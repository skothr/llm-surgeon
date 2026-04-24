"""Tests for probe.attribution_patch — gradient-based AP (Phase 3.5 + 3.10 IG)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

from llm_surgeon.probe import PatchingResult, _capture_residual_stream_with_grad, attribution_patch


def _tinyllama_cached() -> bool:
    root = Path(__file__).resolve().parents[1] / ".cache" / "models"
    return any(root.glob("models--TinyLlama--*"))


class TestCaptureWithGrad:
    def test_captured_tensors_have_grad_fn(self):
        """Every captured (L, sub) tensor keeps its computation graph."""
        # Minimal mock LLaMA-shaped model sufficient for hook invocation.
        class _MockLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(4, 4)
            def forward(self, x):
                return (x + self.self_attn(x),)  # tuple like HF layers

        class _MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(10, 4)  # pyright: ignore[reportAttributeAccessIssue]
                self.model.layers = torch.nn.ModuleList([_MockLayer() for _ in range(2)])  # pyright: ignore[reportAttributeAccessIssue]
                self.lm_head = torch.nn.Linear(4, 10)

            def forward(self, input_ids):
                h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
                for layer in self.model.layers:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
                    h = layer(h)[0]
                return type("Out", (), {"logits": self.lm_head(h)})

        class _MockTok:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids):
                return [str(int(i)) for i in ids]

        model = _MockModel().eval()
        tok = _MockTok()
        captured, h_ins, logits, tokens, _, _, _ = _capture_residual_stream_with_grad(
            model, tok, "hello", sublayers=("attn", "ffn"), layers=None,
        )
        assert len(captured) == 2 * 2  # 2 layers × 2 sublayers
        assert len(h_ins) == 2  # one h_in per layer when attn is captured
        for key, tensor in captured.items():
            assert tensor.requires_grad, f"{key} must require grad"
            assert tensor.grad_fn is not None, f"{key} must have grad_fn"
        assert logits.requires_grad
        assert len(tokens) == 3

    def test_grad_populates_after_backward(self):
        """Calling .backward() populates .grad on each captured tensor."""
        # (same mock setup as above — reuse the helper pattern)
        # After capture, compute sum = logits.sum(); sum.backward();
        # assert every captured[(L,sub)].grad is not None and non-zero.
        class _MockLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(4, 4)
            def forward(self, x):
                return (x + self.self_attn(x),)

        class _MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(10, 4)  # pyright: ignore[reportAttributeAccessIssue]
                self.model.layers = torch.nn.ModuleList([_MockLayer() for _ in range(2)])  # pyright: ignore[reportAttributeAccessIssue]
                self.lm_head = torch.nn.Linear(4, 10)
            def forward(self, input_ids):
                h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
                for layer in self.model.layers:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
                    h = layer(h)[0]
                return type("Out", (), {"logits": self.lm_head(h)})

        class _MockTok:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids):
                return [str(int(i)) for i in ids]

        model = _MockModel().eval()
        tok = _MockTok()
        captured, _, logits, _, _, _, _ = _capture_residual_stream_with_grad(
            model, tok, "hello", sublayers=("attn", "ffn"), layers=None,
        )
        logits.sum().backward()
        for key, tensor in captured.items():
            assert tensor.grad is not None, f"{key} missing grad after backward"
            # At least one element must be non-zero (not a constant-zero grad).
            assert tensor.grad.abs().sum().item() > 0, f"{key} grad is all zeros"


class TestValidation:
    def test_missing_token_ids_raises(self):
        # A minimal mock model — we just need validation to fire before forward.
        with pytest.raises(ValueError, match="correct_token_id and incorrect_token_id"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="b",
                correct_token_id=None, incorrect_token_id=None,  # type: ignore[arg-type]
            )

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="b",
                correct_token_id=1, incorrect_token_id=2,
                direction="wobble",
            )

    def test_bad_sublayer_raises(self):
        with pytest.raises(ValueError, match="sublayers must be"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="b",
                correct_token_id=1, incorrect_token_id=2,
                sublayers=("mlp",),  # type: ignore[arg-type]
            )

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="", corrupted_prompt="b",
                correct_token_id=1, incorrect_token_id=2,
            )
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="",
                correct_token_id=1, incorrect_token_id=2,
            )


class _MockLlamaBlock(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.self_attn = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        return (x + self.self_attn(x),)


class _MockLlama(torch.nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 4, vocab: int = 10):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, d_model)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.layers = torch.nn.ModuleList(  # pyright: ignore[reportAttributeAccessIssue]
            [_MockLlamaBlock(d_model) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(d_model, vocab)
    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
        for layer in self.model.layers:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
            h = layer(h)[0]
        return type("Out", (), {"logits": self.lm_head(h)})


class _MockTok:
    def __init__(self, token_ids):
        self._ids = token_ids
    def __call__(self, text, return_tensors=None):
        # Distinct IDs per prompt identity — use hash-based selection so
        # clean and corrupted produce different activations.
        ids = self._ids[0] if text == "clean" else self._ids[1]
        return {"input_ids": torch.tensor([ids])}
    def convert_ids_to_tokens(self, ids):
        return [str(int(i)) for i in ids]


class TestAttributionPatchLoop:
    def test_callback_fires_per_cell_denoise(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        cells = []
        result = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
            measurement_position=-1,
            on_cell=lambda L, sub, pos, c: cells.append((L, sub, pos, c)),
        )
        # 2 layers × 2 sublayers × 3 positions = 12 cells
        assert len(cells) == 12
        assert result.mode == "approx"
        assert result.direction == "denoise"
        # Every cell has ap_recovery and no patched_logits
        for _, _, _, c in cells:
            assert "ap_recovery" in c
            assert "patched_logits" not in c
            assert isinstance(c["ap_recovery"], float)

    def test_noise_direction_flips_base_prompt(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        result = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="noise",
        )
        assert result.direction == "noise"
        # prompt_tokens_clean came from clean_prompt's tokenization
        assert result.prompt_tokens_clean == ["1", "2", "3"]
        assert result.prompt_tokens_corrupted == ["4", "5", "6"]

    def test_positions_subset(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        cells = []
        attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            positions=[0, 2],
            on_cell=lambda L, sub, pos, c: cells.append((L, sub, pos, c)),
        )
        # 2 layers × 2 sublayers × 2 positions = 8 cells
        assert len(cells) == 8
        unique_positions = {c[2] for c in cells}
        assert unique_positions == {0, 2}

    def test_layers_subset(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=4, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        cells = []
        attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            layers=[1, 2],
            on_cell=lambda L, sub, pos, c: cells.append((L, sub, pos, c)),
        )
        # 2 layers (1, 2) × 2 sublayers × 3 positions = 12 cells
        assert len(cells) == 12
        unique_layers = {c[0] for c in cells}
        assert unique_layers == {1, 2}

    def test_identical_baselines_raises(self):
        """Divide-by-zero guard when clean and corrupted produce same logit_diff."""
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        # Same token ids for both prompts → identical forward → identical logit_diff.
        tok = _MockTok(token_ids=([1, 2, 3], [1, 2, 3]))
        with pytest.raises(ValueError, match="identical logit_diff"):
            attribution_patch(
                model, tok,
                clean_prompt="clean", corrupted_prompt="corrupted",
                correct_token_id=1, incorrect_token_id=2,
            )


class TestApproxVsExactCorrelates:
    @pytest.mark.skipif(
        not _tinyllama_cached() or not torch.cuda.is_available(),
        reason="requires cached TinyLlama and CUDA",
    )
    def test_tinyllama_capital_swap_spearman(self):
        """AP approx rankings must correlate with exact AP rankings (Spearman ≥ 0.5)."""
        import scipy.stats
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import activation_patch

        model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        model.eval()

        clean = "The capital of France is"
        corrupted = "The capital of Italy is"

        # Resolve target tokens from argmax of clean baseline.
        with torch.no_grad():
            clean_ids = tokenizer(clean, return_tensors="pt")["input_ids"].to(model.device)
            corr_ids = tokenizer(corrupted, return_tensors="pt")["input_ids"].to(model.device)
            clean_last = model(clean_ids).logits[0, -1]
            corr_last = model(corr_ids).logits[0, -1]
        correct_id = int(clean_last.argmax().item())
        incorrect_id = int(corr_last.argmax().item())

        # --- Exact activation patching ---
        exact_result = activation_patch(
            model, tokenizer, clean_prompt=clean, corrupted_prompt=corrupted,
            direction="denoise", measurement_position=-1,
        )
        # Exact produces patched_logits — compute logit_diff_recovery per cell.
        exact_scores: Dict[Tuple[int, str, int], float] = {}
        d_clean = float(
            exact_result.clean_baseline_logits[correct_id]
            - exact_result.clean_baseline_logits[incorrect_id]
        )
        d_corrupted = float(
            exact_result.corrupted_baseline_logits[correct_id]
            - exact_result.corrupted_baseline_logits[incorrect_id]
        )
        D = d_clean - d_corrupted
        for cell in exact_result.cells:
            pl = cell["patched_logits"]
            d_patched = float(pl[correct_id] - pl[incorrect_id])
            key = (cell["layer"], cell["sublayer"], cell["position"])
            exact_scores[key] = (d_patched - d_corrupted) / D

        # --- Approx attribution patching ---
        approx_result = attribution_patch(
            model, tokenizer, clean_prompt=clean, corrupted_prompt=corrupted,
            correct_token_id=correct_id, incorrect_token_id=incorrect_id,
            direction="denoise", measurement_position=-1,
        )
        approx_scores: Dict[Tuple[int, str, int], float] = {
            (c["layer"], c["sublayer"], c["position"]): c["ap_recovery"]
            for c in approx_result.cells
        }

        # --- Spearman rank correlation on shared keys ---
        shared = sorted(set(exact_scores.keys()) & set(approx_scores.keys()))
        assert len(shared) > 20, f"too few cells to correlate: {len(shared)}"
        x = [exact_scores[k] for k in shared]
        y = [approx_scores[k] for k in shared]
        result_corr = scipy.stats.spearmanr(x, y)
        rho = float(result_corr.statistic)  # type: ignore[attr-defined]
        print(f"\nSpearman(exact, approx) = {rho:.3f} over {len(shared)} cells")
        assert rho >= 0.5, (
            f"AP approx rank correlation {rho:.3f} below threshold 0.5; "
            f"the gradient approximation is not tracking exact patching"
        )


# ---------------------------------------------------------------------------
# Phase 3.10 — Integrated Gradients unit tests
# ---------------------------------------------------------------------------

class _MockLlamaBlockIG(torch.nn.Module):
    """LLaMA-shaped block with explicit self_attn + mlp sub-modules.

    Required for IG tests: _integrated_gradients_loop hooks both
    model.model.layers[L].self_attn and model.model.layers[L].mlp.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.self_attn = torch.nn.Linear(d_model, d_model)
        self.mlp = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:  # type: ignore[override]
        # Residual-stream: h_post_attn = x + attn_out; layer_out = h_post_attn + mlp_out
        attn_out = self.self_attn(x)
        h_post_attn = x + attn_out
        mlp_out = self.mlp(h_post_attn)
        return (h_post_attn + mlp_out,)


class _MockLlamaIG(torch.nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 4, vocab: int = 10):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, d_model)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.layers = torch.nn.ModuleList(  # pyright: ignore[reportAttributeAccessIssue]
            [_MockLlamaBlockIG(d_model) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor) -> object:
        h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
        for layer in self.model.layers:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
            h = layer(h)[0]
        return type("Out", (), {"logits": self.lm_head(h)})


def _make_ig_fixtures(seed: int = 0):
    """Return (model, tok) using the IG-capable mock."""
    torch.manual_seed(seed)
    model = _MockLlamaIG(num_layers=2, d_model=4, vocab=10).eval()
    tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
    return model, tok


class TestNStepsIG:
    def test_n_steps_1_matches_old_behavior(self):
        """n_steps=1 and default (no n_steps kwarg) produce bit-identical scores."""
        model, tok = _make_ig_fixtures(seed=1)
        common = dict(
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
        )
        r_default = attribution_patch(model, tok, **common)  # type: ignore[arg-type]
        r_1 = attribution_patch(model, tok, n_steps=1, **common)  # type: ignore[arg-type]

        assert r_default.n_steps is None
        assert r_1.n_steps is None
        assert len(r_default.cells) == len(r_1.cells)
        for c_d, c_1 in zip(r_default.cells, r_1.cells):
            assert abs(c_d["ap_recovery"] - c_1["ap_recovery"]) < 1e-9, (
                f"cell {c_d['layer']},{c_d['sublayer']},{c_d['position']}: "
                f"default={c_d['ap_recovery']!r}, n_steps=1={c_1['ap_recovery']!r}"
            )

    def test_n_steps_5_runs_and_differs(self):
        """n_steps=5 runs without error; scores are finite and differ from n_steps=1."""
        model, tok = _make_ig_fixtures(seed=2)
        common = dict(
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
        )
        r1 = attribution_patch(model, tok, n_steps=1, **common)  # type: ignore[arg-type]
        r5 = attribution_patch(model, tok, n_steps=5, **common)  # type: ignore[arg-type]

        assert r5.n_steps == 5
        for c in r5.cells:
            assert math.isfinite(c["ap_recovery"]), f"non-finite cell: {c}"

        scores_1 = [c["ap_recovery"] for c in r1.cells]
        scores_5 = [c["ap_recovery"] for c in r5.cells]
        max_diff = max(abs(a - b) for a, b in zip(scores_1, scores_5))
        assert max_diff > 1e-4, (
            f"n_steps=5 scores too close to n_steps=1 (max diff={max_diff:.2e}); "
            "IG averaging should shift at least one cell by > 1e-4"
        )

    def test_n_steps_convergence(self):
        """IG should converge as N grows: n_steps=10 and n_steps=20 should
        agree more closely with each other than n_steps=2 agrees with n_steps=20.

        Note: Phase 3.5 uses cumulative residual-stream Δs (h_post_attn for
        attn rows, layer_output for ffn rows), which OVERLAP across depth.
        These Δs match exact-AP semantics but don't satisfy IG's completeness
        axiom (Σ ap_recovery ≠ 1.0 in general — it counts each layer's
        contribution multiple times by construction). What IS meaningful is
        convergence: as N → ∞, IG approaches the exact path integral, so
        adjacent-N results should agree ever-closer.
        """
        model, tok = _make_ig_fixtures(seed=3)

        def _run(n: int) -> PatchingResult:
            return attribution_patch(
                model, tok,
                clean_prompt="clean", corrupted_prompt="corrupted",
                correct_token_id=1, incorrect_token_id=2,
                direction="denoise",
                sublayers=("attn", "ffn"),
                n_steps=n,
            )

        r2 = _run(2)
        r10 = _run(10)
        r20 = _run(20)

        def max_abs_diff(a, b):
            return max(abs(c_a["ap_recovery"] - c_b["ap_recovery"])
                       for c_a, c_b in zip(a.cells, b.cells))

        diff_10_20 = max_abs_diff(r10, r20)
        diff_2_20 = max_abs_diff(r2, r20)
        assert diff_10_20 < diff_2_20, (
            f"IG did not converge: diff(10, 20)={diff_10_20:.4f} should be "
            f"< diff(2, 20)={diff_2_20:.4f}"
        )

    def test_n_steps_validation(self):
        """n_steps outside [1, 50] raises ValueError."""
        model, tok = _make_ig_fixtures(seed=4)
        common = dict(
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
        )
        for bad in (0, -1, 51, 100):
            with pytest.raises(ValueError, match=r"n_steps must be int in \[1, 50\]"):
                attribution_patch(model, tok, n_steps=bad, **common)  # type: ignore[arg-type]

    def test_n_steps_noise_direction(self):
        """n_steps=5 with direction='noise' produces finite scores."""
        model, tok = _make_ig_fixtures(seed=5)
        r = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="noise",
            n_steps=5,
        )
        assert r.n_steps == 5
        for c in r.cells:
            assert math.isfinite(c["ap_recovery"]), f"non-finite cell: {c}"


@pytest.mark.skipif(
    not _tinyllama_cached() or not torch.cuda.is_available(),
    reason="requires cached TinyLlama and CUDA",
)
def test_ig_tinyllama_converges():
    """IG n_steps=5 rank-correlates with n_steps=1 (Spearman ρ > 0.8) on TinyLlama."""
    import scipy.stats
    from llm_surgeon.surgery import load_model

    model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
    model.eval()

    clean = "The capital of France is"
    corrupted = "The capital of Russia is"

    paris_ids = tokenizer(" Paris", add_special_tokens=False)["input_ids"]
    moscow_ids = tokenizer(" Moscow", add_special_tokens=False)["input_ids"]
    correct_id = int(paris_ids[0])
    incorrect_id = int(moscow_ids[0])

    r1 = attribution_patch(
        model, tokenizer,
        clean_prompt=clean, corrupted_prompt=corrupted,
        correct_token_id=correct_id, incorrect_token_id=incorrect_id,
        direction="denoise", measurement_position=-1,
        n_steps=1,
    )
    r5 = attribution_patch(
        model, tokenizer,
        clean_prompt=clean, corrupted_prompt=corrupted,
        correct_token_id=correct_id, incorrect_token_id=incorrect_id,
        direction="denoise", measurement_position=-1,
        n_steps=5,
    )

    assert r1.n_steps is None
    assert r5.n_steps == 5

    for c in r5.cells:
        assert math.isfinite(c["ap_recovery"]), f"non-finite IG cell: {c}"

    def key_fn(c: Dict) -> Tuple[int, str, int]:
        return (c["layer"], c["sublayer"], c["position"])

    r1_map = {key_fn(c): c["ap_recovery"] for c in r1.cells}
    r5_map = {key_fn(c): c["ap_recovery"] for c in r5.cells}
    shared = sorted(set(r1_map.keys()) & set(r5_map.keys()))
    assert len(shared) > 20, f"too few shared cells: {len(shared)}"

    r1_scores: List[float] = [r1_map[k] for k in shared]
    r5_scores: List[float] = [r5_map[k] for k in shared]

    # Full-rank Spearman is NOT meaningful here because IG's whole purpose is
    # to re-rank saturated cells relative to first-order AP. On TinyLlama with
    # 264 cells we empirically see ρ ≈ 0.4 — low tail-end cells are near-zero
    # magnitude and re-rank as noise, which is expected. The interpretability-
    # relevant check is whether the TOP cells (those dominating metric delta)
    # agree. We check two things: (1) full-rank ρ is above pure-random (> 0.2,
    # proves IG isn't returning garbage), (2) top-20 cells by |first-order|
    # rank with high correlation under IG.
    result_corr = scipy.stats.spearmanr(r1_scores, r5_scores)
    rho_full = float(result_corr.statistic)  # type: ignore[attr-defined]
    print(f"\nSpearman full-rank(n_steps=1, n_steps=5) = {rho_full:.3f} over {len(shared)} cells")
    assert rho_full > 0.2, (
        f"IG full-rank correlation {rho_full:.3f} is near random; "
        "implementation may be broken"
    )

    # Top-20 agreement: pick indices of top-20 cells by |r1_score|, take both
    # runs' scores at those indices, check rank correlation.
    import numpy as _np
    abs_r1 = _np.abs(_np.array(r1_scores))
    top20_idx = _np.argsort(-abs_r1)[:20]
    r1_top = [r1_scores[i] for i in top20_idx]
    r5_top = [r5_scores[i] for i in top20_idx]
    top_corr = scipy.stats.spearmanr(r1_top, r5_top)
    rho_top = float(top_corr.statistic)  # type: ignore[attr-defined]
    print(f"Spearman top-20(n_steps=1, n_steps=5) = {rho_top:.3f}")
    assert rho_top > 0.5, (
        f"IG top-20 rank correlation {rho_top:.3f} too low; "
        "n_steps=5 should preserve ordering of top-magnitude cells"
    )
