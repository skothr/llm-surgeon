"""Tests for probe.attribution_patch_per_head — per-head gradient AP (Phase 3.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import torch

from llm_surgeon.probe import PatchingResult, _capture_residual_stream_with_grad


def _tinyllama_cached() -> bool:
    root = Path(__file__).resolve().parents[1] / ".cache" / "models"
    return any(root.glob("models--TinyLlama--*"))


class TestPatchingResultNHeads:
    def test_n_heads_defaults_to_none(self) -> None:
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a"],
            prompt_tokens_corrupted=["b"],
            direction="denoise",
            measurement_position=0,
        )
        assert result.n_heads is None

    def test_n_heads_set_explicitly(self) -> None:
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a"],
            prompt_tokens_corrupted=["b"],
            direction="denoise",
            measurement_position=0,
            n_heads=32,
        )
        assert result.n_heads == 32


# Reusable mock model for capture tests (hidden=8, n_heads=2, head_dim=4)
class _MockOProj(torch.nn.Linear):
    """Drop-in o_proj — preserves the [batch, seq, hidden] signature."""
    pass


class _MockSelfAttn(torch.nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.o_proj = _MockOProj(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # Simulate concat_z → o_proj; o_proj receives concat_z as its input.
        return (self.o_proj(x),)


class _MockLayer(torch.nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.self_attn = _MockSelfAttn(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return (x + self.self_attn(x)[0],)


class _MockModel(torch.nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 8, vocab: int = 10) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, d_model)
        self.model.layers = torch.nn.ModuleList(
            [_MockLayer(d_model) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor):  # type: ignore[override]
        h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportCallIssue]
        for layer in self.model.layers:  # pyright: ignore[reportGeneralTypeIssues]
            h = layer(h)[0]
        return type("Out", (), {"logits": self.lm_head(h)})()


class _MockTok:
    def __call__(self, text: str, return_tensors: Optional[str] = None) -> Dict:
        ids = [1, 2, 3] if text == "clean" else [4, 5, 6]
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> list:
        return [str(int(i)) for i in ids]


class TestCaptureConcat_z:
    def test_concat_z_shape(self) -> None:
        """concat_z dict has keys for all requested layers; shape [1, seq, hidden]."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z, _, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=True,
            )
        assert set(concat_z.keys()) == {0, 1}, f"expected layers {{0,1}}, got {set(concat_z.keys())}"
        for L, z in concat_z.items():
            assert z.shape == (1, 3, 8), f"layer {L}: expected [1,3,8], got {z.shape}"

    def test_concat_z_in_graph(self) -> None:
        """Base-side concat_z tensors are in the autograd graph."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z, _, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=True,
            )
        for L, z in concat_z.items():
            assert z.requires_grad, f"layer {L} concat_z must require grad"
            assert z.grad_fn is not None, f"layer {L} concat_z must have grad_fn"

    def test_concat_z_grad_populates_after_backward(self) -> None:
        """After backward, concat_z.grad is non-None and non-zero."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, logits, _, concat_z, _, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=True,
            )
            logits.sum().backward()
        for L, z in concat_z.items():
            assert z.grad is not None, f"layer {L} concat_z.grad is None after backward"
            assert z.grad.abs().sum().item() > 0, f"layer {L} concat_z.grad is all zeros"

    def test_concat_z_subset_layers(self) -> None:
        """layers=[1] only captures concat_z at layer 1, not layer 0."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z, _, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=[1],
                capture_concat_z=True,
            )
        assert set(concat_z.keys()) == {1}, f"expected only layer 1, got {set(concat_z.keys())}"

    def test_capture_concat_z_false_returns_empty(self) -> None:
        """Default flag (False) returns an empty concat_z dict."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z, _, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=False,
            )
        assert concat_z == {}, f"expected empty dict, got {concat_z}"


from llm_surgeon.probe import attribution_patch, attribution_patch_per_head


class TestPerHeadAP:
    def test_sum_invariant_mock(self) -> None:
        """sum_h AP_head(L, h, pos) == (Δattn_out · attn_out.grad) / D at 1e-5.

        Validates the chain-rule decomposition through W_O: reconstructing
        per-head AP from concat_z and summing over heads recovers exactly
        (Δattn_out · attn_out.grad)/D up to floating-point rounding.

        NOTE: this is NOT equal to Phase 3.5's AP_attn cells. Phase 3.5
        linearizes at h_post_attn = h_in + attn_out to match exact AP's
        patched quantity; per-head AP decomposes attn_out's contribution
        alone, which is the right unit for mechanistic interpretability of
        individual heads. The two agree only when Δh_in = 0.
        """
        torch.manual_seed(42)
        # hidden=8, n_heads=2, head_dim=4, num_layers=2
        model = _MockModel(num_layers=2, d_model=8).eval()
        # Teach the model to pretend it has a config.
        model.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()

        # --- Expected target: (Δattn_out · attn_out.grad) / D, computed directly ---
        with torch.no_grad():
            from_cap_x, _, from_logits_x, _, _, _, _ = _capture_residual_stream_with_grad(
                model, tok, "clean", sublayers=("attn",),
            )
            from_attn_detached = {k: v.detach().clone() for k, v in from_cap_x.items()}

        with torch.enable_grad():
            base_cap_x, _, base_logits_x, _, _, _, _ = _capture_residual_stream_with_grad(
                model, tok, "corrupted", sublayers=("attn",),
            )
            d_clean = (from_logits_x[-1, 1] - from_logits_x[-1, 2]).detach()
            d_corrupted = (base_logits_x[-1, 1] - base_logits_x[-1, 2]).detach()
            D = (d_clean - d_corrupted).item()
            (base_logits_x[-1, 1] - base_logits_x[-1, 2]).backward()

        seq_len = base_logits_x.shape[0]
        expected: Dict[Tuple[int, int], float] = {}
        for (L, _sub) in list(base_cap_x.keys()):
            base_attn = base_cap_x[(L, "attn")]
            grad = base_attn.grad
            assert grad is not None
            from_attn_L = from_attn_detached[(L, "attn")]
            for pos in range(seq_len):
                v = (
                    (from_attn_L[0, pos] - base_attn[0, pos].detach())
                    * grad[0, pos]
                ).sum().item()
                expected[(L, pos)] = v / D

        # --- Per-head AP (denoise) ---
        head_result = attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
            measurement_position=-1,
        )
        assert head_result.mode == "approx_head"
        assert head_result.n_heads == 2

        head_sum: Dict[Tuple[int, int], float] = {}
        for c in head_result.cells:
            unit: str = c["unit"]
            if not unit.startswith("attn."):
                continue
            key = (c["layer"], c["position"])
            head_sum[key] = head_sum.get(key, 0.0) + c["ap_recovery"]

        # --- Invariant: sum_h ≈ (Δattn_out · attn_out.grad) / D ---
        for key in expected:
            assert key in head_sum, f"missing key {key} in per-head result"
            diff = abs(head_sum[key] - expected[key])
            assert diff < 1e-5, (
                f"sum invariant failed at {key}: "
                f"sum_heads={head_sum[key]:.8f}, expected={expected[key]:.8f}, "
                f"diff={diff:.2e}"
            )

    def test_ffn_anchor_matches_phase35(self) -> None:
        """FFN cells from attribution_patch_per_head match attribution_patch FFN cells."""
        torch.manual_seed(7)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()

        node_result = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
            sublayers=("ffn",),
        )
        node_ffn = {
            (c["layer"], c["position"]): c["ap_recovery"]
            for c in node_result.cells
            if c.get("sublayer") == "ffn"
        }

        head_result = attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
        )
        head_ffn = {
            (c["layer"], c["position"]): c["ap_recovery"]
            for c in head_result.cells
            if c.get("unit") == "ffn"
        }

        for key in node_ffn:
            assert key in head_ffn, f"missing FFN key {key}"
            diff = abs(head_ffn[key] - node_ffn[key])
            assert diff < 1e-5, f"FFN mismatch at {key}: {diff:.2e}"

    def test_cell_count_mock(self) -> None:
        """L=2, n_heads=2, seq=3, positions=all → 18 cells (2 heads + 1 ffn) × 2 layers × 3 pos."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        cells: list = []
        attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            on_cell=lambda L, unit, pos, c: cells.append(c),
        )
        # 2 layers × (2 heads + 1 ffn) × 3 positions = 18
        assert len(cells) == 18, f"expected 18 cells, got {len(cells)}"

    def test_on_cell_unit_strings(self) -> None:
        """on_cell receives correct unit strings: 'attn.h0', 'attn.h1', 'ffn'."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        units: set = set()
        attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            on_cell=lambda L, unit, pos, c: units.add(unit),
        )
        assert "attn.h0" in units
        assert "attn.h1" in units
        assert "ffn" in units
        assert "attn" not in units         # must NOT use old sublayer names
        assert "attn.h2" not in units      # exactly 2 heads

    def test_noise_direction(self) -> None:
        """Noise direction applies 1 + ap_raw/D convention."""
        torch.manual_seed(5)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        result = attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="noise",
        )
        assert result.direction == "noise"
        # All cells must have ap_recovery key
        for c in result.cells:
            assert "ap_recovery" in c
            assert "patched_logits" not in c

    def test_positions_subset(self) -> None:
        """positions=[0,2] yields only those positions."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        cells: list = []
        attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            positions=[0, 2],
            on_cell=lambda L, unit, pos, c: cells.append(c),
        )
        unique_positions = {c["position"] for c in cells}
        assert unique_positions == {0, 2}

    def test_identical_baselines_raises(self) -> None:
        """Divide-by-zero guard."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        # Same token ids for both prompts → identical forward → identical logit_diff.
        class _SameTok:
            def __call__(self, text: str, return_tensors: Optional[str] = None) -> Dict:
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids: torch.Tensor) -> list:
                return [str(int(i)) for i in ids]

        tok2 = _SameTok()
        with pytest.raises(ValueError, match="identical logit_diff"):
            attribution_patch_per_head(
                model, tok2,
                clean_prompt="clean", corrupted_prompt="corrupted",
                correct_token_id=1, incorrect_token_id=2,
            )


class TestTinyLlamaSpearman:
    @pytest.mark.skipif(
        not _tinyllama_cached() or not torch.cuda.is_available(),
        reason="requires cached TinyLlama and CUDA",
    )
    def test_head_sum_vs_attn_out_target_spearman(self) -> None:
        """sum_h AP_head(L,h,pos) vs (Δattn_out · attn_out.grad)/D: ρ > 0.999.

        Validates the chain-rule decomposition on real LLaMA weights. The two
        quantities are algebraically identical (up to fp rounding), so ρ should
        be ~1.0. A dip indicates a W_O orientation or hook-args bug.

        NOTE: this does NOT compare against Phase 3.5's AP_attn cells (which
        linearize at h_post_attn = h_in + attn_out). See the test-class
        docstring of TestPerHeadAP::test_sum_invariant_mock for the derivation.
        """
        import scipy.stats
        from llm_surgeon.surgery import load_model

        model, tokenizer = load_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16"
        )
        model.eval()

        clean = "The capital of France is"
        corrupted = "The capital of Italy is"

        device = next(model.parameters()).device
        with torch.no_grad():
            clean_ids = tokenizer(clean, return_tensors="pt")["input_ids"].to(device)
            corr_ids = tokenizer(corrupted, return_tensors="pt")["input_ids"].to(device)
            clean_logits = model(clean_ids).logits[0, -1]
            corr_logits = model(corr_ids).logits[0, -1]
        correct_id = int(clean_logits.argmax().item())
        incorrect_id = int(corr_logits.argmax().item())

        # --- Direct target: (Δattn_out · attn_out.grad) / D per (L, pos) ---
        with torch.no_grad():
            from_cap, _, from_logits, _, _, _, _ = _capture_residual_stream_with_grad(
                model, tokenizer, clean, sublayers=("attn",),
            )
            from_attn_detached = {
                k: v.detach().clone() for k, v in from_cap.items()
            }

        with torch.enable_grad():
            base_cap, _, base_logits, _, _, _, _ = _capture_residual_stream_with_grad(
                model, tokenizer, corrupted, sublayers=("attn",),
            )
            d_clean = (
                from_logits[-1, correct_id] - from_logits[-1, incorrect_id]
            ).detach()
            d_corrupted = (
                base_logits[-1, correct_id] - base_logits[-1, incorrect_id]
            ).detach()
            D = (d_clean - d_corrupted).item()
            (
                base_logits[-1, correct_id] - base_logits[-1, incorrect_id]
            ).backward()

        seq_len = base_logits.shape[0]
        attn_target: Dict[Tuple[int, int], float] = {}
        for (L, _sub) in list(base_cap.keys()):
            base_attn = base_cap[(L, "attn")]
            grad = base_attn.grad
            assert grad is not None
            from_attn_L = from_attn_detached[(L, "attn")]
            for pos in range(seq_len):
                v = (
                    (from_attn_L[0, pos] - base_attn[0, pos].detach())
                    * grad[0, pos]
                ).sum().item()
                attn_target[(L, pos)] = v / D

        # --- Per-head AP (Phase 3.6), sum heads per (L, pos) ---
        head_result = attribution_patch_per_head(
            model, tokenizer,
            clean_prompt=clean, corrupted_prompt=corrupted,
            correct_token_id=correct_id, incorrect_token_id=incorrect_id,
            direction="denoise", measurement_position=-1,
        )
        head_sum: Dict[Tuple[int, int], float] = {}
        for c in head_result.cells:
            if c.get("unit", "").startswith("attn."):
                key = (c["layer"], c["position"])
                head_sum[key] = head_sum.get(key, 0.0) + c["ap_recovery"]

        shared = sorted(set(attn_target.keys()) & set(head_sum.keys()))
        assert len(shared) > 50, f"too few cells: {len(shared)}"

        x = [attn_target[k] for k in shared]
        y = [head_sum[k] for k in shared]
        result_corr = scipy.stats.spearmanr(x, y)
        rho = float(result_corr.statistic)  # type: ignore[attr-defined]
        print(
            f"\nSpearman(attn_out_target, sum_heads) = {rho:.4f} "
            f"over {len(shared)} cells"
        )
        assert rho > 0.999, (
            f"Spearman ρ={rho:.4f} < 0.999; reconstruction deviates from "
            f"(Δattn_out · attn_out.grad)/D. Likely a W_O orientation or "
            f"concat_z hook bug."
        )

        max_dev = max(abs(head_sum[k] - attn_target[k]) for k in shared)
        print(f"Max absolute deviation: {max_dev:.6f}")
        assert max_dev < 0.01, (
            f"Max deviation {max_dev:.6f} too large; should be fp epsilon."
        )


# ---------------------------------------------------------------------------
# Phase 3.10.1 — IG extension for per-head AP
# ---------------------------------------------------------------------------

class _MockMLP(torch.nn.Module):
    """Minimal MLP with down_proj so IG loop can attach mlp hooks."""
    def __init__(self, hidden: int = 8, intermediate: int = 16) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.act_fn = torch.nn.GELU()
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))


class _MockLayerFull(torch.nn.Module):
    """Layer with both self_attn and mlp — needed for IG hook registration."""
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.self_attn = _MockSelfAttn(d_model)
        self.mlp = _MockMLP(d_model, d_model * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = x + self.self_attn(x)[0]
        x = x + self.mlp(x)
        return (x,)


class _MockModelFull(torch.nn.Module):
    """Mock model with both self_attn and mlp per layer (needed for IG)."""
    def __init__(self, num_layers: int = 2, d_model: int = 8, vocab: int = 10) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, d_model)
        self.model.layers = torch.nn.ModuleList(
            [_MockLayerFull(d_model) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(d_model, vocab, bias=False)
        self.config = type("cfg", (), {  # pyright: ignore[reportArgumentType]
            "num_attention_heads": 2,
            "hidden_size": d_model,
        })()

    def forward(self, input_ids: torch.Tensor):  # type: ignore[override]
        h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportCallIssue]
        for layer in self.model.layers:  # pyright: ignore[reportGeneralTypeIssues]
            h = layer(h)[0]
        return type("Out", (), {"logits": self.lm_head(h)})()


def _pick_head_tokens(model: "_MockModelFull", tok: "_MockTok") -> Tuple[int, int]:
    """Pick (correct, incorrect) ids guaranteed to give a nonzero AP denominator."""
    with torch.no_grad():
        cl = model(tok("clean")["input_ids"]).logits[0, -1]
        co = model(tok("corrupted")["input_ids"]).logits[0, -1]
    c_id = int(cl.argmax().item())
    i_id = int(co.argmax().item())
    if c_id == i_id:
        topk = co.topk(2).indices
        i_id = int(topk[1].item())
    denom = abs((cl[c_id] - cl[i_id]).item() - (co[c_id] - co[i_id]).item())
    if denom < 1e-4:
        vocab = cl.shape[-1]
        best: Tuple[int, int, float] = (0, 1, 0.0)
        for ci in range(vocab):
            for ii in range(vocab):
                if ii == ci:
                    continue
                d = abs((cl[ci] - cl[ii]).item() - (co[ci] - co[ii]).item())
                if d > best[2]:
                    best = (ci, ii, d)
        c_id, i_id = best[0], best[1]
    return c_id, i_id


class TestPerHeadIG:
    def test_per_head_n_steps_converges(self) -> None:
        """n_steps=10 differs from n_steps=1 in at least one cell; n_steps returns correctly."""
        torch.manual_seed(42)
        model = _MockModelFull(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        correct_id, incorrect_id = _pick_head_tokens(model, tok)

        def _run(n: int) -> PatchingResult:
            return attribution_patch_per_head(
                model,
                tok,
                clean_prompt="clean",
                corrupted_prompt="corrupted",
                correct_token_id=correct_id,
                incorrect_token_id=incorrect_id,
                direction="denoise",
                measurement_position=-1,
                n_steps=n,
            )

        r1 = _run(1)
        r10 = _run(10)

        assert r1.n_steps is None, f"n_steps=1 should set n_steps=None on result, got {r1.n_steps}"
        assert r10.n_steps == 10, f"n_steps=10 should set n_steps=10 on result, got {r10.n_steps}"

        for c in r1.cells:
            assert isinstance(c["ap_recovery"], float) and not (c["ap_recovery"] != c["ap_recovery"]), \
                f"Non-finite ap_recovery in r1: {c}"
        for c in r10.cells:
            assert isinstance(c["ap_recovery"], float) and not (c["ap_recovery"] != c["ap_recovery"]), \
                f"Non-finite ap_recovery in r10: {c}"

        r1_map = {(c["layer"], c["unit"], c["position"]): c["ap_recovery"] for c in r1.cells}
        r10_map = {(c["layer"], c["unit"], c["position"]): c["ap_recovery"] for c in r10.cells}
        shared_keys = set(r1_map.keys()) & set(r10_map.keys())
        assert shared_keys, "No shared cells between n_steps=1 and n_steps=10"
        max_diff = max(abs(r1_map[k] - r10_map[k]) for k in shared_keys)
        assert max_diff > 1e-4, (
            f"n_steps=10 and n_steps=1 agree within 1e-4 on all cells (max_diff={max_diff:.2e}); "
            "IG should differ from first-order on a nonlinear mock."
        )


@pytest.mark.skipif(
    not _tinyllama_cached() or not torch.cuda.is_available(),
    reason="requires cached TinyLlama and CUDA",
)
def test_per_head_ig_tinyllama() -> None:
    """Top-20 Spearman between n_steps=1 and n_steps=5 |ap_recovery| rankings > 0.5."""
    import math
    import numpy as np
    import scipy.stats
    from llm_surgeon.surgery import load_model

    model, tokenizer = load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16"
    )
    model.eval()

    clean = "The capital of France is"
    corrupted = "The capital of Russia is"

    device = next(model.parameters()).device
    with torch.no_grad():
        paris_ids = tokenizer(" Paris", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        moscow_ids = tokenizer(" Moscow", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    paris_id = int(paris_ids[0, -1].item())
    moscow_id = int(moscow_ids[0, -1].item())

    def _run_head(n: int) -> PatchingResult:
        return attribution_patch_per_head(
            model,
            tokenizer,
            clean_prompt=clean,
            corrupted_prompt=corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=moscow_id,
            direction="denoise",
            measurement_position=-1,
            n_steps=n,
        )

    r1 = _run_head(1)
    r5 = _run_head(5)

    assert r1.n_steps is None
    assert r5.n_steps == 5

    r1_map = {(c["layer"], c["unit"], c["position"]): c["ap_recovery"] for c in r1.cells}
    r5_map = {(c["layer"], c["unit"], c["position"]): c["ap_recovery"] for c in r5.cells}

    shared = sorted(set(r1_map.keys()) & set(r5_map.keys()))
    assert len(shared) > 20, f"too few shared cells: {len(shared)}"

    top20 = sorted(shared, key=lambda k: abs(r1_map[k]), reverse=True)[:20]
    x = [abs(r1_map[k]) for k in top20]
    y = [abs(r5_map[k]) for k in top20]
    result_corr = scipy.stats.spearmanr(x, y)
    rho = float(result_corr.statistic)  # type: ignore[attr-defined]
    print(f"\nper-head IG TinyLlama: Spearman(n1, n5) top-20 = {rho:.4f}")
    assert rho > 0.5, (
        f"Spearman ρ={rho:.4f} < 0.5 on top-20 cells; IG per-head rankings diverged unexpectedly."
    )
    assert all(math.isfinite(c["ap_recovery"]) for c in r5.cells), "Non-finite ap_recovery in IG result"
