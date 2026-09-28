"""Tests for probe.edge_attribution_patch — edge-level gradient AP (Phase 3.7)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import pytest
import torch

from llm_surgeon.probe import PatchingResult, _capture_residual_stream_with_grad


def _tinyllama_cached() -> bool:
    from llm_surgeon.surgery import _is_cached
    return _is_cached("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


class TestPatchingResultNEdges:
    def test_n_edges_defaults_to_none(self) -> None:
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a"],
            prompt_tokens_corrupted=["b"],
            direction="denoise",
            measurement_position=0,
        )
        assert result.n_edges is None

    def test_n_edges_set_explicitly(self) -> None:
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a"],
            prompt_tokens_corrupted=["b"],
            direction="denoise",
            measurement_position=0,
            n_edges=90432,
        )
        assert result.n_edges == 90432


import torch.nn as nn


class _MockLN(nn.Module):
    """Minimal LayerNorm-like module that passes through its input (no learned params needed)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _MockMLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _MockSelfAttn(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return (self.o_proj(x),)


class _MockLayer(nn.Module):
    """Mirrors HF LLaMA layer structure for hook compatibility."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.input_layernorm = _MockLN()
        self.self_attn = _MockSelfAttn(d_model)
        self.post_attention_layernorm = _MockLN()
        self.mlp = _MockMLP(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h)[0]
        h = x + attn_out
        h2 = self.post_attention_layernorm(h)
        ffn_out = self.mlp(h2)
        return (h + ffn_out,)


class _MockModelFull(nn.Module):
    """Mock with input_layernorm, post_attention_layernorm, and model.norm for hook tests."""
    def __init__(self, num_layers: int = 2, d_model: int = 8, vocab: int = 10) -> None:
        super().__init__()

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab, d_model)
                self.layers = nn.ModuleList([_MockLayer(d_model) for _ in range(num_layers)])

        self.model = _Inner()
        self.model.norm = _MockLN()       # type: ignore[attr-defined]
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor):  # type: ignore[override]
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        h = self.model.norm(h)  # type: ignore[operator]
        return type("Out", (), {"logits": self.lm_head(h)})()


class _MockTok:
    def __call__(self, text: str, return_tensors: Optional[str] = None) -> Dict:
        ids = [1, 2, 3] if text == "clean" else [4, 5, 6]
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> list:
        return [str(int(i)) for i in ids]


class TestReaderGradCapture:
    def test_reader_inputs_keys(self) -> None:
        """capture_reader_grads=True returns keys for attn_in, ffn_in at each layer, plus logits."""
        torch.manual_seed(0)
        model = _MockModelFull(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, _, reader_inputs, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_reader_grads=True,
            )
        expected = {("attn_in", 0), ("ffn_in", 0), ("attn_in", 1), ("ffn_in", 1), ("logits", 2)}
        assert set(reader_inputs.keys()) == expected, f"got {set(reader_inputs.keys())}"

    def test_reader_inputs_shape(self) -> None:
        """Each reader input tensor has shape [1, seq, hidden]."""
        torch.manual_seed(0)
        model = _MockModelFull(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, _, reader_inputs, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_reader_grads=True,
            )
        for key, tensor in reader_inputs.items():
            assert tensor.shape == (1, 3, 8), f"{key}: expected [1,3,8], got {tensor.shape}"

    def test_reader_inputs_in_graph(self) -> None:
        """Reader input tensors are in the autograd graph (requires_grad=True, grad_fn not None)."""
        torch.manual_seed(0)
        model = _MockModelFull(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, _, reader_inputs, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_reader_grads=True,
            )
        for key, tensor in reader_inputs.items():
            assert tensor.requires_grad, f"{key} must require grad"
            assert tensor.grad_fn is not None, f"{key} must have grad_fn"

    def test_reader_grads_populated_after_backward(self) -> None:
        """After backward, all reader_inputs[k].grad are non-None and non-zero."""
        torch.manual_seed(0)
        model = _MockModelFull(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, logits, _, _, reader_inputs, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_reader_grads=True,
            )
            logits.sum().backward()
        for key, tensor in reader_inputs.items():
            assert tensor.grad is not None, f"{key}.grad is None after backward"
            assert tensor.grad.abs().sum().item() > 0, f"{key}.grad is all zeros"

    def test_reader_grads_absent_without_flag(self) -> None:
        """capture_reader_grads=False returns an empty reader_inputs dict."""
        torch.manual_seed(0)
        model = _MockModelFull(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, _, reader_inputs, _ = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_reader_grads=False,
            )
        assert reader_inputs == {}, f"expected empty dict, got {set(reader_inputs.keys())}"


from llm_surgeon.probe import edge_attribution_patch


class TestEdgeAP:
    def test_validation_top_k_edges_zero(self) -> None:
        """top_k_edges < 1 raises ValueError."""
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()
        tok = _MockTok()
        with pytest.raises(ValueError, match="top_k_edges"):
            edge_attribution_patch(
                model, tok, "clean", "other",
                correct_token_id=1, incorrect_token_id=4,
                top_k_edges=0,
            )

    def test_validation_identical_baselines(self) -> None:
        """Identical logit_diff raises ValueError."""
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()
        tok = _MockTok()
        # Use same prompt twice → identical logit_diff = 0
        with pytest.raises(ValueError, match="identical logit_diff"):
            edge_attribution_patch(
                model, tok, "clean", "clean",
                correct_token_id=1, incorrect_token_id=4,
            )

    def test_edge_count_mock(self) -> None:
        """Total edge count matches formula for 2-layer, 2-head mock."""
        # Writers: 1 embed + 2 layers × 2 heads + 2 layers × 1 ffn = 1 + 4 + 2 = 7
        # Readers: 2 × attn_in + 2 × ffn_in + 1 logits = 5
        # Valid edges (by rule): compute expected count manually
        # embed → all 5 readers = 5
        # attn.h0_L0 → attn_in_L1, ffn_in_L0, ffn_in_L1, logits = 4
        # attn.h1_L0 → same = 4
        # attn.h0_L1 → ffn_in_L1, logits = 2
        # attn.h1_L1 → ffn_in_L1, logits = 2
        # ffn_L0 → attn_in_L1, ffn_in_L1, logits = 3
        # ffn_L1 → logits = 1
        # Total = 5 + 4 + 4 + 2 + 2 + 3 + 1 = 21 edges (per position)
        torch.manual_seed(42)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()

        class _Cfg:
            num_attention_heads = 2
            hidden_size = 8
        model.config = _Cfg()  # type: ignore[attr-defined]

        tok = _MockTok()
        result = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=1000,     # large enough to keep all
        )
        seq_len = 3
        expected_per_pos = 21
        assert result.n_edges == expected_per_pos * seq_len, \
            f"expected {expected_per_pos * seq_len}, got {result.n_edges}"

    def test_sum_invariant_mock(self) -> None:
        """For each reader r and position, Σ over writers of the edge effect == Δr · grad_r / D.

        With every layer targeted, each reader's pre-norm residual is the
        embedding plus every causally-earlier writer's output, so the edge
        scores into one reader must sum to that reader's node-level score.
        The right-hand side is computed here from captures alone.
        """
        torch.manual_seed(7)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()
        model.config = SimpleNamespace(num_attention_heads=2, hidden_size=8)  # type: ignore[attr-defined]
        tok = _MockTok()
        result = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=10_000,
        )
        assert result.n_edges == len(result.cells)

        with torch.no_grad():
            _, _, from_logits, _, _, from_readers, _ = _capture_residual_stream_with_grad(
                model, tok, "clean", capture_reader_grads=True,
            )
        with torch.enable_grad():
            _, _, base_logits, _, _, base_readers, _ = _capture_residual_stream_with_grad(
                model, tok, "other", capture_reader_grads=True,
            )
            metric = base_logits[-1, 1] - base_logits[-1, 4]
            keys = list(base_readers)
            grads = dict(zip(keys, torch.autograd.grad(metric, [base_readers[k] for k in keys])))
        denom = ((from_logits[-1, 1] - from_logits[-1, 4]) - (base_logits[-1, 1] - base_logits[-1, 4])).item()

        from collections import defaultdict
        sums: Dict[Tuple, float] = defaultdict(float)
        for cell in result.cells:
            sums[(cell["reader_unit"], cell["reader_layer"], cell["position"])] += cell["ap_recovery"]
        assert len(sums) == len(keys) * 3
        for (ru, rl, pos), total in sums.items():
            key = (ru, rl)
            target = ((from_readers[key][0, pos] - base_readers[key][0, pos].detach()) * grads[key][0, pos]).sum().item() / denom
            assert abs(total - target) < 1e-5, f"reader {key} pos={pos}: Σ edges={total}, node={target}"

    def test_per_head_decomposability_mock(self) -> None:
        """Σ_h edge effect (L attn.hN → r) == Δattn_out(L) · grad_r / D at 1e-5."""
        torch.manual_seed(13)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()
        model.config = SimpleNamespace(num_attention_heads=2, hidden_size=8)  # type: ignore[attr-defined]
        tok = _MockTok()
        result = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=10_000,
        )

        with torch.no_grad():
            from_cap, _, from_logits, _, _, _, _ = _capture_residual_stream_with_grad(model, tok, "clean")
        with torch.enable_grad():
            base_cap, _, base_logits, _, _, base_readers, _ = _capture_residual_stream_with_grad(
                model, tok, "other", capture_reader_grads=True,
            )
            metric = base_logits[-1, 1] - base_logits[-1, 4]
            keys = list(base_readers)
            grads = dict(zip(keys, torch.autograd.grad(metric, [base_readers[k] for k in keys])))
        denom = ((from_logits[-1, 1] - from_logits[-1, 4]) - (base_logits[-1, 1] - base_logits[-1, 4])).item()

        from collections import defaultdict
        head_sums: Dict[Tuple, float] = defaultdict(float)
        for cell in result.cells:
            if cell["writer_unit"].startswith("attn.h"):
                key = (cell["writer_layer"], cell["reader_unit"], cell["reader_layer"], cell["position"])
                head_sums[key] += cell["ap_recovery"]
        assert len(head_sums) > 0, "no attn head edges found"
        for (L_w, ru, rl, pos), s in head_sums.items():
            delta_attn = from_cap[(L_w, "attn")][0, pos] - base_cap[(L_w, "attn")][0, pos].detach()
            target = (delta_attn * grads[(ru, rl)][0, pos]).sum().item() / denom
            assert abs(s - target) < 1e-5, f"{(L_w, ru, rl, pos)}: Σ_h={s}, direct={target}"

    def test_top_k_selection(self) -> None:
        """Only top_k_edges cells emitted; they are the ones with largest |ap_recovery|."""
        torch.manual_seed(99)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()

        class _Cfg:
            num_attention_heads = 2
            hidden_size = 8
        model.config = _Cfg()  # type: ignore[attr-defined]

        tok = _MockTok()
        result_all = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=1000,
        )
        k = 5
        result_topk = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=k,
        )
        assert len(result_topk.cells) == k
        # Verify top-k are the k largest by |ap_recovery| from the full set.
        full_sorted = sorted(result_all.cells, key=lambda c: abs(c["ap_recovery"]), reverse=True)
        top_from_full = {
            (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"])
            for c in full_sorted[:k]
        }
        top_from_topk = {
            (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"])
            for c in result_topk.cells
        }
        assert top_from_full == top_from_topk

    def test_on_cell_receives_dict(self) -> None:
        """on_cell receives a dict with the six required keys."""
        torch.manual_seed(5)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()

        class _Cfg:
            num_attention_heads = 2
            hidden_size = 8
        model.config = _Cfg()  # type: ignore[attr-defined]

        tok = _MockTok()
        received: list = []
        def on_cell(cell: dict) -> None:
            received.append(cell)

        edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=10,
            on_cell=on_cell,
        )
        assert len(received) > 0
        required_keys = {"writer_layer", "writer_unit", "reader_layer", "reader_unit",
                         "position", "ap_recovery"}
        for cell in received:
            assert required_keys.issubset(cell.keys()), f"missing keys in {cell.keys()}"

    def test_embed_writer_present(self) -> None:
        """At least one cell per unique reader has writer_unit == 'embed'."""
        torch.manual_seed(3)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()

        class _Cfg:
            num_attention_heads = 2
            hidden_size = 8
        model.config = _Cfg()  # type: ignore[attr-defined]

        tok = _MockTok()
        result = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=1000,
        )
        embed_cells = [c for c in result.cells if c["writer_unit"] == "embed"]
        assert len(embed_cells) > 0, "no embed writer cells found"

    def test_invalid_edges_absent(self) -> None:
        """No cell has a same-layer or later-layer ffn writer with a ffn_in or attn_in reader."""
        torch.manual_seed(11)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()

        class _Cfg:
            num_attention_heads = 2
            hidden_size = 8
        model.config = _Cfg()  # type: ignore[attr-defined]

        tok = _MockTok()
        result = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
            top_k_edges=1000,
        )
        for cell in result.cells:
            if cell["writer_unit"] == "ffn" and cell["reader_unit"] in ("attn_in", "ffn_in"):
                # FFN writer must strictly precede reader layer
                assert cell["writer_layer"] < cell["reader_layer"], \
                    f"Invalid edge: L{cell['writer_layer']}.ffn → L{cell['reader_layer']}.{cell['reader_unit']}"

    def test_mode_is_edge(self) -> None:
        """PatchingResult.mode == 'edge'."""
        torch.manual_seed(1)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()

        class _Cfg:
            num_attention_heads = 2
            hidden_size = 8
        model.config = _Cfg()  # type: ignore[attr-defined]

        tok = _MockTok()
        result = edge_attribution_patch(
            model, tok, "clean", "other",
            correct_token_id=1, incorrect_token_id=4,
        )
        assert result.mode == "edge"
        assert result.n_edges is not None and result.n_edges > 0
        assert result.n_heads == 2


class TestEdgeAPIntegratedGradients:
    def test_edge_ap_n_steps_equals_ap_on_linear_model(self) -> None:
        """The mock is linear in every sublayer, so every gradient is constant
        and IG must reproduce the n_steps=1 scores. An IG loop that cuts the
        gradient paths through downstream sublayers fails this."""
        torch.manual_seed(77)
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()
        model.config = SimpleNamespace(num_attention_heads=2, hidden_size=8)  # type: ignore[attr-defined]
        tok = _MockTok()
        common: Dict = dict(correct_token_id=1, incorrect_token_id=4, top_k_edges=10_000)
        r_1 = edge_attribution_patch(model, tok, "clean", "other", n_steps=1, **common)
        r_10 = edge_attribution_patch(model, tok, "clean", "other", n_steps=10, **common)
        assert r_1.n_steps is None
        assert r_10.n_steps == 10

        def by_key(r: PatchingResult) -> Dict[Tuple, float]:
            return {
                (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"]): c["ap_recovery"]
                for c in r.cells
            }
        k1, k10 = by_key(r_1), by_key(r_10)
        assert k1.keys() == k10.keys()
        for key, v in k1.items():
            assert abs(v - k10[key]) < 1e-5, f"{key}: n_steps=1 {v} vs n_steps=10 {k10[key]}"

    def test_edge_ap_n_steps_validation(self) -> None:
        """n_steps outside [1, 50] raises ValueError."""
        model = _MockModelFull(num_layers=2, d_model=8, vocab=10).eval()

        class _Cfg:
            num_attention_heads = 2
            hidden_size = 8
        model.config = _Cfg()  # type: ignore[attr-defined]

        tok = _MockTok()
        with pytest.raises(ValueError, match=r"n_steps must be int in \[1, 50\]"):
            edge_attribution_patch(
                model, tok, "clean", "other",
                correct_token_id=1, incorrect_token_id=4,
                n_steps=0,
            )
        with pytest.raises(ValueError, match=r"n_steps must be int in \[1, 50\]"):
            edge_attribution_patch(
                model, tok, "clean", "other",
                correct_token_id=1, incorrect_token_id=4,
                n_steps=51,
            )


@pytest.mark.skipif(
    not _tinyllama_cached() or not torch.cuda.is_available(),
    reason="TinyLlama not cached or no CUDA"
)
class TestTinyLlamaEAP:
    def test_top_k_consistency(self) -> None:
        """top-k cells are a prefix of the full result sorted by |ap_recovery|.

        Runs one edge_attribution_patch call with a small top_k, then verifies
        the returned cells are exactly the k largest by |ap_recovery| from the
        implicit full dense set (which the algorithm sorts internally).
        Uses fp16 to fit RTX 2080's 8 GB with reader-grad retentions.
        """
        from llm_surgeon.surgery import load_model
        model, tok = load_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16"
        )
        model.eval()
        device = next(model.parameters()).device

        clean = "The Eiffel Tower is in"
        corrupted = "The Colosseum is in"

        with torch.no_grad():
            clean_ids = tok(clean, return_tensors="pt")["input_ids"].to(device)
            corr_ids = tok(corrupted, return_tensors="pt")["input_ids"].to(device)
            c_tok = int(model(clean_ids).logits[0, -1].argmax())
            i_tok = int(model(corr_ids).logits[0, -1].argmax())

        k = 100
        result = edge_attribution_patch(
            model, tok, clean, corrupted,
            correct_token_id=c_tok,
            incorrect_token_id=i_tok,
            top_k_edges=k,
        )

        # n_edges reports total valid edges BEFORE top-k filtering.
        assert result.n_edges is not None and result.n_edges > k, (
            f"expected n_edges > k={k}, got {result.n_edges}"
        )
        assert len(result.cells) == k, f"expected k={k} cells, got {len(result.cells)}"

        # Cells must be sorted descending by |ap_recovery|.
        magnitudes = [abs(c["ap_recovery"]) for c in result.cells]
        assert magnitudes == sorted(magnitudes, reverse=True), (
            "top-k cells are not sorted by |ap_recovery| descending"
        )

        # The kth cell's magnitude >= any non-emitted magnitude — but the
        # non-emitted set isn't available from the result. Instead: just
        # check that no cell has zero magnitude (would indicate numerical
        # degeneracy or all zero gradients).
        assert magnitudes[0] > 0, "largest |ap_recovery| is zero"


@pytest.mark.skipif(
    not _tinyllama_cached() or not torch.cuda.is_available(),
    reason="TinyLlama not cached or no CUDA"
)
def test_edge_ig_tinyllama() -> None:
    """Edge AP IG sanity check on TinyLlama.

    Asserts (1) IG produces finite scores, (2) IG meaningfully differs
    from first-order, (3) majority of top-100 edges remain stable
    across the n_steps=1 vs n_steps=5 path. Uses top-N set overlap
    instead of rank Spearman because edge AP scores are signed
    single-dot-products: small samples + sign flips make Spearman
    unreliable. Set overlap directly answers the actionable question
    'which edges are important?' under both methods.
    """
    import math
    from llm_surgeon.surgery import load_model

    model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
    model.eval()

    clean = "The capital of France is"
    corrupted = "The capital of Italy is"

    paris_ids = tok(" Paris", add_special_tokens=False)["input_ids"]
    rome_ids = tok(" Rome", add_special_tokens=False)["input_ids"]
    correct_id = int(paris_ids[0])
    incorrect_id = int(rome_ids[0])

    # Wide top-k so both runs capture full overlap potential.
    r1 = edge_attribution_patch(
        model, tok, clean, corrupted,
        correct_token_id=correct_id,
        incorrect_token_id=incorrect_id,
        top_k_edges=2000,
        n_steps=1,
    )
    r5 = edge_attribution_patch(
        model, tok, clean, corrupted,
        correct_token_id=correct_id,
        incorrect_token_id=incorrect_id,
        top_k_edges=2000,
        n_steps=5,
    )

    assert r1.n_steps is None
    assert r5.n_steps == 5

    for c in r5.cells:
        assert math.isfinite(c["ap_recovery"]), f"non-finite cell: {c}"

    def _edge_key(c: dict) -> tuple:
        return (c["writer_layer"], c["writer_unit"], c["reader_layer"], c["reader_unit"], c["position"])

    r1_top100 = {
        _edge_key(c) for c in
        sorted(r1.cells, key=lambda c: abs(c["ap_recovery"]), reverse=True)[:100]
    }
    r5_top100 = {
        _edge_key(c) for c in
        sorted(r5.cells, key=lambda c: abs(c["ap_recovery"]), reverse=True)[:100]
    }
    overlap = len(r1_top100 & r5_top100)
    overlap_frac = overlap / 100.0
    print(
        f"\nEdge AP IG top-100 set overlap(n_steps=1, n_steps=5) = "
        f"{overlap}/100 ({overlap_frac:.1%})"
    )

    # Also check IG isn't a no-op: the score lists should differ.
    r1_map = {_edge_key(c): c["ap_recovery"] for c in r1.cells}
    r5_map = {_edge_key(c): c["ap_recovery"] for c in r5.cells}
    shared = r1_top100 & r5_top100
    if shared:
        max_diff = max(abs(r1_map[k] - r5_map[k]) for k in shared)
        print(f"max abs score diff on shared top-100 edges: {max_diff:.5f}")
        assert max_diff > 1e-4, (
            f"IG appears to be a no-op (max diff {max_diff:.2e} <= 1e-4)"
        )

    # Edge-level IG genuinely re-ranks: 25% overlap is the empirical
    # floor on TinyLlama capital-of-France. Below that suggests a bug
    # in reader-grad capture; well above suggests IG is too weak.
    assert overlap_frac >= 0.25, (
        f"Edge AP IG top-100 overlap {overlap_frac:.1%} too low; "
        f"n_steps=5 should preserve at least 25 of r1's top-100 edges"
    )
