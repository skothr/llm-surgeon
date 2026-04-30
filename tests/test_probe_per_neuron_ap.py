"""Unit tests for probe.attribution_patch_per_neuron (Phase 3.9).

Mock infrastructure is adapted from test_probe_circuit.py but extends
_MockLayer with a proper mlp.down_proj attribute so the per-neuron hook
can attach to it, and adds intermediate_size to the config.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import os

import pytest
import torch
from torch import nn

from llm_surgeon.probe import (
    PatchingResult,
    attribution_patch_per_neuron,
    _capture_residual_stream_with_grad,
)


# ---- Shared mock infrastructure (adapted from test_probe_circuit.py) ----

def _stable_word_hash(s: str) -> int:
    """Deterministic across runs (unlike Python's hash())."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) % (10 ** 9)
    return h


class _MockTokenizer:
    def __init__(self, vocab: List[str]) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def __call__(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        ids = [_stable_word_hash(w) % self.vocab_size for w in text.split()]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> List[str]:
        return [self.vocab[int(i) % self.vocab_size] for i in ids.flatten()]


class _MockMLP(nn.Module):
    """MLP with an explicit down_proj named attribute so the pre-hook can attach."""

    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.act_fn = nn.GELU()
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(h)))


class _MockLayerAttn(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        concat_z = torch.tanh(h)
        return self.o_proj(concat_z)


class _MockLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int, intermediate: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = _MockLayerAttn(hidden, n_heads)
        self.mlp = _MockMLP(hidden, intermediate)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_attn = h + self.self_attn(self.input_layernorm(h))
        h_ffn = h_attn + self.mlp(self.post_attention_layernorm(h_attn))
        return h_ffn


class _MockInner(nn.Module):
    def __init__(self, vocab: int, hidden: int, n_heads: int, n_layers: int, intermediate: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_MockLayer(hidden, n_heads, intermediate) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)


class _MockModel(nn.Module):
    def __init__(
        self,
        vocab: int = 16,
        hidden: int = 8,
        n_heads: int = 2,
        n_layers: int = 2,
        intermediate: int = 12,
    ) -> None:
        super().__init__()
        self.model = _MockInner(vocab, hidden, n_heads, n_layers, intermediate)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        self.config = SimpleNamespace(
            num_attention_heads=n_heads,
            hidden_size=hidden,
            num_hidden_layers=n_layers,
            intermediate_size=intermediate,
        )
        self._device = torch.device("cpu")

    def forward(self, input_ids: torch.Tensor) -> Any:
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.norm(h)
        logits = self.lm_head(h)

        class _Out:
            pass
        out = _Out()
        out.logits = logits  # pyright: ignore[reportAttributeAccessIssue]
        return out


def _make_mock() -> Tuple[_MockModel, _MockTokenizer]:
    torch.manual_seed(42)
    model = _MockModel()
    tokenizer = _MockTokenizer(vocab=["w" + str(i) for i in range(16)])
    return model, tokenizer


def _pick_tokens(
    model: "_MockModel",
    tok: "_MockTokenizer",
    clean: str,
    corrupted: str,
    meas_pos: int = -1,
) -> Tuple[int, int]:
    """Pick (correct_id, incorrect_id) guaranteed to yield a nonzero AP denominator."""
    with torch.no_grad():
        clean_out = model(tok(clean)["input_ids"])
        corr_out = model(tok(corrupted)["input_ids"])
    cl = clean_out.logits[0]
    co = corr_out.logits[0]
    seq = cl.shape[0]
    mp = meas_pos % seq
    c_id = int(cl[mp].argmax().item())
    i_id = int(co[mp].argmax().item())
    if c_id == i_id:
        topk = co[mp].topk(2).indices
        i_id = int(topk[1].item())
    d_clean = (cl[mp, c_id] - cl[mp, i_id]).item()
    d_corr = (co[mp, c_id] - co[mp, i_id]).item()
    if abs(d_clean - d_corr) < 1e-4:
        vocab = cl.shape[-1]
        best: Tuple[int, int, float] = (0, 1, 0.0)
        for ci in range(vocab):
            for ii in range(vocab):
                if ii == ci:
                    continue
                denom = abs((cl[mp, ci] - cl[mp, ii]).item() - (co[mp, ci] - co[mp, ii]).item())
                if denom > best[2]:
                    best = (ci, ii, denom)
        c_id, i_id = best[0], best[1]
    return c_id, i_id


# ---- Shared prompts + dynamically-picked token IDs (non-degenerate AP denominator) ----

CLEAN_PROMPT = "one two three four"
CORR_PROMPT = "one two three five"
_pick_model, _pick_tok = _make_mock()
CORRECT_ID, INCORRECT_ID = _pick_tokens(_pick_model, _pick_tok, CLEAN_PROMPT, CORR_PROMPT)


# ---- Tests ----

class TestPerNeuronMock:
    def test_returns_patching_result(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            top_k_neurons=20,
        )
        assert isinstance(result, PatchingResult)
        assert result.mode == "approx_neuron"
        assert result.n_neurons == model.config.intermediate_size
        assert len(result.cells) == 20

    def test_cells_have_required_fields(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            top_k_neurons=10,
        )
        for c in result.cells:
            assert "layer" in c and isinstance(c["layer"], int)
            assert "unit" in c and c["unit"] == f"neuron.n{c['neuron']}"
            assert "neuron" in c and isinstance(c["neuron"], int)
            assert "position" in c and isinstance(c["position"], int)
            assert "ap_recovery" in c and isinstance(c["ap_recovery"], float)
            assert 0 <= c["neuron"] < model.config.intermediate_size

    def test_cells_sorted_desc_by_abs(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            top_k_neurons=30,
        )
        mags = [abs(c["ap_recovery"]) for c in result.cells]
        assert mags == sorted(mags, reverse=True)

    def test_top_k_exceeds_total_caps(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            top_k_neurons=10**9,
        )
        # Total = n_layers * intermediate_size * seq_len
        expected = (
            model.config.num_hidden_layers
            * model.config.intermediate_size
            * len(tok(CLEAN_PROMPT)["input_ids"][0])
        )
        assert len(result.cells) == expected

    def test_sum_invariant_mock(self) -> None:
        """Σ_i ap_neuron_raw(L, i, pos) == (Δffn_out · grad_ffn_out)_pos.

        Computes the right-hand side directly from captures and compares
        against the left-hand side (reconstructed by summing per-neuron
        ap_raw from an undivided-by-D path).
        """
        model, tok = _make_mock()

        from_prompt = CLEAN_PROMPT
        base_prompt = CORR_PROMPT

        # Capture from pass (no grad)
        with torch.no_grad():
            from_captured, _, from_logits, _, _, _, from_ffn_acts = \
                _capture_residual_stream_with_grad(
                    model, tok, from_prompt,
                    sublayers=("attn", "ffn"),
                    capture_ffn_out=True,
                    capture_ffn_act=True,
                )

        # Capture base pass (with grad)
        with torch.enable_grad():
            base_captured, _, base_logits, _, _, _, base_ffn_acts = \
                _capture_residual_stream_with_grad(
                    model, tok, base_prompt,
                    sublayers=("attn", "ffn"),
                    capture_ffn_out=True,
                    capture_ffn_act=True,
                )
            meas_pos = base_logits.shape[0] - 1
            metric = (
                base_logits[meas_pos, CORRECT_ID]
                - base_logits[meas_pos, INCORRECT_ID]
            )
            metric.backward()

        # For each layer, at every position, verify the invariant.
        for L in range(model.config.num_hidden_layers):
            if (L, "ffn_out") not in base_captured:
                continue
            base_ffn_out = base_captured[(L, "ffn_out")]
            from_ffn_out_L = from_captured[(L, "ffn_out")]
            if base_ffn_out.grad is None:
                continue
            W_down: torch.Tensor = model.model.layers[L].mlp.down_proj.weight  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType]
            for pos in range(base_ffn_out.shape[1]):
                grad_ffn_out = base_ffn_out.grad[0, pos].detach()
                delta_ffn_out = (from_ffn_out_L[0, pos] - base_ffn_out[0, pos].detach())
                target = (delta_ffn_out * grad_ffn_out).sum().item()

                grad_act = grad_ffn_out @ W_down
                delta_act = from_ffn_acts[L][0, pos] - base_ffn_acts[L][0, pos].detach()
                reconstructed = (delta_act * grad_act).sum().item()

                assert abs(target - reconstructed) < 1e-4, \
                    f"Sum invariant broken at L={L}, pos={pos}: target={target}, sum={reconstructed}"

    def test_validation(self) -> None:
        model, tok = _make_mock()
        with pytest.raises(ValueError, match="top_k_neurons"):
            attribution_patch_per_neuron(
                model, tok, CLEAN_PROMPT, CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                top_k_neurons=0,
            )
        with pytest.raises(ValueError, match="prompts cannot be empty"):
            attribution_patch_per_neuron(
                model, tok, "", CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            )
        with pytest.raises(ValueError, match="same length"):
            attribution_patch_per_neuron(
                model, tok, "a b c", "d e",
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            )
        with pytest.raises(ValueError, match="direction"):
            attribution_patch_per_neuron(
                model, tok, CLEAN_PROMPT, CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                direction="nonsense",
            )


class TestCaptureFFNAct:
    def test_capture_ffn_act_flag_populates_dict(self) -> None:
        model, tok = _make_mock()
        with torch.no_grad():
            out = _capture_residual_stream_with_grad(
                model, tok, CLEAN_PROMPT,
                sublayers=("attn", "ffn"),
                capture_ffn_act=True,
            )
        assert len(out) == 7
        ffn_acts = out[6]
        assert isinstance(ffn_acts, dict)
        for L in range(model.config.num_hidden_layers):
            assert L in ffn_acts
            assert ffn_acts[L].shape[-1] == model.config.intermediate_size

    def test_capture_ffn_act_false_default_empty(self) -> None:
        model, tok = _make_mock()
        with torch.no_grad():
            out = _capture_residual_stream_with_grad(
                model, tok, CLEAN_PROMPT,
                sublayers=("attn", "ffn"),
                # capture_ffn_act defaults to False
            )
        ffn_acts = out[6]
        assert ffn_acts == {}


# -------------------------------------------------------------------------
# TinyLlama integration (skipif GPU missing; fp16 to avoid OOM on 8GB)
# -------------------------------------------------------------------------

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestTinyLlamaPerNeuron:
    def test_top_neurons_identifiable(self) -> None:
        from llm_surgeon.surgery import load_model
        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        clean = "The capital of France is"
        corrupted = "The capital of Italy is"
        paris_id = int(tok(" Paris", return_tensors="pt")["input_ids"][0, 1].item())
        rome_id = int(tok(" Rome", return_tensors="pt")["input_ids"][0, 1].item())

        r = attribution_patch_per_neuron(
            model, tok, clean, corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=rome_id,
            direction="denoise",
            top_k_neurons=50,
        )
        assert r.mode == "approx_neuron"
        assert r.n_neurons == 5632
        assert len(r.cells) == 50
        for c in r.cells:
            assert 0 <= c["neuron"] < 5632
            assert 0 <= c["layer"] < 22
        mags = [abs(c["ap_recovery"]) for c in r.cells]
        assert mags == sorted(mags, reverse=True)
        assert mags[0] > 0.001, "top neuron should have nonzero AP on a real task"


# ---------------------------------------------------------------------------
# Phase 3.10.1 — IG extension for per-neuron AP
# ---------------------------------------------------------------------------

class TestPerNeuronIG:
    def test_per_neuron_n_steps_converges(self) -> None:
        """n_steps=10 differs from n_steps=1 in at least one cell; n_steps stored correctly."""
        model, tok = _make_mock()

        def _run(n: int) -> PatchingResult:
            return attribution_patch_per_neuron(
                model,
                tok,
                clean_prompt=CLEAN_PROMPT,
                corrupted_prompt=CORR_PROMPT,
                correct_token_id=CORRECT_ID,
                incorrect_token_id=INCORRECT_ID,
                top_k_neurons=50,
                n_steps=n,
            )

        r1 = _run(1)
        r10 = _run(10)

        assert r1.n_steps is None, f"n_steps=1 should store None, got {r1.n_steps}"
        assert r10.n_steps == 10, f"n_steps=10 should store 10, got {r10.n_steps}"

        for c in r1.cells:
            val = c["ap_recovery"]
            assert isinstance(val, float) and val == val, f"Non-finite ap_recovery in r1: {c}"
        for c in r10.cells:
            val = c["ap_recovery"]
            assert isinstance(val, float) and val == val, f"Non-finite ap_recovery in r10: {c}"

        r1_map = {(c["layer"], c["neuron"], c["position"]): c["ap_recovery"] for c in r1.cells}
        r10_map = {(c["layer"], c["neuron"], c["position"]): c["ap_recovery"] for c in r10.cells}
        shared_keys = set(r1_map.keys()) & set(r10_map.keys())
        assert shared_keys, "No shared cells between n_steps=1 and n_steps=10"
        max_diff = max(abs(r1_map[k] - r10_map[k]) for k in shared_keys)
        assert max_diff > 1e-4, (
            f"n_steps=10 agrees with n_steps=1 within 1e-4 on all cells (max_diff={max_diff:.2e}); "
            "IG should differ from first-order on a nonlinear mock."
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
def test_per_neuron_ig_tinyllama() -> None:
    """Top-20 Spearman between n_steps=1 and n_steps=5 |ap_recovery| rankings > 0.5."""
    import math
    import numpy as np
    import scipy.stats
    from llm_surgeon.surgery import load_model

    model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
    model.eval()

    clean = "The capital of France is"
    corrupted = "The capital of Russia is"

    device = next(model.parameters()).device
    with torch.no_grad():
        paris_ids = tok(" Paris", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        moscow_ids = tok(" Moscow", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    paris_id = int(paris_ids[0, -1].item())
    moscow_id = int(moscow_ids[0, -1].item())

    def _run_neuron(n: int) -> PatchingResult:
        return attribution_patch_per_neuron(
            model,
            tok,
            clean_prompt=clean,
            corrupted_prompt=corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=moscow_id,
            direction="denoise",
            top_k_neurons=50,
            n_steps=n,
        )

    r1 = _run_neuron(1)
    r5 = _run_neuron(5)

    assert r1.n_steps is None
    assert r5.n_steps == 5

    r1_map = {(c["layer"], c["neuron"], c["position"]): c["ap_recovery"] for c in r1.cells}
    r5_map = {(c["layer"], c["neuron"], c["position"]): c["ap_recovery"] for c in r5.cells}

    shared = sorted(set(r1_map.keys()) & set(r5_map.keys()))
    assert len(shared) >= 1, f"no shared cells between n_steps=1 and n_steps=5: {len(shared)}"

    n_top = min(20, len(shared))
    top20 = sorted(shared, key=lambda k: abs(r1_map[k]), reverse=True)[:n_top]
    x = [abs(r1_map[k]) for k in top20]
    y = [abs(r5_map[k]) for k in top20]
    result_corr = scipy.stats.spearmanr(x, y)
    rho = float(result_corr.statistic)  # type: ignore[attr-defined]
    print(f"\nper-neuron IG TinyLlama: Spearman(n1, n5) top-{n_top} = {rho:.4f}")
    assert rho > 0.5, (
        f"Spearman ρ={rho:.4f} < 0.5 on top-{n_top} cells; IG per-neuron rankings diverged unexpectedly."
    )
    assert all(math.isfinite(c["ap_recovery"]) for c in r5.cells), "Non-finite ap_recovery in IG result"
