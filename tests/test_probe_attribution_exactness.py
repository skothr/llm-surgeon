"""Exactness checks for probe attribution patching on a real LLaMA graph.

The other probe test modules use hand-written mocks with MHA-shaped,
fp32, bias-free sublayers. These tests run the attribution functions on
tiny random ``LlamaForCausalLM`` models (RMSNorm, rotary attention,
SwiGLU MLP), including a GQA config whose ``head_dim`` differs from
``hidden_size // num_attention_heads``, and compare against quantities
computed independently with autograd or exact patching.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llm_surgeon.probe import (
    PatchingResult,
    _capture_residual_stream_with_grad,
    _integrated_gradients_loop,
    attribution_patch,
    attribution_patch_per_head,
    attribution_patch_per_neuron,
    edge_attribution_patch,
    extract_circuit,
)
from llm_surgeon.probe import _attribution

CLEAN = "t1 t2 t3 t4 t5"
CORRUPTED = "t1 t2 t9 t4 t5"
CORRECT_ID = 5
INCORRECT_ID = 11


class _WordTok:
    """Maps "tN" to token id N."""

    def __call__(self, text: str, return_tensors: str | None = None) -> dict:
        return {"input_ids": torch.tensor([[int(w[1:]) for w in text.split()]])}

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> list[str]:
        return [f"t{int(i)}" for i in ids.flatten()]


def _tiny_llama(*, gqa: bool) -> LlamaForCausalLM:
    """3-layer random LLaMA. gqa=True: 4 query heads, 2 kv heads, head_dim=16
    so o_proj is [32, 64] rather than square."""
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,  # pyright: ignore[reportCallIssue]
        hidden_size=32,  # pyright: ignore[reportCallIssue]
        intermediate_size=48,  # pyright: ignore[reportCallIssue]
        num_hidden_layers=3,  # pyright: ignore[reportCallIssue]
        num_attention_heads=4,  # pyright: ignore[reportCallIssue]
        num_key_value_heads=2 if gqa else 4,  # pyright: ignore[reportCallIssue]
        head_dim=16 if gqa else 8,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=64,  # pyright: ignore[reportCallIssue]
        initializer_range=0.2,  # pyright: ignore[reportCallIssue]
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(params=[False, True], ids=["mha", "gqa"])
def llama(request: pytest.FixtureRequest) -> LlamaForCausalLM:
    return _tiny_llama(gqa=request.param)


def _metric(logits: torch.Tensor) -> torch.Tensor:
    return logits[-1, CORRECT_ID] - logits[-1, INCORRECT_ID]


def _denominator(model) -> float:
    tok = _WordTok()
    with torch.no_grad():
        clean = model(tok(CLEAN)["input_ids"]).logits[0]
        corr = model(tok(CORRUPTED)["input_ids"]).logits[0]
    return (_metric(clean) - _metric(corr)).item()


def _captures(model, prompt: str, **kwargs):
    return _capture_residual_stream_with_grad(model, _WordTok(), prompt, **kwargs)


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------


class TestIntegratedGradients:
    @pytest.mark.parametrize("site", [(0, "attn"), (0, "ffn"), (1, "attn")])
    def test_single_site_ig_matches_exact_patch(self, llama, site) -> None:
        """IG along a single sublayer's path integrates to the exact effect of
        patching that sublayer: Σ Δ·avg_grad == metric(patched) - metric(base).
        Holds only if the gradient flows through the downstream sublayers."""
        L, sub = site
        cap_key = (L, "attn") if sub == "attn" else (L, "ffn_out")
        with torch.no_grad():
            from_cap = _captures(llama, CLEAN, capture_ffn_out=True)[0]
            base_cap, _, base_logits, *_ = _captures(
                llama, CORRUPTED, capture_ffn_out=True
            )
        base_val, from_val = base_cap[cap_key], from_cap[cap_key]

        avg_grad, _ = _integrated_gradients_loop(
            model=llama,
            input_ids=_WordTok()(CORRUPTED)["input_ids"],
            base_components={site: base_val},
            from_components={site: from_val},
            measurement_position=4,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            n_steps=40,
        )
        ig_total = ((from_val - base_val) * avg_grad[site]).sum().item()

        module = (
            llama.model.layers[L].self_attn
            if sub == "attn"
            else llama.model.layers[L].mlp
        )

        def patch(_mod, _inp, out):
            return (from_val,) + tuple(out[1:]) if isinstance(out, tuple) else from_val

        handle = module.register_forward_hook(patch)
        try:
            with torch.no_grad():
                patched_logits = llama(_WordTok()(CORRUPTED)["input_ids"]).logits[0]
        finally:
            handle.remove()
        exact = (_metric(patched_logits) - _metric(base_logits)).item()

        assert abs(exact) > 1e-3, "degenerate fixture: the patch has no effect"
        assert ig_total == pytest.approx(exact, rel=1e-3, abs=1e-5)

    def test_ig_equals_plain_gradient_when_from_equals_base(self, llama) -> None:
        """With from == base the path is a single point, so every IG average
        must equal the plain gradient at the base forward, for every site."""
        n_layers = len(llama.model.layers)
        with torch.enable_grad():
            cap, _, logits, _, _, readers, _ = _captures(
                llama,
                CORRUPTED,
                capture_ffn_out=True,
                capture_reader_grads=True,
            )
            keys = [
                (L, s) for L in range(n_layers) for s in ("attn", "ffn_out")
            ] + list(readers)
            tensors = [cap[k] if k in cap else readers[k] for k in keys]
            plain = dict(zip(keys, torch.autograd.grad(_metric(logits), tensors)))

        components = {(L, "attn"): cap[(L, "attn")].detach() for L in range(n_layers)}
        components.update(
            {(L, "ffn"): cap[(L, "ffn_out")].detach() for L in range(n_layers)}
        )
        avg_grad, avg_reader = _integrated_gradients_loop(
            model=llama,
            input_ids=_WordTok()(CORRUPTED)["input_ids"],
            base_components=components,
            from_components=components,
            measurement_position=4,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            n_steps=3,
            capture_reader_grads=True,
        )
        for L in range(n_layers):
            torch.testing.assert_close(
                avg_grad[(L, "attn")], plain[(L, "attn")], rtol=1e-4, atol=1e-6
            )
            torch.testing.assert_close(
                avg_grad[(L, "ffn")], plain[(L, "ffn_out")], rtol=1e-4, atol=1e-6
            )
        assert set(avg_reader) == set(readers)
        for k in readers:
            torch.testing.assert_close(avg_reader[k], plain[k], rtol=1e-4, atol=1e-6)

    def test_hooks_removed_when_registration_fails(self, llama, monkeypatch) -> None:
        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(_attribution, "_attach_reader_grad_hooks", boom)
        components = {(0, "attn"): torch.zeros(1, 5, 32)}
        with pytest.raises(RuntimeError, match="boom"):
            _integrated_gradients_loop(
                model=llama,
                input_ids=_WordTok()(CORRUPTED)["input_ids"],
                base_components=components,
                from_components=components,
                measurement_position=4,
                correct_token_id=CORRECT_ID,
                incorrect_token_id=INCORRECT_ID,
                n_steps=1,
                capture_reader_grads=True,
            )
        assert not llama.model.layers[0].self_attn._forward_hooks


# ---------------------------------------------------------------------------
# Per-head math with GQA and an explicit head_dim
# ---------------------------------------------------------------------------


def _attn_out_grads(model) -> tuple[dict, dict, dict]:
    n_layers = len(model.model.layers)
    with torch.no_grad():
        from_cap = _captures(model, CLEAN)[0]
    with torch.enable_grad():
        base_cap, _, logits, _, _, readers, _ = _captures(
            model, CORRUPTED, capture_reader_grads=True
        )
        keys = [(L, "attn") for L in range(n_layers)] + list(readers)
        tensors = [base_cap[k] if k in base_cap else readers[k] for k in keys]
        grads = dict(zip(keys, torch.autograd.grad(_metric(logits), tensors)))
    return from_cap, base_cap, grads


class TestPerHeadShapes:
    def test_per_head_sums_to_attn_out_score(self, llama) -> None:
        """Σ_h head effect · D == Δattn_out · grad_attn_out at every (L, pos)."""
        result = attribution_patch_per_head(
            llama,
            _WordTok(),
            CLEAN,
            CORRUPTED,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
        )
        assert result.n_heads == 4
        denom = _denominator(llama)
        from_cap, base_cap, grads = _attn_out_grads(llama)

        sums: dict[tuple[int, int], float] = defaultdict(float)
        for c in result.cells:
            if c["unit"].startswith("attn.h"):
                sums[(c["layer"], c["position"])] += c["ap_recovery"] * denom
        assert len(sums) == 3 * 5
        for (L, pos), total in sums.items():
            delta = (
                from_cap[(L, "attn")][0, pos] - base_cap[(L, "attn")][0, pos].detach()
            )
            target = (delta * grads[(L, "attn")][0, pos]).sum().item()
            assert total == pytest.approx(target, rel=1e-4, abs=1e-6), (L, pos)

    def test_edge_heads_sum_to_attn_out_score(self, llama) -> None:
        """Σ_h edge effect (L attn.hN → r) · D == Δattn_out(L) · grad_r."""
        result = edge_attribution_patch(
            llama,
            _WordTok(),
            CLEAN,
            CORRUPTED,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            top_k_edges=100_000,
        )
        denom = _denominator(llama)
        from_cap, base_cap, grads = _attn_out_grads(llama)

        sums: dict[tuple, float] = defaultdict(float)
        for c in result.cells:
            if c["writer_unit"].startswith("attn.h"):
                sums[
                    (
                        c["writer_layer"],
                        c["reader_unit"],
                        c["reader_layer"],
                        c["position"],
                    )
                ] += c["ap_recovery"] * denom
        assert sums
        for (L_w, ru, rl, pos), total in sums.items():
            delta = (
                from_cap[(L_w, "attn")][0, pos]
                - base_cap[(L_w, "attn")][0, pos].detach()
            )
            target = (delta * grads[(ru, rl)][0, pos]).sum().item()
            assert total == pytest.approx(target, rel=1e-4, abs=1e-6), (
                L_w,
                ru,
                rl,
                pos,
            )


# ---------------------------------------------------------------------------
# Side effects, validation, unsupported models
# ---------------------------------------------------------------------------

_ENTRY_POINTS: dict[str, Callable[..., PatchingResult]] = {
    "cell": attribution_patch,
    "head": attribution_patch_per_head,
    "neuron": attribution_patch_per_neuron,
    "edge": edge_attribution_patch,
    "circuit": extract_circuit,
}


def _call(name: str, model, **kwargs) -> PatchingResult:
    return _ENTRY_POINTS[name](
        model,
        _WordTok(),
        CLEAN,
        CORRUPTED,
        correct_token_id=CORRECT_ID,
        incorrect_token_id=INCORRECT_ID,
        **kwargs,
    )


@pytest.mark.parametrize("name", list(_ENTRY_POINTS))
class TestEntryPoints:
    @pytest.mark.parametrize("n_steps", [1, 2])
    def test_no_parameter_grads_accumulate(self, name: str, n_steps: int) -> None:
        model = _tiny_llama(gqa=True)
        assert all(p.requires_grad for p in model.parameters())
        _call(name, model, n_steps=n_steps)
        leaked = [n for n, p in model.named_parameters() if p.grad is not None]
        assert leaked == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"positions": [-6]},
            {"positions": [5]},
            {"measurement_position": 5},
            {"measurement_position": -6},
            {"layers": [3]},
            {"layers": [-4]},
        ],
    )
    def test_out_of_range_indices_raise(self, name: str, kwargs: dict) -> None:
        with pytest.raises(IndexError):
            _call(name, _tiny_llama(gqa=False), **kwargs)

    def test_negative_layer_normalized(self, name: str) -> None:
        model = _tiny_llama(gqa=False)
        neg = _call(name, model, layers=[-1])
        pos = _call(name, model, layers=[2])
        assert neg.cells == pos.cells
        layer_keys = {"layer", "writer_layer", "reader_layer"}
        seen = {
            c[k]
            for c in neg.cells
            for k in layer_keys & c.keys()
            if c.get("reader_unit") != "logits"
        }
        assert seen <= {0, 2}  # 0 is the embed writer's placeholder layer


@pytest.mark.parametrize("name", ["head", "neuron", "edge", "circuit"])
def test_quantized_model_rejected(name: str) -> None:
    model = _tiny_llama(gqa=False)
    setattr(model, "hf_quantizer", object())
    with pytest.raises(ValueError, match="quantized"):
        _call(name, model)


def test_quantized_model_warns_for_per_cell() -> None:
    model = _tiny_llama(gqa=False)
    setattr(model, "hf_quantizer", object())
    with pytest.warns(UserWarning, match="quantized"):
        _call("cell", model)


@pytest.mark.parametrize("name", ["head", "neuron", "edge"])
def test_meta_device_weights_rejected(name: str) -> None:
    model = _tiny_llama(gqa=False)
    layer = model.model.layers[0]
    assert isinstance(layer, LlamaDecoderLayer)
    layer.mlp.down_proj.to("meta")
    with pytest.raises(ValueError, match="offloaded"):
        _call(name, model)


# ---------------------------------------------------------------------------
# Noise direction ranks by effect size, not by 1 + effect
# ---------------------------------------------------------------------------


class TestNoiseRanking:
    def test_per_neuron_noise_top_k_is_largest_effect(self) -> None:
        model = _tiny_llama(gqa=False)
        full = _call("neuron", model, direction="noise", top_k_neurons=10**6)
        top = _call("neuron", model, direction="noise", top_k_neurons=10)
        want = sorted((abs(c["ap_recovery"] - 1.0) for c in full.cells), reverse=True)[
            :10
        ]
        got = [abs(c["ap_recovery"] - 1.0) for c in top.cells]
        assert got == pytest.approx(want, abs=1e-7)
        for c in top.cells:
            assert c["ap_effect"] == pytest.approx(c["ap_recovery"] - 1.0, abs=1e-6)

    def test_edge_noise_top_k_is_largest_effect(self) -> None:
        model = _tiny_llama(gqa=False)
        full = _call("edge", model, direction="noise", top_k_edges=10**6)
        top = _call("edge", model, direction="noise", top_k_edges=10)
        want = sorted((abs(c["ap_recovery"] - 1.0) for c in full.cells), reverse=True)[
            :10
        ]
        got = [abs(c["ap_recovery"] - 1.0) for c in top.cells]
        assert got == pytest.approx(want, abs=1e-7)

    def test_circuit_noise_tau_filters_on_effect(self) -> None:
        model = _tiny_llama(gqa=False)
        full = _call(
            "circuit", model, direction="noise", tau=0.0, top_k_candidates=10**6
        )
        # Positions before the corrupted token have zero effect; take the median
        # of the nonzero effects so tau splits the edges that matter.
        effects = sorted(e for c in full.cells if (e := abs(c["ap_recovery"] - 1.0)) > 1e-6)
        tau = effects[len(effects) // 2]
        result = _call(
            "circuit", model, direction="noise", tau=tau, top_k_candidates=10**6
        )
        below = [c for c in result.cells if abs(c["ap_recovery"] - 1.0) < tau]
        assert below, "fixture must have edges under tau"
        assert not any(c["in_circuit"] for c in below)
        assert any(c["in_circuit"] for c in result.cells)
