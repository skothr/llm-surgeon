"""Regression tests for the probe core (intervention, capture, logit lens, NLA).

Covers the issue #8 audit findings. Besides the shared MHA fp32 fixture, these
tests use a GQA model (num_key_value_heads < num_attention_heads, head_dim !=
hidden_size / num_attention_heads) and a bf16 copy, because several bugs only
show outside the MHA fp32 configuration.
"""

from __future__ import annotations

import copy
import warnings

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon.probe import (
    HiddenStates,
    Intervention,
    activation_patch,
    compare_logit_lens,
    extract_hidden_states,
    intervene,
    logit_lens,
    ops,
)
from llm_surgeon.probe._capture import (
    _capture_residual_stream,
    _capture_residual_stream_with_grad,
)
from tests.conftest import _make_tiny_tokenizer

PROMPT_A = "word10 word11 word12 word13"
PROMPT_B = "word20 word21 word22 word23"


@pytest.fixture
def tokenizer():
    return _make_tiny_tokenizer(64)


@pytest.fixture
def gqa_llama():
    """4-layer GQA LLaMA: 4 query heads share 2 KV heads, head_dim=16 != 32/4."""
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64,  # pyright: ignore[reportCallIssue]
        hidden_size=32,  # pyright: ignore[reportCallIssue]
        intermediate_size=64,  # pyright: ignore[reportCallIssue]
        num_hidden_layers=4,  # pyright: ignore[reportCallIssue]
        num_attention_heads=4,  # pyright: ignore[reportCallIssue]
        num_key_value_heads=2,  # pyright: ignore[reportCallIssue]
        head_dim=16,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=128,  # pyright: ignore[reportCallIssue]
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture(params=["mha", "gqa"])
def any_llama(request, tiny_llama, gqa_llama):
    return tiny_llama if request.param == "mha" else gqa_llama


def _baseline_logits(model, tokenizer, prompt: str) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return model(ids).logits[0]


# ---------------------------------------------------------------------------
# ops.noise
# ---------------------------------------------------------------------------


class TestNoise:
    def test_unseeded_noise_differs_between_calls(self):
        h = torch.zeros(4, 8)
        op = ops.noise(1.0)
        assert not torch.equal(op(h, 0), op(h, 0))

    def test_unseeded_noise_differs_between_ops(self):
        h = torch.zeros(4, 8)
        assert not torch.equal(ops.noise(1.0)(h, 0), ops.noise(1.0)(h, 0))

    def test_seeded_noise_is_reproducible(self):
        h = torch.zeros(4, 8)
        assert torch.equal(ops.noise(1.0, seed=7)(h, 0), ops.noise(1.0, seed=7)(h, 0))


# ---------------------------------------------------------------------------
# intervene: validation, negative layers, duplicates, attn path
# ---------------------------------------------------------------------------


class TestInterveneValidation:
    def test_out_of_range_layer_raises(self, tiny_llama, tokenizer):
        n = len(tiny_llama.model.layers)
        with pytest.raises(IndexError, match="layer"):
            intervene(
                tiny_llama,
                tokenizer,
                PROMPT_A,
                [Intervention(layer=n, sublayer="ffn", fn=ops.scale(0.0))],
            )

    def test_unknown_sublayer_raises(self, tiny_llama, tokenizer):
        with pytest.raises(ValueError, match="sublayer"):
            intervene(
                tiny_llama,
                tokenizer,
                PROMPT_A,
                [Intervention(layer=0, sublayer="mlp", fn=ops.scale(0.0))],
            )

    def test_negative_layer_targets_from_the_end(self, tiny_llama, tokenizer):
        n = len(tiny_llama.model.layers)
        neg = intervene(
            tiny_llama,
            tokenizer,
            PROMPT_A,
            [Intervention(layer=-1, sublayer="ffn", fn=ops.scale(0.5))],
        )
        pos = intervene(
            tiny_llama,
            tokenizer,
            PROMPT_A,
            [Intervention(layer=n - 1, sublayer="ffn", fn=ops.scale(0.5))],
        )
        base = _baseline_logits(tiny_llama, tokenizer, PROMPT_A)
        assert not torch.allclose(neg.output_logits, base)
        assert torch.equal(neg.output_logits, pos.output_logits)
        assert neg.interventions_applied[0]["layer"] == n - 1

    def test_duplicate_key_composes_in_order(self, tiny_llama, tokenizer):
        both = intervene(
            tiny_llama,
            tokenizer,
            PROMPT_A,
            [
                Intervention(layer=3, sublayer="ffn", fn=ops.zero_dims([0])),
                Intervention(layer=3, sublayer="ffn", fn=ops.scale(2.0)),
            ],
        )
        only_scale = intervene(
            tiny_llama,
            tokenizer,
            PROMPT_A,
            [
                Intervention(layer=3, sublayer="ffn", fn=ops.scale(2.0)),
            ],
        )

        def composed(h, i):
            return ops.scale(2.0)(ops.zero_dims([0])(h, i), i)

        expected = intervene(
            tiny_llama,
            tokenizer,
            PROMPT_A,
            [
                Intervention(layer=3, sublayer="ffn", fn=composed),
            ],
        )
        assert torch.allclose(both.output_logits, expected.output_logits)
        assert not torch.allclose(both.output_logits, only_scale.output_logits)


class TestInterveneSublayers:
    @pytest.mark.parametrize("sublayer", ["attn", "ffn"])
    def test_identity_op_matches_baseline(self, any_llama, tokenizer, sublayer):
        res = intervene(
            any_llama,
            tokenizer,
            PROMPT_A,
            [Intervention(layer=1, sublayer=sublayer, fn=ops.scale(1.0))],
        )
        base = _baseline_logits(any_llama, tokenizer, PROMPT_A)
        assert torch.allclose(res.output_logits, base, atol=1e-5)

    def test_attn_identity_is_exact_in_bf16(self, gqa_llama, tokenizer):
        """The attn write-back returns ``state - h_in`` and the layer re-adds
        ``h_in``. Guard that an identity op leaves a bf16 run bit-identical
        even when |h_in| >> |attn_out| (embeddings scaled 100x; RMSNorm keeps
        the attention output at its usual size)."""
        model = copy.deepcopy(gqa_llama)
        with torch.no_grad():
            model.model.embed_tokens.weight.mul_(100.0)
        model = model.to(torch.bfloat16)
        for layer in range(len(model.model.layers)):
            res = intervene(
                model,
                tokenizer,
                PROMPT_A,
                [Intervention(layer=layer, sublayer="attn", fn=ops.scale(1.0))],
            )
            base = _baseline_logits(model, tokenizer, PROMPT_A)
            assert torch.equal(res.output_logits, base), f"layer {layer} perturbed"

    @pytest.mark.parametrize("sublayer", ["attn", "ffn"])
    def test_replacing_full_state_reproduces_other_prompt(
        self, any_llama, tokenizer, sublayer
    ):
        """Every later layer reads only the residual stream, so replacing it at
        (L, sub) with prompt A's state makes prompt B's run produce A's logits."""
        layer = 1
        captured, _ = _capture_residual_stream(
            any_llama,
            tokenizer,
            PROMPT_A,
            sublayers=(sublayer,),
            layers=[layer],
        )
        res = intervene(
            any_llama,
            tokenizer,
            PROMPT_B,
            [
                Intervention(
                    layer=layer,
                    sublayer=sublayer,
                    fn=ops.replace(captured[(layer, sublayer)]),
                ),
            ],
        )
        target = _baseline_logits(any_llama, tokenizer, PROMPT_A)
        assert torch.allclose(res.output_logits, target, atol=1e-4)


# ---------------------------------------------------------------------------
# Negative layer indices in capture paths
# ---------------------------------------------------------------------------


class TestNegativeLayers:
    def test_extract_hidden_states_normalizes_negative_layers(
        self, tiny_llama, tokenizer
    ):
        n = len(tiny_llama.model.layers)
        hs = extract_hidden_states(tiny_llama, tokenizer, PROMPT_A, layers=[-1])
        assert set(hs.states) == {(n - 1, "ffn")}

    def test_capture_out_of_range_layer_raises(self, tiny_llama, tokenizer):
        with pytest.raises(IndexError):
            _capture_residual_stream(tiny_llama, tokenizer, PROMPT_A, layers=[99])

    def test_capture_unknown_sublayer_raises(self, tiny_llama, tokenizer):
        with pytest.raises(ValueError, match="sublayer"):
            _capture_residual_stream(
                tiny_llama, tokenizer, PROMPT_A, sublayers=("mlp",)
            )

    def test_capture_with_grad_normalizes_negative_layers(self, tiny_llama, tokenizer):
        n = len(tiny_llama.model.layers)
        with torch.enable_grad():
            captured, *_ = _capture_residual_stream_with_grad(
                tiny_llama,
                tokenizer,
                PROMPT_A,
                sublayers=("ffn",),
                layers=[-1],
            )
        assert set(captured) == {(n - 1, "ffn")}

    def test_activation_patch_negative_layer_patches(self, any_llama, tokenizer):
        """Denoise-patching the last layer's output at the measurement position
        must reproduce the clean logits there."""
        n = len(any_llama.model.layers)
        result = activation_patch(
            any_llama,
            tokenizer,
            PROMPT_A,
            PROMPT_B,
            direction="denoise",
            layers=[-1],
            sublayers=("ffn",),
            positions=[3],
        )
        assert [c["layer"] for c in result.cells] == [n - 1]
        assert torch.allclose(
            result.cells[0]["patched_logits"], result.clean_baseline_logits, atol=1e-4
        )
        assert not torch.allclose(
            result.clean_baseline_logits, result.corrupted_baseline_logits, atol=1e-4
        )


# ---------------------------------------------------------------------------
# Hidden-grad capture: narrow embedding fallback
# ---------------------------------------------------------------------------


def test_capture_with_grad_does_not_swallow_embedding_errors(
    tiny_llama, tokenizer, monkeypatch
):
    class _Broken(torch.nn.Module):
        def forward(self, _ids):
            raise AttributeError("broken embedding internals")

    monkeypatch.setattr(tiny_llama, "get_input_embeddings", lambda: _Broken())
    with torch.enable_grad(), pytest.raises(AttributeError, match="broken embedding"):
        _capture_residual_stream_with_grad(tiny_llama, tokenizer, PROMPT_A)


# ---------------------------------------------------------------------------
# logit_lens positions and LogitLensResult methods
# ---------------------------------------------------------------------------


class TestLogitLensPositions:
    def test_out_of_range_position_raises(self, tiny_llama, tokenizer):
        with pytest.raises(IndexError, match="position"):
            logit_lens(tiny_llama, tokenizer, PROMPT_A, positions=[100])

    def test_negative_positions_resolve_in_result_methods(self, tiny_llama, tokenizer):
        res = logit_lens(tiny_llama, tokenizer, PROMPT_A, top_k=3)
        last = len(res.prompt_tokens) - 1
        top = next(p for p in res.predictions if p["position"] == last)["top_k"][0]
        assert res.prediction_flips(-1) == res.prediction_flips(last)
        assert res.first_correct_layer(-1, top["token"]) == res.first_correct_layer(
            last, top["token"]
        )
        assert res.first_correct_layer(-1, top["token"]) is not None
        assert res.summary(-2) == res.summary(last - 1)
        assert len(res.summary(-2).splitlines()) > 2

    def test_first_correct_layer_accepts_token_id(self, tiny_llama, tokenizer):
        res = logit_lens(tiny_llama, tokenizer, PROMPT_A, top_k=3)
        cell = res.predictions[0]
        tid = cell["top_k"][0]["token_id"]
        assert res.first_correct_layer(cell["position"], tid) == cell["layer"]

    def test_sublayer_filter(self):
        from llm_surgeon.probe import LogitLensResult

        def row(layer, sub, tid):
            return {
                "layer": layer,
                "sublayer": sub,
                "position": 0,
                "top_k": [
                    {"token": f"t{tid}", "token_id": tid, "prob": 1.0, "rank": 0}
                ],
                "metrics": {},
            }

        res = LogitLensResult(
            predictions=[
                row(0, "attn", 1),
                row(0, "ffn", 2),
                row(1, "attn", 1),
                row(1, "ffn", 2),
            ],
            logits=None,
            prompt_tokens=["x"],
        )
        assert res.prediction_flips(0) == 3
        assert res.prediction_flips(0, sublayer="ffn") == 0
        assert res.first_correct_layer(0, 1, sublayer="ffn") is None
        assert res.first_correct_layer(0, 2, sublayer="ffn") == 0

    def test_prediction_flips_compares_token_ids(self):
        from llm_surgeon.probe import LogitLensResult

        def row(layer, tid):
            # Two distinct ids that decode to the same string.
            return {
                "layer": layer,
                "sublayer": "ffn",
                "position": 0,
                "top_k": [{"token": "Paris", "token_id": tid, "prob": 1.0, "rank": 0}],
                "metrics": {},
            }

        res = LogitLensResult(
            predictions=[row(0, 5), row(1, 6)], logits=None, prompt_tokens=["x"]
        )
        assert res.prediction_flips(0) == 1


def test_extract_hidden_states_embed_fires_first(tiny_llama, tokenizer):
    order = []
    extract_hidden_states(
        tiny_llama,
        tokenizer,
        PROMPT_A,
        sublayers=("embed", "attn", "ffn"),
        on_layer=lambda L, sub, _d: order.append((L, sub)),
    )
    assert order[:3] == [(0, "embed"), (0, "attn"), (0, "ffn")]


def test_extract_hidden_states_rejects_detach_false(tiny_llama, tokenizer):
    with pytest.raises(ValueError, match="detach"):
        extract_hidden_states(tiny_llama, tokenizer, PROMPT_A, detach=False)


def test_hidden_states_load_uses_weights_only(tmp_path, monkeypatch):
    hs = HiddenStates(states={(0, "ffn"): torch.ones(2, 3)}, prompt_tokens=["a", "b"])
    path = str(tmp_path / "hs.pt")
    hs.save(path)
    seen = {}
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    loaded = HiddenStates.load(path)
    assert seen.get("weights_only") is True
    assert torch.equal(loaded.states[(0, "ffn")], torch.ones(2, 3))
    assert loaded.prompt_tokens == ["a", "b"]


# ---------------------------------------------------------------------------
# compare_logit_lens layer maps and devices
# ---------------------------------------------------------------------------


class TestCompareLayerMaps:
    def test_pairs_the_mapped_layers(self, tiny_llama, tokenizer):
        """With B's map reversed, original layer k must pair A's layer k with
        B's compressed layer n-1-k (same model, so B's state is A's n-1-k)."""
        n = len(tiny_llama.model.layers)
        hs = extract_hidden_states(tiny_llama, tokenizer, PROMPT_A, sublayers=("ffn",))
        seen = {}

        def cb(orig, sub, data):
            if sub == "ffn":
                seen[orig] = data["hidden_state_b"]

        compare_logit_lens(
            tiny_llama,
            tiny_llama,
            tokenizer,
            PROMPT_A,
            layer_map_b=list(reversed(range(n))),
            on_layer=cb,
        )
        assert set(seen) == set(range(n))
        for k in range(n):
            assert torch.equal(seen[k], hs.states[(n - 1 - k, "ffn")])

    def test_short_layer_map_raises(self, tiny_llama, tokenizer):
        n = len(tiny_llama.model.layers)
        with pytest.raises(ValueError, match="layer_map_b"):
            compare_logit_lens(
                tiny_llama,
                tiny_llama,
                tokenizer,
                PROMPT_A,
                layer_map_b=list(range(n - 1)),
            )

    def test_duplicate_layer_map_warns_and_keeps_last_copy(self, tiny_llama, tokenizer):
        n = len(tiny_llama.model.layers)
        dup_map = [0, 1, 2, 2] + list(
            range(3, n - 1)
        )  # compressed 2 and 3 both map to 2
        assert len(dup_map) == n
        hs = extract_hidden_states(tiny_llama, tokenizer, PROMPT_A, sublayers=("ffn",))
        seen = {}

        def cb(orig, sub, data):
            if sub == "ffn":
                seen[orig] = data["hidden_state_b"]

        with pytest.warns(RuntimeWarning, match="duplicate"):
            compare_logit_lens(
                tiny_llama,
                tiny_llama,
                tokenizer,
                PROMPT_A,
                layer_map_b=dup_map,
                on_layer=cb,
            )
        assert torch.equal(seen[2], hs.states[(3, "ffn")])

    def test_identity_maps_do_not_warn(self, tiny_llama, tokenizer):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compare_logit_lens(tiny_llama, tiny_llama, tokenizer, PROMPT_A)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second device (CUDA)"
)
def test_compare_logit_lens_across_devices(tiny_llama, tokenizer):
    model_b = copy.deepcopy(tiny_llama).to("cuda")
    result = compare_logit_lens(tiny_llama, model_b, tokenizer, PROMPT_A)
    assert result.comparisons


# ---------------------------------------------------------------------------
# GQA smoke coverage for the logit-lens / patch paths
# ---------------------------------------------------------------------------


def test_gqa_logit_lens_and_patch_shapes(gqa_llama, tokenizer):
    n = len(gqa_llama.model.layers)
    ll = logit_lens(gqa_llama, tokenizer, PROMPT_A, top_k=2, positions=[-1])
    assert len(ll.predictions) == n * 2
    ap = activation_patch(gqa_llama, tokenizer, PROMPT_A, PROMPT_B, positions=[0, 3])
    assert len(ap.cells) == n * 2 * 2


# ---------------------------------------------------------------------------
# NLA helpers (no downloads)
# ---------------------------------------------------------------------------


class TestNlaScore:
    def test_identical_vectors(self):
        from llm_surgeon.probe import nla_score

        h = torch.randn(16)
        s = nla_score(h, h.clone())
        assert s["cosine"] == pytest.approx(1.0, abs=1e-6)
        assert s["normalized_mse"] == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        from llm_surgeon.probe import nla_score

        a = torch.zeros(16)
        b = torch.zeros(16)
        a[0] = 3.0
        b[1] = 5.0
        s = nla_score(a, b)
        assert s["cosine"] == pytest.approx(0.0, abs=1e-6)
        assert s["normalized_mse"] == pytest.approx(2.0, abs=1e-5)

    def test_opposite_vectors(self):
        from llm_surgeon.probe import nla_score

        h = torch.randn(16)
        s = nla_score(h, -2.0 * h)
        assert s["cosine"] == pytest.approx(-1.0, abs=1e-6)
        assert s["normalized_mse"] == pytest.approx(4.0, abs=1e-5)


def _nla_meta(d: int = 4) -> dict:
    return {
        "d_model": d,
        "prompt_templates": {"av": "x {injection_char} y"},
        "tokens": {
            "injection_char": "@",
            "injection_token_id": 9,
            "injection_left_neighbor_id": 8,
            "injection_right_neighbor_id": 10,
        },
        "extraction": {"injection_scale": 1.0},
    }


class _StubTok:
    def __init__(self, ids: list[int]):
        self._ids = ids

    def apply_chat_template(self, *_a, **_k):
        ids = torch.tensor([self._ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TestNlaVerbalizeGuards:
    def test_zero_activation_raises(self):
        from llm_surgeon.probe import nla_verbalize

        with pytest.raises(ValueError, match="zero"):
            nla_verbalize(
                torch.zeros(4), model=None, tok=_StubTok([8, 9, 10]), meta=_nla_meta()
            )

    @pytest.mark.parametrize("ids", [[9, 10, 8], [10, 8, 9]])
    def test_injection_at_sequence_edge_raises(self, ids):
        from llm_surgeon.probe import nla_verbalize

        with pytest.raises(RuntimeError, match="injection"):
            nla_verbalize(
                torch.ones(4), model=None, tok=_StubTok(ids), meta=_nla_meta()
            )
