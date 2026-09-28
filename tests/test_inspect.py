"""Tests for inspect module."""

import gc

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon.inspect import (
    _compute_metrics,
    inspect_head,
    block_influence,
    magnitude_influence,
    sublayer_influence,
    weight_norms,
    weight_svd,
    attention_entropy,
    residual_stream_norms,
)
from tests.conftest import _make_tiny_tokenizer


@pytest.fixture
def tokenizer():
    return _make_tiny_tokenizer(64)


@pytest.fixture
def gqa_llama():
    """GQA model: 4 query heads share 2 KV heads, and head_dim (16) differs
    from hidden_size / num_attention_heads (32 / 4 = 8)."""
    cfg = LlamaConfig(
        vocab_size=64,  # pyright: ignore[reportCallIssue]
        hidden_size=32,  # pyright: ignore[reportCallIssue]
        intermediate_size=64,  # pyright: ignore[reportCallIssue]
        num_hidden_layers=3,  # pyright: ignore[reportCallIssue]
        num_attention_heads=4,  # pyright: ignore[reportCallIssue]
        num_key_value_heads=2,  # pyright: ignore[reportCallIssue]
        head_dim=16,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=128,  # pyright: ignore[reportCallIssue]
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def _zero_writes(model, layer: int, *, attn: bool, mlp: bool) -> None:
    """Zero a layer's residual writes (o_proj and/or down_proj)."""
    with torch.no_grad():
        if attn:
            model.model.layers[layer].self_attn.o_proj.weight.zero_()
        if mlp:
            model.model.layers[layer].mlp.down_proj.weight.zero_()


# ---------------------------------------------------------------------------
# Task 1: block_influence
# ---------------------------------------------------------------------------


class TestBlockInfluence:
    def test_returns_dict_with_8_keys(self, tiny_llama, tokenizer):
        scores = block_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        assert isinstance(scores, dict)
        assert len(scores) == 8

    def test_keys_are_integer_layer_indices(self, tiny_llama, tokenizer):
        scores = block_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        assert set(scores.keys()) == set(range(8))

    def test_scores_between_0_and_1(self, tiny_llama, tokenizer):
        scores = block_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for idx, val in scores.items():
            assert 0.0 <= val <= 1.0, f"Layer {idx} score {val} out of range"

    def test_multiple_prompts_produces_valid_output(self, tiny_llama, tokenizer):
        prompts = ["word4 word5", "word6 word7 word8", "word9 word10 word11 word12"]
        scores = block_influence(tiny_llama, tokenizer, prompts)
        assert len(scores) == 8
        for val in scores.values():
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# magnitude_influence
# ---------------------------------------------------------------------------


class TestMagnitudeInfluence:
    def test_returns_dict_with_8_keys(self, tiny_llama, tokenizer):
        scores = magnitude_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        assert isinstance(scores, dict)
        assert len(scores) == 8

    def test_keys_are_integer_layer_indices(self, tiny_llama, tokenizer):
        scores = magnitude_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        assert set(scores.keys()) == set(range(8))

    def test_each_layer_has_three_metrics(self, tiny_llama, tokenizer):
        scores = magnitude_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        expected_keys = {"magnitude_ratio", "contribution_norm", "bi_score"}
        for idx, layer_scores in scores.items():
            assert set(layer_scores.keys()) == expected_keys, (
                f"Layer {idx} keys: {set(layer_scores.keys())}"
            )

    def test_magnitude_ratio_positive(self, tiny_llama, tokenizer):
        scores = magnitude_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for idx, layer_scores in scores.items():
            assert layer_scores["magnitude_ratio"] > 0.0, (
                f"Layer {idx} magnitude_ratio should be positive"
            )

    def test_contribution_norm_non_negative(self, tiny_llama, tokenizer):
        scores = magnitude_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for idx, layer_scores in scores.items():
            assert layer_scores["contribution_norm"] >= 0.0, (
                f"Layer {idx} contribution_norm should be non-negative"
            )

    def test_bi_score_between_0_and_1(self, tiny_llama, tokenizer):
        scores = magnitude_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for idx, layer_scores in scores.items():
            assert 0.0 <= layer_scores["bi_score"] <= 1.0, (
                f"Layer {idx} bi_score {layer_scores['bi_score']} out of range"
            )

    def test_identity_layer_known_answer(self, tiny_llama, tokenizer):
        """A layer whose attention and MLP write nothing is the identity:
        bi_score 0, magnitude_ratio 1, contribution_norm 0."""
        _zero_writes(tiny_llama, 2, attn=True, mlp=True)
        scores = magnitude_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        assert scores[2]["bi_score"] == pytest.approx(0.0, abs=1e-6)
        assert scores[2]["magnitude_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert scores[2]["contribution_norm"] == pytest.approx(0.0, abs=1e-6)
        # An untouched layer does contribute.
        assert scores[3]["contribution_norm"] > 1e-3

    def test_multiple_prompts(self, tiny_llama, tokenizer):
        prompts = ["word4 word5", "word6 word7 word8", "word9 word10 word11 word12"]
        scores = magnitude_influence(tiny_llama, tokenizer, prompts)
        assert len(scores) == 8
        for layer_scores in scores.values():
            assert layer_scores["magnitude_ratio"] > 0.0
            assert layer_scores["contribution_norm"] >= 0.0


# ---------------------------------------------------------------------------
# sublayer_influence
# ---------------------------------------------------------------------------


class TestSublayerInfluence:
    def test_returns_dict_with_8_keys(self, tiny_llama, tokenizer):
        scores = sublayer_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        assert isinstance(scores, dict)
        assert len(scores) == 8

    def test_each_layer_has_attention_mlp_total(self, tiny_llama, tokenizer):
        scores = sublayer_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for idx, layer_scores in scores.items():
            assert set(layer_scores.keys()) == {"attention", "mlp", "total"}, (
                f"Layer {idx} keys: {set(layer_scores.keys())}"
            )

    def test_each_sublayer_has_three_metrics(self, tiny_llama, tokenizer):
        scores = sublayer_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        expected = {"magnitude_ratio", "contribution_norm", "bi_score"}
        for idx, layer_scores in scores.items():
            for sublayer in ("attention", "mlp", "total"):
                assert set(layer_scores[sublayer].keys()) == expected, (
                    f"Layer {idx} {sublayer} keys: {set(layer_scores[sublayer].keys())}"
                )

    def test_total_matches_magnitude_influence(self, tiny_llama, tokenizer):
        """The 'total' sub-dict should match magnitude_influence results."""
        prompts = ["word4 word5 word6"]
        mi = magnitude_influence(tiny_llama, tokenizer, prompts)
        sl = sublayer_influence(tiny_llama, tokenizer, prompts)
        for idx in range(8):
            for key in ("magnitude_ratio", "contribution_norm", "bi_score"):
                assert abs(sl[idx]["total"][key] - mi[idx][key]) < 1e-5, (
                    f"Layer {idx} total.{key}: sublayer={sl[idx]['total'][key]} "
                    f"!= magnitude={mi[idx][key]}"
                )

    def test_contribution_norms_non_negative(self, tiny_llama, tokenizer):
        scores = sublayer_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for _idx, layer_scores in scores.items():
            for sublayer in ("attention", "mlp", "total"):
                assert layer_scores[sublayer]["contribution_norm"] >= 0.0

    def test_magnitude_ratios_positive(self, tiny_llama, tokenizer):
        scores = sublayer_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for _idx, layer_scores in scores.items():
            for sublayer in ("attention", "mlp", "total"):
                assert layer_scores[sublayer]["magnitude_ratio"] > 0.0

    def test_multiple_prompts(self, tiny_llama, tokenizer):
        prompts = ["word4 word5", "word6 word7 word8", "word9 word10 word11 word12"]
        scores = sublayer_influence(tiny_llama, tokenizer, prompts)
        assert len(scores) == 8
        for layer_scores in scores.values():
            for sublayer in ("attention", "mlp", "total"):
                assert layer_scores[sublayer]["magnitude_ratio"] > 0.0


# ---------------------------------------------------------------------------
# Task 2: weight_norms and weight_svd
# ---------------------------------------------------------------------------


class TestWeightNorms:
    def test_returns_list_of_8_dicts(self, tiny_llama):
        norms = weight_norms(tiny_llama)
        assert isinstance(norms, list)
        assert len(norms) == 8

    def test_dicts_have_expected_keys(self, tiny_llama):
        norms = weight_norms(tiny_llama)
        for entry in norms:
            assert "layer" in entry
            assert "attn_norm" in entry
            assert "mlp_norm" in entry
            assert "total_norm" in entry

    def test_layer_indices_sequential(self, tiny_llama):
        norms = weight_norms(tiny_llama)
        for i, entry in enumerate(norms):
            assert entry["layer"] == i

    def test_all_values_positive(self, tiny_llama):
        norms = weight_norms(tiny_llama)
        for entry in norms:
            assert entry["attn_norm"] > 0.0
            assert entry["mlp_norm"] > 0.0
            assert entry["total_norm"] > 0.0


class TestWeightSVD:
    def test_all_layers_returned_by_default(self, tiny_llama):
        result = weight_svd(tiny_llama)
        assert isinstance(result, dict)
        assert len(result) == 8

    def test_specific_layers_only(self, tiny_llama):
        result = weight_svd(tiny_llama, layers=[0, 3, 7])
        assert set(result.keys()) == {0, 3, 7}

    def test_values_are_tensors_with_svd_keys(self, tiny_llama):
        result = weight_svd(tiny_llama, layers=[0])
        layer0 = result[0]
        for key in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]:
            assert key in layer0
            assert isinstance(layer0[key], torch.Tensor)

    def test_singular_values_non_negative(self, tiny_llama):
        result = weight_svd(tiny_llama, layers=[0])
        for key, svs in result[0].items():
            assert (svs >= 0).all(), f"{key} has negative singular values"


# ---------------------------------------------------------------------------
# Task 3: attention_entropy and residual_stream_norms
# ---------------------------------------------------------------------------


class TestAttentionEntropy:
    def test_returns_dict_of_8_layers(self, tiny_llama, tokenizer):
        result = attention_entropy(tiny_llama, tokenizer, "word4 word5 word6")
        assert isinstance(result, dict)
        assert len(result) == 8

    def test_each_layer_has_4_heads(self, tiny_llama, tokenizer):
        result = attention_entropy(tiny_llama, tokenizer, "word4 word5 word6")
        for layer_idx, entropies in result.items():
            assert len(entropies) == 4, (
                f"Layer {layer_idx} has {len(entropies)} heads, expected 4"
            )

    def test_entropies_are_non_negative(self, tiny_llama, tokenizer):
        result = attention_entropy(tiny_llama, tokenizer, "word4 word5 word6")
        for layer_idx, entropies in result.items():
            for h, e in enumerate(entropies):
                assert e >= 0.0, f"Layer {layer_idx} head {h} entropy {e} is negative"


class TestResidualStreamNorms:
    def test_returns_list_of_9_values(self, tiny_llama, tokenizer):
        norms = residual_stream_norms(tiny_llama, tokenizer, "word4 word5 word6")
        assert isinstance(norms, list)
        assert len(norms) == 9  # 8 layers + embedding

    def test_all_positive(self, tiny_llama, tokenizer):
        norms = residual_stream_norms(tiny_llama, tokenizer, "word4 word5 word6")
        for i, n in enumerate(norms):
            assert n > 0.0, f"Norm at position {i} is not positive: {n}"

    def test_raises_when_a_layer_hook_never_fires(self, tiny_llama, tokenizer):
        # LlamaModel.forward runs layers[: config.num_hidden_layers].
        tiny_llama.config.num_hidden_layers = 7
        with pytest.raises(RuntimeError, match=r"\[7\]"):
            residual_stream_norms(tiny_llama, tokenizer, "word4 word5 word6")


# ---------------------------------------------------------------------------
# Audit fixes (issue #10): known answers, memory, GQA, head_dim, config restore
# ---------------------------------------------------------------------------


class TestComputeMetricsKnownAnswer:
    def test_hand_computed_values(self):
        flat_in = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        flat_out = torch.tensor([[6.0, 8.0], [0.0, 1.0]])
        m = _compute_metrics(flat_in, flat_out)
        # ratios: 10/5 = 2, 1/1 = 1 -> mean 1.5
        assert m["magnitude_ratio"] == pytest.approx(1.5)
        # contributions: ||(3,4)|| = 5, ||(-1,1)|| = sqrt(2)
        assert m["contribution_norm"] == pytest.approx((5.0 + 2**0.5) / 2)
        # cosines: 1 and 0 -> mean 0.5 -> bi = 0.5
        assert m["bi_score"] == pytest.approx(0.5)

    def test_identity_is_zero_influence(self):
        x = torch.randn(5, 8)
        m = _compute_metrics(x, x.clone())
        assert m["bi_score"] == pytest.approx(0.0, abs=1e-6)
        assert m["magnitude_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert m["contribution_norm"] == pytest.approx(0.0, abs=1e-6)


class TestSublayerKnownAnswer:
    def test_zeroed_attention_write(self, tiny_llama, tokenizer):
        _zero_writes(tiny_llama, 1, attn=True, mlp=False)
        scores = sublayer_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        attn = scores[1]["attention"]
        assert attn["contribution_norm"] == pytest.approx(0.0, abs=1e-6)
        assert attn["magnitude_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert scores[1]["mlp"]["contribution_norm"] > 1e-3
        # With no attention write, the MLP span is the whole block.
        for key in ("magnitude_ratio", "contribution_norm", "bi_score"):
            assert scores[1]["mlp"][key] == pytest.approx(
                scores[1]["total"][key], abs=1e-5
            )


class _WatchedPrompts(list):
    """Prompt list that, before yielding each prompt after the first, counts
    live (1, seq, hidden) tensors, i.e. hidden states retained from earlier
    prompts."""

    def __init__(self, prompts, seq, hidden):
        super().__init__(prompts)
        self.seq, self.hidden = seq, hidden
        self.counts: list[int] = []

    def __iter__(self):
        for k, prompt in enumerate(list.__iter__(self)):
            if k:
                gc.collect()
                self.counts.append(
                    sum(
                        1
                        for o in gc.get_objects()
                        # type() rather than isinstance(): isinstance touches
                        # deprecated module proxies and emits FutureWarnings.
                        if type(o) is torch.Tensor
                        and o.dim() == 3
                        and tuple(o.shape[1:]) == (self.seq, self.hidden)
                    )
                )
            yield prompt


class TestInfluenceMemory:
    @pytest.mark.parametrize("fn", [magnitude_influence, sublayer_influence])
    def test_does_not_retain_hidden_states_across_prompts(
        self, fn, tiny_llama, tokenizer
    ):
        prompt = " ".join(f"word{i}" for i in range(4, 11))  # 7 tokens
        prompts = _WatchedPrompts([prompt] * 4, seq=7, hidden=32)
        fn(tiny_llama, tokenizer, prompts)
        # At most one prompt's worth (3 tensors per layer) may be alive.
        assert max(prompts.counts) <= 3 * 8, prompts.counts


class TestAttnImplementationRestore:
    def test_attention_entropy_restores(self, tiny_llama, tokenizer):
        orig = tiny_llama.config._attn_implementation
        attention_entropy(tiny_llama, tokenizer, "word4 word5")
        assert tiny_llama.config._attn_implementation == orig

    def test_attention_entropy_restores_none(self, tiny_llama, tokenizer):
        # The getter can return None; the property has no deleter, so a
        # `del` restore raised AttributeError here.
        tiny_llama.config._attn_implementation = None
        attention_entropy(tiny_llama, tokenizer, "word4 word5")
        assert tiny_llama.config._attn_implementation is None

    def test_inspect_head_restores(self, tiny_llama, tokenizer):
        orig = tiny_llama.config._attn_implementation
        inspect_head(tiny_llama, tokenizer, "word4 word5", layer=0, head=1)
        assert tiny_llama.config._attn_implementation == orig

    def test_inspect_head_restores_when_hook_setup_fails(
        self, tiny_llama, tokenizer, monkeypatch
    ):
        orig = tiny_llama.config._attn_implementation
        o_proj = tiny_llama.model.layers[0].self_attn.o_proj

        def boom(*_a, **_k):
            raise RuntimeError("hook setup failed")

        monkeypatch.setattr(o_proj, "register_forward_pre_hook", boom)
        with pytest.raises(RuntimeError, match="hook setup failed"):
            inspect_head(tiny_llama, tokenizer, "word4 word5", layer=0, head=0)
        assert tiny_llama.config._attn_implementation == orig


class TestInspectHead:
    def test_returns_expected_shapes(self, tiny_llama, tokenizer):
        out = inspect_head(tiny_llama, tokenizer, "word4 word5 word6", layer=2, head=3)
        assert out["attention_pattern"].shape == (3, 3)
        assert out["output_norm"] > 0.0
        assert out["entropy"] >= 0.0

    @pytest.mark.parametrize("layer,head", [(-1, 0), (8, 0), (0, -1), (0, 4)])
    def test_rejects_out_of_range(self, tiny_llama, tokenizer, layer, head):
        with pytest.raises(IndexError):
            inspect_head(tiny_llama, tokenizer, "word4", layer=layer, head=head)

    def test_uses_config_head_dim(self, gqa_llama, tokenizer):
        """head_dim (16) != hidden/heads (8): the head's slice of the o_proj
        input must be 16 wide."""
        layer, head, hd = 1, 2, 16
        captured = {}
        o_proj = gqa_llama.model.layers[layer].self_attn.o_proj
        handle = o_proj.register_forward_pre_hook(
            lambda _m, args: captured.__setitem__("x", args[0].detach())
        )
        try:
            out = inspect_head(gqa_llama, tokenizer, "word4 word5 word6", layer, head)
        finally:
            handle.remove()
        expected = (
            captured["x"][0, :, head * hd : (head + 1) * hd].float().norm(dim=-1).mean()
        )
        assert out["output_norm"] == pytest.approx(expected.item(), rel=1e-5)


class TestGQA:
    def test_attention_entropy_reports_query_heads(self, gqa_llama, tokenizer):
        result = attention_entropy(gqa_llama, tokenizer, "word4 word5 word6")
        assert len(result) == 3
        for entropies in result.values():
            assert len(entropies) == 4

    def test_weight_svd_kv_shapes(self, gqa_llama):
        svd = weight_svd(gqa_llama, layers=[0])[0]
        # k_proj: (2 kv heads * 16, 32) -> min dim 32
        assert svd["k_proj"].shape == (32,)
        # q_proj: (4 * 16, 32) -> 32 singular values
        assert svd["q_proj"].shape == (32,)

    def test_influence_runs(self, gqa_llama, tokenizer):
        scores = sublayer_influence(gqa_llama, tokenizer, ["word4 word5 word6"])
        assert set(scores) == {0, 1, 2}
