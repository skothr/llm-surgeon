"""Tests for inspect module."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon.inspect import (
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

    def test_bi_scores_match_block_influence(self, tiny_llama, tokenizer):
        """bi_score from magnitude_influence should match block_influence."""
        prompts = ["word4 word5 word6"]
        bi = block_influence(tiny_llama, tokenizer, prompts)
        mi = magnitude_influence(tiny_llama, tokenizer, prompts)
        for idx in range(8):
            assert abs(mi[idx]["bi_score"] - bi[idx]) < 1e-5, (
                f"Layer {idx}: magnitude_influence bi_score {mi[idx]['bi_score']} "
                f"!= block_influence {bi[idx]}"
            )

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
        for idx, layer_scores in scores.items():
            for sublayer in ("attention", "mlp", "total"):
                assert layer_scores[sublayer]["contribution_norm"] >= 0.0

    def test_magnitude_ratios_positive(self, tiny_llama, tokenizer):
        scores = sublayer_influence(tiny_llama, tokenizer, ["word4 word5 word6"])
        for idx, layer_scores in scores.items():
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
        for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
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
            assert len(entropies) == 4, f"Layer {layer_idx} has {len(entropies)} heads, expected 4"

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
