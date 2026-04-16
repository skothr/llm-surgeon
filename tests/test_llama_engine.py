import numpy as np
import pytest
from llm_surgeon.llama_engine import GenerateStep, compare_logits


class TestCompareLogits:
    def test_identical_logits(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = compare_logits(a, a)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert result["kl_divergence"] == pytest.approx(0.0, abs=1e-6)
        assert result["max_logit_diff"] == pytest.approx(0.0, abs=1e-6)
        assert result["top_k_agreement"] == 4

    def test_different_logits(self):
        a = np.array([10.0, 1.0, 0.5, 0.1], dtype=np.float32)
        b = np.array([0.1, 0.5, 1.0, 10.0], dtype=np.float32)
        result = compare_logits(a, b)
        assert result["cosine_similarity"] < 0.5
        assert result["kl_divergence"] > 1.0
        assert result["top_k_agreement"] >= 0

    def test_top_k_agreement_partial(self):
        a = np.array([10.0, 5.0, 1.0, 0.1], dtype=np.float32)
        b = np.array([10.0, 4.0, 0.5, 5.0], dtype=np.float32)
        result = compare_logits(a, b, top_k=2)
        assert result["top_k_agreement"] == 1

    def test_returns_all_keys(self):
        a = np.ones(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        result = compare_logits(a, b)
        assert set(result.keys()) == {
            "cosine_similarity", "kl_divergence",
            "max_logit_diff", "mean_logit_diff", "top_k_agreement",
        }


class TestGenerateStep:
    def test_fields(self):
        step = GenerateStep(token_id=42, token_str="hello", logits=None)
        assert step.token_id == 42
        assert step.token_str == "hello"
        assert step.logits is None

    def test_with_logits(self):
        logits = np.zeros(100, dtype=np.float32)
        step = GenerateStep(token_id=1, token_str="a", logits=logits)
        assert step.logits is not None
        assert step.logits.shape == (100,)
