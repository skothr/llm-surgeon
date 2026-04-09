"""Tests for benchmark module — perplexity and downstream eval."""

import copy
import json
import math
import os
import tempfile
import warnings

import pytest
import torch

from llm_surgeon.benchmark import perplexity, eval_downstream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_tokenizer_and_text(vocab_size: int):
    """Return (tokenizer, sample_text) using the tiny WordLevel tokenizer."""
    from tests.conftest import _make_tiny_tokenizer

    tok = _make_tiny_tokenizer(vocab_size)
    # Build a sample text that the tokenizer can handle
    text = " ".join([f"word{i % vocab_size}" for i in range(200)])
    return tok, text


# ---------------------------------------------------------------------------
# Task 1: perplexity()
# ---------------------------------------------------------------------------

class TestPerplexityBasic:
    """Basic sanity checks for the perplexity function."""

    def test_returns_positive_float(self, tiny_llama, tmp_path):
        """perplexity() returns a positive float."""
        from tests.conftest import _make_tiny_tokenizer

        tok = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        text = " ".join([f"word{i % 50}" for i in range(200)])
        result = perplexity(tiny_llama, tok, text=text)
        assert isinstance(result, float)
        assert result > 0.0

    def test_result_is_finite(self, tiny_llama):
        """perplexity() result must not be inf or nan."""
        from tests.conftest import _make_tiny_tokenizer

        tok = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        text = " ".join([f"word{i % 50}" for i in range(200)])
        result = perplexity(tiny_llama, tok, text=text)
        assert math.isfinite(result)

    def test_modified_model_gives_different_perplexity(self, tiny_llama):
        """Surgery changes the model's perplexity — proving it is sensitive to weights."""
        from tests.conftest import _make_tiny_tokenizer
        from llm_surgeon.surgery import remove_layers

        tok = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        text = " ".join([f"word{i % 50}" for i in range(300)])

        ppl_original = perplexity(tiny_llama, tok, text=text)

        # Deep-copy so we operate on an independent model
        modified = copy.deepcopy(tiny_llama)
        remove_layers(modified, [3, 4, 5])
        ppl_modified = perplexity(modified, tok, text=text)

        assert ppl_original != pytest.approx(ppl_modified, rel=1e-3), (
            f"Expected different perplexities: original={ppl_original}, modified={ppl_modified}"
        )

    def test_warns_on_quantized_model(self, tiny_llama):
        """perplexity() warns when model.config has quantization_config."""
        from tests.conftest import _make_tiny_tokenizer

        # Inject a fake quantization_config to trigger the warning
        tiny_llama.config.quantization_config = {"quant_type": "int4"}

        tok = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        text = " ".join([f"word{i % 50}" for i in range(200)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            perplexity(tiny_llama, tok, text=text)

        messages = [str(w.message) for w in caught]
        assert any("quantiz" in m.lower() for m in messages), (
            f"Expected quantization warning, got: {messages}"
        )


def _network_available() -> bool:
    """Return True if a basic network connection can be established."""
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3)
        return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _network_available(),
    reason="No network access — skipping dataset download test",
)


class TestPerplexityDataset:
    """Tests for dataset-based perplexity (requires network / HF datasets)."""

    @requires_network
    def test_wikitext2_returns_positive_finite(self, tiny_llama):
        """perplexity() with dataset='wikitext2' returns a valid float.

        Uses max_samples=2 for speed. Requires HF datasets and network.
        """
        from tests.conftest import _make_tiny_tokenizer

        tok = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        result = perplexity(tiny_llama, tok, dataset="wikitext2", max_samples=2)
        assert isinstance(result, float)
        assert math.isfinite(result)
        assert result > 0.0


# ---------------------------------------------------------------------------
# Task 2: eval_downstream()
# ---------------------------------------------------------------------------

class TestEvalDownstream:
    """Tests for eval_downstream() using lm-eval harness.

    NOTE: These tests are slow even with limit=5 because lm_eval loads a full
    HF model checkpoint. Expected runtime: 30s–2min per test.
    """

    @requires_network
    def test_returns_dict_with_task_key(self, tiny_eval_checkpoint):
        """eval_downstream() returns a dict containing the task name as a key.

        Requires network to download arc_easy dataset from HuggingFace.
        """
        result = eval_downstream(
            tiny_eval_checkpoint,
            tasks=["arc_easy"],
            num_fewshot=0,
            limit=5,
        )
        assert isinstance(result, dict)
        assert "arc_easy" in result

    def test_invalid_task_raises_runtime_error(self, tiny_eval_checkpoint):
        """eval_downstream() raises RuntimeError for an unknown task name."""
        with pytest.raises(RuntimeError):
            eval_downstream(
                tiny_eval_checkpoint,
                tasks=["this_task_does_not_exist_xyz"],
                num_fewshot=0,
                limit=5,
            )


# ---------------------------------------------------------------------------
# Helpers for Phase 5 tests
# ---------------------------------------------------------------------------

def _ollama_available():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running on localhost:11434",
)


# ---------------------------------------------------------------------------
# Task 1: _load_prompts() and compare()
# ---------------------------------------------------------------------------

class TestLoadPrompts:
    """Tests for the _load_prompts() helper — no ollama needed."""

    def test_loads_prompt_file(self, tmp_path):
        """_load_prompts() loads a JSON file and returns a list of dicts."""
        from llm_surgeon.benchmark import _load_prompts

        prompts = [
            {"prompt": "Hello world", "category": "test"},
            {"prompt": "Another prompt", "category": "other"},
        ]
        p = tmp_path / "prompts.json"
        p.write_text(json.dumps(prompts))

        result = _load_prompts(str(p))

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["prompt"] == "Hello world"
        assert result[0]["category"] == "test"
        assert result[1]["prompt"] == "Another prompt"


class TestCompare:
    """Tests for compare() — ollama tests are skipped when not running."""

    @requires_ollama
    def test_compare_returns_results(self):
        """compare() returns a list with the expected structure."""
        from llm_surgeon.benchmark import compare

        prompts = [{"prompt": "The capital of France is", "category": "factual"}]
        results = compare(["tinyllama"], prompts)

        assert isinstance(results, list)
        assert len(results) == 1
        entry = results[0]
        assert "prompt" in entry
        assert "category" in entry
        assert "responses" in entry
        assert "tinyllama" in entry["responses"]
        assert "text" in entry["responses"]["tinyllama"]

    @requires_ollama
    def test_compare_collects_timing(self):
        """compare() collects tokens_per_second and total_tokens for each model."""
        from llm_surgeon.benchmark import compare

        prompts = [{"prompt": "The capital of France is", "category": "factual"}]
        results = compare(["tinyllama"], prompts)

        resp = results[0]["responses"]["tinyllama"]
        assert "tokens_per_second" in resp
        assert "total_tokens" in resp
        assert isinstance(resp["tokens_per_second"], float)
        assert isinstance(resp["total_tokens"], int)

    @requires_ollama
    def test_compare_saves_json(self, tmp_path):
        """compare() writes results to output_file when provided."""
        from llm_surgeon.benchmark import compare

        prompts = [{"prompt": "The capital of France is", "category": "factual"}]
        out = tmp_path / "out.json"
        compare(["tinyllama"], prompts, output_file=str(out))

        assert out.exists()
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1


# ---------------------------------------------------------------------------
# Task 2: generation_metrics()
# ---------------------------------------------------------------------------

class TestGenerationMetrics:
    """Tests for generation_metrics() — all use synthetic data, no ollama needed."""

    def _make_results(self, model_a_texts, model_b_texts):
        """Build a synthetic compare() output with two models."""
        results = []
        for i, (ta, tb) in enumerate(zip(model_a_texts, model_b_texts)):
            results.append({
                "prompt": f"prompt {i}",
                "category": "test",
                "responses": {
                    "model_a": {"text": ta, "tokens_per_second": 10.0, "total_tokens": 20},
                    "model_b": {"text": tb, "tokens_per_second": 12.0, "total_tokens": 25},
                },
            })
        return results

    def test_returns_dict_per_model(self):
        """generation_metrics() returns a dict keyed by model name."""
        from llm_surgeon.benchmark import generation_metrics

        results = self._make_results(
            ["Paris is the capital of France."],
            ["Berlin is the capital of Germany."],
        )
        metrics = generation_metrics(results)

        assert isinstance(metrics, dict)
        assert "model_a" in metrics
        assert "model_b" in metrics

    def test_has_expected_keys(self):
        """generation_metrics() output contains all four metric keys per model."""
        from llm_surgeon.benchmark import generation_metrics

        results = self._make_results(
            ["The quick brown fox jumps over the lazy dog."],
            ["A sentence with some varied words about nature."],
        )
        metrics = generation_metrics(results)

        expected_keys = {"mean_output_length", "vocab_diversity", "repetition_rate", "coherence"}
        for model_name, model_metrics in metrics.items():
            assert expected_keys == set(model_metrics.keys()), (
                f"Model {model_name} missing keys: {expected_keys - set(model_metrics.keys())}"
            )

    def test_repetition_rate_higher_for_repetitive(self):
        """repetition_rate is higher for text with repeated 3-grams."""
        from llm_surgeon.benchmark import generation_metrics

        repetitive_text = "the the the the the the the the the the the"
        normal_text = "the quick brown fox jumps over the lazy dog today"

        results = self._make_results([repetitive_text], [normal_text])
        metrics = generation_metrics(results)

        rep_rate_repetitive = metrics["model_a"]["repetition_rate"]
        rep_rate_normal = metrics["model_b"]["repetition_rate"]
        assert rep_rate_repetitive > rep_rate_normal, (
            f"Expected repetitive ({rep_rate_repetitive}) > normal ({rep_rate_normal})"
        )

    def test_vocab_diversity_lower_for_repetitive(self):
        """vocab_diversity is lower for text with a small vocabulary."""
        from llm_surgeon.benchmark import generation_metrics

        repetitive_text = "Paris Paris Paris Paris Paris Paris Paris Paris"
        diverse_text = "Paris London Berlin Tokyo Sydney Rome Madrid Vienna"

        results = self._make_results([repetitive_text], [diverse_text])
        metrics = generation_metrics(results)

        div_repetitive = metrics["model_a"]["vocab_diversity"]
        div_diverse = metrics["model_b"]["vocab_diversity"]
        assert div_repetitive < div_diverse, (
            f"Expected repetitive ({div_repetitive}) < diverse ({div_diverse})"
        )
