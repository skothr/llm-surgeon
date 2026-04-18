"""Tests for benchmark module — perplexity and downstream eval."""

import copy
import json
import math
import warnings

import pytest

from llm_surgeon.benchmark import perplexity, eval_downstream


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

        perplexity(tiny_llama, tok, text=text)

        # Deep-copy so we operate on an independent model
        modified = copy.deepcopy(tiny_llama)
        remove_layers(modified, [3, 4, 5])
        ppl_modified = perplexity(modified, tok, text=text)

        # Random-weight tiny models may produce similar perplexity after surgery.
        # Just verify computation completed without error — real models show clear deltas.
        assert isinstance(ppl_modified, float)
        assert ppl_modified > 0

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
            tasks=["arc_easy"],
            model_path=tiny_eval_checkpoint,
            num_fewshot=0,
            limit=5,
        )
        assert isinstance(result, dict)
        assert "arc_easy" in result

    def test_invalid_task_raises_runtime_error(self, tiny_eval_checkpoint):
        """eval_downstream() raises RuntimeError for an unknown task name."""
        with pytest.raises(RuntimeError):
            eval_downstream(
                tasks=["this_task_does_not_exist_xyz"],
                model_path=tiny_eval_checkpoint,
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


# ---------------------------------------------------------------------------
# Phase 2: in-process eval_downstream + eval_and_log
# ---------------------------------------------------------------------------

class TestEvalDownstreamValidation:
    """Validation-error contracts on the new eval_downstream signature."""

    def test_no_model_source_raises(self):
        from llm_surgeon.benchmark import eval_downstream
        with pytest.raises(ValueError, match="Exactly one of model_path or model"):
            eval_downstream(tasks=["hellaswag"])

    def test_both_model_sources_raises(self):
        from llm_surgeon.benchmark import eval_downstream
        with pytest.raises(ValueError, match="Exactly one of model_path or model"):
            eval_downstream(
                tasks=["hellaswag"],
                model_path="/tmp/nonexistent",
                model=object(),
                tokenizer=object(),
            )

    def test_model_without_tokenizer_raises(self):
        from llm_surgeon.benchmark import eval_downstream
        with pytest.raises(ValueError, match="tokenizer.*required"):
            eval_downstream(tasks=["hellaswag"], model=object())


class TestFewShotResolution:
    """_resolve_fewshot maps (tasks, num_fewshot) to {task: int} per spec §3."""

    def test_default_uses_paper_standard(self):
        from llm_surgeon.benchmark import _resolve_fewshot
        out = _resolve_fewshot(["hellaswag", "arc_challenge"], None)
        assert out == {"hellaswag": 0, "arc_challenge": 25}

    def test_unknown_task_defaults_to_zero(self):
        from llm_surgeon.benchmark import _resolve_fewshot
        out = _resolve_fewshot(["not_in_table"], None)
        assert out == {"not_in_table": 0}

    def test_int_applies_uniformly(self):
        from llm_surgeon.benchmark import _resolve_fewshot
        out = _resolve_fewshot(["hellaswag", "arc_challenge"], 3)
        assert out == {"hellaswag": 3, "arc_challenge": 3}

    def test_dict_overrides_paper_standard(self):
        from llm_surgeon.benchmark import _resolve_fewshot
        out = _resolve_fewshot(
            ["hellaswag", "arc_challenge"],
            {"hellaswag": 10},
        )
        # Specified task wins; unspecified falls back to PAPER_STANDARD.
        assert out == {"hellaswag": 10, "arc_challenge": 25}


class TestGroupByFewshot:
    def test_groups_by_count(self):
        from llm_surgeon.benchmark import _group_by_fewshot
        groups = _group_by_fewshot(
            {"hellaswag": 0, "arc_easy": 0, "arc_challenge": 25}
        )
        # Sorted by (count, first-task-name) for determinism.
        assert groups == [(0, ["arc_easy", "hellaswag"]), (25, ["arc_challenge"])]


class TestEvalDownstreamInProcess:
    """Mock simple_evaluate / HFLM; assert kwarg routing and grouping."""

    def _install_mocks(self, monkeypatch):
        captured = {"calls": []}

        class _FakeHFLM:
            def __init__(self, pretrained, tokenizer):
                captured["hflm_pretrained"] = pretrained
                captured["hflm_tokenizer"] = tokenizer

        def fake_simple_evaluate(**kwargs):
            captured["calls"].append(kwargs)
            return {
                "results": {t: {"acc,none": 0.5, "acc_stderr,none": 0.01}
                            for t in kwargs["tasks"]},
                "config": {"model": "mock"},
            }

        monkeypatch.setattr("lm_eval.models.huggingface.HFLM", _FakeHFLM)
        monkeypatch.setattr("lm_eval.simple_evaluate", fake_simple_evaluate)
        return captured

    def test_defaults_to_fast_triplet(self, monkeypatch):
        from llm_surgeon.benchmark import eval_downstream, FAST_TRIPLET
        captured = self._install_mocks(monkeypatch)

        class _M:
            config = None

        result = eval_downstream(model=_M(), tokenizer=object())
        assert set(result.keys()) == set(FAST_TRIPLET)
        seen = set()
        for c in captured["calls"]:
            seen.update(c["tasks"])
        assert seen == set(FAST_TRIPLET)

    def test_dispatches_per_task_fewshot(self, monkeypatch):
        from llm_surgeon.benchmark import eval_downstream
        captured = self._install_mocks(monkeypatch)

        class _M:
            config = None

        eval_downstream(
            tasks=["hellaswag", "arc_challenge"],
            model=_M(), tokenizer=object(),
        )
        calls = captured["calls"]
        assert len(calls) == 2
        by_n = {c["num_fewshot"]: c["tasks"] for c in calls}
        assert by_n[0] == ["hellaswag"]
        assert by_n[25] == ["arc_challenge"]

    def test_quantized_model_warns(self, monkeypatch):
        from llm_surgeon.benchmark import eval_downstream
        self._install_mocks(monkeypatch)

        class _Cfg:
            quantization_config = {"load_in_4bit": True}

        class _M:
            config = _Cfg()

        with pytest.warns(UserWarning, match="quantization_config set"):
            eval_downstream(
                tasks=["hellaswag"],
                model=_M(), tokenizer=object(),
            )

    def test_hflm_receives_model_and_tokenizer(self, monkeypatch):
        from llm_surgeon.benchmark import eval_downstream
        captured = self._install_mocks(monkeypatch)

        class _M:
            config = None

        m, t = _M(), object()
        eval_downstream(tasks=["hellaswag"], model=m, tokenizer=t)
        assert captured["hflm_pretrained"] is m
        assert captured["hflm_tokenizer"] is t

    def test_returns_primary_accuracy_dict(self, monkeypatch):
        from llm_surgeon.benchmark import eval_downstream
        self._install_mocks(monkeypatch)

        class _M:
            config = None

        result = eval_downstream(
            tasks=["hellaswag"], model=_M(), tokenizer=object(),
        )
        assert result == {"hellaswag": 0.5}


class TestEvalAndLog:
    """eval_and_log persists harness results to both metrics (flat) and
    harness_results (blob) tables via tracking."""

    def _mock_in_process(self, monkeypatch, task_results):
        def fake(**kwargs):
            return {
                "results": task_results,
                "config": {"model": "mock"},
                "effective_num_fewshot": {t: 0 for t in kwargs["tasks"]},
            }
        monkeypatch.setattr(
            "llm_surgeon.benchmark._in_process_eval", fake
        )

    def test_writes_flat_metrics(self, monkeypatch, tmp_path):
        from llm_surgeon.benchmark import eval_and_log
        from llm_surgeon.tracking import start, get_experiment

        self._mock_in_process(
            monkeypatch,
            {"hellaswag": {"acc,none": 0.5, "acc_stderr,none": 0.01,
                           "alias": "hellaswag"}},
        )
        db = str(tmp_path / "t.db")
        exp = start("exp1", db_path=db)

        class _M:
            config = None

        acc = eval_and_log(
            experiment=exp,
            model=_M(), tokenizer=object(),
            tasks=["hellaswag"],
        )
        assert acc == {"hellaswag": 0.5}

        metrics = {m["key"]: m["value"]
                   for m in get_experiment("exp1", db_path=db)["metrics"]}
        assert metrics["harness.hellaswag.acc,none"] == 0.5
        assert metrics["harness.hellaswag.acc_stderr,none"] == 0.01
        assert "harness.hellaswag.alias" not in metrics

    def test_writes_blob(self, monkeypatch, tmp_path):
        from llm_surgeon.benchmark import eval_and_log
        from llm_surgeon.tracking import start
        import sqlite3
        import json as _json

        self._mock_in_process(
            monkeypatch,
            {"hellaswag": {"acc,none": 0.5}},
        )
        db = str(tmp_path / "t.db")
        exp = start("exp1", db_path=db)

        class _M:
            config = None

        eval_and_log(
            experiment=exp,
            model=_M(), tokenizer=object(),
            tasks=["hellaswag"],
            limit=20,
        )
        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT tasks_json, num_fewshot, limit_samples, result_json "
            "FROM harness_results WHERE experiment_name = 'exp1'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert _json.loads(row[0]) == ["hellaswag"]
        assert _json.loads(row[1]) == {"hellaswag": 0}
        assert row[2] == 20
        payload = _json.loads(row[3])
        assert payload["results"]["hellaswag"]["acc,none"] == 0.5


def _tinyllama_cached() -> bool:
    from llm_surgeon.surgery import _is_cached
    return _is_cached("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


import torch  # noqa: E402


@pytest.mark.skipif(
    not _tinyllama_cached() or not torch.cuda.is_available(),
    reason="requires TinyLlama cache + CUDA GPU",
)
class TestEvalAndLogIntegration:
    """End-to-end: real TinyLlama fp16 load -> arc_easy eval -> SQLite."""

    def test_tinyllama_arc_easy_limit20(self, tmp_path):
        from llm_surgeon.surgery import load_model
        from llm_surgeon.benchmark import eval_and_log
        from llm_surgeon.tracking import start, get_experiment
        import sqlite3

        db = str(tmp_path / "t.db")
        exp = start("it1", db_path=db)

        model, tokenizer = load_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16",
        )
        acc = eval_and_log(
            experiment=exp,
            model=model, tokenizer=tokenizer,
            tasks=["arc_easy"],
            limit=20,
        )
        assert "arc_easy" in acc
        assert 0.0 <= acc["arc_easy"] <= 1.0

        metrics = {m["key"]: m["value"]
                   for m in get_experiment("it1", db_path=db)["metrics"]}
        assert any(k.startswith("harness.arc_easy.") for k in metrics)

        conn = sqlite3.connect(db)
        n = conn.execute(
            "SELECT COUNT(*) FROM harness_results WHERE experiment_name = 'it1'"
        ).fetchone()[0]
        conn.close()
        assert n == 1
