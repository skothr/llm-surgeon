# Phase 5: Generation Comparison + Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
>
> **Tool rules (for subagents):**
> - Use Read (not cat/head/tail), Grep (not grep/rg/awk), Glob (not find/ls), Edit (not sed/awk) for all file operations
> - You are already in the project root (/home/ai/ai-projects/llm) — never cd
> - Python venv: `/home/ai/ai-projects/llm/testing/.venv/bin/python`
> - Run tests: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`

**Goal:** Add ollama-based generation comparison and automated generation quality metrics to `benchmark.py`, plus a default prompt set.

**Architecture:** `compare()` sends prompts to ollama models via HTTP API and collects responses. `generation_metrics()` computes heuristic quality metrics on those responses. Results saved as JSON.

**Tech Stack:** requests (ollama API), json

**Reference:** `docs/superpowers/specs/2026-04-08-llm-surgeon-design.md` (v2), Phase 5 section of phase plan.

---

## File Map

```
testing/
  llm_surgeon/
    benchmark.py         — MODIFY — add compare, generation_metrics
  tests/
    test_benchmark.py    — MODIFY — add tests for compare + generation_metrics
  prompts/
    default.json         — CREATE — default prompt set
```

---

### Task 1: Default prompt set + compare()

**Files:**
- Create: `testing/prompts/default.json`
- Modify: `testing/llm_surgeon/benchmark.py`
- Modify: `testing/tests/test_benchmark.py`

- [ ] **Step 1: Create default prompt set**

Create `testing/prompts/default.json`:

```json
[
    {"prompt": "The capital of France is", "category": "factual"},
    {"prompt": "Explain gravity in one sentence.", "category": "reasoning"},
    {"prompt": "Write a haiku about rain.", "category": "creative"},
    {"prompt": "List three prime numbers.", "category": "factual"},
    {"prompt": "What happens when water freezes?", "category": "reasoning"}
]
```

- [ ] **Step 2: Write tests for compare()**

Add to `testing/tests/test_benchmark.py`:

```python
import json
import os
import requests
from llm_surgeon.benchmark import compare, generation_metrics


def _ollama_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


class TestCompare:
    def test_loads_prompt_file(self, tmp_path):
        """Test that compare can load a JSON prompt file."""
        prompts = [
            {"prompt": "Hello", "category": "test"},
            {"prompt": "World", "category": "test"},
        ]
        prompt_file = str(tmp_path / "prompts.json")
        with open(prompt_file, "w") as f:
            json.dump(prompts, f)

        # Without ollama, we just test the prompt loading
        loaded = compare._load_prompts(prompt_file)
        assert len(loaded) == 2
        assert loaded[0]["prompt"] == "Hello"

    @pytest.mark.skipif(not _ollama_available(), reason="ollama not running")
    def test_compare_returns_results(self):
        """Test actual comparison with ollama (requires tinyllama model)."""
        results = compare(
            models=["tinyllama"],
            prompts=[{"prompt": "Hello", "category": "test"}],
            temperature=0.0,
            max_tokens=20,
        )
        assert isinstance(results, list)
        assert len(results) == 1  # one prompt
        assert "tinyllama" in results[0]["responses"]

    @pytest.mark.skipif(not _ollama_available(), reason="ollama not running")
    def test_compare_collects_timing(self):
        results = compare(
            models=["tinyllama"],
            prompts=[{"prompt": "Count to three.", "category": "test"}],
            temperature=0.0,
            max_tokens=30,
        )
        resp = results[0]["responses"]["tinyllama"]
        assert "text" in resp
        assert "tokens_per_second" in resp
        assert "total_tokens" in resp

    @pytest.mark.skipif(not _ollama_available(), reason="ollama not running")
    def test_compare_saves_json(self, tmp_path):
        output_file = str(tmp_path / "results.json")
        results = compare(
            models=["tinyllama"],
            prompts=[{"prompt": "Hi", "category": "test"}],
            temperature=0.0,
            max_tokens=10,
            output_file=output_file,
        )
        assert os.path.exists(output_file)
        with open(output_file) as f:
            saved = json.load(f)
        assert len(saved) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Expected: FAIL — ImportError for `compare`

- [ ] **Step 4: Implement compare()**

Add to `testing/llm_surgeon/benchmark.py`:

```python
import json as json_module
import requests as requests_lib


def compare(
    models: list,
    prompts,
    temperature: float = 0.0,
    max_tokens: int = 256,
    output_file: Optional[str] = None,
) -> list:
    """Compare generation from multiple ollama models on the same prompts.

    Args:
        models: List of ollama model names
        prompts: List of prompt dicts [{"prompt": str, "category": str}]
                 or path to a JSON file
        temperature: Sampling temperature (0.0 for near-deterministic)
        max_tokens: Max tokens to generate per prompt
        output_file: Optional path to save results as JSON

    Returns:
        List of result dicts, one per prompt.

    Note: temperature=0.0 is near-deterministic but quantized models in
    llama.cpp have numerical non-determinism from parallel reduction.
    Don't chase phantom differences across runs.
    """
    if isinstance(prompts, str):
        prompts = _load_prompts(prompts)

    results = []
    for prompt_entry in prompts:
        prompt_text = prompt_entry["prompt"]
        category = prompt_entry.get("category", "")

        entry = {
            "prompt": prompt_text,
            "category": category,
            "responses": {},
        }

        for model_name in models:
            try:
                r = requests_lib.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": prompt_text,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()

                # Extract timing info
                total_duration = data.get("total_duration", 0)  # nanoseconds
                eval_count = data.get("eval_count", 0)
                eval_duration = data.get("eval_duration", 1)  # nanoseconds
                tokens_per_sec = (eval_count / eval_duration * 1e9) if eval_duration > 0 else 0

                entry["responses"][model_name] = {
                    "text": data.get("response", ""),
                    "tokens_per_second": round(tokens_per_sec, 1),
                    "total_tokens": eval_count,
                }
            except Exception as e:
                entry["responses"][model_name] = {
                    "text": f"[ERROR: {e}]",
                    "tokens_per_second": 0,
                    "total_tokens": 0,
                }

        results.append(entry)

    # Print side-by-side
    for entry in results:
        print(f"\nPrompt: \"{entry['prompt']}\"")
        for model_name, resp in entry["responses"].items():
            text_preview = resp["text"][:200].replace("\n", " ")
            print(f"  {model_name:30s} -> {text_preview}  ({resp['tokens_per_second']} tok/s)")

    # Save if requested
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json_module.dump(results, f, indent=2)

    return results


# Attach as a static-like method for testing prompt loading
def _load_prompts(path: str) -> list:
    with open(path) as f:
        return json_module.load(f)

compare._load_prompts = staticmethod(_load_prompts)
```

- [ ] **Step 5: Run tests to verify they pass**

Expected: TestCompare.test_loads_prompt_file PASSES. Ollama tests pass if ollama running, skip otherwise.

- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/benchmark.py testing/tests/test_benchmark.py testing/prompts/default.json
git commit -m "feat: add compare() for ollama generation comparison + default prompt set"
```

---

### Task 2: generation_metrics()

**Files:**
- Modify: `testing/llm_surgeon/benchmark.py`
- Modify: `testing/tests/test_benchmark.py`

- [ ] **Step 1: Write tests for generation_metrics()**

Add to `testing/tests/test_benchmark.py`:

```python
class TestGenerationMetrics:
    def test_returns_dict_per_model(self):
        """Test with synthetic results (no ollama needed)."""
        fake_results = [
            {
                "prompt": "Hello",
                "category": "test",
                "responses": {
                    "model_a": {"text": "Hello world this is a test response with varied words.", "tokens_per_second": 10, "total_tokens": 10},
                    "model_b": {"text": "the the the the the the the the the the", "tokens_per_second": 10, "total_tokens": 10},
                },
            },
        ]
        metrics = generation_metrics(fake_results)
        assert "model_a" in metrics
        assert "model_b" in metrics

    def test_repetition_rate_higher_for_repetitive(self):
        fake_results = [
            {
                "prompt": "test",
                "category": "test",
                "responses": {
                    "good": {"text": "The quick brown fox jumps over the lazy dog near the river.", "tokens_per_second": 10, "total_tokens": 12},
                    "bad": {"text": "the the the the the the the the the the the the", "tokens_per_second": 10, "total_tokens": 12},
                },
            },
        ]
        metrics = generation_metrics(fake_results)
        assert metrics["bad"]["repetition_rate"] > metrics["good"]["repetition_rate"]

    def test_vocab_diversity_lower_for_repetitive(self):
        fake_results = [
            {
                "prompt": "test",
                "category": "test",
                "responses": {
                    "good": {"text": "Paris is a beautiful city with many historic landmarks and museums.", "tokens_per_second": 10, "total_tokens": 11},
                    "bad": {"text": "Paris Paris Paris Paris Paris Paris Paris Paris Paris Paris", "tokens_per_second": 10, "total_tokens": 10},
                },
            },
        ]
        metrics = generation_metrics(fake_results)
        assert metrics["bad"]["vocab_diversity"] < metrics["good"]["vocab_diversity"]

    def test_has_expected_keys(self):
        fake_results = [
            {
                "prompt": "test",
                "category": "test",
                "responses": {
                    "model": {"text": "A normal response.", "tokens_per_second": 10, "total_tokens": 3},
                },
            },
        ]
        metrics = generation_metrics(fake_results)
        m = metrics["model"]
        assert "mean_output_length" in m
        assert "vocab_diversity" in m
        assert "repetition_rate" in m
        assert "coherence" in m
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — ImportError for `generation_metrics`

- [ ] **Step 3: Implement generation_metrics()**

Add to `testing/llm_surgeon/benchmark.py`:

```python
def generation_metrics(results: list) -> dict:
    """Compute heuristic quality metrics on generation comparison results.

    Metrics per model:
        mean_output_length: Average character length of responses
        vocab_diversity: Unique words / total words (0 to 1)
        repetition_rate: Fraction of repeated 3-grams
        coherence: Fraction of responses that are valid, non-empty text

    These are failure detectors, not quality metrics. A model passing all
    four can still be bad, but failing any means it's definitely broken.

    Args:
        results: Output from compare()

    Returns:
        Dict mapping model name to metrics dict.
    """
    # Collect all responses per model
    model_texts: dict = {}
    for entry in results:
        for model_name, resp in entry["responses"].items():
            if model_name not in model_texts:
                model_texts[model_name] = []
            model_texts[model_name].append(resp["text"])

    metrics = {}
    for model_name, texts in model_texts.items():
        # Mean output length
        lengths = [len(t) for t in texts]
        mean_length = sum(lengths) / len(lengths) if lengths else 0

        # Vocab diversity (unique words / total words)
        all_words = []
        for t in texts:
            all_words.extend(t.lower().split())
        if all_words:
            vocab_div = len(set(all_words)) / len(all_words)
        else:
            vocab_div = 0.0

        # Repetition rate (fraction of repeated 3-grams)
        all_ngrams = []
        for t in texts:
            words = t.lower().split()
            ngrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
            all_ngrams.extend(ngrams)
        if all_ngrams:
            unique_ngrams = set(all_ngrams)
            rep_rate = 1.0 - (len(unique_ngrams) / len(all_ngrams))
        else:
            rep_rate = 0.0

        # Coherence (fraction of non-empty, valid text responses)
        coherent = sum(
            1 for t in texts
            if t.strip() and not t.startswith("[ERROR") and t.isprintable()
        )
        coherence = coherent / len(texts) if texts else 0.0

        metrics[model_name] = {
            "mean_output_length": round(mean_length, 1),
            "vocab_diversity": round(vocab_div, 4),
            "repetition_rate": round(rep_rate, 4),
            "coherence": round(coherence, 4),
        }

    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

Expected: All generation_metrics tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`
Expected: All 137+ tests PASS.

- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/benchmark.py testing/tests/test_benchmark.py
git commit -m "feat: add generation_metrics for automated failure detection"
```

---

## Final State

```
testing/
  llm_surgeon/
    benchmark.py         — perplexity, eval_downstream, compare, generation_metrics
  tests/
    test_benchmark.py    — tests for all benchmark functions
  prompts/
    default.json         — 5 categorized test prompts
```
