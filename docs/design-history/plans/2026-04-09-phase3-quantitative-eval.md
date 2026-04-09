# Phase 3: Quantitative Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.
>
> **Tool rules (for subagents):**
> - Use Read (not cat/head/tail), Grep (not grep/rg/awk), Glob (not find/ls), Edit (not sed/awk) for all file operations
> - You are already in the project root (/home/ai/ai-projects/llm) — never cd
> - Python venv: `/home/ai/ai-projects/llm/testing/.venv/bin/python`
> - Run tests: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`

**Goal:** Add perplexity measurement and downstream task evaluation to `benchmark.py`, giving quantitative metrics for comparing original vs surgically modified models.

**Architecture:** Two functions in `benchmark.py`: `perplexity()` computes perplexity directly via PyTorch forward passes with sliding window; `eval_downstream()` shells out to EleutherAI's `lm-evaluation-harness`. Both work on HuggingFace models (not GGUF/ollama).

**Tech Stack:** PyTorch, HuggingFace datasets, lm-eval

**Reference:** `docs/superpowers/specs/2026-04-08-llm-surgeon-design.md` (v2), Phase 3 section of phase plan.

---

## File Map

```
testing/
  llm_surgeon/
    benchmark.py         — CREATE — perplexity, eval_downstream
    __init__.py          — MODIFY — add benchmark import
  tests/
    test_benchmark.py    — CREATE — tests for benchmark functions
```

---

### Task 1: Perplexity measurement

**Files:**
- Create: `testing/llm_surgeon/benchmark.py`
- Create: `testing/tests/test_benchmark.py`
- Modify: `testing/llm_surgeon/__init__.py`

- [ ] **Step 1: Write tests for perplexity**

Create `testing/tests/test_benchmark.py`:

```python
"""Tests for benchmark module."""

import pytest
import torch
from llm_surgeon.benchmark import perplexity
from llm_surgeon.surgery import remove_layers


class TestPerplexity:
    def test_returns_float(self, tiny_llama, tiny_llama_config):
        """Perplexity on a tiny model with random weights — just check it returns a number."""
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        # Create a minimal tokenizer for the tiny model
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        ppl = perplexity(tiny_llama, tokenizer, text="Hello world this is a test of the model")
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_perplexity_is_finite(self, tiny_llama, tiny_llama_config):
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        ppl = perplexity(tiny_llama, tokenizer, text="Hello world this is a test")
        assert not torch.isinf(torch.tensor(ppl))
        assert not torch.isnan(torch.tensor(ppl))

    def test_modified_model_different_perplexity(self, tiny_llama, tiny_llama_config):
        """Removing layers should change perplexity."""
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        text = "Hello world this is a test of the perplexity measurement"
        ppl_original = perplexity(tiny_llama, tokenizer, text=text)

        # Create a fresh model and modify it
        from transformers import LlamaForCausalLM
        model2 = LlamaForCausalLM(tiny_llama_config)
        model2.eval()
        # Copy weights so they start identical
        model2.load_state_dict(tiny_llama.state_dict())
        remove_layers(model2, [3, 4, 5])
        ppl_modified = perplexity(model2, tokenizer, text=text)

        # They should be different (removing layers changes the model)
        assert ppl_original != ppl_modified

    def test_warns_on_quantized_model(self, tiny_llama, tiny_llama_config):
        """Should warn if model appears to be quantized (has bnb config)."""
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        # Simulate quantized model by setting the config flag
        tiny_llama.config.quantization_config = {"quant_method": "bitsandbytes"}
        with pytest.warns(UserWarning, match="quantiz"):
            perplexity(tiny_llama, tokenizer, text="Hello world test")

    def test_dataset_mode(self, tiny_llama, tiny_llama_config):
        """Test perplexity with dataset='wikitext2' — loads a small slice."""
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        ppl = perplexity(tiny_llama, tokenizer, dataset="wikitext2", max_samples=2)
        assert isinstance(ppl, float)
        assert ppl > 0
```

Note: We need a `_make_tiny_tokenizer` helper. We'll add it to conftest.py as a module-level function (not a fixture) so tests can call it directly.

- [ ] **Step 2: Update conftest.py with tokenizer helper**

Add to `testing/tests/conftest.py` a module-level function:

```python
def _make_tiny_tokenizer(vocab_size=64):
    """Create a minimal tokenizer for testing. Returns a tokenizer with the given vocab size."""
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {f"tok{i}": i for i in range(vocab_size)}
    tokenizer_model = WordLevel(vocab=vocab, unk_token="tok0")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = Whitespace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="tok0",
        bos_token="tok1",
        eos_token="tok2",
        pad_token="tok0",
    )
    return fast_tokenizer
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_benchmark.py -v`
Expected: FAIL — `ImportError: cannot import name 'perplexity'`

- [ ] **Step 4: Implement perplexity**

Create `testing/llm_surgeon/benchmark.py`:

```python
"""Benchmarking: perplexity measurement and downstream evaluation."""

import warnings
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss


def perplexity(
    model,
    tokenizer,
    text: Optional[str] = None,
    dataset: Optional[str] = None,
    max_samples: Optional[int] = None,
    stride: Optional[int] = None,
) -> float:
    """Compute perplexity of a model on text or a standard dataset.

    Args:
        model: HuggingFace model (should be fp16/fp32, not quantized)
        tokenizer: Tokenizer matching the model
        text: Direct text to evaluate on
        dataset: Dataset name ('wikitext2' or 'c4'). Ignored if text is provided.
        max_samples: Max dataset samples to use (for speed)
        stride: Sliding window stride. Defaults to model's max_position_embeddings // 2.

    Returns:
        Perplexity as a float.
    """
    # Warn if model appears quantized
    if hasattr(model.config, "quantization_config") and model.config.quantization_config:
        warnings.warn(
            "Model appears to be quantized. Perplexity measurements on quantized models "
            "are noisy and may not accurately reflect surgery impact. "
            "Use eval or export mode (fp16) for reliable perplexity.",
            UserWarning,
        )

    if text is None and dataset is None:
        raise ValueError("Must provide either text or dataset")

    if text is None:
        text = _load_dataset_text(dataset, max_samples=max_samples)

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    max_length = getattr(model.config, "max_position_embeddings", 2048)
    if stride is None:
        stride = max_length // 2

    seq_len = input_ids.size(1)
    device = next(model.parameters()).device

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # only score the new tokens

        input_chunk = input_ids[:, begin:end].to(device)

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_chunk[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Only count the non-overlapping portion
        if begin > 0:
            overlap = begin + max_length - end if end < seq_len else 0
            loss = loss[-(target_len - 1):]

        nlls.append(loss.sum())
        prev_end = end

        if end == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    total_tokens = seq_len - 1  # exclude first token (no prediction for it)
    ppl = torch.exp(total_nll / total_tokens).item()

    return ppl


def _load_dataset_text(dataset_name: str, max_samples: Optional[int] = None) -> str:
    """Load text from a standard dataset."""
    from datasets import load_dataset

    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: wikitext2, c4")

    texts = []
    for i, item in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        t = item.get("text", "")
        if t.strip():
            texts.append(t)

    if not texts:
        raise RuntimeError(f"No text found in dataset {dataset_name}")

    return "\n\n".join(texts)
```

- [ ] **Step 5: Update __init__.py**

```python
"""LLM Surgeon — surgical layer-level manipulation of LLaMA models."""

from llm_surgeon import surgery, verify, export, benchmark
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_benchmark.py -v`
Expected: All perplexity tests PASS.

- [ ] **Step 7: Commit**

```bash
git add testing/llm_surgeon/benchmark.py testing/tests/test_benchmark.py testing/llm_surgeon/__init__.py testing/tests/conftest.py
git commit -m "feat: add perplexity measurement with sliding window and quantization warning"
```

---

### Task 2: eval_downstream (lm-evaluation-harness integration)

**Files:**
- Modify: `testing/llm_surgeon/benchmark.py`
- Modify: `testing/tests/test_benchmark.py`

- [ ] **Step 1: Write tests for eval_downstream**

Add to `testing/tests/test_benchmark.py`:

```python
from llm_surgeon.benchmark import eval_downstream


class TestEvalDownstream:
    def test_returns_dict(self, tiny_llama, tmp_path):
        """Run eval on tiny model with a fast task. Results will be random but the function should work."""
        # Save the model to a checkpoint (lm_eval needs a path)
        from llm_surgeon.export import save_checkpoint
        ckpt = str(tmp_path / "model")
        save_checkpoint(tiny_llama, ckpt)

        results = eval_downstream(
            model_path=ckpt,
            tasks=["arc_easy"],
            num_fewshot=0,
            limit=5,  # only evaluate 5 examples for speed
        )
        assert isinstance(results, dict)
        assert "arc_easy" in results

    def test_invalid_task_raises(self, tiny_llama, tmp_path):
        from llm_surgeon.export import save_checkpoint
        ckpt = str(tmp_path / "model")
        save_checkpoint(tiny_llama, ckpt)

        with pytest.raises(RuntimeError):
            eval_downstream(
                model_path=ckpt,
                tasks=["nonexistent_task_xyz"],
                num_fewshot=0,
                limit=5,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_benchmark.py::TestEvalDownstream -v`
Expected: FAIL — `ImportError: cannot import name 'eval_downstream'`

- [ ] **Step 3: Implement eval_downstream**

Add to `testing/llm_surgeon/benchmark.py`:

```python
import subprocess
import sys
import json as json_module


def eval_downstream(
    model_path: str,
    tasks: list,
    num_fewshot: int = 5,
    limit: Optional[int] = None,
) -> dict:
    """Evaluate a model on downstream tasks using lm-evaluation-harness.

    Args:
        model_path: Path to HuggingFace checkpoint (local) or model ID
        tasks: List of task names (e.g. ['arc_challenge', 'hellaswag'])
        num_fewshot: Number of few-shot examples
        limit: Max examples per task (for speed). None = full dataset.

    Returns:
        Dict mapping task name to accuracy (or main metric).
    """
    import tempfile

    tasks_str = ",".join(tasks)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "results.json")

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", tasks_str,
            "--num_fewshot", str(num_fewshot),
            "--output_path", tmpdir,
            "--log_samples",
        ]
        if limit is not None:
            cmd.extend(["--limit", str(limit)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"lm_eval failed (exit code {result.returncode}):\n"
                f"{result.stderr[-2000:]}"
            )

        # Parse results — lm_eval writes results to a subdirectory
        # Find the results JSON file
        results_dict = {}
        for root, dirs, files in os.walk(tmpdir):
            for fname in files:
                if fname == "results.json":
                    fpath = os.path.join(root, fname)
                    with open(fpath) as f:
                        data = json_module.load(f)
                    # Extract metrics per task
                    for task_name in tasks:
                        task_results = data.get("results", {}).get(task_name, {})
                        # Try common metric names
                        for metric_key in ["acc,none", "acc_norm,none", "acc", "exact_match,none"]:
                            if metric_key in task_results:
                                results_dict[task_name] = task_results[metric_key]
                                break
                        else:
                            # Fall back to first numeric metric
                            for k, v in task_results.items():
                                if isinstance(v, (int, float)) and "stderr" not in k:
                                    results_dict[task_name] = v
                                    break

        if not results_dict:
            raise RuntimeError(
                f"Could not parse lm_eval results. stdout:\n{result.stdout[-1000:]}"
            )

        return results_dict
```

Add `import os` at the top of benchmark.py if not already present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_benchmark.py::TestEvalDownstream -v --timeout=120`
Expected: `test_returns_dict` PASSES (slow — runs 5 arc_easy examples on tiny model). `test_invalid_task_raises` PASSES.

Note: The lm_eval test is slow even with limit=5 because it loads the evaluation harness. This is expected.

- [ ] **Step 5: Run full test suite**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`
Expected: All Phase 1 + Phase 2 + new benchmark tests PASS.

- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/benchmark.py testing/tests/test_benchmark.py
git commit -m "feat: add eval_downstream via lm-evaluation-harness"
```

---

## Final State

```
testing/
  llm_surgeon/
    __init__.py          — imports surgery, verify, export, benchmark
    surgery.py           — (Phase 1)
    verify.py            — (Phase 1)
    export.py            — (Phase 2)
    benchmark.py         — perplexity, eval_downstream
  tests/
    conftest.py          — tiny_llama fixture + _make_tiny_tokenizer helper
    test_surgery.py      — (Phase 1)
    test_verify.py       — (Phase 1)
    test_export.py       — (Phase 2)
    test_benchmark.py    — perplexity + eval_downstream tests
```

All tests pass. Perplexity + downstream evaluation ready.
