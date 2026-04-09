"""Quantitative evaluation: perplexity and downstream task benchmarks."""

import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def perplexity(
    model,
    tokenizer,
    text: Optional[str] = None,
    dataset: Optional[str] = None,
    max_samples: Optional[int] = None,
    stride: Optional[int] = None,
    verbose: bool = False,
) -> float:
    """Compute perplexity of *model* on the given text or dataset.

    Args:
        model: A HuggingFace ``CausalLM`` model (already loaded, in eval mode).
        tokenizer: Matching tokenizer.
        text: Raw text string to evaluate on.  Mutually exclusive with *dataset*.
        dataset: Dataset shorthand — ``"wikitext2"`` or ``"c4"``.
        max_samples: Maximum number of dataset examples to concatenate.
        stride: Sliding-window stride (tokens).  Defaults to
            ``max_position_embeddings // 2``.

    Returns:
        Perplexity as a ``float``.

    Warns:
        UserWarning — if the model config contains ``quantization_config``
        (perplexity measurement on quantised models is unreliable).
    """
    if text is None and dataset is None:
        raise ValueError("Provide either 'text' or 'dataset'.")
    if text is not None and dataset is not None:
        raise ValueError("Provide either 'text' or 'dataset', not both.")

    # Warn if the model appears quantized
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        if model.config.quantization_config:
            warnings.warn(
                "Model has quantization_config set — perplexity on quantized "
                "models is noisy and may not reflect true model quality.",
                UserWarning,
                stacklevel=2,
            )

    # ---- Resolve text -------------------------------------------------------
    if dataset is not None:
        text = _load_dataset_text(dataset, max_samples=max_samples)
    # else text is already set

    # ---- Tokenize -----------------------------------------------------------
    # Tokenize the full text without truncation — the sliding window below
    # handles chunking. Temporarily raise model_max_length so the tokenizer
    # doesn't warn about sequence length exceeding max_position_embeddings.
    _saved_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e12)
    encodings = tokenizer(text, return_tensors="pt")
    tokenizer.model_max_length = _saved_max
    input_ids = encodings.input_ids  # shape (1, seq_len)

    # ---- Sliding window parameters -----------------------------------------
    max_length = getattr(model.config, "max_position_embeddings", 512)
    if stride is None:
        stride = max_length // 2

    seq_len = input_ids.size(1)
    device = model.model.embed_tokens.weight.device

    # ---- Sliding window NLL -------------------------------------------------
    nlls = []
    prev_end = 0
    total_windows = (seq_len - 1) // stride + 1
    window_idx = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        # The "target" tokens are those not covered by the previous window
        target_begin = max(begin, prev_end)

        chunk = input_ids[:, begin:end].to(device)
        target_len = end - target_begin

        if target_len <= 0:
            prev_end = end
            continue

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)

        # outputs.loss is the mean NLL over all positions in the chunk.
        # We want the NLL only over the non-overlapping suffix.
        # Re-compute using the logits for precision.
        logits = outputs.logits  # (1, chunk_len, vocab)
        # Shift: logits[0..T-1] predict tokens[1..T]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        # Only count positions that fall in the non-overlapping part
        # relative positions within the chunk: [target_begin - begin : end - begin - 1]
        rel_start = target_begin - begin  # first position we care about
        # shift_labels has length (end - begin - 1)
        # we want positions rel_start .. (end - begin - 1)
        sl = shift_logits[:, rel_start:, :]  # (1, target_len, vocab) approximately
        lb = shift_labels[:, rel_start:]     # (1, target_len)

        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        nll = loss_fct(sl.view(-1, sl.size(-1)), lb.view(-1))
        nlls.append(nll.item())
        window_idx += 1

        if verbose and (window_idx % 8 == 0 or end == seq_len):
            running_ppl = float(torch.exp(torch.tensor(sum(nlls) / max(1, _count_scored_tokens_partial(input_ids, max_length, stride, end)))).item())
            print(f"  [perplexity] window {window_idx}/{total_windows} "
                  f"({end}/{seq_len} tokens, running ppl: {running_ppl:.2f})")

        prev_end = end
        if end == seq_len:
            break

    if not nlls:
        raise ValueError("No tokens were evaluated — text may be too short.")

    total_nll = sum(nlls)
    # Total number of predicted tokens (denominator)
    # prev_end - 1 because the last token has no next token to predict
    n_tokens = max(prev_end - 1, 1)
    # More precisely: count tokens we actually scored
    # Recount from the window logic: it's the sum of target_len values, but
    # adjusted for the shift.  Approximate with the closed-form version.
    # Simpler: count tokens directly.
    avg_nll = total_nll / max(1, _count_scored_tokens(input_ids, max_length, stride))
    return float(torch.exp(torch.tensor(avg_nll)).item())


def _count_scored_tokens_partial(input_ids: torch.Tensor, max_length: int, stride: int, up_to: int) -> int:
    """Count scored tokens up to a given position (for running ppl display)."""
    seq_len = input_ids.size(1)
    total = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        if end > up_to:
            end = up_to
        target_begin = max(begin, prev_end)
        target_len = end - target_begin
        if target_len > 0:
            total += target_len
        prev_end = end
        if end >= up_to:
            break
    return max(total, 1)


def _count_scored_tokens(input_ids: torch.Tensor, max_length: int, stride: int) -> int:
    """Count total non-overlapping tokens scored across all windows."""
    seq_len = input_ids.size(1)
    total = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_begin = max(begin, prev_end)
        target_len = end - target_begin
        if target_len > 0:
            # Subtract 1 because shifted prediction: last token has no label
            total += max(0, target_len - (1 if end == seq_len and begin == 0 else 0))
        prev_end = end
        if end == seq_len:
            break
    return max(total, 1)


def _load_dataset_text(name: str, max_samples: Optional[int] = None) -> str:
    """Load and concatenate text from a HuggingFace dataset."""
    from datasets import load_dataset

    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = ds["text"]
    elif name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = (ex["text"] for ex in ds)
    else:
        raise ValueError(f"Unknown dataset: '{name}'. Supported: 'wikitext2', 'c4'.")

    if max_samples is not None:
        collected = []
        for i, t in enumerate(texts):
            if i >= max_samples:
                break
            collected.append(t)
        texts = collected

    return "\n\n".join(t for t in texts if t and t.strip())


# ---------------------------------------------------------------------------
# Downstream evaluation via lm-evaluation-harness
# ---------------------------------------------------------------------------

def eval_downstream(
    model_path: str,
    tasks: List[str],
    num_fewshot: int = 5,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a HuggingFace checkpoint on downstream tasks via lm-eval.

    Shells out to ``lm_eval`` CLI so the harness manages its own model loading.

    Args:
        model_path: Local path or HuggingFace model ID.
        tasks: List of lm-eval task names (e.g. ``["arc_easy", "hellaswag"]``).
        num_fewshot: Number of few-shot examples (default 5).
        limit: Cap examples per task (useful for fast testing).

    Returns:
        ``dict`` mapping task name to accuracy (``float``).

    Raises:
        RuntimeError: If lm_eval exits with a non-zero return code or the
            output cannot be parsed.
    """
    tasks_str = ",".join(tasks)

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", tasks_str,
            "--num_fewshot", str(num_fewshot),
            "--output_path", tmpdir,
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]

        # Build a clean environment: remove SOCKS/HTTP proxy variables that can
        # cause socksio ImportError when socksio is not installed.
        env = os.environ.copy()
        for key in list(env):
            if key.upper() in (
                "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                "NO_PROXY", "FTP_PROXY",
            ):
                env.pop(key, None)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"lm_eval failed (exit {result.returncode}).\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        # lm_eval writes results JSON to <output_path>/<model_name>/results.json
        # Walk the tmpdir to find it
        results_data = _find_and_parse_results(tmpdir)
        return _extract_accuracies(results_data, tasks)


def _find_and_parse_results(output_dir: str) -> dict:
    """Recursively search for results.json produced by lm_eval."""
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname == "results.json":
                path = os.path.join(root, fname)
                with open(path) as f:
                    return json.load(f)
    raise RuntimeError(
        f"No results.json found under {output_dir}. "
        "lm_eval may have failed silently."
    )


def _extract_accuracies(data: dict, tasks: List[str]) -> Dict[str, float]:
    """Extract per-task accuracy from lm_eval JSON output."""
    results = data.get("results", {})
    out: Dict[str, float] = {}

    for task in tasks:
        if task not in results:
            # Try with comma-separated group fallback
            raise RuntimeError(
                f"Task '{task}' not found in lm_eval results. "
                f"Available keys: {list(results.keys())}"
            )
        task_data = results[task]
        # lm_eval stores accuracy under different keys depending on the task
        for key in ("acc,none", "acc_norm,none", "acc", "acc_norm"):
            if key in task_data:
                out[task] = float(task_data[key])
                break
        else:
            # Return first numeric value
            for v in task_data.values():
                if isinstance(v, (int, float)):
                    out[task] = float(v)
                    break
            else:
                out[task] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Generation comparison via Ollama
# ---------------------------------------------------------------------------

def _load_prompts(path: str) -> List[Dict[str, str]]:
    """Load a prompt list from a JSON file.

    Args:
        path: Path to a JSON file containing a list of dicts with at least
              a ``"prompt"`` key and optionally a ``"category"`` key.

    Returns:
        List of prompt dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(
    models: List[str],
    prompts: Union[List[Dict[str, str]], str],
    temperature: float = 0.0,
    max_tokens: int = 256,
    output_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compare multiple ollama models across a set of prompts.

    Sends each prompt to each model via the Ollama HTTP API and collects the
    generated text and timing statistics.

    Args:
        models: List of ollama model names (e.g. ``["tinyllama", "mistral"]``).
        prompts: Either a list of dicts ``[{"prompt": str, "category": str}]``
                 or a path to a JSON file in that format.
        temperature: Sampling temperature.  ``0.0`` is near-deterministic.
        max_tokens: Maximum number of tokens to generate per response.
        output_file: If provided, write results as JSON to this path.

    Returns:
        List of result dicts, one per prompt::

            [
                {
                    "prompt": str,
                    "category": str,
                    "responses": {
                        "<model>": {
                            "text": str,
                            "tokens_per_second": float,
                            "total_tokens": int,
                        }
                    }
                },
                ...
            ]

    Note:
        ``temperature=0.0`` is near-deterministic but not exact due to
        quantized parallel reduction in GPU matrix operations.
    """
    import requests

    if isinstance(prompts, str):
        prompts = _load_prompts(prompts)

    results: List[Dict[str, Any]] = []

    for prompt_entry in prompts:
        prompt_text = prompt_entry["prompt"]
        category = prompt_entry.get("category", "")

        responses: Dict[str, Dict[str, Any]] = {}

        for model in models:
            payload = {
                "model": model,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data.get("response", "")
            eval_count = data.get("eval_count", 0)
            eval_duration_ns = data.get("eval_duration", 0)

            # eval_duration is in nanoseconds; compute tokens/second
            if eval_duration_ns and eval_duration_ns > 0:
                tps = eval_count / (eval_duration_ns / 1e9)
            else:
                tps = 0.0

            responses[model] = {
                "text": text,
                "tokens_per_second": float(tps),
                "total_tokens": int(eval_count),
            }

        results.append({
            "prompt": prompt_text,
            "category": category,
            "responses": responses,
        })

    # Print side-by-side summary
    _print_compare_summary(results, models)

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def _print_compare_summary(
    results: List[Dict[str, Any]],
    models: List[str],
) -> None:
    """Print a human-readable side-by-side comparison of model responses."""
    col_width = 60
    header = " | ".join(f"{m:<{col_width}}" for m in models)
    separator = "-+-".join("-" * col_width for _ in models)

    print("\n=== Generation Comparison ===\n")
    for entry in results:
        print(f"[{entry['category']}] {entry['prompt']!r}")
        print(separator)
        # Truncate long responses for display
        cols = []
        for model in models:
            text = entry["responses"].get(model, {}).get("text", "")
            tps = entry["responses"].get(model, {}).get("tokens_per_second", 0.0)
            snippet = text[:col_width - 10].replace("\n", " ")
            cols.append(f"{snippet:<{col_width - 10}}  {tps:5.1f}t/s")
        print(" | ".join(cols))
        print()


# ---------------------------------------------------------------------------
# Automated generation quality metrics
# ---------------------------------------------------------------------------

def generation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute per-model failure-detection metrics from compare() output.

    These are diagnostic metrics intended to detect obvious generation
    failures, not to measure absolute quality.

    Metrics computed per model:

    - ``mean_output_length``: average character length of responses.
    - ``vocab_diversity``: ``unique_words / total_words`` (0–1).
    - ``repetition_rate``: fraction of 3-grams that are repeated
      (0 = no repetition, 1 = all repeated).
    - ``coherence``: fraction of responses that are non-empty,
      non-error, and contain only printable text.

    Args:
        results: Output from :func:`compare`.

    Returns:
        Dict mapping model name to a dict of metric name → float value.
    """
    # Gather all model names from the first result entry
    if not results:
        return {}

    model_names: List[str] = list(results[0]["responses"].keys())

    # Collect all texts per model
    texts_per_model: Dict[str, List[str]] = {m: [] for m in model_names}
    for entry in results:
        for model in model_names:
            text = entry["responses"].get(model, {}).get("text", "")
            texts_per_model[model].append(text)

    out: Dict[str, Dict[str, float]] = {}
    for model, texts in texts_per_model.items():
        out[model] = {
            "mean_output_length": _mean_output_length(texts),
            "vocab_diversity": _vocab_diversity(texts),
            "repetition_rate": _repetition_rate(texts),
            "coherence": _coherence(texts),
        }

    return out


def _mean_output_length(texts: List[str]) -> float:
    """Average character length across a list of response strings."""
    if not texts:
        return 0.0
    return sum(len(t) for t in texts) / len(texts)


def _vocab_diversity(texts: List[str]) -> float:
    """Ratio of unique words to total words across all texts (0–1)."""
    all_words: List[str] = []
    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        all_words.extend(words)
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def _repetition_rate(texts: List[str]) -> float:
    """Fraction of 3-grams that are repeated within the combined text.

    A 3-gram is "repeated" if it appears more than once.  The rate is
    ``repeated_3gram_count / total_3gram_count``, or 0 if there are
    fewer than 3 words total.
    """
    all_words: List[str] = []
    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        all_words.extend(words)

    if len(all_words) < 3:
        return 0.0

    trigrams = [
        (all_words[i], all_words[i + 1], all_words[i + 2])
        for i in range(len(all_words) - 2)
    ]
    total = len(trigrams)
    counts: Dict[tuple, int] = {}
    for tg in trigrams:
        counts[tg] = counts.get(tg, 0) + 1

    repeated = sum(1 for tg in trigrams if counts[tg] > 1)
    return repeated / total


def _coherence(texts: List[str]) -> float:
    """Fraction of responses that are non-empty and contain printable text."""
    if not texts:
        return 0.0
    coherent = 0
    for text in texts:
        if not text or not text.strip():
            continue
        # Check that at least 90% of characters are printable
        printable_count = sum(1 for c in text if c.isprintable())
        if printable_count / len(text) >= 0.9:
            coherent += 1
    return coherent / len(texts)
