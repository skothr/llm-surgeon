"""Quantitative evaluation: perplexity and downstream task benchmarks."""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Downstream-eval defaults and helpers (Phase 2)
# ---------------------------------------------------------------------------

FAST_TRIPLET: list[str] = ["hellaswag", "arc_easy", "arc_challenge"]

PAPER_STANDARD_FEWSHOT: dict[str, int] = {
    "hellaswag": 0,
    "arc_easy": 0,
    "arc_challenge": 25,
    "mmlu": 5,
}


def _resolve_fewshot(
    tasks: list[str],
    num_fewshot: int | dict[str, int] | None,
) -> dict[str, int]:
    """Resolve the (tasks, num_fewshot) pair into a full per-task dict.

    None -> PAPER_STANDARD_FEWSHOT per task, fallback 0 for unknown tasks.
    int -> applied uniformly.
    dict -> specified tasks use the dict value; unspecified fall back to
            PAPER_STANDARD_FEWSHOT (then 0).
    """
    if num_fewshot is None:
        return {t: PAPER_STANDARD_FEWSHOT.get(t, 0) for t in tasks}
    if isinstance(num_fewshot, int):
        return {t: num_fewshot for t in tasks}
    # dict: explicit > paper > 0
    out: dict[str, int] = {}
    for t in tasks:
        if t in num_fewshot:
            out[t] = num_fewshot[t]
        else:
            out[t] = PAPER_STANDARD_FEWSHOT.get(t, 0)
    return out


def _group_by_fewshot(fewshot_map: dict[str, int]) -> list[tuple[int, list[str]]]:
    """Group tasks sharing the same num_fewshot count.

    Returns a sorted list of (count, [task, ...]) pairs. Ordering is
    deterministic: ascending by count, then by task name inside each group.
    This lets _in_process_eval call simple_evaluate once per unique count.
    """
    buckets: dict[int, list[str]] = {}
    for task, n in fewshot_map.items():
        buckets.setdefault(n, []).append(task)
    return [(n, sorted(buckets[n])) for n in sorted(buckets)]


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def perplexity(
    model,
    tokenizer,
    text: str | None = None,
    dataset: str | None = None,
    max_samples: int | None = None,
    stride: int | None = None,
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
    max_length: int = int(getattr(model.config, "max_position_embeddings", 512))
    if stride is None:
        stride = max_length // 2
    assert stride is not None

    seq_len = input_ids.size(1)
    # Use get_input_embeddings() for portability across HF architectures
    # (not just the LLaMA-specific model.model.embed_tokens path).
    device = model.get_input_embeddings().weight.device

    # ---- Sliding window NLL -------------------------------------------------
    nlls = []
    n_scored = 0
    prev_end = 0
    total_windows = (seq_len - 1) // stride + 1
    window_idx = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_begin = max(begin, prev_end)

        chunk = input_ids[:, begin:end].to(device)
        target_len = end - target_begin

        if target_len <= 0:
            prev_end = end
            continue

        with torch.no_grad():
            outputs = model(chunk, labels=chunk)

        # NLL only over the non-overlapping suffix. Re-compute from logits
        # because outputs.loss is averaged across the full chunk.
        logits = outputs.logits  # (1, chunk_len, vocab)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        rel_start = target_begin - begin
        sl = shift_logits[:, rel_start:, :]
        lb = shift_labels[:, rel_start:]

        scored = lb.numel()
        if scored == 0:
            prev_end = end
            continue

        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        nll = loss_fct(sl.view(-1, sl.size(-1)), lb.view(-1))
        nlls.append(nll.item())
        n_scored += scored
        window_idx += 1

        if verbose and (window_idx % 8 == 0 or end == seq_len):
            running_ppl = float(torch.exp(torch.tensor(sum(nlls) / max(1, n_scored))).item())
            print(f"  [perplexity] window {window_idx}/{total_windows} "
                  f"({end}/{seq_len} tokens, running ppl: {running_ppl:.2f})")

        prev_end = end
        if end == seq_len:
            break

    if not nlls:
        raise ValueError("No tokens were evaluated — text may be too short.")

    avg_nll = sum(nlls) / max(1, n_scored)
    return float(torch.exp(torch.tensor(avg_nll)).item())


def _load_dataset_text(name: str, max_samples: int | None = None) -> str:
    """Load and concatenate text from a HuggingFace dataset."""
    from datasets import load_dataset  # pyright: ignore[reportMissingImports]

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
    tasks: list[str] | None = None,
    *,
    model_path: str | None = None,
    model: Any = None,
    tokenizer: Any = None,
    num_fewshot: int | dict[str, int] | None = None,
    limit: int | None = None,
) -> dict[str, float]:
    """Evaluate a HuggingFace checkpoint on downstream tasks via lm-eval.

    Exactly one of *model_path* or *model* must be given:

    - ``model_path=...`` -> shells out to ``lm_eval`` CLI (legacy path).
    - ``model=..., tokenizer=...`` -> runs in-process via ``HFLM`` (new).
    """
    # Validate model-source arguments.
    if (model_path is None) == (model is None):
        raise ValueError(
            "Exactly one of model_path or model must be provided."
        )
    if model is not None and tokenizer is None:
        raise ValueError("tokenizer is required when model is provided.")

    if tasks is None:
        tasks = list(FAST_TRIPLET)

    if model is not None:
        full_result = _in_process_eval(
            model=model, tokenizer=tokenizer,
            tasks=tasks, num_fewshot=num_fewshot, limit=limit,
        )
        return _extract_accuracies(full_result, tasks)

    # Subprocess path (legacy).
    if isinstance(num_fewshot, dict):
        raise ValueError(
            "Dict num_fewshot is only supported with in-memory model; "
            "pass an int or None for the model_path subprocess path."
        )
    effective_nf = num_fewshot if num_fewshot is not None else 0
    assert model_path is not None
    return _subprocess_eval(
        model_path=model_path,
        tasks=tasks,
        num_fewshot=effective_nf,
        limit=limit,
    )


def _subprocess_eval_full(
    *,
    model_path: str,
    tasks: list[str],
    num_fewshot: int,
    limit: int | None,
) -> dict[str, Any]:
    """Shell out to ``lm_eval`` CLI and return the full output dict.

    Core subprocess path; both `_subprocess_eval` (narrowed) and
    `eval_and_log`'s subprocess branch build on this.
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
        env = os.environ.copy()
        for key in list(env):
            if key.upper() in (
                "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
                "NO_PROXY", "FTP_PROXY",
            ):
                env.pop(key, None)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"lm_eval failed (exit {result.returncode}).\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )
        return _find_and_parse_results(tmpdir)


def _subprocess_eval(
    *,
    model_path: str,
    tasks: list[str],
    num_fewshot: int,
    limit: int | None,
) -> dict[str, float]:
    """Narrowed-output subprocess path — delegates to `_subprocess_eval_full`."""
    results_data = _subprocess_eval_full(
        model_path=model_path, tasks=tasks,
        num_fewshot=num_fewshot, limit=limit,
    )
    return _extract_accuracies(results_data, tasks)


def _in_process_eval(
    *,
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    num_fewshot: int | dict[str, int] | None,
    limit: int | None,
) -> dict[str, Any]:
    """Run lm_eval.simple_evaluate in-process against an in-memory model."""
    from lm_eval import simple_evaluate  # pyright: ignore[reportMissingImports]
    from lm_eval.models.huggingface import HFLM  # pyright: ignore[reportMissingImports]

    cfg = getattr(model, "config", None)
    if getattr(cfg, "quantization_config", None):
        warnings.warn(
            "Model has quantization_config set — harness accuracy on "
            "quantized models is typically 1-3 pp below fp16 reference "
            "numbers.",
            UserWarning, stacklevel=3,
        )

    lm = HFLM(pretrained=model, tokenizer=tokenizer)  # pyright: ignore[reportCallIssue]
    fewshot_map = _resolve_fewshot(tasks, num_fewshot)

    merged: dict[str, Any] = {"results": {}, "config": None}
    for n, group in _group_by_fewshot(fewshot_map):
        # pyright resolves simple_evaluate through lm_eval's lazy __getattr__
        # and can't see the real signature — runtime call is correct.
        partial: Any = simple_evaluate(model=lm, tasks=group, num_fewshot=n, limit=limit)  # pyright: ignore[reportCallIssue, reportArgumentType]
        merged["results"].update(partial["results"])
        if merged["config"] is None:
            merged["config"] = partial.get("config", {})
    merged["effective_num_fewshot"] = fewshot_map
    return merged


def _serialize_harness_metrics(task_result: dict[str, Any]) -> dict[str, float]:
    """Flatten a single task's harness result into float-valued metrics."""
    out: dict[str, float] = {}
    for k, v in task_result.items():
        if isinstance(v, (int, float)) and not (
            isinstance(v, float) and math.isnan(v)
        ):
            out[k] = float(v)
    return out


def eval_and_log(
    experiment: Any,
    *,
    model_path: str | None = None,
    model: Any = None,
    tokenizer: Any = None,
    tasks: list[str] | None = None,
    num_fewshot: int | dict[str, int] | None = None,
    limit: int | None = None,
) -> dict[str, float]:
    """Run eval_downstream and persist results to experiment tracking."""
    if (model_path is None) == (model is None):
        raise ValueError(
            "Exactly one of model_path or model must be provided."
        )
    if model is not None and tokenizer is None:
        raise ValueError("tokenizer is required when model is provided.")

    if tasks is None:
        tasks = list(FAST_TRIPLET)

    if model is not None:
        full = _in_process_eval(
            model=model, tokenizer=tokenizer,
            tasks=tasks, num_fewshot=num_fewshot, limit=limit,
        )
    else:
        if isinstance(num_fewshot, dict):
            raise ValueError(
                "Dict num_fewshot is only supported with in-memory model."
            )
        effective_nf = num_fewshot if num_fewshot is not None else 0
        assert model_path is not None
        full = _subprocess_eval_full(
            model_path=model_path, tasks=tasks,
            num_fewshot=effective_nf, limit=limit,
        )

    for task in tasks:
        task_result = full.get("results", {}).get(task, {})
        flat = _serialize_harness_metrics(task_result)
        for metric_key, value in flat.items():
            experiment.log_metric(f"harness.{task}.{metric_key}", value)

    from llm_surgeon.tracking import log_harness_result
    if isinstance(num_fewshot, int):
        nf_to_store: Any = num_fewshot
    else:
        nf_to_store = _resolve_fewshot(tasks, num_fewshot)
    log_harness_result(
        db_path=experiment.db_path,
        experiment_name=experiment.name,
        tasks=tasks,
        num_fewshot=nf_to_store,
        limit=limit,
        result=full,
    )

    return _extract_accuracies(full, tasks)


def _find_and_parse_results(output_dir: str) -> dict:
    """Recursively search for results JSON produced by lm_eval."""
    for root, _dirs, files in os.walk(output_dir):
        for fname in sorted(files, reverse=True):
            if fname.startswith("results") and fname.endswith(".json"):
                path = os.path.join(root, fname)
                with open(path) as f:
                    return json.load(f)
    raise RuntimeError(
        f"No results JSON found under {output_dir}. "
        "lm_eval may have failed silently."
    )


def _extract_accuracies(data: dict, tasks: list[str]) -> dict[str, float]:
    """Extract per-task accuracy from lm_eval JSON output."""
    results = data.get("results", {})
    out: dict[str, float] = {}

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
            raise RuntimeError(
                f"Task '{task}' has no recognized accuracy key. "
                f"Expected one of acc,none / acc_norm,none / acc / acc_norm. "
                f"Got: {list(task_data.keys())}"
            )

    return out


# ---------------------------------------------------------------------------
# Generation comparison via Ollama
# ---------------------------------------------------------------------------

def _load_prompts(path: str) -> list[dict[str, str]]:
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
    models: list[str],
    prompts: list[dict[str, str]] | str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    output_file: str | None = None,
) -> list[dict[str, Any]]:
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

    results: list[dict[str, Any]] = []

    for prompt_entry in prompts:
        prompt_text = prompt_entry["prompt"]
        category = prompt_entry.get("category", "")

        responses: dict[str, dict[str, Any]] = {}

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
    results: list[dict[str, Any]],
    models: list[str],
) -> None:
    """Print a human-readable side-by-side comparison of model responses."""
    col_width = 60
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

def generation_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
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

    model_names: list[str] = list(results[0]["responses"].keys())

    # Collect all texts per model
    texts_per_model: dict[str, list[str]] = {m: [] for m in model_names}
    for entry in results:
        for model in model_names:
            text = entry["responses"].get(model, {}).get("text", "")
            texts_per_model[model].append(text)

    out: dict[str, dict[str, float]] = {}
    for model, texts in texts_per_model.items():
        out[model] = {
            "mean_output_length": _mean_output_length(texts),
            "vocab_diversity": _vocab_diversity(texts),
            "repetition_rate": _repetition_rate(texts),
            "coherence": _coherence(texts),
        }

    return out


def _mean_output_length(texts: list[str]) -> float:
    """Average character length across a list of response strings."""
    if not texts:
        return 0.0
    return sum(len(t) for t in texts) / len(texts)


def _vocab_diversity(texts: list[str]) -> float:
    """Ratio of unique words to total words across all texts (0–1)."""
    all_words: list[str] = []
    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        all_words.extend(words)
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def _repetition_rate(texts: list[str]) -> float:
    """Fraction of 3-grams that are repeated within the combined text.

    A 3-gram is "repeated" if it appears more than once.  The rate is
    ``repeated_3gram_count / total_3gram_count``, or 0 if there are
    fewer than 3 words total.
    """
    all_words: list[str] = []
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
    counts: dict[tuple, int] = {}
    for tg in trigrams:
        counts[tg] = counts.get(tg, 0) + 1

    repeated = sum(1 for tg in trigrams if counts[tg] > 1)
    return repeated / total


def _coherence(texts: list[str]) -> float:
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
