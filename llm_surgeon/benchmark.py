"""Quantitative evaluation: perplexity and downstream task benchmarks."""

import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from llm_surgeon.tracking import Experiment


# Downstream-eval defaults and helpers

FAST_TRIPLET: list[str] = ["hellaswag", "arc_easy", "arc_challenge"]

PAPER_STANDARD_FEWSHOT: dict[str, int] = {
    "hellaswag": 0,
    "arc_easy": 0,
    "arc_challenge": 25,
    "mmlu": 5,
}

PRIMARY_METRIC: dict[str, str] = {
    "hellaswag": "acc_norm",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "mmlu": "acc",
}
"""Metric reported per task by eval_downstream / eval_and_log.

Length-normalised accuracy for the multiple-choice completion tasks, as in
the Open LLM Leaderboard settings that PAPER_STANDARD_FEWSHOT follows.
Tasks not listed report ``acc``.
"""


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
    if isinstance(num_fewshot, bool):
        raise TypeError("num_fewshot must be an int, a dict or None, not bool.")
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
    Both harness paths run one lm_eval call per unique count.
    """
    buckets: dict[int, list[str]] = {}
    for task, n in fewshot_map.items():
        buckets.setdefault(n, []).append(task)
    return [(n, sorted(buckets[n])) for n in sorted(buckets)]


# Perplexity

DEFAULT_PPL_WINDOW: int = 2048
"""Default sliding-window length for :func:`perplexity`, in tokens.

The window is ``min(max_position_embeddings, DEFAULT_PPL_WINDOW)`` unless the
caller passes ``max_length``. Long-context configs (e.g. 131072 positions)
would otherwise put the whole corpus in one forward pass.
"""

C4_DEFAULT_MAX_SAMPLES: int = 256
"""Number of C4 validation documents read when ``max_samples`` is None.

C4 is streamed; without a bound the whole validation split would be read
into memory.
"""


def perplexity(
    model,
    tokenizer,
    text: str | None = None,
    dataset: str | None = None,
    max_samples: int | None = None,
    stride: int | None = None,
    verbose: bool = False,
    max_length: int | None = None,
) -> float:
    """Compute perplexity of *model* on the given text or dataset.

    Every token after the first is scored exactly once. Each window of
    *max_length* tokens scores only the tokens the previous window did not
    reach, conditioned on up to ``max_length - 1`` tokens of context.

    Args:
        model: A HuggingFace ``CausalLM`` model (already loaded, in eval mode).
        tokenizer: Matching tokenizer.
        text: Raw text string to evaluate on.  Mutually exclusive with *dataset*.
        dataset: Dataset shorthand — ``"wikitext2"`` or ``"c4"``.
        max_samples: Number of dataset rows to concatenate: raw wikitext-2
            rows (blank rows included), or C4 documents (default
            ``C4_DEFAULT_MAX_SAMPLES``).  None reads all of wikitext-2.
        stride: Sliding-window stride (tokens), ``0 < stride < max_length``
            (consecutive windows overlap so each window's first scored
            token has context).
            Defaults to ``max_length // 2``.
        max_length: Window length (tokens).  Defaults to
            ``min(max_position_embeddings, DEFAULT_PPL_WINDOW)``.

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

    # ---- Sliding window parameters -----------------------------------------
    if max_length is None:
        max_pos = int(getattr(model.config, "max_position_embeddings", 512))
        max_length = min(max_pos, DEFAULT_PPL_WINDOW)
    if max_length < 2:
        raise ValueError(f"max_length must be >= 2, got {max_length}.")
    if stride is None:
        stride = max_length // 2
    if not 0 < stride < max_length:
        raise ValueError(
            f"stride must satisfy 0 < stride < max_length ({max_length}), "
            f"got {stride}."
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
    try:
        encodings = tokenizer(text, return_tensors="pt")
    finally:
        tokenizer.model_max_length = _saved_max
    input_ids = encodings.input_ids  # shape (1, seq_len)

    seq_len = input_ids.size(1)
    # Use get_input_embeddings() for portability across HF architectures
    # (not just the LLaMA-specific model.model.embed_tokens path).
    device = model.get_input_embeddings().weight.device

    # ---- Sliding window NLL -------------------------------------------------
    nlls = []
    n_scored = 0
    prev_end = 0
    if seq_len <= max_length:
        total_windows = 1
    else:
        total_windows = math.ceil((seq_len - max_length) / stride) + 1
    window_idx = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        # Score tokens [target_begin, end). Token 0 has no context, and
        # tokens before prev_end were scored by an earlier window.
        target_begin = max(begin + 1, prev_end)
        chunk = input_ids[:, begin:end].to(device)

        with torch.no_grad():
            outputs = model(chunk)

        # The output at chunk position j predicts token begin + j + 1, so
        # the first scored token (target_begin) is predicted at j = rel_start.
        rel_start = target_begin - begin - 1
        sl = outputs.logits[:, rel_start:-1, :]  # (1, n_target, vocab)
        lb = chunk[:, rel_start + 1:]

        scored = lb.numel()
        prev_end = end
        if scored == 0:
            continue

        nll = loss_fct(sl.reshape(-1, sl.size(-1)).float(), lb.reshape(-1))
        nlls.append(nll.item())
        n_scored += scored
        window_idx += 1

        if verbose and (window_idx % 8 == 0 or end == seq_len):
            running_ppl = float(torch.exp(torch.tensor(sum(nlls) / max(1, n_scored))).item())
            print(f"  [perplexity] window {window_idx}/{total_windows} "
                  f"({end}/{seq_len} tokens, running ppl: {running_ppl:.2f})")

        if end == seq_len:
            break

    if not nlls:
        raise ValueError("No tokens were evaluated — text may be too short.")

    avg_nll = sum(nlls) / max(1, n_scored)
    return float(torch.exp(torch.tensor(avg_nll)).item())


def _load_dataset_text(name: str, max_samples: int | None = None) -> str:
    """Load and concatenate text from a HuggingFace dataset.

    wikitext2 follows the reference recipe, ``"\\n\\n".join(test["text"])``
    over the raw rows (blank rows included), so the result is comparable
    with published wikitext-2 perplexities; *max_samples* keeps the first N
    raw rows.  c4 reads the first *max_samples* validation documents
    (``C4_DEFAULT_MAX_SAMPLES`` when None).
    """
    from datasets import load_dataset

    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        rows = list(ds["text"])
        if max_samples is not None:
            rows = rows[:max_samples]
        return "\n\n".join(rows)
    if name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        n = C4_DEFAULT_MAX_SAMPLES if max_samples is None else max_samples
        docs = (ex["text"] for ex in itertools.islice(ds, n))
        return "\n\n".join(t for t in docs if t and t.strip())
    raise ValueError(f"Unknown dataset: '{name}'. Supported: 'wikitext2', 'c4'.")


# Downstream evaluation via lm-evaluation-harness

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

    Both paths resolve *num_fewshot* the same way (see ``_resolve_fewshot``):
    None uses ``PAPER_STANDARD_FEWSHOT`` per task, an int applies to every
    task, and a dict overrides per task.  Tasks with different counts run
    as separate harness calls.

    Returns:
        ``{task: score}`` using each task's primary metric from
        ``PRIMARY_METRIC`` (``acc`` for tasks not listed there).
    """
    if tasks is None:
        tasks = list(FAST_TRIPLET)
    full = _run_harness(
        model_path=model_path, model=model, tokenizer=tokenizer,
        tasks=tasks, num_fewshot=num_fewshot, limit=limit,
    )
    return _extract_accuracies(full, tasks)


def _run_harness(
    *,
    model_path: str | None,
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    num_fewshot: int | dict[str, int] | None,
    limit: int | None,
) -> dict[str, Any]:
    """Validate the model source and run lm_eval on the chosen path.

    Returns the merged harness output; ``full["effective_num_fewshot"]`` is
    the per-task few-shot count that actually ran.
    """
    if (model_path is None) == (model is None):
        raise ValueError(
            "Exactly one of model_path or model must be provided."
        )
    if model is not None and tokenizer is None:
        raise ValueError("tokenizer is required when model is provided.")

    if model is not None:
        return _in_process_eval(
            model=model, tokenizer=tokenizer,
            tasks=tasks, num_fewshot=num_fewshot, limit=limit,
        )
    assert model_path is not None
    return _subprocess_eval_grouped(
        model_path=model_path, tasks=tasks,
        num_fewshot=num_fewshot, limit=limit,
    )


def _merge_harness_outputs(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge the outputs of several harness calls (one per few-shot group).

    Dict-valued top-level keys (``results``, ``versions``, ``n-shot``,
    ``configs``, ``higher_is_better``, ...) are merged across groups.  The
    run-level ``config`` and other non-dict values come from the first group.
    """
    merged: dict[str, Any] = {"results": {}}
    for partial in partials:
        for key, value in partial.items():
            if key == "config" or not isinstance(value, dict):
                merged.setdefault(key, value)
            else:
                merged.setdefault(key, {}).update(value)
    return merged


def _subprocess_eval_full(
    *,
    model_path: str,
    tasks: list[str],
    num_fewshot: int,
    limit: int | None,
) -> dict[str, Any]:
    """Shell out to ``lm_eval`` CLI once and return the full output dict."""
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


def _subprocess_eval_grouped(
    *,
    model_path: str,
    tasks: list[str],
    num_fewshot: int | dict[str, int] | None,
    limit: int | None,
) -> dict[str, Any]:
    """Run the ``lm_eval`` CLI once per few-shot group and merge the outputs."""
    fewshot_map = _resolve_fewshot(tasks, num_fewshot)
    partials = [
        _subprocess_eval_full(
            model_path=model_path, tasks=group,
            num_fewshot=n, limit=limit,
        )
        for n, group in _group_by_fewshot(fewshot_map)
    ]
    merged = _merge_harness_outputs(partials)
    merged["effective_num_fewshot"] = fewshot_map
    return merged


def _in_process_eval(
    *,
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    num_fewshot: int | dict[str, int] | None,
    limit: int | None,
) -> dict[str, Any]:
    """Run lm_eval.simple_evaluate in-process against an in-memory model."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

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

    partials: list[dict[str, Any]] = []
    for n, group in _group_by_fewshot(fewshot_map):
        # pyright resolves simple_evaluate through lm_eval's lazy __getattr__
        # and can't see the real signature — runtime call is correct.
        partial: Any = simple_evaluate(model=lm, tasks=group, num_fewshot=n, limit=limit)  # pyright: ignore[reportCallIssue, reportArgumentType]
        partials.append(partial)
    merged = _merge_harness_outputs(partials)
    merged["effective_num_fewshot"] = fewshot_map
    return merged


def _serialize_harness_metrics(task_result: dict[str, Any]) -> dict[str, float]:
    """Flatten a single task's harness result into float-valued metrics.

    Keeps finite int/float values; drops strings, bools and NaN/inf.
    """
    out: dict[str, float] = {}
    for k, v in task_result.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if math.isfinite(v):
            out[k] = float(v)
    return out


def eval_and_log(
    experiment: "Experiment",
    *,
    model_path: str | None = None,
    model: Any = None,
    tokenizer: Any = None,
    tasks: list[str] | None = None,
    num_fewshot: int | dict[str, int] | None = None,
    limit: int | None = None,
) -> dict[str, float]:
    """Run eval_downstream and persist results to experiment tracking.

    The stored ``num_fewshot`` is the per-task count that actually ran.
    """
    if tasks is None:
        tasks = list(FAST_TRIPLET)

    full = _run_harness(
        model_path=model_path, model=model, tokenizer=tokenizer,
        tasks=tasks, num_fewshot=num_fewshot, limit=limit,
    )

    for task in tasks:
        task_result = full.get("results", {}).get(task, {})
        flat = _serialize_harness_metrics(task_result)
        for metric_key, value in flat.items():
            experiment.log_metric(f"harness.{task}.{metric_key}", value)

    from llm_surgeon.tracking import log_harness_result
    log_harness_result(
        db_path=experiment.db_path,
        experiment_name=experiment.name,
        tasks=tasks,
        num_fewshot=full["effective_num_fewshot"],
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
    """Extract per-task primary-metric scores from lm_eval JSON output.

    The metric for each task is ``PRIMARY_METRIC[task]`` (default ``acc``);
    if the harness did not report it, the other of ``acc`` / ``acc_norm``
    is used.
    """
    results = data.get("results", {})
    out: dict[str, float] = {}

    for task in tasks:
        if task not in results:
            raise RuntimeError(
                f"Task '{task}' not found in lm_eval results. "
                f"Available keys: {list(results.keys())}"
            )
        task_data = results[task]
        primary = PRIMARY_METRIC.get(task, "acc")
        metrics = [primary] + [m for m in ("acc", "acc_norm") if m != primary]
        for metric in metrics:
            key = next((k for k in (f"{metric},none", metric) if k in task_data), None)
            if key is not None:
                out[task] = float(task_data[key])
                break
        else:
            raise RuntimeError(
                f"Task '{task}' has no recognized accuracy key. "
                f"Expected one of acc,none / acc_norm,none / acc / acc_norm. "
                f"Got: {list(task_data.keys())}"
            )

    return out


# Generation comparison via Ollama

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
    host: str = "http://localhost:11434",
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
        host: Base URL of the Ollama server.

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
                            "error": str,  # only when the request failed
                        }
                    }
                },
                ...
            ]

        A failed request (connection error, timeout, HTTP error, bad JSON)
        is recorded with ``text=""`` and an ``error`` message instead of
        aborting the run.

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
            try:
                resp = requests.post(
                    f"{host.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                responses[model] = {
                    "text": "",
                    "tokens_per_second": 0.0,
                    "total_tokens": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue

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


# Automated generation quality metrics

def generation_metrics(results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute per-model failure-detection metrics from compare() output.

    These are diagnostic metrics intended to detect obvious generation
    failures, not to measure absolute quality.

    Metrics computed per model:

    - ``mean_output_length``: average character length of responses.
    - ``vocab_diversity``: ``unique_words / total_words`` per response,
      averaged over responses with words (0–1).
    - ``repetition_rate``: fraction of 3-grams repeated within a response,
      averaged over responses with at least 3 words
      (0 = no repetition, 1 = all repeated).
    - ``coherence``: fraction of responses that are non-empty, non-error
      (no ``error`` key from :func:`compare`), and at least 90% printable
      text (newlines, tabs and carriage returns count as printable).

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
            resp = entry["responses"].get(model, {})
            # A failed request scores as an empty (incoherent) response.
            text = "" if resp.get("error") else resp.get("text", "")
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


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _vocab_diversity(texts: list[str]) -> float:
    """Mean per-response ratio of unique words to total words (0–1).

    Computed per response so the score does not fall with the number or
    length of responses pooled together.
    """
    ratios = [len(set(w)) / len(w) for w in map(_words, texts) if w]
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def _repetition_rate(texts: list[str]) -> float:
    """Mean per-response fraction of 3-grams that are repeated.

    A 3-gram is "repeated" if it appears more than once in the same
    response.  Each response's rate is
    ``repeated_3gram_count / total_3gram_count``; responses with fewer
    than 3 words are skipped, and the result is 0 if none remain.  Phrases
    shared across different responses do not count.
    """
    rates: list[float] = []
    for words in map(_words, texts):
        if len(words) < 3:
            continue
        trigrams = list(zip(words, words[1:], words[2:]))
        counts: dict[tuple, int] = {}
        for tg in trigrams:
            counts[tg] = counts.get(tg, 0) + 1
        repeated = sum(1 for tg in trigrams if counts[tg] > 1)
        rates.append(repeated / len(trigrams))
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


_ALLOWED_WHITESPACE = frozenset("\n\t\r")


def _coherence(texts: list[str]) -> float:
    """Fraction of responses that are non-empty and contain printable text.

    ``str.isprintable`` is False for newlines and tabs, so those are
    counted as printable here; multi-line answers are not failures.
    """
    if not texts:
        return 0.0
    coherent = 0
    for text in texts:
        if not text or not text.strip():
            continue
        # Check that at least 90% of characters are printable
        printable_count = sum(
            1 for c in text if c.isprintable() or c in _ALLOWED_WHITESPACE
        )
        if printable_count / len(text) >= 0.9:
            coherent += 1
    return coherent / len(texts)
