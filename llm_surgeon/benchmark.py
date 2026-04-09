"""Quantitative evaluation: perplexity and downstream task benchmarks."""

import json
import os
import subprocess
import sys
import tempfile
import warnings
from typing import Dict, List, Optional

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
    encodings = tokenizer(text, return_tensors="pt")
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
