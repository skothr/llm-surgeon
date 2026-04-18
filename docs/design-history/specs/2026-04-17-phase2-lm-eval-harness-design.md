# Phase 2 — lm-evaluation-harness integration (Design Spec)

**Date:** 2026-04-17
**Status:** Approved for planning
**Roadmap ref:** `project_llm_surgeon_roadmap.md` — Phase 2
**Preceded by:** Phase 1 model-loading refactor (shipped 2026-04-17; commits `8901e8b` … `4995069`)

## Problem

`llm_surgeon/benchmark.py::eval_downstream` already integrates the `lm-evaluation-harness` — but via a subprocess `sys.executable -m lm_eval --model hf --model_args pretrained=<path>`. That requires a **disk-resident checkpoint**, which is wrong for the Phase 2 goal: *"any surgically-modified in-memory checkpoint becomes instantly comparable."* A user who runs `load_model → remove_layers → eval_downstream` today has to `save_pretrained()` to a temp dir first — a multi-GB round trip.

Additionally:

- The harness output is thrown away: `eval_downstream` returns only a `{task: accuracy}` dict. Stderr, confidence intervals, config metadata, and sample counts are discarded.
- Nothing writes to `tracking.py` — the caller must manually `experiment.log_metric("hellaswag_acc", 0.61)`.
- Few-shot is hardcoded `5` uniform, ignoring paper-standard per-task conventions (0-shot HellaSwag, 25-shot ARC-c).
- No sensible default task set for a "quick baseline" comparison.

## Goal

1. Upgrade `eval_downstream` to accept **in-memory** `model` + `tokenizer` via `lm_eval.simple_evaluate(model=HFLM(pretrained=model, tokenizer=tokenizer))`.
2. Preserve the existing subprocess path for callers passing `model_path=...`.
3. Add `eval_and_log(experiment, ...)` that runs evaluation and writes results to both the flat `metrics` table and a new `harness_results` blob table.
4. Default to a fast triplet (`hellaswag`, `arc_easy`, `arc_challenge`) and paper-standard few-shot per task.
5. Warn — don't block — when a quantized model is evaluated.

## Non-goals

- Perplexity (`perplexity()` stays unchanged).
- Generation-quality metrics (`compare()`, `generation_metrics()`).
- New module — everything lands in existing files.
- Phase 3 (activation patching).
- `lm_eval` dependency install — `0.4.11` is already in `testing/.venv`.

## Decisions

| # | Question | Choice | Why |
|---|---|---|---|
| 1 | Integration style | Upgrade `eval_downstream` in-place; add `eval_and_log`; keep subprocess fallback | Preserves existing callers; new in-process path is the genuinely new capability. |
| 2 | Default task set | `FAST_TRIPLET = ["hellaswag", "arc_easy", "arc_challenge"]` | Full MMLU is ~20 min on TinyLlama/2080 — too slow for the iteration loop. MMLU remains opt-in. |
| 3 | Few-shot policy | Per-task paper-standard via `PAPER_STANDARD_FEWSHOT` dict; `num_fewshot=` kwarg overrides | External comparability requires per-task conventions. |
| 4 | SQLite storage | Flat `metrics` rows (`harness.<task>.<metric>`) + full JSON blob in new `harness_results` table | Preserves existing query paths; blob captures stderr/CI for later rigor. |
| 5 | Tracking helper | Function `eval_and_log(experiment, ...)` in `benchmark.py` | Avoids `tracking → benchmark` import coupling. |
| 6 | Call-signature change | Mutually exclusive `model_path` / `model` kwargs; `tasks` optional first positional | Explicit codepath selection; clearer than type-dispatch. |
| 7 | Quantization handling | `UserWarning`, don't block | Mirrors precedent in `perplexity()` (benchmark.py:53). |
| 8 | Test scope | Unit (mocked) + one real integration (`arc_easy`, `limit=20`, TinyLlama) | Mocks cover contracts; one real run closes the "does it work?" gap. |

## New `eval_downstream` signature

```python
FAST_TRIPLET: list[str] = ["hellaswag", "arc_easy", "arc_challenge"]

PAPER_STANDARD_FEWSHOT: dict[str, int] = {
    "hellaswag": 0,
    "arc_easy": 0,
    "arc_challenge": 25,
    "mmlu": 5,
}

def eval_downstream(
    tasks: list[str] | None = None,
    *,
    model_path: str | None = None,
    model: Any = None,
    tokenizer: Any = None,
    num_fewshot: int | dict[str, int] | None = None,
    limit: int | None = None,
) -> dict[str, float]:
    """Evaluate on downstream tasks via lm-eval.

    Exactly one of `model_path` or `model` must be given. When `model` is
    given, `tokenizer` must also be given; evaluation runs in-process via
    HFLM. When `model_path` is given, evaluation shells out to the lm_eval
    CLI (unchanged legacy path).
    """
```

Validation rules:
- Exactly one of `model_path` / `model` non-None → else `ValueError`.
- `model` without `tokenizer` → `ValueError`.
- `tasks=None` → `FAST_TRIPLET`.
- `num_fewshot=None` → `PAPER_STANDARD_FEWSHOT[task]` per-task, fallback `0` for unknown tasks.

## In-process evaluation path

```python
def _in_process_eval(model, tokenizer, tasks, num_fewshot, limit) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    if getattr(getattr(model, "config", None), "quantization_config", None):
        warnings.warn(
            "Model has quantization_config set — harness accuracy on "
            "quantized models is typically 1–3 pp below fp16 reference.",
            UserWarning, stacklevel=3,
        )

    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    fewshot_map = _resolve_fewshot(tasks, num_fewshot)

    # simple_evaluate takes a single num_fewshot per call; group tasks
    # sharing the same few-shot count to minimise invocations.
    merged: dict[str, Any] = {"results": {}, "config": None}
    for n, group in _group_by_fewshot(fewshot_map):
        partial = simple_evaluate(
            model=lm, tasks=group, num_fewshot=n, limit=limit,
        )
        merged["results"].update(partial["results"])
        if merged["config"] is None:
            merged["config"] = partial.get("config", {})
    return merged
```

- `_group_by_fewshot(fewshot_map) -> list[tuple[int, list[str]]]` — groups tasks by their few-shot count; returns (n, tasks) pairs.
- `_resolve_fewshot(tasks, num_fewshot) -> dict[str, int]` — resolves None/int/dict into a full per-task dict using `PAPER_STANDARD_FEWSHOT` with fallback 0.

Return is the full `lm_eval` dict (unpruned) so `eval_and_log` can serialize it; `eval_downstream` itself narrows to `{task: primary_accuracy}` before returning.

Primary-accuracy rule (mirrors existing `_extract_accuracies` at `benchmark.py:269`): try `acc_norm,none` → `acc,none` → `acc_norm` → `acc` → first numeric value → `nan`.

## `eval_and_log` — tracking integration

```python
def eval_and_log(
    experiment: Experiment,
    *,
    model_path: str | None = None,
    model: Any = None,
    tokenizer: Any = None,
    tasks: list[str] | None = None,
    num_fewshot: int | dict[str, int] | None = None,
    limit: int | None = None,
) -> dict[str, float]:
    """Run evaluation and persist to experiment tracking.

    Writes:
      - One row per (task, metric) into `metrics`, keyed
        `harness.<task>.<metric>`:
            harness.hellaswag.acc_norm
            harness.hellaswag.acc_norm_stderr
            harness.hellaswag.n_samples
      - One blob into `harness_results` with the full lm_eval output.

    Returns the flat {task: primary_accuracy} dict.
    """
```

Blob fields: `tasks_json` (JSON list), `num_fewshot` (JSON, int or dict), `limit_samples` (int or NULL), `result_json` (full `lm_eval` output), `created_at` (ISO-8601).

**Flat-metric serialization rule (precise):** for each `task` in the harness's `results` dict, iterate `results[task]`; for every leaf whose value is `int` or `float` (and not NaN), write one `metrics` row with:

```
key   = f"harness.{task}.{leaf_key}"     # e.g. "harness.hellaswag.acc_norm"
value = float(leaf_value)
```

Non-numeric leaves (version strings, alias flags) are skipped — they still live in the full JSON blob. Keys like `acc,none` (the harness's natural key style) are written verbatim: `harness.hellaswag.acc,none`. Consumers (`compare_experiments`, GUI) that `LIKE 'harness.%'` already handle arbitrary suffixes.

## Schema addition

Appended to `_SCHEMA_SQL` in `tracking.py`:

```sql
CREATE TABLE IF NOT EXISTS harness_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT    NOT NULL,
    tasks_json      TEXT    NOT NULL,
    num_fewshot     TEXT    NOT NULL,
    limit_samples   INTEGER,
    result_json     TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);
```

`start()` gains one additional `DELETE FROM harness_results WHERE experiment_name = ?` line in its rerun-clean path to match existing `metrics` / `surgery_ops` / `samples` handling.

No FK — matches the existing pattern (`metrics.experiment_name` is also un-FK'd). The write helper is a module-level `_log_harness_result(db_path, experiment_name, tasks, num_fewshot, limit, result)` in `tracking.py`, called only from `benchmark.py::eval_and_log`.

## Tests

### Unit (new, mocked)

- `test_eval_downstream_requires_exactly_one_model_source` — 0 or 2 sources → `ValueError`.
- `test_eval_downstream_requires_tokenizer_with_model` — `model=m` without tokenizer → `ValueError`.
- `test_eval_downstream_defaults_to_fast_triplet` — no `tasks` → harness called with `FAST_TRIPLET`.
- `test_eval_downstream_dispatches_per_task_fewshot` — `["hellaswag", "arc_challenge"]` produces two `simple_evaluate` calls (0-shot group + 25-shot group).
- `test_eval_downstream_in_memory_quant_warns` — mock `config.quantization_config = {...}` → `UserWarning` matches `"quantization_config set"`.
- `test_eval_downstream_subprocess_path_unchanged` — `model_path=...` still shells out.
- `test_eval_and_log_writes_flat_metrics` — mock `eval_downstream`; assert `metrics` rows written with `harness.<task>.<metric>` keys.
- `test_eval_and_log_writes_blob` — same mock; assert one `harness_results` row with correct columns.

### Integration (new, real run)

- `test_eval_and_log_tinyllama_arc_easy` — real TinyLlama fp16 load via `load_model`; `eval_and_log(exp, model=m, tokenizer=t, tasks=["arc_easy"], limit=20)`. Asserts:
  - Return dict has `"arc_easy"` → float in `[0.0, 1.0]`.
  - `metrics` rows for `harness.arc_easy.*` present.
  - Exactly one `harness_results` row for the experiment.
  - Gated by `@pytest.mark.skipif(not _tinyllama_cached() or not torch.cuda.is_available())`.
  - Expected runtime ~30 s on RTX 2080.

### Subagent GPU policy

Per the updated `feedback_gpu_sandbox.md` memory, the integration-test subagent's prompt must explicitly instruct it to use `dangerouslyDisableSandbox: true` when running GPU-dependent pytest commands — not skip them.

## Migration & risk

### Migration

- Callers of the old `eval_downstream(model_path, tasks, ...)` positional break: `tasks` is the new first positional; `model_path` is keyword-only. Planning phase must audit `grep -rn eval_downstream\\( testing/` and update call sites.
- `experiments.db` gets the `harness_results` table on first `_connect()` after deploy (via `CREATE TABLE IF NOT EXISTS`). No ALTER, no migration script.

### Risks

- In-process `HFLM` on an nf4-quantized model runs but returns unreliable numbers. Mitigated by `UserWarning`.
- Large harness result blobs (up to ~40 KB for full MMLU). SQLite handles this comfortably; noted.
- `simple_evaluate` holds GPU state across calls — grouping by few-shot count deliberately re-uses the same HFLM instance.

### Rollback

Single-commit revert restores prior `benchmark.py` and `tracking.py`. The `harness_results` table remains on disk, orphaned — cosmetic only.

## Commit strategy

One commit:

1. `feat(benchmark): in-process lm-eval-harness + tracking integration` —
   `benchmark.py` signature + implementation; `tracking.py` schema + write
   helper; new test file (unit + integration).

## Expected outcome

- A single call — `eval_and_log(exp, model=m, tokenizer=t)` — runs the fast triplet with paper-standard few-shot and persists every metric the harness emits. Ready for direct use in surgery-variant A/B comparisons.
- Zero user-facing breaking change for the subprocess path; all new surface is additive for callers that want in-process evaluation.
- Clean foundation for Phase 3 (activation patching) — which will use the same `eval_and_log` to quantify the downstream effect of a causal intervention.
