# Phase 2 lm-evaluation-harness Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `llm_surgeon/benchmark.py::eval_downstream` to run the `lm-evaluation-harness` on in-memory HuggingFace models via `lm_eval.simple_evaluate(HFLM(pretrained=model))`, add `eval_and_log(experiment, ...)` for SQLite tracking integration, and add a `harness_results` blob table to `tracking.py`. Subprocess fallback preserved for path-based callers.

**Architecture:** One function's signature rewritten (`eval_downstream`), one new function added (`eval_and_log`), three internal helpers (`_in_process_eval`, `_resolve_fewshot`, `_group_by_fewshot`, `_serialize_harness_metrics`), one SQLite schema addition (`harness_results` table + a rerun-clean DELETE + a module-level `_log_harness_result` writer). Two existing in-repo callers migrate to the new signature. Tests live alongside the existing `test_benchmark.py` and `test_tracking.py`.

**Tech Stack:** Python 3 · PyTorch · transformers · lm-evaluation-harness 0.4.11 (`lm_eval.simple_evaluate`, `lm_eval.models.huggingface.HFLM`) · SQLite · pytest

**Spec:** `testing/docs/superpowers/specs/2026-04-17-phase2-lm-eval-harness-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `testing/llm_surgeon/benchmark.py` | Modify | Rewrite `eval_downstream` (kwarg-split signature + in-process path); add `eval_and_log`, `FAST_TRIPLET`, `PAPER_STANDARD_FEWSHOT`, `_in_process_eval`, `_resolve_fewshot`, `_group_by_fewshot`, `_serialize_harness_metrics`. Keep subprocess `model_path=` path and all existing helpers (`_find_and_parse_results`, `_extract_accuracies`). Leave `perplexity()`, `compare()`, `generation_metrics()` untouched. |
| `testing/llm_surgeon/tracking.py` | Modify | Add `harness_results` CREATE TABLE clause to `_SCHEMA_SQL`; add matching `DELETE FROM harness_results` in `start()`'s rerun-clean block; add module-level `_log_harness_result(db_path, experiment_name, tasks, num_fewshot, limit, result)` writer. Leave `Experiment` class and public API untouched. |
| `testing/llm_surgeon/recipe.py` | Modify | Lines 214–230: swap `save_checkpoint → tempdir → eval_downstream(ckpt, ...)` for direct in-memory `benchmark.eval_downstream(tasks=..., model=..., tokenizer=..., ...)`. Drops the tempdir + `_export.save_checkpoint` block. |
| `testing/tests/test_benchmark.py` | Modify | Migrate existing `TestEvalDownstream` (lines 117–147) to new kwarg signature (`model_path=...`). Add new `TestEvalDownstreamInProcess`, `TestEvalDownstreamValidation`, `TestFewShotResolution`, `TestEvalAndLog`, `TestEvalAndLogIntegration` classes. |
| `testing/tests/test_tracking.py` | Modify | Add `TestHarnessResultsTable` class — one test for `_log_harness_result`, one for rerun-clean cascade. |
| `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md` | Modify | Update status block to mark Phase 2 shipped and point to the commit SHA. No git commit (memory lives outside the repo). |

---

## Task Ordering

Single implementation commit (per spec §8) plus a roadmap-memory update:

1. **Task 1** — `feat(benchmark): in-process lm-eval-harness + tracking integration` (all code + tests, one commit).
2. **Task 2** — Update roadmap memory. No commit.

---

## Task 1 — In-process lm-eval-harness + tracking integration

**Files:**
- Modify: `testing/llm_surgeon/benchmark.py`
- Modify: `testing/llm_surgeon/tracking.py`
- Modify: `testing/llm_surgeon/recipe.py`
- Modify: `testing/tests/test_benchmark.py`
- Modify: `testing/tests/test_tracking.py`

---

### Subtask 1A — Schema addition

- [ ] **Step 1.1 — Write failing test for `harness_results` table creation**

Append to `testing/tests/test_tracking.py`:

```python
class TestHarnessResultsTable:
    def test_log_harness_result_writes_row(self, tmp_path):
        """_log_harness_result inserts a harness_results row and _connect
        creates the table via CREATE TABLE IF NOT EXISTS."""
        from llm_surgeon.tracking import start, _log_harness_result
        import sqlite3

        db = str(tmp_path / "t.db")
        start("exp1", db_path=db)

        _log_harness_result(
            db_path=db,
            experiment_name="exp1",
            tasks=["hellaswag", "arc_easy"],
            num_fewshot={"hellaswag": 0, "arc_easy": 0},
            limit=20,
            result={"results": {"hellaswag": {"acc_norm,none": 0.61}}},
        )

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT experiment_name, tasks_json, num_fewshot, limit_samples, "
            "result_json, created_at FROM harness_results WHERE experiment_name = ?",
            ("exp1",),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "exp1"
        import json as _json
        assert _json.loads(row[1]) == ["hellaswag", "arc_easy"]
        assert _json.loads(row[2]) == {"hellaswag": 0, "arc_easy": 0}
        assert row[3] == 20
        payload = _json.loads(row[4])
        assert payload["results"]["hellaswag"]["acc_norm,none"] == 0.61
        assert row[5]  # created_at non-empty

    def test_start_reruns_purge_harness_results(self, tmp_path):
        """Re-calling start(name) deletes any prior harness_results rows
        for that experiment, matching the existing metrics/surgery_ops
        cascade behavior."""
        from llm_surgeon.tracking import start, _log_harness_result
        import sqlite3

        db = str(tmp_path / "t.db")
        start("exp1", db_path=db)
        _log_harness_result(
            db_path=db, experiment_name="exp1",
            tasks=["hellaswag"], num_fewshot=0, limit=None,
            result={"results": {}},
        )
        # Re-run the experiment — prior rows should be wiped.
        start("exp1", db_path=db)

        conn = sqlite3.connect(db)
        n = conn.execute(
            "SELECT COUNT(*) FROM harness_results WHERE experiment_name = ?",
            ("exp1",),
        ).fetchone()[0]
        conn.close()
        assert n == 0
```

- [ ] **Step 1.2 — Run tests to confirm they fail**

Run from `/home/ai/ai-projects/llm`:
```
testing/.venv/bin/python -m pytest testing/tests/test_tracking.py::TestHarnessResultsTable -v
```
Expected: both tests FAIL with `ImportError: cannot import name '_log_harness_result'`.

- [ ] **Step 1.3 — Add `harness_results` table to `_SCHEMA_SQL`**

In `testing/llm_surgeon/tracking.py`, extend the `_SCHEMA_SQL` string (currently ends around line 50 with the `samples` table). Append before the closing `"""`:

```python
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

- [ ] **Step 1.4 — Add `DELETE FROM harness_results` to `start()`'s rerun-clean block**

In `tracking.py`, find `start()` (around line 138). After the existing line `conn.execute("DELETE FROM samples WHERE experiment_name = ?", (name,))` and before `conn.execute("DELETE FROM experiments WHERE name = ?", (name,))`, add:

```python
conn.execute("DELETE FROM harness_results WHERE experiment_name = ?", (name,))
```

So the block becomes:

```python
conn.execute("DELETE FROM metrics WHERE experiment_name = ?", (name,))
conn.execute("DELETE FROM surgery_ops WHERE experiment_name = ?", (name,))
conn.execute("DELETE FROM samples WHERE experiment_name = ?", (name,))
conn.execute("DELETE FROM harness_results WHERE experiment_name = ?", (name,))
conn.execute("DELETE FROM experiments WHERE name = ?", (name,))
```

- [ ] **Step 1.5 — Add `_log_harness_result` module-level writer**

Append to `tracking.py` after the `compare_experiments` function (end of file):

```python
def _log_harness_result(
    *,
    db_path: str,
    experiment_name: str,
    tasks: List[str],
    num_fewshot: Any,
    limit: int | None,
    result: Dict[str, Any],
) -> None:
    """Insert one row into harness_results with the full lm_eval output.

    Called by benchmark.eval_and_log. Module-level (not an Experiment method)
    to avoid a tracking -> benchmark import cycle.
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO harness_results
                (experiment_name, tasks_json, num_fewshot, limit_samples,
                 result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_name,
                json.dumps(tasks),
                json.dumps(num_fewshot),
                limit,
                json.dumps(result),
                _now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
```

Add the import if `Any` isn't already used from `typing` at the top of the file — check line 7 (`from typing import Any, Dict, List`). If not present, extend it.

- [ ] **Step 1.6 — Run tracking tests to confirm they pass**

Run:
```
testing/.venv/bin/python -m pytest testing/tests/test_tracking.py::TestHarnessResultsTable -v
```
Expected: both tests PASS.

---

### Subtask 1B — `benchmark.py` constants, helpers, new signature

- [ ] **Step 1.7 — Write failing tests for the signature validation rules**

Append to `testing/tests/test_benchmark.py`:

```python
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
```

- [ ] **Step 1.8 — Run new tests to confirm they fail**

Run:
```
testing/.venv/bin/python -m pytest testing/tests/test_benchmark.py::TestEvalDownstreamValidation testing/tests/test_benchmark.py::TestFewShotResolution testing/tests/test_benchmark.py::TestGroupByFewshot -v
```
Expected: all tests FAIL (`ImportError` for `_resolve_fewshot`, `_group_by_fewshot`; signature errors for validation tests).

- [ ] **Step 1.9 — Add constants and helpers to `benchmark.py`**

In `testing/llm_surgeon/benchmark.py`, after the existing imports (around line 14) and before `perplexity()`, add:

```python
# ---------------------------------------------------------------------------
# Downstream-eval defaults and helpers (Phase 2)
# ---------------------------------------------------------------------------

FAST_TRIPLET: List[str] = ["hellaswag", "arc_easy", "arc_challenge"]

PAPER_STANDARD_FEWSHOT: Dict[str, int] = {
    "hellaswag": 0,
    "arc_easy": 0,
    "arc_challenge": 25,
    "mmlu": 5,
}


def _resolve_fewshot(
    tasks: List[str],
    num_fewshot: Union[int, Dict[str, int], None],
) -> Dict[str, int]:
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
    out: Dict[str, int] = {}
    for t in tasks:
        if t in num_fewshot:
            out[t] = num_fewshot[t]
        else:
            out[t] = PAPER_STANDARD_FEWSHOT.get(t, 0)
    return out


def _group_by_fewshot(fewshot_map: Dict[str, int]) -> List[Tuple[int, List[str]]]:
    """Group tasks sharing the same num_fewshot count.

    Returns a sorted list of (count, [task, ...]) pairs. Ordering is
    deterministic: ascending by count, then by task name inside each group.
    This lets _in_process_eval call simple_evaluate once per unique count.
    """
    buckets: Dict[int, List[str]] = {}
    for task, n in fewshot_map.items():
        buckets.setdefault(n, []).append(task)
    return [(n, sorted(buckets[n])) for n in sorted(buckets)]
```

Also extend the `typing` import at the top (currently `from typing import Any, Dict, List, Optional, Union`) — all names already present, no change needed. Add `Tuple` to the import if missing.

- [ ] **Step 1.10 — Rewrite `eval_downstream` signature**

Replace the entire current `eval_downstream` function (lines ~174–238 in the pre-patch file) with the new version. Keep the existing `_find_and_parse_results` and `_extract_accuracies` helpers (they're still used by the subprocess path).

```python
def eval_downstream(
    tasks: Optional[List[str]] = None,
    *,
    model_path: Optional[str] = None,
    model: Any = None,
    tokenizer: Any = None,
    num_fewshot: Union[int, Dict[str, int], None] = None,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a HuggingFace checkpoint on downstream tasks via lm-eval.

    Exactly one of *model_path* or *model* must be given:

    - `model_path=...` -> shells out to ``lm_eval`` CLI (legacy path).
    - `model=..., tokenizer=...` -> runs in-process via ``HFLM`` (new).

    Args:
        tasks: List of lm-eval task names. Defaults to FAST_TRIPLET.
        model_path: Local path or HuggingFace model ID. Mutually exclusive
            with ``model``.
        model: An already-loaded HuggingFace CausalLM. Mutually exclusive
            with ``model_path``. Requires ``tokenizer``.
        tokenizer: Required companion to ``model``.
        num_fewshot: int (uniform), dict (per-task override), or None
            (PAPER_STANDARD_FEWSHOT).
        limit: Cap examples per task (useful for fast testing).

    Returns:
        dict mapping task name to primary accuracy (acc_norm if present,
        else acc).
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

    # Subprocess path (legacy). Resolve an int for --num_fewshot: accept
    # only int or None here; dict isn't supported by the CLI single-value
    # flag. If None, use 0 (the CLI's own default).
    if isinstance(num_fewshot, dict):
        raise ValueError(
            "Dict num_fewshot is only supported with in-memory model; "
            "pass an int or None for the model_path subprocess path."
        )
    effective_nf = num_fewshot if num_fewshot is not None else 0
    return _subprocess_eval(
        model_path=model_path,
        tasks=tasks,
        num_fewshot=effective_nf,
        limit=limit,
    )
```

- [ ] **Step 1.11 — Extract subprocess path into `_subprocess_eval`**

Below `eval_downstream`, add the helper (the body is the old `eval_downstream` body, unmodified except for the signature):

```python
def _subprocess_eval(
    *,
    model_path: str,
    tasks: List[str],
    num_fewshot: int,
    limit: Optional[int],
) -> Dict[str, float]:
    """Legacy subprocess path: shells out to lm_eval CLI."""
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

        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"lm_eval failed (exit {result.returncode}).\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )

        results_data = _find_and_parse_results(tmpdir)
        return _extract_accuracies(results_data, tasks)
```

- [ ] **Step 1.12 — Add `_in_process_eval`**

Below `_subprocess_eval`, add:

```python
def _in_process_eval(
    *,
    model: Any,
    tokenizer: Any,
    tasks: List[str],
    num_fewshot: Union[int, Dict[str, int], None],
    limit: Optional[int],
) -> Dict[str, Any]:
    """Run lm_eval.simple_evaluate in-process against an in-memory model.

    Returns the full lm_eval output dict (unnarrowed) so callers can
    persist the full blob before narrowing to flat accuracies.
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM  # pyright: ignore[reportMissingImports]

    # Mirror perplexity()'s warning at benchmark.py:53 for the same scenario.
    cfg = getattr(model, "config", None)
    if getattr(cfg, "quantization_config", None):
        warnings.warn(
            "Model has quantization_config set — harness accuracy on "
            "quantized models is typically 1-3 pp below fp16 reference "
            "numbers.",
            UserWarning, stacklevel=3,
        )

    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    fewshot_map = _resolve_fewshot(tasks, num_fewshot)

    merged: Dict[str, Any] = {"results": {}, "config": None}
    for n, group in _group_by_fewshot(fewshot_map):
        partial = simple_evaluate(
            model=lm, tasks=group, num_fewshot=n, limit=limit,
        )
        merged["results"].update(partial["results"])
        if merged["config"] is None:
            merged["config"] = partial.get("config", {})
    merged["effective_num_fewshot"] = fewshot_map
    return merged
```

The `effective_num_fewshot` field captures the resolved per-task count — useful for post-hoc inspection of a stored blob.

- [ ] **Step 1.13 — Run validation + helper tests to confirm they pass**

Run:
```
testing/.venv/bin/python -m pytest testing/tests/test_benchmark.py::TestEvalDownstreamValidation testing/tests/test_benchmark.py::TestFewShotResolution testing/tests/test_benchmark.py::TestGroupByFewshot -v
```
Expected: all 10 tests PASS.

---

### Subtask 1C — In-process dispatch tests (mocked)

- [ ] **Step 1.14 — Write failing tests for in-process dispatch**

Append to `testing/tests/test_benchmark.py`:

```python
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
            # Return a minimal well-formed lm_eval output.
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
            config = None  # no quantization_config

        result = eval_downstream(model=_M(), tokenizer=object())
        assert set(result.keys()) == set(FAST_TRIPLET)
        # Collect every task name across every simple_evaluate call.
        seen = set()
        for c in captured["calls"]:
            seen.update(c["tasks"])
        assert seen == set(FAST_TRIPLET)

    def test_dispatches_per_task_fewshot(self, monkeypatch):
        """Hellaswag is 0-shot, arc_challenge is 25-shot -> two calls."""
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
        # One call at 0-shot with hellaswag, one at 25-shot with arc_challenge.
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
```

- [ ] **Step 1.15 — Run in-process dispatch tests**

Run:
```
testing/.venv/bin/python -m pytest testing/tests/test_benchmark.py::TestEvalDownstreamInProcess -v
```
Expected: all 5 tests PASS.

---

### Subtask 1D — `eval_and_log` helper

- [ ] **Step 1.16 — Write failing tests for `eval_and_log`**

Append to `testing/tests/test_benchmark.py`:

```python
class TestEvalAndLog:
    """eval_and_log persists harness results to both metrics (flat) and
    harness_results (blob) tables via tracking."""

    def _mock_in_process(self, monkeypatch, task_results):
        """Pin _in_process_eval to return a canned full lm_eval output."""
        def fake(**kwargs):
            # Echo the resolved few-shot back for blob verification.
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
        # Non-numeric alias field must NOT appear in flat metrics.
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
        # num_fewshot is a dict (resolved per-task) when num_fewshot=None.
        assert _json.loads(row[1]) == {"hellaswag": 0}
        assert row[2] == 20
        payload = _json.loads(row[3])
        assert payload["results"]["hellaswag"]["acc,none"] == 0.5
```

- [ ] **Step 1.17 — Run eval_and_log tests to confirm they fail**

Run:
```
testing/.venv/bin/python -m pytest testing/tests/test_benchmark.py::TestEvalAndLog -v
```
Expected: both tests FAIL — `ImportError: cannot import name 'eval_and_log'`.

- [ ] **Step 1.18 — Implement `eval_and_log` in `benchmark.py`**

Append to `benchmark.py` after `_in_process_eval`:

```python
def _serialize_harness_metrics(task_result: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a single task's harness result into float-valued metrics.

    Only numeric leaves (int/float, not NaN) are kept — strings like 'alias'
    go into the JSON blob instead. Keys are preserved verbatim, including
    comma suffixes like 'acc,none'.
    """
    out: Dict[str, float] = {}
    for k, v in task_result.items():
        if isinstance(v, (int, float)) and not (
            isinstance(v, float) and math.isnan(v)
        ):
            out[k] = float(v)
    return out


def eval_and_log(
    experiment: Any,
    *,
    model_path: Optional[str] = None,
    model: Any = None,
    tokenizer: Any = None,
    tasks: Optional[List[str]] = None,
    num_fewshot: Union[int, Dict[str, int], None] = None,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """Run eval_downstream and persist results to experiment tracking.

    Writes:
        - One row per (task, metric) in ``experiment.metrics`` with key
          ``harness.<task>.<metric>`` for every numeric leaf in the
          harness's per-task result dict.
        - One blob into ``harness_results`` with the full lm_eval output.

    Returns the flat ``{task: primary_accuracy}`` dict.
    """
    if (model_path is None) == (model is None):
        raise ValueError(
            "Exactly one of model_path or model must be provided."
        )
    if model is not None and tokenizer is None:
        raise ValueError("tokenizer is required when model is provided.")

    if tasks is None:
        tasks = list(FAST_TRIPLET)

    # Run harness. For the in-memory path we want the *full* blob, so we
    # call the helper directly instead of eval_downstream (which narrows).
    if model is not None:
        full = _in_process_eval(
            model=model, tokenizer=tokenizer,
            tasks=tasks, num_fewshot=num_fewshot, limit=limit,
        )
    else:
        # Subprocess path: read the raw results JSON rather than going
        # through eval_downstream's narrowing, so the blob captures stderr.
        if isinstance(num_fewshot, dict):
            raise ValueError(
                "Dict num_fewshot is only supported with in-memory model."
            )
        effective_nf = num_fewshot if num_fewshot is not None else 0
        full = _subprocess_eval_full(
            model_path=model_path, tasks=tasks,
            num_fewshot=effective_nf, limit=limit,
        )

    # Persist flat metrics.
    for task in tasks:
        task_result = full.get("results", {}).get(task, {})
        flat = _serialize_harness_metrics(task_result)
        for metric_key, value in flat.items():
            experiment.log_metric(f"harness.{task}.{metric_key}", value)

    # Persist full blob.
    from llm_surgeon.tracking import _log_harness_result
    # Compute the num_fewshot actually used — a dict when None/dict, else int.
    if isinstance(num_fewshot, int):
        nf_to_store: Any = num_fewshot
    else:
        nf_to_store = _resolve_fewshot(tasks, num_fewshot)
    _log_harness_result(
        db_path=experiment.db_path,
        experiment_name=experiment.name,
        tasks=tasks,
        num_fewshot=nf_to_store,
        limit=limit,
        result=full,
    )

    return _extract_accuracies(full, tasks)
```

Also add `import math` to `benchmark.py`'s top-of-file imports if not present (check line 1–13 — `math` is not currently imported; add it alphabetically after `json`).

- [ ] **Step 1.19 — Add `_subprocess_eval_full` helper**

Below `_subprocess_eval`, add:

```python
def _subprocess_eval_full(
    *,
    model_path: str,
    tasks: List[str],
    num_fewshot: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    """Subprocess path, returning the full lm_eval output dict (not narrowed).

    Used by eval_and_log when model_path= is passed.
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
```

- [ ] **Step 1.20 — Run `eval_and_log` tests to confirm they pass**

Run:
```
testing/.venv/bin/python -m pytest testing/tests/test_benchmark.py::TestEvalAndLog -v
```
Expected: both tests PASS.

---

### Subtask 1E — Migrate existing callers

- [ ] **Step 1.21 — Migrate `recipe.py:214–230`**

In `testing/llm_surgeon/recipe.py`, replace the existing downstream-eval block (currently reads approximately the lines below) with the in-memory equivalent:

```python
    if "downstream" in eval_cfg:
        from llm_surgeon import benchmark
        ds_cfg = eval_cfg["downstream"] or {}
        tasks = ds_cfg.get("tasks", [])
        num_fewshot = ds_cfg.get("num_fewshot", 5)
        _log(f"Running downstream eval: {tasks} ({num_fewshot}-shot)...", verbose)
        ds_results = benchmark.eval_downstream(
            tasks=tasks,
            model=model, tokenizer=tokenizer,
            num_fewshot=num_fewshot,
        )
        for task, score in ds_results.items():
            exp.log_metric(task, score)
            results[task] = score
            _log(f"  {task}: {score:.4f}", verbose)
```

The deletions: `import tempfile`, `from llm_surgeon import export as _export`, `with tempfile.TemporaryDirectory() as tmpdir:`, `ckpt = os.path.join(tmpdir, "checkpoint")`, `_export.save_checkpoint(model, ckpt, tokenizer=tokenizer)`. That's ~5 lines removed plus one indentation level dropped from the block inside the `if "downstream"` branch.

- [ ] **Step 1.22 — Update the existing `TestEvalDownstream` test class**

Two existing tests in `testing/tests/test_benchmark.py` call `eval_downstream(tiny_eval_checkpoint, tasks=[...], num_fewshot=0, limit=5)` positionally. Rewrite their calls to the new signature:

At line ~130 (inside `test_returns_dict_with_task_key`):

```python
result = eval_downstream(
    tasks=["arc_easy"],
    model_path=tiny_eval_checkpoint,
    num_fewshot=0,
    limit=5,
)
```

At line ~142 (inside `test_invalid_task_raises_runtime_error`):

```python
with pytest.raises(RuntimeError):
    eval_downstream(
        tasks=["this_task_does_not_exist_xyz"],
        model_path=tiny_eval_checkpoint,
        num_fewshot=0,
        limit=5,
    )
```

Both tests keep their existing decorators (`@requires_network` on the first).

- [ ] **Step 1.23 — Type-check modified modules**

From `testing/`:
```
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright \
    llm_surgeon/benchmark.py llm_surgeon/tracking.py llm_surgeon/recipe.py \
    tests/test_benchmark.py tests/test_tracking.py
```

Expected: 0 errors / 0 warnings / 0 informations. Fix any diagnostics surfaced before running the test suite.

- [ ] **Step 1.24 — Run the full benchmark + tracking test suite**

```
testing/.venv/bin/python -m pytest testing/tests/test_benchmark.py testing/tests/test_tracking.py -v
```

Expected: all existing + new tests PASS or skip cleanly (e.g. `@requires_network` tests skip offline). No failures.

---

### Subtask 1F — Integration test (real TinyLlama + GPU)

- [ ] **Step 1.25 — Add the integration test**

Append to `testing/tests/test_benchmark.py`:

```python
def _tinyllama_cached() -> bool:
    """Same cache probe used by test_surgery.py::TestLoadModelIntegration."""
    from llm_surgeon.surgery import _is_cached
    return _is_cached("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


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

        # Flat metrics populated.
        metrics = {m["key"]: m["value"]
                   for m in get_experiment("it1", db_path=db)["metrics"]}
        assert any(k.startswith("harness.arc_easy.") for k in metrics)

        # Exactly one blob row.
        conn = sqlite3.connect(db)
        n = conn.execute(
            "SELECT COUNT(*) FROM harness_results WHERE experiment_name = 'it1'"
        ).fetchone()[0]
        conn.close()
        assert n == 1
```

- [ ] **Step 1.26 — Run the integration test with GPU sandbox disabled**

**This subagent must use `dangerouslyDisableSandbox: true` for this Bash call** — per the user-approved memory `feedback_gpu_sandbox.md`. Do not skip.

Run:
```
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_benchmark.py::TestEvalAndLogIntegration -v
```

Expected: 1 PASS in ~30–90 s (includes a one-time TinyLlama fp16 load from the local cache plus arc_easy eval over 20 items). If runtime exceeds 3 min, something's wrong — investigate before committing.

---

### Subtask 1G — Commit

- [ ] **Step 1.27 — Final full-suite sanity check**

```
testing/.venv/bin/python -m pytest testing/tests/ -v 2>&1 | tail -30
```

Expected: no new failures. Pre-existing skips are acceptable.

- [ ] **Step 1.28 — Confirm clean working tree intent**

```
git -C /home/ai/ai-projects/llm status
```

Expected: the only changes are `testing/llm_surgeon/benchmark.py`, `testing/llm_surgeon/tracking.py`, `testing/llm_surgeon/recipe.py`, `testing/tests/test_benchmark.py`, `testing/tests/test_tracking.py`. No stray files.

- [ ] **Step 1.29 — Commit**

```bash
cd /home/ai/ai-projects/llm
git add \
    testing/llm_surgeon/benchmark.py \
    testing/llm_surgeon/tracking.py \
    testing/llm_surgeon/recipe.py \
    testing/tests/test_benchmark.py \
    testing/tests/test_tracking.py
git commit -m "$(cat <<'EOF'
feat(benchmark): in-process lm-eval-harness + tracking integration

Upgrade eval_downstream to run lm_eval.simple_evaluate in-process via
HFLM on an in-memory (model, tokenizer) pair — previously it shelled
out to the CLI, requiring a disk-resident checkpoint. Subprocess path
retained under model_path= kwarg. Add eval_and_log helper that writes
flat harness.<task>.<metric> rows to `metrics` and a full JSON blob
to the new `harness_results` table. Defaults: FAST_TRIPLET (hellaswag,
arc_easy, arc_challenge) with paper-standard per-task few-shot
(0/0/25). Quantized models warn. Migrates recipe.py off the tempdir+
save_checkpoint dance. Unit tests + one integration test against
TinyLlama fp16 (limit=20, ~30 s on 2080).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run: `git status` — expected clean tree afterwards. Report the commit SHA.

---

## Task 2 — Mark Phase 2 shipped in roadmap memory

**Files:**
- Modify: `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md`

No git commit — memory lives outside the repo.

---

- [ ] **Step 2.1 — Update the `### Status` block**

Append a new bullet after the existing "Phase 1 shipped" line:

```
- 2026-04-17: **Phase 2 shipped** — eval_downstream accepts in-memory (model, tokenizer) via lm_eval.simple_evaluate + HFLM. New eval_and_log helper writes flat `harness.<task>.<metric>` rows + full JSON blob to new `harness_results` table. Defaults: FAST_TRIPLET with paper-standard per-task few-shot. recipe.py migrated off tempdir/save_checkpoint. Commit on `master`: <SHA>. Spec: testing/docs/superpowers/specs/2026-04-17-phase2-lm-eval-harness-design.md. Plan: testing/docs/superpowers/plans/2026-04-17-phase2-lm-eval-harness.md. Pyright clean; unit tests + one TinyLlama integration run (arc_easy, limit=20) green.
- Next up: **Phase 3 — activation patching**.
```

Replace `<SHA>` with the Task 1 commit SHA from Step 1.29.

Also, update the "Next up" line at the bottom (which currently says "Phase 2 — lm-evaluation-harness integration") to point to Phase 3. Leave the rest of the document untouched.

---

## Spec-Coverage Self-Review

Cross-check against `2026-04-17-phase2-lm-eval-harness-design.md`:

| Spec section | Task(s) |
|---|---|
| §2 new `eval_downstream` signature (tasks positional, kwargs model_path/model/tokenizer/num_fewshot/limit) | Step 1.10 |
| §2 validation rules (exactly one source, tokenizer required, tasks default) | Steps 1.7, 1.10 |
| §3 in-process path — `_in_process_eval`, `HFLM` wrap, quant warn, group-by-fewshot | Steps 1.12, 1.14, 1.9 |
| §3 primary-accuracy narrowing via `_extract_accuracies` | Step 1.10 (reuses existing helper) |
| §4 `eval_and_log` signature + behavior | Step 1.18 |
| §4 flat-key convention `harness.<task>.<metric>` | Step 1.18 (`_serialize_harness_metrics` + the loop in `eval_and_log`) |
| §5 `harness_results` table | Step 1.3 |
| §5 `start()` rerun-clean cascade | Step 1.4 |
| §5 `_log_harness_result` module-level writer | Step 1.5 |
| §6 unit tests (validation, fewshot resolution, grouping, dispatch, warn, flat metrics, blob) | Steps 1.7, 1.14, 1.16 |
| §6 integration test (TinyLlama arc_easy limit=20) | Step 1.25 |
| §6 GPU-sandbox policy for the integration test | Step 1.26 (explicit prompt text) |
| §7 migration of in-repo callers (`recipe.py`, existing `test_benchmark.py` tests) | Steps 1.21, 1.22 |
| §8 one-commit strategy | Step 1.29 |
| Roadmap memory bump | Task 2 |

No placeholders (no TBD/TODO/"similar to…"/"add validation"). Type consistency: `_resolve_fewshot` / `_group_by_fewshot` signatures match between Step 1.9 definition and Step 1.12 callsite. `_serialize_harness_metrics` defined (Step 1.18) and used (Step 1.18). `_log_harness_result`'s keyword-only signature matches between tracking.py definition (Step 1.5) and benchmark.py call (Step 1.18). `eval_and_log` parameter names are identical between test (Step 1.16) and implementation (Step 1.18).

---

## Execution Handoff

Plan complete and saved to `testing/docs/superpowers/plans/2026-04-17-phase2-lm-eval-harness.md`.

Per saved feedback memory (`feedback_execution_choice.md`), dispatching via **subagent-driven-development** without asking. The implementation subagent for Step 1.26 will receive the GPU-sandbox instruction in its prompt per `feedback_gpu_sandbox.md`.
