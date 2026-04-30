"""SQLite-backed experiment tracking for llm-surgeon."""

import json
import sqlite3
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_DB = str(Path(__file__).parent.parent / "experiments/experiments.db")


# Schema

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    description  TEXT,
    base_model   TEXT,
    recipe_yaml  TEXT,
    status       TEXT    NOT NULL DEFAULT 'running',
    notes        TEXT,
    created_at   TEXT    NOT NULL,
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name  TEXT    NOT NULL,
    key              TEXT    NOT NULL,
    value            REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS surgery_ops (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name     TEXT    NOT NULL,
    operation           TEXT    NOT NULL,
    description         TEXT,
    layer_count_before  INTEGER,
    layer_count_after   INTEGER
);

CREATE TABLE IF NOT EXISTS samples (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name  TEXT    NOT NULL,
    data             TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS harness_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT    NOT NULL,
    tasks_json      TEXT    NOT NULL,
    num_fewshot     TEXT    NOT NULL,
    limit_samples   INTEGER,
    result_json     TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);
"""


_SCHEMA_INITIALIZED: set[str] = set()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if db_path not in _SCHEMA_INITIALIZED:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        _SCHEMA_INITIALIZED.add(db_path)
    return conn


@contextmanager
def _connection(db_path: str) -> Generator[sqlite3.Connection]:
    """Open a connection, run a transaction, close.

    Wraps the body in ``with conn:`` so writes commit on success and roll
    back on exception. Caller does **not** call ``conn.commit()``.
    """
    conn = _connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Experiment class

class Experiment:
    """Handle for a running experiment — log ops, metrics, and samples."""

    def __init__(self, name: str, db_path: str) -> None:
        self.name = name
        self.db_path = db_path

    def log_surgery(self, surgery_log) -> None:
        """Record all ops from a SurgeryLog."""
        with _connection(self.db_path) as conn:
            for op in surgery_log.ops:
                conn.execute(
                    """
                    INSERT INTO surgery_ops
                        (experiment_name, operation, description, layer_count_before, layer_count_after)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self.name, op.operation, op.description, op.layer_count_before, op.layer_count_after),
                )

    def log_metric(self, key: str, value: float) -> None:
        """Record a scalar metric."""
        with _connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO metrics (experiment_name, key, value) VALUES (?, ?, ?)",
                (self.name, key, float(value)),
            )

    def log_samples(self, samples: list[str]) -> None:
        """Record generation samples as a single JSON blob."""
        with _connection(self.db_path) as conn:
            conn.execute(
                "INSERT INTO samples (experiment_name, data) VALUES (?, ?)",
                (self.name, json.dumps(samples)),
            )

    def finish(self, notes: str = "") -> None:
        """Mark experiment completed."""
        with _connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE experiments
                SET status = 'completed', notes = ?, finished_at = ?
                WHERE name = ?
                """,
                (notes, _now(), self.name),
            )


# Public API

def start(
    name: str,
    description: str = "",
    base_model: str = "",
    recipe: Mapping[str, Any] | None = None,
    db_path: str = _DEFAULT_DB,
) -> Experiment:
    """Create a new experiment record and return an Experiment handle."""
    recipe_yaml = json.dumps(recipe) if recipe is not None else None
    with _connection(db_path) as conn:
        # If an experiment with this name already exists, replace it
        for tbl in ("metrics", "surgery_ops", "samples", "harness_results"):
            conn.execute(f"DELETE FROM {tbl} WHERE experiment_name = ?", (name,))
        conn.execute("DELETE FROM experiments WHERE name = ?", (name,))
        conn.execute(
            """
            INSERT INTO experiments (name, description, base_model, recipe_yaml, status, created_at)
            VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (name, description, base_model, recipe_yaml, _now()),
        )
    return Experiment(name=name, db_path=db_path)


def list_experiments(db_path: str = _DEFAULT_DB) -> list[dict]:
    """Return all experiments as a list of dicts."""
    with _connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM experiments ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]


def get_experiment(name: str, db_path: str = _DEFAULT_DB) -> dict:
    """Return a single experiment with its metrics, ops, and samples."""
    with _connection(db_path) as conn:
        exp_row = conn.execute(
            "SELECT * FROM experiments WHERE name = ?", (name,)
        ).fetchone()
        if exp_row is None:
            raise KeyError(f"Experiment '{name}' not found")

        result = dict(exp_row)
        result["metrics"] = [
            dict(r) for r in conn.execute(
                "SELECT key, value FROM metrics WHERE experiment_name = ?", (name,)
            ).fetchall()
        ]
        result["ops"] = [
            dict(r) for r in conn.execute(
                "SELECT operation, description, layer_count_before, layer_count_after "
                "FROM surgery_ops WHERE experiment_name = ?",
                (name,),
            ).fetchall()
        ]
        result["samples"] = [
            dict(r) for r in conn.execute(
                "SELECT data FROM samples WHERE experiment_name = ?", (name,)
            ).fetchall()
        ]
        return result


def compare_experiments(names: list[str], db_path: str = _DEFAULT_DB) -> dict[str, dict]:
    """Return side-by-side metric dicts for the named experiments.

    Returns:
        { experiment_name: { metric_key: value, ... }, ... }
    """
    result = {}
    for name in names:
        exp = get_experiment(name, db_path=db_path)
        result[name] = {m["key"]: m["value"] for m in exp["metrics"]}
    return result


def log_harness_result(
    *,
    db_path: str,
    experiment_name: str,
    tasks: list[str],
    num_fewshot: int | dict[str, int] | None,
    limit: int | None,
    result: dict[str, Any],
) -> None:
    """Insert one row into harness_results with the full lm_eval output.

    Called by benchmark.eval_and_log via deferred import to avoid a
    tracking -> benchmark cycle. Name is semi-public (no leading underscore)
    to signal cross-module use.
    """
    with _connection(db_path) as conn:
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
                # lm_eval's result dict can contain torch.dtype and other
                # non-JSON types in its config section. Stringify them.
                json.dumps(result, default=str),
                _now(),
            ),
        )
