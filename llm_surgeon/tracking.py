"""SQLite-backed experiment tracking for llm-surgeon."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_DEFAULT_DB = str(Path(__file__).parent.parent / "experiments/experiments.db")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

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
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class Experiment:
    """Handle for a running experiment — log ops, metrics, and samples."""

    def __init__(self, name: str, db_path: str) -> None:
        self.name = name
        self.db_path = db_path

    def log_surgery(self, surgery_log) -> None:
        """Record all ops from a SurgeryLog."""
        conn = _connect(self.db_path)
        try:
            for op in surgery_log.ops:
                conn.execute(
                    """
                    INSERT INTO surgery_ops
                        (experiment_name, operation, description, layer_count_before, layer_count_after)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self.name, op.operation, op.description, op.layer_count_before, op.layer_count_after),
                )
            conn.commit()
        finally:
            conn.close()

    def log_metric(self, key: str, value: float) -> None:
        """Record a scalar metric."""
        conn = _connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO metrics (experiment_name, key, value) VALUES (?, ?, ?)",
                (self.name, key, float(value)),
            )
            conn.commit()
        finally:
            conn.close()

    def log_samples(self, samples: List[str]) -> None:
        """Record generation samples as a single JSON blob."""
        conn = _connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO samples (experiment_name, data) VALUES (?, ?)",
                (self.name, json.dumps(samples)),
            )
            conn.commit()
        finally:
            conn.close()

    def finish(self, notes: str = "") -> None:
        """Mark experiment completed."""
        conn = _connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE experiments
                SET status = 'completed', notes = ?, finished_at = ?
                WHERE name = ?
                """,
                (notes, _now(), self.name),
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(
    name: str,
    description: str = "",
    base_model: str = "",
    recipe: Any = None,
    db_path: str = _DEFAULT_DB,
) -> Experiment:
    """Create a new experiment record and return an Experiment handle."""
    recipe_yaml = json.dumps(recipe) if recipe is not None else None
    conn = _connect(db_path)
    try:
        # If experiment with this name already exists, delete the old one
        conn.execute("DELETE FROM metrics WHERE experiment_name = ?", (name,))
        conn.execute("DELETE FROM surgery_ops WHERE experiment_name = ?", (name,))
        conn.execute("DELETE FROM samples WHERE experiment_name = ?", (name,))
        conn.execute("DELETE FROM experiments WHERE name = ?", (name,))
        conn.execute(
            """
            INSERT INTO experiments (name, description, base_model, recipe_yaml, status, created_at)
            VALUES (?, ?, ?, ?, 'running', ?)
            """,
            (name, description, base_model, recipe_yaml, _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return Experiment(name=name, db_path=db_path)


def list_experiments(db_path: str = _DEFAULT_DB) -> List[Dict]:
    """Return all experiments as a list of dicts."""
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM experiments ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_experiment(name: str, db_path: str = _DEFAULT_DB) -> Dict:
    """Return a single experiment with its metrics, ops, and samples."""
    conn = _connect(db_path)
    try:
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
    finally:
        conn.close()


def compare_experiments(names: List[str], db_path: str = _DEFAULT_DB) -> Dict[str, Dict]:
    """Return side-by-side metric dicts for the named experiments.

    Returns:
        { experiment_name: { metric_key: value, ... }, ... }
    """
    result = {}
    for name in names:
        exp = get_experiment(name, db_path=db_path)
        result[name] = {m["key"]: m["value"] for m in exp["metrics"]}
    return result
