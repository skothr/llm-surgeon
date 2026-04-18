"""Tests for experiment tracking module."""

from llm_surgeon.tracking import start, list_experiments, get_experiment, compare_experiments
from llm_surgeon.surgery import SurgeryLog


class TestStartCreatesExperiment:
    def test_start_creates_experiment(self, tmp_path):
        db = str(tmp_path / "exp.db")
        exp = start("test-exp", description="a test", base_model="tiny", recipe={}, db_path=db)
        assert exp is not None
        exps = list_experiments(db_path=db)
        assert len(exps) == 1
        assert exps[0]["name"] == "test-exp"


class TestLogSurgery:
    def test_log_surgery(self, tmp_path):
        db = str(tmp_path / "exp.db")
        exp = start("test-surgery", base_model="tiny", db_path=db)
        log = SurgeryLog()
        log.add("remove_layers", "Removed [3,4]", 8, 6)
        exp.log_surgery(log)
        result = get_experiment("test-surgery", db_path=db)
        assert len(result["ops"]) == 1
        assert result["ops"][0]["operation"] == "remove_layers"
        assert result["ops"][0]["layer_count_before"] == 8
        assert result["ops"][0]["layer_count_after"] == 6


class TestLogMetric:
    def test_log_metric(self, tmp_path):
        db = str(tmp_path / "exp.db")
        exp = start("test-metric", base_model="tiny", db_path=db)
        exp.log_metric("perplexity", 42.5)
        result = get_experiment("test-metric", db_path=db)
        metrics = {m["key"]: m["value"] for m in result["metrics"]}
        assert "perplexity" in metrics
        assert abs(metrics["perplexity"] - 42.5) < 1e-6


class TestLogSamples:
    def test_log_samples(self, tmp_path):
        db = str(tmp_path / "exp.db")
        exp = start("test-samples", base_model="tiny", db_path=db)
        samples = ["hello world", "foo bar baz"]
        exp.log_samples(samples)
        result = get_experiment("test-samples", db_path=db)
        assert len(result["samples"]) == 1
        import json
        stored = json.loads(result["samples"][0]["data"])
        assert stored == samples


class TestFinishSetsStatus:
    def test_finish_sets_status(self, tmp_path):
        db = str(tmp_path / "exp.db")
        exp = start("test-finish", base_model="tiny", db_path=db)
        exp.finish(notes="done")
        result = get_experiment("test-finish", db_path=db)
        assert result["status"] == "completed"
        assert result["notes"] == "done"
        assert result["finished_at"] is not None


class TestListExperiments:
    def test_list_experiments(self, tmp_path):
        db = str(tmp_path / "exp.db")
        start("exp-a", base_model="tiny", db_path=db)
        start("exp-b", base_model="tiny", db_path=db)
        exps = list_experiments(db_path=db)
        names = [e["name"] for e in exps]
        assert "exp-a" in names
        assert "exp-b" in names
        assert len(exps) == 2


class TestGetExperimentReturnsMetrics:
    def test_get_experiment_returns_metrics(self, tmp_path):
        db = str(tmp_path / "exp.db")
        exp = start("test-get", base_model="tiny", db_path=db)
        exp.log_metric("perplexity", 10.0)
        exp.log_metric("accuracy", 0.9)
        result = get_experiment("test-get", db_path=db)
        metrics = {m["key"]: m["value"] for m in result["metrics"]}
        assert metrics["perplexity"] == 10.0
        assert abs(metrics["accuracy"] - 0.9) < 1e-6


class TestCompareExperiments:
    def test_compare_experiments(self, tmp_path):
        db = str(tmp_path / "exp.db")
        exp_a = start("cmp-a", base_model="tiny", db_path=db)
        exp_a.log_metric("perplexity", 20.0)
        exp_b = start("cmp-b", base_model="tiny", db_path=db)
        exp_b.log_metric("perplexity", 15.0)
        result = compare_experiments(["cmp-a", "cmp-b"], db_path=db)
        assert "cmp-a" in result
        assert "cmp-b" in result
        assert result["cmp-a"]["perplexity"] == 20.0
        assert result["cmp-b"]["perplexity"] == 15.0


class TestDbPersists:
    def test_db_persists(self, tmp_path):
        db = str(tmp_path / "exp.db")
        start("persist-exp", base_model="tiny", db_path=db)
        # New call — simulates a fresh process reading the same db
        exps = list_experiments(db_path=db)
        assert any(e["name"] == "persist-exp" for e in exps)


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
