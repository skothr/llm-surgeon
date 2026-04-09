# Phase 6: Calibration, Recipes, Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
>
> **Tool rules (for subagents):**
> - Use Read (not cat/head/tail), Grep (not grep/rg/awk), Glob (not find/ls), Edit (not sed/awk) for all file operations
> - You are already in the project root (/home/ai/ai-projects/llm) — never cd
> - Python venv: `/home/ai/ai-projects/llm/testing/.venv/bin/python`
> - Run tests: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`

**Goal:** Add LayerNorm calibration to surgery.py, declarative YAML recipe execution to recipe.py, and SQLite experiment tracking to tracking.py.

**Architecture:** `calibrate()` adjusts RMSNorm parameters after surgery. `recipe.py` parses YAML experiment definitions and orchestrates the full pipeline. `tracking.py` logs experiments to SQLite with surgery ops, metrics, and samples.

**Tech Stack:** PyTorch (for calibration), pyyaml, sqlite3 (stdlib)

**Reference:** `docs/superpowers/specs/2026-04-08-llm-surgeon-design.md` (v2), Phase 6 section of phase plan.

---

## File Map

```
testing/
  llm_surgeon/
    surgery.py           — MODIFY — add calibrate()
    recipe.py            — CREATE — run(), run_batch(), generate_layer_sweep()
    tracking.py          — CREATE — start(), Experiment class, list/get/compare
    __init__.py          — MODIFY — add recipe, tracking imports
  tests/
    test_surgery.py      — MODIFY — add calibration tests
    test_recipe.py       — CREATE — recipe parsing and execution tests
    test_tracking.py     — CREATE — experiment logging tests
```

---

### Task 1: calibrate() in surgery.py

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for calibrate**

Add to `testing/tests/test_surgery.py`:

```python
from llm_surgeon.surgery import calibrate


class TestCalibrate:
    def test_runs_without_error(self, tiny_llama, tiny_llama_config):
        """Calibrate should run on an unmodified model without crashing."""
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        remove_layers(tiny_llama, [3, 4])
        calibrate(tiny_llama, tokenizer, text="tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok10", num_samples=2)

    def test_modifies_norm_parameters(self, tiny_llama, tiny_llama_config):
        """Calibration should change at least some RMSNorm gain parameters."""
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        remove_layers(tiny_llama, [3, 4])

        # Capture norm weights before
        norms_before = {}
        for i, layer in enumerate(tiny_llama.model.layers):
            norms_before[i] = layer.input_layernorm.weight.data.clone()

        calibrate(tiny_llama, tokenizer, text="tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok10", num_samples=2)

        # At least some norms should have changed
        changed = False
        for i, layer in enumerate(tiny_llama.model.layers):
            if not torch.equal(norms_before[i], layer.input_layernorm.weight.data):
                changed = True
                break
        assert changed, "Calibration did not modify any norm parameters"

    def test_model_still_runs_after_calibration(self, tiny_llama, tiny_llama_config):
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        remove_layers(tiny_llama, [5, 6, 7])
        calibrate(tiny_llama, tokenizer, text="tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok10", num_samples=2)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement calibrate**

Add to `testing/llm_surgeon/surgery.py`:

```python
def calibrate(
    model,
    tokenizer,
    text: Optional[str] = None,
    dataset: Optional[str] = None,
    num_samples: int = 128,
) -> None:
    """Calibrate RMSNorm parameters after surgery.

    Runs calibration text through the model, collects residual stream
    statistics at each layer boundary, and rescales RMSNorm gain parameters
    to compensate for distribution shift caused by surgery.

    This is the simplest calibration strategy (LayerNorm rescaling).
    It adjusts normalization parameters only — no weight modification
    beyond norms, no gradient computation.

    Args:
        model: Modified model to calibrate
        tokenizer: Tokenizer for the model
        text: Direct calibration text. If None, loads from dataset.
        dataset: Dataset name ('wikitext2'). Ignored if text provided.
        num_samples: Number of text samples to use from dataset.
    """
    if text is None and dataset is None:
        dataset = "wikitext2"

    if text is None:
        from datasets import load_dataset
        if dataset == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        texts = [item["text"] for item in ds if item["text"].strip()][:num_samples]
        text = "\n".join(texts)

    device = model.model.embed_tokens.weight.device
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=getattr(model.config, "max_position_embeddings", 2048))
    input_ids = encodings["input_ids"].to(device)

    # Collect residual stream statistics at each layer boundary
    layer_stats = {}
    hooks = []

    def make_input_hook(idx):
        def hook(module, inp, out):
            hidden = inp[0].detach().float()
            mean = hidden.mean(dim=(0, 1))  # (hidden_size,)
            var = hidden.var(dim=(0, 1))    # (hidden_size,)
            if idx not in layer_stats:
                layer_stats[idx] = {"mean_sum": torch.zeros_like(mean), "var_sum": torch.zeros_like(var), "count": 0}
            layer_stats[idx]["mean_sum"] += mean
            layer_stats[idx]["var_sum"] += var
            layer_stats[idx]["count"] += 1
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_input_hook(i)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    # Rescale RMSNorm gain parameters based on collected statistics
    for i, layer in enumerate(model.model.layers):
        if i in layer_stats and layer_stats[i]["count"] > 0:
            stats = layer_stats[i]
            avg_var = stats["var_sum"] / stats["count"]
            # RMS of the input to this layer
            rms = (avg_var + 1e-6).sqrt()
            # Scale the norm weights to compensate for distribution shift
            # If rms is large, scale down; if small, scale up
            target_rms = rms.mean()  # target uniform RMS
            scale = target_rms / rms
            scale = scale.to(layer.input_layernorm.weight.device)
            layer.input_layernorm.weight.data *= scale
```

- [ ] **Step 4: Run tests to verify they pass**
- [ ] **Step 5: Commit**

```bash
git add testing/llm_surgeon/surgery.py testing/tests/test_surgery.py
git commit -m "feat: add calibrate() for RMSNorm rescaling after surgery"
```

---

### Task 2: tracking.py

**Files:**
- Create: `testing/llm_surgeon/tracking.py`
- Create: `testing/tests/test_tracking.py`
- Modify: `testing/llm_surgeon/__init__.py`

- [ ] **Step 1: Write tests for tracking**

Create `testing/tests/test_tracking.py`:

```python
"""Tests for experiment tracking."""

import os
import pytest
from llm_surgeon.tracking import start, list_experiments, get_experiment, compare_experiments
from llm_surgeon.surgery import SurgeryLog


class TestTracking:
    def test_start_creates_experiment(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        exp = start(name="test-exp", description="A test", base_model="tiny", db_path=db_path)
        assert exp.name == "test-exp"

    def test_log_surgery(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        exp = start(name="test-exp", description="A test", base_model="tiny", db_path=db_path)
        log = SurgeryLog()
        log.add("remove_layers", "Removed [3]", 8, 7)
        exp.log_surgery(log)
        exp.finish()

    def test_log_metric(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        exp = start(name="test-exp", description="A test", base_model="tiny", db_path=db_path)
        exp.log_metric("perplexity", 8.7)
        exp.log_metric("accuracy", 0.45)
        exp.finish()

    def test_log_samples(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        exp = start(name="test-exp", description="A test", base_model="tiny", db_path=db_path)
        exp.log_samples([{"prompt": "test", "response": "output"}])
        exp.finish()

    def test_finish_sets_status(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        exp = start(name="test-exp", description="A test", base_model="tiny", db_path=db_path)
        exp.finish(notes="all done")
        data = get_experiment("test-exp", db_path=db_path)
        assert data["status"] == "completed"
        assert data["notes"] == "all done"

    def test_list_experiments(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        start(name="exp-1", description="First", base_model="tiny", db_path=db_path).finish()
        start(name="exp-2", description="Second", base_model="tiny", db_path=db_path).finish()
        exps = list_experiments(db_path=db_path)
        names = [e["name"] for e in exps]
        assert "exp-1" in names
        assert "exp-2" in names

    def test_get_experiment_returns_metrics(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        exp = start(name="test-exp", description="A test", base_model="tiny", db_path=db_path)
        exp.log_metric("perplexity", 6.5)
        exp.finish()
        data = get_experiment("test-exp", db_path=db_path)
        assert data["metrics"]["perplexity"] == 6.5

    def test_compare_experiments(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        e1 = start(name="exp-a", description="A", base_model="tiny", db_path=db_path)
        e1.log_metric("perplexity", 6.5)
        e1.finish()
        e2 = start(name="exp-b", description="B", base_model="tiny", db_path=db_path)
        e2.log_metric("perplexity", 12.3)
        e2.finish()
        comparison = compare_experiments(["exp-a", "exp-b"], db_path=db_path)
        assert "exp-a" in comparison
        assert "exp-b" in comparison
        assert comparison["exp-a"]["perplexity"] == 6.5
        assert comparison["exp-b"]["perplexity"] == 12.3

    def test_db_persists(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        start(name="persist-test", description="Test", base_model="tiny", db_path=db_path).finish()
        # Simulate process restart by calling list_experiments fresh
        exps = list_experiments(db_path=db_path)
        assert any(e["name"] == "persist-test" for e in exps)
```

- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Implement tracking.py**

Create `testing/llm_surgeon/tracking.py`:

```python
"""Experiment tracking with SQLite."""

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional, List


_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments.db",
)


def _get_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or _DEFAULT_DB
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            base_model TEXT,
            recipe_yaml TEXT,
            status TEXT DEFAULT 'running',
            notes TEXT,
            created_at TEXT,
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            key TEXT NOT NULL,
            value REAL NOT NULL,
            FOREIGN KEY (experiment_name) REFERENCES experiments(name)
        );
        CREATE TABLE IF NOT EXISTS surgery_ops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            operation TEXT,
            description TEXT,
            layer_count_before INTEGER,
            layer_count_after INTEGER,
            FOREIGN KEY (experiment_name) REFERENCES experiments(name)
        );
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            data TEXT,
            FOREIGN KEY (experiment_name) REFERENCES experiments(name)
        );
    """)
    conn.commit()


class Experiment:
    """An active experiment being tracked."""

    def __init__(self, name: str, db_path: Optional[str] = None):
        self.name = name
        self._db_path = db_path

    def log_surgery(self, surgery_log) -> None:
        conn = _get_db(self._db_path)
        for op in surgery_log.ops:
            conn.execute(
                "INSERT INTO surgery_ops (experiment_name, operation, description, layer_count_before, layer_count_after) VALUES (?, ?, ?, ?, ?)",
                (self.name, op.operation, op.description, op.layer_count_before, op.layer_count_after),
            )
        conn.commit()
        conn.close()

    def log_metric(self, key: str, value: float) -> None:
        conn = _get_db(self._db_path)
        conn.execute(
            "INSERT INTO metrics (experiment_name, key, value) VALUES (?, ?, ?)",
            (self.name, key, value),
        )
        conn.commit()
        conn.close()

    def log_samples(self, samples: list) -> None:
        conn = _get_db(self._db_path)
        conn.execute(
            "INSERT INTO samples (experiment_name, data) VALUES (?, ?)",
            (self.name, json.dumps(samples)),
        )
        conn.commit()
        conn.close()

    def finish(self, notes: str = "") -> None:
        conn = _get_db(self._db_path)
        conn.execute(
            "UPDATE experiments SET status = 'completed', notes = ?, finished_at = ? WHERE name = ?",
            (notes, datetime.now().isoformat(), self.name),
        )
        conn.commit()
        conn.close()


def start(
    name: str,
    description: str = "",
    base_model: str = "",
    recipe: str = "",
    db_path: Optional[str] = None,
) -> Experiment:
    """Start a new experiment."""
    conn = _get_db(db_path)
    conn.execute(
        "INSERT INTO experiments (name, description, base_model, recipe_yaml, created_at) VALUES (?, ?, ?, ?, ?)",
        (name, description, base_model, recipe, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return Experiment(name, db_path=db_path)


def list_experiments(db_path: Optional[str] = None) -> list:
    conn = _get_db(db_path)
    rows = conn.execute("SELECT name, description, base_model, status, created_at FROM experiments ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_experiment(name: str, db_path: Optional[str] = None) -> dict:
    conn = _get_db(db_path)
    row = conn.execute("SELECT * FROM experiments WHERE name = ?", (name,)).fetchone()
    if row is None:
        conn.close()
        raise KeyError(f"Experiment '{name}' not found")
    result = dict(row)

    # Attach metrics
    metrics_rows = conn.execute("SELECT key, value FROM metrics WHERE experiment_name = ?", (name,)).fetchall()
    result["metrics"] = {r["key"]: r["value"] for r in metrics_rows}

    # Attach surgery ops
    ops_rows = conn.execute("SELECT operation, description, layer_count_before, layer_count_after FROM surgery_ops WHERE experiment_name = ?", (name,)).fetchall()
    result["surgery_ops"] = [dict(r) for r in ops_rows]

    conn.close()
    return result


def compare_experiments(names: list, db_path: Optional[str] = None) -> dict:
    result = {}
    for name in names:
        data = get_experiment(name, db_path=db_path)
        result[name] = data["metrics"]
    return result
```

- [ ] **Step 4: Update __init__.py**

```python
from llm_surgeon import surgery, verify, export, benchmark, inspect, tracking
```

- [ ] **Step 5: Run tests to verify they pass**
- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/tracking.py testing/tests/test_tracking.py testing/llm_surgeon/__init__.py
git commit -m "feat: add SQLite experiment tracking"
```

---

### Task 3: recipe.py

**Files:**
- Create: `testing/llm_surgeon/recipe.py`
- Create: `testing/tests/test_recipe.py`
- Modify: `testing/llm_surgeon/__init__.py`

- [ ] **Step 1: Write tests for recipe**

Create `testing/tests/test_recipe.py`:

```python
"""Tests for recipe module."""

import os
import pytest
import yaml
from llm_surgeon.recipe import parse_recipe, generate_layer_sweep


class TestParseRecipe:
    def test_parses_yaml(self, tmp_path):
        recipe_content = {
            "name": "test-recipe",
            "base_model": "some-model",
            "description": "A test recipe",
            "surgery": [
                {"remove_layers": [3, 4, 5]},
            ],
        }
        recipe_file = str(tmp_path / "recipe.yaml")
        with open(recipe_file, "w") as f:
            yaml.dump(recipe_content, f)

        recipe = parse_recipe(recipe_file)
        assert recipe["name"] == "test-recipe"
        assert recipe["surgery"][0]["remove_layers"] == [3, 4, 5]

    def test_validates_required_fields(self, tmp_path):
        recipe_file = str(tmp_path / "bad.yaml")
        with open(recipe_file, "w") as f:
            yaml.dump({"description": "missing name and base_model"}, f)
        with pytest.raises(ValueError, match="name"):
            parse_recipe(recipe_file)


class TestGenerateLayerSweep:
    def test_generates_correct_count(self, tmp_path):
        output_dir = str(tmp_path / "sweep")
        files = generate_layer_sweep(
            num_layers=8,
            base_model="test-model",
            output_dir=output_dir,
        )
        assert len(files) == 8
        assert all(os.path.exists(f) for f in files)

    def test_each_file_removes_one_layer(self, tmp_path):
        output_dir = str(tmp_path / "sweep")
        files = generate_layer_sweep(
            num_layers=8,
            base_model="test-model",
            output_dir=output_dir,
        )
        for i, fpath in enumerate(files):
            with open(fpath) as f:
                recipe = yaml.safe_load(f)
            assert recipe["surgery"][0]["remove_layers"] == [i]

    def test_file_names_are_descriptive(self, tmp_path):
        output_dir = str(tmp_path / "sweep")
        files = generate_layer_sweep(
            num_layers=4,
            base_model="test-model",
            output_dir=output_dir,
        )
        for f in files:
            assert "remove-layer" in os.path.basename(f)
```

Also add a test for `run()` using the tiny_llama fixture (but this needs the full pipeline, so we'll make it conditional):

```python
class TestRun:
    def test_run_surgery_only(self, tiny_llama, tiny_llama_config, tmp_path):
        """Test recipe execution with surgery only (no export/eval)."""
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)

        recipe_content = {
            "name": "test-run",
            "base_model": "unused",  # we'll pass model directly
            "description": "Test surgery-only recipe",
            "surgery": [
                {"remove_layers": [3, 4, 5]},
            ],
        }
        recipe_file = str(tmp_path / "recipe.yaml")
        with open(recipe_file, "w") as f:
            yaml.dump(recipe_content, f)

        from llm_surgeon.recipe import run
        result = run(
            recipe_file,
            model=tiny_llama,
            tokenizer=tokenizer,
            db_path=str(tmp_path / "test.db"),
            skip_export=True,
            skip_eval=True,
        )
        assert result["name"] == "test-run"
        assert len(tiny_llama.model.layers) == 5
```

- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Implement recipe.py**

Create `testing/llm_surgeon/recipe.py`:

```python
"""Declarative experiment recipes (YAML)."""

import glob
import os
from typing import Optional, List

import yaml

from llm_surgeon import surgery, verify, tracking


def parse_recipe(path: str) -> dict:
    """Parse a YAML recipe file and validate required fields."""
    with open(path, "r") as f:
        recipe = yaml.safe_load(f)

    if "name" not in recipe:
        raise ValueError("Recipe must have a 'name' field")
    if "base_model" not in recipe:
        raise ValueError("Recipe must have a 'base_model' field")

    return recipe


def run(
    recipe_path: str,
    model=None,
    tokenizer=None,
    db_path: Optional[str] = None,
    skip_export: bool = False,
    skip_eval: bool = False,
) -> dict:
    """Execute a recipe file.

    If model/tokenizer are not provided, loads from recipe's base_model.

    Args:
        recipe_path: Path to YAML recipe file
        model: Pre-loaded model (optional, for testing)
        tokenizer: Pre-loaded tokenizer (optional)
        db_path: SQLite database path (optional)
        skip_export: Skip GGUF export step
        skip_eval: Skip evaluation steps

    Returns:
        Dict with experiment name and results.
    """
    recipe = parse_recipe(recipe_path)
    recipe_yaml = open(recipe_path).read()

    # Start tracking
    exp = tracking.start(
        name=recipe["name"],
        description=recipe.get("description", ""),
        base_model=recipe["base_model"],
        recipe=recipe_yaml,
        db_path=db_path,
    )

    # Load model if not provided
    if model is None:
        model, tokenizer = surgery.load_model(recipe["base_model"], mode="export")

    # Execute surgery steps
    for step in recipe.get("surgery", []):
        if "remove_layers" in step:
            log = surgery.remove_layers(model, step["remove_layers"])
            exp.log_surgery(log)
        elif "keep_layers" in step:
            log = surgery.keep_layers(model, step["keep_layers"])
            exp.log_surgery(log)
        elif "reorder_layers" in step:
            log = surgery.reorder_layers(model, step["reorder_layers"])
            exp.log_surgery(log)
        elif "swap_layers" in step:
            pair = step["swap_layers"]
            log = surgery.swap_layers(model, pair[0], pair[1])
            exp.log_surgery(log)
        elif "duplicate_layer" in step:
            d = step["duplicate_layer"]
            log = surgery.duplicate_layer(model, d["src"], d["dst"])
            exp.log_surgery(log)
        elif "calibrate" in step:
            cal = step["calibrate"]
            surgery.calibrate(
                model, tokenizer,
                dataset=cal.get("dataset", "wikitext2"),
                num_samples=cal.get("num_samples", 128),
            )

    # Structural verification (automatic)
    verify.check_structure(model)

    # Evaluation
    if not skip_eval:
        eval_config = recipe.get("evaluate", {})

        if "perplexity" in eval_config and tokenizer is not None:
            from llm_surgeon import benchmark
            ppl_cfg = eval_config["perplexity"]
            ppl = benchmark.perplexity(
                model, tokenizer,
                dataset=ppl_cfg.get("dataset", "wikitext2"),
                max_samples=ppl_cfg.get("max_samples"),
            )
            exp.log_metric(f"perplexity_{ppl_cfg.get('dataset', 'wikitext2')}", ppl)

    # Export
    if not skip_export:
        export_config = recipe.get("export", {})
        if export_config:
            from llm_surgeon import export
            result = export.full_pipeline(
                model,
                tokenizer=tokenizer,
                name=export_config.get("ollama_name", recipe["name"]),
                quantization=export_config.get("quantization", "Q4_K_M"),
                output_dir=export_config.get("output_dir", "outputs"),
            )

    exp.finish(notes="Recipe completed successfully")

    return {"name": recipe["name"], "recipe": recipe}


def run_batch(pattern: str, **kwargs) -> list:
    """Run all recipe files matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        print(f"Running recipe: {f}")
        results.append(run(f, **kwargs))
    return results


def generate_layer_sweep(
    num_layers: int,
    base_model: str,
    output_dir: str,
    description_prefix: str = "Remove layer",
) -> List[str]:
    """Generate one recipe file per layer removal.

    Creates num_layers YAML files, each removing a single layer.

    Returns list of created file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = []

    for i in range(num_layers):
        recipe = {
            "name": f"remove-layer-{i}",
            "base_model": base_model,
            "description": f"{description_prefix} {i}",
            "surgery": [
                {"remove_layers": [i]},
            ],
            "evaluate": {
                "perplexity": {"dataset": "wikitext2"},
            },
        }
        path = os.path.join(output_dir, f"remove-layer-{i:02d}.yaml")
        with open(path, "w") as f:
            yaml.dump(recipe, f, default_flow_style=False)
        files.append(path)

    return files
```

- [ ] **Step 4: Update __init__.py**

```python
from llm_surgeon import surgery, verify, export, benchmark, inspect, tracking, recipe
```

- [ ] **Step 5: Install pyyaml if not present**

```bash
/home/ai/ai-projects/llm/testing/.venv/bin/pip install pyyaml
```

- [ ] **Step 6: Run tests to verify they pass**
- [ ] **Step 7: Run full test suite**
- [ ] **Step 8: Commit**

```bash
git add testing/llm_surgeon/recipe.py testing/tests/test_recipe.py testing/llm_surgeon/__init__.py
git commit -m "feat: add YAML recipe execution and layer sweep generation"
```

---

## Final State

```
testing/
  llm_surgeon/
    __init__.py          — imports all 7 modules
    surgery.py           — + calibrate()
    verify.py            — (Phase 4)
    export.py            — (Phase 2)
    benchmark.py         — (Phases 3+5)
    inspect.py           — (Phase 4)
    tracking.py          — SQLite experiment tracking
    recipe.py            — YAML recipe parsing + execution
  tests/
    test_surgery.py      — + calibration tests
    test_verify.py       — (Phase 4)
    test_export.py       — (Phase 2)
    test_benchmark.py    — (Phases 3+5)
    test_inspect.py      — (Phase 4)
    test_tracking.py     — experiment tracking tests
    test_recipe.py       — recipe parsing + execution tests
  prompts/
    default.json         — (Phase 5)
  experiments.db         — auto-created on first use
```

All tests pass. Full toolkit complete.
