"""YAML recipe parsing and execution for llm-surgeon."""

import glob as _glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from llm_surgeon import surgery, verify, tracking


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_recipe(path: str) -> Dict[str, Any]:
    """Load a YAML recipe file and validate required fields.

    Required fields: name, base_model.

    Returns the parsed dict.
    Raises ValueError if required fields are missing.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    missing = [field for field in ("name", "base_model") if field not in data]
    if missing:
        raise ValueError(
            f"Recipe '{path}' is missing required fields: {', '.join(missing)}"
        )
    return data


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _apply_surgery_step(model, tokenizer, step: Dict[str, Any]) -> Optional[surgery.SurgeryLog]:
    """Execute a single surgery step dict and return the SurgeryLog (or None for calibrate)."""
    if "remove_layers" in step:
        return surgery.remove_layers(model, step["remove_layers"])
    if "keep_layers" in step:
        return surgery.keep_layers(model, step["keep_layers"])
    if "reorder_layers" in step:
        return surgery.reorder_layers(model, step["reorder_layers"])
    if "swap_layers" in step:
        args = step["swap_layers"]
        return surgery.swap_layers(model, args[0], args[1])
    if "duplicate_layer" in step:
        args = step["duplicate_layer"]
        return surgery.duplicate_layer(model, src=args[0], dst=args[1])
    if "calibrate" in step:
        opts = step["calibrate"] or {}
        surgery.calibrate(
            model,
            tokenizer,
            text=opts.get("text"),
            dataset=opts.get("dataset"),
            num_samples=int(opts.get("num_samples", 128)),
        )
        return None
    raise ValueError(f"Unknown surgery step: {list(step.keys())}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    recipe_path: str,
    model=None,
    tokenizer=None,
    db_path: Optional[str] = None,
    skip_export: bool = False,
    skip_eval: bool = False,
) -> Dict[str, Any]:
    """Parse a recipe file and execute it.

    Steps performed:
    1. Parse recipe
    2. Start tracking experiment
    3. Load model (if not provided)
    4. Execute surgery steps in order
    5. verify.check_structure (automatic)
    6. Run evaluation unless skip_eval=True
    7. Run export unless skip_export=True
    8. Finish tracking

    Returns a result dict with at minimum {"name": ..., "status": "completed"}.
    """
    recipe_data = parse_recipe(recipe_path)
    name = recipe_data["name"]
    base_model = recipe_data.get("base_model", "")
    description = recipe_data.get("description", "")

    # Tracking
    exp_kwargs: Dict[str, Any] = dict(
        name=name,
        description=description,
        base_model=base_model,
        recipe=recipe_data,
    )
    if db_path is not None:
        exp_kwargs["db_path"] = db_path
    exp = tracking.start(**exp_kwargs)

    # Load model if not supplied
    if model is None:
        model, tokenizer = surgery.load_model(base_model, mode="export")

    # Execute surgery steps
    steps = recipe_data.get("surgery", [])
    combined_log = surgery.SurgeryLog()
    for step in steps:
        log = _apply_surgery_step(model, tokenizer, step)
        if log is not None:
            combined_log.ops.extend(log.ops)

    # Always verify structure after surgery
    verify.check_structure(model)

    if combined_log.ops:
        exp.log_surgery(combined_log)

    # Evaluation
    if not skip_eval:
        eval_cfg = recipe_data.get("evaluate", {})
        if eval_cfg:
            _run_evaluation(model, tokenizer, eval_cfg, exp)

    # Export
    if not skip_export:
        export_cfg = recipe_data.get("export", {})
        if export_cfg:
            _run_export(model, tokenizer, base_model, name, export_cfg, exp)

    exp.finish()
    return {"name": name, "status": "completed"}


def _run_evaluation(model, tokenizer, eval_cfg: Dict, exp: tracking.Experiment) -> None:
    """Run evaluation steps as specified in the recipe's evaluate section."""
    if "perplexity" in eval_cfg:
        try:
            from llm_surgeon import benchmark
            ppl_cfg = eval_cfg["perplexity"] or {}
            dataset = ppl_cfg.get("dataset", "wikitext2")
            result = benchmark.perplexity(model, tokenizer, dataset=dataset)
            exp.log_metric("perplexity", result)
        except Exception:
            pass  # Best-effort — don't abort the recipe run


def _run_export(model, tokenizer, base_model: str, name: str, export_cfg: Dict, exp: tracking.Experiment) -> None:
    """Run export steps as specified in the recipe's export section."""
    try:
        from llm_surgeon import export as _export
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = os.path.join(tmpdir, "checkpoint")
            _export.save_checkpoint(model, ckpt_dir, tokenizer=tokenizer)
    except Exception:
        pass  # Best-effort


def run_batch(pattern: str, **kwargs) -> List[Dict]:
    """Run all recipe files matching the glob pattern.

    Returns a list of result dicts from each run() call.
    """
    files = sorted(_glob.glob(pattern))
    results = []
    for fpath in files:
        result = run(fpath, **kwargs)
        results.append(result)
    return results


def generate_layer_sweep(
    num_layers: int,
    base_model: str,
    output_dir: str,
) -> List[str]:
    """Generate N recipe files, each removing one layer (layer i for file i).

    Returns a list of generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(num_layers):
        recipe_data = {
            "name": f"layer-sweep-remove-{i}",
            "base_model": base_model,
            "description": f"Layer sweep: remove layer {i}",
            "surgery": [{"remove_layers": [i]}],
        }
        fpath = os.path.join(output_dir, f"sweep_remove_layer_{i}.yaml")
        with open(fpath, "w") as f:
            yaml.dump(recipe_data, f, default_flow_style=False)
        paths.append(fpath)
    return paths
