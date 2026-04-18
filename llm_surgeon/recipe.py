"""YAML recipe parsing and execution for llm-surgeon."""

import glob as _glob
import os
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

def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[recipe] {msg}")


def _apply_surgery_step(
    model, tokenizer, step: Dict[str, Any], verbose: bool = False,
    baseline_stats: Optional[surgery.CalibrationStats] = None,
) -> Optional[surgery.SurgeryLog]:
    """Execute a single surgery step dict and return the SurgeryLog (or None for calibrate)."""
    if "remove_layers" in step:
        _log(f"remove_layers({step['remove_layers']})", verbose)
        return surgery.remove_layers(model, step["remove_layers"])
    if "keep_layers" in step:
        _log(f"keep_layers({step['keep_layers']})", verbose)
        return surgery.keep_layers(model, step["keep_layers"])
    if "reorder_layers" in step:
        _log(f"reorder_layers({step['reorder_layers']})", verbose)
        return surgery.reorder_layers(model, step["reorder_layers"])
    if "swap_layers" in step:
        args = step["swap_layers"]
        _log(f"swap_layers({args[0]}, {args[1]})", verbose)
        return surgery.swap_layers(model, args[0], args[1])
    if "duplicate_layer" in step:
        args = step["duplicate_layer"]
        _log(f"duplicate_layer(src={args[0]}, dst={args[1]})", verbose)
        return surgery.duplicate_layer(model, src=args[0], dst=args[1])
    if "calibrate" in step:
        opts = step["calibrate"] or {}
        _log(f"calibrate(dataset={opts.get('dataset', 'wikitext2')}, num_samples={opts.get('num_samples', 128)})", verbose)
        surgery.calibrate(
            model,
            tokenizer,
            baseline_stats=baseline_stats,
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
    verbose: bool = True,
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

    Returns a result dict with name, status, and any collected metrics/paths.
    """
    recipe_data = parse_recipe(recipe_path)
    name = recipe_data["name"]
    base_model = recipe_data.get("base_model", "")
    description = recipe_data.get("description", "")

    _log(f"Starting experiment: {name}", verbose)
    _log(f"Base model: {base_model}", verbose)

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

    result: Dict[str, Any] = {"name": name}

    # Load model if not supplied
    if model is None:
        _log(f"Loading model in export mode...", verbose)
        model, tokenizer = surgery.load_model(base_model, mode="export")
        _log(f"Model loaded ({len(model.model.layers)} layers)", verbose)

    # Execute surgery steps
    steps = recipe_data.get("surgery", [])

    # If calibration is requested, capture baseline stats BEFORE surgery
    baseline_stats: Optional[surgery.CalibrationStats] = None
    has_calibrate = any("calibrate" in s for s in steps)
    if has_calibrate:
        _log("Capturing baseline calibration stats (pre-surgery)...", verbose)
        cal_step = next(s for s in steps if "calibrate" in s)
        cal_opts = cal_step["calibrate"] or {}
        baseline_stats = surgery.capture_calibration_stats(
            model, tokenizer,
            text=cal_opts.get("text"),
            dataset=cal_opts.get("dataset"),
            num_samples=int(cal_opts.get("num_samples", 128)),
        )
        _log(f"Captured RMS for {len(baseline_stats)} layers", verbose)

    combined_log = surgery.SurgeryLog()
    for step in steps:
        log = _apply_surgery_step(model, tokenizer, step, verbose=verbose, baseline_stats=baseline_stats)
        if log is not None:
            combined_log.ops.extend(log.ops)

    # Always verify structure after surgery
    _log("Verifying model structure...", verbose)
    report = verify.check_structure(model)
    _log(f"Structure verification: {report}", verbose)

    if combined_log.ops:
        exp.log_surgery(combined_log)
        _log(f"Logged {len(combined_log.ops)} surgery operations", verbose)

    # Analyze
    analyze_cfg = recipe_data.get("analyze", {})
    if analyze_cfg:
        _log("Running analysis...", verbose)
        analyze_results = _run_analyze(model, tokenizer, analyze_cfg, exp, verbose)
        result["analyze"] = analyze_results

    # Evaluation
    if not skip_eval:
        eval_cfg = recipe_data.get("evaluate", {})
        if eval_cfg:
            _log("Running evaluation...", verbose)
            eval_results = _run_evaluation(model, tokenizer, eval_cfg, exp, verbose)
            result["eval"] = eval_results

    # Export
    if not skip_export:
        export_cfg = recipe_data.get("export", {})
        if export_cfg:
            _log("Running export pipeline...", verbose)
            export_results = _run_export(model, tokenizer, name, export_cfg, verbose)
            result["export"] = export_results

    exp.finish()
    result["status"] = "completed"
    _log(f"Experiment '{name}' completed.", verbose)
    return result


def _run_evaluation(
    model, tokenizer, eval_cfg: Dict, exp: tracking.Experiment, verbose: bool,
) -> Dict[str, Any]:
    """Run evaluation steps as specified in the recipe's evaluate section."""
    results = {}

    if "perplexity" in eval_cfg:
        from llm_surgeon import benchmark
        ppl_cfg = eval_cfg["perplexity"] or {}
        dataset = ppl_cfg.get("dataset", "wikitext2")
        max_samples = ppl_cfg.get("max_samples")
        _log(f"Measuring perplexity on {dataset} (max_samples={max_samples})...", verbose)
        ppl = benchmark.perplexity(model, tokenizer, dataset=dataset, max_samples=max_samples, verbose=verbose)
        exp.log_metric(f"perplexity_{dataset}", ppl)
        results[f"perplexity_{dataset}"] = ppl
        _log(f"Perplexity ({dataset}): {ppl:.2f}", verbose)

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

    return results


def _run_analyze(
    model, tokenizer, analyze_cfg: Dict, exp: tracking.Experiment, verbose: bool,
) -> Dict[str, Any]:
    """Run analysis steps as specified in the recipe's analyze section."""
    from llm_surgeon import probe

    results = {}

    if "logit_lens" in analyze_cfg:
        ll_cfg = analyze_cfg["logit_lens"] or {}
        prompt = ll_cfg.get("prompt", "")
        top_k = int(ll_cfg.get("top_k", 5))
        _log(f"Running logit lens (top_k={top_k})...", verbose)
        ll_result = probe.logit_lens(model, tokenizer, prompt, top_k=top_k)

        num_positions = len(ll_result.prompt_tokens)
        last_pos = num_positions - 1
        flips = ll_result.prediction_flips(last_pos)
        exp.log_metric("logit_lens_prediction_flips", flips)

        results["logit_lens"] = {
            "num_layers_captured": len(set(p["layer"] for p in ll_result.predictions)),
            "prediction_flips": flips,
        }
        _log(f"Logit lens: {results['logit_lens']}", verbose)

    if "hidden_states" in analyze_cfg:
        hs_cfg = analyze_cfg["hidden_states"] or {}
        prompt = hs_cfg.get("prompt", "")
        _log(f"Extracting hidden states...", verbose)
        hs = probe.extract_hidden_states(model, tokenizer, prompt)
        results["hidden_states"] = {
            "num_capture_points": len(hs.states),
        }
        _log(f"Hidden states: {len(hs.states)} capture points", verbose)

    return results


def _run_export(
    model, tokenizer, name: str, export_cfg: Dict, verbose: bool,
) -> Dict[str, Any]:
    """Run export steps as specified in the recipe's export section."""
    from llm_surgeon import export as _export

    quantization = export_cfg.get("quantization", "Q4_K_M")
    ollama_name = export_cfg.get("ollama_name", name)
    output_dir = export_cfg.get("output_dir", "outputs")

    _log(f"Exporting: quantization={quantization}, ollama_name={ollama_name}", verbose)
    result = _export.full_pipeline(
        model,
        tokenizer=tokenizer,
        name=ollama_name,
        quantization=quantization,
        output_dir=output_dir,
    )
    _log(f"GGUF saved to {result['gguf_path']}", verbose)
    if result.get("registered"):
        _log(f"Registered as '{ollama_name}' in ollama", verbose)

    return result


def run_batch(pattern: str, **kwargs) -> List[Dict]:
    """Run all recipe files matching the glob pattern.

    Returns a list of result dicts from each run() call.
    """
    files = sorted(_glob.glob(pattern))
    results = []
    for fpath in files:
        print(f"\n{'='*60}")
        print(f"Running recipe: {fpath}")
        print(f"{'='*60}")
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
