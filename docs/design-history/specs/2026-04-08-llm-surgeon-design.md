# LLM Surgeon — Local Model Surgery Toolkit

## Goal

A Python toolkit for surgical layer-level manipulation of LLaMA 3 8B models, with quantitative evaluation (perplexity, downstream benchmarks, activation analysis), export to GGUF for fast inference in ollama, and experiment tracking. Enables systematic experimentation with layer reordering, removal, duplication, and other structural changes to study their effects on model behavior.

## Target Model

LLaMA 3 8B (decoder-only, 32 layers, d=4096, h=32). Single architecture to start; generalization to other architectures is out of scope for v1.

**Terminology:** A "layer" in this toolkit refers to a full `LlamaDecoderLayer` — both the self-attention block and the MLP block, plus their associated RMSNorm layers. When you `remove_layers(model, [16])`, you remove the attention, MLP, `input_layernorm`, and `post_attention_layernorm` at position 16. Sub-layer surgery (removing just the MLP or just the attention from a layer) is out of scope for v1.

## Hardware Constraints

- GPU: NVIDIA RTX 2080 (8GB VRAM)
- RAM: 32GB
- CPU: Intel i7-8700K (6 cores / 12 threads)
- Storage: NVMe SSDs

Implications: 4-bit quantization (bitsandbytes) for GPU inspection, fp16 on CPU for export and perplexity evaluation. Both fit within available resources.

**Memory limits for export mode:** LLaMA 3 8B in fp16 uses ~16GB RAM. After surgery that increases model size (e.g. `duplicate_layer`), the model can grow. With OS overhead (~4-6GB), the practical ceiling is roughly 40 layers at fp16 on 32GB RAM. Operations that increase model size should warn when approaching this limit. Operations that only remove or reorder layers stay within bounds.

## Project Structure

```
testing/
  llm_surgeon/
    __init__.py
    surgery.py        — model loading, layer manipulation, calibration
    verify.py         — structural checks, activation comparison, cached baselines
    inspect.py        — weight/activation analysis (norms, BI scores, entropy, SVD)
    export.py         — HF checkpoint -> GGUF -> quantize -> ollama
    benchmark.py      — perplexity, downstream evals, generation comparison
    tracking.py       — experiment logging (SQLite)
    recipe.py         — declarative experiment definitions (YAML)
  experiments/        — experiment scripts and recipe files
    example_skip_layer.py
    example_skip_layer.yaml
  prompts/            — test prompt sets for generation comparison
    default.json
  outputs/            — generated checkpoints, GGUF files, results
    baselines/        — cached activation baselines
  experiments.db      — SQLite experiment log (auto-created)
  requirements.txt
  pyproject.toml
```

## Module: `surgery.py`

### Model Loading

Three modes depending on purpose:

- **`inspect` mode**: bitsandbytes 4-bit + `device_map="auto"`. Fits in 8GB VRAM. For quick inspection, activation analysis, and generation comparison.
- **`eval` mode**: fp16 on CPU with GPU offload where possible. For perplexity and downstream evaluation where quantization noise would contaminate measurements.
- **`export` mode**: fp16 on CPU (~16GB RAM). Full precision weights for clean surgery + GGUF export.

Both `eval` and `export` use fp16 weights. The distinction is intent: `eval` is for measurement (may use `device_map="auto"` to split across GPU+CPU for speed), `export` forces CPU-only to ensure consistent serialization.

```python
model, tokenizer = surgeon.load_model("meta-llama/Meta-Llama-3-8B", mode="inspect")
model, tokenizer = surgeon.load_model("meta-llama/Meta-Llama-3-8B", mode="eval")
model, tokenizer = surgeon.load_model("meta-llama/Meta-Llama-3-8B", mode="export")
```

### Layer Operations

All operations modify the model in-place and update `config.num_hidden_layers`. All return a `SurgeryLog` (list of changes made).

| Function | Description |
|----------|-------------|
| `remove_layers(model, [16, 17, 18])` | Delete specified layers |
| `keep_layers(model, [0, 1, ..., 15])` | Keep only listed layers, remove the rest |
| `reorder_layers(model, [0, 1, 2, 15, 14, 13, ...])` | Rearrange to specified order |
| `duplicate_layer(model, src=10, dst=11)` | Deep-copy a layer and insert at position (warns if approaching memory limit) |
| `swap_layers(model, 5, 12)` | Swap two layers' positions |
| `get_layer_info(model)` | Print summary: layer count, param counts per layer, total params, estimated memory |

**Index semantics:** Operations use layer indices as they exist at call time, not original indices. This is simpler and avoids ambiguity when chaining multiple operations.

**`SurgeryLog`:** A dataclass containing a list of `SurgeryOp` entries (each has `operation`, `description`, and `layer_count_before`/`layer_count_after`). Passed to `verify.check_structure()` for cross-referencing. Also recorded by experiment tracking.

### Calibration

When layers are reordered or removed, the input distribution to downstream layers shifts — the residual stream statistics no longer match what those layers were trained to expect. Without correction, most non-trivial rearrangements produce degraded or garbage output.

**v1 approach: LayerNorm rescaling.** This is the simplest calibration strategy — run calibration samples through the modified model, collect residual stream statistics at each layer boundary, and rescale RMSNorm parameters to normalize the shifted distributions.

```python
surgery.calibrate(model, tokenizer, dataset="wikitext2", num_samples=128)
```

- Runs forward pass over calibration samples
- Collects residual stream mean/variance at each layer boundary
- Adjusts RMSNorm gain parameters to compensate
- Lightweight — no gradient computation, no weight modification beyond norms
- Optional: skip to measure raw surgery impact without correction

**Future calibration approaches (not in v1):**
- Calibration-aware weight projection (SliceGPT approach) — projects weight matrices to account for changed input distributions
- Knowledge distillation on calibration set — a few gradient steps to realign modified model outputs to match original
- DARE-style weight rescaling — scale remaining weights to compensate for removed capacity

## Module: `verify.py`

### Level 1: Structural Checks (fast, no inference)

```python
report = verify.check_structure(model, surgery_log)
```

- `len(model.model.layers)` matches `model.config.num_hidden_layers`
- All layers properly indexed (no gaps)
- Embedding dimensions match layer input dimensions
- Output head (lm_head) dimensions consistent
- Cross-references `SurgeryLog` — e.g. if 3 layers removed, confirms count dropped by 3
- Returns `VerifyReport` with pass/fail + details. Raises on critical inconsistencies.

### Level 2: Activation Comparison (runs inference)

```python
diff = verify.compare_activations(original, modified, prompt, layers="all")
```

- Runs forward pass on both models with same input
- Captures activations at every layer boundary via PyTorch forward hooks
- Per-layer metrics: cosine similarity, L2 distance, max absolute difference
- Output table shows exactly where divergence starts
- Both models in `inspect` mode (4-bit), one forward pass at a time, activations moved to CPU
- For different layer counts: aligns from layer 0, compares up to shorter model's depth

### Cached Baselines

Loading the original model fresh for every comparison is slow. Instead, cache activations once and reuse across experiments:

```python
# Cache once (saves activations to disk as .pt files)
verify.cache_baseline(
    model=original_model,
    tokenizer=tokenizer,
    prompts=["The capital of France is", "Explain gravity"],
    cache_dir="outputs/baselines/llama3-8b/"
)

# Compare against cache (only loads the modified model)
diff = verify.compare_to_baseline(
    model=modified_model,
    tokenizer=tokenizer,
    prompts=["The capital of France is"],
    cache_dir="outputs/baselines/llama3-8b/"
)
```

- Baseline cache stores per-layer activations as PyTorch tensors on disk
- Keyed by prompt text (hashed) so lookups are fast
- One-time cost to build the cache, then reusable across many experiments

Verification is separate from surgery — optional, not baked in.

## Module: `inspect.py`

Analysis tools for understanding individual layers before and after surgery. Helps decide *which* layers to target rather than doing it blind.

### Block Influence (BI) Score

The single most predictive metric for "can I safely remove this layer" (from ShortGPT). Measures how much each layer transforms the residual stream:

```python
bi_scores = inspect.block_influence(model, tokenizer, prompts=["The capital of France is"])
# → dict mapping layer index to BI score (0.0 = identity/safe to remove, 1.0 = maximal transformation)
```

- Computes cosine similarity between each layer's input and output
- BI = 1 - cosine_similarity (so higher = more influential, lower = more redundant)
- Averaged across tokens and prompts for stability
- This is the primary decision tool: sort layers by BI score, candidates for removal are those with the lowest scores

### Weight Analysis

```python
inspect.weight_norms(model)
# → table: layer index, attn weight Frobenius norm, MLP weight norm, total norm

inspect.weight_svd(model, layers=[10, 11, 12])
# → singular value spectrum of weight matrices for specified layers
# → helps identify redundant layers (similar spectra) or critical layers (distinct spectra)
```

### Activation Analysis (requires forward pass)

```python
inspect.attention_entropy(model, tokenizer, prompt="The capital of France is")
# → per-layer, per-head entropy of attention distributions
# → low entropy = focused attention, high = diffuse/possibly redundant

inspect.residual_stream_norms(model, tokenizer, prompt="The capital of France is")
# → magnitude of residual stream at each layer boundary
# → sudden jumps or drops indicate layers with outsized impact
```

- All functions work on 4-bit (inspect mode) models
- Output is both printed tables and returned as dicts for programmatic use
- These are read-only analysis tools — they don't modify the model

## Module: `export.py`

### Step 1: Save HuggingFace Checkpoint

```python
checkpoint_path = export.save_checkpoint(model, "outputs/llama3-no-layers-16-18")
```

- `model.save_pretrained()` + `tokenizer.save_pretrained()`
- Guarantees sequential layer numbering 0 to N-1
- Writes `config.json` with correct `num_hidden_layers`
- Validates checkpoint is loadable before returning

### Step 2: Convert to GGUF + Quantize

```python
gguf_path = export.to_gguf(checkpoint_path, output_dir="outputs/gguf/", quantization="Q4_K_M")
```

- Shells out to `convert_hf_to_gguf.py` (produces f16 GGUF)
- Shells out to `llama-quantize` (produces quantized GGUF)
- Cleans up intermediate f16 GGUF
- Requires local `llama.cpp` build. Path set via `LLAMA_CPP_PATH` env var or config file.

### Step 3: Register with Ollama

```python
export.register_ollama(gguf_path, name="llama3-no-layers-16-18")
```

- Generates `Modelfile` with `FROM /absolute/path/to/model.gguf`
- Runs `ollama create <name> -f Modelfile`
- Confirms registration via `ollama list`

### Convenience Wrapper

```python
export.full_pipeline(model, name="llama3-no-layers-16-18", quantization="Q4_K_M", output_dir="outputs/")
```

Runs all three steps in sequence.

## Module: `benchmark.py`

### Perplexity Measurement

The primary quantitative metric for evaluating surgery impact.

```python
ppl = benchmark.perplexity(model, tokenizer, dataset="wikitext2")
# → returns a single float: perplexity on the dataset
```

- Supports standard datasets: WikiText-2, C4 (small subset)
- Sliding window with stride = context_length / 2 (4096 for LLaMA 3's 8192 context)
- **Must run on fp16 weights, not 4-bit.** Quantization noise contaminates perplexity measurements and can mask or exaggerate the effect of surgery. Use `eval` mode or `export` mode for this.
- This is the number you compare across experiments: "baseline perplexity is 6.2, removing layers 16-18 gives 8.7"

### Downstream Task Evaluation

Perplexity measures general language modeling, but layer surgery often has selective damage — a model can keep low perplexity but lose reasoning or factual recall. Standard benchmarks catch this.

```python
results = benchmark.eval_downstream(
    model_path="outputs/llama3-no-layers-16-18",  # HF checkpoint path
    tasks=["arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc2"],
    num_fewshot=5
)
# → dict of task → accuracy
```

- Shells out to EleutherAI's `lm-evaluation-harness` (`lm_eval` CLI)
- Does not reimplement any evaluation logic — just orchestrates the tool and collects results
- These are the benchmarks every pruning paper reports alongside perplexity
- `model_path` accepts both HuggingFace model IDs (e.g. `"meta-llama/Meta-Llama-3-8B"` for baseline) and local checkpoint paths (e.g. `"outputs/llama3-no-layers-16-18"` for modified)
- Runs on fp16 weights, not GGUF — `lm_eval` works with HF models natively
- Slow (potentially hours for full MMLU) — run selectively. ARC-Challenge + HellaSwag are fastest and most informative for surgery damage assessment.

### Generation Comparison via Ollama

```python
results = benchmark.compare(
    models=["llama3:8b", "llama3-no-layers-16-18"],
    prompts="prompts/default.json",
    temperature=0.0
)
```

- Sends each prompt to each model via ollama HTTP API (`localhost:11434/api/generate`)
- Collects: response text, tokens/second, total tokens generated
- Temperature 0.0 for near-reproducibility (note: quantized models in llama.cpp have numerical non-determinism from parallel reduction — outputs may vary slightly across runs even at temp 0, don't chase phantom differences)

### Generation Metrics

Automated heuristics that catch common surgery failure modes without needing an LLM judge:

```python
metrics = benchmark.generation_metrics(results)
```

| Metric | What it catches |
|--------|----------------|
| Mean output length | Model that collapses to short/empty outputs |
| Vocabulary diversity (unique tokens / total tokens) | Model that becomes monotonic |
| Repetition rate (fraction of repeated n-grams, n=3) | Model that loops |
| Coherence heuristic (% of outputs that are valid UTF-8 / parseable text) | Model that outputs garbled tokens |

- Computed automatically on generation comparison results
- Not quality metrics — these are failure detectors. A model that passes all four can still be bad, but a model that fails any of them is definitely broken.

### Prompt Sets

JSON files with categorized prompts:

```json
[
    {"prompt": "The capital of France is", "category": "factual"},
    {"prompt": "Explain quantum entanglement simply", "category": "reasoning"},
    {"prompt": "Write a haiku about rain", "category": "creative"}
]
```

## Module: `tracking.py`

Local experiment tracking — structured logging so you don't lose track across dozens of experiments.

```python
# Start an experiment
exp = tracking.start(
    name="remove-layers-16-18",
    description="Remove middle layers 16-18, no calibration",
    base_model="meta-llama/Meta-Llama-3-8B",
    recipe="experiments/remove_mid_layers.yaml"  # optional: link to recipe file
)

# Record surgery
exp.log_surgery(surgery_log)

# Record metrics
exp.log_metric("perplexity_wikitext2", 8.7)
exp.log_metric("arc_challenge_acc", 0.41)
exp.log_metric("hellaswag_acc", 0.68)

# Record generation samples
exp.log_samples(benchmark_results)

# Record generation metrics
exp.log_metric("mean_output_length", 47.3)
exp.log_metric("repetition_rate", 0.02)

# Finish
exp.finish(notes="Coherent output but factual accuracy degraded")
```

### Storage

- SQLite database (`experiments.db`) in the project root
- Schema: experiments table (id, name, description, base_model, recipe_path, timestamp, status, notes), metrics table (experiment_id, key, value), surgery_ops table (experiment_id, operation details)
- Query with standard SQL or convenience functions:

```python
tracking.list_experiments()
tracking.get_experiment("remove-layers-16-18")
tracking.compare_experiments(["remove-layers-16-18", "remove-layers-28-30"])
# → side-by-side table of metrics
```

- No external dependencies (sqlite3 is in Python stdlib)

## Module: `recipe.py`

Declarative experiment definitions for reproducibility and batch runs. Once you're past the exploratory phase and running systematic sweeps ("remove each layer individually, measure perplexity"), writing a Python script per experiment becomes tedious. Recipes solve this.

### Format

```yaml
name: remove-layers-16-18
base_model: meta-llama/Meta-Llama-3-8B
description: Remove middle layers 16-18, calibrate, evaluate

surgery:
  - remove_layers: [16, 17, 18]
  - calibrate:
      dataset: wikitext2
      num_samples: 128

evaluate:
  perplexity:
    dataset: wikitext2
  downstream:
    tasks: [arc_challenge, hellaswag]
    num_fewshot: 5
  generation:
    prompts: prompts/default.json

export:
  quantization: Q4_K_M
  ollama_name: llama3-no-mid
```

### Execution

```python
recipe.run("experiments/remove_mid_layers.yaml")
# → loads model, applies surgery, verifies structure, calibrates, evaluates, exports, logs to tracking
```

### Batch Runs

```python
recipe.run_batch("experiments/sweep_*.yaml")
# → runs all matching recipes sequentially, tracking each as a separate experiment
```

- Structural verification (`verify.check_structure`) runs automatically after surgery steps — no need to specify it in the recipe
- Recipes are logged verbatim in the experiment tracker (the YAML content, not just a path)
- A recipe can also be generated programmatically for sweeps:

```python
recipe.generate_layer_sweep(
    layers=range(32),
    template="experiments/single_layer_removal_template.yaml",
    output_dir="experiments/sweep/"
)
# → generates 32 YAML files, one per layer removal
```

## Typical End-to-End Workflow

```
 1. inspect.block_influence(model, prompts)        — rank layers by removability (BI score)
 2. inspect.weight_norms(model)                     — secondary signal for layer importance
 3. inspect.attention_entropy(model, prompt)         — identify diffuse/redundant heads

 4. Load model in EXPORT mode (fp16, CPU)            — full precision for surgery + save
 5. Perform surgery (remove_layers, reorder, etc.)
 6. surgery.calibrate(model, dataset)                — optional: correct distribution shift
 7. verify.check_structure(model, surgery_log)       — fast structural sanity check
 8. export.full_pipeline(model, name, quant)         — save checkpoint → GGUF → ollama

 9. Load MODIFIED checkpoint in EVAL mode            — fp16 for clean measurement
10. benchmark.perplexity(model, dataset)             — primary quantitative metric
11. benchmark.eval_downstream(checkpoint, tasks)     — ARC, HellaSwag, etc.

12. verify.compare_to_baseline(model, cache_dir)     — see where activations diverge
13. benchmark.compare(models, prompts)               — side-by-side generation in ollama
14. benchmark.generation_metrics(results)            — automated failure detection

15. tracking: log surgery, metrics, samples          — runs throughout all steps
```

Or equivalently, via a recipe file:

```
recipe.run("experiments/my_experiment.yaml")   — does steps 4-15 automatically
```

## Prerequisites

- Python 3.10+
- `llama.cpp` cloned and built locally
- `ollama` installed and running
- `lm-evaluation-harness` installed (`pip install lm-eval`)
- HuggingFace access token for LLaMA 3 (gated model)

### Python Dependencies

- `torch`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `datasets` (for WikiText-2 / C4 loading)
- `lm-eval` (EleutherAI's evaluation harness)
- `requests` (for ollama API)
- `pyyaml` (for recipe files)

## Implementation Phases

### Phase 1: Core Surgery + Structural Verification
- `surgery.py`: model loading (all three modes), all layer operations, `SurgeryLog`
- `verify.py`: structural checks only (Level 1)
- `get_layer_info` with memory estimation
- **Test:** load model → remove layers → verify structure → save checkpoint → reload and confirm

### Phase 2: Export Pipeline
- `export.py`: all three steps + convenience wrapper
- **Test:** modified model → GGUF → ollama → generate text and confirm it runs

### Phase 3: Quantitative Evaluation
- `benchmark.py`: perplexity measurement (fp16, proper stride)
- `benchmark.py`: `lm-evaluation-harness` integration (downstream tasks)
- **Test:** measure perplexity on original → surgery → measure on modified → confirm delta makes sense

### Phase 4: Inspection + Activation Analysis
- `inspect.py`: Block Influence scores, weight norms, SVD, attention entropy, residual stream norms
- `verify.py`: activation comparison (Level 2) + cached baselines
- **Test:** compute BI scores → remove lowest-BI layer → confirm perplexity impact is minimal

### Phase 5: Generation Comparison + Metrics
- `benchmark.py`: ollama generation comparison + generation metrics (repetition, diversity, etc.)
- Prompt sets
- **Test:** compare original vs modified side-by-side, confirm metrics catch a deliberately broken model

### Phase 6: Calibration, Recipes, Tracking
- `surgery.py`: calibration (norm rescaling)
- `recipe.py`: YAML parsing, execution, batch runs, sweep generation
- `tracking.py`: full experiment logging with SQLite
- **Test:** run a complete experiment via recipe file, confirm it's logged and queryable

## Out of Scope (v1)

- Sub-layer surgery (removing just attention or just MLP from a layer)
- Attention head-level surgery (ablate/modify individual heads)
- Multiple architecture support (Qwen, Mistral, etc.)
- Training / fine-tuning integration
- TransformerLens / nnsight integration
- Direct GGUF-level tensor manipulation (bypass PyTorch)
- Advanced calibration (weight projection, knowledge distillation)
- LLM-as-judge evaluation
- Remote/cloud execution
