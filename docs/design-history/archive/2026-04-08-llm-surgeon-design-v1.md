# LLM Surgeon — Local Model Surgery Toolkit

## Goal

A Python toolkit for surgical layer-level manipulation of LLaMA 3 8B models, with quantitative verification (perplexity, activation analysis), export to GGUF for fast inference in ollama, and experiment tracking. Enables systematic experimentation with layer reordering, removal, duplication, and other structural changes to study their effects on model behavior.

## Target Model

LLaMA 3 8B (decoder-only, 32 layers, d=4096, h=32). Single architecture to start; generalization to other architectures is out of scope for v1.

## Hardware Constraints

- GPU: NVIDIA RTX 2080 (8GB VRAM)
- RAM: 32GB
- CPU: Intel i7-8700K (6 cores / 12 threads)
- Storage: NVMe SSDs

Implications: 4-bit quantization (bitsandbytes) for GPU inspection/evaluation, fp16 on CPU for export. Both fit within available resources.

## Project Structure

```
testing/
  llm_surgeon/
    __init__.py
    surgery.py        — model loading, layer manipulation, calibration
    verify.py         — structural checks, activation comparison, cached baselines
    inspect.py        — weight/activation inspection (norms, entropy, SVD)
    export.py         — HF checkpoint -> GGUF -> quantize -> ollama
    benchmark.py      — perplexity measurement + ollama generation comparison
    tracking.py       — experiment logging (local JSON/SQLite)
  experiments/        — experiment scripts that use the toolkit
    example_skip_layer.py
  prompts/            — test prompt sets for benchmarking
    default.json
  outputs/            — generated checkpoints, GGUF files, benchmark results
  experiments.db      — SQLite experiment log (auto-created)
  requirements.txt
  pyproject.toml
```

## Module: `surgery.py`

### Model Loading

Two modes depending on purpose:

- **`inspect` mode**: bitsandbytes 4-bit + `device_map="auto"`. Fits in 8GB VRAM. For inspection, evaluation, and activation analysis.
- **`export` mode**: fp16 on CPU (~16GB RAM). Full precision weights for clean surgery + GGUF export.

Both return a standard HuggingFace `LlamaForCausalLM` + tokenizer.

```python
model, tokenizer = surgeon.load_model("meta-llama/Meta-Llama-3-8B", mode="inspect")
model, tokenizer = surgeon.load_model("meta-llama/Meta-Llama-3-8B", mode="export")
```

### Layer Operations

All operations modify the model in-place and update `config.num_hidden_layers`. All return a `SurgeryLog` (list of changes made).

| Function | Description |
|----------|-------------|
| `remove_layers(model, [16, 17, 18])` | Delete specified layers |
| `keep_layers(model, [0, 1, ..., 15])` | Keep only listed layers, remove the rest |
| `reorder_layers(model, [0, 1, 2, 15, 14, 13, ...])` | Rearrange to specified order |
| `duplicate_layer(model, src=10, dst=11)` | Deep-copy a layer and insert at position |
| `swap_layers(model, 5, 12)` | Swap two layers' positions |
| `get_layer_info(model)` | Print summary: layer count, param counts per layer, total params |

**Index semantics:** Operations use layer indices as they exist at call time, not original indices. This is simpler and avoids ambiguity when chaining multiple operations.

**`SurgeryLog`:** A dataclass containing a list of `SurgeryOp` entries (each has `operation`, `description`, and `layer_count_before`/`layer_count_after`). Passed to `verify.check_structure()` for cross-referencing. Also recorded by experiment tracking.

### Calibration

When layers are reordered or removed, the input distribution to downstream layers shifts — the residual stream statistics no longer match what those layers were trained to expect. Without correction, most non-trivial rearrangements produce degraded or garbage output.

```python
surgery.calibrate(model, tokenizer, dataset="wikitext2", num_samples=128)
```

- Runs a forward pass over calibration samples
- Collects residual stream statistics (mean, variance) at each layer boundary
- Rescales layer norm parameters to compensate for distribution shift
- This is a lightweight correction, not fine-tuning — it adjusts normalization, not weights
- Optional: call it after surgery, before export. Skip it if you want to measure the raw effect of the surgery without correction.

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

Analysis tools for understanding individual layers before and after surgery. Helps decide *which* layers to cut or rearrange, rather than doing it blind.

```python
# Per-layer weight statistics
inspect.weight_norms(model)
# → table: layer index, attn weight norm, MLP weight norm, total norm

inspect.weight_svd(model, layers=[10, 11, 12])
# → singular value spectrum of weight matrices for specified layers
# → helps identify redundant layers (similar spectra) or critical layers (distinct spectra)

# Activation-based analysis (requires a forward pass)
inspect.attention_entropy(model, tokenizer, prompt="The capital of France is")
# → per-layer, per-head entropy of attention distributions
# → low entropy = focused attention, high entropy = diffuse/possibly redundant

inspect.residual_stream_norms(model, tokenizer, prompt="The capital of France is")
# → magnitude of residual stream at each layer boundary
# → sudden jumps or drops indicate layers with outsized impact
```

- All functions work on 4-bit (inspect mode) models
- Output is both printed tables and returned as DataFrames/dicts for programmatic use
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

The primary quantitative metric for evaluating surgery impact. Runs directly through the PyTorch model (no ollama needed).

```python
ppl = benchmark.perplexity(model, tokenizer, dataset="wikitext2", stride=512)
# → returns a single float: perplexity on the dataset
```

- Supports standard datasets: WikiText-2, C4 (small subset)
- Uses sliding window with configurable stride for long documents
- Works on 4-bit (inspect mode) models
- This is the number you compare across experiments: "baseline perplexity is 6.2, removing layers 16-18 gives 8.7, removing layers 28-30 gives 6.4"

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
- Temperature 0.0 for near-reproducibility (note: quantized models in llama.cpp have numerical non-determinism from parallel reduction, so outputs may vary slightly across runs even at temp 0 — don't chase phantom differences)

### Prompt Sets

JSON files with categorized prompts:

```json
[
    {"prompt": "The capital of France is", "category": "factual"},
    {"prompt": "Explain quantum entanglement simply", "category": "reasoning"},
    {"prompt": "Write a haiku about rain", "category": "creative"}
]
```

### Output

- Side-by-side comparison printed to terminal (prompt, each model's response, speed)
- Perplexity scores displayed alongside generation results when available
- Full results saved as JSON for later analysis

## Module: `tracking.py`

Local experiment tracking — structured logging so you don't lose track across dozens of experiments.

```python
# Start an experiment
exp = tracking.start(
    name="remove-layers-16-18",
    description="Remove middle layers 16-18, no calibration",
    base_model="meta-llama/Meta-Llama-3-8B"
)

# Record surgery
exp.log_surgery(surgery_log)

# Record metrics
exp.log_metric("perplexity_wikitext2", 8.7)
exp.log_metric("perplexity_c4", 12.3)

# Record generation samples
exp.log_samples(benchmark_results)

# Finish
exp.finish(notes="Coherent output but factual accuracy degraded")
```

### Storage

- SQLite database (`experiments.db`) in the project root
- Schema: experiments table (id, name, description, base_model, timestamp, status, notes), metrics table (experiment_id, key, value), surgery_ops table (experiment_id, operation details)
- Query with standard SQL or convenience functions:

```python
tracking.list_experiments()
tracking.get_experiment("remove-layers-16-18")
tracking.compare_experiments(["remove-layers-16-18", "remove-layers-28-30"])
# → side-by-side table of metrics
```

- No external dependencies (sqlite3 is in Python stdlib)
- Not MLflow or W&B — just structured local logging

## Typical End-to-End Workflow

```
 1. inspect.weight_norms(model)                   — understand layer importance
 2. inspect.attention_entropy(model, prompt)       — identify redundant heads/layers

 3. Load model in EXPORT mode (fp16, CPU)          — full precision for surgery + save
 4. Perform surgery (remove_layers, reorder, etc.)
 5. surgery.calibrate(model, dataset)              — optional: correct distribution shift
 6. verify.check_structure(model, surgery_log)     — fast structural sanity check
 7. export.full_pipeline(model, name, quant)       — save checkpoint → GGUF → ollama

 8. Load MODIFIED checkpoint in INSPECT mode       — 4-bit on GPU
 9. benchmark.perplexity(model, dataset)           — quantitative comparison
10. verify.compare_to_baseline(model, cache_dir)   — see where activations diverge

11. benchmark.compare(models, prompts)             — side-by-side generation in ollama

12. tracking: log surgery, metrics, samples        — runs throughout all steps
```

Steps 1-2 are pre-surgery analysis (decide what to cut). Steps 3-7 are surgery+export. Steps 8-10 are quantitative evaluation. Step 11 is qualitative evaluation. Tracking wraps the whole thing.

## Prerequisites

- Python 3.10+
- `llama.cpp` cloned and built locally
- `ollama` installed and running
- HuggingFace access token for LLaMA 3 (gated model)

### Python Dependencies

- `torch`
- `transformers`
- `accelerate`
- `bitsandbytes`
- `datasets` (for WikiText-2 / C4 loading)
- `requests` (for ollama API)

## Implementation Phases

Given the expanded scope, implementation should proceed in testable phases:

### Phase 1: Core Surgery + Structural Verification
- `surgery.py`: model loading (both modes), all layer operations, `SurgeryLog`
- `verify.py`: structural checks only (Level 1)
- Basic `get_layer_info`
- Test: load model → remove layers → verify structure → save checkpoint → reload and confirm

### Phase 2: Export Pipeline
- `export.py`: all three steps + convenience wrapper
- Test: modified model → GGUF → ollama → generate text and confirm it runs

### Phase 3: Perplexity + Generation Benchmarking
- `benchmark.py`: perplexity measurement + ollama comparison
- Prompt sets
- Test: measure perplexity on original → surgery → measure perplexity on modified → compare

### Phase 4: Inspection + Activation Analysis
- `inspect.py`: weight norms, SVD, attention entropy, residual stream norms
- `verify.py`: activation comparison (Level 2) + cached baselines
- Test: inspect original model → perform surgery → compare activations against cached baseline

### Phase 5: Calibration + Experiment Tracking
- `surgery.py`: calibration function
- `tracking.py`: full experiment logging
- Test: run a complete experiment with tracking, compare calibrated vs uncalibrated results

## Out of Scope (v1)

- Attention head-level surgery (ablate/modify individual heads within a layer)
- Multiple architecture support (Qwen, Mistral, etc.)
- Training / fine-tuning integration
- TransformerLens / nnsight integration
- Direct GGUF-level tensor manipulation (bypass PyTorch)
- Automated quality scoring beyond perplexity
- Remote/cloud execution
