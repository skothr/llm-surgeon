# LLM Surgeon — Phase Plan

Reference spec: `2026-04-08-llm-surgeon-design.md` (v2)

Each phase is independently testable. After each phase, the user runs verification examples to confirm before moving to the next. Phases should be implemented sequentially — later phases depend on earlier ones.

---

## Phase 1: Core Surgery + Structural Verification

### Scope
- `llm_surgeon/__init__.py` — package init, convenience imports
- `llm_surgeon/surgery.py`:
  - `load_model(model_id, mode)` — three modes: `inspect` (4-bit GPU), `eval` (fp16 GPU+CPU), `export` (fp16 CPU-only)
  - `remove_layers(model, layer_indices)` → SurgeryLog
  - `keep_layers(model, layer_indices)` → SurgeryLog
  - `reorder_layers(model, new_order)` → SurgeryLog
  - `duplicate_layer(model, src, dst)` → SurgeryLog (with memory warning)
  - `swap_layers(model, i, j)` → SurgeryLog
  - `get_layer_info(model)` — layer count, param counts, estimated memory
  - `SurgeryLog` and `SurgeryOp` dataclasses
- `llm_surgeon/verify.py`:
  - `check_structure(model, surgery_log)` → VerifyReport (Level 1 only, no inference)
- `pyproject.toml` + `requirements.txt`

### NOT in this phase
No export, no GGUF, no benchmarking, no inspection, no calibration, no tracking.

### Testing Process
1. `pip install -e .` — confirm package installs
2. Load model in `inspect` mode — confirm fits in VRAM, returns (model, tokenizer)
3. Load model in `export` mode — confirm loads on CPU, check RAM usage (~16GB)
4. `get_layer_info(model)` — confirm prints 32 layers with param counts
5. `remove_layers(model, [16, 17, 18])` — confirm 29 layers, correct surgery log
6. `check_structure(model, log)` — confirm passes
7. `model.save_pretrained("outputs/test")` — confirm checkpoint saves
8. Reload saved checkpoint — confirm loads with 29 layers
9. Test each operation on fresh model loads
10. Test error cases: invalid indices, empty list, duplicate layer memory warning

### Completion Criteria
- All six operations work and return correct SurgeryLog
- check_structure passes after every valid operation
- check_structure fails/warns on deliberately broken state
- Saved checkpoints reload with correct layer count

### User Verification
```python
from llm_surgeon import surgery, verify

# Load (downloads LLaMA 3 8B on first run — ~16GB)
model, tokenizer = surgery.load_model("meta-llama/Meta-Llama-3-8B", mode="inspect")
surgery.get_layer_info(model)

# Remove some layers
log = surgery.remove_layers(model, [16, 17, 18])
print(log)

# Verify
report = verify.check_structure(model, log)
print(report)

# Save and reload test
model_export, tok = surgery.load_model("meta-llama/Meta-Llama-3-8B", mode="export")
log = surgery.remove_layers(model_export, [16, 17, 18])
model_export.save_pretrained("outputs/test-phase1")
tok.save_pretrained("outputs/test-phase1")
```

---

## Phase 2: Export Pipeline

### Scope
- `llm_surgeon/export.py`:
  - `save_checkpoint(model, output_dir)` — save HF checkpoint with sequential layer numbering
  - `to_gguf(checkpoint_path, output_dir, quantization)` — convert via convert_hf_to_gguf.py + llama-quantize
  - `register_ollama(gguf_path, name)` — create Modelfile + ollama create
  - `full_pipeline(model, tokenizer, name, quantization, output_dir)` — all three steps

### Dependencies
- Phase 1 (needs modified model to export)
- llama.cpp built locally
- ollama installed and running

### Prerequisites to Verify Before Starting
- [ ] llama.cpp repo is cloned and built (`make` or `cmake`)
- [ ] `LLAMA_CPP_PATH` env var set or config mechanism decided
- [ ] ollama is running (`ollama list` works)
- [ ] HuggingFace token configured for LLaMA 3 access

### Testing Process
1. `save_checkpoint(model, "outputs/test-export")` — confirm HF checkpoint with correct config.json
2. `to_gguf("outputs/test-export", quantization="Q4_K_M")` — confirm GGUF file created
3. `register_ollama(gguf_path, name="test-surgeon")` — confirm `ollama list` shows model
4. `ollama run test-surgeon "Hello"` — confirm generates text (quality doesn't matter yet)
5. `full_pipeline(model, name="test-pipeline", quantization="Q4_K_M")` — confirm all steps in one call
6. Test error cases: missing llama.cpp, ollama not running, invalid quantization type

### Completion Criteria
- Modified model exports to GGUF and runs in ollama
- Full pipeline works end-to-end with a single call
- Clear error messages when prerequisites missing

### User Verification
```python
from llm_surgeon import surgery, export

model, tok = surgery.load_model("meta-llama/Meta-Llama-3-8B", mode="export")
log = surgery.remove_layers(model, [28, 29, 30, 31])  # remove last 4

export.full_pipeline(model, tokenizer=tok, name="llama3-28layer", quantization="Q4_K_M", output_dir="outputs/")
```
Then in terminal:
```bash
ollama run llama3-28layer "What is the capital of France?"
```

---

## Phase 3: Quantitative Evaluation

### Scope
- `llm_surgeon/benchmark.py`:
  - `perplexity(model, tokenizer, dataset, stride)` — WikiText-2 / C4, sliding window, fp16 required
  - `eval_downstream(model_path, tasks, num_fewshot)` — shells out to lm-evaluation-harness

### Dependencies
- Phase 1 (needs models)
- Phase 2 (for eval_downstream on local checkpoints)
- `datasets` package (for WikiText-2 loading)
- `lm-eval` package (for downstream tasks)

### Testing Process
1. Baseline perplexity on unmodified LLaMA 3 8B in eval mode — confirm ~6-7 on WikiText-2
2. Perplexity on modified model (remove 3 layers) — confirm higher than baseline
3. Perplexity on trivially modified model (remove 1 low-impact layer) — confirm only slightly higher
4. `eval_downstream` on original model with ARC-Challenge — confirm ~50-55% (published range)
5. `eval_downstream` on modified model — confirm lower scores
6. Attempt perplexity in inspect mode (4-bit) — confirm warning/error about quantization noise
7. Verify stride defaults to context_length/2

### Completion Criteria
- Baseline perplexity matches published values for LLaMA 3 8B
- Perplexity correctly degrades proportional to surgery severity
- eval_downstream shells out to lm_eval and parses results correctly
- Clear warning when trying to measure perplexity on 4-bit model

### User Verification
```python
from llm_surgeon import surgery, benchmark

# Baseline
model, tok = surgery.load_model("meta-llama/Meta-Llama-3-8B", mode="eval")
baseline_ppl = benchmark.perplexity(model, tok, dataset="wikitext2")
print(f"Baseline perplexity: {baseline_ppl}")  # expect ~6-7

# Modified
model2, tok2 = surgery.load_model("meta-llama/Meta-Llama-3-8B", mode="eval")
surgery.remove_layers(model2, [16, 17, 18])
modified_ppl = benchmark.perplexity(model2, tok2, dataset="wikitext2")
print(f"Modified perplexity: {modified_ppl}")  # expect notably higher

# Downstream (slow — ARC only for quick test)
results = benchmark.eval_downstream("meta-llama/Meta-Llama-3-8B", tasks=["arc_challenge"])
print(results)
```

---

## Phase 4: Inspection + Activation Analysis

### Scope
- `llm_surgeon/inspect.py`:
  - `block_influence(model, tokenizer, prompts)` — BI scores per layer (ShortGPT metric)
  - `weight_norms(model)` — per-layer Frobenius norms for attn + MLP
  - `weight_svd(model, layers)` — singular value spectra
  - `attention_entropy(model, tokenizer, prompt)` — per-layer, per-head entropy
  - `residual_stream_norms(model, tokenizer, prompt)` — residual magnitude at each boundary
- `llm_surgeon/verify.py` additions:
  - `compare_activations(original, modified, prompt, layers)` — Level 2
  - `cache_baseline(model, tokenizer, prompts, cache_dir)` — save activations to disk
  - `compare_to_baseline(model, tokenizer, prompts, cache_dir)` — compare against cached

### Dependencies
- Phase 1 (needs models)
- Phase 3 (for validation test: BI scores should predict perplexity impact)
- Independent of Phase 2 (no export needed)

### Testing Process
1. `block_influence` — confirm 32 scores, all in [0, 1]
2. Verify BI intuition: early/late layers generally higher BI, middle layers often lower
3. `weight_norms` — confirm per-layer values, no NaN/inf
4. `weight_svd(model, layers=[0, 15, 31])` — confirm correct shape arrays
5. `attention_entropy` — confirm per-layer, per-head values
6. `residual_stream_norms` — confirm 33 values (input + 32 layer outputs)
7. `cache_baseline` — confirm .pt files written to disk
8. `compare_to_baseline` after surgery — confirm divergence table matches expected pattern
9. **Validation test:** remove lowest-BI layer → measure perplexity (Phase 3). Remove highest-BI layer → measure perplexity. Confirm BI scores are predictive (low BI removal causes less damage).

### Completion Criteria
- BI scores are predictive: low-BI removal < high-BI removal in perplexity damage
- Cached baselines load correctly and match live comparisons
- All inspect functions produce consistent, non-garbage output

### User Verification
```python
from llm_surgeon import surgery, inspect

model, tok = surgery.load_model("meta-llama/Meta-Llama-3-8B", mode="inspect")

# Which layers can I safely remove?
bi = inspect.block_influence(model, tok, prompts=["The capital of France is", "Explain gravity simply"])
for layer, score in sorted(bi.items(), key=lambda x: x[1]):
    print(f"Layer {layer:2d}: BI={score:.4f}")

# Weight structure
inspect.weight_norms(model)

# Attention patterns
inspect.attention_entropy(model, tok, prompt="The quick brown fox jumps over the lazy dog")
```

---

## Phase 5: Generation Comparison + Metrics

### Scope
- `llm_surgeon/benchmark.py` additions:
  - `compare(models, prompts, temperature)` — side-by-side generation via ollama API
  - `generation_metrics(results)` — repetition rate, vocab diversity, mean length, coherence
- `prompts/default.json` — default prompt set

### Dependencies
- Phase 2 (needs models registered in ollama)

### Testing Process
1. `compare(["llama3:8b", "test-surgeon"], prompts)` — confirm side-by-side output
2. `generation_metrics(results)` — confirm all four metrics computed
3. Test with deliberately broken model (remove 20 layers) — confirm metrics flag it
4. Test with mildly modified model — confirm metrics show mostly healthy
5. Confirm results saved as JSON
6. Test prompt set loading from JSON file

### Completion Criteria
- Side-by-side comparison displays cleanly with timing info
- Generation metrics correctly distinguish healthy from broken models
- Results saved as JSON for later analysis

### User Verification
```python
from llm_surgeon import benchmark

# Requires models in ollama from Phase 2
results = benchmark.compare(
    models=["llama3:8b", "llama3-28layer"],
    prompts="prompts/default.json"
)
metrics = benchmark.generation_metrics(results)
print(metrics)
```

---

## Phase 6: Calibration, Recipes, Tracking

### Scope
- `llm_surgeon/surgery.py` addition:
  - `calibrate(model, tokenizer, dataset, num_samples)` — LayerNorm rescaling
- `llm_surgeon/recipe.py`:
  - `run(recipe_path)` — execute a single YAML recipe (full pipeline)
  - `run_batch(glob_pattern)` — execute multiple recipes sequentially
  - `generate_layer_sweep(layers, template, output_dir)` — generate sweep YAML files
- `llm_surgeon/tracking.py`:
  - `start(name, description, base_model, recipe)` → Experiment
  - `Experiment.log_surgery(surgery_log)`
  - `Experiment.log_metric(key, value)`
  - `Experiment.log_samples(results)`
  - `Experiment.finish(notes)`
  - `list_experiments()`, `get_experiment(name)`, `compare_experiments(names)`
- `experiments.db` SQLite schema (auto-created on first use)

### Dependencies
- All prior phases (recipes orchestrate the full pipeline)

### Testing Process
1. `calibrate(model, tok, "wikitext2", 128)` — confirm runs without error
2. Compare perplexity pre/post calibration on a modified model — confirm improvement
3. Write recipe YAML → `recipe.run(path)` — confirm full pipeline executes
4. `generate_layer_sweep(range(32), template, dir)` — confirm 32 YAML files
5. `run_batch("experiments/sweep_*.yaml")` — test with 2-3 recipes
6. `tracking.list_experiments()` — confirm experiments appear
7. `tracking.compare_experiments([...])` — confirm side-by-side metrics
8. Test recipe with invalid operations — confirm clear error
9. Kill and restart Python — confirm experiments.db survives

### Completion Criteria
- Calibration measurably helps perplexity on rearranged models
- Recipes execute full pipeline from single YAML file
- Tracking records everything and supports queries
- Batch runs work with sweep-generated recipes
- DB persists across sessions

### User Verification
```yaml
# experiments/test_recipe.yaml
name: test-remove-layer-20
base_model: meta-llama/Meta-Llama-3-8B
description: Remove layer 20, calibrate, evaluate

surgery:
  - remove_layers: [20]
  - calibrate:
      dataset: wikitext2
      num_samples: 128

evaluate:
  perplexity:
    dataset: wikitext2

export:
  quantization: Q4_K_M
  ollama_name: llama3-no-layer-20
```
```python
from llm_surgeon import recipe, tracking

recipe.run("experiments/test_recipe.yaml")

# Check what we've done
tracking.list_experiments()
```

---

## Phase Summary

| Phase | What | Key Deliverable | Depends On |
|-------|------|----------------|------------|
| 1 | Core Surgery + Verify | Load → cut → verify → save | — |
| 2 | Export Pipeline | Modified model → GGUF → ollama | Phase 1 |
| 3 | Quantitative Eval | Perplexity + downstream benchmarks | Phase 1, 2 |
| 4 | Inspection + Analysis | BI scores, activation comparison | Phase 1, 3 |
| 5 | Generation Comparison | Side-by-side output + failure metrics | Phase 2 |
| 6 | Calibration + Recipes + Tracking | Full automated pipeline | All |

**Note:** Phases 4 and 5 are somewhat independent of each other and could be done in either order. All other phases are strictly sequential.
