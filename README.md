# llm-surgeon

Surgical, layer-level manipulation and interpretability probing of local
LLaMA-family models. `llm-surgeon` loads a model (HuggingFace checkpoint or
GGUF), lets you edit it at the level of individual layers, heads, and neurons,
and inspect what the residual stream is doing as it computes — all on a single
local GPU, with TinyLlama-scale models that iterate in seconds.

This is the shared root library: a standalone, reusable Python package with no
GUI or experiment-runner dependencies. Higher-level tools (the inspection GUI,
research arcs) build on top of it.

## What it does

The package is `llm_surgeon/`. Its modules:

- **`surgery`** — model loading (`load_model`) and layer-level surgery
  operations. Loads HF checkpoints (safetensors-first, with a documented
  fallback for repos/caches that never shipped safetensors) and applies
  structural edits.
- **`probe/`** — hidden-state probing and intervention subpackage:
  - `logit_lens`, `compare_logit_lens`, `extract_hidden_states`,
    `layer_predictions_table` — project the residual stream at each layer
    through the unembedding to read off intermediate predictions.
  - `intervene`, `activation_patch`, `ops` — capture-and-replace edits to
    activations during a forward pass.
  - `attribution_patch`, `attribution_patch_per_head`,
    `attribution_patch_per_neuron`, `edge_attribution_patch`,
    `extract_circuit` — gradient-based attribution and circuit extraction.
  - `nla_verbalize`, `nla_reconstruct`, `nla_score`, `load_av`/`load_ar` —
    natural-language-of-activations verbalization helpers.
- **`inspect`** — inspection and activation-analysis tools.
- **`benchmark`** — quantitative evaluation: perplexity and downstream-task
  benchmarks.
- **`recipe`** — YAML recipe parsing and execution, so a surgery + eval
  pipeline can be declared as data (see `experiments/test_full_recipe.yaml`).
- **`tracking`** — SQLite-backed experiment tracking.
- **`verify`** — structural verification of modified models.
- **`export`** — export pipeline: HF checkpoint → GGUF → Ollama registration
  (shells out to a local `llama.cpp` install via `LLAMA_CPP_PATH`).
- **`gguf_reader` / `gguf_writer`** — parse/dequantize GGUF into a standard
  `LlamaForCausalLM`, and the inverse export back to GGUF. The read and write
  tensor-name maps and Q/K head permutation are shared to keep the two paths
  from drifting.
- **`llama_engine`** — native GGUF inference (`LlamaEngine`) via
  `llama-cpp-python`, for fast generation / logits / perplexity on quantized
  models. The `llama_cpp` import is lazy: the package imports and the
  HF-side functionality works without it installed.

## Install

Editable install from a clone:

```bash
pip install -e .
```

With the optional dependency groups:

```bash
pip install -e ".[dev,gui,eval]"
```

Optional groups:

- `dev` — `pytest` (the test runner).
- `gui` — `fastapi` + `uvicorn`, for a service wrapping the library.
- `eval` — `lm_eval` + `datasets`, for the harness benchmarks in `benchmark`.
- `gguf` — `llama-cpp-python`, for the native `LlamaEngine` GGUF inference
  path. Not needed for HF loading, surgery, or probing.

Requires Python ≥ 3.10. Core dependencies (torch, transformers, accelerate,
bitsandbytes, numpy, huggingface_hub, ...) are declared in `pyproject.toml`;
a CUDA-capable GPU is assumed for non-trivial models.

## Quickstart

```python
from llm_surgeon import surgery, probe

model, tokenizer = surgery.load_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="inspect"
)
result = probe.logit_lens(model, tokenizer, "The capital of France is", top_k=5)
print(result.summary())
```

Runnable examples live in `examples/`: `quickstart.py` (load + logit lens),
`probe_demo.py` (probing walkthrough), and `qwen_load_check.py` (loading a
non-LLaMA-named checkpoint).

## Dev models

Iterate against small local models cached in the HuggingFace
`models--{org}--{name}` layout:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — 22 layers, 2048 hidden, 1.1B params.
  **Default** for examples, fixtures, and fast test iteration.
- `openlm-research/open_llama_3b_v2` — 26 layers, 3200 hidden, 3B params.
  For a slightly-larger-scale sanity check.

## Tests

```bash
pytest
```

`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["."]`, so a
bare `pytest` from the repo root runs the suite (22 test files covering
surgery, export, inspect, recipe, tracking, verify, the GGUF reader, the
llama engine, and the full probe surface). The suite assumes torch and the
dev models are available; install with `pip install -e ".[dev]"` first.

## Design history

`docs/design-history/` holds the phased design record: `specs/` (per-phase
design documents) and `plans/` (the running implementation plans) that built
out the library — Phase 1 core surgery through the attribution/circuit and
logit-lens phases. It is a historical record, not current API documentation;
the code and its module docstrings are authoritative for the present surface.

