# probe.py — Hidden State Probing and Intervention

**Date:** 2026-04-11
**Module:** `testing/llm_surgeon/probe.py`
**Tests:** `testing/tests/test_probe.py`

## Purpose

Add the ability to extract hidden states at sub-layer granularity, project them into token space (logit lens), and intervene on the forward pass by modifying hidden states at specific points. Enables layer-by-layer analysis of what a model "believes" and how representations evolve.

## Phase 1: Observation (logit lens + hidden state extraction)

### Capture Points

Capture occurs on the **residual stream** — after each residual add, before the next normalization. In a Pre-LN LLaMA layer:

```
input h
  -> input_layernorm(h) -> self_attn -> + h  ->  h'      <- "attn" capture point
  -> post_attention_layernorm(h') -> mlp -> + h' ->  h''  <- "ffn" capture point
```

The logit lens projection at each capture point is: `residual stream state -> model.model.norm (final RMSNorm) -> model.lm_head -> logits`. This matches the transform the model applies at the final layer.

Future extension: mid-FFN capture (after gate/up projection, before down projection) can be added as a third sublayer type `"ffn_mid"` without API changes.

### Hook Implementation

- **Post-attention (`"attn"`)**: Forward hook on `layer.self_attn`. The hook computes the residual add (`h + attn_output`) to capture the post-attention residual stream state. This requires accessing the layer's input (via a pre-hook or input capture on the layer block) and adding the attention output.
- **Post-FFN (`"ffn"`)**: Forward hook on the full layer block (`model.model.layers[i]`). Captures layer output directly — same pattern as existing `block_influence` in `inspect.py`.

All hooks are registered before the forward pass and removed immediately after. No persistent model mutation.

### `logit_lens()`

```python
def logit_lens(
    model,
    tokenizer,
    prompt: str,
    top_k: int = 10,
    full_logits: bool = False,
    positions: list[int] | None = None,
    on_layer: Callable[[int, str, dict], None] | None = None,
) -> LogitLensResult:
```

**Parameters:**
- `model` — HuggingFace `AutoModelForCausalLM`
- `tokenizer` — corresponding tokenizer
- `prompt` — input text
- `top_k` — number of top predictions to return per capture point (default 10)
- `full_logits` — if `True`, also store full logit tensors per capture point
- `positions` — sequence positions to analyze (default: all; `-1` or specific indices)
- `on_layer` — optional callback, called per capture point during the forward pass

**Returns `LogitLensResult`:**

```python
@dataclass
class LogitLensResult:
    predictions: list[dict]
    # Each dict: {layer: int, sublayer: str, position: int,
    #             top_k: list[{token: str, token_id: int, prob: float, rank: int}]}
    logits: dict[tuple[int, str], Tensor] | None
    # Keyed by (layer, sublayer), each tensor is (seq_len, vocab_size). None if full_logits=False.
    prompt_tokens: list[str]
    # Tokenized prompt for position reference.

    def summary(self, position: int = -1) -> str:
        """Readable table: layers down, top-1 prediction at each capture point."""

    def first_correct_layer(self, position: int, target_token: str) -> int | None:
        """Layer index where target_token first appears as top-1 prediction."""

    def prediction_flips(self, position: int) -> int:
        """Number of times the top-1 prediction changes across layers."""
```

**Callback `on_layer` signature:** `(layer: int, sublayer: str, data: dict) -> None`
- `data` contains: `{"hidden_state": Tensor, "top_k": list[tuple[str, float]], "logits": Tensor | None}` (`"logits"` is `None` unless `full_logits=True`)
- Called during the forward pass as each capture point is processed.

### `extract_hidden_states()`

```python
def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layers: list[int] | None = None,
    sublayers: tuple[str, ...] = ("ffn",),
    detach: bool = True,
    on_layer: Callable[[int, str, dict], None] | None = None,
) -> HiddenStates:
```

**Parameters:**
- `layers` — which layers to capture (default: all)
- `sublayers` — `("attn",)`, `("ffn",)`, or `("attn", "ffn")` (default: FFN only)
- `detach` — detach tensors from computation graph (default: True)
- `on_layer` — optional callback

**Returns `HiddenStates`:**

```python
@dataclass
class HiddenStates:
    states: dict[tuple[int, str], Tensor]
    # Keyed by (layer, sublayer). Each tensor: (seq_len, d_model).
    prompt_tokens: list[str]

    def cosine_similarity(self, a: tuple[int, str], b: tuple[int, str], position: int = -1) -> float:
        """Cosine similarity between two capture points at a given position."""

    def save(self, path: str) -> None:
        """Save to .pt file."""

    @staticmethod
    def load(path: str) -> "HiddenStates":
        """Load from .pt file."""
```

**Callback `on_layer`:** `(layer: int, sublayer: str, data: dict) -> None`
- `data` contains: `{"hidden_state": Tensor}`

### `layer_predictions_table()`

```python
def layer_predictions_table(result: LogitLensResult, position: int = -1) -> str:
```

Formats a single position's predictions across all layers as a readable table. Columns: layer, sublayer, top-1 token, probability, top-3 tokens. Utility for interactive exploration.

## Phase 2: Intervention

### `intervene()`

```python
def intervene(
    model,
    tokenizer,
    prompt: str,
    interventions: list[Intervention],
    capture_logit_lens: bool = False,
    top_k: int = 10,
    on_layer: Callable[[int, str, dict], None] | None = None,
) -> InterventionResult:
```

**`Intervention` dataclass:**

```python
@dataclass
class Intervention:
    layer: int
    sublayer: str  # "attn" or "ffn"
    fn: Callable[[Tensor, int], Tensor]  # (hidden_state, layer_idx) -> modified_state
```

The `fn` callable receives the residual stream tensor at the specified point and returns a modified tensor of the same shape. The hook replaces the layer's output with the modified state.

**Returns `InterventionResult`:**

```python
@dataclass
class InterventionResult:
    output_logits: Tensor  # Final model output (seq_len, vocab_size)
    logit_lens_result: LogitLensResult | None  # If capture_logit_lens=True
    interventions_applied: list[dict]  # [{layer, sublayer, op_repr}, ...] for tracking
```

**Callback `on_layer`:** `(layer: int, sublayer: str, data: dict) -> None`
- `data` contains: `{"hidden_state": Tensor, "modified": bool, "top_k": list | None}`

### Predefined Operations (`probe.ops`)

Factory functions returning callables with descriptive `__repr__` for experiment logging:

| Function | Signature | Description |
|---|---|---|
| `ops.scale` | `(factor: float)` | Multiply hidden state by scalar |
| `ops.zero_dims` | `(dims: list[int])` | Zero specific dimensions |
| `ops.clamp` | `(min: float, max: float)` | Clamp all values |
| `ops.noise` | `(std: float)` | Add Gaussian noise (seeded for reproducibility) |
| `ops.replace` | `(tensor: Tensor)` | Substitute a cached hidden state |
| `ops.project_out` | `(direction: Tensor)` | Remove a direction from the hidden state |

Each returns a callable `(Tensor, int) -> Tensor` that also has a string representation (e.g., `"scale(0.5)"`, `"noise(std=0.1)"`).

## Tracking Integration

### SQLite (scalar summaries via existing `metrics` / `samples` tables)

- Top-1 accuracy per layer: does layer N's top prediction match final layer?
- Entropy per layer per position
- Layer where correct prediction first appears
- Prediction flip count across layers
- Intervention metadata: layer, sublayer, op repr

### Disk (tensors as `.pt` files)

```
outputs/<experiment>/probe/
  hidden_states/
    <prompt_hash>.pt          # extract_hidden_states().save()
  logit_lens/
    <prompt_hash>.pt          # full logit tensors
  interventions/
    <prompt_hash>_<op_repr>.pt  # intervention capture data
```

Prompt hash: `sha256(prompt)[:12]`, same convention as `verify.cache_baseline`.

### Recipe Integration

New `analyze` section in YAML recipes, runs after surgery and before export:

```yaml
analyze:
  logit_lens:
    prompt: "The capital of France is"
    top_k: 5
  hidden_states:
    prompt: "The capital of France is"
    save: true
```

`recipe.py` gains an `_run_analyze()` step that calls the `probe` functions and logs results to the experiment.

## Module Structure

`probe.py` imports:
- `torch`, `torch.nn.functional`
- `dataclasses`, `hashlib`
- No imports from other `llm_surgeon` modules

Other modules that change:
- `recipe.py` — new `analyze` phase calling `probe` functions
- `__init__.py` — export `probe` module
- No changes to `tracking.py`, `inspect.py`, `surgery.py`, or other existing modules

## Testing

Tests use the existing `tiny_llama` fixture (8 layers, 32 hidden dims).

**Phase 1 tests:**
- `logit_lens` returns correct structure, top-k per layer, probabilities sum to ~1
- `logit_lens` with `full_logits=True` returns tensors of shape `(seq_len, vocab_size)`
- `logit_lens` with `positions` parameter filters correctly
- `extract_hidden_states` returns tensors of correct shape `(seq_len, d_model)`
- `extract_hidden_states` with `sublayers=("attn", "ffn")` returns 2x capture points
- `HiddenStates.save()` / `.load()` roundtrip
- `HiddenStates.cosine_similarity()` returns 1.0 for same point
- `layer_predictions_table` returns non-empty string
- `on_layer` callback fires correct number of times
- `LogitLensResult.summary()` returns readable output
- `LogitLensResult.first_correct_layer()` and `.prediction_flips()` return sane values

**Phase 2 tests:**
- `intervene` with `ops.scale(1.0)` produces same output as unmodified forward pass
- `intervene` with `ops.scale(0.0)` at layer 0 produces different output
- `intervene` with `capture_logit_lens=True` returns valid `LogitLensResult`
- Each predefined op: correct tensor shape, correct transformation, correct `__repr__`
- `ops.replace` with tensor from `extract_hidden_states` produces valid output
- `ops.project_out` reduces component in specified direction
- `on_layer` callback `data["modified"]` is `True` at intervention points, `False` elsewhere
- Intervention metadata logged correctly for experiment tracking

## Future: llama.cpp Live Streaming

Not in scope for this build, but the design accommodates it. llama.cpp's `llama_new_context_params` supports a `cb_eval` callback that receives per-layer activations during inference. A future module could:

1. Build a custom llama.cpp callback (C++) that emits residual stream states over a Unix socket / shared memory
2. A Python receiver converts the raw buffers into tensors
3. Feed those tensors into the same `on_layer` callback interface that `probe.py` defines

The `on_layer` signature is runtime-agnostic: it receives `(layer, sublayer, data_dict)` regardless of whether the source is a PyTorch hook or a llama.cpp callback. The GUI visualization layer would consume this same interface.
