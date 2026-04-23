# Phase 3.9 — Per-Neuron FFN Attribution Patching

**Date:** 2026-04-23
**Status:** Spec (awaiting review)
**Depends on:** Phase 3.5 (gradient AP), Phase 3.6 (per-head AP), Phase 3.7 (edge AP helper plumbing)
**References:**
- Nanda 2023 — *Attribution Patching* primer
- Geva et al. 2020 — *Transformer Feed-Forward Layers Are Key-Value Memories*
- Dar et al. 2022 — *Analyzing Transformers in Embedding Space*

## 1. Motivation

Phase 3.6 decomposed attention's contribution to a metric per head via the chain rule through `W_O`. FFN is the other half of every transformer layer, and per-FFN-block attribution (Phase 3.5) treats the MLP as a single opaque writer. Published interpretability work routinely names individual *neurons* (rows of the intermediate activation) as "responsible for" capitalization, grammar, code, etc. (Gurnee & Tegmark 2023, Bills et al. 2023). This phase ships the decomposition that makes those claims testable in this toolkit.

Same O(forward + backward) compute as Phase 3.6. Just swap `W_O` for `W_down` and `concat_z` for `act`.

## 2. Goals

- **G1.** `probe.attribution_patch_per_neuron(...)` returns a `PatchingResult` with per-(layer, neuron, position) scores. Top-k selection on the backend (intermediate_size is ~5632 for TinyLlama — exhaustive streaming is ~620k cells, too many).
- **G2.** WS route handles `mode="approx_neuron"` on the existing `/activation-patching` endpoint, accepting `top_k_neurons`.
- **G3.** `PerNeuronPatchingPanel.tsx` renders a ranked-list view (not a heatmap — 5632 cols is unrenderable). Position selector + optional layer filter + "copy TSV" export.
- **G4.** Shared `_capture_residual_stream_with_grad` gains a `capture_ffn_act` flag (new pre-hook on `mlp.down_proj` capturing its input) and `retain_grad` on ffn_out tensors so `ffn_out.grad` is available after `metric.backward()`.

## 3. Non-Goals

- **N1.** Per-reader edge-style framing (neuron-as-writer with multiple reader endpoints like Phase 3.7). Phase 3.9 uses the Phase 3.6 one-reader-per-layer convention: `grad_r = ffn_out.grad`, which is the gradient flowing into the MLP output via the residual stream. That one number per (L, pos) is enough for "how much does neuron n contribute to the metric."
- **N2.** Cross-prompt neuron ranking. Single clean/corrupted pair per run.
- **N3.** Neuron-level circuit extraction. If demand materializes, that's Phase 3.9.1 — treat top-k per-neuron cells as writers and run the Phase 3.8 BFS.
- **N4.** GQA/MQA-specific handling. FFN decomposition is independent of attention grouping, so this phase is orthogonal to the GQA tangent.
- **N5.** Neuron *activation* visualization (dataset-level analysis like "what prompts activate neuron X"). This phase is causal attribution only.

## 4. Math

### 4.1 LLaMA MLP structure

```
act = silu(gate_proj(h_ln)) * up_proj(h_ln)   # [B, T, intermediate]
ffn_out = down_proj(act)                       # [B, T, hidden]
```

where `down_proj.weight.shape == [hidden, intermediate]` and (PyTorch Linear convention) `ffn_out = act @ W_down.T`.

### 4.2 Per-neuron gradient decomposition

For a single reader (the residual stream at position `pos`) with `grad_ffn_out = ffn_out.grad[0, pos]  # [hidden]`:

```
∂m/∂ffn_out[h] = grad_ffn_out[h]
∂ffn_out[h] / ∂act[i] = W_down[h, i]
∂m/∂act[i] = Σ_h grad_ffn_out[h] · W_down[h, i]
           = (grad_ffn_out @ W_down)[i]                 # NO transpose
```

Per-neuron AP raw at (L, neuron_i, pos):

```
grad_act = grad_ffn_out @ W_down                         # [intermediate]
delta_act = from_act[0, pos] - base_act[0, pos]          # [intermediate]
ap_neuron_raw = delta_act * grad_act                     # [intermediate]
```

Normalized:

```
ap_recovery_neuron = ap_neuron_raw / D              if direction == "denoise"
                     1.0 + ap_neuron_raw / D         if direction == "noise"
```

where `D = Δ_clean − Δ_corrupted` (same as Phase 3.5/3.6/3.7).

### 4.3 Sum invariant

`Σ_i ap_neuron_raw(L, i, pos) == (delta_ffn_out[0, pos] · grad_ffn_out[0, pos])`

which is exactly Phase 3.5's `ap_ffn_raw` at `(L, pos)`. This is a clean target for mock tests.

### 4.4 Per-head vs per-neuron parity

| | Phase 3.6 (per-head) | Phase 3.9 (per-neuron) |
|---|---|---|
| Decomposed quantity | `concat_z @ W_O.T` | `act @ W_down.T` |
| Projection matrix | `W_O: [hidden, hidden]` | `W_down: [hidden, intermediate]` |
| Grad transform | `grad_r @ W_O` | `grad_r @ W_down` |
| Reshape step | `.view(n_heads, head_dim)` | none (neurons are atomic) |
| Unit count | 32 (TinyLlama) | 5632 (TinyLlama) |
| Total cells per run | 22 × 32 × seq_len ≈ 3.5k | 22 × 5632 × seq_len ≈ 620k |
| UI | Flat heatmap | Ranked list + filter |

## 5. API

### 5.1 New public function

```python
def attribution_patch_per_neuron(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    top_k_neurons: int = 200,
    on_cell: Optional[Callable[[Dict], None]] = None,
) -> PatchingResult:
    """Per-neuron FFN attribution patching (Phase 3.9).

    Decomposes Δffn_out's contribution to the metric into per-
    (layer, neuron, position) AP scores via chain rule through
    W_down. One forward + one backward pass (shared with Phase 3.6
    per-head if both were requested in the same pipeline, though
    the WS route treats them as distinct modes).

    Returns PatchingResult with mode='approx_neuron',
    n_neurons=intermediate_size, and `cells` containing only the
    top-k tuples by |ap_recovery|.

    If `top_k_neurons > num_layers * intermediate_size * n_positions`,
    silently caps at the actual count.
    """
```

Each emitted cell:
```python
{
    "layer": int,
    "unit": str,           # "neuron.n<i>"  (matches Phase 3.6 "attn.h<N>" convention)
    "neuron": int,         # the raw intermediate-size index i
    "position": int,
    "ap_recovery": float,
}
```

The `neuron` field is redundant-with-parseable-unit but included for cheap client-side sorting/filtering (avoid parsing `"neuron.n1234"` on every click).

### 5.2 `PatchingResult` extension

```python
@dataclass
class PatchingResult:
    # ...existing fields...
    mode: str = "exact"  # now: "exact" | "approx" | "approx_head" | "edge" | "circuit" | "approx_neuron"
    # ...existing Optional fields...
    n_neurons: Optional[int] = None   # NEW: set by attribution_patch_per_neuron (intermediate_size)
```

All new fields are `Optional` / default `None` — zero-impact on existing call sites.

### 5.3 Capture helper extension

`_capture_residual_stream_with_grad` gains `capture_ffn_act: bool = False`. When true:
- Adds a forward-pre-hook on each `layer.mlp.down_proj` that stores `args[0]` in a new `ffn_acts: Dict[int, torch.Tensor]` dict keyed by layer index.
- Extends the return tuple from **6-tuple** (Phase 3.7) to **7-tuple**:
  ```
  (captured, h_ins, logits, tokens, concat_z, reader_inputs, ffn_acts)
  ```
- When `capture_ffn_act=False`, `ffn_acts` is an empty dict.

All existing callers updated to unpack the 7-tuple (adding `, _` for the new slot). Phase 3.5/3.6/3.7/3.8 tests must still pass unchanged.

Additionally: when `capture_ffn_out=True`, the ffn_out tensors stored by the post-hook must have `retain_grad()` called so `.grad` is populated after `metric.backward()`. This is NEW — Phases 3.7/3.8 use ffn_out only as a writer delta (`delta_ffn_out`, no grad read); Phase 3.9 uses `ffn_out.grad` as the per-layer reader gradient. The hook should check `output.requires_grad` and call `retain_grad()` on it before returning; when running under `torch.no_grad()` (the `from` pass) this is a no-op.

### 5.4 Implementation sketch

```python
def attribution_patch_per_neuron(model, tokenizer, clean, corrupted, ...):
    # Validation (same pattern as Phase 3.6 per-head)
    if top_k_neurons < 1:
        raise ValueError("top_k_neurons must be >= 1")
    # ...prompt validation...

    # Direction handling (same as Phase 3.5/3.6)
    from_prompt = clean if direction == "denoise" else corrupted
    base_prompt = corrupted if direction == "denoise" else clean

    with torch.no_grad():
        from_captured_raw, _, from_logits, from_tokens, _, _, from_ffn_acts_raw = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=("attn", "ffn"), layers=layers,
                capture_concat_z=False,
                capture_reader_grads=False,
                capture_ffn_out=True,
                capture_ffn_act=True,
            )
        from_ffn_out = {k[0]: v.detach().clone() for k, v in from_captured_raw.items() if k[1] == "ffn_out"}
        from_ffn_acts = {k: v.detach().clone() for k, v in from_ffn_acts_raw.items()}

    with torch.enable_grad():
        base_captured, _, base_logits, base_tokens, _, _, base_ffn_acts = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=("attn", "ffn"), layers=layers,
                capture_concat_z=False,
                capture_reader_grads=False,
                capture_ffn_out=True,
                capture_ffn_act=True,
            )

        # Compute D and run backward (same as Phase 3.5/3.6)
        # ...

        metric = (base_logits[meas_pos, correct_token_id]
                  - base_logits[meas_pos, incorrect_token_id])
        metric.backward()

    # Emit per-neuron cells
    num_layers = len(model.model.layers)
    target_layers_set = set(range(num_layers)) if layers is None else set(layers)
    intermediate_size = model.config.intermediate_size

    all_cells: List[Dict] = []
    for L in sorted(target_layers_set):
        if (L, "ffn_out") not in base_captured:
            continue
        base_ffn_out = base_captured[(L, "ffn_out")]
        if base_ffn_out.grad is None:
            continue
        W_down = model.model.layers[L].mlp.down_proj.weight  # [hidden, intermediate]
        from_act = from_ffn_acts[L]
        base_act = base_ffn_acts[L]

        for pos in normalized_positions:
            grad_ffn_out = base_ffn_out.grad[0, pos].detach()          # [hidden]
            grad_act = grad_ffn_out @ W_down                           # [intermediate]
            delta_act = from_act[0, pos] - base_act[0, pos].detach()   # [intermediate]
            ap_raw = (delta_act * grad_act)                             # [intermediate]
            ap_recovery = (
                ap_raw / denominator if direction == "denoise"
                else 1.0 + ap_raw / denominator
            )
            for i in range(intermediate_size):
                all_cells.append({
                    "layer": L,
                    "unit": f"neuron.n{i}",
                    "neuron": i,
                    "position": pos,
                    "ap_recovery": float(ap_recovery[i].item()),
                })

    n_total = len(all_cells)
    all_cells.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
    top_cells = all_cells[:top_k_neurons]

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    return PatchingResult(
        cells=top_cells,
        # ...same as Phase 3.6...
        mode="approx_neuron",
        n_neurons=intermediate_size,
    )
```

**Performance note**: the per-i loop in the emit is ~124k iterations for TinyLlama across all layers × positions (22 × 5632 × ~5). Each iteration is a Python-level dict construction and `.item()` call. Benchmark: expected ~1–2s per prompt on CPU, O(n_layers × intermediate_size × seq_len). If this proves slow, vectorize by computing `topk` per (L, pos) in torch and appending only `top_k_neurons / (n_layers * seq_len)` rows per slice. Defer the vectorization until the straightforward implementation is shown to be slow.

## 6. WS protocol

Route `/ws/sessions/{name}/activation-patching` gains a sixth mode.

**Client cfg:**
```jsonc
{
  "op": "activation-patching",
  "cfg": {
    "mode": "approx_neuron",
    "clean_prompt": "...",
    "corrupted_prompt": "...",
    "direction": "denoise",
    "measurement_position": -1,
    "correct_token_id": 1234,
    "incorrect_token_id": 5678,
    "top_k_neurons": 200
  }
}
```

**Server streams**: `status` → `data` (one per cell, in top-k order) → `baselines` → `complete`.

Complete summary extras:
```jsonc
{
  "mode": "approx_neuron",
  "n_neurons": 5632,
  "top_k_neurons": 200
}
```

Auto-pick token IDs extends to `approx_neuron` (same logic as edge/circuit).

## 7. Frontend

### 7.1 Types

`api.ts`:
- `PatchingMode` gains `"approx_neuron"`.
- `PatchingCellData.neuron?: number`.
- `PatchingCompleteData.summary.n_neurons?: number`, `top_k_neurons?: number`.

### 7.2 Controls

`PatchingControls.tsx`:
- Sixth radio "per-neuron FFN (approx)".
- When `mode === "approx_neuron"`, show one numeric input: `top_k_neurons` (default 200, min 1).
- `PatchingState` gains `top_k_neurons: number`.

### 7.3 Visualization: `PerNeuronPatchingPanel.tsx`

New top-level panel (sibling of `EdgeAttributionPanel.tsx` etc.). Routed from `VisualizationArea.tsx` when `mode === "approx_neuron"`.

**Layout** (top to bottom):
1. **Controls row**: position selector (default = measurement_position), layer filter (`All` or per-layer chip), search-by-neuron-id input, "copy TSV" button.
2. **Stats**: `Showing N of M cells (filtered from top_k_neurons)` + min/max/mean of `ap_recovery` across visible rows.
3. **Ranked table**: sortable by `ap_recovery` (default desc) or `layer` or `neuron`. Columns: `#`, `layer`, `neuron`, `position`, `ap_recovery`. Row background colored with PiYG scale mapped to `[-0.5, 1.0]` (same as Phase 3.5+). Truncated to 200 rows max displayed (virtual scroll if needed — probably unnecessary at 200 default).

**No heatmap** — 5632 neurons per layer makes per-position grid unrenderable. If later demanded, add a "top-20-per-layer" heatmap variant.

**Pinned card**: clicking a row pins it below the table. Shows `layer`, `neuron index`, `position`, `ap_recovery`, and a placeholder "decode top logits for this neuron" button (non-functional v1; wired in Phase 3.9.1 if needed).

### 7.4 Fixture & smoke test

- `frontend/tests/e2e/fixtures/activation-patching-per-neuron.json` — mirrors `activation-patching-per-head.json` structure:
  - `"schema": "llm-surgeon-gui-experiment/v1"`
  - cells carry `layer`, `unit: "neuron.n<i>"`, `neuron`, `position`, `ap_recovery`
  - summary: `mode: "approx_neuron"`, `n_neurons: 5632`, `top_k_neurons: 5` (use a small k in the fixture).
- 15th Playwright test: imports fixture, asserts the panel renders with a ranked table, position selector, layer filter, "copy TSV" button.

## 8. Testing

### 8.1 Python unit tests (`test_probe_per_neuron_ap.py`)

New test file. Mock-model cases (fast, no GPU):

1. **`test_returns_patching_result`** — smoke.
2. **`test_mode_and_n_neurons_set`** — `mode == "approx_neuron"`, `n_neurons == model.config.intermediate_size`.
3. **`test_sum_invariant_mock`** — compute `Σ_i ap_neuron_raw(L, i, pos)` directly from captures and assert it equals `(delta_ffn_out · grad_ffn_out) / D` within 1e-5. Mirrors the Phase 3.6 sum-invariant test but for neurons.
4. **`test_top_k_respected`** — `top_k_neurons=10` ⇒ `len(result.cells) == 10`; cells sorted desc by `|ap_recovery|`.
5. **`test_top_k_exceeds_total_caps`** — `top_k_neurons=10**9` ⇒ `len(cells) == total neurons available`.
6. **`test_cells_have_required_fields`** — every cell has `layer`, `unit`, `neuron`, `position`, `ap_recovery`; `unit == f"neuron.n{neuron}"`.
7. **`test_validation`** — `top_k_neurons < 1`, empty prompt, mismatched-length prompts all raise.
8. **`test_capture_helper_7_tuple`** — direct call to `_capture_residual_stream_with_grad(capture_ffn_act=True)` returns a 7-tuple and `ffn_acts` is populated per-layer.
9. **`test_capture_ffn_act_false_default`** — default `capture_ffn_act=False` produces an empty `ffn_acts` dict (backward compatibility).
10. **`test_phase_3_regression_unpacks`** — a quick test that Phase 3.6/3.7 callers in the tree still work after the tuple-arity change. This runs the existing `edge_attribution_patch` / `attribution_patch_per_head` on the mock and asserts they succeed.

### 8.2 TinyLlama integration test (GPU-guarded)

`TestTinyLlamaPerNeuron::test_top_neurons_identifiable` — skipif no GPU / no cached model. fp16 for OOM headroom (reader-grad tensors + ffn_out retentions ~ Phase 3.7 footprint).

Assertions:
- `mode == "approx_neuron"`, `n_neurons == 5632`, `top_k_neurons` respected.
- All cells have `0 <= neuron < 5632`, `0 <= layer < 22`.
- Cells sorted desc by `|ap_recovery|`.
- `|ap_recovery|` of top cell is > 0.01 (sanity: the top neuron on a capital-of-France prompt is not trivially zero).

### 8.3 Frontend tests

- **Vitest**: none needed. `PerNeuronPatchingPanel` is display-only (no new pure utility functions). If a sort helper is extracted, add a Vitest case.
- **Playwright**: fixture-based smoke (§7.4).

## 9. Commit plan

8 tasks, one commit each (combine where subagent edits the same file — same policy as Phases 3.6/3.7/3.8).

1. **Spec commit** — this file.
2. **Capture helper 7-tuple refactor** — `capture_ffn_act` flag + update all existing callers. Phase 3.5/3.6/3.7/3.8 tests must still pass.
3. **`attribution_patch_per_neuron` implementation** — probe.py + `PatchingResult.n_neurons` field.
4. **Python unit + TinyLlama tests** — `test_probe_per_neuron_ap.py`.
5. **Backend WS `approx_neuron` branch** — `routes/probes.py`.
6. **Frontend types** — `api.ts`.
7. **`PatchingControls`** sixth radio + `top_k_neurons` input. **`ProbePanel`** forwards `top_k_neurons`.
8. **`PerNeuronPatchingPanel.tsx`** + `VisualizationArea` routing + Playwright fixture + smoke test.

## 10. Verification matrix

Before declaring phase shipped:

- **pyright**: 0/0/0 across `probe.py`, `routes/probes.py`, all test files.
- **tsc**: clean on frontend.
- **pytest (non-GPU)**: Phase 3.5/3.6/3.7/3.8 regression suites still pass + new per-neuron suite passes.
- **TinyLlama integration**:
  - Phase 3.5 Spearman vs exact ≥ 0.95.
  - Phase 3.6 sum-heads vs attn_out target ρ = 1.0000.
  - Phase 3.7 EAP top-k consistency passes.
  - Phase 3.8 circuit integration test passes.
  - New per-neuron integration passes.
- **Vitest**: existing suites still green.
- **Playwright**: existing 14 + 1 new = 15 passing.

## 11. Open questions

*(None — resolved in §1–10.)*
