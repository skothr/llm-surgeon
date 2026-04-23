# Phase 3.6 — Per-Head Attribution Patching Design

**Date:** 2026-04-21
**Roadmap:** `llm_surgeon/project_llm_surgeon_roadmap.md` (Phase 3.6)
**Goal:** extend gradient-based attribution patching to per-attention-head granularity. Given clean and corrupted prompts, produce per-`(layer, head, position)` AP scores for the attention sublayer via a chain-rule decomposition of the existing `concat_z` pre-`o_proj` tensor, plus FFN AP as an anchor row (same math as Phase 3.5, unchanged).

**Why per-head (not per-neuron):** attention heads are the natural unit of mechanistic interpretability — circuits are almost always described in terms of heads (induction head, name-mover head, etc.). FFN neurons are an alternative unit; per-head gives immediate circuit-extraction utility without the dimensionality explosion of per-neuron FFN attribution.

**Why a separate function (not a flag on `attribution_patch`):** `attribution_patch` is already shipped and tested. The new function has a different capture structure, different output keys, and different frontend contract. Keeping them separate allows clean TDD of each without entangling the Phase 3.5 invariants.

**Non-goals for Phase 3.6:** per-neuron FFN attribution, edge attribution (Syed et al. 2023), multi-head grouping (GQA), QK/OV circuit decomposition. All deferred.

---

## 1. Architecture Overview

### 1.1 Residual-stream → per-head decomposition

In HuggingFace LLaMA, the attention output at layer L is:

```
attn_out[L] = concat_z[L] @ W_O[L].T
```

where:
- `concat_z[L]`: shape `[batch, seq, hidden]` — the concatenation of all head output vectors before the output projection. Also called the "pre-o_proj" tensor or "z-space" tensor.
- `W_O[L]`: `self_attn.o_proj.weight`, shape `[hidden, hidden]`.
- `@` is batched matrix multiply over (batch, seq) positions.

Sliced by head h (with `d = hidden / n_heads = head_dim`):

```
concat_z[L, :, h*d:(h+1)*d]  →  head h's contribution to attn_out via W_O[h*d:(h+1)*d, :]
```

The full `attn_out` is the sum of each head's contribution.

### 1.2 Chain-rule gradient to concat_z-space

Phase 3.5 captures `attn_out.grad = ∂metric/∂attn_out` via `retain_grad()` on the `attn_out` tensor. The chain rule through the linear `o_proj` gives the gradient with respect to `concat_z`:

```
∂metric/∂concat_z[L, pos, :] = (∂metric/∂attn_out[L, pos, :]) @ W_O[L]
```

i.e., right-multiply `attn_out.grad[0, pos, :]` (shape `[hidden]`) by `W_O` (shape `[hidden, hidden]`) to get `concat_z_grad[0, pos, :]` (shape `[hidden]`).

This is a free O(hidden²) computation — no additional backward pass required.

### 1.3 Per-head AP score

Reshape `concat_z` and `concat_z_grad` from `[1, seq, hidden]` to `[1, seq, n_heads, head_dim]`:

```
concat_z_reshaped[L]      = concat_z[L].view(1, seq, n_heads, head_dim)
concat_z_grad_reshaped[L] = concat_z_grad[L].view(1, seq, n_heads, head_dim)
```

Then for head h at (L, pos):

```
AP_head(L, h, pos) = sum_d(
    (from_concat_z[L, 0, pos, h, :] - base_concat_z[L, 0, pos, h, :])
    * concat_z_grad[L, 0, pos, h, :]
)
```

Vectorized (all heads simultaneously):

```python
delta_z = from_cz[0, pos] - base_cz[0, pos]          # [n_heads, head_dim]
ap_heads = (delta_z * cz_grad[0, pos]).sum(dim=-1)   # [n_heads]
```

Normalization (denoise, same as Phase 3.5):

```
ap_recovery_head(L, h, pos) = AP_head(L, h, pos) / (d_clean - d_corrupted)
```

Noise direction:

```
ap_recovery_head(L, h, pos) = 1 + AP_head(L, h, pos) / (d_clean - d_corrupted)
```

### 1.4 Sum invariant

Because the per-head AP values are a linear decomposition of the full attn-row AP (via the chain rule through a linear map), their sum equals the Phase 3.5 attn-row AP:

```
sum_h AP_head(L, h, pos) ≈ AP_attn(L, pos)
```

This is exact up to floating-point arithmetic (tolerance ~1e-5), and is the primary correctness unit test.

**Why "≈" and not "=":** Phase 3.5 `attribution_patch` captures `attn_out` directly and computes `attn_out.grad` via `retain_grad()`. Phase 3.6 reconstructs `concat_z_grad` from `attn_out.grad @ W_O`, then sums. Any floating-point deviation comes from the matmul; the algebra is exact.

### 1.5 FFN AP (unchanged)

`AP_ffn(L, pos)` is computed identically to Phase 3.5 using the captured `ffn` tensor and its gradient. It appears as the first column in the per-head heatmap ("ffn" unit) to provide a reference scale and to let users see whether the attention heads at a layer collectively contribute more than the FFN.

---

## 2. Python API

### 2.1 New function: `attribution_patch_per_head`

```python
def attribution_patch_per_head(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",          # "denoise" | "noise"
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    layers: Optional[List[int]] = None,
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult: ...
```

Notes:
- `sublayers` parameter is dropped (always captures both attn and ffn; the sublayer axis is the head axis for attn, plus a single "ffn" column).
- `on_cell` callback signature is unchanged: `(layer: int, unit: str, position: int, cell: dict)`. The `unit` string is `"attn.h{N}"` (0-indexed) for head N, or `"ffn"` for the FFN anchor.
- Returns a `PatchingResult` with `mode="approx_head"`.

### 2.2 Extended `_capture_residual_stream_with_grad`

Add a `capture_concat_z: bool = False` keyword argument:

```python
def _capture_residual_stream_with_grad(
    model,
    tokenizer,
    prompt: str,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    capture_concat_z: bool = False,                # NEW
) -> Tuple[
    Dict[Tuple[int, str], torch.Tensor],           # captured residual states
    Dict[int, torch.Tensor],                        # h_ins (layer pre-hook inputs)
    torch.Tensor,                                   # output logits
    List[str],                                      # prompt tokens
    Dict[int, torch.Tensor],                        # concat_z per layer (NEW, empty if flag=False)
]: ...
```

When `capture_concat_z=True`, registers an additional `register_forward_pre_hook` on `model.model.layers[i].self_attn.o_proj`. This hook fires just before `o_proj`'s forward, and `args[0]` is the `concat_z` tensor (shape `[batch, seq, hidden]`) that is the pre-projection attention output:

```python
def make_concat_z_hook(idx):
    def hook(_module, args):
        z = args[0]                          # [batch, seq, hidden]
        if z.requires_grad:
            z.retain_grad()
        concat_z_captured[idx] = z
    return hook
hooks.append(
    model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(
        make_concat_z_hook(i)
    )
)
```

The captured `concat_z` tensors remain in the autograd graph (no `.detach()`). When `capture_concat_z=False` (default), the dict is empty and the return signature is backwards-compatible except for the extra return element. Callers that don't use `capture_concat_z` (i.e., `attribution_patch` from Phase 3.5) must be updated to unpack 5 values instead of 4.

### 2.3 `PatchingResult` dataclass extension

Add `n_heads: Optional[int] = None` field (default `None` preserves all existing call sites):

```python
@dataclass
class PatchingResult:
    cells: List[Dict]
    clean_baseline_logits: torch.Tensor
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: List[str]
    prompt_tokens_corrupted: List[str]
    direction: str
    measurement_position: int
    mode: str = "exact"                  # "exact" | "approx" | "approx_head"
    n_heads: Optional[int] = None        # NEW: set by attribution_patch_per_head
```

### 2.4 Cell dict format

Per-head cells:

```python
{"layer": L, "unit": "attn.h3", "position": pos, "ap_recovery": 0.83}
```

FFN cells (anchor):

```python
{"layer": L, "unit": "ffn", "position": pos, "ap_recovery": 0.12}
```

The `on_cell` callback receives `unit` as the second argument (where Phase 3.5 received `sublayer`). The existing `on_cell` signature `(int, str, int, dict)` is unchanged — the string just carries a different vocabulary of values.

### 2.5 Core algorithm (denoise direction)

```python
# --- Step 1: Forward 'from' prompt (no_grad) ---
with torch.no_grad():
    from_captured, from_h_ins_raw, from_logits, from_tokens, from_concat_z_raw = \
        _capture_residual_stream_with_grad(
            model, tokenizer, from_prompt,
            sublayers=("attn", "ffn"), layers=layers,
            capture_concat_z=True,
        )
    from_states = {k: v.detach().clone() for k, v in from_captured.items()}
    from_h_ins = {idx: v.detach().clone() for idx, v in from_h_ins_raw.items()}
    from_concat_z = {idx: v.detach().clone() for idx, v in from_concat_z_raw.items()}

# --- Step 2: Forward 'base' prompt (with grad) ---
with torch.enable_grad():
    base_captured, base_h_ins, base_logits, base_tokens, base_concat_z = \
        _capture_residual_stream_with_grad(
            model, tokenizer, base_prompt,
            sublayers=("attn", "ffn"), layers=layers,
            capture_concat_z=True,
        )
    ...
    metric.backward()

# --- Step 3: Compute n_heads and W_O ---
# n_heads from model config; head_dim = hidden // n_heads.
n_heads: int = model.config.num_attention_heads
hidden: int = model.config.hidden_size
head_dim: int = hidden // n_heads

# --- Step 4: Per-cell AP ---
for L in sorted_layers:
    # FFN anchor (same as Phase 3.5)
    base_ffn = base_captured[(L, "ffn")]           # [1, seq, hidden]
    ffn_grad = base_ffn.grad                        # [1, seq, hidden]
    from_ffn = from_states[(L, "ffn")]
    for pos in normalized_positions:
        ap_raw = ((from_ffn[0, pos] - base_ffn[0, pos].detach()) * ffn_grad[0, pos]).sum().item()
        ap_recovery = ap_raw / denominator          # denoise
        on_cell(L, "ffn", pos, {"layer": L, "unit": "ffn", "position": pos, "ap_recovery": ap_recovery})

    # Attention heads via concat_z
    base_cz = base_concat_z[L]                     # [1, seq, hidden], in graph
    from_cz = from_concat_z[L]                     # [1, seq, hidden], detached
    W_O = model.model.layers[L].self_attn.o_proj.weight   # [hidden, hidden]
    attn_out_grad = base_captured[(L, "attn")].grad        # [1, seq, hidden]

    # Chain rule: ∂metric/∂concat_z = attn_out_grad @ W_O
    # (attn_out = concat_z @ W_O.T, so dL/dconcat_z = dL/dattn_out @ W_O, NO transpose)
    # attn_out_grad[0]: [seq, hidden]; W_O: [hidden, hidden]
    # => concat_z_grad[0]: [seq, hidden]
    concat_z_grad = (attn_out_grad[0] @ W_O)               # [seq, hidden]

    for pos in normalized_positions:
        delta_z = (from_cz[0, pos] - base_cz[0, pos].detach())     # [hidden]
        cz_grad_pos = concat_z_grad[pos]                             # [hidden]

        # Reshape to [n_heads, head_dim] for per-head sum
        dz_heads = delta_z.view(n_heads, head_dim)           # [n_heads, head_dim]
        cz_grad_heads = cz_grad_pos.view(n_heads, head_dim)  # [n_heads, head_dim]
        ap_heads = (dz_heads * cz_grad_heads).sum(dim=-1)    # [n_heads]

        for h in range(n_heads):
            ap_raw_h = ap_heads[h].item()
            ap_recovery_h = ap_raw_h / denominator            # denoise
            unit = f"attn.h{h}"
            on_cell(L, unit, pos, {"layer": L, "unit": unit, "position": pos,
                                   "ap_recovery": float(ap_recovery_h)})
```

For `noise` direction the sign convention is identical to Phase 3.5: `ap_recovery = 1 + ap_raw / denominator`, applied independently to each head and the FFN anchor.

### 2.6 Memory profile

For TinyLlama (22 layers, hidden=2048, n_heads=32, head_dim=64, seq≈6):
- `concat_z` per layer: `[1, 6, 2048]` × f32 = 48 KB. All layers: 1 MB.
- `concat_z_grad` (recomputed from `attn_out_grad @ W_O`): same shape. 1 MB.
- `W_O` per layer: `[2048, 2048]` × f32 = 16 MB. Already in model weights — no copy.
- Total new overhead vs Phase 3.5: ~2 MB activations. Negligible on RTX 2080.

### 2.7 Validation rules (superset of `attribution_patch`)

| Condition | Error |
|---|---|
| All Phase 3.5 validation conditions | as before |
| `correct_token_id` or `incorrect_token_id` is `None` | `ValueError("attribution_patch_per_head requires correct_token_id and incorrect_token_id")` |
| `abs(d_clean - d_corrupted) < 1e-6` | `ValueError("clean and corrupted baselines have identical logit_diff; AP would divide by zero")` |
| `model.config.num_attention_heads` is unavailable | `AttributeError` propagates naturally; no special handling |

---

## 3. Backend WS Route

**No new route.** `/ws/sessions/{name}/activation-patching` gains a third `mode` value:

```python
mode = cfg.get("mode", "exact")
if mode not in ("exact", "approx", "approx_head"):
    send error; return
```

For `mode == "approx_head"`, token-pair resolution is identical to `mode == "approx"`. The dispatch:

```python
elif mode == "approx_head":
    assert correct_token_id is not None and incorrect_token_id is not None
    result = attribution_patch_per_head(
        info.model, info.tokenizer,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        direction=direction,
        measurement_position=measurement_position,
        positions=positions,
        layers=layers,
        on_cell=on_cell,
    )
```

### Frame format

Data frames for per-head mode carry `unit` instead of `sublayer`:

```json
{"type": "data", "layer": 5, "unit": "attn.h3", "position": 2, "ap_recovery": 0.71}
{"type": "data", "layer": 5, "unit": "ffn",     "position": 2, "ap_recovery": 0.12}
```

The `on_cell` closure in the route handler already checks `"ap_recovery" in cell` to populate the frame. It also needs to forward `unit` instead of `sublayer` for `approx_head` mode:

```python
def on_cell(layer_idx: int, unit: str, position: int, cell: dict) -> None:
    msg: dict = {
        "type": "data",
        "layer": layer_idx,
        "original_layer": info.original_layer(layer_idx),
        "position": position,
    }
    # In approx_head mode, cell carries "unit"; in exact/approx, it carries "sublayer".
    if "unit" in cell:
        msg["unit"] = cell["unit"]
    else:
        msg["sublayer"] = unit        # existing exact/approx path
    if "patched_logits" in cell:
        msg["patched_logits"] = _encode_hidden_state(cell["patched_logits"])
    if "ap_recovery" in cell:
        msg["ap_recovery"] = cell["ap_recovery"]
    ...
```

The `complete` frame summary gains `n_heads`:

```json
{
  "type": "complete",
  "summary": {
    "num_cells": 924,
    "direction": "denoise",
    "measurement_position": 5,
    "mode": "approx_head",
    "n_heads": 32
  }
}
```

---

## 4. Frontend Types

### `api.ts` additions

```typescript
export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer?: "attn" | "ffn";       // present in exact/approx modes
  unit?: string;                    // present in approx_head mode: "attn.hN" or "ffn"
  head?: number | null;             // derived on the frontend: parsed from unit
  position: number;
  patched_logits?: EncodedTensor;
  ap_recovery?: number;
}

export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
    mode?: "exact" | "approx" | "approx_head";   // extended
    n_heads?: number;                              // present when mode === "approx_head"
  };
}
```

`head` and `unit` are optional and absent in exact/approx modes; the frontend parses them where needed rather than requiring the backend to duplicate the field.

---

## 5. Frontend: `PerHeadPatchingHeatmap.tsx`

New component. Does NOT modify `ActivationPatchingHeatmap.tsx` (clean separation of concerns; the per-head layout is fundamentally different).

### 5.1 Layout

```
┌──────────────────────────────────────────────────┐
│  Per-head Attribution Patching — session — "prompt…"   [position: ▼ 5]  │
├──────────────────────────────────────────────────┤
│         │ ffn │ h0 │ h1 │ h2 │ … │ h31 │         │
│  L0     │     │    │    │    │   │     │         │
│  L1     │     │    │    │    │   │     │         │
│  …      │     │    │    │    │   │     │         │
│  L21    │     │    │    │    │   │     │         │
└──────────────────────────────────────────────────┘
```

- **Rows**: layers, top = first (L0) to bottom = last (L21 for TinyLlama).
- **Columns**: `ffn` | `head 0` | `head 1` | … | `head N-1`. FFN anchor is the leftmost data column (after the row-label column).
- **Position selector**: dropdown above the grid. Options are token strings (from `prompt_tokens_clean`). Changing position re-renders the grid for the selected position's cells.
- **Color scale**: `d3.interpolatePiYG`, fixed domain `[-0.5, 1.0]`. Same as Phase 3.5 for visual consistency.

### 5.2 Position selector

```tsx
const [selectedPos, setSelectedPos] = useState<number>(completeFrame?.summary.measurement_position ?? 0);
```

Rendered as:

```tsx
<select value={selectedPos} onChange={e => setSelectedPos(Number(e.target.value))}>
  {promptTokens.map((tok, i) => (
    <option key={i} value={i}>{i}: {tok}</option>
  ))}
</select>
```

### 5.3 Pinned-card

Click on any cell pins a card showing:

```
Layer N, head H (or "ffn")
AP recovery: 0.831
─────────────────────────────────────────────
First-order approximation — run exact mode to confirm.
```

No logit top-K shown (approx mode has no patched forward pass).

### 5.4 Routing / display

`ProbePanel.tsx` or the parent viz-router reads `completeFrame.summary.mode`:
- `"exact"` → `<ActivationPatchingHeatmap>`
- `"approx"` → `<ActivationPatchingHeatmap>` (existing, unchanged)
- `"approx_head"` → `<PerHeadPatchingHeatmap>` (new)

### 5.5 Frontend controls

`PatchingControls.tsx` mode radio gains a third option:

```tsx
type PatchingMode = "exact" | "approx" | "approx_head";

<label>
  <input type="radio" checked={state.mode === "approx_head"}
         onChange={() => onChange({ ...state, mode: "approx_head" })} />
  per-head <span style={{ color: "#888", fontSize: 11 }}>(gradient AP, head resolution)</span>
</label>
```

`PatchingState.mode` becomes `"exact" | "approx" | "approx_head"`.

---

## 6. Verification Plan

### 6.1 Unit tests (`testing/tests/test_probe_per_head_ap.py`)

| Test | What it checks |
|---|---|
| `TestCaptureConcat_z::test_concat_z_shape` | `capture_concat_z=True` returns `concat_z[L]` shape `[1, seq, hidden]` for all target layers |
| `TestCaptureConcat_z::test_concat_z_in_graph` | Base-side `concat_z` tensors have `requires_grad=True` and non-None `grad_fn` |
| `TestCaptureConcat_z::test_concat_z_grad_populates` | After `metric.backward()`, `concat_z[L].grad` is non-None and non-zero |
| `TestPerHeadAP::test_sum_invariant_mock` | On a mock model (hidden=8, n_heads=2, head_dim=4, L=2): `sum_h AP_head(L,h,pos) ≈ AP_attn(L,pos)` at tolerance 1e-5 |
| `TestPerHeadAP::test_ffn_anchor_matches_phase35` | FFN cells from `attribution_patch_per_head` match `attribution_patch` FFN cells exactly (same code path) |
| `TestPerHeadAP::test_cell_count_mock` | With L=2, n_heads=2, seq=3, positions=all: expect `(L * n_heads + L * 1) * seq = (2*2 + 2*1) * 3 = 18` cells |
| `TestPerHeadAP::test_noise_direction` | Noise direction applies `1 + ap_raw/D` correctly for head cells |
| `TestPerHeadAP::test_on_cell_unit_strings` | `on_cell` receives `"attn.h0"`, `"attn.h1"`, `"ffn"` as expected unit strings |
| `TestSpearman::test_tinyllama_head_sum_vs_node_ap` | TinyLlama: `sum_h AP_head(L,h,pos)` vs `AP_attn(L,pos)` Spearman ρ > 0.95. Guarded by `@pytest.mark.skipif(not _tinyllama_cached() or not torch.cuda.is_available())` |

### 6.2 Backend (no new pytest — WS tested via Playwright)

Verify `cfg.mode == "approx_head"` branch is handled: pyright 0/0/0 on `probes.py` after changes.

### 6.3 Frontend type-check

`tsc --noEmit` clean on all modified files after each task.

### 6.4 Playwright smoke

One new test in `smoke.spec.ts`:
- Seed fixture `activation-patching-per-head.json` with `mode: "approx_head"` and a handful of `unit`-keyed cells.
- Assert heading `/Per-head Attribution/` is visible.
- Assert position selector dropdown is present.

Total suite: 11 existing + 1 new = 12 tests.

---

## 7. Data Flow Summary

```
User: mode = "approx_head" → ProbePanel.handleRun → WS cfg
  ↓
Backend: receive cfg → auto-pick token IDs → attribution_patch_per_head()
  ↓
  _capture_residual_stream_with_grad(capture_concat_z=True) × 2 (from, base)
  metric.backward()
  for each (L, pos):
    concat_z_grad = attn_out_grad @ W_O  [chain rule, no extra backward]
    per_head_ap = (delta_z * concat_z_grad).sum(dim=-1)  [vectorized]
    ffn_ap = (delta_ffn * ffn_grad).sum()
  → PatchingResult(mode="approx_head", n_heads=N)
  ↓
Backend: stream data frames with "unit" key → complete frame with n_heads
  ↓
Frontend: PatchingCompleteData.summary.mode === "approx_head"
  → render <PerHeadPatchingHeatmap>
  → position selector → grid [rows=layers, cols=ffn|h0|h1|…|hN-1]
  → click cell → pinned card (ap_recovery scalar, no top-k)
```

---

## 8. File Map

| File | Change |
|---|---|
| `testing/llm_surgeon/probe.py` | **+** `attribution_patch_per_head()`, extend `_capture_residual_stream_with_grad` with `capture_concat_z` flag, add `n_heads` field to `PatchingResult`; update `attribution_patch` callers to unpack 5-tuple return |
| `testing/tests/test_probe_per_head_ap.py` | **new** — unit tests + TinyLlama Spearman |
| `testing/gui/backend/routes/probes.py` | **~** `approx_head` mode branch; `on_cell` emits `unit` for head mode; complete frame gains `n_heads` |
| `testing/gui/frontend/src/types/api.ts` | **~** `PatchingCellData.unit?`, `PatchingCellData.head?`, `PatchingCompleteData.summary.mode` extended, `summary.n_heads?` |
| `testing/gui/frontend/src/components/PatchingControls.tsx` | **~** third mode radio (`"approx_head"`), extend `PatchingMode` type |
| `testing/gui/frontend/src/components/ProbePanel.tsx` | **~** route `mode === "approx_head"` to `<PerHeadPatchingHeatmap>` |
| `testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx` | **new** — position selector, layer×head grid, pinned card |
| `testing/gui/frontend/tests/e2e/smoke.spec.ts` | **+** one per-head smoke test |
| `testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-head.json` | **new** — fixture with `mode: "approx_head"` + unit-keyed cells |

---

## 9. Explicit Non-Goals

1. **Per-neuron FFN attribution** — dimensionality explosion (intermediate_size ≈ 11008 for TinyLlama). Deferred.
2. **GQA / MQA support** — TinyLlama uses standard MHA (`num_key_value_heads == num_attention_heads`). GQA would need `o_proj` weight partitioning adjusted. Not in scope.
3. **Exact per-head patching loop** — `O(L × n_heads × P)` forward passes. Possible future work; current phase is gradient-only.
4. **Top-K logits in pin card** — approx_head has no patched forward pass; exact mode still provides it via `ActivationPatchingHeatmap`. No regression.
5. **Circuit extraction UI** — out of scope. The head scores are an input to circuit analysis; automating extraction is a separate phase.
6. **Side-by-side node-level vs per-head** — shared color scale allows manual comparison (rerun in each mode), but no simultaneous multi-mode display.
7. **`_capture_residual_stream_with_grad` return signature breakage** — the 5th return element (`concat_z` dict) is added unconditionally. The single Phase 3.5 call site (`attribution_patch`) must be updated to unpack 5 values. This is a contained one-line fix.
