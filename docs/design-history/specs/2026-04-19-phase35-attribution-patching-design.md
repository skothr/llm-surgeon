# Phase 3.5 — Attribution Patching Design

**Date:** 2026-04-19
**Roadmap:** `llm_surgeon/project_llm_surgeon_roadmap.md` (Phase 3.5)
**Goal:** add **gradient-based attribution patching** (Nanda, 2023; Kramár et al., 2024) as a cheaper *approximation* of exact activation patching — one forward + one backward pass replaces the `layers × sublayers × positions` inner loop. Ships as a **mode toggle** on the existing `activation-patching` operation; exact AP stays the default.

**Why mode-toggle (not new op):** exact AP and attribution patching answer the *same* causal question ("does patching at this cell recover clean behavior?"). The only difference is compute cost and approximation quality. A shared operation lets future work show exact vs approx side-by-side for validation (Kramár et al. 2024 evaluates AP quality exactly this way).

**Non-goal for Phase 3.5:** edge attribution, per-head AP, circuit extraction. Deferred.

---

## 1. Architecture overview

Attribution patching replaces the `O(L·S·P)` patching loop with a single first-order Taylor approximation. For a cell `c = (layer, sublayer, position)` the AP score is:

```
AP(c) = (act_from[c] − act_base[c]) · ∂metric / ∂act_base[c]
```

- **`denoise` direction**: base = corrupted, from = clean, metric backpropped = `logit_diff_clean = logit[correct] − logit[incorrect]` (clean direction). Positive AP(c) ≈ "patching clean activations at c *recovers* clean behavior" ≈ exact `logit_diff_recovery`.
- **`noise` direction**: base = clean, from = corrupted, metric on clean run. Positive AP(c) = "patching corrupted activations at c *destroys* clean behavior." Sign convention matches exact AP's `logit_diff_recovery` value (close to 0 = destroyed, close to 1 = untouched).

To make approx scores visually comparable to exact `logit_diff_recovery`, the backend normalises (letting `D = Δ_clean − Δ_corrupted`):

```
# denoise: from=clean, base=corrupted; AP ≈ Δ(logit_diff) from patching clean in
ap_recovery(c) = AP(c) / D

# noise: from=corrupted, base=clean; AP ≈ Δ(logit_diff) from patching corrupted in
# patched_diff ≈ Δ_clean + AP(c); recovery = (patched_diff − Δ_corrupted) / D
ap_recovery(c) = 1 + AP(c) / D
```

Both conventions produce values that line up with exact `logit_diff_recovery`: near 1 = patching matches clean behavior; near 0 = patching doesn't help (denoise) or fully destroys (noise). Normalisation keeps the heatmap's color scale (PiYG, fixed `[-0.5, 1.0]`) reusable across modes.

**User flow:** ProbePanel → operation = `activation-patching` → `PatchingControls` renders → new **mode radio** (`exact` / `approx`) — exact is default → Run → same heatmap renders. In `approx` mode, the metric dropdown collapses to a single "AP recovery" entry (AP has only one natural metric).

---

## 2. Backend: `probe.py::attribution_patch()`

New public function alongside `activation_patch`. Shares dataclasses (`PatchingResult`) with one added field.

```python
def attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    direction: str = "denoise",               # "denoise" | "noise"
    measurement_position: int = -1,
    correct_token_id: int,                    # required — AP needs a scalar metric
    incorrect_token_id: int,                  # required
    positions: Optional[List[int]] = None,    # None = all
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,       # None = all
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult: ...
```

### Why `correct_token_id` / `incorrect_token_id` are required

Exact AP can lazily argmax the clean baseline to pick a target token — the result is a full logit vector, metrics are chosen later on the frontend. Attribution patching has to backprop through a scalar, chosen before the backward pass. We require explicit IDs rather than recomputing `argmax` inside: the backend already resolves auto-pick in the WS route before invoking `attribution_patch`, so the contract is clean.

### Extended dataclass

```python
@dataclass
class PatchingResult:
    cells: List[Dict]                         # per (layer, sublayer, position) frame
    clean_baseline_logits: torch.Tensor
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: List[str]
    prompt_tokens_corrupted: List[str]
    direction: str
    measurement_position: int
    mode: str = "exact"                       # NEW: "exact" | "approx"
```

Cell dicts carry whichever of `patched_logits` (exact) or `ap_recovery` (approx) applies:

```python
# exact cell (unchanged)
{"layer": L, "sublayer": sub, "position": pos, "patched_logits": <tensor>}
# approx cell (new)
{"layer": L, "sublayer": sub, "position": pos, "ap_recovery": <float>}
```

### Algorithm (denoise)

```
1. tokenize both prompts; assert same length (same validation as activation_patch)
2. With torch.no_grad():
     clean_states   = _capture_residual_stream(model, tok, clean_prompt,
                          sublayers=sublayers, layers=layers)
     clean_logits   = forward(clean_prompt).logits[0, measurement_position]
3. With torch.enable_grad():
     Register hooks on corrupted forward that:
       - capture each (L, sub) residual-stream tensor in a dict AS LIVE TENSORS
         (no .detach()), and call tensor.requires_grad_(True) + retain_grad()
     Run forward(corrupted_prompt); read corrupted_logits[0, measurement_position]
4. Compute scalar metric:
     d_clean     = clean_logits[correct_id]     - clean_logits[incorrect_id]
     d_corrupted = corrupted_logits[correct_id] - corrupted_logits[incorrect_id]
     metric      = d_corrupted                     # denoise: ∂d_corrupted / ∂act
5. metric.backward()
6. For each (L, sub) in captured corrupted tensors:
     corr_act     = captured_corrupted[(L, sub)]        # shape (seq, d_model)
     corr_grad    = corr_act.grad                        # shape (seq, d_model)
     clean_act    = clean_states[(L, sub)]
     for pos in positions:
         ap_raw      = ((clean_act[pos] - corr_act[pos]) * corr_grad[pos]).sum()
         ap_recovery = ap_raw / (d_clean - d_corrupted)
         on_cell(L, sub, pos, {"ap_recovery": float(ap_recovery)})
7. Build PatchingResult(mode="approx", cells=..., direction="denoise", ...)
```

### Algorithm (noise)

Same structure but: base = clean, backward from `d_clean`, from = corrupted activations. Final step:

```
     ap_raw = ((corr_act[pos] - clean_act[pos]) * clean_grad[pos]).sum()
     ap_recovery = 1 + ap_raw / (d_clean - d_corrupted)
```

`ap_raw` here approximates `ΔLD` when corrupted activations are patched into the clean run — expected negative for cells that carry clean-specific signal. The `1 +` shift converts the signed delta into a recovery fraction that matches exact AP's `logit_diff_recovery` in noise direction (close to 1 = patching didn't destroy; close to 0 = patching fully destroyed clean behavior).

### Hook shape vs `_capture_residual_stream`

`_capture_residual_stream` currently calls `.detach()` on captured tensors (safe with `torch.no_grad()`). For AP we need a different capture helper that keeps the computation graph alive. New internal helper:

```python
def _capture_residual_stream_with_grad(
    model, tokenizer, prompt: str,
    sublayers: Tuple[str, ...], layers: Optional[List[int]],
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], torch.Tensor, List[str]]:
    """Same as _capture_residual_stream but tensors retain grad + graph.
    Returns (captured_states, output_logits, prompt_tokens).
    """
```

Hooks register via `register_forward_hook`, the captured tensor has `.retain_grad()` called on it so `.grad` is populated after backward. No `.detach()`. Model must be in `.eval()` but with `torch.enable_grad()` — we can't rely on global state; caller-side responsibility.

### Validation rules (superset of `activation_patch`)

| Condition | Error |
|---|---|
| All Phase 3 validation conditions (same-length, direction, sublayer, position range, non-empty) | as before |
| `correct_token_id` or `incorrect_token_id` is `None` | `ValueError("attribution_patch requires correct_token_id and incorrect_token_id")` |
| `abs(d_clean - d_corrupted) < 1e-6` | `ValueError("clean and corrupted baselines have identical logit_diff; AP would divide by zero")` |

### Quantized models

Same warning as exact AP. Gradient flow through bitsandbytes 4-bit/8-bit linear layers is supported (LoRA uses the same path), so AP runs; precision is reduced.

### Memory

One backward pass on full forward activations. For TinyLlama (22 layers × 2 sublayers × seq_len ≈ 6 × 2048 hidden × f32): ≈ 2 MB of activations + same for gradients + model's own autograd tape (~10–20 MB for 1.1B params). Total well under 500 MB added overhead on top of forward-pass footprint. Fine for RTX 2080's 8 GB.

---

## 3. WS route changes

**No new route.** `/ws/sessions/{name}/activation-patching` handler branches on `cfg.mode`:

```python
mode = cfg.get("mode", "exact")
if mode not in ("exact", "approx"):
    send error; return
if mode == "exact":
    result = activation_patch(...)
else:
    # Resolve token IDs server-side (auto or manual) BEFORE the call
    correct_id, incorrect_id = _resolve_token_pair(cfg, clean_logits, corrupted_logits)
    result = attribution_patch(..., correct_token_id=correct_id, incorrect_token_id=incorrect_id)
```

### Frame changes

`data` frame gains optional `ap_recovery` field; `patched_logits` becomes optional. `complete` frame's `summary` gains `mode`.

```json
// exact (unchanged)
{"type": "data", "layer": 5, "sublayer": "attn", "position": 3,
 "patched_logits": {"shape": [32000], "b64": "..."}}

// approx (new)
{"type": "data", "layer": 5, "sublayer": "attn", "position": 3,
 "ap_recovery": 0.83}
```

Frame order same as exact: status → data (N, streamed in layer-major/position-minor order) → baselines → complete. Since approx has no real loop progress, the backend iterates the cells dict post-backward and emits frames in deterministic order. User sees the heatmap fill effectively all at once (sub-second for TinyLlama); that's fine.

### Token-pair resolution

Unchanged surface — `auto` mode argmaxes on the backend, `manual` mode tokenizes + validates single-token. In approx mode these IDs are now **required** (backend errors out if resolution fails).

---

## 4. Frontend: `PatchingControls.tsx`

Add a **mode radio** below the existing direction radio:

```
mode       (●) exact        ( ) approx (gradient AP, fast)
```

`exact` is default. State:

```tsx
const [mode, setMode] = useState<"exact" | "approx">("exact");
```

Extends `PatchingState` interface in the same file:

```tsx
export interface PatchingState {
  /* existing fields */
  mode: "exact" | "approx";                // new
}
export const DEFAULT_PATCHING_STATE: PatchingState = {
  /* existing defaults */
  mode: "exact",                           // new
};
```

`ProbePanel.handleRun()` forwards the mode:

```tsx
const cfg = {
  clean_prompt, corrupted_prompt, direction, measurement_position,
  mode: patchingState.mode,                // new
  ...(tokenPairMode === "manual" && { correct_token, incorrect_token }),
};
```

When `mode === "approx"` and `tokenPairMode === "auto"`, UI shows a tiny info line under the radio: `"auto-pick uses clean argmax; switch to manual for a specific target"` — so users understand AP's scalar-metric constraint. No UI disabling; backend picks sensibly.

---

## 5. Frontend: `ActivationPatchingHeatmap.tsx`

### Detect mode

Read from the complete frame (already part of `result.data`):

```tsx
const completeFrame = result.data.find((m): m is PatchingCompleteData => m.type === "complete");
const mode: "exact" | "approx" = completeFrame?.summary.mode ?? "exact";
```

### Approx-mode rendering

- **Metric dropdown is hidden** (there's only one metric — AP recovery).
- **Heading** replaces the metric-selector area: `"Attribution patching (∂logit_diff / ∂activation)"`.
- **Cell value** comes directly from `cell.ap_recovery` (no `decodeLogits` call, no math on the client).
- **Color scale** reuses the existing `logit_diff_recovery` interpolator (`d3.interpolatePiYG`, fixed domain `[-0.5, 1.0]`) — same as exact default, so side-by-side comparison is apples-to-apples.
- **Click-to-pin card** shows only the scalar + a reminder: `"AP score is a first-order approximation; run exact mode to confirm"`.

### Exact-mode rendering

Unchanged.

### Shared plumbing

Row computation, column layout, SVG sizing, legend rendering, ExportButtons — all unchanged. CSV export in approx mode has one metric column (`ap_recovery`) instead of four; JSON export includes `mode`.

---

## 6. Data types

### `testing/gui/frontend/src/types/api.ts`

```tsx
export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer: "attn" | "ffn";
  position: number;
  patched_logits?: EncodedTensor;            // present only in exact mode
  ap_recovery?: number;                      // present only in approx mode
}

export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
    mode: "exact" | "approx";                // new
  };
}
```

Both `patched_logits` and `ap_recovery` are optional in the type, but the backend contract guarantees exactly one is set per cell based on mode. Frontend checks `mode` before indexing.

### No new store reducers

Patching reducers (`setPendingResult` / `updatePendingResult` / `finalizePendingResult`) already accumulate arbitrary frames into `ProbeResult.data`.

---

## 7. Testing strategy

### Python: `testing/tests/test_probe_attribution_patch.py` (new file)

| Test | Covers |
|---|---|
| `TestValidation::test_missing_token_ids_raises` | both IDs required |
| `TestValidation::test_identical_baselines_raises` | divide-by-zero guard |
| `TestValidation::test_inherits_activation_patch_validation` | same-length, direction, sublayer, position range |
| `TestCallbackFiresPerCell` | mock model; `on_cell` called `L × S × P` times, each with `ap_recovery` |
| `TestDirectionNoiseFlipsSign` | mock gradients; noise direction's `1 −` shift applied |
| `TestPositionsSubset` / `TestLayersSubset` | scoping preserved from exact AP |
| `TestGradFlowPreserved` | small model; assert every `(L, sub)` tensor has non-zero `.grad` after backward |
| `TestApproxVsExactCorrelates` (integration, skipif-guarded) | TinyLlama, "capital of France vs Italy", Spearman rank correlation between `activation_patch` and `attribution_patch` cell scores ≥ 0.5 |

Integration test is the load-bearing one — it validates that the AP approximation actually tracks exact AP. Threshold 0.5 picked conservatively: Kramár et al. 2024 report 0.7–0.9 on well-behaved tasks, but residual-stream AP on a 1.1B model at short context can be noisier. If the test passes comfortably at 0.5, a follow-up can tighten it.

### Frontend: no new Vitest tests needed

AP approx rendering adds no new pure functions (the math runs on the backend). Existing `patchingMetrics.test.ts` already covers exact-mode metrics. Heatmap mode-switching behavior is covered by Playwright.

### Playwright: extend `smoke.spec.ts`

One new test:

- Seed an `activation-patching` result with `mode: "approx"` + a handful of `ap_recovery` cells via the existing fixture-import path.
- Assert the heatmap heading reads "Attribution patching" and the metric dropdown is **not** visible.
- Fixture reuses `activation-patching.json` shape with `ap_recovery` cells + `mode: "approx"` in the complete frame.

Total suite: 10 existing + 1 new = 11 tests.

### Type-check tiers

Before commit: pyright 0/0/0 on `probe.py`, `test_probe_attribution_patch.py`, `routes/probes.py`. tsc clean. Before merge: Playwright 11/11, 22 existing probe unit tests + N new AP tests all green.

---

## 8. Explicit non-goals

1. **Edge attribution** (attribution across residual-stream edges, as in Syed et al. 2023). Future if circuit-extraction is ever in scope.
2. **Per-head AP** — requires new hooks inside `self_attn.o_proj`; residual-stream hooks can't see per-head gradients.
3. **Multi-target AP** — single `(correct, incorrect)` pair only.
4. **Clamping gradient precision** in fp16 models — AP runs but may be noisier. No mitigation this phase.
5. **Streaming real progress** — approx mode's "streaming" is post-hoc chunking for UI parity, not real compute progress.
6. **Side-by-side exact/approx viz** — spec allows it (shared component + shared interpolator), but no new UI to run both simultaneously. Users toggle mode and rerun.
7. **`PatchingResult.mode` migration for existing exact callers** — defaulted to `"exact"` via dataclass default; no call-site churn.

---

## 9. File map

| File | Change |
|---|---|
| `testing/llm_surgeon/probe.py` | **+** `attribution_patch()`, `_capture_residual_stream_with_grad()` helper, `mode` field on `PatchingResult` (default `"exact"`) |
| `testing/gui/backend/routes/probes.py` | **~** `/activation-patching` handler: branch on `cfg.mode`; token-pair resolution before `attribution_patch` call; `ap_recovery` in data frames |
| `testing/gui/frontend/src/types/api.ts` | **~** `PatchingCellData.ap_recovery?`, `PatchingCompleteData.summary.mode` |
| `testing/gui/frontend/src/components/PatchingControls.tsx` | **~** mode radio + `PatchingState.mode` field + `DEFAULT_PATCHING_STATE.mode` |
| `testing/gui/frontend/src/components/ProbePanel.tsx` | **~** forward `mode` in cfg payload |
| `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx` | **~** mode-branch: hide metric dropdown in approx, read `ap_recovery` directly, different heading |
| `testing/tests/test_probe_attribution_patch.py` | **new** — unit + TinyLlama correlation integration |
| `testing/gui/frontend/tests/e2e/smoke.spec.ts` | **+** one approx-mode heatmap test |
| `testing/gui/frontend/tests/e2e/fixtures/activation-patching-approx.json` | **new** — sibling fixture with `mode: "approx"` + `ap_recovery` cells |

Roadmap memory update in the last plan task, same pattern as Phases 1 / 2 / 3.
