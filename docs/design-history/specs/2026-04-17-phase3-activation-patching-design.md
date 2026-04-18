# Phase 3 — Activation Patching Design

**Date:** 2026-04-17
**Roadmap:** `llm_surgeon/project_llm_surgeon_roadmap.md` (Phase 3)
**Goal:** add causal attribution via the clean/corrupted counterfactual pattern (Heimersheim & Nanda, 2024) — upgrade `probe.py`'s interventions from correlational ("patch X, see what changes") to causal ("point (L, sub, pos) is *sufficient* / *necessary* for this behavior").

**Non-goal for this phase:** attribution patching (gradient approximation). Deferred to Phase 3.5; adding it later means one more backend route and a new op name, reusing the heatmap.

---

## 1. Architecture overview

Given two prompts (`clean`, `corrupted`) that tokenize to the **same length**, run three forward-pass families on a single model:

1. **Clean pass** — capture residual-stream states at every `(layer, sublayer, position)`. Record output logits at the measurement position as the "clean baseline."
2. **Corrupted pass** — same, for the corrupted prompt. "Corrupted baseline."
3. **Patching loop** — for each `(L, sub, pos)` triple, run one forward pass on the *base* prompt with that single point's activation replaced by the *other* prompt's cached state at the same triple. Capture the output logits at the measurement position.

**Direction** picks which prompt is the base:

- `denoise` (default): base = corrupted, patches from clean → high recovery = **sufficient** for clean behavior.
- `noise`: base = clean, patches from corrupted → low recovery = **necessary** for clean behavior.

The same `logit_diff_recovery = (Δ_patched − Δ_corrupted) / (Δ_clean − Δ_corrupted)` formula applies in both directions — the *interpretation* of bright vs dark flips, not the math. In denoise, bright cells (recovery ≈ 1) mean "clean-like"; in noise, bright cells mean "patching didn't degrade" and dark cells (recovery ≈ 0) mean "patch successfully knocked this out of circuit." The frontend heatmap does not flip color scales per direction — users read the meaning from the direction toggle in context.

**Per-cell metrics** are computed client-side from three logit vectors (clean baseline, corrupted baseline, patched). Backend streams only the patched logits per cell; frontend does the metric math so the metric dropdown switches instantly without a backend round-trip.

**User flow:** ProbePanel → operation dropdown → `activation-patching` → conditional `PatchingControls` renders → user enters clean + corrupted prompts, picks direction → Run → heatmap fills live as per-cell frames stream in → click cell to pin per-cell logit top-k + metric value.

---

## 2. Backend: `probe.py::activation_patch()`

Extends `probe.py` (does not create a new module — all required primitives already live there).

```python
@dataclass
class PatchingResult:
    cells: List[Dict]                         # per (layer, sublayer, position) frame
    clean_baseline_logits: torch.Tensor       # (vocab_size,) at measurement_position
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: List[str]
    prompt_tokens_corrupted: List[str]
    direction: str                            # "denoise" | "noise"
    measurement_position: int                 # resolved absolute index

def activation_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    direction: str = "denoise",               # "denoise" | "noise"
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,    # None = all
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,       # None = all
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult: ...
```

### Algorithm (denoise — noise is the mirror, only the base prompt flips)

```
1. tokenize both prompts; assert len(clean_ids) == len(corr_ids)
   → ValueError("prompts must tokenize to same length (clean=N, corrupted=M)")
2. captured_clean, _ = _capture_residual_stream(model, tok, clean_prompt,
                           sublayers=sublayers, layers=layers)
3. captured_corr,  _ = _capture_residual_stream(model, tok, corrupted_prompt,
                           sublayers=sublayers, layers=layers)
4. clean_base_logits     = forward(clean_prompt).logits[:, measurement_position]
   corrupted_base_logits = forward(corrupted_prompt).logits[:, measurement_position]
5. for (L, sub) in sorted(captured_clean.keys()):          # layer-major
       for pos in positions:                               # position-minor
           patch_vec = captured_clean[(L, sub)][pos]       # shape: (d_model,)
           iv = Intervention(
               layer=L, sublayer=sub,
               fn=_make_position_patch(pos, patch_vec),
           )
           result = intervene(model, tok, corrupted_prompt, [iv])
           patched = result.output_logits[measurement_position]
           cell = {"layer": L, "sublayer": sub, "position": pos,
                   "patched_logits": patched.cpu()}
           if on_cell: on_cell(L, sub, pos, cell)
           cells.append(cell)
```

### New helper: `_make_position_patch(pos, vec)`

One genuinely new primitive. `ops.replace(tensor)` overwrites the entire `(seq_len, d_model)` hidden state — we need position-scoped replace:

```python
def _make_position_patch(pos: int, clean_vec: torch.Tensor) -> Callable:
    def fn(h, _layer_idx):
        out = h.clone()
        out[pos] = clean_vec.to(device=h.device, dtype=h.dtype)
        return out
    return _Op(fn, f"patch_pos({pos})")
```

Lives in `probe.py` as a module-level private helper (not exposed under `ops`).

### Validation rules

| Condition | Error |
|---|---|
| `len(clean_ids) != len(corr_ids)` | `ValueError("prompts must tokenize to same length (clean={a}, corrupted={b})")` |
| `direction not in {"denoise", "noise"}` | `ValueError("direction must be 'denoise' or 'noise'")` |
| `not set(sublayers).issubset({"attn", "ffn"})` | `ValueError("sublayers must be subset of {'attn','ffn'}")` |
| `measurement_position` out of `[-seq_len, seq_len-1]` | `IndexError` |
| Empty clean or corrupted prompt | `ValueError("prompt cannot be empty")` |

### Quantized models

Mirrors `benchmark.perplexity()`: emit `warnings.warn(...)` if `getattr(model, "hf_quantizer", None)` is set; still run. Patching through bitsandbytes quant layers works, but round-trips through dequant — slower, slightly less precise.

---

## 3. WS route: `/sessions/{name}/activation-patching`

Append new handler to `testing/gui/backend/routes/probes.py` using the same pattern as `/logit-lens`: `ws.accept()` → read config → `ensure_pytorch` → `async with info.lock` wrapping `loop.run_in_executor`.

### Config (client → server, first JSON frame)

```json
{
  "clean_prompt": "The capital of France is",
  "corrupted_prompt": "The capital of Italy is",
  "direction": "denoise",
  "measurement_position": -1,
  "positions": null,
  "sublayers": ["attn", "ffn"],
  "layers": null,
  "correct_token": "Paris",          // optional; only if manual token-pair mode
  "incorrect_token": "Rome"          // optional
}
```

### Frame types (server → client)

| Type | When | Payload |
|---|---|---|
| `status` | setup progress | `{type, message}` |
| `baselines` | once, before loop | `{type, clean_logits: EncodedTensor, corrupted_logits: EncodedTensor, prompt_tokens_clean, prompt_tokens_corrupted, measurement_position, correct_token_id?, incorrect_token_id?}` |
| `data` | once per cell | `{type, layer, original_layer?, sublayer, position, patched_logits: EncodedTensor}` |
| `complete` | end of loop | `{type, summary: {num_cells, direction, measurement_position}}` |
| `error` | validation / runtime | `{type, message}` |

`EncodedTensor` reuses the existing `_encode_hidden_state` helper: `{shape: [vocab_size], b64: <base64 float32 bytes>}`.

`original_layer` mirrors the logit-lens convention — populated via `info._layer_map[layer]` for surgically-compressed sessions.

### Concurrency

Single session, one `info.lock`. No cross-session locking (unlike `compare-logit-lens`). `dirty` triggers the same "waiting for export" hint message before lock acquisition.

### Streaming cost

TinyLlama vocab = 32k × f32 = 128 KB per cell. 22 layers × 2 sublayers × 10 positions = 440 cells → ~56 MB raw. Base64 adds ~33%. Acceptable for local dev; future top-k-only optimization is possible but not needed now.

### Token-pair resolution

- `auto` mode (config omits `correct_token`/`incorrect_token`): backend computes `argmax(clean_baseline_logits)` and `argmax(corrupted_baseline_logits)`, includes both IDs in the `baselines` frame.
- `manual` mode: backend tokenizes both strings; if either produces more than one token, error out with `ValueError("correct_token/incorrect_token must tokenize to exactly one token")`. Echo resolved IDs in the `baselines` frame.

---

## 4. Frontend: `PatchingControls.tsx`

**File:** `testing/gui/frontend/src/components/PatchingControls.tsx` (new, ~180 LOC).

Conditional form rendered by `ProbePanel` when `operation === "activation-patching"`. Owns patching-only state. Does **not** own the Run button or Stop/Cancel — those remain in `ProbePanel`.

### State (panel-local React `useState`, not Zustand)

```tsx
const [cleanPrompt, setCleanPrompt]     = useState("");
const [corruptedPrompt, setCorruptedPrompt] = useState("");
const [direction, setDirection]         = useState<"denoise" | "noise">("denoise");
const [measurementPos, setMeasurementPos] = useState<number>(-1);
const [tokenPairMode, setTokenPairMode] = useState<"auto" | "manual">("auto");
const [manualCorrect, setManualCorrect] = useState("");
const [manualIncorrect, setManualIncorrect] = useState("");
const [cleanTokens, setCleanTokens]     = useState<number | null>(null);
const [corrTokens, setCorrTokens]       = useState<number | null>(null);
```

Patching state is local (not persisted) because it's a tight interactive loop and doesn't need round-trip through experiment import/export. Users can recall prompts from existing results.

### Layout

```
┌─────────────────────────────────────────────────────────┐
│ Clean prompt   [textarea, 2 rows]                       │
│                                    [6 tokens]           │
│                                                         │
│ Corrupted prompt [textarea, 2 rows]                     │
│                                    [6 tokens ✓]         │  green ✓ if match
│                                    [7 tokens ✗ (6 vs 7)]│  red ✗ if mismatch
│                                                         │
│ direction  (●) denoise   ( ) noise                      │
│ measure @  [-1]  (-1 = last token)                      │
│                                                         │
│ target    (●) auto-pick   ( ) manual                    │
│   correct:   [Paris]    incorrect: [Rome]               │
│              (disabled when auto-pick)                  │
└─────────────────────────────────────────────────────────┘
```

### Length-match indicator

Debounced tokenize POST to `/api/sessions/{name}/tokenize` (endpoint already exists — used by context-window budget display in ProbePanel). 250 ms debounce. `AbortController` cancels stale requests. Green ✓ when lengths equal and both > 0; red ✗ + "lengths differ (N vs M)" when not. `ProbePanel` Run button gets `disabled` when `operation === "activation-patching"` and lengths don't match.

### Wiring

`PatchingControls` takes props for its state + setters (controlled component pattern). `ProbePanel` owns the state slice, renders `<PatchingControls ... />` when `operation === "activation-patching"`. In `ProbePanel.handleRun()`:

```tsx
if (operation === "activation-patching") {
  const cfg = {
    clean_prompt: cleanPrompt,
    corrupted_prompt: corruptedPrompt,
    direction,
    measurement_position: measurementPos,
    ...(tokenPairMode === "manual" && {
      correct_token: manualCorrect,
      incorrect_token: manualIncorrect,
    }),
  };
  const path = `/ws/sessions/${targetSession}/activation-patching`;
  connect(resultId, path, cfg, makeWsHandlers(resultId), targetSession);
  return;
}
```

### Integration with existing ProbePanel features

- **Fan-out / A-B / sweeps** are disabled for `activation-patching`. Add to the same disable-conditions that already hide sampling knobs for `logit-lens`.
- **Prompt library:** two `PromptLibraryBar` instances, one per textarea.

---

## 5. Frontend: `ActivationPatchingHeatmap.tsx`

**File:** `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx` (new, ~300 LOC). Template: `LogitLensHeatmap.tsx`.

### Inputs

A `ProbeResult` where `operation === "activation-patching"`. `result.data` is:

1. One `baselines` frame (first).
2. N `data` frames (one per `(layer, sublayer, position)` cell, layer-major streaming order).
3. One `complete` frame.

### Layout

- Rows = `(layer, sublayer)` — e.g. `L0.attn`, `L0.ffn`, `L1.attn`, … — matching logit-lens row labeling. Surgically-compressed sessions show `L3(←5).ffn` notation, same as logit-lens.
- Cols = positions patched. Column count = `len(positions)` (user-selected subset) or `seq_len` (default).
- Cell fill = selected metric, color-mapped with per-metric interpolator.

### Metric selector (dropdown, mirrors `LogitLensHeatmap` pattern)

| Metric key | Formula | Domain | Interpolator |
|---|---|---|---|
| `logit_diff_recovery` (default) | `(Δ_patched − Δ_corrupted) / (Δ_clean − Δ_corrupted)` where `Δ = logit(correct) − logit(incorrect)` | typically [0, 1], can go negative | `d3.interpolatePiYG` (signed) |
| `kl_from_clean` | `KL(softmax(patched) ‖ softmax(clean))` | [0, ∞) | `d3.interpolateInferno` (reverse) |
| `top1_match` | `argmax(patched) == argmax(clean)` | {0, 1} | binary dark/light |
| `prob_delta` | `p_patched(top1_clean) − p_corrupted(top1_clean)` | [-1, 1] | `d3.interpolatePiYG` |

### Client-side metric utilities

**New file:** `testing/gui/frontend/src/utils/patchingMetrics.ts`.

```typescript
export function decodeLogits(b64: EncodedTensor): Float32Array
export function logitDiffRecovery(
  patched: Float32Array, clean: Float32Array, corrupted: Float32Array,
  correctId: number, incorrectId: number,
): number
export function klFromClean(patched: Float32Array, clean: Float32Array): number
export function top1Match(patched: Float32Array, clean: Float32Array): boolean
export function probDelta(patched: Float32Array, corrupted: Float32Array, cleanTopId: number): number
```

`decodeLogits` reuses the existing EncodedTensor decode path used by hidden-state pin cards.

### Token-pair resolution (frontend view)

- `auto` mode: frontend reads `correct_token_id` / `incorrect_token_id` from the `baselines` frame (backend argmax'd them).
- `manual` mode: same — backend tokenizes the user's strings and echoes the IDs back. Frontend doesn't re-tokenize.

### Click-to-pin card

Shows:
- Header: `L{layer}.{sublayer} pos {pos}`
- Metric value display: `logit_diff_recovery = 0.83`
- Top-5 from `patched` distribution, decoded via `tokenizer.decode` on token IDs.
- Top-5 from `clean` baseline for visual comparison.

Does **not** show hidden-state bar strip or PCA — we don't stream hidden states for patching (logits are the streamed payload). Future enhancement: `include_hidden_state` config flag.

### Export

- **CSV:** flat table `(layer, sublayer, position, logit_diff_recovery, kl_from_clean, top1_match, prob_delta)`, one row per cell, all four metrics pre-computed.
- **JSON:** full result including raw b64 logits so re-import replays identically.
- **SVG:** `getSVG={() => svgRef.current}` via `ExportButtons`.

---

## 6. Data types + store wiring

### `testing/gui/frontend/src/types/api.ts`

```typescript
export type ProbeOperation =
  | "logit-lens"
  | "influence"
  | "attention"
  | "residual-norms"
  | "generate"
  | "activation-patching";       // new

export interface PatchingBaselinesData {
  type: "baselines";
  clean_logits: EncodedTensor;
  corrupted_logits: EncodedTensor;
  prompt_tokens_clean: string[];
  prompt_tokens_corrupted: string[];
  measurement_position: number;
  correct_token_id?: number;
  incorrect_token_id?: number;
}

export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer: "attn" | "ffn";
  position: number;
  patched_logits: EncodedTensor;
}

export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
  };
}

export type WsMessage =
  | /* existing variants */
  | PatchingBaselinesData
  | PatchingCellData
  | PatchingCompleteData;
```

### Store

No new reducers. Existing `setPendingResult` / `updatePendingResult` / `finalizePendingResult` handle patching frames — `ProbeResult.data` accumulates messages regardless of op.

### `VisualizationArea.tsx`

Extend the op-to-component switch to route `activation-patching` results to `ActivationPatchingHeatmap`.

---

## 7. Testing strategy

Unit + one real-TinyLlama integration, per Q8 (matches Phase 2).

### New Python test file: `testing/tests/test_probe_activation_patch.py`

| Test class / test | Covers |
|---|---|
| `TestValidation::test_mismatched_lengths_raises` | same-length enforcement, error message includes both counts |
| `TestValidation::test_empty_prompt_raises` | empty-string on either side |
| `TestValidation::test_bad_direction_raises` | `direction="wobble"` |
| `TestValidation::test_bad_sublayer_raises` | `sublayers=("mlp",)` |
| `TestValidation::test_measurement_pos_out_of_range` | out-of-bounds `measurement_position` |
| `TestPositionPatchOp::test_only_replaces_target_pos` | `_make_position_patch(3, vec)` leaves positions ≠ 3 untouched |
| `TestPositionPatchOp::test_preserves_dtype` | fp16 in → fp16 out |
| `TestCallbackFiresPerCell` | mock model; `on_cell` called `num_layers × num_sublayers × num_positions` times |
| `TestDirectionSwapsBase` | denoise calls `intervene(..., corrupted_prompt, ...)`; noise calls on clean |
| `TestPositionsSubset` | `positions=[2,4]` loop runs only 2 cells per (L, sub) |
| `TestLayersSubset` | `layers=[5,10]` restricts loop to those layers |
| `TestQuantizedModelWarns` | mock `hf_quantizer` → `warnings.warn` fires; run proceeds |
| `TestActivationPatchIntegration::test_tinyllama_capital_swap` | real TinyLlama, "The capital of France is" vs "The capital of Italy is"; late-layer recovery > early-layer + 0.1 |

Integration test gated by `@pytest.mark.skipif` on missing TinyLlama, same pattern as existing fixtures. Expected runtime: ~10–30 s on RTX 2080. Must be run with `dangerouslyDisableSandbox: true` (GPU access).

### Frontend unit tests: `testing/gui/frontend/tests/unit/patchingMetrics.test.ts` (Vitest)

| Suite | Tests |
|---|---|
| `decodeLogits` | round-trips f32 bytes through b64 |
| `logitDiffRecovery` | returns 1.0 when patched == clean; 0.0 when patched == corrupted; negative when patching makes it worse |
| `klFromClean` | returns 0 when distributions identical; handles zero-prob bins (xlogy semantics) |
| `top1Match` / `probDelta` | basic identity + signed cases |

### Playwright smoke suite extension

Add one test to `testing/gui/frontend/tests/e2e/smoke.spec.ts`:
- Seeds a mock `activation-patching` result via experiment-import using an extended fixture JSON.
- Asserts the heatmap SVG renders with expected row/col counts.
- Switches metric dropdown through all four options, asserts no console errors (using the existing `isBackendlessNoise()` filter).

No backend required — tests component wiring, not streaming.

### Type-check tiers

Before commit: pyright 0/0/0 and tsc clean. Before merge: Playwright 10/10 (9 existing + 1 new).

---

## 8. Explicit non-goals (Phase 3 YAGNI boundary)

**Out of scope for this phase:**

1. **Attribution patching** (gradient-based approximation). Phase 3.5 if scale demands it.
2. **Per-head granularity** — requires new hooks inside `self_attn`; current residual-stream hooks can't see per-head outputs.
3. **Cross-prompt patching** (clean and corrupted of different lengths).
4. **Multi-token target strings** for manual mode. Single-token pairs only.
5. **Saving patched checkpoints.** AP is analysis-only; model weights are never modified.
6. **Hidden-state streaming in patching frames.** Only logits are streamed; pin card shows logit top-k + metric only.
7. **Fan-out / A/B / sweeps for AP.** Disabled in UI.
8. **Persistence of patching panel state across reloads.** Local React state only; users can recall from existing results.
9. **WS end-to-end test.** Unit + TinyLlama integration cover the algorithm. Adding WS test infra is a separate effort (would cover existing untested routes too).
10. **Multiple simultaneous token pairs** for signed logit-diff.

**In scope — explicit commitments:**

- LLaMA-style models (TinyLlama, OpenLLaMA 3B, anything `intervene()` already supports).
- Surgically-compressed sessions (uses `info._layer_map` for original-layer labeling).
- Quantized models warn but run.
- Zero diagnostics on pyright + tsc.

---

## 9. File map (what changes)

| File | Change |
|---|---|
| `testing/llm_surgeon/probe.py` | **+** `activation_patch()`, `PatchingResult`, `_make_position_patch()` |
| `testing/gui/backend/routes/probes.py` | **+** `/sessions/{name}/activation-patching` WS handler |
| `testing/gui/frontend/src/types/api.ts` | **+** `ProbeOperation` extension, `PatchingBaselines/Cell/CompleteData`, `WsMessage` union entries |
| `testing/gui/frontend/src/components/PatchingControls.tsx` | **new** — conditional patching form |
| `testing/gui/frontend/src/components/ProbePanel.tsx` | **+** op option, conditional render of `PatchingControls`, `handleRun` branch, disable fan-out/A-B for AP |
| `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx` | **new** — heatmap with metric selector + pin card |
| `testing/gui/frontend/src/components/VisualizationArea.tsx` | **+** op → component dispatch entry |
| `testing/gui/frontend/src/utils/patchingMetrics.ts` | **new** — pure-function metric helpers |
| `testing/tests/test_probe_activation_patch.py` | **new** — unit + integration tests |
| `testing/gui/frontend/tests/unit/patchingMetrics.test.ts` | **new** — frontend metric-fn tests |
| `testing/gui/frontend/tests/e2e/smoke.spec.ts` | **+** one mock-fixture heatmap-render test |
| `testing/gui/frontend/tests/e2e/fixtures/sample.json` | **+** or sibling — patching-result fixture |

Roadmap memory update happens in the last plan task (same pattern as Phases 1 & 2).
