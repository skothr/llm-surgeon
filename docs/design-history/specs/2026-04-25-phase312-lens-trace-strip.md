# Phase 3.12 — Bulk Lens Grid + Lens-Trace Strip on AP Heatmap

**Date:** 2026-04-25
**Status:** approved (autonomy directive)

## 1. Problem

Phase 3.11 shipped per-cell residual lens decode behind a pin click. Useful, but cells are tiny and the user sees lens info only for one cell at a time. The natural next move is **a vertical strip rendered alongside the AP heatmap that shows the top-1 prediction at every layer/sublayer for the currently-selected column**.

This makes one visualization read as two columns:
- **Left:** existing AP heatmap (causal importance — "patching here recovers X").
- **Right:** lens trace strip ("…and the residual stream at this column reads as 'the', 'the', 'Paris', 'Paris', 'Paris'" down the layers).

Together: the user sees both **where** computation matters (heatmap intensity) and **what** the model is computing at that column (lens trace), in a single glance.

## 2. Goals

- One bulk decode endpoint that returns top-1 lens decode for every (layer, sublayer, position).
- Frontend hook fires once per AP result (cached, AbortController on arg change).
- New `LensTraceStrip` component embedded inside `ActivationPatchingHeatmap`, layout: heatmap (left, flex 1) + strip (right, fixed width).
- Strip shows `(layer × sublayer)` rows, each labeled with `L{n}.{sublayer}` and the top-1 token at the current column.
- Position selector: a `<select>` at the strip's top, default = baseline measurement_position.
- No changes to other AP panels in V1 (per-head, per-neuron, edge, circuit) — they get the strip in follow-up phases as their UX needs differ.

## 3. Non-goals

- No top-3 / bottom-3 in the strip — top-1 only. Pin-card already shows top-10/bottom-10 for full detail.
- No final-norm-toggle, no diff-vs-clean lens (those are Phase 3.13+).
- No client-side caching shared with `useResidualDecode`. Pin-card still does its own single-cell fetch (pin clicks remain per-cell). Sharing the cache is a Phase 3.13 cleanup.
- No bulk decode for arbitrary sublayer set. V1 always returns both `attn` and `ffn`.

## 4. Backend

**Route:** `POST /api/sessions/{name}/decode-residual-grid`

**Request:**
```python
class DecodeResidualGridRequest(BaseModel):
    prompt: str
    top_k: int = 1  # only top-K promoted tokens per cell
```

**Response:**
```json
{
  "cells": [
    {"layer": 0, "sublayer": "attn", "position": 0, "tokens": [{"token": "the", "logit": 5.2}]},
    {"layer": 0, "sublayer": "ffn", "position": 0, "tokens": [{"token": "the", "logit": 5.4}]},
    ...
  ],
  "prompt_tokens": ["The", "Eiffel", ...],
  "num_layers": 22
}
```

**Implementation** (in `gui/backend/routes/sessions.py`, after `decode_residual`):

```python
class DecodeResidualGridRequest(BaseModel):
    prompt: str
    top_k: int = 1


_DECODE_RESIDUAL_GRID_TOPK_MAX = 5
_DECODE_RESIDUAL_GRID_MAX_TOKENS = 200  # safety cap on prompt length


@router.post("/sessions/{name}/decode-residual-grid")
async def decode_residual_grid(name: str, req: DecodeResidualGridRequest):
    """Bulk top-K logit-lens decode at every (layer, sublayer, position) point.

    Used by the AP heatmap's lens-trace strip (Phase 3.12). Single forward
    pass + N_layers * 2 sublayer projections. Returns one entry per
    (layer, sublayer, position) cell.
    """
    import torch
    from llm_surgeon.probe import _capture_residual_stream, _project_to_logits

    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    if info.model is None:
        raise HTTPException(500, "Session has no PyTorch model loaded")
    if info.tokenizer is None:
        raise HTTPException(500, "Session has no tokenizer loaded")

    model = info.model
    tok = info.tokenizer
    num_layers = len(model.model.layers)

    def _compute() -> dict:
        captured, prompt_tokens = _capture_residual_stream(
            model, tok, req.prompt, sublayers=("attn", "ffn"),
        )
        seq_len = len(prompt_tokens)
        if seq_len > _DECODE_RESIDUAL_GRID_MAX_TOKENS:
            raise HTTPException(
                413,
                f"prompt too long ({seq_len} tokens; max {_DECODE_RESIDUAL_GRID_MAX_TOKENS})",
            )

        k = max(1, min(req.top_k, _DECODE_RESIDUAL_GRID_TOPK_MAX))
        cells = []
        with torch.no_grad():
            for (layer_idx, sublayer), hidden in sorted(captured.items()):
                logits = _project_to_logits(model, hidden)  # (seq_len, vocab)
                vocab_size = int(logits.shape[1])
                k_eff = min(k, vocab_size)
                top_vals, top_ids = torch.topk(logits, k_eff, dim=-1, largest=True)
                # top_vals/top_ids: (seq_len, k_eff)
                for pos in range(seq_len):
                    tokens = [
                        {
                            "token": tok.decode([int(top_ids[pos, j])], skip_special_tokens=False),
                            "logit": float(top_vals[pos, j]),
                        }
                        for j in range(k_eff)
                    ]
                    cells.append({
                        "layer": layer_idx,
                        "sublayer": sublayer,
                        "position": pos,
                        "tokens": tokens,
                    })

        return {
            "cells": cells,
            "prompt_tokens": prompt_tokens,
            "num_layers": num_layers,
        }

    return await asyncio.to_thread(_compute)
```

**Validation:**
- 404 if session missing.
- 500 if model or tokenizer not loaded.
- 413 if prompt > 200 tokens (cap is generous; protects against accidental MB-sized responses).
- `top_k` clamped to [1, 5] (anything more than ~3 is wasteful for an at-a-glance strip).

**Cost on TinyLlama:** 22 layers × 2 sublayers × 5 positions × 1 top-token ≈ 220 entries, ~5KB JSON. Forward pass: ~50ms fp16.

## 5. Frontend

### 5.1 Hook (new): `utils/useResidualGrid.ts`

```typescript
export type ResidualGridToken = { token: string; logit: number };
export type ResidualGridCell = {
  layer: number;
  sublayer: "attn" | "ffn";
  position: number;
  tokens: ResidualGridToken[];
};
export type ResidualGridResponse = {
  cells: ResidualGridCell[];
  prompt_tokens: string[];
  num_layers: number;
};

export function useResidualGrid(
  sessionName: string | undefined,
  prompt: string | undefined,
  topK: number = 1,
): { data: ResidualGridResponse | null; error: string | null; loading: boolean }
```

Same shape as `useResidualDecode`: AbortController, inert-when-undefined, fires on arg change.

### 5.2 Component (new): `components/visualizations/LensTraceStrip.tsx`

```typescript
type Props = {
  sessionName: string;
  prompt: string;
  promptTokens?: string[];   // from result.baselines if available; falls back to grid.prompt_tokens
  initialPosition?: number;  // defaults to last token
};
```

Renders:
- `<select>` position selector (label: `pos N (token-string)`).
- A flex-column of rows, one per `(layer, sublayer)`. Order: layer 0 attn, layer 0 ffn, layer 1 attn, ..., layer N-1 ffn.
- Each row: `<label>L{n}.{sublayer}</label>` left, `<span>{token}</span>` right (monospace).
- Loading: "decoding lens grid…"
- Error: "lens-grid error: {msg}"

Width: ~180px fixed. Scrollable when there are many layers (CSS `max-height: 600px; overflow-y: auto`).

### 5.3 Wire into `ActivationPatchingHeatmap`

Wrap the existing `<svg>` and a new `<LensTraceStrip>` in a flex row:

```tsx
<div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
  <div style={{ flex: 1, overflowX: "auto" }}>
    <svg ref={svgRef} />
  </div>
  <LensTraceStrip
    sessionName={result.sessionName}
    prompt={result.prompt}
    promptTokens={baselines?.prompt_tokens_clean}
    initialPosition={baselines?.measurement_position}
  />
</div>
```

The pin-card behavior is unchanged (sits inside the heatmap container; absolutely positioned).

## 6. Testing

### 6.1 Backend (new file `tests/test_decode_residual_grid.py`)

Reuses the same `_MockTokenizer` / `_MockModel` mock stack from `test_decode_residual.py`. Five tests:

1. `test_404_missing_session` — non-existent session → 404.
2. `test_500_no_model` — model is None → 500.
3. `test_413_prompt_too_long` — > 200 tokens → 413.
4. `test_response_shape` — happy path returns expected keys, cell counts, top-1 only when top_k=1.
5. `test_top_k_clamped_to_5` — top_k=999 → at most 5 tokens per cell.

Plus one TinyLlama parity test:

6. `test_grid_matches_logit_lens_at_last_layer` — at last_layer + ffn sublayer, the grid's top-1 token at every position must equal `probe.logit_lens()`'s top-1 at the same point.

### 6.2 Frontend

Playwright smoke test that mocks `/decode-residual-grid` and asserts the strip renders. No Vitest (no RTL — same pattern as Phase 3.11).

## 7. Implementation plan

**Single batch — same-file changes; no need to split.**

### Task 1 — Backend (parent does this directly)

1. Add `DecodeResidualGridRequest` + `decode_residual_grid` endpoint to `gui/backend/routes/sessions.py`.
2. Create `tests/test_decode_residual_grid.py` (mirror `test_decode_residual.py` shape).
3. Run pyright + unit tests + TinyLlama integration. Commit.

### Task 2 — Frontend (parent does this directly)

1. Create `utils/useResidualGrid.ts`.
2. Create `components/visualizations/LensTraceStrip.tsx`.
3. Modify `ActivationPatchingHeatmap.tsx` — wrap svg + strip in flex row.
4. Add Playwright smoke test.
5. Run tsc + Vitest + Playwright. Commit.

**Why parent-direct (no subagent):** This phase is small (3 files new, 1 modified, ~200 lines total). Subagent overhead exceeds the work itself. Phase 3.11 also caught subagent-misread issues that wasted time; the recipe here is template-perfect from Phase 3.11.

## 8. Verification matrix

| Lane | Expected |
|------|----------|
| pyright | 0/0/0 across `sessions.py` + `test_decode_residual_grid.py` |
| pytest unit | 5/5 mock tests |
| pytest TinyLlama | 1/1 (~5s with cached model in VRAM) |
| pytest regression | preserved (no probe.py changes) |
| tsc | clean |
| Vitest | 19/19 (no new tests) |
| Playwright | 20/20 (19 prior + 1 new) |
