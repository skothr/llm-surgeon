# Phase 3.11 — Logit Lens × Activation-Patching Pin-Card Integration

**Date:** 2026-04-25
**Author:** llm_surgeon project
**Status:** approved (autonomy directive)

## 1. Problem

Activation-patching panels currently answer "WHICH (layer, position) cells causally matter" but not "WHAT tokens does the model think it's producing at that point." The logit-lens infrastructure (`probe.logit_lens`, `_project_to_logits`) already projects any residual-stream point through the final norm + `lm_head` to get vocab-space logits — but the AP pin cards do not consume it.

Effect: a user sees "patching layer 8 attn at position 3 recovers 0.7 logit-diff" with no semantic context. After this phase, the same pin card also shows "the residual stream at (layer 8, after-attn, position 3) is currently promoting tokens [Paris, France, …] and suppressing [Tokyo, Berlin, …]". Causal map + semantic readout in one view.

## 2. Goals

- One reusable backend endpoint that decodes a single residual-stream point on demand.
- Every AP visualization pin card (5 modes) gains a "Logit lens at this point" block.
- Backwards compatible: cells pre-dating this phase still pin the way they did before; the lens block is additive.
- Reuses `_capture_residual_stream` + `_project_to_logits` from `probe.py` — no new capture infrastructure.

## 3. Non-goals

- No bulk decoding (every cell on every render). On-demand, fired only when user pins a cell.
- No "embed" sublayer in V1. Edge AP writers with `unit="embed"` show a caption ("residual decode at embed unsupported in V1") instead of the lens block. Adding embed support is a follow-up: it's a forward-hook on `model.model.embed_tokens` plus one extra `sublayer` enum value — easy but out of scope.
- No A/B (clean vs corrupted) decode. Single-prompt residual lens only. Future work could decode both prompts side-by-side, mirroring Phase 3.9.1/3.9.2 patterns.
- No final-norm scaling toggle. We use the standard `_project_to_logits` (which DOES apply final norm) — same path as `logit_lens()`. (Phase 3.9.1's neuron decode skipped final norm because a single-direction-from-W_down is unaffected by per-row scaling; here we project a real residual state, so final norm matters and is included.)

## 4. Backend — `decode-residual` endpoint

**Route:** `POST /api/sessions/{name}/decode-residual`

**Request body:**
```python
class DecodeResidualRequest(BaseModel):
    prompt: str
    layer: int
    sublayer: str           # "attn" | "ffn"
    position: int
    top_k: int = 10
```

**Response:**
```json
{
  "top_tokens":    [{"token": "Paris",  "logit": 21.34}, ...],
  "bottom_tokens": [{"token": "Tokyo",  "logit": -8.12}, ...],
  "prompt_tokens": ["The", "Eiffel", "Tower", "is", "in"]
}
```

**Validation:**
- 404 if session not found.
- 500 if model or tokenizer not loaded.
- 400 if `sublayer not in {"attn", "ffn"}`.
- 400 if `layer not in [0, num_layers)`.
- 400 if `position` out of `[0, len(prompt_tokens))` — checked AFTER tokenization so we can quote the exact bound.
- `top_k` clamped to `max(1, min(req.top_k, _DECODE_RESIDUAL_TOPK_MAX, vocab_size))` where the cap is **50**, mirroring `decode-neuron`.

**Implementation (in `gui/backend/routes/sessions.py`, alongside the three existing decode endpoints):**

```python
class DecodeResidualRequest(BaseModel):
    prompt: str
    layer: int
    sublayer: str
    position: int
    top_k: int = 10


_DECODE_RESIDUAL_TOPK_MAX = 50
_DECODE_RESIDUAL_VALID_SUBLAYERS = {"attn", "ffn"}


@router.post("/sessions/{name}/decode-residual")
async def decode_residual(name: str, req: DecodeResidualRequest):
    """Project the residual stream at (layer, sublayer, position) through
    the final norm + lm_head, return top/bottom-k tokens.

    Standard logit lens (nostalgebraist 2020) applied to a single
    residual-stream slot — used by AP pin cards to surface what the
    model "thinks" it's predicting at the patched point.
    """
    import torch
    from llm_surgeon.probe import _capture_residual_stream, _project_to_logits

    if req.sublayer not in _DECODE_RESIDUAL_VALID_SUBLAYERS:
        raise HTTPException(
            400,
            f"sublayer must be one of {sorted(_DECODE_RESIDUAL_VALID_SUBLAYERS)}; got {req.sublayer!r}",
        )

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
    if not 0 <= req.layer < num_layers:
        raise HTTPException(
            400,
            f"layer out of range: got {req.layer}, valid [0, {num_layers})",
        )

    def _compute() -> dict:
        captured, prompt_tokens = _capture_residual_stream(
            model, tok, req.prompt, sublayers=(req.sublayer,),
        )
        seq_len = len(prompt_tokens)
        if not 0 <= req.position < seq_len:
            raise HTTPException(
                400,
                f"position out of range: got {req.position}, valid [0, {seq_len})",
            )
        hidden = captured[(req.layer, req.sublayer)]  # (seq_len, d_model)
        with torch.no_grad():
            logits = _project_to_logits(model, hidden)  # (seq_len, vocab)
        scores = logits[req.position]  # (vocab,)
        vocab_size = int(scores.shape[0])
        k = max(1, min(req.top_k, _DECODE_RESIDUAL_TOPK_MAX, vocab_size))
        top_vals, top_ids = torch.topk(scores, k, largest=True)
        bot_vals, bot_ids = torch.topk(scores, k, largest=False)
        top_tokens = [
            {"token": tok.decode([int(i)], skip_special_tokens=False), "logit": float(v)}
            for i, v in zip(top_ids.tolist(), top_vals.tolist())
        ]
        bottom_tokens = [
            {"token": tok.decode([int(i)], skip_special_tokens=False), "logit": float(v)}
            for i, v in zip(bot_ids.tolist(), bot_vals.tolist())
        ]
        return {
            "top_tokens": top_tokens,
            "bottom_tokens": bottom_tokens,
            "prompt_tokens": prompt_tokens,
        }

    return await asyncio.to_thread(_compute)
```

**Note on the inner HTTPException:** `_capture_residual_stream` runs after layer-validation so seq_len is only knowable post-tokenize. We re-raise from inside the threaded computation — FastAPI catches it via the standard exception handler chain because `asyncio.to_thread` re-raises in the calling coroutine.

## 5. Frontend — pin-card integration

### 5.1 Hook (new): `utils/useResidualDecode.ts`

```typescript
export type ResidualDecodeToken = { token: string; logit: number };
export type ResidualDecodeResponse = {
  top_tokens: ResidualDecodeToken[];
  bottom_tokens: ResidualDecodeToken[];
  prompt_tokens: string[];
};

export function useResidualDecode(
  sessionName: string | undefined,
  prompt: string | undefined,
  layer: number | undefined,
  sublayer: "attn" | "ffn" | undefined,
  position: number | undefined,
  topK: number = 10,
): { data: ResidualDecodeResponse | null; error: string | null; loading: boolean }
```

When any required arg is `undefined`, the hook is inert (no fetch, returns `null/null/false`). On all-defined args, fires `POST /api/sessions/{name}/decode-residual` with `AbortController` lifecycle (cancels on unmount or arg change). Match the pattern from PerNeuronPatchingPanel's existing decode-neuron fetch (Phase 3.9.1) — same hook shape.

### 5.2 Pin-card extension — five panels

Each panel adds a `<ResidualDecodeBlock>` component (or inline JSX) below the existing pin-card content, gated on:
- The cell's writer is identifiable as `(layer, sublayer ∈ {attn, ffn}, position)`.
- For embed-writers (only edge / circuit panels), render a caption instead.

Per-panel cell → decode args mapping:

| Panel | Cell shape | Decode args |
|---|---|---|
| **ActivationPatchingHeatmap** (Phase 3 + 3.5) | `{ layer, sublayer ∈ {"attn","ffn"}, position }` | direct |
| **PerHeadPatchingHeatmap** (Phase 3.6) | `{ layer, unit ∈ {"attn.hN","ffn"}, position }` | sublayer = unit.startsWith("attn") ? "attn" : "ffn" |
| **PerNeuronPatchingPanel** (Phase 3.9) | `{ layer, neuron, position }` | sublayer = "ffn" (always) |
| **EdgeAttributionPanel** (Phase 3.7) | `{ writer_layer, writer_unit, position, ... }` | sublayer = writer_unit.startsWith("attn") ? "attn" : writer_unit === "ffn" ? "ffn" : null (embed → null → caption) |
| **CircuitPanel** (Phase 3.8) | same as Edge | same |

**Visual style** (consistent across all 5 panels):
- Section heading: `"— Logit lens at (L{layer}, {sublayer}, pos {position})"`
- Two-column grid: **Promoted** (green tint, top-10) | **Suppressed** (red tint, bottom-10)
- Per-row: `{token}` left-aligned, `{logit.toFixed(2)}` right-aligned monospace
- Loading state: "decoding…" caption
- Error state: red caption with the error string
- Embed-writer fallback: `"Residual decode at writer 'embed' is not supported in V1."`

Reuse the visual idiom from `PerNeuronPatchingPanel`'s existing decode-neuron block (Phase 3.9.1) — same color tints, same two-column layout. Aim for ≤ 80 lines of JSX per panel including the conditional gate.

### 5.3 Cleanliness — encapsulate in a shared component

To avoid five copies of the same JSX, extract `components/visualizations/ResidualDecodeBlock.tsx`:

```typescript
type Props = {
  sessionName: string;
  prompt: string;
  layer: number;
  sublayer: "attn" | "ffn";
  position: number;
};
```

Each panel imports and renders this component when it has the required props, or renders a caption otherwise. This keeps every pin card one block-of-JSX-line shorter.

## 6. Testing

### 6.1 Backend (in `testing/tests/test_decode_residual.py`)

Six unit tests with mock model + tokenizer (mirroring the existing `test_decode_neuron.py` / `test_decode_head.py` pattern):

1. `test_404_missing_session` — non-existent session name → 404.
2. `test_500_no_model` — session present but `model is None` → 500.
3. `test_400_invalid_sublayer` — `sublayer="embed"` (or any non-attn/ffn) → 400.
4. `test_400_layer_out_of_range` — layer = num_layers → 400.
5. `test_400_position_out_of_range` — position = seq_len → 400 with bound in message.
6. `test_top_k_clamped_to_50` — `top_k=999` returns exactly 50 entries (or vocab if smaller).
7. `test_response_shape` — happy path returns `top_tokens`, `bottom_tokens`, `prompt_tokens` with correct types and ordered (top descending, bottom ascending — by logit).

Plus one TinyLlama integration test:

8. `test_decode_residual_tinyllama_matches_logit_lens` — at the last layer's "ffn" sublayer, the top-1 token must equal the top-1 from `probe.logit_lens()` for the same prompt. (Both go through the same final norm + lm_head, so they MUST agree exactly modulo torch nondeterminism.) Skip-if no GPU; fp16 weights to keep memory low.

### 6.2 Frontend

- 1 Vitest test for `useResidualDecode`: AbortController is called on unmount.
- 1 Playwright test extending the existing exact-AP smoke (`activation-patching-circuit.json` already in fixtures): wait for an AP cell to be clickable, click it, assert the lens block renders with `getByText(/Logit lens at/i)` and at least one promoted-token row. Backend isn't running in Playwright; we mock the response by intercepting the network call via `page.route("**/decode-residual", ...)`.

## 7. Risks & non-issues

- **Forward pass on every pin click** — mitigated by AbortController + small TinyLlama-class models. For LLaMA-3-8B this might be 200-400ms on RTX 2080; acceptable for an interactive pin click.
- **Caching** — V1 doesn't memoize across pin clicks. If user pins the same cell twice they pay twice. Future: keyed memoization in the hook.
- **Existing per-neuron / per-head decodes are NOT made redundant** — they show what the WEIGHT (W_down column / SVD direction) writes, independent of input. The new residual-lens block shows what the ACTUAL RESIDUAL STREAM is currently promoting (input-dependent). Both useful; both shown side-by-side in the per-head and per-neuron pin cards.

## 8. Verification matrix

| Lane | Command (from `testing/`) | Expected |
|------|---------------------------|----------|
| pyright (backend) | `.venv/bin/pyright gui/backend/routes/sessions.py tests/test_decode_residual.py` | 0/0/0 |
| pyright (probe) | `.venv/bin/pyright llm_surgeon/probe.py` | preserved |
| pytest (unit) | `.venv/bin/pytest tests/test_decode_residual.py -v -k "not tinyllama"` | 7/7 |
| pytest (TinyLlama) | `.venv/bin/pytest tests/test_decode_residual.py -v -k "tinyllama"` | 1/1 (~30s) |
| pytest (regression) | `.venv/bin/pytest tests/ -v --co -q | wc -l` then full | preserved (no probe.py changes) |
| tsc | `(cd gui/frontend && ./node_modules/.bin/tsc --noEmit)` | clean |
| Vitest | `(cd gui/frontend && ./node_modules/.bin/vitest run)` | 20/20 (19 prior + 1 new) |
| Playwright | `(cd gui/frontend && ./node_modules/.bin/playwright test)` | 19/19 (18 prior + 1 new) |

## 9. Out of scope follow-ups (logged for later)

- Embed-position decode (forward hook on `model.model.embed_tokens`).
- Side-by-side clean-vs-corrupted residual lens decode.
- Bulk pre-compute of all (layer, sublayer, position) decodes when an AP run completes (so pin clicks are instant).
- Final-norm-scaled vs raw lens toggle (matches Phase 3.9.1 future-work note).
