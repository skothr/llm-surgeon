# Phase 3.9.1 — Top-logit Decoding per Neuron

**Date:** 2026-04-23
**Status:** Spec (awaiting review)
**Depends on:** Phase 3.5 (decode-ids endpoint pattern), Phase 3.9 (per-neuron AP)
**References:**
- nostalgebraist 2020 — *Interpreting GPT: the logit lens*
- Geva et al. 2020 — *Transformer Feed-Forward Layers Are Key-Value Memories*
- Dar et al. 2022 — *Analyzing Transformers in Embedding Space*

## 1. Motivation

Phase 3.9 ranks FFN neurons by causal contribution to a logit-diff metric, but only as integers: "neuron 1234 in layer 11 has `ap_recovery=0.45`." A researcher needs to know what that neuron **does**. The classic move (Geva 2020, nostalgebraist) is to treat each neuron's column in `W_down` as a "value vector" written to the residual stream, then pass that vector through the unembedding to see which tokens it promotes.

This phase adds the one endpoint + one UI card that makes Phase 3.9's ranked list actionable: click a neuron, see "this neuron writes strongly toward [Paris, Lyon, ...] and suppresses [Rome, Milan, ...]".

## 2. Goals

- **G1.** New `POST /api/sessions/{name}/decode-neuron` endpoint: input `{layer, neuron, top_k}` → output `{top_tokens: [{token, logit}], bottom_tokens: [{token, logit}]}`.
- **G2.** `PerNeuronPatchingPanel.tsx` renders a pinned-row card when a table row is clicked. Card shows top-10 promoted + bottom-10 suppressed tokens with logit values and a "copy" button.
- **G3.** Backend unit test on mock model (small vocab). Integration test on TinyLlama. Frontend Vitest for the fetcher. Playwright smoke exercises the click → card flow with a mocked backend response.

## 3. Non-Goals

- **N1.** Rigorous logit-lens normalization (dividing by the mean-hidden-state-norm at that layer, per Dar et al. 2022). V1 uses raw `W_U @ W_down[:, i]` and notes the limitation in the UI caption. Within-neuron token ranking is unbiased by the omission — bias cancels — but across-neuron magnitude comparisons are noisy.
- **N2.** Gate/up projection interpretation. LLaMA MLP is `down_proj(silu(gate_proj(h)) * up_proj(h))`. Phase 3.9.1 only decodes `down_proj` columns — not gate/up directions. Neuron-level gate/up interpretation requires a different framing (Dar 2022 handles this separately).
- **N3.** Caching across requests. Each click triggers a fresh `W_U @ W_down[:, i]` compute. On a 32k-vocab × 2048-hidden-dim model this is ~65M multiplies — ~10 ms on GPU, ~100 ms on CPU. Per-click cost is acceptable; caching adds complexity without clear wins.
- **N4.** Attention-head OV decoding. Same idea exists for attn (`W_O` columns per head), but that's out of scope — save for a Phase 3.9.2 if demand materializes.
- **N5.** Neuron-level visualization of "which dataset prompts activate this neuron most" (the Bills et al. 2023 pipeline). That's a different, much bigger feature.

## 4. Math

### 4.1 Decode formula

Given a LLaMA-style model with:
- `W_down[L]: [hidden, intermediate]` — the MLP output projection at layer L.
- `W_U: [vocab, hidden]` — the unembedding (= `lm_head.weight`).
- `final_norm`: `model.model.norm` (RMSNorm).

For layer `L`, neuron index `i`:

```
direction_i = W_down[L][:, i]                # [hidden]
normed_dir  = final_norm(direction_i)         # v1 deliberately skips this (see §3 N1)
vocab_scores = W_U @ direction_i              # [vocab]
top_ids = vocab_scores.topk(k).indices
top_logits = vocab_scores.topk(k).values
```

Notes:
- `W_U @ direction_i` where `W_U: [vocab, hidden]` and `direction_i: [hidden]` yields `[vocab]`. This is the standard matmul.
- Bottom-k: `vocab_scores.topk(k, largest=False)`.
- "Logit" in the output is the raw dot product — not a probability. UI labels as "logit" to match `logit_lens` semantics.

### 4.2 Why skip final-norm in v1

`final_norm(x) = x / sqrt(mean(x^2) + eps) * gamma` where gamma is the RMSNorm scale. The division by `sqrt(mean(x^2))` depends on the actual residual-stream magnitude at inference time — not knowable from the weight alone. Approximations exist (use the dataset mean; see Dar 2022), but they inject a free parameter.

V1 ranks tokens by `W_U @ W_down[:, i]` directly, which is equivalent to ranking by `W_U @ normed_dir` up to the neuron-specific scalar `1/sqrt(mean(dir^2))`. **Within a single neuron's top-k**, the ranking is identical. **Across neurons**, magnitudes are biased — but for the UI's "what does *this* neuron do?" purpose, the within-neuron ranking is what matters.

The UI caption will say: "Raw `W_U @ W_down[:, neuron]`. Normalize-for-magnitudes deferred."

## 5. API

### 5.1 Backend endpoint

```python
# routes/sessions.py

class DecodeNeuronRequest(BaseModel):
    layer: int
    neuron: int
    top_k: int = 10  # clamp to [1, 50]

@router.post("/sessions/{name}/decode-neuron")
async def decode_neuron(name: str, req: DecodeNeuronRequest) -> Dict:
    """Return the top-k and bottom-k tokens most strongly promoted/suppressed
    by neuron `neuron` at layer `layer`, computed as W_U @ W_down[:, neuron].

    Response shape:
      {
        "top_tokens":    [{"token": str, "logit": float}, ...],
        "bottom_tokens": [{"token": str, "logit": float}, ...]
      }

    Returns 400 on out-of-range layer/neuron, 404 on unknown session, 500
    if no PyTorch model is loaded.
    """
```

Clamp `top_k` to `[1, 50]`. Validate `0 <= layer < num_hidden_layers` and `0 <= neuron < intermediate_size`. Runs in a thread executor (`asyncio.to_thread`) because the matmul is synchronous.

### 5.2 Frontend API wrapper

New typed fetch in `utils/api.ts` (or wherever Phase 3.5's `decodeIds` lives):

```ts
export interface DecodeNeuronResponse {
  top_tokens: Array<{ token: string; logit: number }>;
  bottom_tokens: Array<{ token: string; logit: number }>;
}

export async function decodeNeuron(
  session: string,
  layer: number,
  neuron: number,
  top_k: number = 10,
): Promise<DecodeNeuronResponse>;
```

## 6. Frontend

### 6.1 PerNeuronPatchingPanel changes

Current panel has a table where each row represents a (layer, neuron, position, ap_recovery) cell. Phase 3.9.1 adds:

1. **Row click state**: clicking a row sets `pinnedRow: { layer, neuron } | null`. Clicking the same row again unpins.
2. **Pinned card**: rendered below the stats strip (above the table so it's always visible). Layout:
   ```
   ┌─ Neuron L11.n1234 ──────────────────── [close] ─┐
   │ Raw W_U @ W_down[:, 1234]                        │
   │                                                  │
   │ Top 10 promoted                Bottom 10 suppressed│
   │   Paris    +2.34                 Rome      −1.89   │
   │   Lyon     +1.87                 Milan     −1.52   │
   │   …                              …                 │
   │                                                  │
   │ [copy top]                       [copy bottom]   │
   └──────────────────────────────────────────────────┘
   ```
3. **Fetch-on-pin**: when `pinnedRow` changes, fetch `decodeNeuron(sessionName, layer, neuron, 10)`. Show loading state. On error, show error message.
4. **Session name**: panel currently receives `cells` and `complete`. Needs `sessionName: string` prop. VisualizationArea knows this from `activeResult.sessionName`.

### 6.2 Hover affordance

Rows get `cursor: pointer` and a subtle `:hover` highlight. Pinned row gets a persistent highlight (e.g., left border).

### 6.3 Fetch behavior

- Single in-flight request per panel. If the user clicks another row while a request is pending, abort the first (AbortController) and start the second.
- No caching across pins — simplest correct behavior.

## 7. Testing

### 7.1 Backend unit test

`test_decode_neuron.py` or append to existing `test_sessions_api.py`:

1. **`test_decode_neuron_mock_model_returns_topk`** — mock model with 10-vocab, 4-hidden, 8-intermediate. Call endpoint for `(layer=0, neuron=3, top_k=3)`. Assert response shape, assert `top_tokens` sorted by logit desc, assert `bottom_tokens` sorted by logit asc.
2. **`test_decode_neuron_invalid_layer`** — layer out of range → 400.
3. **`test_decode_neuron_invalid_neuron`** — neuron out of range → 400.
4. **`test_decode_neuron_top_k_clamp`** — top_k=1000 → clamped to 50 (or rejected; pick one). Spec says clamp; implementation must match.
5. **`test_decode_neuron_unknown_session`** — 404.
6. **`test_decode_neuron_no_model`** — session exists but no PyTorch model loaded → 500.

### 7.2 TinyLlama integration

`test_decode_neuron_tinyllama.py`:

1. **`test_capital_of_france_neuron_promotes_paris`** — call Phase 3.9 `attribution_patch_per_neuron` on the capital-of-France task with TinyLlama, take the top-ranked (layer, neuron) with positive `ap_recovery`, then decode that neuron. Assert that at least one of the top-10 promoted tokens has "Paris" or "paris" or " Paris" as a substring (HuggingFace tokenizers vary spacing). Loose assertion because 1B-param models are noisy; the goal is "this neuron's top tokens are semantically coherent," not "the top token is literally 'Paris'."
2. Skip if CUDA unavailable or TinyLlama not cached.

### 7.3 Frontend Vitest

`utils/decodeNeuron.test.ts`:

1. **`test_fetchesAndParsesResponse`** — mock `fetch`, assert correct URL + body, assert response parsed to `DecodeNeuronResponse`.
2. **`test_abortsPriorRequest`** — start request, start another, assert first was aborted.

### 7.4 Playwright smoke

16th test: loads the per-neuron fixture, clicks first row, intercepts `/api/sessions/.../decode-neuron` to return a stub response, asserts the pinned card renders with the stub tokens.

## 8. Commit plan

5 tasks, one commit each unless same-file forces combination.

1. **Spec commit** — this file.
2. **Backend endpoint + unit tests** — `routes/sessions.py` + `test_sessions_api.py` (or new test file).
3. **TinyLlama integration test** — `test_decode_neuron_tinyllama.py` (new) OR append to `test_probe_per_neuron_ap.py`.
4. **Frontend `decodeNeuron` fetcher + Vitest** — `utils/api.ts` (or new file) + test.
5. **`PerNeuronPatchingPanel` pinned card + Playwright smoke** — component changes + new 16th smoke test + backend route interception.

## 9. Verification matrix

- **pyright**: 0/0/0 on `routes/sessions.py`, any new test files.
- **tsc**: clean.
- **pytest (non-GPU)**: all existing + new unit tests pass.
- **TinyLlama**: new integration test passes.
- **Vitest**: all existing + new decodeNeuron tests pass.
- **Playwright**: 15 existing + 1 new = 16 passing.

## 10. Open questions

*(None — resolved in §1–9.)*
