# Phase 3.9.2 — Attention Head OV Output Decoding

**Date:** 2026-04-23
**Status:** Spec (awaiting review)
**Depends on:** Phase 3.6 (per-head AP identifies causal heads), Phase 3.9.1 (endpoint + pin-card pattern for per-neuron)
**References:**
- Elhage et al. 2021 — *A Mathematical Framework for Transformer Circuits* (OV-circuit framing)
- Anthropic — various *Toy Models of Superposition* follow-ups using OV decomposition

## 1. Motivation

Phase 3.6 ships per-head attribution: "attention head 4 at layer 11 drives the metric by 0.35." Phase 3.9.1 closed the same loop for FFN neurons by decoding them into promoted/suppressed tokens. Phase 3.9.2 does the analogous thing for attention heads.

Unlike FFN (`W_down[:, i]` is a 1D direction), each attention head's output projection `W_O_h: [hidden, head_dim]` is a `head_dim`-dim subspace, not a single direction. The standard move (Elhage et al. 2021 §2.2) is to decompose via SVD and decode the dominant left-singular vector — "the most-used output direction of head h in vocabulary space."

## 2. Goals

- **G1.** New `POST /api/sessions/{name}/decode-head` endpoint. Body: `{layer, head, top_k}`. Returns top-k and bottom-k tokens along the dominant write direction, plus `singular_value_ratio` (how dominant the first singular value is — e.g., 0.62 means the first SV captures 62% of the matrix's Frobenius energy).
- **G2.** `PerHeadPatchingHeatmap.tsx` extends its existing pin card. When the pinned cell's `unit` starts with `"attn.h"`, fetch `decode-head` and render promoted/suppressed tokens inside the pin card. For `unit === "ffn"`, no decode (that's Phase 3.9.1's territory — suggest switching to approx_neuron mode in a caption).
- **G3.** Backend unit tests (mock) + TinyLlama integration + Playwright smoke via route interception.

## 3. Non-Goals

- **N1.** Multi-rank decomposition. Decode only the top singular vector. Showing rank-2 or rank-3 would double/triple the pin-card content for marginal interpretability gain (the dominant singular direction typically captures 40-70% of the matrix's energy on small models).
- **N2.** Per-head QK decomposition. This phase only decomposes the V→O pathway ("write side"), not the Q→K pathway ("read side"). Attn-score-decoding is a different feature and much larger scope.
- **N3.** Full OV matrix `W_U @ W_O_h @ W_V_h @ W_E`. That's a `[vocab, vocab]` matrix (~50000²) and unrenderable. Could be reintroduced later as "given input token X, what does head h promote?" (row-wise view) but not now.
- **N4.** Caching. Each click triggers fresh SVD. One SVD on `[2048, 64]` is <5 ms on GPU; no caching needed.
- **N5.** GQA/MQA handling. TinyLlama is standard MHA (`num_key_value_heads == num_attention_heads`). If GQA is later added, the decode formula changes because multiple Q heads share a K/V head — documented as TODO in the endpoint.

## 4. Math

### 4.1 OV output direction per head

For layer `L`, head `h`:

```
W_O = model.model.layers[L].self_attn.o_proj.weight  # [hidden, hidden]
head_dim = hidden // n_heads
W_O_h = W_O[:, h*head_dim:(h+1)*head_dim]            # [hidden, head_dim]

U, S, Vt = torch.linalg.svd(W_O_h, full_matrices=False)
# U: [hidden, head_dim], S: [head_dim], Vt: [head_dim, head_dim]

write_direction = U[:, 0] * S[0]                     # [hidden]
vocab_scores = W_U @ write_direction                  # [vocab]
```

### 4.2 Singular value ratio

```
sv_ratio = (S[0] ** 2) / (S ** 2).sum()  # fraction of matrix energy in rank-1 approximation
```

Typically 0.3-0.7 on small models. UI shows this so the user knows how faithful the rank-1 decoding is. A head with `sv_ratio = 0.9` has a near-1D output; `sv_ratio = 0.2` means the head's output is high-dimensional and the top tokens are a weak summary.

### 4.3 Sign handling

`W_U @ write_direction` has an arbitrary sign (flipping U's first column flips the whole decoding). Convention: orient so that the **top-k promoted tokens have more total energy than the bottom-k** — i.e., flip sign if `sum(top.logits) < -sum(bottom.logits)`. This keeps "top promoted" semantically meaningful.

## 5. API

### 5.1 Backend endpoint

```python
class DecodeHeadRequest(BaseModel):
    layer: int
    head: int
    top_k: int = 10


@router.post("/sessions/{name}/decode-head")
async def decode_head(name: str, req: DecodeHeadRequest):
    """Return top-k and bottom-k tokens along the dominant write direction
    of attention head `head` at `layer`, computed via SVD of W_O_h =
    o_proj.weight[:, h*head_dim:(h+1)*head_dim].

    Sign is oriented so promoted tokens dominate (sum of top logits > sum
    of -bottom logits). Response:
      {
        "top_tokens":    [{"token": str, "logit": float}, ...],
        "bottom_tokens": [{"token": str, "logit": float}, ...],
        "singular_value_ratio": float   # S[0]^2 / sum(S^2) — rank-1 fidelity
      }
    """
```

Validation: `0 <= layer < num_hidden_layers`, `0 <= head < num_attention_heads`, `top_k` clamp to `[1, min(50, vocab_size)]`. Same 404/500 error shapes as `decode-neuron`.

### 5.2 Frontend

`PerHeadPatchingHeatmap` already has a `pinned` state and a floating pin card (left over from Phase 3.6's implementation). Extend that card with:

1. When `pinned.cell.unit` starts with `"attn.h"`, parse `head = parseInt(unit.slice(6))`.
2. `useEffect` watching `[pinned, sessionName]`: fetch `decode-head` with abort-on-unpin.
3. Render two columns (top promoted / bottom suppressed) + singular-value-ratio caption.
4. When `unit === "ffn"`, show a caption: "FFN blocks are decoded per-neuron — switch to approx_neuron mode for interpretation."

Pass `sessionName` prop from `VisualizationArea.tsx` like Phase 3.9.1 did for PerNeuronPatchingPanel.

## 6. Testing

### 6.1 Backend unit tests (`test_decode_head.py`)

1. **`test_returns_topk_bottomk_and_sv_ratio`** — mock model, vocab=10, hidden=8, n_heads=2 (head_dim=4). Call `(layer=0, head=1, top_k=3)`. Assert response shape, `top_tokens` sorted desc, `bottom_tokens` sorted asc, `0 <= singular_value_ratio <= 1`.
2. **`test_invalid_layer_400`**.
3. **`test_invalid_head_400`**.
4. **`test_top_k_clamped_to_50_and_vocab`** — `top_k=1000` with vocab=10 → returns `min(50, 10) = 10` entries.
5. **`test_unknown_session_404`**.
6. **`test_sign_oriented_by_promoted_magnitude`** — check that `abs(sum(top)) >= abs(sum(bottom))`.

### 6.2 TinyLlama integration

`test_decode_head_tinyllama.py` or appended to `test_decode_head.py`:
- Run Phase 3.6 `attribution_patch_per_head` on capital-of-France, pick the top positive-AP head, call decode-head, assert 10 non-empty decoded top tokens and `0 <= sv_ratio <= 1`.

### 6.3 Playwright smoke (17th test)

Load per-head fixture, click a cell, intercept `/api/sessions/*/decode-head` with a stub response, assert pin card shows stub tokens + close button.

## 7. Commit plan

3 tasks, each one commit:

1. **Spec commit** — this file.
2. **Backend endpoint + unit tests + TinyLlama integration** (combined since subagent pattern shows same-file work ships together).
3. **Frontend pin-card decode extension + Playwright smoke**.

## 8. Verification matrix

- pyright 0/0/0.
- tsc clean.
- Phase 3.6-3.9.1 regressions preserved.
- Vitest still 19/19.
- Playwright 16 + 1 new = 17 passing.

## 9. Open questions

*(None — resolved in §1–8.)*
