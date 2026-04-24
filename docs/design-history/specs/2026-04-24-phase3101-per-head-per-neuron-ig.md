# Phase 3.10.1 — Integrated Gradients for Per-Head and Per-Neuron AP

**Date:** 2026-04-24
**Status:** Spec (awaiting review)
**Depends on:** Phase 3.6 (`attribution_patch_per_head`), Phase 3.9 (`attribution_patch_per_neuron`), Phase 3.10 (`_integrated_gradients_loop` helper)

## 1. Motivation

Phase 3.10 shipped IG for `attribution_patch` (Phase 3.5 block-level AP). Per-head (Phase 3.6) and per-neuron (Phase 3.9) AP variants still use first-order linearization. The upgrade costs almost nothing: `_integrated_gradients_loop` already computes the exact gradients those variants need, because per-head's math (`concat_z_grad = attn_out.grad @ W_O`) and per-neuron's math (`grad_act = ffn_out.grad @ W_down`) are chain-rule derivatives of the same `attn_out.grad` and `ffn_out.grad` that the IG helper already averages. Swap the `.grad` read for an `avg_grad` dict lookup in each variant's per-cell loop — no new infrastructure.

## 2. Goals

- **G1.** `attribution_patch_per_head(..., n_steps: int = 1)`. When `n_steps=1` preserves existing Phase 3.6 behavior bit-identical. When `n_steps >= 2`, runs `_integrated_gradients_loop(sublayers=("attn", "ffn"))` and uses `avg_grad[(L, "attn")]` in place of `base_captured[(L, "attn")].grad` for per-head chain rule, and `avg_grad[(L, "ffn")]` in place of `base_ffn.grad` for the FFN anchor rows.
- **G2.** `attribution_patch_per_neuron(..., n_steps: int = 1)`. Same pattern: `avg_grad[(L, "ffn")]` replaces `base_captured[(L, "ffn_out")].grad` in the `grad_act = grad_ffn_out @ W_down` computation.
- **G3.** `PatchingResult.n_steps` populated on per-head and per-neuron returns (already exists from Phase 3.10).
- **G4.** Backend WS `approx_head` and `approx_neuron` branches in `routes/probes.py` parse and forward `cfg.n_steps` identically to Phase 3.10's `approx` branch. `complete.summary.n_steps` populated.
- **G5.** Frontend `PatchingControls.tsx` extends the `n_steps` input visibility from `mode === "approx"` to `mode === "approx"` OR `mode === "approx_head"` OR `mode === "approx_neuron"`.
- **G6.** Tests: 2 unit tests (one per variant, using existing mock fixtures) + 2 TinyLlama integration tests (top-20 Spearman ≥ 0.5 between n=1 and n=5).

## 3. Non-Goals

- **N1.** IG for edge AP (Phase 3.7) and circuit extraction (Phase 3.8). Edge AP captures reader-side gradients at LN pre-hooks; its IG variant needs a different replacement strategy. Deferred.
- **N2.** New helper variants. `_integrated_gradients_loop` is unchanged.
- **N3.** New Playwright smoke tests. The only UI change is `n_steps` input visibility expanding to 2 more modes — the existing Phase 3.10 smoke covers the input rendering; the new modes' smokes already cover the heatmap + panel shells.
- **N4.** Changing Phase 3.6's cumulative-Δ anchor or Phase 3.9's per-neuron Δ anchor. Δ definitions unchanged; only the gradient averaging switches on at `n_steps ≥ 2`.

## 4. Math

### 4.1 Per-head (Phase 3.6) under IG

Phase 3.6 per-head score at `(L, h, pos)`:
```
AP_head_first_order = Σ_head_dim [Δconcat_z[0, pos, h, :] · concat_z_grad[pos, h, :]]
```
where `concat_z_grad[pos] = base_captured[(L, "attn")].grad[0, pos] @ W_O`.

Under IG with N steps, `concat_z_grad` becomes `(1/N) Σ_k (attn_out_grad_k @ W_O)` where `attn_out_grad_k = interp_attn[L].grad` at step k. Since `W_O` is a constant (layer weight), this equals `(1/N Σ_k attn_out_grad_k) @ W_O = avg_grad[(L, "attn")] @ W_O`. **No per-step @ W_O required** — one @ at the end is mathematically equivalent and more efficient.

### 4.2 Per-neuron (Phase 3.9) under IG

Phase 3.9 per-neuron score at `(L, neuron_i, pos)`:
```
AP_neuron_first_order = Δact[0, pos, i] · grad_act[pos, i]
```
where `grad_act[pos] = base_captured[(L, "ffn_out")].grad[0, pos] @ W_down` (no transpose — same pattern as Phase 3.6's `grad_r @ W_O`; see `probe.py` Phase 3.9 implementation).

Under IG: `grad_act[pos] = (1/N Σ_k ffn_out_grad_k @ W_down) = avg_grad[(L, "ffn")][0, pos] @ W_down`. Same pattern as per-head: average first, then chain-rule once.

### 4.3 Key observation

Phase 3.10's `_integrated_gradients_loop` with `sublayers=("attn", "ffn")` already does all the gradient averaging. Per-head and per-neuron simply *consume* the averaged gradients through their existing chain-rule recipes. The entire per-head IG implementation is ~3 lines of change, and per-neuron is ~2 lines.

## 5. API

### 5.1 Backend

```python
def attribution_patch_per_head(
    model, tokenizer, clean_prompt, corrupted_prompt, *,
    correct_token_id, incorrect_token_id,
    direction="denoise", measurement_position=-1,
    positions=None, layers=None,
    n_steps=1,                # NEW
    on_cell=None,
) -> PatchingResult:
```

```python
def attribution_patch_per_neuron(
    model, tokenizer, clean_prompt, corrupted_prompt, *,
    correct_token_id, incorrect_token_id,
    direction="denoise", measurement_position=-1,
    positions=None, layers=None,
    top_k_neurons=200,
    n_steps=1,                # NEW
    on_cell=None,
) -> PatchingResult:
```

Both validate `n_steps` via the same `isinstance(int) and 1 <= n_steps <= 50` check. Result's `n_steps` populated as `n_steps if n_steps > 1 else None`.

### 5.2 WebSocket route

In `routes/probes.py`, both `approx_head` and `approx_neuron` mode branches gain `n_steps` parsing, forwarding, and `complete.summary.n_steps` population — lifted from Phase 3.10's `approx` branch verbatim.

### 5.3 Frontend

Only change: `PatchingControls.tsx` IG-steps input visibility condition expands from `mode === "approx"` to `["approx", "approx_head", "approx_neuron"].includes(mode)`. ProbePanel.tsx forwards `n_steps` in the WS cfg for all three modes.

## 6. Testing

### 6.1 Per-head unit test

`test_per_head_n_steps_converges` — mock fixture (reuse Phase 3.6's `_make_per_head_fixtures` or equivalent). Call `attribution_patch_per_head(..., n_steps=1)` and `n_steps=10`. Assert:
- `r_10.n_steps == 10`
- all per-head cells finite
- at least one cell's `ap_recovery` differs between runs by > 1e-4

### 6.2 Per-neuron unit test

`test_per_neuron_n_steps_converges` — mock fixture. Same shape of assertions.

### 6.3 TinyLlama per-head IG

`test_per_head_ig_tinyllama` — capital-of-France prompt. Run `n_steps=1` and `n_steps=5`. Top-20 Spearman ρ > 0.5 between the two rankings (by `|ap_recovery|` on Phase 3.6 output).

### 6.4 TinyLlama per-neuron IG

`test_per_neuron_ig_tinyllama` — same prompt. Same assertion, `top_k_neurons=50` for speed.

### 6.5 No Playwright additions

Existing Phase 3.6 and Phase 3.9 smokes cover panel rendering. n_steps input visibility is covered by Phase 3.10's smoke (fixture-driven). Keep count at 18.

## 7. Commit plan

4 commits:
1. Spec — this file.
2. Plan.
3. Python core + unit tests + TinyLlama (all probe.py + test file edits — ship as one commit because they're mechanically coupled).
4. Backend route + frontend visibility expansion.

## 8. Verification matrix

- pyright 0/0/0.
- tsc clean.
- Phase 3.6 and Phase 3.9 regressions preserved (n_steps=1 bit-identical).
- 2 new unit tests + 2 new TinyLlama integrations pass.
- Vitest 19/19, Playwright 18/18 unchanged.

## 9. Open questions

*(None — resolved under full-autonomy brainstorm.)*
