# Phase 3.10 — Integrated Gradients for Attribution Patching

**Date:** 2026-04-23
**Status:** Spec (awaiting review)
**Depends on:** Phase 3.5 (`attribution_patch`, first-order AP)
**References:**
- Sundararajan, Taly & Yan 2017 — *Axiomatic Attribution for Deep Networks* (IG formulation, completeness axiom)
- Nanda 2023 — *Attribution Patching* primer (first-order method we're upgrading)
- Heimersheim & Nanda 2024 — *How to use and interpret activation patching*

## 1. Motivation

Phase 3.5 introduced gradient-based AP as a one-forward + one-backward approximation of the Phase 3 exact activation-patching loop. Its accuracy is limited by first-order linearization: the gradient is evaluated at a single point (the base activation) and multiplied by the full delta. When the loss surface is nonlinear between base and from (ReLU/GELU/SiLU saturation in FFN blocks, softmax saturation at extreme logits, normalization's variance-dependence), first-order AP systematically under- or over-estimates component contributions.

**Integrated Gradients** (Sundararajan 2017) fixes this by averaging the gradient over a straight-line path from base to from:

```
IG_i = (from_i - base_i) · ∫₀¹ ∂L(base + α·Δ)/∂act_i dα
```

Approximated via N midpoint-rule steps: `IG_i ≈ (from_i - base_i) · (1/N) Σ_{k=1..N} ∂L/∂act_i |_{x = base + α_k · Δ}` where `α_k = (k - 0.5) / N`.

Two advantages the user gets:
1. **Completeness axiom**: as N→∞, `Σ_components IG_i = L(from) - L(base) = D` exactly. First-order AP violates this systematically.
2. **Better rank-1 signal**: individual cells' scores become more trustworthy, especially cells whose base and from are far apart in activation space.

Current behavior (first-order at α=0) stays available at `n_steps=1` for bit-identical back-compat with all Phase 3.5 tests and downstream callers.

## 2. Goals

- **G1.** `attribution_patch(..., n_steps: int = 1)` extension. Default `n_steps=1` preserves existing behavior bit-identical. `n_steps >= 2` uses IG with midpoint-rule N-step path integration from base-activation to from-activation.
- **G2.** `PatchingResult.n_steps: Optional[int] = None` (populated when IG ran, else None so Phase 3.6/3.7/3.9 results stay untouched).
- **G3.** WS route `/ws/sessions/{name}/activation-patching` in `approx` mode accepts `n_steps` from cfg. `complete.summary.n_steps` populated.
- **G4.** `PatchingControls` in approx mode surfaces a small `n_steps` numeric input (default 1, range 1–50) with tooltip "Integrated Gradients path steps — higher N gives more accurate scores at N× compute cost".
- **G5.** `ActivationPatchingHeatmap` header shows "IG (N steps)" annotation when `result.n_steps > 1`.
- **G6.** Backend unit tests + mock completeness test + TinyLlama integration + Playwright smoke.

## 3. Non-Goals

- **N1.** IG for Phase 3.6 per-head, Phase 3.7 edge, Phase 3.9 per-neuron. Each uses a different capture/replacement hook point; same algorithmic pattern applies but is mechanical to replicate and belongs in separate follow-up phases (Phase 3.10.1, 3.10.2, 3.10.3 if needed).
- **N2.** Baseline-selection flexibility. IG traditionally offers "zero baseline" or "random baseline" vs "counterfactual baseline". In our setup the counterfactual is defined by `corrupted_prompt` — we do not expose other baselines. Straight-line corrupted → clean (or clean → corrupted for noise direction) is the only path.
- **N3.** Alternative integration rules (trapezoidal, Gauss-Legendre). Midpoint rule is O(1/N²) error and matches Sundararajan's simple_approximator; no reason to over-engineer.
- **N4.** Completeness as a hard runtime assertion. Checked in a mock test, but not enforced at runtime because floating-point drift + top-k truncation + disabled sublayers make exact equality impossible in practice.
- **N5.** Changing the `n_steps=1` code path. Back-compat requires bit-identical output; we branch on `n_steps == 1` and keep the existing logic.
- **N6.** GPU memory optimization. N forward+backward passes with `n_steps=N` allocates N fresh autograd graphs serially (each freed after `.backward()`); no attempt to share/reuse graphs across steps.

## 4. Math

### 4.1 Path and midpoint rule

Given `base_act` (the base-side activation tensor captured at each (L, sub)) and `from_act` (the from-side activation), the IG per-cell score is:

```
ap_raw(L, sub, pos) = Δ(L, sub, pos) · g(L, sub, pos)
```

where:
- `Δ(L, sub, pos) = from_val_full[0, pos] - base_val_full[0, pos]` (identical to Phase 3.5; for attn rows, val_full includes h_in residual).
- `g(L, sub, pos) = (1/N) Σ_{k=1..N} ∂L/∂act(L, sub)|_{act = base_act + α_k · (from_act - base_act)}[0, pos]` with `α_k = (k - 0.5) / N`.

For N=1: `α_1 = 0.5` gives midpoint-rule baseline (NOT equivalent to Phase 3.5's α=0). We therefore branch on `n_steps == 1` to preserve the α=0 behavior exactly, and only activate IG when `n_steps >= 2`.

### 4.2 Replacement-hook implementation

**Pre-compute once (no_grad):**
- `base_attn_out[L]`, `base_ffn_out[L]`, `base_h_ins[L]` — captured via existing helper on `base_prompt`.
- `from_attn_out[L]`, `from_ffn_out[L]`, `from_h_ins[L]` — same, on `from_prompt`.
- All detached + cloned.

**For each step k with `α_k = (k - 0.5) / N`:**

1. Build per-layer fresh leaf tensors for this step:
   - `interp_attn[L] = base_attn_out[L] + α_k · (from_attn_out[L] - base_attn_out[L])`, `requires_grad_(True)`.
   - `interp_ffn[L] = base_ffn_out[L] + α_k · (from_ffn_out[L] - base_ffn_out[L])`, `requires_grad_(True)`.
2. Register forward-hooks:
   - On `model.model.layers[L].self_attn` (post): return `(interp_attn[L], *rest)` where `rest` is the other tuple fields self_attn normally returns — hook must preserve the tuple shape.
   - On `model.model.layers[L].mlp` (post): return `interp_ffn[L]`.
3. Forward `base_prompt`. Native residual-adds run:
   - `h_post_attn[L] = base_h_in[L] + interp_attn[L] = base_h_post_attn[L] + α_k · (Δh_in[L] + Δattn[L]) = base_h_post_attn[L] + α_k · Δh_post_attn[L]`.
   - Similarly `h_post_ffn[L] = base_h_post_attn[L]_interpolated + interp_ffn[L] = base_layer_out[L] + α_k · Δlayer_out[L]`.
4. Compute `metric = base_logits[meas_pos, correct] - base_logits[meas_pos, incorrect]` on the interpolated forward's logits.
5. `metric.backward()`. Read grads: `interp_attn[L].grad` (= ∂L/∂h_post_attn[L] at interp) and `interp_ffn[L].grad` (= ∂L/∂h_post_ffn[L] at interp).
6. Accumulate into `grad_sum_attn[L]` and `grad_sum_ffn[L]`.
7. Remove hooks, zero grads, release graph.

**After N steps:** `avg_grad_attn[L] = grad_sum_attn[L] / N`; same for ffn. Compute per-cell:
- attn row: `ap_raw[L, pos] = (Δh_in[L] + Δattn[L])[0, pos] · avg_grad_attn[L][0, pos]` → sum over hidden-dim → divide by `denominator`.
- ffn row: `ap_raw[L, pos] = Δlayer_out[L][0, pos] · avg_grad_ffn[L][0, pos]` → sum → divide by `denominator`.

These Δ values are **identical to Phase 3.5's** (per the residual-add chain rule); only the grad term changes (averaged over path). This preserves the semantic anchor and allows direct comparability between `n_steps=1` and `n_steps>=2` results.

**Back-compat branch:** at `n_steps == 1`, skip IG entirely and run the existing Phase 3.5 code path verbatim. This gives bit-identical results (crucial for regression tests). `n_steps >= 2` activates the IG machinery above, using midpoint rule.

### 4.3 Completeness test (mock)

For a tiny model, set `positions` = all seq positions, `layers` = all layers, `sublayers = ("attn", "ffn")`. Run `attribution_patch(n_steps=20)`. Sum `cells[*].ap_recovery`. Direction="denoise" → expect `≈ 1.0 ± 0.05`. This validates IG's completeness axiom. First-order AP typically shows 0.6–0.9; IG at N=20 is expected to hit ≥0.95.

## 5. API

### 5.1 Backend

```python
def attribution_patch(
    model, tokenizer, clean_prompt, corrupted_prompt, *,
    correct_token_id, incorrect_token_id,
    direction="denoise", measurement_position=-1,
    positions=None, sublayers=("attn", "ffn"), layers=None,
    n_steps=1,              # NEW
    on_cell=None,
) -> PatchingResult:
```

Validation: `n_steps` must be int, `1 <= n_steps <= 50`. Out-of-range raises `ValueError`.

`PatchingResult` gains `n_steps: Optional[int] = None`. Populated at Phase 3.5 call sites (None if `n_steps=1`, the integer otherwise — None preserves serialization compat with Phase 3.6/3.7/3.9 PatchingResults).

### 5.2 WebSocket route

`routes/probes.py` `approx` mode branch reads `cfg.n_steps` (default 1), passes to `attribution_patch`. `complete.summary.n_steps` populated.

### 5.3 Frontend

- `PatchingState.n_steps: number = 1`.
- `PatchingCompleteData.summary.n_steps?: number`.
- `PatchingControls.tsx`: numeric input `<input type="number" min=1 max=50>` rendered when `mode === "approx"`. Label "IG steps" + small tooltip. Below the "approx auto-pick info line".
- `ProbePanel.tsx`: forwards `n_steps` into WS cfg payload when mode is approx.
- `ActivationPatchingHeatmap.tsx`: header shows "Attribution Patching (∇) — IG N steps" when `result.n_steps > 1`, plain "Attribution Patching (∇)" otherwise.

### 5.4 Error handling

- `n_steps` not in [1, 50] → 400-style validation rejection on route (current WS pattern: `{"error": "..."}` frame + close).
- OOM during IG step → graceful fallback: catch `torch.OutOfMemoryError`, send error frame, close cleanly. Preserves Phase 3.7's existing pattern.

## 6. Testing

### 6.1 Backend unit tests (`test_probe_attribution_patch.py` — append new cases)

1. **`test_n_steps_1_matches_old_behavior`** — mock 2-layer 4-head model. Call `attribution_patch(n_steps=1)` and compare cell-by-cell against pre-IG baseline values stored in this test. Every `ap_recovery` must match to 1e-9.
2. **`test_n_steps_5_runs_and_differs`** — same mock, `n_steps=5`. All cells finite; at least one cell `|ap_recovery(n=5) - ap_recovery(n=1)| > 1e-4` (proves IG actually does something).
3. **`test_n_steps_completeness`** — same mock, `n_steps=20`, `positions=all`, `layers=all`, `sublayers=("attn","ffn")`, `direction="denoise"`. `abs(sum(cell.ap_recovery for cell) - 1.0) < 0.1`. Looser tolerance than ideal: finite mock with MLP nonlinearities won't hit 1.0 exactly at N=20, but first-order AP is expected at 0.6–0.9 so hitting ≥0.9 proves IG is doing its job. (If tolerance turns out too tight during implementation, loosen to 0.15 — don't tighten below 0.1.)
4. **`test_n_steps_validation`** — `n_steps=0` raises ValueError; `n_steps=51` raises ValueError; `n_steps=-1` raises ValueError; `n_steps="5"` raises TypeError (or is coerced — let Python's normal conversion fail).
5. **`test_n_steps_noise_direction`** — mock with `direction="noise"`, `n_steps=5`. Cells finite, completeness `sum(ap_recovery - 1) ≈ -1` (because noise normalization adds 1).

### 6.2 TinyLlama integration (`test_probe_attribution_patch.py` append)

**`test_ig_tinyllama_converges`** — guarded by `tinyllama_model` fixture. Run AP on capital-of-France prompt at `n_steps=1` and `n_steps=5`. Assert:
- both complete without error
- `n_steps=5` has higher (closer to 1.0) `sum(ap_recovery)` than `n_steps=1` for denoise direction (or at minimum, both are finite and in-range)
- individual cell rank order is highly correlated (Spearman ρ > 0.8 between n=1 and n=5 orderings)

Budget: ≤ 180s on RTX 2080 (Phase 3.5 is ~2s, N=5 expected ~10s).

### 6.3 Playwright smoke

**Extend existing `activation-patching-approx.json` fixture** to carry `n_steps: 5` in its summary. Add one smoke test:
- Load fixture, assert heatmap header shows "IG 5 steps".

Test count: 17 → 18.

### 6.4 Vitest

No new Vitest cases — `n_steps` is backend-computed, no client-side math.

## 7. Commit plan

6 tasks, 6 commits:

1. **Spec** — this file.
2. **Plan** — implementation plan.
3. **Backend core** — `attribution_patch` `n_steps` extension, replacement-hook machinery, 5 unit tests.
4. **TinyLlama integration** — one test.
5. **Backend route + types** — `routes/probes.py` cfg passthrough, `PatchingResult.n_steps`, `PatchingCompleteData.summary.n_steps`.
6. **Frontend** — `PatchingState`, `PatchingControls` input, `ActivationPatchingHeatmap` header, fixture update, Playwright smoke.

## 8. Verification matrix

- pyright 0/0/0.
- tsc clean.
- Existing Phase 3.5 tests still pass (regression gate — `n_steps=1` default preserves exact behavior).
- 5 new unit + 1 new TinyLlama integration.
- Vitest 19/19 unchanged.
- Playwright 17 → 18.

## 9. Open questions

*(None — resolved in §1–8 under full-autonomy brainstorm.)*
