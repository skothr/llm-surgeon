# Phase 3.10.2 — Integrated Gradients for Edge AP and Circuit Extraction

**Date:** 2026-04-24
**Status:** Spec (autonomous ship — matches Phase 3.10/3.10.1 pattern)
**Depends on:** Phase 3.7 (`edge_attribution_patch` + `_compute_all_edges`), Phase 3.8 (`extract_circuit`), Phase 3.10 (`_integrated_gradients_loop` helper), Phase 3.10.1 (per-head/per-neuron IG via chain-rule reuse)

## 1. Motivation

Phase 3.10 and 3.10.1 established IG for block-level AP, per-head AP, and per-neuron AP — all three via a single `_integrated_gradients_loop` helper plus chain-rule consumers. Edge AP (Phase 3.7) and circuit extraction (Phase 3.8) are the last remaining gradient-based AP variants without an IG path, and they share a single reader-side gradient surface: `reader_inputs[key].grad` at `input_layernorm` / `post_attention_layernorm` / `model.norm` pre-hooks.

The math for IG here is:
```
AP_edge(w → r, pos) = (Δwrite_w · avg_grad_reader_r).sum() / D
```
Only `grad_reader_r` is averaged; writer-side Δs are static `(from − base)` quantities.

## 2. Goals

- **G1.** Extend `_integrated_gradients_loop` with `capture_reader_grads: bool = False`. When True, registers the same LN pre-hooks as `_capture_residual_stream_with_grad` during each IG step, `retain_grad()`s the reader inputs, accumulates their grads, and returns them averaged.
- **G2.** Helper return signature changes from `Dict[Tuple[int, str], torch.Tensor]` → `Tuple[Dict[Tuple[int, str], torch.Tensor], Dict[Tuple[str, int], torch.Tensor]]`. Second dict is empty when `capture_reader_grads=False`. Existing 3 call sites (Phase 3.10/3.10.1) updated to unpack `avg_grad, _ = _integrated_gradients_loop(...)`.
- **G3.** `edge_attribution_patch(..., n_steps: int = 1)`. When `n_steps=1` preserves existing Phase 3.7 behavior bit-identical. When `n_steps >= 2`, `_compute_all_edges` routes reader-grad reads to averaged versions from the IG helper.
- **G4.** `extract_circuit(..., n_steps: int = 1)`. Same pattern — signature passthrough through `_compute_all_edges`.
- **G5.** `_compute_all_edges` gains `n_steps: int = 1` parameter. When `n_steps >= 2`, after capturing `base_captured`/`base_cz`/`reader_inputs` via the existing forward, runs the IG helper with `capture_reader_grads=True` and `sublayers=("attn", "ffn")`, replaces `reader_tensor.grad[0, pos]` reads with `avg_reader_grads[reader_key][0, pos]` in the enumeration loop.
- **G6.** Backend WS `edge` and `circuit` branches in `routes/probes.py` parse and forward `cfg.n_steps` identically to Phase 3.10's `approx` branch. `complete.summary.n_steps` populated.
- **G7.** Frontend `PatchingControls.tsx` IG-steps input visibility condition expands from `["approx", "approx_head", "approx_neuron"]` to include `"edge"` and `"circuit"`. `ProbePanel.tsx` forwards `n_steps` in WS cfg for edge/circuit modes (already forwarded for approx* modes).
- **G8.** Tests: 2 mock unit tests (edge + circuit) + 2 TinyLlama integration tests (top-20 Spearman ≥ 0.5 between n=1 and n=5).

## 3. Non-Goals

- **N1.** New IG helper variants — single generalized helper handles both cases.
- **N2.** Expanding Phase 3.8's τ filter / reverse-BFS math. Circuit-extraction logic is unchanged; only the edge scores it consumes get higher fidelity under IG.
- **N3.** GQA/MQA support. Deferred — TinyLlama is MHA; LLaMA-3-8B validation is its own phase.
- **N4.** New Playwright smokes. Existing Phase 3.7 (edge) and Phase 3.8 (circuit) smokes cover panel rendering. IG annotation visibility is covered by Phase 3.10's smoke (the annotation `<span>` lives in `ActivationPatchingHeatmap` which isn't the edge/circuit render path — no new annotation needed; the `complete.summary.n_steps` echo is sufficient for backend coverage).

## 4. Math

### 4.1 Reader gradient under IG

Phase 3.7 captures reader inputs at LN pre-hooks: `input_layernorm` (attn_in, read by attn sublayer), `post_attention_layernorm` (ffn_in, read by FFN), `model.norm` (logits, read by LM head). Their gradients are `∂L/∂residual_stream` at those points.

Under IG with N steps replacing module outputs `self_attn[L]` and `mlp[L]` with `base + α_k · (from − base)`:
```
avg_grad_reader[r] = (1/N) Σ_k grad_reader_r_at_step_k
```
where each `grad_reader_r_at_step_k` is the backward gradient through the interpolated network. Same α_k = `(k + 0.5) / N` midpoint rule as Phase 3.10.

### 4.2 Edge score composition

Phase 3.7 per-edge score:
```
AP_edge(w → r, pos) = (Δwrite_w[0, pos] · grad_reader_r[0, pos]).sum() / D
```

Under IG: `grad_reader_r` → `avg_grad_reader_r`. Everything else (Δembed, Δffn_out, Δconcat_z) is unchanged.

For per-head attn writers: Phase 3.7 already chain-rules through `W_O`:
```
grad_z_r = grad_reader_r @ W_O
```
Under IG: `grad_z_r = avg_grad_reader_r @ W_O` (W_O is a constant; same as Phase 3.10.1 per-head chain-rule reuse).

### 4.3 Circuit extraction

`extract_circuit` calls `_compute_all_edges` to get the edge list, then sorts, filters by τ, reverse-BFSes from `logits`. None of the postprocessing depends on how edge scores were computed (first-order vs IG). Pure passthrough.

## 5. API

### 5.1 Helper

```python
def _integrated_gradients_loop(
    *,
    model, tokenizer, base_prompt, base_captured, base_h_ins,
    from_states, from_h_ins, sublayers, layers,
    measurement_position, correct_token_id, incorrect_token_id,
    n_steps: int,
    capture_reader_grads: bool = False,    # NEW
) -> Tuple[
    Dict[Tuple[int, str], torch.Tensor],   # avg_grad (module outputs)
    Dict[Tuple[str, int], torch.Tensor],   # avg_reader_grads (empty if flag off)
]:
```

Return-type change is breaking for the 3 existing call sites; they update to `avg_grad, _ = _integrated_gradients_loop(...)` (trivial).

### 5.2 Edge / circuit APIs

```python
def edge_attribution_patch(..., n_steps: int = 1, ...) -> PatchingResult
def extract_circuit(..., n_steps: int = 1, ...) -> PatchingResult
```

Validate `n_steps` via `isinstance(int) and 1 <= n_steps <= 50`. Result's `n_steps` field populated as `n_steps if n_steps > 1 else None`.

### 5.3 _compute_all_edges

```python
def _compute_all_edges(
    model, tokenizer, clean_prompt, corrupted_prompt, *,
    correct_token_id, incorrect_token_id, direction,
    measurement_position, positions, layers,
    n_steps: int = 1,    # NEW
) -> Tuple[...]:
```

When `n_steps >= 2`:
1. Do the existing clean/corrupted no-grad captures (for Δembed, Δffn_out, Δconcat_z, baseline logits) — unchanged.
2. Do the grad-enabled `base_prompt` capture (for base_captured, base_cz, reader_inputs refs, base_h_ins) — but SKIP the `metric.backward()` call.
3. Call `_integrated_gradients_loop(capture_reader_grads=True, sublayers=("attn", "ffn"))` with the captured state.
4. In the reader-enumeration loop, route `grad_r` reads to `avg_reader_grads[reader_key][0, pos]` instead of `reader_tensor.grad[0, pos]`.

When `n_steps == 1`: existing behavior, bit-identical.

### 5.4 WebSocket route

In `routes/probes.py`, `edge` and `circuit` mode branches gain `n_steps` parsing (bounds [1, 50]) + forwarding — lifted from Phase 3.10's `approx` branch. The validation at lines 932-940 already runs universally (per Phase 3.10); only the kwarg forwarding is new per branch.

### 5.5 Frontend

Single condition extension:
```tsx
// PatchingControls.tsx
const showNSteps = ["approx", "approx_head", "approx_neuron", "edge", "circuit"].includes(state.mode);
```

`ProbePanel.tsx` ensures `cfg.n_steps` is included in the WS payload for edge and circuit modes (may already be via shared code path — verify).

## 6. Testing

### 6.1 Edge mock unit (`test_edge_ap_n_steps_converges`)

Reuse Phase 3.7's `_make_edge_fixtures`. Call `edge_attribution_patch(..., n_steps=1)` and `n_steps=10`. Assert:
- `r_10.n_steps == 10`, `r_1.n_steps is None`
- all cells finite
- at least one cell's `ap_recovery` differs between runs by > 1e-4
- sort order preserved (highest-magnitude cells present in both)

### 6.2 Circuit mock unit (`test_circuit_n_steps_converges`)

Reuse Phase 3.8's `_make_circuit_fixtures`. Same shape of assertions; additionally:
- `in_circuit` labels can differ (IG may reshuffle which edges cross τ).

### 6.3 Edge TinyLlama (`test_edge_ig_tinyllama`)

Capital-of-France prompt. Run `n_steps=1` and `n_steps=5`, `top_k_edges=50`. Top-20 Spearman ρ > 0.5 between the two rankings (by `|ap_recovery|`). fp16 for compute budget (matches Phase 3.7's EAP fp16 pattern).

### 6.4 Circuit TinyLlama (`test_circuit_ig_tinyllama`)

Same prompt. τ=0.05, `top_k_candidates=200`, `n_steps=1` vs `n_steps=5`. Assert: at least one edge's `in_circuit` flag differs OR top-20 edge ranking Spearman > 0.5.

### 6.5 No Playwright additions

Phase 3.7 and 3.8 smokes cover edge/circuit panel rendering. n_steps plumbing coverage for edge/circuit is implicit via the Phase 3.10 smoke's fixture (which renders the IG annotation span on the `approx` heatmap). A dedicated smoke would require wiring the IG annotation into `EdgeAttributionPanel` / `CircuitPanel`, which is out of scope here — the backend passes `n_steps` through; UI rendering of the annotation is deferred (Phase 3.10.3 if it becomes worth surfacing).

Test count: 19 vitest / 18 Playwright stays unchanged.

## 7. Commit plan

4 commits:
1. Spec — this file.
2. Plan.
3. Python core + unit tests + TinyLlama integrations (probe.py + test file edits — ship as one commit because helper signature change touches 3 existing sites + edge AP + circuit in lockstep).
4. Backend route + frontend visibility expansion.

## 8. Verification matrix

- pyright 0/0/0 across probe.py + routes/probes.py + test files.
- tsc clean.
- Phase 3.7 (edge) and Phase 3.8 (circuit) regressions preserved (n_steps=1 bit-identical; sum invariants preserved).
- Phase 3.10 and 3.10.1 regressions preserved (3 call sites updated to unpack tuple).
- 2 new mock unit tests + 2 new TinyLlama integrations pass.
- Vitest 19/19, Playwright 18/18 unchanged.

## 9. Key design decisions

- **Tuple return over heterogeneous dict**: keeps helper's typing clean — two dicts with distinct key shapes (`Tuple[int, str]` for modules vs `Tuple[str, int]` for readers). Union types on keys would leak into every call site.
- **No `metric.backward()` in IG path**: when `n_steps >= 2`, the existing first-order backward is REPLACED (not augmented) by the IG loop. Avoids double-computing and grad-accumulation bugs from residual `.grad` state.
- **Writer-side Δs remain first-order**: mathematically correct. `Δembed`, `Δffn_out`, `Δconcat_z` are static `(from − base)` quantities — there's nothing to path-integrate over them. IG only makes sense for the gradient side where the network's response depends on intermediate activations.
- **No Phase 3.10.3 for UI annotation**: explicitly deferred. The `complete.summary.n_steps` echo covers backend observability; UI annotation on `EdgeAttributionPanel` / `CircuitPanel` can be added later if users request it.
