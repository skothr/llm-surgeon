# Phase 3.10.2 — IG for Edge AP and Circuit Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `n_steps` parameter to `edge_attribution_patch` and `extract_circuit` for Integrated Gradients (Sundararajan 2017) via a single generalized `_integrated_gradients_loop` helper — the last remaining AP variants without an IG path.

**Architecture:** Extend the Phase 3.10 helper with `capture_reader_grads: bool = False` → when True, runs the same interpolation + forward + backward N times but also registers LN pre-hooks that `retain_grad()` on reader inputs and averages those grads. Helper return signature becomes a two-tuple `(avg_grad_modules, avg_reader_grads)`; existing 3 call sites (Phase 3.10/3.10.1) update to `avg_grad, _ = ...`. `_compute_all_edges` gains `n_steps` and routes reader-grad reads through `avg_reader_grads` when ≥ 2.

**Tech Stack:** PyTorch, FastAPI WebSockets, React/TypeScript, Playwright, Vitest

---

## Task 1: Spec

**Files:**
- Create: `testing/docs/superpowers/specs/2026-04-24-phase3102-edge-circuit-ig.md`

- [ ] **Step 1: Write spec (already done by controller).**

- [ ] **Step 2: Commit.**

```bash
git add testing/docs/superpowers/specs/2026-04-24-phase3102-edge-circuit-ig.md
git commit -m "docs(phase3102): spec — IG for edge AP and circuit"
```

---

## Task 2: Plan

**Files:**
- Create: `testing/docs/superpowers/plans/2026-04-24-phase3102-edge-circuit-ig.md`

- [ ] **Step 1: Write plan (this file).**

- [ ] **Step 2: Commit.**

```bash
git add testing/docs/superpowers/plans/2026-04-24-phase3102-edge-circuit-ig.md
git commit -m "docs(phase3102): plan — IG for edge AP and circuit"
```

---

## Task 3: Python core + unit tests + TinyLlama integrations

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe_edge_ap.py`
- Modify: `testing/tests/test_probe_circuit.py`

### 3A. Extend `_integrated_gradients_loop`

- [ ] **Step 1: Add `capture_reader_grads: bool = False` parameter** to helper signature (probe.py ~line 1094).

- [ ] **Step 2: Change return type** from `Dict[Tuple[int, str], torch.Tensor]` to `Tuple[Dict[Tuple[int, str], torch.Tensor], Dict[Tuple[str, int], torch.Tensor]]`.

- [ ] **Step 3: In the step loop**, when `capture_reader_grads=True`, register forward pre-hooks on:
  - each `model.model.layers[L].input_layernorm` → key `("attn_in", L)`
  - each `model.model.layers[L].post_attention_layernorm` → key `("ffn_in", L)`
  - `model.model.norm` → key `("logits", num_layers)`

  Pre-hook handler: grab `args[0]` (the reader-input tensor), call `.retain_grad()` on it if it requires grad, store the tensor ref in a per-step dict `step_readers: Dict[Tuple[str, int], torch.Tensor]`.

  These pre-hooks must run on the INTERPOLATED forward (where the interpolation hooks have replaced module outputs) — same hook lifetime scope.

- [ ] **Step 4: After each step's backward**, accumulate:
  ```python
  for reader_key, reader_tensor in step_readers.items():
      if reader_tensor.grad is not None:
          if reader_key not in grad_sum_reader:
              grad_sum_reader[reader_key] = torch.zeros_like(reader_tensor.grad.detach())
          grad_sum_reader[reader_key] += reader_tensor.grad.detach()
  ```

- [ ] **Step 5: At end**, build `avg_reader_grads: Dict[Tuple[str, int], torch.Tensor]` by dividing each accumulated tensor by `n_steps`.

- [ ] **Step 6: Return tuple** `(avg_grad, avg_reader_grads)` where `avg_reader_grads` is `{}` when `capture_reader_grads=False`.

### 3B. Update existing 3 call sites

- [ ] **Step 7: In `attribution_patch`** (probe.py ~line 1342), change:
  ```python
  avg_grad = _integrated_gradients_loop(...)
  ```
  to:
  ```python
  avg_grad, _ = _integrated_gradients_loop(...)
  ```

- [ ] **Step 8: In `attribution_patch_per_head`** (probe.py ~line 1539), same change.

- [ ] **Step 9: In `attribution_patch_per_neuron`** (probe.py ~line 1753), same change.

### 3C. Extend `_compute_all_edges` with `n_steps`

- [ ] **Step 10: Add `n_steps: int = 1`** parameter to `_compute_all_edges` (probe.py ~line 1862).

- [ ] **Step 11: Refactor the grad-enabled backward**: When `n_steps == 1`, existing `metric.backward()` path runs. When `n_steps >= 2`:
  - Still capture `base_captured`, `base_cz`, `reader_inputs`, `base_h_ins` from the grad-enabled forward (needed for Δffn_out and Δconcat_z reconstruction, and h_ins for IG helper). But `base_h_ins` is the 2nd return value — update the unpack.
  - Do NOT call `metric.backward()`.
  - Call:
    ```python
    _, avg_reader_grads = _integrated_gradients_loop(
        model=model, tokenizer=tokenizer, base_prompt=base_prompt,
        base_captured=base_captured, base_h_ins=base_h_ins,
        from_states=from_states, from_h_ins={},  # from_h_ins only used for ffn reconstruction; compute similarly
        sublayers=("attn", "ffn"),
        layers=layers,
        measurement_position=meas_pos,
        correct_token_id=correct_token_id,
        incorrect_token_id=incorrect_token_id,
        n_steps=n_steps,
        capture_reader_grads=True,
    )
    ```

  Caveat on `from_h_ins`: the existing _compute_all_edges unpack at line 1926-1933 discards h_ins (`_` in position 2). Need to change that unpack to `from_captured_raw, from_h_ins, ...` so we have from_h_ins for the IG helper's ffn_out reconstruction.

- [ ] **Step 12: In the reader-enumeration loop** (line 1987+), route `grad_r`:
  ```python
  if n_steps == 1:
      grad_r_tensor = reader_tensor.grad
      if grad_r_tensor is None:
          continue
  else:
      grad_r_tensor = avg_reader_grads.get(reader_key)
      if grad_r_tensor is None:
          continue
  grad_r = grad_r_tensor[0, pos].detach() if n_steps == 1 else grad_r_tensor[0, pos]
  ```

  Pyright gotcha (same lesson as Phase 3.10.1): declare as non-Optional `torch.Tensor` in one branch, use `maybe_` intermediate in the other. Example:
  ```python
  maybe_grad = (
      reader_tensor.grad if n_steps == 1
      else avg_reader_grads.get(reader_key)
  )
  if maybe_grad is None:
      continue
  grad_r_full: torch.Tensor = maybe_grad
  grad_r = grad_r_full[0, pos].detach() if n_steps == 1 else grad_r_full[0, pos]
  ```

### 3D. Extend public APIs

- [ ] **Step 13: Add `n_steps: int = 1`** to `edge_attribution_patch` (probe.py ~line 2082) with validation (isinstance int, [1, 50]). Forward to `_compute_all_edges(..., n_steps=n_steps)`.

- [ ] **Step 14: Populate `result.n_steps = n_steps if n_steps > 1 else None`** on return.

- [ ] **Step 15: Add `n_steps: int = 1`** to `extract_circuit` with the same validation + forwarding.

- [ ] **Step 16: Populate `result.n_steps`** on circuit's return.

### 3E. Unit tests

- [ ] **Step 17: Add `TestEdgeAPIntegratedGradients` class** to `test_probe_edge_ap.py` with `test_edge_ap_n_steps_converges`:
  - Reuse existing `_make_edge_fixtures` (or whichever fixture helper exists).
  - Run `edge_attribution_patch(n_steps=1)` and `edge_attribution_patch(n_steps=10)` with `top_k_edges=50`.
  - Assert `r_10.n_steps == 10`, `r_1.n_steps is None`.
  - Assert all cells finite.
  - Assert max abs diff between matched cells > 1e-4.
  - Assert `n_steps` validation: `n_steps=0` raises, `n_steps=51` raises.

- [ ] **Step 18: Add `TestCircuitIntegratedGradients` class** to `test_probe_circuit.py` with `test_circuit_n_steps_converges`:
  - Same pattern. Assert at least one of: `in_circuit` labels differ for ≥1 edge, OR top-20 edge ranking Spearman > 0.5.

### 3F. TinyLlama integrations

- [ ] **Step 19: Add `test_edge_ig_tinyllama`** — capital-of-France prompt, `top_k_edges=50`, fp16, `n_steps=1` vs `n_steps=5`. Compute top-20 Spearman ρ between rankings by `|ap_recovery|`. Assert ρ > 0.5.

- [ ] **Step 20: Add `test_circuit_ig_tinyllama`** — same prompt, τ=0.05, `top_k_candidates=200`, fp16, `n_steps=1` vs `n_steps=5`. Top-20 Spearman > 0.5.

### 3G. Verify + commit

- [ ] **Step 21: Run pyright** on probe.py + both edited test files.

- [ ] **Step 22: Run pytest** on test_probe_edge_ap.py + test_probe_circuit.py + regression check on test_probe_attribution_patch.py + test_probe_per_head_ap.py + test_probe_per_neuron_ap.py (all three must still pass — helper signature change).

- [ ] **Step 23: Commit.**

```bash
git add testing/llm_surgeon/probe.py \
        testing/tests/test_probe_edge_ap.py \
        testing/tests/test_probe_circuit.py
git commit -m "feat(probe): IG for edge AP and circuit — Phase 3.10.2"
```

---

## Task 4: Backend WS route + frontend visibility

**Files:**
- Modify: `testing/gui/backend/routes/probes.py`
- Modify: `testing/gui/frontend/src/components/PatchingControls.tsx`
- Modify: `testing/gui/frontend/src/components/ProbePanel.tsx`

### 4A. Backend

- [ ] **Step 1: In routes/probes.py edge mode branch**, add `n_steps=_n_steps_edge` kwarg to `edge_attribution_patch` call. Use the existing `n_steps` parsing from Phase 3.10 (lines 932-940 validate once for all modes). Pattern:
  ```python
  _n_steps_edge: int = n_steps
  # ... later, in edge branch:
  result = edge_attribution_patch(..., n_steps=_n_steps_edge)
  ```

- [ ] **Step 2: In routes/probes.py circuit mode branch**, same — add `n_steps=_n_steps_circuit` kwarg.

- [ ] **Step 3: Verify summary frame `complete.summary.n_steps`** populates correctly for edge + circuit — it already reads `result.n_steps` generically so no route changes needed.

### 4B. Frontend

- [ ] **Step 4: In PatchingControls.tsx**, extend the IG-steps input visibility condition:
  ```tsx
  const showNSteps = ["approx", "approx_head", "approx_neuron", "edge", "circuit"].includes(state.mode);
  ```

- [ ] **Step 5: In ProbePanel.tsx**, ensure `cfg.n_steps` is forwarded in the WS payload for edge and circuit modes. Check existing else-if structure; extend the same branch used for approx*.

### 4C. Verify + commit

- [ ] **Step 6: Run tsc** from testing/gui/frontend.

- [ ] **Step 7: Run pyright** on routes/probes.py.

- [ ] **Step 8: Run vitest** (should stay 19/19).

- [ ] **Step 9: Run Playwright** (should stay 18/18).

- [ ] **Step 10: Commit.**

```bash
git add testing/gui/backend/routes/probes.py \
        testing/gui/frontend/src/components/PatchingControls.tsx \
        testing/gui/frontend/src/components/ProbePanel.tsx
git commit -m "feat(gui): Phase 3.10.2 — n_steps for edge and circuit modes"
```

---

## Verification matrix (end-of-plan)

- pyright 0/0/0 across probe.py, routes/probes.py, all test files.
- tsc clean.
- Phase 3.7 edge AP regressions preserved.
- Phase 3.8 circuit regressions preserved.
- Phase 3.10 + 3.10.1 regressions preserved (3 call sites adapted).
- 2 new unit tests + 2 new TinyLlama integrations pass.
- Vitest 19/19, Playwright 18/18.

---

## Pyright gotcha reminder

Phase 3.10.1 hit a narrowing bug at per-neuron's branch:
```python
# BAD: Optional annotation blocks flow-based None elimination
grad_ffn_out_tensor: Optional[torch.Tensor] = base_ffn_out.grad
```

Fix pattern (also used in this plan's Step 12):
```python
# GOOD: non-Optional in happy path, local maybe_ intermediate in the branch
if n_steps == 1:
    grad_tensor: torch.Tensor = base_ffn_out.grad  # narrowed by prior `is not None` check
else:
    maybe_grad = avg_grad.get((L, "ffn"))
    if maybe_grad is None:
        continue
    grad_tensor = maybe_grad
```
