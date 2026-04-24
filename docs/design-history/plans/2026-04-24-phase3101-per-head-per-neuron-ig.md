# Phase 3.10.1 — Per-Head and Per-Neuron IG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `attribution_patch_per_head` (Phase 3.6) and `attribution_patch_per_neuron` (Phase 3.9) with an `n_steps: int = 1` parameter that activates Integrated Gradients via the existing Phase 3.10 `_integrated_gradients_loop` helper.

**Architecture:** `n_steps=1` preserves existing Phase 3.6/3.9 behavior bit-identical via a conditional branch. `n_steps >= 2` calls `_integrated_gradients_loop(sublayers=("attn","ffn"))` — the Phase 3.10 helper, unchanged — and substitutes `avg_grad[(L, "attn")]` for `base_captured[(L, "attn")].grad` in the per-head chain rule, and `avg_grad[(L, "ffn")]` for `base_captured[(L, "ffn_out")].grad` in the per-neuron chain rule. No new infrastructure.

**Tech Stack:** PyTorch (reusing existing autograd machinery), FastAPI (two WS mode branches — `approx_head`, `approx_neuron` — get cfg passthrough), React+TS (one input-visibility condition expansion).

---

## Files

- **Modify:** `testing/llm_surgeon/probe.py:1418-1607` (`attribution_patch_per_head`).
- **Modify:** `testing/llm_surgeon/probe.py:1610-1810` (`attribution_patch_per_neuron`).
- **Modify:** `testing/tests/test_probe_per_head_ap.py` — append 1 unit test + 1 TinyLlama.
- **Modify:** `testing/tests/test_probe_per_neuron_ap.py` — append 1 unit test + 1 TinyLlama.
- **Modify:** `testing/gui/backend/routes/probes.py` — `approx_head` and `approx_neuron` mode branches gain `n_steps` parsing identical to `approx` branch.
- **Modify:** `testing/gui/frontend/src/components/PatchingControls.tsx` — expand `state.mode === "approx"` condition to `["approx", "approx_head", "approx_neuron"].includes(state.mode)` for the IG-steps input.
- **Modify:** `testing/gui/frontend/src/components/ProbePanel.tsx` — forward `n_steps` in WS cfg for the two new modes too.

---

## Task 1: Per-head IG in `attribution_patch_per_head`

- [ ] **Step 1: Extend signature.**

Find `attribution_patch_per_head` signature near line 1418. Insert `n_steps: int = 1,` immediately before `on_cell:`. Add the same validation line `attribution_patch` uses (at roughly line 1258 in probe.py for reference):

```python
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")
```

Place the validation near the other validations at the top of the function body (around line 1456).

- [ ] **Step 2: Branch on `n_steps`.**

Find the single-backward block around line 1526-1530:
```python
        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        metric.backward()
```

Replace with:
```python
        if n_steps == 1:
            metric = (
                base_logits[meas_pos, correct_token_id]
                - base_logits[meas_pos, incorrect_token_id]
            )
            metric.backward()
            avg_grad: Optional[Dict[Tuple[int, str], torch.Tensor]] = None
        else:
            # Need from_h_ins to reconstruct ffn_out for the IG helper.
            # Capture separately in a no_grad forward (same cost as the
            # existing from-side capture; we only need h_ins).
            with torch.no_grad():
                _, from_h_ins_for_ig, _, _, _, _, _ = \
                    _capture_residual_stream_with_grad(
                        model, tokenizer, from_prompt,
                        sublayers=sublayers, layers=layers,
                    )
                _, base_h_ins_for_ig, _, _, _, _, _ = \
                    _capture_residual_stream_with_grad(
                        model, tokenizer, base_prompt,
                        sublayers=sublayers, layers=layers,
                    )
                from_h_ins_ig = {
                    k: v.detach().clone() for k, v in from_h_ins_for_ig.items()
                }
                base_h_ins_ig = {
                    k: v.detach().clone() for k, v in base_h_ins_for_ig.items()
                }
            avg_grad = _integrated_gradients_loop(
                model=model,
                tokenizer=tokenizer,
                base_prompt=base_prompt,
                base_captured=base_captured,
                base_h_ins=base_h_ins_ig,
                from_states=from_states,
                from_h_ins=from_h_ins_ig,
                sublayers=sublayers,
                layers=layers,
                measurement_position=meas_pos,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
                n_steps=n_steps,
            )
```

NOTE on h_ins: Phase 3.6 currently doesn't capture h_ins (the existing helper call at line 1501 doesn't need them for per-head math). But `_integrated_gradients_loop` requires `base_h_ins` and `from_h_ins` to reconstruct `ffn_out` for the FFN-row IG. We capture them in no_grad before entering the IG loop. This adds ~2 tiny extra forward passes for the h_in capture but keeps the helper unchanged.

**Simpler alternative worth preferring:** modify the existing `from` and `base` captures (lines 1491-1506) to also return h_ins. The captures already happen; we just need to store `from_h_ins` and `base_h_ins` alongside `from_states` and `base_captured`. This avoids the two extra forward passes. Do this if it's a clean edit; otherwise fall back to the no_grad capture above.

- [ ] **Step 3: Substitute `avg_grad` in cell loops.**

In the FFN-anchor loop around line 1540-1555:
```python
            base_ffn = base_captured[(L, "ffn")]
            ffn_grad = base_ffn.grad
```
Change to:
```python
            base_ffn = base_captured[(L, "ffn")]
            if n_steps == 1:
                ffn_grad = base_ffn.grad
            else:
                assert avg_grad is not None
                ffn_grad = avg_grad.get((L, "ffn"))
```

In the per-head loop around line 1558-1565:
```python
            attn_out_grad = base_captured[(L, "attn")].grad
            if attn_out_grad is None:
                continue
            W_O: torch.Tensor = model.model.layers[L].self_attn.o_proj.weight
            concat_z_grad = attn_out_grad[0] @ W_O
```
Change to:
```python
            if n_steps == 1:
                attn_out_grad = base_captured[(L, "attn")].grad
            else:
                assert avg_grad is not None
                attn_out_grad = avg_grad.get((L, "attn"))
            if attn_out_grad is None:
                continue
            W_O: torch.Tensor = model.model.layers[L].self_attn.o_proj.weight
            concat_z_grad = attn_out_grad[0] @ W_O
```

- [ ] **Step 4: Populate `n_steps` on return.**

Find the return around line 1597-1607. Add `n_steps=(n_steps if n_steps > 1 else None),` as a new kwarg.

- [ ] **Step 5: Pyright.**

Run: `testing/.venv/bin/python -m pyright testing/llm_surgeon/probe.py`
Expected: 0/0/0.

- [ ] **Step 6: Regression — existing per-head tests pass.**

Run: `testing/.venv/bin/python -m pytest testing/tests/test_probe_per_head_ap.py -v`
Expected: all existing tests pass (n_steps=1 preserves behavior).

No commit yet — combine with Task 2.

---

## Task 2: Per-neuron IG in `attribution_patch_per_neuron`

- [ ] **Step 1: Extend signature.**

Find `attribution_patch_per_neuron` signature near line 1610. Insert `n_steps: int = 1,` immediately before `on_cell:`. Add validation near the other validations at the top of the function body:

```python
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")
```

- [ ] **Step 2: Branch on `n_steps`.**

Find the single-backward block in per-neuron body (where `metric.backward()` is called). Apply the same `if n_steps == 1 / else _integrated_gradients_loop(...)` pattern as Task 1 Step 2. Capture h_ins if not already captured.

- [ ] **Step 3: Substitute `avg_grad`.**

Find where per-neuron reads `base_captured[(L, "ffn_out")].grad` (the chain-rule `grad_act = grad_ffn_out @ W_down` line). Change to use `avg_grad[(L, "ffn")]` when `n_steps >= 2`. Note: Phase 3.9 captures via `capture_ffn_out=True` flag which adds `(L, "ffn_out")` keys to `base_captured`. The IG helper produces `avg_grad[(L, "ffn")]` which is the SAME quantity (mlp post-hook output averaged) — use it directly.

- [ ] **Step 4: Populate `n_steps` on return.**

Add `n_steps=(n_steps if n_steps > 1 else None),` kwarg.

- [ ] **Step 5: Pyright.**

Run: `testing/.venv/bin/python -m pyright testing/llm_surgeon/probe.py`
Expected: 0/0/0.

- [ ] **Step 6: Regression.**

Run: `testing/.venv/bin/python -m pytest testing/tests/test_probe_per_neuron_ap.py -v`
Expected: all existing tests pass.

- [ ] **Step 7: Commit Tasks 1+2.**

```bash
git add testing/llm_surgeon/probe.py
git commit -m "$(cat <<'EOF'
feat(probe): n_steps IG for attribution_patch_per_head and _per_neuron

Reuses Phase 3.10's _integrated_gradients_loop. Per-head chain-rules
avg_grad[(L,"attn")] @ W_O; per-neuron chain-rules avg_grad[(L,"ffn")]
@ W_down. n_steps=1 branch preserves Phase 3.6/3.9 behavior exactly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Unit tests + TinyLlama for per-head

- [ ] **Step 1: Write `test_per_head_n_steps_converges`.**

Append to `testing/tests/test_probe_per_head_ap.py`. Use the existing per-head mock fixture (read the file to find its exact name). Template:

```python
def test_per_head_n_steps_converges(existing_per_head_fixture):
    model, tok = existing_per_head_fixture  # adjust to actual fixture name
    import math
    kwargs = dict(
        clean_prompt="A B C D",
        corrupted_prompt="A B C E",
        correct_token_id=1,
        incorrect_token_id=2,
        direction="denoise",
    )
    r1 = attribution_patch_per_head(model, tok, **kwargs, n_steps=1)
    r10 = attribution_patch_per_head(model, tok, **kwargs, n_steps=10)
    assert r1.n_steps is None
    assert r10.n_steps == 10
    for c in r10.cells:
        assert math.isfinite(c["ap_recovery"])
    assert any(
        abs(c1["ap_recovery"] - c10["ap_recovery"]) > 1e-4
        for c1, c10 in zip(r1.cells, r10.cells)
    )
```

Note: use explicit kwargs in `attribution_patch_per_head(...)` calls, NOT `**dict` unpacking — pyright can't type-check heterogeneous dicts with `**` into typed kwargs (hit this in Phase 3.10). If the existing tests use `**kwargs` anyway, match that pattern; but extract a local helper `def _run(n): ...` if type errors appear.

- [ ] **Step 2: Write `test_per_head_ig_tinyllama`.**

Append:
```python
def test_per_head_ig_tinyllama(tinyllama_model):  # adjust fixture name
    import math
    import scipy.stats
    import numpy as np
    model, tokenizer = tinyllama_model
    correct_id = tokenizer(" Paris", add_special_tokens=False)["input_ids"][0]
    incorrect_id = tokenizer(" Moscow", add_special_tokens=False)["input_ids"][0]

    def _run(n: int):
        return attribution_patch_per_head(
            model, tokenizer,
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Russia is",
            correct_token_id=correct_id,
            incorrect_token_id=incorrect_id,
            direction="denoise",
            measurement_position=-1,
            n_steps=n,
        )

    r1 = _run(1)
    r5 = _run(5)
    assert r5.n_steps == 5
    for c in r5.cells:
        assert math.isfinite(c["ap_recovery"])

    def keyfn(c):
        return (c["layer"], c["unit"], c["position"])
    r1_map = {keyfn(c): c["ap_recovery"] for c in r1.cells}
    r5_map = {keyfn(c): c["ap_recovery"] for c in r5.cells}
    shared = sorted(set(r1_map) & set(r5_map))
    r1_scores = np.array([r1_map[k] for k in shared])
    r5_scores = np.array([r5_map[k] for k in shared])
    top20 = np.argsort(-np.abs(r1_scores))[:20]
    rho = float(scipy.stats.spearmanr([r1_scores[i] for i in top20], [r5_scores[i] for i in top20]).statistic)
    print(f"\nper-head top-20 Spearman(n=1, n=5) = {rho:.3f}")
    assert rho > 0.5, f"per-head IG re-ranked top-20 cells too aggressively (ρ={rho:.3f})"
```

Use the SAME skipif guard for GPU/TinyLlama that existing TinyLlama tests in the file use — copy verbatim.

- [ ] **Step 3: Run tests.**

```bash
testing/.venv/bin/python -m pytest testing/tests/test_probe_per_head_ap.py -v -k "n_steps or ig_tinyllama"
```
Expected: 2 PASS (or TinyLlama SKIPPED if no GPU).

- [ ] **Step 4: Pyright.**

Run: `testing/.venv/bin/python -m pyright testing/tests/test_probe_per_head_ap.py`
Expected: 0/0/0.

---

## Task 4: Unit tests + TinyLlama for per-neuron

- [ ] **Step 1: Write `test_per_neuron_n_steps_converges`.**

Append to `testing/tests/test_probe_per_neuron_ap.py`. Mirror Task 3 Step 1 but:
- call `attribution_patch_per_neuron` instead
- assert `r.n_neurons == intermediate_size` and `r.n_steps == 10`
- compare cells by `(layer, neuron, position)` key (Phase 3.9 cell keys)

- [ ] **Step 2: Write `test_per_neuron_ig_tinyllama`.**

Mirror Task 3 Step 2 but call `attribution_patch_per_neuron` with `top_k_neurons=50`, keyfn `(c["layer"], c["neuron"], c["position"])`. Assert top-20 Spearman > 0.5.

- [ ] **Step 3: Run.**

```bash
testing/.venv/bin/python -m pytest testing/tests/test_probe_per_neuron_ap.py -v -k "n_steps or ig_tinyllama"
```
Expected: 2 PASS.

- [ ] **Step 4: Pyright on both test files.**

```bash
testing/.venv/bin/python -m pyright testing/tests/test_probe_per_head_ap.py testing/tests/test_probe_per_neuron_ap.py
```
Expected: 0/0/0.

- [ ] **Step 5: Commit Tasks 3+4.**

```bash
git add testing/tests/test_probe_per_head_ap.py testing/tests/test_probe_per_neuron_ap.py
git commit -m "$(cat <<'EOF'
test(probe): IG unit + TinyLlama tests for per-head and per-neuron AP

Each variant gains: 1 mock unit test (n_steps=10 runs, n_steps!=n_steps=1
in at least one cell) + 1 TinyLlama integration (top-20 Spearman > 0.5
between n_steps=1 and n_steps=5 rankings).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Backend WS route — `approx_head` and `approx_neuron` n_steps passthrough

- [ ] **Step 1: Locate the two branches.**

In `testing/gui/backend/routes/probes.py`, find the `cfg.mode == "approx_head"` branch and the `cfg.mode == "approx_neuron"` branch.

- [ ] **Step 2: Copy Phase 3.10's `n_steps` parsing to both.**

Phase 3.10's `approx` branch now has `n_steps` parsing + validation + forwarding. Mirror that logic in the `approx_head` and `approx_neuron` branches:
- Parse `cfg.get("n_steps", 1)`, validate, send error frame on failure.
- Pass `n_steps=n_steps,` into the respective `attribution_patch_per_head` or `attribution_patch_per_neuron` call.
- Add `"n_steps": result.n_steps` to the `complete.summary` dict in both branches.

- [ ] **Step 3: Pyright.**

Run: `testing/.venv/bin/python -m pyright testing/gui/backend/routes/probes.py`
Expected: 0/0/0.

No commit yet — combine with Task 6.

---

## Task 6: Frontend — expand n_steps input visibility

- [ ] **Step 1: Extend `PatchingControls.tsx` visibility condition.**

Find the `state.mode === "approx"` conditional around the `n_steps` input (added in Phase 3.10). Change to:
```tsx
{["approx", "approx_head", "approx_neuron"].includes(state.mode) && (
  <label ... >IG steps: <input ... /></label>
)}
```

- [ ] **Step 2: Extend `ProbePanel.tsx` cfg forwarding.**

Find where `n_steps` is added to the WS cfg payload (Phase 3.10 added it conditionally for `state.mode === "approx"`). Expand the condition identically.

- [ ] **Step 3: tsc.**

Run: `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit`
Expected: clean.

- [ ] **Step 4: Run Vitest + Playwright regression.**

```bash
cd testing/gui/frontend && ./node_modules/.bin/vitest run
cd testing/gui/frontend && npm run e2e
```
Expected: Vitest 19/19, Playwright 18/18 (unchanged — no new smokes).

- [ ] **Step 5: Commit Tasks 5+6.**

```bash
git add testing/gui/backend/routes/probes.py testing/gui/frontend/src/components/PatchingControls.tsx testing/gui/frontend/src/components/ProbePanel.tsx
git commit -m "$(cat <<'EOF'
feat(backend+gui): n_steps passthrough for approx_head and approx_neuron

WS route branches parse cfg.n_steps identically to approx branch, forward
to per-head and per-neuron AP, populate complete.summary.n_steps. Frontend
IG-steps input visibility expands to cover all three AP modes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Verification matrix

- pyright 0/0/0 on probe.py + routes/probes.py + both modified test files.
- tsc clean.
- pytest: 2 new unit + 2 new TinyLlama tests pass; existing Phase 3.6/3.9 tests preserved bit-identical.
- Vitest 19/19.
- Playwright 18/18.

## Commit plan summary

4 commits (after spec):
1. `docs(phase 3.10.1): plan` (this file).
2. `feat(probe): n_steps IG for per-head and per-neuron`.
3. `test(probe): IG unit + TinyLlama tests for per-head and per-neuron`.
4. `feat(backend+gui): n_steps passthrough for approx_head and approx_neuron`.
