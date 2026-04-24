# Phase 3.10 — Integrated Gradients Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `attribution_patch` (Phase 3.5) with a path-integration option (`n_steps >= 2`) that activates Integrated Gradients (Sundararajan 2017), while preserving bit-identical behavior when `n_steps=1`.

**Architecture:** Add `n_steps: int = 1` keyword-only parameter to `attribution_patch`. When `n_steps == 1`, execute the existing Phase 3.5 single-forward-backward path unchanged. When `n_steps >= 2`, run N forward+backward passes where each pass attaches `self_attn` and `mlp` post-hooks that REPLACE module outputs with interpolated values `base_act + α_k · (from_act - base_act)`. Accumulate the fresh-leaf tensors' `.grad` across steps; average; multiply by Phase 3.5's existing Δ definitions. Path: midpoint rule (`α_k = (k - 0.5) / N`).

**Tech Stack:** PyTorch (forward hooks, autograd), HuggingFace LLaMA (self_attn + mlp module structure), FastAPI (WS route passthrough), React + TypeScript (PatchingControls numeric input + heatmap header annotation), Playwright (smoke test).

---

## Files

- **Modify:** `testing/llm_surgeon/probe.py:911-925` — add `n_steps: Optional[int] = None` to `PatchingResult`.
- **Modify:** `testing/llm_surgeon/probe.py:1079-1246` — extend `attribution_patch` with `n_steps` parameter + IG branch.
- **Modify:** `testing/tests/test_probe_attribution_patch.py` — add 5 unit tests + 1 TinyLlama integration test.
- **Modify:** `testing/gui/backend/routes/probes.py:~1162-1181` — pass `n_steps` from cfg dict to `attribution_patch`; add to `complete.summary`.
- **Modify:** `testing/gui/frontend/src/api/types.ts` (or wherever `PatchingCompleteData.summary` is declared) — add optional `n_steps?: number`.
- **Modify:** `testing/gui/frontend/src/components/PatchingControls.tsx:18-46` — add `n_steps: number` to `PatchingState` (default 1); add numeric input visible when `mode === "approx"`.
- **Modify:** `testing/gui/frontend/src/components/ProbePanel.tsx` — forward `n_steps` into WS cfg when mode is approx.
- **Modify:** `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx:257-259` — extend header to show "IG N steps" when `result.n_steps > 1`.
- **Modify:** `testing/gui/frontend/tests/e2e/fixtures/activation-patching-approx.json` — add `n_steps: 5` to the embedded `result.summary` (or however fixture represents result summary).
- **Modify:** `testing/gui/frontend/tests/e2e/smoke.spec.ts` — add one new test asserting the IG header renders when fixture carries `n_steps > 1`.

---

## Task 1: Add `n_steps` field to `PatchingResult`

**Files:**
- Modify: `testing/llm_surgeon/probe.py:925` (just after `n_neurons`).

- [ ] **Step 1: Edit dataclass**

Add one field after line 925. Find:

```python
    n_neurons: Optional[int] = None                # set by attribution_patch_per_neuron (= intermediate_size)
```

Replace with:

```python
    n_neurons: Optional[int] = None                # set by attribution_patch_per_neuron (= intermediate_size)
    n_steps: Optional[int] = None                  # set by attribution_patch when n_steps > 1 (IG path steps)
```

- [ ] **Step 2: Pyright check**

Run: `.venv/bin/python -m pyright testing/llm_surgeon/probe.py`
Expected: 0 errors, 0 warnings, 0 infos.

- [ ] **Step 3: Run existing AP tests as regression gate**

Run: `.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py -v`
Expected: all existing tests still pass (the new field is optional with None default, no call site change required).

- [ ] **Step 4: No commit yet — combine with Task 2.**

---

## Task 2: Add `n_steps` parameter + validation to `attribution_patch` (no IG behavior yet)

**Files:**
- Modify: `testing/llm_surgeon/probe.py:1079-1093` (signature) and `:1113-1116` (validation block).

- [ ] **Step 1: Extend signature**

Find:

```python
def attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult:
```

Add `n_steps: int = 1,` on its own line just before `on_cell`:

```python
def attribution_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    correct_token_id: int,
    incorrect_token_id: int,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    n_steps: int = 1,
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult:
```

- [ ] **Step 2: Add validation after existing validation block** (after the `if not clean_prompt or not corrupted_prompt:` check, around line 1116)

Insert:

```python
    if not isinstance(n_steps, int) or n_steps < 1 or n_steps > 50:
        raise ValueError(f"n_steps must be int in [1, 50], got {n_steps!r}")
```

- [ ] **Step 3: Populate `n_steps` on the return**

Find the `return PatchingResult(` near line 1236. Add `n_steps=(n_steps if n_steps > 1 else None),` as a new kwarg. The return becomes:

```python
    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx",
        n_steps=(n_steps if n_steps > 1 else None),
    )
```

(None when n_steps=1 preserves the "field unset" serialization semantics.)

- [ ] **Step 4: Pyright check**

Run: `.venv/bin/python -m pyright testing/llm_surgeon/probe.py`
Expected: 0/0/0.

- [ ] **Step 5: Regression test — n_steps=1 still produces old values bit-identical**

Run: `.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py -v`
Expected: all pre-IG tests still pass (default `n_steps=1` preserves behavior).

- [ ] **Step 6: No commit yet — combine with Task 3.**

---

## Task 3: Implement IG algorithm for `n_steps >= 2`

**Files:**
- Modify: `testing/llm_surgeon/probe.py:1188-1194` (the scalar metric + single backward call).

This is the core algorithmic change. Replace the single-backward block with a branch.

- [ ] **Step 1: Locate the existing single-backward block**

Find the block inside `attribution_patch` (currently around lines 1188-1193):

```python
        # --- Step 3: Metric scalar on base-side logits, backward ---
        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        metric.backward()
```

This runs inside the `with torch.enable_grad():` block that captured `base_captured`, `base_h_ins`, `base_logits`, `base_tokens` via `_capture_residual_stream_with_grad`.

- [ ] **Step 2: Replace with conditional branch**

```python
        if n_steps == 1:
            # --- Step 3: Metric scalar on base-side logits, backward ---
            metric = (
                base_logits[meas_pos, correct_token_id]
                - base_logits[meas_pos, incorrect_token_id]
            )
            metric.backward()
            # base_captured[(L, sub)].grad is now populated for Step 4.
        else:
            # n_steps >= 2: Integrated Gradients path integration.
            # The grad on base_captured won't be used — we run N separate
            # forward+backward passes with replacement hooks and accumulate
            # grads on fresh leaf tensors, then stuff the averaged grads
            # back into base_captured-like dict for Step 4 to consume.
            pass  # implementation below
```

- [ ] **Step 3: Implement IG branch**

Replace the `pass  # implementation below` with the full IG machinery. Insert this inside the `with torch.enable_grad():` block at the `else:` branch:

```python
            # Pre-compute from-side attn_out and ffn_out per layer for
            # interpolation. We already have from_states[(L, sub)] from
            # Step 1's capture; it's attn_out for "attn" rows and layer
            # output (= h_post_ffn) for "ffn" rows.
            #
            # For IG we need from_attn_out[L] and from_ffn_out[L] as
            # *separate* components (not layer output). ffn_out = layer_out
            # - h_post_attn, reconstructable from from_states and from_h_ins:
            #   from_ffn_out[L] = from_states[(L, "ffn")] - (from_h_ins[L] + from_states[(L, "attn")])
            #
            # Same derivation for base-side: base_ffn_out[L] =
            # base_captured[(L, "ffn")] - (base_h_ins[L] + base_captured[(L, "attn")])
            # using .detach() values only (fresh leaf tensors below).

            import torch as _torch

            # Build per-layer base and from components as detached values.
            # (Everything here is value-only — no graph retention needed.)
            num_layers = len(model.model.layers)
            target_layers_set = (
                set(range(num_layers)) if layers is None else set(layers)
            )
            base_attn: Dict[int, _torch.Tensor] = {}
            base_ffn: Dict[int, _torch.Tensor] = {}
            from_attn: Dict[int, _torch.Tensor] = {}
            from_ffn: Dict[int, _torch.Tensor] = {}
            for L in sorted(target_layers_set):
                b_attn = base_captured[(L, "attn")].detach() if "attn" in sublayers else None
                b_layer_out = base_captured[(L, "ffn")].detach() if "ffn" in sublayers else None
                b_hin = base_h_ins[L].detach() if "attn" in sublayers else None
                f_attn = from_states[(L, "attn")] if "attn" in sublayers else None
                f_layer_out = from_states[(L, "ffn")] if "ffn" in sublayers else None
                f_hin = from_h_ins[L] if "attn" in sublayers else None
                if b_attn is not None:
                    base_attn[L] = b_attn
                if f_attn is not None:
                    from_attn[L] = f_attn
                if b_layer_out is not None and b_attn is not None and b_hin is not None:
                    base_ffn[L] = b_layer_out - (b_hin + b_attn)
                elif b_layer_out is not None:
                    # ffn-only case: no attn capture, so layer_out ≈ h_in + ffn_out
                    # and we don't have h_in separately. Skip ffn interp in
                    # ffn-only mode — fall through to n_steps=1-style behavior
                    # would require more plumbing; we allow n_steps >= 2 only
                    # when "attn" in sublayers so we always have h_ins available.
                    pass
                if f_layer_out is not None and f_attn is not None and f_hin is not None:
                    from_ffn[L] = f_layer_out - (f_hin + f_attn)

            # Accumulator grads at each interp tensor.
            grad_sum_attn: Dict[int, _torch.Tensor] = {
                L: _torch.zeros_like(base_attn[L]) for L in base_attn
            }
            grad_sum_ffn: Dict[int, _torch.Tensor] = {
                L: _torch.zeros_like(base_ffn[L]) for L in base_ffn
            }

            # N forward+backward passes with interpolation hooks.
            device = _get_input_device(model)
            enc = tokenizer(base_prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)

            for k in range(n_steps):
                alpha = (k + 0.5) / n_steps

                # Build this step's interp leaf tensors (fresh per step).
                interp_attn: Dict[int, _torch.Tensor] = {}
                interp_ffn: Dict[int, _torch.Tensor] = {}
                for L in base_attn:
                    t = base_attn[L] + alpha * (from_attn[L] - base_attn[L])
                    t = t.detach().clone().requires_grad_(True)
                    interp_attn[L] = t
                for L in base_ffn:
                    t = base_ffn[L] + alpha * (from_ffn[L] - base_ffn[L])
                    t = t.detach().clone().requires_grad_(True)
                    interp_ffn[L] = t

                hooks: List = []

                def make_attn_replace(L_captured: int):
                    def hook(_mod, _inp, out):
                        new = interp_attn[L_captured]
                        if isinstance(out, tuple):
                            return (new,) + tuple(out[1:])
                        return new
                    return hook

                def make_mlp_replace(L_captured: int):
                    def hook(_mod, _inp, _out):
                        return interp_ffn[L_captured]
                    return hook

                for L in interp_attn:
                    hooks.append(
                        model.model.layers[L].self_attn.register_forward_hook(
                            make_attn_replace(L)
                        )
                    )
                for L in interp_ffn:
                    hooks.append(
                        model.model.layers[L].mlp.register_forward_hook(
                            make_mlp_replace(L)
                        )
                    )

                try:
                    out = model(input_ids)
                    step_logits = out.logits[0] if hasattr(out, "logits") else out[0][0]
                    step_metric = (
                        step_logits[meas_pos, correct_token_id]
                        - step_logits[meas_pos, incorrect_token_id]
                    )
                    step_metric.backward()
                finally:
                    for h in hooks:
                        h.remove()

                for L, t in interp_attn.items():
                    if t.grad is not None:
                        grad_sum_attn[L] += t.grad.detach()
                for L, t in interp_ffn.items():
                    if t.grad is not None:
                        grad_sum_ffn[L] += t.grad.detach()

            # Average; then splice back into a dict keyed like base_captured.grad.
            avg_grad: Dict[Tuple[int, str], _torch.Tensor] = {}
            for L in grad_sum_attn:
                avg_grad[(L, "attn")] = grad_sum_attn[L] / n_steps
            for L in grad_sum_ffn:
                avg_grad[(L, "ffn")] = grad_sum_ffn[L] / n_steps
```

- [ ] **Step 4: Adapt Step 4 (per-cell computation) to read averaged grads for n_steps >= 2**

The existing Step 4 loop at `for (L, sub) in sorted_keys:` reads `base_act.grad`. For IG we need to read from `avg_grad[(L, sub)]` instead when `n_steps >= 2`. Replace the `base_grad = base_act.grad` line with:

```python
        if n_steps == 1:
            if base_act.grad is None:
                continue  # shouldn't happen after backward but guard defensively
            base_grad = base_act.grad  # (1, seq_len, d_model)
        else:
            if (L, sub) not in avg_grad:
                continue
            base_grad = avg_grad[(L, sub)]
```

- [ ] **Step 5: Pyright check**

Run: `.venv/bin/python -m pyright testing/llm_surgeon/probe.py`
Expected: 0/0/0.

- [ ] **Step 6: Regression — n_steps=1 still bit-identical**

Run: `.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py -v`
Expected: all Phase 3.5 tests still pass unchanged.

- [ ] **Step 7: Commit Tasks 1+2+3 together**

```bash
git add testing/llm_surgeon/probe.py
git commit -m "$(cat <<'EOF'
feat(probe): attribution_patch n_steps for Integrated Gradients

n_steps=1 preserves Phase 3.5 behavior exactly. n_steps>=2 runs N
midpoint-rule forward+backward passes with self_attn and mlp output
hooks that replace module outputs with interpolated base→from values.
Gradients read off fresh leaf tensors; averaged across steps; consumed
by existing per-cell Δ · grad reduction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Backend unit tests for IG

**Files:**
- Modify: `testing/tests/test_probe_attribution_patch.py` (append tests at end).

- [ ] **Step 1: Write `test_n_steps_1_matches_old_behavior`**

Append to the test file. The existing file already has mock fixtures for `attribution_patch`; reuse them.

```python
def test_n_steps_1_matches_old_behavior(mock_ap_model, mock_ap_tokenizer):
    """n_steps=1 must produce bit-identical results to the no-n_steps call."""
    kwargs = dict(
        clean_prompt="A B C D",
        corrupted_prompt="A B C E",
        correct_token_id=1,
        incorrect_token_id=2,
        direction="denoise",
    )
    baseline = attribution_patch(mock_ap_model, mock_ap_tokenizer, **kwargs)
    new = attribution_patch(mock_ap_model, mock_ap_tokenizer, **kwargs, n_steps=1)
    assert baseline.n_steps is None
    assert new.n_steps is None  # n_steps=1 returns None (unset field semantics)
    assert len(baseline.cells) == len(new.cells)
    for b, n in zip(baseline.cells, new.cells):
        assert b["layer"] == n["layer"]
        assert b["sublayer"] == n["sublayer"]
        assert b["position"] == n["position"]
        assert abs(b["ap_recovery"] - n["ap_recovery"]) < 1e-9
```

- [ ] **Step 2: Write `test_n_steps_5_runs_and_differs`**

```python
def test_n_steps_5_runs_and_differs(mock_ap_model, mock_ap_tokenizer):
    """n_steps=5 should run, produce finite values, and differ from n_steps=1."""
    kwargs = dict(
        clean_prompt="A B C D",
        corrupted_prompt="A B C E",
        correct_token_id=1,
        incorrect_token_id=2,
        direction="denoise",
    )
    r1 = attribution_patch(mock_ap_model, mock_ap_tokenizer, **kwargs, n_steps=1)
    r5 = attribution_patch(mock_ap_model, mock_ap_tokenizer, **kwargs, n_steps=5)
    assert r5.n_steps == 5
    assert len(r1.cells) == len(r5.cells)
    import math
    for c in r5.cells:
        assert math.isfinite(c["ap_recovery"])
    assert any(
        abs(c1["ap_recovery"] - c5["ap_recovery"]) > 1e-4
        for c1, c5 in zip(r1.cells, r5.cells)
    ), "IG at n_steps=5 produced no change anywhere — integration is a no-op"
```

- [ ] **Step 3: Write `test_n_steps_completeness`**

```python
def test_n_steps_completeness(mock_ap_model, mock_ap_tokenizer):
    """With n_steps=20 over all layers/positions/sublayers, IG's
    completeness axiom implies Σ ap_recovery ≈ 1.0 for denoise direction.
    First-order (n_steps=1) typically deviates more from 1.0."""
    kwargs = dict(
        clean_prompt="A B C D",
        corrupted_prompt="A B C E",
        correct_token_id=1,
        incorrect_token_id=2,
        direction="denoise",
        sublayers=("attn", "ffn"),
    )
    r20 = attribution_patch(mock_ap_model, mock_ap_tokenizer, **kwargs, n_steps=20)
    total = sum(c["ap_recovery"] for c in r20.cells)
    assert abs(total - 1.0) < 0.1, (
        f"IG completeness axiom violated: sum(ap_recovery) = {total} (expected ≈1.0)"
    )
```

- [ ] **Step 4: Write `test_n_steps_validation`**

```python
import pytest

def test_n_steps_validation(mock_ap_model, mock_ap_tokenizer):
    base_kwargs = dict(
        clean_prompt="A B C D",
        corrupted_prompt="A B C E",
        correct_token_id=1,
        incorrect_token_id=2,
    )
    for bad in (0, -1, 51, 100):
        with pytest.raises(ValueError, match=r"n_steps must be int in \[1, 50\]"):
            attribution_patch(mock_ap_model, mock_ap_tokenizer, **base_kwargs, n_steps=bad)
```

- [ ] **Step 5: Write `test_n_steps_noise_direction`**

```python
def test_n_steps_noise_direction(mock_ap_model, mock_ap_tokenizer):
    """IG also works in noise direction; ap_recovery still finite."""
    import math
    kwargs = dict(
        clean_prompt="A B C D",
        corrupted_prompt="A B C E",
        correct_token_id=1,
        incorrect_token_id=2,
        direction="noise",
    )
    r = attribution_patch(mock_ap_model, mock_ap_tokenizer, **kwargs, n_steps=5)
    assert r.n_steps == 5
    for c in r.cells:
        assert math.isfinite(c["ap_recovery"])
```

- [ ] **Step 6: Run all 5 new tests**

Run:
```bash
.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py -v -k "n_steps"
```
Expected: 5 passed.

- [ ] **Step 7: Run full AP test file (regression gate)**

Run:
```bash
.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py -v
```
Expected: all old + 5 new tests pass.

- [ ] **Step 8: Pyright**

Run: `.venv/bin/python -m pyright testing/tests/test_probe_attribution_patch.py`
Expected: 0/0/0.

- [ ] **Step 9: Commit**

```bash
git add testing/tests/test_probe_attribution_patch.py
git commit -m "$(cat <<'EOF'
test(probe): IG unit tests for attribution_patch n_steps

5 mock-model tests covering: n_steps=1 bit-identical back-compat,
n_steps=5 runs and differs, n_steps=20 completeness axiom within 0.1
of 1.0, validation bounds [1, 50], noise direction support.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: TinyLlama integration test

**Files:**
- Modify: `testing/tests/test_probe_attribution_patch.py` (append).

- [ ] **Step 1: Write the TinyLlama test**

Append:

```python
def test_ig_tinyllama_converges(tinyllama_model):
    """On a real model, IG at n_steps=5 should stay finite, have high
    rank correlation with n_steps=1, and typically sum closer to 1.0
    than first-order AP for denoise direction. Budget ≤ 180s on RTX 2080."""
    import math
    from scipy.stats import spearmanr

    model, tokenizer = tinyllama_model
    kwargs = dict(
        clean_prompt="The capital of France is",
        corrupted_prompt="The capital of Russia is",
        correct_token_id=tokenizer(" Paris", add_special_tokens=False)["input_ids"][0],
        incorrect_token_id=tokenizer(" Moscow", add_special_tokens=False)["input_ids"][0],
        direction="denoise",
        measurement_position=-1,
    )
    r1 = attribution_patch(model, tokenizer, **kwargs, n_steps=1)
    r5 = attribution_patch(model, tokenizer, **kwargs, n_steps=5)

    assert r1.n_steps is None
    assert r5.n_steps == 5
    for c in r5.cells:
        assert math.isfinite(c["ap_recovery"])

    # Rank correlation: IG at N=5 should rank cells similarly to N=1.
    scores_1 = [c["ap_recovery"] for c in r1.cells]
    scores_5 = [c["ap_recovery"] for c in r5.cells]
    rho, _ = spearmanr(scores_1, scores_5)
    assert rho > 0.8, f"IG rank order deviates too far from n_steps=1 (ρ={rho})"
```

- [ ] **Step 2: Run it (if tinyllama_model fixture is configured — else skip-guarded)**

Run:
```bash
.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py::test_ig_tinyllama_converges -v
```
Expected: PASS in ≤180s, or SKIPPED if the fixture is absent.

- [ ] **Step 3: Commit**

```bash
git add testing/tests/test_probe_attribution_patch.py
git commit -m "$(cat <<'EOF'
test(probe): TinyLlama IG integration — n_steps=5 rank correlation

Asserts Spearman ρ > 0.8 between n_steps=1 and n_steps=5 cell scores on
capital-of-France. IG should re-rank slightly but not wildly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Backend WS route cfg passthrough

**Files:**
- Modify: `testing/gui/backend/routes/probes.py:~1162-1181` (the `approx` branch) plus the `complete.summary` construction nearby.

- [ ] **Step 1: Read `n_steps` from cfg dict**

Near the top of the handler (where other cfg fields like `direction`, `measurement_position` are read from the dict), add:

```python
n_steps_raw = config.get("n_steps", 1)
n_steps = int(n_steps_raw) if isinstance(n_steps_raw, (int, float, str)) else 1
if n_steps < 1 or n_steps > 50:
    # Validation mirrors attribution_patch — surface a clear error frame.
    await ws.send_json({"type": "error", "message": f"n_steps must be int in [1, 50], got {n_steps_raw!r}"})
    await ws.close()
    return
```

(The exact placement follows the existing `correct_token_id`/`incorrect_token_id` parsing pattern in the file. Use it as a template.)

- [ ] **Step 2: Pass `n_steps` into `attribution_patch` call**

In the `approx` branch (lambda invoking `attribution_patch`), add `n_steps=n_steps,` as a keyword argument:

```python
lambda: attribution_patch(
    info.model, info.tokenizer,
    clean_prompt=clean_prompt,
    corrupted_prompt=corrupted_prompt,
    correct_token_id=_cid,
    incorrect_token_id=_iid,
    direction=direction,
    measurement_position=measurement_position,
    positions=positions,
    sublayers=sublayers,
    layers=layers,
    n_steps=n_steps,
    on_cell=on_cell,
),
```

- [ ] **Step 3: Include `n_steps` in `complete.summary`**

Find the `complete.summary = {...}` or equivalent dict construction for `approx` mode (the route currently populates `mode`, `n_cells`, etc.). Add:

```python
"n_steps": result.n_steps,
```

(None for n_steps=1 runs; an int for IG runs. Clients treat None / absent identically.)

- [ ] **Step 4: Pyright check**

Run: `.venv/bin/python -m pyright testing/gui/backend/routes/probes.py`
Expected: 0/0/0.

- [ ] **Step 5: Commit**

```bash
git add testing/gui/backend/routes/probes.py
git commit -m "$(cat <<'EOF'
feat(backend): WS approx mode accepts n_steps for IG

Parses cfg.n_steps (default 1), validates [1,50], forwards to
attribution_patch, reports back in complete.summary.n_steps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Frontend types

**Files:**
- Modify: `testing/gui/frontend/src/api/types.ts` (or wherever `PatchingCompleteData` / `PatchingSummary` is declared — check for `PatchingCompleteData.summary.mode` to locate it).

- [ ] **Step 1: Add `n_steps?` field to the summary type**

Find the interface or type alias that covers `summary` for the complete frame. Add:

```ts
export interface PatchingCompleteSummary {
  mode: PatchingMode;
  // ... existing fields
  n_steps?: number;
}
```

(If the summary is currently an inline type inside `PatchingCompleteData`, promote/extend it as appropriate following the existing pattern the file uses for other optional summary fields like `n_heads`, `n_neurons`, `n_edges`, `tau`.)

- [ ] **Step 2: tsc check**

Run: `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit`
Expected: clean.

- [ ] **Step 3: No commit yet — bundle with Tasks 8+9.**

---

## Task 8: PatchingControls — add `n_steps` input

**Files:**
- Modify: `testing/gui/frontend/src/components/PatchingControls.tsx:18-46` (interface + default) and the render block around line 180 (approx radio).

- [ ] **Step 1: Extend `PatchingState` interface**

Find:

```ts
export interface PatchingState {
  // ... existing fields
  top_k_neurons: number;
}
```

Add:

```ts
export interface PatchingState {
  // ... existing fields
  top_k_neurons: number;
  n_steps: number;
}
```

- [ ] **Step 2: Extend `DEFAULT_PATCHING_STATE`**

Add `n_steps: 1,` as the last field.

- [ ] **Step 3: Add numeric input**

Inside the `mode` selector block, after the approx radio but before the per-head radio (OR in a conditional section that renders when `mode === "approx"`, similar to how `top_k_edges` is conditional on `mode === "edge"`), insert:

```tsx
{state.mode === "approx" && (
  <label style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
    IG steps:
    <input
      type="number"
      min={1}
      max={50}
      value={state.n_steps}
      onChange={(e) =>
        onChange({ n_steps: Math.max(1, Math.min(50, Number(e.target.value) || 1)) })
      }
      style={{ width: 60 }}
      title="n_steps=1: first-order AP (fast). n_steps>=2: Integrated Gradients (N× cost, more accurate)."
    />
    <span style={{ color: "#888", fontSize: 11 }}>
      {state.n_steps === 1 ? "(first-order)" : `(IG, ${state.n_steps}×)`}
    </span>
  </label>
)}
```

- [ ] **Step 4: Extend `ProbePanel` to forward n_steps to WS cfg**

In `ProbePanel.tsx`, find the section that builds the WS config for `activation-patching`. Add `n_steps: state.n_steps,` to the cfg payload when `state.mode === "approx"` (pattern should match how `top_k_edges` and `tau` are conditionally added).

- [ ] **Step 5: tsc check**

Run: `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit`
Expected: clean.

- [ ] **Step 6: No commit yet.**

---

## Task 9: Heatmap header — annotate IG

**Files:**
- Modify: `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx:257-259`.

- [ ] **Step 1: Extend header text**

Find:

```tsx
<h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
  {mode === "approx" ? "Attribution Patching (∇)" : "Activation Patching"}
  {" — "}{result.sessionName}{" — \""}{result.prompt.slice(0, 40)}{"\""}
</h3>
```

Replace with:

```tsx
<h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
  {mode === "approx" ? "Attribution Patching (∇)" : "Activation Patching"}
  {mode === "approx" && result.n_steps != null && result.n_steps > 1
    ? ` — IG ${result.n_steps} steps`
    : ""}
  {" — "}{result.sessionName}{" — \""}{result.prompt.slice(0, 40)}{"\""}
</h3>
```

(Note: `result.n_steps` must be threaded through from the complete-frame handler into the result object. If `result` is built in a parent and carries `summary.n_steps`, access it as `result.summary?.n_steps` instead. Check `ActivationPatchingHeatmap`'s props to confirm.)

- [ ] **Step 2: tsc check**

Run: `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit`
Expected: clean.

- [ ] **Step 3: No commit yet.**

---

## Task 10: Playwright fixture + smoke test

**Files:**
- Modify: `testing/gui/frontend/tests/e2e/fixtures/activation-patching-approx.json` — add `n_steps: 5` to the fixture's result summary.
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts` — add test.

- [ ] **Step 1: Update fixture**

Open `activation-patching-approx.json`. Find the `results` array → first result → `summary` object. Add:

```json
"n_steps": 5,
```

(Do NOT remove any existing fields. The fixture may already have `mode: "approx"`, `measurement_position`, etc. Just add `n_steps`.)

- [ ] **Step 2: Write Playwright smoke test**

Append to `smoke.spec.ts`. Pattern should mirror existing AP-approx smoke:

```ts
test("approx mode with IG shows step count in heatmap header", async ({ page }) => {
  await page.goto("/");
  const fixture = path.join(__dirname, "fixtures", "activation-patching-approx.json");
  const fileChooser = page.locator('input[type="file"]');
  await fileChooser.setInputFiles(fixture);

  // Navigate to the probe result that has mode=approx.
  await page.getByRole("tab", { name: /probe/i }).click();

  // Assert the IG annotation is rendered.
  await expect(page.getByRole("heading", { name: /Attribution Patching.*IG 5 steps/ }))
    .toBeVisible();
});
```

(Adjust selectors to match existing tests in the file — the import/seed pattern is established.)

- [ ] **Step 3: Run Playwright**

Run: `cd testing/gui/frontend && npm run e2e`
Expected: 18 tests pass (17 existing + 1 new).

- [ ] **Step 4: Run Vitest (regression)**

Run: `cd testing/gui/frontend && ./node_modules/.bin/vitest run`
Expected: 19/19 pass.

- [ ] **Step 5: Commit Tasks 7+8+9+10 together (all frontend)**

```bash
git add testing/gui/frontend/src/api/types.ts \
        testing/gui/frontend/src/components/PatchingControls.tsx \
        testing/gui/frontend/src/components/ProbePanel.tsx \
        testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx \
        testing/gui/frontend/tests/e2e/fixtures/activation-patching-approx.json \
        testing/gui/frontend/tests/e2e/smoke.spec.ts
git commit -m "$(cat <<'EOF'
feat(gui/frontend): n_steps control + IG header annotation

PatchingState.n_steps (default 1) plus numeric input in PatchingControls
visible when mode=approx. Heatmap header appends " — IG N steps" when
result.n_steps > 1. Fixture extended, 18th Playwright smoke asserts
the annotation renders.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification matrix

Run each and confirm output:

- **pyright**: `.venv/bin/python -m pyright testing/llm_surgeon/probe.py testing/gui/backend/routes/probes.py testing/tests/test_probe_attribution_patch.py` → 0/0/0.
- **tsc**: `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit` → clean.
- **pytest (AP)**: `.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py -v` → all pass (including 5 new unit + 1 TinyLlama).
- **pytest (regressions)**: `.venv/bin/python -m pytest testing/tests/ -v --ignore=testing/tests/test_probe_attribution_patch.py` → unchanged from pre-Phase-3.10 baseline.
- **vitest**: `cd testing/gui/frontend && ./node_modules/.bin/vitest run` → 19/19.
- **playwright**: `cd testing/gui/frontend && npm run e2e` → 18/18.

---

## Commit plan summary

6 commits total (4 Python, 1 backend, 1 frontend):

1. `docs(phase 3.10): spec ...` (already shipped: `e20763e`)
2. `docs(phase 3.10): plan ...` (this file)
3. `feat(probe): attribution_patch n_steps for Integrated Gradients`
4. `test(probe): IG unit tests for attribution_patch n_steps`
5. `test(probe): TinyLlama IG integration — n_steps=5 rank correlation`
6. `feat(backend): WS approx mode accepts n_steps for IG`
7. `feat(gui/frontend): n_steps control + IG header annotation`
