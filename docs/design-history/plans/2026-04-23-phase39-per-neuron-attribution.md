# Phase 3.9 — Per-Neuron FFN Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `mode="approx_neuron"` on the activation-patching WS route: per-(layer, neuron, position) AP decomposition via chain rule through `mlp.down_proj`, with a ranked-list viz on the frontend.

**Architecture:** New private pre-hook on each `layer.mlp.down_proj` captures the MLP intermediate activation (its forward input). Phase 3.7's `ffn_out` post-hook gains a `retain_grad()` call so `ffn_out.grad` is available as the per-layer reader gradient. New `probe.attribution_patch_per_neuron(...)` computes `(Δact * (grad_ffn_out @ W_down)) / D` per neuron and returns top-k. Backend adds a sixth mode branch. Frontend adds `PerNeuronPatchingPanel.tsx` (ranked table — not a heatmap — because intermediate_size=5632 is unrenderable as a grid).

**Tech Stack:** Python 3.11 + PyTorch + transformers + FastAPI WebSockets + React 18 + TypeScript + Zustand (existing). No new deps.

**Spec:** `testing/docs/superpowers/specs/2026-04-23-phase39-per-neuron-attribution.md` (commit `d57aef1`).

**Tool rules (for every subagent prompt):**
- Use Read (not cat), Edit (not Bash sed/awk/cat), Grep (not Bash grep/rg/awk), Glob (not find)
- For git ops: `git -C /home/ai/ai-projects/llm <cmd>`
- For CUDA/GPU/tsc/vitest/pyright/git operations: pass `dangerouslyDisableSandbox: true` to the Bash call
- If a Bash call returns "Permission denied" or sandbox error twice in a row, STOP and report BLOCKED — the parent will run it
- Pyright/tsc must be 0 errors / 0 warnings / 0 info after every task

---

## File Structure

**Python**
- **Modify** `testing/llm_surgeon/probe.py`
  - `_capture_residual_stream_with_grad`: add `capture_ffn_act: bool = False` param, extend return to **7-tuple** with `ffn_acts: Dict[int, Tensor]`. Also: inside the existing `capture_ffn_out` post-hook, call `retain_grad()` on `mlp_out` when it requires grad.
  - Update all existing callers (4 call sites: `attribution_patch`, `attribution_patch_per_head`, `_compute_all_edges`, any test) to unpack the new 7-tuple shape.
  - `PatchingResult`: add `n_neurons: Optional[int] = None`.
  - New public `attribution_patch_per_neuron(...)` function.
- **Create** `testing/tests/test_probe_per_neuron_ap.py` — mock + TinyLlama tests.

**Backend**
- **Modify** `testing/gui/backend/routes/probes.py` — add `elif cfg.mode == "approx_neuron"` branch.

**Frontend**
- **Modify** `testing/gui/frontend/src/types/api.ts`
- **Modify** `testing/gui/frontend/src/components/PatchingControls.tsx` — sixth radio + `top_k_neurons` input, `PatchingState` field
- **Modify** `testing/gui/frontend/src/components/ProbePanel.tsx` — forward `top_k_neurons`
- **Modify** `testing/gui/frontend/src/components/VisualizationArea.tsx` — route `mode === "approx_neuron"`
- **Create** `testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx` — ranked-list view
- **Create** `testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-neuron.json`
- **Modify** `testing/gui/frontend/tests/e2e/smoke.spec.ts` — 15th test

---

## Task 1: Capture helper — 7-tuple + retain_grad on ffn_out

**Files:**
- Modify: `testing/llm_surgeon/probe.py:178-344` (helper itself)
- Modify: `testing/llm_surgeon/probe.py` — every call site of `_capture_residual_stream_with_grad`
- Modify: `testing/tests/test_probe_edge_ap.py`, `testing/tests/test_probe_per_head_ap.py`, `testing/tests/test_probe_attribution_patch.py`, `testing/tests/test_probe_circuit.py` — any test that unpacks the helper

**Goal:** Extend the capture helper to optionally record MLP intermediate activations (the input to `down_proj`) AND ensure `ffn_out` tensors have `retain_grad()` so `.grad` is populated. Return 7-tuple.

- [ ] **Step 1: Run all existing probe tests as a baseline**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest testing/tests/test_probe_edge_ap.py testing/tests/test_probe_per_head_ap.py testing/tests/test_probe_attribution_patch.py testing/tests/test_probe_circuit.py -v -k "not TinyLlama"
```
Expected: 54 tests pass (snapshot the exact count).

- [ ] **Step 2: Extend the helper signature + return type**

In `testing/llm_surgeon/probe.py`, find `_capture_residual_stream_with_grad` (line 178). Replace the signature block (lines 178-194) with:

```python
def _capture_residual_stream_with_grad(
    model,
    tokenizer,
    prompt: str,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    capture_concat_z: bool = False,
    capture_reader_grads: bool = False,
    capture_ffn_out: bool = False,
    capture_ffn_act: bool = False,
) -> Tuple[
    Dict[Tuple[int, str], torch.Tensor],
    Dict[int, torch.Tensor],
    torch.Tensor,
    List[str],
    Dict[int, torch.Tensor],
    Dict[Tuple, torch.Tensor],
    Dict[int, torch.Tensor],
]:
```

Also update the docstring's Returns section (around lines 214-222). Replace the existing Returns block with:

```
    Returns: (captured_states, h_ins, output_logits, prompt_tokens,
              concat_z_captured, reader_inputs, ffn_acts).
        concat_z_captured is empty when capture_concat_z=False.
        reader_inputs is empty when capture_reader_grads=False; otherwise
        holds pre-LN residual tensors keyed by ("attn_in", L), ("ffn_in", L),
        ("logits", N_L) with retain_grad() called so .grad is populated after
        backward().
        When capture_ffn_out=True, captured also contains (L, "ffn_out") keys
        holding the raw MLP output before the residual add; retain_grad() is
        called on the ffn_out tensor so .grad is populated after backward()
        (needed for Phase 3.9 per-neuron attribution).
        ffn_acts is empty when capture_ffn_act=False; otherwise holds the
        input tensor to each mlp.down_proj (i.e. the MLP intermediate
        activation, shape [batch, seq, intermediate_size]) keyed by layer
        index.
```

- [ ] **Step 3: Add `retain_grad()` inside the `capture_ffn_out` hook**

In the same file, find the `capture_ffn_out` block (line 288-294):

```python
        if capture_ffn_out:
            def make_mlp_hook(idx: int):
                def hook(_module: torch.nn.Module, _inp: Tuple, out: object) -> None:
                    mlp_out = out[0] if isinstance(out, tuple) else out  # type: ignore[index]
                    captured[(idx, "ffn_out")] = mlp_out  # type: ignore[assignment]
                return hook
            hooks.append(model.model.layers[i].mlp.register_forward_hook(make_mlp_hook(i)))
```

Replace with:

```python
        if capture_ffn_out:
            def make_mlp_hook(idx: int):
                def hook(_module: torch.nn.Module, _inp: Tuple, out: object) -> None:
                    mlp_out = out[0] if isinstance(out, tuple) else out  # type: ignore[index]
                    if mlp_out.requires_grad:  # type: ignore[union-attr]
                        mlp_out.retain_grad()  # type: ignore[union-attr]
                    captured[(idx, "ffn_out")] = mlp_out  # type: ignore[assignment]
                return hook
            hooks.append(model.model.layers[i].mlp.register_forward_hook(make_mlp_hook(i)))
```

- [ ] **Step 4: Add the `capture_ffn_act` pre-hook block**

In the same file, directly AFTER the `capture_ffn_out` block (right before the `capture_reader_grads` block), add:

```python
        if capture_ffn_act:
            def make_ffn_act_hook(idx: int):
                def hook(_module: torch.nn.Module, args: Tuple) -> None:
                    a = args[0]  # [batch, seq, intermediate]
                    if a.requires_grad:
                        a.retain_grad()
                    ffn_acts[idx] = a
                return hook
            hooks.append(
                model.model.layers[i].mlp.down_proj.register_forward_pre_hook(
                    make_ffn_act_hook(i)
                )
            )
```

Also — near line 235 where the local dicts are initialized — add `ffn_acts: Dict[int, torch.Tensor] = {}` after `reader_inputs`:

```python
    captured: Dict[Tuple[int, str], torch.Tensor] = {}
    h_ins: Dict[int, torch.Tensor] = {}
    concat_z_captured: Dict[int, torch.Tensor] = {}
    reader_inputs: Dict[Tuple, torch.Tensor] = {}
    ffn_acts: Dict[int, torch.Tensor] = {}
    hooks: List = []
```

- [ ] **Step 5: Update the return statement**

Find the return at line 344:

```python
    return captured, h_ins, model_output.logits[0], prompt_tokens, concat_z_captured, reader_inputs
```

Replace with:

```python
    return captured, h_ins, model_output.logits[0], prompt_tokens, concat_z_captured, reader_inputs, ffn_acts
```

- [ ] **Step 6: Update existing callers of the helper to unpack 7-tuple**

Run this grep to find all call sites:
```bash
grep -n "_capture_residual_stream_with_grad(" testing/llm_surgeon/probe.py testing/tests/
```

For each occurrence of a tuple unpack (pattern `X, Y, Z, W, U, V = _capture_residual_stream_with_grad(...)`), add one more `_` at the end. Example transformation:

Before:
```python
from_captured_raw, _, from_logits, from_tokens, from_cz_raw, _ = \
    _capture_residual_stream_with_grad(...)
```
After:
```python
from_captured_raw, _, from_logits, from_tokens, from_cz_raw, _, _ = \
    _capture_residual_stream_with_grad(...)
```

Known call sites to update:
- `testing/llm_surgeon/probe.py` — inside `attribution_patch`, `attribution_patch_per_head`, and `_compute_all_edges` (4 total unpack sites in probe.py, two per function for "from" and "base")
- `testing/tests/test_probe_edge_ap.py` — check for direct helper calls in tests
- `testing/tests/test_probe_per_head_ap.py` — same
- `testing/tests/test_probe_attribution_patch.py` — same
- `testing/tests/test_probe_circuit.py` — same

- [ ] **Step 7: Run the baseline tests — must still pass unchanged**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest testing/tests/test_probe_edge_ap.py testing/tests/test_probe_per_head_ap.py testing/tests/test_probe_attribution_patch.py testing/tests/test_probe_circuit.py -v -k "not TinyLlama"
```
Expected: same pass count as Step 1 (54). If any test fails with a tuple-arity mismatch, you missed a call site in Step 6.

- [ ] **Step 8: Pyright clean**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pyright testing/llm_surgeon/probe.py testing/tests/
```
Expected: 0/0/0.

- [ ] **Step 9: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_edge_ap.py testing/tests/test_probe_per_head_ap.py testing/tests/test_probe_attribution_patch.py testing/tests/test_probe_circuit.py
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
refactor(probe): _capture_residual_stream_with_grad 7-tuple + ffn_out retain_grad

Adds capture_ffn_act flag to the capture helper (new pre-hook on
mlp.down_proj storing the MLP intermediate activation). Return tuple
extends from 6 to 7 elements; all existing callers updated to unpack
the new trailing slot. Also: the existing capture_ffn_out post-hook
now calls retain_grad() on mlp_out when it requires grad — Phase 3.9
per-neuron attribution reads ffn_out.grad as the per-layer reader
gradient, which was not previously populated.

Behavior-preserving: Phase 3.5/3.6/3.7/3.8 tests all still pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `attribution_patch_per_neuron` implementation

**Files:**
- Modify: `testing/llm_surgeon/probe.py` — add `PatchingResult.n_neurons` field; add new function after `attribution_patch_per_head` (line ~1218) and before `_is_valid_attn_writer` (line ~1418).

- [ ] **Step 1: Extend `PatchingResult`**

Find `PatchingResult` (line 885-899 after Phase 3.8). Change:

```python
    mode: str = "exact"                           # "exact" | "approx" | "approx_head" | "edge" | "circuit"
    n_heads: Optional[int] = None                  # ...
    n_edges: Optional[int] = None                  # ...
    n_edges_in_circuit: Optional[int] = None       # ...
    n_nodes_in_circuit: Optional[int] = None       # ...
    tau: Optional[float] = None                    # ...
```

to include a new line:

```python
    mode: str = "exact"                           # "exact" | "approx" | "approx_head" | "edge" | "circuit" | "approx_neuron"
    n_heads: Optional[int] = None                  # set by attribution_patch_per_head / edge_attribution_patch / extract_circuit
    n_edges: Optional[int] = None                  # set by edge_attribution_patch / extract_circuit (pre-filter count)
    n_edges_in_circuit: Optional[int] = None       # set by extract_circuit
    n_nodes_in_circuit: Optional[int] = None       # set by extract_circuit (includes the logits sink)
    tau: Optional[float] = None                    # set by extract_circuit (applied threshold)
    n_neurons: Optional[int] = None                # set by attribution_patch_per_neuron (= intermediate_size)
```

- [ ] **Step 2: Add `attribution_patch_per_neuron` function**

Insert directly after `attribution_patch_per_head`. Exact body:

```python
def attribution_patch_per_neuron(
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
    layers: Optional[List[int]] = None,
    top_k_neurons: int = 200,
    on_cell: Optional[Callable[[Dict], None]] = None,
) -> PatchingResult:
    """Per-neuron FFN attribution patching (Phase 3.9).

    Decomposes Δffn_out's contribution to the metric into per-
    (layer, neuron, position) AP scores via chain rule through W_down.
    One forward + one backward pass. For each layer, each FFN output
    position, and each neuron index i in [0, intermediate_size):

        grad_act = grad_ffn_out @ W_down             # [intermediate]
        delta_act = from_act[pos] - base_act[pos]    # [intermediate]
        ap_raw[i] = delta_act[i] * grad_act[i]
        ap_recovery[i] = ap_raw[i] / D  (denoise) or 1 + ap_raw[i]/D (noise)

    Returns PatchingResult with mode='approx_neuron',
    n_neurons=intermediate_size, and `cells` containing only the top-k
    tuples by |ap_recovery|. If top_k_neurons exceeds the total
    neuron-cell count, silently caps.
    """
    if top_k_neurons < 1:
        raise ValueError("top_k_neurons must be >= 1")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompts cannot be empty")
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")

    device = _get_input_device(model)

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    meas_pos = measurement_position % seq_len

    normalized_positions: List[int] = (
        list(range(seq_len)) if positions is None
        else [p if p >= 0 else seq_len + p for p in positions]
    )

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt

    sublayers: Tuple[str, ...] = ("attn", "ffn")

    # --- From pass (no_grad, capture ffn_act + ffn_out) ---
    with torch.no_grad():
        from_captured_raw, _, from_logits, from_tokens, _, _, from_ffn_acts_raw = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=False,
                capture_reader_grads=False,
                capture_ffn_out=True,
                capture_ffn_act=True,
            )
        from_ffn_acts = {k: v.detach().clone() for k, v in from_ffn_acts_raw.items()}

    # --- Base pass (enable_grad, backward through metric) ---
    with torch.enable_grad():
        base_captured, _, base_logits, base_tokens, _, _, base_ffn_acts = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=False,
                capture_reader_grads=False,
                capture_ffn_out=True,
                capture_ffn_act=True,
            )

        clean_baseline = from_logits if direction == "denoise" else base_logits
        corrupted_baseline = base_logits if direction == "denoise" else from_logits

        d_clean = (
            clean_baseline[meas_pos, correct_token_id]
            - clean_baseline[meas_pos, incorrect_token_id]
        ).detach()
        d_corrupted = (
            corrupted_baseline[meas_pos, correct_token_id]
            - corrupted_baseline[meas_pos, incorrect_token_id]
        ).detach()
        denominator = (d_clean - d_corrupted).item()

        if abs(denominator) < 1e-6:
            raise ValueError(
                "clean and corrupted baselines have identical logit_diff; "
                "AP would divide by zero"
            )

        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        metric.backward()

    intermediate_size: int = model.config.intermediate_size
    num_layers = len(model.model.layers)
    target_layers_set = set(range(num_layers)) if layers is None else set(layers)

    all_cells: List[Dict] = []
    for L in sorted(target_layers_set):
        if (L, "ffn_out") not in base_captured:
            continue
        base_ffn_out = base_captured[(L, "ffn_out")]
        if base_ffn_out.grad is None:
            continue
        if L not in base_ffn_acts or L not in from_ffn_acts:
            continue
        W_down: torch.Tensor = model.model.layers[L].mlp.down_proj.weight  # [hidden, intermediate]
        from_act = from_ffn_acts[L]
        base_act_L = base_ffn_acts[L]

        for pos in normalized_positions:
            grad_ffn_out = base_ffn_out.grad[0, pos].detach()            # [hidden]
            grad_act = grad_ffn_out @ W_down                             # [intermediate]
            delta_act = from_act[0, pos] - base_act_L[0, pos].detach()   # [intermediate]
            ap_raw = (delta_act * grad_act)                              # [intermediate]
            if direction == "denoise":
                ap_recovery = ap_raw / denominator
            else:
                ap_recovery = 1.0 + ap_raw / denominator

            ap_recovery_cpu = ap_recovery.detach().cpu().tolist()
            for i in range(intermediate_size):
                all_cells.append({
                    "layer": L,
                    "unit": f"neuron.n{i}",
                    "neuron": i,
                    "position": pos,
                    "ap_recovery": float(ap_recovery_cpu[i]),
                })

    n_total = len(all_cells)
    all_cells.sort(key=lambda c: abs(c["ap_recovery"]), reverse=True)
    top_cells = all_cells[:top_k_neurons]

    if on_cell is not None:
        for cell in top_cells:
            on_cell(cell)

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return PatchingResult(
        cells=top_cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx_neuron",
        n_neurons=intermediate_size,
    )
```

- [ ] **Step 3: Pyright clean**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pyright testing/llm_surgeon/probe.py
```
Expected: 0/0/0.

- [ ] **Step 4: Quick smoke — call it with mock-ish inputs**

Run a one-liner to ensure the function is importable and doesn't crash on obvious shape bugs:

```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -c "
from llm_surgeon.probe import attribution_patch_per_neuron
print('import ok:', attribution_patch_per_neuron.__name__)
"
```
Expected: `import ok: attribution_patch_per_neuron`

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(probe): attribution_patch_per_neuron — per-neuron FFN attribution

Decomposes Δffn_out's contribution to the metric into per-(layer,
neuron, position) AP scores via chain rule through W_down. One forward
+ one backward pass (same cost as Phase 3.6 per-head). Returns top-k
cells by |ap_recovery|.

PatchingResult gains n_neurons: Optional[int] (= intermediate_size).

Reference: Nanda 2023 (Attribution Patching primer), Gurnee & Tegmark
2023 (neuron attribution methodology).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Python tests (mock + TinyLlama integration)

**Files:**
- Create: `testing/tests/test_probe_per_neuron_ap.py`

- [ ] **Step 1: Write the test file**

Create `testing/tests/test_probe_per_neuron_ap.py`. Copy the mock infrastructure from `test_probe_circuit.py` (the `_stable_word_hash`, `_MockTokenizer`, `_MockLayer`, `_MockModel`, `_make_mock`, `_pick_tokens`, `CLEAN_PROMPT`, `CORR_PROMPT`, `CORRECT_ID`, `INCORRECT_ID` block — all of it) verbatim so tests are independent of Phase 3.8. Below that, write the tests:

```python
# ---- Tests ----

class TestPerNeuronMock:
    def test_returns_patching_result(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok,
            clean_prompt=CLEAN_PROMPT,
            corrupted_prompt=CORR_PROMPT,
            correct_token_id=CORRECT_ID,
            incorrect_token_id=INCORRECT_ID,
            top_k_neurons=20,
        )
        assert isinstance(result, PatchingResult)
        assert result.mode == "approx_neuron"
        assert result.n_neurons == model.config.intermediate_size
        assert len(result.cells) == 20

    def test_cells_have_required_fields(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            top_k_neurons=10,
        )
        for c in result.cells:
            assert "layer" in c and isinstance(c["layer"], int)
            assert "unit" in c and c["unit"] == f"neuron.n{c['neuron']}"
            assert "neuron" in c and isinstance(c["neuron"], int)
            assert "position" in c and isinstance(c["position"], int)
            assert "ap_recovery" in c and isinstance(c["ap_recovery"], float)
            assert 0 <= c["neuron"] < model.config.intermediate_size

    def test_cells_sorted_desc_by_abs(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            top_k_neurons=30,
        )
        mags = [abs(c["ap_recovery"]) for c in result.cells]
        assert mags == sorted(mags, reverse=True)

    def test_top_k_exceeds_total_caps(self) -> None:
        model, tok = _make_mock()
        result = attribution_patch_per_neuron(
            model, tok, CLEAN_PROMPT, CORR_PROMPT,
            correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            top_k_neurons=10**9,
        )
        # Total = n_layers * intermediate_size * seq_len
        expected = (
            model.config.num_hidden_layers
            * model.config.intermediate_size
            * len(tok(CLEAN_PROMPT)["input_ids"][0])
        )
        assert len(result.cells) == expected

    def test_sum_invariant_mock(self) -> None:
        """Σ_i ap_neuron_raw(L, i, pos) == (Δffn_out · grad_ffn_out)_pos.

        Computes the right-hand side directly from captures and compares
        against the left-hand side (reconstructed by summing per-neuron
        ap_raw from an undivided-by-D path).
        """
        import torch
        model, tok = _make_mock()

        from_prompt = CLEAN_PROMPT
        base_prompt = CORR_PROMPT
        from_ids = tok(from_prompt)["input_ids"]
        base_ids = tok(base_prompt)["input_ids"]

        # Capture from pass (no grad)
        with torch.no_grad():
            from_captured, _, from_logits, _, _, _, from_ffn_acts = \
                _capture_residual_stream_with_grad(
                    model, tok, from_prompt,
                    sublayers=("attn", "ffn"),
                    capture_ffn_out=True,
                    capture_ffn_act=True,
                )

        # Capture base pass (with grad)
        with torch.enable_grad():
            base_captured, _, base_logits, _, _, _, base_ffn_acts = \
                _capture_residual_stream_with_grad(
                    model, tok, base_prompt,
                    sublayers=("attn", "ffn"),
                    capture_ffn_out=True,
                    capture_ffn_act=True,
                )
            meas_pos = base_logits.shape[0] - 1
            metric = (
                base_logits[meas_pos, CORRECT_ID]
                - base_logits[meas_pos, INCORRECT_ID]
            )
            metric.backward()

        # For each layer, at every position, verify the invariant.
        for L in range(model.config.num_hidden_layers):
            if (L, "ffn_out") not in base_captured:
                continue
            base_ffn_out = base_captured[(L, "ffn_out")]
            from_ffn_out_L = from_captured[(L, "ffn_out")]
            if base_ffn_out.grad is None:
                continue
            W_down = model.model.layers[L].mlp.down_proj.weight
            for pos in range(base_ffn_out.shape[1]):
                grad_ffn_out = base_ffn_out.grad[0, pos].detach()
                delta_ffn_out = (from_ffn_out_L[0, pos] - base_ffn_out[0, pos].detach())
                target = (delta_ffn_out * grad_ffn_out).sum().item()

                grad_act = grad_ffn_out @ W_down
                delta_act = from_ffn_acts[L][0, pos] - base_ffn_acts[L][0, pos].detach()
                reconstructed = (delta_act * grad_act).sum().item()

                assert abs(target - reconstructed) < 1e-4, \
                    f"Sum invariant broken at L={L}, pos={pos}: target={target}, sum={reconstructed}"

    def test_validation(self) -> None:
        model, tok = _make_mock()
        with pytest.raises(ValueError, match="top_k_neurons"):
            attribution_patch_per_neuron(
                model, tok, CLEAN_PROMPT, CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                top_k_neurons=0,
            )
        with pytest.raises(ValueError, match="prompts cannot be empty"):
            attribution_patch_per_neuron(
                model, tok, "", CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            )
        with pytest.raises(ValueError, match="same length"):
            attribution_patch_per_neuron(
                model, tok, "a b c", "d e",
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
            )
        with pytest.raises(ValueError, match="direction"):
            attribution_patch_per_neuron(
                model, tok, CLEAN_PROMPT, CORR_PROMPT,
                correct_token_id=CORRECT_ID, incorrect_token_id=INCORRECT_ID,
                direction="nonsense",
            )


class TestCaptureFFNAct:
    def test_capture_ffn_act_flag_populates_dict(self) -> None:
        import torch
        model, tok = _make_mock()
        with torch.no_grad():
            out = _capture_residual_stream_with_grad(
                model, tok, CLEAN_PROMPT,
                sublayers=("attn", "ffn"),
                capture_ffn_act=True,
            )
        assert len(out) == 7
        ffn_acts = out[6]
        assert isinstance(ffn_acts, dict)
        for L in range(model.config.num_hidden_layers):
            assert L in ffn_acts
            assert ffn_acts[L].shape[-1] == model.config.intermediate_size

    def test_capture_ffn_act_false_default_empty(self) -> None:
        import torch
        model, tok = _make_mock()
        with torch.no_grad():
            out = _capture_residual_stream_with_grad(
                model, tok, CLEAN_PROMPT,
                sublayers=("attn", "ffn"),
                # capture_ffn_act defaults to False
            )
        ffn_acts = out[6]
        assert ffn_acts == {}


# -------------------------------------------------------------------------
# TinyLlama integration (skipif GPU missing; fp16 to avoid OOM on 8GB)
# -------------------------------------------------------------------------

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestTinyLlamaPerNeuron:
    def test_top_neurons_identifiable(self) -> None:
        from llm_surgeon.surgery import load_model
        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        clean = "The capital of France is"
        corrupted = "The capital of Italy is"
        paris_id = int(tok(" Paris", return_tensors="pt")["input_ids"][0, 1].item())
        rome_id = int(tok(" Rome", return_tensors="pt")["input_ids"][0, 1].item())

        r = attribution_patch_per_neuron(
            model, tok, clean, corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=rome_id,
            direction="denoise",
            top_k_neurons=50,
        )
        assert r.mode == "approx_neuron"
        assert r.n_neurons == 5632
        assert len(r.cells) == 50
        for c in r.cells:
            assert 0 <= c["neuron"] < 5632
            assert 0 <= c["layer"] < 22
        mags = [abs(c["ap_recovery"]) for c in r.cells]
        assert mags == sorted(mags, reverse=True)
        assert mags[0] > 0.001, "top neuron should have nonzero AP on a real task"
```

Also make sure the imports at the top of the file include:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import pytest
import torch
from torch import nn

from llm_surgeon.probe import (
    PatchingResult,
    attribution_patch_per_neuron,
    _capture_residual_stream_with_grad,
)
```

- [ ] **Step 2: Run mock tests (non-GPU)**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest testing/tests/test_probe_per_neuron_ap.py -v -k "not TinyLlama"
```
Expected: 7 or 8 tests pass (TestPerNeuronMock × 6 + TestCaptureFFNAct × 2).

If any test fails with a denominator-zero error, the mock model or prompts need adjustment — use the same `_pick_tokens` helper from Phase 3.8's tests and prompts that differ in the last token.

- [ ] **Step 3: Run TinyLlama integration (GPU required)**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest testing/tests/test_probe_per_neuron_ap.py::TestTinyLlamaPerNeuron -v -s
```
Expected: passes in ~1-2 min on RTX 2080 (fp16).

- [ ] **Step 4: Full regression — all Phase 3.5-3.9 tests**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py testing/tests/test_probe_per_head_ap.py testing/tests/test_probe_edge_ap.py testing/tests/test_probe_circuit.py testing/tests/test_probe_per_neuron_ap.py -v -k "not TinyLlama"
```
Expected: all pass (~62 tests).

- [ ] **Step 5: Pyright clean**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pyright testing/tests/test_probe_per_neuron_ap.py testing/llm_surgeon/probe.py
```
Expected: 0/0/0.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/tests/test_probe_per_neuron_ap.py
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
test(probe): mock + TinyLlama tests for attribution_patch_per_neuron

Mock-model suite (6 tests): shape, cell fields, sort-desc invariant,
top_k cap, validation, sum invariant (Σ_i ap_neuron_raw == Δffn_out ·
grad_ffn_out). Capture-helper suite (2 tests): capture_ffn_act
populates per-layer acts; default False yields empty dict.

TinyLlama integration (1 test, GPU/fp16): n_neurons==5632, all indices
within bounds, sort invariant, top neuron has nonzero AP on
capital-of-France task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Backend WS `approx_neuron` branch

**Files:**
- Modify: `testing/gui/backend/routes/probes.py`

- [ ] **Step 1: Find the current mode branches**

Run:
```bash
grep -n 'cfg.mode ==' testing/gui/backend/routes/probes.py
```
Note the line numbers — there should be branches for `"exact"`, `"approx"`, `"approx_head"`, `"edge"`, and `"circuit"`. You'll add `"approx_neuron"` as the sixth.

- [ ] **Step 2: Import `attribution_patch_per_neuron`**

Find the `from llm_surgeon.probe import (...)` block inside the AP route handler. Add `attribution_patch_per_neuron,` alongside `attribution_patch_per_head`.

- [ ] **Step 3: Extend the Pydantic config model**

Find the cfg Pydantic model (the one that already accepts `top_k_edges`, `top_k_candidates`, `tau`). Add:

```python
    top_k_neurons: int = 200
```

Update the `mode` type constraint to include `"approx_neuron"`.

- [ ] **Step 4: Add the `approx_neuron` branch**

After the existing `elif cfg.mode == "circuit":` branch, insert:

```python
        elif cfg.mode == "approx_neuron":
            top_k_neurons = int(getattr(cfg, "top_k_neurons", 200))
            if top_k_neurons < 1:
                raise HTTPException(status_code=400, detail="top_k_neurons must be >= 1")

            def on_cell_neuron(cell: dict) -> None:
                loop.call_soon_threadsafe(
                    asyncio.create_task,
                    ws.send_json({"type": "data", **cell}),
                )

            result = await asyncio.to_thread(
                attribution_patch_per_neuron,
                info.model,
                info.tokenizer,
                cfg.clean_prompt,
                cfg.corrupted_prompt,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
                direction=cfg.direction,
                measurement_position=cfg.measurement_position,
                positions=positions,
                layers=layers,
                top_k_neurons=top_k_neurons,
                on_cell=on_cell_neuron,
            )
            summary_extra = {
                "n_neurons": result.n_neurons,
                "top_k_neurons": top_k_neurons,
            }
```

Important: mirror the structure of the `"circuit"` branch exactly. Read it first and make sure `summary_extra` is assembled into the `complete` frame the same way.

- [ ] **Step 5: Pyright clean**

Run:
```bash
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pyright testing/gui/backend/routes/probes.py
```
Expected: 0/0/0.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/backend/routes/probes.py
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(backend): approx_neuron mode branch on activation-patching WS route

Sixth mode on /ws/sessions/{name}/activation-patching. Accepts
top_k_neurons config. Streams per-cell {layer, unit, neuron,
position, ap_recovery}. Summary extras: n_neurons, top_k_neurons.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Frontend types

**Files:**
- Modify: `testing/gui/frontend/src/types/api.ts`

- [ ] **Step 1: Extend PatchingMode, PatchingCellData, summary**

Find `PatchingMode` and replace:

```ts
export type PatchingMode = "exact" | "approx" | "approx_head" | "edge" | "circuit";
```

with:

```ts
export type PatchingMode = "exact" | "approx" | "approx_head" | "edge" | "circuit" | "approx_neuron";
```

Find `PatchingCellData` and add (in the existing "optional per-mode fields" block):

```ts
  // approx_neuron mode fields
  neuron?: number;
```

(the `layer` and `unit` fields are already optional on the type from Phase 3.6.)

Find `PatchingCompleteData.summary` and add:

```ts
    n_neurons?: number;
    top_k_neurons?: number;
```

- [ ] **Step 2: Tsc clean**

Run:
```bash
cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/types/api.ts
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(gui/frontend): approx_neuron mode types

PatchingMode extends to 'approx_neuron'. PatchingCellData gains
optional neuron index. Summary gains optional n_neurons and
top_k_neurons.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `PatchingControls` — sixth radio + `top_k_neurons` input

**Files:**
- Modify: `testing/gui/frontend/src/components/PatchingControls.tsx`
- Modify: `testing/gui/frontend/src/components/ProbePanel.tsx`

- [ ] **Step 1: Extend `PatchingMode`, `PatchingState`, `DEFAULT_PATCHING_STATE`**

In `PatchingControls.tsx`, replace:

```ts
export type PatchingMode = "exact" | "approx" | "approx_head" | "edge" | "circuit";
```

with:

```ts
export type PatchingMode = "exact" | "approx" | "approx_head" | "edge" | "circuit" | "approx_neuron";
```

In `PatchingState`, add:

```ts
  top_k_neurons: number;
```

In `DEFAULT_PATCHING_STATE`, add:

```ts
  top_k_neurons: 200,
```

- [ ] **Step 2: Add sixth radio + conditional input**

Find the circuit radio block. After it, insert:

```tsx
              <label>
                <input
                  type="radio"
                  checked={state.mode === "approx_neuron"}
                  onChange={() => onChange({ mode: "approx_neuron" })}
                />
                per-neuron FFN (approx)
              </label>
```

After the circuit conditional input block (the one showing `tau` + `top_k_candidates`), insert:

```tsx
          {state.mode === "approx_neuron" && (
            <div className="row">
              <label>
                top_k_neurons:
                <input
                  type="number"
                  min={1}
                  value={state.top_k_neurons}
                  onChange={(e) =>
                    onChange({ top_k_neurons: Math.max(1, Number(e.target.value)) })
                  }
                />
              </label>
            </div>
          )}
```

Update the auto-pick info-line condition to include `"approx_neuron"`:

```tsx
          {(state.mode === "approx" || state.mode === "approx_head" || state.mode === "edge" || state.mode === "circuit" || state.mode === "approx_neuron") && state.tokenPairMode === "auto" && (
```

- [ ] **Step 3: Forward `top_k_neurons` from `ProbePanel.tsx`**

Find the existing mode-switched cfg forwarding in `ProbePanel.tsx`. It currently looks like:

```ts
        if (patchingState.mode === "edge") {
          cfg.top_k_edges = patchingState.top_k_edges;
        } else if (patchingState.mode === "circuit") {
          cfg.top_k_candidates = patchingState.top_k_candidates;
          cfg.tau = patchingState.tau;
        }
```

Extend to:

```ts
        if (patchingState.mode === "edge") {
          cfg.top_k_edges = patchingState.top_k_edges;
        } else if (patchingState.mode === "circuit") {
          cfg.top_k_candidates = patchingState.top_k_candidates;
          cfg.tau = patchingState.tau;
        } else if (patchingState.mode === "approx_neuron") {
          cfg.top_k_neurons = patchingState.top_k_neurons;
        }
```

- [ ] **Step 4: Tsc clean**

Run:
```bash
cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/PatchingControls.tsx testing/gui/frontend/src/components/ProbePanel.tsx
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(gui/frontend): per-neuron FFN (approx_neuron) mode radio + input

Sixth radio in PatchingControls. Conditional top_k_neurons numeric
input shown only in approx_neuron mode. ProbePanel forwards
top_k_neurons into the WS cfg.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `PerNeuronPatchingPanel.tsx` + `VisualizationArea` routing

**Files:**
- Create: `testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx`
- Modify: `testing/gui/frontend/src/components/VisualizationArea.tsx`

- [ ] **Step 1: Read EdgeAttributionPanel's Top-list tab for table-styling idiom**

Run:
```bash
grep -n "Top-list\|TopList\|<table" testing/gui/frontend/src/components/visualizations/EdgeAttributionPanel.tsx
```
Mirror the table styling (row colors, hover states) if possible. If EdgeAttributionPanel uses a different mechanism, adapt — don't copy blindly.

- [ ] **Step 2: Write `PerNeuronPatchingPanel.tsx`**

Create `testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx`:

```tsx
import { useMemo, useState } from "react";
import type { PatchingCellData, PatchingCompleteData } from "../../types/api";

interface Props {
  cells: PatchingCellData[];
  complete?: PatchingCompleteData;
}

type SortKey = "ap_recovery" | "layer" | "neuron";

function apColor(ap: number): string {
  // Piecewise PiYG mapped to [-0.5, 1.0] — same as Phase 3.5+.
  const clamped = Math.max(-0.5, Math.min(1.0, ap));
  if (clamped >= 0) {
    const t = clamped;
    const g = Math.floor(200 * t + 30);
    return `rgb(30, ${g}, 60)`;
  } else {
    const t = -clamped / 0.5;
    const r = Math.floor(180 * t + 40);
    return `rgb(${r}, 40, 80)`;
  }
}

export function PerNeuronPatchingPanel({ cells, complete }: Props) {
  const neuronCells = useMemo(
    () => cells.filter((c) => c.neuron !== undefined && c.layer !== undefined),
    [cells],
  );

  const positions = useMemo(() => {
    const s = new Set<number>();
    for (const c of neuronCells) if (c.position !== undefined) s.add(c.position);
    return Array.from(s).sort((a, b) => a - b);
  }, [neuronCells]);

  const layers = useMemo(() => {
    const s = new Set<number>();
    for (const c of neuronCells) if (c.layer !== undefined) s.add(c.layer);
    return Array.from(s).sort((a, b) => a - b);
  }, [neuronCells]);

  const initialPos =
    complete?.summary?.measurement_position ?? positions[positions.length - 1] ?? 0;

  const [selectedPos, setSelectedPos] = useState<number | "all">(initialPos);
  const [selectedLayer, setSelectedLayer] = useState<number | "all">("all");
  const [neuronSearch, setNeuronSearch] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("ap_recovery");
  const [sortDesc, setSortDesc] = useState<boolean>(true);

  const visible = useMemo(() => {
    let rows = neuronCells;
    if (selectedPos !== "all") rows = rows.filter((c) => c.position === selectedPos);
    if (selectedLayer !== "all") rows = rows.filter((c) => c.layer === selectedLayer);
    if (neuronSearch.trim() !== "") {
      const q = neuronSearch.trim();
      const qNum = Number(q);
      if (!Number.isNaN(qNum)) {
        rows = rows.filter((c) => c.neuron === qNum);
      }
    }
    const cmp = (a: PatchingCellData, b: PatchingCellData): number => {
      const aa = (a[sortKey] ?? 0) as number;
      const bb = (b[sortKey] ?? 0) as number;
      if (sortKey === "ap_recovery") {
        return Math.abs(bb) - Math.abs(aa);
      }
      return aa - bb;
    };
    const sorted = [...rows].sort(cmp);
    return sortDesc ? sorted : sorted.reverse();
  }, [neuronCells, selectedPos, selectedLayer, neuronSearch, sortKey, sortDesc]);

  const stats = useMemo(() => {
    if (visible.length === 0) return { min: 0, max: 0, mean: 0 };
    let min = Infinity, max = -Infinity, sum = 0;
    for (const c of visible) {
      const v = c.ap_recovery ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    return { min, max, mean: sum / visible.length };
  }, [visible]);

  const exportTSV = () => {
    const header = "layer\tneuron\tposition\tap_recovery";
    const body = visible
      .map((c) => `${c.layer}\t${c.neuron}\t${c.position}\t${(c.ap_recovery ?? 0).toFixed(6)}`)
      .join("\n");
    navigator.clipboard?.writeText(`${header}\n${body}`).catch(() => undefined);
  };

  return (
    <div className="per-neuron-panel">
      <h3>Per-Neuron FFN Attribution</h3>
      <div className="controls" style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
        <label>
          position:
          <select
            value={selectedPos}
            onChange={(e) => {
              const v = e.target.value;
              setSelectedPos(v === "all" ? "all" : Number(v));
            }}
          >
            <option value="all">all</option>
            {positions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <label>
          layer:
          <select
            value={selectedLayer}
            onChange={(e) => {
              const v = e.target.value;
              setSelectedLayer(v === "all" ? "all" : Number(v));
            }}
          >
            <option value="all">all</option>
            {layers.map((L) => (
              <option key={L} value={L}>
                L{L}
              </option>
            ))}
          </select>
        </label>
        <label>
          neuron#:
          <input
            type="text"
            placeholder="e.g. 1234"
            value={neuronSearch}
            onChange={(e) => setNeuronSearch(e.target.value)}
            style={{ width: 80 }}
          />
        </label>
        <label>
          sort:
          <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
            <option value="ap_recovery">|ap_recovery|</option>
            <option value="layer">layer</option>
            <option value="neuron">neuron</option>
          </select>
        </label>
        <button onClick={() => setSortDesc((d) => !d)}>{sortDesc ? "↓" : "↑"}</button>
        <button onClick={exportTSV}>copy TSV</button>
      </div>

      <div className="stats" style={{ color: "#aaa", marginBottom: 8 }}>
        Showing <b>{visible.length}</b> of {neuronCells.length} cells. min={stats.min.toFixed(4)}, max={stats.max.toFixed(4)}, mean={stats.mean.toFixed(4)}.
      </div>

      <div style={{ maxHeight: 600, overflowY: "auto", border: "1px solid #333" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "monospace", fontSize: 12 }}>
          <thead style={{ position: "sticky", top: 0, background: "#1a1a1a" }}>
            <tr>
              <th style={{ textAlign: "left", padding: 4 }}>#</th>
              <th style={{ textAlign: "left", padding: 4 }}>layer</th>
              <th style={{ textAlign: "left", padding: 4 }}>neuron</th>
              <th style={{ textAlign: "left", padding: 4 }}>pos</th>
              <th style={{ textAlign: "right", padding: 4 }}>ap_recovery</th>
            </tr>
          </thead>
          <tbody>
            {visible.slice(0, 200).map((c, i) => (
              <tr key={`${c.layer}-${c.neuron}-${c.position}`} style={{ background: apColor(c.ap_recovery ?? 0) }}>
                <td style={{ padding: 4 }}>{i + 1}</td>
                <td style={{ padding: 4 }}>L{c.layer}</td>
                <td style={{ padding: 4 }}>n{c.neuron}</td>
                <td style={{ padding: 4 }}>{c.position}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{(c.ap_recovery ?? 0).toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Wire routing in `VisualizationArea.tsx`**

Find the existing `mode === "circuit"` / `"edge"` / `"approx_head"` routing. Add `"approx_neuron"` as a new branch BEFORE the `approx_head` branch (keep the order mode-specific → fallback):

```tsx
mode === "circuit" ? (
  <CircuitPanel cells={cellMsgs} complete={completeMsg} />
) : mode === "edge" ? (
  <EdgeAttributionPanel ... />
) : mode === "approx_neuron" ? (
  <PerNeuronPatchingPanel cells={cellMsgs} complete={completeMsg} />
) : mode === "approx_head" ? (
  <PerHeadPatchingHeatmap ... />
) : (
  <ActivationPatchingHeatmap ... />
)
```

Add the import at the top of the file:

```tsx
import { PerNeuronPatchingPanel } from "./visualizations/PerNeuronPatchingPanel";
```

- [ ] **Step 4: Tsc clean**

Run:
```bash
cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/visualizations/PerNeuronPatchingPanel.tsx testing/gui/frontend/src/components/VisualizationArea.tsx
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
feat(gui/frontend): PerNeuronPatchingPanel — ranked-list viz for per-neuron AP

Ranked table (layer × neuron × position × ap_recovery) with:
- position selector (+ "all")
- layer filter (+ "all")
- neuron-id search
- sort by |ap_recovery|, layer, or neuron index
- stats strip (min/max/mean of visible rows)
- copy TSV export

No heatmap — intermediate_size=5632 makes a full grid unrenderable.
VisualizationArea routes mode === 'approx_neuron' to this panel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Playwright fixture + smoke test

**Files:**
- Create: `testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-neuron.json`
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Read the per-head fixture for the schema**

Run:
```bash
cat testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-head.json
```
(Use Read.) Note the outer `"schema": "llm-surgeon-gui-experiment/v1"` wrapper and result shape.

- [ ] **Step 2: Write the fixture**

Create `testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-neuron.json` — copy the per-head fixture's skeleton and:
- Change the result's summary `mode` to `"approx_neuron"`.
- Add summary fields: `n_neurons: 5632`, `top_k_neurons: 5`.
- Replace cells with 5 per-neuron cells, each of shape:
  ```json
  {
    "type": "data",
    "layer": 10,
    "unit": "neuron.n1234",
    "neuron": 1234,
    "position": 4,
    "ap_recovery": 0.12
  }
  ```
- Vary `layer` (e.g., 5, 10, 12, 15, 21), `neuron` (e.g., 42, 1234, 2500, 3333, 5000), and `ap_recovery` (e.g., 0.45, 0.32, 0.18, -0.11, 0.05) so the sort-desc-by-abs ordering is demonstrable.

- [ ] **Step 3: Add 15th Playwright test**

Append to `testing/gui/frontend/tests/e2e/smoke.spec.ts` (after the 14th circuit test):

```ts
const PER_NEURON_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-per-neuron.json");

test("per-neuron FFN panel renders with table and filters", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(PER_NEURON_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-neuron.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  await page.getByRole("heading", { name: /Per-Neuron FFN Attribution/i })
    .waitFor({ state: "visible", timeout: 5000 });

  // Stats strip
  await expect(page.getByText(/Showing \d+ of \d+ cells/i)).toBeVisible();

  // Table header
  await expect(page.getByRole("columnheader", { name: /ap_recovery/i })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: /neuron/i })).toBeVisible();

  // Copy TSV button
  await expect(page.getByRole("button", { name: /copy tsv/i })).toBeVisible();

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});
```

Also add `const PER_NEURON_FIXTURE_PATH = ...` near the other `FIXTURE_PATH` declarations at the top of the file if your test style defines them there.

- [ ] **Step 4: Run Playwright**

Run:
```bash
cd testing/gui/frontend && npm run e2e
```
Expected: 15/15 tests pass.

- [ ] **Step 5: Final verification matrix**

Run each:
```bash
# Tsc
cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
# Vitest (regression — Phase 3.8 BFS + existing)
cd testing/gui/frontend && npx vitest run
# Pyright
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pyright testing/llm_surgeon/probe.py testing/gui/backend/routes/probes.py testing/tests/test_probe_per_neuron_ap.py
# All Python tests (no GPU)
/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest testing/tests/ -v -k "not TinyLlama"
# Playwright
cd testing/gui/frontend && npm run e2e
```
Expected: all green, 0/0/0 pyright, 62+ Python tests pass, Vitest 19/19, Playwright 15/15.

- [ ] **Step 6: Commit + update roadmap memory**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-neuron.json testing/gui/frontend/tests/e2e/smoke.spec.ts
git -C /home/ai/ai-projects/llm commit -m "$(cat <<'EOF'
test(gui/frontend): Playwright smoke for PerNeuronPatchingPanel

15th test. Imports activation-patching-per-neuron.json fixture
(5 per-neuron cells, mode='approx_neuron', n_neurons=5632) and
asserts the panel heading, stats strip, table columns, and copy-TSV
button render.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Then update the roadmap memory at `~/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md` with a Phase 3.9 shipped entry (commit SHAs + verification matrix + subagent lessons) matching the Phase 3.8 entry format.

---

## Verification Matrix (run before declaring phase shipped)

| Check | Command | Expected |
|-------|---------|----------|
| Pyright | `.venv/bin/python -m pyright testing/llm_surgeon/probe.py testing/gui/backend/routes/probes.py testing/tests/test_probe_*.py` | 0/0/0 |
| Tsc | `cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit` | clean |
| Python unit (no GPU) | `.venv/bin/python -m pytest testing/tests/ -v -k "not TinyLlama"` | 62+ pass |
| Python TinyLlama (GPU) | `.venv/bin/python -m pytest testing/tests/test_probe_per_neuron_ap.py::TestTinyLlamaPerNeuron -v` | pass ~1–2 min |
| Phase 3.5 regression | `.venv/bin/python -m pytest testing/tests/test_probe_attribution_patch.py::TestTinyLlamaAttributionPatch -v` | ρ=0.956 preserved |
| Phase 3.6 regression | `.venv/bin/python -m pytest testing/tests/test_probe_per_head_ap.py::TestTinyLlamaPerHead -v` | ρ=1.0000 preserved |
| Phase 3.7 regression | `.venv/bin/python -m pytest testing/tests/test_probe_edge_ap.py::TestTinyLlamaEAP -v` | top-k consistency passes |
| Phase 3.8 regression | `.venv/bin/python -m pytest testing/tests/test_probe_circuit.py::TestTinyLlamaCircuit -v` | circuit at tau=0.02 passes |
| Vitest | `cd testing/gui/frontend && npx vitest run` | 19/19 |
| Playwright | `cd testing/gui/frontend && npm run e2e` | 15/15 |

---

## Plan Self-Review Notes

**Spec coverage:**
- §2 G1 → Task 2.
- §2 G2 → Task 4.
- §2 G3 → Task 7.
- §2 G4 → Task 1 (7-tuple + retain_grad).
- §4 math → Task 2 (`attribution_patch_per_neuron` body mirrors the spec's pseudocode exactly).
- §5.1 signature → Task 2 exactly.
- §5.2 `n_neurons` field → Task 2 Step 1.
- §5.3 capture helper extension → Task 1.
- §5.4 implementation sketch → Task 2 Step 2.
- §6 WS protocol → Task 4.
- §7.1 types → Task 5.
- §7.2 controls → Task 6.
- §7.3 panel → Task 7.
- §7.4 fixture + smoke → Task 8.
- §8.1 Python unit tests → Task 3 (9 tests covering all §8.1 items).
- §8.2 TinyLlama integration → Task 3.
- §8.3 Vitest/Playwright → Task 8 (Playwright) + no Vitest per spec.
- §9 commit plan → Tasks 1-8 map to spec commits.

**Placeholder scan:** None.

**Type consistency:**
- `top_k_neurons` same name across Python signature, WS cfg, Pydantic model, frontend `PatchingState`, `PatchingControls` prop, and `PerNeuronPatchingPanel` summary reference.
- `n_neurons` same name Python → WS → frontend.
- Cell field `neuron` (int) same across Python emission and TypeScript optional field.
- `mode` string `"approx_neuron"` same everywhere.

Ready for execution.
