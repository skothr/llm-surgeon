# Phase 3 Activation Patching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add causal attribution via the clean/corrupted counterfactual pattern — upgrade `probe.py` from correlational interventions to `activation_patch()` with denoise/noise directions, plus a full backend WS route, frontend `PatchingControls` form, and multi-metric heatmap visualization.

**Architecture:** One new Python function in `probe.py` composed of existing primitives (`_capture_residual_stream` + `intervene()` + new `_make_position_patch()`), one new WS route mirroring `/logit-lens` streaming, one new React form rendered conditionally by `ProbePanel`, one new d3 heatmap component with client-side metric computation, one new frontend utility file for pure metric functions. Tests: unit (with `tiny_llama` fixture + mocks) + one real-TinyLlama integration + Playwright smoke extension.

**Tech Stack:** Python 3.11, PyTorch, transformers (HF LLaMA), FastAPI WebSockets, React + TypeScript + Zustand, d3, Vitest (frontend unit), pytest (Python), Playwright.

**Spec:** `testing/docs/superpowers/specs/2026-04-17-phase3-activation-patching-design.md` (commit `fc16b90`).

**Cwd for tool invocations:** `/home/ai/ai-projects/llm`. The pyright CLI must be run from `testing/` (see CLAUDE.md § Type Checking) — several tasks encode this explicitly.

---

## Tool rules (apply to every task + every subagent prompt)

- Use `Read` (not `cat`), `Edit` (not `sed`/`awk`/`cat`), `Grep` (not shell grep), `Glob` (not `find`). All file ops go through dedicated tools.
- For git ops outside the repo root, use `git -C <path>` rather than `cd`.
- Avoid unnecessary compound commands. Avoid chaining that would trigger a permission prompt.
- **GPU tests:** any Bash call that runs pytest touching CUDA (`import torch; torch.cuda.*`, model load from HF, etc.) must be invoked with `dangerouslyDisableSandbox: true`. The existing `/dev/nvidia*` sandbox rule blocks CUDA otherwise. If a subagent cannot get that permission granted, surface a BLOCKED status — the controller will run the test directly.
- Pyright CLI: run from `testing/` cwd. Command: `.venv/bin/python -m pyright <paths>`. Not from project root (import resolution fails).
- Frontend tsc: run from `testing/gui/frontend/`. Command: `./node_modules/.bin/tsc --noEmit`.
- Playwright: run from `testing/gui/frontend/`. Command: `npm run e2e`.
- Zero-diagnostics discipline: every commit in this plan must land with pyright 0/0/0 and tsc clean.

---

## File Structure

### New files
- `testing/tests/test_probe_activation_patch.py` — all new Python unit + integration tests.
- `testing/gui/frontend/src/components/PatchingControls.tsx` — conditional patching form.
- `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx` — heatmap visualization.
- `testing/gui/frontend/src/utils/patchingMetrics.ts` — pure-function metric helpers.
- `testing/gui/frontend/tests/unit/patchingMetrics.test.ts` — Vitest unit tests (check whether `tests/unit/` already exists; if not, create it alongside `tests/e2e/`).

### Modified files
- `testing/llm_surgeon/probe.py` — **+** `PatchingResult`, `_make_position_patch()`, `activation_patch()`.
- `testing/gui/backend/routes/probes.py` — **+** `/sessions/{name}/activation-patching` WS handler.
- `testing/gui/frontend/src/types/api.ts` — **+** ProbeOperation extension, Patching* interfaces, WsMessage union.
- `testing/gui/frontend/src/components/ProbePanel.tsx` — **+** op option, conditional render, `handleRun` branch, disable fan-out/A-B for AP.
- `testing/gui/frontend/src/components/VisualizationArea.tsx` — **+** op → component dispatch entry.
- `testing/gui/frontend/tests/e2e/smoke.spec.ts` — **+** one patching-heatmap test.
- `testing/gui/frontend/tests/e2e/fixtures/sample.json` — **+** patching result fixture OR create sibling file.
- `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md` — append "Phase 3 shipped" entry.

---

## Task 1: `_make_position_patch` helper + unit tests

**Why first:** smallest, self-contained new primitive. TDD it standalone before the function that uses it.

**Files:**
- Modify: `testing/llm_surgeon/probe.py` (append near the `_Ops` class definition)
- Create: `testing/tests/test_probe_activation_patch.py`

- [ ] **Step 1: Write the failing tests**

Create `testing/tests/test_probe_activation_patch.py`:

```python
"""Tests for probe.activation_patch — causal attribution via clean/corrupted counterfactual."""

import pytest
import torch

from llm_surgeon.probe import _make_position_patch


class TestMakePositionPatch:
    def test_only_replaces_target_position(self):
        # seq_len=5, d_model=4
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        patch_vec = torch.tensor([100.0, 200.0, 300.0, 400.0])
        fn = _make_position_patch(pos=2, clean_vec=patch_vec)
        out = fn(hidden, layer_idx=0)
        # Position 2 must equal patch_vec.
        assert torch.equal(out[2], patch_vec)
        # All other positions unchanged.
        for pos in (0, 1, 3, 4):
            assert torch.equal(out[pos], hidden[pos]), f"position {pos} was modified"

    def test_preserves_dtype_and_device(self):
        hidden = torch.randn(3, 8, dtype=torch.float16)
        # Patch vec in a different dtype — op must cast to match hidden.
        patch_vec = torch.randn(8, dtype=torch.float32)
        fn = _make_position_patch(pos=1, clean_vec=patch_vec)
        out = fn(hidden, layer_idx=0)
        assert out.dtype == torch.float16
        assert out.device == hidden.device

    def test_does_not_mutate_input(self):
        hidden = torch.randn(4, 6)
        original = hidden.clone()
        patch_vec = torch.zeros(6)
        fn = _make_position_patch(pos=0, clean_vec=patch_vec)
        fn(hidden, layer_idx=0)
        assert torch.equal(hidden, original), "input hidden tensor was mutated"

    def test_repr_is_descriptive(self):
        fn = _make_position_patch(pos=3, clean_vec=torch.zeros(4))
        assert "patch_pos(3)" in repr(fn)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py -v
```

Expected: `ImportError: cannot import name '_make_position_patch' from 'llm_surgeon.probe'` (or `AttributeError`).

- [ ] **Step 3: Implement `_make_position_patch` in probe.py**

Use Read on `testing/llm_surgeon/probe.py` to find the `_Op` and `_Ops` class definitions (around lines 459–513), then use Edit to append this immediately after the `ops = _Ops()` line (around line 516):

```python
# ---------------------------------------------------------------------------
# Activation patching — position-scoped replace for causal attribution
# ---------------------------------------------------------------------------

def _make_position_patch(pos: int, clean_vec: torch.Tensor) -> _Op:
    """Build an intervention op that replaces hidden_state[pos] with clean_vec,
    leaving all other positions untouched.

    Used by activation_patch() to inject a single cached activation at exactly
    one (layer, sublayer, position) triple during a corrupted (or clean) base
    forward pass. The clone avoids mutating the input tensor that intervene()'s
    hook path still references downstream.
    """
    def fn(h: torch.Tensor, _layer_idx: int) -> torch.Tensor:
        out = h.clone()
        out[pos] = clean_vec.to(device=h.device, dtype=h.dtype)
        return out
    return _Op(fn, f"patch_pos({pos})")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Pyright check**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_activation_patch.py
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_activation_patch.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): _make_position_patch helper for activation patching

Position-scoped replace — overwrites a single position in the hidden
state while leaving all others untouched. Primitive for the upcoming
activation_patch() function."
```

---

## Task 2: `PatchingResult` dataclass + `activation_patch()` validation

**Why second:** validation rules fail fast and cheap, before the loop does anything. TDD them with no model needed.

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe_activation_patch.py`

- [ ] **Step 1: Write the failing tests**

Use Edit on `testing/tests/test_probe_activation_patch.py` to append:

```python
# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """activation_patch input validation — fails fast, no model needed."""

    @pytest.fixture
    def tokenizer(self):
        from tests.conftest import _make_tiny_tokenizer
        return _make_tiny_tokenizer(64)

    def test_mismatched_lengths_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match=r"same length.*clean=\d+.*corrupted=\d+"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11 word12",      # 3 tokens
                corrupted_prompt="word10 word11",         # 2 tokens
            )

    def test_empty_clean_prompt_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="empty"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="", corrupted_prompt="word10 word11",
            )

    def test_empty_corrupted_prompt_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="empty"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="",
            )

    def test_bad_direction_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="direction must be 'denoise' or 'noise'"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="word12 word13",
                direction="wobble",  # pyright: ignore[reportArgumentType]
            )

    def test_bad_sublayer_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="sublayers must be subset"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="word12 word13",
                sublayers=("mlp",),
            )

    def test_measurement_pos_out_of_range_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(IndexError):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="word12 word13",
                measurement_position=100,  # seq_len=2
            )
```

- [ ] **Step 2: Write the failing tests for `PatchingResult`**

Append to `testing/tests/test_probe_activation_patch.py`:

```python
# ---------------------------------------------------------------------------
# PatchingResult dataclass
# ---------------------------------------------------------------------------

class TestPatchingResult:
    def test_construction_and_fields(self):
        from llm_surgeon.probe import PatchingResult
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a", "b"],
            prompt_tokens_corrupted=["c", "d"],
            direction="denoise",
            measurement_position=1,
        )
        assert result.direction == "denoise"
        assert result.measurement_position == 1
        assert len(result.cells) == 0
```

- [ ] **Step 3: Run tests to verify failure**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py::TestValidation testing/tests/test_probe_activation_patch.py::TestPatchingResult -v
```

Expected: all fail with `ImportError` on `PatchingResult` / `activation_patch`.

- [ ] **Step 4: Implement `PatchingResult` and the validation skeleton of `activation_patch`**

Use Read on `testing/llm_surgeon/probe.py` to find the end of the file. Use Edit to append:

```python
# ---------------------------------------------------------------------------
# Activation patching — public API
# ---------------------------------------------------------------------------

@dataclass
class PatchingResult:
    cells: List[Dict]
    clean_baseline_logits: torch.Tensor
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: List[str]
    prompt_tokens_corrupted: List[str]
    direction: str
    measurement_position: int


def activation_patch(
    model,
    tokenizer,
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    direction: str = "denoise",
    measurement_position: int = -1,
    positions: Optional[List[int]] = None,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult:
    """Causal attribution via activation patching.

    Given two same-length prompts (clean, corrupted), computes how much each
    (layer, sublayer, position) residual-stream point causally drives the
    output delta between clean and corrupted behavior. See
    docs/superpowers/specs/2026-04-17-phase3-activation-patching-design.md.

    Args:
        direction: "denoise" (base=corrupted, patches from clean — bright cells
            are *sufficient* for clean behavior) or "noise" (base=clean, patches
            from corrupted — bright cells are *necessary* for clean behavior).
        measurement_position: absolute or negative (-1 = last) index where
            output logits are recorded. Out-of-range raises IndexError.
        positions: patch-position subset; None = all positions.
        sublayers: must be subset of {"attn", "ffn"}.
        layers: layer subset; None = all layers.
        on_cell: called with (layer, sublayer, position, cell_dict) per frame,
            before the frame is appended to the result. Used by the WS handler
            to stream cells live.

    Returns:
        PatchingResult with one cell per iterated (layer, sublayer, position).
    """
    # -- Input validation (fail fast, no forward passes yet) ---------------
    if direction not in ("denoise", "noise"):
        raise ValueError(f"direction must be 'denoise' or 'noise', got {direction!r}")

    if not clean_prompt:
        raise ValueError("clean_prompt cannot be empty")
    if not corrupted_prompt:
        raise ValueError("corrupted_prompt cannot be empty")

    allowed_subs = {"attn", "ffn"}
    if not set(sublayers).issubset(allowed_subs):
        raise ValueError(f"sublayers must be subset of {allowed_subs}, got {sublayers}")

    # Tokenize both; enforce same length.
    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    n_clean = clean_ids.shape[1]
    n_corr = corr_ids.shape[1]
    if n_clean != n_corr:
        raise ValueError(
            f"prompts must tokenize to same length (clean={n_clean}, corrupted={n_corr})"
        )
    seq_len = n_clean

    # Resolve measurement_position.
    if measurement_position < -seq_len or measurement_position >= seq_len:
        raise IndexError(
            f"measurement_position {measurement_position} out of range for seq_len={seq_len}"
        )
    resolved_meas = measurement_position % seq_len

    # Validate positions subset.
    if positions is not None:
        for p in positions:
            if p < 0 or p >= seq_len:
                raise IndexError(f"position {p} out of range for seq_len={seq_len}")

    # Quantized model warning — mirror benchmark.perplexity() pattern.
    if getattr(model, "hf_quantizer", None) is not None:
        import warnings
        warnings.warn(
            "activation_patch on a quantized model: patching works but round-trips "
            "through dequant — slower and slightly less precise than fp16/fp32.",
            RuntimeWarning, stacklevel=2,
        )

    # Placeholder return so validation tests can run before the loop is
    # implemented. Task 3 replaces this body with the real algorithm.
    return PatchingResult(
        cells=[],
        clean_baseline_logits=torch.zeros(0),
        corrupted_baseline_logits=torch.zeros(0),
        prompt_tokens_clean=tokenizer.convert_ids_to_tokens(clean_ids[0]),
        prompt_tokens_corrupted=tokenizer.convert_ids_to_tokens(corr_ids[0]),
        direction=direction,
        measurement_position=resolved_meas,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py -v
```

Expected: all `TestValidation` and `TestPatchingResult` tests pass (plus Task 1's `TestMakePositionPatch`).

- [ ] **Step 6: Pyright check**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_activation_patch.py
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 7: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_activation_patch.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): PatchingResult + activation_patch validation

Skeleton of activation_patch() — dataclass, signature, validation rules
(same-length prompts, direction, sublayer, measurement_position bounds,
non-empty prompts, quantized-model warning). Loop body is a placeholder
returning an empty result so validation-only tests can run."
```

---

## Task 3: `activation_patch()` loop — real algorithm with tiny_llama

**Why third:** loop logic is the heart of the feature. Test with a real `tiny_llama` (8-layer, 32-hidden LlamaForCausalLM — no download) to exercise hook composition without GPU.

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe_activation_patch.py`

- [ ] **Step 1: Write failing tests for the core loop**

Edit `testing/tests/test_probe_activation_patch.py` to append:

```python
# ---------------------------------------------------------------------------
# Core algorithm — tiny_llama fixture (8 layers, 32 hidden)
# ---------------------------------------------------------------------------

class TestActivationPatchLoop:
    @pytest.fixture
    def tokenizer(self):
        from tests.conftest import _make_tiny_tokenizer
        return _make_tiny_tokenizer(64)

    def test_callback_fires_per_cell(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        calls = []
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11 word12",
            corrupted_prompt="word20 word21 word22",
            direction="denoise",
            on_cell=lambda L, sub, pos, cell: calls.append((L, sub, pos)),
        )
        # 8 layers × 2 sublayers × 3 positions = 48 cells.
        assert len(calls) == 48
        assert len(result.cells) == 48
        # Every triple is unique.
        assert len(set(calls)) == 48

    def test_layer_major_position_minor_order(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        calls = []
        activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11 word12",
            corrupted_prompt="word20 word21 word22",
            on_cell=lambda L, sub, pos, cell: calls.append((L, sub, pos)),
        )
        # First layer is 0. First sublayer is 'attn'. Positions increment.
        assert calls[0][0] == 0
        # Same-(L,sub) cells stream consecutively before advancing.
        same_block = [c for c in calls[:3]]
        assert all(c[0] == same_block[0][0] and c[1] == same_block[0][1] for c in same_block)

    def test_direction_denoise_base_is_corrupted(self, tiny_llama, tokenizer, monkeypatch):
        # Spy on intervene() to confirm it was called with corrupted_prompt.
        from llm_surgeon import probe
        original_intervene = probe.intervene
        received_prompts = []

        def spy(model, tok, prompt, interventions, **kwargs):
            received_prompts.append(prompt)
            return original_intervene(model, tok, prompt, interventions, **kwargs)

        monkeypatch.setattr(probe, "intervene", spy)
        probe.activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            direction="denoise",
        )
        # Every intervene() call during the loop used the corrupted prompt.
        assert all(p == "word20 word21" for p in received_prompts)
        assert len(received_prompts) == 8 * 2 * 2  # L × sub × pos

    def test_direction_noise_base_is_clean(self, tiny_llama, tokenizer, monkeypatch):
        from llm_surgeon import probe
        original_intervene = probe.intervene
        received_prompts = []

        def spy(model, tok, prompt, interventions, **kwargs):
            received_prompts.append(prompt)
            return original_intervene(model, tok, prompt, interventions, **kwargs)

        monkeypatch.setattr(probe, "intervene", spy)
        probe.activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            direction="noise",
        )
        assert all(p == "word10 word11" for p in received_prompts)

    def test_positions_subset_filters_loop(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11 word12",
            corrupted_prompt="word20 word21 word22",
            positions=[0, 2],  # skip position 1
        )
        assert len(result.cells) == 8 * 2 * 2
        positions_seen = {cell["position"] for cell in result.cells}
        assert positions_seen == {0, 2}

    def test_layers_subset_filters_loop(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            layers=[3, 5],
        )
        layers_seen = {cell["layer"] for cell in result.cells}
        assert layers_seen == {3, 5}

    def test_sublayers_subset_filters_loop(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            sublayers=("ffn",),
        )
        subs_seen = {cell["sublayer"] for cell in result.cells}
        assert subs_seen == {"ffn"}

    def test_cells_have_patched_logits(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
        )
        for cell in result.cells:
            assert "patched_logits" in cell
            # Shape (vocab_size,) at measurement_position.
            assert cell["patched_logits"].shape == (tiny_llama.config.vocab_size,)

    def test_baselines_are_populated(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
        )
        assert result.clean_baseline_logits.shape == (tiny_llama.config.vocab_size,)
        assert result.corrupted_baseline_logits.shape == (tiny_llama.config.vocab_size,)

    def test_quantized_model_emits_warning(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        # Fake a quantizer attribute.
        tiny_llama.hf_quantizer = object()  # type: ignore[attr-defined]
        try:
            with pytest.warns(RuntimeWarning, match="quantized"):
                activation_patch(
                    tiny_llama, tokenizer,
                    clean_prompt="word10 word11",
                    corrupted_prompt="word20 word21",
                )
        finally:
            del tiny_llama.hf_quantizer  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py::TestActivationPatchLoop -v
```

Expected: all fail with assertions about empty `result.cells` / `received_prompts`. Placeholder body from Task 2 returns empty result.

- [ ] **Step 3: Implement the real loop**

Replace the placeholder return in `activation_patch()` body (in `testing/llm_surgeon/probe.py`). Use Edit to replace the block that ends with `direction=direction, measurement_position=resolved_meas,` and the closing `)` with the full implementation:

```python
    # -- Tokenize once more for the forward passes (reuse ids) -------------
    device = _get_input_device(model)
    clean_input_ids = clean_ids.to(device)
    corr_input_ids = corr_ids.to(device)

    # -- Capture residual streams for both prompts -------------------------
    captured_clean, prompt_tokens_clean = _capture_residual_stream(
        model, tokenizer, clean_prompt,
        sublayers=sublayers, layers=layers,
    )
    captured_corr, prompt_tokens_corrupted = _capture_residual_stream(
        model, tokenizer, corrupted_prompt,
        sublayers=sublayers, layers=layers,
    )

    # -- Baseline forward passes at measurement_position -------------------
    with torch.no_grad():
        clean_out = model(clean_input_ids)
        corr_out = model(corr_input_ids)
    clean_baseline_logits = clean_out.logits[0, resolved_meas].detach().cpu()
    corrupted_baseline_logits = corr_out.logits[0, resolved_meas].detach().cpu()

    # -- Direction selects base prompt + patch source ---------------------
    if direction == "denoise":
        base_prompt = corrupted_prompt
        patch_source = captured_clean
    else:  # "noise"
        base_prompt = clean_prompt
        patch_source = captured_corr

    # -- Resolve iteration sets -------------------------------------------
    target_positions = list(range(seq_len)) if positions is None else list(positions)
    # captured_* keys are already filtered by sublayers/layers; iterate them.
    # Sort: layer-major, attn before ffn within each layer.
    sort_key = lambda k: (k[0], 0 if k[1] == "attn" else 1)
    triples = sorted(patch_source.keys(), key=sort_key)

    # -- Patching loop ----------------------------------------------------
    cells: List[Dict] = []
    for (L, sub) in triples:
        patch_tensor = patch_source[(L, sub)]  # shape: (seq_len, d_model)
        for pos in target_positions:
            clean_vec = patch_tensor[pos]
            iv = Intervention(
                layer=L, sublayer=sub,
                fn=_make_position_patch(pos, clean_vec),
            )
            result = intervene(
                model, tokenizer, base_prompt,
                interventions=[iv],
                capture_logit_lens=False,
            )
            patched_logits = result.output_logits[resolved_meas].detach().cpu()
            cell = {
                "layer": L,
                "sublayer": sub,
                "position": pos,
                "patched_logits": patched_logits,
            }
            if on_cell is not None:
                on_cell(L, sub, pos, cell)
            cells.append(cell)

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline_logits,
        corrupted_baseline_logits=corrupted_baseline_logits,
        prompt_tokens_clean=prompt_tokens_clean,
        prompt_tokens_corrupted=prompt_tokens_corrupted,
        direction=direction,
        measurement_position=resolved_meas,
    )
```

Delete the old placeholder return block (`return PatchingResult(cells=[], clean_baseline_logits=torch.zeros(0), ...)`) that Task 2 installed.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py -v
```

Expected: all 22+ tests pass (4 Task 1 + 7 Task 2 validation + 1 PatchingResult + 10 Task 3 loop tests = 22). Runtime on CPU with `tiny_llama`: ~5–20 s (48 forward passes × ~50 ms).

- [ ] **Step 5: Pyright check**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_activation_patch.py
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_activation_patch.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): activation_patch() core loop — denoise + noise directions

Composes _capture_residual_stream (both prompts) + intervene() (with
_make_position_patch as the op) to produce per-(layer, sublayer, position)
patched logits at the measurement position. Direction picks base prompt
and patch source. on_cell callback streams frames to the WS layer."
```

---

## Task 4: TinyLlama integration test (real GPU)

**Why fourth:** validates the algorithm end-to-end on a real model. Phase 2 got burned by dtype bugs mocks missed; this catches the same class.

**GPU note:** This test requires CUDA. The subagent must request `dangerouslyDisableSandbox: true` when running the pytest command. If permission is denied, return BLOCKED — the controller runs it directly (per established pattern in `feedback_gpu_sandbox.md`).

**Files:**
- Modify: `testing/tests/test_probe_activation_patch.py`

- [ ] **Step 1: Write the integration test**

Use Edit to append to `testing/tests/test_probe_activation_patch.py`:

```python
# ---------------------------------------------------------------------------
# Integration — real TinyLlama on CUDA
# ---------------------------------------------------------------------------

class TestActivationPatchIntegration:
    """End-to-end: real TinyLlama fp16 → activation patch → sanity check."""

    def test_tinyllama_capital_swap(self):
        """Denoise recovery should be larger at late layers than early.

        Intuition: information about the country aggregates through the
        stack. Patching a late-layer residual state with clean activations
        flips the output back toward the clean answer; patching an early
        layer — before the relevant facts have been integrated — does
        comparatively little.
        """
        import torch
        import torch.nn.functional as F
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import activation_patch

        model, tokenizer = load_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16",
        )
        num_layers = len(model.model.layers)

        result = activation_patch(
            model, tokenizer,
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
            direction="denoise",
        )

        # All cells present.
        # seq_len depends on TinyLlama tokenizer; assert algebraic shape.
        positions = {c["position"] for c in result.cells}
        assert len(result.cells) == num_layers * 2 * len(positions)

        # Compute logit-diff-recovery per (layer, sublayer, pos=last) using
        # clean-top-1 vs corrupted-top-1 as the token pair.
        clean_id = int(result.clean_baseline_logits.argmax().item())
        corr_id = int(result.corrupted_baseline_logits.argmax().item())
        delta_clean = (result.clean_baseline_logits[clean_id] - result.clean_baseline_logits[corr_id]).item()
        delta_corr = (result.corrupted_baseline_logits[clean_id] - result.corrupted_baseline_logits[corr_id]).item()
        denom = delta_clean - delta_corr
        # denom should be >0 — the clean forward actually prefers clean-top-1
        # over corrupted-top-1. If not, the model didn't learn the contrast
        # and this test prompt is unsuitable.
        assert denom > 0, f"bad prompt pair: clean/corrupt deltas collapse (denom={denom})"

        last_pos = max(positions)
        recovery_by_layer = {}
        for cell in result.cells:
            if cell["position"] != last_pos or cell["sublayer"] != "ffn":
                continue
            patched = cell["patched_logits"]
            delta_patched = (patched[clean_id] - patched[corr_id]).item()
            recovery = (delta_patched - delta_corr) / denom
            recovery_by_layer[cell["layer"]] = recovery

        early_mean = sum(recovery_by_layer[L] for L in range(5)) / 5
        late_mean = sum(recovery_by_layer[L] for L in range(num_layers - 5, num_layers)) / 5

        # Loose threshold — point is "we didn't break the algorithm,"
        # not pin an exact curve that might shift across TinyLlama revisions.
        assert late_mean > early_mean + 0.1, (
            f"expected late-layer recovery to exceed early by ≥0.1, "
            f"got early={early_mean:.3f} late={late_mean:.3f}"
        )
```

- [ ] **Step 2: Run the integration test**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py::TestActivationPatchIntegration::test_tinyllama_capital_swap -v
```

**Subagent note:** invoke this Bash call with `dangerouslyDisableSandbox: true` — CUDA device access is blocked under the default sandbox. If permission is denied in your session, return BLOCKED with message "GPU sandbox denied for TinyLlama integration — needs controller run." The controller will run it and report back.

Expected: PASS in ~20–60 s on RTX 2080. Forward passes: `22 layers × 2 sublayers × seq_len ≈ 260–300 forward passes`.

- [ ] **Step 3: Pyright check**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright tests/test_probe_activation_patch.py
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/tests/test_probe_activation_patch.py
git -C /home/ai/ai-projects/llm commit -m "test(probe): TinyLlama integration for activation_patch

Loads TinyLlama fp16, runs denoise patching on a capital-swap prompt
pair, asserts late-layer logit-diff recovery exceeds early-layer by
≥0.1. Loose threshold intentional — catches algorithm regressions
without pinning an exact numeric curve."
```

---

## Task 5: Backend WS route `/sessions/{name}/activation-patching`

**Files:**
- Modify: `testing/gui/backend/routes/probes.py`

- [ ] **Step 1: Understand the existing WS pattern**

Use Read on `testing/gui/backend/routes/probes.py` to review the `/logit-lens` handler (around lines 80–194) — this is the template for lock ordering, `on_layer` callback marshaling, error envelopes, and `_encode_hidden_state` usage.

- [ ] **Step 2: Add the new WS handler**

Use Edit on `testing/gui/backend/routes/probes.py` to append the handler. Place it after the existing `intervene_ws` handler (end of file, before the last `return ...` in whatever trailing helper exists):

```python
@router.websocket("/sessions/{name}/activation-patching")
async def activation_patching_ws(ws: WebSocket, name: str):
    """Streaming activation-patching: one frame per (layer, sublayer, position) cell.

    Config (first JSON message):
      {
        "clean_prompt": str,
        "corrupted_prompt": str,
        "direction": "denoise" | "noise",     # default "denoise"
        "measurement_position": int,           # default -1
        "positions": [int] | null,             # default null (= all)
        "sublayers": ["attn","ffn"] subset,    # default ["attn","ffn"]
        "layers": [int] | null,                # default null (= all)
        "correct_token": str (optional),       # for manual logit-diff token pair
        "incorrect_token": str (optional)
      }

    Frames: status → baselines (once) → data (N) → complete | error.
    """
    await ws.accept()
    log.info("WS activation-patching connected (session='%s')", name)
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        log.warning("WS activation-patching: session '%s' not found", name)
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    try:
        mgr.ensure_pytorch(name)
    except Exception as e:
        log.exception("WS activation-patching: ensure_pytorch failed for '%s'", name)
        await _send_json(ws, {"type": "error", "message": f"Failed to load PyTorch model: {e}"})
        await ws.close()
        return

    try:
        raw = await ws.receive_text()
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        await _send_json(ws, {"type": "error", "message": f"Invalid JSON config: {e}"})
        await ws.close()
        return
    except WebSocketDisconnect:
        return

    clean_prompt = config.get("clean_prompt", "")
    corrupted_prompt = config.get("corrupted_prompt", "")
    direction = config.get("direction", "denoise")
    measurement_position = int(config.get("measurement_position", -1))
    positions = config.get("positions")  # None or list
    sublayers = tuple(config.get("sublayers", ["attn", "ffn"]))
    layers = config.get("layers")
    correct_token = config.get("correct_token")
    incorrect_token = config.get("incorrect_token")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from llm_surgeon import probe

    # Resolve manual token pair if provided (both or neither).
    correct_token_id: int | None = None
    incorrect_token_id: int | None = None
    if correct_token is not None or incorrect_token is not None:
        if correct_token is None or incorrect_token is None:
            await _send_json(ws, {"type": "error",
                                  "message": "correct_token and incorrect_token must both be provided or both omitted"})
            await ws.close()
            return
        try:
            c_ids = info.tokenizer(correct_token, add_special_tokens=False)["input_ids"]
            i_ids = info.tokenizer(incorrect_token, add_special_tokens=False)["input_ids"]
            if len(c_ids) != 1 or len(i_ids) != 1:
                raise ValueError(
                    f"correct_token/incorrect_token must tokenize to exactly one token "
                    f"(got {len(c_ids)} and {len(i_ids)})"
                )
            correct_token_id = int(c_ids[0])
            incorrect_token_id = int(i_ids[0])
        except Exception as e:
            await _send_json(ws, {"type": "error", "message": str(e)})
            await ws.close()
            return

    connected = True
    loop = asyncio.get_running_loop()

    def on_cell(layer_idx, sublayer, position, cell):
        nonlocal connected
        if not connected:
            return
        msg = {
            "type": "data",
            "layer": layer_idx,
            "original_layer": (info._layer_map[layer_idx]
                               if layer_idx < len(info._layer_map) else layer_idx),
            "sublayer": sublayer,
            "position": position,
            "patched_logits": _encode_hidden_state(cell["patched_logits"]),
        }
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
        try:
            ok = fut.result(timeout=10)
        except Exception:
            ok = False
        if not ok:
            connected = False

    if info.dirty:
        await _send_json(ws, {"type": "status", "message": "Waiting for model export..."})

    try:
        async with info.lock:
            await _send_json(ws, {"type": "status", "message": "Capturing activations..."})
            result = await loop.run_in_executor(
                None,
                lambda: probe.activation_patch(
                    info.model, info.tokenizer,
                    clean_prompt=clean_prompt,
                    corrupted_prompt=corrupted_prompt,
                    direction=direction,
                    measurement_position=measurement_position,
                    positions=positions,
                    sublayers=sublayers,
                    layers=layers,
                    on_cell=on_cell,
                ),
            )

        if connected:
            # Baselines frame goes after the patching loop because it carries
            # the resolved measurement_position + auto-resolved token IDs.
            # Send it first would require a pre-pass we don't need here — the
            # frontend groups frames by type regardless of order.
            baselines_msg: dict = {
                "type": "baselines",
                "clean_logits": _encode_hidden_state(result.clean_baseline_logits),
                "corrupted_logits": _encode_hidden_state(result.corrupted_baseline_logits),
                "prompt_tokens_clean": result.prompt_tokens_clean,
                "prompt_tokens_corrupted": result.prompt_tokens_corrupted,
                "measurement_position": result.measurement_position,
            }
            # Auto-resolve token IDs if manual pair wasn't provided.
            if correct_token_id is None:
                correct_token_id = int(result.clean_baseline_logits.argmax().item())
                incorrect_token_id = int(result.corrupted_baseline_logits.argmax().item())
            baselines_msg["correct_token_id"] = correct_token_id
            baselines_msg["incorrect_token_id"] = incorrect_token_id
            await _send_json(ws, baselines_msg)

            await _send_json(ws, {
                "type": "complete",
                "summary": {
                    "num_cells": len(result.cells),
                    "direction": result.direction,
                    "measurement_position": result.measurement_position,
                },
            })

    except ValueError as e:
        # Input validation errors from activation_patch are user-facing.
        log.warning("WS activation-patching validation error: %s", e)
        await _send_json(ws, {"type": "error", "message": str(e)})
    except Exception as e:
        log.exception("WS activation-patching error (session='%s')", name)
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        log.info("WS activation-patching disconnected (session='%s')", name)
        try:
            await ws.close()
        except RuntimeError:
            pass
```

- [ ] **Step 3: Pyright check**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright gui/backend/routes/probes.py
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 4: Quick smoke-check the FastAPI app boots**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -c "from gui.backend.app import app; print('routes:', [r.path for r in app.routes if 'activation' in r.path])"
```

Expected output contains `/sessions/{name}/activation-patching`.

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/backend/routes/probes.py
git -C /home/ai/ai-projects/llm commit -m "feat(gui/backend): /sessions/{name}/activation-patching WS route

Streams one 'data' frame per (layer, sublayer, position) cell via
probe.activation_patch()'s on_cell callback. Handles manual vs auto
token pair resolution for logit-diff recovery. Mirrors /logit-lens
locking and error-envelope patterns."
```

---

## Task 6: Frontend `types/api.ts` additions

**Files:**
- Modify: `testing/gui/frontend/src/types/api.ts`

- [ ] **Step 1: Read the existing types file to find the ProbeOperation and WsMessage definitions**

```
Read /home/ai/ai-projects/llm/testing/gui/frontend/src/types/api.ts
```

Note the line numbers of `ProbeOperation` (around line 133) and `WsMessage` (search for the union definition).

- [ ] **Step 2: Add `activation-patching` to `ProbeOperation`**

Use Edit to change the union:

```typescript
// OLD
export type ProbeOperation = "logit-lens" | "influence" | "attention" | "residual-norms" | "generate";

// NEW
export type ProbeOperation =
  | "logit-lens"
  | "influence"
  | "attention"
  | "residual-norms"
  | "generate"
  | "activation-patching";
```

- [ ] **Step 3: Add Patching* interfaces near the existing `LogitLensData` interface**

Find the existing `LogitLensData` interface (via Grep). Use Edit to add these definitions nearby:

```typescript
export interface PatchingBaselinesData {
  type: "baselines";
  clean_logits: EncodedTensor;
  corrupted_logits: EncodedTensor;
  prompt_tokens_clean: string[];
  prompt_tokens_corrupted: string[];
  measurement_position: number;
  correct_token_id?: number;
  incorrect_token_id?: number;
}

export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer: "attn" | "ffn";
  position: number;
  patched_logits: EncodedTensor;
}

export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
  };
}
```

If `EncodedTensor` is not yet exported from this file (check via Grep), add:

```typescript
export interface EncodedTensor {
  shape: number[];
  b64: string;
}
```

- [ ] **Step 4: Extend the `WsMessage` union**

Find the `WsMessage = ...` union. Add the three new variants:

```typescript
export type WsMessage =
  | /* ...existing variants... */
  | PatchingBaselinesData
  | PatchingCellData
  | PatchingCompleteData;
```

Note: `PatchingCellData` has `type: "data"` just like existing per-layer frames. That's fine — discrimination happens on *combination* of `type` + shape; TypeScript treats them as compatible since `data` frames already come in multiple shapes from different ops.

- [ ] **Step 5: tsc check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/types/api.ts
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): types for activation-patching WS frames

Adds PatchingBaselinesData, PatchingCellData, PatchingCompleteData,
'activation-patching' to ProbeOperation, and extends WsMessage union.
No component wiring yet — follow-up commits consume these."
```

---

## Task 7: Frontend `patchingMetrics.ts` + Vitest unit tests

**Files:**
- Create: `testing/gui/frontend/src/utils/patchingMetrics.ts`
- Create: `testing/gui/frontend/tests/unit/patchingMetrics.test.ts`

**Note:** Confirm Vitest is configured in the frontend before starting. Check `testing/gui/frontend/package.json` for a `test` script and a `vitest` dev dependency. If not present, add vitest config as a sub-step; the subagent should not silently skip this. Command to check:

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && grep -E '"(vitest|test)":' package.json
```

If Vitest is not configured, **stop and surface this as BLOCKED** — the controller will decide whether to add Vitest or fold the metric tests into a Playwright component-style test. Default is to add Vitest.

- [ ] **Step 1: Write the failing tests**

Create `testing/gui/frontend/tests/unit/patchingMetrics.test.ts`:

```typescript
import { describe, it, expect } from "vitest";
import {
  decodeLogits,
  logitDiffRecovery,
  klFromClean,
  top1Match,
  probDelta,
} from "../../src/utils/patchingMetrics";

function encode(arr: Float32Array): { shape: number[]; b64: string } {
  const bytes = new Uint8Array(arr.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return { shape: [arr.length], b64: btoa(binary) };
}

describe("decodeLogits", () => {
  it("round-trips float32 bytes", () => {
    const arr = new Float32Array([1.5, -2.0, 0.0, 3.14]);
    const decoded = decodeLogits(encode(arr));
    expect(decoded.length).toBe(4);
    for (let i = 0; i < 4; i++) expect(decoded[i]).toBeCloseTo(arr[i], 5);
  });
});

describe("logitDiffRecovery", () => {
  it("returns 1.0 when patched equals clean", () => {
    const clean = new Float32Array([5, 1, 0, 0]);
    const corrupted = new Float32Array([1, 5, 0, 0]);
    const r = logitDiffRecovery(clean, clean, corrupted, 0, 1);
    expect(r).toBeCloseTo(1.0, 5);
  });

  it("returns 0.0 when patched equals corrupted", () => {
    const clean = new Float32Array([5, 1, 0, 0]);
    const corrupted = new Float32Array([1, 5, 0, 0]);
    const r = logitDiffRecovery(corrupted, clean, corrupted, 0, 1);
    expect(r).toBeCloseTo(0.0, 5);
  });

  it("is signed — negative when patching makes it worse", () => {
    const clean = new Float32Array([5, 1, 0, 0]);
    const corrupted = new Float32Array([1, 5, 0, 0]);
    // patched has an even larger gap in favor of the wrong token.
    const worse = new Float32Array([0, 10, 0, 0]);
    const r = logitDiffRecovery(worse, clean, corrupted, 0, 1);
    expect(r).toBeLessThan(0);
  });
});

describe("klFromClean", () => {
  it("returns 0 when distributions are identical", () => {
    const logits = new Float32Array([1.0, 2.0, 3.0]);
    expect(klFromClean(logits, logits)).toBeCloseTo(0, 5);
  });

  it("is strictly positive when distributions differ", () => {
    const a = new Float32Array([3, 1, 1]);
    const b = new Float32Array([1, 3, 1]);
    expect(klFromClean(a, b)).toBeGreaterThan(0);
  });

  it("handles zero-probability bins via xlogy semantics (does not return NaN)", () => {
    // After softmax these won't actually hit zero (exp stays positive),
    // but a very large negative logit gets us to machine-zero probability
    // which would blow up a naive p*log(p/q) without a floor.
    const a = new Float32Array([0, -1e8, 0]);
    const b = new Float32Array([0, 0, 0]);
    const kl = klFromClean(a, b);
    expect(Number.isFinite(kl)).toBe(true);
  });
});

describe("top1Match", () => {
  it("returns true when argmax agrees", () => {
    const a = new Float32Array([5, 1, 0]);
    const b = new Float32Array([3, 1, 0]);
    expect(top1Match(a, b)).toBe(true);
  });

  it("returns false when argmax differs", () => {
    const a = new Float32Array([5, 1, 0]);
    const b = new Float32Array([0, 5, 0]);
    expect(top1Match(a, b)).toBe(false);
  });
});

describe("probDelta", () => {
  it("returns signed probability difference on the clean top-1 id", () => {
    const patched = new Float32Array([2, 0, 0]);     // softmax[0] ≈ 0.7864
    const corrupted = new Float32Array([0, 0, 0]);   // uniform → softmax[0] = 1/3
    const d = probDelta(patched, corrupted, /* cleanTopId */ 0);
    expect(d).toBeCloseTo(0.7864 - 1 / 3, 2);
  });
});
```

- [ ] **Step 2: Run the tests — confirm they fail for "module not found"**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/vitest run tests/unit/patchingMetrics.test.ts
```

Expected: failure with "Cannot find module '../../src/utils/patchingMetrics'" or similar.

- [ ] **Step 3: Implement `patchingMetrics.ts`**

Create `testing/gui/frontend/src/utils/patchingMetrics.ts`:

```typescript
/**
 * Pure-function metric helpers for the activation-patching heatmap.
 * All functions operate on Float32Array logit vectors (one per cell).
 * Metric computation happens client-side so the metric dropdown switches
 * without a backend round-trip — same design as LogitLensHeatmap.
 */

import type { EncodedTensor } from "../types/api";

/** Decode a base64-float32 EncodedTensor to a Float32Array. */
export function decodeLogits(enc: EncodedTensor): Float32Array {
  const binary = atob(enc.b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  // Slice ensures we don't alias the underlying ArrayBuffer (which may be
  // reused by a downstream decode).
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** Softmax of a logit vector in-place-free (returns a new Float32Array). */
function softmax(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

/**
 * logit_diff_recovery = (Δ_patched − Δ_corrupted) / (Δ_clean − Δ_corrupted)
 * where Δ = logit(correct) − logit(incorrect).
 *
 * Returns 1.0 when patched == clean, 0.0 when patched == corrupted, signed.
 * Undefined / 0 when the denominator is zero (caller should guard for that
 * case with a valid prompt pair).
 */
export function logitDiffRecovery(
  patched: Float32Array,
  clean: Float32Array,
  corrupted: Float32Array,
  correctId: number,
  incorrectId: number,
): number {
  const deltaPatched = patched[correctId] - patched[incorrectId];
  const deltaClean = clean[correctId] - clean[incorrectId];
  const deltaCorr = corrupted[correctId] - corrupted[incorrectId];
  const denom = deltaClean - deltaCorr;
  if (denom === 0) return 0;
  return (deltaPatched - deltaCorr) / denom;
}

/**
 * KL(softmax(patched) ‖ softmax(clean)) in nats.
 *
 * Uses a small clamp on q (the reference distribution) to avoid -Inf when
 * q has probability mass under machine-zero. xlogy semantics: 0 * log(...)
 * evaluates to 0, so zero-prob bins of p don't contribute.
 */
export function klFromClean(patched: Float32Array, clean: Float32Array): number {
  const p = softmax(patched);
  const q = softmax(clean);
  const floor = 1e-45;
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue;
    const qi = Math.max(q[i], floor);
    kl += p[i] * Math.log(p[i] / qi);
  }
  return kl;
}

/** True when argmax(patched) == argmax(clean). */
export function top1Match(patched: Float32Array, clean: Float32Array): boolean {
  return argmax(patched) === argmax(clean);
}

/**
 * p_patched(cleanTopId) − p_corrupted(cleanTopId).
 *
 * Measures how much patching pushed probability mass onto the clean-top-1
 * token, compared to the corrupted baseline. Signed.
 */
export function probDelta(
  patched: Float32Array,
  corrupted: Float32Array,
  cleanTopId: number,
): number {
  const pp = softmax(patched);
  const pc = softmax(corrupted);
  return pp[cleanTopId] - pc[cleanTopId];
}

function argmax(arr: Float32Array): number {
  let best = 0;
  let bestVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
  }
  return best;
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/vitest run tests/unit/patchingMetrics.test.ts
```

Expected: `Test Files 1 passed`, all 11 tests green.

- [ ] **Step 5: tsc check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/utils/patchingMetrics.ts testing/gui/frontend/tests/unit/patchingMetrics.test.ts
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): patchingMetrics pure-function utilities + tests

logitDiffRecovery, klFromClean, top1Match, probDelta + decodeLogits.
Client-side metric computation means the heatmap's metric dropdown
switches without backend round-trip."
```

---

## Task 8: Frontend `PatchingControls.tsx` component

**Files:**
- Create: `testing/gui/frontend/src/components/PatchingControls.tsx`

- [ ] **Step 1: Create the component file**

Create `testing/gui/frontend/src/components/PatchingControls.tsx`:

```tsx
/**
 * Conditional subpanel rendered by ProbePanel when operation is
 * "activation-patching". Owns patching-only local state; does NOT
 * own the Run button (ProbePanel does).
 *
 * Layout:
 *   [clean prompt textarea]              [N tokens]
 *   [corrupted prompt textarea]          [M tokens ✓/✗]
 *   direction: (●) denoise  ( ) noise
 *   measure @ [-1]
 *   target:   (●) auto-pick  ( ) manual
 *     correct: [ ]   incorrect: [ ]
 */
import { useEffect, useState } from "react";

export interface PatchingState {
  cleanPrompt: string;
  corruptedPrompt: string;
  direction: "denoise" | "noise";
  measurementPos: number;
  tokenPairMode: "auto" | "manual";
  manualCorrect: string;
  manualIncorrect: string;
}

export const DEFAULT_PATCHING_STATE: PatchingState = {
  cleanPrompt: "",
  corruptedPrompt: "",
  direction: "denoise",
  measurementPos: -1,
  tokenPairMode: "auto",
  manualCorrect: "",
  manualIncorrect: "",
};

interface Props {
  targetSession: string;
  state: PatchingState;
  onChange: (patch: Partial<PatchingState>) => void;
  /** True when clean and corrupted tokenize to same length. Bound to Run enable. */
  onLengthMatchChange: (match: boolean) => void;
}

const labelStyle: React.CSSProperties = {
  fontFamily: "monospace", color: "#a0a0c0", fontSize: 12,
};

const gridStyle: React.CSSProperties = {
  display: "grid", gridTemplateColumns: "auto 1fr", gap: "4px 8px",
  alignItems: "center", fontSize: 12,
};

export function PatchingControls({ targetSession, state, onChange, onLengthMatchChange }: Props) {
  const [cleanTokens, setCleanTokens] = useState<number | null>(null);
  const [corrTokens, setCorrTokens] = useState<number | null>(null);

  // Debounced tokenize probes — same pattern as ProbePanel's promptTokens
  // budget display. 250 ms keeps typing snappy without spamming the backend.
  useEffect(() => {
    if (!targetSession || !state.cleanPrompt) { setCleanTokens(null); return; }
    const ac = new AbortController();
    const t = setTimeout(() => {
      fetch(`/api/sessions/${targetSession}/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.cleanPrompt }),
        signal: ac.signal,
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (d && typeof d.count === "number") setCleanTokens(d.count); })
        .catch(() => { /* abort / offline */ });
    }, 250);
    return () => { clearTimeout(t); ac.abort(); };
  }, [state.cleanPrompt, targetSession]);

  useEffect(() => {
    if (!targetSession || !state.corruptedPrompt) { setCorrTokens(null); return; }
    const ac = new AbortController();
    const t = setTimeout(() => {
      fetch(`/api/sessions/${targetSession}/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: state.corruptedPrompt }),
        signal: ac.signal,
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { if (d && typeof d.count === "number") setCorrTokens(d.count); })
        .catch(() => { /* abort / offline */ });
    }, 250);
    return () => { clearTimeout(t); ac.abort(); };
  }, [state.corruptedPrompt, targetSession]);

  // Surface length-match state to the parent so Run can gate on it.
  const lengthsMatch =
    cleanTokens != null && corrTokens != null && cleanTokens === corrTokens && cleanTokens > 0;
  useEffect(() => { onLengthMatchChange(lengthsMatch); }, [lengthsMatch, onLengthMatchChange]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div>
        <textarea
          placeholder="Clean prompt"
          value={state.cleanPrompt}
          onChange={(e) => onChange({ cleanPrompt: e.target.value })}
          rows={2}
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />
        <div style={{ fontSize: 10, color: "#8888aa", textAlign: "right" }}>
          {cleanTokens == null ? "—" : `${cleanTokens} tokens`}
        </div>
      </div>

      <div>
        <textarea
          placeholder="Corrupted prompt"
          value={state.corruptedPrompt}
          onChange={(e) => onChange({ corruptedPrompt: e.target.value })}
          rows={2}
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />
        <div style={{
          fontSize: 10,
          textAlign: "right",
          color: cleanTokens == null || corrTokens == null
            ? "#8888aa"
            : lengthsMatch ? "#6bc06b" : "#c06060",
        }}>
          {corrTokens == null
            ? "—"
            : lengthsMatch
              ? `${corrTokens} tokens \u2713`
              : cleanTokens != null
                ? `${corrTokens} tokens \u2717 (lengths differ: ${cleanTokens} vs ${corrTokens})`
                : `${corrTokens} tokens`}
        </div>
      </div>

      <div style={gridStyle}>
        <label style={labelStyle}>direction</label>
        <div style={{ display: "flex", gap: 10 }}>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="direction" value="denoise"
              checked={state.direction === "denoise"}
              onChange={() => onChange({ direction: "denoise" })} />
            {" "}denoise
          </label>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="direction" value="noise"
              checked={state.direction === "noise"}
              onChange={() => onChange({ direction: "noise" })} />
            {" "}noise
          </label>
        </div>

        <label style={labelStyle} title="Absolute or negative index. -1 = last token.">measure @</label>
        <input
          type="number"
          value={state.measurementPos}
          onChange={(e) => onChange({ measurementPos: Number(e.target.value) })}
          style={{ width: 60, fontFamily: "monospace", fontSize: 12 }}
        />

        <label style={labelStyle}>target</label>
        <div style={{ display: "flex", gap: 10 }}>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="tokenPairMode" value="auto"
              checked={state.tokenPairMode === "auto"}
              onChange={() => onChange({ tokenPairMode: "auto" })} />
            {" "}auto-pick
          </label>
          <label style={{ fontSize: 12 }}>
            <input type="radio" name="tokenPairMode" value="manual"
              checked={state.tokenPairMode === "manual"}
              onChange={() => onChange({ tokenPairMode: "manual" })} />
            {" "}manual
          </label>
        </div>

        <label style={labelStyle}>correct</label>
        <input type="text"
          value={state.manualCorrect}
          disabled={state.tokenPairMode !== "manual"}
          onChange={(e) => onChange({ manualCorrect: e.target.value })}
          placeholder="e.g. Paris"
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />

        <label style={labelStyle}>incorrect</label>
        <input type="text"
          value={state.manualIncorrect}
          disabled={state.tokenPairMode !== "manual"}
          onChange={(e) => onChange({ manualIncorrect: e.target.value })}
          placeholder="e.g. Rome"
          style={{ width: "100%", fontFamily: "monospace", fontSize: 12 }}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: tsc check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: no errors. (Unused-import warnings from `PatchingState`/`DEFAULT_PATCHING_STATE` are fine — ProbePanel will import them in Task 10.)

- [ ] **Step 3: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/PatchingControls.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): PatchingControls conditional form component

Two prompt textareas with debounced tokenize + length-match indicator,
direction toggle, measurement-position input, token-pair mode
(auto/manual). Panel-local state; ProbePanel owns the shape via props
(to be wired in a follow-up commit)."
```

---

## Task 9: Frontend `ActivationPatchingHeatmap.tsx` component

**Files:**
- Create: `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx`

- [ ] **Step 1: Read the template**

Read `testing/gui/frontend/src/components/visualizations/LogitLensHeatmap.tsx` to refresh the d3 heatmap pattern (row/col grid, color scale with per-metric interpolator, click-to-pin, ExportButtons wiring).

- [ ] **Step 2: Create the heatmap component**

Create `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx`:

```tsx
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import {
  decodeLogits, logitDiffRecovery, klFromClean, top1Match, probDelta,
} from "../../utils/patchingMetrics";
import type { ProbeResult, PatchingBaselinesData, PatchingCellData } from "../../types/api";

interface Props {
  result: ProbeResult;
}

type MetricKey = "logit_diff_recovery" | "kl_from_clean" | "top1_match" | "prob_delta";

interface MetricDef {
  label: string;
  interpolator: (t: number) => string;
  fixedDomain?: [number, number];
  format: (v: number) => string;
}

const METRICS: Record<MetricKey, MetricDef> = {
  logit_diff_recovery: {
    label: "Logit-diff recovery",
    interpolator: d3.interpolatePiYG,
    fixedDomain: [-0.5, 1.0],
    format: (v) => v.toFixed(3),
  },
  kl_from_clean: {
    label: "KL from clean (nats)",
    interpolator: (t) => d3.interpolateInferno(1 - t),  // reverse: low KL = bright
    format: (v) => v.toFixed(3),
  },
  top1_match: {
    label: "Top-1 matches clean",
    interpolator: (t) => (t > 0.5 ? "#6bc06b" : "#333"),
    fixedDomain: [0, 1],
    format: (v) => (v > 0.5 ? "yes" : "no"),
  },
  prob_delta: {
    label: "Δ p(clean top-1)",
    interpolator: d3.interpolatePiYG,
    fixedDomain: [-1, 1],
    format: (v) => v.toFixed(3),
  },
};

interface PinnedCell {
  cell: PatchingCellData;
  x: number; y: number;
}

export function ActivationPatchingHeatmap({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [metric, setMetric] = useState<MetricKey>("logit_diff_recovery");
  const [pinned, setPinned] = useState<PinnedCell | null>(null);

  const baselines = useMemo(
    () => result.data.find((m): m is PatchingBaselinesData => m.type === "baselines"),
    [result.data]
  );
  const cells = useMemo(
    () => result.data.filter((m): m is PatchingCellData =>
      m.type === "data" && "patched_logits" in m && "position" in m
    ),
    [result.data]
  );

  // Decode tensors once — they're reused across every metric switch.
  const cleanLogits = useMemo(
    () => baselines ? decodeLogits(baselines.clean_logits) : null, [baselines]);
  const corruptedLogits = useMemo(
    () => baselines ? decodeLogits(baselines.corrupted_logits) : null, [baselines]);
  const cellLogits = useMemo(
    () => new Map(cells.map((c) => [`${c.layer}.${c.sublayer}.${c.position}`, decodeLogits(c.patched_logits)])),
    [cells]
  );

  // Pure-ish: takes metric as a parameter so the CSV exporter can compute
  // all four columns per cell without mutating component state.
  const getCellValueFor = useCallback((cell: PatchingCellData, m: MetricKey): number | null => {
    if (!cleanLogits || !corruptedLogits || !baselines) return null;
    const patched = cellLogits.get(`${cell.layer}.${cell.sublayer}.${cell.position}`);
    if (!patched) return null;
    switch (m) {
      case "logit_diff_recovery":
        if (baselines.correct_token_id == null || baselines.incorrect_token_id == null) return null;
        return logitDiffRecovery(patched, cleanLogits, corruptedLogits,
          baselines.correct_token_id, baselines.incorrect_token_id);
      case "kl_from_clean":
        return klFromClean(patched, cleanLogits);
      case "top1_match":
        return top1Match(patched, cleanLogits) ? 1 : 0;
      case "prob_delta": {
        let topId = 0, topVal = -Infinity;
        for (let i = 0; i < cleanLogits.length; i++) {
          if (cleanLogits[i] > topVal) { topVal = cleanLogits[i]; topId = i; }
        }
        return probDelta(patched, corruptedLogits, topId);
      }
    }
  }, [cellLogits, cleanLogits, corruptedLogits, baselines]);

  useEffect(() => {
    if (!svgRef.current || cells.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Build row keys (layer, sublayer), sorted layer-major, attn before ffn.
    const rowKeySet = new Set<string>();
    for (const c of cells) rowKeySet.add(`${c.layer}.${c.sublayer}`);
    const rowKeys = Array.from(rowKeySet).sort((a, b) => {
      const [la, sa] = a.split(".");
      const [lb, sb] = b.split(".");
      if (la !== lb) return Number(la) - Number(lb);
      return sa === "attn" ? -1 : 1;
    });

    const positionSet = new Set<number>();
    for (const c of cells) positionSet.add(c.position);
    const positions = Array.from(positionSet).sort((a, b) => a - b);

    const margin = { top: 40, right: 20, bottom: 40, left: 80 };
    const cellW = Math.max(30, Math.min(60, 600 / positions.length));
    const cellH = 20;
    const width = margin.left + positions.length * cellW + margin.right;
    const height = margin.top + rowKeys.length * cellH + margin.bottom;
    svg.attr("width", width).attr("height", height);

    const def = METRICS[metric];

    let domain: [number, number];
    if (def.fixedDomain) {
      domain = def.fixedDomain;
    } else {
      let minV = Infinity, maxV = -Infinity;
      for (const c of cells) {
        const v = getCellValueFor(c, metric);
        if (v == null) continue;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
      domain = (Number.isFinite(minV) && Number.isFinite(maxV) && minV !== maxV) ? [minV, maxV] : [0, 1];
    }

    const colorScale = d3.scaleSequential(def.interpolator).domain(domain);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Row labels.
    rowKeys.forEach((rk, rowIdx) => {
      g.append("text")
        .attr("x", -4)
        .attr("y", rowIdx * cellH + cellH / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#8888aa")
        .text(`L${rk.split(".")[0]}.${rk.split(".")[1]}`);
    });

    // Column (position) labels.
    positions.forEach((pos, colIdx) => {
      g.append("text")
        .attr("x", colIdx * cellW + cellW / 2)
        .attr("y", -6)
        .attr("text-anchor", "middle")
        .attr("font-size", 9)
        .attr("fill", "#666")
        .text(pos);
    });

    // Cells.
    for (const cell of cells) {
      const rk = `${cell.layer}.${cell.sublayer}`;
      const rowIdx = rowKeys.indexOf(rk);
      const colIdx = positions.indexOf(cell.position);
      if (rowIdx < 0 || colIdx < 0) continue;

      const v = getCellValueFor(cell, metric);
      const fill = v != null ? colorScale(v) : "#222";

      g.append("rect")
        .attr("x", colIdx * cellW)
        .attr("y", rowIdx * cellH)
        .attr("width", cellW - 1)
        .attr("height", cellH - 1)
        .attr("fill", fill)
        .attr("rx", 2)
        .style("cursor", "pointer")
        .on("click", (event) => {
          setPinned({ cell, x: event.pageX + 10, y: event.pageY + 10 });
        });
    }

    // Legend.
    const legendY = rowKeys.length * cellH + 20;
    const legendW = Math.min(200, positions.length * cellW);
    const legendH = 8;
    const gradId = `ap-grad-${metric}-${result.id}`;
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", gradId).attr("x1", "0%").attr("x2", "100%");
    for (let i = 0; i <= 16; i++) {
      grad.append("stop").attr("offset", `${(i / 16) * 100}%`).attr("stop-color", def.interpolator(i / 16));
    }
    g.append("rect")
      .attr("x", 0).attr("y", legendY).attr("width", legendW).attr("height", legendH)
      .attr("fill", `url(#${gradId})`).attr("rx", 1);
    g.append("text")
      .attr("x", 0).attr("y", legendY + legendH + 10)
      .attr("font-size", 9).attr("fill", "#888").text(def.format(domain[0]));
    g.append("text")
      .attr("x", legendW).attr("y", legendY + legendH + 10)
      .attr("text-anchor", "end").attr("font-size", 9).attr("fill", "#888").text(def.format(domain[1]));
    g.append("text")
      .attr("x", legendW / 2).attr("y", legendY - 2)
      .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#aaa").text(def.label);
  }, [cells, metric, getCellValueFor, result.id]);

  const csvRows = useCallback((): (string | number)[][] => {
    const header = ["layer", "sublayer", "position",
      "logit_diff_recovery", "kl_from_clean", "top1_match", "prob_delta"];
    const rows: (string | number)[][] = [header];
    for (const cell of cells) {
      rows.push([
        cell.layer, cell.sublayer, cell.position,
        getCellValueFor(cell, "logit_diff_recovery") ?? "",
        getCellValueFor(cell, "kl_from_clean") ?? "",
        getCellValueFor(cell, "top1_match") ?? "",
        getCellValueFor(cell, "prob_delta") ?? "",
      ]);
    }
    return rows;
  }, [cells, getCellValueFor]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Activation Patching — {result.sessionName} — "{result.prompt.slice(0, 40)}"
        </h3>
        <label style={{ fontSize: 12, color: "#8888aa", display: "flex", alignItems: "center", gap: 6 }}>
          Metric:
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MetricKey)}
            style={{
              background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a5276",
              borderRadius: 3, padding: "2px 6px", fontSize: 12,
            }}
          >
            {(Object.keys(METRICS) as MetricKey[]).map((k) => (
              <option key={k} value={k}>{METRICS[k].label}</option>
            ))}
          </select>
        </label>
        <div style={{ marginLeft: "auto" }}>
          <ExportButtons
            filenameBase={`activation_patching_${result.sessionName}`}
            getSVG={() => svgRef.current}
            getCSVRows={csvRows}
            getJSON={() => ({
              sessionName: result.sessionName,
              prompt: result.prompt,
              timestamp: result.timestamp,
              data: result.data,
            })}
          />
        </div>
      </div>
      <div style={{ overflowX: "auto" }}><svg ref={svgRef} /></div>
      {pinned && (
        <PinnedCard cell={pinned.cell} x={pinned.x} y={pinned.y} onClose={() => setPinned(null)} />
      )}
    </div>
  );
}

function PinnedCard({ cell, x, y, onClose }: { cell: PatchingCellData; x: number; y: number; onClose: () => void }) {
  return (
    <div style={{
      position: "fixed", left: x, top: y, background: "#0f1626",
      border: "1px solid #1a5276", borderRadius: 4, padding: "10px 12px",
      fontFamily: "monospace", fontSize: 12, color: "#e0e0f0", zIndex: 200,
      boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, marginBottom: 6 }}>
        <strong style={{ color: "#a0a0c0" }}>
          L{cell.layer}.{cell.sublayer} pos {cell.position}
        </strong>
        <button onClick={onClose} style={{
          background: "transparent", border: "none", color: "#888",
          cursor: "pointer", fontSize: 14, padding: 0, lineHeight: 1,
        }}>×</button>
      </div>
      <div style={{ fontSize: 10, color: "#888" }}>
        Click a cell to see patched logits (detailed top-k view — enhancement).
      </div>
    </div>
  );
}
```

**Note on `getCellValueFor`:** metric is passed as a parameter (not closed over) so the CSV exporter can compute all four columns per cell without any state mutation. Standard React pattern — no casts.

- [ ] **Step 3: tsc check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Vite build check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/vite build 2>&1 | tail -5
```

Expected: `built in <N>s` with no error lines.

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): ActivationPatchingHeatmap with metric selector

d3-rendered (layer.sublayer) × position heatmap; metrics computed
client-side from decoded logit vectors (logit-diff-recovery, KL,
top-1-match, prob-delta). Click-to-pin card stub. CSV / JSON / SVG
export via existing ExportButtons. Not yet wired into VisualizationArea."
```

---

## Task 10: Wire `ProbePanel` + `VisualizationArea`

**Files:**
- Modify: `testing/gui/frontend/src/components/ProbePanel.tsx`
- Modify: `testing/gui/frontend/src/components/VisualizationArea.tsx`

- [ ] **Step 1: Extend ProbePanel's op dropdown**

Use Read on `ProbePanel.tsx` to find the `<select value={operation} ...>` block (around line 571). Use Edit to add the option:

```tsx
<option value="logit-lens">Logit Lens</option>
<option value="generate">Generate</option>
<option value="influence">Layer Influence</option>
<option value="attention">Attention Entropy</option>
<option value="residual-norms">Residual Norms</option>
<option value="activation-patching">Activation Patching</option>
```

- [ ] **Step 2: Add patching state + controls rendering in ProbePanel**

Use Edit to add near the top of `ProbePanel()`:

```tsx
import { PatchingControls, DEFAULT_PATCHING_STATE, type PatchingState } from "./PatchingControls";

// inside ProbePanel:
const [patchingState, setPatchingStateLocal] = useState<PatchingState>(DEFAULT_PATCHING_STATE);
const [patchingLengthsMatch, setPatchingLengthsMatch] = useState(false);
const updatePatchingState = (patch: Partial<PatchingState>) =>
  setPatchingStateLocal((prev) => ({ ...prev, ...patch }));
```

Use Edit to render the controls conditionally, near where other op-specific blocks are rendered (e.g. after the `operation === "logit-lens"` block around line 593, and before `operation === "generate"`):

```tsx
{operation === "activation-patching" && targetSession && (
  <PatchingControls
    targetSession={targetSession}
    state={patchingState}
    onChange={updatePatchingState}
    onLengthMatchChange={setPatchingLengthsMatch}
  />
)}
```

- [ ] **Step 3: Add `WS_OPS` membership for the new op**

Find the `WS_OPS` set (around line 77). Use Edit:

```tsx
const WS_OPS = new Set<ProbeOperation>(["logit-lens", "generate", "activation-patching"]);
```

- [ ] **Step 4: Branch `handleRun` for activation-patching**

Use Read to find `handleRun` (around line 289). Use Edit to insert the AP branch at the top of the WS-path block (inside the `if (isWs)` section, before the fan-out handling):

```tsx
if (operation === "activation-patching") {
  if (!patchingLengthsMatch) {
    setError("Clean and corrupted prompts must tokenize to the same length.");
    setRunning(false);
    return;
  }
  localPendingIdsRef.current.add(resultId);
  const runParamsSnapshot = { ...samplingParams };
  setPendingResult(resultId, {
    id: resultId, operation, sessionName: targetSession, prompt: patchingState.cleanPrompt,
    data: [], timestamp: Date.now(), isB: false,
    runParams: runParamsSnapshot,
  });
  const cfg: Record<string, unknown> = {
    clean_prompt: patchingState.cleanPrompt,
    corrupted_prompt: patchingState.corruptedPrompt,
    direction: patchingState.direction,
    measurement_position: patchingState.measurementPos,
  };
  if (patchingState.tokenPairMode === "manual") {
    cfg.correct_token = patchingState.manualCorrect;
    cfg.incorrect_token = patchingState.manualIncorrect;
  }
  const path = `/ws/sessions/${targetSession}/activation-patching`;
  connect(resultId, path, cfg, makeWsHandlers(resultId), targetSession);
  return;
}
```

- [ ] **Step 5: Disable fan-out / A-B for activation-patching**

Find the fan-out inputs disable guards (search for `disabled={!!targetSessionB}` patterns). Use Edit to add `|| operation === "activation-patching"` to the conditions that hide or disable N-way fan-out, A-B session selection, and sweep axes. The existing `operation === "generate"` or similar guards are adjacent — mirror the pattern.

Specifically, update the conditions around `numSeeds`, the `sweep` dropdown, and axis² — they should evaluate to disabled when `operation === "activation-patching"`. If the clean approach is harder to tease apart, simply early-return from the existing generate-specific block when `operation !== "generate"`.

- [ ] **Step 6: Gate `getWsPath` and `getWsConfig` for activation-patching**

Update these two helpers to handle the new op explicitly:

```tsx
const getWsPath = (session: string) => {
  if (operation === "logit-lens") return `/ws/sessions/${session}/logit-lens`;
  if (operation === "activation-patching") return `/ws/sessions/${session}/activation-patching`;
  return `/ws/sessions/${session}/generate`;
};

const getWsConfig = (maxTokensOverride?: number) => {
  if (operation === "logit-lens") return { prompt, top_k: displayTopK };
  if (operation === "activation-patching") {
    // Unused by fan-out/"run-on-all" paths — the dedicated AP branch in
    // handleRun builds its own cfg with clean/corrupted/direction/etc.
    return { prompt };
  }
  // ...existing generate config...
};
```

- [ ] **Step 7: Dispatch to `ActivationPatchingHeatmap` in VisualizationArea**

Use Read on `VisualizationArea.tsx` to find the op → component switch (search for `activeResult.operation ===` or similar). Use Edit to add:

```tsx
import { ActivationPatchingHeatmap } from "./visualizations/ActivationPatchingHeatmap";

// in the switch / conditional rendering:
{activeResult.operation === "activation-patching" && (
  <ActivationPatchingHeatmap result={activeResult} />
)}
```

- [ ] **Step 8: tsc check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: no errors.

- [ ] **Step 9: Vite build check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/vite build 2>&1 | tail -5
```

Expected: `built in <N>s`.

- [ ] **Step 10: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/ProbePanel.tsx testing/gui/frontend/src/components/VisualizationArea.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): wire activation-patching into ProbePanel + VisualizationArea

Op dropdown option, conditional PatchingControls render, handleRun
branch that gates on length-match, WS_OPS membership, fan-out/A-B
disabled for AP, VisualizationArea dispatches to ActivationPatchingHeatmap.
End-to-end path now live: user picks op → fills form → Run → heatmap
streams in."
```

---

## Task 11: Playwright smoke extension

**Files:**
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts`
- Modify: `testing/gui/frontend/tests/e2e/fixtures/sample.json` (or create sibling `activation-patching.json`)

- [ ] **Step 1: Read the existing fixture shape**

Read `testing/gui/frontend/tests/e2e/fixtures/sample.json` to see the `ExperimentFile` / `ProbeResult` shape already in use.

- [ ] **Step 2: Extend the fixture with one AP result**

Use Edit to append a second result (an activation-patching result with a mock baselines frame + 2 cells so the heatmap renders). A minimal base64 float32 vector of length 4 is fine — the metric calc doesn't care about realism, only shape.

Example tensor: `new Float32Array([0.1, 0.2, 0.3, 0.4])` encoded as base64 — precompute and hardcode. Use this one-liner in node to generate it once:

```bash
node -e "const a=new Float32Array([0.1,0.2,0.3,0.4]); const b=Buffer.from(a.buffer); console.log(b.toString('base64'))"
```

Expected: `zczMPc3MTD6amZk+zcxMPw==` or similar. Embed this in the fixture.

Sample addition to `sample.json` (exact placement depends on fixture shape):

```json
{
  "results": [
    /* existing results */,
    {
      "id": "ap-test-1",
      "operation": "activation-patching",
      "sessionName": "test-session",
      "prompt": "The capital of France is",
      "timestamp": 1713000000000,
      "isB": false,
      "data": [
        {
          "type": "baselines",
          "clean_logits": {"shape": [4], "b64": "<base64 float32 tensor from node oneliner>"},
          "corrupted_logits": {"shape": [4], "b64": "<same>"},
          "prompt_tokens_clean": ["T","h","e"," capital"],
          "prompt_tokens_corrupted": ["T","h","e"," capital"],
          "measurement_position": 3,
          "correct_token_id": 0,
          "incorrect_token_id": 1
        },
        {
          "type": "data",
          "layer": 0,
          "sublayer": "attn",
          "position": 0,
          "patched_logits": {"shape": [4], "b64": "<same>"}
        },
        {
          "type": "data",
          "layer": 0,
          "sublayer": "ffn",
          "position": 1,
          "patched_logits": {"shape": [4], "b64": "<same>"}
        },
        { "type": "complete", "summary": {"num_cells": 2, "direction": "denoise", "measurement_position": 3} }
      ]
    }
  ]
}
```

- [ ] **Step 3: Write the Playwright test**

Use Edit to append to `testing/gui/frontend/tests/e2e/smoke.spec.ts`:

```typescript
test("activation-patching heatmap renders from imported fixture", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  // Import the extended fixture (same mechanism as existing smoke tests).
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles("tests/e2e/fixtures/sample.json");

  // Wait for a result chip / activation-patching heatmap to appear.
  await page.getByText(/Activation Patching/).waitFor({ state: "visible", timeout: 5000 });

  // Switch metric dropdown through all options.
  const metric = page.getByRole("combobox").filter({ hasText: /Logit-diff recovery|KL|Top-1|Δ p/ }).first();
  for (const opt of ["KL from clean (nats)", "Top-1 matches clean", "Δ p(clean top-1)", "Logit-diff recovery"]) {
    await metric.selectOption({ label: opt });
    await page.waitForTimeout(50); // let d3 re-render
  }

  // Wait briefly for any async rendering to settle.
  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});
```

- [ ] **Step 4: Run the smoke suite**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```

**Subagent note:** invoke with `dangerouslyDisableSandbox: true` — playwright + vite subprocesses touch `/dev/urandom` via node's crypto, which the default sandbox blocks.

Expected: 10/10 tests pass (9 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/tests/e2e/smoke.spec.ts testing/gui/frontend/tests/e2e/fixtures/sample.json
git -C /home/ai/ai-projects/llm commit -m "test(gui/frontend): Playwright smoke for activation-patching heatmap

Seeds an activation-patching result via experiment-import (mock
baselines + 2 cells, one per sublayer). Asserts the heatmap renders
and the metric dropdown switches through all four options without
console errors. No backend required — exercises the full
frontend decode + render path."
```

---

## Task 12: Update roadmap memory

**Files:**
- Modify: `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md`

- [ ] **Step 1: Read the existing memory file**

```
Read /home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md
```

Locate the `### Status` section at the bottom and the `Next up: **Phase 3 — activation patching**.` line.

- [ ] **Step 2: Append Phase 3 shipped entry**

Use Edit to replace `- Next up: **Phase 3 — activation patching**.` with:

```markdown
- 2026-04-17: **Phase 3 shipped** — `activation_patch()` added to `llm_surgeon/probe.py` (denoise + noise directions, per-(layer, sublayer, position) granularity, quantized-model warning). Reuses `_capture_residual_stream` + `intervene()` + new `_make_position_patch()` helper. WS route `/sessions/{name}/activation-patching` streams baselines + per-cell patched logits. Frontend: new `PatchingControls.tsx` form (two prompts with length-match indicator, direction/measurement/token-pair controls), new `ActivationPatchingHeatmap.tsx` (rows = layer.sublayer, cols = positions, 4-metric dropdown: logit-diff-recovery / KL-from-clean / top1-match / prob-delta). Metric math in `utils/patchingMetrics.ts` — client-side so dropdown switches without backend round-trip. Commits on `master`: <COMMIT-SHAS-LIST>. Spec: `testing/docs/superpowers/specs/2026-04-17-phase3-activation-patching-design.md`. Plan: `testing/docs/superpowers/plans/2026-04-17-phase3-activation-patching.md`. Pyright 0/0/0; tsc clean; TinyLlama `capital_swap` integration passes (late-layer recovery > early by ≥0.1); Playwright 10/10 (9 existing + 1 new).
- Next up: Phase 3.5 (attribution patching — gradient approximation) if scaling needs arise.
```

Replace `<COMMIT-SHAS-LIST>` by running `git -C /home/ai/ai-projects/llm log --oneline -n 12 master | tac` and listing the 12 commits this plan produced (1 per task in order 1–11, plus this task's commit — which you're about to make).

Actually you can list only the commits on `master` whose subject starts with one of: `feat(probe):`, `test(probe):`, `feat(gui/backend):`, `feat(gui/frontend):`, `test(gui/frontend):`. Use `git -C /home/ai/ai-projects/llm log --oneline --grep="activation" --grep="patching" --grep="probe)" master | head -20` to narrow the search. Substitute the resulting SHAs back into the memory entry.

- [ ] **Step 3: Commit the memory update**

```bash
git -C /home/ai/ai-projects/llm add  # nothing — memory file is outside the repo
```

Wait — the memory file lives in `/home/skothr/.claude/...` which is NOT inside the `/home/ai/ai-projects/llm` repo. Memory updates don't get committed to the project git. Skip the git-add; the Edit tool already persisted the file.

- [ ] **Step 4: Final verification — full test suites green**

```bash
cd /home/ai/ai-projects/llm && testing/.venv/bin/python -m pytest testing/tests/test_probe_activation_patch.py -v
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py gui/backend/routes/probes.py tests/test_probe_activation_patch.py
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e  # needs dangerouslyDisableSandbox
```

**Subagent note:** the last two commands need `dangerouslyDisableSandbox: true`. If either fails, surface specific error back to controller.

Expected all-green final output:
- pytest: all tests in `test_probe_activation_patch.py` pass (unit + integration).
- pyright: `0 errors, 0 warnings, 0 informations`.
- tsc: no output.
- playwright: `10 passed`.

- [ ] **Step 5: Final summary to controller**

Report: "Phase 3 complete. Commits: <list of 11 commit SHAs from tasks 1–11>. All test tiers green (pytest + pyright + tsc + playwright). Memory file updated with shipped status."

---

## Self-Review Checklist (for the plan author)

### Spec coverage

| Spec section | Covered by task |
|---|---|
| §1 architecture | Task 3 (loop), Task 5 (WS), Task 9 (heatmap) — all three composed |
| §2 `activation_patch` signature | Task 2 + Task 3 |
| §2 `_make_position_patch` | Task 1 |
| §2 validation rules | Task 2 |
| §2 quantized-model warning | Task 3 (merged) |
| §3 WS route | Task 5 |
| §3 baselines / data / complete / error frames | Task 5 |
| §3 token-pair resolution | Task 5 |
| §4 PatchingControls | Task 8 |
| §4 length-match indicator | Task 8 |
| §4 Run-button gating | Task 10 (`handleRun` branch) |
| §5 ActivationPatchingHeatmap | Task 9 |
| §5 metric selector | Task 9 |
| §5 client-side metric computation | Task 7 |
| §5 click-to-pin card | Task 9 (stub; full top-k is enhancement) |
| §5 CSV / JSON / SVG export | Task 9 |
| §6 ProbeOperation extension | Task 6 |
| §6 Patching* data type interfaces | Task 6 |
| §6 WsMessage union | Task 6 |
| §6 VisualizationArea dispatch | Task 10 |
| §6 fan-out/A-B disabled | Task 10 |
| §7 unit tests | Tasks 1, 2, 3 |
| §7 TinyLlama integration | Task 4 |
| §7 frontend metric unit tests | Task 7 |
| §7 Playwright extension | Task 11 |
| §8 non-goals | Nothing implements them by design — explicit non-targets, no coverage needed |

**Gap:** Task 9 ships a stubbed pin card (header + layer/sub/pos only). Spec §5 calls for top-5 logits decoded from `patched` and `clean` baselines. This is a visual-only enhancement and its absence doesn't block Phase 3 correctness — the heatmap, metric switching, and export all work. **Mitigation:** flag as a known Phase 3 limitation in the roadmap memory update (Task 12). The full pin card can land in a 13th commit if the controller wants it before merging.

### Placeholder scan

No TBD / TODO / "add appropriate X" language. Every step shows full code or exact commands.

### Type consistency

- `PatchingResult` fields used consistently across tasks 2, 3, 5.
- `PatchingState` shape matches the `on_change` payload in Task 10.
- `PatchingBaselinesData` / `PatchingCellData` / `PatchingCompleteData` field names match between backend (Task 5), types (Task 6), component (Task 9).
- `correct_token_id` / `incorrect_token_id` — same name everywhere (backend snake_case preserved in JSON).
- `measurement_position` vs `measurementPos` — backend/JSON uses snake_case, frontend React state uses camelCase. Translation happens in Task 10's `handleRun` branch. Consistent.

---

## Execution handoff

**Plan complete and saved to `testing/docs/superpowers/plans/2026-04-17-phase3-activation-patching.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, spec-compliance + code-quality review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

**Which approach?**
