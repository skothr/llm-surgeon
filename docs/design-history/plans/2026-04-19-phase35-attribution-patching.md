# Phase 3.5 Attribution Patching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add gradient-based attribution patching (Nanda, 2023; Kramár et al., 2024) as a `mode="approx"` variant of the existing `activation-patching` operation. One forward + one backward pass replaces the `layers × sublayers × positions` inner loop of exact AP.

**Architecture:** One new Python function `attribution_patch()` in `probe.py` + one new internal capture helper that keeps the autograd graph alive. The existing `/ws/sessions/{name}/activation-patching` WS handler branches on `cfg.mode`. Frontend `PatchingControls` gains a mode radio; `ActivationPatchingHeatmap` reads `result.mode` from the `complete` frame and branches rendering (single "AP recovery" metric vs the existing 4-metric dropdown). No new routes, no new viz components — pure extensions.

**Tech Stack:** Python 3.11, PyTorch (autograd), transformers (HF LLaMA), FastAPI WebSockets, React + TypeScript + Zustand, d3, pytest (Python), Playwright (frontend E2E).

**Spec:** `testing/docs/superpowers/specs/2026-04-19-phase35-attribution-patching-design.md` (commit `38b0461`).

**Cwd for tool invocations:** `/home/ai/ai-projects/llm`. The pyright CLI must be run from `testing/` (see CLAUDE.md § Type Checking).

---

## Tool rules (apply to every task + every subagent prompt)

- Use `Read` (not `cat`), `Edit` (not `sed`/`awk`/`cat`), `Grep` (not shell grep), `Glob` (not `find`). All file ops go through dedicated tools.
- For git ops outside the repo root, use `git -C <path>` rather than `cd`.
- Avoid unnecessary compound commands. Avoid chaining that would trigger a permission prompt.
- **GPU tests:** any Bash call that runs pytest touching CUDA must be invoked with `dangerouslyDisableSandbox: true`. If a subagent cannot get that permission granted, surface a BLOCKED status — the controller will run the test directly.
- Pyright CLI: run from `testing/` cwd. Command: `.venv/bin/python -m pyright <paths>`.
- Frontend tsc: run from `testing/gui/frontend/`. Command: `./node_modules/.bin/tsc --noEmit`.
- Playwright: run from `testing/gui/frontend/`. Command: `npm run e2e`.
- Zero-diagnostics discipline: every commit must land with pyright 0/0/0 and tsc clean.
- **Model selection for subagent dispatch:** sonnet or opus only. Never haiku.

---

## File Structure

### New files
- `testing/tests/test_probe_attribution_patch.py` — unit + TinyLlama correlation integration.
- `testing/gui/frontend/tests/e2e/fixtures/activation-patching-approx.json` — mock AP-approx result for Playwright.

### Modified files
- `testing/llm_surgeon/probe.py` — **+** `attribution_patch()`, `_capture_residual_stream_with_grad()`, `mode` field on `PatchingResult`.
- `testing/gui/backend/routes/probes.py` — **~** `/activation-patching` WS handler gains `cfg.mode` branch + `ap_recovery` frame emission.
- `testing/gui/frontend/src/types/api.ts` — **~** `PatchingCellData.ap_recovery?`, `PatchingCompleteData.summary.mode`.
- `testing/gui/frontend/src/components/PatchingControls.tsx` — **~** mode radio + `PatchingState.mode`.
- `testing/gui/frontend/src/components/ProbePanel.tsx` — **~** forward `mode` in cfg payload.
- `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx` — **~** mode-branch rendering.
- `testing/gui/frontend/tests/e2e/smoke.spec.ts` — **+** one approx-mode test.
- `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md` — append "Phase 3.5 shipped" entry.

---

## Task 1: Add `mode` field to `PatchingResult` dataclass

**Why first:** every subsequent task references `PatchingResult.mode`; adding it with default `"exact"` preserves all Phase 3 call sites without a churn commit.

**Files:**
- Modify: `testing/llm_surgeon/probe.py` (the existing `PatchingResult` dataclass near the activation-patching block)
- Modify: `testing/tests/test_probe_activation_patch.py` (add a dataclass-default assertion)

- [ ] **Step 1: Write failing test**

Append to `testing/tests/test_probe_activation_patch.py` (in the existing `TestPatchingResult` class — grep for it):

```python
    def test_mode_defaults_to_exact(self):
        """Default mode is 'exact' so existing Phase 3 call sites don't churn."""
        from llm_surgeon.probe import PatchingResult
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a"],
            prompt_tokens_corrupted=["b"],
            direction="denoise",
            measurement_position=-1,
        )
        assert result.mode == "exact"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_activation_patch.py::TestPatchingResult::test_mode_defaults_to_exact -v
```

Expected: FAIL with `TypeError: PatchingResult.__init__() got unexpected ...` or `AttributeError: ... has no attribute 'mode'`.

- [ ] **Step 3: Add field to dataclass**

Edit the `PatchingResult` dataclass in `testing/llm_surgeon/probe.py` (near line ~716). Add a final field **with default** so call sites that construct it positionally don't break:

```python
@dataclass
class PatchingResult:
    cells: List[Dict]
    clean_baseline_logits: torch.Tensor
    corrupted_baseline_logits: torch.Tensor
    prompt_tokens_clean: List[str]
    prompt_tokens_corrupted: List[str]
    direction: str
    measurement_position: int
    mode: str = "exact"  # "exact" | "approx"
```

- [ ] **Step 4: Run test to verify it passes + existing tests unaffected**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_activation_patch.py -v
```

Expected: all existing `TestPatchingResult` + `TestValidation` + `TestMakePositionPatch` + `TestActivationPatchLoop` tests PASS, plus the new `test_mode_defaults_to_exact` PASS.

- [ ] **Step 5: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_activation_patch.py
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_activation_patch.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): PatchingResult.mode field defaults to 'exact'"
```

---

## Task 2: `_capture_residual_stream_with_grad` helper

**Why:** `_capture_residual_stream` uses `.detach()` which severs autograd. AP needs the captured tensors to keep their computation graph so we can call `.backward()` through the metric and read `.grad` off each captured tensor.

**Files:**
- Modify: `testing/llm_surgeon/probe.py` (add new helper near the existing capture helper)
- Modify: `testing/tests/test_probe_attribution_patch.py` — **new file** — start it with this task.

- [ ] **Step 1: Write failing test**

Create `testing/tests/test_probe_attribution_patch.py`:

```python
"""Tests for probe.attribution_patch — gradient-based AP (Phase 3.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_surgeon.probe import _capture_residual_stream_with_grad


def _tinyllama_cached() -> bool:
    root = Path(__file__).resolve().parents[1] / ".cache" / "models"
    return any(root.glob("models--TinyLlama--*"))


class TestCaptureWithGrad:
    def test_captured_tensors_have_grad_fn(self):
        """Every captured (L, sub) tensor keeps its computation graph."""
        # Minimal mock LLaMA-shaped model sufficient for hook invocation.
        class _MockLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(4, 4)
            def forward(self, x):
                return (x + self.self_attn(x),)  # tuple like HF layers

        class _MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(10, 4)
                self.model.layers = torch.nn.ModuleList([_MockLayer() for _ in range(2)])
                self.lm_head = torch.nn.Linear(4, 10)

            def forward(self, input_ids):
                h = self.model.embed_tokens(input_ids)
                for layer in self.model.layers:
                    h = layer(h)[0]
                return type("Out", (), {"logits": self.lm_head(h)})

        class _MockTok:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids):
                return [str(int(i)) for i in ids]

        model = _MockModel().eval()
        tok = _MockTok()
        captured, logits, tokens = _capture_residual_stream_with_grad(
            model, tok, "hello", sublayers=("attn", "ffn"), layers=None,
        )
        assert len(captured) == 2 * 2  # 2 layers × 2 sublayers
        for key, tensor in captured.items():
            assert tensor.requires_grad, f"{key} must require grad"
            assert tensor.grad_fn is not None, f"{key} must have grad_fn"
        assert logits.requires_grad
        assert len(tokens) == 3

    def test_grad_populates_after_backward(self):
        """Calling .backward() populates .grad on each captured tensor."""
        # (same mock setup as above — reuse the helper pattern)
        # After capture, compute sum = logits.sum(); sum.backward();
        # assert every captured[(L,sub)].grad is not None and non-zero.
        class _MockLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(4, 4)
            def forward(self, x):
                return (x + self.self_attn(x),)

        class _MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(10, 4)
                self.model.layers = torch.nn.ModuleList([_MockLayer() for _ in range(2)])
                self.lm_head = torch.nn.Linear(4, 10)
            def forward(self, input_ids):
                h = self.model.embed_tokens(input_ids)
                for layer in self.model.layers:
                    h = layer(h)[0]
                return type("Out", (), {"logits": self.lm_head(h)})

        class _MockTok:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids):
                return [str(int(i)) for i in ids]

        model = _MockModel().eval()
        tok = _MockTok()
        captured, logits, _ = _capture_residual_stream_with_grad(
            model, tok, "hello", sublayers=("attn", "ffn"), layers=None,
        )
        logits.sum().backward()
        for key, tensor in captured.items():
            assert tensor.grad is not None, f"{key} missing grad after backward"
            # At least one element must be non-zero (not a constant-zero grad).
            assert tensor.grad.abs().sum().item() > 0, f"{key} grad is all zeros"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py::TestCaptureWithGrad -v
```

Expected: FAIL with `ImportError: cannot import name '_capture_residual_stream_with_grad'`.

- [ ] **Step 3: Implement the helper**

Add to `testing/llm_surgeon/probe.py` (immediately after `_capture_residual_stream`):

```python
def _capture_residual_stream_with_grad(
    model,
    tokenizer,
    prompt: str,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], torch.Tensor, List[str]]:
    """Capture residual-stream states with autograd graph intact.

    Mirrors _capture_residual_stream but keeps tensors attached to the graph
    so a downstream .backward() populates .grad on each captured tensor.
    Caller is responsible for providing a torch.enable_grad() context.

    Returns: (captured_states, output_logits, prompt_tokens)
    """
    device = _get_input_device(model)
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    num_layers = len(model.model.layers)
    target_layers = set(range(num_layers)) if layers is None else set(layers)

    captured: Dict[Tuple[int, str], torch.Tensor] = {}
    layer_block_inputs: Dict[int, torch.Tensor] = {}
    hooks: List = []

    for i in range(num_layers):
        if i not in target_layers:
            continue

        def make_pre(idx):
            def hook(_module, args):
                layer_block_inputs[idx] = args[0]
            return hook
        hooks.append(model.model.layers[i].register_forward_pre_hook(make_pre(i)))

        if "attn" in sublayers:
            def make_attn(idx):
                def hook(_module, _inp, out):
                    attn_out = out[0] if isinstance(out, tuple) else out
                    h_post_attn = layer_block_inputs[idx] + attn_out
                    state = h_post_attn[0]
                    state.requires_grad_(True)
                    state.retain_grad()
                    captured[(idx, "attn")] = state
                return hook
            hooks.append(model.model.layers[i].self_attn.register_forward_hook(make_attn(i)))

        if "ffn" in sublayers:
            def make_ffn(idx):
                def hook(_module, _inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    state = hidden[0]
                    state.requires_grad_(True)
                    state.retain_grad()
                    captured[(idx, "ffn")] = state
                return hook
            hooks.append(model.model.layers[i].register_forward_hook(make_ffn(i)))

    try:
        model_output = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return captured, model_output.logits[0], prompt_tokens
```

Note: NO `torch.no_grad()` inside — caller controls grad context. No `.detach()`. `requires_grad_(True)` + `retain_grad()` ensures `.grad` is populated after a downstream backward.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py::TestCaptureWithGrad -v
```

Expected: both tests PASS.

- [ ] **Step 5: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_attribution_patch.py
```

Expected: `0 errors, 0 warnings, 0 informations`. If pyright flags the new helper as unused — it IS called by the test — then leave the suppression off; the test import makes it referenced.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_attribution_patch.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): _capture_residual_stream_with_grad helper for AP"
```

---

## Task 3: `attribution_patch` validation + scaffolding

**Why:** lock in the public contract and error shapes before the core algorithm lands. Mirror the `activation_patch` validation pattern for parity.

**Files:**
- Modify: `testing/llm_surgeon/probe.py` (append `attribution_patch` scaffold after `activation_patch`)
- Modify: `testing/tests/test_probe_attribution_patch.py`

- [ ] **Step 1: Write failing validation tests**

Append to `testing/tests/test_probe_attribution_patch.py`:

```python
from llm_surgeon.probe import attribution_patch


class TestValidation:
    def test_missing_token_ids_raises(self):
        # A minimal mock model — we just need validation to fire before forward.
        with pytest.raises(ValueError, match="correct_token_id and incorrect_token_id"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="b",
                correct_token_id=None, incorrect_token_id=None,  # type: ignore[arg-type]
            )

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="b",
                correct_token_id=1, incorrect_token_id=2,
                direction="wobble",
            )

    def test_bad_sublayer_raises(self):
        with pytest.raises(ValueError, match="sublayers must be"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="b",
                correct_token_id=1, incorrect_token_id=2,
                sublayers=("mlp",),  # type: ignore[arg-type]
            )

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="", corrupted_prompt="b",
                correct_token_id=1, incorrect_token_id=2,
            )
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            attribution_patch(
                model=None, tokenizer=None,
                clean_prompt="a", corrupted_prompt="",
                correct_token_id=1, incorrect_token_id=2,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py::TestValidation -v
```

Expected: FAIL — `ImportError: cannot import name 'attribution_patch'`.

- [ ] **Step 3: Implement the scaffold**

Append to `testing/llm_surgeon/probe.py` (after the existing `activation_patch` function):

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
    """Gradient-based attribution patching (approximate, Phase 3.5).

    One forward + one backward pass produces a per-cell AP score that
    approximates exact activation_patch's logit_diff_recovery. Much cheaper
    than the O(L·S·P) exact loop.

    See: Nanda 2023 (attribution patching primer) and Kramár et al. 2024
    (Attribution Patching Outperforms Automated Circuit Discovery).
    """
    # Validation (raise before any forward pass).
    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(
            "attribution_patch requires correct_token_id and incorrect_token_id"
        )
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not set(sublayers).issubset({"attn", "ffn"}):
        raise ValueError("sublayers must be subset of {'attn', 'ffn'}")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompt cannot be empty")

    # Core algorithm lives in Task 4. Scaffold just raises NotImplementedError.
    raise NotImplementedError("attribution_patch core algorithm is Task 4")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py::TestValidation -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_attribution_patch.py
```

Expected: `0/0/0`.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_attribution_patch.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): attribution_patch signature + validation"
```

---

## Task 4: `attribution_patch` core algorithm

**Why:** the real work. Replace the `NotImplementedError` scaffold with the full forward+backward+per-cell computation.

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe_attribution_patch.py`

- [ ] **Step 1: Write failing test — denoise direction on a mock model**

Append to `testing/tests/test_probe_attribution_patch.py`:

```python
class _MockLlamaBlock(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.self_attn = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        return (x + self.self_attn(x),)


class _MockLlama(torch.nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 4, vocab: int = 10):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, d_model)
        self.model.layers = torch.nn.ModuleList(
            [_MockLlamaBlock(d_model) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(d_model, vocab)
    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return type("Out", (), {"logits": self.lm_head(h)})


class _MockTok:
    def __init__(self, token_ids):
        self._ids = token_ids
    def __call__(self, text, return_tensors=None):
        # Distinct IDs per prompt identity — use hash-based selection so
        # clean and corrupted produce different activations.
        ids = self._ids[0] if text == "clean" else self._ids[1]
        return {"input_ids": torch.tensor([ids])}
    def convert_ids_to_tokens(self, ids):
        return [str(int(i)) for i in ids]


class TestAttributionPatchLoop:
    def test_callback_fires_per_cell_denoise(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        cells = []
        result = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
            measurement_position=-1,
            on_cell=lambda L, sub, pos, c: cells.append((L, sub, pos, c)),
        )
        # 2 layers × 2 sublayers × 3 positions = 12 cells
        assert len(cells) == 12
        assert result.mode == "approx"
        assert result.direction == "denoise"
        # Every cell has ap_recovery and no patched_logits
        for _, _, _, c in cells:
            assert "ap_recovery" in c
            assert "patched_logits" not in c
            assert isinstance(c["ap_recovery"], float)

    def test_noise_direction_flips_base_prompt(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        result = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="noise",
        )
        assert result.direction == "noise"
        # prompt_tokens_clean came from clean_prompt's tokenization
        assert result.prompt_tokens_clean == ["1", "2", "3"]
        assert result.prompt_tokens_corrupted == ["4", "5", "6"]

    def test_positions_subset(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        cells = []
        attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            positions=[0, 2],
            on_cell=lambda L, sub, pos, c: cells.append((L, sub, pos, c)),
        )
        # 2 layers × 2 sublayers × 2 positions = 8 cells
        assert len(cells) == 8
        unique_positions = {c[2] for c in cells}
        assert unique_positions == {0, 2}

    def test_layers_subset(self):
        torch.manual_seed(0)
        model = _MockLlama(num_layers=4, d_model=4, vocab=10).eval()
        tok = _MockTok(token_ids=([1, 2, 3], [4, 5, 6]))
        cells = []
        attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            layers=[1, 2],
            on_cell=lambda L, sub, pos, c: cells.append((L, sub, pos, c)),
        )
        # 2 layers (1, 2) × 2 sublayers × 3 positions = 12 cells
        assert len(cells) == 12
        unique_layers = {c[0] for c in cells}
        assert unique_layers == {1, 2}

    def test_identical_baselines_raises(self):
        """Divide-by-zero guard when clean and corrupted produce same logit_diff."""
        torch.manual_seed(0)
        model = _MockLlama(num_layers=2, d_model=4, vocab=10).eval()
        # Same token ids for both prompts → identical forward → identical logit_diff.
        tok = _MockTok(token_ids=([1, 2, 3], [1, 2, 3]))
        with pytest.raises(ValueError, match="identical logit_diff"):
            attribution_patch(
                model, tok,
                clean_prompt="clean", corrupted_prompt="corrupted",
                correct_token_id=1, incorrect_token_id=2,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py::TestAttributionPatchLoop -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Replace scaffold with full algorithm**

Edit `testing/llm_surgeon/probe.py` — replace the entire `attribution_patch` body (keeping the validation at the top) with:

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
    """Gradient-based attribution patching (Phase 3.5)."""
    import warnings

    # --- Validation (raise before any forward pass) ---
    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(
            "attribution_patch requires correct_token_id and incorrect_token_id"
        )
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not set(sublayers).issubset({"attn", "ffn"}):
        raise ValueError("sublayers must be subset of {'attn', 'ffn'}")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompt cannot be empty")

    if getattr(model, "hf_quantizer", None) is not None:
        warnings.warn(
            "attribution_patch on a quantized model: gradient flow works but "
            "precision is reduced (fp16/int8 through bitsandbytes).",
            stacklevel=2,
        )

    # --- Tokenize + length check ---
    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    if positions is None:
        positions = list(range(seq_len))
    for pos in positions:
        if pos < -seq_len or pos >= seq_len:
            raise IndexError(f"position {pos} out of range for seq_len={seq_len}")
    positions = [p if p >= 0 else seq_len + p for p in positions]
    meas_pos = measurement_position if measurement_position >= 0 else seq_len + measurement_position
    if meas_pos < 0 or meas_pos >= seq_len:
        raise IndexError(
            f"measurement_position {measurement_position} out of range for seq_len={seq_len}"
        )

    # --- Step 1: Forward 'from' prompt in no_grad to cache activations ---
    # denoise: from=clean, base=corrupted
    # noise:   from=corrupted, base=clean
    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt

    with torch.no_grad():
        from_captured, from_logits, from_tokens = _capture_residual_stream_with_grad(
            model, tokenizer, from_prompt, sublayers=sublayers, layers=layers,
        )
        # Detach + clone so the 'from' tensors don't pollute the upcoming
        # base-side graph. They're used purely as values.
        from_states = {k: v.detach().clone() for k, v in from_captured.items()}

    # --- Step 2: Forward 'base' prompt WITH grad to build the graph ---
    with torch.enable_grad():
        base_captured, base_logits, base_tokens = _capture_residual_stream_with_grad(
            model, tokenizer, base_prompt, sublayers=sublayers, layers=layers,
        )

        # Also run both prompts for baseline logits (for d_clean / d_corrupted).
        # base_logits already has the base-prompt logits — reuse them.
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

        # --- Step 3: Metric scalar on base-side logits, backward ---
        metric = (
            base_logits[meas_pos, correct_token_id]
            - base_logits[meas_pos, incorrect_token_id]
        )
        metric.backward()

    # --- Step 4: Compute AP per cell ---
    cells: List[Dict] = []
    sorted_keys = sorted(base_captured.keys(), key=lambda k: (k[0], k[1]))
    for (L, sub) in sorted_keys:
        base_act = base_captured[(L, sub)]
        if base_act.grad is None:
            continue  # shouldn't happen after backward but guard defensively
        base_grad = base_act.grad
        from_act = from_states[(L, sub)]
        for pos in positions:
            if direction == "denoise":
                ap_raw = ((from_act[pos] - base_act[pos].detach()) * base_grad[pos]).sum().item()
                ap_recovery = ap_raw / denominator
            else:  # noise
                ap_raw = ((from_act[pos] - base_act[pos].detach()) * base_grad[pos]).sum().item()
                ap_recovery = 1.0 + ap_raw / denominator
            cell: Dict = {
                "layer": L,
                "sublayer": sub,
                "position": pos,
                "ap_recovery": float(ap_recovery),
            }
            cells.append(cell)
            if on_cell is not None:
                on_cell(L, sub, pos, cell)

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=from_tokens if direction == "denoise" else base_tokens,
        prompt_tokens_corrupted=base_tokens if direction == "denoise" else from_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx",
    )
```

**Key invariants** (worth re-checking):
- `from_states` is detached — it's data, not computation.
- `metric` is computed on `base_logits` which flow through `base_captured` tensors (they `requires_grad_(True)` inside the capture helper).
- `base_act[pos].detach()` in the AP raw formula — we want the *value* of base_act at pos, not a graph reference, because the multiplication with `base_grad[pos]` is a numerical computation.
- For `noise`, the Taylor approximation signs: `ap_raw = (from_act - base_act) · ∂d_base/∂base_act` approximates `Δd_base` when base_act is replaced by from_act. In noise direction, from = corrupted, base = clean → patching corrupted in gives negative Δd_clean → recovery = 1 + Δd_clean/D. The formula in the code computes this correctly.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py -v
```

Expected: all 5 tests in `TestAttributionPatchLoop` PASS plus the earlier `TestCaptureWithGrad` + `TestValidation`.

- [ ] **Step 5: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_attribution_patch.py
```

Expected: `0/0/0`. If pyright flags `positions = [p if p >= 0 else seq_len + p for p in positions]` as a reassignment-with-narrower-type, use `normalized_positions: List[int] = [...]`.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_attribution_patch.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): attribution_patch core algorithm"
```

---

## Task 5: TinyLlama integration — Spearman correlation vs exact AP

**Why:** the load-bearing test of correctness. Confirms the gradient approximation actually tracks exact AP on a real model.

**Files:**
- Modify: `testing/tests/test_probe_attribution_patch.py`

- [ ] **Step 1: Write the failing test**

Append to `testing/tests/test_probe_attribution_patch.py`:

```python
class TestApproxVsExactCorrelates:
    @pytest.mark.skipif(
        not _tinyllama_cached() or not torch.cuda.is_available(),
        reason="requires cached TinyLlama and CUDA",
    )
    def test_tinyllama_capital_swap_spearman(self):
        """AP approx rankings must correlate with exact AP rankings (Spearman ≥ 0.5)."""
        import scipy.stats
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import activation_patch

        model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda")
        model.eval()

        clean = "The capital of France is"
        corrupted = "The capital of Italy is"

        # Resolve target tokens from argmax of clean baseline.
        with torch.no_grad():
            clean_ids = tokenizer(clean, return_tensors="pt")["input_ids"].to(model.device)
            corr_ids = tokenizer(corrupted, return_tensors="pt")["input_ids"].to(model.device)
            clean_last = model(clean_ids).logits[0, -1]
            corr_last = model(corr_ids).logits[0, -1]
        correct_id = int(clean_last.argmax().item())
        incorrect_id = int(corr_last.argmax().item())

        # --- Exact activation patching ---
        exact_result = activation_patch(
            model, tokenizer, clean_prompt=clean, corrupted_prompt=corrupted,
            direction="denoise", measurement_position=-1,
        )
        # Exact produces patched_logits — compute logit_diff_recovery per cell.
        exact_scores: Dict[Tuple[int, str, int], float] = {}
        d_clean = float(
            exact_result.clean_baseline_logits[correct_id]
            - exact_result.clean_baseline_logits[incorrect_id]
        )
        d_corrupted = float(
            exact_result.corrupted_baseline_logits[correct_id]
            - exact_result.corrupted_baseline_logits[incorrect_id]
        )
        D = d_clean - d_corrupted
        for cell in exact_result.cells:
            pl = cell["patched_logits"]
            d_patched = float(pl[correct_id] - pl[incorrect_id])
            key = (cell["layer"], cell["sublayer"], cell["position"])
            exact_scores[key] = (d_patched - d_corrupted) / D

        # --- Approx attribution patching ---
        approx_result = attribution_patch(
            model, tokenizer, clean_prompt=clean, corrupted_prompt=corrupted,
            correct_token_id=correct_id, incorrect_token_id=incorrect_id,
            direction="denoise", measurement_position=-1,
        )
        approx_scores: Dict[Tuple[int, str, int], float] = {
            (c["layer"], c["sublayer"], c["position"]): c["ap_recovery"]
            for c in approx_result.cells
        }

        # --- Spearman rank correlation on shared keys ---
        shared = sorted(set(exact_scores.keys()) & set(approx_scores.keys()))
        assert len(shared) > 20, f"too few cells to correlate: {len(shared)}"
        x = [exact_scores[k] for k in shared]
        y = [approx_scores[k] for k in shared]
        rho, _ = scipy.stats.spearmanr(x, y)
        print(f"\nSpearman(exact, approx) = {rho:.3f} over {len(shared)} cells")
        assert rho >= 0.5, (
            f"AP approx rank correlation {rho:.3f} below threshold 0.5; "
            f"the gradient approximation is not tracking exact patching"
        )
```

- [ ] **Step 2: Run the test (requires GPU + TinyLlama)**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py::TestApproxVsExactCorrelates -v -s
```

This call **must** be run with `dangerouslyDisableSandbox: true`. Expected: test runs in ~60–120 s, prints Spearman correlation, passes with rho ≥ 0.5.

**If the test BLOCKS** (subagent lacks GPU-sandbox permission): surface BLOCKED status; the controller will run it directly.

**If rho < 0.5:** investigate first — this is the primary validity signal. Possible causes (in order of likelihood): sign bug in noise/denoise normalization (Task 4), grad flowing through the wrong tensor identity, baseline confusion (clean vs corrupted logits swapped), measurement at wrong position. Do NOT lower the threshold.

- [ ] **Step 3: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright tests/test_probe_attribution_patch.py
```

Expected: `0/0/0`.

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/tests/test_probe_attribution_patch.py
git -C /home/ai/ai-projects/llm commit -m "test(probe): TinyLlama Spearman correlation exact vs approx AP"
```

---

## Task 6: Extend WS route with `cfg.mode` branch

**Files:**
- Modify: `testing/gui/backend/routes/probes.py`

- [ ] **Step 1: Read the existing `/activation-patching` handler**

Locate it via `Grep` for `activation-patching` in `probes.py`. The handler follows the pattern: `ws.accept()` → read cfg → validate → resolve token pair → call `activation_patch()` with `on_cell` → send baselines → send complete.

- [ ] **Step 2: Add mode branch**

Modify the handler — inside the `try:` block after cfg is parsed and token IDs resolved:

```python
mode = cfg.get("mode", "exact")
if mode not in ("exact", "approx"):
    await ws.send_json({"type": "error", "message": f"mode must be 'exact' or 'approx', got {mode!r}"})
    return

def _emit_cell(L: int, sub: str, pos: int, cell: dict) -> None:
    """Marshal per-cell frame onto the main loop thread."""
    frame: dict = {"type": "data", "layer": L, "sublayer": sub, "position": pos}
    if info._layer_map is not None:
        frame["original_layer"] = info._layer_map.get(L, L)
    if "patched_logits" in cell:
        frame["patched_logits"] = _encode_hidden_state(cell["patched_logits"])
    if "ap_recovery" in cell:
        frame["ap_recovery"] = cell["ap_recovery"]
    asyncio.run_coroutine_threadsafe(ws.send_json(frame), loop)

if mode == "exact":
    result = activation_patch(
        model, tokenizer,
        clean_prompt=clean_prompt, corrupted_prompt=corrupted_prompt,
        direction=direction, measurement_position=measurement_position,
        positions=positions_cfg, sublayers=sublayers, layers=layers,
        on_cell=_emit_cell,
    )
else:  # approx
    result = attribution_patch(
        model, tokenizer,
        clean_prompt=clean_prompt, corrupted_prompt=corrupted_prompt,
        correct_token_id=correct_id, incorrect_token_id=incorrect_id,
        direction=direction, measurement_position=measurement_position,
        positions=positions_cfg, sublayers=sublayers, layers=layers,
        on_cell=_emit_cell,
    )
```

And update the `complete` frame assembly to include `mode`:

```python
await ws.send_json({
    "type": "complete",
    "summary": {
        "num_cells": len(result.cells),
        "direction": result.direction,
        "measurement_position": result.measurement_position,
        "mode": result.mode,  # new
    },
})
```

**Token-pair resolution**: ensure it happens **before** the mode branch, because `attribution_patch` requires IDs. If the existing handler resolves them lazily (e.g., only on auto-pick and inside the `activation_patch` loop), move the resolution up.

- [ ] **Step 3: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright gui/backend/routes/probes.py
```

Expected: `0/0/0`.

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/backend/routes/probes.py
git -C /home/ai/ai-projects/llm commit -m "feat(gui/backend): AP approx mode branch + ap_recovery frames"
```

No test in this step — WS-level E2E testing is not part of this phase (see spec §8). The integration is exercised by the Playwright test in Task 9 using a mocked fixture.

---

## Task 7: Extend frontend types

**Files:**
- Modify: `testing/gui/frontend/src/types/api.ts`

- [ ] **Step 1: Extend `PatchingCellData` + `PatchingCompleteData`**

Find the existing types and extend them:

```typescript
export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer: "attn" | "ffn";
  position: number;
  patched_logits?: EncodedTensor;   // now optional — only present in exact mode
  ap_recovery?: number;             // present only in approx mode
}

export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
    mode?: "exact" | "approx";      // new — defaults to 'exact' when absent (old fixtures)
  };
}
```

- [ ] **Step 2: Type-check**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: clean. If making `patched_logits` optional triggers errors in `ActivationPatchingHeatmap.tsx` where it's currently accessed directly, **that's expected** — Task 9 will address it. You may need to add a narrow `if (m.patched_logits) { ... }` guard as a temporary patch to keep tsc green for this commit, then refactor in Task 9.

Actually, simpler: change the access path in `ActivationPatchingHeatmap.tsx` from `m.patched_logits` to `m.patched_logits!` ONLY in existing exact-path code, since Task 9 will properly branch on mode. Do not add behavior changes in this commit.

- [ ] **Step 3: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/types/api.ts testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): AP frame types for approx mode"
```

---

## Task 8: Add mode radio to `PatchingControls` + forward in cfg

**Files:**
- Modify: `testing/gui/frontend/src/components/PatchingControls.tsx`
- Modify: `testing/gui/frontend/src/components/ProbePanel.tsx`

- [ ] **Step 1: Extend `PatchingState` + `DEFAULT_PATCHING_STATE`**

In `PatchingControls.tsx`:

```tsx
export interface PatchingState {
  /* existing fields */
  mode: "exact" | "approx";
}

export const DEFAULT_PATCHING_STATE: PatchingState = {
  /* existing defaults */
  mode: "exact",
};
```

- [ ] **Step 2: Add the mode radio**

Inside the `PatchingControls` component JSX, below the existing direction radio:

```tsx
<div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 4 }}>
  <span style={{ color: "#8888aa", fontSize: 12 }}>mode</span>
  <label>
    <input
      type="radio"
      checked={state.mode === "exact"}
      onChange={() => onChange({ ...state, mode: "exact" })}
    />
    exact
  </label>
  <label>
    <input
      type="radio"
      checked={state.mode === "approx"}
      onChange={() => onChange({ ...state, mode: "approx" })}
    />
    approx <span style={{ color: "#888", fontSize: 11 }}>(gradient AP, fast)</span>
  </label>
</div>
{state.mode === "approx" && state.tokenPairMode === "auto" && (
  <div style={{ color: "#7f7", fontSize: 11, marginLeft: 40 }}>
    auto-pick uses clean argmax; switch to manual for a specific target
  </div>
)}
```

- [ ] **Step 3: Forward `mode` in `ProbePanel.handleRun`**

In `ProbePanel.tsx` — in the `activation-patching` branch where the cfg is built:

```tsx
const cfg = {
  clean_prompt: /* ... */,
  corrupted_prompt: /* ... */,
  direction: /* ... */,
  measurement_position: /* ... */,
  mode: patchingState.mode,   // new
  ...(tokenPairMode === "manual" && { correct_token, incorrect_token }),
};
```

- [ ] **Step 4: tsc + Playwright smoke**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: clean.

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e -- --grep "activation-patching"
```

Expected: existing 1 AP test passes (mode radio doesn't break Phase 3 flow; defaults preserve exact behavior).

**NOTE**: vite/Playwright call must be invoked with `dangerouslyDisableSandbox: true`.

- [ ] **Step 5: Commit**

```bash
git -C /home/ai/ai-projects/llm add \
  testing/gui/frontend/src/components/PatchingControls.tsx \
  testing/gui/frontend/src/components/ProbePanel.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): PatchingControls mode radio"
```

---

## Task 9: Mode-branch in `ActivationPatchingHeatmap`

**Files:**
- Modify: `testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx`

- [ ] **Step 1: Read current heatmap**

Get a mental model of the `useEffect` that computes cell values + the metric dropdown.

- [ ] **Step 2: Detect mode + branch rendering**

Near the top of the component body, after computing `baselines` and `cells`:

```tsx
const completeFrame = useMemo(
  () => result.data.find((m): m is PatchingCompleteData => m.type === "complete"),
  [result.data]
);
const mode: "exact" | "approx" = completeFrame?.summary.mode ?? "exact";
```

Replace the `getCellValueFor` path inside the d3 `useEffect` so that when `mode === "approx"` it reads `ap_recovery` directly:

```tsx
// Inside the cell-drawing loop, replace the value computation with:
let v: number | null;
if (mode === "approx") {
  v = typeof cell.ap_recovery === "number" ? cell.ap_recovery : null;
} else {
  v = getCellValueFor(cell as PatchingCellData, metric);
}
```

Hide the metric dropdown in approx mode:

```tsx
{mode === "exact" && (
  <label style={/* existing metric-label styles */}>
    Metric:
    <select /* existing select */>...</select>
  </label>
)}
```

Change the heading + color scale in approx mode:

```tsx
<h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
  {mode === "approx" ? "Attribution Patching (\u2207)" : "Activation Patching"}
  {" \u2014 "}{result.sessionName}{" \u2014 \""}{result.prompt.slice(0, 40)}{"\""}
</h3>
```

For the color scale in approx mode, force the `logit_diff_recovery` interpolator (PiYG, fixed `[-0.5, 1.0]`) regardless of the `metric` state:

```tsx
const effectiveMetric: MetricKey = mode === "approx" ? "logit_diff_recovery" : metric;
const def = METRICS[effectiveMetric];
```

CSV export changes: when `mode === "approx"`, the rows should be `(layer, sublayer, position, ap_recovery)` only:

```tsx
const csvRows = useCallback((): (string | number)[][] => {
  if (mode === "approx") {
    const header = ["layer", "sublayer", "position", "ap_recovery"];
    const rows: (string | number)[][] = [header];
    for (const cell of cells) {
      rows.push([cell.layer, cell.sublayer, cell.position, cell.ap_recovery ?? ""]);
    }
    return rows;
  }
  /* existing exact-mode rows */
}, [cells, getCellValueFor, mode]);
```

Pin card: in approx mode, show just the scalar + a caveat:

```tsx
<PinnedCard
  cell={pinned.cell}
  x={pinned.x}
  y={pinned.y}
  mode={mode}
  onClose={() => setPinned(null)}
/>
```

And update `PinnedCard` to render the caveat when `mode === "approx"`:

```tsx
{mode === "approx" && typeof cell.ap_recovery === "number" && (
  <div style={{ marginTop: 4 }}>
    AP recovery: <strong>{cell.ap_recovery.toFixed(3)}</strong>
    <div style={{ fontSize: 10, color: "#888", marginTop: 2 }}>
      First-order approximation — run exact mode to confirm.
    </div>
  </div>
)}
```

- [ ] **Step 3: tsc clean**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/visualizations/ActivationPatchingHeatmap.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): AP heatmap mode-branch rendering"
```

---

## Task 10: Playwright smoke test for approx mode

**Files:**
- Create: `testing/gui/frontend/tests/e2e/fixtures/activation-patching-approx.json`
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Create the approx fixture**

Read `testing/gui/frontend/tests/e2e/fixtures/activation-patching.json` first for structure. Then create the approx sibling — same shape but cells use `ap_recovery` and the complete frame has `mode: "approx"`:

```json
{
  "version": 1,
  "exportedAt": "2026-04-19T00:00:00.000Z",
  "sessions": [
    {
      "name": "ap-approx-demo",
      "modelId": "mock",
      "modelLabel": "mock",
      "createdAt": "2026-04-19T00:00:00.000Z",
      "dirty": false
    }
  ],
  "results": [
    {
      "id": "ap-approx-1",
      "sessionName": "ap-approx-demo",
      "operation": "activation-patching",
      "prompt": "The capital of France is",
      "timestamp": "2026-04-19T00:00:00.000Z",
      "tags": [],
      "data": [
        {
          "type": "baselines",
          "clean_logits": {"shape": [4], "b64": "zczMPc3MTD6amZk+zcxMPw=="},
          "corrupted_logits": {"shape": [4], "b64": "zczMPc3MTD6amZk+zcxMPw=="},
          "prompt_tokens_clean": ["The", "capital", "of", "France"],
          "prompt_tokens_corrupted": ["The", "capital", "of", "Italy"],
          "measurement_position": 3,
          "correct_token_id": 0,
          "incorrect_token_id": 1
        },
        {"type": "data", "layer": 0, "sublayer": "attn", "position": 0, "ap_recovery": 0.12},
        {"type": "data", "layer": 0, "sublayer": "ffn", "position": 0, "ap_recovery": 0.05},
        {"type": "data", "layer": 1, "sublayer": "attn", "position": 2, "ap_recovery": 0.83},
        {"type": "data", "layer": 1, "sublayer": "ffn", "position": 3, "ap_recovery": 0.91},
        {
          "type": "complete",
          "summary": {"num_cells": 4, "direction": "denoise", "measurement_position": 3, "mode": "approx"}
        }
      ]
    }
  ]
}
```

- [ ] **Step 2: Write the failing Playwright test**

Append to `testing/gui/frontend/tests/e2e/smoke.spec.ts`:

```typescript
const AP_APPROX_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-approx.json");

test("attribution-patching heatmap renders without metric dropdown", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(AP_APPROX_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-approx.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  // Heatmap heading includes the gradient-mode label.
  await page.getByRole("heading", { name: /Attribution Patching/ }).waitFor({ state: "visible", timeout: 5000 });

  // Metric dropdown is NOT rendered in approx mode.
  const metricSelects = page.locator("select").filter({
    has: page.locator("option", { hasText: "Logit-diff recovery" }),
  });
  await expect(metricSelects).toHaveCount(0);

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});
```

- [ ] **Step 3: Run Playwright**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```

Invoke with `dangerouslyDisableSandbox: true`. Expected: 11/11 tests pass (10 existing + 1 new).

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add \
  testing/gui/frontend/tests/e2e/fixtures/activation-patching-approx.json \
  testing/gui/frontend/tests/e2e/smoke.spec.ts
git -C /home/ai/ai-projects/llm commit -m "test(gui/frontend): Playwright smoke for AP approx heatmap"
```

---

## Task 11: Final verification + roadmap memory update

- [ ] **Step 1: Full verification sweep**

Run in parallel where possible:

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_attribution_patch.py tests/test_probe_activation_patch.py -v
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py gui/backend/routes/probes.py tests/test_probe_attribution_patch.py tests/test_probe_activation_patch.py
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```

All must be green. GPU test (Task 5) + Playwright require `dangerouslyDisableSandbox: true`.

- [ ] **Step 2: Update roadmap memory**

Edit `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md`: replace the "Next up: Phase 3.5" line with a "Phase 3.5 shipped" entry in the same style as the Phase 3 entry. Include:

- Brief description of what shipped (gradient AP as mode-toggle).
- Commit SHAs per task.
- Verification matrix (pytest counts, pyright result, tsc result, Playwright count).
- Updated "Next up" line (what's genuinely next — could be Phase 3 follow-ups from the existing list, or nothing pending).

- [ ] **Step 3: No commit for the memory update**

Memory lives outside the repo — no git commit needed.

---

## Self-review checklist

**Spec coverage:**
- [ ] `attribution_patch()` with denoise + noise — Tasks 3 + 4
- [ ] `_capture_residual_stream_with_grad` — Task 2
- [ ] `PatchingResult.mode` — Task 1
- [ ] WS route `cfg.mode` branch — Task 6
- [ ] Frontend types — Task 7
- [ ] `PatchingControls` mode radio — Task 8
- [ ] Heatmap mode branch — Task 9
- [ ] TinyLlama correlation test — Task 5
- [ ] Playwright approx smoke — Task 10

**Type signatures used consistently:**
- `attribution_patch` signature matches between Task 3 scaffold and Task 4 full.
- `PatchingResult.mode` defaults to `"exact"` (Task 1) everywhere.
- `PatchingCellData.ap_recovery?: number` (Task 7) matches backend emission (Task 6) and frontend consumption (Task 9).
- `PatchingCompleteData.summary.mode?: "exact" | "approx"` matches backend (Task 6) + frontend default-to-exact (Task 9).

**No placeholders:** every code block is complete runnable code, not pseudo.
