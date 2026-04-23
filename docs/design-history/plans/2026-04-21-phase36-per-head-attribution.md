# Phase 3.6 Per-Head Attribution Patching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add per-attention-head gradient AP as `mode="approx_head"` in the `/activation-patching` operation. New Python function `attribution_patch_per_head()` decomposes the existing `attn_out.grad` into per-head gradients via chain rule through `W_O`, producing per-`(layer, head, position)` AP scores plus FFN anchor rows. New frontend component `PerHeadPatchingHeatmap.tsx` displays the result.

**Architecture:** One new Python function + one extended capture helper in `probe.py`. Existing `attribution_patch` (Phase 3.5) is unchanged except for a one-line unpack update. Backend WS handler gains a third `mode` branch. Frontend gains one new component and a mode radio extension. No new routes.

**Tech Stack:** Python 3.11, PyTorch (autograd), transformers (HF LLaMA), FastAPI WebSockets, React + TypeScript + Zustand, d3, pytest (Python), Playwright (frontend E2E).

**Spec:** `testing/docs/superpowers/specs/2026-04-21-phase36-per-head-attribution.md`.

**Cwd for tool invocations:** `/home/ai/ai-projects/llm`. Pyright runs from `testing/`. tsc and Playwright run from `testing/gui/frontend/`.

---

## Tool rules (apply to every task + every subagent prompt)

- Use `Read` (not `cat`), `Edit` (not `sed`/`awk`/`cat`), `Grep` (not shell grep), `Glob` (not `find`). All file ops go through dedicated tools.
- For git ops outside the repo root, use `git -C <path>` rather than `cd`.
- Avoid unnecessary compound commands. Avoid chaining that would trigger a permission prompt.
- **GPU tests:** any Bash call that runs pytest touching CUDA must use `dangerouslyDisableSandbox: true`. If a subagent cannot get that permission, surface BLOCKED status.
- **Playwright / Vite:** invoke with `dangerouslyDisableSandbox: true` (they touch `/dev/urandom`).
- Pyright CLI: run from `testing/` cwd. Command: `.venv/bin/python -m pyright <paths>`.
- Frontend tsc: run from `testing/gui/frontend/`. Command: `./node_modules/.bin/tsc --noEmit`.
- Playwright: run from `testing/gui/frontend/`. Command: `npm run e2e`.
- Zero-diagnostics discipline: every commit must land with pyright 0/0/0 and tsc clean.
- **Model selection for subagent dispatch:** sonnet or opus only. Never haiku.

---

## File Structure

### New files
- `testing/tests/test_probe_per_head_ap.py` — unit + TinyLlama Spearman integration.
- `testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx` — new viz component.
- `testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-head.json` — Playwright fixture.

### Modified files
- `testing/llm_surgeon/probe.py` — extend `_capture_residual_stream_with_grad` (5-tuple return + `capture_concat_z` flag); add `n_heads` field to `PatchingResult`; add `attribution_patch_per_head()`; update `attribution_patch` to unpack 5-tuple.
- `testing/gui/backend/routes/probes.py` — `approx_head` mode branch + `unit`-keyed frames + `n_heads` in complete frame.
- `testing/gui/frontend/src/types/api.ts` — `PatchingCellData.unit?`, `head?`; `PatchingCompleteData.summary.mode` extended to `"approx_head"`; `summary.n_heads?`.
- `testing/gui/frontend/src/components/PatchingControls.tsx` — third mode radio + `PatchingMode` type extension.
- `testing/gui/frontend/src/components/ProbePanel.tsx` — route `mode === "approx_head"` to `<PerHeadPatchingHeatmap>`.
- `testing/gui/frontend/tests/e2e/smoke.spec.ts` — one new per-head smoke test.

---

## Task 1: Add `n_heads` field to `PatchingResult` + extend `_capture_residual_stream_with_grad` return signature

**Why first:** both `attribution_patch` (Phase 3.5) and the new `attribution_patch_per_head` call `_capture_residual_stream_with_grad`. Extending the return from 4-tuple to 5-tuple must happen before any new code references the 5th element, and the existing Phase 3.5 caller must be updated atomically to avoid breaking tests.

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify (if it exists): `testing/tests/test_probe_attribution_patch.py` (add assert for `n_heads` default)

- [ ] **Step 1: Write failing test**

Create `testing/tests/test_probe_per_head_ap.py` (new file):

```python
"""Tests for probe.attribution_patch_per_head — per-head gradient AP (Phase 3.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import torch

from llm_surgeon.probe import PatchingResult, _capture_residual_stream_with_grad


def _tinyllama_cached() -> bool:
    root = Path(__file__).resolve().parents[1] / ".cache" / "models"
    return any(root.glob("models--TinyLlama--*"))


class TestPatchingResultNHeads:
    def test_n_heads_defaults_to_none(self) -> None:
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a"],
            prompt_tokens_corrupted=["b"],
            direction="denoise",
            measurement_position=0,
        )
        assert result.n_heads is None

    def test_n_heads_set_explicitly(self) -> None:
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a"],
            prompt_tokens_corrupted=["b"],
            direction="denoise",
            measurement_position=0,
            n_heads=32,
        )
        assert result.n_heads == 32
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py::TestPatchingResultNHeads -v
```

Expected: FAIL with `TypeError` (no `n_heads` parameter) or `AttributeError`.

- [ ] **Step 3: Add `n_heads` field to `PatchingResult`**

Edit `testing/llm_surgeon/probe.py` — locate the `PatchingResult` dataclass (around line 801). Add the new field after `mode`:

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
    mode: str = "exact"                  # "exact" | "approx" | "approx_head"
    n_heads: Optional[int] = None        # set by attribution_patch_per_head
```

- [ ] **Step 4: Extend `_capture_residual_stream_with_grad` to return 5-tuple**

Edit `testing/llm_surgeon/probe.py` — change the function signature and body to accept `capture_concat_z: bool = False` and add the `o_proj` pre-hook:

New signature:
```python
def _capture_residual_stream_with_grad(
    model,
    tokenizer,
    prompt: str,
    sublayers: Tuple[str, ...] = ("attn", "ffn"),
    layers: Optional[List[int]] = None,
    capture_concat_z: bool = False,
) -> Tuple[
    Dict[Tuple[int, str], torch.Tensor],
    Dict[int, torch.Tensor],
    torch.Tensor,
    List[str],
    Dict[int, torch.Tensor],
]:
```

Inside the function, add a new dict and conditionally register the pre-hook on `o_proj`:

```python
concat_z_captured: Dict[int, torch.Tensor] = {}

# (existing capture_h_in logic unchanged)

for i in range(num_layers):
    if i not in target_layers:
        continue
    # ... existing pre-hook for h_ins ...
    # ... existing attn hook ...
    # ... existing ffn hook ...
    if capture_concat_z and "attn" in sublayers:
        def make_concat_z_hook(idx: int):
            def hook(_module: torch.nn.Module, args: Tuple) -> None:
                z = args[0]             # [batch, seq, hidden]
                if z.requires_grad:
                    z.retain_grad()
                concat_z_captured[idx] = z
            return hook
        hooks.append(
            model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(
                make_concat_z_hook(i)
            )
        )

# (existing try/finally forward pass unchanged)

return captured, h_ins, model_output.logits[0], prompt_tokens, concat_z_captured
```

- [ ] **Step 5: Update the `attribution_patch` (Phase 3.5) caller to unpack 5 values**

Edit `testing/llm_surgeon/probe.py` — inside `attribution_patch()`, there are two calls to `_capture_residual_stream_with_grad`. Update both from 4-tuple to 5-tuple unpack:

```python
# In attribution_patch — from-prompt call (no_grad branch):
from_captured, from_h_ins_raw, from_logits, from_tokens, _ = \
    _capture_residual_stream_with_grad(
        model, tokenizer, from_prompt, sublayers=sublayers, layers=layers,
    )

# base-prompt call (enable_grad branch):
base_captured, base_h_ins, base_logits, base_tokens, _ = \
    _capture_residual_stream_with_grad(
        model, tokenizer, base_prompt, sublayers=sublayers, layers=layers,
    )
```

The 5th return element (empty dict when `capture_concat_z=False`) is discarded via `_`.

- [ ] **Step 6: Run all tests to verify no regression**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py::TestPatchingResultNHeads tests/test_probe_attribution_patch.py tests/test_probe_activation_patch.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_per_head_ap.py tests/test_probe_attribution_patch.py
```

Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 8: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_per_head_ap.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): PatchingResult.n_heads + capture_concat_z flag on _capture_residual_stream_with_grad"
```

---

## Task 2: `TestCaptureConcat_z` — verify concat_z shape, graph membership, grad population

**Why:** before implementing the full `attribution_patch_per_head`, lock in that `_capture_residual_stream_with_grad(capture_concat_z=True)` delivers a correctly shaped, graph-attached tensor whose `.grad` is populated after backward. This is the foundational correctness claim for per-head decomposition.

**Files:**
- Modify: `testing/tests/test_probe_per_head_ap.py`

- [ ] **Step 1: Write failing tests**

Append to `testing/tests/test_probe_per_head_ap.py`:

```python
# Reusable mock model for capture tests (hidden=8, n_heads=2, head_dim=4)
class _MockOProj(torch.nn.Linear):
    """Drop-in o_proj — preserves the [batch, seq, hidden] signature."""
    pass


class _MockSelfAttn(torch.nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.o_proj = _MockOProj(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # Simulate concat_z → o_proj; o_proj receives concat_z as its input.
        return (self.o_proj(x),)


class _MockLayer(torch.nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        self.self_attn = _MockSelfAttn(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return (x + self.self_attn(x)[0],)


class _MockModel(torch.nn.Module):
    def __init__(self, num_layers: int = 2, d_model: int = 8, vocab: int = 10) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(vocab, d_model)
        self.model.layers = torch.nn.ModuleList(
            [_MockLayer(d_model) for _ in range(num_layers)]
        )
        self.lm_head = torch.nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor):  # type: ignore[override]
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return type("Out", (), {"logits": self.lm_head(h)})()


class _MockTok:
    def __call__(self, text: str, return_tensors: Optional[str] = None) -> Dict:
        ids = [1, 2, 3] if text == "clean" else [4, 5, 6]
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids: torch.Tensor) -> list:
        return [str(int(i)) for i in ids]


class TestCaptureConcat_z:
    def test_concat_z_shape(self) -> None:
        """concat_z dict has keys for all requested layers; shape [1, seq, hidden]."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=True,
            )
        assert set(concat_z.keys()) == {0, 1}, f"expected layers {{0,1}}, got {set(concat_z.keys())}"
        for L, z in concat_z.items():
            assert z.shape == (1, 3, 8), f"layer {L}: expected [1,3,8], got {z.shape}"

    def test_concat_z_in_graph(self) -> None:
        """Base-side concat_z tensors are in the autograd graph."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=True,
            )
        for L, z in concat_z.items():
            assert z.requires_grad, f"layer {L} concat_z must require grad"
            assert z.grad_fn is not None, f"layer {L} concat_z must have grad_fn"

    def test_concat_z_grad_populates_after_backward(self) -> None:
        """After backward, concat_z.grad is non-None and non-zero."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, logits, _, concat_z = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=True,
            )
            logits.sum().backward()
        for L, z in concat_z.items():
            assert z.grad is not None, f"layer {L} concat_z.grad is None after backward"
            assert z.grad.abs().sum().item() > 0, f"layer {L} concat_z.grad is all zeros"

    def test_concat_z_subset_layers(self) -> None:
        """layers=[1] only captures concat_z at layer 1, not layer 0."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=[1],
                capture_concat_z=True,
            )
        assert set(concat_z.keys()) == {1}, f"expected only layer 1, got {set(concat_z.keys())}"

    def test_capture_concat_z_false_returns_empty(self) -> None:
        """Default flag (False) returns an empty concat_z dict."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        tok = _MockTok()
        with torch.enable_grad():
            _, _, _, _, concat_z = _capture_residual_stream_with_grad(
                model, tok, "clean",
                sublayers=("attn", "ffn"), layers=None,
                capture_concat_z=False,
            )
        assert concat_z == {}, f"expected empty dict, got {concat_z}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py::TestCaptureConcat_z -v
```

Expected: most FAIL (shape wrong, or import error, or concat_z is empty when it shouldn't be).

- [ ] **Step 3: Verify implementation already in place from Task 1**

If Task 1's implementation is correct, these tests may already pass. Run them — if they all pass, skip to Step 4.

If any fail: revisit the `capture_concat_z` hook in `_capture_residual_stream_with_grad`. The most common issue is hooking `o_proj` via `register_forward_pre_hook` with the wrong `args` index — verify `args[0]` is the `concat_z` tensor (the pre-projection input, shape `[batch, seq, hidden]`).

- [ ] **Step 4: Run full test suite to verify no regression**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py tests/test_probe_attribution_patch.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_per_head_ap.py
```

Expected: `0/0/0`.

- [ ] **Step 6: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_per_head_ap.py
git -C /home/ai/ai-projects/llm commit -m "test(probe): TestCaptureConcat_z — concat_z shape, graph, grad population"
```

---

## Task 3: `attribution_patch_per_head` — mock model validation + core algorithm

**Why:** implement the new function and lock in correctness via the sum-over-heads invariant on a small mock model before running on a real model.

**Files:**
- Modify: `testing/llm_surgeon/probe.py`
- Modify: `testing/tests/test_probe_per_head_ap.py`

- [ ] **Step 1: Write failing tests**

Append to `testing/tests/test_probe_per_head_ap.py`:

```python
from llm_surgeon.probe import attribution_patch, attribution_patch_per_head


class TestPerHeadAP:
    def test_sum_invariant_mock(self) -> None:
        """sum_h AP_head(L, h, pos) == AP_attn(L, pos) at tolerance 1e-5.

        This is the primary correctness test: per-head scores are a linear
        decomposition of the node-level attn AP via chain rule through W_O.
        """
        torch.manual_seed(42)
        # hidden=8, n_heads=2, head_dim=4, num_layers=2
        model = _MockModel(num_layers=2, d_model=8).eval()
        # Teach the model to pretend it has a config.
        model.config = type("cfg", (), {
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()

        # Phase 3.5 node-level AP for attn rows (denoise)
        node_result = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
            measurement_position=-1,
            sublayers=("attn",),
        )
        node_attn: Dict[Tuple[int, int], float] = {
            (c["layer"], c["position"]): c["ap_recovery"]
            for c in node_result.cells
            if c.get("sublayer") == "attn"
        }

        # Phase 3.6 per-head AP (denoise)
        head_result = attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
            measurement_position=-1,
        )
        assert head_result.mode == "approx_head"
        assert head_result.n_heads == 2

        # Group head cells by (layer, pos) and sum
        head_cells: Dict[Tuple[int, int], float] = {}
        for c in head_result.cells:
            unit: str = c["unit"]
            if not unit.startswith("attn."):
                continue
            key = (c["layer"], c["position"])
            head_cells[key] = head_cells.get(key, 0.0) + c["ap_recovery"]

        # Invariant: sum_h ≈ node-level
        for key in node_attn:
            assert key in head_cells, f"missing key {key} in per-head result"
            diff = abs(head_cells[key] - node_attn[key])
            assert diff < 1e-5, (
                f"sum invariant failed at {key}: "
                f"sum_heads={head_cells[key]:.8f}, node={node_attn[key]:.8f}, diff={diff:.2e}"
            )

    def test_ffn_anchor_matches_phase35(self) -> None:
        """FFN cells from attribution_patch_per_head match attribution_patch FFN cells."""
        torch.manual_seed(7)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()

        node_result = attribution_patch(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
            sublayers=("ffn",),
        )
        node_ffn = {
            (c["layer"], c["position"]): c["ap_recovery"]
            for c in node_result.cells
            if c.get("sublayer") == "ffn"
        }

        head_result = attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="denoise",
        )
        head_ffn = {
            (c["layer"], c["position"]): c["ap_recovery"]
            for c in head_result.cells
            if c.get("unit") == "ffn"
        }

        for key in node_ffn:
            assert key in head_ffn, f"missing FFN key {key}"
            diff = abs(head_ffn[key] - node_ffn[key])
            assert diff < 1e-5, f"FFN mismatch at {key}: {diff:.2e}"

    def test_cell_count_mock(self) -> None:
        """L=2, n_heads=2, seq=3, positions=all → 18 cells (2 heads + 1 ffn) × 2 layers × 3 pos."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        cells: list = []
        attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            on_cell=lambda L, unit, pos, c: cells.append(c),
        )
        # 2 layers × (2 heads + 1 ffn) × 3 positions = 18
        assert len(cells) == 18, f"expected 18 cells, got {len(cells)}"

    def test_on_cell_unit_strings(self) -> None:
        """on_cell receives correct unit strings: 'attn.h0', 'attn.h1', 'ffn'."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        units: set = set()
        attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            on_cell=lambda L, unit, pos, c: units.add(unit),
        )
        assert "attn.h0" in units
        assert "attn.h1" in units
        assert "ffn" in units
        assert "attn" not in units         # must NOT use old sublayer names
        assert "attn.h2" not in units      # exactly 2 heads

    def test_noise_direction(self) -> None:
        """Noise direction applies 1 + ap_raw/D convention."""
        torch.manual_seed(5)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        result = attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            direction="noise",
        )
        assert result.direction == "noise"
        # All cells must have ap_recovery key
        for c in result.cells:
            assert "ap_recovery" in c
            assert "patched_logits" not in c

    def test_positions_subset(self) -> None:
        """positions=[0,2] yields only those positions."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        tok = _MockTok()
        cells: list = []
        attribution_patch_per_head(
            model, tok,
            clean_prompt="clean", corrupted_prompt="corrupted",
            correct_token_id=1, incorrect_token_id=2,
            positions=[0, 2],
            on_cell=lambda L, unit, pos, c: cells.append(c),
        )
        unique_positions = {c["position"] for c in cells}
        assert unique_positions == {0, 2}

    def test_identical_baselines_raises(self) -> None:
        """Divide-by-zero guard."""
        torch.manual_seed(0)
        model = _MockModel(num_layers=2, d_model=8).eval()
        model.config = type("cfg", (), {
            "num_attention_heads": 2,
            "hidden_size": 8,
        })()
        # Same token ids for both prompts → identical forward → identical logit_diff.
        class _SameTok:
            def __call__(self, text: str, return_tensors: Optional[str] = None) -> Dict:
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids: torch.Tensor) -> list:
                return [str(int(i)) for i in ids]

        tok = _SameTok()
        with pytest.raises(ValueError, match="identical logit_diff"):
            attribution_patch_per_head(
                model, tok,
                clean_prompt="clean", corrupted_prompt="corrupted",
                correct_token_id=1, incorrect_token_id=2,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py::TestPerHeadAP -v
```

Expected: FAIL with `ImportError: cannot import name 'attribution_patch_per_head'`.

- [ ] **Step 3: Implement `attribution_patch_per_head`**

Append to `testing/llm_surgeon/probe.py` after `attribution_patch`:

```python
def attribution_patch_per_head(
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
    on_cell: Optional[Callable[[int, str, int, Dict], None]] = None,
) -> PatchingResult:
    """Per-attention-head gradient attribution patching (Phase 3.6).

    Decomposes the node-level attn AP from Phase 3.5 into per-head scores via
    chain rule through W_O (o_proj). Produces per-(layer, head, position) AP
    values plus FFN anchor rows. One forward + one backward pass, same cost as
    Phase 3.5.

    Unit strings in on_cell / cells: "attn.h{N}" (0-indexed) for head N,
    "ffn" for FFN anchor. Invariant: sum_h AP_head(L,h,pos) == AP_attn(L,pos)
    at ~1e-5 tolerance.
    """
    import warnings

    if correct_token_id is None or incorrect_token_id is None:
        raise ValueError(
            "attribution_patch_per_head requires correct_token_id and incorrect_token_id"
        )
    if direction not in ("denoise", "noise"):
        raise ValueError("direction must be 'denoise' or 'noise'")
    if not clean_prompt or not corrupted_prompt:
        raise ValueError("prompt cannot be empty")

    if getattr(model, "hf_quantizer", None) is not None:
        warnings.warn(
            "attribution_patch_per_head on a quantized model: gradient flow works "
            "but precision is reduced.",
            stacklevel=2,
        )

    clean_ids = tokenizer(clean_prompt, return_tensors="pt")["input_ids"]
    corr_ids = tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"]
    if clean_ids.shape[1] != corr_ids.shape[1]:
        raise ValueError(
            f"prompts must tokenize to same length "
            f"(clean={clean_ids.shape[1]}, corrupted={corr_ids.shape[1]})"
        )
    seq_len = clean_ids.shape[1]
    normalized_positions: List[int] = (
        list(range(seq_len)) if positions is None
        else [p if p >= 0 else seq_len + p for p in positions]
    )
    meas_pos = measurement_position % seq_len

    n_heads: int = model.config.num_attention_heads
    hidden: int = model.config.hidden_size
    head_dim: int = hidden // n_heads

    from_prompt = clean_prompt if direction == "denoise" else corrupted_prompt
    base_prompt = corrupted_prompt if direction == "denoise" else clean_prompt
    sublayers: Tuple[str, ...] = ("attn", "ffn")

    with torch.no_grad():
        from_captured, from_h_ins_raw, from_logits, from_tokens, from_concat_z_raw = \
            _capture_residual_stream_with_grad(
                model, tokenizer, from_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
            )
        from_states = {k: v.detach().clone() for k, v in from_captured.items()}
        from_h_ins = {i: v.detach().clone() for i, v in from_h_ins_raw.items()}
        from_concat_z = {i: v.detach().clone() for i, v in from_concat_z_raw.items()}

    with torch.enable_grad():
        base_captured, base_h_ins, base_logits, base_tokens, base_concat_z = \
            _capture_residual_stream_with_grad(
                model, tokenizer, base_prompt,
                sublayers=sublayers, layers=layers,
                capture_concat_z=True,
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

    num_layers = len(model.model.layers)
    target_layers = sorted(
        set(range(num_layers)) if layers is None else set(layers)
    )

    cells: List[Dict] = []

    for L in target_layers:
        # --- FFN anchor (identical math to Phase 3.5) ---
        if (L, "ffn") in base_captured:
            base_ffn = base_captured[(L, "ffn")]    # [1, seq, hidden]
            ffn_grad = base_ffn.grad
            from_ffn = from_states.get((L, "ffn"))
            if ffn_grad is not None and from_ffn is not None:
                for pos in normalized_positions:
                    # FFN uses the full layer output (residual stream post-layer)
                    # same as Phase 3.5.
                    ap_raw = (
                        (from_ffn[0, pos] - base_ffn[0, pos].detach()) * ffn_grad[0, pos]
                    ).sum().item()
                    ap_recovery = ap_raw / denominator if direction == "denoise" else 1.0 + ap_raw / denominator
                    cell: Dict = {"layer": L, "unit": "ffn", "position": pos,
                                  "ap_recovery": float(ap_recovery)}
                    cells.append(cell)
                    if on_cell is not None:
                        on_cell(L, "ffn", pos, cell)

        # --- Per-head AP via chain rule through W_O ---
        if (L, "attn") in base_captured and L in base_concat_z:
            attn_out_grad = base_captured[(L, "attn")].grad    # [1, seq, hidden]
            if attn_out_grad is None:
                continue
            W_O: torch.Tensor = model.model.layers[L].self_attn.o_proj.weight  # [hidden, hidden]
            # Chain rule: ∂metric/∂concat_z = attn_out_grad @ W_O
            # attn_out_grad[0]: [seq, hidden]; W_O: [hidden, hidden]
            concat_z_grad = attn_out_grad[0] @ W_O             # [seq, hidden]

            base_cz = base_concat_z[L]              # [1, seq, hidden], in graph
            from_cz = from_concat_z.get(L)
            if from_cz is None:
                continue

            for pos in normalized_positions:
                delta_z = (from_cz[0, pos] - base_cz[0, pos].detach())      # [hidden]
                cz_grad_pos = concat_z_grad[pos]                              # [hidden]

                dz_heads = delta_z.view(n_heads, head_dim)           # [n_heads, head_dim]
                cz_grad_heads = cz_grad_pos.view(n_heads, head_dim)  # [n_heads, head_dim]
                ap_heads_raw = (dz_heads * cz_grad_heads).sum(dim=-1)  # [n_heads]

                for h in range(n_heads):
                    ap_raw_h = ap_heads_raw[h].item()
                    ap_recovery_h = (
                        ap_raw_h / denominator
                        if direction == "denoise"
                        else 1.0 + ap_raw_h / denominator
                    )
                    unit = f"attn.h{h}"
                    hcell: Dict = {"layer": L, "unit": unit, "position": pos,
                                   "ap_recovery": float(ap_recovery_h)}
                    cells.append(hcell)
                    if on_cell is not None:
                        on_cell(L, unit, pos, hcell)

    clean_tokens = from_tokens if direction == "denoise" else base_tokens
    corrupted_tokens = base_tokens if direction == "denoise" else from_tokens

    return PatchingResult(
        cells=cells,
        clean_baseline_logits=clean_baseline.detach(),
        corrupted_baseline_logits=corrupted_baseline.detach(),
        prompt_tokens_clean=clean_tokens,
        prompt_tokens_corrupted=corrupted_tokens,
        direction=direction,
        measurement_position=meas_pos,
        mode="approx_head",
        n_heads=n_heads,
    )
```

**Key correctness note on `W_O` orientation:** `o_proj.weight` in HF LLaMA is shape `[out_features=hidden, in_features=hidden]`. For a linear `y = x @ W.T`, we have `∂metric/∂x = ∂metric/∂y @ W`. So `concat_z_grad = attn_out_grad[0] @ W_O` is correct (not `W_O.T`). Verify this against the shapes if tests fail.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py::TestPerHeadAP -v
```

Expected: all 7 tests PASS.

If `test_sum_invariant_mock` fails with diff > 1e-5: this is a sign error in the `W_O` matmul direction. Debug by checking `W_O` shape and whether `@` should be `attn_out_grad[0] @ W_O` vs `attn_out_grad[0] @ W_O.T`. The correct formula is `∂L/∂(concat_z) = ∂L/∂(attn_out) @ W_O` because `attn_out = concat_z @ W_O.T` means `∂L/∂(concat_z) = (∂L/∂(attn_out)) @ (W_O.T).T = ∂L/∂(attn_out) @ W_O`.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py tests/test_probe_attribution_patch.py tests/test_probe_activation_patch.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py tests/test_probe_per_head_ap.py
```

Expected: `0/0/0`.

- [ ] **Step 7: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/llm_surgeon/probe.py testing/tests/test_probe_per_head_ap.py
git -C /home/ai/ai-projects/llm commit -m "feat(probe): attribution_patch_per_head with sum-over-heads invariant"
```

---

## Task 4: TinyLlama Spearman — sum-over-heads vs node-level AP

**Why:** the mock-model sum invariant proves the chain-rule algebra is correct in principle. The TinyLlama test proves (a) the invariant holds on real LLaMA weights at numerical precision, and (b) the `o_proj.weight` orientation is correct for a real HF model.

**Files:**
- Modify: `testing/tests/test_probe_per_head_ap.py`

- [ ] **Step 1: Write the failing test**

Append to `testing/tests/test_probe_per_head_ap.py`:

```python
class TestTinyLlamaSpearman:
    @pytest.mark.skipif(
        not _tinyllama_cached() or not torch.cuda.is_available(),
        reason="requires cached TinyLlama and CUDA",
    )
    def test_head_sum_vs_node_ap_spearman(self) -> None:
        """sum_h AP_head(L,h,pos) vs AP_attn(L,pos): Spearman ρ > 0.95.

        Should be ~1.0 (same quantity reconstructed), tolerance >0.95 guards
        against floating-point catastrophe or sign errors.
        """
        import scipy.stats
        from llm_surgeon.surgery import load_model

        model, tokenizer = load_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda"
        )
        model.eval()

        clean = "The capital of France is"
        corrupted = "The capital of Italy is"

        with torch.no_grad():
            device = next(model.parameters()).device
            clean_ids = tokenizer(clean, return_tensors="pt")["input_ids"].to(device)
            corr_ids = tokenizer(corrupted, return_tensors="pt")["input_ids"].to(device)
            clean_logits = model(clean_ids).logits[0, -1]
            corr_logits = model(corr_ids).logits[0, -1]
        correct_id = int(clean_logits.argmax().item())
        incorrect_id = int(corr_logits.argmax().item())

        # Node-level AP (Phase 3.5) — attn rows only
        node_result = attribution_patch(
            model, tokenizer,
            clean_prompt=clean, corrupted_prompt=corrupted,
            correct_token_id=correct_id, incorrect_token_id=incorrect_id,
            direction="denoise", measurement_position=-1,
            sublayers=("attn",),
        )
        node_attn: Dict[Tuple[int, int], float] = {
            (c["layer"], c["position"]): c["ap_recovery"]
            for c in node_result.cells
            if c.get("sublayer") == "attn"
        }

        # Per-head AP (Phase 3.6) — sum heads per (layer, pos)
        head_result = attribution_patch_per_head(
            model, tokenizer,
            clean_prompt=clean, corrupted_prompt=corrupted,
            correct_token_id=correct_id, incorrect_token_id=incorrect_id,
            direction="denoise", measurement_position=-1,
        )
        head_sum: Dict[Tuple[int, int], float] = {}
        for c in head_result.cells:
            if c.get("unit", "").startswith("attn."):
                key = (c["layer"], c["position"])
                head_sum[key] = head_sum.get(key, 0.0) + c["ap_recovery"]

        shared = sorted(set(node_attn.keys()) & set(head_sum.keys()))
        assert len(shared) > 50, f"too few cells: {len(shared)}"

        x = [node_attn[k] for k in shared]
        y = [head_sum[k] for k in shared]
        rho, _ = scipy.stats.spearmanr(x, y)
        print(f"\nSpearman(node_attn, sum_heads) = {rho:.4f} over {len(shared)} cells")
        assert rho > 0.95, (
            f"Spearman ρ={rho:.4f} < 0.95; sum-over-heads deviates from node-level AP. "
            f"Likely a W_O orientation bug or wrong concat_z tensor captured."
        )

        # Bonus: also check the absolute max deviation is small
        max_dev = max(abs(head_sum[k] - node_attn[k]) for k in shared)
        print(f"Max absolute deviation: {max_dev:.6f}")
        assert max_dev < 0.01, (
            f"Max deviation {max_dev:.6f} too large; should be near floating-point epsilon. "
            f"The chain rule reconstruction is exact by construction."
        )
```

- [ ] **Step 2: Run the test (requires GPU + TinyLlama)**

Run with `dangerouslyDisableSandbox: true`:

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py::TestTinyLlamaSpearman -v -s
```

Expected: runs ~30–60 s, prints ρ (should be >0.999), passes.

**If ρ < 0.95:** Do NOT lower the threshold. Diagnose:
1. **W_O orientation:** print `model.model.layers[0].self_attn.o_proj.weight.shape` — should be `[2048, 2048]`. The formula `attn_out_grad @ W_O` requires `W_O` of shape `[hidden, hidden]` (same shape), which is correct.
2. **concat_z captured correctly:** print `concat_z[0].shape` during a test forward — should be `[1, seq, 2048]`. If it's `[seq, 2048]` the hook is capturing the wrong arg.
3. **Batch dim:** ensure all indexing uses `[0, pos, :]` (batch index 0).

- [ ] **Step 3: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright tests/test_probe_per_head_ap.py
```

Expected: `0/0/0`.

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/tests/test_probe_per_head_ap.py
git -C /home/ai/ai-projects/llm commit -m "test(probe): TinyLlama Spearman sum-over-heads vs node-level attn AP"
```

---

## Task 5: Backend — `approx_head` mode branch in `probes.py`

**Why:** wire the new Python function into the existing WS handler. No new route; just a third branch.

**Files:**
- Modify: `testing/gui/backend/routes/probes.py`

- [ ] **Step 1: Read the current `activation_patching_ws` handler**

Locate the `mode` validation block (currently checks `not in ("exact", "approx")`) and the `on_cell` closure. Understand the current structure before editing.

- [ ] **Step 2: Extend mode validation**

Find:
```python
if mode not in ("exact", "approx"):
    await _send_json(ws, {"type": "error",
                          "message": f"mode must be 'exact' or 'approx', got {mode!r}"})
```

Change to:
```python
if mode not in ("exact", "approx", "approx_head"):
    await _send_json(ws, {"type": "error",
                          "message": f"mode must be 'exact', 'approx', or 'approx_head', got {mode!r}"})
```

- [ ] **Step 3: Update `on_cell` closure to emit `unit` for `approx_head` mode**

The current `on_cell` always emits `sublayer`. For `approx_head`, cells carry `"unit"` instead. Update:

```python
def on_cell(layer_idx: int, unit_or_sub: str, position: int, cell: dict) -> None:
    nonlocal connected
    if not connected:
        return
    msg: dict = {
        "type": "data",
        "layer": layer_idx,
        "original_layer": info.original_layer(layer_idx),
        "position": position,
    }
    if mode == "approx_head":
        msg["unit"] = cell.get("unit", unit_or_sub)
    else:
        msg["sublayer"] = unit_or_sub
    if "patched_logits" in cell:
        msg["patched_logits"] = _encode_hidden_state(cell["patched_logits"])
    if "ap_recovery" in cell:
        msg["ap_recovery"] = cell["ap_recovery"]
    fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
    try:
        ok = fut.result(timeout=10)
    except Exception:
        ok = False
    if not ok:
        connected = False
```

- [ ] **Step 4: Add `approx_head` dispatch branch**

Inside the `async with info.lock:` block, after the existing `else:` (approx branch), add:

```python
elif mode == "approx_head":
    assert correct_token_id is not None and incorrect_token_id is not None
    _cid: int = correct_token_id
    _iid: int = incorrect_token_id
    from llm_surgeon.probe import attribution_patch_per_head
    result = await loop.run_in_executor(
        None,
        lambda: attribution_patch_per_head(
            info.model, info.tokenizer,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            correct_token_id=_cid,
            incorrect_token_id=_iid,
            direction=direction,
            measurement_position=measurement_position,
            positions=positions,
            layers=layers,
            on_cell=on_cell,
        ),
    )
```

Note: the auto-pick token IDs block (`if mode == "approx" and correct_token_id is None`) must also trigger for `mode == "approx_head"`:

```python
if mode in ("approx", "approx_head") and correct_token_id is None:
    # (existing auto-pick block unchanged)
    ...
```

- [ ] **Step 5: Extend complete frame to include `n_heads`**

```python
summary: dict = {
    "num_cells": len(result.cells),
    "direction": result.direction,
    "measurement_position": result.measurement_position,
    "mode": result.mode,
}
if result.n_heads is not None:
    summary["n_heads"] = result.n_heads
await _send_json(ws, {"type": "complete", "summary": summary})
```

- [ ] **Step 6: Pyright clean**

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright gui/backend/routes/probes.py
```

Expected: `0/0/0`.

- [ ] **Step 7: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/backend/routes/probes.py
git -C /home/ai/ai-projects/llm commit -m "feat(gui/backend): approx_head mode branch + unit-keyed frames + n_heads in complete"
```

---

## Task 6: Frontend types — extend `api.ts`

**Why:** `unit` and `n_heads` must be typed before any consuming component is written.

**Files:**
- Modify: `testing/gui/frontend/src/types/api.ts`

- [ ] **Step 1: Extend `PatchingCellData`**

Find the existing interface:
```typescript
export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer?: "attn" | "ffn";
  position: number;
  patched_logits?: EncodedTensor;
  ap_recovery?: number;
}
```

Update `sublayer` to optional (it already is in Phase 3.5 types — verify), and add `unit` and `head`:

```typescript
export interface PatchingCellData {
  type: "data";
  layer: number;
  original_layer?: number;
  sublayer?: "attn" | "ffn";     // present in exact/approx modes
  unit?: string;                  // present in approx_head mode: "attn.hN" or "ffn"
  head?: number | null;           // derived client-side from unit; not sent by backend
  position: number;
  patched_logits?: EncodedTensor;
  ap_recovery?: number;
}
```

- [ ] **Step 2: Extend `PatchingCompleteData`**

```typescript
export interface PatchingCompleteData {
  type: "complete";
  summary: {
    num_cells: number;
    direction: "denoise" | "noise";
    measurement_position: number;
    mode?: "exact" | "approx" | "approx_head";
    n_heads?: number;
  };
}
```

- [ ] **Step 3: tsc clean**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: clean. If making `sublayer` optional causes downstream errors in `ActivationPatchingHeatmap.tsx`, add a narrow `cell.sublayer!` guard there (Task 7 will cover any more surgical fix).

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/types/api.ts
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): api.ts types for per-head AP (unit, head, n_heads)"
```

---

## Task 7: Extend `PatchingControls.tsx` mode radio + `ProbePanel.tsx` routing

**Why:** expose the new mode to users, and route `approx_head` results to the new component.

**Files:**
- Modify: `testing/gui/frontend/src/components/PatchingControls.tsx`
- Modify: `testing/gui/frontend/src/components/ProbePanel.tsx`

- [ ] **Step 1: Extend `PatchingMode` type and add radio in `PatchingControls.tsx`**

Find `PatchingState.mode` type (`"exact" | "approx"`). Change to:

```typescript
export type PatchingMode = "exact" | "approx" | "approx_head";

export interface PatchingState {
  /* existing fields */
  mode: PatchingMode;
}

export const DEFAULT_PATCHING_STATE: PatchingState = {
  /* existing defaults */
  mode: "exact",
};
```

In the JSX, after the existing `approx` radio, add:

```tsx
<label>
  <input
    type="radio"
    checked={state.mode === "approx_head"}
    onChange={() => onChange({ ...state, mode: "approx_head" })}
  />
  per-head{" "}
  <span style={{ color: "#888", fontSize: 11 }}>(gradient AP, head resolution)</span>
</label>
```

- [ ] **Step 2: Route `approx_head` results in `ProbePanel.tsx`**

Find the viz-dispatch section where `result.operation === "activation-patching"` selects the heatmap. Add a branch for `approx_head`:

```tsx
import { PerHeadPatchingHeatmap } from "./visualizations/PerHeadPatchingHeatmap";

// In the render / switch on operation + mode:
const completeFrame = result.data.find(
  (m): m is PatchingCompleteData => m.type === "complete"
);
const patchingMode = completeFrame?.summary.mode ?? "exact";

if (result.operation === "activation-patching") {
  if (patchingMode === "approx_head") {
    return <PerHeadPatchingHeatmap result={result} />;
  }
  return <ActivationPatchingHeatmap result={result} />;
}
```

(Adapt to match the actual conditional structure in `ProbePanel.tsx` — read the file first.)

- [ ] **Step 3: tsc clean**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: clean. The `PerHeadPatchingHeatmap` import may fail until Task 8 creates the file — stub it first with an empty component:

```typescript
// PerHeadPatchingHeatmap.tsx (stub, will be replaced in Task 8)
import type { ProbeResult } from "../../types/api";
interface Props { result: ProbeResult; }
export function PerHeadPatchingHeatmap({ result: _result }: Props) {
  return <div>PerHeadPatchingHeatmap (stub)</div>;
}
```

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add \
  testing/gui/frontend/src/components/PatchingControls.tsx \
  testing/gui/frontend/src/components/ProbePanel.tsx \
  testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): approx_head mode radio + ProbePanel routing stub"
```

---

## Task 8: `PerHeadPatchingHeatmap.tsx` — full implementation

**Why:** the core new viz. Rows = layers, cols = (ffn | head 0 | head 1 | … | head N-1) at a selected position.

**Files:**
- Modify (replace stub): `testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx`

- [ ] **Step 1: Write the component**

```typescript
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { ExportButtons } from "../ExportButtons";
import type {
  ProbeResult, PatchingBaselinesData, PatchingCellData, PatchingCompleteData,
} from "../../types/api";

interface Props {
  result: ProbeResult;
}

interface PinnedCell {
  cell: PatchingCellData;
  x: number;
  y: number;
}

export function PerHeadPatchingHeatmap({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedPos, setSelectedPos] = useState<number>(0);
  const [pinned, setPinned] = useState<PinnedCell | null>(null);

  const completeFrame = useMemo(
    () => result.data.find((m): m is PatchingCompleteData => m.type === "complete"),
    [result.data]
  );
  const n_heads: number = completeFrame?.summary.n_heads ?? 0;
  const measPos: number = completeFrame?.summary.measurement_position ?? 0;

  // Initialize selectedPos to measurement_position on mount.
  const [posInit, setPosInit] = useState(false);
  if (!posInit && measPos !== 0) {
    setSelectedPos(measPos);
    setPosInit(true);
  }

  const baselines = useMemo(
    () => result.data.find((m): m is PatchingBaselinesData => m.type === "baselines"),
    [result.data]
  );
  const promptTokens: string[] = baselines?.prompt_tokens_clean ?? [];

  const allCells = useMemo(
    () => result.data.filter((m): m is PatchingCellData =>
      m.type === "data" && "unit" in m
    ),
    [result.data]
  );

  // Cells at the selected position.
  const positionCells = useMemo(
    () => allCells.filter((c) => c.position === selectedPos),
    [allCells, selectedPos]
  );

  // Unique layer indices (sorted ascending).
  const layerIds = useMemo(
    () => Array.from(new Set(allCells.map((c) => c.layer))).sort((a, b) => a - b),
    [allCells]
  );

  // Column order: "ffn" first, then "attn.h0" … "attn.h{N-1}".
  const columnUnits: string[] = useMemo(() => {
    const units = ["ffn", ...Array.from({ length: n_heads }, (_, h) => `attn.h${h}`)];
    return units;
  }, [n_heads]);

  const colorScale = useMemo(
    () => d3.scaleSequential(d3.interpolatePiYG).domain([-0.5, 1.0]),
    []
  );

  useEffect(() => {
    if (!svgRef.current || positionCells.length === 0 || columnUnits.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 24, right: 16, bottom: 32, left: 48 };
    const cellW = Math.max(12, Math.min(28, 700 / columnUnits.length));
    const cellH = 18;
    const width = margin.left + columnUnits.length * cellW + margin.right;
    const height = margin.top + layerIds.length * cellH + margin.bottom;
    svg.attr("width", width).attr("height", height);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Row labels (layer indices).
    layerIds.forEach((L, rowIdx) => {
      g.append("text")
        .attr("x", -4).attr("y", rowIdx * cellH + cellH / 2)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle")
        .attr("font-size", 9).attr("fill", "#8888aa")
        .text(`L${L}`);
    });

    // Column labels ("ffn", "h0", "h1", …).
    columnUnits.forEach((unit, colIdx) => {
      const label = unit === "ffn" ? "ffn" : unit.replace("attn.h", "h");
      g.append("text")
        .attr("x", colIdx * cellW + cellW / 2).attr("y", -6)
        .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#666")
        .text(label);
    });

    // Build lookup: (layer, unit) → ap_recovery.
    const cellMap = new Map<string, number>();
    for (const c of positionCells) {
      const u = c.unit ?? "";
      if (u) cellMap.set(`${c.layer}.${u}`, c.ap_recovery ?? 0);
    }

    // Draw cells.
    layerIds.forEach((L, rowIdx) => {
      columnUnits.forEach((unit, colIdx) => {
        const v = cellMap.get(`${L}.${unit}`);
        const fill = v != null ? colorScale(v) : "#1a1a2e";
        g.append("rect")
          .attr("x", colIdx * cellW).attr("y", rowIdx * cellH)
          .attr("width", cellW - 1).attr("height", cellH - 1)
          .attr("fill", fill).attr("rx", 2)
          .style("cursor", v != null ? "pointer" : "default")
          .on("click", (event) => {
            if (v == null) return;
            const syntheticCell: PatchingCellData = {
              type: "data", layer: L, unit, position: selectedPos, ap_recovery: v,
            };
            setPinned({ cell: syntheticCell, x: event.pageX + 10, y: event.pageY + 10 });
          });
      });
    });

    // Legend bar.
    const legendY = layerIds.length * cellH + 12;
    const legendW = Math.min(180, columnUnits.length * cellW);
    const gradId = `ph-grad-${result.id}`;
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", gradId).attr("x1", "0%").attr("x2", "100%");
    for (let i = 0; i <= 16; i++) {
      grad.append("stop")
        .attr("offset", `${(i / 16) * 100}%`)
        .attr("stop-color", d3.interpolatePiYG(i / 16));
    }
    g.append("rect")
      .attr("x", 0).attr("y", legendY).attr("width", legendW).attr("height", 7)
      .attr("fill", `url(#${gradId})`).attr("rx", 1);
    g.append("text")
      .attr("x", 0).attr("y", legendY + 16)
      .attr("font-size", 8).attr("fill", "#888").text("-0.5");
    g.append("text")
      .attr("x", legendW).attr("y", legendY + 16)
      .attr("text-anchor", "end").attr("font-size", 8).attr("fill", "#888").text("1.0");
    g.append("text")
      .attr("x", legendW / 2).attr("y", legendY - 2)
      .attr("text-anchor", "middle").attr("font-size", 8).attr("fill", "#aaa")
      .text("AP recovery (PiYG)");
  }, [positionCells, layerIds, columnUnits, colorScale, selectedPos, result.id]);

  const csvRows = useCallback((): (string | number)[][] => {
    const header = ["layer", "unit", "position", "ap_recovery"];
    const rows: (string | number)[][] = [header];
    for (const c of allCells) {
      rows.push([c.layer, c.unit ?? "", c.position, c.ap_recovery ?? ""]);
    }
    return rows;
  }, [allCells]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0", margin: 0 }}>
          Per-head Attribution Patching
          {" — "}{result.sessionName}
          {" — \""}{result.prompt.slice(0, 40)}{"\""}
        </h3>
        <label style={{ fontSize: 12, color: "#8888aa", display: "flex", alignItems: "center", gap: 6 }}>
          position:
          <select
            value={selectedPos}
            onChange={(e) => setSelectedPos(Number(e.target.value))}
            style={{
              background: "#0f1626", color: "#e0e0f0", border: "1px solid #1a5276",
              borderRadius: 3, padding: "2px 6px", fontSize: 11,
            }}
          >
            {promptTokens.map((tok, i) => (
              <option key={i} value={i}>{i}: {tok}</option>
            ))}
          </select>
        </label>
        <div style={{ marginLeft: "auto" }}>
          <ExportButtons
            filenameBase={`per_head_ap_${result.sessionName}`}
            svgRef={svgRef}
            csvRows={csvRows}
          />
        </div>
      </div>
      <svg ref={svgRef} />
      {pinned && (
        <div
          style={{
            position: "fixed", top: pinned.y, left: pinned.x,
            background: "#0d1b2a", border: "1px solid #1a5276",
            borderRadius: 6, padding: "10px 14px", fontSize: 12,
            color: "#c0c8e0", zIndex: 1000, maxWidth: 280,
            boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <strong>
              L{pinned.cell.layer}, {pinned.cell.unit ?? pinned.cell.sublayer}
            </strong>
            <button
              onClick={() => setPinned(null)}
              style={{ background: "none", border: "none", color: "#888", cursor: "pointer", fontSize: 14 }}
            >
              ×
            </button>
          </div>
          {typeof pinned.cell.ap_recovery === "number" && (
            <div>
              AP recovery:{" "}
              <strong style={{ color: "#a0e0a0" }}>
                {pinned.cell.ap_recovery.toFixed(4)}
              </strong>
            </div>
          )}
          <div style={{ fontSize: 10, color: "#666", marginTop: 6 }}>
            First-order approximation — run exact mode to confirm.
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: tsc clean**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

Expected: clean.

- [ ] **Step 3: Vite build (Tier 2)**

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/vite build
```

Run with `dangerouslyDisableSandbox: true`. Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add testing/gui/frontend/src/components/visualizations/PerHeadPatchingHeatmap.tsx
git -C /home/ai/ai-projects/llm commit -m "feat(gui/frontend): PerHeadPatchingHeatmap component"
```

---

## Task 9: Playwright smoke — fixture + per-head test

**Files:**
- Create: `testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-head.json`
- Modify: `testing/gui/frontend/tests/e2e/smoke.spec.ts`

- [ ] **Step 1: Create the fixture**

The fixture must have `mode: "approx_head"` in the complete frame and cells with `unit` keys (not `sublayer`). Use 2 layers, 2 heads, and 2 positions for compactness:

```json
{
  "version": 1,
  "exportedAt": "2026-04-21T00:00:00.000Z",
  "sessions": [
    {
      "name": "ph-demo",
      "modelId": "mock",
      "modelLabel": "mock",
      "createdAt": "2026-04-21T00:00:00.000Z",
      "dirty": false
    }
  ],
  "results": [
    {
      "id": "ph-ap-1",
      "sessionName": "ph-demo",
      "operation": "activation-patching",
      "prompt": "The capital of France is",
      "timestamp": "2026-04-21T00:00:00.000Z",
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
        {"type": "data", "layer": 0, "unit": "ffn",     "position": 0, "ap_recovery": 0.05},
        {"type": "data", "layer": 0, "unit": "attn.h0", "position": 0, "ap_recovery": 0.10},
        {"type": "data", "layer": 0, "unit": "attn.h1", "position": 0, "ap_recovery": 0.08},
        {"type": "data", "layer": 1, "unit": "ffn",     "position": 3, "ap_recovery": 0.22},
        {"type": "data", "layer": 1, "unit": "attn.h0", "position": 3, "ap_recovery": 0.71},
        {"type": "data", "layer": 1, "unit": "attn.h1", "position": 3, "ap_recovery": 0.15},
        {
          "type": "complete",
          "summary": {
            "num_cells": 6,
            "direction": "denoise",
            "measurement_position": 3,
            "mode": "approx_head",
            "n_heads": 2
          }
        }
      ]
    }
  ]
}
```

- [ ] **Step 2: Write the failing Playwright test**

Append to `testing/gui/frontend/tests/e2e/smoke.spec.ts`:

```typescript
const PH_FIXTURE_PATH = path.join(__dirname, "fixtures", "activation-patching-per-head.json");

test("per-head attribution heatmap renders with position selector", async ({ page }) => {
  await page.goto("/");
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error" && !isBackendlessNoise(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });

  const fixture = fs.readFileSync(PH_FIXTURE_PATH, "utf8");
  await page.locator('input[type="file"]').setInputFiles({
    name: "activation-patching-per-head.json",
    mimeType: "application/json",
    buffer: Buffer.from(fixture),
  });

  // Heading must include "Per-head Attribution"
  await page.getByRole("heading", { name: /Per-head Attribution/ })
    .waitFor({ state: "visible", timeout: 5000 });

  // Position selector dropdown must be present (distinguishes this viz from others)
  const positionSelect = page.locator("select").filter({
    has: page.locator("option", { hasText: /^0:/ }),
  });
  await expect(positionSelect).toHaveCount(1);

  await page.waitForTimeout(100);
  expect(consoleErrors).toEqual([]);
});
```

- [ ] **Step 3: Run Playwright suite**

Run with `dangerouslyDisableSandbox: true`:

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```

Expected: 12/12 tests pass (11 existing + 1 new).

If the heading test fails: check that `ProbePanel.tsx` routes `mode === "approx_head"` to `PerHeadPatchingHeatmap` and that the heading string matches `/Per-head Attribution/` exactly.

- [ ] **Step 4: Commit**

```bash
git -C /home/ai/ai-projects/llm add \
  testing/gui/frontend/tests/e2e/fixtures/activation-patching-per-head.json \
  testing/gui/frontend/tests/e2e/smoke.spec.ts
git -C /home/ai/ai-projects/llm commit -m "test(gui/frontend): Playwright smoke for per-head AP heatmap"
```

---

## Task 10: Final verification + roadmap memory update

- [ ] **Step 1: Full sweep**

Run in parallel where possible:

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pytest tests/test_probe_per_head_ap.py tests/test_probe_attribution_patch.py tests/test_probe_activation_patch.py -v
```

```bash
cd /home/ai/ai-projects/llm/testing && .venv/bin/python -m pyright llm_surgeon/probe.py gui/backend/routes/probes.py tests/test_probe_per_head_ap.py tests/test_probe_attribution_patch.py
```

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && ./node_modules/.bin/tsc --noEmit
```

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run e2e
```

GPU test and Playwright require `dangerouslyDisableSandbox: true`.

All must be green: pyright 0/0/0, tsc clean, pytest (non-GPU) passing, Playwright 12/12.

- [ ] **Step 2: Update roadmap memory**

Edit `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md`:
append a "Phase 3.6 shipped" entry in the same style as the Phase 3.5 entry. Include:
- What shipped (per-head AP as `approx_head` mode).
- Commit SHAs per task.
- Verification matrix.
- Updated "Next up" line.

- [ ] **Step 3: No commit for memory update**

Memory lives outside the repo.

---

## Self-review checklist

**Spec coverage:**
- [ ] `_capture_residual_stream_with_grad` extended with `capture_concat_z` — Tasks 1 + 2
- [ ] `PatchingResult.n_heads` — Task 1
- [ ] `attribution_patch_per_head` signature + algorithm — Task 3
- [ ] Sum-over-heads invariant mock test — Task 3
- [ ] TinyLlama Spearman test — Task 4
- [ ] Backend `approx_head` mode branch — Task 5
- [ ] Frontend types `unit`, `head`, `n_heads` — Task 6
- [ ] `PatchingControls` third radio + `ProbePanel` routing — Task 7
- [ ] `PerHeadPatchingHeatmap` component — Task 8
- [ ] Playwright per-head smoke — Task 9

**Invariants to verify:**
- `sum_h AP_head(L,h,pos) == AP_attn(L,pos)` at 1e-5 tolerance (mock) and > 0.95 Spearman / < 0.01 max-dev (TinyLlama).
- `W_O` orientation: `concat_z_grad = attn_out_grad[0] @ W_O` (NOT `@ W_O.T`).
- `concat_z` captured via `register_forward_pre_hook` on `o_proj`, where `args[0]` is the pre-projection tensor.
- Noise direction: `1 + ap_raw/D` applied independently per head and for FFN.
- `on_cell` second argument is `unit` string (`"attn.hN"` or `"ffn"`), matching Phase 3.5 interface shape but with different vocabulary.

**No placeholders:** every code block in this plan is complete runnable code, not pseudo-code.
