# Phase 1: Core Surgery + Structural Verification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Tool rules (for subagents):**
> - Use Read (not cat/head/tail), Grep (not grep/rg/awk), Glob (not find/ls), Edit (not sed/awk) for all file operations
> - You are already in the project root — never `cd` into a directory before running a command
> - Each unique compound command triggers a new permission prompt, so avoid unnecessary chaining

**Goal:** Build the core `llm_surgeon` package with model loading, six layer surgery operations, and structural verification — all tested against a tiny in-memory LLaMA model.

**Architecture:** A Python package (`llm_surgeon`) with two modules: `surgery.py` (load models + manipulate layers) and `verify.py` (validate structural integrity after surgery). All operations work on standard HuggingFace `LlamaForCausalLM` instances. Unit tests use a tiny 8-layer LLaMA model created in-memory (no downloads needed).

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace transformers, accelerate, bitsandbytes, pytest

**Reference:** `docs/superpowers/specs/2026-04-08-llm-surgeon-design.md` (v2), Phase 1 section of `docs/superpowers/specs/2026-04-08-llm-surgeon-phase-plan.md`

---

## File Map

```
testing/
  llm_surgeon/
    __init__.py          — CREATE — package init, imports surgery + verify modules
    surgery.py           — CREATE — SurgeryOp, SurgeryLog, load_model, 6 operations
    verify.py            — CREATE — VerifyReport, check_structure
  tests/
    __init__.py          — CREATE — test package init
    conftest.py          — CREATE — tiny_llama fixture (8-layer model, no download)
    test_surgery.py      — CREATE — tests for all surgery operations
    test_verify.py       — CREATE — tests for check_structure
  pyproject.toml         — CREATE — package config, dependencies, pytest config
  requirements.txt       — CREATE — pinned dependencies
```

---

## Prerequisites

Before starting, verify:
- Python 3.10+ available
- `pip install torch transformers accelerate bitsandbytes pytest` succeeds
- Git initialized in `/home/ai/ai-projects/llm` (run `git init` if needed — commit steps assume git is available)

---

### Task 1: Project Scaffolding

**Files:**
- Create: `testing/pyproject.toml`
- Create: `testing/requirements.txt`
- Create: `testing/llm_surgeon/__init__.py`
- Create: `testing/tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-surgeon"
version = "0.1.0"
description = "Surgical layer-level manipulation of LLaMA models"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.40",
    "accelerate>=0.27",
    "bitsandbytes>=0.43",
    "requests",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.setuptools.packages.find]
where = ["."]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create requirements.txt**

```
torch>=2.0
transformers>=4.40
accelerate>=0.27
bitsandbytes>=0.43
requests
pytest>=8.0
```

- [ ] **Step 3: Create llm_surgeon/__init__.py**

```python
"""LLM Surgeon — surgical layer-level manipulation of LLaMA models."""

from llm_surgeon import surgery, verify
```

- [ ] **Step 4: Create tests/__init__.py**

Empty file:
```python
```

- [ ] **Step 5: Install package in editable mode and verify**

Run: `cd /home/ai/ai-projects/llm/testing && pip install -e ".[dev]"`
Expected: Successful install, no errors.

Run: `cd /home/ai/ai-projects/llm/testing && python -c "import llm_surgeon; print('OK')"`
Expected: Will fail (surgery and verify modules don't exist yet). This is expected — we'll create them in the next tasks.

- [ ] **Step 6: Create empty module files so import works**

Create `testing/llm_surgeon/surgery.py`:
```python
"""Model loading and layer surgery operations."""
```

Create `testing/llm_surgeon/verify.py`:
```python
"""Structural verification of modified models."""
```

- [ ] **Step 7: Verify package imports**

Run: `cd /home/ai/ai-projects/llm/testing && python -c "from llm_surgeon import surgery, verify; print('OK')"`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add pyproject.toml requirements.txt llm_surgeon/__init__.py llm_surgeon/surgery.py llm_surgeon/verify.py tests/__init__.py && git commit -m "feat: scaffold llm-surgeon package with empty modules"
```

---

### Task 2: Data Classes (SurgeryOp, SurgeryLog)

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Create: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for data classes**

Create `testing/tests/test_surgery.py`:

```python
"""Tests for surgery module."""

from llm_surgeon.surgery import SurgeryOp, SurgeryLog


class TestSurgeryOp:
    def test_creation(self):
        op = SurgeryOp(
            operation="remove_layers",
            description="Removed layers [16, 17, 18]",
            layer_count_before=32,
            layer_count_after=29,
        )
        assert op.operation == "remove_layers"
        assert op.layer_count_before == 32
        assert op.layer_count_after == 29

    def test_str(self):
        op = SurgeryOp("remove_layers", "Removed layers [16]", 32, 31)
        s = str(op)
        assert "remove_layers" in s


class TestSurgeryLog:
    def test_empty(self):
        log = SurgeryLog()
        assert len(log.ops) == 0

    def test_add_operation(self):
        log = SurgeryLog()
        log.add("remove_layers", "Removed layers [16]", 32, 31)
        assert len(log.ops) == 1
        assert log.ops[0].operation == "remove_layers"
        assert log.ops[0].layer_count_before == 32
        assert log.ops[0].layer_count_after == 31

    def test_multiple_operations(self):
        log = SurgeryLog()
        log.add("remove_layers", "Removed layers [16]", 32, 31)
        log.add("swap_layers", "Swapped 0 and 5", 31, 31)
        assert len(log.ops) == 2

    def test_str(self):
        log = SurgeryLog()
        log.add("remove_layers", "Removed layers [16]", 32, 31)
        s = str(log)
        assert "SurgeryLog" in s
        assert "remove_layers" in s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py -v`
Expected: FAIL — `ImportError: cannot import name 'SurgeryOp' from 'llm_surgeon.surgery'`

- [ ] **Step 3: Implement data classes**

Replace contents of `testing/llm_surgeon/surgery.py`:

```python
"""Model loading and layer surgery operations."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SurgeryOp:
    """A single surgery operation record."""
    operation: str
    description: str
    layer_count_before: int
    layer_count_after: int

    def __str__(self) -> str:
        return (
            f"{self.operation}: {self.description} "
            f"({self.layer_count_before} -> {self.layer_count_after} layers)"
        )


@dataclass
class SurgeryLog:
    """Log of surgery operations performed on a model."""
    ops: List[SurgeryOp] = field(default_factory=list)

    def add(self, operation: str, description: str, before: int, after: int) -> None:
        self.ops.append(SurgeryOp(operation, description, before, after))

    def __str__(self) -> str:
        if not self.ops:
            return "SurgeryLog: (empty)"
        lines = ["SurgeryLog:"]
        for op in self.ops:
            lines.append(f"  {op}")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add SurgeryOp and SurgeryLog data classes"
```

---

### Task 3: Test Fixtures (Tiny LLaMA Model)

**Files:**
- Create: `testing/tests/conftest.py`

- [ ] **Step 1: Create conftest.py with tiny_llama fixture**

Create `testing/tests/conftest.py`:

```python
"""Shared test fixtures."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def tiny_llama_config():
    """LLaMA config with small dimensions for fast testing."""
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_llama(tiny_llama_config):
    """An 8-layer LLaMA model with tiny dimensions. No download needed."""
    model = LlamaForCausalLM(tiny_llama_config)
    model.eval()
    return model
```

- [ ] **Step 2: Write a test that uses the fixture to verify it works**

Add to the top of `testing/tests/test_surgery.py`, after existing imports:

```python
class TestTinyLlamaFixture:
    def test_fixture_creates_model(self, tiny_llama):
        assert tiny_llama is not None
        assert len(tiny_llama.model.layers) == 8

    def test_fixture_has_correct_config(self, tiny_llama):
        assert tiny_llama.config.num_hidden_layers == 8
        assert tiny_llama.config.hidden_size == 32
        assert tiny_llama.config.vocab_size == 64

    def test_fixture_can_forward(self, tiny_llama):
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
```

- [ ] **Step 3: Run tests to verify fixture works**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestTinyLlamaFixture -v`
Expected: All 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add tests/conftest.py tests/test_surgery.py && git commit -m "feat: add tiny_llama test fixture for fast model testing"
```

---

### Task 4: get_layer_info

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for get_layer_info**

Add to `testing/tests/test_surgery.py`:

```python
from llm_surgeon.surgery import get_layer_info


class TestGetLayerInfo:
    def test_returns_dict(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert isinstance(info, dict)

    def test_layer_count(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["num_layers"] == 8

    def test_hidden_size(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["hidden_size"] == 32

    def test_total_params_positive(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["total_params"] > 0

    def test_layer_params_list_length(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert len(info["layer_params"]) == 8

    def test_estimated_memory_positive(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["estimated_memory_gb"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestGetLayerInfo -v`
Expected: FAIL — `ImportError: cannot import name 'get_layer_info'`

- [ ] **Step 3: Implement get_layer_info**

Add to `testing/llm_surgeon/surgery.py`:

```python
from typing import Dict, Any


def get_layer_info(model) -> Dict[str, Any]:
    """Print and return summary of model layer structure."""
    layers = model.model.layers
    total_params = sum(p.numel() for p in model.parameters())
    est_memory_gb = total_params * 2 / 1e9

    layer_params = []
    for i, layer in enumerate(layers):
        lp = sum(p.numel() for p in layer.parameters())
        layer_params.append(lp)
        print(f"  Layer {i:2d}: {lp:,} params")

    print(f"\nModel: {model.config.model_type}")
    print(f"Layers: {len(layers)}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated memory (fp16): {est_memory_gb:.2f} GB")

    return {
        "num_layers": len(layers),
        "hidden_size": model.config.hidden_size,
        "total_params": total_params,
        "estimated_memory_gb": est_memory_gb,
        "layer_params": layer_params,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestGetLayerInfo -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add get_layer_info function"
```

---

### Task 5: remove_layers

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for remove_layers**

Add to `testing/tests/test_surgery.py`:

```python
from llm_surgeon.surgery import remove_layers


class TestRemoveLayers:
    def test_removes_single_layer(self, tiny_llama):
        log = remove_layers(tiny_llama, [3])
        assert len(tiny_llama.model.layers) == 7
        assert tiny_llama.config.num_hidden_layers == 7

    def test_removes_multiple_layers(self, tiny_llama):
        log = remove_layers(tiny_llama, [2, 4, 6])
        assert len(tiny_llama.model.layers) == 5
        assert tiny_llama.config.num_hidden_layers == 5

    def test_returns_surgery_log(self, tiny_llama):
        log = remove_layers(tiny_llama, [0])
        assert len(log.ops) == 1
        assert log.ops[0].operation == "remove_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 7

    def test_preserves_remaining_layers(self, tiny_llama):
        # Capture weight of layer 0 before surgery
        weight_before = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        remove_layers(tiny_llama, [7])  # remove last layer
        weight_after = tiny_llama.model.layers[0].self_attn.q_proj.weight.data
        assert torch.equal(weight_before, weight_after)

    def test_invalid_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            remove_layers(tiny_llama, [99])

    def test_negative_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            remove_layers(tiny_llama, [-1])

    def test_model_still_runs_after_surgery(self, tiny_llama):
        remove_layers(tiny_llama, [3, 4, 5])
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
```

Add `import torch` and `import pytest` at the top of the file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestRemoveLayers -v`
Expected: FAIL — `ImportError: cannot import name 'remove_layers'`

- [ ] **Step 3: Implement remove_layers**

Add to `testing/llm_surgeon/surgery.py`:

```python
def remove_layers(model, layer_indices: List[int]) -> SurgeryLog:
    """Remove layers at the specified indices. Indices are current positions."""
    layers = model.model.layers
    num_before = len(layers)

    for idx in layer_indices:
        if idx < 0 or idx >= num_before:
            raise IndexError(
                f"Layer index {idx} out of range [0, {num_before - 1}]"
            )

    # Remove in reverse sorted order to keep indices stable
    for idx in sorted(layer_indices, reverse=True):
        del layers[idx]

    model.config.num_hidden_layers = len(layers)

    log = SurgeryLog()
    log.add("remove_layers", f"Removed layers {layer_indices}", num_before, len(layers))
    return log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestRemoveLayers -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add remove_layers operation"
```

---

### Task 6: keep_layers

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for keep_layers**

Add to `testing/tests/test_surgery.py`:

```python
from llm_surgeon.surgery import keep_layers


class TestKeepLayers:
    def test_keeps_specified_layers(self, tiny_llama):
        log = keep_layers(tiny_llama, [0, 1, 2])
        assert len(tiny_llama.model.layers) == 3
        assert tiny_llama.config.num_hidden_layers == 3

    def test_returns_surgery_log(self, tiny_llama):
        log = keep_layers(tiny_llama, [0, 1])
        assert log.ops[0].operation == "keep_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 2

    def test_preserves_correct_layers(self, tiny_llama):
        # Capture weights of layers 0 and 7
        w0 = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        w7 = tiny_llama.model.layers[7].self_attn.q_proj.weight.data.clone()
        keep_layers(tiny_llama, [0, 7])
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w0)
        assert torch.equal(tiny_llama.model.layers[1].self_attn.q_proj.weight.data, w7)

    def test_invalid_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            keep_layers(tiny_llama, [0, 99])

    def test_model_still_runs(self, tiny_llama):
        keep_layers(tiny_llama, [0, 3, 7])
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestKeepLayers -v`
Expected: FAIL — `ImportError: cannot import name 'keep_layers'`

- [ ] **Step 3: Implement keep_layers**

Add to `testing/llm_surgeon/surgery.py`:

```python
import torch.nn as nn


def keep_layers(model, layer_indices: List[int]) -> SurgeryLog:
    """Keep only the layers at the specified indices, remove all others."""
    layers = model.model.layers
    num_before = len(layers)

    for idx in layer_indices:
        if idx < 0 or idx >= num_before:
            raise IndexError(
                f"Layer index {idx} out of range [0, {num_before - 1}]"
            )

    new_layers = nn.ModuleList([layers[i] for i in layer_indices])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)

    log = SurgeryLog()
    log.add("keep_layers", f"Kept layers {layer_indices}", num_before, len(new_layers))
    return log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestKeepLayers -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add keep_layers operation"
```

---

### Task 7: reorder_layers

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for reorder_layers**

Add to `testing/tests/test_surgery.py`:

```python
from llm_surgeon.surgery import reorder_layers


class TestReorderLayers:
    def test_reverses_layers(self, tiny_llama):
        w_first = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        w_last = tiny_llama.model.layers[7].self_attn.q_proj.weight.data.clone()
        log = reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        # First layer should now have what was last
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w_last)
        assert torch.equal(tiny_llama.model.layers[7].self_attn.q_proj.weight.data, w_first)

    def test_layer_count_unchanged(self, tiny_llama):
        reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        assert len(tiny_llama.model.layers) == 8
        assert tiny_llama.config.num_hidden_layers == 8

    def test_returns_surgery_log(self, tiny_llama):
        log = reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        assert log.ops[0].operation == "reorder_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 8

    def test_wrong_length_raises(self, tiny_llama):
        with pytest.raises(ValueError, match="must match layer count"):
            reorder_layers(tiny_llama, [0, 1, 2])

    def test_not_permutation_raises(self, tiny_llama):
        with pytest.raises(ValueError, match="must be a permutation"):
            reorder_layers(tiny_llama, [0, 0, 0, 0, 0, 0, 0, 0])

    def test_model_still_runs(self, tiny_llama):
        reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestReorderLayers -v`
Expected: FAIL — `ImportError: cannot import name 'reorder_layers'`

- [ ] **Step 3: Implement reorder_layers**

Add to `testing/llm_surgeon/surgery.py`:

```python
def reorder_layers(model, new_order: List[int]) -> SurgeryLog:
    """Rearrange layers to the specified order. new_order must be a permutation."""
    layers = model.model.layers
    num_before = len(layers)

    if len(new_order) != num_before:
        raise ValueError(
            f"new_order length ({len(new_order)}) must match layer count ({num_before})"
        )
    if set(new_order) != set(range(num_before)):
        raise ValueError(
            f"new_order must be a permutation of [0, {num_before - 1}]"
        )

    new_layers = nn.ModuleList([layers[i] for i in new_order])
    model.model.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)

    log = SurgeryLog()
    log.add("reorder_layers", f"Reordered to {new_order}", num_before, len(new_layers))
    return log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestReorderLayers -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add reorder_layers operation"
```

---

### Task 8: swap_layers

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for swap_layers**

Add to `testing/tests/test_surgery.py`:

```python
from llm_surgeon.surgery import swap_layers


class TestSwapLayers:
    def test_swaps_weights(self, tiny_llama):
        w0 = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        w5 = tiny_llama.model.layers[5].self_attn.q_proj.weight.data.clone()
        swap_layers(tiny_llama, 0, 5)
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w5)
        assert torch.equal(tiny_llama.model.layers[5].self_attn.q_proj.weight.data, w0)

    def test_layer_count_unchanged(self, tiny_llama):
        swap_layers(tiny_llama, 0, 7)
        assert len(tiny_llama.model.layers) == 8
        assert tiny_llama.config.num_hidden_layers == 8

    def test_returns_surgery_log(self, tiny_llama):
        log = swap_layers(tiny_llama, 2, 6)
        assert log.ops[0].operation == "swap_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 8

    def test_invalid_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            swap_layers(tiny_llama, 0, 99)

    def test_swap_same_index(self, tiny_llama):
        w0 = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        swap_layers(tiny_llama, 0, 0)
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w0)

    def test_model_still_runs(self, tiny_llama):
        swap_layers(tiny_llama, 1, 6)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestSwapLayers -v`
Expected: FAIL — `ImportError: cannot import name 'swap_layers'`

- [ ] **Step 3: Implement swap_layers**

Add to `testing/llm_surgeon/surgery.py`:

```python
def swap_layers(model, i: int, j: int) -> SurgeryLog:
    """Swap two layers' positions."""
    layers = model.model.layers
    num_before = len(layers)

    for idx in (i, j):
        if idx < 0 or idx >= num_before:
            raise IndexError(
                f"Layer index {idx} out of range [0, {num_before - 1}]"
            )

    layers[i], layers[j] = layers[j], layers[i]

    log = SurgeryLog()
    log.add("swap_layers", f"Swapped layers {i} and {j}", num_before, len(layers))
    return log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestSwapLayers -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add swap_layers operation"
```

---

### Task 9: duplicate_layer

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for duplicate_layer**

Add to `testing/tests/test_surgery.py`:

```python
from llm_surgeon.surgery import duplicate_layer


class TestDuplicateLayer:
    def test_increases_layer_count(self, tiny_llama):
        log = duplicate_layer(tiny_llama, src=3, dst=4)
        assert len(tiny_llama.model.layers) == 9
        assert tiny_llama.config.num_hidden_layers == 9

    def test_duplicate_has_same_weights(self, tiny_llama):
        w_src = tiny_llama.model.layers[3].self_attn.q_proj.weight.data.clone()
        duplicate_layer(tiny_llama, src=3, dst=4)
        w_dup = tiny_llama.model.layers[4].self_attn.q_proj.weight.data
        assert torch.equal(w_src, w_dup)

    def test_duplicate_is_deep_copy(self, tiny_llama):
        duplicate_layer(tiny_llama, src=3, dst=4)
        # Modifying the duplicate should not affect the original
        tiny_llama.model.layers[4].self_attn.q_proj.weight.data.zero_()
        assert not torch.equal(
            tiny_llama.model.layers[3].self_attn.q_proj.weight.data,
            tiny_llama.model.layers[4].self_attn.q_proj.weight.data,
        )

    def test_returns_surgery_log(self, tiny_llama):
        log = duplicate_layer(tiny_llama, src=0, dst=1)
        assert log.ops[0].operation == "duplicate_layer"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 9

    def test_invalid_src_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            duplicate_layer(tiny_llama, src=99, dst=0)

    def test_invalid_dst_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            duplicate_layer(tiny_llama, src=0, dst=99)

    def test_dst_at_end_allowed(self, tiny_llama):
        duplicate_layer(tiny_llama, src=0, dst=8)  # append at end
        assert len(tiny_llama.model.layers) == 9

    def test_model_still_runs(self, tiny_llama):
        duplicate_layer(tiny_llama, src=3, dst=4)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestDuplicateLayer -v`
Expected: FAIL — `ImportError: cannot import name 'duplicate_layer'`

- [ ] **Step 3: Implement duplicate_layer**

Add to `testing/llm_surgeon/surgery.py`, with `import copy` and `import warnings` at the top of the file:

```python
import copy
import warnings


def duplicate_layer(model, src: int, dst: int) -> SurgeryLog:
    """Deep-copy a layer and insert it at the destination position."""
    layers = model.model.layers
    num_before = len(layers)

    if src < 0 or src >= num_before:
        raise IndexError(
            f"Source index {src} out of range [0, {num_before - 1}]"
        )
    if dst < 0 or dst > num_before:
        raise IndexError(
            f"Destination index {dst} out of range [0, {num_before}]"
        )

    # Memory warning for large models
    total_params = sum(p.numel() for p in model.parameters())
    est_gb = total_params * 2 / 1e9
    if est_gb > 28:
        warnings.warn(
            f"Model is ~{est_gb:.1f} GB in fp16, approaching 32 GB RAM limit. "
            f"Duplicating a layer will increase this.",
            ResourceWarning,
        )

    new_layer = copy.deepcopy(layers[src])
    layers.insert(dst, new_layer)
    model.config.num_hidden_layers = len(layers)

    log = SurgeryLog()
    log.add(
        "duplicate_layer",
        f"Duplicated layer {src} -> position {dst}",
        num_before,
        len(layers),
    )
    return log
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestDuplicateLayer -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add duplicate_layer operation with memory warning"
```

---

### Task 10: check_structure (verify.py)

**Files:**
- Modify: `testing/llm_surgeon/verify.py`
- Create: `testing/tests/test_verify.py`

- [ ] **Step 1: Write tests for VerifyReport and check_structure**

Create `testing/tests/test_verify.py`:

```python
"""Tests for verify module."""

import pytest
import torch
from llm_surgeon.verify import VerifyReport, check_structure
from llm_surgeon.surgery import (
    SurgeryLog,
    remove_layers,
    keep_layers,
    swap_layers,
    duplicate_layer,
)


class TestVerifyReport:
    def test_starts_passed(self):
        report = VerifyReport()
        assert report.passed is True

    def test_add_passing_check(self):
        report = VerifyReport()
        report.add_check("test_check", True, "all good")
        assert report.passed is True
        assert len(report.checks) == 1

    def test_add_failing_check_sets_failed(self):
        report = VerifyReport()
        report.add_check("test_check", False, "mismatch")
        assert report.passed is False

    def test_str_shows_status(self):
        report = VerifyReport()
        report.add_check("check1", True, "ok")
        s = str(report)
        assert "PASSED" in s

    def test_str_shows_failed(self):
        report = VerifyReport()
        report.add_check("check1", False, "bad")
        s = str(report)
        assert "FAILED" in s


class TestCheckStructure:
    def test_passes_on_unmodified_model(self, tiny_llama):
        report = check_structure(tiny_llama)
        assert report.passed is True

    def test_passes_after_remove_layers(self, tiny_llama):
        log = remove_layers(tiny_llama, [3, 4, 5])
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_passes_after_keep_layers(self, tiny_llama):
        log = keep_layers(tiny_llama, [0, 1, 7])
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_passes_after_swap(self, tiny_llama):
        log = swap_layers(tiny_llama, 0, 7)
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_passes_after_duplicate(self, tiny_llama):
        log = duplicate_layer(tiny_llama, src=0, dst=1)
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_catches_config_mismatch(self, tiny_llama):
        remove_layers(tiny_llama, [0])
        # Deliberately break config
        tiny_llama.config.num_hidden_layers = 999
        with pytest.raises(ValueError, match="Structural verification failed"):
            check_structure(tiny_llama)

    def test_catches_surgery_log_mismatch(self, tiny_llama):
        # Create a log that says we removed 3 layers, but only remove 1
        remove_layers(tiny_llama, [0])
        fake_log = SurgeryLog()
        fake_log.add("remove_layers", "Removed 3 layers", 8, 5)
        with pytest.raises(ValueError, match="Structural verification failed"):
            check_structure(tiny_llama, fake_log)

    def test_no_surgery_log_still_validates(self, tiny_llama):
        remove_layers(tiny_llama, [0, 1])
        report = check_structure(tiny_llama)  # no log passed
        assert report.passed is True

    def test_checks_embedding_consistency(self, tiny_llama):
        report = check_structure(tiny_llama)
        check_names = [c["name"] for c in report.checks]
        assert "embedding_dim_consistent" in check_names
        assert "lm_head_vocab_consistent" in check_names
        assert "lm_head_hidden_consistent" in check_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_verify.py -v`
Expected: FAIL — `ImportError: cannot import name 'VerifyReport'`

- [ ] **Step 3: Implement VerifyReport and check_structure**

Replace contents of `testing/llm_surgeon/verify.py`:

```python
"""Structural verification of modified models."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VerifyReport:
    """Result of structural verification checks."""
    passed: bool = True
    checks: List[dict] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"VerifyReport: {status}"]
        for check in self.checks:
            mark = "[pass]" if check["passed"] else "[FAIL]"
            lines.append(f"  {mark} {check['name']}: {check['detail']}")
        return "\n".join(lines)


def check_structure(model, surgery_log=None) -> VerifyReport:
    """Validate model structural integrity after surgery.

    Raises ValueError if any critical check fails.
    """
    report = VerifyReport()

    # Check 1: layer count matches config
    actual_layers = len(model.model.layers)
    config_layers = model.config.num_hidden_layers
    report.add_check(
        "layer_count_matches_config",
        actual_layers == config_layers,
        f"actual={actual_layers}, config={config_layers}",
    )

    # Check 2: embedding dimension matches hidden size
    embed_dim = model.model.embed_tokens.embedding_dim
    hidden_size = model.config.hidden_size
    report.add_check(
        "embedding_dim_consistent",
        embed_dim == hidden_size,
        f"embed_dim={embed_dim}, hidden_size={hidden_size}",
    )

    # Check 3: lm_head output matches vocab size
    lm_head_out = model.lm_head.out_features
    vocab_size = model.config.vocab_size
    report.add_check(
        "lm_head_vocab_consistent",
        lm_head_out == vocab_size,
        f"lm_head_out={lm_head_out}, vocab_size={vocab_size}",
    )

    # Check 4: lm_head input matches hidden size
    lm_head_in = model.lm_head.in_features
    report.add_check(
        "lm_head_hidden_consistent",
        lm_head_in == hidden_size,
        f"lm_head_in={lm_head_in}, hidden_size={hidden_size}",
    )

    # Check 5: cross-reference surgery log
    if surgery_log is not None:
        for op in surgery_log.ops:
            report.add_check(
                f"surgery_log_{op.operation}",
                actual_layers == op.layer_count_after,
                f"expected={op.layer_count_after} after {op.operation}, "
                f"actual={actual_layers}",
            )

    if not report.passed:
        raise ValueError(f"Structural verification failed:\n{report}")

    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_verify.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/ -v`
Expected: All tests PASS (surgery + verify).

- [ ] **Step 6: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/verify.py tests/test_verify.py && git commit -m "feat: add check_structure with VerifyReport"
```

---

### Task 11: load_model

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write tests for load_model**

Add to `testing/tests/test_surgery.py`:

```python
import os
from llm_surgeon.surgery import load_model


class TestLoadModel:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_model("nonexistent-model", mode="invalid")

    def test_valid_modes_accepted(self):
        """Test that valid mode strings don't raise ValueError.
        We don't actually load a model here — just test mode validation.
        """
        # This will fail with OSError (model not found), not ValueError
        for mode in ("inspect", "eval", "export"):
            with pytest.raises(OSError):
                load_model("nonexistent/model-id-that-does-not-exist", mode=mode)

    def test_returns_tuple(self, tiny_llama, tmp_path):
        """Test load_model with a local saved model."""
        # Save the tiny model to a temp dir
        save_path = str(tmp_path / "tiny_model")
        tiny_llama.save_pretrained(save_path)
        # Create a minimal tokenizer config so AutoTokenizer can load
        import json
        tokenizer_config = {
            "model_type": "llama",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
        with open(os.path.join(save_path, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)
        # Create minimal tokenizer.json
        vocab = {f"token_{i}": i for i in range(64)}
        tokenizer_data = {
            "version": "1.0",
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1, "content": "<s>", "special": True},
                {"id": 2, "content": "</s>", "special": True},
            ],
        }
        with open(os.path.join(save_path, "tokenizer.json"), "w") as f:
            json.dump(tokenizer_data, f)

        model, tokenizer = load_model(save_path, mode="export")
        assert model is not None
        assert tokenizer is not None
        assert len(model.model.layers) == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestLoadModel -v`
Expected: FAIL — `ImportError: cannot import name 'load_model'`

- [ ] **Step 3: Implement load_model**

Add to `testing/llm_surgeon/surgery.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple


def load_model(model_id: str, mode: str = "inspect") -> Tuple:
    """Load a model and tokenizer.

    Modes:
        inspect: 4-bit quantized on GPU (fast, for inspection)
        eval: fp16 with auto device map (for perplexity measurement)
        export: fp16 on CPU only (for clean checkpoint export)
    """
    if mode not in ("inspect", "eval", "export"):
        raise ValueError(
            f"Unknown mode: '{mode}'. Must be 'inspect', 'eval', or 'export'."
        )

    if mode == "inspect":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif mode == "eval":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif mode == "export":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestLoadModel -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add llm_surgeon/surgery.py tests/test_surgery.py && git commit -m "feat: add load_model with inspect/eval/export modes"
```

---

### Task 12: Chained Operations Test

**Files:**
- Modify: `testing/tests/test_surgery.py`
- Modify: `testing/tests/test_verify.py`

- [ ] **Step 1: Write integration tests for chaining multiple operations**

Add to `testing/tests/test_surgery.py`:

```python
class TestChainedOperations:
    def test_remove_then_swap(self, tiny_llama):
        remove_layers(tiny_llama, [6, 7])  # 8 -> 6 layers
        swap_layers(tiny_llama, 0, 5)  # swap first and last
        assert len(tiny_llama.model.layers) == 6
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_keep_then_reorder(self, tiny_llama):
        keep_layers(tiny_llama, [0, 2, 4, 6])  # 8 -> 4 layers
        reorder_layers(tiny_llama, [3, 2, 1, 0])  # reverse
        assert len(tiny_llama.model.layers) == 4
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_duplicate_then_remove(self, tiny_llama):
        duplicate_layer(tiny_llama, src=0, dst=1)  # 8 -> 9
        remove_layers(tiny_llama, [0])  # 9 -> 8 (removed original, kept copy)
        assert len(tiny_llama.model.layers) == 8
```

Add to `testing/tests/test_verify.py`:

```python
from llm_surgeon.surgery import reorder_layers


class TestCheckStructureChained:
    def test_verify_after_multiple_ops(self, tiny_llama):
        log1 = remove_layers(tiny_llama, [6, 7])
        log2 = swap_layers(tiny_llama, 0, 5)
        # Verify with last log only (layer count should match final state)
        report = check_structure(tiny_llama, log2)
        assert report.passed is True

    def test_verify_no_log_after_chain(self, tiny_llama):
        remove_layers(tiny_llama, [0, 1])
        swap_layers(tiny_llama, 0, 5)
        reorder_layers(tiny_llama, list(range(5, -1, -1)))
        report = check_structure(tiny_llama)
        assert report.passed is True
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add tests/test_surgery.py tests/test_verify.py && git commit -m "test: add chained operation and verification integration tests"
```

---

### Task 13: Save and Reload Checkpoint Test

**Files:**
- Modify: `testing/tests/test_surgery.py`

- [ ] **Step 1: Write test for save/reload roundtrip**

Add to `testing/tests/test_surgery.py`:

```python
from transformers import AutoModelForCausalLM


class TestSaveReload:
    def test_modified_model_saves_and_reloads(self, tiny_llama, tmp_path):
        # Remove layers
        remove_layers(tiny_llama, [3, 4, 5])
        assert len(tiny_llama.model.layers) == 5

        # Save
        save_path = str(tmp_path / "modified_model")
        tiny_llama.save_pretrained(save_path)

        # Reload
        reloaded = AutoModelForCausalLM.from_pretrained(save_path)
        assert len(reloaded.model.layers) == 5
        assert reloaded.config.num_hidden_layers == 5

    def test_reloaded_model_produces_output(self, tiny_llama, tmp_path):
        remove_layers(tiny_llama, [6, 7])
        save_path = str(tmp_path / "modified_model")
        tiny_llama.save_pretrained(save_path)

        reloaded = AutoModelForCausalLM.from_pretrained(save_path)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = reloaded(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_reloaded_weights_match(self, tiny_llama, tmp_path):
        w0_before = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        remove_layers(tiny_llama, [7])  # remove last layer only
        save_path = str(tmp_path / "modified_model")
        tiny_llama.save_pretrained(save_path)

        reloaded = AutoModelForCausalLM.from_pretrained(save_path)
        w0_after = reloaded.model.layers[0].self_attn.q_proj.weight.data
        assert torch.equal(w0_before, w0_after)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/test_surgery.py::TestSaveReload -v`
Expected: All 3 tests PASS.

- [ ] **Step 3: Run full test suite — final check**

Run: `cd /home/ai/ai-projects/llm/testing && python -m pytest tests/ -v`
Expected: ALL tests PASS. This is the Phase 1 completion gate.

- [ ] **Step 4: Commit**

```bash
cd /home/ai/ai-projects/llm/testing && git add tests/test_surgery.py && git commit -m "test: add save/reload roundtrip tests for modified models"
```

---

## Final State

After all tasks are complete, the project should have:

```
testing/
  llm_surgeon/
    __init__.py          — imports surgery, verify
    surgery.py           — SurgeryOp, SurgeryLog, load_model, get_layer_info,
                           remove_layers, keep_layers, reorder_layers,
                           swap_layers, duplicate_layer
    verify.py            — VerifyReport, check_structure
  tests/
    __init__.py
    conftest.py          — tiny_llama fixture
    test_surgery.py      — ~50 tests covering all operations + edge cases
    test_verify.py       — ~12 tests covering structural verification
  pyproject.toml
  requirements.txt
```

All tests pass. No external model downloads needed for tests. Ready for Phase 2 (export pipeline).
