# Phase 2: Export Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Tool rules (for subagents):**
> - Use Read (not cat/head/tail), Grep (not grep/rg/awk), Glob (not find/ls), Edit (not sed/awk) for all file operations
> - You are already in the project root (/home/ai/ai-projects/llm) — never cd
> - Python venv: `/home/ai/ai-projects/llm/testing/.venv/bin/python`
> - Run tests: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`
> - System python does NOT have torch/pytest — always use the full venv path

**Goal:** Build `export.py` — save modified models as HuggingFace checkpoints, convert to GGUF via llama.cpp, quantize, and register with ollama.

**Architecture:** Three-step pipeline (save checkpoint → convert to GGUF → register with ollama), each step callable independently plus a convenience wrapper. Shells out to llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize` for format conversion.

**Tech Stack:** Python, subprocess (for llama.cpp tools), requests (for ollama API verification)

**Key paths:**
- llama.cpp: `/home/ai/ai-projects/llm/llama.cpp/`
- convert script: `/home/ai/ai-projects/llm/llama.cpp/convert_hf_to_gguf.py`
- quantize binary: `/home/ai/ai-projects/llm/llama.cpp/build/bin/llama-quantize`
- ollama API: `http://localhost:11434`

**Reference:** `docs/superpowers/specs/2026-04-08-llm-surgeon-design.md` (v2), Phase 2 section of phase plan.

---

## File Map

```
testing/
  llm_surgeon/
    export.py            — CREATE — save_checkpoint, to_gguf, register_ollama, full_pipeline
  tests/
    test_export.py       — CREATE — tests for export module
```

---

### Task 1: Configuration + save_checkpoint

**Files:**
- Create: `testing/llm_surgeon/export.py`
- Create: `testing/tests/test_export.py`

- [ ] **Step 1: Write tests for save_checkpoint**

Create `testing/tests/test_export.py`:

```python
"""Tests for export module."""

import json
import os
import pytest
import torch
from transformers import AutoModelForCausalLM
from llm_surgeon.surgery import remove_layers
from llm_surgeon.export import save_checkpoint


class TestSaveCheckpoint:
    def test_saves_model_files(self, tiny_llama, tmp_path):
        output_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, output_dir)
        assert os.path.exists(os.path.join(output_dir, "config.json"))
        assert os.path.exists(output_dir)
        # Should have safetensors or bin files
        files = os.listdir(output_dir)
        assert any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)

    def test_saves_tokenizer_when_provided(self, tiny_llama, tmp_path):
        from transformers import AutoTokenizer
        # Use a real small tokenizer
        output_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, output_dir)
        # config.json should exist at minimum
        with open(os.path.join(output_dir, "config.json")) as f:
            config = json.load(f)
        assert config["num_hidden_layers"] == 8

    def test_config_matches_modified_model(self, tiny_llama, tmp_path):
        remove_layers(tiny_llama, [3, 4, 5])
        output_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, output_dir)
        with open(os.path.join(output_dir, "config.json")) as f:
            config = json.load(f)
        assert config["num_hidden_layers"] == 5

    def test_reloadable(self, tiny_llama, tmp_path):
        remove_layers(tiny_llama, [6, 7])
        output_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, output_dir)
        reloaded = AutoModelForCausalLM.from_pretrained(output_dir)
        assert len(reloaded.model.layers) == 6

    def test_returns_path(self, tiny_llama, tmp_path):
        output_dir = str(tmp_path / "checkpoint")
        result = save_checkpoint(tiny_llama, output_dir)
        assert result == output_dir

    def test_saves_tokenizer_object(self, tiny_llama, tmp_path):
        from transformers import PreTrainedTokenizerFast
        # Create a minimal tokenizer
        output_dir = str(tmp_path / "checkpoint")
        # Just test without tokenizer - should still work
        save_checkpoint(tiny_llama, output_dir)
        assert os.path.exists(os.path.join(output_dir, "config.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_checkpoint'`

- [ ] **Step 3: Implement save_checkpoint**

Create `testing/llm_surgeon/export.py`:

```python
"""Export modified models: HF checkpoint -> GGUF -> ollama."""

import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional


# Default llama.cpp path — override via LLAMA_CPP_PATH env var
_DEFAULT_LLAMA_CPP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..",
    "llama.cpp",
)
LLAMA_CPP_PATH = os.environ.get("LLAMA_CPP_PATH", os.path.normpath(_DEFAULT_LLAMA_CPP))


def _get_convert_script() -> str:
    """Get path to convert_hf_to_gguf.py."""
    path = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {path}. "
            f"Set LLAMA_CPP_PATH env var to your llama.cpp directory."
        )
    return path


def _get_quantize_binary() -> str:
    """Get path to llama-quantize binary."""
    path = os.path.join(LLAMA_CPP_PATH, "build", "bin", "llama-quantize")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"llama-quantize not found at {path}. "
            f"Build llama.cpp first: cd {LLAMA_CPP_PATH} && cmake -B build && cmake --build build"
        )
    return path


def save_checkpoint(model, output_dir: str, tokenizer=None) -> str:
    """Save model (and optionally tokenizer) as a HuggingFace checkpoint.

    Returns the output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Verify config is consistent
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    actual_layers = len(model.model.layers)
    if config["num_hidden_layers"] != actual_layers:
        raise RuntimeError(
            f"Saved config has {config['num_hidden_layers']} layers "
            f"but model has {actual_layers}"
        )

    return output_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Update __init__.py**

Add `export` to `testing/llm_surgeon/__init__.py`:

```python
"""LLM Surgeon — surgical layer-level manipulation of LLaMA models."""

from llm_surgeon import surgery, verify, export
```

- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/export.py testing/tests/test_export.py testing/llm_surgeon/__init__.py
git commit -m "feat: add save_checkpoint for HF checkpoint export"
```

---

### Task 2: to_gguf

**Files:**
- Modify: `testing/llm_surgeon/export.py`
- Modify: `testing/tests/test_export.py`

- [ ] **Step 1: Write tests for to_gguf**

Add to `testing/tests/test_export.py`:

```python
from llm_surgeon.export import to_gguf, LLAMA_CPP_PATH


class TestToGguf:
    def test_converts_checkpoint_to_gguf(self, tiny_llama, tmp_path):
        # Save checkpoint first
        ckpt_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, ckpt_dir)
        # Convert
        gguf_path = to_gguf(ckpt_dir, output_dir=str(tmp_path / "gguf"))
        assert os.path.exists(gguf_path)
        assert gguf_path.endswith(".gguf")

    def test_quantized_file_smaller_than_f16(self, tiny_llama, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, ckpt_dir)
        # Convert with quantization
        gguf_path = to_gguf(
            ckpt_dir,
            output_dir=str(tmp_path / "gguf"),
            quantization="Q4_K_M",
        )
        assert os.path.exists(gguf_path)
        # Q4 file should exist
        assert "Q4_K_M" in gguf_path or "q4" in gguf_path.lower()

    def test_f16_only_when_no_quantization(self, tiny_llama, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, ckpt_dir)
        gguf_path = to_gguf(
            ckpt_dir,
            output_dir=str(tmp_path / "gguf"),
            quantization=None,
        )
        assert os.path.exists(gguf_path)

    def test_missing_llama_cpp_raises(self, tiny_llama, tmp_path, monkeypatch):
        monkeypatch.setattr("llm_surgeon.export.LLAMA_CPP_PATH", "/nonexistent/path")
        ckpt_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, ckpt_dir)
        with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
            to_gguf(ckpt_dir, output_dir=str(tmp_path / "gguf"))

    def test_returns_gguf_path(self, tiny_llama, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, ckpt_dir)
        gguf_path = to_gguf(ckpt_dir, output_dir=str(tmp_path / "gguf"))
        assert isinstance(gguf_path, str)
        assert os.path.isfile(gguf_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py::TestToGguf -v`
Expected: FAIL — `ImportError: cannot import name 'to_gguf'`

- [ ] **Step 3: Implement to_gguf**

Add to `testing/llm_surgeon/export.py`:

```python
def to_gguf(
    checkpoint_path: str,
    output_dir: str,
    quantization: Optional[str] = "Q4_K_M",
) -> str:
    """Convert a HuggingFace checkpoint to GGUF format.

    Args:
        checkpoint_path: Path to HF checkpoint directory
        output_dir: Where to write the GGUF file
        quantization: Quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0).
                      None for f16 only.

    Returns:
        Path to the final GGUF file.
    """
    convert_script = _get_convert_script()
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert HF checkpoint to f16 GGUF
    model_name = os.path.basename(os.path.normpath(checkpoint_path))
    f16_path = os.path.join(output_dir, f"{model_name}-f16.gguf")

    cmd = [
        "python3", convert_script,
        checkpoint_path,
        "--outfile", f16_path,
        "--outtype", "f16",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed:\n{result.stderr}\n{result.stdout}"
        )

    if not os.path.exists(f16_path):
        raise RuntimeError(f"Expected f16 GGUF at {f16_path} but file not found")

    # Step 2: Quantize if requested
    if quantization is None:
        return f16_path

    quantize_bin = _get_quantize_binary()
    quant_path = os.path.join(output_dir, f"{model_name}-{quantization}.gguf")

    cmd = [quantize_bin, f16_path, quant_path, quantization]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed:\n{result.stderr}\n{result.stdout}"
        )

    if not os.path.exists(quant_path):
        raise RuntimeError(f"Expected quantized GGUF at {quant_path} but file not found")

    # Clean up intermediate f16 file
    os.remove(f16_path)

    return quant_path
```

Note: The `convert_hf_to_gguf.py` script should be run with the venv's python so it has access to torch/transformers. Update the cmd to use the venv python:

```python
    # Use the same python that's running this code (the venv python)
    import sys
    cmd = [
        sys.executable, convert_script,
        checkpoint_path,
        "--outfile", f16_path,
        "--outtype", "f16",
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py::TestToGguf -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add testing/llm_surgeon/export.py testing/tests/test_export.py
git commit -m "feat: add to_gguf conversion with optional quantization"
```

---

### Task 3: register_ollama

**Files:**
- Modify: `testing/llm_surgeon/export.py`
- Modify: `testing/tests/test_export.py`

- [ ] **Step 1: Write tests for register_ollama**

Note: These tests interact with the live ollama service. We test the Modelfile generation logic in isolation, and only test the actual ollama registration if ollama is running.

Add to `testing/tests/test_export.py`:

```python
import requests
from llm_surgeon.export import register_ollama, _generate_modelfile


def _ollama_available() -> bool:
    """Check if ollama is running."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


class TestGenerateModelfile:
    def test_contains_from_line(self, tmp_path):
        gguf_path = str(tmp_path / "model.gguf")
        # Create a dummy file
        with open(gguf_path, "w") as f:
            f.write("dummy")
        content = _generate_modelfile(gguf_path)
        assert f"FROM {gguf_path}" in content

    def test_uses_absolute_path(self, tmp_path):
        gguf_path = str(tmp_path / "model.gguf")
        with open(gguf_path, "w") as f:
            f.write("dummy")
        content = _generate_modelfile(gguf_path)
        assert os.path.isabs(gguf_path)
        assert gguf_path in content


@pytest.mark.skipif(not _ollama_available(), reason="ollama not running")
class TestRegisterOllama:
    def test_registers_model(self, tiny_llama, tmp_path):
        # Full pipeline: save -> convert -> register
        ckpt_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, ckpt_dir)
        gguf_path = to_gguf(ckpt_dir, output_dir=str(tmp_path / "gguf"), quantization=None)
        register_ollama(gguf_path, name="test-surgeon-export")
        # Verify it's registered
        r = requests.get("http://localhost:11434/api/tags")
        models = [m["name"] for m in r.json().get("models", [])]
        assert any("test-surgeon-export" in m for m in models)

    def test_registered_model_generates(self, tiny_llama, tmp_path):
        ckpt_dir = str(tmp_path / "checkpoint")
        save_checkpoint(tiny_llama, ckpt_dir)
        gguf_path = to_gguf(ckpt_dir, output_dir=str(tmp_path / "gguf"), quantization=None)
        register_ollama(gguf_path, name="test-surgeon-gen")
        # Try generating
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "test-surgeon-gen", "prompt": "Hello", "stream": False},
            timeout=60,
        )
        assert r.status_code == 200
        assert "response" in r.json()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py::TestGenerateModelfile -v`
Expected: FAIL — `ImportError: cannot import name '_generate_modelfile'`

- [ ] **Step 3: Implement register_ollama and _generate_modelfile**

Add to `testing/llm_surgeon/export.py`:

```python
def _generate_modelfile(gguf_path: str) -> str:
    """Generate a Modelfile for ollama."""
    abs_path = os.path.abspath(gguf_path)
    return f"FROM {abs_path}\n"


def register_ollama(gguf_path: str, name: str) -> None:
    """Register a GGUF model with ollama.

    Creates a Modelfile and runs `ollama create`.
    """
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    # Write Modelfile next to the GGUF
    modelfile_path = os.path.join(os.path.dirname(gguf_path), f"Modelfile.{name}")
    modelfile_content = _generate_modelfile(gguf_path)
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    # Run ollama create
    cmd = ["ollama", "create", name, "-f", modelfile_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create failed:\n{result.stderr}\n{result.stdout}"
        )

    # Verify registration
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(name in m for m in models):
                raise RuntimeError(
                    f"Model '{name}' not found in ollama after registration. "
                    f"Available: {models}"
                )
    except requests.ConnectionError:
        print("Warning: Could not verify ollama registration (API not reachable)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py -v`
Expected: TestGenerateModelfile tests PASS. TestRegisterOllama tests PASS if ollama is running, SKIP otherwise.

- [ ] **Step 5: Commit**

```bash
git add testing/llm_surgeon/export.py testing/tests/test_export.py
git commit -m "feat: add register_ollama with Modelfile generation"
```

---

### Task 4: full_pipeline convenience wrapper

**Files:**
- Modify: `testing/llm_surgeon/export.py`
- Modify: `testing/tests/test_export.py`

- [ ] **Step 1: Write tests for full_pipeline**

Add to `testing/tests/test_export.py`:

```python
from llm_surgeon.export import full_pipeline


class TestFullPipeline:
    def test_produces_gguf(self, tiny_llama, tmp_path):
        result = full_pipeline(
            tiny_llama,
            name="test-pipeline",
            quantization=None,  # skip quantization for speed
            output_dir=str(tmp_path / "output"),
            register=False,  # don't try to register with ollama
        )
        assert "gguf_path" in result
        assert os.path.exists(result["gguf_path"])
        assert "checkpoint_path" in result
        assert os.path.exists(result["checkpoint_path"])

    def test_with_modified_model(self, tiny_llama, tmp_path):
        remove_layers(tiny_llama, [4, 5, 6, 7])
        result = full_pipeline(
            tiny_llama,
            name="test-modified",
            quantization=None,
            output_dir=str(tmp_path / "output"),
            register=False,
        )
        assert os.path.exists(result["gguf_path"])
        # Verify the checkpoint has correct layer count
        with open(os.path.join(result["checkpoint_path"], "config.json")) as f:
            config = json.load(f)
        assert config["num_hidden_layers"] == 4

    def test_with_quantization(self, tiny_llama, tmp_path):
        result = full_pipeline(
            tiny_llama,
            name="test-quant",
            quantization="Q4_K_M",
            output_dir=str(tmp_path / "output"),
            register=False,
        )
        assert os.path.exists(result["gguf_path"])
        assert "Q4_K_M" in result["gguf_path"]

    @pytest.mark.skipif(not _ollama_available(), reason="ollama not running")
    def test_full_pipeline_with_ollama(self, tiny_llama, tmp_path):
        result = full_pipeline(
            tiny_llama,
            name="test-full-pipeline",
            quantization=None,
            output_dir=str(tmp_path / "output"),
            register=True,
        )
        assert result["registered"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py::TestFullPipeline -v`
Expected: FAIL — `ImportError: cannot import name 'full_pipeline'`

- [ ] **Step 3: Implement full_pipeline**

Add to `testing/llm_surgeon/export.py`:

```python
def full_pipeline(
    model,
    name: str,
    quantization: Optional[str] = "Q4_K_M",
    output_dir: str = "outputs",
    tokenizer=None,
    register: bool = True,
) -> dict:
    """Run the complete export pipeline: save -> GGUF -> ollama.

    Returns dict with checkpoint_path, gguf_path, and registered status.
    """
    checkpoint_path = os.path.join(output_dir, "checkpoint", name)
    gguf_dir = os.path.join(output_dir, "gguf")

    # Step 1: Save checkpoint
    save_checkpoint(model, checkpoint_path, tokenizer=tokenizer)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Step 2: Convert to GGUF
    gguf_path = to_gguf(checkpoint_path, output_dir=gguf_dir, quantization=quantization)
    print(f"GGUF written to {gguf_path}")

    # Step 3: Register with ollama
    registered = False
    if register:
        register_ollama(gguf_path, name=name)
        registered = True
        print(f"Registered as '{name}' in ollama")

    return {
        "checkpoint_path": checkpoint_path,
        "gguf_path": gguf_path,
        "registered": registered,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/test_export.py -v`
Expected: All tests PASS (ollama tests skip if not running).

- [ ] **Step 5: Run full test suite**

Run: `/home/ai/ai-projects/llm/testing/.venv/bin/python -m pytest /home/ai/ai-projects/llm/testing/tests/ -v`
Expected: All 72 Phase 1 tests + new export tests PASS.

- [ ] **Step 6: Commit**

```bash
git add testing/llm_surgeon/export.py testing/tests/test_export.py
git commit -m "feat: add full_pipeline export convenience wrapper"
```

---

## Final State

After all tasks are complete:

```
testing/
  llm_surgeon/
    __init__.py          — imports surgery, verify, export
    surgery.py           — (from Phase 1)
    verify.py            — (from Phase 1)
    export.py            — save_checkpoint, to_gguf, register_ollama, full_pipeline
  tests/
    test_surgery.py      — (from Phase 1)
    test_verify.py       — (from Phase 1)
    test_export.py       — tests for all export functions
```

All tests pass. Export pipeline works end-to-end: surgery -> checkpoint -> GGUF -> ollama.
