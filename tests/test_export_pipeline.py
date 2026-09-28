"""Regression tests for export.py paths that need no llama.cpp or Ollama.

llama.cpp's converter and quantizer are replaced by a fake ``subprocess.run``
that records each command and writes the output file it names; the Ollama
API is replaced by a fake ``requests.get``.
"""

import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_surgeon import export
from llm_surgeon.export import (
    to_gguf,
)


@pytest.fixture
def llama_cpp_dir(tmp_path, monkeypatch) -> Path:
    """A directory that passes to_gguf's llama.cpp tool-existence checks."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").touch()
    (root / "build" / "bin" / "llama-quantize").touch()
    monkeypatch.setenv("LLAMA_CPP_PATH", str(root))
    return root


class FakeRun:
    """Stand-in for subprocess.run: records calls, writes the named outputs."""

    def __init__(self, quantize_rc: int = 0):
        self.calls: list[tuple[list[str], dict]] = []
        self.quantize_rc = quantize_rc

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if "--outfile" in cmd:  # convert_hf_to_gguf.py
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF-f16")
            return MagicMock(returncode=0, stdout="", stderr="")
        # llama-quantize <in> <out> <type>
        if self.quantize_rc == 0:
            Path(cmd[2]).write_bytes(b"GGUF-quant")
        return MagicMock(returncode=self.quantize_rc, stdout="", stderr="boom")

    @property
    def quantize_calls(self):
        return [(c, kw) for c, kw in self.calls if "--outfile" not in c]


# ---------------------------------------------------------------------------
# to_gguf
# ---------------------------------------------------------------------------


class TestToGgufQuantization:
    def test_f16_quantization_skips_quantize_and_keeps_file(
        self, llama_cpp_dir, tmp_path
    ):
        run = FakeRun()
        with patch("subprocess.run", run):
            result = to_gguf(
                str(tmp_path / "ckpt"), str(tmp_path / "out"), quantization="F16"
            )
        assert run.quantize_calls == []
        assert os.path.exists(result)
        assert result.endswith("ckpt-F16.gguf")

    def test_quantized_output_named_and_intermediate_removed(
        self, llama_cpp_dir, tmp_path
    ):
        run = FakeRun()
        out_dir = tmp_path / "out"
        with patch("subprocess.run", run):
            result = to_gguf(
                str(tmp_path / "ckpt"), str(out_dir), quantization="Q4_K_M"
            )
        assert result.endswith("ckpt-Q4_K_M.gguf")
        assert os.path.exists(result)
        assert sorted(p.name for p in out_dir.iterdir()) == ["ckpt-Q4_K_M.gguf"]

    def test_intermediate_removed_when_quantize_fails(self, llama_cpp_dir, tmp_path):
        out_dir = tmp_path / "out"
        with patch("subprocess.run", FakeRun(quantize_rc=1)):
            with pytest.raises(RuntimeError, match="quantization failed"):
                to_gguf(str(tmp_path / "ckpt"), str(out_dir), quantization="Q4_K_M")
        assert list(out_dir.iterdir()) == []

    def test_ld_library_path_has_no_empty_entry(
        self, llama_cpp_dir, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        run = FakeRun()
        with patch("subprocess.run", run):
            to_gguf(
                str(tmp_path / "ckpt"), str(tmp_path / "out"), quantization="Q4_K_M"
            )
        [(_cmd, kwargs)] = run.quantize_calls
        assert kwargs["env"]["LD_LIBRARY_PATH"] == str(llama_cpp_dir / "build" / "bin")

    def test_ld_library_path_prepends_to_existing(
        self, llama_cpp_dir, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")
        run = FakeRun()
        with patch("subprocess.run", run):
            to_gguf(
                str(tmp_path / "ckpt"), str(tmp_path / "out"), quantization="Q4_K_M"
            )
        [(_cmd, kwargs)] = run.quantize_calls
        bin_dir = str(llama_cpp_dir / "build" / "bin")
        assert kwargs["env"]["LD_LIBRARY_PATH"] == f"{bin_dir}:/opt/lib"



# ---------------------------------------------------------------------------
# Ollama registration and verification
# ---------------------------------------------------------------------------


def _fake_tags(*names: str):
    """A requests.get stand-in answering /api/tags with ``names``."""
    seen: list[str] = []

    def get(url, timeout=None):
        del timeout
        seen.append(url)
        models = [{"name": n, "model": n} for n in names]
        return types.SimpleNamespace(status_code=200, json=lambda: {"models": models})

    return get, seen








# ---------------------------------------------------------------------------
# full_pipeline and save_checkpoint
# ---------------------------------------------------------------------------




