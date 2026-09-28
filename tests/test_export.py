"""Tests for export module — Phase 2."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoModelForCausalLM

from llm_surgeon.export import (
    _generate_modelfile,
    full_pipeline,
    register_ollama,
    save_checkpoint,
    to_gguf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llama_cpp_tool(*parts: str) -> bool:
    """True if $LLAMA_CPP_PATH is set and contains the given file."""
    root = os.environ.get("LLAMA_CPP_PATH")
    if not root:
        return False
    return os.path.exists(os.path.join(root, *parts))


requires_llama_cpp = pytest.mark.skipif(
    not _llama_cpp_tool("convert_hf_to_gguf.py"),
    reason="needs LLAMA_CPP_PATH pointing at a llama.cpp checkout",
)
requires_llama_quantize = pytest.mark.skipif(
    not (_llama_cpp_tool("convert_hf_to_gguf.py")
         and _llama_cpp_tool("build", "bin", "llama-quantize")),
    reason="needs LLAMA_CPP_PATH pointing at a built llama.cpp (build/bin/llama-quantize)",
)


def _ollama_available() -> bool:
    try:
        import requests

        from llm_surgeon.export import _ollama_base_url

        r = requests.get(f"{_ollama_base_url()}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture
def ollama_models():
    """Skip unless Ollama is up; `ollama rm` every name appended on teardown.

    Probing inside a fixture keeps collection free of network calls.
    """
    if not _ollama_available():
        pytest.skip("ollama not running")
    names: list[str] = []
    yield names
    for name in names:
        subprocess.run(["ollama", "rm", name], capture_output=True, text=True)


class _CopyTokenizer:
    """Tokenizer stand-in whose save_pretrained copies tiny_checkpoint's SPM files."""

    def __init__(self, src: str):
        self.src = src

    def save_pretrained(self, out_dir: str) -> None:
        for fname in ("tokenizer.model", "tokenizer_config.json"):
            shutil.copy(os.path.join(self.src, fname), out_dir)


# ---------------------------------------------------------------------------
# Task 1: save_checkpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_returns_output_dir(self, tiny_llama, tmp_path):
        out = save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))
        assert out == str(tmp_path / "ckpt")

    def test_creates_config_json(self, tiny_llama, tmp_path):
        out = save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))
        assert os.path.exists(os.path.join(out, "config.json"))

    def test_creates_model_weights(self, tiny_llama, tmp_path):
        out = save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))
        weight_files = (
            list(Path(out).glob("*.bin"))
            + list(Path(out).glob("*.safetensors"))
        )
        assert len(weight_files) > 0

    def test_config_num_hidden_layers_correct(self, tiny_llama, tmp_path):
        out = save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["num_hidden_layers"] == 8

    def test_config_reflects_surgery(self, tiny_llama, tmp_path):
        from llm_surgeon.surgery import remove_layers
        remove_layers(tiny_llama, [0, 1, 2])
        out = save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["num_hidden_layers"] == 5

    def test_raises_on_config_mismatch(self, tiny_llama, tmp_path):
        # Corrupt the config to simulate a mismatch between layers and config
        tiny_llama.config.num_hidden_layers = 999
        with pytest.raises(ValueError, match="num_hidden_layers"):
            save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))

    def test_model_reloadable(self, tiny_llama, tmp_path):
        out = save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))
        reloaded = AutoModelForCausalLM.from_pretrained(out)
        assert len(reloaded.model.layers) == 8

    def test_tokenizer_saved_when_provided(self, tiny_llama, tmp_path):
        mock_tokenizer = MagicMock()
        save_checkpoint(tiny_llama, str(tmp_path / "ckpt"), tokenizer=mock_tokenizer)
        mock_tokenizer.save_pretrained.assert_called_once_with(str(tmp_path / "ckpt"))

    def test_tokenizer_not_required(self, tiny_llama, tmp_path):
        out = save_checkpoint(tiny_llama, str(tmp_path / "ckpt"))
        assert os.path.exists(os.path.join(out, "config.json"))
        assert not list(Path(out).glob("tokenizer*"))

    def test_output_dir_created_if_missing(self, tiny_llama, tmp_path):
        nested = str(tmp_path / "deep" / "nested" / "ckpt")
        out = save_checkpoint(tiny_llama, nested)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Task 2: to_gguf
#
# Tests that invoke the actual llama.cpp converter use `tiny_checkpoint`,
# which includes a sentencepiece tokenizer that the converter accepts.
# ---------------------------------------------------------------------------

class TestToGguf:
    def test_raises_if_convert_script_missing(self, tiny_checkpoint, tmp_path):
        with pytest.raises(FileNotFoundError):
            to_gguf(tiny_checkpoint, str(tmp_path / "gguf"), llama_cpp_path="/nonexistent/path")

    def test_raises_if_quantize_binary_missing(self, tiny_checkpoint, tmp_path):
        with pytest.raises(FileNotFoundError):
            to_gguf(
                tiny_checkpoint,
                str(tmp_path / "gguf"),
                quantization="Q4_K_M",
                llama_cpp_path="/nonexistent/path",
            )

    @requires_llama_cpp
    def test_convert_produces_f16_gguf(self, tiny_checkpoint, tmp_path):
        out_dir = str(tmp_path / "gguf")
        result = to_gguf(tiny_checkpoint, out_dir, quantization=None)
        assert result.endswith(".gguf")
        assert os.path.exists(result)

    @requires_llama_quantize
    def test_convert_with_quantization_produces_quantized_gguf(self, tiny_checkpoint, tmp_path):
        out_dir = str(tmp_path / "gguf")
        result = to_gguf(tiny_checkpoint, out_dir, quantization="Q4_K_M")
        assert os.path.exists(result)
        assert result.endswith("-Q4_K_M.gguf")

    @requires_llama_quantize
    def test_f16_intermediate_cleaned_up_after_quantization(self, tiny_checkpoint, tmp_path):
        out_dir = str(tmp_path / "gguf")
        to_gguf(tiny_checkpoint, out_dir, quantization="Q4_K_M")
        gguf_files = list(Path(out_dir).glob("*F16*.gguf"))
        assert len(gguf_files) == 0

    @requires_llama_cpp
    def test_no_quantization_returns_f16_path(self, tiny_checkpoint, tmp_path):
        out_dir = str(tmp_path / "gguf")
        result = to_gguf(tiny_checkpoint, out_dir, quantization=None)
        assert os.path.exists(result)
        assert result.endswith(".gguf")

    def test_raises_without_llama_cpp_path(self, tiny_checkpoint, tmp_path, monkeypatch):
        monkeypatch.delenv("LLAMA_CPP_PATH", raising=False)
        with pytest.raises(FileNotFoundError, match="LLAMA_CPP_PATH"):
            to_gguf(tiny_checkpoint, str(tmp_path / "gguf"), quantization=None)

    def test_env_var_overrides_llama_cpp_path(self, tiny_checkpoint, tmp_path, monkeypatch):
        monkeypatch.setenv("LLAMA_CPP_PATH", "/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            to_gguf(tiny_checkpoint, str(tmp_path / "gguf"))

    @requires_llama_cpp
    def test_raises_on_conversion_failure(self, tiny_llama, tmp_path):
        """A checkpoint with no tokenizer fails conversion and raises RuntimeError."""
        ckpt = str(tmp_path / "bad_ckpt")
        save_checkpoint(tiny_llama, ckpt)
        with pytest.raises(RuntimeError, match="GGUF conversion failed"):
            to_gguf(ckpt, str(tmp_path / "gguf"), quantization=None)


# ---------------------------------------------------------------------------
# Task 3: register_ollama
# ---------------------------------------------------------------------------

class TestGenerateModelfile:
    def test_contains_from_directive(self, tmp_path):
        gguf = str(tmp_path / "model.gguf")
        Path(gguf).touch()
        content = _generate_modelfile(gguf)
        assert content.startswith("FROM ")

    def test_contains_absolute_path(self, tmp_path, monkeypatch):
        (tmp_path / "model.gguf").touch()
        monkeypatch.chdir(tmp_path)
        content = _generate_modelfile("model.gguf")  # relative input
        assert content == f"FROM {tmp_path / 'model.gguf'}\n"

    def test_returns_string(self, tmp_path):
        gguf = str(tmp_path / "model.gguf")
        Path(gguf).touch()
        content = _generate_modelfile(gguf)
        assert isinstance(content, str)


class TestRegisterOllama:
    def test_raises_if_gguf_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            register_ollama(str(tmp_path / "nonexistent.gguf"), "test-model")

    def test_modelfile_written_to_disk(self, tmp_path):
        gguf = str(tmp_path / "model.gguf")
        Path(gguf).touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch("llm_surgeon.export._verify_ollama_registration", return_value=True):
                register_ollama(gguf, "test-model")
        modelfiles = list(Path(tmp_path).rglob("*Modelfile"))
        assert len(modelfiles) > 0

    def test_calls_ollama_create(self, tmp_path):
        gguf = str(tmp_path / "model.gguf")
        Path(gguf).touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch("llm_surgeon.export._verify_ollama_registration", return_value=True):
                register_ollama(gguf, "my-model")
        cmd = mock_run.call_args[0][0]
        assert "ollama" in cmd[0] or "ollama" in " ".join(cmd)
        assert "create" in cmd
        assert "my-model" in cmd

    def test_raises_on_ollama_failure(self, tmp_path):
        gguf = str(tmp_path / "model.gguf")
        Path(gguf).touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="ollama error")
            with pytest.raises(RuntimeError, match="ollama create"):
                register_ollama(gguf, "bad-model")

    @requires_llama_cpp
    def test_registers_model_with_ollama(self, tiny_checkpoint, tmp_path, ollama_models):
        gguf_path = to_gguf(tiny_checkpoint, str(tmp_path / "gguf"), quantization=None)
        model_name = "test-llm-surgeon-tiny"
        ollama_models.append(model_name)
        register_ollama(gguf_path, model_name)
        import requests

        from llm_surgeon.export import _ollama_base_url

        r = requests.get(f"{_ollama_base_url()}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        assert f"{model_name}:latest" in names


# ---------------------------------------------------------------------------
# Task 4: full_pipeline
#
# The full_pipeline runs save_checkpoint + to_gguf internally.
# Tests that skip mocking to_gguf need tiny_checkpoint pre-populated,
# but full_pipeline creates its own checkpoint directory — so those tests
# use a mock for to_gguf (since TestToGguf already covers real conversion).
# The end-to-end test injects SPM tokenizer via a mock tokenizer.
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _fake_gguf(self, tmp_path, name="fake.gguf"):
        """Create a dummy GGUF file to use as a mock return value."""
        p = tmp_path / name
        p.touch()
        return str(p)

    def test_returns_dict_with_required_keys(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = self._fake_gguf(tmp_path)
            result = full_pipeline(
                tiny_llama,
                name="test-pipeline",
                quantization=None,
                output_dir=str(tmp_path),
                tokenizer=None,
                register=False,
            )
        assert "checkpoint_path" in result
        assert "gguf_path" in result
        assert "registered" in result

    def test_registered_false_when_skipped(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = self._fake_gguf(tmp_path)
            result = full_pipeline(
                tiny_llama,
                name="test-pipeline",
                quantization=None,
                output_dir=str(tmp_path),
                tokenizer=None,
                register=False,
            )
        assert result["registered"] is False

    def test_checkpoint_path_exists(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = self._fake_gguf(tmp_path)
            result = full_pipeline(
                tiny_llama,
                name="test-pipeline",
                quantization=None,
                output_dir=str(tmp_path),
                tokenizer=None,
                register=False,
            )
        assert os.path.exists(result["checkpoint_path"])

    def test_gguf_path_in_result(self, tiny_llama, tmp_path):
        fake = self._fake_gguf(tmp_path)
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = fake
            result = full_pipeline(
                tiny_llama,
                name="test-pipeline",
                quantization=None,
                output_dir=str(tmp_path),
                tokenizer=None,
                register=False,
            )
        assert result["gguf_path"] == fake

    def test_uses_name_in_checkpoint_path(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = self._fake_gguf(tmp_path)
            result = full_pipeline(
                tiny_llama,
                name="my-custom-model",
                quantization=None,
                output_dir=str(tmp_path),
                tokenizer=None,
                register=False,
            )
        assert "my-custom-model" in result["checkpoint_path"]

    def test_register_true_calls_register_ollama(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = self._fake_gguf(tmp_path)
            with patch("llm_surgeon.export.register_ollama") as mock_reg:
                mock_reg.return_value = None
                result = full_pipeline(
                    tiny_llama,
                    name="test-pipeline",
                    quantization=None,
                    output_dir=str(tmp_path),
                    tokenizer=None,
                    register=True,
                )
        mock_reg.assert_called_once()
        assert result["registered"] is True

    def test_tokenizer_passed_through(self, tiny_llama, tmp_path):
        mock_tokenizer = MagicMock()
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = self._fake_gguf(tmp_path)
            full_pipeline(
                tiny_llama,
                name="test-pipeline",
                quantization=None,
                output_dir=str(tmp_path),
                tokenizer=mock_tokenizer,
                register=False,
            )
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_quantization_forwarded_to_gguf(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = self._fake_gguf(tmp_path)
            full_pipeline(
                tiny_llama,
                name="test-pipeline",
                quantization="Q4_K_M",
                output_dir=str(tmp_path),
                tokenizer=None,
                register=False,
            )
        assert mock_gguf.call_args.kwargs["quantization"] == "Q4_K_M"

    @requires_llama_cpp
    def test_end_to_end_no_ollama(self, tiny_llama, tiny_checkpoint, tmp_path):
        """Full pipeline from model to GGUF with the real converter, no registration."""
        result = full_pipeline(
            tiny_llama,
            name="tiny-e2e",
            quantization=None,
            output_dir=str(tmp_path / "out"),
            tokenizer=_CopyTokenizer(tiny_checkpoint),
            register=False,
        )
        assert os.path.exists(result["gguf_path"])
        assert result["gguf_path"].endswith("tiny-e2e-F16.gguf")
        assert result["registered"] is False

    @requires_llama_cpp
    def test_registers_with_ollama_when_requested(
        self, tiny_llama, tiny_checkpoint, tmp_path, ollama_models
    ):
        """End-to-end with real ollama registration."""
        name = "test-full-pipeline-tiny"
        ollama_models.append(name)
        result = full_pipeline(
            tiny_llama,
            name=name,
            quantization=None,
            output_dir=str(tmp_path / "out"),
            tokenizer=_CopyTokenizer(tiny_checkpoint),
            register=True,
        )
        assert result["registered"] is True
        assert os.path.exists(result["gguf_path"])
