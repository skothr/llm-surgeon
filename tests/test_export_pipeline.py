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
    _ollama_base_url,
    _verify_ollama_registration,
    full_pipeline,
    register_ollama,
    save_checkpoint,
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

    def test_model_name_sets_output_stem(self, llama_cpp_dir, tmp_path):
        with patch("subprocess.run", FakeRun()):
            result = to_gguf(
                str(tmp_path / "checkpoint"),
                str(tmp_path / "out"),
                quantization="Q4_K_M",
                model_name="my-model",
            )
        assert os.path.basename(result) == "my-model-Q4_K_M.gguf"


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


class TestVerifyOllamaRegistration:
    def test_prefix_or_substring_match_is_not_registration(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        get, _ = _fake_tags("foo-old:latest", "myfoo:latest")
        with patch("requests.get", get):
            assert _verify_ollama_registration("foo") is False

    def test_untagged_name_matches_latest(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        get, _ = _fake_tags("foo:latest")
        with patch("requests.get", get):
            assert _verify_ollama_registration("foo") is True

    def test_tag_must_match(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        get, _ = _fake_tags("foo:latest")
        with patch("requests.get", get):
            assert _verify_ollama_registration("foo:q4") is False

    def test_queries_ollama_host(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "10.1.2.3:9999")
        get, seen = _fake_tags("foo:latest")
        with patch("requests.get", get):
            assert _verify_ollama_registration("foo") is True
        assert seen == ["http://10.1.2.3:9999/api/tags"]


class TestOllamaBaseUrl:
    @pytest.mark.parametrize(
        ("host", "url"),
        [
            (None, "http://127.0.0.1:11434"),
            ("", "http://127.0.0.1:11434"),
            ("0.0.0.0", "http://0.0.0.0:11434"),
            ("example.com:8080", "http://example.com:8080"),
            ("http://example.com", "http://example.com:80"),
            ("https://example.com", "https://example.com:443"),
            ("https://example.com:8443/ollama/", "https://example.com:8443/ollama"),
            ("[::1]:11500", "http://[::1]:11500"),
        ],
    )
    def test_parses_like_ollama(self, monkeypatch, host, url):
        if host is None:
            monkeypatch.delenv("OLLAMA_HOST", raising=False)
        else:
            monkeypatch.setenv("OLLAMA_HOST", host)
        assert _ollama_base_url() == url


class TestRegisterOllamaFiles:
    def _register(self, gguf: Path, name: str, verified: bool = True):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch(
                "llm_surgeon.export._verify_ollama_registration", return_value=verified
            ):
                register_ollama(str(gguf), name)
        return mock_run.call_args[0][0]

    def test_modelfile_named_after_gguf(self, tmp_path):
        a, b = tmp_path / "a-Q4_K_M.gguf", tmp_path / "b-Q4_K_M.gguf"
        a.touch()
        b.touch()
        cmd_a = self._register(a, "a")
        cmd_b = self._register(b, "b")
        mf_a, mf_b = Path(cmd_a[-1]), Path(cmd_b[-1])
        assert mf_a != mf_b
        assert mf_a.read_text() == f"FROM {a}\n"
        assert mf_b.read_text() == f"FROM {b}\n"

    def test_unverified_registration_raises(self, tmp_path):
        gguf = tmp_path / "m.gguf"
        gguf.touch()
        with pytest.raises(RuntimeError, match="not listed by the Ollama API"):
            self._register(gguf, "m", verified=False)


# ---------------------------------------------------------------------------
# full_pipeline and save_checkpoint
# ---------------------------------------------------------------------------


class TestFullPipelineNaming:
    def test_gguf_named_after_model_not_checkpoint_dir(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = str(tmp_path / "x.gguf")
            full_pipeline(
                tiny_llama, "my-model", "Q4_K_M", str(tmp_path), register=False
            )
        assert mock_gguf.call_args.kwargs["model_name"] == "my-model"
        assert mock_gguf.call_args.kwargs["quantization"] == "Q4_K_M"

    def test_ollama_style_name_is_one_path_component(self, tiny_llama, tmp_path):
        with patch("llm_surgeon.export.to_gguf") as mock_gguf:
            mock_gguf.return_value = str(tmp_path / "x.gguf")
            with patch("llm_surgeon.export.register_ollama") as mock_reg:
                result = full_pipeline(
                    tiny_llama, "user/model:tag", None, str(tmp_path), register=True
                )
        ckpt = Path(result["checkpoint_path"])
        assert ckpt == tmp_path / "user_model_tag" / "checkpoint"
        assert mock_gguf.call_args.kwargs["model_name"] == "user_model_tag"
        # Ollama still gets the name as given.
        assert mock_reg.call_args[0][1] == "user/model:tag"

    def test_end_to_end_with_fake_llama_cpp(self, tiny_llama, llama_cpp_dir, tmp_path):
        del llama_cpp_dir
        with patch("subprocess.run", FakeRun()):
            result = full_pipeline(
                tiny_llama, "tiny", "Q4_K_M", str(tmp_path), register=False
            )
        assert result["gguf_path"] == str(
            tmp_path / "tiny" / "gguf" / "tiny-Q4_K_M.gguf"
        )
        assert os.path.exists(result["gguf_path"])
        assert result["registered"] is False


class TestSaveCheckpointShape:
    def test_wrapped_model_raises_type_error(self, tiny_llama, tmp_path):
        wrapper = types.SimpleNamespace(module=tiny_llama, config=tiny_llama.config)
        with pytest.raises(TypeError, match="model.model.layers"):
            save_checkpoint(wrapper, str(tmp_path / "ckpt"))

    def test_uses_export_module(self):
        # Guard against the tests above silently importing another copy.
        assert export.__file__.endswith(os.path.join("llm_surgeon", "export.py"))
