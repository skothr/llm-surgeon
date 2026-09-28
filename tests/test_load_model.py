"""load_model: device-map / max_memory handling, the .bin fallback, and the
Ollama (GGUF) in-place quantization path. Loading is mocked; nothing is
downloaded."""

import copy

import pytest
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig

from llm_surgeon import surgery


@pytest.fixture
def captured_loads(monkeypatch):
    """Record every from_pretrained call; optionally raise from the first one."""
    calls = []
    first_error: list[OSError] = []

    def fake_model_from_pretrained(load_id, **kwargs):
        calls.append(kwargs)
        if first_error and len(calls) == 1:
            raise first_error[0]
        return object()

    monkeypatch.setattr(
        "llm_surgeon.surgery.AutoModelForCausalLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "llm_surgeon.surgery.AutoTokenizer.from_pretrained",
        lambda load_id, **kwargs: object(),
    )
    monkeypatch.setattr(surgery, "_is_cached", lambda *a, **kw: False)
    return calls, first_error


class TestMaxMemory:
    def test_without_device_map_raises_for_dense_modes(self, captured_loads):
        with pytest.raises(ValueError, match="max_memory"):
            surgery.load_model("Org/Model", mode="fp16", max_memory={0: "4GiB"})

    def test_fp32_cpu_raises(self, captured_loads):
        with pytest.raises(ValueError, match="max_memory"):
            surgery.load_model("Org/Model", mode="export", max_memory={0: "4GiB"})

    def test_with_device_map_is_passed(self, captured_loads):
        calls, _ = captured_loads
        surgery.load_model(
            "Org/Model", mode="bf16", device_map="auto", max_memory={0: "4GiB"}
        )
        assert calls[0]["max_memory"] == {0: "4GiB"}
        assert calls[0]["device_map"] == "auto"
        assert calls[0]["torch_dtype"] is torch.bfloat16


class TestSafetensorsFallback:
    def test_missing_safetensors_retries_without_it(self, captured_loads):
        calls, first_error = captured_loads
        first_error.append(
            OSError(
                "Org/Model does not appear to have a file named model.safetensors "
                "or model.safetensors.index.json"
            )
        )
        surgery.load_model("Org/Model", mode="fp16")
        assert len(calls) == 2
        assert calls[0]["use_safetensors"] is True
        assert "use_safetensors" not in calls[1]

    def test_other_safetensors_error_is_raised(self, captured_loads):
        calls, first_error = captured_loads
        first_error.append(
            OSError("Error while deserializing header: corrupt safetensors file")
        )
        with pytest.raises(OSError, match="corrupt"):
            surgery.load_model("Org/Model", mode="fp16")
        assert len(calls) == 1  # no pickle .bin retry


class TestQuantizeInPlace:
    def test_skips_lm_head_and_honours_device(self, tiny_llama):
        bnb = pytest.importorskip("bitsandbytes")
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )
        model = surgery._quantize_in_place(tiny_llama, cfg, device="cpu")
        assert type(model.lm_head) is nn.Linear
        assert isinstance(model.model.layers[0].self_attn.q_proj, bnb.nn.Linear4bit)
        assert model.model.layers[0].self_attn.q_proj.weight.device.type == "cpu"


class TestOllamaPath:
    @pytest.fixture
    def fake_gguf(self, monkeypatch, tiny_llama):
        from llm_surgeon import gguf_reader

        monkeypatch.setattr(gguf_reader, "resolve_ollama_blob", lambda _id: "blob")
        monkeypatch.setattr(
            gguf_reader,
            "load_gguf_as_hf",
            lambda blob, dtype: (copy.deepcopy(tiny_llama), object()),
        )

    def test_nf4_on_cpu_device_map(self, fake_gguf):
        bnb = pytest.importorskip("bitsandbytes")
        model, _ = surgery.load_model("tiny:latest", mode="nf4", device_map="cpu")
        assert isinstance(model.model.layers[0].mlp.down_proj, bnb.nn.Linear4bit)
        assert type(model.lm_head) is nn.Linear

    def test_single_device_dict_map(self, fake_gguf):
        pytest.importorskip("bitsandbytes")
        model, _ = surgery.load_model(
            "tiny:latest", mode="int8", device_map={"": "cpu"}
        )
        assert model.model.layers[0].mlp.down_proj.weight.device.type == "cpu"

    def test_max_memory_raises(self, fake_gguf):
        pytest.importorskip("bitsandbytes")
        with pytest.raises(ValueError, match="max_memory"):
            surgery.load_model("tiny:latest", mode="nf4", max_memory={0: "4GiB"})

    @pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error")
    def test_no_cuda_without_device_map_raises(self, fake_gguf):
        pytest.importorskip("bitsandbytes")
        with pytest.raises(RuntimeError, match="CUDA"):
            surgery.load_model("tiny:latest", mode="nf4")
