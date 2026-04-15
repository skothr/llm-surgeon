import pytest
import numpy as np
from pathlib import Path
from llm_surgeon.gguf_reader import (
    GGUFFile,
    resolve_ollama_blob,
    load_gguf_as_hf,
    gguf_model_meta,
    _dequant_f32,
    _dequant_f16,
    _dequant_q4_0,
    _dequant_q8_0,
    _map_tensor_name,
)

OLLAMA_DIR = Path("/usr/share/ollama/.ollama/models")
TINYLLAMA_EXISTS = (
    OLLAMA_DIR / "manifests/registry.ollama.ai/library/tinyllama/latest"
).exists()


# ── Unit tests (no model needed) ─────────────────────────────────────

class TestDequant:
    def test_f32_roundtrip(self):
        vals = np.array([1.0, -2.5, 0.0, 3.14], dtype=np.float32)
        result = _dequant_f32(vals.tobytes(), 4)
        np.testing.assert_allclose(result, vals)

    def test_f16_roundtrip(self):
        vals = np.array([1.0, -2.5, 0.0, 3.14], dtype=np.float16)
        result = _dequant_f16(vals.tobytes(), 4)
        np.testing.assert_allclose(result, vals.astype(np.float32), rtol=1e-3)

    def test_q4_0_shape(self):
        nb = 4
        block = np.zeros(18, dtype=np.uint8)
        block[:2] = np.array([1.0], dtype=np.float16).view(np.uint8)
        data = np.tile(block, nb).tobytes()
        result = _dequant_q4_0(data, nb * 32)
        assert result.shape == (nb * 32,)

    def test_q8_0_known_values(self):
        block = bytearray(34)
        scale = np.array([0.5], dtype=np.float16)
        block[:2] = scale.tobytes()
        for i in range(32):
            block[2 + i] = np.int8(i - 16).view(np.uint8)
        result = _dequant_q8_0(bytes(block), 32)
        expected = np.array([i - 16 for i in range(32)], dtype=np.float32) * 0.5
        np.testing.assert_allclose(result, expected, rtol=1e-3)


class TestNameMapping:
    def test_global_tensors(self):
        assert _map_tensor_name("token_embd.weight") == "model.embed_tokens.weight"
        assert _map_tensor_name("output_norm.weight") == "model.norm.weight"
        assert _map_tensor_name("output.weight") == "lm_head.weight"

    def test_layer_tensors(self):
        assert _map_tensor_name("blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight"
        assert _map_tensor_name("blk.15.ffn_down.weight") == "model.layers.15.mlp.down_proj.weight"
        assert _map_tensor_name("blk.3.attn_norm.weight") == "model.layers.3.input_layernorm.weight"

    def test_unknown_returns_none(self):
        assert _map_tensor_name("unknown_tensor") is None
        assert _map_tensor_name("blk.0.unknown_thing") is None


# ── Integration tests (need Ollama models) ────────────────────────────

@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestOllamaResolution:
    def test_resolve_with_tag(self):
        blob = resolve_ollama_blob("tinyllama:latest")
        assert blob is not None
        assert blob.exists()

    def test_resolve_default_tag(self):
        blob = resolve_ollama_blob("tinyllama")
        assert blob is not None

    def test_resolve_nonexistent(self):
        assert resolve_ollama_blob("nonexistent-model-xyz:latest") is None


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestGGUFFile:
    @pytest.fixture
    def gguf(self):
        blob = resolve_ollama_blob("tinyllama:latest")
        g = GGUFFile(blob)
        yield g
        g.close()

    def test_architecture(self, gguf):
        assert gguf.architecture == "llama"

    def test_metadata(self, gguf):
        assert gguf.metadata["llama.block_count"] == 22
        assert gguf.metadata["llama.embedding_length"] == 2048
        assert gguf.metadata["llama.attention.head_count"] == 32

    def test_tensor_count(self, gguf):
        assert len(gguf.tensor_infos) == 201

    def test_read_tensor(self, gguf):
        arr = gguf.read_tensor_numpy("blk.0.attn_q.weight")
        assert arr.shape == (2048, 2048)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()

    def test_read_tensor_torch(self, gguf):
        t = gguf.read_tensor("blk.0.attn_q.weight")
        assert t.shape == (2048, 2048)
        assert t.dtype == torch.float16

    def test_norm_tensor_f32(self, gguf):
        arr = gguf.read_tensor_numpy("blk.0.attn_norm.weight")
        assert arr.shape == (2048,)


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestGGUFModelMeta:
    def test_meta_fields(self):
        blob = resolve_ollama_blob("tinyllama:latest")
        meta = gguf_model_meta(blob)
        assert meta["architecture"] == "llama"
        assert meta["quantization"] == "Q4_0"
        assert meta["num_layers"] == 22
        assert meta["hidden_size"] == 2048
        assert meta["num_heads"] == 32
        assert meta["num_kv_heads"] == 4
        assert meta["vocab_size"] == 32000
        assert meta["intermediate_size"] == 5632
        assert meta["max_position_embeddings"] == 2048
        assert meta["num_tensors"] == 201
        assert meta["model_name"] == "TinyLlama"
        assert meta["total_params"] > 1e9
        assert meta["total_bytes"] > 0
        assert 4.0 <= meta["bits_per_weight"] <= 5.0
        assert "Q4_0" in meta["tensor_type_counts"]


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestLoadGGUFAsHF:
    @pytest.fixture(scope="class")
    def model_and_tok(self):
        import warnings
        warnings.filterwarnings("ignore", "invalid value")
        blob = resolve_ollama_blob("tinyllama:latest")
        return load_gguf_as_hf(blob)

    def test_model_type(self, model_and_tok):
        model, _ = model_and_tok
        assert type(model).__name__ == "LlamaForCausalLM"

    def test_config(self, model_and_tok):
        model, _ = model_and_tok
        assert model.config.num_hidden_layers == 22
        assert model.config.hidden_size == 2048
        assert model.config.num_attention_heads == 32
        assert model.config.num_key_value_heads == 4

    def test_layer_structure(self, model_and_tok):
        model, _ = model_and_tok
        assert len(model.model.layers) == 22
        layer = model.model.layers[0]
        assert hasattr(layer, "self_attn")
        assert hasattr(layer.self_attn, "q_proj")
        assert hasattr(layer.self_attn, "o_proj")
        assert hasattr(layer, "mlp")
        assert hasattr(layer.mlp, "gate_proj")

    def test_weight_shapes(self, model_and_tok):
        model, _ = model_and_tok
        assert model.model.embed_tokens.weight.shape == (32000, 2048)
        assert model.lm_head.weight.shape == (32000, 2048)
        assert model.model.layers[0].self_attn.q_proj.weight.shape == (2048, 2048)
        assert model.model.layers[0].mlp.gate_proj.weight.shape == (5632, 2048)

    def test_tokenizer(self, model_and_tok):
        _, tokenizer = model_and_tok
        assert tokenizer is not None
        ids = tokenizer.encode("Hello")
        assert len(ids) > 0
        assert tokenizer.decode(ids) == "Hello"

    def test_surgery_compatible(self, model_and_tok):
        """Verify the model works with core surgery operations."""
        import copy
        from llm_surgeon import surgery
        model, _ = model_and_tok
        m = copy.deepcopy(model)
        surgery.zero_heads(m, 0, [0])
        head_dim = 2048 // 32
        norm = m.model.layers[0].self_attn.o_proj.weight[:, :head_dim].float().norm()
        assert norm.item() == 0.0


import torch
