import os

import pytest
import numpy as np
import torch
from pathlib import Path
from llm_surgeon.gguf_reader import (
    GGML_TYPE_BF16,
    GGML_TYPE_I32,
    GGUFFile,
    resolve_ollama_blob,
    load_gguf_as_hf,
    gguf_model_meta,
    _DEQUANT,
    _build_config,
    _dequant_f32,
    _dequant_f16,
    _dequant_q4_0,
    _dequant_q8_0,
    _map_tensor_name,
    _reverse_permute,
    TensorInfo,
)

OLLAMA_DIR = Path(os.environ.get("OLLAMA_MODELS", "/usr/share/ollama/.ollama/models"))
TINYLLAMA_EXISTS = (
    OLLAMA_DIR / "manifests/registry.ollama.ai/library/tinyllama/latest"
).exists()


def _write_gguf(path: Path, tensors: dict, raw_dtypes: dict | None = None) -> Path:
    """Write a GGUF holding only ``tensors`` (name -> array); uint8 arrays
    with an entry in ``raw_dtypes`` are stored as that ggml type's raw bytes."""
    gguf = pytest.importorskip("gguf")
    writer = gguf.GGUFWriter(str(path), arch="llama")
    for name, arr in tensors.items():
        raw = (raw_dtypes or {}).get(name)
        writer.add_tensor(name, arr, raw_dtype=raw)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return path


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


# Byte offsets of the f16 scale/min fields in one block of each type. Random
# bytes there can decode to inf/NaN, so tests overwrite them with finite values.
_F16_FIELDS = {
    "Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0],
    "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208],
}


class TestDequantMatchesGGUFPy:
    """Each native dequantizer against gguf-py's reference implementation."""

    @pytest.mark.parametrize("type_name", sorted(_F16_FIELDS))
    def test_quant_types(self, type_name):
        gguf = pytest.importorskip("gguf")
        from gguf import quants

        qtype = gguf.GGMLQuantizationType[type_name]
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        rng = np.random.default_rng(0)
        nb = 8
        raw = rng.integers(0, 256, size=(nb, type_size), dtype=np.uint8)
        for off in _F16_FIELDS[type_name]:
            scale = rng.uniform(-2, 2, size=nb).astype(np.float16)
            raw[:, off:off + 2] = scale.view(np.uint8).reshape(nb, 2)

        ours = _DEQUANT[qtype.value](raw.tobytes(), nb * block_size)
        ref = quants.dequantize(raw.ravel(), qtype).ravel()
        np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)

    def test_bf16(self):
        gguf = pytest.importorskip("gguf")
        from gguf import quants

        rng = np.random.default_rng(1)
        vals = rng.standard_normal(64).astype(np.float32)
        bits = (vals.view(np.uint32) >> 16).astype(np.uint16)
        ours = _DEQUANT[GGML_TYPE_BF16](bits.tobytes(), 64)
        ref = quants.dequantize(bits.view(np.uint8), gguf.GGMLQuantizationType.BF16).ravel()
        np.testing.assert_array_equal(ours, ref)


class TestGGMLTypeIds:
    def test_ids_match_ggml_enum(self):
        gguf = pytest.importorskip("gguf")
        assert GGML_TYPE_BF16 == gguf.GGMLQuantizationType.BF16.value
        assert GGML_TYPE_I32 == gguf.GGMLQuantizationType.I32.value

    def test_reads_bf16_tensor(self, tmp_path):
        gguf = pytest.importorskip("gguf")
        vals = np.array([1.0, -2.5, 0.15625, 3.0], dtype=np.float32)
        bits = (vals.view(np.uint32) >> 16).astype(np.uint16)
        path = _write_gguf(
            tmp_path / "bf16.gguf", {"t": bits.view(np.uint8)},
            {"t": gguf.GGMLQuantizationType.BF16},
        )
        with GGUFFile(path) as g:
            assert g.tensor_infos[0].type_name == "BF16"
            np.testing.assert_array_equal(g.read_tensor_numpy("t"), vals)

    def test_reads_i32_tensor(self, tmp_path):
        vals = np.array([1, -7, 1 << 20, 0], dtype=np.int32)
        path = _write_gguf(tmp_path / "i32.gguf", {"t": vals})
        with GGUFFile(path) as g:
            assert g.tensor_infos[0].type_name == "I32"
            np.testing.assert_array_equal(g.read_tensor_numpy("t"), vals.astype(np.float32))

    def test_types_without_native_dequant_fall_back_to_gguf_py(self, tmp_path):
        gguf = pytest.importorskip("gguf")
        from gguf import quants

        qtype = gguf.GGMLQuantizationType.Q2_K
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        rng = np.random.default_rng(2)
        raw = rng.integers(0, 256, size=(4, type_size), dtype=np.uint8)
        # Q2_K ends with f16 d, dmin
        raw[:, -4:] = rng.uniform(-1, 1, size=(4, 2)).astype(np.float16).view(np.uint8).reshape(4, 4)
        path = _write_gguf(tmp_path / "q2k.gguf", {"t": raw.ravel()}, {"t": qtype})
        with GGUFFile(path) as g:
            arr = g.read_tensor_numpy("t")
        np.testing.assert_allclose(arr.ravel(), quants.dequantize(raw.ravel(), qtype).ravel())
        assert arr.size == 4 * block_size


class TestFileValidation:
    def test_truncated_tensor_data_raises(self, tmp_path):
        path = _write_gguf(tmp_path / "t.gguf", {"t": np.arange(64, dtype=np.float32)})
        data = path.read_bytes()
        path.write_bytes(data[:-16])
        with GGUFFile(path) as g:
            with pytest.raises(ValueError, match="Truncated"):
                g.read_tensor_numpy("t")

    @pytest.mark.parametrize("version_bytes", [b"\x01\x00\x00\x00", b"\x00\x00\x00\x03"])
    def test_unsupported_version_raises(self, tmp_path, version_bytes):
        """GGUF v1 and big-endian files (version reads byte-swapped) are refused."""
        bad = tmp_path / "v.gguf"
        bad.write_bytes(b"GGUF" + version_bytes + b"\x00" * 16)
        with pytest.raises(ValueError, match="version"):
            GGUFFile(bad)


def _llama_cpp_permute(w: np.ndarray, n_head: int, n_head_kv: int) -> np.ndarray:
    """``LlamaModel.permute`` from llama.cpp's convert_hf_to_gguf.py."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
            .swapaxes(1, 2).reshape(w.shape))


class TestPermute:
    @pytest.mark.parametrize("n_heads,n_kv,head_dim", [(4, 4, 8), (8, 2, 6), (6, 3, 10)])
    def test_reverse_undoes_llama_cpp_permute(self, n_heads, n_kv, head_dim):
        rng = np.random.default_rng(0)
        q = rng.standard_normal((n_heads * head_dim, 12)).astype(np.float32)
        k = rng.standard_normal((n_kv * head_dim, 12)).astype(np.float32)
        q_gguf = _llama_cpp_permute(q, n_heads, n_heads)
        k_gguf = _llama_cpp_permute(k, n_heads, n_kv)
        np.testing.assert_array_equal(_reverse_permute(torch.from_numpy(q_gguf), n_heads).numpy(), q)
        np.testing.assert_array_equal(_reverse_permute(torch.from_numpy(k_gguf), n_kv).numpy(), k)


def _meta(**extra):
    meta = {
        "general.architecture": "llama",
        "llama.embedding_length": 64,
        "llama.block_count": 2,
        "llama.attention.head_count": 4,
        "llama.feed_forward_length": 128,
        "tokenizer.ggml.tokens": [f"t{i}" for i in range(10)],
    }
    meta.update(extra)
    return meta


def _rope_params(config) -> dict:
    params = getattr(config, "rope_parameters", None)
    return params if isinstance(params, dict) else (config.rope_scaling or {})


class TestBuildConfig:
    @pytest.mark.parametrize("key", [
        "llama.embedding_length", "llama.block_count",
        "llama.attention.head_count", "llama.feed_forward_length",
    ])
    def test_missing_required_key_raises(self, key):
        meta = _meta()
        del meta[key]
        with pytest.raises(ValueError, match=key):
            _build_config(meta)

    def test_head_dim_from_key_length(self):
        config = _build_config(_meta(**{"llama.attention.key_length": 32}))
        assert config.head_dim == 32

    def test_vocab_size_from_token_embd(self):
        embd = TensorInfo(name="token_embd.weight", shape=(64, 48), ggml_type=1, offset=0)
        assert _build_config(_meta(), [embd]).vocab_size == 48

    def test_linear_rope_scaling(self):
        config = _build_config(_meta(**{
            "llama.rope.scaling.type": "linear", "llama.rope.scaling.factor": 4.0,
        }))
        params = _rope_params(config)
        assert params.get("rope_type", params.get("type")) == "linear"
        assert params["factor"] == 4.0

    def test_unsupported_rope_scaling_raises(self):
        with pytest.raises(ValueError, match="yarn"):
            _build_config(_meta(**{
                "llama.rope.scaling.type": "yarn", "llama.rope.scaling.factor": 4.0,
            }))


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


def test_resolve_nonexistent(tmp_path):
    assert resolve_ollama_blob("nonexistent-model-xyz:latest", models_dir=str(tmp_path)) is None


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


@pytest.fixture(scope="module")
def model_and_tok():
    import warnings
    warnings.filterwarnings("ignore", "invalid value")
    blob = resolve_ollama_blob("tinyllama:latest")
    return load_gguf_as_hf(blob)


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestLoadGGUFAsHF:

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
        assert ids[0] == tokenizer.bos_token_id == 1
        assert len(ids) > 1
        assert tokenizer.decode(ids, skip_special_tokens=True) == "Hello"

    def test_tokenizer_byte_fallback(self, model_and_tok):
        """Characters outside the vocab encode as <0xNN> byte tokens, not <unk>."""
        _, tokenizer = model_and_tok
        text = "a\tb\n\U0001f999 end"
        ids = tokenizer.encode(text)
        assert tokenizer.unk_token_id not in ids
        assert tokenizer.decode(ids, skip_special_tokens=True) == text

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


class TestGGUFParseFailureClosesFile:
    def test_truncated_file_releases_fd(self, tmp_path):
        """A parse failure must not leak the file descriptor.

        Captures the partially-constructed GGUFFile via a subclass so we
        can inspect _file without relying on implicit GC of the frame
        holding the failed __init__.
        """
        captured: dict = {}

        class Probe(GGUFFile):
            def _parse(self):
                captured["self"] = self
                super()._parse()

        bad = tmp_path / "truncated.gguf"
        bad.write_bytes(b"GGUF" + b"\x03\x00\x00\x00")

        with pytest.raises(Exception):
            Probe(bad)

        obj = captured["self"]
        assert obj._file is None, "partially-constructed GGUFFile leaked its file handle"


class TestGGUFAlignment:
    def test_honors_non_default_alignment(self, tmp_path):
        """Writer with alignment=64 must be readable by GGUFFile."""
        gguf = pytest.importorskip("gguf")

        out = tmp_path / "align64.gguf"
        writer = gguf.GGUFWriter(str(out), arch="llama")
        writer.add_custom_alignment(64)
        vals = np.arange(8, dtype=np.float32)
        writer.add_tensor("t", vals)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()

        with GGUFFile(out) as g:
            assert g.metadata.get("general.alignment") == 64
            arr = g.read_tensor_numpy("t")
            np.testing.assert_array_equal(arr.ravel(), vals)
