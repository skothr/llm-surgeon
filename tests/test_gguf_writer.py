"""export_hf_to_gguf → load_gguf_as_hf round trips on configs the shared
MHA fixture does not cover: GQA, head_dim != hidden / heads, RoPE scaling.
"""

import logging
import warnings

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon import gguf_reader
from llm_surgeon.gguf_reader import GGUFFile, load_gguf_as_hf

gguf = pytest.importorskip("gguf")

from llm_surgeon.gguf_writer import _forward_permute, export_hf_to_gguf  # noqa: E402
from tests.test_gguf_reader import _llama_cpp_permute  # noqa: E402

VOCAB = 40


def _word_tokenizer(n: int):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    specials = ["<unk>", "<s>", "</s>"]
    vocab = {t: i for i, t in enumerate(specials)}
    for i in range(len(specials), n):
        vocab[f"w{i}"] = i
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
    )


def _gqa_config(**overrides) -> LlamaConfig:
    """GQA: 8 query heads share 2 KV heads."""
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        max_position_embeddings=256, rope_theta=500000.0,
    )
    kwargs.update(overrides)
    return LlamaConfig(**kwargs)  # pyright: ignore[reportArgumentType]


def _f16_exact_model(config: LlamaConfig) -> LlamaForCausalLM:
    """A random model whose weights survive the F16 export unchanged."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())
    return model.eval()


def _logits(model, n_pos: int = 120) -> torch.Tensor:
    ids = torch.arange(n_pos).remainder(VOCAB).unsqueeze(0)
    with torch.no_grad():
        return model(ids).logits


def _roundtrip(model, tmp_path, tokenizer=None):
    out = export_hf_to_gguf(model, tokenizer or _word_tokenizer(VOCAB), tmp_path / "m.gguf")
    loaded, _ = load_gguf_as_hf(out, dtype=torch.float32)
    return out, loaded


class TestRoundTrip:
    def test_gqa_and_explicit_head_dim(self, tmp_path):
        """head_dim 16 != hidden 64 / 8 heads."""
        model = _f16_exact_model(_gqa_config(head_dim=16))
        out, loaded = _roundtrip(model, tmp_path)
        with GGUFFile(out) as g:
            assert g.metadata["llama.attention.key_length"] == 16
            assert g.metadata["llama.rope.dimension_count"] == 16
        assert loaded.config.head_dim == 16
        torch.testing.assert_close(_logits(loaded), _logits(model), rtol=1e-4, atol=1e-4)

    def test_llama3_rope_scaling(self, tmp_path):
        rope = {
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 512,
        }
        model = _f16_exact_model(_gqa_config(rope_scaling=rope))
        out, loaded = _roundtrip(model, tmp_path)
        with GGUFFile(out) as g:
            assert any(ti.name == "rope_freqs.weight" for ti in g.tensor_infos)
        torch.testing.assert_close(
            loaded.model.rotary_emb.inv_freq, model.model.rotary_emb.inv_freq, rtol=1e-6, atol=0,
        )
        torch.testing.assert_close(_logits(loaded), _logits(model), rtol=1e-4, atol=1e-4)

    def test_linear_rope_scaling(self, tmp_path):
        model = _f16_exact_model(_gqa_config(rope_scaling={"rope_type": "linear", "factor": 4.0}))
        out, loaded = _roundtrip(model, tmp_path)
        with GGUFFile(out) as g:
            assert g.metadata["llama.rope.scaling.type"] == "linear"
            assert g.metadata["llama.rope.scaling.factor"] == 4.0
        torch.testing.assert_close(_logits(loaded), _logits(model), rtol=1e-4, atol=1e-4)

    def test_load_emits_no_future_warning(self, tmp_path):
        """transformers 5.18 drops LlamaRotaryEmbedding's device kwarg."""
        model = _f16_exact_model(_gqa_config())
        out = export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "m.gguf")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            load_gguf_as_hf(out, dtype=torch.float32)

    def test_num_key_value_heads_none_means_mha(self, tmp_path):
        model = _f16_exact_model(_gqa_config(num_key_value_heads=8))
        model.config.num_key_value_heads = None
        out, loaded = _roundtrip(model, tmp_path)
        with GGUFFile(out) as g:
            assert g.metadata["llama.attention.head_count_kv"] == 8
        model.config.num_key_value_heads = 8
        torch.testing.assert_close(_logits(loaded), _logits(model), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_heads,n_kv,head_dim", [(4, 4, 8), (8, 2, 6)])
def test_forward_permute_matches_llama_cpp(n_heads, n_kv, head_dim):
    rng = np.random.default_rng(1)
    k = rng.standard_normal((n_kv * head_dim, 5)).astype(np.float32)
    np.testing.assert_array_equal(_forward_permute(k, n_kv), _llama_cpp_permute(k, n_heads, n_kv))


class TestLoaderRefusesMismatch:
    def test_missing_tensor_raises(self, tmp_path, monkeypatch):
        model = _f16_exact_model(_gqa_config())
        full = model.state_dict()
        dropped = "model.layers.1.mlp.up_proj.weight"
        monkeypatch.setattr(model, "state_dict", lambda: {k: v for k, v in full.items() if k != dropped})
        out = export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "m.gguf")
        with pytest.raises(ValueError, match="missing"):
            load_gguf_as_hf(out, dtype=torch.float32)

    def test_unmapped_tensor_raises(self, tmp_path, monkeypatch):
        model = _f16_exact_model(_gqa_config())
        out = export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "m.gguf")
        monkeypatch.delitem(gguf_reader._GGUF_LAYER, "ffn_up.weight")
        with pytest.raises(ValueError, match="no LlamaForCausalLM mapping"):
            load_gguf_as_hf(out, dtype=torch.float32)


class TestExportRefuses:
    def test_non_llama_architecture(self, tmp_path):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(  # pyright: ignore[reportCallIssue]
            vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2,
        )
        with pytest.raises(ValueError, match="model_type='qwen2'"):
            export_hf_to_gguf(Qwen2ForCausalLM(config), _word_tokenizer(VOCAB), tmp_path / "q.gguf")
        assert not (tmp_path / "q.gguf").exists()

    def test_unmapped_parameter(self, tmp_path):
        model = LlamaForCausalLM(_gqa_config(attention_bias=True))
        with pytest.raises(ValueError, match="q_proj.bias"):
            export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "b.gguf")

    def test_tokenizer_larger_than_embedding(self, tmp_path):
        model = LlamaForCausalLM(_gqa_config())
        with pytest.raises(ValueError, match="embedding"):
            export_hf_to_gguf(model, _word_tokenizer(VOCAB + 5), tmp_path / "v.gguf")

    def test_unsupported_rope_type(self, tmp_path):
        rope = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64}
        model = LlamaForCausalLM(_gqa_config(rope_scaling=rope))
        with pytest.raises(ValueError, match="yarn"):
            export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "y.gguf")


class TestExportedVocab:
    def test_padded_embedding_gets_named_unused_tokens(self, tmp_path):
        model = LlamaForCausalLM(_gqa_config(vocab_size=VOCAB + 8))
        out = export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "p.gguf")
        with GGUFFile(out) as g:
            tokens = g.metadata["tokenizer.ggml.tokens"]
            types = g.metadata["tokenizer.ggml.token_type"]
            assert len(tokens) == VOCAB + 8 == g.metadata["llama.vocab_size"]
            assert tokens[VOCAB] == f"[PAD{VOCAB}]"
            assert types[VOCAB:] == [5] * 8
            assert "" not in tokens
        loaded, _ = load_gguf_as_hf(out, dtype=torch.float32)
        assert loaded.config.vocab_size == VOCAB + 8

    def test_token_types(self, tmp_path):
        model = LlamaForCausalLM(_gqa_config())
        out = export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "t.gguf")
        with GGUFFile(out) as g:
            types = g.metadata["tokenizer.ggml.token_type"]
            assert types[:3] == [2, 3, 3]  # <unk> unknown; <s>, </s> control
            assert g.metadata["tokenizer.ggml.add_bos_token"] is False

    def test_fp16_overflow_is_clamped_with_warning(self, tmp_path, caplog):
        model = LlamaForCausalLM(_gqa_config())
        with torch.no_grad():
            model.get_parameter("model.layers.0.mlp.down_proj.weight")[0, 0] = 1e6
        with caplog.at_level(logging.WARNING, logger="llm_surgeon.gguf_writer"):
            out = export_hf_to_gguf(model, _word_tokenizer(VOCAB), tmp_path / "o.gguf")
        assert "float16 range" in caplog.text
        with GGUFFile(out) as g:
            arr = g.read_tensor_numpy("blk.0.ffn_down.weight")
        assert np.isfinite(arr).all()
        assert arr[0, 0] == np.finfo(np.float16).max
