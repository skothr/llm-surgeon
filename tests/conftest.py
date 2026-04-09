"""Shared test fixtures."""

import json
import os

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


@pytest.fixture
def tiny_checkpoint(tiny_llama, tmp_path):
    """Saved HF checkpoint with a minimal SPM tokenizer — ready for GGUF conversion."""
    import sentencepiece as spm

    checkpoint_dir = str(tmp_path / "tiny_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tiny_llama.save_pretrained(checkpoint_dir)

    # Create a tiny sentencepiece model with the correct vocab_size
    corpus_file = str(tmp_path / "corpus.txt")
    vocab_size = tiny_llama.config.vocab_size
    with open(corpus_file, "w") as f:
        f.write("\n".join([f"tok{i}" for i in range(vocab_size)] + ["hello world"]))

    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=os.path.join(checkpoint_dir, "tokenizer"),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=1.0,
        num_threads=1,
    )

    tokenizer_config = {
        "model_type": "llama",
        "tokenizer_class": "LlamaTokenizer",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }
    with open(os.path.join(checkpoint_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    return checkpoint_dir
