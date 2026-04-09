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
