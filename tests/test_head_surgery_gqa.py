"""Head surgery on GQA models with an explicit head_dim and projection biases.

The shared ``tiny_llama`` fixture is MHA with ``head_dim == hidden / heads``
and no biases, which hides every bug that depends on those differing.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon.surgery import (
    scale_heads,
    swap_heads,
    zero_attention,
    zero_heads,
    zero_mlp,
)

HEAD_DIM = 16  # != hidden_size // num_attention_heads (32 // 4 == 8)


@pytest.fixture
def gqa_llama():
    """2-layer GQA model: 4 query heads, 2 KV heads, head_dim 16, biased projections."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,  # pyright: ignore[reportCallIssue]
        hidden_size=32,  # pyright: ignore[reportCallIssue]
        intermediate_size=64,  # pyright: ignore[reportCallIssue]
        num_hidden_layers=2,  # pyright: ignore[reportCallIssue]
        num_attention_heads=4,  # pyright: ignore[reportCallIssue]
        num_key_value_heads=2,  # pyright: ignore[reportCallIssue]
        head_dim=HEAD_DIM,  # pyright: ignore[reportCallIssue]
        attention_bias=True,  # pyright: ignore[reportCallIssue]
        mlp_bias=True,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=128,  # pyright: ignore[reportCallIssue]
    )
    model = LlamaForCausalLM(config)
    # HF initialises biases to zero; randomise them so bias handling matters.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.normal_(std=0.5)
    model.eval()
    return model


def _logits(model) -> torch.Tensor:
    input_ids = torch.arange(1, 13).unsqueeze(0)
    with torch.no_grad():
        return model(input_ids).logits


class TestHeadDim:
    def test_fixture_head_dim_differs_from_hidden_over_heads(self, gqa_llama):
        attn = gqa_llama.model.layers[0].self_attn
        assert attn.o_proj.weight.shape == (32, 4 * HEAD_DIM)

    def test_zero_heads_uses_config_head_dim(self, gqa_llama):
        o_before = gqa_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        zero_heads(gqa_llama, layer=0, heads=[1])
        o_after = gqa_llama.model.layers[0].self_attn.o_proj.weight.data
        assert torch.all(o_after[:, HEAD_DIM : 2 * HEAD_DIM] == 0)
        assert torch.equal(o_after[:, :HEAD_DIM], o_before[:, :HEAD_DIM])
        assert torch.equal(o_after[:, 2 * HEAD_DIM :], o_before[:, 2 * HEAD_DIM :])

    def test_scale_heads_uses_config_head_dim(self, gqa_llama):
        o_before = gqa_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        scale_heads(gqa_llama, layer=0, heads=[3], factor=0.5)
        o_after = gqa_llama.model.layers[0].self_attn.o_proj.weight.data
        assert torch.allclose(
            o_after[:, 3 * HEAD_DIM :], o_before[:, 3 * HEAD_DIM :] * 0.5
        )
        assert torch.equal(o_after[:, : 3 * HEAD_DIM], o_before[:, : 3 * HEAD_DIM])


class TestSwapHeadsGQA:
    def test_intra_group_swap_preserves_logits(self, gqa_llama):
        before = _logits(gqa_llama)
        swap_heads(gqa_llama, layer=0, h1=0, h2=1)  # both use KV head 0
        assert torch.allclose(_logits(gqa_llama), before, atol=1e-5)

    def test_intra_group_swap_moves_q_rows_and_o_columns(self, gqa_llama):
        attn = gqa_llama.model.layers[0].self_attn
        q_before = attn.q_proj.weight.data.clone()
        k_before = attn.k_proj.weight.data.clone()
        swap_heads(gqa_llama, layer=0, h1=2, h2=3)
        assert torch.equal(
            attn.q_proj.weight.data[2 * HEAD_DIM : 3 * HEAD_DIM],
            q_before[3 * HEAD_DIM :],
        )
        # The shared K head is untouched.
        assert torch.equal(attn.k_proj.weight.data, k_before)

    def test_cross_group_swap_raises(self, gqa_llama):
        w_before = {n: p.clone() for n, p in gqa_llama.named_parameters()}
        with pytest.raises(ValueError, match="KV group"):
            swap_heads(gqa_llama, layer=0, h1=0, h2=2)
        for n, p in gqa_llama.named_parameters():
            assert torch.equal(p, w_before[n]), f"{n} changed despite the error"


class TestSwapHeadsMHA:
    def test_swap_is_a_permutation(self, tiny_llama):
        before = _logits(tiny_llama)
        swap_heads(tiny_llama, layer=0, h1=0, h2=3)
        assert torch.allclose(_logits(tiny_llama), before, atol=1e-5)


class TestZeroBlocksWithBias:
    """attention_bias / mlp_bias add a constant after the zeroed weight."""

    def test_zero_mlp_output_is_zero(self, gqa_llama):
        zero_mlp(gqa_llama, 0)
        mlp = gqa_llama.model.layers[0].mlp
        with torch.no_grad():
            out = mlp(torch.randn(1, 5, 32))
        assert torch.all(out == 0)

    def test_zero_attention_o_proj_output_is_zero(self, gqa_llama):
        zero_attention(gqa_llama, 0)
        o_proj = gqa_llama.model.layers[0].self_attn.o_proj
        with torch.no_grad():
            out = o_proj(torch.randn(1, 5, 4 * HEAD_DIM))
        assert torch.all(out == 0)


class TestDuplicateHeads:
    def test_scale_heads_rejects_duplicates(self, tiny_llama):
        with pytest.raises(ValueError, match="Duplicate head"):
            scale_heads(tiny_llama, layer=0, heads=[1, 1], factor=0.5)

    def test_zero_heads_rejects_duplicates(self, tiny_llama):
        with pytest.raises(ValueError, match="Duplicate head"):
            zero_heads(tiny_llama, layer=0, heads=[2, 2])
