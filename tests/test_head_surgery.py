"""Tests for attention head surgery operations."""

import pytest
import torch

from llm_surgeon.surgery import zero_heads, scale_heads, swap_heads, zero_mlp, zero_attention
from llm_surgeon.inspect import inspect_head
from tests.conftest import _make_tiny_tokenizer


class TestZeroHeads:
    def test_zeroed_head_produces_zero_output(self, tiny_llama):
        """After zeroing head 2 in layer 0, that head's o_proj columns should be zero."""
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        zero_heads(tiny_llama, layer=0, heads=[2])
        o_proj = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        # o_proj shape: (hidden, num_heads * head_dim) — columns [h*hd:(h+1)*hd] are head h
        head_slice = o_proj[:, 2 * head_dim : 3 * head_dim]
        assert torch.all(head_slice == 0)

    def test_other_heads_unchanged(self, tiny_llama):
        """Zeroing head 2 should not affect heads 0, 1, 3."""
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        zero_heads(tiny_llama, layer=0, heads=[2])
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        for h in [0, 1, 3]:
            before_slice = o_before[:, h * head_dim : (h + 1) * head_dim]
            after_slice = o_after[:, h * head_dim : (h + 1) * head_dim]
            assert torch.equal(before_slice, after_slice)

    def test_multiple_heads(self, tiny_llama):
        """Can zero multiple heads at once."""
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        zero_heads(tiny_llama, layer=0, heads=[0, 3])
        o_proj = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        assert torch.all(o_proj[:, 0:head_dim] == 0)
        assert torch.all(o_proj[:, 3 * head_dim : 4 * head_dim] == 0)
        # Head 1 and 2 should be non-zero
        assert not torch.all(o_proj[:, head_dim : 2 * head_dim] == 0)

    def test_returns_surgery_log(self, tiny_llama):
        log = zero_heads(tiny_llama, layer=0, heads=[2])
        assert len(log.ops) == 1
        assert "zero" in log.ops[0].operation
        assert "head" in log.ops[0].operation.lower() or "head" in log.ops[0].description.lower()

    def test_model_still_runs(self, tiny_llama):
        zero_heads(tiny_llama, layer=0, heads=[0, 1, 2])
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_invalid_head_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            zero_heads(tiny_llama, layer=0, heads=[99])

    def test_invalid_layer_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            zero_heads(tiny_llama, layer=99, heads=[0])


class TestScaleHeads:
    def test_scaling_by_zero_equals_zeroing(self, tiny_llama):
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        scale_heads(tiny_llama, layer=0, heads=[1], factor=0.0)
        o_proj = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        head_slice = o_proj[:, head_dim : 2 * head_dim]
        assert torch.all(head_slice == 0)

    def test_scaling_by_one_is_noop(self, tiny_llama):
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        scale_heads(tiny_llama, layer=0, heads=[1], factor=1.0)
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        assert torch.equal(o_before, o_after)

    def test_scaling_by_half(self, tiny_llama):
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        scale_heads(tiny_llama, layer=0, heads=[2], factor=0.5)
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        before_slice = o_before[:, 2 * head_dim : 3 * head_dim]
        after_slice = o_after[:, 2 * head_dim : 3 * head_dim]
        assert torch.allclose(after_slice, before_slice * 0.5)

    def test_other_heads_unchanged(self, tiny_llama):
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        scale_heads(tiny_llama, layer=0, heads=[2], factor=0.5)
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        for h in [0, 1, 3]:
            assert torch.equal(
                o_before[:, h * head_dim : (h + 1) * head_dim],
                o_after[:, h * head_dim : (h + 1) * head_dim],
            )

    def test_returns_surgery_log(self, tiny_llama):
        log = scale_heads(tiny_llama, layer=0, heads=[1], factor=0.5)
        assert len(log.ops) == 1

    def test_model_still_runs(self, tiny_llama):
        scale_heads(tiny_llama, layer=3, heads=[0, 1, 2, 3], factor=0.1)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)


class TestSwapHeads:
    def test_swaps_o_proj_columns(self, tiny_llama):
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        swap_heads(tiny_llama, layer=0, h1=0, h2=3)
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        # Head 0's columns should now be what head 3 had
        assert torch.equal(o_after[:, 0:head_dim], o_before[:, 3 * head_dim : 4 * head_dim])
        assert torch.equal(o_after[:, 3 * head_dim : 4 * head_dim], o_before[:, 0:head_dim])

    def test_swaps_q_proj_rows(self, tiny_llama):
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        q_before = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        swap_heads(tiny_llama, layer=0, h1=1, h2=2)
        q_after = tiny_llama.model.layers[0].self_attn.q_proj.weight.data
        assert torch.equal(q_after[head_dim : 2 * head_dim, :], q_before[2 * head_dim : 3 * head_dim, :])
        assert torch.equal(q_after[2 * head_dim : 3 * head_dim, :], q_before[head_dim : 2 * head_dim, :])

    def test_middle_heads_unchanged(self, tiny_llama):
        head_dim = tiny_llama.config.hidden_size // tiny_llama.config.num_attention_heads
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        swap_heads(tiny_llama, layer=0, h1=0, h2=3)
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        # Heads 1 and 2 unchanged
        assert torch.equal(
            o_before[:, head_dim : 3 * head_dim],
            o_after[:, head_dim : 3 * head_dim],
        )

    def test_returns_surgery_log(self, tiny_llama):
        log = swap_heads(tiny_llama, layer=0, h1=0, h2=1)
        assert len(log.ops) == 1

    def test_model_still_runs(self, tiny_llama):
        swap_heads(tiny_llama, layer=0, h1=0, h2=3)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_swap_same_head_is_noop(self, tiny_llama):
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        swap_heads(tiny_llama, layer=0, h1=2, h2=2)
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        assert torch.equal(o_before, o_after)


class TestInspectHead:
    def test_returns_dict(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        result = inspect_head(tiny_llama, tokenizer, prompt="word4 word5 word6 word7", layer=0, head=0)
        assert isinstance(result, dict)

    def test_has_attention_pattern(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        result = inspect_head(tiny_llama, tokenizer, prompt="word4 word5 word6 word7", layer=0, head=0)
        assert "attention_pattern" in result
        attn = result["attention_pattern"]
        assert isinstance(attn, torch.Tensor)
        # Should be (seq_len, seq_len) — attention weights for this head
        seq_len = attn.shape[0]
        assert attn.shape == (seq_len, seq_len)

    def test_attention_rows_sum_to_one(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        result = inspect_head(tiny_llama, tokenizer, prompt="word4 word5 word6 word7", layer=0, head=0)
        attn = result["attention_pattern"]
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_has_output_norm(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        result = inspect_head(tiny_llama, tokenizer, prompt="word4 word5 word6 word7", layer=0, head=0)
        assert "output_norm" in result
        assert isinstance(result["output_norm"], float)
        assert result["output_norm"] >= 0

    def test_has_entropy(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        result = inspect_head(tiny_llama, tokenizer, prompt="word4 word5 word6 word7", layer=0, head=0)
        assert "entropy" in result
        assert isinstance(result["entropy"], float)
        assert result["entropy"] >= 0

    def test_different_heads_different_patterns(self, tiny_llama, tiny_llama_config):
        tokenizer = _make_tiny_tokenizer(tiny_llama_config.vocab_size)
        r0 = inspect_head(tiny_llama, tokenizer, prompt="word4 word5 word6 word7", layer=0, head=0)
        r1 = inspect_head(tiny_llama, tokenizer, prompt="word4 word5 word6 word7", layer=0, head=1)
        # Different heads should have different attention patterns (very unlikely to be identical with random weights)
        assert not torch.equal(r0["attention_pattern"], r1["attention_pattern"])


class TestZeroMlp:
    def test_down_proj_zeroed(self, tiny_llama):
        """After zero_mlp, the down_proj weight should be all zeros."""
        zero_mlp(tiny_llama, layer=0)
        dp = tiny_llama.model.layers[0].mlp.down_proj.weight.data
        assert torch.all(dp == 0)

    def test_other_layers_unchanged(self, tiny_llama):
        """zero_mlp on layer 0 should not affect layer 1's MLP."""
        dp1_before = tiny_llama.model.layers[1].mlp.down_proj.weight.data.clone()
        zero_mlp(tiny_llama, layer=0)
        dp1_after = tiny_llama.model.layers[1].mlp.down_proj.weight.data
        assert torch.equal(dp1_before, dp1_after)

    def test_attention_unchanged(self, tiny_llama):
        """zero_mlp should not affect the attention weights in the same layer."""
        o_before = tiny_llama.model.layers[0].self_attn.o_proj.weight.data.clone()
        zero_mlp(tiny_llama, layer=0)
        o_after = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        assert torch.equal(o_before, o_after)

    def test_returns_surgery_log(self, tiny_llama):
        log = zero_mlp(tiny_llama, layer=0)
        assert len(log.ops) == 1
        assert "mlp" in log.ops[0].operation.lower() or "mlp" in log.ops[0].description.lower()

    def test_model_still_runs(self, tiny_llama):
        zero_mlp(tiny_llama, layer=0)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_invalid_layer_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            zero_mlp(tiny_llama, layer=99)

    def test_multiple_layers(self, tiny_llama):
        """Can zero MLP in multiple layers sequentially."""
        zero_mlp(tiny_llama, layer=0)
        zero_mlp(tiny_llama, layer=3)
        assert torch.all(tiny_llama.model.layers[0].mlp.down_proj.weight.data == 0)
        assert torch.all(tiny_llama.model.layers[3].mlp.down_proj.weight.data == 0)
        # Other layers untouched
        assert not torch.all(tiny_llama.model.layers[1].mlp.down_proj.weight.data == 0)


class TestZeroAttention:
    def test_o_proj_zeroed(self, tiny_llama):
        """After zero_attention, the o_proj weight should be all zeros."""
        zero_attention(tiny_llama, layer=0)
        o = tiny_llama.model.layers[0].self_attn.o_proj.weight.data
        assert torch.all(o == 0)

    def test_mlp_unchanged(self, tiny_llama):
        """zero_attention should not affect the MLP weights in the same layer."""
        dp_before = tiny_llama.model.layers[0].mlp.down_proj.weight.data.clone()
        zero_attention(tiny_llama, layer=0)
        dp_after = tiny_llama.model.layers[0].mlp.down_proj.weight.data
        assert torch.equal(dp_before, dp_after)

    def test_returns_surgery_log(self, tiny_llama):
        log = zero_attention(tiny_llama, layer=0)
        assert len(log.ops) == 1

    def test_model_still_runs(self, tiny_llama):
        zero_attention(tiny_llama, layer=0)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_invalid_layer_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            zero_attention(tiny_llama, layer=99)
