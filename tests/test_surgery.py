"""Tests for surgery module."""

import pytest
import torch
from llm_surgeon.surgery import SurgeryOp, SurgeryLog
from llm_surgeon.surgery import get_layer_info
from llm_surgeon.surgery import remove_layers
from llm_surgeon.surgery import keep_layers
from llm_surgeon.surgery import reorder_layers


class TestSurgeryOp:
    def test_creation(self):
        op = SurgeryOp(
            operation="remove_layers",
            description="Removed layers [16, 17, 18]",
            layer_count_before=32,
            layer_count_after=29,
        )
        assert op.operation == "remove_layers"
        assert op.layer_count_before == 32
        assert op.layer_count_after == 29

    def test_str(self):
        op = SurgeryOp("remove_layers", "Removed layers [16]", 32, 31)
        s = str(op)
        assert "remove_layers" in s


class TestSurgeryLog:
    def test_empty(self):
        log = SurgeryLog()
        assert len(log.ops) == 0

    def test_add_operation(self):
        log = SurgeryLog()
        log.add("remove_layers", "Removed layers [16]", 32, 31)
        assert len(log.ops) == 1
        assert log.ops[0].operation == "remove_layers"
        assert log.ops[0].layer_count_before == 32
        assert log.ops[0].layer_count_after == 31

    def test_multiple_operations(self):
        log = SurgeryLog()
        log.add("remove_layers", "Removed layers [16]", 32, 31)
        log.add("swap_layers", "Swapped 0 and 5", 31, 31)
        assert len(log.ops) == 2

    def test_str(self):
        log = SurgeryLog()
        log.add("remove_layers", "Removed layers [16]", 32, 31)
        s = str(log)
        assert "SurgeryLog" in s
        assert "remove_layers" in s


class TestTinyLlamaFixture:
    def test_fixture_creates_model(self, tiny_llama):
        assert tiny_llama is not None
        assert len(tiny_llama.model.layers) == 8

    def test_fixture_has_correct_config(self, tiny_llama):
        assert tiny_llama.config.num_hidden_layers == 8
        assert tiny_llama.config.hidden_size == 32
        assert tiny_llama.config.vocab_size == 64

    def test_fixture_can_forward(self, tiny_llama):
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)


class TestGetLayerInfo:
    def test_returns_dict(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert isinstance(info, dict)

    def test_layer_count(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["num_layers"] == 8

    def test_hidden_size(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["hidden_size"] == 32

    def test_total_params_positive(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["total_params"] > 0

    def test_layer_params_list_length(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert len(info["layer_params"]) == 8

    def test_estimated_memory_positive(self, tiny_llama):
        info = get_layer_info(tiny_llama)
        assert info["estimated_memory_gb"] > 0


class TestRemoveLayers:
    def test_removes_single_layer(self, tiny_llama):
        log = remove_layers(tiny_llama, [3])
        assert len(tiny_llama.model.layers) == 7
        assert tiny_llama.config.num_hidden_layers == 7

    def test_removes_multiple_layers(self, tiny_llama):
        log = remove_layers(tiny_llama, [2, 4, 6])
        assert len(tiny_llama.model.layers) == 5
        assert tiny_llama.config.num_hidden_layers == 5

    def test_returns_surgery_log(self, tiny_llama):
        log = remove_layers(tiny_llama, [0])
        assert len(log.ops) == 1
        assert log.ops[0].operation == "remove_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 7

    def test_preserves_remaining_layers(self, tiny_llama):
        weight_before = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        remove_layers(tiny_llama, [7])
        weight_after = tiny_llama.model.layers[0].self_attn.q_proj.weight.data
        assert torch.equal(weight_before, weight_after)

    def test_invalid_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            remove_layers(tiny_llama, [99])

    def test_negative_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            remove_layers(tiny_llama, [-1])

    def test_model_still_runs_after_surgery(self, tiny_llama):
        remove_layers(tiny_llama, [3, 4, 5])
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)


class TestKeepLayers:
    def test_keeps_specified_layers(self, tiny_llama):
        log = keep_layers(tiny_llama, [0, 1, 2])
        assert len(tiny_llama.model.layers) == 3
        assert tiny_llama.config.num_hidden_layers == 3

    def test_returns_surgery_log(self, tiny_llama):
        log = keep_layers(tiny_llama, [0, 1])
        assert log.ops[0].operation == "keep_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 2

    def test_preserves_correct_layers(self, tiny_llama):
        w0 = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        w7 = tiny_llama.model.layers[7].self_attn.q_proj.weight.data.clone()
        keep_layers(tiny_llama, [0, 7])
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w0)
        assert torch.equal(tiny_llama.model.layers[1].self_attn.q_proj.weight.data, w7)

    def test_invalid_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            keep_layers(tiny_llama, [0, 99])

    def test_model_still_runs(self, tiny_llama):
        keep_layers(tiny_llama, [0, 3, 7])
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)


class TestReorderLayers:
    def test_reverses_layers(self, tiny_llama):
        w_first = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        w_last = tiny_llama.model.layers[7].self_attn.q_proj.weight.data.clone()
        log = reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w_last)
        assert torch.equal(tiny_llama.model.layers[7].self_attn.q_proj.weight.data, w_first)

    def test_layer_count_unchanged(self, tiny_llama):
        reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        assert len(tiny_llama.model.layers) == 8
        assert tiny_llama.config.num_hidden_layers == 8

    def test_returns_surgery_log(self, tiny_llama):
        log = reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        assert log.ops[0].operation == "reorder_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 8

    def test_wrong_length_raises(self, tiny_llama):
        with pytest.raises(ValueError, match="must match layer count"):
            reorder_layers(tiny_llama, [0, 1, 2])

    def test_not_permutation_raises(self, tiny_llama):
        with pytest.raises(ValueError, match="must be a permutation"):
            reorder_layers(tiny_llama, [0, 0, 0, 0, 0, 0, 0, 0])

    def test_model_still_runs(self, tiny_llama):
        reorder_layers(tiny_llama, [7, 6, 5, 4, 3, 2, 1, 0])
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)
