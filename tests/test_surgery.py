"""Tests for surgery module."""

import pytest
import torch
from llm_surgeon.surgery import SurgeryOp, SurgeryLog
from llm_surgeon.surgery import get_layer_info


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
