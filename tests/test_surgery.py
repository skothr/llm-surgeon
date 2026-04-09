"""Tests for surgery module."""

import os
import json
import pytest
import torch
from llm_surgeon.surgery import SurgeryOp, SurgeryLog
from llm_surgeon.surgery import get_layer_info
from llm_surgeon.surgery import remove_layers
from llm_surgeon.surgery import keep_layers
from llm_surgeon.surgery import reorder_layers
from llm_surgeon.surgery import swap_layers
from llm_surgeon.surgery import duplicate_layer
from llm_surgeon.surgery import load_model
from transformers import AutoModelForCausalLM


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


class TestSwapLayers:
    def test_swaps_weights(self, tiny_llama):
        w0 = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        w5 = tiny_llama.model.layers[5].self_attn.q_proj.weight.data.clone()
        swap_layers(tiny_llama, 0, 5)
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w5)
        assert torch.equal(tiny_llama.model.layers[5].self_attn.q_proj.weight.data, w0)

    def test_layer_count_unchanged(self, tiny_llama):
        swap_layers(tiny_llama, 0, 7)
        assert len(tiny_llama.model.layers) == 8
        assert tiny_llama.config.num_hidden_layers == 8

    def test_returns_surgery_log(self, tiny_llama):
        log = swap_layers(tiny_llama, 2, 6)
        assert log.ops[0].operation == "swap_layers"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 8

    def test_invalid_index_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            swap_layers(tiny_llama, 0, 99)

    def test_swap_same_index(self, tiny_llama):
        w0 = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        swap_layers(tiny_llama, 0, 0)
        assert torch.equal(tiny_llama.model.layers[0].self_attn.q_proj.weight.data, w0)

    def test_model_still_runs(self, tiny_llama):
        swap_layers(tiny_llama, 1, 6)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)


class TestDuplicateLayer:
    def test_increases_layer_count(self, tiny_llama):
        log = duplicate_layer(tiny_llama, src=3, dst=4)
        assert len(tiny_llama.model.layers) == 9
        assert tiny_llama.config.num_hidden_layers == 9

    def test_duplicate_has_same_weights(self, tiny_llama):
        w_src = tiny_llama.model.layers[3].self_attn.q_proj.weight.data.clone()
        duplicate_layer(tiny_llama, src=3, dst=4)
        w_dup = tiny_llama.model.layers[4].self_attn.q_proj.weight.data
        assert torch.equal(w_src, w_dup)

    def test_duplicate_is_deep_copy(self, tiny_llama):
        duplicate_layer(tiny_llama, src=3, dst=4)
        tiny_llama.model.layers[4].self_attn.q_proj.weight.data.zero_()
        assert not torch.equal(
            tiny_llama.model.layers[3].self_attn.q_proj.weight.data,
            tiny_llama.model.layers[4].self_attn.q_proj.weight.data,
        )

    def test_returns_surgery_log(self, tiny_llama):
        log = duplicate_layer(tiny_llama, src=0, dst=1)
        assert log.ops[0].operation == "duplicate_layer"
        assert log.ops[0].layer_count_before == 8
        assert log.ops[0].layer_count_after == 9

    def test_invalid_src_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            duplicate_layer(tiny_llama, src=99, dst=0)

    def test_invalid_dst_raises(self, tiny_llama):
        with pytest.raises(IndexError):
            duplicate_layer(tiny_llama, src=0, dst=99)

    def test_dst_at_end_allowed(self, tiny_llama):
        duplicate_layer(tiny_llama, src=0, dst=8)
        assert len(tiny_llama.model.layers) == 9

    def test_model_still_runs(self, tiny_llama):
        duplicate_layer(tiny_llama, src=3, dst=4)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)


class TestLoadModel:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_model("nonexistent-model", mode="invalid")

    def test_valid_modes_accepted(self):
        # Valid modes are accepted; the model lookup fails (OSError for missing model,
        # or ImportError when a SOCKS proxy is configured but socksio is not installed).
        for mode in ("inspect", "eval", "export"):
            with pytest.raises((OSError, ImportError)):
                load_model("nonexistent/model-id-that-does-not-exist", mode=mode)

    def test_returns_tuple(self, tiny_llama, tmp_path):
        save_path = str(tmp_path / "tiny_model")
        tiny_llama.save_pretrained(save_path)
        tokenizer_config = {
            "model_type": "llama",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
        with open(os.path.join(save_path, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)
        vocab = {f"token_{i}": i for i in range(64)}
        tokenizer_data = {
            "version": "1.0",
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1, "content": "<s>", "special": True},
                {"id": 2, "content": "</s>", "special": True},
            ],
        }
        with open(os.path.join(save_path, "tokenizer.json"), "w") as f:
            json.dump(tokenizer_data, f)

        model, tokenizer = load_model(save_path, mode="export")
        assert model is not None
        assert tokenizer is not None
        assert len(model.model.layers) == 8


class TestChainedOperations:
    def test_remove_then_swap(self, tiny_llama):
        remove_layers(tiny_llama, [6, 7])
        swap_layers(tiny_llama, 0, 5)
        assert len(tiny_llama.model.layers) == 6
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_keep_then_reorder(self, tiny_llama):
        keep_layers(tiny_llama, [0, 2, 4, 6])
        reorder_layers(tiny_llama, [3, 2, 1, 0])
        assert len(tiny_llama.model.layers) == 4
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_duplicate_then_remove(self, tiny_llama):
        duplicate_layer(tiny_llama, src=0, dst=1)
        remove_layers(tiny_llama, [0])
        assert len(tiny_llama.model.layers) == 8


class TestCalibrate:
    def test_capture_stats_returns_list(self, tiny_llama):
        from llm_surgeon.surgery import capture_calibration_stats
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        text = " ".join([f"tok{i}" for i in range(4, 20)])
        stats = capture_calibration_stats(tiny_llama, tokenizer, text=text)
        assert isinstance(stats, list)
        assert len(stats) == 8  # 8 layers in tiny_llama
        assert all(isinstance(v, float) and v > 0 for v in stats)

    def test_calibrate_with_baseline_modifies_norms(self, tiny_llama):
        from llm_surgeon.surgery import calibrate, capture_calibration_stats, remove_layers
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        text = " ".join([f"tok{i}" for i in range(4, 20)])

        # Capture baseline BEFORE surgery
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=text)

        # Surgery
        remove_layers(tiny_llama, [3, 4])

        # Clone norm weights before calibration
        norms_before = [
            layer.input_layernorm.weight.data.clone()
            for layer in tiny_llama.model.layers
        ]

        # Calibrate with baseline
        calibrate(tiny_llama, tokenizer, baseline_stats=baseline, text=text)

        norms_after = [
            layer.input_layernorm.weight.data.clone()
            for layer in tiny_llama.model.layers
        ]
        changed = any(
            not torch.equal(b, a) for b, a in zip(norms_before, norms_after)
        )
        assert changed, "calibrate() did not modify any norm parameters"

    def test_calibrate_without_baseline_warns(self, tiny_llama):
        from llm_surgeon.surgery import calibrate, remove_layers
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        remove_layers(tiny_llama, [3, 4])
        text = " ".join([f"tok{i}" for i in range(4, 20)])
        with pytest.warns(UserWarning, match="baseline_stats"):
            calibrate(tiny_llama, tokenizer, text=text)

    def test_model_still_runs_after_calibration(self, tiny_llama):
        from llm_surgeon.surgery import calibrate, capture_calibration_stats, remove_layers
        from tests.conftest import _make_tiny_tokenizer
        tokenizer = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        text = " ".join([f"tok{i}" for i in range(4, 20)])
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=text)
        remove_layers(tiny_llama, [3, 4])
        calibrate(tiny_llama, tokenizer, baseline_stats=baseline, text=text)
        input_ids = torch.randint(0, tiny_llama.config.vocab_size, (1, 10))
        with torch.no_grad():
            output = tiny_llama(input_ids)
        assert output.logits.shape == (1, 10, tiny_llama.config.vocab_size)


class TestSaveReload:
    def test_modified_model_saves_and_reloads(self, tiny_llama, tmp_path):
        remove_layers(tiny_llama, [3, 4, 5])
        assert len(tiny_llama.model.layers) == 5
        save_path = str(tmp_path / "modified_model")
        tiny_llama.save_pretrained(save_path)
        reloaded = AutoModelForCausalLM.from_pretrained(save_path)
        assert len(reloaded.model.layers) == 5
        assert reloaded.config.num_hidden_layers == 5

    def test_reloaded_model_produces_output(self, tiny_llama, tmp_path):
        remove_layers(tiny_llama, [6, 7])
        save_path = str(tmp_path / "modified_model")
        tiny_llama.save_pretrained(save_path)
        reloaded = AutoModelForCausalLM.from_pretrained(save_path)
        input_ids = torch.randint(0, 64, (1, 10))
        with torch.no_grad():
            output = reloaded(input_ids)
        assert output.logits.shape == (1, 10, 64)

    def test_reloaded_weights_match(self, tiny_llama, tmp_path):
        w0_before = tiny_llama.model.layers[0].self_attn.q_proj.weight.data.clone()
        remove_layers(tiny_llama, [7])
        save_path = str(tmp_path / "modified_model")
        tiny_llama.save_pretrained(save_path)
        reloaded = AutoModelForCausalLM.from_pretrained(save_path)
        w0_after = reloaded.model.layers[0].self_attn.q_proj.weight.data
        assert torch.equal(w0_before, w0_after)
