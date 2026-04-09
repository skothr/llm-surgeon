"""Tests for verify module."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon.verify import (
    VerifyReport,
    check_structure,
    compare_activations,
    cache_baseline,
    compare_to_baseline,
)
from llm_surgeon.surgery import (
    SurgeryLog,
    remove_layers,
    keep_layers,
    swap_layers,
    duplicate_layer,
    reorder_layers,
)
from tests.conftest import _make_tiny_tokenizer


@pytest.fixture
def tokenizer():
    return _make_tiny_tokenizer(64)


@pytest.fixture
def tiny_llama_7layer():
    """7-layer LLaMA model (one fewer than tiny_llama) for comparison tests."""
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=7,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


class TestVerifyReport:
    def test_starts_passed(self):
        report = VerifyReport()
        assert report.passed is True

    def test_add_passing_check(self):
        report = VerifyReport()
        report.add_check("test_check", True, "all good")
        assert report.passed is True
        assert len(report.checks) == 1

    def test_add_failing_check_sets_failed(self):
        report = VerifyReport()
        report.add_check("test_check", False, "mismatch")
        assert report.passed is False

    def test_str_shows_status(self):
        report = VerifyReport()
        report.add_check("check1", True, "ok")
        s = str(report)
        assert "PASSED" in s

    def test_str_shows_failed(self):
        report = VerifyReport()
        report.add_check("check1", False, "bad")
        s = str(report)
        assert "FAILED" in s


class TestCheckStructure:
    def test_passes_on_unmodified_model(self, tiny_llama):
        report = check_structure(tiny_llama)
        assert report.passed is True

    def test_passes_after_remove_layers(self, tiny_llama):
        log = remove_layers(tiny_llama, [3, 4, 5])
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_passes_after_keep_layers(self, tiny_llama):
        log = keep_layers(tiny_llama, [0, 1, 7])
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_passes_after_swap(self, tiny_llama):
        log = swap_layers(tiny_llama, 0, 7)
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_passes_after_duplicate(self, tiny_llama):
        log = duplicate_layer(tiny_llama, src=0, dst=1)
        report = check_structure(tiny_llama, log)
        assert report.passed is True

    def test_catches_config_mismatch(self, tiny_llama):
        remove_layers(tiny_llama, [0])
        tiny_llama.config.num_hidden_layers = 999
        with pytest.raises(ValueError, match="Structural verification failed"):
            check_structure(tiny_llama)

    def test_catches_surgery_log_mismatch(self, tiny_llama):
        remove_layers(tiny_llama, [0])
        fake_log = SurgeryLog()
        fake_log.add("remove_layers", "Removed 3 layers", 8, 5)
        with pytest.raises(ValueError, match="Structural verification failed"):
            check_structure(tiny_llama, fake_log)

    def test_no_surgery_log_still_validates(self, tiny_llama):
        remove_layers(tiny_llama, [0, 1])
        report = check_structure(tiny_llama)
        assert report.passed is True

    def test_checks_embedding_consistency(self, tiny_llama):
        report = check_structure(tiny_llama)
        check_names = [c["name"] for c in report.checks]
        assert "embedding_dim_consistent" in check_names
        assert "lm_head_vocab_consistent" in check_names
        assert "lm_head_hidden_consistent" in check_names


class TestCheckStructureChained:
    def test_verify_after_multiple_ops(self, tiny_llama):
        log1 = remove_layers(tiny_llama, [6, 7])
        log2 = swap_layers(tiny_llama, 0, 5)
        report = check_structure(tiny_llama, log2)
        assert report.passed is True

    def test_verify_no_log_after_chain(self, tiny_llama):
        remove_layers(tiny_llama, [0, 1])
        swap_layers(tiny_llama, 0, 5)
        reorder_layers(tiny_llama, list(range(5, -1, -1)))
        report = check_structure(tiny_llama)
        assert report.passed is True


# ---------------------------------------------------------------------------
# Task 4: compare_activations, cache_baseline, compare_to_baseline
# ---------------------------------------------------------------------------

class TestCompareActivations:
    def test_returns_7_entries_for_8_vs_7_layer(self, tiny_llama, tiny_llama_7layer, tokenizer):
        result = compare_activations(tiny_llama, tiny_llama_7layer, tokenizer, "word4 word5 word6")
        assert isinstance(result, list)
        assert len(result) == 7

    def test_entries_have_required_keys(self, tiny_llama, tiny_llama_7layer, tokenizer):
        result = compare_activations(tiny_llama, tiny_llama_7layer, tokenizer, "word4 word5 word6")
        for entry in result:
            assert "layer" in entry
            assert "cosine_sim" in entry
            assert "l2_dist" in entry
            assert "max_abs_diff" in entry

    def test_identical_models_have_cosine_sim_near_1(self, tiny_llama, tokenizer):
        result = compare_activations(tiny_llama, tiny_llama, tokenizer, "word4 word5 word6")
        for entry in result:
            assert abs(entry["cosine_sim"] - 1.0) < 1e-4, \
                f"Layer {entry['layer']} cosine_sim={entry['cosine_sim']}, expected ~1.0"

    def test_layer_indices_sequential(self, tiny_llama, tiny_llama_7layer, tokenizer):
        result = compare_activations(tiny_llama, tiny_llama_7layer, tokenizer, "word4 word5 word6")
        for i, entry in enumerate(result):
            assert entry["layer"] == i


class TestCacheBaseline:
    def test_creates_pt_files(self, tiny_llama, tokenizer, tmp_path):
        prompts = ["word4 word5", "word6 word7 word8"]
        cache_dir = str(tmp_path / "cache")
        cache_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        import os
        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        assert len(pt_files) == len(prompts)

    def test_cache_dir_created_if_missing(self, tiny_llama, tokenizer, tmp_path):
        import os
        cache_dir = str(tmp_path / "new_cache" / "subdir")
        assert not os.path.exists(cache_dir)
        cache_baseline(tiny_llama, tokenizer, ["word4 word5"], cache_dir)
        assert os.path.exists(cache_dir)


class TestCompareToBaseline:
    def test_returns_results_for_each_prompt(self, tiny_llama, tokenizer, tmp_path):
        prompts = ["word4 word5", "word6 word7 word8"]
        cache_dir = str(tmp_path / "cache")
        cache_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        results = compare_to_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        assert set(results.keys()) == set(prompts)

    def test_identical_model_has_high_cosine_sim(self, tiny_llama, tokenizer, tmp_path):
        prompts = ["word4 word5 word6"]
        cache_dir = str(tmp_path / "cache")
        cache_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        results = compare_to_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        for entry in results[prompts[0]]:
            assert abs(entry["cosine_sim"] - 1.0) < 1e-4, \
                f"Layer {entry['layer']} cosine_sim={entry['cosine_sim']}, expected ~1.0"

    def test_result_entries_match_compare_activations_structure(self, tiny_llama, tokenizer, tmp_path):
        prompts = ["word4 word5"]
        cache_dir = str(tmp_path / "cache")
        cache_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        results = compare_to_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        for entry in results[prompts[0]]:
            assert "layer" in entry
            assert "cosine_sim" in entry
            assert "l2_dist" in entry
            assert "max_abs_diff" in entry
