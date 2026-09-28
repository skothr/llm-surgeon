"""Tests for verify module."""

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon.verify import (
    _compare_activation_lists,
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
        vocab_size=64,  # pyright: ignore[reportCallIssue]
        hidden_size=32,  # pyright: ignore[reportCallIssue]
        intermediate_size=64,  # pyright: ignore[reportCallIssue]
        num_hidden_layers=7,  # pyright: ignore[reportCallIssue]
        num_attention_heads=4,  # pyright: ignore[reportCallIssue]
        num_key_value_heads=4,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=128,  # pyright: ignore[reportCallIssue]
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
        remove_layers(tiny_llama, [6, 7])
        log2 = swap_layers(tiny_llama, 0, 5)
        report = check_structure(tiny_llama, log2)
        assert report.passed is True

    def test_combined_log_with_count_changes(self, tiny_llama):
        """A combined log (as recipe.run builds) whose layer count changes then returns."""
        combined = SurgeryLog()
        combined.ops.extend(remove_layers(tiny_llama, [0]).ops)  # 8 -> 7
        combined.ops.extend(duplicate_layer(tiny_llama, src=0, dst=1).ops)  # 7 -> 8
        report = check_structure(tiny_llama, combined)
        assert report.passed is True

    def test_catches_broken_op_chain(self, tiny_llama):
        remove_layers(tiny_llama, [0, 1])  # 8 -> 6
        log = SurgeryLog()
        log.add("remove_layers", "Removed 1", 8, 7)
        log.add("remove_layers", "Removed 1", 5, 6)  # before != previous after
        with pytest.raises(ValueError, match="surgery_log_chain"):
            check_structure(tiny_llama, log)

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
    def test_returns_7_entries_for_8_vs_7_layer(
        self, tiny_llama, tiny_llama_7layer, tokenizer
    ):
        result = compare_activations(
            tiny_llama, tiny_llama_7layer, tokenizer, "word4 word5 word6"
        )
        assert isinstance(result, list)
        assert len(result) == 7

    def test_entries_have_required_keys(self, tiny_llama, tiny_llama_7layer, tokenizer):
        result = compare_activations(
            tiny_llama, tiny_llama_7layer, tokenizer, "word4 word5 word6"
        )
        for entry in result:
            assert "layer" in entry
            assert "cosine_sim" in entry
            assert "l2_dist" in entry
            assert "max_abs_diff" in entry

    def test_identical_models_have_cosine_sim_near_1(self, tiny_llama, tokenizer):
        result = compare_activations(
            tiny_llama, tiny_llama, tokenizer, "word4 word5 word6"
        )
        for entry in result:
            assert abs(entry["cosine_sim"] - 1.0) < 1e-4, (
                f"Layer {entry['layer']} cosine_sim={entry['cosine_sim']}, expected ~1.0"
            )

    def test_layer_indices_sequential(self, tiny_llama, tiny_llama_7layer, tokenizer):
        result = compare_activations(
            tiny_llama, tiny_llama_7layer, tokenizer, "word4 word5 word6"
        )
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
            assert abs(entry["cosine_sim"] - 1.0) < 1e-4, (
                f"Layer {entry['layer']} cosine_sim={entry['cosine_sim']}, expected ~1.0"
            )

    def test_result_entries_match_compare_activations_structure(
        self, tiny_llama, tokenizer, tmp_path
    ):
        prompts = ["word4 word5"]
        cache_dir = str(tmp_path / "cache")
        cache_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        results = compare_to_baseline(tiny_llama, tokenizer, prompts, cache_dir)
        for entry in results[prompts[0]]:
            assert "layer" in entry
            assert "cosine_sim" in entry
            assert "l2_dist" in entry
            assert "max_abs_diff" in entry


# ---------------------------------------------------------------------------
# Audit fixes (issue #10)
# ---------------------------------------------------------------------------


class TestLayerMap:
    def test_layer_map_aligns_after_removal(self, tiny_llama, tokenizer):
        """With a layer_map, layers before the removed one match exactly and
        each later layer is compared with its own original index."""
        modified = copy.deepcopy(tiny_llama)
        remove_layers(modified, [3])
        # modified index -> original index
        layer_map = {i: (i if i < 3 else i + 1) for i in range(7)}
        result = compare_activations(
            tiny_llama, modified, tokenizer, "word4 word5 word6", layer_map=layer_map
        )
        assert [e["layer"] for e in result] == list(range(7))
        assert [e["original_layer"] for e in result] == [0, 1, 2, 4, 5, 6, 7]
        for e in result[:3]:
            assert e["max_abs_diff"] < 1e-5

    def test_layer_map_out_of_range_raises(self, tiny_llama, tokenizer):
        with pytest.raises(IndexError, match="layer_map"):
            compare_activations(
                tiny_llama, tiny_llama, tokenizer, "word4 word5", layer_map={0: 99}
            )

    def test_positional_entries_report_original_layer(self, tiny_llama, tokenizer):
        result = compare_activations(tiny_llama, tiny_llama, tokenizer, "word4 word5")
        assert [e["original_layer"] for e in result] == list(range(8))


class TestCompareMetrics:
    def test_l2_dist_is_per_token_mean(self):
        """l2_dist uses the same per-token convention as cosine_sim, so it does
        not grow with sequence length."""
        seq, hidden = 5, 4
        a = torch.zeros(1, seq, hidden)
        b = torch.zeros(1, seq, hidden)
        b[..., 0] = 3.0  # every token is 3.0 away
        a[..., 1] = 1.0
        b[..., 1] = 1.0
        [entry] = _compare_activation_lists([a], [b])
        assert entry["l2_dist"] == pytest.approx(3.0)
        assert entry["max_abs_diff"] == pytest.approx(3.0)


class TestMissingLayers:
    def test_capture_raises_when_a_layer_hook_never_fires(self, tiny_llama, tokenizer):
        # LlamaModel.forward runs layers[: config.num_hidden_layers], so layer 7
        # is skipped and its hook never fires.
        tiny_llama.config.num_hidden_layers = 7
        with pytest.raises(RuntimeError, match=r"\[7\]"):
            compare_activations(tiny_llama, tiny_llama, tokenizer, "word4 word5")


class TestBaselineCache:
    def test_cached_tensors_are_on_cpu(self, tiny_llama, tokenizer, tmp_path):
        cache_baseline(tiny_llama, tokenizer, ["word4 word5"], str(tmp_path))
        [path] = list(tmp_path.glob("*.pt"))
        payload = torch.load(path, weights_only=True)
        assert all(t.device.type == "cpu" for t in payload["activations"])

    def test_loads_with_map_location_cpu(
        self, tiny_llama, tokenizer, tmp_path, monkeypatch
    ):
        cache_baseline(tiny_llama, tokenizer, ["word4 word5"], str(tmp_path))
        seen = {}
        real_load = torch.load

        def spy_load(*args, **kwargs):
            seen.update(kwargs)
            return real_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", spy_load)
        compare_to_baseline(tiny_llama, tokenizer, ["word4 word5"], str(tmp_path))
        assert seen.get("map_location") == "cpu"
        assert seen.get("weights_only") is True

    def test_rejects_baseline_from_different_hidden_size(
        self, tiny_llama, tokenizer, tmp_path
    ):
        cache_baseline(tiny_llama, tokenizer, ["word4 word5"], str(tmp_path))
        other = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,  # pyright: ignore[reportCallIssue]
                hidden_size=16,  # pyright: ignore[reportCallIssue]
                intermediate_size=32,  # pyright: ignore[reportCallIssue]
                num_hidden_layers=2,  # pyright: ignore[reportCallIssue]
                num_attention_heads=4,  # pyright: ignore[reportCallIssue]
                num_key_value_heads=4,  # pyright: ignore[reportCallIssue]
            )
        ).eval()
        with pytest.raises(ValueError, match="hidden_size"):
            compare_to_baseline(other, tokenizer, ["word4 word5"], str(tmp_path))

    def test_rejects_baseline_from_different_tokenization(
        self, tiny_llama, tokenizer, tmp_path
    ):
        prompt = "word4 word62"
        cache_baseline(tiny_llama, tokenizer, [prompt], str(tmp_path))
        other_tok = _make_tiny_tokenizer(60)  # word62 -> [UNK]
        with pytest.raises(ValueError, match="token"):
            compare_to_baseline(tiny_llama, other_tok, [prompt], str(tmp_path))

    def test_accepts_legacy_list_cache(self, tiny_llama, tokenizer, tmp_path):
        prompt = "word4 word5"
        cache_baseline(tiny_llama, tokenizer, [prompt], str(tmp_path))
        [path] = list(tmp_path.glob("*.pt"))
        payload = torch.load(path, weights_only=True)
        torch.save(payload["activations"], path)  # pre-metadata format
        results = compare_to_baseline(tiny_llama, tokenizer, [prompt], str(tmp_path))
        assert len(results[prompt]) == 8
