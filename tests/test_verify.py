"""Tests for verify module."""

import pytest
import torch
from llm_surgeon.verify import VerifyReport, check_structure
from llm_surgeon.surgery import (
    SurgeryLog,
    remove_layers,
    keep_layers,
    swap_layers,
    duplicate_layer,
    reorder_layers,
)


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
