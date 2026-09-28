"""calibrate(): baseline pairing after structural surgery, and corpus handling."""

import copy

import pytest
import torch

from llm_surgeon.surgery import (
    CalibrationStats,
    _capture_norm_outputs,
    calibrate,
    capture_calibration_stats,
    duplicate_layer,
    remove_layers,
    reorder_layers,
)
from tests.conftest import _make_tiny_tokenizer

TEXT = " ".join(f"word{i}" for i in range(4, 60))


@pytest.fixture
def tokenizer(tiny_llama):
    return _make_tiny_tokenizer(tiny_llama.config.vocab_size)


def _expected_scale(base_ms, cur_ms, scale_clip=5.0, min_variance=1e-6):
    valid = (base_ms > min_variance) & (cur_ms > min_variance)
    raw = torch.where(
        valid,
        torch.sqrt(base_ms / cur_ms.clamp_min(min_variance)),
        torch.ones_like(base_ms),
    )
    return raw.clamp(1.0 / scale_clip, scale_clip)


def _assert_scaled_toward(model, baseline, origins, tokenizer):
    """Calibrate ``model`` and check layer i was scaled toward baseline[origins[i]]."""
    current = _capture_norm_outputs(copy.deepcopy(model), tokenizer, text=TEXT)
    before = [
        (l.input_layernorm.weight.clone(), l.post_attention_layernorm.weight.clone())
        for l in model.model.layers
    ]
    calibrate(model, tokenizer, baseline_stats=baseline, text=TEXT)
    for i, origin in enumerate(origins):
        layer = model.model.layers[i]
        for norm, w0, base, cur in (
            (
                layer.input_layernorm,
                before[i][0],
                baseline.input_norm,
                current.input_norm,
            ),
            (
                layer.post_attention_layernorm,
                before[i][1],
                baseline.post_attn_norm,
                current.post_attn_norm,
            ),
        ):
            expected = w0 * _expected_scale(base[origin], cur[i])
            assert torch.allclose(norm.weight, expected, rtol=1e-5), (
                f"layer {i} (originally {origin}) not scaled toward baseline {origin}"
            )


class TestDefaultLayerMap:
    def test_follows_removed_layers(self, tiny_llama, tokenizer):
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=TEXT)
        remove_layers(tiny_llama, [3, 4])
        _assert_scaled_toward(tiny_llama, baseline, [0, 1, 2, 5, 6, 7], tokenizer)

    def test_follows_reordered_layers(self, tiny_llama, tokenizer):
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=TEXT)
        order = [7, 6, 5, 4, 3, 2, 1, 0]
        reorder_layers(tiny_llama, order)
        _assert_scaled_toward(tiny_llama, baseline, order, tokenizer)

    def test_duplicated_layer_uses_source_stats(self, tiny_llama, tokenizer):
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=TEXT)
        duplicate_layer(tiny_llama, src=2, dst=3)
        _assert_scaled_toward(
            tiny_llama, baseline, [0, 1, 2, 2, 3, 4, 5, 6, 7], tokenizer
        )

    def test_baseline_captured_after_earlier_surgery(self, tiny_llama, tokenizer):
        remove_layers(tiny_llama, [0])  # layers now originally 1..7
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=TEXT)
        assert baseline.layer_origins == [1, 2, 3, 4, 5, 6, 7]
        remove_layers(tiny_llama, [1])  # drops original layer 2
        # Baseline index of original layers 1, 3, 4, ... is 0, 2, 3, ...
        _assert_scaled_toward(tiny_llama, baseline, [0, 2, 3, 4, 5, 6], tokenizer)


class TestArgumentValidation:
    def test_missing_baseline_raises(self, tiny_llama, tokenizer):
        with pytest.raises(ValueError, match="baseline_stats"):
            calibrate(tiny_llama, tokenizer, text=TEXT)

    def test_negative_layer_map_entry_raises(self, tiny_llama, tokenizer):
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=TEXT)
        with pytest.raises(ValueError, match="layer_map"):
            calibrate(
                tiny_llama,
                tokenizer,
                baseline_stats=baseline,
                layer_map=[-1, 0, 1, 2, 3, 4, 5, 6],
                text=TEXT,
            )

    def test_unsupported_dataset_raises(self, tiny_llama, tokenizer):
        with pytest.raises(ValueError, match="Unsupported calibration dataset"):
            capture_calibration_stats(tiny_llama, tokenizer, dataset="c4")


class TestCorpus:
    @staticmethod
    def _count_forwards(model):
        calls = []
        handle = model.model.register_forward_pre_hook(
            lambda _m, args, kwargs: calls.append(1), with_kwargs=True
        )
        return calls, handle

    def test_num_samples_sets_sequence_count(self, tiny_llama, tokenizer, monkeypatch):
        import datasets

        lines = ["", " = Heading = \n"] + [
            " ".join(f"word{4 + (i + j) % 50}" for j in range(40)) + "\n"
            for i in range(40)
        ]
        requested = []

        def fake_load_dataset(*args, **kwargs):
            requested.append((args, kwargs))
            return {"text": lines}

        monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
        calls, handle = self._count_forwards(tiny_llama)
        try:
            capture_calibration_stats(tiny_llama, tokenizer, num_samples=3)
        finally:
            handle.remove()
        assert requested and requested[0][1].get("split") == "train"
        assert len(calls) == 3

    def test_long_text_is_not_truncated_to_one_sequence(self, tiny_llama, tokenizer):
        long_text = " ".join(
            f"word{4 + i % 50}" for i in range(300)
        )  # > 128-token context
        calls, handle = self._count_forwards(tiny_llama)
        try:
            capture_calibration_stats(
                tiny_llama, tokenizer, text=long_text, num_samples=8
            )
        finally:
            handle.remove()
        assert len(calls) == 3  # ceil(300 / 128)

    def test_stats_are_token_weighted_mean_over_sequences(self, tiny_llama, tokenizer):
        long_text = " ".join(f"word{4 + i % 50}" for i in range(256))
        stats = capture_calibration_stats(tiny_llama, tokenizer, text=long_text)
        first = capture_calibration_stats(
            tiny_llama, tokenizer, text=" ".join(long_text.split()[:128])
        )
        second = capture_calibration_stats(
            tiny_llama, tokenizer, text=" ".join(long_text.split()[128:])
        )
        for a, b, both in zip(first.input_norm, second.input_norm, stats.input_norm):
            assert torch.allclose(both, (a + b) / 2, rtol=1e-4)

    def test_restores_training_mode(self, tiny_llama, tokenizer):
        tiny_llama.train()
        capture_calibration_stats(tiny_llama, tokenizer, text=TEXT)
        assert tiny_llama.training


class TestReport:
    def test_fully_skipped_norm_not_counted_as_calibrated(self, tiny_llama, tokenizer):
        baseline = capture_calibration_stats(tiny_llama, tokenizer, text=TEXT)
        hidden = tiny_llama.config.hidden_size
        poisoned = CalibrationStats(
            input_norm=[torch.zeros(hidden)] + list(baseline.input_norm[1:]),
            post_attn_norm=list(baseline.post_attn_norm),
        )
        with pytest.warns(UserWarning, match="fully skipped"):
            report = calibrate(
                tiny_llama, tokenizer, baseline_stats=poisoned, text=TEXT
            )
        assert report.layers_calibrated == 2 * 8 - 1
