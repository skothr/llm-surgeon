"""Tests for probe.activation_patch — causal attribution via clean/corrupted counterfactual."""

import pytest
import torch

from llm_surgeon.probe import _make_position_patch


class TestMakePositionPatch:
    def test_only_replaces_target_position(self):
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        patch_vec = torch.tensor([100.0, 200.0, 300.0, 400.0])
        fn = _make_position_patch(pos=2, clean_vec=patch_vec)
        out = fn(hidden, layer_idx=0)
        assert torch.equal(out[2], patch_vec)
        for pos in (0, 1, 3, 4):
            assert torch.equal(out[pos], hidden[pos]), f"position {pos} was modified"

    def test_preserves_dtype_and_device(self):
        hidden = torch.randn(3, 8, dtype=torch.float16)
        # Patch vec in a different dtype — op must cast to match hidden.
        patch_vec = torch.randn(8, dtype=torch.float32)
        fn = _make_position_patch(pos=1, clean_vec=patch_vec)
        out = fn(hidden, layer_idx=0)
        assert out.dtype == torch.float16
        assert out.device == hidden.device

    def test_does_not_mutate_input(self):
        hidden = torch.randn(4, 6)
        original = hidden.clone()
        patch_vec = torch.zeros(6)
        fn = _make_position_patch(pos=0, clean_vec=patch_vec)
        fn(hidden, layer_idx=0)
        assert torch.equal(hidden, original), "input hidden tensor was mutated"

    def test_repr_is_descriptive(self):
        fn = _make_position_patch(pos=3, clean_vec=torch.zeros(4))
        assert "patch_pos(3)" in repr(fn)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """activation_patch input validation — fails fast, no model needed."""

    @pytest.fixture
    def tokenizer(self):
        from tests.conftest import _make_tiny_tokenizer
        return _make_tiny_tokenizer(64)

    def test_mismatched_lengths_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match=r"same length.*clean=\d+.*corrupted=\d+"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11 word12",
                corrupted_prompt="word10 word11",
            )

    def test_empty_clean_prompt_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="empty"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="", corrupted_prompt="word10 word11",
            )

    def test_empty_corrupted_prompt_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="empty"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="",
            )

    def test_bad_direction_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="direction must be 'denoise' or 'noise'"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="word12 word13",
                direction="wobble",  # pyright: ignore[reportArgumentType]
            )

    def test_bad_sublayer_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(ValueError, match="sublayers must be subset"):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="word12 word13",
                sublayers=("mlp",),
            )

    def test_measurement_pos_out_of_range_raises(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        with pytest.raises(IndexError):
            activation_patch(
                tiny_llama, tokenizer,
                clean_prompt="word10 word11", corrupted_prompt="word12 word13",
                measurement_position=100,
            )


# ---------------------------------------------------------------------------
# PatchingResult dataclass
# ---------------------------------------------------------------------------

class TestPatchingResult:
    def test_construction_and_fields(self):
        from llm_surgeon.probe import PatchingResult
        result = PatchingResult(
            cells=[],
            clean_baseline_logits=torch.zeros(10),
            corrupted_baseline_logits=torch.zeros(10),
            prompt_tokens_clean=["a", "b"],
            prompt_tokens_corrupted=["c", "d"],
            direction="denoise",
            measurement_position=1,
        )
        assert result.direction == "denoise"
        assert result.measurement_position == 1
        assert len(result.cells) == 0
