"""Tests for probe.activation_patch — causal attribution via clean/corrupted counterfactual."""

import pytest  # pyright: ignore[reportUnusedImport]
import torch

from llm_surgeon.probe import _make_position_patch


class TestMakePositionPatch:
    def test_only_replaces_target_position(self):
        # seq_len=5, d_model=4
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        patch_vec = torch.tensor([100.0, 200.0, 300.0, 400.0])
        fn = _make_position_patch(pos=2, clean_vec=patch_vec)
        out = fn(hidden, layer_idx=0)
        # Position 2 must equal patch_vec.
        assert torch.equal(out[2], patch_vec)
        # All other positions unchanged.
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
