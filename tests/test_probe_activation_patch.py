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


# ---------------------------------------------------------------------------
# Core algorithm — tiny_llama fixture (8 layers, 32 hidden)
# ---------------------------------------------------------------------------

class TestActivationPatchLoop:
    @pytest.fixture
    def tokenizer(self):
        from tests.conftest import _make_tiny_tokenizer
        return _make_tiny_tokenizer(64)

    def test_callback_fires_per_cell(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        calls = []
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11 word12",
            corrupted_prompt="word20 word21 word22",
            direction="denoise",
            on_cell=lambda L, sub, pos, cell: calls.append((L, sub, pos)),
        )
        # 8 layers × 2 sublayers × 3 positions = 48 cells.
        assert len(calls) == 48
        assert len(result.cells) == 48
        # Every triple is unique.
        assert len(set(calls)) == 48

    def test_layer_major_position_minor_order(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        calls = []
        activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11 word12",
            corrupted_prompt="word20 word21 word22",
            on_cell=lambda L, sub, pos, cell: calls.append((L, sub, pos)),
        )
        # First layer is 0. First sublayer is 'attn'. Positions increment.
        assert calls[0][0] == 0
        # Same-(L,sub) cells stream consecutively before advancing.
        same_block = [c for c in calls[:3]]
        assert all(c[0] == same_block[0][0] and c[1] == same_block[0][1] for c in same_block)

    def test_direction_denoise_base_is_corrupted(self, tiny_llama, tokenizer, monkeypatch):
        # Spy on intervene() to confirm it was called with corrupted_prompt.
        from llm_surgeon import probe
        original_intervene = probe.intervene
        received_prompts = []

        def spy(model, tok, prompt, interventions, **kwargs):
            received_prompts.append(prompt)
            return original_intervene(model, tok, prompt, interventions, **kwargs)

        monkeypatch.setattr(probe, "intervene", spy)
        probe.activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            direction="denoise",
        )
        # Every intervene() call during the loop used the corrupted prompt.
        assert all(p == "word20 word21" for p in received_prompts)
        assert len(received_prompts) == 8 * 2 * 2  # L × sub × pos

    def test_direction_noise_base_is_clean(self, tiny_llama, tokenizer, monkeypatch):
        from llm_surgeon import probe
        original_intervene = probe.intervene
        received_prompts = []

        def spy(model, tok, prompt, interventions, **kwargs):
            received_prompts.append(prompt)
            return original_intervene(model, tok, prompt, interventions, **kwargs)

        monkeypatch.setattr(probe, "intervene", spy)
        probe.activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            direction="noise",
        )
        assert all(p == "word10 word11" for p in received_prompts)
        assert len(received_prompts) == 8 * 2 * 2  # L × sub × pos

    def test_positions_subset_filters_loop(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11 word12",
            corrupted_prompt="word20 word21 word22",
            positions=[0, 2],
        )
        assert len(result.cells) == 8 * 2 * 2
        positions_seen = {cell["position"] for cell in result.cells}
        assert positions_seen == {0, 2}

    def test_layers_subset_filters_loop(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            layers=[3, 5],
        )
        layers_seen = {cell["layer"] for cell in result.cells}
        assert layers_seen == {3, 5}

    def test_sublayers_subset_filters_loop(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
            sublayers=("ffn",),
        )
        subs_seen = {cell["sublayer"] for cell in result.cells}
        assert subs_seen == {"ffn"}

    def test_cells_have_patched_logits(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
        )
        for cell in result.cells:
            assert "patched_logits" in cell
            assert cell["patched_logits"].shape == (tiny_llama.config.vocab_size,)

    def test_baselines_are_populated(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        result = activation_patch(
            tiny_llama, tokenizer,
            clean_prompt="word10 word11",
            corrupted_prompt="word20 word21",
        )
        assert result.clean_baseline_logits.shape == (tiny_llama.config.vocab_size,)
        assert result.corrupted_baseline_logits.shape == (tiny_llama.config.vocab_size,)

    def test_quantized_model_emits_warning(self, tiny_llama, tokenizer):
        from llm_surgeon.probe import activation_patch
        tiny_llama.hf_quantizer = object()  # type: ignore[attr-defined]
        try:
            with pytest.warns(RuntimeWarning, match="quantized"):
                activation_patch(
                    tiny_llama, tokenizer,
                    clean_prompt="word10 word11",
                    corrupted_prompt="word20 word21",
                )
        finally:
            del tiny_llama.hf_quantizer  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Integration — real TinyLlama on CUDA
# ---------------------------------------------------------------------------

class TestActivationPatchIntegration:
    """End-to-end: real TinyLlama fp16 → activation patch → sanity check."""

    def test_tinyllama_capital_swap(self):
        """Denoise recovery should be larger at late layers than early.

        Intuition: information about the country aggregates through the
        stack. Patching a late-layer residual state with clean activations
        flips the output back toward the clean answer; patching an early
        layer — before the relevant facts have been integrated — does
        comparatively little.
        """
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import activation_patch

        model, tokenizer = load_model(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16",
        )
        num_layers = len(model.model.layers)

        result = activation_patch(
            model, tokenizer,
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Italy is",
            direction="denoise",
        )

        positions = {c["position"] for c in result.cells}
        assert len(result.cells) == num_layers * 2 * len(positions)

        clean_id = int(result.clean_baseline_logits.argmax().item())
        corr_id = int(result.corrupted_baseline_logits.argmax().item())
        delta_clean = (result.clean_baseline_logits[clean_id] - result.clean_baseline_logits[corr_id]).item()
        delta_corr = (result.corrupted_baseline_logits[clean_id] - result.corrupted_baseline_logits[corr_id]).item()
        denom = delta_clean - delta_corr
        # denom should be >0 — the clean forward actually prefers clean-top-1
        # over corrupted-top-1. If not, the model didn't learn the contrast
        # and this test prompt is unsuitable.
        assert denom > 0, f"bad prompt pair: clean/corrupt deltas collapse (denom={denom})"

        last_pos = max(positions)
        recovery_by_layer = {}
        for cell in result.cells:
            if cell["position"] != last_pos or cell["sublayer"] != "ffn":
                continue
            patched = cell["patched_logits"]
            delta_patched = (patched[clean_id] - patched[corr_id]).item()
            recovery = (delta_patched - delta_corr) / denom
            recovery_by_layer[cell["layer"]] = recovery

        early_mean = sum(recovery_by_layer[L] for L in range(5)) / 5
        late_mean = sum(recovery_by_layer[L] for L in range(num_layers - 5, num_layers)) / 5

        # Loose threshold — point is "we didn't break the algorithm,"
        # not pin an exact numeric curve.
        assert late_mean > early_mean + 0.1, (
            f"expected late-layer recovery to exceed early by ≥0.1, "
            f"got early={early_mean:.3f} late={late_mean:.3f}"
        )
