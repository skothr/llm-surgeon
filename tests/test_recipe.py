"""Tests for recipe module."""

import pytest
import yaml

from llm_surgeon.recipe import parse_recipe, run, generate_layer_sweep


class TestParsesYaml:
    def test_parses_yaml(self, tmp_path):
        recipe_file = tmp_path / "test.yaml"
        recipe_file.write_text(yaml.dump({
            "name": "remove-3-4",
            "base_model": "TinyLlama/TinyLlama-1.1B",
            "description": "Remove layers 3-4",
            "surgery": [{"remove_layers": [3, 4]}],
        }))
        result = parse_recipe(str(recipe_file))
        assert result["name"] == "remove-3-4"
        assert result["base_model"] == "TinyLlama/TinyLlama-1.1B"
        assert result["surgery"] == [{"remove_layers": [3, 4]}]


class TestValidatesRequiredFields:
    def test_missing_name_raises(self, tmp_path):
        recipe_file = tmp_path / "bad.yaml"
        recipe_file.write_text(yaml.dump({"base_model": "TinyLlama"}))
        with pytest.raises(ValueError, match="name"):
            parse_recipe(str(recipe_file))

    def test_missing_base_model_raises(self, tmp_path):
        recipe_file = tmp_path / "bad.yaml"
        recipe_file.write_text(yaml.dump({"name": "test"}))
        with pytest.raises(ValueError, match="base_model"):
            parse_recipe(str(recipe_file))


class TestGenerateLayerSweep:
    def test_generates_correct_count(self, tmp_path):
        files = generate_layer_sweep(num_layers=8, base_model="tiny", output_dir=str(tmp_path))
        assert len(files) == 8

    def test_each_file_removes_one_layer(self, tmp_path):
        files = generate_layer_sweep(num_layers=8, base_model="tiny", output_dir=str(tmp_path))
        for i, fpath in enumerate(files):
            with open(fpath) as f:
                data = yaml.safe_load(f)
            # Each recipe should remove exactly one layer
            remove_steps = [
                step for step in data.get("surgery", [])
                if "remove_layers" in step
            ]
            assert len(remove_steps) == 1, f"File {fpath} should have one remove_layers step"
            assert remove_steps[0]["remove_layers"] == [i], (
                f"File {i} should remove layer {i}, got {remove_steps[0]['remove_layers']}"
            )

    def test_files_have_required_fields(self, tmp_path):
        files = generate_layer_sweep(num_layers=4, base_model="my-model", output_dir=str(tmp_path))
        for fpath in files:
            with open(fpath) as f:
                data = yaml.safe_load(f)
            assert "name" in data
            assert "base_model" in data
            assert data["base_model"] == "my-model"


class TestRunSurgeryOnly:
    def test_run_surgery_only(self, tiny_llama, tmp_path):
        """run() with skip_export=True, skip_eval=True applies surgery to provided model."""
        recipe_file = tmp_path / "surgery_only.yaml"
        recipe_file.write_text(yaml.dump({
            "name": "test-run-surgery",
            "base_model": "tiny",
            "description": "Remove layer 0",
            "surgery": [{"remove_layers": [0]}],
        }))
        db = str(tmp_path / "exp.db")
        result = run(
            str(recipe_file),
            model=tiny_llama,
            skip_export=True,
            skip_eval=True,
            db_path=db,
        )
        # Layer count should have decreased
        assert len(tiny_llama.model.layers) == 7
        assert result is not None

    def test_run_multiple_surgery_steps(self, tiny_llama, tmp_path):
        """run() applies multiple surgery steps in order."""
        recipe_file = tmp_path / "multi_step.yaml"
        recipe_file.write_text(yaml.dump({
            "name": "test-multi-step",
            "base_model": "tiny",
            "surgery": [
                {"remove_layers": [0]},
                {"remove_layers": [0]},
            ],
        }))
        db = str(tmp_path / "exp.db")
        run(
            str(recipe_file),
            model=tiny_llama,
            skip_export=True,
            skip_eval=True,
            db_path=db,
        )
        assert len(tiny_llama.model.layers) == 6


class TestRunWithAnalyze:
    def test_run_with_analyze(self, tiny_llama, tmp_path):
        """Recipe with an analyze section runs logit_lens and hidden_states."""
        from tests.conftest import _make_tiny_tokenizer

        tokenizer = _make_tiny_tokenizer(tiny_llama.config.vocab_size)
        db_path = str(tmp_path / "test.db")

        recipe_data = {
            "name": "test-analyze",
            "base_model": "test",
            "surgery": [{"remove_layers": [7]}],
            "analyze": {
                "logit_lens": {"prompt": "word4 word5 word6", "top_k": 3},
                "hidden_states": {"prompt": "word4 word5 word6"},
            },
        }
        recipe_path = str(tmp_path / "analyze.yaml")
        with open(recipe_path, "w") as f:
            yaml.dump(recipe_data, f)

        result = run(
            recipe_path,
            model=tiny_llama,
            tokenizer=tokenizer,
            db_path=db_path,
            skip_export=True,
            skip_eval=True,
        )
        assert result["status"] == "completed"
        assert "analyze" in result
        assert "logit_lens" in result["analyze"]
        assert "hidden_states" in result["analyze"]
