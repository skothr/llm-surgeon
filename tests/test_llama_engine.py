import numpy as np
import pytest
from llm_surgeon.llama_engine import GenerateStep, compare_logits


class TestCompareLogits:
    def test_identical_logits(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = compare_logits(a, a)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert result["kl_divergence"] == pytest.approx(0.0, abs=1e-6)
        assert result["max_logit_diff"] == pytest.approx(0.0, abs=1e-6)
        assert result["top_k_agreement"] == 4

    def test_different_logits(self):
        a = np.array([10.0, 1.0, 0.5, 0.1], dtype=np.float32)
        b = np.array([0.1, 0.5, 1.0, 10.0], dtype=np.float32)
        result = compare_logits(a, b)
        assert result["cosine_similarity"] < 0.5
        assert result["kl_divergence"] > 1.0
        assert result["top_k_agreement"] >= 0

    def test_top_k_agreement_partial(self):
        a = np.array([10.0, 5.0, 1.0, 0.1], dtype=np.float32)
        b = np.array([10.0, 4.0, 0.5, 5.0], dtype=np.float32)
        result = compare_logits(a, b, top_k=2)
        assert result["top_k_agreement"] == 1

    def test_returns_all_keys(self):
        a = np.ones(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        result = compare_logits(a, b)
        assert set(result.keys()) == {
            "cosine_similarity", "kl_divergence",
            "max_logit_diff", "mean_logit_diff", "top_k_agreement",
        }

    def test_cosine_invariant_to_constant_offset(self):
        # softmax is invariant to a constant offset, so logit vectors
        # differing by an additive constant represent identical
        # distributions. Cosine similarity should reflect that by
        # mean-centering before the dot product.
        rng = np.random.default_rng(0)
        a = rng.standard_normal(2048).astype(np.float32)
        b = a + 1000.0
        result = compare_logits(a, b)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-4)
        assert result["kl_divergence"] == pytest.approx(0.0, abs=1e-5)


class TestGenerateStep:
    def test_fields(self):
        step = GenerateStep(token_id=42, token_str="hello", logits=None)
        assert step.token_id == 42
        assert step.token_str == "hello"
        assert step.logits is None

    def test_with_logits(self):
        logits = np.zeros(100, dtype=np.float32)
        step = GenerateStep(token_id=1, token_str="a", logits=logits)
        assert step.logits is not None
        assert step.logits.shape == (100,)


from pathlib import Path
from llm_surgeon.llama_engine import LlamaEngine
from llm_surgeon.gguf_reader import resolve_ollama_blob

OLLAMA_DIR = Path("/usr/share/ollama/.ollama/models")
TINYLLAMA_EXISTS = (
    OLLAMA_DIR / "manifests/registry.ollama.ai/library/tinyllama/latest"
).exists()


def _tinyllama_blob() -> Path:
    blob = resolve_ollama_blob("tinyllama:latest")
    assert blob is not None, "TINYLLAMA_EXISTS guard failed to prevent None blob"
    return blob


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestLlamaEngineCore:
    @pytest.fixture
    def engine(self):
        blob = _tinyllama_blob()
        eng = LlamaEngine(blob, n_ctx=128)
        yield eng
        eng.close()

    def test_is_loaded(self, engine):
        assert engine.is_loaded

    def test_close(self):
        blob = _tinyllama_blob()
        eng = LlamaEngine(blob, n_ctx=128)
        eng.close()
        assert not eng.is_loaded

    def test_context_manager(self):
        blob = _tinyllama_blob()
        with LlamaEngine(blob, n_ctx=128) as eng:
            assert eng.is_loaded
        assert not eng.is_loaded

    def test_tokenize(self, engine):
        tokens = engine.tokenize("Hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert tokens[0] == 1  # BOS token

    def test_tokenize_no_bos(self, engine):
        tokens = engine.tokenize("Hello world", add_bos=False)
        assert tokens[0] != 1

    def test_detokenize(self, engine):
        tokens = engine.tokenize("Hello world")
        text = engine.detokenize(tokens)
        assert "Hello" in text
        assert "world" in text

    def test_n_vocab(self, engine):
        assert engine.n_vocab == 32000


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestLlamaEngineLogits:
    @pytest.fixture(scope="class")
    def engine(self):
        blob = _tinyllama_blob()
        eng = LlamaEngine(blob, n_ctx=128)
        yield eng
        eng.close()

    def test_logits_shape(self, engine):
        tokens = engine.tokenize("The capital of France is")
        result = engine.logits(tokens)
        assert isinstance(result, np.ndarray)
        assert result.shape == (32000,)
        assert result.dtype == np.float32

    def test_logits_predicts_paris(self, engine):
        tokens = engine.tokenize("The capital of France is")
        result = engine.logits(tokens)
        top_id = int(np.argmax(result))
        top_token = engine.detokenize([top_id])
        assert "Paris" in top_token

    def test_logits_all_shape(self, engine):
        tokens = engine.tokenize("Hello world")
        result = engine.logits_all(tokens)
        assert isinstance(result, list)
        assert len(result) == len(tokens)
        assert all(arr.shape == (32000,) for arr in result)

    def test_logits_all_last_matches_logits(self, engine):
        tokens = engine.tokenize("Test prompt")
        single = engine.logits(tokens)
        all_logits = engine.logits_all(tokens)
        np.testing.assert_allclose(single, all_logits[-1], atol=1e-4)


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestLlamaEngineGenerate:
    @pytest.fixture(scope="class")
    def engine(self):
        blob = _tinyllama_blob()
        eng = LlamaEngine(blob, n_ctx=128)
        yield eng
        eng.close()

    def test_generate_yields_steps(self, engine):
        tokens = engine.tokenize("The capital of France is")
        steps = list(engine.generate(tokens, max_tokens=5, temperature=0))
        assert len(steps) >= 1
        assert all(isinstance(s, GenerateStep) for s in steps)

    def test_generate_step_fields(self, engine):
        tokens = engine.tokenize("Hello")
        steps = list(engine.generate(tokens, max_tokens=1, temperature=0))
        step = steps[0]
        assert isinstance(step.token_id, int)
        assert isinstance(step.token_str, str)

    def test_generate_without_logits(self, engine):
        tokens = engine.tokenize("Hello")
        steps = list(engine.generate(tokens, max_tokens=1, temperature=0, emit_logits=False))
        assert steps[0].logits is None

    def test_generate_with_logits(self, engine):
        tokens = engine.tokenize("Hello")
        steps = list(engine.generate(tokens, max_tokens=1, temperature=0, emit_logits=True))
        assert steps[0].logits is not None
        assert steps[0].logits.shape == (32000,)

    def test_generate_stops_at_max_tokens(self, engine):
        tokens = engine.tokenize("Once upon a time")
        steps = list(engine.generate(tokens, max_tokens=3, temperature=0.8))
        assert len(steps) <= 3

    def test_generate_greedy_deterministic(self, engine):
        tokens = engine.tokenize("The capital of France is")
        steps1 = list(engine.generate(tokens, max_tokens=3, temperature=0))
        steps2 = list(engine.generate(tokens, max_tokens=3, temperature=0))
        assert [s.token_id for s in steps1] == [s.token_id for s in steps2]


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestLlamaEnginePerplexity:
    @pytest.fixture(scope="class")
    def engine(self):
        blob = _tinyllama_blob()
        eng = LlamaEngine(blob, n_ctx=512)
        yield eng
        eng.close()

    def test_perplexity_is_finite(self, engine):
        ppl = engine.perplexity("The quick brown fox jumps over the lazy dog.")
        assert isinstance(ppl, float)
        assert np.isfinite(ppl)
        assert ppl > 0

    def test_coherent_lower_than_random(self, engine):
        coherent = engine.perplexity("The weather today is sunny and warm.")
        garbage = engine.perplexity("xkcd plonk wibble zarg fleem quux narf.")
        assert coherent < garbage


import tempfile


@pytest.mark.skipif(not TINYLLAMA_EXISTS, reason="tinyllama not in Ollama")
class TestExportHfToGguf:
    def test_round_trip_logits(self):
        """Load GGUF -> dequant to PyTorch -> export back -> reload -> compare logits."""
        import gc
        import torch
        from llm_surgeon.llama_engine import export_hf_to_gguf
        from llm_surgeon.surgery import load_model

        model, tokenizer = load_model("tinyllama:latest", mode="fp32")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "exported.gguf"
            export_hf_to_gguf(model, tokenizer, out_path)

            assert out_path.exists()
            assert out_path.stat().st_size > 1e6

            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

            with LlamaEngine(out_path, n_ctx=128) as eng:
                tokens = eng.tokenize("The capital of France is")
                logits_exported = eng.logits(tokens)

            blob = _tinyllama_blob()
            with LlamaEngine(blob, n_ctx=128) as eng_orig:
                tokens_orig = eng_orig.tokenize("The capital of France is")
                logits_original = eng_orig.logits(tokens_orig)

            result = compare_logits(logits_original, logits_exported)
            assert result["cosine_similarity"] > 0.99
            assert result["top_k_agreement"] >= 8

    def test_exported_file_is_valid_gguf(self):
        """Verify the exported file can be parsed as GGUF."""
        from llm_surgeon.llama_engine import export_hf_to_gguf
        from llm_surgeon.surgery import load_model
        from llm_surgeon.gguf_reader import GGUFFile

        model, tokenizer = load_model("tinyllama:latest", mode="fp32")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "exported.gguf"
            export_hf_to_gguf(model, tokenizer, out_path)

            with GGUFFile(out_path) as g:
                assert g.architecture == "llama"
                assert len(g.tensor_infos) > 0

    def test_exported_tokenizer_metadata(self):
        """Exported GGUF carries chat_template, byte-typed byte tokens, and merges."""
        from llm_surgeon.llama_engine import export_hf_to_gguf
        from llm_surgeon.surgery import load_model
        from llm_surgeon.gguf_reader import GGUFFile

        model, tokenizer = load_model("tinyllama:latest", mode="fp32")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "exported.gguf"
            export_hf_to_gguf(model, tokenizer, out_path)

            with GGUFFile(out_path) as g:
                meta = g.metadata
                assert meta.get("tokenizer.chat_template"), "chat_template missing"
                token_types = meta.get("tokenizer.ggml.token_type")
                assert token_types is not None
                # TinyLlama has 256 byte tokens <0x00>..<0xFF> in the base vocab.
                # Before this fix, all were tagged type 1 (normal); now they should be 6 (byte).
                byte_type_count = sum(1 for t in token_types if t == 6)
                assert byte_type_count >= 256, f"expected ≥256 byte-typed tokens, got {byte_type_count}"
                # Merges are present for BPE-backed fast tokenizers.
                merges = meta.get("tokenizer.ggml.merges")
                assert merges is not None and len(merges) > 0


class TestExportHfToGgufRopeTheta:
    def test_raises_on_none_rope_theta(self, tiny_llama):
        """rope_theta=None must raise — silent fallback to 10000.0 would corrupt
        LLaMA 3 / Mistral exports where the true base is 500000 / 1000000."""
        from llm_surgeon.llama_engine import export_hf_to_gguf

        tiny_llama.config.rope_theta = None
        # Newer transformers stashes rope_theta under rope_parameters too —
        # null both paths so the fallback chain exhausts.
        if hasattr(tiny_llama.config, "rope_parameters"):
            tiny_llama.config.rope_parameters = None

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "bad.gguf"
            with pytest.raises(ValueError, match="rope_theta"):
                export_hf_to_gguf(tiny_llama, None, out_path)
