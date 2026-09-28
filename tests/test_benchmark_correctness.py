"""Regression tests for benchmark.py correctness fixes (audit issue #11).

The perplexity tests use a context-independent stub model: its logits are
the same at every position, so the exact perplexity is known in closed form
whatever the windowing, and any token the sliding window fails to score (or
scores twice) moves the result.
"""

import itertools
import json
import math
import sqlite3
import types
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon import benchmark
from llm_surgeon.benchmark import perplexity


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


class _ConstantLogitModel(nn.Module):
    """Causal-LM stand-in whose next-token logits ignore the context.

    Records the shape and kwargs of every forward call.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
        )
        self.embed = nn.Embedding(vocab_size, 4)
        gen = torch.Generator().manual_seed(0)
        self.row = torch.randn(vocab_size, generator=gen) * 3.0
        self.calls: list[dict[str, Any]] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> types.SimpleNamespace:
        self.calls.append({"len": int(input_ids.size(1)), "kwargs": sorted(kwargs)})
        b, n = input_ids.shape
        return types.SimpleNamespace(logits=self.row.expand(b, n, -1).clone())

    def exact_ppl(self, ids: list[int]) -> float:
        """Perplexity with every token except the first scored exactly once."""
        logp = torch.log_softmax(self.row, dim=-1)
        nll = -sum(float(logp[t]) for t in ids[1:])
        return math.exp(nll / (len(ids) - 1))


class _ListTokenizer:
    """Tokenizer stand-in: text is a space-separated list of token ids."""

    def __init__(self) -> None:
        self.model_max_length = 77

    def __call__(self, text: str, return_tensors: str = "pt") -> types.SimpleNamespace:
        ids = [int(t) for t in text.split()]
        return types.SimpleNamespace(input_ids=torch.tensor([ids]))


def _ids_text(n: int, vocab: int = 50) -> tuple[list[int], str]:
    ids = [(i * 7 + 3) % vocab for i in range(n)]
    return ids, " ".join(str(i) for i in ids)


@pytest.fixture
def gqa_llama():
    """GQA LLaMA: fewer KV heads than query heads, head_dim != hidden/heads."""
    cfg = LlamaConfig(
        vocab_size=64,  # pyright: ignore[reportCallIssue]
        hidden_size=48,  # pyright: ignore[reportCallIssue]
        intermediate_size=96,  # pyright: ignore[reportCallIssue]
        num_hidden_layers=4,  # pyright: ignore[reportCallIssue]
        num_attention_heads=4,  # pyright: ignore[reportCallIssue]
        num_key_value_heads=2,  # pyright: ignore[reportCallIssue]
        head_dim=16,  # pyright: ignore[reportCallIssue]
        max_position_embeddings=4096,  # pyright: ignore[reportCallIssue]
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# perplexity(): sliding window
# ---------------------------------------------------------------------------


class TestPerplexityWindow:
    @pytest.mark.parametrize(
        "n,window,stride",
        [(10, 4, 2), (37, 8, 3), (64, 16, 15), (50, 8, 1), (5, 16, 8)],
    )
    def test_every_token_scored_once(self, n, window, stride):
        """Result equals the closed-form perplexity over tokens 1..n-1."""
        model = _ConstantLogitModel(vocab_size=50, max_position_embeddings=128)
        ids, text = _ids_text(n)
        got = perplexity(
            model, _ListTokenizer(), text=text, stride=stride, max_length=window
        )
        assert got == pytest.approx(model.exact_ppl(ids), rel=1e-5)

    def test_default_window_capped_for_long_context_config(self):
        """A 131072-position config does not produce a 131072-token chunk."""
        model = _ConstantLogitModel(vocab_size=50, max_position_embeddings=131072)
        ids, text = _ids_text(5000)
        got = perplexity(model, _ListTokenizer(), text=text)
        assert max(c["len"] for c in model.calls) == benchmark.DEFAULT_PPL_WINDOW
        assert got == pytest.approx(model.exact_ppl(ids), rel=1e-5)

    def test_max_length_argument_bounds_chunks(self):
        model = _ConstantLogitModel(vocab_size=50, max_position_embeddings=128)
        _ids, text = _ids_text(100)
        perplexity(model, _ListTokenizer(), text=text, max_length=32)
        assert max(c["len"] for c in model.calls) == 32

    def test_no_labels_passed_to_model(self):
        """The forward pass gets input ids only; the loss is recomputed here."""
        model = _ConstantLogitModel(vocab_size=50, max_position_embeddings=128)
        _ids, text = _ids_text(40)
        perplexity(model, _ListTokenizer(), text=text, max_length=16)
        assert all(c["kwargs"] == [] for c in model.calls)

    @pytest.mark.parametrize("stride", [0, -4, 16, 17])
    def test_invalid_stride_raises(self, stride):
        model = _ConstantLogitModel(vocab_size=50, max_position_embeddings=128)
        _ids, text = _ids_text(40)
        with pytest.raises(ValueError, match="stride"):
            perplexity(model, _ListTokenizer(), text=text, stride=stride, max_length=16)

    def test_tokenizer_max_length_restored_on_error(self):
        class _Boom(_ListTokenizer):
            def __call__(self, text, return_tensors="pt"):
                raise RuntimeError("tokenizer failure")

        tok = _Boom()
        model = _ConstantLogitModel(vocab_size=50, max_position_embeddings=128)
        with pytest.raises(RuntimeError, match="tokenizer failure"):
            perplexity(model, tok, text="1 2 3")
        assert tok.model_max_length == 77

    def test_verbose_progress_total_matches_windows(self, capsys):
        model = _ConstantLogitModel(vocab_size=50, max_position_embeddings=128)
        _ids, text = _ids_text(100)
        perplexity(
            model, _ListTokenizer(), text=text, max_length=16, stride=8, verbose=True
        )
        last = capsys.readouterr().out.strip().splitlines()[-1]
        n_windows = len(model.calls)
        assert f"window {n_windows}/{n_windows} " in last

    def test_gqa_model_matches_single_pass_loss(self, gqa_llama):
        """With the window covering the text, ppl == exp(HF loss)."""
        from tests.conftest import _make_tiny_tokenizer

        tok = _make_tiny_tokenizer(gqa_llama.config.vocab_size)
        text = " ".join(f"word{4 + i % 50}" for i in range(120))
        got = perplexity(gqa_llama, tok, text=text, max_length=256)
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            ref = math.exp(float(gqa_llama(ids, labels=ids).loss))
        assert got == pytest.approx(ref, rel=1e-4)


# ---------------------------------------------------------------------------
# _load_dataset_text
# ---------------------------------------------------------------------------


class TestLoadDatasetText:
    def test_c4_without_max_samples_is_bounded(self, monkeypatch):
        """c4 streams; max_samples=None must not drain the whole split."""
        pulled = {"n": 0}

        def _stream():
            for i in itertools.count():
                pulled["n"] += 1
                if i > 100_000:
                    raise AssertionError("c4 stream drained without bound")
                yield {"text": f"doc {i}"}

        def fake_load_dataset(*args, **kwargs):
            assert kwargs.get("streaming") is True
            return _stream()

        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
        text = benchmark._load_dataset_text("c4", max_samples=None)
        assert text.count("doc ") == benchmark.C4_DEFAULT_MAX_SAMPLES

    def test_wikitext_uses_standard_join(self, monkeypatch):
        """wikitext2 follows the reference recipe: raw rows joined by blank lines."""
        rows = ["", " = Title = \n", "", " body text \n"]

        def fake_load_dataset(*args, **kwargs):
            return {"text": rows}

        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
        assert benchmark._load_dataset_text("wikitext2") == "\n\n".join(rows)
        assert benchmark._load_dataset_text("wikitext2", max_samples=2) == (
            "\n\n".join(rows[:2])
        )
