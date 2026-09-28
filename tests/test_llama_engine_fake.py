"""LlamaEngine tests against a scripted stand-in for ``llama_cpp.Llama``.

The real-model tests in test_llama_engine.py need TinyLlama in the local
Ollama store and only exercise ASCII output on short prompts. These tests
drive the engine's own logic (logit reads, UTF-8 streaming, context bounds)
with a fake whose logits and byte-level vocab are fully controlled.
"""

import sys
import types
from collections import deque

import numpy as np
import pytest

from llm_surgeon.llama_engine import LlamaEngine

BOS, EOS = 1, 2

# Byte-level vocab: tokens 8..13 are single UTF-8 bytes, so "é" (2 bytes)
# and "😀" (4 bytes) are split across tokens as SPM byte fallback does.
_E_ACUTE = "é".encode()
_GRIN = "😀".encode()
VOCAB: list[bytes] = [
    b"",  # 0 <unk>
    b"",  # 1 BOS
    b"",  # 2 EOS
    b" Hi",  # 3
    b" there",  # 4
    b"!",  # 5
    b" x",  # 6
    b" y",  # 7
    _E_ACUTE[:1],  # 8
    _E_ACUTE[1:],  # 9
    _GRIN[:1],  # 10
    _GRIN[1:2],  # 11
    _GRIN[2:3],  # 12
    _GRIN[3:],  # 13
]


class FakeLlama:
    """The subset of ``llama_cpp.Llama`` that LlamaEngine uses.

    Logits for position ``p`` are a row of ``logit_rows`` when given
    (perplexity tests), else one-hot on ``plan[p]`` so greedy decoding
    follows the plan and falls back to token 6 once the plan runs out.
    """

    def __init__(
        self,
        *,
        n_ctx: int = 64,
        plan: dict[int, int] | None = None,
        logit_rows: np.ndarray | None = None,
        forbid_eval_logits: bool = True,
    ):
        self._n_ctx = n_ctx
        self._plan = plan or {}
        self._rows = logit_rows
        self._forbid_eval_logits = forbid_eval_logits
        self.scores = np.zeros((n_ctx, len(VOCAB)), dtype=np.float32)
        self.n_tokens = 0

    # llama_cpp.Llama API -------------------------------------------------
    def n_vocab(self) -> int:
        return len(VOCAB)

    def n_ctx(self) -> int:
        return self._n_ctx

    def token_eos(self) -> int:
        return EOS

    def reset(self) -> None:
        self.n_tokens = 0

    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        ids = [VOCAB.index(b" " + w.encode()) for w in text.decode().split()]
        return ([BOS] if add_bos else []) + ids

    def detokenize(self, tokens: list[int], prev_tokens=None, special=False) -> bytes:
        return b"".join(VOCAB[t] for t in tokens)

    def eval(self, tokens: list[int]) -> None:
        if not tokens:
            return
        if self.n_tokens + len(tokens) > self._n_ctx:
            raise RuntimeError("llama_decode returned 1")  # what llama.cpp does
        for tok in tokens:
            del tok
            p = self.n_tokens
            if self._rows is not None:
                self.scores[p] = self._rows[p]
            else:
                self.scores[p] = 0.0
                self.scores[p, self._plan.get(p, 6)] = 10.0
            self.n_tokens += 1

    @property
    def eval_logits(self):
        # Real llama-cpp-python rebuilds every row as Python lists here:
        # O(n_tokens * n_vocab) per access.
        if self._forbid_eval_logits:
            raise AssertionError("eval_logits re-materializes all logits; read .scores")
        return deque(self.scores[: self.n_tokens, :].tolist(), maxlen=self._n_ctx)


def _engine(monkeypatch, fake: FakeLlama) -> LlamaEngine:
    fake_mod = types.SimpleNamespace(Llama=lambda **_kw: fake)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)
    return LlamaEngine("fake.gguf", n_ctx=fake.n_ctx())  # pyright: ignore[reportArgumentType]


def _greedy_plan(prompt_len: int, gen: list[int]) -> dict[int, int]:
    """Plan that makes greedy decoding emit ``gen`` right after the prompt."""
    return {prompt_len - 1 + i: tok for i, tok in enumerate(gen)}


# ---------------------------------------------------------------------------
# Logit reads go through the scores ndarray, not eval_logits
# ---------------------------------------------------------------------------


class TestLogitReads:
    def test_perplexity_matches_reference(self, monkeypatch):
        rng = np.random.default_rng(0)
        rows = rng.standard_normal((64, len(VOCAB))).astype(np.float32)
        eng = _engine(monkeypatch, FakeLlama(logit_rows=rows))
        tokens = eng.tokenize("Hi there x y")
        ppl = eng.perplexity("Hi there x y")

        nll = 0.0
        for i in range(len(tokens) - 1):
            row = rows[i].astype(np.float64)
            nll -= row[tokens[i + 1]] - np.logaddexp.reduce(row)
        assert ppl == pytest.approx(float(np.exp(nll / (len(tokens) - 1))), rel=1e-5)

    def test_logits_and_logits_all(self, monkeypatch):
        rng = np.random.default_rng(1)
        rows = rng.standard_normal((64, len(VOCAB))).astype(np.float32)
        eng = _engine(monkeypatch, FakeLlama(logit_rows=rows))
        tokens = [BOS, 3, 4]
        last = eng.logits(tokens)
        np.testing.assert_array_equal(last, rows[2])
        every = eng.logits_all(tokens)
        assert len(every) == 3
        for got, want in zip(every, rows[:3]):
            np.testing.assert_array_equal(got, want)

    def test_returned_logits_do_not_alias_engine_buffer(self, monkeypatch):
        rng = np.random.default_rng(2)
        rows = rng.standard_normal((64, len(VOCAB))).astype(np.float32)
        eng = _engine(monkeypatch, FakeLlama(logit_rows=rows))
        first = eng.logits([BOS, 3])
        snapshot = first.copy()
        eng.logits([BOS, 7, 7, 7])  # overwrites the same scores rows
        np.testing.assert_array_equal(first, snapshot)

    def test_generate_does_not_touch_eval_logits(self, monkeypatch):
        eng = _engine(monkeypatch, FakeLlama(plan=_greedy_plan(2, [4, 5, EOS])))
        steps = list(eng.generate([BOS, 3], max_tokens=5, temperature=0))
        assert [s.token_id for s in steps] == [4, 5, EOS]


# ---------------------------------------------------------------------------
# Streaming text across multi-byte UTF-8 characters
# ---------------------------------------------------------------------------


class TestUtf8Streaming:
    GEN = [8, 9, 10, 11, 12, 13]  # "é😀" as six byte tokens

    def test_split_characters_are_emitted_whole(self, monkeypatch):
        eng = _engine(monkeypatch, FakeLlama(plan=_greedy_plan(2, [*self.GEN, EOS])))
        steps = list(eng.generate([BOS, 3], max_tokens=10, temperature=0))
        text = "".join(s.token_str for s in steps)
        assert text == "é😀"
        assert "�" not in text
        # The character is emitted on the step that completes it.
        assert [s.token_str for s in steps[:6]] == ["", "é", "", "", "", "😀"]

    def test_non_ascii_stop_sequence_matches(self, monkeypatch):
        eng = _engine(monkeypatch, FakeLlama(plan=_greedy_plan(2, [*self.GEN, 3, 4])))
        steps = list(
            eng.generate([BOS, 3], max_tokens=10, temperature=0, stop_sequences=["😀"])
        )
        assert [s.token_id for s in steps] == self.GEN

    def test_incomplete_tail_flushed_at_max_tokens(self, monkeypatch):
        # Generation ends mid-character: the dangling byte is flushed as a
        # replacement character instead of being silently dropped.
        eng = _engine(monkeypatch, FakeLlama(plan=_greedy_plan(2, [3, 10])))
        steps = list(eng.generate([BOS, 3], max_tokens=2, temperature=0))
        assert "".join(s.token_str for s in steps) == " Hi�"

    def test_leading_space_of_first_token_kept(self, monkeypatch):
        eng = _engine(monkeypatch, FakeLlama(plan=_greedy_plan(2, [4, EOS])))
        steps = list(eng.generate([BOS, 3], max_tokens=5, temperature=0))
        assert steps[0].token_str == " there"
