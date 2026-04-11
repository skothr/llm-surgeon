"""Tests for probe module — logit lens, hidden state extraction, intervention."""

import torch

from tests.conftest import _make_tiny_tokenizer

from llm_surgeon.probe import LogitLensResult, HiddenStates


def _make_test_tokenizer(vocab_size):
    return _make_tiny_tokenizer(vocab_size)


# ---------------------------------------------------------------------------
# LogitLensResult
# ---------------------------------------------------------------------------

def test_logit_lens_result_summary():
    result = LogitLensResult(
        predictions=[
            {"layer": 0, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "hello", "token_id": 5, "prob": 0.8, "rank": 0},
                       {"token": "world", "token_id": 6, "prob": 0.1, "rank": 1}]},
            {"layer": 1, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "world", "token_id": 6, "prob": 0.7, "rank": 0},
                       {"token": "hello", "token_id": 5, "prob": 0.2, "rank": 1}]},
        ],
        logits=None,
        prompt_tokens=["the"],
    )
    text = result.summary(position=0)
    assert "hello" in text
    assert "world" in text


def test_logit_lens_result_prediction_flips():
    result = LogitLensResult(
        predictions=[
            {"layer": 0, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "a", "token_id": 1, "prob": 0.5, "rank": 0}]},
            {"layer": 1, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "b", "token_id": 2, "prob": 0.5, "rank": 0}]},
            {"layer": 2, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "b", "token_id": 2, "prob": 0.6, "rank": 0}]},
            {"layer": 3, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "a", "token_id": 1, "prob": 0.7, "rank": 0}]},
        ],
        logits=None,
        prompt_tokens=["x"],
    )
    assert result.prediction_flips(position=0) == 2


def test_logit_lens_result_first_correct_layer():
    result = LogitLensResult(
        predictions=[
            {"layer": 0, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "wrong", "token_id": 1, "prob": 0.5, "rank": 0}]},
            {"layer": 1, "sublayer": "ffn", "position": 0,
             "top_k": [{"token": "right", "token_id": 2, "prob": 0.6, "rank": 0}]},
        ],
        logits=None,
        prompt_tokens=["x"],
    )
    assert result.first_correct_layer(position=0, target_token="right") == 1
    assert result.first_correct_layer(position=0, target_token="missing") is None


# ---------------------------------------------------------------------------
# HiddenStates
# ---------------------------------------------------------------------------

def test_hidden_states_cosine_similarity_identical():
    t = torch.randn(5, 32)
    hs = HiddenStates(
        states={(0, "ffn"): t, (1, "ffn"): t},
        prompt_tokens=["a"] * 5,
    )
    sim = hs.cosine_similarity((0, "ffn"), (1, "ffn"), position=-1)
    assert abs(sim - 1.0) < 1e-5


def test_hidden_states_save_load(tmp_path):
    t = torch.randn(5, 32)
    hs = HiddenStates(
        states={(0, "ffn"): t, (2, "attn"): t * 2},
        prompt_tokens=["a", "b", "c", "d", "e"],
    )
    path = str(tmp_path / "states.pt")
    hs.save(path)
    loaded = HiddenStates.load(path)
    assert set(loaded.states.keys()) == {(0, "ffn"), (2, "attn")}
    assert loaded.prompt_tokens == hs.prompt_tokens
    assert torch.allclose(loaded.states[(0, "ffn")], t)
