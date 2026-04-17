"""Tests for probe module — logit lens, hidden state extraction, intervention."""

import pytest
import torch

from tests.conftest import _make_tiny_tokenizer

from llm_surgeon.probe import (
    LogitLensResult, HiddenStates, extract_hidden_states,
    logit_lens, layer_predictions_table, ops,
    Intervention, InterventionResult, intervene,
    _cell_metrics, _pair_metrics, compare_logit_lens,
)


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


# ---------------------------------------------------------------------------
# extract_hidden_states
# ---------------------------------------------------------------------------

def test_extract_hidden_states_ffn_only(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    hs = extract_hidden_states(tiny_llama, tokenizer, prompt)
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(hs.states) == num_layers
    for i in range(num_layers):
        assert (i, "ffn") in hs.states
        assert hs.states[(i, "ffn")].shape[-1] == tiny_llama_config.hidden_size


def test_extract_hidden_states_both_sublayers(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    hs = extract_hidden_states(tiny_llama, tokenizer, prompt, sublayers=("attn", "ffn"))
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(hs.states) == num_layers * 2
    for i in range(num_layers):
        assert (i, "attn") in hs.states
        assert (i, "ffn") in hs.states


def test_extract_hidden_states_specific_layers(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    hs = extract_hidden_states(tiny_llama, tokenizer, prompt, layers=[0, 3, 7])
    assert set(hs.states.keys()) == {(0, "ffn"), (3, "ffn"), (7, "ffn")}


def test_extract_hidden_states_callback(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    calls = []
    def cb(layer, sublayer, data):
        calls.append((layer, sublayer))
        assert "hidden_state" in data
    extract_hidden_states(tiny_llama, tokenizer, prompt, on_layer=cb)
    assert len(calls) == tiny_llama_config.num_hidden_layers


# ---------------------------------------------------------------------------
# logit_lens
# ---------------------------------------------------------------------------

def test_logit_lens_basic(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=5)

    assert isinstance(result, LogitLensResult)
    assert result.logits is None
    assert len(result.prompt_tokens) > 0

    num_layers = tiny_llama_config.num_hidden_layers
    num_positions = len(result.prompt_tokens)
    expected_predictions = num_layers * 2 * num_positions
    assert len(result.predictions) == expected_predictions

    for p in result.predictions:
        assert "layer" in p
        assert "sublayer" in p
        assert p["sublayer"] in ("attn", "ffn")
        assert "position" in p
        assert "top_k" in p
        assert len(p["top_k"]) <= 5
        if p["top_k"]:
            assert "token" in p["top_k"][0]
            assert "prob" in p["top_k"][0]
            assert p["top_k"][0]["prob"] >= 0.0


def test_logit_lens_full_logits(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=3, full_logits=True)

    assert result.logits is not None
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(result.logits) == num_layers * 2
    for _key, tensor in result.logits.items():
        assert tensor.shape[-1] == tiny_llama_config.vocab_size


def test_logit_lens_positions_filter(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=3, positions=[-1])

    positions_seen = set(p["position"] for p in result.predictions)
    assert len(positions_seen) == 1


def test_logit_lens_callback(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    calls = []
    def cb(layer, sublayer, data):
        calls.append((layer, sublayer))
        assert "hidden_state" in data
        assert "top_k" in data
        assert "metrics" in data
        assert len(data["metrics"]) == len(data["top_k"])
    logit_lens(tiny_llama, tokenizer, prompt, top_k=3, on_layer=cb)
    num_layers = tiny_llama_config.num_hidden_layers
    assert len(calls) == num_layers * 2


# ---------------------------------------------------------------------------
# Per-cell metrics (entropy, top1_prob, top1_margin)
# ---------------------------------------------------------------------------

def test_cell_metrics_uniform_distribution_has_max_entropy():
    vocab = 128
    probs = torch.full((vocab,), 1.0 / vocab)
    m = _cell_metrics(probs)
    # H(uniform) = log(vocab); top1_prob = 1/vocab; margin = 0.
    assert m["entropy"] == pytest.approx(torch.log(torch.tensor(float(vocab))).item(), abs=1e-5)
    assert m["top1_prob"] == pytest.approx(1.0 / vocab, abs=1e-6)
    assert m["top1_margin"] == pytest.approx(0.0, abs=1e-6)


def test_cell_metrics_one_hot_distribution_has_zero_entropy():
    vocab = 128
    probs = torch.zeros(vocab)
    probs[42] = 1.0
    m = _cell_metrics(probs)
    # xlogy handles p=0 correctly, so entropy is exactly 0 here.
    assert m["entropy"] == pytest.approx(0.0, abs=1e-6)
    assert m["top1_prob"] == pytest.approx(1.0, abs=1e-6)
    assert m["top1_margin"] == pytest.approx(1.0, abs=1e-6)


def test_cell_metrics_margin_is_nonnegative():
    torch.manual_seed(0)
    for _ in range(20):
        logits = torch.randn(256)
        probs = torch.softmax(logits, dim=-1)
        m = _cell_metrics(probs)
        assert m["top1_margin"] >= 0.0
        assert 0.0 <= m["top1_prob"] <= 1.0
        assert m["entropy"] >= 0.0


def test_logit_lens_cells_carry_metrics(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    result = logit_lens(tiny_llama, tokenizer, "word4 word5 word6", top_k=3)
    for p in result.predictions:
        assert "metrics" in p
        m = p["metrics"]
        assert set(m.keys()) == {"entropy", "top1_prob", "top1_margin"}
        assert m["top1_margin"] >= 0.0
        assert 0.0 <= m["top1_prob"] <= 1.0
        assert m["entropy"] >= 0.0


# ---------------------------------------------------------------------------
# Pair metrics (A/B comparison)
# ---------------------------------------------------------------------------

def test_pair_metrics_self_comparison_is_identity():
    torch.manual_seed(0)
    logits = torch.randn(256)
    probs = torch.softmax(logits, dim=-1)
    m = _pair_metrics(probs, probs)
    assert m["kl_ab"] == pytest.approx(0.0, abs=1e-5)
    assert m["js"] == pytest.approx(0.0, abs=1e-5)
    assert m["cosine"] == pytest.approx(1.0, abs=1e-5)
    assert m["top1_delta_prob"] == pytest.approx(0.0, abs=1e-6)
    assert m["top1_match"] is True


def test_pair_metrics_js_bounded_and_symmetric():
    torch.manual_seed(1)
    log2 = torch.log(torch.tensor(2.0)).item()
    for _ in range(20):
        p_a = torch.softmax(torch.randn(128), dim=-1)
        p_b = torch.softmax(torch.randn(128), dim=-1)
        m_ab = _pair_metrics(p_a, p_b)
        m_ba = _pair_metrics(p_b, p_a)
        assert 0.0 <= m_ab["js"] <= log2 + 1e-5
        assert m_ab["js"] == pytest.approx(m_ba["js"], abs=1e-5)
        assert m_ab["cosine"] == pytest.approx(m_ba["cosine"], abs=1e-5)
        assert m_ab["top1_match"] == m_ba["top1_match"]


def test_pair_metrics_kl_is_nonnegative():
    torch.manual_seed(2)
    for _ in range(20):
        p_a = torch.softmax(torch.randn(128), dim=-1)
        p_b = torch.softmax(torch.randn(128), dim=-1)
        m = _pair_metrics(p_a, p_b)
        assert m["kl_ab"] >= -1e-6  # tiny float slop allowed at the boundary


def test_pair_metrics_handles_zero_in_a():
    # If p_a has zero support, xlogy(0, ...) = 0 so those positions contribute nothing.
    p_a = torch.zeros(16)
    p_a[3] = 0.7
    p_a[5] = 0.3
    p_b = torch.softmax(torch.arange(16, dtype=torch.float32), dim=-1)
    m = _pair_metrics(p_a, p_b)
    assert torch.isfinite(torch.tensor(m["kl_ab"]))
    assert torch.isfinite(torch.tensor(m["js"]))


def test_compare_logit_lens_against_self(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    result = compare_logit_lens(tiny_llama, tiny_llama, tokenizer, "word4 word5 word6", top_k=3)

    num_layers = tiny_llama_config.num_hidden_layers
    num_positions = len(result.prompt_tokens)
    # All (attn, ffn) pairs from every layer should align when comparing against self.
    assert len(result.aligned_keys) == num_layers * 2
    assert len(result.comparisons) == num_layers * 2 * num_positions

    for cell in result.comparisons:
        cmp = cell["compare"]
        assert cmp["kl_ab"] == pytest.approx(0.0, abs=1e-4)
        assert cmp["js"] == pytest.approx(0.0, abs=1e-4)
        assert cmp["cosine"] == pytest.approx(1.0, abs=1e-4)
        assert cmp["top1_delta_prob"] == pytest.approx(0.0, abs=1e-5)
        assert cmp["top1_match"] is True


def test_compare_logit_lens_streams_via_callback(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    calls = []
    def cb(orig_layer, sublayer, data):
        calls.append((orig_layer, sublayer))
        assert "cells" in data
        for cell in data["cells"]:
            assert "compare" in cell
            assert "top_k_a" in cell
            assert "top_k_b" in cell
            assert "metrics_a" in cell
            assert "metrics_b" in cell

    compare_logit_lens(tiny_llama, tiny_llama, tokenizer, "word4 word5", on_layer=cb)
    assert len(calls) == tiny_llama_config.num_hidden_layers * 2


def test_compare_logit_lens_alignment_with_layer_maps(tiny_llama, tiny_llama_config):
    # Model B has a shifted layer_map: pretend its compressed layers 0..N-1 map to
    # ORIGINAL layers 5..5+N-1. Alignment by original index means B's captured layer 0
    # (which is tagged original=5) should only compare against A's layer 5 — and since
    # A's layer_map is identity, the intersection is layers {5..min(N_a-1, 5+N_b-1)}.
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    num_layers = tiny_llama_config.num_hidden_layers
    shift = max(1, num_layers // 3)
    identity = list(range(num_layers))
    shifted = list(range(shift, shift + num_layers))

    result = compare_logit_lens(
        tiny_llama, tiny_llama, tokenizer, "word4 word5",
        top_k=3, layer_map_a=identity, layer_map_b=shifted,
    )
    expected_aligned_layers = set(range(shift, num_layers))
    actual_layers = set(orig for (orig, _sub) in result.aligned_keys)
    assert actual_layers == expected_aligned_layers


def test_layer_predictions_table(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = logit_lens(tiny_llama, tokenizer, prompt, top_k=3)
    table = layer_predictions_table(result, position=-1)
    assert isinstance(table, str)
    assert len(table) > 0
    assert "Layer" in table


# ---------------------------------------------------------------------------
# Predefined operations (ops)
# ---------------------------------------------------------------------------

def test_ops_scale():
    fn = ops.scale(2.0)
    t = torch.ones(4, 8)
    result = fn(t, 0)
    assert torch.allclose(result, t * 2.0)
    assert "scale(2.0)" in repr(fn)


def test_ops_scale_identity():
    fn = ops.scale(1.0)
    t = torch.randn(4, 8)
    result = fn(t, 0)
    assert torch.allclose(result, t)


def test_ops_zero_dims():
    fn = ops.zero_dims([0, 2, 4])
    t = torch.ones(3, 8)
    result = fn(t, 0)
    assert result[:, 0].sum() == 0
    assert result[:, 2].sum() == 0
    assert result[:, 4].sum() == 0
    assert result[:, 1].sum() == 3
    assert "zero_dims" in repr(fn)


def test_ops_clamp():
    fn = ops.clamp(-0.5, 0.5)
    t = torch.tensor([[-1.0, 0.0, 1.0]])
    result = fn(t, 0)
    assert result.min().item() >= -0.5
    assert result.max().item() <= 0.5
    assert "clamp" in repr(fn)


def test_ops_noise():
    fn = ops.noise(0.1, seed=42)
    t = torch.zeros(4, 8)
    result = fn(t, 0)
    assert not torch.allclose(result, t)
    assert result.abs().max().item() < 1.0
    result2 = ops.noise(0.1, seed=42)(t, 0)
    assert torch.allclose(result, result2)
    assert "noise" in repr(fn)


def test_ops_replace():
    replacement = torch.ones(4, 8) * 5
    fn = ops.replace(replacement)
    t = torch.zeros(4, 8)
    result = fn(t, 0)
    assert torch.allclose(result, replacement)
    assert "replace" in repr(fn)


def test_ops_project_out():
    direction = torch.zeros(8)
    direction[0] = 1.0
    fn = ops.project_out(direction)
    t = torch.ones(3, 8)
    result = fn(t, 0)
    assert result[:, 0].abs().max().item() < 1e-5
    assert torch.allclose(result[:, 1:], t[:, 1:])
    assert "project_out" in repr(fn)


# ---------------------------------------------------------------------------
# intervene()
# ---------------------------------------------------------------------------

def test_intervene_scale_identity(tiny_llama, tiny_llama_config):
    """Scaling by 1.0 should produce the same output as an unmodified forward pass."""
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"

    device = tiny_llama.model.embed_tokens.weight.device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # pyright: ignore[reportAttributeAccessIssue]
    with torch.no_grad():
        baseline_logits = tiny_llama(input_ids).logits[0]

    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=0, sublayer="ffn", fn=ops.scale(1.0))],
    )
    assert isinstance(result, InterventionResult)
    assert torch.allclose(result.output_logits, baseline_logits, atol=1e-4)


def test_intervene_scale_zero_changes_output(tiny_llama, tiny_llama_config):
    """Zeroing layer 0 should produce different output."""
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"

    device = tiny_llama.model.embed_tokens.weight.device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # pyright: ignore[reportAttributeAccessIssue]
    with torch.no_grad():
        baseline_logits = tiny_llama(input_ids).logits[0]

    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=0, sublayer="ffn", fn=ops.scale(0.0))],
    )
    assert not torch.allclose(result.output_logits, baseline_logits, atol=1e-3)


def test_intervene_with_logit_lens(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=2, sublayer="ffn", fn=ops.scale(0.5))],
        capture_logit_lens=True,
        top_k=3,
    )
    assert result.logit_lens_result is not None
    assert len(result.logit_lens_result.predictions) > 0


def test_intervene_callback_modified_flag(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    calls = []
    def cb(layer, sublayer, data):
        calls.append((layer, sublayer, data["modified"]))
    intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=3, sublayer="ffn", fn=ops.scale(2.0))],
        on_layer=cb,
    )
    modified_calls = [(l, s) for l, s, m in calls if m]
    assert (3, "ffn") in modified_calls
    unmodified_calls = [(l, s) for l, s, m in calls if not m]
    assert len(unmodified_calls) > 0


def test_intervene_metadata(tiny_llama, tiny_llama_config):
    tokenizer = _make_test_tokenizer(tiny_llama_config.vocab_size)
    prompt = "word4 word5 word6"
    result = intervene(
        tiny_llama, tokenizer, prompt,
        interventions=[Intervention(layer=1, sublayer="ffn", fn=ops.noise(0.1, seed=0))],
    )
    assert len(result.interventions_applied) == 1
    meta = result.interventions_applied[0]
    assert meta["layer"] == 1
    assert meta["sublayer"] == "ffn"
    assert "noise" in meta["op_repr"]
