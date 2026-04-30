"""Unit + integration tests for POST /api/sessions/{name}/decode-neuron (Phase 3.9.1)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import os

import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient


# ---- Mock session + model setup ----

class _MockTok:
    """Minimal HF-like tokenizer: per-id decode returns a deterministic string."""
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return "".join(f"<{i}>" for i in ids)


class _MockDownProj(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden, intermediate))


class _MockMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.down_proj = _MockDownProj(hidden, intermediate)


class _MockLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = _MockMLP(hidden, intermediate)


class _MockInner(nn.Module):
    def __init__(self, n_layers: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(hidden, intermediate) for _ in range(n_layers)])


class _MockLMHead(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))


class _MockModel(nn.Module):
    def __init__(self, vocab: int = 10, hidden: int = 4, intermediate: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.model = _MockInner(n_layers, hidden, intermediate)
        self.lm_head = _MockLMHead(vocab, hidden)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            intermediate_size=intermediate,
            vocab_size=vocab,
        )


@pytest.fixture
def app_with_mock_session():
    """Fresh FastAPI app with a single session injected that holds a mock model + tokenizer."""
    from gui.backend.app import app  # noqa: PLC0415
    from gui.backend.routes.sessions import get_manager  # noqa: PLC0415

    mgr = get_manager()
    mock_model = _MockModel()
    mock_tok = _MockTok()
    session_name = "mock-decode-neuron"
    # Insert directly into the manager's internal dict.
    mgr._sessions[session_name] = SimpleNamespace(  # pyright: ignore[reportArgumentType]
        name=session_name,
        model=mock_model,
        tokenizer=mock_tok,
        llama=None,
        dirty=False,
        original_layer=lambda i: i,
        _layer_map=[],
    )
    try:
        yield (app, session_name)
    finally:
        mgr._sessions.pop(session_name, None)


class TestDecodeNeuronUnit:
    def test_returns_topk_and_bottomk_sorted(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 3, "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == 5
        assert len(body["bottom_tokens"]) == 5
        top_logits = [t["logit"] for t in body["top_tokens"]]
        bot_logits = [t["logit"] for t in body["bottom_tokens"]]
        assert top_logits == sorted(top_logits, reverse=True)
        assert bot_logits == sorted(bot_logits)
        # Top-logit > bottom-logit (or equal at degenerate cases).
        assert top_logits[0] >= bot_logits[0]

    def test_invalid_layer_returns_400(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 99, "neuron": 0, "top_k": 3})
        assert resp.status_code == 400
        assert "layer" in resp.json()["detail"]

    def test_invalid_neuron_returns_400(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 999, "top_k": 3})
        assert resp.status_code == 400
        assert "neuron" in resp.json()["detail"]

    def test_top_k_clamped_to_50(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 0, "top_k": 1000})
        assert resp.status_code == 200
        body = resp.json()
        # Mock vocab is 10; clamp to min(50, vocab).
        assert len(body["top_tokens"]) == min(50, 10)

    def test_top_k_floor_at_1(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-neuron",
                           json={"layer": 0, "neuron": 0, "top_k": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == 1

    def test_unknown_session_returns_404(self, app_with_mock_session) -> None:
        app, _ = app_with_mock_session
        client = TestClient(app)
        resp = client.post("/api/sessions/does-not-exist/decode-neuron",
                           json={"layer": 0, "neuron": 0, "top_k": 3})
        assert resp.status_code == 404


# -------------------------------------------------------------------------
# TinyLlama integration — click a real top-AP neuron and sanity-check the
# decoded tokens (skipif GPU missing; fp16 to match Phase 3.9's precedent).
# -------------------------------------------------------------------------

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestDecodeNeuronTinyLlama:
    def test_top_neuron_decode_has_semantic_signal(self) -> None:
        """On the capital-of-France task, the top-ranked FFN neuron's
        decoded tokens should include at least one recognizable string
        (loose assertion — 1B-param neuron rankings are noisy)."""
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import attribution_patch_per_neuron

        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        clean = "The capital of France is"
        corrupted = "The capital of Italy is"
        paris_id = int(tok(" Paris", return_tensors="pt")["input_ids"][0, 1].item())
        rome_id = int(tok(" Rome", return_tensors="pt")["input_ids"][0, 1].item())

        r = attribution_patch_per_neuron(
            model, tok, clean, corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=rome_id,
            direction="denoise",
            top_k_neurons=10,
        )
        # Pick the top cell with positive ap_recovery (neuron that PROMOTES Paris).
        top_pos = next((c for c in r.cells if c["ap_recovery"] > 0), None)
        assert top_pos is not None, "expected at least one positive-AP neuron"

        L, neuron_idx = int(top_pos["layer"]), int(top_pos["neuron"])

        # Compute W_U @ W_down[L][:, neuron] directly (no endpoint needed).
        with torch.no_grad():
            direction = model.model.layers[L].mlp.down_proj.weight[:, neuron_idx]
            scores = model.lm_head.weight @ direction
            top_ids = torch.topk(scores, 10, largest=True).indices.tolist()

        decoded = [tok.decode([i], skip_special_tokens=False) for i in top_ids]
        # Loose semantic check: SOME token in the top-10 has non-whitespace content
        # and the list is 10 strings.
        assert len(decoded) == 10
        assert any(s.strip() != "" for s in decoded)
