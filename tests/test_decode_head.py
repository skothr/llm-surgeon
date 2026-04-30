"""Unit + integration tests for POST /api/sessions/{name}/decode-head (Phase 3.9.2)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
import os

import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient


class _MockTok:
    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return "".join(f"<{i}>" for i in ids)


class _MockAttn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _MockLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = _MockAttn(hidden)


class _MockInner(nn.Module):
    def __init__(self, n_layers: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MockLayer(hidden) for _ in range(n_layers)])


class _MockLMHead(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden))


class _MockModel(nn.Module):
    def __init__(self, vocab: int = 10, hidden: int = 8, n_heads: int = 2, n_layers: int = 2) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.model = _MockInner(n_layers, hidden)
        self.lm_head = _MockLMHead(vocab, hidden)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=hidden,
            num_attention_heads=n_heads,
            vocab_size=vocab,
        )


@pytest.fixture
def app_with_mock_session():
    from gui.backend.app import app  # noqa: PLC0415
    from gui.backend.routes.sessions import get_manager  # noqa: PLC0415

    mgr = get_manager()
    session_name = "mock-decode-head"
    mgr._sessions[session_name] = SimpleNamespace(  # pyright: ignore[reportArgumentType]
        name=session_name,
        model=_MockModel(),
        tokenizer=_MockTok(),
        llama=None,
        dirty=False,
        original_layer=lambda i: i,
        _layer_map=[],
    )
    try:
        yield (app, session_name)
    finally:
        mgr._sessions.pop(session_name, None)


class TestDecodeHeadUnit:
    def test_returns_topk_bottomk_and_sv_ratio(self, app_with_mock_session: Any) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 1, "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == 3
        assert len(body["bottom_tokens"]) == 3
        top_logits = [t["logit"] for t in body["top_tokens"]]
        bot_logits = [t["logit"] for t in body["bottom_tokens"]]
        assert top_logits == sorted(top_logits, reverse=True)
        assert bot_logits == sorted(bot_logits)
        assert 0.0 <= body["singular_value_ratio"] <= 1.0

    def test_invalid_layer_returns_400(self, app_with_mock_session: Any) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 99, "head": 0, "top_k": 3})
        assert resp.status_code == 400
        assert "layer" in resp.json()["detail"]

    def test_invalid_head_returns_400(self, app_with_mock_session: Any) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 99, "top_k": 3})
        assert resp.status_code == 400
        assert "head" in resp.json()["detail"]

    def test_top_k_clamped_to_vocab(self, app_with_mock_session: Any) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 1000})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["top_tokens"]) == min(50, 10)  # mock vocab=10

    def test_top_k_floor_at_1(self, app_with_mock_session: Any) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 0})
        assert resp.status_code == 200
        assert len(resp.json()["top_tokens"]) == 1

    def test_unknown_session_returns_404(self, app_with_mock_session: Any) -> None:
        app, _ = app_with_mock_session
        client = TestClient(app)
        resp = client.post("/api/sessions/does-not-exist/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 3})
        assert resp.status_code == 404

    def test_sign_oriented_by_promoted_magnitude(self, app_with_mock_session: Any) -> None:
        """After orientation, |sum(top logits)| >= |sum(bottom logits)|."""
        app, name = app_with_mock_session
        client = TestClient(app)
        resp = client.post(f"/api/sessions/{name}/decode-head",
                           json={"layer": 0, "head": 0, "top_k": 5})
        body = resp.json()
        top_sum = sum(t["logit"] for t in body["top_tokens"])
        bot_sum = sum(t["logit"] for t in body["bottom_tokens"])
        assert abs(top_sum) >= abs(bot_sum)


# ---- TinyLlama integration ----

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestDecodeHeadTinyLlama:
    def test_top_ap_head_decode_has_signal(self) -> None:
        """On capital-of-France, pick the top positive-AP head via
        attribution_patch_per_head, then verify direct SVD+decode yields
        10 non-empty tokens and sv_ratio in [0, 1]."""
        from llm_surgeon.surgery import load_model
        from llm_surgeon.probe import attribution_patch_per_head

        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        clean = "The capital of France is"
        corrupted = "The capital of Italy is"
        paris_id = int(tok(" Paris", return_tensors="pt")["input_ids"][0, 1].item())
        rome_id = int(tok(" Rome", return_tensors="pt")["input_ids"][0, 1].item())

        r = attribution_patch_per_head(
            model, tok, clean, corrupted,
            correct_token_id=paris_id,
            incorrect_token_id=rome_id,
            direction="denoise",
        )
        # Phase 3.6 cells have unit like "attn.hN" or "ffn". Pick first
        # positive-AP attn.h* cell.
        top_head = next(
            (c for c in r.cells
             if c["ap_recovery"] > 0 and isinstance(c.get("unit"), str)
             and c["unit"].startswith("attn.h")),
            None,
        )
        assert top_head is not None, "expected at least one positive-AP attn head"

        L = int(top_head["layer"])
        h = int(str(top_head["unit"])[len("attn.h"):])

        # Direct SVD decode — mirrors endpoint logic.
        with torch.no_grad():
            W_O = model.model.layers[L].self_attn.o_proj.weight  # [hidden, hidden]
            n_heads = model.config.num_attention_heads
            head_dim = W_O.shape[0] // n_heads
            W_O_h = W_O[:, h*head_dim:(h+1)*head_dim].to(dtype=torch.float32)
            U, S, _ = torch.linalg.svd(W_O_h, full_matrices=False)
            direction = U[:, 0] * S[0]
            W_U = model.lm_head.weight.to(dtype=torch.float32)
            scores = W_U @ direction
            top_ids = torch.topk(scores, 10, largest=True).indices.tolist()
            sv_ratio = float((S[0] ** 2).item()) / float((S ** 2).sum().item())

        decoded = [tok.decode([i], skip_special_tokens=False) for i in top_ids]
        assert len(decoded) == 10
        assert any(s.strip() != "" for s in decoded)
        assert 0.0 <= sv_ratio <= 1.0
