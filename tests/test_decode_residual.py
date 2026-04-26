"""Tests for POST /api/sessions/{name}/decode-residual (Phase 3.11).

Mirrors test_decode_neuron.py / test_decode_head.py — a tiny mock model
+ tokenizer, then routing-and-validation tests, plus one TinyLlama
parity test against probe.logit_lens at the last layer's "ffn" point.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import os

import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient


# ----- Mock model/tokenizer -----

class _MockTokenizer:
    """Minimal HF-like tokenizer: whitespace-split + per-id decode."""
    vocab = ("<pad>", "the", "cat", "sat", "on", "mat")

    def __call__(self, text: str, return_tensors: str = "pt", **_):  # noqa: ARG002
        ids = [self.vocab.index(t) if t in self.vocab else 0 for t in text.split()]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def convert_ids_to_tokens(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [self.vocab[i] for i in ids]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:  # noqa: ARG002
        if hasattr(ids, "tolist"):  # type: ignore[attr-defined]
            ids = ids.tolist()  # type: ignore[union-attr]
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(self.vocab[i] for i in ids)


class _MockLayer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(d)
        self.self_attn = nn.Linear(d, d)
        self.post_attention_layernorm = nn.LayerNorm(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class _MockInner(nn.Module):
    def __init__(self, vocab: int, d: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_MockLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)


class _MockModel(nn.Module):
    def __init__(self, vocab: int = 6, d: int = 8, n_layers: int = 3) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.model = _MockInner(vocab, d, n_layers)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = SimpleNamespace(
            num_hidden_layers=n_layers,
            hidden_size=d,
            num_attention_heads=1,
            intermediate_size=d,
            vocab_size=vocab,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(self.model.norm(x))


# ----- Fixtures -----

@pytest.fixture
def app_with_mock_session():
    from gui.backend.app import app  # noqa: PLC0415
    from gui.backend.routes.sessions import get_manager  # noqa: PLC0415

    mgr = get_manager()
    session_name = "mock-decode-residual"
    mgr._sessions[session_name] = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        name=session_name,
        model=_MockModel(),
        tokenizer=_MockTokenizer(),
        llama=None,
        dirty=False,
        original_layer=lambda i: i,
        _layer_map=[],
    )
    try:
        yield (app, session_name)
    finally:
        mgr._sessions.pop(session_name, None)  # pyright: ignore[reportAttributeAccessIssue]


# ----- Validation tests -----

class TestDecodeResidualUnit:
    def test_404_missing_session(self, app_with_mock_session) -> None:
        app, _ = app_with_mock_session
        client = TestClient(app)
        r = client.post("/api/sessions/missing/decode-residual", json={
            "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 0,
        })
        assert r.status_code == 404

    def test_500_no_model(self, app_with_mock_session) -> None:
        from gui.backend.routes.sessions import get_manager  # noqa: PLC0415
        app, name = app_with_mock_session
        get_manager()._sessions[name].model = None  # pyright: ignore[reportAttributeAccessIssue]
        client = TestClient(app)
        r = client.post(f"/api/sessions/{name}/decode-residual", json={
            "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 0,
        })
        assert r.status_code == 500

    def test_400_invalid_sublayer(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        r = client.post(f"/api/sessions/{name}/decode-residual", json={
            "prompt": "the cat sat", "layer": 0, "sublayer": "embed", "position": 0,
        })
        assert r.status_code == 400
        assert "sublayer" in r.json()["detail"]

    def test_400_layer_out_of_range(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        r = client.post(f"/api/sessions/{name}/decode-residual", json={
            "prompt": "the cat sat", "layer": 99, "sublayer": "ffn", "position": 0,
        })
        assert r.status_code == 400
        assert "layer" in r.json()["detail"]

    def test_400_position_out_of_range(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        r = client.post(f"/api/sessions/{name}/decode-residual", json={
            "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 99,
        })
        assert r.status_code == 400
        assert "position" in r.json()["detail"]

    def test_top_k_clamped(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        r = client.post(f"/api/sessions/{name}/decode-residual", json={
            "prompt": "the cat sat", "layer": 0, "sublayer": "ffn", "position": 0, "top_k": 999,
        })
        assert r.status_code == 200
        body = r.json()
        # mock vocab is 6; clamp = min(999, 50, 6) = 6
        assert len(body["top_tokens"]) == 6
        assert len(body["bottom_tokens"]) == 6

    def test_response_shape(self, app_with_mock_session) -> None:
        app, name = app_with_mock_session
        client = TestClient(app)
        r = client.post(f"/api/sessions/{name}/decode-residual", json={
            "prompt": "the cat sat", "layer": 1, "sublayer": "attn", "position": 1, "top_k": 3,
        })
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"top_tokens", "bottom_tokens", "prompt_tokens"}
        assert len(body["top_tokens"]) == 3
        assert len(body["bottom_tokens"]) == 3
        top_logits = [t["logit"] for t in body["top_tokens"]]
        bot_logits = [t["logit"] for t in body["bottom_tokens"]]
        assert top_logits == sorted(top_logits, reverse=True)
        assert bot_logits == sorted(bot_logits)
        assert body["prompt_tokens"] == ["the", "cat", "sat"]


# ----- TinyLlama parity test -----

def _tinyllama_cached() -> bool:
    env_cache = os.environ.get("TINYLLAMA_CACHE")
    if env_cache:
        return Path(env_cache).exists()
    default = Path("testing/.cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0")
    return default.exists() or Path(".cache/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestDecodeResidualTinyLlama:
    def test_decode_residual_matches_logit_lens(self) -> None:
        """At the last layer's "ffn" point, decode-residual's top-1 token must
        equal probe.logit_lens()'s top-1 token at the same (layer, sublayer,
        position) — both go through the same final-norm + lm_head."""
        from llm_surgeon.surgery import load_model  # noqa: PLC0415
        from llm_surgeon.probe import logit_lens  # noqa: PLC0415
        from gui.backend.app import app  # noqa: PLC0415
        from gui.backend.routes.sessions import get_manager  # noqa: PLC0415

        model, tok = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        prompt = "The Eiffel Tower is in"

        mgr = get_manager()
        name = "tinyllama-decode-residual-test"
        mgr._sessions[name] = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            name=name,
            model=model,
            tokenizer=tok,
            llama=None,
            dirty=False,
            original_layer=lambda i: i,
            _layer_map=[],
        )

        try:
            last_layer = len(model.model.layers) - 1
            result = logit_lens(model, tok, prompt, top_k=1)
            seq_len = len(result.prompt_tokens)
            ref_pred = next(p for p in result.predictions
                            if p["layer"] == last_layer
                            and p["sublayer"] == "ffn"
                            and p["position"] == seq_len - 1)
            ref_top1 = ref_pred["top_k"][0]["token"]

            client = TestClient(app)
            r = client.post(f"/api/sessions/{name}/decode-residual", json={
                "prompt": prompt, "layer": last_layer, "sublayer": "ffn",
                "position": seq_len - 1, "top_k": 1,
            })
            assert r.status_code == 200, r.text
            endpoint_top1 = r.json()["top_tokens"][0]["token"]
            assert endpoint_top1 == ref_top1, f"endpoint {endpoint_top1!r} != logit_lens {ref_top1!r}"
        finally:
            mgr._sessions.pop(name, None)  # pyright: ignore[reportAttributeAccessIssue]
            del model
            torch.cuda.empty_cache()
