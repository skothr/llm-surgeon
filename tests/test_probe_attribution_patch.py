"""Tests for probe.attribution_patch — gradient-based AP (Phase 3.5)."""

from __future__ import annotations

import torch

from llm_surgeon.probe import _capture_residual_stream_with_grad


class TestCaptureWithGrad:
    def test_captured_tensors_have_grad_fn(self):
        """Every captured (L, sub) tensor keeps its computation graph."""
        # Minimal mock LLaMA-shaped model sufficient for hook invocation.
        class _MockLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(4, 4)
            def forward(self, x):
                return (x + self.self_attn(x),)  # tuple like HF layers

        class _MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(10, 4)  # pyright: ignore[reportAttributeAccessIssue]
                self.model.layers = torch.nn.ModuleList([_MockLayer() for _ in range(2)])  # pyright: ignore[reportAttributeAccessIssue]
                self.lm_head = torch.nn.Linear(4, 10)

            def forward(self, input_ids):
                h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
                for layer in self.model.layers:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
                    h = layer(h)[0]
                return type("Out", (), {"logits": self.lm_head(h)})

        class _MockTok:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids):
                return [str(int(i)) for i in ids]

        model = _MockModel().eval()
        tok = _MockTok()
        captured, logits, tokens = _capture_residual_stream_with_grad(
            model, tok, "hello", sublayers=("attn", "ffn"), layers=None,
        )
        assert len(captured) == 2 * 2  # 2 layers × 2 sublayers
        for key, tensor in captured.items():
            assert tensor.requires_grad, f"{key} must require grad"
            assert tensor.grad_fn is not None, f"{key} must have grad_fn"
        assert logits.requires_grad
        assert len(tokens) == 3

    def test_grad_populates_after_backward(self):
        """Calling .backward() populates .grad on each captured tensor."""
        # (same mock setup as above — reuse the helper pattern)
        # After capture, compute sum = logits.sum(); sum.backward();
        # assert every captured[(L,sub)].grad is not None and non-zero.
        class _MockLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Linear(4, 4)
            def forward(self, x):
                return (x + self.self_attn(x),)

        class _MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(10, 4)  # pyright: ignore[reportAttributeAccessIssue]
                self.model.layers = torch.nn.ModuleList([_MockLayer() for _ in range(2)])  # pyright: ignore[reportAttributeAccessIssue]
                self.lm_head = torch.nn.Linear(4, 10)
            def forward(self, input_ids):
                h = self.model.embed_tokens(input_ids)  # pyright: ignore[reportAttributeAccessIssue,reportCallIssue]
                for layer in self.model.layers:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
                    h = layer(h)[0]
                return type("Out", (), {"logits": self.lm_head(h)})

        class _MockTok:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2, 3]])}
            def convert_ids_to_tokens(self, ids):
                return [str(int(i)) for i in ids]

        model = _MockModel().eval()
        tok = _MockTok()
        captured, logits, _ = _capture_residual_stream_with_grad(
            model, tok, "hello", sublayers=("attn", "ffn"), layers=None,
        )
        logits.sum().backward()
        for key, tensor in captured.items():
            assert tensor.grad is not None, f"{key} missing grad after backward"
            # At least one element must be non-zero (not a constant-zero grad).
            assert tensor.grad.abs().sum().item() > 0, f"{key} grad is all zeros"
