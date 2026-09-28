"""Logit-lens decoding and hidden-state extraction."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from llm_surgeon.probe._capture import _capture_residual_stream
from llm_surgeon.probe._hooks import _get_input_device
from llm_surgeon.probe._types import (
    CompareLogitLensResult,
    HiddenStates,
    LogitLensResult,
)

# Capture-point order within one layer: the embedding precedes layer 0's
# attn add, which precedes its ffn output.
_SUBLAYER_ORDER = {"embed": 0, "attn": 1, "ffn": 2}


def _capture_sort_key(key: tuple[int, str]) -> tuple[int, int]:
    return (key[0], _SUBLAYER_ORDER.get(key[1], len(_SUBLAYER_ORDER)))


def _topk_tokens(pos_probs: torch.Tensor, top_k: int, tokenizer: Any) -> list[dict]:
    """Top-k ``{"token", "token_id", "prob", "rank"}`` rows for one position.

    ``token`` is ``tokenizer.decode([id])``, which drops SentencePiece's
    leading-space marker; match on ``token_id`` when that matters.
    """
    topk_probs, topk_ids = pos_probs.topk(min(top_k, pos_probs.shape[0]))
    return [
        {
            "token": tokenizer.decode([tid]),
            "token_id": tid,
            "prob": tp,
            "rank": rank,
        }
        for rank, (tid, tp) in enumerate(zip(topk_ids.tolist(), topk_probs.tolist()))
    ]


def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layers: list[int] | None = None,
    sublayers: tuple[str, ...] = ("ffn",),
    detach: bool = True,
    on_layer: Callable[[int, str, dict], None] | None = None,
) -> HiddenStates:
    """Extract raw hidden state tensors at specified residual stream capture points.

    Capture points are on the residual stream: the embedding output ("embed",
    keyed ``(0, "embed")``), the post-attention residual add ("attn") and the
    post-FFN residual add / layer output ("ffn"). ``layers`` accepts negative
    indices; keys always use the non-negative index. ``on_layer`` fires in
    forward order (embed, then attn before ffn within each layer).

    Tensors are always detached (captured under ``torch.no_grad()``);
    ``detach=False`` is rejected rather than silently ignored.
    """
    if not detach:
        raise ValueError(
            "detach=False is not supported: states are captured under no_grad. "
            "Use _capture_residual_stream_with_grad for graph-attached tensors."
        )
    captured, prompt_tokens = _capture_residual_stream(
        model, tokenizer, prompt, sublayers=sublayers, layers=layers,
    )

    if on_layer is not None:
        for (layer_idx, sub), tensor in sorted(
            captured.items(), key=lambda kv: _capture_sort_key(kv[0]),
        ):
            on_layer(layer_idx, sub, {"hidden_state": tensor})

    return HiddenStates(states=captured, prompt_tokens=prompt_tokens)


def _cell_metrics(pos_probs: torch.Tensor) -> dict[str, float]:
    """Per-position scalar metrics from a full-vocab softmax distribution.

    entropy is in nats. top1_margin is p(top1) - p(top2), always >= 0.
    """
    # xlogy handles p=0 cleanly (0*log(0) := 0), unlike a bare p*p.log().
    entropy = -torch.special.xlogy(pos_probs, pos_probs).sum().item()
    top2 = pos_probs.topk(2).values
    return {
        "entropy": entropy,
        "top1_prob": top2[0].item(),
        "top1_margin": (top2[0] - top2[1]).item(),
    }


def _pair_metrics(p_a: torch.Tensor, p_b: torch.Tensor) -> dict[str, float]:
    """Per-position comparison metrics between two softmax distributions.

    kl_ab: KL(A||B) in nats (unbounded above).
    js:    Jensen-Shannon divergence in nats (symmetric, 0 <= js <= log 2).
    cosine: cosine similarity of the two full-vocab distributions in [-1, 1];
            always non-negative in practice since both inputs are non-negative.
    top1_delta_prob: p_a(argmax p_a) - p_b(argmax p_b). Signed.
    top1_match: whether the two argmax tokens agree.
    """
    # A small floor on B (and on the mixture) avoids -inf when A has support
    # where B vanishes. xlogy(0, 0) = 0 keeps KL(A||A) = 0 numerically.
    b_safe = p_b.clamp(min=1e-45).log()
    kl_ab = (torch.special.xlogy(p_a, p_a) - p_a * b_safe).sum()

    m = 0.5 * (p_a + p_b)
    m_safe = m.clamp(min=1e-45).log()
    kl_am = (torch.special.xlogy(p_a, p_a) - p_a * m_safe).sum()
    kl_bm = (torch.special.xlogy(p_b, p_b) - p_b * m_safe).sum()
    js = 0.5 * (kl_am + kl_bm)

    cos = F.cosine_similarity(p_a.unsqueeze(0), p_b.unsqueeze(0)).item()

    top1_a = int(p_a.argmax().item())
    top1_b = int(p_b.argmax().item())
    return {
        "kl_ab": kl_ab.item(),
        "js": js.item(),
        "cosine": cos,
        "top1_delta_prob": (p_a[top1_a] - p_b[top1_b]).item(),
        "top1_match": top1_a == top1_b,
    }


def _project_to_logits(model, hidden_state: torch.Tensor) -> torch.Tensor:
    """Apply final RMSNorm + lm_head to a hidden state tensor.

    Args:
        hidden_state: (seq_len, d_model) tensor from the residual stream.

    Returns:
        (seq_len, vocab_size) logit tensor.
    """
    h = hidden_state.unsqueeze(0).to(_get_input_device(model))
    h = model.model.norm(h)
    return model.lm_head(h)[0]


def logit_lens(
    model,
    tokenizer,
    prompt: str,
    top_k: int = 10,
    full_logits: bool = False,
    positions: list[int] | None = None,
    on_layer: Callable[[int, str, dict], None] | None = None,
) -> LogitLensResult:
    """Project each layer's residual stream state through the output head.

    Captures at both post-attention and post-FFN points (sub-layer granularity).
    ``positions`` may be negative (``-1`` is the last token); an index outside
    ``[-seq_len, seq_len)`` raises IndexError. Rows store the resolved index.
    """
    captured, prompt_tokens = _capture_residual_stream(
        model, tokenizer, prompt, sublayers=("attn", "ffn"),
    )

    seq_len = len(prompt_tokens)
    if positions is not None:
        for p in positions:
            if not -seq_len <= p < seq_len:
                raise IndexError(f"position {p} out of range for seq_len={seq_len}")
        resolved_positions = [p % seq_len for p in positions]
    else:
        resolved_positions = list(range(seq_len))

    predictions = []
    logits_dict: dict[tuple[int, str], torch.Tensor] | None = {} if full_logits else None

    for (layer_idx, sublayer), hidden in sorted(captured.items()):
        with torch.no_grad():
            layer_logits = _project_to_logits(model, hidden)

        if full_logits:
            assert logits_dict is not None
            logits_dict[(layer_idx, sublayer)] = layer_logits.cpu()

        probs = F.softmax(layer_logits.float(), dim=-1)

        cb_top_k_per_position = []
        cb_metrics_per_position = []
        for pos in resolved_positions:
            pos_probs = probs[pos]
            metrics = _cell_metrics(pos_probs)
            top_k_list = _topk_tokens(pos_probs, top_k, tokenizer)
            predictions.append({
                "layer": layer_idx,
                "sublayer": sublayer,
                "position": pos,
                "top_k": top_k_list,
                "metrics": metrics,
            })
            cb_top_k_per_position.append(top_k_list)
            cb_metrics_per_position.append(metrics)

        if on_layer is not None:
            cb_logits = layer_logits.cpu() if full_logits else None
            on_layer(layer_idx, sublayer, {
                "hidden_state": hidden,
                "top_k": cb_top_k_per_position,
                "metrics": cb_metrics_per_position,
                "logits": cb_logits,
            })

    return LogitLensResult(
        predictions=predictions,
        logits=logits_dict,
        prompt_tokens=prompt_tokens,
    )


def compare_logit_lens(
    model_a,
    model_b,
    tokenizer,
    prompt: str,
    top_k: int = 10,
    on_layer: Callable[[int, str, dict], None] | None = None,
    layer_map_a: list[int] | None = None,
    layer_map_b: list[int] | None = None,
) -> CompareLogitLensResult:
    """Run logit lens on two models over the same prompt, compute exact per-cell
    comparison metrics (KL, JS, cosine, top-1 delta/match) from the FULL-vocab
    softmax distributions, and stream results per aligned (original_layer, sublayer).

    Alignment is by ORIGINAL layer index. Callers with compressed models should
    pass layer_map_a / layer_map_b where `layer_map_x[compressed_idx] == original_idx`.
    If a map is None, the compressed index IS the original index (identity).
    A map must have one entry per layer of its model (else ValueError). When a
    map repeats an original index (e.g. after ``duplicate_layer``), only the
    last copy in forward order is compared, and a RuntimeWarning names the
    compressed layers left out.

    The two models may sit on different devices; B's distributions are moved
    to A's device for the pairwise metrics.

    The tokenizer must be shared by both models; KL and JS over different vocabs
    are ill-defined.
    """
    for name, layer_map, model in (
        ("layer_map_a", layer_map_a, model_a), ("layer_map_b", layer_map_b, model_b),
    ):
        n = len(model.model.layers)
        if layer_map is not None and len(layer_map) != n:
            raise ValueError(
                f"{name} has {len(layer_map)} entries but the model has {n} layers"
            )

    captured_a, prompt_tokens = _capture_residual_stream(
        model_a, tokenizer, prompt, sublayers=("attn", "ffn"),
    )
    captured_b, _ = _capture_residual_stream(
        model_b, tokenizer, prompt, sublayers=("attn", "ffn"),
    )

    def _reverse(
        name: str, layer_map: list[int] | None, captured: dict[tuple[int, str], torch.Tensor],
    ) -> dict[tuple[int, str], tuple[int, str]]:
        """(original_layer, sublayer) -> compressed key; last copy wins."""
        reverse: dict[tuple[int, str], tuple[int, str]] = {}
        dropped: set[int] = set()
        for (idx, sub) in sorted(captured.keys(), key=_capture_sort_key):
            orig = idx if layer_map is None else layer_map[idx]
            if (orig, sub) in reverse:
                dropped.add(reverse[(orig, sub)][0])
            reverse[(orig, sub)] = (idx, sub)
        if dropped:
            warnings.warn(
                f"{name} maps several layers to one original index (duplicate "
                f"layers); comparing only the last copy, leaving out compressed "
                f"layer(s) {sorted(dropped)}",
                RuntimeWarning, stacklevel=3,
            )
        return reverse

    reverse_a = _reverse("layer_map_a", layer_map_a, captured_a)
    reverse_b = _reverse("layer_map_b", layer_map_b, captured_b)

    # Preserve original-layer order; (attn, ffn) ordering within each layer.
    aligned_keys = sorted(
        set(reverse_a.keys()) & set(reverse_b.keys()), key=_capture_sort_key,
    )

    seq_len = len(prompt_tokens)
    positions = list(range(seq_len))

    comparisons: list[dict] = []
    for (orig_layer, sublayer) in aligned_keys:
        hidden_a = captured_a[reverse_a[(orig_layer, sublayer)]]
        hidden_b = captured_b[reverse_b[(orig_layer, sublayer)]]

        with torch.no_grad():
            logits_a = _project_to_logits(model_a, hidden_a)
            logits_b = _project_to_logits(model_b, hidden_b)
        probs_a = F.softmax(logits_a.float(), dim=-1)
        probs_b = F.softmax(logits_b.float(), dim=-1).to(probs_a.device)

        cb_frames = []
        for pos in positions:
            pa = probs_a[pos]
            pb = probs_b[pos]
            cell = {
                "original_layer": orig_layer,
                "sublayer": sublayer,
                "position": pos,
                "top_k_a": _topk_tokens(pa, top_k, tokenizer),
                "top_k_b": _topk_tokens(pb, top_k, tokenizer),
                "metrics_a": _cell_metrics(pa),
                "metrics_b": _cell_metrics(pb),
                "compare": _pair_metrics(pa, pb),
            }
            comparisons.append(cell)
            cb_frames.append(cell)

        if on_layer is not None:
            on_layer(orig_layer, sublayer, {
                "cells": cb_frames,
                "hidden_state_a": hidden_a,
                "hidden_state_b": hidden_b,
            })

    return CompareLogitLensResult(
        comparisons=comparisons,
        prompt_tokens=prompt_tokens,
        aligned_keys=aligned_keys,
    )


def layer_predictions_table(result: LogitLensResult, position: int = -1) -> str:
    """Format a single position's logit lens predictions as a readable table."""
    return result.summary(position=position)

