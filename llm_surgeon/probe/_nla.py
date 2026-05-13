"""Natural Language Autoencoder verbalizer + reconstructor (CPU-resident).

The AV (activation → text) and AR (text → activation) form a paired
autoencoder. Round-trip ``h → AV → text → AR → h_pred`` validates
whether the verbalization preserves the residual-stream information.

Both checkpoints from kitft (Anthropic NLA, 2026-05-07). Skips kitft's
SGLang inference path; uses vanilla transformers on CPU at bf16.

First ``load_av()`` downloads ~15 GB; first ``load_ar()`` downloads ~10 GB.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any, cast

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_surgeon import surgery


AV_ID = "kitft/nla-qwen2.5-7b-L20-av"
AR_ID = "kitft/nla-qwen2.5-7b-L20-ar"

_EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def load_av_meta() -> dict[str, Any]:
    """Fetch and parse the AV's nla_meta.yaml sidecar (small file)."""
    path = hf_hub_download(AV_ID, "nla_meta.yaml", cache_dir=surgery.MODEL_CACHE_DIR)
    with open(path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def load_av() -> tuple[Any, Any, dict[str, Any]]:
    """Load AV onto CPU at bf16. First call downloads ~15 GB of safetensors."""
    meta = load_av_meta()
    tok = AutoTokenizer.from_pretrained(AV_ID, cache_dir=surgery.MODEL_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        AV_ID,
        cache_dir=surgery.MODEL_CACHE_DIR,
        dtype="bfloat16",
        device_map="cpu",
    )
    model.eval()
    return model, tok, meta


def nla_verbalize(
    activation: torch.Tensor,
    *,
    model: Any,
    tok: Any,
    meta: dict[str, Any],
    max_new_tokens: int = 200,
) -> str:
    """Return the AV's English explanation of a single ``(d_model,)`` activation.

    Args:
        activation: 1-D tensor of shape ``(meta["d_model"],)``. Any dtype/device —
            the function normalizes to unit L2 then scales by
            ``meta["extraction"]["injection_scale"]`` in fp32 before casting to
            the AV's bf16 input-embed dtype.
        model: The AV model from ``load_av()``.
        tok: The AV tokenizer from ``load_av()``.
        meta: The nla_meta dict from ``load_av()``.
        max_new_tokens: Cap on AV output length. ~1.5 tok/s on CPU.

    Returns:
        The content between ``<explanation>...</explanation>`` tags from the
        AV's output, or the full decoded text if those tags are missing
        (typically signals truncation).
    """
    d = meta["d_model"]
    if activation.shape != (d,):
        raise ValueError(f"expected ({d},), got {tuple(activation.shape)}")

    prompt = meta["prompt_templates"]["av"].format(
        injection_char=meta["tokens"]["injection_char"]
    )
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids: torch.Tensor = enc["input_ids"]
    attn_mask: torch.Tensor = enc["attention_mask"]

    inj_id = meta["tokens"]["injection_token_id"]
    pos = (input_ids[0] == inj_id).nonzero(as_tuple=True)[0]
    if pos.numel() != 1:
        raise RuntimeError(f"expected exactly 1 injection token, found {pos.numel()}")
    p = int(pos.item())
    left = int(input_ids[0, p - 1].item())
    right = int(input_ids[0, p + 1].item())
    if left != meta["tokens"]["injection_left_neighbor_id"]:
        raise RuntimeError(f"injection left-neighbor drift: {left}")
    if right != meta["tokens"]["injection_right_neighbor_id"]:
        raise RuntimeError(f"injection right-neighbor drift: {right}")

    h = activation.detach().float().cpu()
    h = h / h.norm()
    h = h * meta["extraction"]["injection_scale"]

    embed = model.get_input_embeddings()
    inputs_embeds = embed(input_ids).clone()
    inputs_embeds[0, p] = h.to(inputs_embeds.dtype)

    with torch.no_grad():
        out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    m = _EXPLANATION_RE.search(text)
    return m.group(1) if m else text


def load_ar_meta() -> dict[str, Any]:
    """Fetch and parse the AR's nla_meta.yaml sidecar (small file)."""
    path = hf_hub_download(AR_ID, "nla_meta.yaml", cache_dir=surgery.MODEL_CACHE_DIR)
    with open(path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def load_ar() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load AR backbone + value head onto CPU at bf16. First call downloads ~10 GB.

    Returns (backbone, value_head, tokenizer, meta).
    """
    meta = load_ar_meta()
    tok = AutoTokenizer.from_pretrained(AR_ID, cache_dir=surgery.MODEL_CACHE_DIR)
    backbone = AutoModelForCausalLM.from_pretrained(
        AR_ID,
        cache_dir=surgery.MODEL_CACHE_DIR,
        dtype="bfloat16",
        device_map="cpu",
    )
    backbone.eval()
    head_path = hf_hub_download(AR_ID, "value_head.safetensors", cache_dir=surgery.MODEL_CACHE_DIR)
    d = backbone.config.hidden_size
    value_head = torch.nn.Linear(d, d, bias=False, dtype=torch.bfloat16)
    value_head.load_state_dict(OrderedDict(load_file(head_path)))
    value_head.eval()
    return backbone, value_head, tok, meta


def nla_reconstruct(
    explanation: str,
    *,
    backbone: Any, value_head: Any, tok: Any, meta: dict[str, Any],
) -> torch.Tensor:
    """Run AR to reconstruct an activation vector from an AV explanation.

    Returns the raw reconstructed vector (un-normalized). Use ``nla_score``
    for the faithfulness comparison against an original activation.
    """
    prompt = meta["prompt_templates"]["ar"].format(explanation=explanation)
    ids = tok(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = backbone.model(ids).last_hidden_state[0, -1]
        return value_head(out).float().detach().cpu().clone()


def nla_score(
    h_original: torch.Tensor, h_reconstructed: torch.Tensor,
    *, mse_scale: float | None = None,
) -> dict[str, float]:
    """Compare original h to AR's reconstruction.

    Returns:
        ``cosine``: cosine similarity in [-1, 1]. Magnitude-invariant, the
            cleanest direction-fidelity signal.
        ``normalized_mse``: MSE after both vectors L2-normalize-and-rescale
            to ``mse_scale`` (default √d_model). Lower is better; orthogonal
            vectors give ~2.0, identical vectors give 0.
    """
    h = h_original.detach().float().cpu().flatten()
    h_pred = h_reconstructed.detach().float().cpu().flatten()
    cosine = torch.nn.functional.cosine_similarity(
        h.unsqueeze(0), h_pred.unsqueeze(0), dim=1
    ).item()
    scale = mse_scale if mse_scale is not None else math.sqrt(h.numel())
    h_n = h / h.norm().clamp_min(1e-12) * scale
    p_n = h_pred / h_pred.norm().clamp_min(1e-12) * scale
    return {
        "cosine": cosine,
        "normalized_mse": ((h_n - p_n) ** 2).mean().item(),
    }
