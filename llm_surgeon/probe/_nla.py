"""Natural Language Autoencoder verbalizer (CPU-resident).

Loads kitft/nla-qwen2.5-7b-L20-av onto CPU at bf16 and turns a single
(d_model,) residual-stream activation from Qwen2.5-7B layer 20 into a
short English description.

Skips kitft's SGLang inference path; uses vanilla
``transformers.generate(inputs_embeds=...)`` so this runs on CPU
without a GPU server. First ``load_av()`` call downloads ~15 GB.

Released by Anthropic 2026-05-07 (https://transformer-circuits.pub/2026/nla/).
"""

from __future__ import annotations

import re
from typing import Any, cast

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_surgeon import surgery


AV_ID = "kitft/nla-qwen2.5-7b-L20-av"

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
