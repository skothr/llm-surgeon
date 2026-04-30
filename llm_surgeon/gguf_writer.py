"""GGUF writer: export HuggingFace LlamaForCausalLM models to GGUF.

Inverse of ``gguf_reader``. Tokenizer reconstruction handles both
SentencePiece-style ("llama") and GPT-2-style ("gpt2") fast tokenizers.
Tensor name maps and the Q/K head permutation are derived from
``gguf_reader`` to keep the read/write paths from drifting.
"""

import logging
import re
from pathlib import Path

import numpy as np

from llm_surgeon.gguf_reader import _GGUF_GLOBAL, _GGUF_LAYER

log = logging.getLogger("llm_surgeon.gguf_writer")


# Inverse of gguf_reader's GGUF→HF maps. Imported and inverted here so the
# two modules can never drift.
_HF_TO_GGUF_GLOBAL = {v: k for k, v in _GGUF_GLOBAL.items()}
_HF_TO_GGUF_LAYER = {v: k for k, v in _GGUF_LAYER.items()}

_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _forward_permute(t: np.ndarray, n_head: int, n_kv_heads: int) -> np.ndarray:
    """Apply the Q/K head interleaving that llama.cpp expects.

    Inverse of gguf_reader._reverse_permute. Must be applied to Q and K
    weight matrices when writing GGUF from HuggingFace format.
    """
    n = n_kv_heads if n_head != n_kv_heads else n_head
    dim = t.shape[0] // n // 2
    return t.reshape(n, 2, dim, *t.shape[1:]).swapaxes(1, 2).reshape(t.shape)


def _tokenizer_json(tokenizer) -> dict | None:
    """Parse the HF fast tokenizer's serialized JSON, or None if unavailable."""
    if not getattr(tokenizer, "is_fast", False):
        return None
    try:
        import json
        return json.loads(tokenizer.backend_tokenizer.to_str())
    except Exception:
        return None


def _detect_tokenizer_style(tok_json: dict | None) -> str:
    """Classify tokenizer as 'llama' (SentencePiece-style) or 'gpt2' (BPE-style).

    Both internally use BPE in HF fast tokenizers, but they differ in how
    whitespace is handled: SentencePiece uses a Metaspace pre-tokenizer
    (▁-prefixed tokens), BPE/GPT-2 uses a ByteLevel pre-tokenizer.
    """
    if tok_json is None:
        return "llama"
    pre = tok_json.get("pre_tokenizer") or {}
    pre_type = pre.get("type", "")
    if pre_type == "Sequence":
        children = [p.get("type", "") for p in pre.get("pretokenizers", [])]
        if any(t == "ByteLevel" for t in children):
            return "gpt2"
        if any(t == "Metaspace" for t in children):
            return "llama"
    if pre_type == "ByteLevel":
        return "gpt2"
    if pre_type == "Metaspace":
        return "llama"
    return "llama"


def _classify_token_type(tok_str: str, special_token_strs: set[str]) -> int:
    """Map a token string to the GGUF token_type code.

    1 = normal, 2 = unknown, 3 = control, 6 = byte.
    """
    if _BYTE_TOKEN_RE.match(tok_str):
        return 6
    if tok_str in special_token_strs:
        return 3
    return 1


def _write_tokenizer(writer, tokenizer) -> None:
    """Write vocab, merges (if BPE-style), scores, types, and specials to GGUF.

    Handles both SentencePiece-style ("llama") and GPT-2-style ("gpt2")
    tokenizers; picks the right tokenizer_model based on the fast
    tokenizer's pre-tokenizer.
    """
    vocab = tokenizer.get_vocab()
    tokens = [""] * len(vocab)
    for tok, idx in vocab.items():
        if 0 <= idx < len(tokens):
            tokens[idx] = tok

    tok_json = _tokenizer_json(tokenizer)
    style = _detect_tokenizer_style(tok_json)
    writer.add_tokenizer_model(style)
    writer.add_token_list(tokens)

    # Collect special-token strings so we can tag them as control (type 3).
    special_strs: set[str] = set()
    for attr in ("bos_token", "eos_token", "pad_token", "unk_token", "sep_token", "cls_token", "mask_token"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, str):
            special_strs.add(v)
    if hasattr(tokenizer, "additional_special_tokens"):
        for v in tokenizer.additional_special_tokens or []:
            if isinstance(v, str):
                special_strs.add(v)

    # BPE merges live in tokenizer.json under model.merges. Unigram/SP-style
    # llama.cpp runtimes ignore merges, so writing them is harmless, but the
    # gpt2 runtime *requires* them.
    merges: list[str] = []
    if tok_json is not None:
        mdl = tok_json.get("model") or {}
        raw_merges = mdl.get("merges") or []
        for m in raw_merges:
            if isinstance(m, list):
                merges.append(" ".join(m))
            elif isinstance(m, str):
                merges.append(m)
    if merges:
        writer.add_token_merges(merges)
    elif style == "gpt2":
        log.warning("GPT-2-style tokenizer detected but no merges found; "
                    "exported GGUF may fail to tokenize correctly.")

    scores = [0.0] * len(tokens)
    token_types = [_classify_token_type(t, special_strs) for t in tokens]
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    if getattr(tokenizer, "bos_token_id", None) is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        try:
            writer.add_pad_token_id(tokenizer.pad_token_id)
        except Exception:
            pass
    if getattr(tokenizer, "unk_token_id", None) is not None:
        try:
            writer.add_unk_token_id(tokenizer.unk_token_id)
        except Exception:
            pass

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and chat_template.strip():
        try:
            writer.add_chat_template(chat_template)
        except Exception:
            log.exception("Failed to write chat_template")


def export_hf_to_gguf(model, tokenizer, output_path: Path) -> Path:
    """Export a HuggingFace LlamaForCausalLM to F16 GGUF.

    Writes metadata, tokenizer, and all weights as F16 tensors using
    gguf.GGUFWriter. Q and K matrices are forward-permuted to match
    the layout llama.cpp expects.
    """
    import gguf

    output_path = Path(output_path)
    config = model.config
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads

    writer = gguf.GGUFWriter(str(output_path), arch="llama")

    writer.add_block_count(config.num_hidden_layers)
    writer.add_embedding_length(config.hidden_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_vocab_size(config.vocab_size)
    if hasattr(config, "rms_norm_eps"):
        writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        # Newer transformers stores RoPE config under config.rope_parameters.
        rope_params = getattr(config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta")
    if rope_theta is None:
        raise ValueError(
            "Cannot export model with config.rope_theta=None. "
            "Set config.rope_theta explicitly before export "
            "(e.g., 10000.0 for LLaMA 2, 500000.0 for LLaMA 3, 1000000.0 for Mistral)."
        )
    writer.add_rope_freq_base(rope_theta)
    head_dim = config.hidden_size // n_heads
    writer.add_rope_dimension_count(head_dim)
    writer.add_file_type(gguf.GGMLQuantizationType.F16)

    _write_tokenizer(writer, tokenizer)

    state_dict = model.state_dict()
    for hf_name, param in state_dict.items():
        arr = param.float().cpu().numpy()

        if hf_name in _HF_TO_GGUF_GLOBAL:
            gguf_name = _HF_TO_GGUF_GLOBAL[hf_name]
        elif hf_name.startswith("model.layers."):
            parts = hf_name.split(".", 3)
            layer_idx = parts[2]
            suffix = parts[3]
            gguf_suffix = _HF_TO_GGUF_LAYER.get(suffix)
            if gguf_suffix is None:
                log.debug("Skipping unmapped tensor: %s", hf_name)
                continue
            gguf_name = f"blk.{layer_idx}.{gguf_suffix}"
        else:
            log.debug("Skipping unmapped tensor: %s", hf_name)
            continue

        if ".attn_q." in gguf_name:
            arr = _forward_permute(arr, n_heads, n_heads)
        elif ".attn_k." in gguf_name:
            arr = _forward_permute(arr, n_heads, n_kv_heads)

        if arr.ndim == 1:
            writer.add_tensor(gguf_name, arr.astype(np.float32))
        else:
            writer.add_tensor(gguf_name, arr.astype(np.float16))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    log.info("Exported F16 GGUF: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path
