"""GGUF writer: export HuggingFace LlamaForCausalLM models to GGUF.

Inverse of ``gguf_reader``. Tokenizer reconstruction handles both
SentencePiece-style ("llama") and GPT-2-style ("gpt2") fast tokenizers.
The tensor name maps and the Llama-3 split pattern are imported from
``gguf_reader`` so the read/write paths cannot drift; the Q/K head
permutation here is the inverse of ``gguf_reader._reverse_permute``, and
tests check the two against each other.
"""

import logging
import re
from pathlib import Path

import numpy as np

from llm_surgeon.gguf_reader import _GGUF_GLOBAL, _GGUF_LAYER, _ROPE_FREQS, LLAMA3_SPLIT_REGEX

log = logging.getLogger("llm_surgeon.gguf_writer")


# Inverse of gguf_reader's GGUF→HF maps. Imported and inverted here so the
# two modules can never drift.
_HF_TO_GGUF_GLOBAL = {v: k for k, v in _GGUF_GLOBAL.items()}
_HF_TO_GGUF_LAYER = {v: k for k, v in _GGUF_LAYER.items()}

_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

# HF model_type values whose weights and attention match llama.cpp's "llama"
# architecture (convert_hf_to_gguf.py writes Mistral as "llama" too). Others,
# e.g. Qwen2 with its q/k/v biases, would export as a GGUF that loads but
# computes different outputs.
_LLAMA_MODEL_TYPES = ("llama", "mistral")

# GGUF tokenizer.ggml.token_type codes (llama.cpp LLAMA_TOKEN_TYPE_*)
_TOKEN_NORMAL, _TOKEN_UNKNOWN, _TOKEN_CONTROL, _TOKEN_UNUSED, _TOKEN_BYTE = 1, 2, 3, 5, 6


def _forward_permute(t: np.ndarray, n_groups: int) -> np.ndarray:
    """Apply the Q/K head interleaving that llama.cpp expects.

    Same as ``permute`` in llama.cpp's convert_hf_to_gguf.py. ``n_groups``
    is the number of heads in the matrix: the attention head count for Q,
    the KV head count for K. Inverse of gguf_reader._reverse_permute.
    """
    dim = t.shape[0] // n_groups // 2
    return t.reshape(n_groups, 2, dim, *t.shape[1:]).swapaxes(1, 2).reshape(t.shape)


def _tokenizer_json(tokenizer) -> dict | None:
    """Parse the HF fast tokenizer's serialized JSON, or None if unavailable."""
    if not getattr(tokenizer, "is_fast", False):
        return None
    try:
        import json
        return json.loads(tokenizer.backend_tokenizer.to_str())
    except Exception:
        # Without the JSON the style defaults to "llama" and no merges are
        # written, which is wrong for a byte-level tokenizer; say so.
        log.warning(
            "Could not serialize the fast tokenizer; exporting it as "
            "SentencePiece-style without merges", exc_info=True,
        )
        return None


def _pre_tokenizer_steps(tok_json: dict | None) -> list[dict]:
    pre = (tok_json or {}).get("pre_tokenizer") or {}
    if pre.get("type") == "Sequence":
        return list(pre.get("pretokenizers", []))
    return [pre] if pre else []


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


def _detect_pre_tokenizer(tok_json: dict | None) -> str | None:
    """llama.cpp ``tokenizer.ggml.pre`` name for a byte-level tokenizer.

    "llama-bpe" for the Llama-3 split pattern, "gpt-2" for a plain ByteLevel
    pre-tokenizer with its built-in regex, None when neither matches.
    """
    steps = _pre_tokenizer_steps(tok_json)
    for step in steps:
        if step.get("type") == "Split":
            pattern = (step.get("pattern") or {}).get("Regex")
            return "llama-bpe" if pattern == LLAMA3_SPLIT_REGEX else None
    if any(s.get("type") == "ByteLevel" and s.get("use_regex", True) for s in steps):
        return "gpt-2"
    return None


def _classify_token_type(tok_str: str, special_token_strs: set[str], unk_token: str | None) -> int:
    """Map a token string to the GGUF token_type code.

    1 = normal, 2 = unknown, 3 = control, 6 = byte. (5 = unused marks the
    padding entries that ``_write_tokenizer`` adds.)
    """
    if unk_token is not None and tok_str == unk_token:
        return _TOKEN_UNKNOWN
    if _BYTE_TOKEN_RE.match(tok_str):
        return _TOKEN_BYTE
    if tok_str in special_token_strs:
        return _TOKEN_CONTROL
    return _TOKEN_NORMAL


def _token_list(tokenizer, n_vocab: int) -> tuple[list[str], set[int]]:
    """Token strings indexed by id, padded to the embedding's ``n_vocab`` rows.

    Ids with no token (a padded embedding, gaps between added tokens) get a
    ``[PAD{id}]`` placeholder, as convert_hf_to_gguf.py writes them; their
    indices are returned so they can be typed as unused. llama.cpp requires
    the token list length to equal the embedding rows.
    """
    vocab = tokenizer.get_vocab()
    max_id = max(vocab.values(), default=-1)
    if max_id >= n_vocab:
        raise ValueError(
            f"Tokenizer has token id {max_id}, but the model's embedding has only "
            f"{n_vocab} rows; resize the embeddings before export"
        )
    by_id: list[str | None] = [None] * n_vocab
    for tok, idx in vocab.items():
        by_id[idx] = tok
    padded = {i for i, tok in enumerate(by_id) if tok is None}
    return [tok if tok is not None else f"[PAD{i}]" for i, tok in enumerate(by_id)], padded


def _write_tokenizer(writer, tokenizer, n_vocab: int) -> None:
    """Write vocab, merges (if BPE-style), scores, types, and specials to GGUF.

    Handles both SentencePiece-style ("llama") and GPT-2-style ("gpt2")
    tokenizers; picks the right tokenizer_model based on the fast
    tokenizer's pre-tokenizer.
    """
    tokens, padded = _token_list(tokenizer, n_vocab)

    tok_json = _tokenizer_json(tokenizer)
    style = _detect_tokenizer_style(tok_json)
    writer.add_tokenizer_model(style)
    if style == "gpt2":
        pre = _detect_pre_tokenizer(tok_json)
        if pre is not None:
            writer.add_tokenizer_pre(pre)
        else:
            log.warning(
                "Byte-level tokenizer with an unrecognized pre-tokenizer; llama.cpp "
                "will fall back to its default split pattern and token ids may differ."
            )
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
    unk_token = getattr(tokenizer, "unk_token", None)
    unk_token = unk_token if isinstance(unk_token, str) else None

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
    token_types = [
        _TOKEN_UNUSED if i in padded else _classify_token_type(t, special_strs, unk_token)
        for i, t in enumerate(tokens)
    ]
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    if getattr(tokenizer, "bos_token_id", None) is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        writer.add_pad_token_id(tokenizer.pad_token_id)
    if getattr(tokenizer, "unk_token_id", None) is not None:
        writer.add_unk_token_id(tokenizer.unk_token_id)

    # Record whether encode() adds BOS/EOS; without the keys llama.cpp and
    # gguf_reader fall back to per-vocab-type defaults that may differ.
    probe = tokenizer.encode("a")
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    writer.add_add_bos_token(bos_id is not None and probe[:1] == [bos_id])
    writer.add_add_eos_token(eos_id is not None and len(probe) > 1 and probe[-1:] == [eos_id])

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and chat_template.strip():
        try:
            writer.add_chat_template(chat_template)
        except Exception:
            log.exception("Failed to write chat_template")


def _rope_params(config) -> dict:
    """The model's RoPE parameters (``rope_parameters`` on transformers 5,
    ``rope_scaling`` before that), or {} when there are none."""
    params = getattr(config, "rope_parameters", None)
    if not isinstance(params, dict):
        params = getattr(config, "rope_scaling", None)
    return params if isinstance(params, dict) else {}


def _llama3_rope_freqs(model, rope_theta: float, head_dim: int) -> np.ndarray:
    """llama.cpp ``rope_freqs`` factors for a llama3-scaled rotary embedding.

    llama.cpp divides the unscaled inverse frequencies by these factors, so
    they are the ratio of the unscaled to the model's scaled ``inv_freq``.
    """
    inv_freq = next(
        (b for name, b in model.named_buffers() if name.endswith("rotary_emb.inv_freq")),
        None,
    )
    if inv_freq is None:
        raise ValueError("llama3 RoPE scaling set but the model has no rotary_emb.inv_freq buffer")
    base = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    return (base / inv_freq.double().cpu().numpy()).astype(np.float32)


def _gguf_tensor_name(hf_name: str) -> str | None:
    if hf_name in _HF_TO_GGUF_GLOBAL:
        return _HF_TO_GGUF_GLOBAL[hf_name]
    if hf_name.startswith("model.layers."):
        parts = hf_name.split(".", 3)
        gguf_suffix = _HF_TO_GGUF_LAYER.get(parts[3]) if len(parts) == 4 else None
        if gguf_suffix is not None:
            return f"blk.{parts[2]}.{gguf_suffix}"
    return None


def export_hf_to_gguf(model, tokenizer, output_path: Path) -> Path:
    """Export a HuggingFace LlamaForCausalLM to F16 GGUF.

    Writes metadata, tokenizer, and all weights as F16 tensors using
    gguf.GGUFWriter. Q and K matrices are forward-permuted to match
    the layout llama.cpp expects.

    Only LLaMA-architecture models (``model_type`` llama or mistral) are
    accepted; anything else raises ValueError rather than being written as
    a "llama" GGUF with dropped or misread tensors.

    Requires the optional ``gguf`` package (``pip install 'llm-surgeon[gguf]'``).
    """
    try:
        import gguf
    except ImportError as e:
        raise ImportError(
            "export_hf_to_gguf needs the `gguf` package, which is an optional "
            "dependency. Install it with `pip install 'llm-surgeon[gguf]'`."
        ) from e

    output_path = Path(output_path)
    config = model.config
    model_type = getattr(config, "model_type", None)
    if model_type not in _LLAMA_MODEL_TYPES:
        raise ValueError(
            f"export_hf_to_gguf writes llama-architecture GGUFs only; got "
            f"model_type={model_type!r} (supported: {', '.join(_LLAMA_MODEL_TYPES)})"
        )
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads or n_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_heads

    state_dict = model.state_dict()
    names = {hf_name: _gguf_tensor_name(hf_name) for hf_name in state_dict}
    unmapped = [hf_name for hf_name, gguf_name in names.items() if gguf_name is None]
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} model tensors have no llama GGUF mapping, e.g. "
            f"{unmapped[:5]}; exporting without them would change the model's outputs"
        )

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        # Newer transformers stores RoPE config under config.rope_parameters.
        rope_theta = _rope_params(config).get("rope_theta")
    if rope_theta is None:
        raise ValueError(
            "Cannot export model with config.rope_theta=None. "
            "Set config.rope_theta explicitly before export "
            "(e.g., 10000.0 for LLaMA 2, 500000.0 for LLaMA 3, 1000000.0 for Mistral)."
        )
    rope = _rope_params(config)
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    if rope_type not in ("default", "linear", "llama3"):
        raise ValueError(
            f"RoPE scaling type {rope_type!r} cannot be exported to a llama GGUF "
            "(supported: default, linear, llama3)"
        )

    n_vocab = model.get_input_embeddings().weight.shape[0]

    writer = gguf.GGUFWriter(str(output_path), arch="llama")

    writer.add_block_count(config.num_hidden_layers)
    writer.add_embedding_length(config.hidden_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_vocab_size(n_vocab)
    if hasattr(config, "rms_norm_eps"):
        writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_rope_dimension_count(head_dim)
    if rope_type == "linear":
        writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        writer.add_rope_scaling_factor(float(rope["factor"]))
    writer.add_file_type(gguf.GGMLQuantizationType.F16)

    _write_tokenizer(writer, tokenizer, n_vocab)

    if rope_type == "llama3":
        writer.add_tensor(_ROPE_FREQS, _llama3_rope_freqs(model, rope_theta, head_dim))

    f16_max = float(np.finfo(np.float16).max)
    for hf_name, param in state_dict.items():
        gguf_name = names[hf_name]
        assert gguf_name is not None
        arr = param.float().cpu().numpy()

        if ".attn_q." in gguf_name:
            arr = _forward_permute(arr, n_heads)
        elif ".attn_k." in gguf_name:
            arr = _forward_permute(arr, n_kv_heads)

        if arr.ndim == 1:
            writer.add_tensor(gguf_name, arr.astype(np.float32))
        else:
            if arr.size and float(np.abs(arr).max()) > f16_max:
                log.warning(
                    "%s has values beyond the float16 range; clamping to +/-%g",
                    hf_name, f16_max,
                )
                arr = np.clip(arr, -f16_max, f16_max)
            writer.add_tensor(gguf_name, arr.astype(np.float16))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    log.info("Exported F16 GGUF: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path
