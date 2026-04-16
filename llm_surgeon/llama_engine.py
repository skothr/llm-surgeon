"""llama.cpp engine: native GGUF inference via llama-cpp-python.

Provides LlamaEngine for fast generation/logits/perplexity on quantized
GGUF models, plus export_hf_to_gguf for re-exporting modified HF models
back to GGUF format.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger("llm_surgeon.llama_engine")


@dataclass
class GenerateStep:
    """One step of streaming generation."""
    token_id: int
    token_str: str
    logits: np.ndarray | None


def compare_logits(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    top_k: int = 10,
) -> dict:
    """Compare two logit vectors.

    Returns dict with cosine_similarity, kl_divergence, max_logit_diff,
    mean_logit_diff, and top_k_agreement.
    """
    cos_num = np.dot(logits_a, logits_b)
    cos_den = np.linalg.norm(logits_a) * np.linalg.norm(logits_b)
    cosine = float(cos_num / cos_den) if cos_den > 0 else 0.0

    diff = np.abs(logits_a - logits_b)

    pa = np.exp(logits_a - logits_a.max())
    pa /= pa.sum()
    pb = np.exp(logits_b - logits_b.max())
    pb /= pb.sum()
    # Mask zero entries in pa: 0 * log(0 / anything) is defined as 0, but numpy
    # computes it as 0 * -inf = NaN. Only sum over positions with positive pa.
    nz = pa > 0
    kl = float(np.sum(pa[nz] * np.log(pa[nz] / np.clip(pb[nz], 1e-10, None))))

    top_a = set(np.argsort(logits_a)[-top_k:][::-1].tolist())
    top_b = set(np.argsort(logits_b)[-top_k:][::-1].tolist())

    return {
        "cosine_similarity": cosine,
        "kl_divergence": kl,
        "max_logit_diff": float(diff.max()),
        "mean_logit_diff": float(diff.mean()),
        "top_k_agreement": len(top_a & top_b),
    }


class LlamaEngine:
    """Native GGUF inference via llama-cpp-python.

    Wraps llama_cpp.Llama for fast generation, logit extraction,
    and perplexity scoring on quantized GGUF models.
    """

    def __init__(
        self,
        gguf_path: Path,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
    ):
        from llama_cpp import Llama

        self._path = Path(gguf_path)
        log.info("Loading GGUF via llama.cpp: %s", self._path.name)
        self._llm = Llama(
            model_path=str(self._path),
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            n_ctx=n_ctx,
            verbose=False,
        )
        self._n_vocab = self._llm.n_vocab()
        log.info("LlamaEngine ready: %d vocab, ctx=%d", self._n_vocab, n_ctx)

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def n_vocab(self) -> int:
        return self._n_vocab

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if getattr(self, "_llm", None) is not None:
            del self._llm
            self._llm = None
            log.info("LlamaEngine closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    def _require_loaded(self):
        if self._llm is None:
            raise RuntimeError("LlamaEngine is closed")

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        self._require_loaded()
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)  # type: ignore[union-attr]

    def detokenize(self, tokens: list[int]) -> str:
        self._require_loaded()
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")  # type: ignore[union-attr]

    def logits(self, tokens: list[int]) -> np.ndarray:
        """Full vocab logits for the last token position. Shape: (n_vocab,)"""
        self._require_loaded()
        assert self._llm is not None
        llm = self._llm
        llm.reset()
        llm.eval(tokens)
        return np.array(llm.eval_logits[-1], dtype=np.float32)

    def logits_all(self, tokens: list[int]) -> list[np.ndarray]:
        """Full vocab logits for every position. List of (n_vocab,) arrays."""
        self._require_loaded()
        assert self._llm is not None
        llm = self._llm
        llm.reset()
        llm.eval(tokens)
        return [np.array(row, dtype=np.float32) for row in llm.eval_logits]

    def generate(
        self,
        tokens: list[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        emit_logits: bool = True,
    ) -> Iterator[GenerateStep]:
        """Streaming token generation. Greedy when temperature=0."""
        self._require_loaded()
        assert self._llm is not None
        llm = self._llm
        llm.reset()
        llm.eval(tokens)

        generated_ids: list[int] = []
        generated_text = ""

        for _ in range(max_tokens):
            logits_arr = np.array(llm.eval_logits[-1], dtype=np.float32)

            if repetition_penalty != 1.0:
                for tid in set(tokens + generated_ids):
                    if logits_arr[tid] > 0:
                        logits_arr[tid] /= repetition_penalty
                    else:
                        logits_arr[tid] *= repetition_penalty

            if temperature == 0:
                next_id = int(np.argmax(logits_arr))
            else:
                scaled = logits_arr / temperature
                if top_k > 0:
                    threshold = np.partition(scaled, -top_k)[-top_k]
                    scaled[scaled < threshold] = -np.inf
                probs = np.exp(scaled - scaled.max())
                s = probs.sum()
                if not np.isfinite(s) or s <= 0:
                    # Shouldn't happen after top_k + softmax, but guard so we
                    # never hand NaN/zero distribution to np.random.choice.
                    next_id = int(np.argmax(scaled))
                else:
                    probs /= s
                    if top_p < 1.0:
                        sorted_idx = np.argsort(-probs)
                        cum = np.cumsum(probs[sorted_idx])
                        cutoff = np.searchsorted(cum, top_p) + 1
                        mask = np.ones_like(probs, dtype=bool)
                        mask[sorted_idx[:cutoff]] = False
                        probs[mask] = 0.0
                        s2 = probs.sum()
                        if s2 > 0:
                            probs /= s2
                        else:
                            probs[:] = 0.0
                            probs[sorted_idx[0]] = 1.0
                    next_id = int(np.random.choice(len(probs), p=probs))

            token_str = self.detokenize([next_id])
            generated_ids.append(next_id)
            generated_text += token_str

            yield GenerateStep(
                token_id=next_id,
                token_str=token_str,
                logits=logits_arr if emit_logits else None,
            )

            if next_id == llm.token_eos():
                break
            if stop_sequences and any(s in generated_text for s in stop_sequences):
                break

            llm.eval([next_id])

    def perplexity(self, text: str) -> float:
        """Compute perplexity: exp(-1/N * sum(log P(token_i | context)))."""
        tokens = self.tokenize(text, add_bos=True)
        if len(tokens) < 2:
            return float("inf")

        self._require_loaded()
        assert self._llm is not None
        llm = self._llm
        llm.reset()
        llm.eval(tokens)

        nll_sum = 0.0
        count = 0
        for i in range(len(tokens) - 1):
            logits_i = np.array(self._llm.eval_logits[i], dtype=np.float32)
            log_probs = logits_i - np.logaddexp.reduce(logits_i)
            nll_sum -= log_probs[tokens[i + 1]]
            count += 1

        return float(np.exp(nll_sum / count))


def _forward_permute(t: np.ndarray, n_head: int, n_kv_heads: int) -> np.ndarray:
    """Apply the Q/K head interleaving that llama.cpp expects.

    Inverse of gguf_reader._reverse_permute. Must be applied to Q and K
    weight matrices when writing GGUF from HuggingFace format.
    """
    n = n_kv_heads if n_head != n_kv_heads else n_head
    dim = t.shape[0] // n // 2
    return t.reshape(n, 2, dim, *t.shape[1:]).swapaxes(1, 2).reshape(t.shape)


_HF_TO_GGUF_GLOBAL = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

_HF_TO_GGUF_LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


import re as _re

_BYTE_TOKEN_RE = _re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _tokenizer_json(tokenizer):
    """Parse the HF fast tokenizer's serialized JSON, or None if unavailable."""
    if not getattr(tokenizer, "is_fast", False):
        return None
    try:
        import json as _json
        return _json.loads(tokenizer.backend_tokenizer.to_str())
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
        rope_theta = 10000.0
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
