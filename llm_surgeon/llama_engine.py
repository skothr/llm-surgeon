"""llama.cpp engine: native GGUF inference via llama-cpp-python.

Provides LlamaEngine for fast generation/logits/perplexity on quantized
GGUF models, plus export_hf_to_gguf for re-exporting modified HF models
back to GGUF format.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from llama_cpp import Llama  # pyright: ignore[reportMissingImports]

log = logging.getLogger("llm_surgeon.llama_engine")


@dataclass
class GenerateStep:
    """One step of streaming generation."""
    token_id: int
    token_str: str
    logits: np.ndarray | None


def _sample(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    rng: np.random.Generator,
) -> int:
    """Sample one token id from ``logits`` using llama.cpp's filter order:
    temperature → top_k → top_p → min_p → multinomial. Callers should
    handle temperature == 0 (greedy) themselves; this assumes temperature > 0.
    """
    scaled = logits / temperature
    if top_k > 0 and top_k < scaled.shape[-1]:
        threshold = np.partition(scaled, -top_k)[-top_k]
        scaled = np.where(scaled < threshold, -np.inf, scaled)
    probs = np.exp(scaled - scaled.max())
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        # Shouldn't happen after top_k + softmax, but guard so we never hand
        # NaN/zero distribution to rng.choice.
        return int(np.argmax(logits))
    probs = probs / s
    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        cum = np.cumsum(probs[sorted_idx])
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        mask = np.ones_like(probs, dtype=bool)
        mask[sorted_idx[:cutoff]] = False
        probs = np.where(mask, 0.0, probs)
        s2 = probs.sum()
        if s2 > 0:
            probs = probs / s2
        else:
            return int(sorted_idx[0])
    if min_p > 0.0:
        # Keep tokens with prob ≥ min_p × max(prob). Unlike top_p (quantile),
        # min_p is a relative-floor filter robust to long-tail distributions.
        thresh = min_p * probs.max()
        probs = np.where(probs >= thresh, probs, 0.0)
        s3 = probs.sum()
        if s3 > 0:
            probs = probs / s3
        else:
            return int(np.argmax(logits))
    return int(rng.choice(len(probs), p=probs))


def compare_logits(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    top_k: int = 10,
) -> dict:
    """Compare two logit vectors.

    Returns dict with cosine_similarity, kl_divergence, max_logit_diff,
    mean_logit_diff, and top_k_agreement.
    """
    # Mean-center before cosine: softmax(l) == softmax(l + c), so logits that
    # differ only by a constant describe identical distributions. Raw cosine
    # does not respect this shift invariance; centered cosine does.
    a_centered = logits_a - logits_a.mean()
    b_centered = logits_b - logits_b.mean()
    cos_num = np.dot(a_centered, b_centered)
    cos_den = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
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
        from llama_cpp import Llama  # pyright: ignore[reportMissingImports]

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
        # During interpreter shutdown, llama_cpp or logging may already be torn
        # down; swallow any error so __del__ stays silent instead of printing a
        # spurious traceback.
        try:
            self.close()
        except Exception:
            pass

    def _engine(self) -> "Llama":
        if self._llm is None:
            raise RuntimeError("LlamaEngine is closed")
        return self._llm

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        return self._engine().tokenize(text.encode("utf-8"), add_bos=add_bos)

    def detokenize(self, tokens: list[int]) -> str:
        return self._engine().detokenize(tokens).decode("utf-8", errors="replace")

    def logits(self, tokens: list[int]) -> np.ndarray:
        """Full vocab logits for the last token position. Shape: (n_vocab,)"""
        llm = self._engine()
        llm.reset()
        llm.eval(tokens)
        return np.array(llm.eval_logits[-1], dtype=np.float32)

    def logits_all(self, tokens: list[int]) -> list[np.ndarray]:
        """Full vocab logits for every position. List of (n_vocab,) arrays."""
        llm = self._engine()
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
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        stop_sequences: list[str] | None = None,
        emit_logits: bool = True,
        seed: int | None = None,
    ) -> Iterator[GenerateStep]:
        """Streaming token generation. Greedy when temperature=0.

        Sampler order mirrors llama.cpp: repetition_penalty → temperature →
        top_k → top_p → min_p → multinomial. Pass ``seed`` for reproducible
        sampling (temperature > 0 only; greedy is already deterministic).
        """
        llm = self._engine()
        llm.reset()
        llm.eval(tokens)

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        generated_ids: list[int] = []
        generated_text = ""
        # llama.cpp's detokenize on a single-token list strips the leading
        # SentencePiece space (▁→space conversion only fires when there's
        # prior context). Detokenize `[last_prompt_token, *gen_so_far]` and
        # slice off whatever was already emitted so the first generated
        # token carries its leading space — otherwise " Paris" arrives as
        # "Paris" and the panel renders "isParis". O(1) extra cost per step.
        boundary_id = tokens[-1] if tokens else None
        boundary_text = self.detokenize([boundary_id]) if boundary_id is not None else ""
        prev_full = boundary_text

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
                next_id = _sample(
                    logits_arr,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    rng=rng,
                )

            generated_ids.append(next_id)
            if boundary_id is not None:
                full = self.detokenize([boundary_id, *generated_ids])
            else:
                full = self.detokenize(generated_ids)
            token_str = full[len(prev_full):]
            prev_full = full
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

        llm = self._engine()
        llm.reset()
        llm.eval(tokens)

        nll_sum = 0.0
        count = 0
        for i in range(len(tokens) - 1):
            logits_i = np.array(llm.eval_logits[i], dtype=np.float32)
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
    import gguf  # pyright: ignore[reportMissingImports]

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
