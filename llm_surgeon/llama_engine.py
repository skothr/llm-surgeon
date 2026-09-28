"""llama.cpp engine: native GGUF inference via llama-cpp-python.

Provides LlamaEngine for fast generation/logits/perplexity on quantized
GGUF models. The HF-to-GGUF export path lives in ``gguf_writer``;
``export_hf_to_gguf`` is re-exported here for backward compatibility.
"""

import codecs
import logging
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from llama_cpp import Llama

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


def _scores(llm: "Llama") -> np.ndarray:
    """Logit rows for every evaluated position: a view of ``llm.scores``.

    ``Llama.eval_logits`` rebuilds every row as Python lists on each access
    (O(n_tokens * n_vocab)), so reading it once per position or per generated
    token costs O(n^2 * n_vocab). ``scores`` is the float32 ndarray it is built
    from. The view aliases a buffer the next ``eval`` overwrites; copy before
    returning rows to callers.
    """
    return llm.scores[: llm.n_tokens]


def _decode_utf8(data: bytes, *, final: bool) -> str:
    """Decode ``data`` as UTF-8, holding back an incomplete trailing character.

    With ``final=False`` a multi-byte character whose bytes are not all
    present yet (LLaMA byte-fallback tokens split emoji and rare CJK across
    tokens) is left out instead of becoming U+FFFD; it appears once the
    token that completes it arrives. ``final=True`` flushes it as U+FFFD.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    return decoder.decode(data, final=final)


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

    def _eval_prompt(self, tokens: list[int]) -> "Llama":
        """Reset the context and evaluate ``tokens``, validating them first.

        Raises ValueError for an empty list or one longer than n_ctx, which
        llama.cpp would otherwise reject with an opaque decode error.
        """
        llm = self._engine()
        if not tokens:
            raise ValueError("tokens is empty; pass at least one token (e.g. BOS)")
        n_ctx = llm.n_ctx()
        if len(tokens) > n_ctx:
            raise ValueError(
                f"{len(tokens)} tokens exceed the context window (n_ctx={n_ctx}); "
                f"shorten the input or construct LlamaEngine with a larger n_ctx"
            )
        llm.reset()
        llm.eval(tokens)
        return llm

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        return self._engine().tokenize(text.encode("utf-8"), add_bos=add_bos)

    def detokenize(self, tokens: list[int]) -> str:
        return self._engine().detokenize(tokens).decode("utf-8", errors="replace")

    def logits(self, tokens: list[int]) -> np.ndarray:
        """Full vocab logits for the last token position. Shape: (n_vocab,)"""
        llm = self._eval_prompt(tokens)
        return np.array(_scores(llm)[-1], dtype=np.float32)

    def logits_all(self, tokens: list[int]) -> list[np.ndarray]:
        """Full vocab logits for every position. List of (n_vocab,) arrays."""
        llm = self._eval_prompt(tokens)
        return list(np.array(_scores(llm), dtype=np.float32))

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

        Generation stops at EOS, at a stop sequence, after ``max_tokens``
        steps, or when the context window (n_ctx) is full. Raises ValueError
        when ``tokens`` is empty or longer than n_ctx.
        """
        llm = self._eval_prompt(tokens)
        n_ctx = llm.n_ctx()

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        generated_ids: list[int] = []
        generated_text = ""
        # llama.cpp's detokenize on a single-token list strips the leading
        # SentencePiece space (▁→space conversion only fires when there's
        # prior context). Detokenize `[last_prompt_token, *gen_so_far]` and
        # slice off whatever was already emitted so the first generated
        # token carries its leading space — otherwise " Paris" arrives as
        # "Paris" and the panel renders "isParis". Each step re-detokenizes
        # the whole generation, O(len(generated)) per step.
        boundary = [tokens[-1]]
        prev_full = _decode_utf8(llm.detokenize(boundary), final=False)
        eos_id = llm.token_eos()

        for step in range(max_tokens):
            logits_arr = np.array(_scores(llm)[-1], dtype=np.float32)

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
            # Flush a dangling partial character only when no later token
            # can complete it.
            ctx_full = llm.n_tokens >= n_ctx  # no room to evaluate next_id
            last = step == max_tokens - 1 or next_id == eos_id or ctx_full
            full = _decode_utf8(llm.detokenize([*boundary, *generated_ids]), final=last)
            token_str = full[len(prev_full):]
            prev_full = full
            generated_text += token_str

            yield GenerateStep(
                token_id=next_id,
                token_str=token_str,
                logits=logits_arr if emit_logits else None,
            )

            if next_id == eos_id or ctx_full:
                break
            if stop_sequences and any(s in generated_text for s in stop_sequences):
                break

            llm.eval([next_id])

    def perplexity(self, text: str) -> float:
        """Compute perplexity: exp(-1/N * sum(log P(token_i | context)))."""
        tokens = self.tokenize(text, add_bos=True)
        if len(tokens) < 2:
            return float("inf")

        llm = self._eval_prompt(tokens)

        # Row i predicts token i + 1; the last row predicts past the text.
        rows = _scores(llm)[: len(tokens) - 1]
        log_z = np.logaddexp.reduce(rows, axis=-1).astype(np.float64)
        targets = np.asarray(tokens[1:])
        target_logits = rows[np.arange(len(targets)), targets].astype(np.float64)
        return float(np.exp(np.mean(log_z - target_logits)))


from llm_surgeon.gguf_writer import export_hf_to_gguf  # noqa: F401  (re-export)
