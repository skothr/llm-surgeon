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
    kl = float(np.sum(pa * np.log(pa / np.clip(pb, 1e-10, None))))

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
        if self._llm is not None:
            del self._llm
            self._llm = None
            log.info("LlamaEngine closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    def tokenize(self, text: str, add_bos: bool = True) -> list[int]:
        return self._llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def detokenize(self, tokens: list[int]) -> str:
        return self._llm.detokenize(tokens).decode("utf-8", errors="replace")

    def logits(self, tokens: list[int]) -> np.ndarray:
        """Full vocab logits for the last token position. Shape: (n_vocab,)"""
        self._llm.reset()
        self._llm.eval(tokens)
        return np.array(self._llm.eval_logits[-1], dtype=np.float32)

    def logits_all(self, tokens: list[int]) -> list[np.ndarray]:
        """Full vocab logits for every position. List of (n_vocab,) arrays."""
        self._llm.reset()
        self._llm.eval(tokens)
        return [np.array(row, dtype=np.float32) for row in self._llm.eval_logits]

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
        self._llm.reset()
        self._llm.eval(tokens)

        generated_ids: list[int] = []
        generated_text = ""

        for _ in range(max_tokens):
            logits_arr = np.array(self._llm.eval_logits[-1], dtype=np.float32)

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
                probs /= probs.sum()
                if top_p < 1.0:
                    sorted_idx = np.argsort(-probs)
                    cum = np.cumsum(probs[sorted_idx])
                    cutoff = np.searchsorted(cum, top_p) + 1
                    mask = np.ones_like(probs, dtype=bool)
                    mask[sorted_idx[:cutoff]] = False
                    probs[mask] = 0.0
                    probs /= probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))

            token_str = self.detokenize([next_id])
            generated_ids.append(next_id)
            generated_text += token_str

            yield GenerateStep(
                token_id=next_id,
                token_str=token_str,
                logits=logits_arr if emit_logits else None,
            )

            if next_id == self._llm.token_eos():
                break
            if stop_sequences and any(s in generated_text for s in stop_sequences):
                break

            self._llm.eval([next_id])

    def perplexity(self, text: str) -> float:
        """Compute perplexity: exp(-1/N * sum(log P(token_i | context)))."""
        tokens = self.tokenize(text, add_bos=True)
        if len(tokens) < 2:
            return float("inf")

        self._llm.reset()
        self._llm.eval(tokens)

        nll_sum = 0.0
        count = 0
        for i in range(len(tokens) - 1):
            logits_i = np.array(self._llm.eval_logits[i], dtype=np.float32)
            log_probs = logits_i - np.logaddexp.reduce(logits_i)
            nll_sum -= log_probs[tokens[i + 1]]
            count += 1

        return float(np.exp(nll_sum / count))
