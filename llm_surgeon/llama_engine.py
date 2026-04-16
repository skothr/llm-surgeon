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
