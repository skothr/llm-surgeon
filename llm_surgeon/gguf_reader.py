"""GGUF file reader: parse, dequantize, and load into HuggingFace models.

Supports loading Ollama/llama.cpp GGUF models as standard
LlamaForCausalLM instances compatible with the llm_surgeon API.
"""

import json
import logging
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("llm_surgeon.gguf_reader")

# ── GGML type constants ──────────────────────────────────────────────

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 26

GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    26: "BF16",
}

# (values_per_block, bytes_per_block)
GGML_BLOCK_SIZE = {
    GGML_TYPE_F32:  (1, 4),
    GGML_TYPE_F16:  (1, 2),
    GGML_TYPE_BF16: (1, 2),
    GGML_TYPE_Q4_0: (32, 18),
    GGML_TYPE_Q4_1: (32, 20),
    GGML_TYPE_Q5_0: (32, 22),
    GGML_TYPE_Q5_1: (32, 24),
    GGML_TYPE_Q8_0: (32, 34),
    GGML_TYPE_Q2_K: (256, 84),
    GGML_TYPE_Q3_K: (256, 110),
    GGML_TYPE_Q4_K: (256, 144),
    GGML_TYPE_Q5_K: (256, 176),
    GGML_TYPE_Q6_K: (256, 210),
}


# ── Dequantization ───────────────────────────────────────────────────

def _dequant_f32(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)[:n].copy()


def _dequant_f16(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16)[:n].astype(np.float32)


def _dequant_bf16(data: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint16)[:n]
    # BF16 → F32: shift left 16 bits
    f32_bits = raw.astype(np.uint32) << 16
    return f32_bits.view(np.float32).copy()


def _dequant_q4_0(data: bytes, n: int) -> np.ndarray:
    """Q4_0: 32 values per 18-byte block (f16 scale + 16 nibble bytes)."""
    nb = n // 32
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 18).reshape(nb, 18)
    scales = raw[:, :2].view(np.float16).astype(np.float32)  # (nb, 1)
    qs = raw[:, 2:]  # (nb, 16)
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    values = np.concatenate([lo, hi], axis=1)  # (nb, 32)
    return (values * scales).ravel()


def _dequant_q4_1(data: bytes, n: int) -> np.ndarray:
    """Q4_1: 32 values per 20-byte block (f16 scale + f16 min + 16 bytes)."""
    nb = n // 32
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 20).reshape(nb, 20)
    scales = raw[:, :2].view(np.float16).astype(np.float32)
    mins = raw[:, 2:4].view(np.float16).astype(np.float32)
    qs = raw[:, 4:]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    values = np.concatenate([lo, hi], axis=1)
    return (values * scales + mins).ravel()


def _dequant_q5_0(data: bytes, n: int) -> np.ndarray:
    """Q5_0: 32 values per 22-byte block (f16 scale + 4 high-bit bytes + 16 nibble bytes)."""
    nb = n // 32
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 22).reshape(nb, 22)
    scales = raw[:, :2].view(np.float16).astype(np.float32)
    qh = raw[:, 2:6]  # 4 bytes = 32 high bits
    qs = raw[:, 6:]    # 16 bytes = 32 nibbles
    # Unpack high bits: bit j of uint32
    qh32 = qh.view(np.uint32).ravel()  # (nb,)
    hi_bits = np.zeros((nb, 32), dtype=np.float32)
    for bit in range(32):
        hi_bits[:, bit] = ((qh32 >> bit) & 1).astype(np.float32) * 16.0
    lo = (qs & 0x0F).astype(np.float32)
    up = (qs >> 4).astype(np.float32)
    values = np.empty((nb, 32), dtype=np.float32)
    values[:, :16] = lo + hi_bits[:, :16] - 16.0
    values[:, 16:] = up + hi_bits[:, 16:] - 16.0
    return (values * scales).ravel()


def _dequant_q5_1(data: bytes, n: int) -> np.ndarray:
    """Q5_1: 32 values per 24-byte block (f16 scale + f16 min + 4 high-bit bytes + 16 bytes)."""
    nb = n // 32
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 24).reshape(nb, 24)
    scales = raw[:, :2].view(np.float16).astype(np.float32)
    mins = raw[:, 2:4].view(np.float16).astype(np.float32)
    qh = raw[:, 4:8]
    qs = raw[:, 8:]
    qh32 = qh.view(np.uint32).ravel()
    hi_bits = np.zeros((nb, 32), dtype=np.float32)
    for bit in range(32):
        hi_bits[:, bit] = ((qh32 >> bit) & 1).astype(np.float32) * 16.0
    lo = (qs & 0x0F).astype(np.float32)
    up = (qs >> 4).astype(np.float32)
    values = np.empty((nb, 32), dtype=np.float32)
    values[:, :16] = lo + hi_bits[:, :16]
    values[:, 16:] = up + hi_bits[:, 16:]
    return (values * scales + mins).ravel()


def _dequant_q8_0(data: bytes, n: int) -> np.ndarray:
    """Q8_0: 32 values per 34-byte block (f16 scale + 32 int8 values)."""
    nb = n // 32
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 34).reshape(nb, 34)
    scales = raw[:, :2].view(np.float16).astype(np.float32)
    qs = raw[:, 2:].view(np.int8).astype(np.float32)
    return (qs * scales).ravel()


def _unpack_k4_scales(sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack 12-byte K-quant scale/min block into 8 scales + 8 mins.

    Used by Q4_K and Q5_K. sc shape: (nb, 12).
    Returns (scales, mins) each (nb, 8).
    """
    nb = sc.shape[0]
    scales = np.zeros((nb, 8), dtype=np.float32)
    mins = np.zeros((nb, 8), dtype=np.float32)
    scales[:, :4] = (sc[:, :4] & 63).astype(np.float32)
    mins[:, :4] = (sc[:, 4:8] & 63).astype(np.float32)
    for i in range(4):
        scales[:, 4 + i] = (
            (sc[:, 8 + i] & 0x0F) | ((sc[:, i] >> 6) << 4)
        ).astype(np.float32)
        mins[:, 4 + i] = (
            (sc[:, 8 + i] >> 4) | ((sc[:, 4 + i] >> 6) << 4)
        ).astype(np.float32)
    return scales, mins


def _dequant_q4_k(data: bytes, n: int) -> np.ndarray:
    """Q4_K: 256 values per 144-byte block.

    Layout: f16 d, f16 dmin, uint8[12] scales, uint8[128] qs.
    8 sub-blocks of 32 values; qs packed as nibble pairs per 64-value chunk.
    """
    nb = n // 256
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 144).reshape(nb, 144)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = raw[:, :2].view(np.float16).astype(np.float32).reshape(nb, 1)
        dmin = raw[:, 2:4].view(np.float16).astype(np.float32).reshape(nb, 1)
        sc_bytes = raw[:, 4:16]
        qs = raw[:, 16:]  # (nb, 128)

        scales, mins = _unpack_k4_scales(sc_bytes)
        result = np.zeros((nb, 256), dtype=np.float32)

        for j in range(4):
            q32 = qs[:, j * 32:(j + 1) * 32]
            lo = (q32 & 0x0F).astype(np.float32)
            hi = (q32 >> 4).astype(np.float32)
            sc_lo = scales[:, 2 * j:2 * j + 1]
            m_lo = mins[:, 2 * j:2 * j + 1]
            sc_hi = scales[:, 2 * j + 1:2 * j + 2]
            m_hi = mins[:, 2 * j + 1:2 * j + 2]
            result[:, j * 64:j * 64 + 32] = d * sc_lo * lo - dmin * m_lo
            result[:, j * 64 + 32:j * 64 + 64] = d * sc_hi * hi - dmin * m_hi

    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result.ravel()


def _dequant_q5_k(data: bytes, n: int) -> np.ndarray:
    """Q5_K: 256 values per 176-byte block.

    Layout: f16 d, f16 dmin, uint8[12] scales, uint8[32] qh, uint8[128] qs.
    Like Q4_K but with an extra high bit per value from qh.
    """
    nb = n // 256
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 176).reshape(nb, 176)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = raw[:, :2].view(np.float16).astype(np.float32).reshape(nb, 1)
        dmin = raw[:, 2:4].view(np.float16).astype(np.float32).reshape(nb, 1)
        sc_bytes = raw[:, 4:16]
        qh = raw[:, 16:48]   # (nb, 32)
        qs = raw[:, 48:]      # (nb, 128)

        scales, mins = _unpack_k4_scales(sc_bytes)
        result = np.zeros((nb, 256), dtype=np.float32)

        for j in range(4):
            q32 = qs[:, j * 32:(j + 1) * 32]
            lo = (q32 & 0x0F).astype(np.float32)
            hi = (q32 >> 4).astype(np.float32)
            hb_lo = ((qh >> (2 * j)) & 1).astype(np.float32) * 16.0
            hb_hi = ((qh >> (2 * j + 1)) & 1).astype(np.float32) * 16.0
            lo = lo + hb_lo[:, :32]
            hi = hi + hb_hi[:, :32]
            sc_lo = scales[:, 2 * j:2 * j + 1]
            m_lo = mins[:, 2 * j:2 * j + 1]
            sc_hi = scales[:, 2 * j + 1:2 * j + 2]
            m_hi = mins[:, 2 * j + 1:2 * j + 2]
            result[:, j * 64:j * 64 + 32] = d * sc_lo * lo - dmin * m_lo
            result[:, j * 64 + 32:j * 64 + 64] = d * sc_hi * hi - dmin * m_hi

    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result.ravel()


def _dequant_q6_k(data: bytes, n: int) -> np.ndarray:
    """Q6_K: 256 values per 210-byte block.

    Layout: uint8[128] ql, uint8[64] qh, int8[16] scales, f16 d.
    6 bits per value: 4 from ql + 2 from qh.
    """
    nb = n // 256
    raw = np.frombuffer(data, dtype=np.uint8, count=nb * 210).reshape(nb, 210)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ql = raw[:, :128]       # (nb, 128)
        qh = raw[:, 128:192]    # (nb, 64)
        sc = raw[:, 192:208].view(np.int8).astype(np.float32)  # (nb, 16)
        d = raw[:, 208:210].view(np.float16).astype(np.float32).reshape(nb, 1)

        result = np.zeros((nb, 256), dtype=np.float32)

        for chunk in range(2):
            ql_c = ql[:, chunk * 64:(chunk + 1) * 64]   # (nb, 64)
            qh_c = qh[:, chunk * 32:(chunk + 1) * 32]   # (nb, 32)
            sc_c = sc[:, chunk * 8:(chunk + 1) * 8]      # (nb, 8)
            base = chunk * 128

            ql_lo_a = ql_c[:, :32]
            ql_lo_b = ql_c[:, 32:64]

            q1 = ((ql_lo_a & 0x0F) | (((qh_c >> 0) & 3) << 4)).astype(np.float32) - 32.0
            q2 = ((ql_lo_b & 0x0F) | (((qh_c >> 2) & 3) << 4)).astype(np.float32) - 32.0
            q3 = ((ql_lo_a >> 4) | (((qh_c >> 4) & 3) << 4)).astype(np.float32) - 32.0
            q4 = ((ql_lo_b >> 4) | (((qh_c >> 6) & 3) << 4)).astype(np.float32) - 32.0

            for half in range(2):
                lo = half * 16
                hi = lo + 16
                si = half
                result[:, base + lo:base + hi] = d * sc_c[:, si:si + 1] * q1[:, lo:hi]
                result[:, base + 32 + lo:base + 32 + hi] = d * sc_c[:, si + 2:si + 3] * q2[:, lo:hi]
                result[:, base + 64 + lo:base + 64 + hi] = d * sc_c[:, si + 4:si + 5] * q3[:, lo:hi]
                result[:, base + 96 + lo:base + 96 + hi] = d * sc_c[:, si + 6:si + 7] * q4[:, lo:hi]

    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result.ravel()


_DEQUANT = {
    GGML_TYPE_F32:  _dequant_f32,
    GGML_TYPE_F16:  _dequant_f16,
    GGML_TYPE_BF16: _dequant_bf16,
    GGML_TYPE_Q4_0: _dequant_q4_0,
    GGML_TYPE_Q4_1: _dequant_q4_1,
    GGML_TYPE_Q5_0: _dequant_q5_0,
    GGML_TYPE_Q5_1: _dequant_q5_1,
    GGML_TYPE_Q8_0: _dequant_q8_0,
    GGML_TYPE_Q4_K: _dequant_q4_k,
    GGML_TYPE_Q5_K: _dequant_q5_k,
    GGML_TYPE_Q6_K: _dequant_q6_k,
}


# ── GGUF File Parser ─────────────────────────────────────────────────

@dataclass
class TensorInfo:
    name: str
    shape: tuple
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAME.get(self.ggml_type, f"?{self.ggml_type}")

    @property
    def n_elements(self) -> int:
        r = 1
        for d in self.shape:
            r *= d
        return r


class GGUFFile:
    """Read-only interface to a GGUF model file.

    Usage::

        with GGUFFile(path) as g:
            print(g.architecture, len(g.tensor_infos))
            arr = g.read_tensor_numpy("blk.0.attn_q.weight")
            t   = g.read_tensor("blk.0.attn_q.weight")
    """

    def __init__(self, path):
        self.path = Path(path)
        self.version: int = 0
        self.metadata: dict = {}
        self.tensor_infos: list[TensorInfo] = []
        self._tensor_map: dict[str, TensorInfo] = {}
        self._data_offset: int = 0
        self._file: Optional[IO[bytes]] = None
        self._parse()

    # ── parsing internals ────────────────────────────────────────

    def _parse(self):
        self._file = open(self.path, "rb")
        f = self._file

        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (magic={magic!r}): {self.path}")

        self.version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            key = self._read_string()
            vtype = struct.unpack("<I", f.read(4))[0]
            self.metadata[key] = self._read_value(vtype)

        for _ in range(n_tensors):
            name = self._read_string()
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            info = TensorInfo(name=name, shape=dims, ggml_type=dtype, offset=offset)
            self.tensor_infos.append(info)
            self._tensor_map[name] = info

        header_end = f.tell()
        # GGUF writers may specify general.alignment (default 32); data and
        # per-tensor offsets are padded to this value. Honor it instead of
        # assuming 32, else non-default alignments shift all tensor reads.
        align = int(self.metadata.get("general.alignment", 32))
        self._data_offset = ((header_end + align - 1) // align) * align

    def _read_string(self) -> str:
        f = self._file
        assert f is not None
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, vtype: int):
        f = self._file
        assert f is not None
        readers = {
            0: lambda: struct.unpack("<B", f.read(1))[0],
            1: lambda: struct.unpack("<b", f.read(1))[0],
            2: lambda: struct.unpack("<H", f.read(2))[0],
            3: lambda: struct.unpack("<h", f.read(2))[0],
            4: lambda: struct.unpack("<I", f.read(4))[0],
            5: lambda: struct.unpack("<i", f.read(4))[0],
            6: lambda: struct.unpack("<f", f.read(4))[0],
            7: lambda: bool(struct.unpack("<B", f.read(1))[0]),
            8: lambda: self._read_string(),
            10: lambda: struct.unpack("<Q", f.read(8))[0],
            11: lambda: struct.unpack("<q", f.read(8))[0],
            12: lambda: struct.unpack("<d", f.read(8))[0],
        }
        if vtype == 9:  # array
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            return [self._read_value(arr_type) for _ in range(arr_len)]
        if vtype in readers:
            return readers[vtype]()
        raise ValueError(f"Unknown GGUF value type: {vtype}")

    # ── public API ───────────────────────────────────────────────

    @property
    def architecture(self) -> str:
        return self.metadata.get("general.architecture", "unknown")

    def read_tensor_numpy(self, name: str) -> np.ndarray:
        """Read and dequantize a tensor, returning a float32 numpy array."""
        info = self._tensor_map[name]
        n = info.n_elements
        block_size, bytes_per_block = GGML_BLOCK_SIZE.get(info.ggml_type, (0, 0))
        if block_size == 0:
            raise ValueError(f"Unknown block size for type {info.type_name}")
        fn = _DEQUANT.get(info.ggml_type)
        if fn is None:
            raise ValueError(
                f"No dequantizer for {info.type_name} "
                f"(tensor '{name}'). Supported: {sorted(GGML_TYPE_NAME[k] for k in _DEQUANT)}"
            )
        n_bytes = (n // block_size) * bytes_per_block
        f = self._file
        assert f is not None
        f.seek(self._data_offset + info.offset)
        raw = f.read(n_bytes)
        flat = fn(raw, n)
        # GGUF dims are (ne[0], ne[1], ...) where ne[0] is innermost;
        # numpy/PyTorch convention is reversed
        return flat.reshape(info.shape[::-1])

    def read_tensor(self, name: str, dtype=torch.float16) -> torch.Tensor:
        """Read and dequantize a tensor, returning a PyTorch tensor."""
        arr = self.read_tensor_numpy(name)
        t = torch.from_numpy(arr.copy())
        if dtype in (torch.float16, torch.bfloat16):
            finfo = torch.finfo(dtype)
            t = t.clamp(finfo.min, finfo.max)
        return t.to(dtype)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


# ── GGUF ↔ HuggingFace mapping ──────────────────────────────────────

# LLaMA-family name mapping (covers LLaMA 1/2/3, Mistral, etc.)
_GGUF_GLOBAL = {
    "token_embd.weight":  "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight":      "lm_head.weight",
}

_GGUF_LAYER = {
    "attn_norm.weight":   "input_layernorm.weight",
    "ffn_norm.weight":    "post_attention_layernorm.weight",
    "attn_q.weight":      "self_attn.q_proj.weight",
    "attn_k.weight":      "self_attn.k_proj.weight",
    "attn_v.weight":      "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight":    "mlp.gate_proj.weight",
    "ffn_up.weight":      "mlp.up_proj.weight",
    "ffn_down.weight":    "mlp.down_proj.weight",
}


def _map_tensor_name(gguf_name: str) -> Optional[str]:
    """Map a GGUF tensor name to a HuggingFace state_dict key."""
    if gguf_name in _GGUF_GLOBAL:
        return _GGUF_GLOBAL[gguf_name]
    if gguf_name.startswith("blk."):
        parts = gguf_name.split(".", 2)  # ["blk", "0", "attn_q.weight"]
        if len(parts) == 3:
            layer_idx = parts[1]
            suffix = parts[2]
            hf_suffix = _GGUF_LAYER.get(suffix)
            if hf_suffix:
                return f"model.layers.{layer_idx}.{hf_suffix}"
    return None


def _build_config(meta: dict):
    """Build a HuggingFace LlamaConfig from GGUF metadata."""
    from transformers import LlamaConfig

    arch = meta.get("general.architecture", "llama")
    prefix = arch + "."

    vocab_size = len(meta.get("tokenizer.ggml.tokens", []))
    hidden = meta.get(prefix + "embedding_length", 4096)
    n_layers = meta.get(prefix + "block_count", 32)
    n_heads = meta.get(prefix + "attention.head_count", 32)
    n_kv_heads = meta.get(prefix + "attention.head_count_kv", n_heads)
    ffn_size = meta.get(prefix + "feed_forward_length", hidden * 4)
    ctx_len = meta.get(prefix + "context_length", 4096)
    rms_eps = meta.get(prefix + "attention.layer_norm_rms_epsilon", 1e-5)
    rope_theta = meta.get(prefix + "rope.freq_base", 10000.0)

    return LlamaConfig(**{  # type: ignore
        "vocab_size": vocab_size,
        "hidden_size": hidden,
        "intermediate_size": ffn_size,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "max_position_embeddings": ctx_len,
        "rms_norm_eps": rms_eps,
        "rope_theta": rope_theta,
        "tie_word_embeddings": False,
    })


def _build_tokenizer(meta: dict):
    """Build a HuggingFace tokenizer from GGUF metadata.

    Returns a PreTrainedTokenizerFast, or None if vocab data is missing.
    """
    tokens = meta.get("tokenizer.ggml.tokens")
    if not tokens:
        return None

    merges = meta.get("tokenizer.ggml.merges", [])
    scores = meta.get("tokenizer.ggml.scores", [])
    bos_id = meta.get("tokenizer.ggml.bos_token_id", 1)
    eos_id = meta.get("tokenizer.ggml.eos_token_id", 2)
    model_type = meta.get("tokenizer.ggml.model", "llama")

    bos = tokens[bos_id] if bos_id < len(tokens) else "<s>"
    eos = tokens[eos_id] if eos_id < len(tokens) else "</s>"
    unk = "<unk>"

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE, Unigram
        from tokenizers.pre_tokenizers import Metaspace
        from tokenizers.decoders import Metaspace as MetaspaceDecoder
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        log.warning("tokenizers library not available; returning None for tokenizer")
        return None

    vocab = {tok: i for i, tok in enumerate(tokens)}

    if merges:
        merge_pairs = []
        for m in merges:
            pair = m.split(" ", 1)
            if len(pair) == 2:
                merge_pairs.append(tuple(pair))
        tok = Tokenizer(BPE(vocab=vocab, merges=merge_pairs, unk_token=unk))
    elif scores:
        tok = Tokenizer(Unigram([(t, s) for t, s in zip(tokens, scores)]))  # type: ignore
    else:
        log.warning("No merges or scores in GGUF metadata; returning None for tokenizer")
        return None

    if model_type == "llama":
        tok.pre_tokenizer = Metaspace(replacement="\u2581", prepend_scheme="first")  # type: ignore
        tok.decoder = MetaspaceDecoder(replacement="\u2581", prepend_scheme="first")  # type: ignore

    chat_template = meta.get("tokenizer.chat_template")

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=bos,
        eos_token=eos,
        unk_token=unk,
    )
    if chat_template:
        hf_tok.chat_template = chat_template

    return hf_tok


def _reverse_permute(t: torch.Tensor, n_head: int, n_kv_heads: int) -> torch.Tensor:
    """Reverse the Q/K head interleaving that convert_hf_to_gguf.py applies.

    llama.cpp permutes Q and K weights during HF→GGUF conversion to
    interleave the first and second halves of each head's dimensions.
    This undoes that permutation so the weights match HF's layout.
    """
    n = n_kv_heads if n_head != n_kv_heads else n_head
    dim = t.shape[0] // n // 2
    return t.reshape(n, dim, 2, *t.shape[1:]).swapaxes(1, 2).reshape(t.shape)


# ── Main loader ──────────────────────────────────────────────────────

def load_gguf_as_hf(
    gguf_path,
    dtype=torch.float16,
) -> Tuple:
    """Load a GGUF file and return a (model, tokenizer) tuple.

    The returned model is a standard LlamaForCausalLM instance with
    dequantized weights — fully compatible with all llm_surgeon operations.

    Args:
        gguf_path: Path to the GGUF file.
        dtype: Target dtype for model weights (default: float16).

    Returns:
        (model, tokenizer) tuple matching surgery.load_model() interface.
    """
    from transformers import LlamaForCausalLM

    gguf_path = Path(gguf_path)
    log.info("Loading GGUF: %s", gguf_path)

    with GGUFFile(gguf_path) as g:
        arch = g.architecture
        if arch not in ("llama", "mistral"):
            raise ValueError(
                f"Unsupported GGUF architecture: '{arch}'. "
                f"Currently supported: llama, mistral."
            )

        config = _build_config(g.metadata)
        log.info(
            "Config: %d layers, %d hidden, %d heads, %d vocab",
            config.num_hidden_layers, config.hidden_size,
            config.num_attention_heads, config.vocab_size,
        )

        # Build state dict from GGUF tensors
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        state_dict = {}
        skipped = []
        for info in g.tensor_infos:
            hf_name = _map_tensor_name(info.name)
            if hf_name is None:
                skipped.append(info.name)
                continue
            t = g.read_tensor(info.name, dtype=dtype)
            if ".attn_q." in info.name:
                t = _reverse_permute(t, n_heads, n_heads)
            elif ".attn_k." in info.name:
                t = _reverse_permute(t, n_heads, n_kv_heads)
            state_dict[hf_name] = t

        if skipped:
            log.debug("Skipped %d unmapped tensors: %s", len(skipped), skipped[:5])

        # Handle tied embeddings: if output.weight is absent, share embed_tokens
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            config.tie_word_embeddings = True
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
            log.info("Tied embeddings: lm_head shares embed_tokens weight")

        # Create model on meta device (no memory), then fill with real weights
        with torch.device("meta"):  # type: ignore
            model = LlamaForCausalLM(config)

        model.load_state_dict(state_dict, assign=True, strict=False)  # type: ignore

        # Buffers not in the GGUF state dict (e.g. RoPE inv_freq) remain as
        # meta tensors after assign=True + strict=False. Reinitialize all
        # RoPE modules so inv_freq is properly computed from config.
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        for name, module in model.named_modules():
            if isinstance(module, LlamaRotaryEmbedding):
                parent_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name[0]) if len(parent_name) > 1 else model
                attr = parent_name[1] if len(parent_name) > 1 else name
                setattr(parent, attr, LlamaRotaryEmbedding(config, device="cpu"))

        model.eval()
        model.requires_grad_(False)

        loaded = len(state_dict)
        expected = sum(1 for _ in model.state_dict())
        if loaded < expected:
            log.warning(
                "Loaded %d/%d state_dict keys from GGUF", loaded, expected
            )

        tokenizer = _build_tokenizer(g.metadata)
        if tokenizer is None:
            log.warning("Could not build tokenizer from GGUF metadata")

    log.info(
        "Loaded %s: %d params, dtype=%s",
        gguf_path.name,
        sum(p.numel() for p in model.parameters()),
        dtype,
    )
    return model, tokenizer


# ── Ollama resolution ────────────────────────────────────────────────

def resolve_ollama_blob(model_id: str, models_dir: Optional[str] = None) -> Optional[Path]:
    """Resolve an Ollama model ID (e.g. 'tinyllama:latest') to a GGUF blob path.

    Args:
        model_id: Ollama model name, optionally with tag (default: latest).
        models_dir: Override for Ollama models directory.

    Returns:
        Path to the GGUF blob, or None if not found.
    """
    import os

    parts = model_id.split(":", 1)
    name = parts[0]
    tag = parts[1] if len(parts) > 1 else "latest"

    base = Path(models_dir or os.environ.get("OLLAMA_MODELS", "/usr/share/ollama/.ollama/models"))
    manifest_path = base / "manifests" / "registry.ollama.ai" / "library" / name / tag

    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace("sha256:", "sha256-")
            blob_path = base / "blobs" / digest
            if blob_path.exists():
                return blob_path

    return None


_GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K",
}


def gguf_model_meta(path) -> dict:
    """Extract model metadata from a GGUF file without reading tensor data.

    Opens the file, parses only the header (metadata KV pairs + tensor index),
    then closes. Returns a dict with standardized architecture fields.
    """
    with GGUFFile(path) as g:
        md = g.metadata
        arch = md.get("general.architecture", "")

        file_type_id = md.get("general.file_type")
        quantization = _GGUF_FILE_TYPES.get(file_type_id) if file_type_id is not None else None

        type_counts: dict[str, int] = {}
        total_elements = 0
        total_bytes = 0
        for ti in g.tensor_infos:
            tname = GGML_TYPE_NAME.get(ti.ggml_type, f"?{ti.ggml_type}")
            n = ti.n_elements
            type_counts[tname] = type_counts.get(tname, 0) + n
            total_elements += n
            bs = GGML_BLOCK_SIZE.get(ti.ggml_type)
            if bs:
                vals_per_block, bytes_per_block = bs
                total_bytes += (n // vals_per_block) * bytes_per_block

        bpw = round(total_bytes * 8 / total_elements, 2) if total_elements else None

        return {
            "architecture": arch or None,
            "model_name": md.get("general.name"),
            "quantization": quantization,
            "num_layers": md.get(f"{arch}.block_count"),
            "hidden_size": md.get(f"{arch}.embedding_length"),
            "num_heads": md.get(f"{arch}.attention.head_count"),
            "num_kv_heads": md.get(f"{arch}.attention.head_count_kv"),
            "vocab_size": len(md.get("tokenizer.ggml.tokens", [])) or None,
            "intermediate_size": md.get(f"{arch}.feed_forward_length"),
            "max_position_embeddings": md.get(f"{arch}.context_length"),
            "rope_theta": md.get(f"{arch}.rope.freq_base"),
            "num_tensors": len(g.tensor_infos),
            "tensor_type_counts": type_counts if type_counts else None,
            "total_params": total_elements or None,
            "total_bytes": total_bytes or None,
            "bits_per_weight": bpw,
        }
