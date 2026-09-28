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
from typing import IO

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
GGML_TYPE_I32 = 26
GGML_TYPE_BF16 = 30

GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    26: "I32", 30: "BF16",
}

# (values_per_block, bytes_per_block)
GGML_BLOCK_SIZE = {
    GGML_TYPE_F32:  (1, 4),
    GGML_TYPE_F16:  (1, 2),
    GGML_TYPE_I32:  (1, 4),
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


def _dequant_i32(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int32)[:n].astype(np.float32)


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


def _unpack_k4_scales(sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    GGML_TYPE_I32:  _dequant_i32,
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


def _gguf_py_qtype(ggml_type: int):
    """Return gguf-py's ``GGMLQuantizationType`` for ``ggml_type``, or None.

    Types without a dequantizer in this module (Q2_K, Q3_K, the IQ family,
    ...) are decoded with the optional ``gguf`` package when it is installed.
    """
    try:
        import gguf
    except ImportError:
        return None
    try:
        return gguf.GGMLQuantizationType(ggml_type)
    except ValueError:
        return None


def _type_name(ggml_type: int) -> str:
    name = GGML_TYPE_NAME.get(ggml_type)
    if name is None:
        qtype = _gguf_py_qtype(ggml_type)
        name = qtype.name if qtype is not None else f"?{ggml_type}"
    return name


def _block_size(ggml_type: int) -> tuple[int, int] | None:
    """(values_per_block, bytes_per_block), from gguf-py for types not listed here."""
    bs = GGML_BLOCK_SIZE.get(ggml_type)
    if bs is None:
        qtype = _gguf_py_qtype(ggml_type)
        if qtype is not None:
            import gguf
            bs = gguf.GGML_QUANT_SIZES[qtype]
    return bs


# ── GGUF File Parser ─────────────────────────────────────────────────

# struct format of each fixed-size GGUF metadata value type (7 = bool)
_GGUF_SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}

# numpy dtype of each fixed-size GGUF value type, for bulk array reads
_GGUF_ARRAY_DTYPE = {
    0: "<u1", 1: "<i1", 2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4",
    6: "<f4", 7: "?", 10: "<u8", 11: "<i8", 12: "<f8",
}

@dataclass
class TensorInfo:
    name: str
    shape: tuple
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return _type_name(self.ggml_type)

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
        self._file: IO[bytes] | None = None
        self._parse()

    # ── parsing internals ────────────────────────────────────────

    def _parse(self):
        self._file = open(self.path, "rb")
        try:
            f = self._file

            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Not a GGUF file (magic={magic!r}): {self.path}")

            self.version = struct.unpack("<I", f.read(4))[0]
            if self.version not in (2, 3):
                # Version 1 used 32-bit lengths; a big-endian file reads as a
                # huge version number. Neither parses with the layout below.
                raise ValueError(
                    f"Unsupported GGUF version {self.version} in {self.path} "
                    "(supported: little-endian GGUF v2 and v3)"
                )
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
        except Exception:
            # On parse failure the caller never gets a reference to self, so
            # __del__-based cleanup on the partially-constructed object is
            # not guaranteed. Close the fd eagerly — otherwise a service
            # scanning many session files can run out of descriptors.
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            raise

    def _get_file(self) -> IO[bytes]:
        if self._file is None:
            raise RuntimeError(f"GGUFFile is closed: {self.path}")
        return self._file

    def _read_string(self) -> str:
        f = self._get_file()
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, vtype: int):
        f = self._get_file()
        if vtype == 8:
            return self._read_string()
        if vtype == 9:  # array
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            dtype = _GGUF_ARRAY_DTYPE.get(arr_type)
            if dtype is not None:
                # Fixed-size elements: one read instead of one per element
                # (token_type / scores arrays run to 100k+ entries).
                dt = np.dtype(dtype)
                data = f.read(arr_len * dt.itemsize)
                return np.frombuffer(data, dtype=dt, count=arr_len).tolist()
            return [self._read_value(arr_type) for _ in range(arr_len)]
        fmt = _GGUF_SCALAR_FMT.get(vtype)
        if fmt is None:
            raise ValueError(f"Unknown GGUF value type: {vtype}")
        value = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        return bool(value) if vtype == 7 else value

    # ── public API ───────────────────────────────────────────────

    @property
    def architecture(self) -> str:
        return self.metadata.get("general.architecture", "unknown")

    def read_tensor_numpy(self, name: str) -> np.ndarray:
        """Read and dequantize a tensor, returning a float32 numpy array."""
        info = self._tensor_map[name]
        n = info.n_elements
        bs = _block_size(info.ggml_type)
        if bs is None:
            raise ValueError(f"Unknown block size for type {info.type_name} (tensor '{name}')")
        block_size, bytes_per_block = bs
        if n % block_size:
            raise ValueError(
                f"Tensor '{name}' has {n} elements, not a multiple of the "
                f"{info.type_name} block size {block_size}"
            )
        fn = _DEQUANT.get(info.ggml_type)
        qtype = None
        if fn is None:
            qtype = _gguf_py_qtype(info.ggml_type)
            if qtype is None:
                raise ValueError(
                    f"No dequantizer for {info.type_name} "
                    f"(tensor '{name}'). Supported: {sorted(GGML_TYPE_NAME[k] for k in _DEQUANT)}; "
                    "install the `gguf` package for the other ggml types."
                )
        n_bytes = (n // block_size) * bytes_per_block
        f = self._get_file()
        f.seek(self._data_offset + info.offset)
        raw = f.read(n_bytes)
        if len(raw) != n_bytes:
            raise ValueError(
                f"Truncated data for tensor '{name}': expected {n_bytes} bytes, "
                f"read {len(raw)} (file {self.path})"
            )
        if fn is not None:
            flat = fn(raw, n)
        else:
            from gguf import quants
            assert qtype is not None
            flat = quants.dequantize(np.frombuffer(raw, dtype=np.uint8), qtype)
            flat = flat.astype(np.float32, copy=False).ravel()
        # GGUF dims are (ne[0], ne[1], ...) where ne[0] is innermost;
        # numpy/PyTorch convention is reversed
        return flat.reshape(info.shape[::-1])

    def read_tensor(self, name: str, dtype=torch.float16) -> torch.Tensor:
        """Read and dequantize a tensor, returning a PyTorch tensor."""
        arr = self.read_tensor_numpy(name)
        if not arr.flags.writeable:
            arr = arr.copy()
        t = torch.from_numpy(arr)
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


def _map_tensor_name(gguf_name: str) -> str | None:
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


def _rope_scaling_from_meta(meta: dict, prefix: str) -> dict | None:
    """Map GGUF RoPE scaling metadata to an HF ``rope_scaling`` dict.

    Llama-3.1-style scaling is not metadata but a ``rope_freqs.weight``
    tensor; ``load_gguf_as_hf`` applies that one to the rotary module.
    """
    kind = meta.get(prefix + "rope.scaling.type")
    factor = meta.get(prefix + "rope.scaling.factor")
    if kind is None and prefix + "rope.scale_linear" in meta:
        # Pre-2024 GGUFs stored linear scaling under this key.
        kind, factor = "linear", meta[prefix + "rope.scale_linear"]
    if kind is None or kind == "none":
        return None
    if kind == "linear" and factor:
        return None if float(factor) == 1.0 else {"rope_type": "linear", "factor": float(factor)}
    raise ValueError(
        f"Unsupported RoPE scaling type '{kind}' (factor={factor}) in GGUF metadata; "
        "only linear scaling and llama3-style rope_freqs are reproduced."
    )


def _build_config(meta: dict, tensor_infos: "list[TensorInfo] | None" = None):
    """Build a HuggingFace LlamaConfig from GGUF metadata.

    Raises ValueError when a required hyperparameter is missing, rather than
    substituting LLaMA-7B defaults that would not match the tensors.
    ``tensor_infos`` supplies the vocab size from ``token_embd.weight``,
    which is authoritative over the length of the embedded token list.
    """
    from transformers import LlamaConfig

    arch = meta.get("general.architecture", "llama")
    prefix = arch + "."

    def required(key: str):
        if prefix + key not in meta:
            raise ValueError(f"GGUF metadata is missing required key '{prefix + key}'")
        return meta[prefix + key]

    hidden = required("embedding_length")
    n_layers = required("block_count")
    n_heads = required("attention.head_count")
    ffn_size = required("feed_forward_length")
    n_kv_heads = meta.get(prefix + "attention.head_count_kv", n_heads)
    ctx_len = meta.get(prefix + "context_length", 4096)
    rms_eps = meta.get(prefix + "attention.layer_norm_rms_epsilon", 1e-5)
    rope_theta = meta.get(prefix + "rope.freq_base", 10000.0)
    head_dim = meta.get(prefix + "attention.key_length", hidden // n_heads)
    value_dim = meta.get(prefix + "attention.value_length", head_dim)
    rope_dim = meta.get(prefix + "rope.dimension_count", head_dim)
    if value_dim != head_dim:
        raise ValueError(
            f"attention.key_length={head_dim} != attention.value_length={value_dim}; "
            "LlamaForCausalLM uses one head_dim for both"
        )
    if rope_dim != head_dim:
        raise ValueError(
            f"Partial rotary embedding (rope.dimension_count={rope_dim}, "
            f"head_dim={head_dim}) is not supported"
        )

    embd = next((ti for ti in tensor_infos or [] if ti.name == "token_embd.weight"), None)
    if embd is not None and len(embd.shape) == 2:
        vocab_size = embd.shape[1]  # GGUF dims are (ne0=hidden, ne1=vocab)
    else:
        vocab_size = len(meta.get("tokenizer.ggml.tokens", []))
    if not vocab_size:
        raise ValueError(
            "Cannot determine vocab size: no token_embd.weight tensor and no "
            "tokenizer.ggml.tokens metadata"
        )

    kwargs = {
        "vocab_size": vocab_size,
        "hidden_size": hidden,
        "intermediate_size": ffn_size,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": ctx_len,
        "rms_norm_eps": rms_eps,
        "rope_theta": rope_theta,
        "tie_word_embeddings": False,
    }
    rope_scaling = _rope_scaling_from_meta(meta, prefix)
    if rope_scaling is not None:
        kwargs["rope_scaling"] = rope_scaling

    # LlamaConfig forwards arbitrary kwargs through PretrainedConfig.__init__,
    # so the stub can't enumerate every accepted parameter. Splat-form keeps
    # the single rule-scoped ignore on one line instead of fanning out across
    # every kwarg.
    return LlamaConfig(**kwargs)


# Llama-3 pre-tokenizer split pattern: llama.cpp's LLAMA3 pre-type, which
# GGUF files name "llama-bpe", and the Split step of Meta-Llama-3's
# tokenizer.json. gguf_writer matches on it to write that name.
LLAMA3_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# GGUF tokenizer.ggml.token_type codes (llama.cpp LLAMA_TOKEN_TYPE_*)
_TOKEN_NORMAL, _TOKEN_UNKNOWN, _TOKEN_CONTROL, _TOKEN_USER_DEFINED = 1, 2, 3, 4


def _merges_from_scores(
    tokens: list[str], scores: list[float], token_types: list[int]
) -> list[tuple[str, str]]:
    """Derive ordered BPE merges from SentencePiece piece scores.

    SentencePiece BPE repeatedly merges the adjacent pair whose concatenation
    has the highest score. Ranking every in-vocab split of each normal piece
    by that piece's score gives the equivalent HF merge list (the derivation
    transformers' GGUF converter uses). Control, byte and unknown pieces are
    never merge results.
    """
    score_of = dict(zip(tokens, scores))
    ranked: list[tuple[str, str, float]] = []
    for i, piece in enumerate(tokens):
        if token_types and token_types[i] != _TOKEN_NORMAL:
            continue
        local = [
            (piece[:k], piece[k:], score_of[piece])
            for k in range(1, len(piece))
            if piece[:k] in score_of and piece[k:] in score_of
        ]
        local.sort(key=lambda m: (score_of[m[0]], score_of[m[1]]), reverse=True)
        ranked.extend(local)
    ranked.sort(key=lambda m: m[2], reverse=True)
    return [(left, right) for left, right, _ in ranked]


def _build_tokenizer(meta: dict):
    """Build a HuggingFace tokenizer from GGUF metadata.

    ``tokenizer.ggml.model == "llama"`` (SentencePiece) becomes a byte-fallback
    BPE with the ▁ space normalizer; ``"gpt2"`` becomes a byte-level BPE with
    the pre-tokenizer named by ``tokenizer.ggml.pre``. Control tokens are
    registered as special tokens and BOS/EOS are added per
    ``tokenizer.ggml.add_bos_token`` / ``add_eos_token``.

    Returns a PreTrainedTokenizerFast, or None if vocab data is missing.
    """
    tokens = meta.get("tokenizer.ggml.tokens")
    if not tokens:
        return None

    model_type = meta.get("tokenizer.ggml.model", "llama")
    if model_type not in ("llama", "gpt2"):
        log.warning("Unsupported tokenizer.ggml.model %r; returning None for tokenizer", model_type)
        return None

    try:
        from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
        from tokenizers.models import BPE
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        log.warning("tokenizers library not available; returning None for tokenizer")
        return None

    scores = meta.get("tokenizer.ggml.scores") or []
    token_types = meta.get("tokenizer.ggml.token_type") or []
    merges: list[tuple[str, str]] = []
    for m in meta.get("tokenizer.ggml.merges") or []:
        left, sep, right = m.partition(" ")
        if sep:
            merges.append((left, right))

    # llama.cpp's SentencePiece defaults when the ids are absent
    spm = model_type == "llama"

    def token_at(key: str, default: int | None) -> tuple[str | None, int | None]:
        idx = meta.get(key, default)
        if isinstance(idx, int) and 0 <= idx < len(tokens):
            return tokens[idx], idx
        return None, None

    bos, bos_id = token_at("tokenizer.ggml.bos_token_id", 1 if spm else None)
    eos, eos_id = token_at("tokenizer.ggml.eos_token_id", 2 if spm else None)
    unk, _ = token_at("tokenizer.ggml.unknown_token_id", 0 if spm else None)
    pad, _ = token_at("tokenizer.ggml.padding_token_id", None)

    vocab = {tok: i for i, tok in enumerate(tokens)}

    if spm:
        if not merges:
            if not scores:
                log.warning("No merges or scores in GGUF metadata; returning None for tokenizer")
                return None
            merges = _merges_from_scores(tokens, scores, token_types)
        tok = Tokenizer(BPE(
            vocab=vocab, merges=merges, unk_token=unk, fuse_unk=True, byte_fallback=True,
        ))
        add_space_prefix = meta.get("tokenizer.ggml.add_space_prefix", True)
        norm_steps = [normalizers.Prepend("\u2581")] if add_space_prefix else []
        norm_steps.append(normalizers.Replace(" ", "\u2581"))
        tok.normalizer = normalizers.Sequence(norm_steps)
        dec_steps = [decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse()]
        if add_space_prefix:
            dec_steps.append(decoders.Strip(content=" ", left=1, right=0))
        tok.decoder = decoders.Sequence(dec_steps)
    else:
        if not merges:
            log.warning("gpt2-style GGUF vocab has no merges; returning None for tokenizer")
            return None
        pre = meta.get("tokenizer.ggml.pre", "default")
        if pre == "llama-bpe":
            # llama.cpp sets ignore_merges for this pre-type, as does the
            # Llama-3 tokenizer.json: whole-word vocab hits skip the merges.
            tok = Tokenizer(BPE(vocab=vocab, merges=merges, ignore_merges=True))
            tok.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(LLAMA3_SPLIT_REGEX), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ])
        else:
            if pre != "gpt-2":
                log.warning(
                    "GGUF pre-tokenizer %r is not reproduced; using the GPT-2 split "
                    "pattern, so token ids can differ from llama.cpp", pre,
                )
            tok = Tokenizer(BPE(vocab=vocab, merges=merges))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()

    if token_types:
        special = [
            t for t, ty in zip(tokens, token_types)
            if ty in (_TOKEN_CONTROL, _TOKEN_UNKNOWN)
        ]
        user_defined = [t for t, ty in zip(tokens, token_types) if ty == _TOKEN_USER_DEFINED]
    else:
        special = [t for t in (unk, bos, eos) if t is not None]
        user_defined = []
    tok.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in special])
    if user_defined:
        tok.add_tokens([AddedToken(t, normalized=False, special=False) for t in user_defined])

    add_bos = meta.get("tokenizer.ggml.add_bos_token", spm) and bos is not None
    add_eos = meta.get("tokenizer.ggml.add_eos_token", False) and eos is not None
    if add_bos or add_eos:
        head = [bos] if add_bos else []
        tail = [eos] if add_eos else []
        template_ids = [(bos, bos_id)] if add_bos else []
        if add_eos:
            template_ids.append((eos, eos_id))
        tok.post_processor = TemplateProcessing(
            single=[*head, "$A", *tail],
            pair=[*head, "$A", *tail, *head, "$B:1", *tail],
            special_tokens=template_ids,
        )

    chat_template = meta.get("tokenizer.chat_template")

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=bos,
        eos_token=eos,
        unk_token=unk,
        pad_token=pad,
        clean_up_tokenization_spaces=False,
    )
    if chat_template:
        hf_tok.chat_template = chat_template

    return hf_tok


def _reverse_permute(t: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Reverse the Q/K head interleaving that convert_hf_to_gguf.py applies.

    llama.cpp permutes Q and K weights during HF→GGUF conversion to
    interleave the first and second halves of each head's dimensions.
    This undoes that permutation so the weights match HF's layout.
    ``n_groups`` is the number of heads in the matrix: the attention head
    count for Q, the KV head count for K. Inverse of
    ``gguf_writer._forward_permute``.
    """
    dim = t.shape[0] // n_groups // 2
    return t.reshape(n_groups, dim, 2, *t.shape[1:]).swapaxes(1, 2).reshape(t.shape)


# ── Main loader ──────────────────────────────────────────────────────

# Per-frequency divisors that Llama-3.1+ GGUFs carry in place of HF's
# rope_scaling={"rope_type": "llama3", ...} (llama.cpp "freq_factors").
_ROPE_FREQS = "rope_freqs.weight"


def _apply_rope_freqs(rotary: torch.nn.Module, rope_freqs: torch.Tensor) -> None:
    """Divide a rotary module's inverse frequencies by llama.cpp freq factors.

    llama.cpp computes theta = pos * inv_freq / rope_freqs[i]; for llama3
    scaling this equals HF's rescaled inv_freq exactly.
    """
    inv_freq = rotary.inv_freq
    assert isinstance(inv_freq, torch.Tensor)
    if rope_freqs.shape != inv_freq.shape:
        raise ValueError(
            f"{_ROPE_FREQS} has shape {tuple(rope_freqs.shape)}, "
            f"expected {tuple(inv_freq.shape)} (head_dim / 2)"
        )
    scaled = inv_freq / rope_freqs.to(inv_freq.dtype)
    rotary.inv_freq = scaled
    rotary.original_inv_freq = scaled.clone()

def load_gguf_as_hf(
    gguf_path,
    dtype=torch.float16,
) -> tuple:
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

        config = _build_config(g.metadata, g.tensor_infos)
        log.info(
            "Config: %d layers, %d hidden, %d heads, %d vocab",
            config.num_hidden_layers, config.hidden_size,
            config.num_attention_heads, config.vocab_size,
        )

        # A tensor with no HF counterpart would be dropped and the model
        # would compute different outputs from llama.cpp; refuse instead.
        unmapped = [
            info.name for info in g.tensor_infos
            if info.name != _ROPE_FREQS and _map_tensor_name(info.name) is None
        ]
        if unmapped:
            raise ValueError(
                f"GGUF has {len(unmapped)} tensors with no LlamaForCausalLM "
                f"mapping, e.g. {unmapped[:5]}"
            )

        # Build state dict from GGUF tensors
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads or n_heads
        state_dict = {}
        rope_freqs = None
        for info in g.tensor_infos:
            if info.name == _ROPE_FREQS:
                rope_freqs = torch.from_numpy(g.read_tensor_numpy(info.name)).float()
                continue
            hf_name = _map_tensor_name(info.name)
            assert hf_name is not None
            t = g.read_tensor(info.name, dtype=dtype)
            if ".attn_q." in info.name:
                t = _reverse_permute(t, n_heads)
            elif ".attn_k." in info.name:
                t = _reverse_permute(t, n_kv_heads)
            state_dict[hf_name] = t

        # Handle tied embeddings: if output.weight is absent, share embed_tokens
        if "lm_head.weight" not in state_dict and "model.embed_tokens.weight" in state_dict:
            config.tie_word_embeddings = True
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
            log.info("Tied embeddings: lm_head shares embed_tokens weight")

        # Create model on meta device (no memory), then fill with real weights
        with torch.device("meta"):  # type: ignore
            model = LlamaForCausalLM(config)

        # strict=False so both lists come back for one clear error; any
        # missing key would otherwise stay a meta tensor and fail in forward.
        result = model.load_state_dict(state_dict, assign=True, strict=False)  # type: ignore
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"GGUF tensors do not match LlamaForCausalLM: "
                f"{len(result.missing_keys)} missing (e.g. {result.missing_keys[:5]}), "
                f"{len(result.unexpected_keys)} unexpected (e.g. {result.unexpected_keys[:5]})"
            )

        # Non-persistent buffers (RoPE inv_freq) are not in any state dict and
        # remain meta tensors. Rebuild each RoPE module on the CPU from config.
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        for name, module in model.named_modules():
            if isinstance(module, LlamaRotaryEmbedding):
                parent_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name[0]) if len(parent_name) > 1 else model
                attr = parent_name[1] if len(parent_name) > 1 else name
                rotary = LlamaRotaryEmbedding(config)
                if rope_freqs is not None:
                    _apply_rope_freqs(rotary, rope_freqs)
                setattr(parent, attr, rotary)

        model.eval()
        model.requires_grad_(False)

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

def resolve_ollama_blob(model_id: str, models_dir: str | None = None) -> Path | None:
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
            tname = ti.type_name
            n = ti.n_elements
            type_counts[tname] = type_counts.get(tname, 0) + n
            total_elements += n
            bs = _block_size(ti.ggml_type)
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
