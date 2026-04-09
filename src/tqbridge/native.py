"""ctypes bindings to libtqbridge — native C compression for the hot path.

Loads the shared library built from llama-cpp-turboquant/src/tqbridge.c and
exposes a NativeCompressor with the same interface as TinygradCompressor.

The native path eliminates Python/tinygrad overhead for compress/decompress,
which is critical for hitting tensor-core-class throughput on the RTX PRO 6000.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from tqbridge.wire import Format

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

_LIB: ctypes.CDLL | None = None


def _find_library() -> Path | None:
    """Search for libtqbridge in known locations."""
    candidates = []

    # 1. Explicit env var
    env = os.environ.get("TQBRIDGE_LIB")
    if env:
        candidates.append(Path(env))

    # 2. Relative to this package (llama-cpp-turboquant/build/bin/)
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent  # src/tqbridge -> src -> repo
    build_bin = repo_root / "llama-cpp-turboquant" / "build" / "bin"

    if sys.platform == "darwin":
        candidates.append(build_bin / "libtqbridge.dylib")
    elif sys.platform == "linux":
        candidates.append(build_bin / "libtqbridge.so")
    else:
        candidates.append(build_bin / "tqbridge.dll")

    # 3. System library path
    system = ctypes.util.find_library("tqbridge")
    if system:
        candidates.append(Path(system))

    for path in candidates:
        if path.exists():
            return path
    return None


def _load_lib() -> ctypes.CDLL:
    """Load libtqbridge, raising ImportError if not found."""
    global _LIB
    if _LIB is not None:
        return _LIB

    path = _find_library()
    if path is None:
        raise ImportError(
            "libtqbridge not found. Build it with:\n"
            "  cd llama-cpp-turboquant/build && cmake .. && "
            "cmake --build . --target tqbridge"
        )

    _LIB = ctypes.CDLL(str(path))
    _setup_prototypes(_LIB)
    return _LIB


# ---------------------------------------------------------------------------
# C type definitions
# ---------------------------------------------------------------------------

class _TqCompressed(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("n_vectors", ctypes.c_size_t),
        ("head_dim", ctypes.c_uint16),
        ("fmt", ctypes.c_int),
    ]


class _TqWireHeader(ctypes.Structure):
    _fields_ = [
        ("fmt_k", ctypes.c_int),
        ("fmt_v", ctypes.c_int),
        ("n_layers", ctypes.c_uint16),
        ("layer_start", ctypes.c_uint16),
        ("seq_len", ctypes.c_uint32),
        ("n_heads_k", ctypes.c_uint16),
        ("n_heads_v", ctypes.c_uint16),
        ("head_dim", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("payload_bytes", ctypes.c_uint64),
        ("version", ctypes.c_uint8),
    ]


# Opaque bridge pointer
_TqBridgePtr = ctypes.c_void_p

# Status enum
TQ_STATUS_OK = 0


def _setup_prototypes(lib: ctypes.CDLL) -> None:
    """Declare C function signatures for type safety."""

    # tq_bridge_init
    lib.tq_bridge_init.restype = ctypes.c_int
    lib.tq_bridge_init.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # tq_bridge **
        ctypes.c_int,                      # head_dim
        ctypes.c_int,                      # fmt
        ctypes.c_int,                      # seed
    ]

    # tq_bridge_init_precomputed
    lib.tq_bridge_init_precomputed.restype = ctypes.c_int
    lib.tq_bridge_init_precomputed.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # tq_bridge **
        ctypes.c_int,                      # head_dim
        ctypes.c_int,                      # fmt
        ctypes.POINTER(ctypes.c_float),    # rotation
        ctypes.POINTER(ctypes.c_float),    # codebook
        ctypes.c_int,                      # n_centroids
    ]

    # tq_bridge_free
    lib.tq_bridge_free.restype = None
    lib.tq_bridge_free.argtypes = [ctypes.c_void_p]

    # tq_compress
    lib.tq_compress.restype = ctypes.c_int
    lib.tq_compress.argtypes = [
        ctypes.c_void_p,                         # bridge
        ctypes.POINTER(ctypes.c_float),          # input
        ctypes.c_size_t,                         # n_vectors
        ctypes.POINTER(_TqCompressed),           # output
    ]

    # tq_decompress
    lib.tq_decompress.restype = ctypes.c_int
    lib.tq_decompress.argtypes = [
        ctypes.c_void_p,                         # bridge
        ctypes.POINTER(_TqCompressed),           # input
        ctypes.POINTER(ctypes.c_float),          # output
    ]

    # tq_compressed_free
    lib.tq_compressed_free.restype = None
    lib.tq_compressed_free.argtypes = [ctypes.POINTER(_TqCompressed)]

    # tq_compress_q8_0
    lib.tq_compress_q8_0.restype = ctypes.c_int
    lib.tq_compress_q8_0.argtypes = [
        ctypes.POINTER(ctypes.c_float),          # input
        ctypes.c_size_t,                         # n_elements
        ctypes.POINTER(ctypes.c_void_p),         # out_data
        ctypes.POINTER(ctypes.c_size_t),         # out_size
    ]

    # tq_decompress_q8_0
    lib.tq_decompress_q8_0.restype = ctypes.c_int
    lib.tq_decompress_q8_0.argtypes = [
        ctypes.c_void_p,                         # data
        ctypes.c_size_t,                         # n_elements
        ctypes.POINTER(ctypes.c_float),          # output
    ]

    # tq_encode_header
    lib.tq_encode_header.restype = ctypes.c_int
    lib.tq_encode_header.argtypes = [
        ctypes.POINTER(_TqWireHeader),
        ctypes.POINTER(ctypes.c_uint8),
    ]

    # tq_decode_header
    lib.tq_decode_header.restype = ctypes.c_int
    lib.tq_decode_header.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(_TqWireHeader),
    ]

    # tq_compression_ratio
    lib.tq_compression_ratio.restype = ctypes.c_float
    lib.tq_compression_ratio.argtypes = [ctypes.c_int, ctypes.c_int]

    # tq_compressed_size
    lib.tq_compressed_size.restype = ctypes.c_size_t
    lib.tq_compressed_size.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t]

    # tq_status_str
    lib.tq_status_str.restype = ctypes.c_char_p
    lib.tq_status_str.argtypes = [ctypes.c_int]


# ---------------------------------------------------------------------------
# Python format ↔ C format mapping
# ---------------------------------------------------------------------------

_PY_TO_C_FMT = {
    Format.FP16:    0x10,
    Format.Q8_0:    0x08,
    Format.Q5_K_M:  0x05,
    Format.TURBO4:  0x04,
    Format.TURBO3:  0x03,
    Format.TURBO2:  0x02,
}

_FORMAT_BITS = {
    Format.TURBO2: 2,
    Format.TURBO3: 3,
    Format.TURBO4: 4,
}


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

class NativeError(RuntimeError):
    """Error from the native tqbridge library."""
    pass


def _check(lib: ctypes.CDLL, status: int) -> None:
    if status != TQ_STATUS_OK:
        msg = lib.tq_status_str(status)
        raise NativeError(f"tqbridge: {msg.decode()}")


# ---------------------------------------------------------------------------
# NativeBridge — low-level context wrapper
# ---------------------------------------------------------------------------

class NativeBridge:
    """Low-level wrapper around tq_bridge C context.

    Uses precomputed rotation matrix and codebook from the Python NumPy
    reference implementation, ensuring bit-exact parity between C and Python
    compression outputs for the same seed.
    """

    def __init__(self, head_dim: int, fmt: Format, seed: int = 42):
        self._lib = _load_lib()
        self._ptr = ctypes.c_void_p()

        bit_width = _FORMAT_BITS.get(fmt)
        if bit_width is not None:
            # Use precomputed rotation + codebook from the Python oracle
            from tqbridge.compression import _get_rotation, _get_codebook

            rotation = _get_rotation(head_dim, seed).astype(np.float32)
            codebook = _get_codebook(bit_width, head_dim).astype(np.float32)
            rotation_c = np.ascontiguousarray(rotation)
            codebook_c = np.ascontiguousarray(codebook)

            status = self._lib.tq_bridge_init_precomputed(
                ctypes.byref(self._ptr), head_dim, _PY_TO_C_FMT[fmt],
                rotation_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                codebook_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(codebook_c),
            )
        else:
            # Q8_0 / FP16: no rotation needed
            status = self._lib.tq_bridge_init(
                ctypes.byref(self._ptr), head_dim, _PY_TO_C_FMT[fmt], seed
            )

        _check(self._lib, status)
        self.head_dim = head_dim
        self.fmt = fmt
        self.seed = seed

    def compress(self, vectors: np.ndarray) -> tuple[np.ndarray, _TqCompressed]:
        """Compress float32 vectors via the C bridge.

        Args:
            vectors: shape (n_vectors, head_dim) float32, C-contiguous

        Returns:
            (compressed_bytes, c_struct) — bytes as np.ndarray, struct for decompress
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n_vectors = vectors.shape[0]
        comp = _TqCompressed()

        status = self._lib.tq_compress(
            self._ptr,
            vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_vectors,
            ctypes.byref(comp),
        )
        _check(self._lib, status)

        # Copy compressed bytes to Python-owned buffer before we might free the C buffer
        buf = (ctypes.c_uint8 * comp.size).from_address(comp.data)
        compressed_bytes = np.frombuffer(buf, dtype=np.uint8).copy()

        return compressed_bytes, comp

    def decompress(self, comp: _TqCompressed) -> np.ndarray:
        """Decompress back to float32 vectors.

        Args:
            comp: _TqCompressed struct from compress()

        Returns:
            np.ndarray shape (n_vectors, head_dim) float32
        """
        n_vectors = comp.n_vectors
        output = np.empty(n_vectors * self.head_dim, dtype=np.float32)

        status = self._lib.tq_decompress(
            self._ptr,
            ctypes.byref(comp),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        _check(self._lib, status)
        return output.reshape(n_vectors, self.head_dim)

    def free_compressed(self, comp: _TqCompressed) -> None:
        """Free C-allocated compressed payload."""
        self._lib.tq_compressed_free(ctypes.byref(comp))

    def close(self) -> None:
        if self._ptr:
            self._lib.tq_bridge_free(self._ptr)
            self._ptr = ctypes.c_void_p()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# NativeCompressor — drop-in replacement for TinygradCompressor
# ---------------------------------------------------------------------------

class NativeCompressor:
    """Native C compression matching TinygradCompressor interface.

    Operates on NumPy arrays (CPU). For the bridge pipeline, tensors are
    pulled to CPU via .numpy(), compressed natively, transferred as raw
    bytes, then decompressed natively on the destination.

    Usage:
        compressor = NativeCompressor(head_dim=128, seed=42)
        compressed = compressor.compress(np_array, Format.TURBO3)
        restored = compressor.decompress(compressed)
    """

    def __init__(self, head_dim: int = 128, seed: int = 42):
        self.head_dim = head_dim
        self.seed = seed
        self._bridges: dict[Format, NativeBridge] = {}

    def _get_bridge(self, fmt: Format) -> NativeBridge:
        if fmt not in self._bridges:
            self._bridges[fmt] = NativeBridge(self.head_dim, fmt, self.seed)
        return self._bridges[fmt]

    def compress(self, data: np.ndarray, fmt: Format = Format.TURBO3) -> dict:
        """Compress a NumPy array.

        Args:
            data: float32 array, last dim must be head_dim for PolarQuant
            fmt: compression format

        Returns:
            Dict with compressed payload (same structure as TinygradCompressor).
        """
        orig_shape = data.shape
        n_elements = data.size

        if fmt in _FORMAT_BITS:
            vectors = data.reshape(-1, self.head_dim).astype(np.float32, copy=False)
            bridge = self._get_bridge(fmt)
            compressed_bytes, comp = bridge.compress(vectors)
            n_vectors = vectors.shape[0]

            result = {
                "fmt": fmt,
                "compressed_bytes": compressed_bytes,
                "n_vectors": n_vectors,
                "orig_shape": orig_shape,
                "n_elements": n_elements,
                "bit_width": _FORMAT_BITS[fmt],
                "_c_struct": comp,
                "_bridge": bridge,
            }
            return result

        elif fmt == Format.Q8_0:
            flat = np.ascontiguousarray(data.reshape(-1), dtype=np.float32)
            lib = _load_lib()

            out_data = ctypes.c_void_p()
            out_size = ctypes.c_size_t()
            status = lib.tq_compress_q8_0(
                flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                flat.size,
                ctypes.byref(out_data),
                ctypes.byref(out_size),
            )
            _check(lib, status)

            buf = (ctypes.c_uint8 * out_size.value).from_address(out_data.value)
            compressed_bytes = np.frombuffer(buf, dtype=np.uint8).copy()
            lib.free(out_data)

            return {
                "fmt": fmt,
                "compressed_bytes": compressed_bytes,
                "orig_shape": orig_shape,
                "n_elements": n_elements,
            }

        elif fmt == Format.FP16:
            return {
                "fmt": fmt,
                "data": data.astype(np.float16),
                "orig_shape": orig_shape,
                "n_elements": n_elements,
            }
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def decompress(self, compressed: dict) -> np.ndarray:
        """Decompress back to float32 NumPy array."""
        fmt = compressed["fmt"]

        if fmt in _FORMAT_BITS:
            if "_bridge" in compressed:
                # Local path: use the C struct from compress()
                bridge = compressed["_bridge"]
                comp = compressed["_c_struct"]
                vectors = bridge.decompress(comp)
                bridge.free_compressed(comp)
                return vectors.reshape(compressed["orig_shape"])
            else:
                # Remote/TCP path: decompress from raw compressed bytes
                bridge = self._get_bridge(fmt)
                n_vectors = compressed.get("n_vectors",
                    compressed["n_elements"] // self.head_dim)
                raw = np.ascontiguousarray(compressed["compressed_bytes"], dtype=np.uint8)

                # Feed bytes into C bridge for decompression
                c_comp = _TqCompressed()
                c_buf = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
                c_comp.data = ctypes.cast(c_buf, ctypes.c_void_p)
                c_comp.size = len(raw)
                c_comp.n_vectors = n_vectors
                c_comp.head_dim = self.head_dim
                c_comp.fmt = _PY_TO_C_FMT[fmt]

                vectors = bridge.decompress(c_comp)
                return vectors.reshape(compressed["orig_shape"])

        elif fmt == Format.Q8_0:
            lib = _load_lib()
            n_elements = compressed["n_elements"]
            compressed_bytes = compressed["compressed_bytes"]
            output = np.empty(n_elements, dtype=np.float32)

            # Allocate C buffer for compressed data
            c_buf = (ctypes.c_uint8 * len(compressed_bytes))(*compressed_bytes)

            status = lib.tq_decompress_q8_0(
                ctypes.cast(c_buf, ctypes.c_void_p),
                n_elements,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            _check(lib, status)
            return output.reshape(compressed["orig_shape"])

        elif fmt == Format.FP16:
            return compressed["data"].astype(np.float32).reshape(
                compressed["orig_shape"]
            )
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def compressed_size_bytes(self, compressed: dict) -> int:
        """Compressed payload size in bytes."""
        fmt = compressed["fmt"]
        if "compressed_bytes" in compressed:
            return len(compressed["compressed_bytes"])
        if fmt == Format.FP16:
            return compressed["n_elements"] * 2
        return 0

    def close(self) -> None:
        for bridge in self._bridges.values():
            bridge.close()
        self._bridges.clear()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Native wire protocol helpers
# ---------------------------------------------------------------------------

def native_encode_header(
    fmt_k: Format, fmt_v: Format,
    n_layers: int, layer_start: int, seq_len: int,
    n_heads_k: int, n_heads_v: int, head_dim: int,
    payload_bytes: int, flags: int = 0,
) -> bytes:
    """Encode a 40-byte wire header using the C implementation."""
    lib = _load_lib()
    hdr = _TqWireHeader(
        fmt_k=_PY_TO_C_FMT[fmt_k], fmt_v=_PY_TO_C_FMT[fmt_v],
        n_layers=n_layers, layer_start=layer_start, seq_len=seq_len,
        n_heads_k=n_heads_k, n_heads_v=n_heads_v, head_dim=head_dim,
        flags=flags, payload_bytes=payload_bytes, version=1,
    )
    buf = (ctypes.c_uint8 * 40)()
    status = lib.tq_encode_header(ctypes.byref(hdr), buf)
    _check(lib, status)
    return bytes(buf)


def native_decode_header(data: bytes) -> dict:
    """Decode a 40-byte wire header using the C implementation."""
    lib = _load_lib()
    buf = (ctypes.c_uint8 * 40)(*data)
    hdr = _TqWireHeader()
    status = lib.tq_decode_header(buf, ctypes.byref(hdr))
    _check(lib, status)
    return {
        "fmt_k": hdr.fmt_k,
        "fmt_v": hdr.fmt_v,
        "n_layers": hdr.n_layers,
        "layer_start": hdr.layer_start,
        "seq_len": hdr.seq_len,
        "n_heads_k": hdr.n_heads_k,
        "n_heads_v": hdr.n_heads_v,
        "head_dim": hdr.head_dim,
        "flags": hdr.flags,
        "payload_bytes": hdr.payload_bytes,
        "version": hdr.version,
    }
