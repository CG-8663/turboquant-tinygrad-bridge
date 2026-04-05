"""Compression pipeline: TurboQuant compress/decompress for K and V tensors.

Python reference implementation of PolarQuant (TurboQuant without QJL).
Algorithm: norm extraction -> rotation -> Lloyd-Max scalar quantisation -> bit packing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as normal_dist

from tqbridge.wire import Format


# ---------------------------------------------------------------------------
# Rotation matrix generation
# ---------------------------------------------------------------------------

def _rotation_matrix(d: int, seed: int = 42) -> NDArray[np.float64]:
    """Haar-distributed random orthogonal matrix via QR decomposition.

    Deterministic given (d, seed). Same seed on Metal and CUDA produces
    identical rotation -- no need to transmit the matrix.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    # Fix signs for Haar distribution
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q *= signs[np.newaxis, :]
    # Ensure det = +1 (proper rotation)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# Cache rotation matrices (they're expensive to compute, deterministic)
_rotation_cache: dict[tuple[int, int], NDArray[np.float64]] = {}


def _get_rotation(d: int, seed: int = 42) -> NDArray[np.float64]:
    key = (d, seed)
    if key not in _rotation_cache:
        _rotation_cache[key] = _rotation_matrix(d, seed)
    return _rotation_cache[key]


# ---------------------------------------------------------------------------
# Lloyd-Max codebook
# ---------------------------------------------------------------------------

def _lloyd_max_codebook(bit_width: int, d: int, n_iter: int = 100) -> NDArray[np.float64]:
    """Compute optimal Lloyd-Max centroids for N(0, 1/d) distribution.

    Returns sorted array of 2^bit_width centroids.
    """
    n_centroids = 1 << bit_width
    sigma = 1.0 / np.sqrt(d)

    # Initialise centroids from uniform quantiles
    quantiles = np.linspace(0, 1, n_centroids + 2)[1:-1]  # skip 0 and 1
    centroids = normal_dist.ppf(quantiles, scale=sigma)

    for _ in range(n_iter):
        # Boundaries = midpoints between centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids via conditional expectation E[X | a < X < b]
        # where X ~ N(0, sigma^2)
        lo = np.concatenate([[-np.inf], boundaries])
        hi = np.concatenate([boundaries, [np.inf]])

        new_centroids = np.empty(n_centroids)
        for i in range(n_centroids):
            a, b = lo[i], hi[i]
            # E[X | a < X < b] = sigma^2 * (phi(a/sigma) - phi(b/sigma)) / (Phi(b/sigma) - Phi(a/sigma))
            a_norm = a / sigma
            b_norm = b / sigma
            pdf_diff = normal_dist.pdf(a_norm) - normal_dist.pdf(b_norm)
            cdf_diff = normal_dist.cdf(b_norm) - normal_dist.cdf(a_norm)
            if cdf_diff > 1e-12:
                new_centroids[i] = sigma * pdf_diff / cdf_diff
            else:
                new_centroids[i] = centroids[i]

        centroids = new_centroids

    return np.sort(centroids)


_codebook_cache: dict[tuple[int, int], NDArray[np.float64]] = {}


def _get_codebook(bit_width: int, d: int) -> NDArray[np.float64]:
    key = (bit_width, d)
    if key not in _codebook_cache:
        _codebook_cache[key] = _lloyd_max_codebook(bit_width, d)
    return _codebook_cache[key]


# ---------------------------------------------------------------------------
# Bit packing / unpacking
# ---------------------------------------------------------------------------

def _pack_indices(indices: NDArray[np.uint8], bit_width: int) -> bytes:
    """Pack an array of b-bit indices into a compact byte array."""
    n = len(indices)
    total_bits = n * bit_width
    n_bytes = (total_bits + 7) // 8
    buf = bytearray(n_bytes)

    for i, idx in enumerate(indices):
        bit_pos = i * bit_width
        byte_pos = bit_pos // 8
        bit_off = bit_pos % 8

        # Write bits, possibly spanning two bytes
        buf[byte_pos] |= (idx & ((1 << bit_width) - 1)) << bit_off
        overflow = bit_off + bit_width - 8
        if overflow > 0 and byte_pos + 1 < n_bytes:
            buf[byte_pos + 1] |= idx >> (bit_width - overflow)

    return bytes(buf)


def _unpack_indices(data: bytes, n: int, bit_width: int) -> NDArray[np.uint8]:
    """Unpack n b-bit indices from a compact byte array."""
    mask = (1 << bit_width) - 1
    indices = np.empty(n, dtype=np.uint8)

    for i in range(n):
        bit_pos = i * bit_width
        byte_pos = bit_pos // 8
        bit_off = bit_pos % 8

        val = data[byte_pos] >> bit_off
        if bit_off + bit_width > 8 and byte_pos + 1 < len(data):
            val |= data[byte_pos + 1] << (8 - bit_off)
        indices[i] = val & mask

    return indices


# ---------------------------------------------------------------------------
# q8_0 block format (ggml-compatible)
# ---------------------------------------------------------------------------

QK8_0 = 32  # block size


def _compress_q8_0(data: NDArray[np.float32]) -> bytes:
    """Compress float32 array to q8_0 format (ggml-compatible blocks).

    Block layout: 2 bytes (fp16 scale) + 32 bytes (int8 values) = 34 bytes per block.
    """
    flat = data.ravel().astype(np.float32)
    # Pad to multiple of QK8_0
    remainder = len(flat) % QK8_0
    if remainder:
        flat = np.concatenate([flat, np.zeros(QK8_0 - remainder, dtype=np.float32)])

    n_blocks = len(flat) // QK8_0
    buf = bytearray()

    for b in range(n_blocks):
        block = flat[b * QK8_0 : (b + 1) * QK8_0]
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 0.0

        # Scale as float16
        scale_f16 = np.float16(scale)
        buf.extend(scale_f16.tobytes())

        # Quantise to int8
        if scale > 0:
            qs = np.round(block / scale).clip(-128, 127).astype(np.int8)
        else:
            qs = np.zeros(QK8_0, dtype=np.int8)
        buf.extend(qs.tobytes())

    return bytes(buf)


def _decompress_q8_0(data: bytes, n_elements: int) -> NDArray[np.float32]:
    """Decompress q8_0 format back to float32 array."""
    n_blocks = (n_elements + QK8_0 - 1) // QK8_0
    block_size = 2 + QK8_0  # 34 bytes per block
    result = np.empty(n_blocks * QK8_0, dtype=np.float32)

    for b in range(n_blocks):
        offset = b * block_size
        scale = np.frombuffer(data[offset : offset + 2], dtype=np.float16)[0]
        qs = np.frombuffer(data[offset + 2 : offset + block_size], dtype=np.int8)
        result[b * QK8_0 : (b + 1) * QK8_0] = float(scale) * qs.astype(np.float32)

    return result[:n_elements]


# ---------------------------------------------------------------------------
# PolarQuant (turbo3/turbo4) compress / decompress
# ---------------------------------------------------------------------------

# Per-vector wire layout:
#   4 bytes: float32 norm
#   ceil(bit_width * d / 8) bytes: packed indices

_FORMAT_BITS = {
    Format.TURBO2: 2,
    Format.TURBO3: 3,
    Format.TURBO4: 4,
}


def _polar_compress_vectors(
    vectors: NDArray[np.float32],
    bit_width: int,
    head_dim: int,
    seed: int = 42,
) -> bytes:
    """Compress a batch of vectors using PolarQuant.

    Args:
        vectors: shape (n_vectors, head_dim) float32
        bit_width: 2, 3, or 4
        head_dim: dimension per head
        seed: rotation matrix seed

    Returns:
        Packed bytes: for each vector, [float32 norm | packed indices]
    """
    rotation = _get_rotation(head_dim, seed)
    codebook = _get_codebook(bit_width, head_dim)
    boundaries = (codebook[:-1] + codebook[1:]) / 2.0

    n_vectors = vectors.shape[0]
    packed_index_bytes = (bit_width * head_dim + 7) // 8
    bytes_per_vector = 4 + packed_index_bytes  # norm + indices
    buf = bytearray(n_vectors * bytes_per_vector)

    for i in range(n_vectors):
        v = vectors[i].astype(np.float64)

        # Extract norm
        norm = np.linalg.norm(v)
        safe_norm = norm if norm > 0 else 1.0
        v_unit = v / safe_norm

        # Rotate
        y = rotation @ v_unit

        # Quantise: find nearest centroid index for each coordinate
        indices = np.searchsorted(boundaries, y).astype(np.uint8)

        # Pack
        offset = i * bytes_per_vector
        struct.pack_into("<f", buf, offset, float(norm))
        packed = _pack_indices(indices, bit_width)
        buf[offset + 4 : offset + 4 + packed_index_bytes] = packed

    return bytes(buf)


def _polar_decompress_vectors(
    data: bytes,
    n_vectors: int,
    bit_width: int,
    head_dim: int,
    seed: int = 42,
) -> NDArray[np.float32]:
    """Decompress PolarQuant-compressed vectors.

    Returns: shape (n_vectors, head_dim) float32
    """
    rotation = _get_rotation(head_dim, seed)
    codebook = _get_codebook(bit_width, head_dim)

    packed_index_bytes = (bit_width * head_dim + 7) // 8
    bytes_per_vector = 4 + packed_index_bytes
    result = np.empty((n_vectors, head_dim), dtype=np.float32)

    for i in range(n_vectors):
        offset = i * bytes_per_vector
        norm = struct.unpack_from("<f", data, offset)[0]
        indices = _unpack_indices(
            data[offset + 4 : offset + 4 + packed_index_bytes],
            head_dim,
            bit_width,
        )

        # Look up centroids
        y_hat = codebook[indices]

        # Inverse rotation
        x_hat = rotation.T @ y_hat

        # Rescale
        result[i] = (x_hat * norm).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# CompressionPipeline (public API)
# ---------------------------------------------------------------------------

@dataclass
class CompressedKV:
    """Compressed K and V tensors with metadata."""
    k_data: bytes
    v_data: bytes
    fmt_k: Format
    fmt_v: Format
    n_elements_k: int
    n_elements_v: int
    head_dim: int


class CompressionPipeline:
    """Compress and decompress K/V tensors using TurboQuant or q8_0.

    Supports asymmetric K/V configurations (e.g. q8_0 K + turbo3 V).
    """

    def __init__(self, seed: int = 42):
        self._seed = seed

    def compress_kv(
        self,
        keys: NDArray[np.float32],
        values: NDArray[np.float32],
        fmt_k: Format,
        fmt_v: Format,
        head_dim: int,
    ) -> CompressedKV:
        """Compress K and V tensors independently.

        Args:
            keys: shape (n_vectors_k, head_dim) float32
            values: shape (n_vectors_v, head_dim) float32
            fmt_k: compression format for keys
            fmt_v: compression format for values
            head_dim: dimension per attention head
        """
        k_data = self._compress_tensor(keys, fmt_k, head_dim)
        v_data = self._compress_tensor(values, fmt_v, head_dim)
        return CompressedKV(
            k_data=k_data,
            v_data=v_data,
            fmt_k=fmt_k,
            fmt_v=fmt_v,
            n_elements_k=keys.size,
            n_elements_v=values.size,
            head_dim=head_dim,
        )

    def decompress_kv(
        self,
        compressed: CompressedKV,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Decompress K and V tensors.

        Returns: (keys, values) as flat float32 arrays.
        """
        n_vectors_k = compressed.n_elements_k // compressed.head_dim
        n_vectors_v = compressed.n_elements_v // compressed.head_dim
        keys = self._decompress_tensor(
            compressed.k_data, compressed.fmt_k, n_vectors_k, compressed.head_dim
        )
        values = self._decompress_tensor(
            compressed.v_data, compressed.fmt_v, n_vectors_v, compressed.head_dim
        )
        return keys, values

    def _compress_tensor(
        self,
        tensor: NDArray[np.float32],
        fmt: Format,
        head_dim: int,
    ) -> bytes:
        if fmt == Format.Q8_0:
            return _compress_q8_0(tensor)
        elif fmt == Format.FP16:
            return tensor.astype(np.float16).tobytes()
        elif fmt in _FORMAT_BITS:
            vectors = tensor.reshape(-1, head_dim)
            return _polar_compress_vectors(
                vectors, _FORMAT_BITS[fmt], head_dim, self._seed
            )
        else:
            raise ValueError(f"Unsupported compression format: {fmt}")

    def _decompress_tensor(
        self,
        data: bytes,
        fmt: Format,
        n_vectors: int,
        head_dim: int,
    ) -> NDArray[np.float32]:
        if fmt == Format.Q8_0:
            return _decompress_q8_0(data, n_vectors * head_dim)
        elif fmt == Format.FP16:
            return np.frombuffer(data, dtype=np.float16).astype(np.float32)
        elif fmt in _FORMAT_BITS:
            return _polar_decompress_vectors(
                data, n_vectors, _FORMAT_BITS[fmt], head_dim, self._seed
            )
        else:
            raise ValueError(f"Unsupported decompression format: {fmt}")
