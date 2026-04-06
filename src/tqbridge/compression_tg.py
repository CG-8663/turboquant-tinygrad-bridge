"""Tinygrad-native compression: PolarQuant + q8_0 using tinygrad tensor ops.

This module mirrors compression.py but operates on tinygrad Tensors instead
of NumPy arrays. The NumPy reference in compression.py is the correctness
oracle; this module is for on-device performance.

Rotation matrices and codebooks are precomputed in NumPy (one-time, cached)
and stored as tinygrad Tensors on the target device.
"""

from __future__ import annotations

from tinygrad import Tensor, dtypes, Device
import numpy as np

from tqbridge.compression import _get_rotation, _get_codebook
from tqbridge.wire import Format


# ---------------------------------------------------------------------------
# Cached tinygrad tensors for rotation matrices and codebooks
# ---------------------------------------------------------------------------

_tg_rotation_cache: dict[tuple[int, int, str], Tensor] = {}
_tg_codebook_cache: dict[tuple[int, int, str], Tensor] = {}
_tg_boundary_cache: dict[tuple[int, int, str], Tensor] = {}


def _get_tg_rotation(d: int, seed: int, device: str) -> Tensor:
    key = (d, seed, device)
    if key not in _tg_rotation_cache:
        R = _get_rotation(d, seed).astype(np.float32)
        _tg_rotation_cache[key] = Tensor(R, device=device)
    return _tg_rotation_cache[key]


def _get_tg_codebook(bit_width: int, d: int, device: str) -> Tensor:
    key = (bit_width, d, device)
    if key not in _tg_codebook_cache:
        cb = _get_codebook(bit_width, d).astype(np.float32)
        _tg_codebook_cache[key] = Tensor(cb, device=device)
    return _tg_codebook_cache[key]


def _get_tg_boundaries(bit_width: int, d: int, device: str) -> Tensor:
    key = (bit_width, d, device)
    if key not in _tg_boundary_cache:
        cb = _get_codebook(bit_width, d).astype(np.float32)
        boundaries = (cb[:-1] + cb[1:]) / 2.0
        _tg_boundary_cache[key] = Tensor(boundaries, device=device)
    return _tg_boundary_cache[key]


# ---------------------------------------------------------------------------
# Format mapping
# ---------------------------------------------------------------------------

_FORMAT_BITS = {
    Format.TURBO2: 2,
    Format.TURBO3: 3,
    Format.TURBO4: 4,
}

# ---------------------------------------------------------------------------
# PolarQuant compress / decompress (tinygrad tensors)
# ---------------------------------------------------------------------------


def polar_compress(
    vectors: Tensor,
    bit_width: int,
    head_dim: int,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Compress vectors using PolarQuant on-device.

    Args:
        vectors: shape (n_vectors, head_dim) on any device
        bit_width: 2, 3, or 4
        head_dim: dimension per head
        seed: rotation matrix seed

    Returns:
        (norms, indices): norms is (n_vectors,) float32,
                          indices is (n_vectors, head_dim) uint8
    """
    device = vectors.device
    rotation = _get_tg_rotation(head_dim, seed, device)
    boundaries = _get_tg_boundaries(bit_width, head_dim, device)

    # Norm extraction
    norms = (vectors * vectors).sum(axis=-1).sqrt()  # (n_vectors,)
    safe_norms = norms.maximum(Tensor(1e-12, device=device))
    v_unit = vectors / safe_norms.unsqueeze(-1)  # (n_vectors, head_dim)

    # Rotate: v_unit @ rotation.T
    y = v_unit @ rotation.transpose()  # (n_vectors, head_dim)

    # Quantize via broadcasting: for each value, count how many boundaries it exceeds
    # y: (n_vectors, head_dim), boundaries: (n_boundaries,)
    # Compare: (n_vectors, head_dim, 1) >= (1, 1, n_boundaries)
    indices = (y.unsqueeze(-1) >= boundaries.reshape(1, 1, -1)).sum(axis=-1).cast(dtypes.uint8)

    return norms.realize(), indices.realize()


def polar_decompress(
    norms: Tensor,
    indices: Tensor,
    bit_width: int,
    head_dim: int,
    seed: int = 42,
    device: str | None = None,
) -> Tensor:
    """Decompress PolarQuant vectors on-device.

    Args:
        norms: shape (n_vectors,) float32
        indices: shape (n_vectors, head_dim) uint8 (codebook indices)
        bit_width: 2, 3, or 4
        head_dim: dimension per head
        seed: rotation matrix seed
        device: target device (defaults to indices.device)

    Returns:
        Reconstructed vectors: shape (n_vectors, head_dim) float32
    """
    device = device or indices.device
    rotation = _get_tg_rotation(head_dim, seed, device)
    codebook = _get_tg_codebook(bit_width, head_dim, device)

    # Codebook lookup
    y_hat = codebook[indices.cast(dtypes.int)]  # (n_vectors, head_dim)

    # Inverse rotation: y_hat @ rotation (since rotation is orthogonal, R^T = R^-1)
    x_hat = y_hat @ rotation  # (n_vectors, head_dim)

    # Rescale
    return (x_hat * norms.unsqueeze(-1)).realize()


# ---------------------------------------------------------------------------
# q8_0 compress / decompress (tinygrad tensors)
# ---------------------------------------------------------------------------

QK8_0 = 32


def q8_0_compress(data: Tensor) -> tuple[Tensor, Tensor]:
    """Compress float32 tensor to q8_0 format on-device.

    Args:
        data: flat float32 tensor

    Returns:
        (scales, quants): scales is (n_blocks,) float16,
                          quants is (n_blocks, 32) int8
    """
    flat = data.reshape(-1)
    n = flat.shape[0]
    # Pad to multiple of QK8_0
    remainder = n % QK8_0
    if remainder:
        pad = Tensor.zeros(QK8_0 - remainder, device=data.device)
        flat = flat.cat(pad)

    n_blocks = flat.shape[0] // QK8_0
    blocks = flat.reshape(n_blocks, QK8_0)  # (n_blocks, 32)

    # Per-block absolute max
    amax = blocks.abs().max(axis=-1)  # (n_blocks,)
    scales = (amax / 127.0).cast(dtypes.float16)  # (n_blocks,)

    # Quantize
    safe_scales = scales.cast(dtypes.float32).maximum(Tensor(1e-12, device=data.device))
    quants = (blocks / safe_scales.unsqueeze(-1)).round().clip(-128, 127).cast(dtypes.int8)

    return scales.realize(), quants.realize()


def q8_0_decompress(scales: Tensor, quants: Tensor, n_elements: int) -> Tensor:
    """Decompress q8_0 back to float32 on-device.

    Args:
        scales: (n_blocks,) float16
        quants: (n_blocks, 32) int8
        n_elements: original number of elements

    Returns:
        float32 tensor of shape (n_elements,)
    """
    result = quants.cast(dtypes.float32) * scales.cast(dtypes.float32).unsqueeze(-1)
    return result.reshape(-1)[:n_elements].realize()


# ---------------------------------------------------------------------------
# TinygradCompressor (public API matching CompressionPipeline interface)
# ---------------------------------------------------------------------------

class TinygradCompressor:
    """Compress/decompress KV tensors using tinygrad ops.

    Mirrors CompressionPipeline from compression.py but operates on
    tinygrad Tensors for on-device execution.
    """

    def __init__(self, head_dim: int = 128, seed: int = 42):
        self.head_dim = head_dim
        self.seed = seed

    def compress(
        self, tensor: Tensor, fmt: Format = Format.TURBO3
    ) -> dict:
        """Compress a KV tensor.

        Args:
            tensor: shape (..., head_dim) float32 on any device
            fmt: compression format

        Returns:
            Dict with compressed components (format-dependent).
        """
        orig_shape = tensor.shape
        n_elements = 1
        for s in orig_shape:
            n_elements *= s

        if fmt in _FORMAT_BITS:
            vectors = tensor.reshape(-1, self.head_dim)
            norms, indices = polar_compress(
                vectors, _FORMAT_BITS[fmt], self.head_dim, self.seed
            )
            return {
                "fmt": fmt,
                "norms": norms,
                "indices": indices,
                "orig_shape": orig_shape,
                "n_elements": n_elements,
                "bit_width": _FORMAT_BITS[fmt],
            }
        elif fmt == Format.Q8_0:
            scales, quants = q8_0_compress(tensor)
            return {
                "fmt": fmt,
                "scales": scales,
                "quants": quants,
                "orig_shape": orig_shape,
                "n_elements": n_elements,
            }
        elif fmt == Format.FP16:
            return {
                "fmt": fmt,
                "data": tensor.cast(dtypes.float16),
                "orig_shape": orig_shape,
                "n_elements": n_elements,
            }
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def decompress(self, compressed: dict) -> Tensor:
        """Decompress a compressed KV tensor.

        Returns: float32 tensor with original shape.
        """
        fmt = compressed["fmt"]

        if fmt in _FORMAT_BITS:
            vectors = polar_decompress(
                compressed["norms"],
                compressed["indices"],
                compressed["bit_width"],
                self.head_dim,
                self.seed,
            )
            return vectors.reshape(*compressed["orig_shape"])
        elif fmt == Format.Q8_0:
            flat = q8_0_decompress(
                compressed["scales"],
                compressed["quants"],
                compressed["n_elements"],
            )
            return flat.reshape(*compressed["orig_shape"])
        elif fmt == Format.FP16:
            return compressed["data"].cast(dtypes.float32).reshape(*compressed["orig_shape"])
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def compressed_size_bytes(self, compressed: dict) -> int:
        """Estimate compressed size in bytes for metrics."""
        fmt = compressed["fmt"]
        if fmt in _FORMAT_BITS:
            n_vectors = compressed["norms"].shape[0]
            # 4 bytes norm + ceil(bit_width * head_dim / 8) bytes indices per vector
            bits = _FORMAT_BITS[fmt]
            packed_bytes = (bits * self.head_dim + 7) // 8
            return n_vectors * (4 + packed_bytes)
        elif fmt == Format.Q8_0:
            n_blocks = compressed["scales"].shape[0]
            return n_blocks * 34  # 2 bytes scale + 32 bytes quants
        elif fmt == Format.FP16:
            return compressed["n_elements"] * 2
        return 0
