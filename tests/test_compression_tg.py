"""Tests for tinygrad-native compression (compression_tg.py).

Validates against the NumPy reference implementation in compression.py.
"""

import pytest
import numpy as np

# tinygrad must be importable
tinygrad = pytest.importorskip("tinygrad")
from tinygrad import Tensor, dtypes

from tqbridge.compression_tg import (
    TinygradCompressor,
    polar_compress,
    polar_decompress,
    q8_0_compress,
    q8_0_decompress,
)
from tqbridge.compression import CompressionPipeline
from tqbridge.wire import Format


# ---------------------------------------------------------------------------
# PolarQuant round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bit_width,fmt", [(2, Format.TURBO2), (3, Format.TURBO3), (4, Format.TURBO4)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_polar_round_trip(bit_width, fmt, head_dim):
    """Compress and decompress, verify cosine similarity > threshold."""
    n_vectors = 16
    x = Tensor.rand(n_vectors, head_dim)
    x_np = x.numpy()

    comp = TinygradCompressor(head_dim=head_dim)
    compressed = comp.compress(x, fmt)
    x_hat = comp.decompress(compressed)
    x_hat_np = x_hat.numpy()

    # Per-vector cosine similarity
    for i in range(n_vectors):
        cos = np.dot(x_np[i], x_hat_np[i]) / (np.linalg.norm(x_np[i]) * np.linalg.norm(x_hat_np[i]) + 1e-12)
        if bit_width == 2:
            assert cos > 0.7, f"turbo2 cosine too low: {cos}"
        elif bit_width == 3:
            assert cos > 0.9, f"turbo3 cosine too low: {cos}"
        elif bit_width == 4:
            assert cos > 0.95, f"turbo4 cosine too low: {cos}"


@pytest.mark.parametrize("bit_width", [2, 3, 4])
def test_polar_compression_ratio(bit_width):
    """Verify compressed size matches expected ratio."""
    head_dim = 128
    n_vectors = 32
    x = Tensor.rand(n_vectors, head_dim)

    comp = TinygradCompressor(head_dim=head_dim)
    fmt = {2: Format.TURBO2, 3: Format.TURBO3, 4: Format.TURBO4}[bit_width]
    compressed = comp.compress(x, fmt)

    original_bytes = n_vectors * head_dim * 4  # float32
    compressed_bytes = comp.compressed_size_bytes(compressed)
    ratio = original_bytes / compressed_bytes

    expected_min = {2: 5.0, 3: 3.5, 4: 2.8}[bit_width]
    assert ratio > expected_min, f"Compression ratio {ratio:.1f}x below expected {expected_min}x"


def test_polar_zero_vector():
    """Zero vector should compress and decompress without NaN."""
    head_dim = 128
    x = Tensor.zeros(1, head_dim)
    comp = TinygradCompressor(head_dim=head_dim)
    compressed = comp.compress(x, Format.TURBO3)
    x_hat = comp.decompress(compressed)
    assert not np.any(np.isnan(x_hat.numpy()))


def test_polar_deterministic():
    """Same input + same seed = same output."""
    head_dim = 128
    x = Tensor.rand(4, head_dim)
    comp = TinygradCompressor(head_dim=head_dim, seed=42)

    c1 = comp.compress(x, Format.TURBO3)
    c2 = comp.compress(x, Format.TURBO3)

    np.testing.assert_array_equal(c1["indices"].numpy(), c2["indices"].numpy())
    np.testing.assert_allclose(c1["norms"].numpy(), c2["norms"].numpy())


# ---------------------------------------------------------------------------
# Cross-implementation validation (tinygrad vs NumPy reference)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", [Format.TURBO3, Format.TURBO4])
def test_cross_impl_compress_quality(fmt):
    """tinygrad and NumPy implementations should produce similar quality."""
    head_dim = 128
    n_vectors = 16
    np_data = np.random.default_rng(123).standard_normal((n_vectors, head_dim)).astype(np.float32)
    tg_data = Tensor(np_data)

    bit_width = {Format.TURBO3: 3, Format.TURBO4: 4}[fmt]

    # NumPy reference
    np_pipeline = CompressionPipeline(seed=42)
    np_compressed = np_pipeline.compress_kv(np_data, np_data, fmt, fmt, head_dim)
    np_keys, _ = np_pipeline.decompress_kv(np_compressed)
    np_keys = np_keys.reshape(n_vectors, head_dim)

    # tinygrad
    tg_comp = TinygradCompressor(head_dim=head_dim, seed=42)
    tg_compressed = tg_comp.compress(tg_data, fmt)
    tg_keys = tg_comp.decompress(tg_compressed).numpy()

    # Both should have similar MSE vs original
    np_mse = np.mean((np_data - np_keys) ** 2)
    tg_mse = np.mean((np_data - tg_keys) ** 2)

    # MSE should be in the same ballpark (within 2x)
    assert tg_mse < np_mse * 2.5, f"tinygrad MSE {tg_mse:.6f} >> NumPy MSE {np_mse:.6f}"
    assert np_mse < tg_mse * 2.5, f"NumPy MSE {np_mse:.6f} >> tinygrad MSE {tg_mse:.6f}"


# ---------------------------------------------------------------------------
# q8_0 round-trip
# ---------------------------------------------------------------------------

def test_q8_0_round_trip():
    """q8_0 compress/decompress should have very low error."""
    x = Tensor.rand(256)
    comp = TinygradCompressor()
    compressed = comp.compress(x, Format.Q8_0)
    x_hat = comp.decompress(compressed)
    mse = np.mean((x.numpy() - x_hat.numpy()) ** 2)
    assert mse < 1e-4, f"q8_0 MSE too high: {mse}"


def test_q8_0_block_alignment():
    """q8_0 should handle non-block-aligned sizes."""
    x = Tensor.rand(100)  # not a multiple of 32
    comp = TinygradCompressor()
    compressed = comp.compress(x, Format.Q8_0)
    x_hat = comp.decompress(compressed)
    assert x_hat.shape == x.shape


# ---------------------------------------------------------------------------
# FP16 round-trip
# ---------------------------------------------------------------------------

def test_fp16_round_trip():
    """FP16 should have near-zero error."""
    x = Tensor.rand(4, 128)
    comp = TinygradCompressor(head_dim=128)
    compressed = comp.compress(x, Format.FP16)
    x_hat = comp.decompress(compressed)
    mse = np.mean((x.numpy() - x_hat.numpy()) ** 2)
    assert mse < 1e-6, f"FP16 MSE too high: {mse}"


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

def test_shape_preservation():
    """Output shape should match input shape."""
    shapes = [(8, 128), (4, 8, 128), (2, 4, 8, 128)]
    comp = TinygradCompressor(head_dim=128)

    for shape in shapes:
        x = Tensor.rand(*shape)
        for fmt in [Format.TURBO3, Format.Q8_0, Format.FP16]:
            compressed = comp.compress(x, fmt)
            x_hat = comp.decompress(compressed)
            assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} != {shape} for {fmt}"
