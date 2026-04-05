"""Tests for compression pipeline: turbo3, q8_0, asymmetric K/V, round-trips."""

import numpy as np
import pytest

from tqbridge.compression import (
    CompressionPipeline,
    _compress_q8_0,
    _decompress_q8_0,
    _pack_indices,
    _unpack_indices,
    _polar_compress_vectors,
    _polar_decompress_vectors,
    _get_rotation,
    _get_codebook,
    QK8_0,
)
from tqbridge.wire import Format


# ---------------------------------------------------------------------------
# Rotation matrix tests
# ---------------------------------------------------------------------------


def test_rotation_is_orthogonal():
    d = 128
    R = _get_rotation(d, seed=42)
    I = R @ R.T
    np.testing.assert_allclose(I, np.eye(d), atol=1e-10)


def test_rotation_deterministic():
    R1 = _get_rotation(128, seed=42)
    R2 = _get_rotation(128, seed=42)
    np.testing.assert_array_equal(R1, R2)


def test_rotation_different_seeds():
    R1 = _get_rotation(128, seed=42)
    R2 = _get_rotation(128, seed=99)
    assert not np.allclose(R1, R2)


def test_rotation_det_positive():
    R = _get_rotation(128, seed=42)
    assert np.linalg.det(R) > 0


# ---------------------------------------------------------------------------
# Codebook tests
# ---------------------------------------------------------------------------


def test_codebook_sorted():
    cb = _get_codebook(3, 128)
    assert np.all(cb[:-1] <= cb[1:])


def test_codebook_size():
    assert len(_get_codebook(2, 128)) == 4
    assert len(_get_codebook(3, 128)) == 8
    assert len(_get_codebook(4, 128)) == 16


def test_codebook_symmetric():
    """Codebook for N(0, sigma) should be approximately symmetric around 0."""
    cb = _get_codebook(3, 128)
    np.testing.assert_allclose(cb, -cb[::-1], atol=1e-6)


# ---------------------------------------------------------------------------
# Bit packing tests
# ---------------------------------------------------------------------------


def test_pack_unpack_3bit():
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1], dtype=np.uint8)
    packed = _pack_indices(indices, 3)
    unpacked = _unpack_indices(packed, len(indices), 3)
    np.testing.assert_array_equal(indices, unpacked)


def test_pack_unpack_4bit():
    indices = np.arange(16, dtype=np.uint8)
    packed = _pack_indices(indices, 4)
    unpacked = _unpack_indices(packed, len(indices), 4)
    np.testing.assert_array_equal(indices, unpacked)


def test_pack_unpack_2bit():
    indices = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
    packed = _pack_indices(indices, 2)
    unpacked = _unpack_indices(packed, len(indices), 2)
    np.testing.assert_array_equal(indices, unpacked)


def test_pack_unpack_large():
    """Round-trip for head_dim=128 with 3-bit packing."""
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 8, size=128, dtype=np.uint8)
    packed = _pack_indices(indices, 3)
    assert len(packed) == (128 * 3 + 7) // 8  # 48 bytes
    unpacked = _unpack_indices(packed, 128, 3)
    np.testing.assert_array_equal(indices, unpacked)


# ---------------------------------------------------------------------------
# q8_0 tests
# ---------------------------------------------------------------------------


def test_q8_0_round_trip():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(128).astype(np.float32)
    compressed = _compress_q8_0(data)
    decompressed = _decompress_q8_0(compressed, 128)

    # q8_0 should have very small error (8-bit quantisation)
    rel_error = np.linalg.norm(data - decompressed) / np.linalg.norm(data)
    assert rel_error < 0.01  # < 1% relative error


def test_q8_0_block_size():
    """Each q8_0 block is 34 bytes (2 scale + 32 values)."""
    data = np.ones(QK8_0, dtype=np.float32)
    compressed = _compress_q8_0(data)
    assert len(compressed) == 34


def test_q8_0_zero_vector():
    data = np.zeros(64, dtype=np.float32)
    compressed = _compress_q8_0(data)
    decompressed = _decompress_q8_0(compressed, 64)
    np.testing.assert_array_equal(decompressed, 0.0)


def test_q8_0_padding():
    """Non-multiple-of-32 input should be padded."""
    data = np.ones(50, dtype=np.float32)
    compressed = _compress_q8_0(data)
    decompressed = _decompress_q8_0(compressed, 50)
    np.testing.assert_allclose(decompressed, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# PolarQuant (turbo3/turbo4) tests
# ---------------------------------------------------------------------------


def _random_kv_vectors(n_vectors: int, head_dim: int, seed: int = 123) -> np.ndarray:
    """Generate random KV-like vectors with realistic norms.

    Default seed must differ from the rotation matrix seed (42) to avoid
    pathological correlation where test vectors align with rotation eigendirections.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n_vectors, head_dim)).astype(np.float32)
    # Scale to realistic KV cache norms (~10-50)
    norms = rng.uniform(10, 50, size=(n_vectors, 1)).astype(np.float32)
    return vectors * norms


@pytest.mark.parametrize("bit_width", [2, 3, 4])
def test_polar_round_trip(bit_width):
    head_dim = 128
    vectors = _random_kv_vectors(16, head_dim)
    compressed = _polar_compress_vectors(vectors, bit_width, head_dim)
    decompressed = _polar_decompress_vectors(compressed, 16, bit_width, head_dim)

    # Relative error should be within expected bounds
    rel_error = np.linalg.norm(vectors - decompressed) / np.linalg.norm(vectors)
    # turbo2 (~6.5% PPL), turbo3 (~1.1% PPL), turbo4 (~0.23% PPL)
    max_errors = {2: 0.40, 3: 0.20, 4: 0.10}
    assert rel_error < max_errors[bit_width], f"Relative error {rel_error:.4f} too high for {bit_width}-bit"


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_polar_various_head_dims(head_dim):
    vectors = _random_kv_vectors(8, head_dim, seed=99)
    compressed = _polar_compress_vectors(vectors, 3, head_dim)
    decompressed = _polar_decompress_vectors(compressed, 8, 3, head_dim)
    rel_error = np.linalg.norm(vectors - decompressed) / np.linalg.norm(vectors)
    assert rel_error < 0.25


def test_polar_zero_vector():
    """Zero vectors should compress/decompress to zero."""
    vectors = np.zeros((4, 128), dtype=np.float32)
    compressed = _polar_compress_vectors(vectors, 3, 128)
    decompressed = _polar_decompress_vectors(compressed, 4, 3, 128)
    np.testing.assert_allclose(decompressed, 0.0, atol=1e-6)


def test_polar_deterministic():
    vectors = _random_kv_vectors(8, 128)
    c1 = _polar_compress_vectors(vectors, 3, 128, seed=42)
    c2 = _polar_compress_vectors(vectors, 3, 128, seed=42)
    assert c1 == c2


def test_polar_different_seeds_differ():
    vectors = _random_kv_vectors(8, 128)
    c1 = _polar_compress_vectors(vectors, 3, 128, seed=42)
    c2 = _polar_compress_vectors(vectors, 3, 128, seed=99)
    assert c1 != c2


# ---------------------------------------------------------------------------
# CompressionPipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_symmetric_turbo3():
    pipe = CompressionPipeline(seed=42)
    keys = _random_kv_vectors(16, 128)
    values = _random_kv_vectors(16, 128, seed=99)

    compressed = pipe.compress_kv(keys, values, Format.TURBO3, Format.TURBO3, head_dim=128)
    k_out, v_out = pipe.decompress_kv(compressed)

    k_out = k_out.reshape(keys.shape)
    v_out = v_out.reshape(values.shape)

    k_err = np.linalg.norm(keys - k_out) / np.linalg.norm(keys)
    v_err = np.linalg.norm(values - v_out) / np.linalg.norm(values)
    assert k_err < 0.20
    assert v_err < 0.20


def test_pipeline_asymmetric_q8_0_turbo3():
    """Default config: q8_0 K + turbo3 V."""
    pipe = CompressionPipeline(seed=42)
    keys = _random_kv_vectors(16, 128)
    values = _random_kv_vectors(16, 128, seed=99)

    compressed = pipe.compress_kv(keys, values, Format.Q8_0, Format.TURBO3, head_dim=128)
    assert compressed.fmt_k == Format.Q8_0
    assert compressed.fmt_v == Format.TURBO3

    k_out, v_out = pipe.decompress_kv(compressed)
    k_out = k_out.reshape(keys.shape)
    v_out = v_out.reshape(values.shape)

    # K with q8_0 should be very accurate
    k_err = np.linalg.norm(keys - k_out) / np.linalg.norm(keys)
    assert k_err < 0.01  # q8_0 is very precise

    # V with turbo3 should be within tolerance
    v_err = np.linalg.norm(values - v_out) / np.linalg.norm(values)
    assert v_err < 0.20


def test_pipeline_fp16():
    pipe = CompressionPipeline()
    keys = _random_kv_vectors(8, 128)
    values = _random_kv_vectors(8, 128, seed=99)

    compressed = pipe.compress_kv(keys, values, Format.FP16, Format.FP16, head_dim=128)
    k_out, v_out = pipe.decompress_kv(compressed)

    # FP16 round-trip should be very close
    np.testing.assert_allclose(keys.ravel(), k_out, rtol=1e-3)
    np.testing.assert_allclose(values.ravel(), v_out, rtol=1e-3)


def test_pipeline_symmetric_q8_0():
    pipe = CompressionPipeline()
    keys = _random_kv_vectors(8, 128)
    values = _random_kv_vectors(8, 128, seed=99)

    compressed = pipe.compress_kv(keys, values, Format.Q8_0, Format.Q8_0, head_dim=128)
    k_out, v_out = pipe.decompress_kv(compressed)
    k_out = k_out.reshape(keys.shape)
    v_out = v_out.reshape(values.shape)

    k_err = np.linalg.norm(keys - k_out) / np.linalg.norm(keys)
    v_err = np.linalg.norm(values - v_out) / np.linalg.norm(values)
    assert k_err < 0.01
    assert v_err < 0.01
