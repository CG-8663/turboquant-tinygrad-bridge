"""Tests for native C bridge bindings (tqbridge.native).

Cross-validates NativeCompressor against the Python NumPy reference
implementation (compression.py) to ensure C and Python produce
compatible results.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from tqbridge.native import (
        NativeCompressor,
        NativeBridge,
        NativeError,
        native_encode_header,
        native_decode_header,
        _find_library,
    )
    HAS_NATIVE = _find_library() is not None
except ImportError:
    HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not HAS_NATIVE, reason="libtqbridge not built")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def compressor():
    c = NativeCompressor(head_dim=128, seed=42)
    yield c
    c.close()


@pytest.fixture
def vectors_128(rng):
    """16 random vectors of dim 128."""
    return rng.standard_normal((16, 128)).astype(np.float32)


@pytest.fixture
def vectors_small(rng):
    """4 random vectors of dim 128."""
    return rng.standard_normal((4, 128)).astype(np.float32)


# ---------------------------------------------------------------------------
# NativeBridge low-level tests
# ---------------------------------------------------------------------------

class TestNativeBridge:

    def test_init_and_close(self):
        from tqbridge.wire import Format
        bridge = NativeBridge(128, Format.TURBO3, 42)
        bridge.close()

    def test_compress_decompress_round_trip(self, vectors_128):
        from tqbridge.wire import Format
        with NativeBridge(128, Format.TURBO3, 42) as bridge:
            compressed_bytes, comp = bridge.compress(vectors_128)
            assert len(compressed_bytes) > 0
            assert comp.n_vectors == 16
            assert comp.head_dim == 128

            result = bridge.decompress(comp)
            assert result.shape == (16, 128)
            assert result.dtype == np.float32

            mse = np.mean((result - vectors_128) ** 2)
            assert mse < 0.5, f"turbo3 MSE too high: {mse}"

            bridge.free_compressed(comp)

    def test_context_manager(self, vectors_small):
        from tqbridge.wire import Format
        with NativeBridge(128, Format.TURBO3, 42) as bridge:
            _, comp = bridge.compress(vectors_small)
            bridge.decompress(comp)
            bridge.free_compressed(comp)


# ---------------------------------------------------------------------------
# NativeCompressor tests (high-level API)
# ---------------------------------------------------------------------------

class TestNativeCompressor:

    def test_turbo3_round_trip(self, compressor, vectors_128):
        from tqbridge.wire import Format
        compressed = compressor.compress(vectors_128, Format.TURBO3)
        result = compressor.decompress(compressed)

        assert result.shape == vectors_128.shape
        mse = np.mean((result - vectors_128) ** 2)
        assert mse < 0.5, f"turbo3 MSE: {mse}"

    def test_turbo2_round_trip(self, compressor, vectors_128):
        from tqbridge.wire import Format
        compressed = compressor.compress(vectors_128, Format.TURBO2)
        result = compressor.decompress(compressed)

        assert result.shape == vectors_128.shape
        mse = np.mean((result - vectors_128) ** 2)
        assert mse < 0.5, f"turbo2 MSE: {mse}"

    def test_turbo4_round_trip(self, compressor, vectors_128):
        from tqbridge.wire import Format
        compressed = compressor.compress(vectors_128, Format.TURBO4)
        result = compressor.decompress(compressed)

        assert result.shape == vectors_128.shape
        mse = np.mean((result - vectors_128) ** 2)
        assert mse < 0.3, f"turbo4 MSE: {mse}"

    def test_q8_0_round_trip(self, compressor, vectors_128):
        from tqbridge.wire import Format
        flat = vectors_128.reshape(-1)
        compressed = compressor.compress(flat, Format.Q8_0)
        result = compressor.decompress(compressed)

        assert result.shape == flat.shape
        mse = np.mean((result - flat) ** 2)
        assert mse < 1e-3, f"q8_0 MSE: {mse}"

    def test_fp16_round_trip(self, compressor, vectors_128):
        from tqbridge.wire import Format
        compressed = compressor.compress(vectors_128, Format.FP16)
        result = compressor.decompress(compressed)

        assert result.shape == vectors_128.shape
        np.testing.assert_allclose(result, vectors_128, atol=1e-3)

    def test_compressed_size(self, compressor, vectors_128):
        from tqbridge.wire import Format
        compressed = compressor.compress(vectors_128, Format.TURBO3)
        size = compressor.compressed_size_bytes(compressed)
        # turbo3: 16 vectors × (4 bytes norm + 48 bytes packed) = 16 × 52 = 832
        assert size == 16 * 52

    def test_zero_vector(self, compressor):
        from tqbridge.wire import Format
        zeros = np.zeros((1, 128), dtype=np.float32)
        compressed = compressor.compress(zeros, Format.TURBO3)
        result = compressor.decompress(compressed)
        assert np.max(np.abs(result)) < 1e-6

    def test_deterministic(self, compressor, vectors_small):
        from tqbridge.wire import Format
        c1 = compressor.compress(vectors_small, Format.TURBO3)
        bytes1 = c1["compressed_bytes"].copy()
        compressor.decompress(c1)

        c2 = compressor.compress(vectors_small, Format.TURBO3)
        bytes2 = c2["compressed_bytes"].copy()
        compressor.decompress(c2)

        np.testing.assert_array_equal(bytes1, bytes2)

    def test_3d_shape_preserved(self, compressor, rng):
        from tqbridge.wire import Format
        data = rng.standard_normal((4, 8, 128)).astype(np.float32)
        compressed = compressor.compress(data, Format.TURBO3)
        result = compressor.decompress(compressed)
        assert result.shape == (4, 8, 128)


# ---------------------------------------------------------------------------
# Cross-validation: Native vs Python oracle
# ---------------------------------------------------------------------------

class TestCrossValidation:

    def test_turbo3_native_vs_python(self, vectors_128):
        """Native and Python compressors should produce similar MSE."""
        from tqbridge.compression import (
            _polar_compress_vectors, _polar_decompress_vectors,
        )
        from tqbridge.wire import Format

        native = NativeCompressor(head_dim=128, seed=42)

        # Python reference: compress then decompress
        n_vectors = vectors_128.shape[0]
        py_compressed = _polar_compress_vectors(vectors_128, 3, 128, seed=42)
        py_result = _polar_decompress_vectors(py_compressed, n_vectors, 3, 128, seed=42)

        # Native C (now uses precomputed rotation/codebook from Python oracle)
        n_compressed = native.compress(vectors_128, Format.TURBO3)
        n_result = native.decompress(n_compressed)

        # Compressed bytes should be identical (same rotation, codebook, quantization)
        py_bytes = np.frombuffer(py_compressed, dtype=np.uint8)
        np.testing.assert_array_equal(
            py_bytes, n_compressed["compressed_bytes"],
            err_msg="Compressed bytes must be identical (shared rotation/codebook)"
        )

        # Decompressed should be near-identical (float rounding only)
        np.testing.assert_allclose(n_result, py_result, atol=1e-6)

        native.close()

    def test_q8_0_native_vs_python(self, vectors_128):
        """Q8_0 results should be very close between C and Python."""
        from tqbridge.compression import _compress_q8_0, _decompress_q8_0
        from tqbridge.wire import Format

        flat = vectors_128.reshape(-1)

        native = NativeCompressor(head_dim=128, seed=42)

        py_compressed = _compress_q8_0(flat)
        py_result = _decompress_q8_0(py_compressed, len(flat))

        n_compressed = native.compress(flat, Format.Q8_0)
        n_result = native.decompress(n_compressed)

        # Q8_0 should produce nearly identical results.
        # C and Python float16 scale conversions may differ at the LSB level,
        # causing ~0.002 max difference in dequantized values.
        np.testing.assert_allclose(n_result, py_result, atol=3e-3)

        native.close()


# ---------------------------------------------------------------------------
# Wire protocol native tests
# ---------------------------------------------------------------------------

class TestNativeWire:

    def test_header_round_trip(self):
        from tqbridge.wire import Format
        encoded = native_encode_header(
            fmt_k=Format.Q8_0, fmt_v=Format.TURBO3,
            n_layers=32, layer_start=0, seq_len=2048,
            n_heads_k=32, n_heads_v=8, head_dim=128,
            payload_bytes=1048576,
        )
        assert len(encoded) == 40

        decoded = native_decode_header(encoded)
        assert decoded["fmt_k"] == Format.Q8_0
        assert decoded["fmt_v"] == Format.TURBO3
        assert decoded["n_layers"] == 32
        assert decoded["seq_len"] == 2048
        assert decoded["n_heads_k"] == 32
        assert decoded["n_heads_v"] == 8
        assert decoded["head_dim"] == 128
        assert decoded["payload_bytes"] == 1048576

    def test_header_cross_validate_python(self):
        """Native and Python wire headers should be byte-identical."""
        from tqbridge.wire import Format, WireHeader, encode_header

        hdr = WireHeader(
            fmt_k=Format.Q8_0, fmt_v=Format.TURBO3,
            n_layers=8, layer_start=2, seq_len=512,
            n_heads_k=16, n_heads_v=4, head_dim=128,
            flags=0, payload_bytes=65536,
        )
        py_encoded = encode_header(hdr)

        native_encoded = native_encode_header(
            fmt_k=Format.Q8_0, fmt_v=Format.TURBO3,
            n_layers=8, layer_start=2, seq_len=512,
            n_heads_k=16, n_heads_v=4, head_dim=128,
            payload_bytes=65536,
        )

        assert py_encoded == native_encoded, "Wire headers must be byte-identical"

    def test_corrupted_header(self):
        from tqbridge.wire import Format
        encoded = bytearray(native_encode_header(
            fmt_k=Format.Q8_0, fmt_v=Format.TURBO3,
            n_layers=1, layer_start=0, seq_len=128,
            n_heads_k=4, n_heads_v=4, head_dim=64,
            payload_bytes=256,
        ))
        encoded[10] ^= 0xFF
        with pytest.raises(NativeError, match="CRC"):
            native_decode_header(bytes(encoded))
