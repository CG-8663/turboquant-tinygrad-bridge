"""Tests for CUDA PolarQuant kernels (kernels/cuda.py).

Tests require NV device with Docker-based nvcc compilation.
"""

from __future__ import annotations

import numpy as np
import pytest

tinygrad = pytest.importorskip("tinygrad")
from tinygrad import Tensor, Device

try:
    Device["NV"]
    _has_nv = True
except Exception:
    _has_nv = False

try:
    from tqbridge.kernels.cuda import CUDACompressor, CUDAKernelError
    if _has_nv:
        # Test compilation
        _comp = CUDACompressor(head_dim=128, seed=42)
        _comp.close()
        _has_cuda_kernels = True
    else:
        _has_cuda_kernels = False
except Exception:
    _has_cuda_kernels = False

cuda_kernels = pytest.mark.skipif(
    not _has_cuda_kernels,
    reason="Requires NV device with CUDA kernel compilation (Docker + nvcc)",
)


@pytest.fixture
def compressor():
    c = CUDACompressor(head_dim=128, seed=42)
    yield c
    c.close()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@cuda_kernels
class TestCUDACompress:

    def test_compress_shape(self, compressor, rng):
        data = rng.standard_normal((16, 128)).astype(np.float32)
        vectors = Tensor(data, device="NV").realize()

        from tqbridge.wire import Format
        norms, indices = compressor.compress(vectors, Format.TURBO3)

        assert norms.shape == (16,)
        assert indices.shape == (16, 128)

    def test_compress_norms_positive(self, compressor, rng):
        data = rng.standard_normal((8, 128)).astype(np.float32)
        vectors = Tensor(data, device="NV").realize()

        from tqbridge.wire import Format
        norms, _ = compressor.compress(vectors, Format.TURBO3)
        norms_np = norms.numpy()

        assert np.all(norms_np >= 0), "Norms must be non-negative"

    def test_compress_indices_in_range(self, compressor, rng):
        data = rng.standard_normal((8, 128)).astype(np.float32)
        vectors = Tensor(data, device="NV").realize()

        from tqbridge.wire import Format
        _, indices = compressor.compress(vectors, Format.TURBO3)
        idx_np = indices.numpy()

        # turbo3 = 3 bits = 8 centroids, indices 0-7
        assert np.all(idx_np >= 0)
        assert np.all(idx_np <= 7), f"Max index: {idx_np.max()}"


@cuda_kernels
class TestCUDADecompress:

    def test_round_trip_turbo3(self, compressor, rng):
        data = rng.standard_normal((16, 128)).astype(np.float32)
        vectors = Tensor(data, device="NV").realize()

        from tqbridge.wire import Format
        norms, indices = compressor.compress(vectors, Format.TURBO3)
        result = compressor.decompress(norms, indices, Format.TURBO3)

        result_np = result.numpy()
        mse = np.mean((result_np - data) ** 2)
        assert mse < 0.5, f"turbo3 MSE too high: {mse}"

    def test_round_trip_turbo4(self, rng):
        from tqbridge.wire import Format
        comp = CUDACompressor(head_dim=128, seed=42)

        data = rng.standard_normal((8, 128)).astype(np.float32)
        vectors = Tensor(data, device="NV").realize()

        norms, indices = comp.compress(vectors, Format.TURBO4)
        result = comp.decompress(norms, indices, Format.TURBO4)

        mse = np.mean((result.numpy() - data) ** 2)
        assert mse < 0.3, f"turbo4 MSE too high: {mse}"
        comp.close()

    def test_round_trip_turbo2(self, rng):
        from tqbridge.wire import Format
        comp = CUDACompressor(head_dim=128, seed=42)

        data = rng.standard_normal((8, 128)).astype(np.float32)
        vectors = Tensor(data, device="NV").realize()

        norms, indices = comp.compress(vectors, Format.TURBO2)
        result = comp.decompress(norms, indices, Format.TURBO2)

        mse = np.mean((result.numpy() - data) ** 2)
        assert mse < 1.0, f"turbo2 MSE too high: {mse}"
        comp.close()

    def test_zero_vector(self, compressor):
        from tqbridge.wire import Format
        data = np.zeros((1, 128), dtype=np.float32)
        vectors = Tensor(data, device="NV").realize()

        norms, indices = compressor.compress(vectors, Format.TURBO3)
        result = compressor.decompress(norms, indices, Format.TURBO3)

        assert np.max(np.abs(result.numpy())) < 1e-6


@cuda_kernels
class TestCUDACrossValidation:

    def test_matches_tinygrad_compressor(self, rng):
        """CUDA and tinygrad backends should produce similar quality."""
        from tqbridge.compression_tg import TinygradCompressor
        from tqbridge.wire import Format

        data = rng.standard_normal((32, 128)).astype(np.float32)

        # CUDA on NV
        cuda_comp = CUDACompressor(head_dim=128, seed=42)
        nv_vectors = Tensor(data, device="NV").realize()
        norms, indices = cuda_comp.compress(nv_vectors, Format.TURBO3)
        cuda_result = cuda_comp.decompress(norms, indices, Format.TURBO3).numpy()
        cuda_mse = np.mean((cuda_result - data) ** 2)
        cuda_comp.close()

        # tinygrad on Metal
        tg_comp = TinygradCompressor(head_dim=128, seed=42)
        metal_vectors = Tensor(data, device="METAL").realize()
        comp = tg_comp.compress(metal_vectors, Format.TURBO3)
        tg_result = tg_comp.decompress(comp).numpy()
        tg_mse = np.mean((tg_result - data) ** 2)

        # Both should be in the same ballpark
        assert abs(cuda_mse - tg_mse) < 0.1, (
            f"CUDA MSE ({cuda_mse:.6f}) too far from tinygrad ({tg_mse:.6f})"
        )
