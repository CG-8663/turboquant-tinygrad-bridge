"""CUDA kernel manager for PolarQuant compress/decompress.

Compiles polar_quant.cu via tinygrad's NV device compiler and manages
GPU-resident rotation matrices and codebooks. Eliminates CPU round-trips
by running the entire compress/decompress pipeline on the GPU.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np

from tqbridge.compression import _get_rotation, _get_codebook
from tqbridge.wire import Format

_FORMAT_BITS = {
    Format.TURBO2: 2,
    Format.TURBO3: 3,
    Format.TURBO4: 4,
}

_KERNEL_SRC = (Path(__file__).parent / "polar_quant.cu").read_text()


class CUDAKernelError(RuntimeError):
    """Failed to compile or launch CUDA kernel."""
    pass


class CUDACompressor:
    """GPU-native PolarQuant compression via custom CUDA kernels.

    Keeps rotation matrices and codebooks in GPU VRAM. Compress and decompress
    operate entirely on-device with no CPU round-trip.

    Usage:
        compressor = CUDACompressor(head_dim=128, seed=42)
        norms, indices = compressor.compress(gpu_tensor, Format.TURBO3)
        result = compressor.decompress(norms, indices, Format.TURBO3)
        compressor.close()
    """

    def __init__(self, head_dim: int = 128, seed: int = 42, device: str = "NV"):
        from tinygrad import Tensor, Device

        self.head_dim = head_dim
        self.seed = seed
        self.device = device

        # Compile CUDA kernels
        try:
            dev = Device[device]
            self._compiler = dev.compiler
            self._lib = self._compiler.compile(_KERNEL_SRC)
            self._dev = dev
        except Exception as e:
            raise CUDAKernelError(f"Failed to compile CUDA kernels: {e}") from e

        # Cache GPU-resident rotation matrices and codebooks per format
        self._gpu_rotation: dict[int, Tensor] = {}
        self._gpu_codebook: dict[int, Tensor] = {}
        self._gpu_boundaries: dict[int, Tensor] = {}

    def _ensure_tables(self, fmt: Format):
        """Lazily load rotation matrix and codebook to GPU for a format."""
        from tinygrad import Tensor

        bit_width = _FORMAT_BITS[fmt]
        if bit_width in self._gpu_rotation:
            return

        rotation = _get_rotation(self.head_dim, self.seed).astype(np.float32)
        codebook = _get_codebook(bit_width, self.head_dim).astype(np.float32)
        boundaries = ((codebook[:-1] + codebook[1:]) / 2.0).astype(np.float32)

        self._gpu_rotation[bit_width] = Tensor(rotation, device=self.device).realize()
        self._gpu_codebook[bit_width] = Tensor(codebook, device=self.device).realize()
        self._gpu_boundaries[bit_width] = Tensor(boundaries, device=self.device).realize()

    def compress(self, vectors, fmt: Format = Format.TURBO3):
        """Compress float32 vectors on GPU using CUDA kernels.

        Args:
            vectors: tinygrad Tensor shape (n_vectors, head_dim) on NV device
            fmt: compression format (TURBO2/3/4)

        Returns:
            (norms, indices) — norms: (n_vectors,) float32 on device,
                                indices: (n_vectors, head_dim) uint8 on device
        """
        from tinygrad import Tensor, dtypes

        self._ensure_tables(fmt)
        bit_width = _FORMAT_BITS[fmt]

        n_vectors = vectors.shape[0]
        rotation = self._gpu_rotation[bit_width]
        boundaries = self._gpu_boundaries[bit_width]
        n_boundaries = boundaries.shape[0]

        # Allocate output buffers
        norms = Tensor.empty(n_vectors, device=self.device).realize()
        indices = Tensor.empty(n_vectors, self.head_dim, device=self.device).cast(dtypes.uint8).realize()

        # Launch kernel
        block = 256
        grid = (n_vectors + block - 1) // block

        self._launch("polar_compress_kernel", grid, block,
                      vectors, rotation, boundaries, norms, indices,
                      n_vectors, self.head_dim, n_boundaries)

        return norms, indices

    def decompress(self, norms, indices, fmt: Format = Format.TURBO3):
        """Decompress PolarQuant vectors on GPU using CUDA kernels.

        Args:
            norms: tinygrad Tensor (n_vectors,) float32 on device
            indices: tinygrad Tensor (n_vectors, head_dim) uint8 on device
            fmt: compression format

        Returns:
            Reconstructed vectors: (n_vectors, head_dim) float32 on device
        """
        from tinygrad import Tensor

        self._ensure_tables(fmt)
        bit_width = _FORMAT_BITS[fmt]

        n_vectors = norms.shape[0]
        rotation = self._gpu_rotation[bit_width]
        codebook = self._gpu_codebook[bit_width]

        output = Tensor.empty(n_vectors, self.head_dim, device=self.device).realize()

        block = 256
        grid = (n_vectors + block - 1) // block

        self._launch("polar_decompress_kernel", grid, block,
                      norms, indices, rotation, codebook, output,
                      n_vectors, self.head_dim)

        return output

    def _launch(self, kernel_name: str, grid: int, block: int, *args):
        """Launch a compiled CUDA kernel with the given arguments.

        Tensor args are passed as device pointers; int args as values.
        """
        from tinygrad import Tensor

        # Extract raw buffer pointers from tinygrad tensors
        c_args = []
        for a in args:
            if isinstance(a, Tensor):
                # Get the underlying buffer pointer
                a.realize()
                buf = a.lazydata.buffer
                buf.ensure_allocated()
                c_args.append(buf._buf)
            elif isinstance(a, int):
                c_args.append(a)
            else:
                raise TypeError(f"Unsupported kernel arg type: {type(a)}")

        # Use the device's program runner
        from tinygrad.runtime.ops_nv import NVProgram
        prg = NVProgram(self._dev, kernel_name, self._lib)
        prg(c_args, global_size=[grid * block, 1, 1], local_size=[block, 1, 1])

    def close(self):
        """Release GPU resources."""
        self._gpu_rotation.clear()
        self._gpu_codebook.clear()
        self._gpu_boundaries.clear()

    def __del__(self):
        self.close()
