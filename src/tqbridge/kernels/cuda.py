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

_KERNEL_SRC_TEMPLATE = (Path(__file__).parent / "polar_quant.cu").read_text()


class CUDAKernelError(RuntimeError):
    """Failed to compile or launch CUDA kernel."""
    pass


# GPU architecture families
_ARCH_NAMES = {
    "sm_61": "Pascal (GTX 10xx)",
    "sm_75": "Turing (RTX 20xx)",
    "sm_86": "Ampere (RTX 30xx)",
    "sm_89": "Ada Lovelace (RTX 40xx)",
    "sm_120": "Blackwell (RTX 50xx / PRO 6000)",
    "sm_121": "Blackwell (GB10)",
}


class CUDACompressor:
    """GPU-native PolarQuant compression via custom CUDA kernels.

    Keeps rotation matrices and codebooks in GPU VRAM. Compress and decompress
    operate entirely on-device with no CPU round-trip.

    Supports all NVIDIA architectures from Pascal (sm_61, GTX 1060) through
    Blackwell (sm_120, RTX PRO 6000). Kernels use only basic CUDA features
    (__shared__, __syncthreads__, sqrtf) — no warp shuffles or tensor cores.

    Usage:
        compressor = CUDACompressor(head_dim=128, seed=42)
        norms, indices = compressor.compress(gpu_tensor, Format.TURBO3)
        result = compressor.decompress(norms, indices, Format.TURBO3)
        compressor.close()
    """

    # Minimum supported architecture
    MIN_SM = 61  # Pascal

    def __init__(self, head_dim: int = 128, seed: int = 42, device: str = "NV"):
        from tinygrad import Tensor, Device

        self.head_dim = head_dim
        self.seed = seed
        self.device = device

        try:
            self._dev = Device[device]
        except Exception as e:
            raise CUDAKernelError(f"Failed to access device {device}: {e}") from e

        # Detect and report GPU architecture
        self.arch = self._dev.compiler.arch
        arch_name = _ARCH_NAMES.get(self.arch, "Unknown")
        sm_num = int(self.arch.replace("sm_", "")) if self.arch.startswith("sm_") else 0
        if sm_num < self.MIN_SM:
            raise CUDAKernelError(
                f"GPU architecture {self.arch} ({arch_name}) is below minimum "
                f"sm_{self.MIN_SM}. TQBridge kernels require Pascal (GTX 10xx) or newer."
            )

        # Cache compiled libs, programs, GPU-resident tables, and pre-allocated buffers
        self._libs: dict[int, bytes] = {}
        self._programs: dict[str, object] = {}
        self._gpu_rotation: dict[int, Tensor] = {}
        self._gpu_codebook: dict[int, Tensor] = {}
        self._gpu_boundaries: dict[int, Tensor] = {}
        self._prealloc: dict[tuple, dict] = {}

    def _compile_for_format(self, bit_width: int) -> bytes:
        """Compile kernels with HEAD_DIM and N_BOUNDARIES baked in."""
        if bit_width in self._libs:
            return self._libs[bit_width]

        n_boundaries = (1 << bit_width) - 1
        src = f"#define HEAD_DIM {self.head_dim}\n#define N_BOUNDARIES {n_boundaries}\n" + _KERNEL_SRC_TEMPLATE

        try:
            lib = self._dev.compiler.compile(src)
        except Exception as e:
            raise CUDAKernelError(f"Failed to compile CUDA kernels: {e}") from e

        self._libs[bit_width] = lib
        return lib

    def _ensure_tables(self, fmt: Format):
        """Lazily compile kernels and load rotation/codebook to GPU for a format."""
        from tinygrad import Tensor

        bit_width = _FORMAT_BITS[fmt]
        if bit_width in self._gpu_rotation:
            return

        # Compile kernels for this format
        self._compile_for_format(bit_width)

        rotation = _get_rotation(self.head_dim, self.seed).astype(np.float32)
        codebook = _get_codebook(bit_width, self.head_dim).astype(np.float32)
        boundaries = ((codebook[:-1] + codebook[1:]) / 2.0).astype(np.float32)

        self._gpu_rotation[bit_width] = Tensor(rotation, device=self.device).realize()
        self._gpu_codebook[bit_width] = Tensor(codebook, device=self.device).realize()
        self._gpu_boundaries[bit_width] = Tensor(boundaries, device=self.device).realize()

    def preallocate(self, n_vectors: int, fmt: Format = Format.TURBO3):
        """Pre-allocate all GPU buffers for a fixed vector count.

        Call once at init to eliminate per-token allocation overhead.
        Each Tensor.empty().realize() costs ~1ms over the eGPU link.
        Pre-allocating removes 4+ sync round-trips per token.

        Args:
            n_vectors: number of vectors (e.g. n_layers * n_kv_heads * seq_len)
            fmt: compression format
        """
        from tinygrad import Tensor, dtypes

        self._ensure_tables(fmt)
        bit_width = _FORMAT_BITS[fmt]
        key = (bit_width, n_vectors)

        if key in self._prealloc:
            return

        norms = Tensor.empty(n_vectors, device=self.device).realize()
        indices = Tensor.empty(n_vectors, self.head_dim, device=self.device).cast(dtypes.uint8).realize()
        output = Tensor.empty(n_vectors, self.head_dim, device=self.device).realize()
        params = Tensor(np.array([n_vectors], dtype=np.int32), device=self.device).realize()

        self._prealloc[key] = {
            "norms": norms, "indices": indices,
            "output": output, "params": params,
        }

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
        key = (bit_width, n_vectors)

        rotation = self._gpu_rotation[bit_width]
        boundaries = self._gpu_boundaries[bit_width]

        # Use pre-allocated buffers if available, else allocate
        if key in self._prealloc:
            norms = self._prealloc[key]["norms"]
            indices = self._prealloc[key]["indices"]
            params = self._prealloc[key]["params"]
        else:
            norms = Tensor.empty(n_vectors, device=self.device).realize()
            indices = Tensor.empty(n_vectors, self.head_dim, device=self.device).cast(dtypes.uint8).realize()
            params = Tensor(np.array([n_vectors], dtype=np.int32), device=self.device).realize()

        lib = self._libs[bit_width]
        self._launch("polar_compress_kernel", lib, n_vectors, self.head_dim,
                      vectors, rotation, boundaries, norms, indices, params)

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
        key = (bit_width, n_vectors)

        rotation = self._gpu_rotation[bit_width]
        codebook = self._gpu_codebook[bit_width]

        if key in self._prealloc:
            output = self._prealloc[key]["output"]
            params = self._prealloc[key]["params"]
        else:
            output = Tensor.empty(n_vectors, self.head_dim, device=self.device).realize()
            params = Tensor(np.array([n_vectors], dtype=np.int32), device=self.device).realize()

        lib = self._libs[bit_width]
        self._launch("polar_decompress_kernel", lib, n_vectors, self.head_dim,
                      norms, indices, rotation, codebook, output, params)

        return output

    def _launch(self, kernel_name: str, lib: bytes, grid: int, block: int, *tensor_args):
        """Launch a compiled CUDA kernel with the given tensor arguments.

        All arguments must be tinygrad Tensors (passed as device buffers).
        Scalar parameters must be packed into a Tensor buffer by the caller.
        """
        from tinygrad import Tensor
        from tinygrad.runtime.ops_nv import NVProgram

        bufs = []
        for a in tensor_args:
            a.realize()
            bufs.append(a._buffer()._buf)

        cache_key = (kernel_name, id(lib))
        if cache_key not in self._programs:
            self._programs[cache_key] = NVProgram(self._dev, kernel_name, lib)

        prg = self._programs[cache_key]
        prg(*bufs, global_size=(grid, 1, 1), local_size=(block, 1, 1))

    def close(self):
        """Release GPU resources."""
        for attr in ("_gpu_rotation", "_gpu_codebook", "_gpu_boundaries"):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.clear()

    def __del__(self):
        self.close()
