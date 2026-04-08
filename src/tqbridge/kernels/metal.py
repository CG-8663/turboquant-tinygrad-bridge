"""Metal compute shader manager for PolarQuant compress/decompress.

Compiles polar_quant.metal via tinygrad's Metal compiler and manages
GPU-resident rotation matrices and codebooks on Apple Silicon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tqbridge.compression import _get_rotation, _get_codebook
from tqbridge.wire import Format

_FORMAT_BITS = {
    Format.TURBO2: 2,
    Format.TURBO3: 3,
    Format.TURBO4: 4,
}

# Inline Metal source — tinygrad's Metal compiler has a ~1500 char source limit.
# The .metal file is the readable version; this is the compiled version.
_KERNEL_SRC_TEMPLATE = """#include <metal_stdlib>
using namespace metal;
kernel void polar_compress_kernel(device const float*input[[buffer(0)]],device const float*rotation[[buffer(1)]],device const float*boundaries[[buffer(2)]],device float*norms_out[[buffer(3)]],device uchar*indices_out[[buffer(4)]],device const int*params[[buffer(5)]],uint vid[[threadgroup_position_in_grid]],uint tid[[thread_position_in_threadgroup]]){
int nv=params[0];if((int)vid>=nv)return;
device const float*vec=input+vid*HEAD_DIM;
threadgroup float sv[HEAD_DIM],sr[HEAD_DIM];
sv[tid]=vec[tid];threadgroup_barrier(mem_flags::mem_threadgroup);
sr[tid]=sv[tid]*sv[tid];threadgroup_barrier(mem_flags::mem_threadgroup);
for(int s=HEAD_DIM/2;s>0;s>>=1){if((int)tid<s)sr[tid]+=sr[tid+s];threadgroup_barrier(mem_flags::mem_threadgroup);}
float norm=sqrt(sr[0]),inv=(norm>0.0f)?(1.0f/norm):1.0f;
if(tid==0)norms_out[vid]=norm;
device const float*R=rotation+tid*HEAD_DIM;float y=0.0f;
for(int j=0;j<HEAD_DIM;j++)y+=R[j]*sv[j];y*=inv;
int idx=0;for(int b=0;b<N_BOUNDARIES;b++){if(y>=boundaries[b])idx=b+1;else break;}
indices_out[vid*HEAD_DIM+tid]=(uchar)idx;}
kernel void polar_decompress_kernel(device const float*norms[[buffer(0)]],device const uchar*indices[[buffer(1)]],device const float*rotation[[buffer(2)]],device const float*codebook[[buffer(3)]],device float*output[[buffer(4)]],device const int*params[[buffer(5)]],uint vid[[threadgroup_position_in_grid]],uint tid[[thread_position_in_threadgroup]]){
int nv=params[0];if((int)vid>=nv)return;
device const uchar*idx=indices+vid*HEAD_DIM;
threadgroup float sy[HEAD_DIM];sy[tid]=codebook[idx[tid]];threadgroup_barrier(mem_flags::mem_threadgroup);
float v=0.0f;for(int j=0;j<HEAD_DIM;j++)v+=rotation[j*HEAD_DIM+tid]*sy[j];
output[vid*HEAD_DIM+tid]=v*norms[vid];}
"""


class MetalKernelError(RuntimeError):
    pass


class MetalCompressor:
    """GPU-native PolarQuant compression via Metal compute shaders.

    Same architecture as CUDACompressor but for Apple Silicon.
    Pre-allocates buffers to eliminate per-token allocation overhead.
    """

    def __init__(self, head_dim: int = 128, seed: int = 42, device: str = "METAL"):
        from tinygrad import Tensor, Device

        self.head_dim = head_dim
        self.seed = seed
        self.device = device

        try:
            self._dev = Device[device]
        except Exception as e:
            raise MetalKernelError(f"Failed to access device {device}: {e}") from e

        self._libs: dict[int, bytes] = {}
        self._programs: dict[str, object] = {}
        self._gpu_rotation: dict[int, Tensor] = {}
        self._gpu_codebook: dict[int, Tensor] = {}
        self._gpu_boundaries: dict[int, Tensor] = {}
        self._prealloc: dict[tuple, dict] = {}

    def _compile_for_format(self, bit_width: int) -> bytes:
        if bit_width in self._libs:
            return self._libs[bit_width]

        n_boundaries = (1 << bit_width) - 1
        defines = f"#define HEAD_DIM {self.head_dim}\n#define N_BOUNDARIES {n_boundaries}\n"
        src = defines + _KERNEL_SRC_TEMPLATE

        try:
            lib = self._dev.compiler.compile(src)
        except Exception as e:
            raise MetalKernelError(f"Failed to compile Metal shaders: {e}") from e

        self._libs[bit_width] = lib
        return lib

    def _ensure_tables(self, fmt: Format):
        from tinygrad import Tensor

        bit_width = _FORMAT_BITS[fmt]
        if bit_width in self._gpu_rotation:
            return

        self._compile_for_format(bit_width)

        rotation = _get_rotation(self.head_dim, self.seed).astype(np.float32)
        codebook = _get_codebook(bit_width, self.head_dim).astype(np.float32)
        boundaries = ((codebook[:-1] + codebook[1:]) / 2.0).astype(np.float32)

        self._gpu_rotation[bit_width] = Tensor(rotation, device=self.device).realize()
        self._gpu_codebook[bit_width] = Tensor(codebook, device=self.device).realize()
        self._gpu_boundaries[bit_width] = Tensor(boundaries, device=self.device).realize()

    def preallocate(self, n_vectors: int, fmt: Format = Format.TURBO3):
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
        from tinygrad import Tensor, dtypes

        self._ensure_tables(fmt)
        bit_width = _FORMAT_BITS[fmt]
        n_vectors = vectors.shape[0]
        key = (bit_width, n_vectors)

        rotation = self._gpu_rotation[bit_width]
        boundaries = self._gpu_boundaries[bit_width]

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
        from tinygrad import Tensor
        from tinygrad.runtime.ops_metal import MetalProgram

        bufs = []
        for a in tensor_args:
            a.realize()
            bufs.append(a._buffer()._buf)

        cache_key = (kernel_name, id(lib))
        if cache_key not in self._programs:
            self._programs[cache_key] = MetalProgram(self._dev, kernel_name, lib)

        prg = self._programs[cache_key]
        prg(*bufs, global_size=(grid, 1, 1), local_size=(block, 1, 1))

    def close(self):
        self._gpu_rotation.clear()
        self._gpu_codebook.clear()
        self._gpu_boundaries.clear()
        self._prealloc.clear()

    def __del__(self):
        self.close()
