"""GPU detection and kernel backend selection.

Auto-detects available GPU backends and selects the best kernel path:
  NVIDIA (NV/CUDA) → polar_quant.cu (CUDA kernels)
  AMD (AMD/HIP)    → polar_quant.hip (HIP kernels, CUDA-compatible syntax)
  Intel (CL)       → polar_quant.cl (OpenCL kernels)
  Apple (METAL)    → polar_quant.metal (Metal compute shaders)
  CPU fallback     → native C library (tqbridge.c)

Supports discrete GPUs, eGPUs (Thunderbolt/USB4), integrated GPUs (iGPU),
and vGPU instances. Integrated GPUs share system memory with the CPU —
no DMA overhead, making them ideal for decode nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class GPUInfo:
    """Detected GPU information."""
    backend: Literal["cuda", "metal", "amd", "opencl", "cpu"]
    device_name: str
    device_id: str
    memory_mb: int
    is_integrated: bool  # iGPU shares system memory
    is_vgpu: bool        # virtual GPU instance
    compute_capability: str  # e.g. "sm_61", "gfx1100", "gen12"


def detect_gpus() -> list[GPUInfo]:
    """Detect all available GPU backends.

    Returns list of GPUInfo, ordered by preference (fastest first).
    """
    gpus = []

    # NVIDIA (NV backend — tinygrad's HCQ path)
    try:
        from tinygrad import Device
        dev = Device["NV"]
        arch = dev.compiler.arch if hasattr(dev, 'compiler') else "unknown"
        gpus.append(GPUInfo(
            backend="cuda", device_name=f"NVIDIA ({arch})",
            device_id="NV", memory_mb=0,
            is_integrated=False, is_vgpu=False,
            compute_capability=arch,
        ))
    except Exception:
        pass

    # NVIDIA (CUDA backend — standard CUDA driver)
    if not any(g.backend == "cuda" for g in gpus):
        try:
            from tinygrad import Device
            dev = Device["CUDA"]
            arch = dev.compiler.arch if hasattr(dev, 'compiler') else "unknown"
            gpus.append(GPUInfo(
                backend="cuda", device_name=f"NVIDIA CUDA ({arch})",
                device_id="CUDA", memory_mb=0,
                is_integrated=False, is_vgpu=False,
                compute_capability=arch,
            ))
        except Exception:
            pass

    # Apple Metal
    try:
        from tinygrad import Device
        dev = Device["METAL"]
        gpus.append(GPUInfo(
            backend="metal", device_name="Apple Metal",
            device_id="METAL", memory_mb=0,
            is_integrated=True, is_vgpu=False,  # unified memory = integrated
            compute_capability="metal",
        ))
    except Exception:
        pass

    # AMD (HIP or AMD backend)
    for dev_name in ["AMD", "HIP"]:
        try:
            from tinygrad import Device
            dev = Device[dev_name]
            arch = dev.compiler.arch if hasattr(dev, 'compiler') else "unknown"
            is_igpu = "vega" in arch.lower() or "renoir" in arch.lower() or "cezanne" in arch.lower()
            gpus.append(GPUInfo(
                backend="amd", device_name=f"AMD ({arch})",
                device_id=dev_name, memory_mb=0,
                is_integrated=is_igpu, is_vgpu=False,
                compute_capability=arch,
            ))
            break
        except Exception:
            pass

    # OpenCL (Intel Arc, Intel UHD/Iris, AMD fallback, CPU OpenCL)
    try:
        from tinygrad import Device
        dev = Device["CL"]
        gpus.append(GPUInfo(
            backend="opencl", device_name="OpenCL device",
            device_id="CL", memory_mb=0,
            is_integrated=True, is_vgpu=False,  # assume iGPU for CL
            compute_capability="opencl",
        ))
    except Exception:
        pass

    # CPU fallback is always available
    gpus.append(GPUInfo(
        backend="cpu", device_name="CPU (native C)",
        device_id="CPU", memory_mb=0,
        is_integrated=True, is_vgpu=False,
        compute_capability="c11",
    ))

    return gpus


def select_best_backend(gpus: list[GPUInfo] | None = None) -> GPUInfo:
    """Select the best available GPU backend for TQBridge.

    Priority: CUDA > Metal > AMD > OpenCL > CPU
    Integrated GPUs (iGPU) are preferred for decode nodes
    since they share system memory (zero-copy DMA).
    """
    if gpus is None:
        gpus = detect_gpus()

    if not gpus:
        raise RuntimeError("No GPU backends available")

    # Priority order
    priority = {"cuda": 0, "metal": 1, "amd": 2, "opencl": 3, "cpu": 4}
    gpus.sort(key=lambda g: priority.get(g.backend, 99))
    return gpus[0]


def print_gpu_report():
    """Print detected GPU backends."""
    gpus = detect_gpus()
    print(f"Detected {len(gpus)} GPU backend(s):\n")
    for i, gpu in enumerate(gpus):
        marker = "→" if i == 0 else " "
        igpu = " (iGPU, shared mem)" if gpu.is_integrated else ""
        vgpu = " (vGPU)" if gpu.is_vgpu else ""
        print(f"  {marker} {gpu.backend:8s} {gpu.device_name}{igpu}{vgpu}")

    best = select_best_backend(gpus)
    print(f"\n  Selected: {best.backend} ({best.device_name})")


if __name__ == "__main__":
    print_gpu_report()
