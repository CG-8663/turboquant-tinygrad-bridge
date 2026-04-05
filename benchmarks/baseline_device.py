"""Device baselines: raw compute + memory bandwidth on NV and Metal backends.

Measures per-device performance independent of the bridge, so we can later
show the cost/benefit of cross-device transfer with compression.

Usage:
    PYTHONPATH=./tinygrad python3.13 benchmarks/baseline_device.py --device NV
    PYTHONPATH=./tinygrad python3.13 benchmarks/baseline_device.py --device METAL
    PYTHONPATH=./tinygrad python3.13 benchmarks/baseline_device.py --device NV,METAL
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Memory budget: reserve 20% of device memory to avoid driver crashes
# ---------------------------------------------------------------------------
MEMORY_RESERVE = 0.20

def _get_total_memory_bytes(device_name: str) -> int:
    if device_name == "METAL" or (device_name == "NV" and platform.system() == "Darwin"):
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        except Exception:
            return 0
    if device_name == "NV":
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True,
            )
            return int(out.strip().split("\n")[0]) * 1024 * 1024
        except Exception:
            return 0
    return 0

def get_memory_budget(device_name: str) -> int:
    total = _get_total_memory_bytes(device_name)
    if total == 0:
        return 0
    budget = int(total * (1.0 - MEMORY_RESERVE))
    print(f"  Memory budget: {budget / (1024**3):.1f} GB "
          f"({100*(1-MEMORY_RESERVE):.0f}% of {total / (1024**3):.1f} GB)")
    return budget


def _alloc_fits(nbytes: int, budget: int) -> bool:
    """Check if an allocation fits within the memory budget."""
    if budget <= 0:
        return True  # unknown budget — optimistically allow
    return nbytes <= budget


@dataclass
class DeviceResult:
    device: str
    # Memory bandwidth (GB/s) — large tensor copy
    membw_gbs: float
    # Matmul throughput (GFLOPS) — square matmul
    matmul_gflops: float
    # KV cache simulation: time to allocate + fill a realistic KV cache
    kv_alloc_ms: float
    kv_fill_ms: float
    kv_read_ms: float
    kv_size_mb: float
    # Elementwise throughput (GB/s) — add two large tensors
    elemwise_gbs: float


def benchmark_device(device_name: str, warmup: int = 3, trials: int = 10) -> DeviceResult:
    """Run all baselines on a single device."""
    import os
    os.environ["DEV"] = device_name

    from tinygrad import Device, Tensor, dtypes

    budget = get_memory_budget(device_name)

    def _sync():
        """Force device synchronization to prevent command buffer accumulation."""
        Device[device_name].synchronize()

    print(f"\n{'='*60}")
    print(f"  Baselines: {device_name} ({Device.DEFAULT})")
    print(f"{'='*60}")

    # --- Memory bandwidth: copy 256 MB ---
    print("\n[1/4] Memory bandwidth (256 MB copy)...")
    size = 256 * 1024 * 1024 // 4  # 256 MB of float32
    a = Tensor.rand(size).realize()

    for _ in range(warmup):
        b = (a + 0).realize()
    _sync()  # flush warmup command buffers before timed trials

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        b = (a + 0).realize()
        _ = b.numpy()  # force sync
        t1 = time.perf_counter()
        times.append(t1 - t0)

    copy_bytes = size * 4 * 2  # read + write
    membw_median = copy_bytes / np.median(times) / 1e9
    print(f"  Median: {membw_median:.1f} GB/s ({np.median(times)*1000:.2f} ms)")

    # --- Matmul throughput: 2048x2048 ---
    print("\n[2/4] Matmul throughput (2048x2048 fp32)...")
    N = 2048
    x = Tensor.rand(N, N).realize()
    y = Tensor.rand(N, N).realize()

    for _ in range(warmup):
        z = (x @ y).realize()
    _sync()  # flush warmup command buffers before timed trials

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        z = (x @ y).realize()
        _ = z.numpy()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    flops = 2 * N * N * N  # 2*N^3 for matmul
    matmul_gflops = flops / np.median(times) / 1e9
    print(f"  Median: {matmul_gflops:.1f} GFLOPS ({np.median(times)*1000:.2f} ms)")

    # --- Elementwise throughput: add 256 MB ---
    print("\n[3/4] Elementwise add (256 MB + 256 MB)...")
    size_elem = 256 * 1024 * 1024 // 4
    a2 = Tensor.rand(size_elem).realize()
    b2 = Tensor.rand(size_elem).realize()

    for _ in range(warmup):
        c2 = (a2 + b2).realize()
    _sync()  # flush warmup command buffers before timed trials

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        c2 = (a2 + b2).realize()
        _ = c2.numpy()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    elem_bytes = size_elem * 4 * 3  # 2 reads + 1 write
    elemwise_gbs = elem_bytes / np.median(times) / 1e9
    print(f"  Median: {elemwise_gbs:.1f} GB/s ({np.median(times)*1000:.2f} ms)")

    # --- KV cache simulation ---
    # Realistic: 32 layers, 32 heads, head_dim=128, seq_len=2048
    print("\n[4/4] KV cache simulation (32L x 32H x 128D x 2048 seq)...")
    _sync()  # flush all prior command buffers before large allocation
    n_layers, n_heads, head_dim, seq_len = 32, 32, 128, 2048
    kv_elements = n_layers * n_heads * head_dim * seq_len
    kv_bytes = kv_elements * 4  # float32
    kv_size_mb = kv_bytes / (1024 * 1024)
    print(f"  KV cache size: {kv_size_mb:.1f} MB (fp32, K only)")

    if not _alloc_fits(kv_bytes * 3, budget):  # K + fill + overhead
        print(f"  SKIP (KV allocation exceeds memory budget)")
        return DeviceResult(
            device=device_name, membw_gbs=round(membw_median, 2),
            matmul_gflops=round(matmul_gflops, 2),
            kv_alloc_ms=0, kv_fill_ms=0, kv_read_ms=0, kv_size_mb=round(kv_size_mb, 2),
            elemwise_gbs=round(elemwise_gbs, 2),
        )

    # Allocate
    t0 = time.perf_counter()
    kv = Tensor.rand(n_layers, n_heads, seq_len, head_dim).realize()
    t1 = time.perf_counter()
    kv_alloc_ms = (t1 - t0) * 1000

    # Fill (overwrite with new random data)
    times_fill = []
    for _ in range(min(warmup, 2)):
        kv2 = Tensor.rand(n_layers, n_heads, seq_len, head_dim).realize()
    _sync()  # flush warmup command buffers

    for _ in range(min(trials, 5)):
        t0 = time.perf_counter()
        kv2 = Tensor.rand(n_layers, n_heads, seq_len, head_dim).realize()
        t1 = time.perf_counter()
        times_fill.append(t1 - t0)
    kv_fill_ms = np.median(times_fill) * 1000

    # Read back to host
    times_read = []
    for _ in range(min(trials, 5)):
        t0 = time.perf_counter()
        _ = kv.numpy()
        t1 = time.perf_counter()
        times_read.append(t1 - t0)
    kv_read_ms = np.median(times_read) * 1000

    print(f"  Alloc: {kv_alloc_ms:.1f} ms")
    print(f"  Fill:  {kv_fill_ms:.1f} ms (median)")
    print(f"  Read:  {kv_read_ms:.1f} ms (median, device→host)")
    print(f"  Read BW: {kv_bytes / (np.median(times_read)) / 1e9:.1f} GB/s")

    return DeviceResult(
        device=device_name,
        membw_gbs=round(membw_median, 2),
        matmul_gflops=round(matmul_gflops, 2),
        kv_alloc_ms=round(kv_alloc_ms, 2),
        kv_fill_ms=round(kv_fill_ms, 2),
        kv_read_ms=round(kv_read_ms, 2),
        kv_size_mb=round(kv_size_mb, 2),
        elemwise_gbs=round(elemwise_gbs, 2),
    )


def print_comparison(results: list[DeviceResult]) -> None:
    """Print side-by-side comparison table."""
    if len(results) < 2:
        return

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")

    header = f"{'Metric':<30}"
    for r in results:
        header += f"  {r.device:>12}"
    print(header)
    print("-" * (30 + 14 * len(results)))

    rows = [
        ("Mem BW (GB/s)", "membw_gbs"),
        ("Matmul (GFLOPS)", "matmul_gflops"),
        ("Elemwise BW (GB/s)", "elemwise_gbs"),
        ("KV Alloc (ms)", "kv_alloc_ms"),
        ("KV Fill (ms)", "kv_fill_ms"),
        ("KV Read D→H (ms)", "kv_read_ms"),
    ]

    for label, field in rows:
        line = f"{label:<30}"
        for r in results:
            val = getattr(r, field)
            line += f"  {val:>12.2f}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Device baseline benchmarks")
    parser.add_argument(
        "--device", "-d",
        default="NV",
        help="Comma-separated device list, e.g. NV,METAL (default: NV)",
    )
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument(
        "--reserve", type=float, default=0.20,
        help="Fraction of device memory to reserve (default: 0.20 = 20%%)",
    )
    args = parser.parse_args()

    global MEMORY_RESERVE
    MEMORY_RESERVE = args.reserve

    devices = [d.strip().upper() for d in args.device.split(",")]
    results: list[DeviceResult] = []

    for dev in devices:
        result = benchmark_device(dev)
        results.append(result)
        print(f"\n  ✓ {dev} baselines complete")

    print_comparison(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
