"""Benchmark: native C vs tinygrad compression backend.

Compares compress/decompress throughput for NativeCompressor vs TinygradCompressor.
Runs on CPU (CLANG device) since the benchmark isolates compression cost,
not DMA transfer. No GPU required.

Usage:
    python benchmarks/native_vs_tinygrad.py
    python benchmarks/native_vs_tinygrad.py --head-dim 128 --n-vectors 256 --iters 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tqbridge.wire import Format


def benchmark_native(data: np.ndarray, fmt: Format, head_dim: int, iters: int) -> dict:
    """Benchmark NativeCompressor (C via ctypes)."""
    from tqbridge.native import NativeCompressor

    nc = NativeCompressor(head_dim=head_dim, seed=42)

    # Warmup
    comp = nc.compress(data, fmt)
    nc.decompress(comp)

    compress_times = []
    decompress_times = []

    for _ in range(iters):
        t0 = time.perf_counter()
        comp = nc.compress(data, fmt)
        t1 = time.perf_counter()
        result = nc.decompress(comp)
        t2 = time.perf_counter()

        compress_times.append((t1 - t0) * 1000)
        decompress_times.append((t2 - t1) * 1000)

    nc.close()

    mse = float(np.mean((result - data) ** 2))
    return {
        "backend": "native",
        "compress_ms": compress_times,
        "decompress_ms": decompress_times,
        "mse": mse,
    }


def benchmark_numpy(data: np.ndarray, fmt: Format, head_dim: int, iters: int) -> dict:
    """Benchmark NumPy reference implementation."""
    from tqbridge.compression import CompressionPipeline

    pipeline = CompressionPipeline(seed=42)

    # For Q8_0, data is flat; for PolarQuant, data is (n_vectors, head_dim)
    if fmt == Format.Q8_0:
        n_vectors = data.size // head_dim
    else:
        n_vectors = data.shape[0]

    # Warmup
    comp = pipeline._compress_tensor(data, fmt, head_dim)
    pipeline._decompress_tensor(comp, fmt, n_vectors, head_dim)

    compress_times = []
    decompress_times = []

    for _ in range(iters):
        t0 = time.perf_counter()
        comp = pipeline._compress_tensor(data, fmt, head_dim)
        t1 = time.perf_counter()
        result = pipeline._decompress_tensor(comp, fmt, n_vectors, head_dim)
        t2 = time.perf_counter()

        compress_times.append((t1 - t0) * 1000)
        decompress_times.append((t2 - t1) * 1000)

    mse = float(np.mean((result - data) ** 2))
    return {
        "backend": "numpy",
        "compress_ms": compress_times,
        "decompress_ms": decompress_times,
        "mse": mse,
    }


def benchmark_tinygrad(data: np.ndarray, fmt: Format, head_dim: int, iters: int) -> dict:
    """Benchmark TinygradCompressor (tensor ops on CPU device)."""
    from tinygrad import Tensor
    from tqbridge.compression_tg import TinygradCompressor

    tc = TinygradCompressor(head_dim=head_dim, seed=42)
    t_data = Tensor(data).realize()

    # Warmup
    comp = tc.compress(t_data, fmt)
    tc.decompress(comp)

    compress_times = []
    decompress_times = []

    for _ in range(iters):
        t0 = time.perf_counter()
        comp = tc.compress(t_data, fmt)
        t1 = time.perf_counter()
        result = tc.decompress(comp)
        t2 = time.perf_counter()

        compress_times.append((t1 - t0) * 1000)
        decompress_times.append((t2 - t1) * 1000)

    result_np = result.numpy()
    mse = float(np.mean((result_np - data) ** 2))
    return {
        "backend": "tinygrad",
        "compress_ms": compress_times,
        "decompress_ms": decompress_times,
        "mse": mse,
    }


def print_results(label: str, results: dict, n_vectors: int, head_dim: int):
    """Print formatted benchmark results."""
    compress = results["compress_ms"]
    decompress = results["decompress_ms"]
    total = [c + d for c, d in zip(compress, decompress)]

    c_med = sorted(compress)[len(compress) // 2]
    d_med = sorted(decompress)[len(decompress) // 2]
    t_med = sorted(total)[len(total) // 2]

    data_mb = n_vectors * head_dim * 4 / 1e6
    throughput = data_mb / (t_med / 1000) if t_med > 0 else 0

    print(f"  {label:12s}  compress {c_med:8.2f} ms  decompress {d_med:8.2f} ms  "
          f"total {t_med:8.2f} ms  {throughput:8.1f} MB/s  MSE {results['mse']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Native vs tinygrad compression benchmark")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-vectors", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    data = rng.standard_normal((args.n_vectors, args.head_dim)).astype(np.float32)
    data_mb = args.n_vectors * args.head_dim * 4 / 1e6

    print(f"\n{'=' * 78}")
    print(f"  Native C vs Tinygrad vs NumPy Compression Benchmark")
    print(f"  {args.n_vectors} vectors × {args.head_dim} dim = {data_mb:.2f} MB float32")
    print(f"  {args.iters} iterations (median)")
    print(f"{'=' * 78}\n")

    formats = [
        ("TURBO3", Format.TURBO3),
        ("TURBO4", Format.TURBO4),
        ("TURBO2", Format.TURBO2),
        ("Q8_0", Format.Q8_0),
    ]

    all_results = {}

    for fmt_name, fmt in formats:
        print(f"[{fmt_name}]")

        # For Q8_0, NativeCompressor and NumPy operate on flat data
        bench_data = data.reshape(-1) if fmt == Format.Q8_0 else data

        native_res = benchmark_native(bench_data, fmt, args.head_dim, args.iters)
        numpy_res = benchmark_numpy(bench_data, fmt, args.head_dim, args.iters)

        # Tinygrad only supports PolarQuant formats and Q8_0
        tg_res = benchmark_tinygrad(data, fmt, args.head_dim, args.iters)

        print_results("native (C)", native_res, args.n_vectors, args.head_dim)
        print_results("tinygrad", tg_res, args.n_vectors, args.head_dim)
        print_results("numpy (ref)", numpy_res, args.n_vectors, args.head_dim)

        # Speedup
        n_total = sorted([c + d for c, d in zip(native_res["compress_ms"], native_res["decompress_ms"])])
        t_total = sorted([c + d for c, d in zip(tg_res["compress_ms"], tg_res["decompress_ms"])])
        p_total = sorted([c + d for c, d in zip(numpy_res["compress_ms"], numpy_res["decompress_ms"])])
        n_med = n_total[len(n_total) // 2]
        t_med = t_total[len(t_total) // 2]
        p_med = p_total[len(p_total) // 2]

        print(f"  {'speedup':12s}  native vs tinygrad: {t_med/n_med:.1f}x  "
              f"native vs numpy: {p_med/n_med:.1f}x")
        print()

        all_results[fmt_name] = {
            "native": {
                "median_compress_ms": sorted(native_res["compress_ms"])[len(native_res["compress_ms"]) // 2],
                "median_decompress_ms": sorted(native_res["decompress_ms"])[len(native_res["decompress_ms"]) // 2],
                "mse": native_res["mse"],
            },
            "tinygrad": {
                "median_compress_ms": sorted(tg_res["compress_ms"])[len(tg_res["compress_ms"]) // 2],
                "median_decompress_ms": sorted(tg_res["decompress_ms"])[len(tg_res["decompress_ms"]) // 2],
                "mse": tg_res["mse"],
            },
            "numpy": {
                "median_compress_ms": sorted(numpy_res["compress_ms"])[len(numpy_res["compress_ms"]) // 2],
                "median_decompress_ms": sorted(numpy_res["decompress_ms"])[len(numpy_res["decompress_ms"]) // 2],
                "mse": numpy_res["mse"],
            },
        }

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps({
            "config": {
                "head_dim": args.head_dim,
                "n_vectors": args.n_vectors,
                "iters": args.iters,
                "data_mb": data_mb,
            },
            "results": all_results,
        }, indent=2))
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
