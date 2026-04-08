"""Hardware benchmark: warmup impact on Metal→NV bridge transfer.

Measures end-to-end KV cache transfer over Thunderbolt 5 with and
without kernel warmup. Tests both tinygrad and native backends.

Requires: Metal (M3 Ultra) + NV (RTX PRO 6000 via TB5) live.

Usage:
    python benchmarks/hardware_warmup.py
    python benchmarks/hardware_warmup.py --layers 8 --heads 32 --seq-len 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tinygrad"))

import numpy as np
from tinygrad import Tensor, Device

from tqbridge.bridge import KVBridge
from tqbridge.wire import Format


def check_devices():
    try:
        Device["METAL"]
        Device["NV"]
        return True
    except Exception:
        return False


def bench_transfer(bridge, k_cache, v_cache, label: str) -> dict:
    """Run a transfer and return timing dict."""
    t0 = time.perf_counter()
    k_out, v_out, pipeline = bridge.transfer_kv_cache(k_cache, v_cache)
    t1 = time.perf_counter()

    wall_ms = (t1 - t0) * 1000
    n_layers = len(k_out)

    # Verify output is on NV
    for i in range(n_layers):
        assert k_out[i].device == "NV", f"Layer {i} K not on NV"
        assert v_out[i].device == "NV", f"Layer {i} V not on NV"

    return {
        "label": label,
        "wall_ms": wall_ms,
        "pipeline_total_ms": pipeline.total_time_ms,
        "avg_compress_ms": np.mean([m.compress_time_ms for m in pipeline.layers]),
        "avg_transfer_ms": np.mean([m.transfer_time_ms for m in pipeline.layers]),
        "avg_decompress_ms": np.mean([m.decompress_time_ms for m in pipeline.layers]),
        "avg_ratio": pipeline.avg_compression_ratio,
        "n_layers": n_layers,
    }


def print_result(r: dict, data_mb: float):
    throughput = data_mb / (r["wall_ms"] / 1000) if r["wall_ms"] > 0 else 0
    print(f"  {r['label']:30s}  wall {r['wall_ms']:8.1f} ms  "
          f"compress {r['avg_compress_ms']:6.1f}  transfer {r['avg_transfer_ms']:6.1f}  "
          f"decompress {r['avg_decompress_ms']:6.1f}  "
          f"ratio {r['avg_ratio']:4.1f}x  {throughput:8.1f} MB/s")


def main():
    parser = argparse.ArgumentParser(description="Hardware warmup benchmark")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not check_devices():
        print("ERROR: Need both Metal and NV devices. Is the eGPU connected?")
        sys.exit(1)

    shape = (args.layers, args.heads, args.seq_len, args.head_dim)
    data_mb = np.prod(shape) * 4 * 2 / 1e6  # K + V, float32

    print(f"\n{'=' * 90}")
    print(f"  Hardware Warmup Benchmark — Metal (M3 Ultra) → NV (RTX PRO 6000 via TB5)")
    print(f"  KV cache: {args.layers} layers × {args.heads} heads × {args.seq_len} seq × {args.head_dim} dim")
    print(f"  Data: {data_mb:.2f} MB (K + V float32), {args.iters} iterations")
    print(f"{'=' * 90}\n")

    all_results = {}

    for backend in ["tinygrad", "native"]:
        print(f"[Backend: {backend}]")

        # Fresh bridge — no cached kernels
        bridge = KVBridge(
            head_dim=args.head_dim, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3,
            backend=backend,
        )

        # Generate test data
        k_cache = Tensor.rand(*shape, device="METAL").realize()
        v_cache = Tensor.rand(*shape, device="METAL").realize()

        # Cold run (first-ever, includes kernel compilation)
        cold = bench_transfer(bridge, k_cache, v_cache, f"{backend} cold (no warmup)")
        print_result(cold, data_mb)

        # Subsequent runs (kernels cached in memory)
        warm_results = []
        for i in range(args.iters):
            k_cache = Tensor.rand(*shape, device="METAL").realize()
            v_cache = Tensor.rand(*shape, device="METAL").realize()
            r = bench_transfer(bridge, k_cache, v_cache, f"{backend} warm iter {i+1}")
            warm_results.append(r)

        warm_median = sorted(warm_results, key=lambda x: x["wall_ms"])[len(warm_results) // 2]
        warm_median["label"] = f"{backend} warm (median of {args.iters})"
        print_result(warm_median, data_mb)

        bridge.close()

        # Fresh bridge with explicit warmup
        bridge2 = KVBridge(
            head_dim=args.head_dim, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3,
            backend=backend,
        )

        warmup_ms = bridge2.warmup(n_heads=args.heads, seq_len=args.seq_len)
        print(f"  {'warmup time':30s}  {warmup_ms:8.1f} ms")

        k_cache = Tensor.rand(*shape, device="METAL").realize()
        v_cache = Tensor.rand(*shape, device="METAL").realize()
        post_warmup = bench_transfer(bridge2, k_cache, v_cache, f"{backend} post-warmup (1st)")
        print_result(post_warmup, data_mb)

        bridge2.close()

        speedup = cold["wall_ms"] / warm_median["wall_ms"] if warm_median["wall_ms"] > 0 else 0
        print(f"  {'speedup (cold → warm)':30s}  {speedup:.1f}x\n")

        all_results[backend] = {
            "cold": cold,
            "warm_median": warm_median,
            "post_warmup": post_warmup,
            "warmup_ms": warmup_ms,
        }

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps({
            "config": {
                "layers": args.layers,
                "heads": args.heads,
                "seq_len": args.seq_len,
                "head_dim": args.head_dim,
                "data_mb": data_mb,
            },
            "results": all_results,
        }, indent=2, default=float))
        print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
