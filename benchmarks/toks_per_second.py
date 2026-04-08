"""End-to-end tok/s benchmark: KV cache bridge throughput on live hardware.

Measures how many per-token KV cache updates the bridge can push per second
across all backends: tinygrad, native C, and CUDA kernels.

Simulates realistic decode-phase KV transfer for split Metal↔CUDA inference:
  - Each "token" produces 1 KV update per layer (n_kv_heads × 1 × head_dim)
  - Bridge compresses on Metal, transfers over TB5, decompresses on NV
  - tok/s = tokens processed / wall time

Hardware: M3 Ultra (Metal) → RTX PRO 6000 (NV) via Thunderbolt 5.

Usage:
    python benchmarks/toks_per_second.py
    python benchmarks/toks_per_second.py --model qwen3-8b --tokens 100
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

# Model configs: (n_layers, n_kv_heads, head_dim)
MODELS = {
    "qwen3-8b":    (32, 8, 128),
    "qwen3-14b":   (40, 8, 128),
    "qwen3.5-27b": (48, 4, 128),
    "llama3-8b":   (32, 8, 128),
    "llama3-70b":  (80, 8, 128),
}


def bench_toks(
    bridge: KVBridge,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    n_tokens: int,
    bulk: bool = True,
) -> dict:
    """Measure tok/s for n_tokens sequential KV cache transfers."""
    # Per-token KV shape: (n_layers, n_kv_heads, 1, head_dim)
    shape = (n_layers, n_kv_heads, 1, head_dim)

    # Warmup
    k = Tensor.rand(*shape, device="METAL").realize()
    v = Tensor.rand(*shape, device="METAL").realize()
    if bulk:
        bridge.transfer_kv_bulk(k, v)
    else:
        bridge.transfer_kv_cache(k, v)

    # Benchmark
    token_times = []
    t_start = time.perf_counter()

    for tok in range(n_tokens):
        k = Tensor.rand(*shape, device="METAL").realize()
        v = Tensor.rand(*shape, device="METAL").realize()

        t0 = time.perf_counter()
        if bulk:
            _, _, metrics = bridge.transfer_kv_bulk(k, v)
        else:
            _, _, pipeline = bridge.transfer_kv_cache(k, v)
            metrics = pipeline
        t1 = time.perf_counter()
        token_times.append((t1 - t0) * 1000)

    t_end = time.perf_counter()
    wall_s = t_end - t_start
    toks_per_s = n_tokens / wall_s

    per_token_ms = np.median(token_times)
    kv_bytes = n_layers * n_kv_heads * head_dim * 4 * 2
    raw_bw = kv_bytes * toks_per_s / 1e9

    return {
        "toks_per_s": toks_per_s,
        "wall_s": wall_s,
        "per_token_ms": float(per_token_ms),
        "n_tokens": n_tokens,
        "kv_bytes_per_token": kv_bytes,
        "raw_bw_gbps": raw_bw,
    }


def main():
    parser = argparse.ArgumentParser(description="KV bridge tok/s benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen3-8b")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Check hardware
    try:
        Device["METAL"]
        Device["NV"]
    except Exception:
        print("ERROR: Need Metal + NV. Is the eGPU connected?")
        sys.exit(1)

    n_layers, n_kv_heads, head_dim = MODELS[args.model]
    kv_bytes = n_layers * n_kv_heads * head_dim * 4 * 2

    print(f"\n{'=' * 80}")
    print(f"  KV Bridge tok/s — {args.model}")
    print(f"  {n_layers} layers × {n_kv_heads} KV heads × {head_dim} dim")
    print(f"  Per-token KV: {kv_bytes/1024:.0f} KB, {args.tokens} tokens")
    print(f"  Metal (M3 Ultra) → NV (RTX PRO 6000) via TB5")
    print(f"{'=' * 80}\n")

    configs = [
        ("tinygrad (Q8_0 K + TURBO3 V)",  "tinygrad", Format.Q8_0,   Format.TURBO3),
        ("tinygrad (TURBO3 symmetric)",    "tinygrad", Format.TURBO3, Format.TURBO3),
        ("native C (Q8_0 K + TURBO3 V)",   "native",  Format.Q8_0,   Format.TURBO3),
        ("native C (TURBO3 symmetric)",     "native",  Format.TURBO3, Format.TURBO3),
    ]

    # Try CUDA backend
    try:
        from tqbridge.kernels.cuda import CUDACompressor
        _test = CUDACompressor(head_dim=head_dim, seed=42)
        _test.close()
        configs.extend([
            ("CUDA kernels (Q8_0 K + TURBO3 V)", "cuda", Format.Q8_0,   Format.TURBO3),
            ("CUDA kernels (TURBO3 symmetric)",   "cuda", Format.TURBO3, Format.TURBO3),
        ])
    except Exception as e:
        print(f"  (CUDA kernels unavailable: {e})\n")

    results = {}

    for label, backend, fmt_k, fmt_v in configs:
        bridge = KVBridge(
            head_dim=head_dim, fmt_k=fmt_k, fmt_v=fmt_v,
            backend=backend, seed=42,
        )

        # Pre-allocate CUDA buffers for the expected shape
        if backend == "cuda":
            bridge.warmup(n_heads=n_kv_heads, seq_len=1, n_layers=n_layers)

        res = bench_toks(bridge, n_layers, n_kv_heads, head_dim, args.tokens)
        bridge.close()

        print(f"  {label:42s}  {res['toks_per_s']:7.1f} tok/s  "
              f"{res['per_token_ms']:6.1f} ms/tok  "
              f"{res['raw_bw_gbps']:.3f} GB/s raw")

        results[label] = res

    # Summary
    print(f"\n{'─' * 80}")
    best = max(results.items(), key=lambda x: x[1]["toks_per_s"])
    print(f"  Best: {best[0]} — {best[1]['toks_per_s']:.1f} tok/s")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps({
            "model": args.model,
            "config": {"n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim},
            "tokens": args.tokens,
            "results": results,
        }, indent=2, default=float))
        print(f"  Saved to {out}")

    print()


if __name__ == "__main__":
    main()
