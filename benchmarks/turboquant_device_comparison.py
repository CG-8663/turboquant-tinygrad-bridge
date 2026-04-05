"""TurboQuant device comparison: Metal vs NV (RTX PRO 6000 eGPU via TB5).

Runs identical workloads on both devices to show why the bridge exists:
- NV dominates compute-bound (prefill, batch matmul)
- Metal dominates memory-bound (decode, KV read/write)
- TurboQuant compression makes TB5 transfer viable

Usage:
    PYTHONPATH=./tinygrad python3.13 benchmarks/turboquant_device_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class DeviceComparison:
    device: str
    # Prefill: (batch, seq, hidden) @ (hidden, hidden) FP16
    prefill_512_gflops: float = 0.0
    prefill_2048_gflops: float = 0.0
    prefill_4096_gflops: float = 0.0
    # Decode: single token through 32 layers
    decode_tok_per_sec: float = 0.0
    # KV cache ops
    kv_alloc_ms: float = 0.0
    kv_fill_ms: float = 0.0
    kv_read_ms: float = 0.0
    # Simulated turbo compress/decompress bandwidth
    turbo_compress_gbs: float = 0.0
    turbo_decompress_gbs: float = 0.0
    # Transfer simulation: how long to move KV over TB5 (raw vs compressed)
    transfer_raw_ms: float = 0.0
    transfer_turbo3_ms: float = 0.0
    transfer_turbo4_ms: float = 0.0


def _timed(fn, warmup=3, trials=6):
    """Median wall-clock with warmup, sync via .sum().numpy()."""
    for _ in range(warmup):
        r = fn()
        if hasattr(r, 'sum'):
            r.sum().numpy()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        if hasattr(r, 'sum'):
            r.sum().numpy()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def benchmark_device(device_name: str) -> DeviceComparison:
    import os
    os.environ["DEV"] = device_name

    from tinygrad import Device, Tensor, dtypes

    res = DeviceComparison(device=device_name)

    print(f"\n{'='*64}")
    print(f"  TurboQuant Device Comparison: {device_name}")
    print(f"{'='*64}")

    # --- Prefill simulation: FP16 matmul at various batch sizes ---
    print("\n[1/5] Prefill simulation (FP16 matmul)...")
    hidden = 4096
    for seq in [512, 2048, 4096]:
        batch = seq  # tokens
        a = Tensor.rand(batch, hidden).cast(dtypes.half).realize()
        w = Tensor.rand(hidden, hidden).cast(dtypes.half).realize()

        def do_prefill(a=a, w=w):
            return (a @ w).realize()

        dt = _timed(do_prefill, warmup=2, trials=6)
        flops = 2 * batch * hidden * hidden
        gflops = flops / dt / 1e9
        setattr(res, f"prefill_{seq}_gflops", round(gflops, 1))
        print(f"  seq={seq:>5}: {gflops:>10.1f} GFLOPS  ({dt*1000:.2f} ms)")
        del a, w

    # --- Decode simulation: 32 layers, single token ---
    print("\n[2/5] Decode simulation (32 layers, hidden=4096)...")
    n_layers = 32
    weights = [Tensor.rand(hidden, hidden).realize() for _ in range(n_layers)]
    token = Tensor.rand(1, hidden).realize()

    def decode_step(token=token, weights=weights):
        x = token
        for w in weights:
            x = (x @ w).realize()
        return x

    dt = _timed(decode_step, warmup=2, trials=8)
    res.decode_tok_per_sec = round(1.0 / dt, 1)
    print(f"  {res.decode_tok_per_sec:.1f} tok/s  ({dt*1000:.1f} ms/token)")
    del weights, token

    # --- KV cache ops ---
    print("\n[3/5] KV cache ops (32L x 32H x 128D x 2048 seq)...")
    n_layers, n_heads, head_dim, seq_len = 32, 32, 128, 2048
    kv_bytes = n_layers * n_heads * head_dim * seq_len * 4

    t0 = time.perf_counter()
    kv = Tensor.rand(n_layers, n_heads, seq_len, head_dim).realize()
    _ = kv.sum().numpy()
    t1 = time.perf_counter()
    res.kv_alloc_ms = round((t1 - t0) * 1000, 1)

    # Fill
    def kv_fill():
        return Tensor.rand(n_layers, n_heads, seq_len, head_dim).realize()
    dt = _timed(kv_fill, warmup=1, trials=5)
    res.kv_fill_ms = round(dt * 1000, 1)

    # Read
    times_read = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = kv.numpy()
        t1 = time.perf_counter()
        times_read.append(t1 - t0)
    res.kv_read_ms = round(np.median(times_read) * 1000, 1)
    print(f"  Alloc: {res.kv_alloc_ms:.1f} ms")
    print(f"  Fill:  {res.kv_fill_ms:.1f} ms")
    print(f"  Read:  {res.kv_read_ms:.1f} ms")
    del kv

    # --- Simulated turbo compress/decompress ---
    # Turbo compress = scale extraction + rotation + quantize (compute-bound)
    # Simulate with: matmul (rotation) + elementwise ops
    print("\n[4/5] Simulated turbo compress/decompress...")
    n_vecs = n_layers * n_heads * seq_len  # 2M vectors
    # Rotation is the expensive part: (n_vecs, head_dim) @ (head_dim, head_dim)
    chunk = min(n_vecs, 65536)  # process in chunks
    vecs = Tensor.rand(chunk, head_dim).realize()
    rot = Tensor.rand(head_dim, head_dim).realize()

    def compress_sim(v=vecs, r=rot):
        rotated = (v @ r).realize()
        # Quantize simulation: abs + compare + pack (elementwise)
        return (rotated.abs() * 127).cast(dtypes.int8).realize()

    dt = _timed(compress_sim, warmup=2, trials=6)
    compress_bytes = chunk * head_dim * 4
    res.turbo_compress_gbs = round(compress_bytes / dt / 1e9, 1)
    print(f"  Compress:   {res.turbo_compress_gbs:.1f} GB/s  ({dt*1000:.2f} ms for {chunk} vecs)")

    def decompress_sim(v=vecs, r=rot):
        # Dequantize + inverse rotation
        return (v @ r.T).realize()

    dt = _timed(decompress_sim, warmup=2, trials=6)
    res.turbo_decompress_gbs = round(compress_bytes / dt / 1e9, 1)
    print(f"  Decompress: {res.turbo_decompress_gbs:.1f} GB/s  ({dt*1000:.2f} ms for {chunk} vecs)")
    del vecs, rot

    # --- TB5 transfer simulation ---
    print("\n[5/5] TB5 transfer simulation (1 GB KV cache)...")
    TB5_BW_GBS = 5.0  # practical TB5 PCIe 4.0 x4 bandwidth
    kv_gb = kv_bytes / 1e9
    res.transfer_raw_ms = round(kv_gb / TB5_BW_GBS * 1000, 1)
    res.transfer_turbo3_ms = round(kv_gb / 4.6 / TB5_BW_GBS * 1000, 1)  # 4.6x compression
    res.transfer_turbo4_ms = round(kv_gb / 3.8 / TB5_BW_GBS * 1000, 1)  # 3.8x compression
    print(f"  Raw (fp32):   {res.transfer_raw_ms:.0f} ms  ({kv_gb:.2f} GB)")
    print(f"  turbo3 (4.6x): {res.transfer_turbo3_ms:.0f} ms  ({kv_gb/4.6:.2f} GB)")
    print(f"  turbo4 (3.8x): {res.transfer_turbo4_ms:.0f} ms  ({kv_gb/3.8:.2f} GB)")

    return res


def print_comparison(metal: DeviceComparison, nv: DeviceComparison):
    print(f"\n{'='*72}")
    print("  TURBOQUANT BRIDGE: WHY BOTH DEVICES MATTER")
    print(f"{'='*72}")

    def row(label, mv, nv_val, unit, higher_better=True):
        if mv == 0 or nv_val == 0:
            print(f"  {label:<36} {mv:>10.1f} {nv_val:>10.1f} {unit}")
            return
        if higher_better:
            winner = "Metal" if mv > nv_val else "NV"
            ratio = max(mv, nv_val) / min(mv, nv_val)
        else:
            winner = "Metal" if mv < nv_val else "NV"
            ratio = max(mv, nv_val) / min(mv, nv_val)
        print(f"  {label:<36} {mv:>10.1f} {nv_val:>10.1f} {unit:<8} {ratio:.1f}x {winner}")

    print(f"  {'Metric':<36} {'Metal':>10} {'NV(eGPU)':>10} {'Unit':<8} {'Ratio'}")
    print("  " + "-" * 72)

    print("  -- Prefill (NV territory: Blackwell tensor cores) --")
    row("Prefill 512 FP16", metal.prefill_512_gflops, nv.prefill_512_gflops, "GFLOPS")
    row("Prefill 2048 FP16", metal.prefill_2048_gflops, nv.prefill_2048_gflops, "GFLOPS")
    row("Prefill 4096 FP16", metal.prefill_4096_gflops, nv.prefill_4096_gflops, "GFLOPS")

    print("  -- Decode (Metal territory: zero-copy unified memory) --")
    row("Decode (tok/s)", metal.decode_tok_per_sec, nv.decode_tok_per_sec, "tok/s")
    row("KV read (ms)", metal.kv_read_ms, nv.kv_read_ms, "ms", higher_better=False)
    row("KV fill (ms)", metal.kv_fill_ms, nv.kv_fill_ms, "ms", higher_better=False)

    print("  -- Turbo compress/decompress throughput --")
    row("Compress BW", metal.turbo_compress_gbs, nv.turbo_compress_gbs, "GB/s")
    row("Decompress BW", metal.turbo_decompress_gbs, nv.turbo_decompress_gbs, "GB/s")

    print("  -- TB5 transfer (1 GB KV cache) --")
    row("Raw transfer", metal.transfer_raw_ms, nv.transfer_raw_ms, "ms", higher_better=False)
    row("turbo3 (4.6x)", metal.transfer_turbo3_ms, nv.transfer_turbo3_ms, "ms", higher_better=False)
    row("turbo4 (3.8x)", metal.transfer_turbo4_ms, nv.transfer_turbo4_ms, "ms", higher_better=False)

    nv_prefill_lead = nv.prefill_4096_gflops / metal.prefill_4096_gflops if metal.prefill_4096_gflops > 0 else 0
    metal_read_lead = nv.kv_read_ms / metal.kv_read_ms if metal.kv_read_ms > 0 else 0
    turbo3_speedup = metal.transfer_raw_ms / metal.transfer_turbo3_ms if metal.transfer_turbo3_ms > 0 else 0

    print(f"""
  Summary:
    NV prefill advantage:     {nv_prefill_lead:.1f}x (Blackwell tensor cores)
    Metal KV read advantage:  {metal_read_lead:.1f}x (unified memory, no TB5)
    turbo3 transfer speedup:  {turbo3_speedup:.1f}x (4.6x compression over TB5)

    Bridge strategy:
      NV  → heavy prefill, batch attention (compute-bound)
      Metal → decode, KV serving, host access (memory-bound)
      TB5 + turbo3 → ~{5.0 * 4.6:.0f} GB/s effective vs {5.0:.0f} GB/s raw
""")


def main():
    results = {}

    for dev in ["METAL", "NV"]:
        try:
            results[dev] = benchmark_device(dev)
            print(f"\n  ✓ {dev} complete")
        except Exception as e:
            print(f"\n  ✗ {dev} failed: {e}")

    if "METAL" in results and "NV" in results:
        print_comparison(results["METAL"], results["NV"])

    with open("benchmarks/turboquant_comparison.json", "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)
    print("Results saved to benchmarks/turboquant_comparison.json")


if __name__ == "__main__":
    main()
