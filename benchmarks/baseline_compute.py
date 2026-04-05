"""Loaded compute baselines: on-device workloads that reveal each device's strengths.

RTX PRO 6000 Blackwell: raw compute density, batch throughput, FP16 tensor ops
M3 Ultra: memory bandwidth utilisation, large working sets, serving-style access

We avoid host readback in timing loops so TB5 doesn't mask the GPU's true capability.

Usage:
    PYTHONPATH=./tinygrad python3.13 benchmarks/baseline_compute.py --device NV
    PYTHONPATH=./tinygrad python3.13 benchmarks/baseline_compute.py --device METAL
    PYTHONPATH=./tinygrad python3.13 benchmarks/baseline_compute.py --device NV,METAL
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Memory budget: reserve 20% of device memory to avoid driver crashes
# ---------------------------------------------------------------------------
MEMORY_RESERVE = 0.20  # keep 20% free

def _get_total_memory_bytes(device_name: str) -> int:
    """Return total device memory in bytes."""
    if device_name == "METAL" or (device_name == "NV" and platform.system() == "Darwin"):
        # macOS unified memory — both METAL and NV (eGPU) share system RAM pressure
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        except Exception:
            return 0
    if device_name == "NV":
        # Linux / Windows: query VRAM via nvidia-smi
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True,
            )
            return int(out.strip().split("\n")[0]) * 1024 * 1024  # MiB → bytes
        except Exception:
            return 0
    return 0


def get_memory_budget(device_name: str) -> int:
    """Return usable memory budget in bytes (total minus reserve)."""
    total = _get_total_memory_bytes(device_name)
    if total == 0:
        return 0  # unknown — caller should fall back to conservative defaults
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
class ComputeResult:
    device: str
    # Matmul scaling: on-device GFLOPS at various sizes (no readback)
    matmul_512_gflops: float = 0.0
    matmul_1024_gflops: float = 0.0
    matmul_2048_gflops: float = 0.0
    matmul_4096_gflops: float = 0.0
    # FP16 matmul (tensor core territory)
    matmul_fp16_2048_gflops: float = 0.0
    matmul_fp16_4096_gflops: float = 0.0
    # Batch attention simulation: (batch, heads, seq, head_dim) matmul
    attn_score_gflops: float = 0.0  # Q @ K^T
    attn_value_gflops: float = 0.0  # scores @ V
    # Memory-bound: large sequential scan (bandwidth utilisation)
    scan_1gb_gbs: float = 0.0
    scan_4gb_gbs: float = 0.0
    # Serving pattern: many small matmuls (single-token decode simulation)
    decode_tok_per_sec: float = 0.0
    # Multi-head KV gather: random access across heads (cache-friendly test)
    kv_gather_gbs: float = 0.0
    # Chained compute: multi-layer MLP, one sync (shows true on-device throughput)
    chained_mlp_gflops: float = 0.0
    chained_mlp_ms: float = 0.0
    # Batch prefill: large batch FP16 matmul
    prefill_gflops: float = 0.0
    prefill_ms: float = 0.0
    # Memory capacity: max KV cache (seq_len) that fits
    max_kv_seq: int = 0
    max_kv_gb: float = 0.0
    # Reduction: layernorm-style reduce over large tensor
    reduce_gbs: float = 0.0


def _timed(fn, warmup: int = 3, trials: int = 10) -> float:
    """Return median wall-clock seconds for fn(), with warmup.

    Forces device sync via .sum().numpy() — this requires the full tensor
    to be computed before the scalar can be read back. Using .sum() instead
    of .flatten()[0] because a slice/view on Metal unified memory can read
    stale buffer data without waiting for the GPU command buffer to finish.
    """
    for _ in range(warmup):
        r = fn()
        if hasattr(r, 'sum'):
            r.sum().numpy()  # sync

    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        r = fn()
        if hasattr(r, 'sum'):
            r.sum().numpy()  # force full device sync
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.median(times))


def benchmark_compute(device_name: str) -> ComputeResult:
    """Run compute-focused baselines on a single device."""
    import os
    os.environ["DEV"] = device_name

    from tinygrad import Device, Tensor, dtypes

    res = ComputeResult(device=device_name)
    budget = get_memory_budget(device_name)

    print(f"\n{'='*64}")
    print(f"  Loaded Compute Baselines: {device_name}")
    print(f"{'='*64}")

    # ----------------------------------------------------------------
    # 1. Matmul scaling (FP32, on-device, no readback)
    # ----------------------------------------------------------------
    print("\n[1/10] Matmul scaling (FP32, on-device)...")
    for N in [512, 1024, 2048, 4096]:
        a = Tensor.rand(N, N).realize()
        b = Tensor.rand(N, N).realize()

        def do_matmul(a=a, b=b):
            return (a @ b).realize()

        dt = _timed(do_matmul, warmup=3, trials=8)
        flops = 2 * N * N * N
        gflops = flops / dt / 1e9
        setattr(res, f"matmul_{N}_gflops", round(gflops, 1))
        print(f"  {N:>5}x{N}: {gflops:>10.1f} GFLOPS  ({dt*1000:.2f} ms)")
        del a, b

    # ----------------------------------------------------------------
    # 2. FP16 matmul (tensor core / AMX territory)
    # ----------------------------------------------------------------
    print("\n[2/10] Matmul FP16 (tensor cores / AMX)...")
    for N in [2048, 4096]:
        a = Tensor.rand(N, N).cast(dtypes.half).realize()
        b = Tensor.rand(N, N).cast(dtypes.half).realize()

        def do_fp16(a=a, b=b):
            return (a @ b).realize()

        dt = _timed(do_fp16, warmup=3, trials=8)
        flops = 2 * N * N * N
        gflops = flops / dt / 1e9
        setattr(res, f"matmul_fp16_{N}_gflops", round(gflops, 1))
        print(f"  {N:>5}x{N} fp16: {gflops:>10.1f} GFLOPS  ({dt*1000:.2f} ms)")
        del a, b

    # ----------------------------------------------------------------
    # 3. Batched attention score simulation
    #    Q @ K^T: (batch=8, heads=32, seq=2048, dim=128) x (batch=8, heads=32, dim=128, seq=2048)
    # ----------------------------------------------------------------
    print("\n[3/10] Batched attention (8 batch, 32 heads, 2048 seq, 128 dim)...")
    B, H, S, D = 8, 32, 2048, 128

    Q = Tensor.rand(B, H, S, D).realize()
    K = Tensor.rand(B, H, S, D).realize()
    V = Tensor.rand(B, H, S, D).realize()

    # Q @ K^T → (B, H, S, S)
    def attn_score(Q=Q, K=K):
        return (Q @ K.transpose(-1, -2)).realize()

    dt = _timed(attn_score, warmup=2, trials=6)
    flops_qk = 2 * B * H * S * S * D
    res.attn_score_gflops = round(flops_qk / dt / 1e9, 1)
    print(f"  Q@K^T:    {res.attn_score_gflops:>10.1f} GFLOPS  ({dt*1000:.2f} ms)")

    # scores @ V → (B, H, S, D)
    scores = Tensor.rand(B, H, S, S).realize()

    def attn_value(scores=scores, V=V):
        return (scores @ V).realize()

    dt = _timed(attn_value, warmup=2, trials=6)
    flops_sv = 2 * B * H * S * D * S
    res.attn_value_gflops = round(flops_sv / dt / 1e9, 1)
    print(f"  scores@V: {res.attn_value_gflops:>10.1f} GFLOPS  ({dt*1000:.2f} ms)")
    del Q, K, V, scores

    # ----------------------------------------------------------------
    # 4. Memory-bound: large sequential scan (sum reduction)
    #    This is where unified memory shines — pure bandwidth test
    # ----------------------------------------------------------------
    print("\n[4/10] Sequential scan (bandwidth utilisation)...")
    for size_gb, label in [(1, "1gb"), (4, "4gb")]:
        n_elem = size_gb * 1024 * 1024 * 1024 // 4
        alloc_bytes = n_elem * 4 * 2  # input + output tensors
        if not _alloc_fits(alloc_bytes, budget):
            print(f"  {size_gb} GB scan: SKIP (exceeds memory budget)")
            continue
        try:
            t = Tensor.rand(n_elem).realize()

            def scan(t=t):
                return (t + 1.0).realize()

            dt = _timed(scan, warmup=2, trials=6)
            bw = n_elem * 4 * 2 / dt / 1e9  # read + write
            setattr(res, f"scan_{label}_gbs", round(bw, 1))
            print(f"  {size_gb} GB scan: {bw:>8.1f} GB/s  ({dt*1000:.1f} ms)")
            del t
        except Exception as e:
            print(f"  {size_gb} GB scan: SKIP ({e})")

    # ----------------------------------------------------------------
    # 5. Single-token decode simulation (memory-bound, serving pattern)
    #    Simulates autoregressive decode: small matmul per layer
    #    32 layers, each doing (1, 4096) @ (4096, 4096) — one token
    # ----------------------------------------------------------------
    print("\n[5/10] Single-token decode sim (32 layers, hidden=4096)...")
    n_layers = 32
    hidden = 4096
    weights = [Tensor.rand(hidden, hidden).realize() for _ in range(n_layers)]
    token = Tensor.rand(1, hidden).realize()

    def decode_step(token=token, weights=weights):
        x = token
        for w in weights:
            x = (x @ w).realize()
        return x

    dt = _timed(decode_step, warmup=2, trials=8)
    # Each layer: 2 * 1 * 4096 * 4096 flops, but this is memory-bound
    # Report as tokens/sec (1 token per call)
    res.decode_tok_per_sec = round(1.0 / dt, 1)
    print(f"  {res.decode_tok_per_sec:.1f} tok/s  ({dt*1000:.1f} ms/token)")
    del weights, token

    # ----------------------------------------------------------------
    # 6. Multi-head KV gather (random head access pattern)
    #    Simulates reading specific heads from a large KV cache
    # ----------------------------------------------------------------
    print("\n[6/10] KV cache gather (32 heads, 128 dim, 4096 seq)...")
    n_heads, head_dim, seq_len = 32, 128, 4096
    kv_cache = Tensor.rand(n_heads, seq_len, head_dim).realize()

    def kv_gather(kv=kv_cache):
        # Read all heads, sum across seq dim (simulates attention gather)
        return kv.sum(axis=1).realize()

    dt = _timed(kv_gather, warmup=3, trials=8)
    kv_bytes = n_heads * seq_len * head_dim * 4
    res.kv_gather_gbs = round(kv_bytes / dt / 1e9, 1)
    print(f"  {res.kv_gather_gbs:.1f} GB/s  ({dt*1000:.2f} ms)")
    del kv_cache

    # ----------------------------------------------------------------
    # 7. Chained compute: 32-layer MLP forward pass, ONE sync at end
    #    This is where NV shines — all work stays on GPU, no per-op TB5 tax
    # ----------------------------------------------------------------
    print("\n[7/10] Chained MLP forward (32 layers, 4096 hidden, batch=32)...")
    n_layers_mlp = 32
    batch_mlp = 32
    hidden_mlp = 4096
    # 32 weight matrices (4096x4096 x 4B each) + input + intermediates
    mlp_alloc = n_layers_mlp * hidden_mlp * hidden_mlp * 4 + batch_mlp * hidden_mlp * 4
    if not _alloc_fits(mlp_alloc, budget):
        print(f"  SKIP (needs {mlp_alloc / (1024**3):.1f} GB, exceeds budget)")
    else:
        try:
            mlp_weights = [Tensor.rand(hidden_mlp, hidden_mlp).realize() for _ in range(n_layers_mlp)]
            mlp_input = Tensor.rand(batch_mlp, hidden_mlp).realize()

            def chained_mlp(x=mlp_input, ws=mlp_weights):
                for w in ws:
                    x = (x @ w).relu()
                return x.realize()

            dt = _timed(chained_mlp, warmup=2, trials=6)
            total_flops_mlp = n_layers_mlp * 2 * batch_mlp * hidden_mlp * hidden_mlp
            res.chained_mlp_gflops = round(total_flops_mlp / dt / 1e9, 1)
            res.chained_mlp_ms = round(dt * 1000, 1)
            print(f"  {res.chained_mlp_gflops:.1f} GFLOPS  ({dt*1000:.1f} ms, {n_layers_mlp} layers)")
            del mlp_weights, mlp_input
        except Exception as e:
            print(f"  SKIP ({e})")

    # ----------------------------------------------------------------
    # 8. Batch prefill: large batch matmul (simulates prompt processing)
    #    (batch=64, seq=512, hidden=4096) @ (4096, 4096) — prefill territory
    # ----------------------------------------------------------------
    print("\n[8/10] Batch prefill sim (batch=64, seq=512, hidden=4096)...")
    prefill_batch = 64 * 512  # 64 prompts x 512 tokens
    prefill_hidden = 4096
    # input (batch x hidden x 2B) + weight (hidden x hidden x 2B) + output (batch x hidden x 2B)
    prefill_alloc = (prefill_batch * prefill_hidden + prefill_hidden * prefill_hidden + prefill_batch * prefill_hidden) * 2
    if not _alloc_fits(prefill_alloc, budget):
        print(f"  SKIP (needs {prefill_alloc / (1024**3):.1f} GB, exceeds budget)")
    else:
        try:
            prefill_in = Tensor.rand(prefill_batch, prefill_hidden).cast(dtypes.half).realize()
            prefill_w = Tensor.rand(prefill_hidden, prefill_hidden).cast(dtypes.half).realize()

            def batch_prefill(x=prefill_in, w=prefill_w):
                return (x @ w).realize()

            dt = _timed(batch_prefill, warmup=2, trials=6)
            prefill_flops = 2 * prefill_batch * prefill_hidden * prefill_hidden
            res.prefill_gflops = round(prefill_flops / dt / 1e9, 1)
            res.prefill_ms = round(dt * 1000, 1)
            print(f"  {res.prefill_gflops:.1f} GFLOPS  ({dt*1000:.1f} ms)")
            del prefill_in, prefill_w
        except Exception as e:
            print(f"  SKIP ({e})")

    # ----------------------------------------------------------------
    # 9. Memory capacity: largest KV cache that fits
    #    32 layers, 32 heads, 128 dim, increase seq_len until OOM
    #    Metal's 96 GB unified memory vs NV's 96 GB GDDR7
    # ----------------------------------------------------------------
    print("\n[9/10] Memory capacity (max KV cache, fp16)...")
    n_heads_cap, head_dim_cap = 32, 128
    # KV for both K and V: 2 * n_layers * n_heads * seq * dim * 2 bytes (fp16)
    # Try progressively larger seq_len, but respect the memory budget
    max_seq = 0
    for seq_try in [4096, 8192, 16384, 32768, 65536, 131072]:
        kv_bytes_try = 2 * 32 * n_heads_cap * seq_try * head_dim_cap * 2  # K+V, fp16
        kv_gb = kv_bytes_try / (1024**3)
        if not _alloc_fits(kv_bytes_try, budget):
            print(f"  seq={seq_try:>6}: {kv_gb:.1f} GB  ⊘ exceeds budget")
            break
        try:
            kv_test = Tensor.rand(2, 32, n_heads_cap, seq_try, head_dim_cap).cast(dtypes.half).realize()
            _ = kv_test.sum().numpy()
            max_seq = seq_try
            res.max_kv_seq = max_seq
            res.max_kv_gb = round(kv_gb, 2)
            print(f"  seq={seq_try:>6}: {kv_gb:.1f} GB  ✓")
            del kv_test
        except Exception:
            print(f"  seq={seq_try:>6}: {kv_gb:.1f} GB  ✗ OOM")
            break
    if max_seq > 0:
        print(f"  Max KV cache: seq={max_seq}, {res.max_kv_gb:.1f} GB (fp16, 32L x 32H x 128D)")

    # ----------------------------------------------------------------
    # 10. Reduction: layernorm-style (mean + variance over last dim)
    #    Large batch, tests reduction pipeline
    # ----------------------------------------------------------------
    print("\n[10/10] Reduction (layernorm-style, 8192x4096)...")
    rows, cols = 8192, 4096
    x = Tensor.rand(rows, cols).realize()

    def layernorm_reduce(x=x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        return ((x - mean) / (var + 1e-5).sqrt()).realize()

    dt = _timed(layernorm_reduce, warmup=2, trials=6)
    # Multiple passes over data
    data_bytes = rows * cols * 4
    res.reduce_gbs = round(data_bytes * 4 / dt / 1e9, 1)  # ~4 passes (read-heavy)
    print(f"  {res.reduce_gbs:.1f} GB/s effective  ({dt*1000:.2f} ms)")
    del x

    return res


def print_comparison(results: list[ComputeResult]) -> None:
    """Print comparison highlighting each device's strengths."""
    if len(results) < 2:
        return

    r_nv = next((r for r in results if "NV" in r.device), None)
    r_mt = next((r for r in results if "METAL" in r.device), None)
    if not r_nv or not r_mt:
        return

    print(f"\n{'='*72}")
    print("  LOADED COMPUTE COMPARISON: RTX PRO 6000 vs M3 Ultra")
    print(f"{'='*72}")

    def row(label: str, nv_val: float, mt_val: float, unit: str, higher_better: bool = True):
        if nv_val == 0 or mt_val == 0:
            winner = ""
            ratio = ""
        elif higher_better:
            if nv_val > mt_val:
                ratio = f"NV {nv_val/mt_val:.1f}x"
                winner = "◀ NV"
            else:
                ratio = f"Metal {mt_val/nv_val:.1f}x"
                winner = "Metal ▶"
        else:
            if nv_val < mt_val:
                ratio = f"NV {mt_val/nv_val:.1f}x"
                winner = "◀ NV"
            else:
                ratio = f"Metal {nv_val/mt_val:.1f}x"
                winner = "Metal ▶"
        print(f"  {label:<32} {nv_val:>10.1f} {mt_val:>10.1f} {unit:<8} {ratio:>12}  {winner}")

    header = f"  {'Metric':<32} {'NV':>10} {'Metal':>10} {'Unit':<8} {'Ratio':>12}  {'Winner'}"
    print(header)
    print("  " + "-" * 88)

    print("  ── Compute Density (NV territory) ──")
    row("Matmul 4096 FP32", r_nv.matmul_4096_gflops, r_mt.matmul_4096_gflops, "GFLOPS")
    row("Matmul 4096 FP16", r_nv.matmul_fp16_4096_gflops, r_mt.matmul_fp16_4096_gflops, "GFLOPS")
    row("Matmul 2048 FP16", r_nv.matmul_fp16_2048_gflops, r_mt.matmul_fp16_2048_gflops, "GFLOPS")
    row("Attn Q@K^T", r_nv.attn_score_gflops, r_mt.attn_score_gflops, "GFLOPS")
    row("Attn scores@V", r_nv.attn_value_gflops, r_mt.attn_value_gflops, "GFLOPS")

    print("  ── Chained On-Device Compute (NV advantage without per-op TB5 tax) ──")
    row("32-layer MLP chain", r_nv.chained_mlp_gflops, r_mt.chained_mlp_gflops, "GFLOPS")
    row("  └ latency", r_nv.chained_mlp_ms, r_mt.chained_mlp_ms, "ms", higher_better=False)
    row("Batch prefill FP16", r_nv.prefill_gflops, r_mt.prefill_gflops, "GFLOPS")
    row("  └ latency", r_nv.prefill_ms, r_mt.prefill_ms, "ms", higher_better=False)

    print("  ── Memory Bandwidth (Metal territory) ──")
    row("1 GB scan", r_nv.scan_1gb_gbs, r_mt.scan_1gb_gbs, "GB/s")
    row("4 GB scan", r_nv.scan_4gb_gbs, r_mt.scan_4gb_gbs, "GB/s")
    row("KV gather", r_nv.kv_gather_gbs, r_mt.kv_gather_gbs, "GB/s")
    row("Reduction (LN-style)", r_nv.reduce_gbs, r_mt.reduce_gbs, "GB/s")

    print("  ── Serving & Capacity ──")
    row("Decode (tok/s)", r_nv.decode_tok_per_sec, r_mt.decode_tok_per_sec, "tok/s")
    if r_nv.max_kv_seq > 0 and r_mt.max_kv_seq > 0:
        row("Max KV seq len", float(r_nv.max_kv_seq), float(r_mt.max_kv_seq), "tokens")
        row("Max KV cache", r_nv.max_kv_gb, r_mt.max_kv_gb, "GB")

    # Summary
    print(f"\n{'='*72}")
    print("  WHY THEY COMBINE")
    print(f"{'='*72}")

    nv_compute_lead = r_nv.matmul_fp16_4096_gflops / r_mt.matmul_fp16_4096_gflops if r_mt.matmul_fp16_4096_gflops > 0 else 0
    mt_bw_lead = r_mt.scan_1gb_gbs / r_nv.scan_1gb_gbs if r_nv.scan_1gb_gbs > 0 else 0

    print(f"""
  RTX PRO 6000 Blackwell (NV):
    • Raw compute monster — {nv_compute_lead:.1f}x FP16 matmul throughput
    • 24,064 CUDA cores + 4th-gen tensor cores
    • Best at: batch inference, prefill, compute-bound layers
    • Bottleneck: TB5 PCIe 4.0 x4 (~6 GB/s) for host transfers

  M3 Ultra (Metal):
    • Memory bandwidth champion — {mt_bw_lead:.1f}x effective bandwidth
    • 96 GB unified memory, zero-copy CPU↔GPU
    • Best at: serving, decode, KV cache management, large working sets
    • Bottleneck: fewer FLOPs for heavy batch compute

  Together (192 GB combined):
    • NV handles compute-heavy prefill + batch attention
    • Metal handles memory-heavy decode + KV cache serving
    • Bridge compresses KV cache for TB5 transfer: turbo3 at 4.6x means
      ~6 GB/s effective becomes ~27 GB/s effective on compressed data
    • Neither device replaces the other — they fill each other's gaps
""")


def print_scaling(results: list[ComputeResult]) -> None:
    """Print matmul scaling curve for each device."""
    for r in results:
        print(f"\n  Matmul scaling ({r.device}):")
        for N in [512, 1024, 2048, 4096]:
            gf = getattr(r, f"matmul_{N}_gflops")
            bar = "█" * int(gf / max(r.matmul_4096_gflops, 1) * 40)
            print(f"    {N:>5}: {gf:>10.1f} GFLOPS  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Loaded compute baselines")
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
    results: list[ComputeResult] = []

    for dev in devices:
        result = benchmark_compute(dev)
        results.append(result)
        print(f"\n  ✓ {dev} loaded compute baselines complete")

    print_scaling(results)
    print_comparison(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
