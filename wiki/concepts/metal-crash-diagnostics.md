# Metal Crash Diagnostics

> Documented: 2026-04-05 — from repeated crashes during TurboQuant Metal benchmarking

## Summary

Running GPU benchmarks on the Metal backend (M3 Ultra, tinygrad) produces intermittent crashes. Root cause analysis identified **5 crash vectors** in the tinygrad Metal runtime and our benchmark scripts.

## Crash Vectors

### 1. Command Buffer Queue Overflow (HIGH)

**Location:** `tinygrad/runtime/ops_metal.py:34`

Metal command queues are created with a 1024 buffer limit. Every `.realize()` appends a command buffer to `mtl_buffers_in_flight`, but this list only clears on explicit `synchronize()`. Long benchmark loops without sync exhaust the queue.

**Fix:** Sync after warmup loops. Add periodic `Device['METAL'].synchronize()` in long-running tests.

### 2. ICB Pipeline Use-After-Free (HIGH on M1/M2)

**Location:** `tinygrad/runtime/graph/metal.py:29,83`

Metal Indirect Command Buffers can reference deallocated pipeline states on pre-M3 GPUs (M1/M2 families). Tinygrad auto-detects this via GPU family regex (`AGXG(\d+)XFamily`). M3+ (AGXG15+) is not affected.

**Fix:** Auto-detection works for M3 Ultra. Set `FIX_METAL_ICB=1` on M1 Max (32 GB) or CI runners.

### 3. Memory Budget Not Enforced

**Location:** `benchmarks/baseline_device.py`

The benchmark computes a memory budget (80% of total) but never checks allocations against it. Sequential 256 MB + 1 GB allocations can push Metal into an unstable state.

**Fix:** Port `_alloc_fits()` from `baseline_compute.py` and check before each allocation.

### 4. Warmup Without Sync

**Location:** `benchmarks/baseline_device.py:91-93`

Warmup loops call `.realize()` without `.numpy()`, accumulating in-flight buffers before timed trials even start.

**Fix:** Add `.numpy()` or explicit sync after warmup phases.

### 5. Paravirtualized Metal Detection

**Location:** `tinygrad/runtime/ops_metal.py:43-46`

Simple `'virtual' not in name` check can miss non-standard virtualized environments.

**Fix:** Not a concern on bare metal M3 Ultra. For CI, set `METAL_GRAPH=0`.

## Recommended Environment

```bash
# For M3 Ultra (local development)
export DEV=METAL

# For M1 Max (32 GB) or CI runners
export DEV=METAL
export FIX_METAL_ICB=1
export METAL_GRAPH=0  # if paravirtualized
```

## Related

- [TurboQuant Compression](turboquant-compression.md) — compression pipeline (CPU-safe)
- [Cross-Backend Wire Format](cross-backend-wire-format.md) — wire protocol (CPU-safe)
- Full diagnostics: [`tests/test.md`](../../tests/test.md)
