---
tags: [source, benchmarks, apple-silicon, mac, performance]
date: 2026-03
url: https://asiai.dev/turboquant/
author: asiai.dev
---

# TurboQuant Benchmarks — Apple Silicon

## Summary

Independent benchmark analysis of TurboQuant on Apple Silicon Macs. Covers real-world performance across model sizes.

## Key Results

- Llama 70B Q4_K_M fits in 64GB with 32K context using turbo3 KV cache
- 6.3 tok/s measured on Mac Mini M4 Pro
- 104B at 128K context on MacBook: PPL 4.024, 74GB peak memory
- Prefill: 70B turbo3 at 32K = 80.8 t/s vs 75.2 t/s for q8_0 (turbo3 is FASTER)

## Significance for Bridge

These benchmarks establish the Metal-side performance baseline. Our bridge's Metal endpoint (M3 Ultra, 96GB) should exceed these numbers given more memory headroom. The prefill speedup with turbo3 is notable — compression isn't just saving memory, it's faster than uncompressed because of reduced memory bandwidth pressure.

## Linked Pages

- [TurboQuant Compression](../concepts/turboquant-compression.md)
