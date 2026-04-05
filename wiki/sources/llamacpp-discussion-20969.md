---
tags: [source, github, llama-cpp, turboquant, discussion]
date: 2026-03
url: https://github.com/ggml-org/llama.cpp/discussions/20969
author: Community
---

# TurboQuant — Extreme KV Cache Quantization (llama.cpp Discussion #20969)

## Summary

Community discussion on the official llama.cpp repo tracking TurboQuant integration. Central hub for implementation status, benchmark results, and cross-platform validation.

## Key Data Points

- turbo3 at block_size=128: 3.125 bits/val, 5.12x compression, identical PPL
- llama.cpp Metal: turbo2/3/4 supported on M1-M5 via `--cache-type-k/v` flags
- Prefill parity with q8_0 (2747 tok/s), 0.9x decode at long context
- CUDA kernel optimisations: 13-69% turbo decode improvement at 32K+
- Llama 70B Q4_K_M (40GB weights) fits in 64GB with 32K context, 6.3 tok/s on Mac Mini M4 Pro

## Multiple Implementations Referenced

- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — Metal + CUDA
- [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) — CUDA TQ3_1S/4S kernels
- [0xSero/turboquant](https://github.com/0xSero/turboquant) — Triton kernels + vLLM integration
- [scos-lab/turboquant](https://github.com/scos-lab/turboquant) — Reference implementation

## Linked Pages

- [TurboQuant Compression](../concepts/turboquant-compression.md)
- [Tom Turney](../entities/tom-turney.md)
