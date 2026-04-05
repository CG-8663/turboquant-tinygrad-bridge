---
tags: [compression, kv-cache, turbo3, turbo4, iclr-2026]
sources: 3
last_updated: 2026-04-04
---

# TurboQuant Compression

Google Research paper, ICLR 2026 ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

## What It Does

Compresses LLM KV cache at runtime to 2-4 bits per element. No retraining, no new model files. Applied during inference by adding `--cache-type-k turbo3 --cache-type-v turbo3` to llama-server.

## Compression Levels

| Format | Bits | Compression | PPL (wikitext-2) | vs q8_0 |
|--------|------|-------------|-----------------|---------|
| q8_0 | 8 | 1.9x | 6.111 | baseline |
| turbo4 | 4 | 3.8x | 6.125 | +0.23% |
| turbo3 | 3 | 4.6-5.1x | 6.176 | +1.06% |
| turbo2 | 2 | 6.4x | higher | significant |

Needle-In-A-Haystack retrieval: 100% accuracy through 32K context for turbo3 and turbo4.

## How It Works

1. **WHT rotation** — Walsh-Hadamard Transform with deterministic seed. Spreads information across all dimensions so no single element is disproportionately important.
2. **Lloyd-Max codebook** — Optimal centroids for N(0,1/d) distribution. Deterministic given bit-width and dimension.
3. **Block quantisation** — On-GPU blocks are 128 bytes (cache-line aligned). Wire blocks can be stripped to 16 bytes (turbo3) by removing padding.

## Key Property for Bridge

The codebook and rotation are deterministic and identical on Metal and CUDA. Same seed, same ggml block structs from `ggml-common.h`. This means compressed blocks are byte-identical across backends — no format conversion needed for [cross-backend wire format](cross-backend-wire-format.md).

## Performance on Apple Silicon

- 70B turbo3 prefill at 32K: 80.8 t/s (faster than q8_0 at 75.2 t/s)
- 104B at 128K on MacBook: PPL 4.024, 74GB peak memory
- llama.cpp Metal: turbo3 prefill parity with q8_0 (2747 tok/s), 0.9x decode at long context

## CUDA Performance

- turbo decode 13-69% improvement at 32K+ context over base implementation
- Verified on RTX 5090, 3090 Ti, 3090, 4090M
- Advantage shows at 32K+ where KV bandwidth dominates

## Implementations

- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — Metal + CUDA kernels (submodule in this repo)
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — Reference implementation, benchmarks
- [scos-lab/turboquant](https://github.com/scos-lab/turboquant) — Independent reference implementation
- [0xSero/turboquant](https://github.com/0xSero/turboquant) — Triton kernels + vLLM integration
- [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) — CUDA TQ3_1S/4S kernels

## See Also

- [TurboQuant+ Weight Compression](turboquant-plus-weights.md)
- [Cross-Backend Wire Format](cross-backend-wire-format.md)
- [Tom Turney](../entities/tom-turney.md)
