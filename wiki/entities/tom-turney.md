---
tags: [person, turboquant, llama-cpp, metal, apple-silicon]
sources: 3
last_updated: 2026-04-04
---

# Tom Turney (@no_stp_on_snek)

GitHub: [TheTom](https://github.com/TheTom)

## Role

Primary implementer of Google's [TurboQuant](../concepts/turboquant-compression.md) paper (ICLR 2026) in llama.cpp. Built Metal GPU kernels for Apple Silicon. Maintains the [turboquant_plus](https://github.com/TheTom/turboquant_plus) reference implementation and [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork.

## Key Achievements

- 4.9x KV cache compression working end-to-end on M5 Max
- Metal kernels for turbo2, turbo3, turbo4 on Apple Silicon (M1-M5)
- 104B model at 128K context on a MacBook — PPL 4.024, 74GB peak memory
- 511+ Python tests, 100% code coverage on diagnostics
- Prefill performance: 70B turbo3 at 32K context: 80.8 t/s vs 75.2 t/s for q8_0
- [TurboQuant+ weight compression](../concepts/turboquant-plus-weights.md) — WHT + Lloyd-Max applied post-training to Q8_0 GGUFs

## Relevance to Bridge

His implementation proves turbo3/4 blocks are byte-identical on Metal and CUDA — same ggml block structs from `ggml-common.h`, same WHT rotation seed, same Lloyd-Max codebook. This is the foundational assumption of our [cross-backend wire format](../concepts/cross-backend-wire-format.md).

The llama-cpp-turboquant submodule in this repo is his fork.

## What He Hasn't Explored (Our Opportunity)

- Cross-device streaming of compressed KV cache as a wire format
- Using turbo3/4 blocks for inter-device DMA over TB5
- Metal-to-CUDA heterogeneous split inference

## Sources

- [TurboQuant llama.cpp implementation announcement](../sources/turney-turboquant-llama-cpp.md)
- [TurboQuant+ weight compression](../sources/turney-turboquant-plus-weights.md)
- [Independent validation thread](../sources/turney-independent-validation.md)
