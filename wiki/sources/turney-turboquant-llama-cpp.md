---
tags: [source, twitter, turboquant, llama-cpp, metal]
date: 2026-03
url: https://x.com/no_stp_on_snek/status/2036792058854121601
author: Tom Turney (@no_stp_on_snek)
---

# Tom Turney — TurboQuant llama.cpp Implementation

## Summary

Tom Turney announced his implementation of Google's TurboQuant paper (ICLR 2026) in llama.cpp with Metal kernels for Apple Silicon.

## Key Claims

- 4.9x KV cache compression achieved
- Working end-to-end on M5 Max with Qwen 3.5 35B MoE and Qwopus v2 27B
- Speed needs work (unoptimized shader), compression target met
- Repo: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)

## Additional Context (from follow-up tweets)

- Compression happens at runtime, not on model weights
- No new model files needed — use existing GGUFs
- Activation: `--cache-type-k turbo3 --cache-type-v turbo3` on llama-server
- Source: https://x.com/no_stp_on_snek/status/2036962332866183344

## Linked Pages

- [Tom Turney](../entities/tom-turney.md)
- [TurboQuant Compression](../concepts/turboquant-compression.md)
