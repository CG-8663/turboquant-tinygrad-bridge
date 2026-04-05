---
tags: [wire-format, dma, tb5, bridge, architecture]
sources: 2
last_updated: 2026-04-04
---

# Cross-Backend Wire Format

Core innovation of this project. Uses [TurboQuant compressed KV blocks](turboquant-compression.md) directly as the DMA wire format between Metal and CUDA backends over Thunderbolt 5.

## Why This Works

TurboQuant's compressed blocks are byte-identical on both backends:
- Same ggml block structs from `ggml-common.h`
- Same deterministic WHT rotation seed
- Same Lloyd-Max codebook (deterministic given bit-width and dimension)

No custom serialization. No format conversion. The compressed block IS the wire format.

## Wire Savings

TB5 PCIe 4.0 x4 maxes out at ~6 GB/s. For 70B at 32K context:

| Approach | Data Size | Transfer Time |
|----------|-----------|---------------|
| FP16 blocking | ~2.56 GB | ~427 ms |
| turbo3 full blocks (128B) | ~555 MB | ~93 ms |
| turbo3 stripped blocks (16B) | ~69 MB | ~12 ms |
| turbo3 stripped + double-buffered | ~69 MB | **~40 ms end-to-end** |

Stripping removes GPU cache-line padding (128B on-GPU blocks to 16B wire blocks) — 7x less data.

## Double-Buffer Pipeline

```
Metal: Compress layer N -> Buf A --DMA--> CUDA: Decompress layer N
       Compress layer N+1 -> Buf B        Compute layer N
       (overlap compress+DMA with decompress+compute)
```

Per-layer timing (70B, 32K, turbo3): ~1.6 ms/layer effective vs 5.3 ms/layer blocking FP16.

## Implementation

Wire protocol defined in `src/tqbridge/wire.py`:
- 40-byte header with magic `0x54514B56` ("TQKV"), CRC32 validation
- Asymmetric format negotiation (K prefers precision, V prefers compression)
- Supports turbo2/3/4, q8_0, Q5_K_M, FP16

## Dependency: Chronara Flywheel

This wire format is a direct dependency of the [chronaracli flywheel project](/Volumes/Chronara-Storage-1/Projects/chronaracli/). The flywheel's LLM Wiki quality gate validates that cross-backend inference with compressed KV produces identical downstream output to single-backend inference. See [LLM Wiki Pattern](llm-wiki-pattern.md).

## See Also

- [TurboQuant Compression](turboquant-compression.md)
- [MLX Cross-Backend](mlx-cross-backend.md)
- RFC: `docs/RFC-double-buffer-kv-bridge.md`
- RFC: `docs/RFC-metal-cuda-kv-bridge.md`
