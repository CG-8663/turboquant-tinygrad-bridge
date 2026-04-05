---
tags: [source, twitter, validation, rtx-4090, asymmetric-kv]
date: 2026-04
url: https://x.com/no_stp_on_snek/status/2039013601965686789
author: Tom Turney (@no_stp_on_snek)
---

# Tom Turney — Independent Validation

## Summary

Independent researchers confirming TurboQuant+ findings across different hardware, backends, and model families. sztlink on GitHub published full asymmetric K/V matrix on Qwen3-4B (RTX 4090).

## Validated Claims

1. V compression tolerates more aggressive quantisation than K
2. Asymmetric K/V settings (e.g. q8_0 K + turbo3 V) give best quality/compression tradeoff
3. Findings hold across model families and hardware (not just Apple Silicon)

## Significance for Bridge

- Confirms cross-hardware consistency — same compression quality on RTX GPUs as Apple Silicon
- Validates our asymmetric format negotiation strategy in `src/tqbridge/wire.py` (K prefers precision, V prefers compression)
- CUDA kernel improvements show 13-69% turbo decode improvement at 32K+ (verified on RTX 5090, 3090 Ti, 3090, 4090M)

## Linked Pages

- [Tom Turney](../entities/tom-turney.md)
- [TurboQuant Compression](../concepts/turboquant-compression.md)
- [Cross-Backend Wire Format](../concepts/cross-backend-wire-format.md)
