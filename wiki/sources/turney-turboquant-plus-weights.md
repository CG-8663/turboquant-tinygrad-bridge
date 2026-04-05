---
tags: [source, twitter, turboquant-plus, weight-compression]
date: 2026-04
url: https://x.com/no_stp_on_snek/status/2039738023944794572
author: Tom Turney (@no_stp_on_snek)
---

# Tom Turney — TurboQuant+ Weight Compression

## Summary

TurboQuant+ weight compression V1 ready for testing. Same WHT rotation + Lloyd-Max principles from KV cache compression, now applied to model weights.

## Key Claims

- Post-training, no retraining needed
- Apply directly to Q8_0 GGUFs
- Stacks with TurboQuant+ KV cache compression
- Qwen3.5-27B: 26.6G to 19.1G (-28%, +1.3% PPL)

## Significance

Extends TurboQuant from runtime-only (KV cache) to persistent compression (weights). Combined with KV compression, this compounds the memory savings and expands the model sizes feasible on constrained hardware.

## Linked Pages

- [Tom Turney](../entities/tom-turney.md)
- [TurboQuant+ Weight Compression](../concepts/turboquant-plus-weights.md)
