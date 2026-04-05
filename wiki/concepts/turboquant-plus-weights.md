---
tags: [compression, weights, turboquant-plus, post-training]
sources: 1
last_updated: 2026-04-04
---

# TurboQuant+ Weight Compression

Extension of [TurboQuant compression](turboquant-compression.md) from KV cache to model weights. Announced by [Tom Turney](../entities/tom-turney.md) as V1 ready for testing.

## How It Differs from KV Cache Compression

Same WHT rotation + Lloyd-Max principles, applied post-training directly to Q8_0 GGUFs. No retraining needed. Stacks with TurboQuant KV cache compression for compounding savings.

## Results

- Qwen3.5-27B: 26.6G to 19.1G (-28%, +1.3% PPL)
- Applied to existing Q8_0 model files — no new training run needed

## Independent Validation

sztlink on GitHub published asymmetric K/V matrix on Qwen3-4B (RTX 4090) confirming:
1. V compression tolerates more aggressive quantisation than K
2. Asymmetric K/V settings (e.g. q8_0 K + turbo3 V) give best quality/compression tradeoff
3. Findings hold across model families and hardware

## Relevance to Bridge

Weight compression reduces the model size that needs to fit on each device in a split inference setup. Combined with KV cache compression for the wire format, this significantly expands the model sizes feasible on our 192GB combined pool (96GB M3 Ultra + 96GB RTX PRO 6000).

## Source

- [Tom Turney — TurboQuant+ Weight Compression](../sources/turney-turboquant-plus-weights.md)
