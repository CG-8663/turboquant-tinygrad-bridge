---
tags: [source, twitter, mlx, voxtral, cross-backend]
date: 2026-03
url: https://x.com/Prince_Canuma/status/1957448874085236918
author: Prince Canuma (@Prince_Canuma)
---

# Prince Canuma — Voxtral on MLX

## Summary

Voxtral now runs on MLX with support for both Apple Silicon and NVIDIA GPUs.

## Key Details

- Install: `pip install -U mlx-audio`
- CUDA: `pip install mlx[cuda] --force`
- Demonstrates MLX cross-backend capability in production

## Significance for Bridge

Proves MLX can serve the same model on both Metal and CUDA backends. Our bridge extends this from full-model inference on one backend to split-model inference across both.

## Linked Pages

- [Prince Canuma](../entities/prince-canuma.md)
- [MLX Cross-Backend](../concepts/mlx-cross-backend.md)
