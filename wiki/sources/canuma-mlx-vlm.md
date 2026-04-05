---
tags: [source, github, mlx-vlm, vision-language-models, fine-tuning]
date: 2026-04
url: https://github.com/Blaizzy/mlx-vlm
author: Prince Canuma (@Prince_Canuma)
---

# Prince Canuma — MLX-VLM

## Summary

MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) and Omni Models on Mac using MLX. 1,800+ GitHub stars, 204 forks.

## Key Details

- Supports inference and fine-tuning on Apple Silicon
- Extras: `ui`, `cuda`, `cpu` — confirms multi-backend support
- Day-0 model support (Gemma 4, Qwen2.5-VL, etc.)
- mxfp8/nvfp4 quantised models on NVIDIA require activation quantisation flag; on Metal they work without it

## Significance for Bridge

The cuda/cpu extras demonstrate MLX's cross-backend inference is production-ready. The mxfp8/nvfp4 asymmetry informs our wire protocol's format negotiation — backend-specific handling is already an established pattern in the ecosystem.

## Linked Pages

- [Prince Canuma](../entities/prince-canuma.md)
- [MLX Cross-Backend](../concepts/mlx-cross-backend.md)
