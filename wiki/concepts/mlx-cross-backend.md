---
tags: [mlx, apple-silicon, cuda, cross-backend, inference]
sources: 2
last_updated: 2026-04-04
---

# MLX Cross-Backend

Apple's MLX framework supports both Metal (Apple Silicon) and CUDA (NVIDIA GPU) backends.

## Installation

```bash
# Metal (default on Apple Silicon)
pip install mlx

# CUDA
pip install mlx[cuda]
```

## Key Details

- [Prince Canuma](../entities/prince-canuma.md) has built the largest MLX model ecosystem (1,000+ models)
- MLX-VLM supports `ui`, `cuda`, and `cpu` extras
- Voxtral runs on both Apple Silicon and NVIDIA GPUs via MLX
- Models quantised with mxfp8 or nvfp4 on NVIDIA require activation quantisation flag; on Metal they work without it

## Relevance to Bridge

MLX's cross-backend capability validates that the same model can run on both Metal and CUDA. Our bridge extends this — instead of running the full model on one backend, we split it across both and stream compressed KV cache between them via [cross-backend wire format](cross-backend-wire-format.md).

The mxfp8/nvfp4 asymmetry is relevant to our format negotiation layer in `src/tqbridge/wire.py` — the bridge must handle backend-specific quantisation requirements.

## See Also

- [Prince Canuma](../entities/prince-canuma.md)
- [Cross-Backend Wire Format](cross-backend-wire-format.md)
