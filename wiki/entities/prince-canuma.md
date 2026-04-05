---
tags: [person, mlx, apple-silicon, vision-language-models, cross-backend]
sources: 3
last_updated: 2026-04-04
---

# Prince Canuma (@Prince_Canuma)

GitHub: [Blaizzy](https://github.com/Blaizzy)

## Role

ML engineer and open-source developer. Most prolific contributor to Apple's MLX ecosystem. Published 1,000+ models. Maintains MLX-VLM (1,800+ GitHub stars), MLX-Audio, and MLX-Embeddings.

## Key Projects

- **[MLX-VLM](https://github.com/Blaizzy/mlx-vlm)** — Vision Language Models on Apple Silicon. Supports inference and fine-tuning. Day-0 support for new model releases (Gemma 4, Qwen2.5-VL, etc.)
- **[MLX-Audio](https://github.com/Blaizzy/mlx-audio)** — TTS, STT, STS on MLX framework
- **Voxtral on MLX** — Inference on both Apple Silicon AND NVIDIA GPUs
- **MLX CUDA support** — `pip install mlx[cuda]` enables cross-backend inference

## Relevance to Bridge

His work proves MLX runs on both Metal and CUDA backends. MLX-VLM supports extras for `ui`, `cuda`, and `cpu`. This cross-backend capability is the foundation for our bridge's Metal-side inference — if the compressed KV format works across MLX backends, it validates our wire format assumption independently.

Key detail: models quantised with mxfp8 or nvfp4 on NVIDIA GPUs require activation quantisation flag. On Apple Silicon (Metal), they work without it. This asymmetry is relevant to our format negotiation layer.

## What He Hasn't Explored (Our Opportunity)

- Compressed KV cache streaming between MLX Metal and CUDA backends
- TB5 as an inference interconnect
- Split-model inference over hardware links using compressed wire format

## Sources

- [Voxtral on MLX announcement](../sources/canuma-voxtral-mlx.md)
- [MLX-VLM](../sources/canuma-mlx-vlm.md)
- [MLX-Audio](../sources/canuma-mlx-audio.md)
