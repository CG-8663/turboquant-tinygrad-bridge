# TriAttention V3 Hybrid Model Fix — Analysis for Tom

## The Failure Mode

On Qwen3.5-27B (16/64 attention layers, rest Mamba2):
- PPL: fine (+0.29%)
- NIAH: fails at middle and end positions
- The model reaches for semantic neighbors instead of the actual needle

## Root Cause Hypothesis

**The eviction budget doesn't account for attention layer sparsity.**

On Qwen2.5-7B: 32/32 layers are attention. Each KV token appears in 32 attention layers. Evicting 10% of tokens removes 10% of attention signal.

On Qwen3.5-27B: 16/64 layers are attention. Each KV token appears in only 16 attention layers. But each attention layer is **4x more critical** because the model has 4x fewer attention layers to retrieve information from.

Evicting 10% of tokens on the hybrid model is equivalent to evicting 40% of the effective attention capacity. That's why middle/end needles disappear — the scoring correctly identifies them as low-trig-score tokens (they're isolated facts, not part of the dominant attention pattern), but removing them is catastrophic because the model has so few attention layers to fall back on.

## Proposed Fix: Scale Budget by Attention Fraction

```
attention_fraction = n_attention_layers / n_total_layers
effective_budget = 1.0 - (1.0 - raw_budget) * attention_fraction

Qwen2.5-7B:  attention_fraction = 32/32 = 1.0
  raw_budget = 0.90 → effective_budget = 0.90 (unchanged)

Qwen3.5-27B: attention_fraction = 16/64 = 0.25
  raw_budget = 0.90 → effective_budget = 1.0 - 0.10 * 0.25 = 0.975 (97.5%)

Qwen3.5-35B: attention_fraction = 10/40 = 0.25
  raw_budget = 0.90 → effective_budget = 0.975 (97.5%)
```

At 97.5% retention on the hybrid model, you're evicting only 2.5% of tokens — which is the equivalent pressure as 10% on a full transformer.

## Why This Should Work

1. **PPL won't change much** — going from 90% to 97.5% retention on PPL that's already +0.29% will stay flat
2. **NIAH should recover** — the needle survives because you're removing 4x fewer tokens
3. **Still saves memory** — 2.5% eviction × 4.6x TurboQuant = still meaningful combined compression
4. **Auto-scales** — the formula adapts to any attention/Mamba ratio

## Alternative: Per-Layer-Type Budget

Instead of a single global budget, split the eviction:

```
attention_layer_budget = 0.975  (protect attention KV heavily)
mamba_layer_budget = N/A        (Mamba state not in KV cache)
```

This is equivalent to the above formula but makes the intent clearer in the code.

## Second Issue: Partial RoPE

Qwen3.5 uses partial RoPE — only 64/256 head dimensions are rotated. The trig scoring formula averages across all frequency bins, but on partial RoPE, 75% of those bins contribute zero signal (no rotation = no position encoding = no trig score difference between tokens).

**Fix**: In the scoring inner loop, only iterate over the rotated frequency bins (the first 64/2 = 32 frequencies), not all head_dim/2 = 128.

```c
// Current (broken on partial RoPE):
for (int f = 0; f < freq_count; f++) {  // freq_count = head_dim/2 = 128

// Fixed:
int rope_dims = model->hparams.n_rot;  // = 64 for Qwen3.5
int freq_count = rope_dims / 2;         // = 32
for (int f = 0; f < freq_count; f++) {
```

This makes the trig score 4x less noisy on Qwen3.5 because it stops averaging in 96 dimensions of pure noise.

## For TQBridge

If both fixes work, the TQBridge value proposition becomes:

| Workload | TurboQuant | TriAttention V3 | Combined | Transfer per token |
|----------|-----------|-----------------|----------|-------------------|
| Standard 7B, 32K | 4.6x | 1.1x (90%) | 5.1x | 50KB → 10KB |
| Reasoning 7B, 32K | 4.6x | ~5x (estimated) | ~23x | 50KB → 2.2KB |
| Hybrid 27B, 32K | 4.6x | 1.03x (97.5%) | 4.7x | 50KB → 10.6KB |

The reasoning case is where TQBridge + TriAttention really shines — 23x compression means a 27B model's KV cache transfers at 2.2KB per token over the network. That's viable over WiFi.
