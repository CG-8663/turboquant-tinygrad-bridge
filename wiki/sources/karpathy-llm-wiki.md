---
tags: [source, pattern, knowledge-management, training, quality-gate]
date: 2026-04
url: https://github.com/karpathy/llm-wiki
author: Andrej Karpathy (@karpathy)
---

# Karpathy — LLM Wiki Pattern

## Summary

Abstract framework for building personal knowledge bases using LLMs. Three layers (raw sources, wiki, schema), three operations (ingest, query, lint). The LLM maintains the wiki; the human curates sources and asks questions.

## Core Insight

RAG re-derives knowledge from scratch on every query. A wiki compiles once and keeps getting richer. Cross-references already exist. Contradictions already flagged. The wiki is a persistent, compounding artifact.

## Key Quotes

- "The tedious part of maintaining a knowledge base is not the reading or the thinking — it's the bookkeeping."
- "LLMs don't get bored, don't forget to update a cross-reference, and can touch 15 files in one pass."
- "The human's job is to curate sources, direct the analysis, ask good questions, and think about what it all means. The LLM's job is everything else."

## Integration with Chronara Projects

### Direct Dependency: chronaracli Flywheel

This wiki pattern is being implemented as a layer in the [chronaracli flywheel training pipeline](/Volumes/Chronara-Storage-1/Projects/chronaracli/). The flywheel's raw sources (Elasticsearch interaction logs, MongoDB metadata) feed into an LLM-maintained wiki that produces structured training signal for QLoRA fine-tuning on the GX10 cluster.

**The TurboQuant bridge is a direct dependency of the flywheel** — cross-device inference capacity is needed to run the training loop at scale.

### Quality Gate for Bridge Validation

Same sources ingested through Metal-only, CUDA-only, and split inference paths. Identical wiki output = compressed KV wire format doesn't degrade downstream quality.

## Linked Pages

- [Andrej Karpathy](../entities/andrej-karpathy.md)
- [LLM Wiki Pattern](../concepts/llm-wiki-pattern.md)
- [Cross-Backend Wire Format](../concepts/cross-backend-wire-format.md)
