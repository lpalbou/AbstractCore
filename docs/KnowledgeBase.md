# KnowledgeBase

This document collects **durable, cross-cutting insights** about AbstractCore’s design and operational behavior.
It is intentionally brief and additive: we append new learnings, and deprecate by moving to a dedicated section.

## Capabilities database: embedding models

- **Canonical signal**: Embedding-only models should be marked with `model_type: "embedding"` in `abstractcore/assets/model_capabilities.json`.
- **Neutral generation fields**: For embedding models, set `max_output_tokens: 0`, `tool_support: "none"`, and `structured_output: "none"` (they do not produce chat text and do not call tools).
- **Alias hygiene**: Provider-specific identifiers (e.g. LMStudio quantization suffixes like `@q6_k`) should be added as aliases of the canonical model entry so architecture detection does not fall back to generative defaults.

## DEPRECATED

(none yet)

