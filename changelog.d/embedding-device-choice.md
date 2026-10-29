### Changed

- **The embedding model's device is the person's choice** (operator ruling 2026-10-29). `EmbeddingManager(device=...)`
  or the embedding route's `options.device` ("gpu" / "cpu"; the route's option example is `{"device": "gpu"}`); unset
  means the GPU when there is room. On the GGUF lane "cpu" stays all on the CPU (no KV or compute buffers on the
  card), "gpu" that does not fit is refused with the sentence ("... could not be placed on the GPU ...; free GPU
  memory or set its Device to CPU"), and the default falls back to the CPU and says so (`device_note`). The GGUF
  embedder measures what it took on the GPU (`device_bytes` on its residency claim).

### Fixed

- **A text longer than the embedding model reads is refused, never cut** (framework ADR 0026). The GGUF embedding lane
  had a 4,096-token context and llama-cpp-python's `embed()` truncated by default; it now loads at the model's own
  sequence length (8,192 for Qwen3-Embedding-0.6B), embeds with `truncate=False`, and a longer text raises
  `ContextWindowExceededError` ("This text is N tokens, but the embedding model X reads at most M; split it before
  embedding it (nothing was cut).").
- **The sentence-transformers embedding lane never cuts a text either.** The library truncated every text at its
  `max_seq_length` (model-defined, often 512) silently, and a non-strict manager turned failures into zero vectors. The
  model now reads its full length (the catalog's `max_sequence_length`, e.g. 8,192 for Qwen3-Embedding), a text one token
  over raises `ContextWindowExceededError` with the same sentence (counted with the model's own tokenizer, specials
  included), and that error is never replaced by a zero vector.
