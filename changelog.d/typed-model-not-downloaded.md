### Added

- **A model that is not downloaded raises `ModelNotDownloadedError`, which names it** (`abstractcore.exceptions`).
  A subclass of `ModelNotFoundError` with `model` (the id to download) and `provider` (the provider that serves
  it) as attributes, so a host says "download X" without parsing the message (the gateway recovered the embedding
  model's id from the message text). The messages keep their wording. Raised where a local-only load finds
  nothing on this computer: embeddings (`EmbeddingManager`), transformers (not cached, or a download that did not
  finish), GGUF, MLX, and an MLX drafter. Existing `except ModelNotFoundError` handlers still catch it.

### Changed

- **A GGUF model that is not downloaded now reaches the caller as `ModelNotDownloadedError`**, not as a
  `RuntimeError("Failed to load GGUF model ...")` wrapping it; the message is the same "not found in local caches"
  text without that prefix. An MLX model that is not downloaded keeps its own message (the caches searched and the
  download tip) instead of the generic "Model not found for MLX provider" list.
