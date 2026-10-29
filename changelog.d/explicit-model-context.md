### Added

- **The reserved context is explicit and the person's** (ADR 0010 point 8, operator ruling 2026-10-29). Residency claims
  of a loaded GGUF carry `advertised_context_length` and `context_source` (`memory`: the fit rule / allocation ladder
  chose it; `configured`: a person set it; `explicit`: the caller passed `max_tokens`) beside `context_length`, so the
  Models page can say "Context 65,536 (model advertises 262,144; limited by memory)". A person sets a model's context
  with `abstractcore.config.model_contexts.set_model_context(provider, model, tokens)` (config key `model_contexts`):
  it is loaded as is and refused with the readable sentence when it does not fit, never shrunk. A request longer than
  the loaded context still fails with `ContextWindowExceededError`; nothing is cut.

### Changed

- **A GGUF reserves the maximum context that fits, computed** (operator ruling 2026-10-29). The smaller of the advertised
  context and the largest whose weights + KV + compute fit the free device memory minus a 5 % safety buffer
  (`model_memory_buffer`), by inverting the KV-size formula; an allocation that still fails halves the context. This
  replaces the "KV at most half of the free memory" rule and the fixed 131,072 / 65,536 / ... rungs, and the
  calibration seed no longer shrinks a load. Measured numbers for Kimi-9B on the 16 GB card: about 135k of 262,144
  (was 65,536). Logged, and on the claim as `context_reservation`.
