### Added

- **The reserved context is explicit and the person's** (ADR 0010 point 8, operator ruling 2026-10-29). Residency claims
  of a loaded GGUF carry `advertised_context_length` and `context_source` (`memory`: the fit rule / allocation ladder
  chose it; `configured`: a person set it; `explicit`: the caller passed `max_tokens`) beside `context_length`, so the
  Models page can say "Context 65,536 (model advertises 262,144; limited by memory)". A person sets a model's context
  with `abstractcore.config.model_contexts.set_model_context(provider, model, tokens)` (config key `model_contexts`):
  it is loaded as is and refused with the readable sentence when it does not fit, never shrunk. A request longer than
  the loaded context still fails with `ContextWindowExceededError`; nothing is cut.
