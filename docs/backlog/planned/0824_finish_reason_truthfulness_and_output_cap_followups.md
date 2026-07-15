# Planned: finish_reason truthfulness + output-cap follow-ups

## Metadata
- Created: 2026-07-15
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: no-silent-degradation (ADR 0001)
- ADR impact: the "no silent truncation" guarantee (0822) only holds where
  `finish_reason` is truthful; this item closes the remaining lanes.

## Context
Backlog 0822 made the output cap omit-when-unspecified and added
`_annotate_output_truncation` (warn + `metadata["output_truncated"]` on
`finish_reason=length`). A fable5 red-team (2026-07-15) confirmed the resolver
core is sound and live-verified omit-safety on OVH vLLM, LM Studio server, and
OpenRouter (incl. Anthropic-via-OpenRouter). It also surfaced follow-ups beyond
what the same-day fix closed. ALREADY FIXED in that pass: async-lane annotation
(F1 — `agenerate` + async stream now annotate), streaming finalize (F3 —
processor emits the real last-seen finish_reason), the TGI-class omit trap (F5 —
instance `requires_output_cap` knob), Ollama NON-streaming `done_reason` mapping
(F2-partial), and the gpt-oss-120b/20b registry under-cap (8192 → 128000).

## Remaining work (this item)
1. **finish_reason truthfulness on the rest of the lanes (F2).** These providers
   hardcode `finish_reason` and never read the backend's real stop reason, so
   `_annotate_output_truncation` is a structural no-op there:
   - Ollama STREAMING terminal chunk (sync + async) — read `done_reason` on the
     final chunk (the non-streaming path is already mapped via
     `_ollama_finish_reason`).
   - MLX (`mlx_provider.py` ~2093, 2167): hardcodes "stop"/None — derive length
     from whether generation hit `max_tokens`.
   - HuggingFace transformers/GGUF: hardcodes — same.
   - Anthropic SYNC STREAMING terminal chunk (`anthropic_provider.py` ~961):
     map `message_delta.stop_reason` (non-streaming already maps it ~837).
   Highest priority: Ollama (default provider of the processing apps —
   `basic_extractor.py`), then Anthropic stream, then MLX/HF.
2. **Constructor-explicit cap clamp (F4).** The clamp ceiling in
   `_prepare_generation_kwargs` is `self.max_output_tokens` itself, so a
   constructor cap is clamped against itself (not the registry hard max);
   per-call caps clamp correctly. Decide the intended semantics: an explicit
   user cap arguably should NOT be silently reduced (no-silent-budget), but
   Anthropic 400s on over-set — so clamp to the TRUE registry/API hard max, not
   a conservative default, and only where the API mandates it.
3. **Runaway/cost note (F6).** On the omit lane, repetition-looping local models
   free-run to context fill (previously 2048-bounded); processing-app chunk
   loops inherit this per chunk. Consider a documented soft cap for local
   generators, or rely on `_requires_output_cap` per deployment. Cosmetic: the
   CLI's `setattr(provider, "max_output_tokens", …)` (`utils/cli.py` ~443, 481)
   no longer reaches the wire on omit-lane providers (the explicit sentinel
   stays False); and per-call `max_output_tokens=None` cannot override a
   constructor cap back to "omit". Align the CLI attribute-write with the
   sentinel.

## Current code reality
- `base.py`: `_resolve_output_token_cap`, `_requires_output_cap` (+ instance
  `_requires_output_cap_override`), `_annotate_output_truncation`,
  `_annotate_async_stream`.
- `ollama_provider.py`: `_ollama_finish_reason` (non-streaming only so far).
- `streaming.py`: finalize chunk now carries last-seen finish_reason.

## Validation
- A `finish_reason=length` from each provider lane (unit-mockable) sets
  `output_truncated` + warns, on sync AND streaming, sync AND async.
- No regression to omit-when-unspecified or explicit-cap-honored behavior.
