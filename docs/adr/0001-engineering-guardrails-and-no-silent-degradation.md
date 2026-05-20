# ADR 0001: Engineering guardrails and no silent degradation

Status: Accepted.

## Context

AbstractCore deliberately spans heterogeneous providers, local runtimes, multimodal inputs,
structured outputs, tool calling, and server-mode routing. That breadth creates recurring pressure
to smooth over provider gaps with compatibility shims, retries, fallback prompts, sampled media, or
looser parsing.

Some of that behavior is useful. The risk is silent degradation: a caller believes they received a
native or complete result when the system actually used a weaker path, changed the request, clipped
content, retried until a timeout, or skipped an unsupported capability.

The current code already has several explicit examples that should remain visible instead of being
hidden: fallback documentation in `docs/fallbacks.md`, timeout normalization in
`abstractcore/providers/base.py`, known structured-output timeout notes in
`docs/known_bugs/structured-timeout.md`, and explicit `UnsupportedFeatureError` paths for media
fallback limitations.

## Decision

AbstractCore will treat fallback, truncation, timeout, and degraded-mode behavior as observable
contracts, not as invisible repair logic.

New or changed behavior must follow these rules:

- Prefer backend-native controls before prompt markers, request rewriting, or compatibility shims.
- Warn, annotate metadata, document, or fail explicitly when behavior becomes best-effort.
- Fail closed on correctness-critical paths where degraded output would be misleading.
- Preserve timeout context when the system knows the configured duration or responsible component.
- Avoid silent lossy truncation on user-visible, model-visible, or correctness-critical content.

Compatibility shims are allowed when they are the least harmful current option, but their scope and
removal path must be clear.

## Consequences

### Positive

- Users and operators can tell when behavior is native, best-effort, unsupported, or timed out.
- Provider-specific gaps can be improved incrementally without pretending they are solved.
- Future native support can replace explicit shims cleanly.

### Negative

- Some paths remain visibly inconvenient until the backend exposes a better surface.
- More code and docs changes must account for warnings, errors, or metadata.

### Neutral

- This ADR does not ban fallbacks.
- This ADR does not require every provider to support every feature.

## Enforcement

- Reviews should reject silent fallback that changes correctness, trust, routing, or output shape.
- New fallback behavior must be documented in the same change unless it is purely internal and
  impossible for callers to observe.
- Timeout changes must preserve useful duration/source context where available.
- Truncation or preview behavior must be explicit in names, metadata, warnings, or docs.

## Validation

- Targeted tests should assert explicit unsupported errors, warning paths, timeout messages, or
  degraded metadata for changed fallback behavior.
- Documentation checks should verify that new fallbacks appear in the relevant user or developer
  docs.
- Completion notes for fallback-heavy work should state what remains best-effort.

## Related

- `abstractcore/providers/base.py`
- `abstractcore/providers/lmstudio_provider.py`
- `abstractcore/media/vision_fallback.py`
- `docs/fallbacks.md`
- `docs/known_bugs/structured-timeout.md`
- `docs/backlog/planned/2026-05-06_robust-fallback-generate.md`
