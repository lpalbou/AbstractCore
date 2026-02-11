# Docs Review: `docs/architecture.md` and Media Docs (2026-01-09)

## Scope

Code is the source of truth (per request). This report compares:
- Framework-level doc: `docs/architecture.md`
- Media doc: `abstractcore/docs/media-handling-system.md`
…against current AbstractCore implementation.

## Executive Summary

- `docs/architecture.md` is largely consistent at the level it operates (framework package boundaries + runtime/tool execution contract). I did not find critical contradictions with AbstractCore behavior.
- The *media documentation* in `abstractcore/docs/media-handling-system.md` is broadly aligned with the current system, but contains a few statements/snippets that no longer match the code’s exact behavior.
- The AbstractCore internal docs (not requested but relevant) also have drift (e.g., providers README describing `openai_provider.py` as “OpenAI & compatible APIs”).

## Findings

### 1) Framework-level architecture doc is consistent with tool execution “pass-through by default”

`docs/architecture.md` describes:
- AbstractCore normalizes tool calls.
- AbstractRuntime executes tools durably via host-configured `ToolExecutor`.

This matches the default in:
- `abstractcore/abstractcore/providers/base.py` (`execute_tools` default is `False`)

No action required, other than keeping the doc’s abstraction level: it should not attempt to mirror provider-by-provider details.

### 2) Media doc: “non-vision model receives image” behavior is more nuanced than described

`abstractcore/docs/media-handling-system.md` describes an “automatic vision fallback system” (two-stage caption → text model) and also includes an earlier snippet implying metadata-only behavior for non-vision models.

Current code:
- The two-stage fallback is implemented via:
  - `abstractcore/abstractcore/media/handlers/local_handler.py:_create_text_embedded_message(...)`
  - `abstractcore/abstractcore/media/vision_fallback.py:VisionFallbackHandler`
- However, OpenAIProvider currently routes media through `OpenAIMediaHandler` and does not use `LocalMediaHandler` for text-only OpenAI models, which can lead to images being skipped rather than described.

This is both a doc mismatch and a behavior inconsistency across providers.

### 3) Provider docs drift: `providers/README.md` describes `openai_provider.py` as “OpenAI & compatible APIs”

Current code has:
- `abstractcore/abstractcore/providers/openai_provider.py` (OpenAI SDK)
- `abstractcore/abstractcore/providers/openai_compatible_provider.py` (httpx protocol implementation)
- plus separate OpenAI-compatible wrappers (LMStudio, vLLM).

So the documentation’s “component structure” section is outdated.

Touch point:
- `abstractcore/abstractcore/providers/README.md` (component list near the top)

### 4) Media docs include pseudo-code that no longer reflects the centralized BaseProvider path

Some docs describe providers manually processing files, but the actual flow is:
- `BaseProvider.generate_with_telemetry` → `_process_media_content` → provider `_generate_internal`

So documentation examples that show providers calling `media_handler.process_file(...)` directly are misleading.

Touch points:
- `abstractcore/abstractcore/media/README.md` (provider integration pseudo-code)
- `abstractcore/docs/media-handling-system.md` (any similar pseudo-code sections)

## Actionable Recommendations

### R1) Update `abstractcore/docs/media-handling-system.md` to reflect current behavior precisely

Touch points:
- `abstractcore/docs/media-handling-system.md`

Minimum update:
- Clarify that two-stage vision fallback happens when the provider uses `LocalMediaHandler` and vision fallback is configured.
- Clarify current OpenAIProvider behavior for non-vision models (or update code to match the doc; see to-investigate).

### R2) Update provider docs to reflect the presence of `openai_compatible_provider.py`

Touch points:
- `abstractcore/abstractcore/providers/README.md`

### R3) Keep `docs/architecture.md` framework-level; link to AbstractCore docs instead of duplicating internals

Touch points:
- `docs/architecture.md`
- `abstractcore/docs/architecture.md`

This avoids constant churn when provider internals evolve.

