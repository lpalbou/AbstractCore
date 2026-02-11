# AbstractCore Architecture Review (2026-01-09)

## Scope

This report is a code-first architecture review of `abstractcore/abstractcore/` in the monorepo (with a bias toward the provider + media subsystems, since those are the most coupling-heavy parts of the package).

Primary sources:
- `abstractcore/abstractcore/providers/base.py` (request lifecycle, retries, streaming, tool normalization)
- `abstractcore/abstractcore/providers/*_provider.py` (provider implementations)
- `abstractcore/abstractcore/media/*` (media ingestion + vision fallback)
- `abstractcore/abstractcore/server/app.py` (OpenAI-compatible server surface)
- `abstractcore/abstractcore/config/*` (configuration + CLI)

## Executive Summary

AbstractCore’s core design is strong and already aligns with the framework-level architecture described in `docs/architecture.md`:
- A provider abstraction with **sync/async + streaming/non-streaming parity**.
- **Tool calling normalization** across native and prompted tool syntaxes.
- A single generation entrypoint (`BaseProvider.generate_with_telemetry`) that centralizes **retry + observability + media preprocessing + tool-call normalization**.

The biggest medium-term maintainability risk is **code duplication across OpenAI-compatible providers** and **parallel “capability” abstractions** in the media subsystem. These will become increasingly costly as you add more providers (OpenRouter, Gemini, etc.) and more modalities (audio/image generation).

## Current Architecture (As Implemented)

### 1) Provider boundary and request lifecycle

`abstractcore/abstractcore/providers/base.py` is the “kernel” for provider calls:
- Normalizes unified params (`max_output_tokens`, `seed`, `temperature`, …).
- Preprocesses media via `_process_media_content` (into `MediaContent` objects).
- Converts tool definitions into dict specs and controls tool execution mode (`execute_tools` default is **pass-through**).
- Wraps the provider call in `RetryManager` and normalizes exceptions.
- For streaming, wraps the provider’s generator with `UnifiedStreamProcessor` to do incremental tool detection + optional tag rewriting.

Touch points:
- `abstractcore/abstractcore/providers/base.py`:
  - `BaseProvider.generate_with_telemetry(...)`
  - `BaseProvider._handle_tools_with_structured_output(...)` (hybrid “tools + structured” flow)
  - `BaseProvider._normalize_tool_calls_passthrough(...)` (non-streaming tool normalization)
  - `abstractcore/abstractcore/providers/streaming.py:UnifiedStreamProcessor` (streaming tool detection + tag rewriting)

**Strengths**
- A single boundary is where “framework invariants” live: token naming, telemetry, tracing, retry semantics.
- Providers can stay focused on “how to call the API”.

**Architectural hazard**
- When providers re-implement message building / streaming parsing / media handler selection, the “single boundary” advantage erodes and behavior drifts.

### 2) Provider registry and multi-provider model discovery

`abstractcore/abstractcore/providers/registry.py` centralizes provider metadata and lazy loading.

Touch points:
- `abstractcore/abstractcore/providers/registry.py:ProviderRegistry`
- `abstractcore/abstractcore/core/factory.py:create_llm(...)`

**Strengths**
- Good long-term “add provider once” locus.

**Risk**
- Providers currently implement `list_available_models` inconsistently (some as instance, some as `@classmethod`). It works, but it’s an attractive source of subtle drift.

### 3) Media ingestion + vision fallback

Media ingestion is a two-layer system:
- **Processing**: `AutoMediaHandler` selects processors (image/pdf/office/text) and returns `MediaContent`.
- **Formatting**: provider handlers (`OpenAIMediaHandler`, `AnthropicMediaHandler`, `LocalMediaHandler`) convert `MediaContent` into the provider’s message format.

Vision fallback for text-only models is currently implemented in `LocalMediaHandler._create_text_embedded_message(...)` by calling `VisionFallbackHandler`, which itself calls `create_llm(...)` to invoke a configured vision model.

Touch points:
- `abstractcore/abstractcore/providers/base.py:_process_media_content(...)`
- `abstractcore/abstractcore/media/auto_handler.py:AutoMediaHandler`
- `abstractcore/abstractcore/media/handlers/local_handler.py:LocalMediaHandler`
- `abstractcore/abstractcore/media/vision_fallback.py:VisionFallbackHandler`
- Config integration:
  - `abstractcore/abstractcore/config/vision_config.py`
  - `abstractcore/abstractcore/config/main.py` (CLI flags `--set-vision-provider`, `--download-vision-model`, etc.)

**Strengths**
- The fallback is capability-based and configurable.
- The implementation is self-contained and works without requiring an agent runtime.

**Risk**
- There are **two different “MediaCapabilities” abstractions**:
  - `abstractcore/abstractcore/media/types.py:MediaCapabilities` (simple dataclass used by handlers)
  - `abstractcore/abstractcore/media/capabilities.py:MediaCapabilities` (richer dataclass used by `ImageProcessor` via `get_media_capabilities`)
  This naming collision is confusing and will be a maintenance trap as media features grow.

### 4) OpenAI-compatible server surface

`abstractcore/abstractcore/server/app.py` exposes an OpenAI-compatible API surface (`/v1/models`, `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, …) and performs syntax rewriting between “provider tool formats” and “OpenAI tool formats”.

Touch points:
- `abstractcore/abstractcore/server/app.py`

**Strengths**
- Matches the framework-level goal: hosts can call AbstractCore over HTTP with OpenAI-compatible semantics.

**Risk**
- Provider duplication will also leak into server behavior (different providers producing slightly different tool_call payload shapes, error semantics, etc.), increasing the complexity of the rewriter layer.

## Recommendations (Actionable)

### R1) Reduce OpenAI-compatible provider duplication now (before adding OpenRouter/Gemini)

Why:
- `abstractcore/abstractcore/providers/openai_compatible_provider.py`, `lmstudio_provider.py`, and most of `vllm_provider.py` overlap heavily (same message building, media routing, streaming parsing, embeddings endpoints).

Action:
- Introduce a single reusable OpenAI-compatible HTTP “chat completions” base, then make LMStudio/vLLM thin wrappers.
- See report: `abstractcore/abstractcore/docs/reports/2026-01-09-providers-openai-compatible-consolidation.md`.

### R2) Fix media capability abstraction naming to prevent long-term confusion

Why:
- The name `MediaCapabilities` currently refers to two unrelated types in different files.

Action (minimal-impact options):
- Rename `abstractcore/abstractcore/media/capabilities.py:MediaCapabilities` → `ModelMediaCapabilities` (or similar), keep `types.py:MediaCapabilities` as the handler capability type.
- Or: consolidate into one capability class and provide a compatibility shim.

Touch points:
- `abstractcore/abstractcore/media/types.py`
- `abstractcore/abstractcore/media/capabilities.py`
- `abstractcore/abstractcore/media/processors/image_processor.py` (uses `get_media_capabilities`)

### R3) Decide the “output modalities” boundary explicitly (text vs. multimodal outputs)

Why:
- Today `GenerateResponse` is fundamentally text-centric (`content: Optional[str]`).
- `providers/model_capabilities.py` already acknowledges that `ModelOutputCapability` may expand to `IMAGE`, `AUDIO`, `VIDEO`.

Action:
- Treat “generative audio/images” as a separate capability surface from LLM chat generation, unless you commit to evolving `GenerateResponse` into a multimodal output container.
- See report: `abstractcore/abstractcore/docs/reports/2026-01-09-media-vision-fallback-and-modality-boundaries.md`.

### R4) Align documentation with current code structure

Why:
- Some docs still describe `openai_provider.py` as “OpenAI & compatible APIs”, but the code now has a distinct `openai_compatible_provider.py` plus multiple OpenAI-compatible providers.

Action:
- Update doc pointers, and remove obsolete pseudo-code that no longer reflects the central `BaseProvider._process_media_content(...)` path.
- See report: `abstractcore/abstractcore/docs/reports/2026-01-09-docs-architecture-and-media-docs-review.md`.

