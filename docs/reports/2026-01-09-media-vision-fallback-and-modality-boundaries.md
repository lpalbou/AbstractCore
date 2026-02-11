# Media, Vision Fallback, and Future Modalities (2026-01-09)

## Scope

This report focuses on:
- Input media handling (documents/images) and the current vision fallback mechanism.
- How media interacts with providers (OpenAI vs OpenAI-compatible vs local providers).
- The design choice for *future generative* image/audio capabilities: inside `abstractcore` vs separate packages.

Primary sources:
- `abstractcore/abstractcore/providers/base.py` (media preprocessing entrypoint)
- `abstractcore/abstractcore/media/*` (handlers/processors/fallback)
- `abstractcore/docs/media-handling-system.md` (doc surface)

## Executive Summary

The current input media system is well-structured:
- Media ingestion is centralized via `BaseProvider._process_media_content(...)`.
- Providers format content using handler classes.
- Vision fallback for text-only models is implemented and integrated via `LocalMediaHandler`.

The main missing clarity is the boundary between:
- “LLM chat generation” (text + tools + structured output), and
- “media generation” (images/audio/video outputs).

To keep AbstractCore clean and maintainable while scaling providers and modalities, I recommend:
- Keep **input media ingestion** inside AbstractCore (already done).
- Keep **media generation** in separate packages (consistent with `abstractvoice`), but align types/capabilities so higher layers can route by capability.

## How input media works today (code-first)

### 1) Media preprocessing lives at the provider boundary

`BaseProvider.generate_with_telemetry(...)` calls:
- `processed_media = self._process_media_content(media, glyph_compression_pref)`

Touch point:
- `abstractcore/abstractcore/providers/base.py:_process_media_content(...)`

This converts:
- file paths
- `MediaContent` objects
- dict-like media specs
…into a list of `abstractcore.media.types.MediaContent`.

### 2) AutoMediaHandler selects processors (image/pdf/office/text)

Touch points:
- `abstractcore/abstractcore/media/auto_handler.py:AutoMediaHandler`
- `abstractcore/abstractcore/media/processors/*`

Notes:
- Images: PIL-based processing (base64 output)
- PDFs: `pymupdf4llm` if available; otherwise fall back to text processor
- Office: `unstructured` if available; otherwise fall back to text processor
- Audio/video: currently not processed; fall back to placeholders

### 3) Provider handlers convert MediaContent into provider message formats

Touch points:
- `abstractcore/abstractcore/media/handlers/openai_handler.py:OpenAIMediaHandler`
- `abstractcore/abstractcore/media/handlers/anthropic_handler.py:AnthropicMediaHandler`
- `abstractcore/abstractcore/media/handlers/local_handler.py:LocalMediaHandler`

Important: `LocalMediaHandler` is where the two-stage vision fallback is integrated.

## Vision fallback design (what exists and what’s missing)

### What exists (and is good)

For text-only models receiving images:
- `LocalMediaHandler._create_text_embedded_message(...)` attempts to call `VisionFallbackHandler` to generate a description and embeds that description into the prompt.
- If not configured, it inserts a minimal placeholder like `[Image 1: file_name]` and logs setup guidance.

Touch points:
- `abstractcore/abstractcore/media/handlers/local_handler.py:_create_text_embedded_message(...)`
- `abstractcore/abstractcore/media/vision_fallback.py:VisionFallbackHandler`
- config CLI:
  - `abstractcore/abstractcore/config/main.py` (`--set-vision-provider`, `--download-vision-model`, …)

This is a clean “capability routing” primitive and matches your intent: users don’t need to know which model did what.

### What’s missing / inconsistent

#### 1) OpenAIProvider does not use vision fallback for non-vision models

`OpenAIProvider` currently always formats media via `OpenAIMediaHandler`. If the model lacks vision, images can be skipped instead of being routed through fallback.

Touch points:
- `abstractcore/abstractcore/providers/openai_provider.py:_generate_internal`
- `abstractcore/abstractcore/media/handlers/openai_handler.py:create_multimodal_message`

Recommendation:
- Apply the same “vision handler vs local handler” selection strategy for OpenAIProvider as used by OpenAI-compatible providers.

#### 2) VisionFallbackHandler ignores `user_prompt`

`VisionFallbackHandler.create_description(image_path, user_prompt=None)` accepts `user_prompt` but does not use it in the caption prompt.

Touch point:
- `abstractcore/abstractcore/media/vision_fallback.py:VisionFallbackHandler._generate_description(...)`

Recommendation:
- Optionally incorporate a short “question focus” derived from the user’s prompt to improve relevance (configurable, off by default to avoid prompt injection surprises).

### Optional “next level” (your idea): interactive vision Q/A

Your proposed extension (“forward the text model’s questions to the vision model”) is architecturally clean if implemented as a *tool*:
- The text-only model can call a tool like `vision_query(image_id, question)`.
- The tool is implemented by a vision model (or a cascade).

This integrates naturally with your existing tool architecture and keeps costs explicit/configurable.

## Future: Generative audio + images — where should it live?

### Reality check: AbstractCore is currently text-centric on output

`abstractcore/abstractcore/core/types.py:GenerateResponse` is fundamentally:
- `content: Optional[str]`
- optional `tool_calls`

So “generate an image/audio” is not a drop-in extension of the existing `generate(...)` surface unless you evolve the response type and server protocol.

### Option A — Integrate generative media into AbstractCore

Pros:
- Single unified surface for multimodal output.
- Shared telemetry/retries/config.

Cons:
- Requires evolving core response types and server API.
- Increases complexity and dependency surface inside AbstractCore.
- Risks destabilizing the already-stable text+tools interface.

### Option B — Keep generative media in separate packages (recommended)

Pros:
- Preserves AbstractCore’s clean responsibility: text+tools+structured outputs + input media ingestion.
- Matches your existing direction (`abstractvoice/`).
- Allows independent iteration and heavier deps without bloating AbstractCore.

Cons:
- Requires a clean integration/routing story at a higher layer (agent/runtime/host).

### Recommendation: Separate packages + shared capability metadata

Concrete suggestion:
- Keep `abstractvoice` as “audio generation + STT/TTS”.
- Create `abstractvision` for “image generation + (future) video generation”.
- Use AbstractCore as the shared “capability metadata + provider config” foundation, but do not force a single “generate() returns everything” API until you’re ready to redesign the response type.

To make this future-proof, you can still do small incremental steps in AbstractCore now:
- Extend `ModelOutputCapability` (in `abstractcore/abstractcore/providers/model_capabilities.py`) to include `IMAGE`, `AUDIO` when you’re ready.
- Consider a new optional field on `GenerateResponse` (e.g., `media_outputs: List[MediaContent] | None`) if you want to carry multimodal outputs through the server, but treat it as a versioned protocol change.

## Actionable Recommendations

### R1) Unify vision fallback behavior across providers

Touch points:
- `abstractcore/abstractcore/providers/openai_provider.py` (choose handler based on `supports_vision(model)`)
- `abstractcore/abstractcore/providers/openai_compatible_provider.py:_get_media_handler_for_model` (existing reference implementation)

### R2) Resolve the “MediaCapabilities” naming collision

Touch points:
- `abstractcore/abstractcore/media/types.py:MediaCapabilities`
- `abstractcore/abstractcore/media/capabilities.py:MediaCapabilities`

### R3) Treat interactive vision Q/A as a tool-driven optional strategy

Touch points:
- `abstractcore/abstractcore/media/vision_fallback.py` (extend strategies)
- `abstractcore/abstractcore/tools/*` (implement a vision-query tool)
- Higher-layer orchestration (likely `abstractagent/`), not the provider layer.

### R4) Keep generative media out of AbstractCore provider surface for now

Touch points:
- `abstractvoice/` (existing precedent)
- proposed `abstractvision/` (new package)
- `abstractcore/abstractcore/providers/model_capabilities.py` (future capability flags)

