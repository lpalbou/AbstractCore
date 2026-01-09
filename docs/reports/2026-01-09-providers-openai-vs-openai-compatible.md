# Providers: OpenAI vs OpenAI-Compatible (2026-01-09)

## Questions Answered

1) Is it justified / SOTA best practice to have two providers to distinguish “true OpenAI” vs “OpenAI-compatible endpoints”?
2) Is the code really that different?
3) Should `openai_compatible_provider.py` be merged back into `openai_provider.py`?

## Executive Summary

Keeping **two providers** is justified and is broadly aligned with state-of-the-art practice:
- `OpenAIProvider` is “vendor-first”: it targets OpenAI’s canonical behavior and leverages the official SDK.
- `OpenAICompatibleProvider` is “protocol-first”: it targets the *OpenAI Chat Completions wire format* but must be tolerant to many partial/quirky implementations.

Merging them into one provider tends to create a “branch explosion” (lots of conditionals based on endpoint quirks) and typically makes both OpenAI and compatibility endpoints harder to maintain. The right direction is: **keep both** but **factor shared protocol logic** so they don’t drift.

## What the code does today (code-first)

### OpenAIProvider (`abstractcore/abstractcore/providers/openai_provider.py`)

Core characteristics:
- Uses the OpenAI Python SDK (`openai.OpenAI`, `openai.AsyncOpenAI`).
- Requires `OPENAI_API_KEY` (or `api_key=`).
- Supports OpenAI-specific behavior:
  - Reasoning model parameter restrictions (`_is_reasoning_model`).
  - Token cap name change (`max_completion_tokens` vs `max_tokens`) via `_uses_max_completion_tokens`.
  - OpenAI structured outputs strict schema requirements via `_ensure_strict_schema`.
  - Optional preflight model validation via `_validate_model_exists`.

Touch points:
- `OpenAIProvider.__init__` (client creation + preflight)
- `OpenAIProvider._generate_internal` / `OpenAIProvider._agenerate_internal`
- `OpenAIProvider._stream_response` (SDK streaming + tool-call reconstruction)
- `OpenAIProvider._supports_structured_output` + `_ensure_strict_schema`

### OpenAICompatibleProvider (`abstractcore/abstractcore/providers/openai_compatible_provider.py`)

Core characteristics:
- Uses raw `httpx` against `${base_url}/chat/completions` and `${base_url}/models`.
- API key is optional (`OPENAI_COMPATIBLE_API_KEY`).
- Designed to work across many servers/proxies (local servers, inference backends).
- Implements:
  - Tolerant message building (history + system prompt + media merge logic).
  - Native tools when available; prompted tools as fallback.
  - SSE streaming parsing (OpenAI-style `data: {...}` lines).
  - Embeddings endpoint (`/embeddings`) support.

Touch points:
- `OpenAICompatibleProvider.__init__` (base_url + optional auth + client setup + model validation)
- `_generate_internal` / `_agenerate_internal`
- `_single_generate` / `_async_single_generate`
- `_stream_generate` / `_async_stream_generate`
- `_get_media_handler_for_model` (vision vs non-vision routing)
- `list_available_models`, `embed`

## Are they “really that different”?

Yes — the *implementation constraints* differ enough to justify separation:

### 1) Dependency + transport differences
- `OpenAIProvider` depends on `openai` SDK and its typed response objects.
- `OpenAICompatibleProvider` must work with only core deps (`httpx`) and handle less strict JSON shapes.

### 2) Semantics and parameter compatibility
- OpenAI’s reasoning models (and ongoing API evolution) require provider-specific mapping and warnings.
- OpenAI-compatible servers often accept a subset (or a different superset) of parameters and can diverge in error formats.

### 3) Structured outputs “strict mode”
- OpenAI’s strict schema rules (`additionalProperties=false`, required fields) are OpenAI-specific and should not be blindly applied to generic endpoints.

### 4) Media handling expectations
- OpenAI vision uses the canonical `content: [{type: text}, {type: image_url}, ...]` format.
- Many OpenAI-compatible servers *claim* compatibility but behave differently; `OpenAICompatibleProvider` and `LocalMediaHandler` are more defensive.

## Should you merge them into one provider file/class?

Recommendation: **No merge** (keep separate), but consolidate shared logic.

### Why not merge (mid/long-term consequences)

If you merge “OpenAI” + “OpenAI-compatible” into a single provider:
- You inevitably add endpoint-detection and compatibility flags (“if base_url != openai.com then …”), producing a growing condition matrix.
- You risk “accidental coupling” where OpenAI-specific behaviors (strict schema, reasoning-model warnings) get applied to generic endpoints, or vice versa.
- You increase the chance of regressions when OpenAI changes behavior (because you must preserve compatibility behavior in the same codepath).

### Why keep separate (and what you still should share)

Keep:
- `OpenAIProvider` as the canonical “OpenAI vendor integration”.
- `OpenAICompatibleProvider` as the canonical “Chat Completions protocol integration”.

Share:
- Pure protocol logic that is not vendor-specific:
  - message list assembly
  - tool selection policy (“native vs prompted fallback”)
  - response normalization into `GenerateResponse`
  - media handler selection rules (vision vs non-vision)

## Actionable Recommendations

### R1) Keep two providers, but extract shared “Chat Completions protocol” helpers

Goal: avoid behavioral drift without collapsing vendor boundaries.

Touch points for extraction (current duplication):
- message building logic in:
  - `abstractcore/abstractcore/providers/openai_provider.py:_generate_internal`
  - `abstractcore/abstractcore/providers/openai_compatible_provider.py:_generate_internal`
- tool-handling policy decisions in both providers (native vs prompted)
- media handler selection logic (OpenAIProvider currently always uses `OpenAIMediaHandler`)

Concrete suggestion:
- Create an internal module (example) `abstractcore/abstractcore/providers/_chat_completions_common.py` to hold:
  - `build_chat_messages(prompt, messages, system_prompt, media, ...)`
  - `apply_tools(payload_or_call_params, tools, tool_handler, ...)`
  - `apply_structured_output(payload_or_call_params, response_model, provider_kind, ...)`

### R2) Add “non-vision model image” behavior consistency across providers

Today:
- `LocalMediaHandler` provides vision fallback (caption model) for text-only models.
- `OpenAIProvider` uses `OpenAIMediaHandler` and may silently drop images for non-vision models.

Recommendation:
- Make OpenAIProvider choose `LocalMediaHandler` when `supports_vision(model)` is false, matching the OpenAI-compatible family’s behavior.

Touch points:
- `abstractcore/abstractcore/providers/openai_provider.py:_generate_internal` and `_agenerate_internal`
- `abstractcore/abstractcore/media/handlers/local_handler.py:_create_text_embedded_message`
- `abstractcore/abstractcore/media/vision_fallback.py:VisionFallbackHandler`

### R3) Treat OpenAI “Responses API” as a separate future integration

You already expose `/v1/responses` in `abstractcore/abstractcore/server/app.py`. The OpenAI SDK and OpenAI platform increasingly treat “responses” as the primary API for multimodal outputs (text+audio and beyond).

Recommendation:
- Do **not** fold OpenAI “responses API” into `OpenAICompatibleProvider` (it’s not a compatibility target).
- If/when needed, add a dedicated `OpenAIResponsesProvider` or extend `OpenAIProvider` behind a clear interface boundary.

Touch points:
- `abstractcore/abstractcore/server/app.py` (already has a `/v1/responses` surface)
- `abstractcore/abstractcore/providers/openai_provider.py` (currently chat.completions-based)

