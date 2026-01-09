# Providers: Consolidating the OpenAI-Compatible Family (2026-01-09)

## Questions Answered

2) Confirm how `LMStudioProvider` works currently; is it only using an OpenAI-compatible endpoint and is it a duplication?
3) Same for `VLLMProvider`?
4) Same question for `OllamaProvider` (covered partially here, fully in the Ollama report)?
5) If most are OpenAI-compatible, should we consolidate with inheritance (or another approach)?

## Executive Summary

Yes: **LMStudioProvider and most of VLLMProvider are OpenAI-compatible `/v1/chat/completions` wrappers** and are currently a major duplication of `OpenAICompatibleProvider`.

The clean, scalable direction (especially with OpenRouter on the roadmap) is:
- Keep **distinct provider names** (good UX, separate defaults/env vars, documentation).
- Share **one OpenAI-compatible HTTP implementation** for chat completions + streaming + embeddings + model discovery.
- Make each provider only specify what it truly needs:
  - base_url + env var names
  - auth header behavior
  - provider-specific request extensions (e.g., vLLM `extra_body`)
  - provider-specific auxiliary endpoints (e.g., vLLM LoRA management)

## What LMStudioProvider does today (confirmed)

Implementation: `abstractcore/abstractcore/providers/lmstudio_provider.py`

Key observations:
- Default base URL is OpenAI-style: `http://localhost:1234/v1` (via `LMSTUDIO_BASE_URL`).
- Generation calls:
  - Sync: POST `${base_url}/chat/completions`
  - Streaming: SSE from `${base_url}/chat/completions` with `data: {...}` lines
- Model discovery:
  - GET `${base_url}/models`
- Embeddings:
  - POST `${base_url}/embeddings`

Conclusion:
- It is **100%** an OpenAI-compatible provider.
- It is also a **direct duplication** of `OpenAICompatibleProvider` with minor differences:
  - env var names / defaults
  - some improved error-body extraction for LMStudio-specific errors
  - provider identifier string

## What VLLMProvider does today (confirmed)

Implementation: `abstractcore/abstractcore/providers/vllm_provider.py`

Key observations:
- Default base URL is OpenAI-style: `http://localhost:8000/v1` (via `VLLM_BASE_URL`).
- Generation calls:
  - POST `${base_url}/chat/completions`
  - Streaming SSE from the same endpoint
- Adds vLLM-specific features via request extension:
  - `payload["extra_body"] = { guided_regex / guided_json / ... }`
- Exposes vLLM-specific management endpoints:
  - `POST /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter`, `GET /v1/lora_adapters` (base_url with `/v1` stripped)
- Also implements embeddings and model listing via OpenAI-compatible endpoints.

Conclusion:
- vLLM is OpenAI-compatible *plus* additional vLLM-specific APIs.
- The “chat completions” part is still largely duplicative of OpenAICompatibleProvider.

## Consolidation: What to do (and why)

### Why consolidation is worth doing now

You’re planning:
- More OpenAI-compatible endpoints (OpenRouter, potentially other aggregators).
- More local servers that mimic OpenAI chat completions.

If each one copies the same 800–900 lines, you’ll see:
- Drift in tool behavior (“native vs prompted” decisions)
- Drift in media fallback behavior (vision fallback, placeholder rules)
- Inconsistent error semantics (some raise, some yield error chunks in streams)
- Higher cost to add new cross-cutting features (structured outputs, retries, tracing)

### SOTA pattern for this problem

Treat “OpenAI Chat Completions” as a *protocol family*:
- Implement the protocol once (HTTP client, payload building, SSE parsing, response normalization).
- Provide thin provider wrappers for each vendor/server.

In Python, you can do that with either:
- Inheritance (base class + small overrides), or
- Composition (a shared “client” object that providers delegate to).

Both work; composition usually stays cleaner as the protocol grows, but inheritance is perfectly fine if you keep the override surface narrow and stable.

## Actionable Refactor Design (recommended)

### Step 1: Turn OpenAICompatibleProvider into a reusable base

File to refactor:
- `abstractcore/abstractcore/providers/openai_compatible_provider.py`

Add a small set of extension points (the “override surface”):
- `PROVIDER_ID: str` (e.g., `"openai-compatible"`, `"lmstudio"`, `"vllm"`, `"openrouter"`)
- `BASE_URL_ENV_VAR: str`
- `API_KEY_ENV_VAR: str | None`
- `DEFAULT_BASE_URL: str`
- `DEFAULT_API_KEY: str | None` (or sentinel for “no auth”)
- `def _get_headers(self) -> Dict[str, str]` (already exists, but should become the primary customization point)
- `def _mutate_payload(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]` (hook for vLLM `extra_body`, OpenRouter vendor routing, etc.)
- `def _extract_error_detail(self, response: httpx.Response) -> str | None` (hook for LMStudio’s error envelope improvements)

### Step 2: Make LMStudioProvider a thin subclass

Replace most of `abstractcore/abstractcore/providers/lmstudio_provider.py` with:
- `class LMStudioProvider(OpenAICompatibleProvider):`
  - override env/defaults
  - set `self.provider = "lmstudio"`
  - optionally override `_extract_error_detail(...)` if you don’t move that behavior into the base

Keep only LMStudio-specific documentation + any genuinely LMStudio-specific behavior.

### Step 3: Make VLLMProvider a thin subclass + keep extra APIs

Refactor `abstractcore/abstractcore/providers/vllm_provider.py` into:
- `class VLLMProvider(OpenAICompatibleProvider):`
  - override env/defaults + auth
  - override `_mutate_payload(...)` to add `extra_body`
  - keep additional methods: `load_adapter`, `unload_adapter`, `list_adapters`

### Step 4: Add OpenRouterProvider using the same base (future-ready)

Expected shape:
- `class OpenRouterProvider(OpenAICompatibleProvider):`
  - `DEFAULT_BASE_URL="https://openrouter.ai/api/v1"`
  - `API_KEY_ENV_VAR="OPENROUTER_API_KEY"`
  - `_get_headers` to include any OpenRouter-recommended metadata (referer/title)
  - `_mutate_payload` to handle OpenRouter-specific routing fields if you choose to support them

This becomes trivial once the base is consolidated.

## Pros / Cons of “inherit from OpenAICompatibleProvider”

### Pros
- Removes hundreds of duplicated lines per provider.
- Makes new providers (OpenRouter, proxies) cheap to add.
- Eliminates drift in:
  - media routing (vision fallback)
  - prompted vs native tools policy
  - SSE streaming parsing and error behavior
- Makes bug-fixing safer: one fix, all providers benefit.

### Cons / Risks
- A too-powerful base class can become “god class”.
  - Mitigation: keep override surface small and stable.
- Provider-specific quirks (LMStudio error shapes, vLLM extra_body) must be supported without “if provider == …” everywhere.
  - Mitigation: use explicit hooks (`_extract_error_detail`, `_mutate_payload`).

## Actionable Recommendations

### R1) Consolidate LMStudio and vLLM onto the OpenAI-compatible base

Touch points:
- `abstractcore/abstractcore/providers/openai_compatible_provider.py`
- `abstractcore/abstractcore/providers/lmstudio_provider.py`
- `abstractcore/abstractcore/providers/vllm_provider.py`
- Provider registry updates:
  - `abstractcore/abstractcore/providers/registry.py:_load_provider_class(...)`

### R2) Fix “explicit parameter” semantics for base_url/api_key resolution

Right now many providers use `param or env or default`, which prevents callers from intentionally overriding with “empty” values and can cause env-var bleed across providers.

Recommendation:
- Use “if param is not None” instead of truthiness.

Touch points:
- `abstractcore/abstractcore/providers/openai_compatible_provider.py:__init__`
- and after consolidation, in the shared base only.

### R3) Standardize streaming error behavior across all OpenAI-compatible providers

Several providers yield an “Error: …” chunk on streaming failures instead of raising. That can look like a successful stream to the caller.

Recommendation:
- Prefer raising a provider exception from streaming generators, and let `BaseProvider.generate_with_telemetry` track failure consistently.

Touch points:
- `_stream_generate` methods in:
  - `abstractcore/abstractcore/providers/openai_compatible_provider.py`
  - `abstractcore/abstractcore/providers/lmstudio_provider.py`
  - `abstractcore/abstractcore/providers/vllm_provider.py`

