# 2026-01-10 — OpenAI Providers Consolidation (Full Report)

## Executive summary
AbstractCore now has a clean split between:
- **`OpenAIProvider`** (`abstractcore/abstractcore/providers/openai_provider.py`): the “true OpenAI” provider using the official OpenAI SDK and handling OpenAI-specific behaviors (notably reasoning models, `max_completion_tokens`, parameter support differences, and SDK-native response shapes).
- **`OpenAICompatibleProvider`** (`abstractcore/abstractcore/providers/openai_compatible_provider.py`): a shared HTTP implementation for any **OpenAI-compatible** `/v1` endpoint (LMStudio, vLLM, OpenRouter, and generic proxies).

The consolidation removed code duplication by making OpenAI-compatible providers thin subclasses and centralizing shared behavior (payload building, streaming SSE parsing, tool/media support, error normalization) in a single base.

## Q1 — Should we keep 2 providers (OpenAI vs OpenAI-compatible)?
### Answer
Yes — it’s justified and still a SOTA best practice to keep them separate in AbstractCore.

### Why (root-cause reasons)
1) **Protocol compatibility ≠ behavior compatibility**
Even if two endpoints speak “OpenAI-compatible chat completions”, the practical behavior differs:
- parameter support (seed/temperature/top_p)
- token parameter semantics (`max_tokens` vs `max_completion_tokens`)
- tool calling payload/response structure maturity
- streaming chunk structure and tool-call streaming behavior

2) **OpenAI’s reasoning models require provider-specific handling**
In `abstractcore/abstractcore/providers/openai_provider.py`, reasoning-family models:
- ignore `seed` and `temperature` (by design) and AbstractCore warns on request
- use `max_completion_tokens` instead of `max_tokens`

This is not reliably portable to generic OpenAI-compatible endpoints and would complicate the “generic” base.

3) **Official SDK integration is a long-term maintenance win**
Keeping `OpenAIProvider` SDK-based avoids:
- re-implementing OpenAI’s evolving response types and streaming behavior in HTTPX
- regressions in OpenAI-specific features that appear first in the SDK

### Why merging would be worse
If `OpenAIProvider` were merged into `OpenAICompatibleProvider`:
- the shared base would grow OpenAI-only branching and lose its clean “any compatible endpoint” contract
- reasoning-model handling would either leak into the generic provider (hurting clarity), or become a maze of conditionals
- SDK-only capabilities (and future changes) would either be lost or re-implemented

### Recommendation
Keep both:
- `OpenAIProvider`: OpenAI SDK + OpenAI-specific semantics
- `OpenAICompatibleProvider`: the reusable HTTP implementation for everyone else

## Q2 — How does LMStudio work now? Is it duplication?
### Current behavior
`LMStudioProvider` is now a thin subclass:
- File: `abstractcore/abstractcore/providers/lmstudio_provider.py`
- Inherits from: `OpenAICompatibleProvider`
- Default endpoint: `http://localhost:1234/v1`
- No API key by default

### Is it a duplication?
No. The OpenAI-compatible logic lives in one place (`OpenAICompatibleProvider`).

### Why keep it as a separate provider at all?
Even if LMStudio is “just OpenAI-compatible”, a dedicated provider is still justified because it provides:
- a stable provider name (`"lmstudio"`) for configs and switching
- a provider-specific base URL env var (`LMSTUDIO_BASE_URL`)
- future room for LMStudio-specific quirks without contaminating the generic base

## Q3 — Same question for vLLM
### Current behavior
`VLLMProvider` is now a thin subclass:
- File: `abstractcore/abstractcore/providers/vllm_provider.py`
- Inherits from: `OpenAICompatibleProvider`
- Default endpoint: `http://localhost:8000/v1`

### What remains vLLM-specific (why separation is justified)
vLLM adds *real* API surface beyond vanilla OpenAI-compat:
- **request extensions** via `payload["extra_body"]` (`guided_*`, beam search, etc.) in `VLLMProvider._mutate_payload()`
- **management endpoints** (`load_adapter`, `unload_adapter`, `list_adapters`)

This is a genuine reason to keep `vllm_provider.py` as a separate provider class even after consolidation.

### TODO
Add an opt-in integration test once a vLLM server is available (see `abstractcore/abstractcore/providers/vllm_provider.py` docstring TODO).

## Q4 — What about Ollama?
`OllamaProvider` remains separate (and should remain separate) because it is **not** “just OpenAI-compatible” in AbstractCore:
- it uses Ollama’s native API and semantics
- it carries Ollama-specific operational UX (local server assumptions, local model inventory behavior, error hints)
- tool calling and streaming are handled with Ollama/Qwen/Gemma format constraints that are not uniformly OpenAI-compatible

If you ever want to support Ollama’s OpenAI-compatible endpoint as well, that should be a distinct provider (or a mode flag), not a replacement.

## Q5 — Consolidation pattern (what we now have)
### Shared base
- `abstractcore/abstractcore/providers/openai_compatible_provider.py`

Responsibilities centralized here:
- base URL + API key resolution
- HTTP request/response handling via `httpx`
- model discovery (`/v1/models`) + best-effort init validation
- streaming SSE parsing
- tool handling via `UniversalToolHandler` (native vs prompted tooling decisions)
- media handling via handler selection (`OpenAIMediaHandler` vs `LocalMediaHandler`)
- consistent error mapping to AbstractCore exception classes

### Thin subclasses
- `abstractcore/abstractcore/providers/lmstudio_provider.py`
- `abstractcore/abstractcore/providers/vllm_provider.py`
- `abstractcore/abstractcore/providers/openrouter_provider.py`
- plus the generic `openai-compatible` provider entry in the registry

### Override surface (intentionally small)
To add future OpenAI-compatible providers (e.g., some gateways), prefer:
- class attributes: `PROVIDER_ID`, `PROVIDER_DISPLAY_NAME`, `BASE_URL_ENV_VAR`, `API_KEY_ENV_VAR`, `DEFAULT_BASE_URL`
- hooks:
  - `_get_headers()` (for per-provider headers like OpenRouter analytics)
  - `_mutate_payload()` (for extra request fields like vLLM `extra_body`)
  - `_get_api_key_from_config()` (for config-manager fallback)
  - `_validate_model()` override only when needed (e.g., skip unauthenticated validation)

## OpenRouter provider integration
### Provider
- File: `abstractcore/abstractcore/providers/openrouter_provider.py`
- Base URL: `https://openrouter.ai/api/v1`
- Key: `OPENROUTER_API_KEY` (or config fallback via `config.api_keys.openrouter`)

### Config integration
OpenRouter is wired into the centralized config system:
- `abstractcore/abstractcore/config/manager.py` (key storage under `api_keys.openrouter`)
- `abstractcore/abstractcore/config/main.py` (CLI wiring for `abstractcore --set-api-key openrouter ...`)

### Metadata headers (optional)
Supported via env vars:
- `OPENROUTER_SITE_URL` → `HTTP-Referer`
- `OPENROUTER_APP_NAME` → `X-Title`

## Tools and streaming: important behavioral contract (impacts tests + clients)
AbstractCore’s tool-call behavior is centralized in `BaseProvider`:
- File: `abstractcore/abstractcore/providers/base.py`
- Key function: `_normalize_tool_calls_passthrough()`

Contract:
- When `tools` are provided, AbstractCore populates structured `GenerateResponse.tool_calls` whenever tool syntax is detected (native or prompted).
- By default (`tool_call_tags is None`), AbstractCore **cleans tool-call markup out of `response.content`** for clean UX/history and keeps the structured tool calls in `response.tool_calls`.

This is why downstream parsing should prefer `response.tool_calls` over scraping tool tags from content unless explicitly requesting passthrough content.

## Media handling interplay (OpenAI-compatible path)
`OpenAICompatibleProvider` uses capability-based media handlers:
- File: `abstractcore/abstractcore/providers/openai_compatible_provider.py`
- Helper: `_get_media_handler_for_model()`

It selects between:
- `OpenAIMediaHandler` for OpenAI-style vision content blocks
- `LocalMediaHandler` for local models (Ollama/LMStudio/vLLM style)

For Anthropic media validation, we updated `AnthropicMediaHandler.validate_media_for_model()` to rely on the centralized vision-capability database rather than brittle name heuristics:
- File: `abstractcore/abstractcore/media/handlers/anthropic_handler.py`

## Tests and verification (what was run)
- Full default suite (offline-safe): `abstractcore/tests` → **871 passed, 236 skipped**
- Targeted live/local subset (opt-in): OpenAI + Anthropic + Ollama streaming/tooling → **46 passed, 2 skipped**
- LMStudio local streaming check (opt-in): passed
- OpenRouter live smoke check (opt-in): passed (API key provided at runtime; not persisted)

## Recommendations (actionable)
1) **Keep `OpenAIProvider` and `OpenAICompatibleProvider` separate**
   - Reasoning-model behavior and SDK evolutions are OpenAI-specific.
   - Keep the OpenAI-compatible base generic and lean.

2) **Add future OpenAI-compatible providers as thin subclasses**
   - Use the existing override surface (`_get_headers`, `_mutate_payload`, config key hook).
   - Avoid copy/paste of HTTP/stream parsing logic.

3) **Standardize “live test” environment flags**
   - Today there are multiple opt-in flags in the suite (e.g., `ABSTRACTCORE_RUN_LIVE_API_TESTS`, `ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS`, and older `ABSTRACTCORE_TEST_REAL_PROVIDERS`).
   - Consider consolidating these into a single naming scheme to reduce confusion.

4) **Add an opt-in vLLM integration test**
   - Cover `extra_body` injection and LoRA endpoints once a vLLM server is available.
   - Start with a minimal smoke test gated behind `ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1`.

5) **Document “authoring a new OpenAI-compatible provider”**
   - Add a short guide that explains the subclass template and override points.
   - This will pay off as you add OpenRouter-like providers and gateways over time.

---

## Appendix — Implementation details (merged)
This appendix was originally written as `abstractcore/docs/reports/2026-01-10-providers-openai-compatible-consolidation-implementation.md` and is now included here so the consolidation has a single “full” report (design + implementation).

### Scope of this appendix
This section documents the *implemented* consolidation of OpenAI-compatible providers in `abstractcore`, plus the addition of an `openrouter` provider, with a focus on mid/long-term maintainability and the existing “inheritance-first” provider pattern.

This follows the design direction in `abstractcore/docs/reports/2026-01-09-providers-openai-compatible-consolidation.md`, but validates against the current code.

### Key design choices (and why)

#### 1) Inheritance over composition (confirmed + maintained)
AbstractCore’s provider architecture is inheritance-driven:
- All providers inherit from `abstractcore/abstractcore/providers/base.py` (`BaseProvider`), which centralizes:
  - telemetry/events,
  - tool-call normalization,
  - tracing,
  - unified streaming processing,
  - timeout defaults via `abstractcore/config`.

The consolidation preserves that pattern:
- `LMStudioProvider`, `VLLMProvider`, and `OpenRouterProvider` now inherit from a shared HTTP implementation: `OpenAICompatibleProvider`.

Why this is a good fit here:
- The OpenAI-compatible surface is largely a “template method” problem (same workflow, small per-provider differences).
- Most of the variability is cleanly expressed as class attributes (provider id + env vars + default base url) and a couple of hooks.

#### 2) Keep `OpenAIProvider` separate
`abstractcore/abstractcore/providers/openai_provider.py` remains an OpenAI SDK-based provider (and is intentionally *not* forced through the OpenAI-compatible HTTPX path).

Rationale:
- The OpenAI SDK often exposes provider-specific capabilities and response formats first.
- Avoids regressions in structured output/vision/native tooling behavior that may be better supported via the official SDK.

### What was consolidated

#### `OpenAICompatibleProvider` is the shared OpenAI-compatible HTTP implementation
File: `abstractcore/abstractcore/providers/openai_compatible_provider.py`

Key responsibilities consolidated into one place:
- Base URL resolution:
  - parameter → env var → default
- API key resolution:
  - parameter → env var → config fallback hook (`_get_api_key_from_config`)
- Common headers + auth behavior:
  - adds `Authorization: Bearer ...` only when a key is present and non-empty
- Model discovery + best-effort model validation on init (with provider-specific overrides to skip when appropriate)
- Error normalization:
  - richer extraction of error details from JSON/text bodies
  - maps common status codes to AbstractCore exceptions (auth/rate-limit/model-not-found)
- Unified tool-call extraction:
  - normalizes tool calls (including `tool_calls`) into AbstractCore canonical shape
  - attaches `_provider_request` metadata for observability/debugging
- Streaming:
  - parses SSE chunks, and raises on HTTP errors early (rather than yielding “error” text chunks)

#### Thin subclasses for provider-specific behavior

##### LM Studio
File: `abstractcore/abstractcore/providers/lmstudio_provider.py`
- No custom behavior beyond defaults:
  - `PROVIDER_ID="lmstudio"`
  - `DEFAULT_BASE_URL="http://localhost:1234/v1"`
  - no API key by default

##### vLLM
File: `abstractcore/abstractcore/providers/vllm_provider.py`
- Keeps vLLM-only features:
  - `_mutate_payload()` injects `payload["extra_body"]` for vLLM extensions:
    - guided decoding (`guided_regex`, `guided_json`, `guided_grammar`)
    - beam search (`best_of`, `use_beam_search`)
    - caller-provided `extra_body` merge
  - management endpoints:
    - `load_adapter`, `unload_adapter`, `list_adapters`

##### OpenRouter
File: `abstractcore/abstractcore/providers/openrouter_provider.py`
- Adds first-class provider `openrouter` via OpenAI-compatible API:
  - `DEFAULT_BASE_URL="https://openrouter.ai/api/v1"`
  - API key required (`OPENROUTER_API_KEY` or config)
  - avoids unauthenticated model validation on init (OpenRouter typically requires auth)
  - supports optional metadata headers:
    - `OPENROUTER_SITE_URL` → `HTTP-Referer`
    - `OPENROUTER_APP_NAME` → `X-Title`

### Configuration integration (centralized config)
OpenRouter API key storage is integrated into the centralized config system:
- `abstractcore/abstractcore/config/manager.py`:
  - `api_keys.openrouter`
  - `set_api_key("openrouter", ...)`
  - status reporting includes OpenRouter
- `abstractcore/abstractcore/config/main.py`:
  - interactive prompt wiring includes OpenRouter

### Provider registry integration
OpenRouter is registered and exported for `create_llm("openrouter", ...)`:
- `abstractcore/abstractcore/providers/registry.py` (lazy import registration)
- `abstractcore/abstractcore/providers/__init__.py` (export)
- `abstractcore/abstractcore/architectures/detection.py`:
  - strips `openrouter/` prefix for capability lookup consistency

### Testing + verification strategy

#### Default test suite behavior (offline/sandbox-safe)
- Unit tests and non-network behavior are exercised by default.
- Live API and local-server tests are opt-in via env flags:
  - `ABSTRACTCORE_RUN_LIVE_API_TESTS=1`
  - `ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1`

This keeps the suite reliable across environments (including sandboxed CI where localhost sockets or outbound network may be blocked).

#### Recommended smoke checks (manual)
1) LMStudio (local):
- Ensure LMStudio server is running.
- Run minimal generation + streaming with tools.

2) OpenRouter (live):
- Export `OPENROUTER_API_KEY` in your shell (do not commit keys).
- Run a minimal request with a known model (e.g. `openai/gpt-4o-mini`).

3) TODO(vLLM):
- Validate `extra_body` behavior + LoRA endpoints when a vLLM server is available.

### Follow-up recommendations
1) Add an explicit “skip init model validation” knob to `OpenAICompatibleProvider`
- Motivation: local servers are often not running; eager validation can be painful in DX.
- OpenRouter already overrides `_validate_model()` for this; formalizing it may reduce subclass boilerplate.

2) Keep OpenAI-compatible providers thin and declarative
- Prefer:
  - class attributes for env/defaults
  - `_mutate_payload()` for payload diffs
  - `_get_headers()` for per-provider headers
  - `_get_api_key_from_config()` for config integration
- Avoid copying request/stream parsing code into subclasses.

3) Document “OpenAI-compatible provider authoring” pattern
- When adding `openrouter`-like providers (e.g., future gateways), document:
  - required env vars
  - config fallback approach
  - what should live in subclass vs shared base

