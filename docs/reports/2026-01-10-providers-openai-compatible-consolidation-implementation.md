# 2026-01-10 — OpenAI-Compatible Consolidation (Implementation Report)

## Scope of this report
This report documents the *implemented* consolidation of OpenAI-compatible providers in `abstractcore`, plus the addition of an `openrouter` provider, with a focus on mid/long-term maintainability and the existing “inheritance-first” provider pattern.

This follows the design direction in `abstractcore/docs/reports/2026-01-09-providers-openai-compatible-consolidation.md`, but validates against the current code.

## Key design choices (and why)

### 1) Inheritance over composition (confirmed + maintained)
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

### 2) Keep `OpenAIProvider` separate
`abstractcore/abstractcore/providers/openai_provider.py` remains an OpenAI SDK-based provider (and is intentionally *not* forced through the OpenAI-compatible HTTPX path).

Rationale:
- The OpenAI SDK often exposes provider-specific capabilities and response formats first.
- Avoids regressions in structured output/vision/native tooling behavior that may be better supported via the official SDK.

## What was consolidated

### `OpenAICompatibleProvider` is the shared OpenAI-compatible HTTP implementation
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

### Thin subclasses for provider-specific behavior

#### LM Studio
File: `abstractcore/abstractcore/providers/lmstudio_provider.py`
- No custom behavior beyond defaults:
  - `PROVIDER_ID="lmstudio"`
  - `DEFAULT_BASE_URL="http://localhost:1234/v1"`
  - no API key by default

#### vLLM
File: `abstractcore/abstractcore/providers/vllm_provider.py`
- Keeps vLLM-only features:
  - `_mutate_payload()` injects `payload["extra_body"]` for vLLM extensions:
    - guided decoding (`guided_regex`, `guided_json`, `guided_grammar`)
    - beam search (`best_of`, `use_beam_search`)
    - caller-provided `extra_body` merge
  - management endpoints:
    - `load_adapter`, `unload_adapter`, `list_adapters`

#### OpenRouter
File: `abstractcore/abstractcore/providers/openrouter_provider.py`
- Adds first-class provider `openrouter` via OpenAI-compatible API:
  - `DEFAULT_BASE_URL="https://openrouter.ai/api/v1"`
  - API key required (`OPENROUTER_API_KEY` or config)
  - avoids unauthenticated model validation on init (OpenRouter typically requires auth)
  - supports optional metadata headers:
    - `OPENROUTER_SITE_URL` → `HTTP-Referer`
    - `OPENROUTER_APP_NAME` → `X-Title`

## Configuration integration (centralized config)
OpenRouter API key storage is integrated into the centralized config system:
- `abstractcore/abstractcore/config/manager.py`:
  - `api_keys.openrouter`
  - `set_api_key("openrouter", ...)`
  - status reporting includes OpenRouter
- `abstractcore/abstractcore/config/main.py`:
  - interactive prompt wiring includes OpenRouter

## Provider registry integration
OpenRouter is registered and exported for `create_llm("openrouter", ...)`:
- `abstractcore/abstractcore/providers/registry.py` (lazy import registration)
- `abstractcore/abstractcore/providers/__init__.py` (export)
- `abstractcore/abstractcore/architectures/detection.py`:
  - strips `openrouter/` prefix for capability lookup consistency

## Testing + verification strategy

### Default test suite behavior (offline/sandbox-safe)
- Unit tests and non-network behavior are exercised by default.
- Live API and local-server tests are opt-in via env flags:
  - `ABSTRACTCORE_RUN_LIVE_API_TESTS=1`
  - `ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1`

This keeps the suite reliable across environments (including sandboxed CI where localhost sockets or outbound network may be blocked).

### Recommended smoke checks (manual)
1) LMStudio (local):
- Ensure LMStudio server is running.
- Run minimal generation + streaming with tools.

2) OpenRouter (live):
- Export `OPENROUTER_API_KEY` in your shell (do not commit keys).
- Run a minimal request with a known model (e.g. `openai/gpt-5-mini`).

3) TODO(vLLM):
- Validate `extra_body` behavior + LoRA endpoints when a vLLM server is available.

## Follow-up recommendations
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

