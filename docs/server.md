# AbstractCore Server

Transform AbstractCore into an OpenAI-compatible API server. One server, all models, any client.

If you want a dedicated **single-model** `/v1` server (one provider/model per worker), see [Endpoint](endpoint.md).

## Interactive API docs (start here)

Visit while the server is running:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

Swagger UI exposes an `Authorize` button. When `ABSTRACTCORE_SERVER_API_KEY` is set,
enter that value there; requests executed from the docs page will send it as
`Authorization: Bearer <token>`. The docs and OpenAPI schema are public by default so
the UI can load before authentication, but API operations remain protected. Set
`ABSTRACTCORE_SERVER_PROTECT_DOCS=1` if you also want `/docs`, `/redoc`, and
`/openapi.json` behind server auth.

The OpenAPI schema includes executable examples for every request body. JSON
examples intentionally show optional aliases as `null` when sending both fields
would be ambiguous; the server drops nulls before routing. For local/custom
OpenAI-compatible endpoints, set `base_url` only when you intentionally want to
route away from the provider's default API host.

## Quick Start

### Install and Run (2 minutes)

```bash
# Install
pip install "abstractcore[server]"

# Configure server auth and provider keys
export ABSTRACTCORE_SERVER_API_KEY="acore-server-secret"
export OPENAI_API_KEY="sk-..."

# Start server
python -m abstractcore.server.app

# Or with uvicorn directly
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/health
# Response: {"status":"healthy"}
```

### First Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or with Python:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

response = client.chat.completions.create(
    model="anthropic/claude-haiku-4-5",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
```

---

## Configuration

You can configure the server through environment variables or through AbstractCore's centralized config. Environment variables always take precedence over config-persisted values.

```bash
# Persisted local/server config
abstractcore --set-server-api-key acore-server-secret
abstractcore --set-api-key openai sk-...
abstractcore --set-api-key anthropic sk-ant-...
abstractcore --set-api-key openrouter sk-or-...
abstractcore --set-api-key portkey pk_...

# Optional hardening/defaults
abstractcore --set-server-base-url-allowlist "https://example.com/v1"
abstractcore --set-server-url-fetch-allowlist "https://files.example.com"
abstractcore --set-server-media-root /srv/abstractcore-media
abstractcore --set-server-host 127.0.0.1
abstractcore --set-server-port 8000
```

### Environment Variables

```bash
# Provider API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
export PORTKEY_API_KEY="pk_..."         # optional (Portkey)
export PORTKEY_CONFIG="pcfg_..."        # required for Portkey routing

# Server master key. Authenticated clients can use all server-configured providers.
export ABSTRACTCORE_SERVER_API_KEY="acore-server-secret"

# Optional: also protect /docs, /redoc, and /openapi.json.
export ABSTRACTCORE_SERVER_PROTECT_DOCS=1

# Local providers
export OLLAMA_BASE_URL="http://localhost:11434"          # (or legacy: OLLAMA_HOST)
export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
export VLLM_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:1234/v1"
export OPENAI_COMPATIBLE_API_KEY="your-endpoint-key"     # optional, if the endpoint requires auth

# Server bind (only used by `python -m abstractcore.server.app`)
export HOST="0.0.0.0"
export PORT="8000"

# Debug mode
export ABSTRACTCORE_DEBUG=true

# Dangerous (multi-tenant hazard): allow unload_after for providers that can unload shared server state (e.g. Ollama)
export ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER=1

# Server security controls (recommended)
#
# - Request-level base_url overrides are loopback-only by default.
#   URL entries match scheme + exact host + default/explicit port + path-segment prefix.
#   Bare entries match hostname globs, e.g. "*.example.com".
export ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST="https://api.openai.com,https://example.com/v1"
#
# - Remote URL fetches for attachments are blocked for private/loopback/link-local targets by default (SSRF protection).
#   To allow specific hosts/prefixes, use the same structured allowlist syntax:
export ABSTRACTCORE_SERVER_URL_FETCH_ALLOWLIST="https://www.berkshirehathaway.com"
#
# - Local file paths in HTTP requests are disabled by default (including @/path/to/file in message strings).
#   To allow local file paths safely, restrict them under a single directory:
export ABSTRACTCORE_SERVER_MEDIA_ROOT="/srv/abstractcore-media"
#
# - Unsafe escape hatch: allow arbitrary local file paths from HTTP requests (not recommended)
export ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1
```

### Startup Options

```bash
# Using AbstractCore's built-in CLI
python -m abstractcore.server.app --help                    # View all options
python -m abstractcore.server.app --debug                   # Debug mode
python -m abstractcore.server.app --host 127.0.0.1 --port 8080  # Custom host/port
python -m abstractcore.server.app --debug --port 8001       # Debug on custom port

# Using uvicorn directly
uvicorn abstractcore.server.app:app --reload                # Development with auto-reload
uvicorn abstractcore.server.app:app --workers 4             # Production with multiple workers
uvicorn abstractcore.server.app:app --port 3000             # Custom port
```

---

## API Endpoints

### Endpoint Map

All API operations except `GET /health` use the same server auth policy:
send `Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY` when
`ABSTRACTCORE_SERVER_API_KEY` is configured. Provider-key overrides use
`X-AbstractCore-Provider-API-Key`; body/query `api_key` fields are intentionally
disabled.

| Group | Method | Endpoint | Purpose | Main parameters |
|---|---:|---|---|---|
| Health | GET | `/health` | Liveness/version probe; never requires auth | none |
| Discovery | GET | `/v1/models` | List models and filter by provider/capabilities | `provider`, `input_type`, `output_type`, `base_url` |
| Discovery | GET | `/providers` | Provider status/capabilities | `include_models` |
| Chat | POST | `/v1/chat/completions` | OpenAI-compatible chat, streaming, tools, media | `model`, `messages`, `stream`, `tools`, `tool_choice`, `temperature`, `max_tokens`, `base_url`, `agent_format`, `thinking` |
| Chat | POST | `/{provider}/v1/chat/completions` | Provider-scoped chat route where body model is unprefixed | path `provider`, body `model`, `messages`, chat parameters |
| Responses | POST | `/v1/responses` | Responses-style input API plus legacy chat body fallback | `model`, `input` or `messages`, `stream`, generation parameters |
| Embeddings | POST | `/v1/embeddings` | OpenAI-compatible embedding vectors | `model`, `input`, `dimensions`, `encoding_format`, `user`, `base_url` |
| Images | POST | `/v1/images/generations` | Text-to-image generation | `prompt`, optional `model`, `width`, `height`, `n`, `steps`, `guidance_scale`, `seed`, `quality`, `extra` |
| Images | POST | `/v1/images/edits` | Image edit/inpaint via multipart form | `prompt`, `image`, optional `mask`, `model`, `size`, `steps`, `guidance_scale`, `seed`, `extra_json` |
| Vision Jobs | POST | `/v1/vision/jobs/images/generations` | Async image generation with polling | same body as `/v1/images/generations` |
| Vision Jobs | POST | `/v1/vision/jobs/images/edits` | Async image edit with polling | same form fields as `/v1/images/edits` |
| Vision Jobs | GET | `/v1/vision/jobs/{job_id}` | Poll/consume async job state | path `job_id`, query `consume` |
| Vision Models | GET | `/v1/vision/models` | Cached local AbstractVision model inventory | none |
| Vision Models | GET | `/v1/vision/model` | Current in-memory local vision model | none |
| Vision Models | POST | `/v1/vision/model/load` | Load a local model into memory | `model_id` or `model` |
| Vision Models | POST | `/v1/vision/model/unload` | Best-effort unload active local model | none |
| Audio | POST | `/v1/audio/transcriptions` | Speech-to-text multipart endpoint | `file`, optional `model`, `language`, `prompt`, `response_format`, `temperature`, `format`, `base_url` |
| Audio | POST | `/v1/audio/speech` | Text-to-speech endpoint | `input`/`text`, optional `model`, `voice`, `response_format`/`format`, `speed`, `instructions`, `base_url` |
| Audio | POST | `/v1/voice/clone` | AbstractVoice-compatible voice-clone/custom-voice extension | `file`, optional `model`, `base_url`, `name`, `reference_text`, `validate` |
| Audio | POST | `/v1/audio/translations` | Reserved OpenAI-compatible translation route | `file`, `model`; returns `501` in this version |
| Audio | POST | `/v1/audio/music` | Extension endpoint for text-to-music plugins | `prompt`/`input`/`text`, `lyrics`, `format`; requires a music capability plugin |
| Prompt Cache | GET | `/acore/prompt_cache/stats` | Proxy cache stats to AbstractEndpoint | `base_url`; provider key header if upstream requires auth |
| Prompt Cache | GET | `/acore/prompt_cache/capabilities` | Proxy cache capability discovery | `base_url`; provider key header if upstream requires auth |
| Prompt Cache | POST | `/acore/prompt_cache/set` | Select/create upstream cache key | `base_url`, `key`, `make_default`, `ttl_s` |
| Prompt Cache | POST | `/acore/prompt_cache/update` | Prepare prompt/messages/tools upstream | `base_url`, `key`, `prompt` or `messages`, `system_prompt`, `tools`, `ttl_s` |
| Prompt Cache | POST | `/acore/prompt_cache/fork` | Fork one upstream cache key to another | `base_url`, `from_key`, `to_key`, `make_default`, `ttl_s` |
| Prompt Cache | POST | `/acore/prompt_cache/clear` | Clear upstream cache state | `base_url`, optional `key` |
| Prompt Cache | POST | `/acore/prompt_cache/prepare_modules` | Prepare reusable module/tool context upstream | `base_url`, `namespace`, `modules`, `make_default`, `ttl_s`, `version` |

### Shared Request Conventions

- `model` usually uses `provider/model` format, for example
  `openai/gpt-4o-mini`, `anthropic/claude-haiku-4-5`,
  `ollama/qwen3:4b`, `lmstudio/qwen/qwen3-vl-4b`, or
  `openai-compatible/my-model`.
- `base_url` is an AbstractCore extension for routing a provider to a specific
  OpenAI-compatible endpoint. Loopback URLs are allowed by default; non-loopback
  URLs require `ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST`.
- `X-AbstractCore-Provider-API-Key` overrides only the requested upstream
  provider for that request. It does not replace the AbstractCore server token.
- `api_key` body/query parameters remain in some schemas only as disabled
  compatibility placeholders; do not use them.
- Remote URL media fetches are SSRF-protected by default. Local file paths are
  disabled unless `ABSTRACTCORE_SERVER_MEDIA_ROOT` or
  `ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1` is configured.

### Chat Completions

**Endpoint:** `POST /v1/chat/completions`

Standard OpenAI-compatible endpoint. Works with all providers.

Server auth:
- If `ABSTRACTCORE_SERVER_API_KEY` is configured, every non-health endpoint requires
  `Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY`. Authenticated clients can use all
  provider keys/endpoints configured on the server.
- If `ABSTRACTCORE_SERVER_API_KEY` is not configured, `Authorization: Bearer <provider-key>`
  may be used as a bring-your-own upstream provider key. That key is forwarded only to the
  requested provider and never unlocks server-configured provider keys.
- Health checks (`GET /health`) are always unauthenticated.

**Request:**
```json
{
  "model": "provider/model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

**Key Parameters:**
- `model` (required): Prefer `"provider/model-name"` (e.g., `"openai/gpt-4o-mini"`). If you pass a bare model name (no `/`), the server will best-effort auto-detect a provider.
- `messages` (required): Array of message objects
- `stream` (optional): Enable streaming responses
- `tools` (optional): Tools for function calling
- `agent_format` (optional, AbstractCore extension): Tool-call syntax output format for agentic clients (`"auto"|"openai"|"codex"|"qwen3"|"llama3"|"gemma"|"xml"|"passthrough"`). When omitted, the server auto-detects from user-agent + model heuristics.
- `api_key` (deprecated/disabled, AbstractCore extension): Provider API keys are no longer accepted in request bodies or query strings. Configure provider keys on the server, use `X-AbstractCore-Provider-API-Key` for a per-request provider override, or use `Authorization` as a provider key only when `ABSTRACTCORE_SERVER_API_KEY` is not configured.
- `base_url` (optional, AbstractCore extension): Override the provider endpoint (include `/v1` for OpenAI-compatible servers like LM Studio / vLLM / OpenRouter)
- `unload_after` (optional, AbstractCore extension): If `true`, calls `llm.unload_model(model)` after the request completes. Disabled for `ollama/*` unless `ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER=1`.
- `prompt_cache_key` (optional, AbstractCore extension): Best-effort prompt caching key (semantics depend on provider/backend). See `docs/prompt-caching.md`.
- `prompt_cache_retention` (optional, AbstractCore extension): Prompt cache retention policy (OpenAI: `"in_memory"` or `"24h"`; ignored by other providers). See `docs/prompt-caching.md`.
- `thinking` (optional, AbstractCore extension): Unified thinking/reasoning control (`null|"auto"|"on"|"off"|"none"` or `"low"|"medium"|"high"|"xhigh"` when supported). Note: `"none"` is treated as an alias for `"off"`.
- `temperature`, `max_tokens`, `top_p`: Standard LLM parameters

#### Thinking (AbstractCore extension)

The server forwards `thinking` to the underlying provider using AbstractCore’s unified thinking mapping (see [Generation Parameters](generation-parameters.md)).

Example (route to LM Studio + Qwen3.5, disable thinking):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio/qwen3.5-27b@q4_k_m",
    "base_url": "http://localhost:1234/v1",
    "messages": [{"role": "user", "content": "Compute 17*23 - 19*11. Reply with the integer only."}],
    "thinking": "none",
    "max_tokens": 64
  }'
```

Notes:
- For **Qwen3 / Qwen3.5 on LM Studio**, `thinking="none"` maps to LM Studio’s template variables (`enable_thinking` / `enableThinking`) plus a Qwen template “hard switch” fallback (empty `<think></think>`) when needed. This avoids injecting “reasoning effort” instructions into the system prompt.
- Not every backend supports per-effort budgets for `low|medium|high`; when unavailable, levels degrade to “thinking enabled”.

**Example with streaming:**

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

stream = client.chat.completions.create(
    model="ollama/qwen3-coder:30b",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### Provider `base_url` override (AbstractCore extension)

Route a provider to a specific endpoint (useful for remote OpenAI-compatible servers):

Security notes:
- Request-level `base_url` overrides are **loopback-only by default**. To allow additional
  origins or host globs, set `ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST`. URL entries are parsed
  and matched on scheme, exact host, effective port, and path-segment prefix.
- If the server has an environment provider key set (e.g. `OPENAI_API_KEY`) and you route to a **non-loopback** `base_url`, the request is refused unless the provider key was supplied explicitly with `X-AbstractCore-Provider-API-Key`, or with `Authorization` when server auth is disabled.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio/qwen/qwen3-4b-2507",
    "base_url": "http://localhost:1234/v1",
    "messages": [{"role": "user", "content": "Hello from a remote LM Studio endpoint"}]
  }'
```

#### Provider Authentication

Do not put provider keys in request bodies or query strings. Those fields are disabled because
they leak through logs, shell history, browser history, and reverse proxies.

```bash
# Preferred: configure provider keys on the server and authenticate to AbstractCore.
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

When `ABSTRACTCORE_SERVER_API_KEY` is not configured, `Authorization: Bearer <provider-key>` may
be used as an upstream provider key. Once server auth is enabled, `Authorization` is reserved for
the AbstractCore server key and is never forwarded upstream.

To override a single upstream provider while still using the server master key, send the provider
key in `X-AbstractCore-Provider-API-Key`. The override applies only to the requested provider:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -H "X-AbstractCore-Provider-API-Key: $ANTHROPIC_API_KEY" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Provider-Specific Chat Route

**Endpoint:** `POST /{provider}/v1/chat/completions`

This route is useful for clients that already route by base URL path and expect
the body `model` to be provider-local. It is equivalent to using
`POST /v1/chat/completions` with `model="{provider}/{model}"`.

Parameters:
- Path `provider` (required): provider route prefix such as `openai`,
  `anthropic`, `ollama`, `openrouter`, `portkey`, `lmstudio`, `vllm`, or
  `openai-compatible`.
- Body `model` (required): provider-local model id, without the provider prefix.
- Body `messages`, `stream`, `tools`, `tool_choice`, `agent_format`,
  `thinking`, `base_url`, and other chat parameters behave like
  `/v1/chat/completions`.

Example:

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Media generation endpoints (optional)

AbstractCore Server can optionally expose OpenAI-compatible **image generation** and **audio** endpoints.

Important notes:
- These are **interoperability-first** endpoints (return `b64_json` or raw bytes), not an artifact-first durability contract.
- If the required plugin/backend is not available, the server returns `501` with actionable messaging.

#### Images (generate/edit)

Endpoints:
- `POST /v1/images/generations`
- `POST /v1/images/edits`

Remote OpenAI-compatible image proxying is included in `abstractcore[server]`
and is enabled by setting `ABSTRACTCORE_VISION_UPSTREAM_BASE_URL`
or the AbstractVision equivalent `ABSTRACTVISION_BASE_URL`.
The synchronous image routes use the same internal `generate(..., output="image")`
dispatcher as the Python API, then serialize the result back to the
OpenAI-compatible `b64_json` response shape.

Install for remote image proxying:
```bash
pip install "abstractcore[server]"
```

Install local image backends only when you want the server to load Diffusers or
stable-diffusion.cpp models itself:
```bash
pip install "abstractcore[server,vision]"
```

Use provider/model-style image ids:
- Omit `model` only when this server has a configured AbstractVision/OpenAI-compatible
  image default, for example via `ABSTRACTCORE_VISION_UPSTREAM_BASE_URL` /
  `ABSTRACTVISION_BASE_URL` plus an optional default model id.
- `diffusers/default` selects the configured local Diffusers default:
  `ABSTRACTCORE_VISION_MODEL_ID` / `ABSTRACTVISION_DIFFUSERS_MODEL_ID` /
  `ABSTRACTVISION_MODEL_ID`.
- `diffusers/<huggingface-repo>` selects an explicit local Diffusers model.
- `sdcpp/default` selects the configured stable-diffusion.cpp model.
- `openai-compatible/<model>` routes to the configured OpenAI-compatible image
  endpoint.

Local Diffusers generation is cache-only by default; set
`ABSTRACTCORE_VISION_ALLOW_DOWNLOAD=1` or
`ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` only when runtime downloads are
intentional.

`POST /v1/images/generations` JSON parameters:

| Field | Required | Notes |
|---|---:|---|
| `prompt` | yes | Text prompt to render. |
| `model` | no | Omit for the server's configured AbstractVision default. If present, use provider/model routing: `diffusers/default`, `diffusers/<huggingface-repo>`, `sdcpp/default`, or `openai-compatible/<model>`. |
| `width`, `height` | no | Requested output dimensions in pixels. These are the documented fields for JSON generation. Legacy `size: "WIDTHxHEIGHT"` is accepted for compatibility but not advertised. |
| `n` | no | Number of images; clamped to `1..10`. |
| `response_format` | no | Server response format. `b64_json` is the supported response shape. |
| `negative_prompt` | no | Local/backend-specific negative prompt. Strict OpenAI-compatible upstreams do not receive this top-level field; use `extra` only when your custom upstream supports it. |
| `seed` | no | Local deterministic seed. Strict OpenAI-compatible upstreams do not receive this top-level field; use `extra.seed` only when your custom upstream supports it. |
| `steps` | no | Local denoising/inference step count. Strict OpenAI-compatible upstreams do not receive this top-level field; use `extra.steps` only when your custom upstream supports it. |
| `guidance_scale` | no | Local classifier-free guidance scale. Strict OpenAI-compatible upstreams do not receive this top-level field; use `extra.guidance_scale` only when your custom upstream supports it. |
| `quality`, `style`, `user`, `background`, `output_format`, `output_compression`, `moderation` | no | Named OpenAI-compatible passthrough fields for upstream image endpoints. |
| `extra` | no | JSON object for backend-specific passthrough fields. Prefer this over arbitrary top-level keys so the schema stays explicit. |

`POST /v1/images/edits` multipart parameters:

| Field | Required | Notes |
|---|---:|---|
| `prompt` | yes | Edit/inpaint instruction. |
| `image` | yes | Source image file. |
| `mask` | no | Optional mask image for inpainting/edit-capable backends. |
| `model` | no | Same provider/model routing as generation; omit for the server default. |
| `size` | no | OpenAI-style edit output size such as `1024x1024`; multipart edit compatibility keeps this field. |
| `response_format` | no | Server response shape; `b64_json` is supported. |
| `negative_prompt`, `seed`, `steps`, `guidance_scale` | no | Local/backend-specific fields. Strict OpenAI-compatible upstreams do not receive them as top-level fields; use `extra_json` only when your custom upstream supports them. |
| `extra_json` | no | JSON object string with backend/upstream-specific parameters. |

Async image jobs are available when a request can take long enough that polling
is preferable:
- `POST /v1/vision/jobs/images/generations` uses the same JSON body as
  `/v1/images/generations` and returns `{"job_id": "..."}`.
- `POST /v1/vision/jobs/images/edits` uses the same multipart fields as
  `/v1/images/edits` and returns `{"job_id": "..."}`.
- `GET /v1/vision/jobs/{job_id}` returns `queued`, `running`, `succeeded`, or
  `failed`. Add `?consume=true` to remove a completed job from the in-memory job
  store after reading it.

Examples:

```bash
# Remote OpenAI-compatible image endpoint.
BASE=http://127.0.0.1:8000
TOKEN=replace-with-server-token

curl -sS -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-compatible/gpt-image-1","prompt":"A clean product photo of a red ceramic mug on a white table.","n":1,"width":1024,"height":1024,"response_format":"b64_json","quality":"low"}' \
  > /tmp/acore-image.json

python - <<'PY'
import base64
import json
from pathlib import Path

data = json.loads(Path("/tmp/acore-image.json").read_text())
Path("/tmp/acore-image.png").write_bytes(base64.b64decode(data["data"][0]["b64_json"]))
PY

# Image edit using the generated image.
curl -sS -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $TOKEN" \
  -F "model=openai-compatible/gpt-image-1" \
  -F "prompt=Make the mug blue while keeping the white table." \
  -F "image=@/tmp/acore-image.png;type=image/png" \
  -F "size=1024x1024" \
  -F "response_format=b64_json" \
  -F 'extra_json={"quality":"low"}' \
  > /tmp/acore-edit.json

python - <<'PY'
import base64
import json
from pathlib import Path

data = json.loads(Path("/tmp/acore-edit.json").read_text())
Path("/tmp/acore-edit.png").write_bytes(base64.b64decode(data["data"][0]["b64_json"]))
PY

# Configured server image default
curl -sS -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red fox in snow","width":512,"height":512,"response_format":"b64_json"}'
```

Local vision model helper endpoints:

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /v1/vision/models` | List local AbstractVision registry models that are present in known local caches. | Useful before trying local generation with `ABSTRACTCORE_VISION_ALLOW_DOWNLOAD=0`. |
| `GET /v1/vision/model` | Inspect the current in-memory local generation backend. | Reports `model_id`, `backend_kind`, `loaded_at_s`, and whether a backend is loaded. |
| `POST /v1/vision/model/load` | Preload a local model into memory. | JSON body accepts `model_id` or `model`; intended for local Diffusers/sdcpp backends, not remote proxy models. |
| `POST /v1/vision/model/unload` | Best-effort unload of the active local model. | Frees memory when the backend supports unloading. |

#### Audio (STT/TTS)

Endpoints:
- `POST /v1/audio/transcriptions` (multipart; `file=...`)
- `POST /v1/audio/speech` (json; `input=...`, optional `voice`, optional `format`)
- `POST /v1/voice/clone` (multipart; extension route for AbstractVoice-compatible voice cloning)
- `POST /v1/audio/translations` (multipart; reserved for compatibility, returns `501`)
- `POST /v1/audio/music` (json; extension endpoint, requires a music capability plugin)

Local plugin fallback is enabled when `model` is omitted. OpenAI SDK-style
clients that require a non-empty model string can use `abstractvoice/default`.

Remote provider routing is enabled when `model` is supplied in `provider/model` format:
- `openai/gpt-4o-mini-transcribe`, `openai/whisper-1`
- `openai/gpt-4o-mini-tts`, `openai/tts-1`
- `openrouter/...` for OpenRouter STT/TTS models
- `portkey/...` for Portkey-routed OpenAI-compatible audio models
- `openai-compatible/...` for endpoints that implement OpenAI-compatible audio routes

For `openai-compatible/...`, request-level `base_url` can point to a local
AbstractVoice/OpenAI-compatible audio server. Loopback URLs are allowed by
default; non-loopback URLs require `ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST`.

If `model` is omitted, the endpoint delegates to local capability plugins
(typically `abstractvoice`) and returns `501` when no suitable plugin is installed.
Those local/plugin paths use the same internal `generate(..., output=...)`
dispatcher as the Python API; provider/model remote routes keep their
OpenAI-compatible HTTP wire behavior.

Install for remote audio:
```bash
pip install "abstractcore[server,remote]"
```

Install for local plugin fallback:
```bash
pip install "abstractcore[server]"
pip install "abstractcore[voice]"
```

Notes:
- `abstractvoice` 0.9.0+ can install the base plugin path on Python 3.9,
  but Python 3.10+ is recommended. Optional/heavier engines and clone backends
  such as OpenF5/F5-TTS, Chroma, and OmniVoice are Python 3.10+ paths; AEC
  requires Python 3.11+.
- `/v1/audio/transcriptions` requires `python-multipart` for form parsing (included in the server extra).
- Uploaded audio is limited by `ABSTRACTCORE_SERVER_AUDIO_MAX_BYTES` (default: 25 MB).

`POST /v1/audio/transcriptions` multipart parameters:

| Field | Required | Notes |
|---|---:|---|
| `file` | yes | Audio file to transcribe, commonly `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, or `webm`. |
| `model` | no | Provider/model id for remote STT (`openai/gpt-4o-mini-transcribe`, `openai/whisper-1`, `openrouter/...`, `portkey/...`, `openai-compatible/...`). Omit for local `abstractvoice` plugin fallback; `abstractvoice/default` is accepted for clients that require a model string. |
| `language` | no | Input language code such as `en` or `fr`. |
| `prompt` | no | Provider transcription prompt/context. |
| `response_format` | no | Provider response format such as `json`, `text`, `srt`, or `vtt`. |
| `temperature` | no | Provider sampling temperature where supported. |
| `format` | no | Audio format override for providers that need it, notably OpenRouter base64 audio input. |
| `base_url` | no | OpenAI-compatible endpoint override for `openai-compatible/...`; loopback is allowed by default, non-loopback requires allowlist. |

`POST /v1/audio/speech` JSON parameters:

| Field | Required | Notes |
|---|---:|---|
| `input` or `text` | yes | Text to synthesize. `text` is the AbstractCore-compatible alias. |
| `model` | no | Provider/model id for remote TTS (`openai/gpt-4o-mini-tts`, `openai/tts-1`, `openrouter/...`, `portkey/...`, `openai-compatible/...`). Omit for local plugin fallback; `abstractvoice/default` is accepted. |
| `voice` | no | Provider/backend voice name; remote OpenAI-compatible routing defaults to `alloy`. OpenAI TTS voices include `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`, `verse`, `marin`, and `cedar`; the Swagger example uses `coral`. |
| `response_format` or `format` | no | Audio output format. Remote providers commonly support `mp3`, `wav`, `opus`, `aac`, `flac`, or `pcm`; local plugin fallback defaults to `wav`. |
| `speed` | no | Speech speed multiplier when supported. |
| `instructions` | no | Provider-specific style/instruction text for expressive TTS. |
| `provider` | no | Optional provider-routing options forwarded to compatible gateways. |
| `base_url` | no | Endpoint override for local/gateway routing. Prefer this with `openai-compatible/...`; if set with `openai/...`, the request is sent to that URL instead of api.openai.com. Loopback is allowed by default, non-loopback requires allowlist. |

Swagger UI can execute `/v1/audio/speech`. AbstractCore serves a small custom
Swagger wrapper that converts authenticated binary audio `POST` responses into
browser `blob:` URLs before Swagger renders the player. The example uses
`response_format="wav"` because WAV has explicit duration metadata and is the
most reliable inline preview format. If a browser still cannot play the inline
preview, use the response download or a curl `--output` command; the endpoint
returns normal `audio/*` bytes and includes a filename in `Content-Disposition`.

`POST /v1/voice/clone` multipart parameters:

| Field | Required | Notes |
|---|---:|---|
| `file` | yes | Reference voice audio file. |
| `model` | no | Provider/model id for remote clone routing. Use `openai-compatible/default` for an AbstractVoice-compatible server, or `openai/default` where OpenAI custom voice creation is available. Omit for local AbstractVoice clone fallback. |
| `name` | no | Friendly cloned voice name. |
| `reference_text` | no | Transcript of the reference audio when available. |
| `validate` | no | Ask compatible clone servers to validate/smoke-test the clone before returning. |
| `base_url` | no | OpenAI-compatible endpoint override for `openai-compatible/...`; loopback is allowed by default, non-loopback requires allowlist. |
| `clone_path` | no | Provider-specific clone path. Defaults to `/voice/clone` for OpenAI-compatible servers and `/audio/voices` for OpenAI. |
| `file_field` | no | Provider-specific multipart file field. Defaults to `file`; OpenAI uses `audio_sample`. |
| `consent` | no | Provider-specific consent id when custom voice creation requires it. |

The returned `voice_id` / `id` can be used as the `voice` value in
`/v1/audio/speech` when the selected backend supports custom voices.

`POST /v1/audio/music` JSON parameters:

| Field | Required | Notes |
|---|---:|---|
| `prompt` or `input` or `text` | yes | Music generation prompt. |
| `lyrics` | no | Optional lyrics for vocal music backends. |
| `response_format` or `format` | no | Only `wav` is supported in this server contract. |
| extra top-level fields | no | Best-effort passthrough to the installed music capability plugin. |

Examples:

```bash
BASE=http://127.0.0.1:8000
TOKEN=replace-with-server-token

# Local/plugin TTS through AbstractCore's unified output dispatcher.
curl -sS -X POST "$BASE/v1/audio/speech" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: audio/wav" \
  -d '{"input":"Hello from the updated AbstractCore server.","voice":"coral","response_format":"wav"}' \
  --output /tmp/acore-speech.wav

# Local/plugin STT through AbstractCore's unified output dispatcher.
curl -sS -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/tmp/acore-speech.wav;type=audio/wav" \
  -F "language=en"

# Remote speech-to-text (STT)
curl -sS -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@speech.wav" \
  -F "model=openai/gpt-4o-mini-transcribe" \
  -F "language=en"

# Remote text-to-speech (TTS)
curl -sS -X POST "$BASE/v1/audio/speech" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-4o-mini-tts","input":"Hello!","voice":"coral","response_format":"wav"}' \
  --output hello.wav

# Local abstractvoice TTS through the OpenAI-compatible endpoint
curl -sS -X POST "$BASE/v1/audio/speech" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"abstractvoice/default","input":"Hello!","voice":"alloy","format":"wav"}' \
  --output hello.wav

# Remote/local OpenAI-compatible voice clone endpoint
curl -sS -X POST "$BASE/v1/voice/clone" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@reference.wav" \
  -F "model=openai-compatible/default" \
  -F "base_url=http://127.0.0.1:5000/v1" \
  -F "name=my_voice" \
  -F "reference_text=Hello from my reference recording." \
  -F "validate=true"
```

If you want to “ask a model about an audio file”, prefer one of:
- Run STT first (`/v1/audio/transcriptions`) then send the transcript to `POST /v1/chat/completions`, or
- Configure the server’s default audio strategy (`config.audio.strategy`) to enable STT fallback for audio attachments, then attach audio in chat requests.

### Multimodal Requests (Images, Documents, Files)

AbstractCore server supports comprehensive file attachments using OpenAI-compatible multimodal message format, plus AbstractCore's convenient `@filename` syntax.

Security note (HTTP server): local file paths are disabled by default (including `@/path/to/file` and `{"url": "/path/to/file"}`).
Use `http(s)` URLs or `data:` base64, or enable local paths via `ABSTRACTCORE_SERVER_MEDIA_ROOT` (safe) / `ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1` (unsafe).

Image analysis example using a local generated image:

```bash
BASE=http://127.0.0.1:8000
TOKEN=replace-with-server-token

python - <<'PY'
import base64
from pathlib import Path

Path("/tmp/acore-image.b64").write_text(base64.b64encode(Path("/tmp/acore-image.png").read_bytes()).decode("ascii"))
PY

jq -n --rawfile img /tmp/acore-image.b64 '{
  model: "openai/gpt-4o-mini",
  messages: [{
    role: "user",
    content: [
      {type: "text", text: "Describe this image in one concise sentence."},
      {type: "image_url", image_url: {url: ("data:image/png;base64," + $img)}}
    ]
  }],
  max_tokens: 80,
  temperature: 0
}' > /tmp/acore-vision-chat.json

curl -sS -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/acore-vision-chat.json \
  | jq -r '.choices[0].message.content'
```

#### Supported File Types

- **Images**: PNG, JPEG, GIF, WEBP, BMP, TIFF
- **Documents**: PDF, DOCX, XLSX, PPTX
- **Data/Text**: CSV, TSV, TXT, MD, JSON, XML
- **Size Limits**: 10MB per file, 32MB total per request

#### Method 1: @filename Syntax (AbstractCore Extension)

Simple syntax that works with all providers (requires local paths enabled via `ABSTRACTCORE_SERVER_MEDIA_ROOT` or `ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1`):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [
      {"role": "user", "content": "What is in this document? @/path/to/report.pdf"}
    ]
  }'
```

#### Method 2: OpenAI Vision API Format (Image URLs)

Standard OpenAI format for images:

```json
{
  "model": "anthropic/claude-haiku-4-5",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/image.jpg"
          }
        }
      ]
    }
  ]
}
```

**Base64 Images:**
```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
  }
}
```

#### Method 3: OpenAI File Format (Forward-Compatible)

AbstractCore supports OpenAI's planned file format with simplified structure (consistent with image_url):

**File URL Format (Recommended - Same Pattern as image_url):**
```json
{
  "model": "ollama/qwen3:4b",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this document"},
        {
          "type": "file",
          "file_url": {
            "url": "https://example.com/documents/report.pdf"
          }
        }
      ]
    }
  ]
}
```

**Local File Path:**
```json
{
  "type": "file",
  "file_url": {
    "url": "/Users/username/documents/data.csv"
  }
}
```

Note: local file paths require `ABSTRACTCORE_SERVER_MEDIA_ROOT` (safe) or `ABSTRACTCORE_SERVER_ALLOW_LOCAL_FILES=1` (unsafe) on the server.

**Base64 Data URL:**
```json
{
  "type": "file",
  "file_url": {
    "url": "data:application/pdf;base64,JVBERi0xLjQKMSAwIG9iago<PAovVHlwZS..."
  }
}
```

**Filename Extraction:**
- **URLs/Paths**: Extracted automatically (`/path/file.pdf` → `file.pdf`)
- **Base64**: Generated from MIME type (`data:application/pdf;base64,...` → `document.pdf`)

#### Mixed Content Example

Combine text, images, and documents in a single request:

```json
{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Compare this chart with the data in the spreadsheet"},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS..."}
        },
        {
          "type": "file",
          "file_url": {
            "url": "https://example.com/data/sales_data.xlsx"
          }
        }
      ]
    }
  ]
}
```

#### Python Client Examples

**Using OpenAI Client:**
```python
import os
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

# Method 1: @filename syntax
response = client.chat.completions.create(
    model="anthropic/claude-haiku-4-5",
    messages=[{"role": "user", "content": "Summarize @document.pdf"}]
)

# Method 2: File URL (HTTP/HTTPS)
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the key findings?"},
            {
                "type": "file",
                "file_url": {
                    "url": "https://example.com/documents/report.pdf"
                }
            }
        ]
    }]
)

# Method 3: Local file path
response = client.chat.completions.create(
    model="anthropic/claude-haiku-4-5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this local document"},
            {
                "type": "file",
                "file_url": {
                    "url": "/Users/username/documents/report.pdf"
                }
            }
        ]
    }]
)

# Method 4: Base64 data URL
with open("report.pdf", "rb") as f:
    file_data = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="lmstudio/qwen/qwen3-next-80b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the key findings?"},
            {
                "type": "file",
                "file_url": {
                    "url": f"data:application/pdf;base64,{file_data}"
                }
            }
        ]
    }]
)
```

**Universal Provider Support:**
```python
# Same syntax works across all providers
providers_models = [
    "openai/gpt-4o",
    "anthropic/claude-haiku-4-5",
    "ollama/qwen2.5vl:7b",
    "lmstudio/qwen/qwen2.5-vl-7b"
]

for model in providers_models:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Analyze @data.csv and @chart.png"}]
    )
    print(f"{model}: {response.choices[0].message.content[:100]}...")
```

---

### OpenAI Responses API

**Endpoint:** `POST /v1/responses`

AbstractCore implements an OpenAI-compatible Responses-style API, including `input_file` support.

#### Why Use /v1/responses?

- **OpenAI Compatible**: Drop-in replacement for OpenAI's Responses API
- **Native File Support**: `input_file` type designed specifically for document attachments
- **Cleaner API**: Explicit separation between text (`input_text`) and files (`input_file`)
- **Backward Compatible**: Existing `messages` format still works alongside new `input` format
- **Optional Streaming**: Streaming opt-in with `"stream": true` (defaults to `false`)

#### Request Format

**OpenAI Responses API Format (Recommended):**
```json
{
  "model": "gpt-4o",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Analyze this document"},
        {"type": "input_file", "file_url": "https://example.com/report.pdf"}
      ]
    }
  ],
  "stream": false,
  "max_tokens": 2000,
  "temperature": 0.7
}
```

Key parameters:

| Field | Required | Notes |
|---|---:|---|
| `model` | yes | Provider/model id. Bare model ids may be auto-detected, but provider/model is preferred. |
| `input` | yes, unless `messages` is used | Responses-style array of input messages. Content items use `input_text` and `input_file`. |
| `messages` | yes, unless `input` is used | Backward-compatible chat-completions request shape. |
| `stream` | no | When `true`, returns server-sent events. |
| `max_tokens`, `temperature`, `top_p` | no | Standard generation controls, forwarded where supported. |
| `base_url`, `agent_format`, `thinking` | no | AbstractCore extensions with the same behavior as `/v1/chat/completions`. |

**Legacy Format (Still Supported):**
```json
{
  "model": "openai/gpt-4",
  "messages": [
    {"role": "user", "content": "Tell me a story"}
  ],
  "stream": false
}
```

#### Automatic Format Detection

The server automatically detects which format you're using:
- **OpenAI Format**: Presence of `input` field → converts to internal format
- **Legacy Format**: Presence of `messages` field → processes directly
- **Error**: Missing both fields → returns 400 error with clear message

#### Examples

**Simple Text Request:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lmstudio/qwen/qwen3-next-80b",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "What is Python?"}
        ]
      }
    ]
  }'
```

**File Analysis:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Analyze the letter and summarize key points"},
          {"type": "input_file", "file_url": "https://www.berkshirehathaway.com/letters/2024ltr.pdf"}
        ]
      }
    ]
  }'
```

**Multiple Files:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-haiku-4-5",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Compare these documents"},
          {"type": "input_file", "file_url": "https://example.com/report1.pdf"},
          {"type": "input_file", "file_url": "https://example.com/report2.pdf"},
          {"type": "input_file", "file_url": "https://example.com/chart.png"}
        ]
      }
    ],
    "max_tokens": 2000
  }'
```

**Streaming Response:**
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "Summarize this document"},
          {"type": "input_file", "file_url": "https://example.com/document.pdf"}
        ]
      }
    ],
    "stream": true
  }' --no-buffer
```

#### Supported Media Types

All file types supported via URL, local path, or base64:

- **Documents**: PDF, DOCX, XLSX, PPTX
- **Data Files**: CSV, TSV, JSON, XML
- **Text Files**: TXT, MD
- **Images**: PNG, JPEG, GIF, WEBP, BMP, TIFF
- **Size Limits**: 10MB per file, 32MB total per request

**Source Options:**
```json
// HTTP/HTTPS URL
{"type": "input_file", "file_url": "https://example.com/report.pdf"}

// Local file path
{"type": "input_file", "file_url": "/path/to/document.xlsx"}

// Base64 data URL
{"type": "input_file", "file_url": "data:application/pdf;base64,JVBERi0x..."}
```

#### Python Client Example

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["ABSTRACTCORE_SERVER_API_KEY"])

# Direct request to /v1/responses endpoint
import requests

response = requests.post(
    "http://localhost:8000/v1/responses",
    json={
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze this document"},
                    {"type": "input_file", "file_url": "https://example.com/report.pdf"}
                ]
            }
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

---

### Embeddings

**Endpoint:** `POST /v1/embeddings`

Generate embedding vectors for semantic search, RAG, and similarity analysis.

**Request:**
```json
{
  "input": "Text to embed",
  "model": "huggingface/sentence-transformers/all-MiniLM-L6-v2"
}
```

**Supported Providers:**
- **HuggingFace**: Local models with ONNX acceleration
- **Ollama**: `ollama/granite-embedding:278m`, etc.
- **LMStudio**: Any loaded embedding model
- **OpenAI**: `openai/text-embedding-3-small`, `openai/text-embedding-3-large`
- **OpenRouter**: `openrouter/openai/text-embedding-3-small`, etc.
- **Portkey**: `portkey/...` with your Portkey routing configuration
- **OpenAI-compatible**: `openai-compatible/...` against configured/local `/v1/embeddings` endpoints

Anthropic does not expose a native embeddings API. Use OpenAI, OpenRouter,
Portkey, an OpenAI-compatible endpoint, or a local embedding provider.

OpenAI-compatible request fields are forwarded where supported:
- `dimensions`
- `encoding_format`
- `user`
- `base_url` (AbstractCore extension; loopback by default, allowlist required for non-loopback)

Parameters:

| Field | Required | Notes |
|---|---:|---|
| `input` | yes | String or array of strings. Arrays return one vector per input item. |
| `model` | yes | Provider/model id such as `openai/text-embedding-3-small`, `openrouter/openai/text-embedding-3-small`, `portkey/...`, `openai-compatible/...`, `ollama/...`, `lmstudio/...`, or `huggingface/...`. |
| `encoding_format` | no | `float` by default; `base64` is accepted where supported by the provider/backend. |
| `dimensions` | no | Requested output dimensions for providers that support native dimension reduction; local backends may truncate when appropriate. |
| `user` | no | End-user identifier forwarded to providers that support abuse monitoring. |
| `base_url` | no | OpenAI-compatible endpoint override with the same allowlist policy as chat. |
| `api_key` | no | Deprecated/disabled in the body. Use `X-AbstractCore-Provider-API-Key` for provider overrides. |

**Batch Embedding:**
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["text 1", "text 2", "text 3"],
    "model": "ollama/granite-embedding:278m"
  }'
```

---

### Model Discovery

**Endpoint:** `GET /v1/models`

List all available models from configured providers.

**Query Parameters:**
- `provider`: Filter by provider (e.g., `ollama`, `openai`, `anthropic`, `lmstudio`, `openai-compatible`).
- `input_type`: Filter by input capability: `text`, `image`, `audio`, or `video`.
- `output_type`: Filter by output capability: `text` or `embeddings`.
- `base_url`: Optional upstream base URL override for providers that support OpenAI-compatible discovery. Loopback is allowed by default; non-loopback requires `ABSTRACTCORE_SERVER_BASE_URL_ALLOWLIST`.
- `api_key`: Deprecated/disabled query parameter. Use `X-AbstractCore-Provider-API-Key` for provider overrides.

**Examples:**
```bash
# All models
curl http://localhost:8000/v1/models

# Ollama models only
curl http://localhost:8000/v1/models?provider=ollama

# Embedding models only
curl http://localhost:8000/v1/models?output_type=embeddings

# Vision-capable input models
curl http://localhost:8000/v1/models?input_type=image

# Ollama embeddings
curl http://localhost:8000/v1/models?provider=ollama&output_type=embeddings
```

---

### Provider Status

**Endpoint:** `GET /providers`

List all available providers and their status.

**Query Parameters:**
- `include_models` (optional, default `false`): Include model lists for each
  provider. This is slower because it may query provider registries/endpoints.

**Response:**
```json
{
  "providers": [
    {
      "name": "ollama",
      "type": "llm",
      "model_count": 15,
      "status": "available"
    }
  ]
}
```

---

### Health Check

**Endpoint:** `GET /health`

Server health check for monitoring.

**Response:** includes `status`, server `version`, and enabled feature flags.

---

### Prompt Cache Control Plane

Prompt-cache routes are AbstractCore extensions for orchestration systems that
want one gateway to talk to an upstream
[AbstractEndpoint](endpoint.md) prompt-cache control plane. They proxy to the
`/acore/prompt_cache/*` routes on the upstream endpoint named by `base_url`.

These routes do not create local cache state in AbstractCore Server itself.
They normalize a supplied `base_url`, enforce the same server-side base URL
allowlist rules as other request-level routing, and forward provider auth only
from `X-AbstractCore-Provider-API-Key` or from `Authorization` when server auth
is disabled.

Common fields:

| Field | Location | Required | Notes |
|---|---|---:|---|
| `base_url` | query or JSON body | yes | Upstream AbstractEndpoint URL. It may include `/v1`; the proxy strips that suffix for control-plane calls. |
| `X-AbstractCore-Provider-API-Key` | header | no | Upstream endpoint token when required. |
| `api_key` | query/body | no | Deprecated/disabled; do not use. |
| `ttl_s` | JSON body | no | Optional upstream cache TTL in seconds, where supported. |

Operations:

| Endpoint | Method | Parameters | Result |
|---|---:|---|---|
| `/acore/prompt_cache/capabilities` | GET | `base_url` | Upstream supported cache features. |
| `/acore/prompt_cache/stats` | GET | `base_url` | Upstream cache stats. |
| `/acore/prompt_cache/set` | POST | `base_url`, `key`, `make_default`, `ttl_s` | Select/create a cache key upstream. |
| `/acore/prompt_cache/update` | POST | `base_url`, `key`, `prompt` or `messages`, `system_prompt`, `tools`, `add_generation_prompt`, `ttl_s` | Prepare prompt/messages/tools into an upstream cache key. |
| `/acore/prompt_cache/fork` | POST | `base_url`, `from_key`, `to_key`, `make_default`, `ttl_s` | Fork an existing upstream key. |
| `/acore/prompt_cache/clear` | POST | `base_url`, optional `key` | Clear a key or upstream default/all cache state, depending on backend support. |
| `/acore/prompt_cache/prepare_modules` | POST | `base_url`, `namespace`, `modules`, `make_default`, `ttl_s`, `version` | Prepare reusable module/tool context upstream. |

Example:

```bash
curl -X POST http://localhost:8000/acore/prompt_cache/update \
  -H "Authorization: Bearer $ABSTRACTCORE_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "http://127.0.0.1:8001/v1",
    "key": "project-default",
    "messages": [{"role": "system", "content": "You are concise."}],
    "ttl_s": 3600
  }'
```

---

## Agentic CLI integration

AbstractCore Server is **OpenAI-compatible**. Most OpenAI-compatible CLIs/SDKs can be pointed at it by setting:

- `OPENAI_BASE_URL="http://localhost:8000/v1"` (or an equivalent flag)
- `OPENAI_API_KEY="unused"` (many clients require a non-empty key even for local servers)

### Tool calling interoperability

- The server **does not execute tools** (it always returns tool calls; your host/runtime executes them).
- It can emit tool calls either as structured `tool_calls` (OpenAI/Codex style) **or** as tagged content for clients that parse tool calls from assistant text.
- Control the output format with `agent_format` (request body, AbstractCore extension), or rely on auto-detection (user-agent + model heuristics).

Supported `agent_format` values: `auto`, `openai`, `codex`, `qwen3`, `llama3`, `gemma`, `xml`, `passthrough`.

### Codex CLI (example)

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="unused"

codex --model "ollama/qwen3-coder:30b" "Write a factorial function"
```

### Forcing a format (curl)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3:4b-instruct-2507-q4_K_M",
    "messages": [{"role": "user", "content": "Use the tool."}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather by city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "agent_format": "llama3"
  }'
```

---

## Deployment

### Docker

Release images are published to GitHub Container Registry after the matching
PyPI release succeeds:

```bash
ghcr.io/lpalbou/abstractcore-server:<version>
```

The image is built from PyPI, not from the repository checkout, and installs:

```bash
abstractcore[server,remote,media,tokens,compression]==<version>
```

It includes remote chat/responses, remote embeddings, remote STT/TTS routing,
remote OpenAI-compatible image proxying, server dependencies, media parsing,
token counting, and compression helpers. It intentionally does not include
AbstractCore local LLM runtimes (`vllm`, `mlx`, `huggingface`), local embedding
dependencies (`sentence-transformers`), or the AbstractVoice/AbstractVision
local plugin runtimes because those pull large native inference stacks. Build a
custom image with `abstractcore[voice]` or `abstractcore[vision]` when local
voice/vision plugin execution is required.

**Run:**
```bash
docker pull ghcr.io/lpalbou/abstractcore-server:2.13.10
```

For local development, keep secrets in an uncommitted `.env` file:

```bash
ABSTRACTCORE_SERVER_API_KEY=replace-with-a-server-token
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
PORTKEY_API_KEY=pk_...
PORTKEY_CONFIG=pcfg_...
OPENAI_COMPATIBLE_BASE_URL=http://host.docker.internal:1234/v1
OPENAI_COMPATIBLE_API_KEY=optional
ABSTRACTCORE_VISION_UPSTREAM_BASE_URL=https://api.openai.com/v1
ABSTRACTCORE_VISION_UPSTREAM_API_KEY=sk-...
```

Then run the image with that environment file:

```bash
docker run --rm --name abstractcore-server \
  -p 127.0.0.1:8000:8000 \
  --env-file .env \
  ghcr.io/lpalbou/abstractcore-server:2.13.10
```

`ABSTRACTCORE_SERVER_API_KEY` is the AbstractCore server auth token. Clients
send it as `Authorization: Bearer <token>`, including from Swagger UI's
`Authorize` button. Provider keys such as `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
`ANTHROPIC_API_KEY`, and `PORTKEY_API_KEY` stay inside the server container.

Set `ABSTRACTCORE_SERVER_PROTECT_DOCS=1` if `/docs`, `/redoc`, and
`/openapi.json` should require the same server token.

For local OpenAI-compatible endpoints such as LM Studio or Ollama's `/v1`
server, point the container at a URL reachable from Docker:

```bash
docker run --rm --name abstractcore-server \
  -p 127.0.0.1:8000:8000 \
  -e ABSTRACTCORE_SERVER_API_KEY="$ABSTRACTCORE_SERVER_API_KEY" \
  -e OPENAI_COMPATIBLE_BASE_URL="http://host.docker.internal:1234/v1" \
  -e OPENAI_COMPATIBLE_API_KEY="$OPENAI_COMPATIBLE_API_KEY" \
  ghcr.io/lpalbou/abstractcore-server:2.13.10
```

### Docker Compose

```yaml
version: '3.8'

services:
  abstractcore:
    image: ghcr.io/lpalbou/abstractcore-server:2.13.10
    ports:
      - "8000:8000"
    environment:
      - ABSTRACTCORE_SERVER_API_KEY=${ABSTRACTCORE_SERVER_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - PORTKEY_API_KEY=${PORTKEY_API_KEY}
      - PORTKEY_CONFIG=${PORTKEY_CONFIG}
      - OPENAI_COMPATIBLE_BASE_URL=${OPENAI_COMPATIBLE_BASE_URL}
      - OPENAI_COMPATIBLE_API_KEY=${OPENAI_COMPATIBLE_API_KEY}
      - ABSTRACTCORE_VISION_UPSTREAM_BASE_URL=${ABSTRACTCORE_VISION_UPSTREAM_BASE_URL}
      - ABSTRACTCORE_VISION_UPSTREAM_API_KEY=${ABSTRACTCORE_VISION_UPSTREAM_API_KEY}
    restart: unless-stopped
```

### Production with Gunicorn

```bash
pip install gunicorn

gunicorn abstractcore.server.app:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

---

## Debug and Monitoring

### Enable Debug Mode

Debug mode provides comprehensive logging and detailed error reporting for troubleshooting API issues.

```bash
# Method 1: Using command line flag (recommended)
python -m abstractcore.server.app --debug

# Method 2: Using environment variable
export ABSTRACTCORE_DEBUG=true
python -m abstractcore.server.app

# Method 3: With uvicorn directly
export ABSTRACTCORE_DEBUG=true
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

### Debug Features

**Enhanced Error Reporting:**
- **Before**: Uninformative "422 Unprocessable Entity" messages
- **After**: Detailed field validation errors with request body capture

**Example Debug Output:**
```json
🔴 Request Validation Error (422) | method=POST | error_count=2 | errors=[
  {"field": "body -> model", "message": "Field required", "type": "missing"},
  {"field": "body -> messages", "message": "Field required", "type": "missing"}
] | client=127.0.0.1

📋 Request Body (Validation Error) | body={"invalid": "data"}
```

**Request/Response Tracking:**
- Full HTTP request details (method, URL, headers, client IP)
- Response status codes and processing times
- Structured JSON logging for machine processing

**Log Files:**
- `logs/abstractcore_TIMESTAMP.log` - Structured events
- `logs/YYYYMMDD-payloads.jsonl` - Full request bodies
- `logs/verbatim_TIMESTAMP.jsonl` - Complete I/O

**Useful Commands:**
```bash
# Find errors
grep '"level": "error"' logs/abstractcore_*.log

# Track token usage
cat logs/verbatim_*.jsonl | jq '.metadata.tokens | .input + .output' | \
  awk '{sum+=$1} END {print "Total:", sum}'

# Monitor specific model
grep '"model": "qwen3-coder:30b"' logs/verbatim_*.jsonl
```

## Common Patterns

### Multi-Provider Fallback

```python
import requests

providers = [
    "ollama/qwen3-coder:30b",
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5"
]

def generate_with_fallback(prompt):
    for model in providers:
        try:
            response = requests.post(
                "http://localhost:8000/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            continue
    raise Exception("All providers failed")
```

### Local Model Gateway

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-coder:30b

# Use via AbstractCore server
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

---

## Troubleshooting

### Server Won't Start

```bash
# Check port availability
lsof -i :8000

# Use different port
uvicorn abstractcore.server.app:app --port 3000
```

### No Models Available

```bash
# Check providers
curl http://localhost:8000/providers

# Check API keys
echo $OPENAI_API_KEY

# Start Ollama
ollama serve
ollama list
```

### Authentication Errors

```bash
# Set API keys
export ABSTRACTCORE_SERVER_API_KEY="acore-server-secret"
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Restart server after setting keys
```

---

## Why AbstractCore Server?

- **Universal**: One API for all providers  
- **OpenAI Compatible**: Drop-in replacement  
- **Simple**: Clean, focused endpoints  
- **Fast**: Lightweight, high-performance  
- **Debuggable**: Comprehensive logging  
- **CLI Ready**: Codex, Gemini CLI, Crush support  
- **Production Ready**: Docker, multi-worker, health checks  

---

## Related Documentation

- **[Getting Started](getting-started.md)** - Core library quick start
- **[Architecture](architecture.md)** - System architecture including server
- **[Python API Reference](api-reference.md)** - Core library API
- **[Embeddings Guide](embeddings.md)** - Embeddings deep dive
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

**AbstractCore Server** - One server, all models, any client.
