# Prompt Caching (KV / Prefix Caches)

AbstractCore supports **best-effort prompt caching** via `prompt_cache_key`. The exact behavior depends on the provider/backend:

- Some providers treat it as a **hint** (server-managed caching).
- Some local runtimes can retain an **in-process KV/prefix cache** keyed by `prompt_cache_key`.

Prompt caching is most useful when many calls share a long, stable prefix (system prompt, tool schema, long context), because it reduces repeated prefill work (TTFT).

## Unified API surface

- `prompt_cache_key` (generation kwarg): forwarded to the provider when supported.
- `prompt_cache_retention` (OpenAI only): optional retention control (`"in_memory"` or `"24h"` when supported).
- `BaseProvider` control plane (best-effort):
  - `prompt_cache_set(key)`
  - `prompt_cache_update(key, ...)`
  - `prompt_cache_fork(from_key, to_key)`
  - `prompt_cache_clear(key=None)`
  - `prompt_cache_prepare_modules(...)` (hierarchical/prefix module caches)

## Provider status (Mar 2026)

- **OpenAI** (`OpenAIProvider`): forwards `prompt_cache_key` (server-managed) and `prompt_cache_retention` (best-effort; some models support `"24h"`).
- **Anthropic** (`AnthropicProvider`): enables Claude prompt caching via `cache_control` when `prompt_cache_key` is provided (server-managed; default ~5-minute TTL).
- **OpenAI-compatible** (`OpenAICompatibleProvider`, `LMStudioProvider`, `VLLMProvider`, …): forwards `prompt_cache_key` when provided (server-managed if the backend implements it).
- **MLX** (`MLXProvider`): supports in-process KV caches via `prompt_cache_key` and AbstractCore’s cache control plane.
  - CLI persistence: `abstractcore-chat` supports `/cache save|load` (writes/reads a `.safetensors` cache; model-locked).
- **HuggingFace GGUF** (`HuggingFaceProvider` with llama.cpp): supports an in-process RAM cache (`LlamaRAMCache`). No save/load yet.
- **Ollama** (`OllamaProvider`): no prompt-cache integration currently (Ollama manages context internally per request).

## OpenAI notes

OpenAI prompt caching is automatic for prompts with **1024+ tokens**. Use `prompt_cache_key` (an official OpenAI parameter) as a stable identifier to improve cache hit rates across similar requests (it replaces the legacy `user` field for caching/bucketing). Use `prompt_cache_retention` to request longer retention when supported:

- `in_memory` (default): typically 5–10 minutes of inactivity, up to ~1 hour (volatile GPU memory).
- `24h` (extended): up to 24 hours (model-dependent; currently includes frontier GPT-5.x and `gpt-4.1` per OpenAI docs).

You can observe cache hits via `usage.prompt_tokens_details.cached_tokens` in OpenAI responses.

## Anthropic notes

Anthropic prompt caching is enabled by sending `cache_control: {"type":"ephemeral"}` in the Messages API request body. Caching applies to the full prompt prefix (`tools`, `system`, then `messages`) up to the last cacheable block, and Anthropic also supports up to 4 explicit cache breakpoints for finer-grained invalidation. Default TTL is ~5 minutes, with an optional 1-hour TTL (`{"ttl":"1h"}`) at higher input-token cost.

In AbstractCore, `AnthropicProvider` enables automatic caching when `prompt_cache_key` is provided (the key itself is not sent to Anthropic; it’s treated as a unified toggle). Optionally set `prompt_cache_ttl="1h"` to request Anthropic’s 1-hour TTL.

## CLI: saving/loading MLX caches

In `abstractcore-chat` (MLX only):

```bash
/cache save chat_cache
/cache save chat_cache --q8
/cache load chat_cache
```

Notes:
- Caches are **model-locked**; loading a cache resets the transcript and uses the KV cache as the context source of truth.
- `--q8` quantizes the cache before saving (smaller, lossy).

## Endpoint server: prompt cache control plane

`abstractcore-endpoint` can expose prompt-cache controls under `/acore/prompt_cache/*` when the underlying provider supports them (see `docs/endpoint.md`).

## Safety / limitations

- KV caches consume memory; large caches can be expensive.
- Reusing a cache key across unrelated prompts can contaminate context.
- Many remote OpenAI-compatible backends ignore unknown fields or differ in cache semantics; treat `prompt_cache_key` as best-effort.

## Next steps (unification ideas)

- Standardize save/load semantics beyond MLX (e.g., GGUF backends where the runtime supports serialization).
- Add retry-based fallbacks for OpenAI-compatible servers that reject cache-related fields.
- Extend docs and capability registry fields to make prompt-cache support and limitations explicit per backend.
