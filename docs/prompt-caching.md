# Prompt Caching (KV / Prefix Caches)

AbstractCore supports **best-effort prompt caching** via `prompt_cache_key`. The exact behavior depends on the provider/backend:

- Some providers treat it as a **hint** (server-managed caching).
- Some local runtimes can retain an **in-process KV/prefix cache** keyed by `prompt_cache_key`.

Prompt caching is most useful when many calls share a long, stable prefix (system prompt, tool schema, long context), because it reduces repeated prefill work (TTFT).

## Unified API surface

- `prompt_cache_key` (generation kwarg): forwarded to the provider when supported.
- `prompt_cache_retention` (OpenAI only): optional retention control (`"in_memory"` or `"24h"` when supported).
- `BaseProvider.get_prompt_cache_capabilities()`: returns a capability profile with a stable mode:
  - `none`: no prompt-cache support
  - `keyed`: accepts `prompt_cache_key` but does not expose a local control plane
  - `local_control_plane`: supports local key management / module preparation
- `BaseProvider.prompt_cache_supports_operation(operation)`: one place to query whether a specific control-plane operation is supported.
- `BaseProvider.prompt_cache_token_count(key=None)`: best-effort *live* token count for an in-process cache key (useful for observability in KV/local modes; typically `None` for server-managed caches).
- `BaseProvider` control plane (best-effort, capability-gated):
  - `prompt_cache_set(key)`
  - `prompt_cache_update(key, ...)`
  - `prompt_cache_fork(from_key, to_key)`
  - `prompt_cache_clear(key=None)`
  - `prompt_cache_prepare_modules(...)` (hierarchical/prefix module caches)
  - Persistence (local providers only):
    - `prompt_cache_save(key, filename, ...)`
    - `prompt_cache_load(filename, ...)`
- Unsupported control-plane calls raise structured prompt-cache errors (for example `PromptCacheUnsupportedError`) with `operation`, `code`, and `capabilities` so higher layers can catch and downgrade cleanly.

## Capability modes (examples)

Query at runtime:

```python
caps = llm.get_prompt_cache_capabilities()
print(caps.to_dict())
```

**Example: `mode="none"`**

```json
{
  "supported": false,
  "mode": "none",
  "supports_set": false,
  "supports_clear": false,
  "supports_update": false,
  "supports_fork": false,
  "supports_prepare_modules": false,
  "supports_stats": false,
  "supports_save": false,
  "supports_load": false,
  "supports_ttl": false,
  "notes": []
}
```

**Example: `mode="keyed"`**

```json
{
  "supported": true,
  "mode": "keyed",
  "supports_set": true,
  "supports_clear": true,
  "supports_update": false,
  "supports_fork": false,
  "supports_prepare_modules": false,
  "supports_stats": true,
  "supports_save": false,
  "supports_load": false,
  "supports_ttl": true,
  "notes": ["Provider accepts prompt cache keys but does not expose the full local prompt-cache control plane."]
}
```

**Example: `mode="local_control_plane"`**

```json
{
  "supported": true,
  "mode": "local_control_plane",
  "supports_set": true,
  "supports_clear": true,
  "supports_update": true,
  "supports_fork": true,
  "supports_prepare_modules": true,
  "supports_stats": true,
  "supports_save": true,
  "supports_load": true,
  "supports_ttl": true,
  "notes": ["…provider-specific notes…"]
}
```

## Provider status (Mar 2026)

- **OpenAI** (`OpenAIProvider`): forwards `prompt_cache_key` (server-managed) and `prompt_cache_retention` (best-effort; some models support `"24h"`).
- **Anthropic** (`AnthropicProvider`): enables Claude prompt caching via `cache_control` when `prompt_cache_key` is provided (server-managed; default ~5-minute TTL).
- **OpenAI-compatible** (`OpenAICompatibleProvider`, `LMStudioProvider`, `VLLMProvider`, …): forwards `prompt_cache_key` when provided (server-managed if the backend implements it).
- **MLX** (`MLXProvider`): supports in-process KV caches via `prompt_cache_key` and AbstractCore’s cache control plane.
  - CLI persistence: `abstractcore-chat` supports `/cache save|load` (writes/reads a `.safetensors` cache; model-locked).
- **HuggingFace (transformers)** (`HuggingFaceProvider` with `model_type="transformers"`): supports in-process KV reuse keyed by `prompt_cache_key` via `past_key_values` (`DynamicCache`).
  - Supports AbstractCore’s local prompt-cache control plane (`prompt_cache_update`, `prompt_cache_prepare_modules`, `prompt_cache_fork`, …).
  - Supports cache persistence via `prompt_cache_save()` / `prompt_cache_load()` (writes/reads `.safetensors`; model-locked).
  - Limitations: enabled only for standard text-generation models (decoder-only); vision/custom transformer backends do not currently expose prompt caching.
- **HuggingFace GGUF** (`HuggingFaceProvider` with llama.cpp): always supports keyed in-process RAM caches (`LlamaRAMCache`), and reports `mode=local_control_plane` when AbstractCore can render the model's llama.cpp chat format exactly for cache reuse.
  - Current exact renderers: `chatml-function-calling`, `llama-3`
  - Other GGUF chat formats remain `mode=keyed` until an exact cached prompt renderer is implemented.
  - Local control plane optimization: append-only updates tokenize/render only the delta segment; tools are kept in a stable prefix position so system/tools caches remain effective as the discussion grows.
  - Local control plane generation: when `prompt_cache_key` is set and the chat format is supported, AbstractCore can prefill from cached state snapshots and generate via `llm.generate(reset=False)` (instead of `create_chat_completion()`), which avoids llama-cpp-python chat handlers that reset/re-evaluate long prompts.
    - Disable via `ABSTRACTCORE_GGUF_CONTROL_PLANE=0` (falls back to llama-cpp-python’s chat completion API).
  - macOS Metal note: llama.cpp Metal offload can SIGABRT when `llama_cpp` is imported *after* PyTorch/transformers in the same process. AbstractCore pre-imports `llama_cpp` (best-effort) when creating providers on Apple Silicon to keep GGUF Metal usable even if you later use MLX / HuggingFace transformers.
    - If PyTorch/transformers is imported *before* AbstractCore can pre-import `llama_cpp` (for example your app imports `torch` first), AbstractCore disables GGUF Metal offload for safety. Override with `ABSTRACTCORE_GGUF_METAL_UNSAFE=1` (unsafe).
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

Implementation note: the CLI now calls `provider.prompt_cache_save()` / `provider.prompt_cache_load()` instead of reaching into provider internals (`_prompt_cache_store`).

## Sessions: `CachedSession`

For long chats, `CachedSession` promotes the CLI’s “prefill stable prefix once, then reuse” pattern into the library:

```python
from abstractcore import create_llm, CachedSession

llm = create_llm("mlx", model="mlx-community/Mistral-7B-Instruct-v0.1-4bit")
session = CachedSession(
    provider=llm,
    system_prompt="You are a helpful assistant.",
    tools=[...],
    prompt_cache_strategy="auto",  # chooses KV mode when supported
)

session.generate("Hello!")
session.generate("Now continue the discussion…")
```

HuggingFace transformers example (KV mode):

```python
from abstractcore import create_llm, CachedSession

llm = create_llm("huggingface", model="sshleifer/tiny-gpt2", device="cpu")
session = CachedSession(provider=llm, system_prompt="You are helpful.", prompt_cache_strategy="auto")

session.generate("Hello!", max_output_tokens=32)
session.generate("Continue.", max_output_tokens=32)
```

Behavior:
- **MLX / HuggingFace (transformers)**: uses the prompt cache as the context source-of-truth (`mode=kv`) and sends only delta prompts each turn after prefix prefill.
- **Others**: keeps a stable `prompt_cache_key` (`mode=key`) so server-managed caches / local prefix caches can hit consistently.

KV mode notes (MLX + HuggingFace transformers):
- `system_prompt`, `tools`, and prior `messages` are **session-level cached state**. Per-call overrides are ignored (and warn).
- `auto_compact=True` is disabled in KV mode because compaction mutates the transcript but cannot mutate the in-process KV cache without an explicit rebuild. Use `session.rebuild_prompt_cache()` after changing transcript state, or use `prompt_cache_strategy="key"` / `off` when you need compaction semantics.
  - Rationale: KV mode treats the in-process cache as the **context source-of-truth**. Allowing per-call overrides for `messages=`, `system_prompt=`, or `tools=` would create a divergence between (a) the transcript you think you sent and (b) the KV cache the model is actually continuing from. That divergence is subtle and can produce hard-to-debug failures (e.g., tool-call parsing mismatches, “memory” that won’t go away, or incorrect citations).
  - Changing `session.system_prompt` or `session.tools` triggers an automatic cache rebuild on the next `generate()` / `attach_files()` call so the prefix modules realign. For other transcript mutations (editing prior messages, clearing files, compaction), call `CachedSession.rebuild_prompt_cache()` so the KV cache and transcript realign.

## “Box caching” with modules (system/tools/discussion)

When a provider supports `prompt_cache_prepare_modules`, you can build stable prefix “boxes” and only invalidate what changed:

- module `system` → stable persona
- module `tools` → stable tool schema
- (optional) module `discussion_prefix` → immutable summary/memory
- session cache key → append-only growth per turn

The module fingerprints are canonicalized to reduce accidental cache invalidation:
- tools are sorted by name for stable ordering
- message dicts are normalized to a stable subset (`role`, `content`, and tool-call fields)

## File attachments as cache “boxes”

For fast iteration on large contexts, you often want file attachments (code, docs, CSVs, PDFs) to be appended **once** and then reused by KV/prefix caches.

`CachedSession` supports this via `attach_files()`:

- Each file becomes **1 dedicated message box** in the transcript (so history persists across turns).
- In `prompt_cache_mode="kv"`, the same box is also appended to the provider KV cache via `prompt_cache_update()`
  (because the KV cache is the context source-of-truth and `generate()` sends only delta prompts).
- In `prompt_cache_mode="key"`, the file box stays in the transcript and is synced into the provider’s cache on the next `generate()` call (or immediately by passing `prefill_key_mode_cache=True`).
  - The prompt-cache REPL demo enables key-mode prefill on attach so your first question after attaching a large file starts streaming quickly.

Example:

```python
from abstractcore import CachedSession, create_llm

llm = create_llm("mlx", model="mlx-community/Qwen3-4B")
session = CachedSession(provider=llm, system_prompt="You are helpful.", prompt_cache_strategy="auto")

session.attach_files(["README.md", "docs/prompt-caching.md"])
session.generate("Summarize the attached files.", max_output_tokens=128)
```

Notes / limitations:

- This helper extracts text only for `MediaType.TEXT` and `MediaType.DOCUMENT`. For images/audio/video, keep using `media=[...]` on `generate()`.
- Dedupe is stat-based (path + size + mtime). If a file changes after being attached, prefer clearing/rebuilding the session cache before re-attaching to avoid conflicting context.
- Performance benefits (KV/prefix reuse) are currently strongest for local providers with in-process caches: **MLX**, **HuggingFace transformers**, **HuggingFace GGUF**.
- `attach_files()` returns a JSON-ish dict with `attached`/`skipped`/`errors` and a `timing` breakdown (`extract_ms`, `cache_update_ms`, `total_ms`) for observability.

See also: `examples/prompt_cache_repl_demo.py` for an interactive demo with:

- `/cache stats` (capabilities + cache keys)
- `/boxes` (graphical per-box context breakdown + live cache token counts)
- `/stream` (toggle live assistant output; TTFT/TIFT are still reported for observability)
- `@file` attachments (file boxes)

Note: when a model emits inline thinking tags and AbstractCore strips them from visible output, the REPL shows a brief `…` indicator so you can still see that streaming has started.

## Endpoint server: prompt cache control plane

`abstractcore-endpoint` can expose prompt-cache controls under `/acore/prompt_cache/*` when the underlying provider supports them (see `docs/endpoint.md`).

Endpoint responses use a stable JSON shape:

- success: `{"supported": true, "operation": "...", ...}`
- unsupported: `{"supported": false, "operation": "...", "code": "prompt_cache_unsupported", "capabilities": {...}}`
- other errors: `{"supported": false, "operation": "...", "code": "prompt_cache_error", "capabilities": {...}}`

This makes the same capability contract available over HTTP, not only in-process.

Gateway/operator note:

- `abstractgateway` can save/load MLX, HuggingFace transformers, and GGUF in-process prompt caches.
- For GGUF, gateway save/load persists both the raw `LlamaRAMCache` state and the provider-side module metadata needed to keep cache keys/module branches meaningful after reload.

## Safety / limitations

- KV caches consume memory; large caches can be expensive.
- Reusing a cache key across unrelated prompts can contaminate context.
- Many remote OpenAI-compatible backends ignore unknown fields or differ in cache semantics; treat `prompt_cache_key` as best-effort.
- GGUF / llama.cpp: if you see crashes with Metal/MPS acceleration, force CPU for stability:
  - per-call/provider init: `create_llm("huggingface", ..., device="cpu", n_gpu_layers=0, ...)`
  - env override: `ABSTRACTCORE_HF_DEVICE=cpu`

## Next steps (unification ideas)

- Standardize save/load semantics beyond MLX/GGUF once more backends expose a serializable local KV state.
- Add retry-based fallbacks for OpenAI-compatible servers that reject cache-related fields.
- Extend exact cached-prompt renderers to additional GGUF chat formats without weakening the control-plane contract.
