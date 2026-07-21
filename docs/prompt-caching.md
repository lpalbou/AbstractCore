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

`prompt_cache_update(...)` accepts the same unified `thinking` control as `generate(...)`. When a backend implements reasoning control by changing prompt serialization, applying `thinking` during cache preparation keeps the cached prefix aligned with later generation calls.

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

## Provider status (May 2026)

- **OpenAI** (`OpenAIProvider`): forwards `prompt_cache_key` (server-managed) and `prompt_cache_retention` (best-effort; some models support `"24h"`).
- **Anthropic** (`AnthropicProvider`): places an explicit `cache_control` breakpoint on the last system block when `prompt_cache_key` is provided (caches the tools+system head; server-managed; default ~5-minute TTL). Cache reads/writes surface as `usage.cached_input_tokens` / `usage.cache_write_tokens`.
- **OpenAI-compatible** (`OpenAICompatibleProvider`, `LMStudioProvider`, `VLLMProvider`, …): forwards `prompt_cache_key` when provided (server-managed if the backend implements it). Servers that reject the field with HTTP 400 (e.g. OVH AI Endpoints) are handled automatically: the key is dropped, the request retried once, and the field suppressed for that provider instance; server-side automatic prefix caching (vLLM APC, LM Studio/llama.cpp slot reuse) still applies because it needs no request field — only a byte-stable prompt prefix.
- **MLX** (`MLXProvider`): supports in-process KV caches via `prompt_cache_key` and AbstractCore’s cache control plane.
  - CLI persistence: `abstractcore-chat` supports `/cache save|load` (writes/reads a `.safetensors` cache; model-locked).
  - Durable memory blocs: supports exact bloc artifacts through `ensure_bloc_kv_artifact(...)` /
    `load_bloc_kv_artifact(...)`.
- **HuggingFace (transformers)** (`HuggingFaceProvider` with `model_type="transformers"`): supports in-process KV reuse keyed by `prompt_cache_key` via provider-native `past_key_values` / `Cache` objects.
  - Supports AbstractCore’s local prompt-cache control plane (`prompt_cache_update`, `prompt_cache_prepare_modules`, `prompt_cache_fork`, …).
  - Supports cache persistence via `prompt_cache_save()` / `prompt_cache_load()` (writes/reads `.safetensors`; model-locked).
  - Durable memory blocs: supports exact bloc artifacts through the same public bloc API as MLX.
  - Current durable formats cover standard `DynamicCache` layer state, Qwen3.5/Qwen3Next-style
    tensor-list hybrid state, and Mamba-style tensor state when the Transformers cache class is
    constructible from model config. Other custom cache classes fail explicitly until an adapter is
    added.
  - Limitations: enabled only for standard text-generation models (decoder-only); vision/custom transformer backends do not currently expose prompt caching. There is no universal HuggingFace KV tensor format.
  - General HuggingFace model-load compatibility, including quantized Transformers checkpoints, is
    covered in `docs/huggingface-model-compatibility.md`.
- **HuggingFace GGUF** (`HuggingFaceProvider` with llama.cpp): always supports keyed in-process RAM caches (`LlamaRAMCache`), and reports `mode=local_control_plane` when AbstractCore can render the model's llama.cpp chat format exactly for cache reuse.
  - Current exact renderers: `chatml-function-calling`, `llama-3`, and Gemma4 `gemma_turn` through llama.cpp's model chat template.
  - Other GGUF chat formats remain `mode=keyed` until an exact cached prompt renderer is implemented.
  - Local control plane optimization: append-only updates tokenize/render only the delta segment; tools are kept in a stable prefix position so system/tools caches remain effective as the discussion grows.
  - Local control plane generation: when `prompt_cache_key` is set and the chat format is supported, AbstractCore can prefill from cached state snapshots and generate via `llm.generate(reset=False)` (instead of `create_chat_completion()`), which avoids llama-cpp-python chat handlers that reset/re-evaluate long prompts.
  - Plain-generate reuse (fixed 2026-07-14): a plain `generate(messages=…, prompt_cache_key="k")` growing-prefix loop — the runtime's actual calling shape, without explicit `prompt_cache_update` — now PERSISTS its prefill snapshot, so each warm turn loads the prior prefix and evaluates only the growing suffix. Live-measured on this lane: a ~10k-token system prompt drops from ~9 s/turn (full re-prefill, the prior behavior) to ~0.6 s/turn warm on Qwen3-4B-2507 and Gemma-4-E4B GGUFs, correct fact-recall throughout. Before the fix this lane gave zero in-process reuse and `llm.reset()` additionally forfeited llama.cpp's own prefix reuse.
  - Durable memory blocs: supports exact bloc artifacts only for exact-renderer chat formats.
    Unsupported chat formats remain keyed-only.
    - Disable via `ABSTRACTCORE_GGUF_CONTROL_PLANE=0` (falls back to llama-cpp-python’s chat completion API).
  - macOS Metal note: llama.cpp Metal offload can SIGABRT when `llama_cpp` is imported *after* PyTorch/transformers in the same process. AbstractCore pre-imports `llama_cpp` (best-effort) when creating providers on Apple Silicon to keep GGUF Metal usable even if you later use MLX / HuggingFace transformers.
    - If PyTorch/transformers is imported *before* AbstractCore can pre-import `llama_cpp` (for example your app imports `torch` first), AbstractCore disables GGUF Metal offload for safety. Override with `ABSTRACTCORE_GGUF_METAL_UNSAFE=1` (unsafe).
- **Ollama** (`OllamaProvider`): no prompt-cache integration currently (Ollama manages context internally per request).

## Measured performance (July 2026)

Measured 2026-07-12 on Apple Silicon (M-series Max, 128 GB unified memory). Two complementary
setups (development-workspace harnesses; raw logs retained with the benchmark report):

- **Prefill probe**: the same bytes sent warm vs cold (8–16k-token prefixes), isolating pure
  prefill cost, with a content-correctness gate (at temperature 0 the warm answer must
  byte-match a fresh-key cold run — speed without correct context is a failure).
- **Agent-loop bench**: a ReAct tool task where every call re-sends the growing transcript,
  run once with a byte-stable prefix (cache ON) and once with the prefix deliberately broken
  every cycle (cache OFF: a fresh nonce line prepended to the system prompt).

### Prefill probe (the cache's true effect, content-gated)

| Backend | Model | Prefix | Cold prefill | Warm prefill | Speedup |
|---|---|---|---|---|---|
| MLX in-process | Llama-3.2-1B-Instruct 4-bit | 12.7k tk | 2.7 s | 0.31 s | ×8.5 |
| MLX in-process | Qwen3-4B-Instruct-2507 4-bit | 8.5k tk | ~8 s (full feed) | ~0.5 s (238-token suffix feed) | ×15+ |
| GGUF / llama.cpp in-process | Llama-3.2-1B Q8_0 | 12.7k tk | 7.0 s | 0.31 s | ×23 |
| HuggingFace transformers in-process | Qwen3.5-4B bf16 | 16.4k tk | 38.9 s (same-bytes fresh key) | 3.3 s | **×12** |

The probe verdicts are content-gated: each warm answer byte-matched a same-bytes cold run at
temperature 0. Absolute seconds vary run to run under machine load (the transformers cold call
varies ±30% on identical work); ratios are the stable quantity.

### Agent loop, cache ON vs OFF (ReAct task, per-cycle view)

All lanes ran the Qwen3.5-4B family except the MLX rows (see the architecture note); different
quantizations per lane — **compare within a row, never across rows.** The broken-prefix arm
receives a different prompt by construction, so trajectories can diverge; rows below state the
comparable quantity per lane.

| Lane | Cache ON | Cache OFF (broken prefix) | The comparable quantity |
|---|---|---|---|
| LM Studio server (4-bit) | ~6.9 s/cycle, 3 cycles, 20.7 s total (10.9 s/cycle at ~10k-tk transcripts) | ~9.4 s/cycle, 3 cycles, 28.4 s total (19.0 s/cycle at ~10k tk) | per-call re-prefill ≈0.7–1.3 s at 3–4k tk, ~15 s at 10k tk; savings grow with transcript size |
| MLX in-process, pure attention (Qwen3-4B-2507 4-bit) | warm generate feeds ~240 tokens instead of ~8,200 | full re-prefill every cycle | the fed-token delta; end-to-end parity additionally requires the host to keep its per-turn cache maintenance incremental (see note ‡) |
| MLX in-process, hybrid (Qwen3.5-4B 4-bit) | snapshot/restore: warm turns feed the suffix only † | full re-prefill every cycle | the fed-token delta — `hit_restore` on warm turns after the cold turn 1 |
| GGUF / llama.cpp in-process (Q4_K_M) | warm prefill flat as the transcript grows | re-prefill grows with prompt size (~0.9→2.9 s over 2.3k→3.5k tk) | the slope: flat vs growing. Per-cycle wall delta ≈+0.5–0.8 s at these sizes |
| HuggingFace transformers in-process (bf16) | warm prefill ≤~2 s | re-prefill ≈7–12 s per call at 1.5–3.2k tk | per-cycle prefill delta ≈6–10 s; end-to-end totals are decode-dominated at bf16 (~8 tok/s) and NOT a cache measurement |

† **Hybrid-architecture note (Qwen3.5-class on MLX — covers Qwen3.5, Qwen3.6, and Ornith 1.0):**
Qwen3.6 is the same `qwen3_5`/`qwen3_5_moe` architecture as Qwen3.5 (a 3:1 Gated DeltaNet + full
attention stack; Hugging Face loads both with the same classes), and Ornith 1.0 (9B/35B/397B) is a
Qwen3.5 post-train — so all three mix attention layers (trimmable `KVCache`) with linear-attention
layers whose recurrent state (`ArraysCache`) is not trimmable. A recurrent state cannot be rewound,
so the trim-based delta path does not apply. MLX instead uses a **snapshot/restore lane** for these
models: it keeps one copied cache boundary per `prompt_cache_key` and, when the next full-context
prompt extends it, restores the copy and feeds only the new suffix (forward-only reuse; a one-time
`#FALLBACK` line records that the snapshot lane, not the trim lane, is active). A growing-prefix
agent loop therefore reuses its prefix on warm turns just as a pure-attention model does. Measured
on `Qwen3.5-4B-MLX-4bit` (6.6k-token transcript, one key): turn 1 does a full prefill; turns 2–3
report `outcome=hit_restore` and feed ~24 of 6.6k tokens, with fact recall correct throughout. The
first turn always pays a full prefill; divergent prompts (a changed prefix) rebuild and re-snapshot.

‡ **Host composition note:** the in-process delta engages when the host re-sends the full
context per call (`messages=` present) over a stable key. Hosts that additionally maintain the
cache through the control plane (`prompt_cache_update` per turn) must keep those updates
incremental — clearing and re-appending the whole transcript each turn re-introduces the full
prefill on the maintenance path even though `generate` itself deltas correctly.

Guidance:

- **What makes caches hit**: a byte-stable prefix — stable system prompt, stable tool schema
  (tool ORDER included), append-only transcript. Any change to the first bytes (even one line)
  forces a full re-prefill everywhere. Agent loops that only append satisfy this by construction.
- **Shared server for many short-lived clients**: an LM Studio-style server cache persists across
  client restarts and needs no request field — CLI tools and restarting processes get warm
  prefills for free. In-process caches die with the process.
- **Long-lived single process with a large context**: in-process caches (MLX, GGUF, transformers)
  give the largest ratios (×8–×23 measured) because nothing leaves the process — provided the
  model architecture supports the delta path (see the compatibility table below).
- **Prefill-vs-decode reality check**: caching removes prefill only. At small prompts (2–4k
  tokens) with slow decode, end-to-end gains look modest even when the cache works perfectly;
  the ratio becomes dramatic at long prefixes (10k+ tokens).
- Client-side prefix-reuse percentages (as printed by instrumented harnesses) measure whether the
  *precondition* held, not server hits; where a backend reports no cached-token usage fields,
  latency is the only hit evidence. Per-call "prefill estimate" columns derived from wall time
  minus decode are lower bounds (the decode-rate calibration is itself prefill-contaminated),
  and the rate-setting call always reads as zero — treat single-call estimates as indicative,
  slopes and matched-position deltas as evidence.

## Cache-compatible models per lane

Whether the full in-process cache path engages depends on the model's cache architecture.
AbstractCore never returns wrong context: unsupported shapes degrade to a correct rebuild with
a `#FALLBACK` warning. What degrades is the SAVINGS, not the answers.

### MLX (`MLXProvider`)

The delta path needs a trimmable cache (`mlx_lm can_trim_prompt_cache`).

| Model family (mlx_lm architecture) | Cache | Warm-call behavior | Verified |
|---|---|---|---|
| Llama 3.x (non-sliding), Qwen3 dense (`Qwen3-*-Instruct-2507`), Qwen3-MoE, Qwen2.x, Mistral-class, SmolLM — any model without a custom `make_cache` | pure `KVCache` | **Full delta**: trim drift, feed only the suffix | ✅ 2026-07-14: `Qwen3-4B-Instruct-2507-4bit` → `outcome=hit_extend`, fed 37 of 485 tk, ×6 end-to-end at a 12.7k-tk prefix, fact-recall correct |
| Falcon-H1, LongCat-Flash, DeepSeek-V3.2 | `CacheList` of `KVCache` | Full delta (countable and trimmable) | prior |
| Gemma-3/3n/4, GPT-OSS, Cohere2, Ministral3, sliding-window Llama variants | mix `KVCache` + `RotatingKVCache` | Full delta only while the transcript is under the smallest sliding window (GPT-OSS: 128 tokens; Gemma: 512–1024) — then rebuild-per-call with `#FALLBACK`. **NOT a functional cache for agent workloads** (operator ruling 2026-07-19: a window smaller than any real transcript means the delta lane is effectively dead — a `RotatingKVCache` physically discards positions past the window, so the rewind the delta lane needs becomes impossible). Serve these models via LM Studio/vLLM instead, where the engine's forward-only slot/prefix cache works regardless of the window. | ✅ 2026-07-14: `gemma-4-31b-mxfp4` (RotatingKVCache local + KVCache global) → `hit_extend` at ~450-tk transcript (under window), fact-recall correct — but see the verdict: under-window only |
| Llama-4 (Scout/Maverick) | `ChunkedKVCache` (8k chunks) | Delta within the live chunk; partial trims refuse → rebuild | prior |
| Qwen3.5, **Qwen3.6** (same `qwen3_5`/`qwen3_5_moe` hybrid, 3:1 Gated DeltaNet), **Ornith 1.0** (9B/35B/397B — Qwen3.5 post-trains), Qwen3-Next, Nemotron-H, Jamba, LFM2/2.5, Granite-hybrid, Plamo2, Baichuan-M1 | hybrid with untrimmable `ArraysCache` (Gated DeltaNet) layers | **Snapshot/restore delta** (`outcome=hit_restore`): the recurrent state can't be trimmed but it can be copied, so a per-key boundary snapshot is restored and only the suffix is fed on warm growing-prefix turns. Turn 1 pays a full prefill; a divergent prefix rebuilds and re-snapshots. Server lane (LM Studio/vLLM) additionally benefits from the deployed engine's hybrid-checkpoint prefix cache. | ✅ 2026-07-15: `Qwen3.5-4B-MLX-4bit` → turns 2–3 `outcome=hit_restore`, fed ~24 of 6.6k tokens, fact-recall correct; `Qwen3.6-27B/35B-A3B`, `Ornith-1.0-9B` share the arch (census on the 4B: 24×`ArraysCache` + 8×`KVCache`, 3:1, `can_trim_prompt_cache=False`) |
| Mamba/Mamba2, RWKV-7 (pure SSM) | all `ArraysCache` | Snapshot/restore delta (`hit_restore`): state not trimmable but copyable — same lane as the hybrids | ✅ 2026-07-15 (lane shared with the Gated-DeltaNet hybrids) |

Check your model: the first warm full-context call logs `#FALLBACK … not trimmable` /
`… not countable` if the model cannot delta; no warning means the delta path engaged.
`get_prompt_cache_stats()` exposes per-key `token_count` to watch reuse across calls. The ✅ rows
were verified live with `scripts/verify_prompt_cache_families.py` (JSON reports + cache-census
probes retained with the 2026-07-14 benchmark logs).

### GGUF / llama.cpp (`HuggingFaceProvider`)

The full control plane (`mode=local_control_plane`: delta-only updates, prefill-snapshot
generation) requires an exact chat renderer; other formats stay `mode=keyed`, where llama.cpp's
own in-process prefix reuse still applies (real savings, less control).

| GGUF family | Renderer | Mode | Verified |
|---|---|---|---|
| Qwen3 dense (`Qwen3-4B-Instruct-2507`), any "qwen"-named GGUF, ChatML-template models | ChatML (`chatml` / `chatml-function-calling`) | `local_control_plane` | ✅ 2026-07-14: warm ~0.6 s vs cold ~9 s at 10k-tk prefix, fact-recall correct |
| Qwen3.5 / Qwen3.6 hybrids (`Qwen3.5-*-GGUF`, arch `qwen35`/`qwen35moe`) | ChatML | `local_control_plane` | ✅ 2026-07-14: loads under llama.cpp 0.3.23, warm ~1 s, fact-recall correct (recurrent state round-trips through the snapshot; llama.cpp's in-place trim would refuse it) |
| Gemma-4 (`gemma-4-*-GGUF`, arch `gemma4`) | GGUF-embedded chat template (`llama-cpp-chat-template`) | `local_control_plane` | ✅ 2026-07-14: warm ~0.5 s vs cold ~9 s, fact-recall correct |
| Llama-3.x (when the embedded template matches llama.cpp's llama-3 template) | `llama-3` | `local_control_plane` | prior |
| **Ornith 1.0 GGUF** (`ornith-*.gguf` — a Qwen3.5 post-train shipping a `chat_template.default` ChatML Jinja template) — and ANY GGUF whose embedded Jinja template is ChatML | GGUF-embedded chat template (`llama-cpp-chat-template`) | `local_control_plane` | ✅ 2026-07-19 (backlog 0821 closed): detection is by template CONTENT (ChatML markers + a probe render proving the template is ChatML-shaped), never by model name or llama.cpp's guessed format id. Live on the real `ornith-1.0-35b-Q4_K_M.gguf`: warm/cold 0.34–0.35, fact-recall correct, zero warnings; adversary-reviewed (probe false positives rejected; snapshot reader hardened to reuse what a state HOLDS, never what its key claims; stream render failures degrade instead of raising). |
| Mistral, Phi, DeepSeek, Granite, everything else | — | `keyed` (llama.cpp-native prefix reuse only) | prior |

Check your model: `get_prompt_cache_capabilities()` reports the honest per-model `mode` and the
active renderer in `notes`. All ✅ rows were verified live with
`scripts/verify_prompt_cache_families.py` (growing-prefix ReAct shape, one cache key, fact-recall
correctness gate) on the quant noted in the benchmark logs.

### HuggingFace transformers (`HuggingFaceProvider` with `model_type="transformers"`)

The full-context delta path crops the cache back to the shared prefix (`Cache.crop`).

| Architecture | Warm-call behavior |
|---|---|
| Pure-attention decoders (Llama-class, SmolLM2, Qwen3 dense) | **Full delta, byte-exact** |
| Hybrids with linear-attention layers (Qwen3.5, Qwen3-Next, Jamba, LFM2) | Delta with exact attention-KV reuse; the linear layers' recurrent state cannot be rolled back and remains an approximation after multi-turn reuse (labeled with a one-time `#FALLBACK`; output quality typically holds — measured ×12 on Qwen3.5-4B with content-gate pass) |
| Sliding-window models past their window (Gemma-4 class, window 512) | Crop refuses → rebuild-per-warm-call with `#FALLBACK` |
| Hybrids whose crop is a no-op on attention KV too (Zamba class) | Detected by a post-crop length verify → rebuild (never wrong context) |

Check your model: `model.config.layer_types` — all `full_attention` means byte-exact delta; any
`linear_attention`/`mamba` means fast-but-approximate; `sliding_attention` means window-bounded.
Empirically, compare a warm call's wall time against a fresh-key cold call of the same bytes.

## Durable memory bloc artifacts

For local providers with a full control plane, AbstractCore can derive one durable prompt-cache
artifact from one bloc:

```text
1 text/file -> 1 bloc -> 1 provider/model cache artifact
```

Use the public helpers from either `abstractcore` or `abstractcore.core`:

```python
from abstractcore import ensure_bloc_kv_artifact, load_bloc_kv_artifact

ensure = ensure_bloc_kv_artifact(provider=llm, store=store, record=record, debug=True)
loaded = load_bloc_kv_artifact(provider=llm, store=store, record=record, key="work:doc")

response = llm.generate(
    "Use the loaded bloc.",
    prompt_cache_binding=loaded.prompt_cache_binding,
)
```

The shared shape works for:

- MLX (`.safetensors`)
- HuggingFace transformers (`.safetensors`)
- HuggingFace GGUF exact-renderer paths (`.npz`)

The artifact payload is provider-native. The shared abstraction is the manifest, binding, Python
helper, and server route shape; saved tensors/state are not portable between provider/model pairs.
The manifest records provider, model, rendered recipe hash, cache backend, artifact format, and
binding metadata so incompatible artifacts are rejected instead of guessed.

`prompt_cache_key` remains a volatile runtime handle. `prompt_cache_binding` is optional; when
supplied, generation verifies that the key is still loaded with the exact artifact returned by
`load_bloc_kv_artifact(...)`. Missing or stale bindings fail with structured prompt-cache errors
before generation or streaming starts. Without a binding, existing best-effort key behavior is
unchanged.

For server and endpoint use, `/acore/blocs/kv/load` returns
`artifact.prompt_cache_binding`; pass that object to `/v1/chat/completions` as
`prompt_cache_binding`.

Deletion and pruning are also part of the public contract. Use
`list_bloc_kv_artifacts(...)`, `find_bloc_kv_live_bindings(...)`,
`delete_bloc_kv_artifact(...)`, `prune_bloc_kv_artifacts(...)`, and `delete_bloc(...)` in Python, or
the matching `/acore/blocs` and `/acore/blocs/kv/*` delete/list/prune routes. Deleting a loaded KV
artifact is blocked by default; pass `clear_loaded=true` to the route, or `clear_loaded=True` to
the helper, when you want Core to clear matching runtime keys before removing artifact files.

Set `debug=true` on the bloc ensure/load routes, pass `debug=True` to the Python helpers, or set
`ABSTRACTCORE_BLOC_KV_DEBUG=1` to return verbose proof fields such as provider backend, artifact
format, manifest path, artifact hash, binding id, rendered recipe hash, and token count when
available.

## OpenAI notes

OpenAI prompt caching is automatic for prompts with **1024+ tokens**. Use `prompt_cache_key` (an official OpenAI parameter) as a stable identifier to improve cache hit rates across similar requests (it replaces the legacy `user` field for caching/bucketing). Use `prompt_cache_retention` to request longer retention when supported:

- `in_memory` (default): typically 5–10 minutes of inactivity, up to ~1 hour (volatile GPU memory).
- `24h` (extended): up to 24 hours (model-dependent; currently includes frontier GPT-5.x and `gpt-4.1` per OpenAI docs).

You can observe cache hits via `usage.prompt_tokens_details.cached_tokens` in OpenAI responses.

## Anthropic notes

Anthropic prompt caching uses explicit `cache_control: {"type":"ephemeral"}` breakpoints on content blocks. Caching applies to the prompt prefix (`tools`, `system`, then `messages`) up to each marked block; up to 4 breakpoints are supported. Default TTL is ~5 minutes, refreshed on hit, with an optional 1-hour TTL (`{"ttl":"1h"}`) at higher cache-write cost. Cache writes bill at 1.25x input (2x for 1h) and cache reads at 0.1x.

In AbstractCore, `AnthropicProvider` places a breakpoint on the last `system` text block when `prompt_cache_key` is provided (the key itself is not sent to Anthropic; it signals caching intent). This caches the tools + system static head — the byte-stable part of agent-loop requests — while the growing message transcript stays unmarked, so the write premium is only ever paid on content later calls can actually re-read. Optionally set `prompt_cache_ttl="1h"` for the 1-hour tier.

Cache traffic is reported in normalized usage keys: `usage.cached_input_tokens` (read) and `usage.cache_write_tokens` (creation). Anthropic's raw `input_tokens` excludes cache traffic; AbstractCore reports the inclusive prompt size in `input_tokens` so totals stay comparable across providers. When these keys are absent the provider did not report cache fields (absent is distinct from a measured zero).

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

See also: `examples/prompt_caching/prompt_cache_repl_demo.py` for an interactive demo with:

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

The HTTP control plane mirrors this: `/acore/prompt_cache/update` accepts optional `thinking` so warm cache state can be prepared with the same reasoning mode you intend to use at inference time.

Server/operator note:

- Core exposes provider-level `prompt_cache_save(...)` / `prompt_cache_load(...)` for Python and
  CLI/operator use, but the HTTP server and endpoint do not currently expose generic
  `/acore/prompt_cache/save` or `/acore/prompt_cache/load` routes.
- Public persistent HTTP cache artifacts are represented by durable memory blocs through
  `/acore/blocs/kv/*`, which return `prompt_cache_binding` for exact request-time verification.
- Generic live prompt-cache snapshot save/load is intentionally not a thin-client contract. The
  backlog tracks it separately as a possible authenticated local-admin snapshot surface or explicit
  de-scope decision.

## Safety / limitations

- KV caches consume memory; large caches can be expensive.
- Reusing a cache key across unrelated prompts can contaminate context.
- Durable bloc artifacts are exact-prefix artifacts, not composable KV blocks. Do not merge
  independent cache artifacts or treat them as the durable source of truth; the bloc text remains
  primary.
- Many remote OpenAI-compatible backends ignore unknown fields or differ in cache semantics; treat `prompt_cache_key` as best-effort.
- GGUF / llama.cpp: if you see crashes with Metal/MPS acceleration, force CPU for stability:
  - per-call/provider init: `create_llm("huggingface", ..., device="cpu", n_gpu_layers=0, ...)`
  - env override: `ABSTRACTCORE_HF_DEVICE=cpu`

## Next steps (unification ideas)

- Add retry-based fallbacks for OpenAI-compatible servers that reject cache-related fields.
- Extend exact cached-prompt renderers to additional GGUF chat formats without weakening the control-plane contract.
