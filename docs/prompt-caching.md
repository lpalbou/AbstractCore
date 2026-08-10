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
  - **Requires torch >= 2.11 on Apple silicon** — see
    [Requirements](#huggingface-transformers-huggingfaceprovider-with-model_typetransformers).
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
  - Plain-generate reuse: a plain `generate(messages=…, prompt_cache_key="k")` growing-prefix loop — the shape a runtime agent loop produces, with no explicit `prompt_cache_update` — persists its prefill snapshot, so each warm turn loads the stored prefix and evaluates only the growing suffix. Live-measured on this lane with Metal offload active: a ~10k-token system prompt costs ~0.6 s/turn warm against ~9 s/turn for a full re-prefill, on Qwen3-4B-2507 and Gemma-4-E4B GGUFs, with fact recall correct throughout.
  - Durable memory blocs: supports exact bloc artifacts only for exact-renderer chat formats.
    Unsupported chat formats remain keyed-only.
    - Disable via `ABSTRACTCORE_GGUF_CONTROL_PLANE=0` (falls back to llama-cpp-python’s chat completion API).
  - macOS Metal note: llama.cpp Metal offload can SIGABRT when `llama_cpp` is imported *after* PyTorch/transformers in the same process. AbstractCore pre-imports `llama_cpp` (best-effort) when creating providers on Apple Silicon to keep GGUF Metal usable even if you later use MLX / HuggingFace transformers.
    - If PyTorch/transformers is imported *before* AbstractCore can pre-import `llama_cpp` (for example your app imports `torch` first), AbstractCore disables GGUF Metal offload for safety. Override with `ABSTRACTCORE_GGUF_METAL_UNSAFE=1` (unsafe).
- **Ollama** (`OllamaProvider`): no prompt-cache integration currently (Ollama manages context internally per request).

## Measured performance

Each subsection states its own scope. **Compare within a subsection, never across them**: a prefill
probe isolates prefill cost, while an end-to-end session measures prefill *and* decode *and* client
overhead, so the two answer different questions and produce different ratios for the same cache.

Every figure in this section was taken on one machine — **Apple M5 Max, 128 GB unified memory,
macOS 26.3** — with one model resident at a time and no sibling inference running.

### How these numbers are taken

Every figure in this section follows the same protocol. If you reproduce a cell, use it; if you
compare against a figure taken any other way, the comparison is not valid.

- **Probe prefill, not end-to-end wall clock.** The cache changes prompt preprocessing, not decode.
  Pin the output length identically across arms, assert the realized length is equal, and measure a
  decode floor per cell so it can be subtracted. Arms with different output lengths measure decode.
- **A ratio is published only when the warm median sits above the measured decode floor.** Three
  transformers cells warm faster than their own decode floor (`Qwen3.5-4B` and `Ornith-1.0-9B` at
  10k, `Qwen3.6-27B` at 10k); for those the wall ratio is the only publishable number and no
  floor-subtracted figure is quoted.
- **Repeat every arm and report the median with its spread.** Single-sample wall clock on a shared
  machine varies by as much as the effect being measured. Cells here run 3, 5 or 7 repetitions per
  arm; each table states its own count, and rows measured at 3 say so. A ratio is published only when
  every arm of the cell completed cleanly (`n_ok == reps`): a failed or empty completion returns in
  well under a millisecond and would otherwise enter the median as a spectacular cache win.
- **A ratio is published as a point estimate only when the cold arms are tight.** The bound is that
  each cold arm spans no more than ×1.5 from its fastest to its slowest repetition, and that the
  two cold arms agree with each other to within ×0.8–×1.25. A cell that misses it is published as a
  **range** — bounded by the slowest cold repetition over the fastest warm one and the fastest cold
  over the slowest warm — and is labelled as a range in its row. Two rows in the transformers table
  below are ranges for this reason.
- **`max_output_tokens` in the constructor is an instance ceiling** that clamps every per-call
  value. Build with a ceiling large enough for the largest arm (64 is a good default for probes),
  pin the value per call, and check the realized budget.
- **Pin `thinking=False` identically across arms.** Reasoning models return empty visible content at
  small budgets and raise `EmptyCompletionError`.
- **Read reuse from a ground-truth token count, not from `usage.input_tokens`** — that field is a
  fast heuristic estimate of prompt length, not a measurement, and it does not agree with the
  measured length. The MLX rows and the hybrid transformers rows read
  `response.metadata["prompt_cache"]` (`outcome`, `cached_tokens`, `fed_tokens`), which the provider
  attests; the dense transformers rows use a forward-pre-hook count and the GGUF rows an `eval()`
  spy, both of which count tokens actually pushed through a forward pass.
- **On untrimmable architectures, compare warm against a plan-matched cold run**, not a single-call
  cold run, or the measurement is dominated by the prefill chunk plan rather than the cache.
- **The matrix `warm_grow` arm is not an agent-loop turn 2.** The matrix arm can contain an
  identical-bytes turn in between, which changes the outcome. Use the agent-loop tables for per-lane
  verdicts.

**Ratios are the comparable quantity; absolute seconds are not.** Two measurements show this from
opposite directions. Within a single cell, the no-cache arm and the cold-with-cache arm do the same
amount of prefill work, and their absolute seconds still differ by up to about 25% (×0.80–×1.23
across the MLX cells). And when the dense transformers cell at 10k was measured twice with its
absolute prefill seconds differing by a factor of six, its speedup differed by 3% — ×138.9 against
×142.5. Quote ratios, quote the context size with them, and treat absolute seconds as descriptive of
the run they came from rather than as a portable number.

### The measured matrix

Three lanes, identical columns. **Cold prefill** is the no-cache arm's prefill median — the same arm
the speedup divides by. **Warm prefill** is the identical-full-context resend. **Speedup** is
cold ÷ warm. **Prompt reused** is the fraction of the prompt that did not go through a forward pass.

Read down a column within one table. Do **not** read across tables for seconds — the three lanes run
different quantizations (see [Comparing the three lanes](#comparing-the-three-lanes)).

#### MLX, 4-bit

`mlx-community` 4-bit conversions, MLX on Metal, output pinned to one token with the realized length
asserted identical across arms. 7 repetitions per arm except the `Qwen3.5-4B` 30k row, which ran 3.

| Model | Context | Cold prefill | Warm prefill | Speedup | Prompt reused |
|---|---|---|---|---|---|
| `Qwen3-4B-Instruct-2507-4bit` (dense) | 10k | 5.390 s | 0.141 s | **×38** | 99.82% (fed 18 of 10,053) |
| `Qwen3-4B-Instruct-2507-4bit` (dense) | 30k | 32.930 s | 0.502 s | **×66** | 99.94% (fed 18 of 29,967) |
| `Qwen3-4B-Instruct-2507-4bit` (dense) | 100k | 263.556 s | 0.618 s | **×426** | 99.96% (fed 45 of 100,090) |
| `Qwen3.5-4B-MLX-4bit` (hybrid) | 10k | 4.615 s | 0.092 s | **×50** | 99.52% (fed 48 of 10,090) |
| `Qwen3.5-4B-MLX-4bit` (hybrid) | 30k | 12.486 s | 0.110 s | **×113** | 99.84% (fed 48 of 30,090) |
| `Ornith-1.0-9B-4bit` (hybrid) | 10k | 13.150 s | 0.056 s | **×233** | 99.52% (fed 48 of 10,090) |
| `Ornith-1.0-9B-4bit` (hybrid) | 30k | 30.188 s | 0.205 s | **×147** | 99.84% (fed 48 of 30,090) |
| `Qwen3.6-27B-4bit` (hybrid) | 10k | 33.214 s | 0.724 s | **×46** | 99.52% (fed 48 of 10,090) |
| `Qwen3.6-27B-4bit` (hybrid) | 30k | 87.253 s | 0.539 s | **×162** | 99.84% (fed 48 of 30,090) |
| `Qwen3.6-35B-A3B-4bit` (hybrid) | 10k | 5.925 s | 0.351 s | **×17** | 99.52% (fed 48 of 10,090) |
| `Qwen3.6-35B-A3B-4bit` (hybrid) | 30k | 19.018 s | 0.188 s | **×101** | 99.84% (fed 48 of 30,090) |

The 4-bit MLX cells report prefill against prefill: both columns are wall clock minus that cell's own
measured decode floor. Wall-against-wall on the same cells gives a different, also correct, number —
`Ornith-1.0-9B` at 30k reads ×147 prefill-against-prefill and ×78 wall-against-wall. Keep one
convention per comparison and say which one you used.

**Building the cache is not a tax on this lane.** The cold-with-cache arm costs ×0.80–×1.23 of the
no-cache arm across these cells, so the first call of a session pays roughly what it would have
paid anyway.

#### HuggingFace transformers, bf16

torch 2.13.0, transformers 5.9.0, stock `sdpa` attention with no override, MPS device, prefill chunk
step 2048 except where noted. bf16 rows run 7 repetitions per arm except `Qwen3.6-27B` at 30k, which
runs 3; the NF4 rows run 4–5 cold and 7 warm. Dense models take the crop lane; hybrids take the
snapshot lane. **All eight bf16 correctness gates pass, and both NF4 gates pass** — see
[Correctness on this lane](#correctness-on-the-transformers-lane).

| Model | Precision | Context | Cold prefill | Warm prefill | Speedup | Prompt reused |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` (dense, crop) | bf16 | 10k | 9.023 s | 0.063 s | **×143** | 99.99% (fed 1 of 10,011) |
| `Qwen/Qwen3-4B-Instruct-2507` (dense, crop) | bf16 | 30k | 66.927 s | 0.522 s | **×107–×188** | 100.0% (fed 1 of 29,980) |
| `Qwen/Qwen3.5-4B` (hybrid, snapshot) | bf16 | 10k | 9.790 s | 0.052 s | **×188** | 99.99% (fed 1 of 10,028) |
| `Qwen/Qwen3.5-4B` (hybrid, snapshot) | bf16 | 30k | 42.742 s | 0.175 s | **×245** | 100.0% (fed 1 of 30,024) |
| `deepreinforce-ai/Ornith-1.0-9B` (hybrid, snapshot) | bf16 | 10k | 13.114 s | 0.069 s | **×191** | 99.99% (fed 1 of 10,028) |
| `deepreinforce-ai/Ornith-1.0-9B` (hybrid, snapshot) | bf16 | 30k | 56.704 s | 0.135 s | **×419** | 100.0% (fed 1 of 30,024) |
| `deepreinforce-ai/Ornith-1.0-9B` (hybrid, snapshot) | bnb NF4 | 10k | 15.527 s | 0.068 s | **×228** | 99.99% (fed 1 of 10,028) |
| `deepreinforce-ai/Ornith-1.0-9B` (hybrid, snapshot) | bnb NF4 | 30k | 50.750 s | 0.156 s | **×325** | 100.0% (fed 1 of 30,024) |
| `Qwen/Qwen3.6-27B` (hybrid, snapshot) | bf16 | 10k | 47.370 s | 0.166 s | **×285** | 99.99% (fed 1 of 10,028) |
| `Qwen/Qwen3.6-27B` (hybrid, snapshot) | bf16 | 30k | 209.296 s | 0.224 s | **×101–×1,200** | 100.0% (fed 1 of 30,024) |

**Two rows are ranges, not point estimates.** `Qwen3-4B-Instruct-2507` at 30k has a cold arm that
spans ×1.53 between its fastest and slowest repetition, and `Qwen3.6-27B` at 30k spans ×2.13 across
three repetitions (116.5–248.4 s cold, 0.207–1.151 s warm). Both miss the tightness bound stated in
[How these numbers are taken](#how-these-numbers-are-taken), so each is published as the interval its
own repetitions support, with the median-over-median value being ×128 and ×933 respectively. The
27B cell also ran with a **prefill chunk step of 512** rather than 2048 — see
[Prefill chunking](#prefill-chunking-transformers-lane) — and its correctness gate is a separate
cell rather than an arm of the timing run.

**The NF4 rows are measured with the bitsandbytes fused Metal kernel live.** That is what you get
today through `create_llm(...)`: warm identical resend at 10k measures **0.049 s** on the product
path against **0.068 s** in a harness that resolved the kernel before AbstractCore loaded. Reaching
the kernel through the product path depends on `offline_first`, which is covered in
[4-bit on MPS](#4-bit-on-mps-bitsandbytes-fused-kernel) below; a build without the fused kernel
decodes about ×4 slower and warms at 0.27 s instead. Read the NF4 rows as the fused-kernel figures
they are, and check the one-time `#FALLBACK` warning if your own numbers look like the slow set.

#### Correctness on the transformers lane

Every cell in the transformers table has a matching correctness gate, and **all ten pass** — the
eight bf16 cells below plus the two NF4 cells, which carry their gate inside the timing run. Each
gate plants a distinctive fact at roughly 5%, 50% and 95% of the context depth and asks for all
three back, cold and warm, with two controls per cell: a determinism control (the same bytes under a
fresh key, twice, must match) and a context-dependence control (the same question with the context
removed must fail, while a general-knowledge positive control still answers).

| Model | Context | Cold recall (3 depths) | Warm recall (3 depths) | Distinct answers per arm | Verdict |
|---|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` (dense) | 10k | 24/24 | 24/24 | 1, 1, 1 | **PASS** |
| `Qwen/Qwen3-4B-Instruct-2507` (dense) | 30k | 24/24 | 24/24 | 1, 1, 1 | **PASS** |
| `Qwen/Qwen3.5-4B` (hybrid) | 10k | 15/15 | 15/15 | 1, 1, 1 | **PASS** |
| `Qwen/Qwen3.5-4B` (hybrid) | 30k | 15/15 | 15/15 | 1, 1, 1 | **PASS** |
| `deepreinforce-ai/Ornith-1.0-9B` (hybrid) | 10k | 15/15 | 15/15 | 1, 1, 1 | **PASS** |
| `deepreinforce-ai/Ornith-1.0-9B` (hybrid) | 30k | 15/15 | 15/15 | 1, 1, 1 | **PASS** |
| `Qwen/Qwen3.6-27B` (hybrid) | 10k | 15/15 | 15/15 | 1, 1, 1 | **PASS** |
| `Qwen/Qwen3.6-27B` (hybrid) | 30k | 15/15 | 15/15 | 1, 1, 1 | **PASS** |

Sampling is 8 repetitions per depth per arm on the dense model and 5 on the hybrids. The primary
statistic is the number of **distinct** answers per arm rather than the pass count: one distinct
answer over n repetitions is stronger evidence than n independent passes, because it rules out a
per-call intermittent fault that a single sample per depth cannot distinguish from correctness.
Every arm at every depth returned exactly one distinct answer.

The two NF4 cells (`Ornith-1.0-9B` at 10k and 30k) carry their gate inside the timing run and both
pass. Each gate above is its own cell rather than an arm of the corresponding timing run, so a gate
result and a timing row are separate evidence about the same model and tier.

#### GGUF / llama.cpp, Q4_K_M

llama.cpp 0.3.23 on Metal, `n_gpu_layers == -1` asserted per cell, output pinned, 7 repetitions per
arm except the two 30k cells marked as ranges, which ran 3. `n_ctx` is held equal across models
within a tier. The cold arm calls `reset()` per repetition; the warm arm is an identical resend on
one key, and it evaluates the prompt tokens the one-token guard requires rather than zero.

| Model | Context | Cold prefill | Warm prefill | Speedup | Prompt reused |
|---|---|---|---|---|---|
| `unsloth/Qwen3-4B-Instruct-2507-GGUF` | 10k | 9.637 s | 0.066 s | **×145** | 99.98% (evaluated 2 of 10,025) |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF` | 30k | 76.506 s | 0.620 s | **×123** | 99.99% (evaluated 2 of 29,994) |
| `Ornith-1.0-9B-GGUF` | 10k | 8.937 s | 0.089 s | **×101** | 99.98% (evaluated 2 of 10,035) |
| `Ornith-1.0-9B-GGUF` | 30k | 38.155 s | 0.207 s | **×184** | 99.99% (evaluated 2 of 30,031) |
| `Qwen3.6-27B-GGUF` | 10k | 36.599 s | 0.296 s | **×124** | 99.98% (evaluated 2 of 10,044) |
| `Qwen3.6-27B-GGUF` | 30k | 109.079 s | 0.303 s | **×82–×524** | 99.99% (evaluated 2 of 30,040) |
| `Qwen3.6-35B-A3B-GGUF` | 30k | 44.301 s | 0.157 s | **×81–×326** | 99.99% (evaluated 2 of 30,040) |
| `Ornith-1.0-35B-GGUF` | 10k | 9.130 s | 0.082 s | **×111** | 99.98% (evaluated 2 of 10,035) |
| `Ornith-1.0-35B-GGUF` | 30k | 19.359 s | 0.169 s | **×34–×132** | 99.99% (evaluated 2 of 30,031) |

Three cells are published as ranges because three repetitions left the gain spanning more than a
factor of two; their median-over-median values are ×360, ×283 and ×114. Extra repetitions were not
run — a wide spread is a result about the measurement, not a reason to keep sampling until it
narrows.

**`Qwen3.6-35B-A3B` at 10k is UNTESTED here.** That cell completed but failed the pre-registered
check on how many prompt tokens its warm arm evaluated, so its numbers are not published.

**Warm on this lane is not context-independent.** The warm call still performs one forward pass over
the full KV, so a larger context does not have to produce a larger ratio: the 4B goes ×145 at 10k and
×123 at 30k, and it carries the heaviest KV geometry in this set at 148 KiB per token against the
35B-A3B's 27 KiB.

### Comparing the three lanes

`Ornith-1.0-9B` is the one model measured on all three lanes, so it is the **same-model** row. It is
not a like-for-like row: each lane runs its own quantization of those weights, and that is the first
thing to read out of the table.

| Lane | Quantization | 10k | 30k |
|---|---|---|---|
| MLX | 4-bit | **×233** (13.150 s → 0.056 s) | **×147** (30.188 s → 0.205 s) |
| HuggingFace transformers | bf16 | **×191** (13.114 s → 0.069 s) | **×419** (56.704 s → 0.135 s) |
| HuggingFace transformers | bnb NF4 | **×228** (15.527 s → 0.068 s) | **×325** (50.750 s → 0.156 s) |
| GGUF / llama.cpp | Q4_K_M | **×101** (8.937 s → 0.089 s) | **×184** (38.155 s → 0.207 s) |

**Only the ratio is comparable across these rows.** Four different quantizations of the same weights
appear here — 4-bit for MLX, bf16 and bitsandbytes NF4 for transformers, Q4_K_M for GGUF — so the
seconds measure the quantization at least as much as the cache. Reading a bf16 cold prefill against a
4-bit one as an engine comparison is a quantization comparison wearing a disguise. The
transformers rows are the closest thing to a quantization control in this set, since they differ
only in dtype on one lane; there is no unquantized MLX or GGUF build of this model on the measurement
host, so the other halves of that control are untested.

What the ratios say:

- **All three lanes are in the same broad performance class** once each is measured with the
  one-token guard active: a warm full-context resend costs tens to a few hundred milliseconds
  everywhere, and the ratio grows into the hundreds at 30k.
- **GGUF warm calls are the most sensitive to KV geometry.** The warm call still runs one forward
  pass over the full KV, so its cost tracks bytes-per-token rather than being flat; a model with
  heavy KV geometry can show a *lower* ratio at 30k than at 10k.
- **The effect scales with context on every lane**, because it is a function of how much prefill is
  avoided. A ratio measured at one context size does not transfer to another — quote the context
  size with the ratio.

### What the three lanes share

All three lanes implement the same design, which is why their shapes agree:

- **A boundary snapshot for growing conversations.** Each key keeps one cache boundary held back to
  the last position two consecutive prompts agree on, so a prompt ending in per-call ephemeral bytes
  (a generation prompt, a loop counter, a fresh-timestamp envelope) still leaves something the next
  turn can extend. Warm turns restore that boundary and feed only the suffix.
- **A one-token guard**, so a repeated prompt still runs a forward pass — see
  [The one-token guard](#the-one-token-guard).
- **Telemetry** on the lanes that can attest it: `response.metadata["prompt_cache"]` carries
  `outcome`, `cached_tokens` and `fed_tokens` on MLX, on the GGUF control-plane path, and on the
  transformers snapshot lane. The transformers crop lane (dense architectures) emits none, so read
  its behavior from wall clock and `#FALLBACK` warnings.

### The one-token guard

A warm call must always push at least the final prompt token through a forward pass, even when the
prompt is byte-identical to the previous one. Otherwise the sampled position would be produced from
logits the *previous* call left behind, and the answer would be a continuation of the last turn
rather than a response to this one.

All three lanes hold that invariant: MLX and transformers keep one token back from a fully-contained
prompt before feeding, and the GGUF lane rolls a restored full-prompt state back by one token (a
checked rollback — architectures that refuse it get an honest full re-prefill instead). On an
identical resend, cold and warm produce the same first token id, verified on both a dense and a
hybrid GGUF model with a determinism control passing in each case.

This is a correctness invariant rather than a policy, and it is not disabled by any environment
switch.

### Agent loop, cache ON vs OFF (ReAct task, per-cycle view)

A ReAct tool task where every call re-sends the growing transcript, run once with a byte-stable
prefix (cache ON) and once with the prefix deliberately broken every cycle (cache OFF: a fresh nonce
line prepended to the system prompt).

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

#### Agent loop, per-turn view at 10k tokens

One key, a monotonically growing transcript, no identical-bytes repeat — the shape a ReAct loop
actually produces. `outcome`, `cached` and `fed` are read from `metadata.prompt_cache`.

| Turn | Hybrid `Qwen3.5-4B-MLX-4bit` (see the scope note below) | Dense `Qwen3-4B-Instruct-2507-4bit` |
|---|---|---|
| 1 | 7.44 s — `rebuilt`, fed 10,072 | 6.29 s — `cold`, fed 10,077 |
| 2 | 5.46 s — `rebuilt`, fed 10,118 | 0.51 s — `hit_extend`, cached 10,077 / fed 46 |
| 3 | 0.67 s — `hit_restore`, cached 10,068 / fed 92 | 0.41 s — `hit_extend`, cached 10,123 / fed 42 |
| 4 | 0.76 s — `hit_restore`, cached 10,114 / fed 73 | 0.43 s — `hit_extend`, cached 10,165 / fed 27 |

The dense lane reuses from turn 2 onward and never rebuilds: 92% and 99% of the prompt is served
from the cache on turns 2–4, and the turn cost collapses from 6.29 s to under half a second.

**Scope of the hybrid column.** The turn-2 `rebuilt` in this run is a consequence of where the
snapshot boundary fell — a boundary that included the generation scaffolding the call appended, which
the next turn replaces, so the forward-only restore had nothing to extend. The boundary selection
described in [Which turn reuse starts on](#which-turn-reuse-starts-on) excludes that scaffolding, and
a CPU reproduction with the real tokenizer and renderer turns this sequence into
`rebuilt, hit_restore, hit_restore, hit_restore`. Hardware confirmation is
[pending measurement 1](#pending-measurements). Read the hybrid column as the shape of the failure
mode to watch for in your own ledger, not as a cost to plan around.

† **Hybrid-architecture note (Qwen3.5-class on MLX).**
Qwen3.5 mixes attention layers (trimmable `KVCache`) with linear-attention layers whose recurrent
state (`ArraysCache`) is not trimmable. `Qwen3.6` loads the same `qwen3_5`/`qwen3_5_moe` classes and
Ornith 1.0 (9B/35B/397B) is a Qwen3.5 post-train, so the same lane selection applies to them by
construction — but lane selection is not a performance result, and each model's measured status is
listed separately in the
[MLX compatibility table](#mlx-mlxprovider). A recurrent state cannot be rewound,
so the trim-based delta path does not apply. MLX instead uses a **snapshot/restore lane** for these
models: it keeps one copied cache boundary per `prompt_cache_key` and, when the next full-context
prompt extends it, restores the copy and feeds only the new suffix (forward-only reuse; a one-time
`#FALLBACK` line records that the snapshot lane, not the trim lane, is active). A growing-prefix
agent loop therefore reuses its prefix on warm turns just as a pure-attention model does. Measured
on `Qwen3.5-4B-MLX-4bit` (6.6k-token transcript, one key): turn 1 does a full prefill; turns 2–3
report `outcome=hit_restore` and feed ~24 of 6.6k tokens, with fact recall correct throughout.
Divergent prompts (a changed prefix) rebuild and re-snapshot.

‡ **Host composition note.** The in-process delta engages when the host re-sends the full context
per call (`messages=` present) over a stable key; `generate()` then does the token-level prefix
match itself. A host that also drives the control plane should use it to build the stable
`(system, tools)` prefix and fork the session key from it **once**, and leave the transcript to
`generate()`. Pushing the transcript through `prompt_cache_update` on every turn duplicates work the
delta feed is about to redo, and `prompt_cache_clear` is not free on this lane — it also drops the
key's boundary snapshot, which is the state an untrimmable architecture restores from.

#### Agent loop on the GGUF lane

The same growing-transcript shape, measured on GGUF with the boundary snapshot active. Both arms run
in one process on one resident model with `n_gpu_layers == -1` asserted, and differ only in cache
policy. Evaluated prompt tokens come from an `llm.eval` spy independent of the provider's own
telemetry. Median of turns 2–5.

| Model | Context | No cache | With the boundary snapshot | Gain | Prompt tokens evaluated per warm turn |
|---|---|---|---|---|---|
| `Ornith-1.0-9B-GGUF` | 10k | 16.33 s | **2.25 s** | ×7.3 | 10,032 → **16** |
| `Qwen3-4B-Instruct-2507-GGUF` (dense) | 10k | 0.61 s | **0.31 s** | ×2.0 | 24 → **13** |
| `Qwen3.6-27B-GGUF` | 30k | 203.34 s | **2.19 s** | ×93.0 | 30,037 → **16** |
| `Qwen3.6-35B-A3B-GGUF` | 10k | 17.01 s | **0.62 s** | ×27.2 | 10,041 → **16** |

On the hybrids the evaluated-token count collapses to a fixed 16 per turn regardless of transcript
size, which is what a working boundary snapshot looks like from the outside. Turn 2 still does one
rebuild — turn 1 has no previous prompt to take a longest-common-prefix against — and every turn from
3 onward restores. The same property holds on the MLX and transformers lanes.

**The "no cache" arm here is not "no reuse at all".** llama.cpp keeps its own instance-internal
`n_past` prefix reuse, which is why the dense model's uncached arm already drops to 24 tokens per
turn. On that architecture there is little left for a cache to win, and the boundary policy
correctly changes nothing.

#### Which turn reuse starts on

The snapshot lane is selected on the ARCHITECTURE, not on how full the cache happens to be, so an
untrimmable model takes it from the first call — including a session
whose key was forked from a `(system, tools)` prefix and is therefore already warm on turn 1. Where
the lane places its boundary is what decides whether turn 2 restores:

- If the key already holds a fed-token record that is a true prefix of this prompt, that record is
  the boundary and the turn restores from the live cache instead of prefilling cold.
- The boundary is held back to the last position two consecutive prompts agree on, so a prompt
  ending in per-call ephemeral bytes still leaves something the next turn can extend.
- On the first turn of a fresh key there is no previous prompt to compare against, so the boundary
  is held back by the generation scaffolding the call appended — the bytes the next turn replaces
  with the assistant turn that actually happened.

Read `outcome` per turn from `metadata.prompt_cache` on your own workload rather than assuming a
turn number: `hit_restore` means the boundary was reused, `rebuilt` means it was not. Recorded
per-turn results for specific models are in
[Which models have been measured](#mlx-mlxprovider); the hybrid turn-2 outcome under the current
boundary selection is a [pending measurement](#pending-measurements).

### End-to-end sessions through a client

The probes above isolate prefill. A session measured end to end through a gateway client answers a
different question — it includes decode and client overhead — and the two must not be compared.
**No end-to-end cache-on versus cache-off ratio is published here.** The prefill matrix above is the
measured result; treat any end-to-end number you take yourself as belonging to your own workload.

What has been measured end to end is **reuse through the product path**: a four-turn `abstractcode-tui`
session on `Qwen3-4B-Instruct-2507-4bit` at a ~13k-token working context reported `outcome=hit_extend`
on all 96 model calls, with 86–90% of prompt tokens served from cache per turn, read from
`result.metadata.prompt_cache` on the `llm_call` records in the run ledger. That confirms the cache
reaches the provider through client → gateway → runtime; it is not a speed claim, because that run
had no valid uncached control.

**Reading reuse.** Compute it as `Σ cached_tokens / Σ (cached_tokens + fed_tokens)` from
`result.metadata.prompt_cache`, not from `usage.input_tokens` — the latter is a fast heuristic
estimate of prompt length, not a measurement. Report turns 2 onward separately from turn 1: a lane
that builds a `(system, tools)` prefix inside turn 1 and forks it produces turn-1 `cached_tokens`
itself, which is not a saving.

**Measure GGUF in its own process.** GGUF Metal offload is disabled when PyTorch or transformers is
already imported, and a gateway that serves MLX or transformers models has imported one. A GGUF
timing taken there describes CPU inference. The Metal-offloaded figures for that lane are in
[the measured matrix](#gguf--llamacpp-q4_k_m), taken with `n_gpu_layers == -1` asserted per cell —
see [Measure this lane in its own process](#gguf--llamacpp-huggingfaceprovider).

**Bound the output budget.** An attempt on `Qwen/Qwen3.5-4B` (bf16) issued with
`max_new_tokens = 81920` did not return. Pin the output length in both lanes and assert the realized
length, or you are measuring decode.

**Running an A/B of your own.** One gateway, one resident model, the same turns per lane in a fresh
session per lane, and both lane orders:

```bash
# uncached lane
abstractcode-tui exec "<turn prompt>" --session pcb-off --workflow react-agent:react \
  --provider mlx --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --no-project-context --no-review --permissions read --max-iterations 4 --no-prompt-cache

# cached lane: identical, minus --no-prompt-cache
```

Then read `result.metadata.prompt_cache` for every `llm_call` effect in each run's ledger.
`--no-prompt-cache` is an `abstractcode-tui exec` option; it sets `_runtime.prompt_cache = false`,
which suppresses cache-key derivation so the provider receives no `prompt_cache_key` and emits no
`prompt_cache` metadata at all. Absence of that metadata on every call is the proof that the
uncached lane was genuinely uncached — check it, do not assume it.

**The posture is inherited by child runs.** `_runtime.prompt_cache` crosses the sub-run hop, so a
workflow whose agent loop executes in a child or grandchild run carries the posture it was sent
rather than falling back to the server default. `false` is treated as a meaningful value, not as
absence, and an explicit posture set on the child wins. That covers both the direct workflows
(`react-agent:react`) and the flow-graph bundles (`coding-agent`, `basic-agent`,
`multiagent-coding`). Verify the lane from the ledger before trusting any A/B built on it.


### What a cache hit guarantees, and what it does not

**What is guaranteed, stated precisely:**

1. **Identical token ids.** A warm call is generated from exactly the token ids you sent. The delta
   path trims or restores the cache back to the true shared token prefix and feeds the rest, so
   nothing is silently dropped from the middle and the cached region holds exactly the tokens the
   record says it holds.
2. **Identical logical context.** The model sees the same conversation, in the same order, at the
   same positions, as it would on a cold call of the same prompt.
3. **The same top-1 token at the first generated position.** This held in every lane measured. A
   cache split at an arbitrary boundary reproduces the warm result exactly, and a split aligned to
   the cold run's own chunk boundaries is bit-exact with cold.

**What is not guaranteed: byte-identical generated text.** Warm and cold produce the same ids
through a different sequence of matrix shapes, because prefill chunking groups the same tokens into
different batches, and floating-point accumulates differently across different batch shapes. That is
enough to move the sampled tokens even at `temperature=0`. The effect is not caused by the cache:
**two cold runs with identical token ids already differ in text when only their chunk plan differs.**

Each path is individually deterministic — repeat it and you get the same bytes — but the two paths
can be deterministic to *different* answers, and the divergence can reach content rather than only
formatting. Treat a warm rerun as a different sample, not as a reproduction of the cold one.

**The cause is prompt batching, not the cache.** Every lane prefills in fixed chunks — `mlx_lm` at
`prefill_step_size` 2048, the transformers lane at
[`ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP`](#prefill-chunking-transformers-lane), default 2048. A warm
call feeds a short suffix and a cold call feeds the whole prompt, so the same ids are grouped into
different batches. Byte equality is therefore a diagnostic, not a correctness criterion, and it gets
less reliable as the context grows. Gate a cache on planted-fact recall and on top-1 agreement at the
first generated position; record byte equality as an observation.

What this means in practice:

- A cached turn and an uncached turn are not interchangeable outputs. Do not mix them inside one
  evaluation, and do not treat a cached rerun as a reproduction of an uncached result.
- Answers stay correct. Every correctness gate in the matrix recalls its planted facts in both arms,
  cold and warm, with one distinct answer per arm. What moves is the free-form text around the
  answer.
- The divergence can change output *length*, and a longer answer costs more decoding time than the
  cache saves on prefill. Gate on output length as well as on content — see
  [Verifying that a cache is helping](#verifying-that-a-cache-is-helping).
- Through a client the effect is larger, because a runtime-grounding envelope can re-render per call
  and shift the prompt by a token or two. Pin what you can, and read `outcome` per call rather than
  assuming byte stability.


Guidance:

- **What makes caches hit**: a byte-stable prefix — stable system prompt, stable tool schema
  (tool ORDER included), append-only transcript. Any change to the first bytes (even one line)
  forces a full re-prefill everywhere. Agent loops that only append satisfy this by construction.
- **Shared server for many short-lived clients**: an LM Studio-style server cache persists across
  client restarts and needs no request field — CLI tools and restarting processes get warm
  prefills for free. In-process caches die with the process.
- **Long-lived single process with a large context**: in-process caches (MLX, GGUF, transformers)
  give the largest *prefill* ratios because nothing leaves the process — provided the model
  architecture supports the delta path (see the compatibility table below). Take the ratio at your
  own context size from [the measured matrix](#the-measured-matrix); it ranges from ×17 to ×426 on
  MLX, ×107 to ×1,200 on transformers, and ×34 to ×524 on GGUF, and it grows with context on every
  lane. **Those are prefill ratios, not end-to-end ones.** An end-to-end session also pays decode and
  client overhead, which the cache does not touch, so it will show a smaller ratio than any figure in
  that matrix.
- **Choosing between the local lanes for speed**: all three reduce a warm full-context resend to
  tens or low hundreds of milliseconds at these context sizes. GGUF's warm call runs one forward pass
  over the full KV, so its warm cost tracks the model's bytes-per-token more strongly than the other
  two lanes do. Choose a lane for its other properties — quantization availability, model coverage,
  memory footprint — and take the ratio at your own context size.
- **Prefill-vs-decode reality check**: caching removes prefill only, so a turn can never drop below
  `output_tokens × decode_time_per_token`. At small prompts (2–4k tokens) with slow decode,
  end-to-end gains look modest even when the cache works perfectly; the ratio becomes dramatic at
  long prefixes (10k+ tokens). Decode also gets slower as the KV grows, which offsets part of the
  prefill saving on long sessions.
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

Two tables. The first says which code path a family takes — that is decided by the cache classes the
architecture builds, so it is a statement about the code. The second says which specific models have
a recorded measurement — that is a statement about evidence, and **no model's result is inferred
from another model, however closely related.** Sharing an architecture predicts the lane; it does not
predict reuse percentages, wall clock, or gate outcomes.

**Which lane a family takes**

The trim-based delta path needs a trimmable cache (`mlx_lm can_trim_prompt_cache`); untrimmable
architectures take the snapshot/restore path instead.

| Model family (mlx_lm architecture) | Cache | Warm-call behavior |
|---|---|---|
| Llama 3.x (non-sliding), Qwen3 dense (`Qwen3-*-Instruct-2507`), Qwen3-MoE, Qwen2.x, Mistral-class — any model without a custom `make_cache` | pure `KVCache` | **Full delta**: trim to the shared token prefix, feed only the suffix |
| Falcon-H1, LongCat-Flash, DeepSeek-V3.2 | `CacheList` of `KVCache` | Full delta (countable and trimmable) |
| Gemma-3/3n/4, GPT-OSS, Cohere2, Ministral3, sliding-window Llama variants | mix `KVCache` + `RotatingKVCache` | Full delta only while the transcript is under the smallest sliding window (GPT-OSS 128 tokens; Gemma 512–1024), then rebuild per call with `#FALLBACK`. **Documented non-target for agent workloads**: a `RotatingKVCache` physically discards positions past the window, so the rewind the delta lane needs is impossible once a real transcript exceeds it. Serve these through LM Studio or vLLM, where the engine's forward-only slot cache works regardless of the window. |
| Llama-4 (Scout/Maverick) | `ChunkedKVCache` (8k chunks) | Delta within the live chunk; partial trims refuse → rebuild |
| Qwen3.5, Qwen3.6 (`qwen3_5` / `qwen3_5_moe`, 3:1 Gated DeltaNet), Ornith 1.0 (Qwen3.5 post-trains), Qwen3-Next, Nemotron-H, Jamba, LFM2/2.5, Granite-hybrid, Plamo2, Baichuan-M1 | hybrid with untrimmable `ArraysCache` layers | **Snapshot/restore delta** (`outcome=hit_restore`): the recurrent state cannot be trimmed but it can be copied, so a per-key boundary snapshot is restored and only the suffix is fed on warm growing-prefix turns. A divergent prefix rebuilds and re-snapshots. |
| Mamba/Mamba2, RWKV-7 (pure SSM) | all `ArraysCache` | Snapshot/restore delta — the same path as the hybrids |

**Which models have been measured**

| Model | Architecture | What was measured | Result |
|---|---|---|---|
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | dense, full attention | Prefill probe at 10k, 30k and 100k, 7 reps, pinned output length | ×38, ×66 and ×426 vs no cache; 99.82% / 99.94% / 99.96% of the prompt reused; gates pass |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | dense, full attention | Four-turn agent loop at 10k, one key | `cold`, then `hit_extend` on every later turn; no rebuild |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | dense, full attention | Four-turn client session through the gateway at ~13k tokens | `hit_extend` on all 96 model calls; 86–90% of prompt tokens reused per turn (reuse only — no valid uncached control in that run) |
| `Qwen3.5-4B-MLX-4bit` | hybrid | Prefill probe at 10k and 30k, 7 and 3 reps | ×50 and ×113 vs no cache; 99.52% / 99.84% reused; the 10k gate passes, the 30k gate is UNTESTED |
| `Qwen3.5-4B-MLX-4bit` | hybrid | Four-turn agent loop at 10k, one key | Turn 2 `rebuilt`, turns 3–4 `hit_restore` — boundary-scoped, see [pending measurement 1](#pending-measurements) |
| `Ornith-1.0-9B-4bit` | hybrid | Prefill probe at 10k and 30k, 7 reps | ×233 and ×147 vs no cache; 99.52% / 99.84% reused; the 10k gate passes, the 30k gate is UNTESTED |
| `Qwen3.6-27B-4bit` | hybrid | Prefill probe at 10k and 30k, 7 reps | ×46 and ×162 vs no cache; 99.52% / 99.84% reused; **both gates pass** |
| `Qwen3.6-35B-A3B-4bit` | hybrid | Prefill probe at 10k and 30k, 7 reps | ×17 and ×101 vs no cache; 99.52% / 99.84% reused; gates pass |
| `mlx-community/gemma-4-31b-mxfp4` | sliding window | Warm call under the window | `hit_extend` at a ~450-token transcript, recall correct — under-window only; documented non-target above the window |

**UNTESTED on this lane.** No run has been recorded for any of the following, and none of their
behavior should be read off the rows above:

- `Qwen3.5-122B-A10B-4bit` — installed locally, no recorded run.
- Every hybrid family other than Qwen3.5/Qwen3.6/Ornith 1.0 — Qwen3-Next, Nemotron-H, Jamba,
  LFM2/2.5, Granite-hybrid, Plamo2, Baichuan-M1.
- Pure-SSM models (Mamba/Mamba2, RWKV-7).
- `CacheList` families (Falcon-H1, LongCat-Flash, DeepSeek-V3.2) and Llama-4 chunked caches.
- Llama 3.x, Qwen2.x and Mistral-class dense models.
- The 100k-token context tier on every model except the dense `Qwen3-4B-Instruct-2507-4bit`.
- The correctness gate on `Qwen3.6-27B-4bit` at 30k.
- Any quantization other than the one named in the measured row.

Check your model: the first warm full-context call logs `#FALLBACK … not trimmable` /
`… not countable` if the model cannot delta; no warning means the delta path engaged.
`get_prompt_cache_stats()` exposes per-key `token_count` so you can watch reuse across calls, and
`scripts/verify_prompt_cache_families.py` runs a growing-prefix session with a cache census and a
fact-recall gate against any model you have installed.

**Which turn reuse starts on, on an untrimmable architecture.** A snapshot is written by the
snapshot lane, and that lane is selected on the architecture. On a session whose key is forked from a
`(system, tools)` prefix cache, turn 1 is already warm. The boundary the lane snapshots is chosen
conservatively: it is held back to what two consecutive prompts share, so a prompt that ends in
per-call ephemeral bytes — a loop counter, a fresh-timestamp envelope, the generation prompt — still
leaves a boundary the next turn can restore from. Read `outcome` per turn from
`metadata.prompt_cache` on your own workload: `hit_restore` means the boundary was reused,
`rebuilt` means it was not.

### GGUF / llama.cpp (`HuggingFaceProvider`)

The full control plane (`mode=local_control_plane`: delta-only updates, prefill-snapshot
generation) requires an exact chat renderer; other formats stay `mode=keyed`, where llama.cpp's
own in-process prefix reuse still applies (real savings, less control).

| GGUF family | Renderer | Mode | Verified |
|---|---|---|---|
| Qwen3 dense (`Qwen3-4B-Instruct-2507`), any "qwen"-named GGUF, ChatML-template models | ChatML (`chatml` / `chatml-function-calling`) | `local_control_plane` | ✅ mode and correctness. Control-plane-maintained session on Metal: warm ~0.6 s vs cold ~9 s at a 10k-token prefix, fact recall correct. A per-call session pattern also stays correct and token-for-token identical, but see the three notes below before timing this lane |
| Qwen3.5 / Qwen3.6 hybrids (`Qwen3.5-*-GGUF`, arch `qwen35`/`qwen35moe`) | ChatML | `local_control_plane` | ✅ 2026-07-14: loads under llama.cpp 0.3.23, warm ~1 s, fact-recall correct (recurrent state round-trips through the snapshot; llama.cpp's in-place trim would refuse it) |
| Gemma-4 (`gemma-4-*-GGUF`, arch `gemma4`) | GGUF-embedded chat template (`llama-cpp-chat-template`) | `local_control_plane` | ✅ 2026-07-14: warm ~0.5 s vs cold ~9 s, fact-recall correct |
| Llama-3.x (when the embedded template matches llama.cpp's llama-3 template) | `llama-3` | `local_control_plane` | prior |
| **Ornith 1.0 GGUF** (`ornith-*.gguf` — a Qwen3.5 post-train shipping a `chat_template.default` ChatML Jinja template) — and ANY GGUF whose embedded Jinja template is ChatML | GGUF-embedded chat template (`llama-cpp-chat-template`) | `local_control_plane` | ✅ 2026-07-19: detection is by template CONTENT (ChatML markers plus a probe render proving the template is ChatML-shaped), never by model name or llama.cpp's guessed format id. Live on `ornith-1.0-35b-Q4_K_M.gguf`: warm/cold 0.34–0.35, fact recall correct, no warnings. A snapshot is reused for what the state HOLDS rather than what its key claims, and a stream render failure degrades instead of raising. |
| Mistral, Phi, DeepSeek, Granite, everything else | — | `keyed` (llama.cpp-native prefix reuse only) | prior |

Check your model: `get_prompt_cache_capabilities()` reports the honest per-model `mode` and the
active renderer in `notes`. All ✅ rows were verified live with
`scripts/verify_prompt_cache_families.py` (growing-prefix ReAct shape, one cache key, fact-recall
correctness gate) on the quant noted in the benchmark logs. The control-plane generate path attaches
`metadata.prompt_cache` with the same vocabulary as the MLX lane (`outcome`, `cached_tokens`,
`fed_tokens`), so reuse can be read from a run ledger on that path; a `mode=keyed` model, which is
served by llama.cpp's native prefix reuse rather than the control plane, emits none.

Three things to know before you set `prompt_cache_key` on this lane.

**llama.cpp keeps its own prefix cache, so there is no uncached baseline inside a live process.**
Suppressing `prompt_cache_key` removes AbstractCore's control plane; it does not stop llama.cpp
reusing KV state from the previous call in the same process. That state also survives across sessions
in the same process, so a second lane started afterwards inherits it. An ON/OFF comparison on this
lane therefore measures *control plane on top of native reuse*, not *cache versus no cache*. To get a
cold baseline, restart the process or use a fresh model instance — which is what the
[measured matrix](#the-measured-matrix) does, resetting per repetition and validating the result
against a fresh-process cold prefill.

**How much the control plane adds over native reuse depends on the architecture.** On a growing
transcript the boundary snapshot collapses warm prefill to a fixed 16 tokens per turn on hybrids
(×7.3 to ×93.0 over the uncached arm), while on a dense model llama.cpp's own `n_past` reuse already
gets the per-turn feed down to 24 tokens and the control plane adds ×2.0 on top. See
[Agent loop on the GGUF lane](#agent-loop-on-the-gguf-lane).

**Budget the warm call by KV size, not by a flat number.** The warm path performs one forward pass
over the full KV, so its cost scales with the model's bytes-per-token and with the context, rather
than staying flat as the in-process copy lanes do. Measured warm medians on this lane run from
0.066 s to 0.62 s across the matrix — see [Comparing the three lanes](#comparing-the-three-lanes).

**Measure this lane in its own process, and assert the offload.** GGUF Metal offload is disabled when
PyTorch or transformers is already imported (`huggingface_provider.py:2991-3007`; see also the GGUF
notes under [Provider status](#provider-status-may-2026)), so a host process that also serves MLX or
transformers models runs llama.cpp with `n_gpu_layers = 0` and every timing taken there describes CPU
inference.

The downgrade is easy to miss: importing `AutoTokenizer` at the top of a harness is enough to trigger
it, and the only signal is a single `RuntimeWarning`. Reproduced deliberately — with `torch` imported
first the provider comes up at `n_gpu_layers = 0` and emits nothing else; with `llama_cpp` imported
first it comes up at `-1`. Read `n_gpu_layers` back from the provider, **assert it is `-1`, and fail
the run loudly if it is not.** A GGUF timing without that assertion is not attributable to a backend.

**Budget for the cache's own memory.** The GGUF prompt cache is a `LlamaRAMCache` created with
capacity `"auto"` — it grows without a byte cap (`huggingface_provider.py:1961-1975`) and holds
full-`n_ctx` llama states, with `n_ctx` defaulting to 16384. A stored state is sized by `n_ctx`, not
by the prompt that produced it, so process residency can exceed the model file by a wide margin and
several keys multiply it. Cap `n_ctx` to the context you actually use rather than leaving the
default, and see [Cache residency and memory](#cache-residency-and-memory) for how to read the
numbers you measure.

### HuggingFace transformers (`HuggingFaceProvider` with `model_type="transformers"`)

Two lanes. Dense, pure-attention models take the **crop lane**: rewind the cache to the shared token
prefix with `Cache.crop`, then feed only the suffix. Architectures whose cache cannot be cropped take
the **snapshot lane**, routed on the architecture from turn 1: keep one boundary copy per key,
restore it on a warm turn, feed the suffix forward-only. The lane a model took is reported in its
measured row.

**Requirements.** This lane needs **torch >= 2.11** on Apple silicon. torch 2.10.x carries a
known MPS `scaled_dot_product_attention` correctness defect
([pytorch#163597](https://github.com/pytorch/pytorch/issues/163597)) that returns non-deterministic,
numerically wrong results for short queries over a long KV cache in float16/bfloat16 — that is, every
decode step past roughly 1,024 tokens of context. It is fixed upstream. AbstractCore emits a one-time
warning when it detects torch 2.10.x on MPS with a half-precision model, and the stopgap it names
(`ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL=eager`) is slower but exact. On torch >= 2.11 no workaround is
applied and stock `sdpa` is the default. torch <= 2.9 has not been measured on this lane.

#### 4-bit on MPS (bitsandbytes fused kernel)

Loading a bitsandbytes NF4 model on MPS needs the fused Metal 4-bit kernel
(`kernels-community/bitsandbytes-mps`). With it, 4-bit warm calls land in the same class as bf16 —
0.049 s for a warm identical resend on `Ornith-1.0-9B` at 10k. Without it every `Linear4bit` forward
runs a `dequantize → F.linear` fallback that is about ×4 slower, and bitsandbytes does not report
that: it swallows the resolution failure and latches it for the life of the process.

AbstractCore resolves the kernel for you and tells you when it cannot:

- **Resolution runs before the weights load**, because loading a bnb-quantized checkpoint quantizes
  as it loads and would otherwise latch the failure before any later hook could help.
- **`offline_first` does not block it.** `offline_first` (default `True`) sets `HF_HUB_OFFLINE=1` to
  keep model *weights* off the network, and the kernel's publisher check has no offline path. For a
  4-bit MPS load only, AbstractCore retries once with its own offline flags lifted and restores them
  immediately, patching `huggingface_hub.constants.HF_HUB_OFFLINE` for the same window because the
  environment variable alone is snapshotted at that library's import.
- **An offline flag you set yourself is never lifted.** The values present before AbstractCore
  touched them are recorded at import; if the flag is yours, the resolution declines and you get the
  warning instead.
- **When the kernel is unavailable, a one-time `#FALLBACK` warning names it**, counts the affected
  `Linear4bit` modules, states the measured cost, and gives the remedy. It never raises.

Cost when it does run: 1.4–1.7 s on the first 4-bit MPS load and about 2 ms afterwards, and it
replaces the resolution bitsandbytes would otherwise attempt on the first forward. On a genuinely
disconnected host the attempt is bounded by `huggingface_hub`'s own connect timeout (about 7 s),
happens at most once per process, and ends in the warning. Non-4-bit and non-MPS loads are untouched.

| Architecture | Warm-call behavior | Status |
|---|---|---|
| Pure-attention decoders (Llama-class, Qwen3 dense) | **Crop lane**: crop applies to every layer, so the rewind is exact | Measured on `Qwen/Qwen3-4B-Instruct-2507` at 10k and 30k; gates pass |
| Hybrids with linear-attention layers (Qwen3.5, Qwen3.6, Ornith 1.0, Qwen3-Next, Jamba, LFM2) | **Snapshot lane**: forward-only restore of a boundary copy, feed the suffix. No rollback is attempted, so the crop limitation below does not apply | Measured on `Qwen/Qwen3.5-4B`, `deepreinforce-ai/Ornith-1.0-9B` and `Qwen/Qwen3.6-27B` at 10k and 30k; gates pass. Qwen3-Next, Jamba and LFM2 are UNTESTED |
| Sliding-window models past their window (Gemma-4 class, window 512) | Crop refuses → rebuild per warm call with `#FALLBACK` | Documented non-target |
| Hybrids whose crop is a no-op on the attention half too (Zamba class) | Caught by the post-crop length verify → rebuild, never wrong context | UNTESTED |

Check your model with `model.config.layer_types`: all `full_attention` means the crop path applies
exactly; any `linear_attention` or `mamba` entry routes the model to the snapshot lane;
`sliding_attention` means window-bounded.

#### Why hybrid models take the snapshot lane instead of a rollback

`Cache.crop` iterates every layer and calls that layer's own `crop`. On linear-attention and mamba
layers that method is an explicit no-op — transformers states so in the implementation itself
("We don't crop the linear attention cache, so simply do nothing here", `transformers.cache_utils`,
5.8.0). A rollback on such a model would rewind only the attention layers and leave the recurrent
state of the linear layers carrying the previous turn's content, and a length check afterwards cannot
detect it: `Cache.get_seq_length()` reads a single layer and skips forward to the first *attention*
layer, that is, to one of the layers that did crop.

`Qwen/Qwen3.5-4B` makes the scale of that concrete — its census is 24 `linear_attention` and 8
`full_attention` layers of 32, so three quarters of the stack cannot be rolled back at all.

AbstractCore therefore does not attempt a rollback on these architectures:

- **Routing is by architecture, from turn 1.** A model with any uncroppable layer takes the snapshot
  lane, which is forward-only — it restores a boundary copy and feeds the suffix. Nothing needs to be
  rewound, so nothing can silently fail to rewind. This is a direct port of the MLX lane's design.
- **The crop path refuses before mutating.** If a crop is reached on a cache with uncroppable layers,
  it is declined up front rather than half-applied, with a one-time `#FALLBACK` warning naming the
  affected layers, and the warm call rebuilds fresh: one cold prefill, correct output, no prefill
  savings. The refusal happens before the mutation precisely because a post-hoc refusal would hand
  back the inconsistent cache it exists to prevent.
- **The post-crop verify is per layer.** Every layer's sequence length is checked, not just the one
  `get_seq_length()` happens to resolve to.

The measured hybrid rows in [the matrix](#the-measured-matrix) are all snapshot-lane rows, and all of
their correctness gates pass.

## Prefill chunking (transformers lane)

The transformers lane splits a long prefill into slices of query positions instead of running it in
one forward pass. **Nothing is truncated and no context is dropped**: the model still receives the
entire prompt, every token still attends to every earlier token, and each chunk's queries attend to
the whole accumulated prefix rather than only their own slice. What changes is the peak memory of one
intermediate tensor.

**Why it exists.** When torch's `scaled_dot_product_attention` falls back to its unfused math path —
which it does on MPS at long sequence lengths — it materialises the full `[heads, L, L]` attention
score matrix. That tensor is quadratic in context length while everything else in attention is
linear. At 30k tokens on a 32-head model in fp32 it is a single **107 GiB** allocation, which Metal
refuses outright, aborting the process. Chunked at 2048 the largest transient is about 7.9 GiB. Other
local lanes are unaffected: MLX and llama.cpp ship their own fused Metal kernels and never build the
matrix, as does CUDA with FlashAttention.

**The knob.**

```bash
# default: 2048 query positions per forward
export ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP=512   # lower peak memory, small throughput cost
export ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP=0     # disable chunking entirely
```

Peak score memory is `heads × step × L × 4` bytes with chunking against `heads × L × L × 4` without
it, so halving the step halves the transient. The cost is more kernel launches — at 30k with step
2048 that is 15 forward passes instead of 1. A step of 512 is what made a 27B bf16 model at 30k fit
alongside its 52 GB of weights on a 128 GB machine.

**What it does not change, and what it does.** Semantics are unchanged: the same tokens attend to the
same tokens with the same weights. Exact output *bytes* are not guaranteed across different step
values, because changing the chunk plan changes the floating-point reduction order — the same
mechanism described in
[What a cache hit guarantees](#what-a-cache-hit-guarantees-and-what-it-does-not). Pin the step if you
need reproducible bytes.

**Checking what you are on.** `model.config._attn_implementation` reports `sdpa`, `eager` or
`flash_attention_2`. `sdpa` on MPS is the combination that may or may not fuse depending on shape,
and it does not report which. Note that `torch.nn.attention.sdpa_kernel(...)` cannot be used to force
a backend on MPS — the selection op is not implemented there, so all backends produce identical
results.

## Pending measurements

These cells are named so that the absence of a number is explicit rather than implied. Each one
states the model, the context tier, and the quantity to be measured, so a result can be dropped in
without renegotiating the method. Until a cell is filled, treat it as UNTESTED and do not infer it
from a related model.

| # | Lane | Model | Context | Quantity to measure | Result |
|---|---|---|---|---|---|
| 1 | MLX | `Qwen3.5-4B-MLX-4bit` (hybrid) | 10k | Per-turn `outcome` over a four-turn agent loop under the current boundary selection — specifically whether turn 2 reports `hit_restore` with `cached ≈ turn 1 length − 7`, or still `rebuilt` | _pending_ |
| 2 | MLX | `Qwen3.5-4B-MLX-4bit`, `Ornith-1.0-9B-4bit` (hybrid) | 30k | Correctness gate (three planted-fact depths, both controls) for the 30k timing rows | _pending_ |
| 3 | MLX | any hybrid | 100k | Prefill probe at the 100k tier — only the dense `Qwen3-4B-Instruct-2507-4bit` has one | _pending_ |
| 4 | transformers | `Qwen/Qwen3.6-27B` (bf16) | 30k | A cold arm tight enough to publish as a point estimate rather than the ×101–×1,200 range in [the matrix](#the-measured-matrix); the constraint is a genuinely idle host, not headroom | _pending_ |
| 5 | GGUF / llama.cpp | `Qwen3.6-35B-A3B-GGUF` | 10k | A cell that passes its own warm-arm checks — the existing one does not, so no number is published for it | _pending_ |
| 6 | transformers | Qwen3-Next, Jamba, LFM2 (bf16) | 10k | Snapshot-lane warm vs cold prefill and the correctness gate on hybrid families other than Qwen3.5/Qwen3.6/Ornith | _pending_ |
| 7 | MLX, GGUF | `Ornith-1.0-9B` unquantized | 10k, 30k | The other halves of the quantization control — only the transformers lane has both a bf16 and a 4-bit row of one model | _pending_ |
| 8 | GGUF / llama.cpp | the three cells published as ranges | 30k | Cold arms tight enough to publish those rows as point estimates rather than ranges | _pending_ |
| 9 | MLX | any hybrid above 4B | 10k | End-to-end bloc gate (`ABSTRACTCORE_RUN_BLOC_GATE=1`): determinism control, context-dependence control, planted-fact recall at three depths, prefix-reuse telemetry | _pending_ |

Cell 7 needs unquantized MLX and GGUF builds of that model, which do not exist on the measurement
host. The GGUF lane has no uncached baseline inside a live process, so its cold arm resets per
repetition and is validated against a fresh-process cold prefill. Every transformers cell needs a
bounded output budget.

## Verifying that a cache is helping

Five checks, in order. Each one rules out a different way a cache can look healthy and not be.

1. **Did the cache engage at all?** A `generate` call that received a `prompt_cache_key` attaches
   `metadata.prompt_cache` to the response (`{mode, key, outcome, cached_tokens, fed_tokens}`, plus
   `degraded_reason` when degraded) on **MLX**, on the **GGUF control-plane path**, and on the
   **transformers snapshot lane** (hybrid architectures). This struct is emitted on the non-streaming
   path, which is the path a durable run ledger records.
   **Two paths emit nothing and must be read another way**: the transformers crop lane (dense
   architectures) and GGUF models that resolve to `mode=keyed`. There the available evidence is wall
   clock and the `#FALLBACK` warnings, so design the comparison accordingly. On the paths that do
   emit it, absence of `prompt_cache` metadata means no key reached the provider.
2. **How much was actually reused?** `Σ cached_tokens / Σ (cached_tokens + fed_tokens)`. Use these
   fields, not `usage.input_tokens`: the latter is a fast heuristic estimate that runs about a third
   high on chat-template prompts. `cached_tokens + fed_tokens` is the measured prompt length.
   Exclude the first call of a process from the ratio if the host prepares a `(system, tools)`
   prefix in that same call — the tokens it reports as cached were prefilled moments earlier.
3. **Did the wall clock move?** Caching removes prefill only, so the floor of any turn is
   `output_tokens × decode_time_per_token`. Measure your decode rate and compute that floor before
   believing a speedup. At a few thousand prompt tokens on a small local model the prefill being
   removed is a few hundred milliseconds; the ratios get large at 10k+ token prefixes.
4. **Did the answer change?** It will. Generation over reused KV is not the same computation as a
   fresh prefill, and the difference reaches the sampled tokens even at temperature 0 — see
   [What a cache hit guarantees, and what it does not](#what-a-cache-hit-guarantees-and-what-it-does-not).
   Run these gates in order; each one is meaningless if the one before it fails.

   1. **Determinism control** — the same bytes under a fresh key, twice, must produce the same
      output. If they do not, nothing below is interpretable.
   2. **Context-dependence control** — ask the question with the context *removed* and require it to
      FAIL, while a general-knowledge control question still answers. This proves the question
      actually reads the context and that removing it did not simply break generation. A question
      answerable without the context measures nothing.
   3. **Semantic gate** — plant distinctive facts at roughly 5%, 50% and 95% of the context depth
      and require all three back, warm and cold. Three depths, because a cache that loses the middle
      still answers the early and late ones. Use text generation whose answer depends on the
      context, with exactly one correct answer — no arithmetic, and nothing with many valid
      answers, which measures a tie-break rather than correctness. Size the context at 1,000 tokens
      minimum; 10k / 30k / 100k are useful tiers.
   4. **Numeric gate** — top-1 agreement at the first generated position, and `max |Δ log p|` under
      a threshold you measured for your model and context size rather than one you assumed.
   5. **Output length** — compare `output_tokens` per turn against the uncached lane. A cached lane
      that answers a bounded question with an order of magnitude more tokens has changed behavior
      even when the content gate passes, and the extra decoding can cost more than the prefill saved.
   6. **Byte equality is diagnostic only.** Record it, do not gate on it: two *cold* runs with
      identical ids diverge in text when only the prefill chunk plan differs. Do not build a gate as
      a disjunction (`answer looks right` OR `bytes match`) — the first branch short-circuits the
      second and the gate never runs.

   Once the two lanes' answers differ, their later turns run over different transcripts and are no
   longer the same measurement. Compare only turns whose prompts are identical in both lanes.
5. **How noisy is your wall clock?** Measure that before quoting any ratio. Run each arm more than
   once and look at the spread, not only the median. On a shared box a single arm can span a factor
   of two between its own repetitions while every start-and-end quiet check passes, because a
   background job can occupy the middle of the arm. Publish a ratio as a point estimate only when
   the cold arms are tight (see [How these numbers are taken](#how-these-numbers-are-taken)), and as
   a range otherwise. Token accounting has no such problem — quote reuse as a measurement and wall
   clock with its spread.

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

### Current limit: an artifact-backed session reuses on turn 1 only

**Know this before you build a multi-turn session on top of a durable bloc artifact.** Forking a
session key from a loaded artifact carries the artifact's identity onto the session key:
`prompt_cache_fork` copies the source key's metadata, which includes the artifact provenance fields,
so the provider treats the session key as artifact-backed too.

Artifact-backed caches are deliberately protected — the delta path will not trim or rebuild one,
because degrading a verified durable artifact to save a prefill would defeat the artifact's purpose.
So the moment a turn's prompt diverges from the recorded prefix, the provider **bypasses the cache
for that call** instead of trimming it. In an agent loop every turn after the first diverges, so
every turn after the first is a bypass. Measured on one key, four turns:

| Session shape | Turn 1 | Turn 2 | Turn 3 | Turn 4 |
|---|---|---|---|---|
| Forked from a durable bloc artifact | `hit_extend` | `bypassed` | `bypassed` | `bypassed` |
| Forked from an ordinary prefix cache | `hit_restore` | `hit_restore` | `hit_restore` | `hit_restore` |

This applies on **every architecture**, trimmable or not. Until it is resolved, plan for
artifact-backed bloc reuse to deliver the first turn only:

- For **single-shot use over a large fixed document** — load the artifact, ask one question — the
  artifact does exactly what it is for.
- For a **multi-turn session**, fork the session key from an ordinary `(system, tools)` prefix cache
  instead, and keep the artifact for prompt-only / KV-source-of-truth flows where the transcript
  lives in the cache rather than being re-sent.
- A one-time `#FALLBACK` warning naming the key is logged on the first bypass, so you can confirm
  which shape you are in from the logs.

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

## Blocs: the composability contract

*(Last revised 2026-08-03. Read this before changing anything in
`prompt_cache_prepare_modules`, `prompt_cache_plan_bloc_chain`, or any
provider's prompt renderer.)*

### What a bloc is

A **bloc** is an independently-keyed slice of **one rendered conversation**. It
is *not* a conversation of its own. That distinction is the whole design.

When a provider supports `prompt_cache_prepare_modules`, you pass an ordered
module list and get one derived cache key per module:

- module `system` → stable persona
- module `tools` → stable tool schema
- (optional) module `discussion_prefix` → immutable summary/memory
- session cache key → forked from the chain's final key, grows per turn

Each key names the cumulative prefix through that module (the key is a chained
hash of all module fingerprints up to and including it), and its cache holds
exactly the tokens of that prefix.

### Why blocs cannot simply be rendered one at a time

Chat templates **fold** content together. ChatML, gemma-turn and every other
format in `mlx_provider._build_prompt_fragment` render the tool instructions
*inside the single system turn* — one `<|im_start|>system` block, never two.
So rendering a `system` module and a `tools` module as two standalone
conversations produces two consecutive system blocks: bytes `generate()` never
emits. The cache then diverges from the real prompt at the end of the system
text and everything after it is unreachable KV.

Two consequences follow, and `tests/test_prompt_cache_bloc_composition.py`
checks both across several real local tokenizers and deterministic toy ones:

- **The standalone shape is anti-composable.** It is acceptable at N=1 and gets
  *worse* with every bloc added, because each extra standalone render inserts
  bytes `generate()` never emits and strands everything after them. The planned
  cut stays a token-level prefix of the single-shot render at every N.
- **The planned cut gives up a small, fixed amount at N=1 and nothing after
  that.** The system turn's closing tag (about two tokens) is deliberately left
  uncached so a later `tools` bloc can extend the same turn. At N≥2 separate
  blocs cache exactly as many tokens as one merged module would.

The `llama3_header` family is **UNTESTED against real weights**: no Llama model
of 4B or above is installed on the measurement host and a smaller sibling is not
an admissible stand-in, so that family is covered only by a deterministic toy
tokenizer reproducing its hazard — a post-processor that injects a
sequence-start token on every encode. That hazard is the reason the checks are
token-level: a standalone render there is byte-identical *text* and still loses
the prefix. **Text-level assertions about bloc rendering prove nothing.**

### The invariant

Blocs `B₁…B_N` with fragments `f₁…f_N`:

- **(C1) Slot order.** Blocs are laid out in the renderer's slot order
  (`system → tools → messages → prompt → generation prompt`). A bloc may only
  occupy slots at or after its predecessor's highest slot. Out-of-order chains
  are refused (they would render into an already-cached region).
- **(C2) Successor independence.** `f_k` is a function of `B₁…B_k` only. It does
  not change when `B_{k+1}` changes or disappears. This is what makes a bloc's
  derived key an honest name for its bytes.
- **(C3) Concatenation equals the single-shot render.** For every `k`,
  `f₁ ++ … ++ f_k` is a token-for-token **prefix** of what the provider's own
  `generate()` builds for the union of `B₁…B_k`. Same token ids, same positions.

`BaseProvider.prompt_cache_plan_bloc_chain` enforces all three. The cut after
bloc `k` is the **maximal successor-independent prefix** of the render of
`B₁…B_k`: the longest string that is a prefix of the render under *every*
admissible continuation. It is derived by rendering the cumulative union against
a fixed family of canonical probes (`+tools`, `+message`, `+prompt`,
`+generation prompt`) and taking their common prefix, then backing off to the
last **token index** all probes agree on. No template knowledge is hard-coded;
new message formats are handled automatically.

Concretely, for ChatML the `system` bloc holds `<|im_start|>system\n{sys}` —
the closing tag deliberately excluded, because the closing tag is exactly the
part a successor can still rewrite — and the `tools` bloc holds
`\n\n{tools}<|im_end|>\n`.

### Seam safety is checked, not assumed

`tokenize(a) + tokenize(b) == tokenize(a + b)` is **not** a property of BPE:
merges straddle raw-text seams (Llama-3 folds `"."` + `"\n\n"` into one token).
The planner runs an explicit plan-time check that the concatenated fragments are
a token prefix of the full single-shot render, and that no bloc contributes zero
tokens. On failure it logs a `#FALLBACK` and **collapses the whole chain into
one fragment** stored under the chain's final key: the blocs stop being
independently reusable, the prefill stays whole and correct. A wrong cache is
never produced silently.

Planning is render + tokenize work only — no forward pass — and it **never runs
when the chain is already warm**, because `prepare_modules` returns on the final
key first.

### Every local lane feeds the planned cut

*(2026-08-07.)* A plan is worth nothing if the lane ignores it. MLX has fed
`bloc_token_ids` verbatim since the bloc work landed; the **HuggingFace
transformers and GGUF lanes did not** — they re-rendered each module, and the
transformers branch treated `tools is not None` as "rebuild", which reset the
cache and re-prefilled the whole system+tools text. Both now feed the planned
fragment. Measured on `Qwen3-4B-Instruct-2507` (bf16/MPS), a 702-token system
bloc plus a 661-token tools bloc:

| | cold build of `[system, tools]` | after editing ONE tool description |
|---|---|---|
| one merged `system+tools` bloc | 1363 tokens prefilled | 1367 tokens re-prefilled |
| transformers, before this change | 2068 tokens prefilled | 1367 tokens re-prefilled |
| transformers, after | 1363 tokens prefilled | **665** tokens re-prefilled |

So the tools bloc used to cost 52 % *more* to build and save nothing on the edit
it exists for. It now saves 51 % of the prefill on a one-tool change, and the
same figure holds on the MLX lane. On the GGUF lane llama.cpp's own live-context
prefix reuse already skips the shared prefix within a process, so the bloc cut
matches it rather than beating it; the cut still matters there for keyed and
durable reuse, where there is no live context to fall back on.

The GGUF lane keeps a text/token pair (`_gguf_compose_cached_prompt_tokens`
concatenates the stored text with a live suffix), so it derives the text from
the ids it just fed via llama.cpp's detokenizer and **verifies the round-trip**
before trusting it. An unverifiable pair falls back to the rebuild, loudly —
never a stored text that disagrees with its tokens.

### A tools bloc must actually contain tools

The seam checks are all about token boundaries. None of them can see a bloc that
is *structurally* perfect and *empty of content*, which is the failure found on
2026-08-07: a tool set whose descriptions exceeded `ToolDefinition`'s 200-char
cap was converted to **zero** definitions, so `format_tools_prompt` returned `""`
and every local renderer emitted a prompt with no tools in it. The planner
reported `boundaries=[1408, 1411]`, `collapsed=False`, `unsound_at=None` — a
"tools bloc" holding three tokens of closing scaffolding. The model saw no
tools; nothing said so, because the only report was a `logger.warning`, which is
dead in a default AbstractCore process.

Two guards, both `warnings.warn` (the channel a caller actually receives):

- **Per tool, at conversion.** `UniversalToolHandler` announces every tool it
  drops, with the tool name, the model, and the reason. This covers the prompted
  lane *and* `prepare_tools_for_native`, i.e. tools dropped on the wire too.
- **Per chain, at plan time.** If a bloc carries tools whose names do not appear
  anywhere in the prompt the chain will send, `prompt_cache_plan_bloc_chain`
  says so. The cache is not wrong — it faithfully matches what `generate()`
  renders — the *render* is missing the tools, which is why this warns rather
  than collapsing the plan.

**Tool descriptions are capped at 200 characters** (`when_to_use` at 240, three
examples). Over-long descriptions are rejected per tool, not truncated: put the
detail in `when_to_use` or in your docs.

### The composability contract: ordered-prefix only

**Bloc composition is ordered-prefix composition. It is not, and on this
architecture cannot be, free recombination.** Cached `K` tensors are
rotary-position-encoded: the rotary embedding is applied at compile time, at the
absolute offsets the tokens occupied then, and it is baked into the stored
tensors. Nothing in AbstractCore re-applies or shifts it — there is no rope or
offset rewrite in `core/bloc_kv.py` or in any provider. A `K` tensor compiled at
offset 0 is only valid at offset 0.

- **Supported: ordered-prefix composition.** A bloc is reusable when it is
  replayed at the **same absolute start offset**, behind the **same ordered
  prefix**. This is what the module chain does. In this model "composability"
  means *sharing a common prefix* — two agents with different tool sets share the
  `system` bloc's key and its prefill, and only the differing tail is rebuilt.
- **Not deliverable: true N-way recombination.** Reordering blocs, dropping a
  predecessor, or splicing one bloc into a different chain would require
  re-encoding every cached `K` at its new positions. Nothing does that, and adding
  it is a research problem rather than a metadata change. Do not design against
  it, and do not ship an API that implies it.

`BlocKVArtifactManifest` carries `start_offset` and `prefix_chain` so the
constraint is representable, and `bloc_kv_composition_verdict()` returns the
usual three-way verdict (`match` / `mismatch` → refuse and recompile /
`abstain` for pre-axis artifacts). Before those fields existed a wrong
composition could not even be expressed as an error.

**It is enforced, not merely decidable.** `load_bloc_kv_artifact()` runs the
verdict on the adopting path, *before* either `prompt_cache_load` call that
would put the tensors into a live key, and **raises** on a mismatch. A shifted
offset or a different or reordered `prefix_chain` refuses the load rather than
producing a plausible wrong answer, and the refusal is total: on a mismatch the
provider's load is never called and the target key is never created, which is
covered by a test that asserts both. Callers adopting a bloc into a chain pass
`at_offset=` and `prefix_chain=`; the defaults (`0`, `[]`) describe the only
composition this package can execute.

A gate that judges correctly while nothing calls it is not a gate. When you add
a verdict to this area, add the call site and a test that the refusal actually
stops the adoption, in the same change.

### What invalidates a bloc

A bloc's key is a chained SHA-256 over `PromptCacheModule.fingerprint()` of
every module up to and including it. That fingerprint covers `module_id`,
`system_prompt`, `prompt`, `messages`, `tools`, `add_generation_prompt` and
`scope`. Consequences worth knowing:

- Editing the system prompt changes **every** downstream bloc key. Blocs are a
  prefix chain; that is inherent, not a bug.
- Tool **order** is part of the identity. `normalized()` deliberately does *not*
  sort tools: the same value is both the fingerprint input and the rendered
  content, so sorting one side would make the module lane render a different
  order than `generate()` does for the same list and break the prefix at the
  second tool. Callers that emit an unstable order get distinct keys — which is
  what distinct bytes deserve.
- `normalized()` **strips** `system_prompt` and `prompt`, and every renderer
  strips to match. A system prompt with surrounding whitespace would otherwise
  render one way through the bloc chain and another through `generate()`,
  diverging at its first token and losing the whole prefix. If you add a
  renderer branch, strip.
- Message dicts are normalized to a stable subset (`role`, `content`, and
  tool-call fields).

### Coupling costs, stated plainly

- Blocs are a **chain**, not a set. Bloc `k` is only reusable behind exactly
  blocs `1…k-1`. There is no "system bloc + a different memory bloc" without a
  full rebuild of everything after the divergence.
- An individual bloc is **not** an independently valid render. The `system`
  bloc's cache ends mid-turn, with the system block unterminated. That is fine
  for a causal prefix KV — but it means a bloc's bytes are not a usable prompt on
  their own, and a durable artifact of one must record where it sits.
- **Cross-lane sharing is impossible today.** MLX hand-rolls its renderer, GGUF
  uses llama.cpp's embedded Jinja template, and transformers uses
  `apply_chat_template`. Three renderers, three token streams — a bloc compiled
  on one lane is not valid on another, which is why `tokenizer_fingerprint` and
  `cache_backend` are reuse gates.
- MLX's renderer implements **two** of the ten registered message formats
  (`im_start_end`, `gemma_turn`); `llama3_header`, `inst`, `harmony`,
  `glm_special_tokens` and the rest fall through to a plain `role: content`
  form with no special tokens. Bloc planning is still *correct* there — the
  same renderer serves `generate()`, so both sides agree — but the prompts
  themselves are out-of-distribution for those models. Fixing that is separate
  work.

### Where the invariant cannot hold

Honest list. Each of these makes (C2) or (C3) unsatisfiable, and the planner
degrades loudly rather than pretending:

1. **Templates that emit a header or BOS more than once per render.** If adding
   a bloc rewrites bytes an earlier bloc already owns, the boundary cannot be
   monotonic. Detected → `unsound_at`, chain truncated at the last sound
   boundary.
2. **Tokenizers that merge unavoidably across every candidate cut.** If no token
   boundary inside the folded turn is stable, backoff yields an empty fragment.
   Detected → `#FALLBACK` collapse to one bloc.
3. **Native tool-calling templates that pass `tools=` to
   `apply_chat_template`.** Tools may then be interleaved with, or reordered
   against, the system content by Jinja logic the planner cannot see. MLX's
   hand-rolled renderer is not affected; a provider that adopts
   `apply_chat_template` must re-verify (C3) for it.
4. **Sliding-window / `RotatingKVCache` models** (Gemma-3/3n/4, window
   512–1024). The KV is not a durable prefix at agent context sizes at all, so
   bloc reuse is meaningless there regardless of rendering. Documented
   non-target.
5. **Recurrent / hybrid-linear-attention layers** (`ArraysCache`, Gated-DeltaNet:
   Qwen3.5/3.6, Ornith). Their state is not trimmable and not countable, so the
   chain's boundaries cannot be verified against the cache after the fact; those
   lanes use the snapshot/restore path instead.
6. **Any change to the probe family.** The probes define where every boundary
   falls. Changing them changes every bloc's bytes under unchanged keys. Treat
   them as part of the cache format: bump `version` if you must touch them.

### Proofs

- `tests/test_prompt_cache_bloc_composition.py` — token-level composition for
  N = 1, 2, 3 across four real local tokenizers (Qwen3-4B, gemma-4-31b,
  Ornith-9B, Qwen3.5-4B — loaded the same way the runtime loads them, so the
  fixture cannot validate a token stream production never emits) plus
  deterministic toy tokenizers whose merges straddle seams and whose
  post-processor injects a BOS on every encode; successor independence;
  whitespace canonicalization; exactly-one sequence-start token across the whole
  chain; the `#FALLBACK` collapse; and the composition-position verdicts. A hard
  module-level assert requires ≥3 real tokenizers, so the suite cannot go green
  on a box that has none.
  **Not covered by a real tokenizer:** the `llama3_header` family — no Llama 4B+
  model is installed locally and substituting a smaller sibling is inadmissible.
  Its hazard (post-processor BOS on every encode) is reproduced deterministically
  by `ToyAutoBosTokenizer`; the family itself is UNTESTED against real weights.
- `tests/test_bloc_kv.py` — the composition gate refusing a shifted offset and a
  changed prefix chain **at load**, and abstaining with a `#FALLBACK` for
  pre-axis artifacts.
- `tests/test_prompt_cache_bloc_e2e_gate.py` — the end-to-end gate at 10k+
  tokens (opt-in: `ABSTRACTCORE_RUN_BLOC_GATE=1`), with a determinism control, a
  context-dependence control, planted-fact recall at ~5%/50%/95% depth, and a
  prefix-reuse telemetry gate.
- `tests/providers/test_mlx_single_system_block_unit.py` — pins the standalone
  per-module render **as a defect**, so the chain can never be reconnected to it.

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

## Cache residency and memory

In-process caches live for the lifetime of the process. Size them before enabling long-lived keys.

**How much a key costs.** For an attention layer the KV state is
`2 (K and V) × attention_layers × kv_heads × head_dim × 2 bytes` per token. Hybrid models carry KV
on their full-attention layers only, plus a fixed recurrent state that does not grow with the
transcript:

| Model | Unbounded-KV layers | KV per token | Fixed recurrent state | One 5.6k-token key | ×32 keys | ×32 keys at 32k tokens |
|---|---|---|---|---|---|---|
| `Qwen3-4B-Instruct-2507` (dense, 36 layers) | all 36 | **144 KiB** | — | 787 MB | **24.6 GB** | **144 GB** |
| `Qwen3.5-4B` (32 layers, 8 full-attention) | 8 | 32 KiB | ~24 MiB | 175 MB | 5.5 GB | 32 GB |
| `Qwen3.6-27B` (64 layers, 16 full-attention) | 16 | 64 KiB | ~72 MiB | 350 MB | 10.9 GB | 64 GB |
| `Ornith-1.0-9B` (32 layers, 8 full-attention) | 8 | 32 KiB | ~24 MiB | 175 MB | 5.5 GB | 32 GB |

**Full-attention models are the expensive case** — a dense 4B costs more KV per token than a 27B
hybrid, and hybrids are 4–8× cheaper for the same context because only their full-attention layers
keep an unbounded KV. At 144 KiB/token a single 32k-token key costs 4.5 GB, and a full 32-entry store
exceeds the RAM of a 128 GB machine. On architectures that use the snapshot/restore lane, budget
roughly **double** — the live cache plus one boundary snapshot per key.

**How many keys.** `PromptCacheStore` keeps at most `max_entries` keys (default 32) and evicts the
least recently used *within* the store. Keys are session-scoped and nothing removes them when a
session ends, so entries accumulate for the life of the provider — observed growing 1 → 2 → 4 across
consecutive benchmark sessions against a limit of 32. Set the limit to the concurrency you actually
need, and clear keys you are finished with:

```python
llm = create_llm("mlx", model="...", prompt_cache_max_entries=4)
llm.prompt_cache_clear(key)   # when a session ends
```

**Measuring residency.** Process RSS is not a reliable instrument for this on Apple Silicon: MLX
returns freed buffers to its own allocator pool rather than to the OS, so RSS behaves as a
high-water mark. On a measured run, unloading a 2.35 GB model moved gateway RSS by 37 MB. Use
`get_prompt_cache_stats()` for per-key `token_count` and multiply by the per-token figure above, or
MLX's own allocator counters.

## Safety / limitations

- KV caches consume memory; large caches can be expensive — see
  [Cache residency and memory](#cache-residency-and-memory).
- Reusing a cache key across unrelated prompts is safe for correctness but not free: the delta path
  trims the cache to the longest shared token prefix and feeds the rest, so the answer is always
  produced from the prompt you sent, and what you lose is the saving, not the context.
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
