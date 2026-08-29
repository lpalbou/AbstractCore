# Memory and Model Residency

AbstractCore gives you a small, dependable set of surfaces for one operational question: **what is
using memory on this host right now, and how do I free it — or keep it?** This page covers the
host memory snapshot, host-wide model residency discovery, per-provider loaded-model listings,
what `unload_model()` actually frees, locking a model against unloading, and estimating how much
context a model can sustain before you load it.

Related pages:

- [Architecture](architecture.md) — the provider lifecycle and the `unload_model()` contract
- [API Reference](api-reference.md) — full method signatures
- [Prompt Caching](prompt-caching.md#cache-residency-and-memory) — sizing KV caches and per-key cost
- [Server](server.md) — the HTTP routes (`/acore/memory`, `/acore/models/loaded`, `/acore/prompt_cache/*`)

## Host memory snapshot

`abstractcore.utils.memory.get_memory_snapshot()` reports system RAM, this process, and the
accelerator backend in one call. It is pure observation: every probe is individually guarded,
unknown values stay `None`, and the function never raises.

```python
from abstractcore.utils.memory import get_memory_snapshot

snap = get_memory_snapshot()
print(snap["ram"]["available_bytes"])       # can another model fit?
print(snap["device"]["allocated_bytes"])    # what does the accelerator hold right now?
```

Shape:

```json
{
  "ts": 1700000000.0,
  "ram": {"total_bytes": 137438953472, "available_bytes": 33013366784, "used_bytes": 95548260352, "percent": 76.0},
  "process": {"rss_bytes": 53182464},
  "device": {"backend": "metal", "allocated_bytes": 0, "total_bytes": 137438953472, "free_bytes": null,
             "host_in_use_bytes": 9911418880, "wired_limit_bytes": 115343360000},
  "host": {"host_id": "a1b2c3d4e5f6", "host_name": "studio.local", "kind": "local"}
}
```

- `ram` comes from `psutil` when available.
- `process.rss_bytes` is this process's resident set size.
- `device` probes MLX first (`backend: "metal"`), then `torch.cuda` (`"cuda"`), then `torch.mps`
  (`"mps"`); with none available, `backend` is `null`.
- `device.allocated_bytes` is **this process's** accelerator memory (mlx active memory on Metal).
  It is truthful per-process but blind to other processes: a model resident inside LM Studio's or
  Ollama's process reads `0` here.
- `device.host_in_use_bytes` (Metal only, else `null`) is the **accelerator heap across
  processes**: driver-allocated Metal buffers, read from IORegistry
  (`ioreg -r -c IOAccelerator -l`, the `"In use system memory"` PerformanceStatistics figure).
  It sees MLX models whichever process allocated them, including MLX-engine servers such as
  LM Studio. It is **not** the host's total memory use, and **memory-mapped GGUF/llama.cpp
  weights do not appear in it at all** — see
  [Reading the snapshot](#reading-the-snapshot-process-local-accelerator-heap-and-process-rss).
- `device.wired_limit_bytes` (Metal only, else `null`) is the enforced accelerator ceiling:
  `sysctl iogpu.wired_limit_mb` when set (> 0), else Metal's
  `max_recommended_working_set_size` — the same ceiling `estimate_context_fit()` budgets against
  (one shared helper, `abstractcore.utils.memory.metal_wired_limit_bytes`).
- `host` names the machine the snapshot was observed on (see
  [Modalities and host identity](#modalities-and-host-identity-on-residency-records)).
- Metal exposes no free-bytes query, so `device.free_bytes` is `null` on Apple Silicon. On unified
  memory machines, `ram.available_bytes` is the practical headroom figure.

The same snapshot is served over HTTP as `GET /acore/memory` on the AbstractCore server.

## Reading the snapshot: process-local, accelerator heap, and process RSS

Three device figures answer three different questions. Picking the wrong one is the quickest way to
conclude "nothing is loaded" while a 93 GB model is resident.

**`device.allocated_bytes` answers "what does *this process's* accelerator allocator hold?"** It is
the authoritative "memory freed" signal for models loaded in-process through MLX: after unloading
an MLX model it drops by roughly the model-plus-KV size, back to near zero when nothing else is
resident.

It is process-local, and blind in two ordinary directions:

- a model resident in **another process** — anything LM Studio or Ollama holds — is not counted;
- a model resident in **this** process through a backend with its own allocator is not counted
  either. A llama.cpp/GGUF model loaded through the HuggingFace provider maps its weights outside
  mlx's allocator, so `allocated_bytes` can read `0` while that model is fully resident in this
  very process.

`allocated_bytes: 0` therefore never means "no model is loaded on this host". It means "this
process's accelerator allocator holds nothing".

**`device.host_in_use_bytes` answers "how much accelerator *heap* is allocated across processes?"**
It is a driver-allocator figure (Metal only): it counts buffers the Metal driver allocated, in any
process. An MLX model shows up here whether this process or an MLX-engine server such as LM Studio
allocated it.

It has one large, systematic blind spot, and the UIs must never paper over it:

- **Memory-mapped GGUF/llama.cpp weights are not counted.** llama.cpp `mmap`s the `.gguf` and wraps
  those pages with `newBufferWithBytesNoCopy`, so the weights are file-backed and never become
  driver-allocated accelerator memory. Measured on an Apple-silicon host: a fully offloaded
  (`n_gpu_layers=-1`) 89.99 GB three-shard GGUF left `host_in_use_bytes` at 0.79 GB. The same host
  reported `allocated_bytes: 0` and `process.rss_bytes` of 75.81 GB.

So `host_in_use_bytes` is **not** the host's total memory use, and it is not a denominator for
"how full is this machine". A resident-model total legitimately and routinely exceeds it — that is
the normal GGUF case, not an inconsistency. Where do mmapped weights show up instead?
`process.rss_bytes`, `ram.used_bytes`, and the model's own `est_weights_bytes`.

Compare `host_in_use_bytes` against `device.wired_limit_bytes` for accelerator-heap headroom, and
read `ram` for the system picture. It is a total, not an attribution: it cannot tell you *which*
model or process holds the memory. For per-model figures, read the residency records —
`size_bytes` / `size_vram_bytes` from a model server, `est_weights_bytes` from an in-process
provider (see
[Per-model memory on residency records](#per-model-memory-on-residency-records)).

**Presenting it.** Every AbstractFramework surface (web console, abstractflow, both TUIs,
`monitor-memory`) renders RAM as the primary system meter and shows this figure as its own
clearly-scoped line labelled **`Accelerator heap · <backend> (all processes)`**, with the note
*"memory-mapped GGUF weights are not counted here"*. No surface labels it as the host's memory use,
and none subtracts RAM-dimensioned quantities from it.

**Process RSS is not a reliable unload signal on Metal hosts.** Freed Metal buffers return to the
process allocator rather than to the operating system, so `process.rss_bytes` behaves as a
high-water mark: it does not drop after an in-process unload and can even grow slightly. Monitoring
or alerting that verifies unload through RSS will report false negatives; read
`device.allocated_bytes` instead, and use `ram.available_bytes` / `ram.percent` for host-level
capacity decisions.

In short: verify an in-process unload with `device.allocated_bytes`, render accelerator-heap usage
from `device.host_in_use_bytes` against `device.wired_limit_bytes` (labelled as the accelerator
heap, never as total memory use), read `ram` for host capacity, and never read any of them as a
per-model number.

## Host-wide residency: which models are loaded on this machine

`abstractcore.utils.residency.sweep_loaded_models()` asks the host's local model servers — Ollama
(`GET /api/ps`) and LM Studio (loaded instances via its native REST API) — which models they
currently hold resident, without constructing any model-bound provider:

```python
from abstractcore.utils.residency import sweep_loaded_models

for record in sweep_loaded_models(timeout_s=2.0):
    print(record["provider"], record["model"], record.get("size_bytes"))
```

Each record is the server's own enumeration, normalized and tagged `source: "provider_server"`:

```json
{
  "provider": "ollama",
  "model": "qwen3:latest",
  "resident": true,
  "loaded": true,
  "source": "provider_server",
  "size_bytes": 5225376768,
  "size_vram_bytes": 5225376768,
  "expires_at": "2026-01-01T00:05:00Z",
  "context_length": 4096
}
```

The sweep is best-effort and never raises: a server that is not running, unreachable, or errors
contributes nothing (unknown stays unknown — it is never guessed). LM Studio records carry
`provider_instance_ids` and `size_bytes` where the server reports them; the LM Studio REST payload
has no VRAM split, so those records never carry `size_vram_bytes`.

Note that residency covers whatever the servers hold — Ollama's running-model list includes
embedding models too, so a sweep record does not imply a text-generation model.

### Matching helpers

The module also exposes the alias rules used to compare sweep records against registry or runtime
records, so every consumer deduplicates the same way:

```python
from abstractcore.utils.residency import (
    SWEEP_PROVIDERS,        # ("ollama", "lmstudio") — the providers the sweep covers
    normalize_sweep_model,  # case-normalized name; strips Ollama's ":latest" alias
    sweep_models_match,     # does (provider, model) name the same resident model as a sweep row?
)

normalize_sweep_model("Qwen3:latest")                    # "qwen3"
sweep_models_match("ollama", "qwen3", {"model": "qwen3:latest"})   # True
```

`sweep_models_match` applies `normalize_sweep_model` equality for all providers, and for LM Studio
additionally the provider's own substring resolution against the server key and loaded instance
ids (so `qwen3-vl-4b` matches the server key `qwen/qwen3-vl-4b`). Empty inputs never match.

## Provider-level listings

Every provider instance offers `list_loaded_models(filters=None)`, returning the models it can
**verify** as loaded (residency is provider-owned truth; unknown residency yields an empty list,
never a guess):

- **Default (BaseProvider)**: derived from `get_model_residency()` — one record when the instance's
  model is verified loaded, otherwise empty.
- **Ollama / LM Studio instances**: enumerate **all** models resident on their server, not just the
  instance's own model. Transport errors propagate (an unreachable server raises rather than
  returning a false "nothing loaded"); `sweep_loaded_models()` is the catching, best-effort layer.
- **MLX / HuggingFace**: the in-process residency record plus a best-effort `est_weights_bytes`
  estimate of the resident weight size.

Records carry normalized `size_bytes` / `size_vram_bytes` when the backend reports sizes. The
`filters` argument is accepted for capability-handler compatibility and ignored.

### Per-model memory on residency records

Two per-model figures ride residency records where they are knowable. Both are best-effort:
absent means unknown, never zero.

- **`est_weights_bytes`** — the in-memory weight footprint reported by the provider's own residency
  claim (`get_model_residency()`), for providers that serve the model **in this process** and
  therefore know it: MLX sums the loaded parameter arrays; HuggingFace uses the total on-disk size
  of the resolved `.gguf` for a GGUF model (a memory-mapped quant's on-disk size *is* its weight
  footprint) and the summed parameter bytes for a transformers model. Ollama and LM Studio do not
  report it — their servers report `size_bytes` / `size_vram_bytes` instead.

  For a **split GGUF** (`<stem>-00001-of-00003.gguf`) this is the sum of **every shard**, not the
  file llama.cpp was handed. llama-cpp's `model_path` names only the first shard, and that shard
  can be a rounding error against the set: measured live, first shard 10,946,624 B against
  89,986,353,824 B for the three-shard quant. A partially-fetched set sums the shards actually
  present rather than claiming completeness.

  Because a GGUF's weights are memory-mapped, this footprint is resident as process RSS and is
  **not** visible in `device.host_in_use_bytes`.
- **`cache_bytes`** — the total bytes the runtime's prompt-cache store holds, on managed residency
  records served by the AbstractCore server: the per-key `bytes` figures plus MLX's hybrid boundary
  snapshot bytes, the same arithmetic `get_prompt_cache_stats()` exposes per key. An empty store is
  a known `0`; a store whose keys carry no byte figures stays unknown.

Weights and cache are **separate** footprints. Adding `cache_bytes` into a model's size
double-counts what the device actually holds; render it as its own figure.

A UI that shows one size per row should read the first field that is known, in this order:
`size_bytes`, `size_vram_bytes`, `est_weights_bytes`. The first two come from a model server that
holds the weights; the third is the only figure an in-process MLX or GGUF runtime can offer,
because there is no server to ask.

For server-wide queries without constructing a provider instance, use the classmethods:

```python
from abstractcore.providers.ollama_provider import OllamaProvider
from abstractcore.providers.lmstudio_provider import LMStudioProvider

OllamaProvider.list_server_loaded_models()      # base_url from OLLAMA_BASE_URL / OLLAMA_HOST, else http://localhost:11434
LMStudioProvider.list_server_loaded_models()    # base_url from LMSTUDIO_BASE_URL, else http://localhost:1234/v1
```

Both accept `base_url=` and `timeout_s=` and raise on transport errors.

## What `unload_model()` frees

`unload_model()` is the single best-effort unload entrypoint across providers (see
[Architecture](architecture.md#memory-management) for the per-provider behavior). For in-process
providers (MLX, HuggingFace), unloading frees the model weights **and drops the instance's session
caches** — the prompt-cache store and, on MLX, the hybrid KV boundary snapshots. Session caches are
only useful while the weights are resident, and they are the memory hogs, so unload means "free the
memory", including them.

```python
from abstractcore import create_llm
from abstractcore.utils.memory import get_memory_snapshot

llm = create_llm("mlx", model="mlx-community/Qwen3-4B-4bit")
llm.generate("Hello", prompt_cache_key="session-1")

llm.unload_model(llm.model)
print(get_memory_snapshot()["device"]["allocated_bytes"])  # back to near zero
```

Verify the unload through `device.allocated_bytes`, not process RSS (see
[Reading the snapshot](#reading-the-snapshot-process-local-accelerator-heap-and-process-rss)).

To size and inspect the session caches themselves — per-key `token_count` and best-effort `bytes` —
use `get_prompt_cache_stats()`; see
[Prompt Caching — Cache residency and memory](prompt-caching.md#cache-residency-and-memory).

## Locking a model in memory

When the AbstractCore server keeps a runtime warm (`POST /acore/models/load`), you can lock it
against unloading:

- `POST /acore/models/lock` sets a registry-level lock on a runtime whose model is verified
  **resident** in provider memory, and `POST /acore/models/unlock` clears it. Both select the
  runtime by `runtime_id` or by `provider` + `model` (optional `base_url`).
- **Lock requires provider-verified residency** (`provider_resident: true`). A warm registry
  runtime alone is configuration, not memory — locking it would present a configured model as
  loaded. Locking a non-resident runtime refuses with HTTP `409` and body
  `{"ok": false, "error": "model_not_resident", "detail": "...", "runtime_id": "..."}`; load the
  model first (`POST /acore/models/load` with `"lock": true`) to lock it at load time.
- **A `provider` + `model` selector naming no managed runtime adopts a sweep-resident model.**
  Models resident on the host's local model servers are not always loaded through this gateway —
  you loaded one in the LM Studio app, or `ollama run` did. Locking such a pair creates the managed
  runtime entry for it through the same path a load uses, **client construction only: no cold
  model load happens**, so nothing is loaded twice and nothing is re-downloaded.
  Residency is then re-verified with the provider's own probe before the lock is set, and the
  response carries `"adopted": true`.

  One precise caveat, because "no provider-side call happens" would be false: on **Ollama** the
  lock is reinforced with the server's residency-pin knob, `load_model(model, keep_alive=-1)`,
  which POSTs `/api/generate` with `prompt: ""` — the standard preload idiom. On a model the
  probe has just verified resident (the only kind the lock rule accepts) that request is a
  keep-alive **refresh**: nothing is generated, no weights are re-read, only the TTL moves.
  Unlock re-verifies residency first and skips the restore when the model is gone, so neither
  direction can load a model back as a side effect. A pair the sweep does not verify resident refuses with the
  same `409 model_not_resident` body (`runtime_id: null` — nothing was adopted); a pair whose
  provider probe disagrees with the sweep refuses too, and the just-created entry is dropped.
  Adoption applies to the sweep providers only (Ollama, LM Studio): MLX and HuggingFace residency
  lives on an owning provider instance, not on a server that can be asked.
- `POST /acore/models/load` with `"lock": true` locks the runtime in the same call. When the
  provider cannot verify the loaded model resident, the load still succeeds and the response's
  `lock` block reports `{"locked": false, "error": "model_not_resident", ...}` additively.
- A locked runtime refuses `POST /acore/models/unload` with HTTP `409` and body
  `{"ok": false, "error": "model_locked", "detail": "...", "runtime_id": "..."}`. Pass
  `"force": true` to unlock and unload in one call; if the provider unload fails, the runtime
  stays registered and locked.
- Automatic cleanup never evicts a locked model: `unload_after` on chat requests skips locked
  runtimes, including requests that address the same server-resident model directly rather than
  through the managed runtime (matched with the same alias rules as the sweep).

The registry lock is the enforcement truth. Where the provider has a residency knob of its own,
lock and unlock also apply it best-effort: on Ollama, lock maps to `keep_alive: -1` and unlock
restores the server default (`"5m"`). The lock/unlock response reports that side channel as
`provider_side` — `{"supported": bool, "applied": bool}` plus a `detail` when it failed — and a
provider-side failure never fails the lock. Providers without such a knob report
`provider_side: {"supported": false, "applied": false}` while the gateway lock still enforces.

**LM Studio has no residency-pin knob, and the lock says so.** A lock on an LM Studio model
answers:

```json
{"supported": false, "applied": false,
 "detail": "lock guards this stack's unloads; the external server may still evict on its own policy"}
```

The lock is real — it blocks this stack's unloads and `unload_after` cleanup — but LM Studio is an
external server running its own eviction policy (idle TTL, JIT model switching, the user clicking
"Eject"). A model locked here can still disappear from under you, and the lock cannot prevent it.
Re-read `GET /acore/models/loaded` rather than assuming a lock guarantees residency.

The Ollama keep-alive knob rides that provider's native load request, which is why lock only
reaches resident models (see the lock rule above) — otherwise setting the knob would itself be a
server-side load. Unlock always works — including on a locked runtime whose
model was since evicted, so locks are never stranded — and skips the keep-alive restore for a
non-resident model (`provider_side.applied: false` with a detail) so unlock never loads a model
back as a side effect either.

Residency records report lock state truthfully:

- managed runtime rows carry `locked`, `lockable: true`, and `locked_at` (present only while
  locked); `pinned` is a compatibility alias carrying the same value as `locked`;
- `locked`/`pinned` reflect the gateway-managed lock only — Ollama's own server-side keep-alive
  shows through the record's `expires_at`;
- sweep-only rows (`source: "provider_server"`) carry `lockable: true`: locking one adopts it into
  the registry, so it is lockable even though no managed runtime exists for it yet.

## Context calibration and estimation

### The calibration store

When the HuggingFace provider loads a GGUF model, it settles a workable context size by walking a
descending ladder of candidate `n_ctx` values and probe-decoding each rung. Where that ladder ran
(that is, when you did not pin the context with an explicit `max_tokens`), the settled context is
recorded to a per-machine calibration store —
`~/.abstractcore/calibration/context_calibration.json`, directory overridable with
`ABSTRACTCORE_CALIBRATION_DIR` — keyed by provider, model artifact, and the machine's device/RAM
totals. Later loads of the same model on the same hardware seed the ladder with the recorded rung
and skip the larger rungs that already failed; the seeded rung is still probed, because memory
conditions change — a calibration entry is a hint, never a fact. Writes are atomic under a
cross-process lock, the file is capped (oldest entries drop), and a corrupt file is abandoned and
rebuilt: the store is a cache, not a source of truth.

Where the ladder ran, the provider's residency claim — and the managed residency record served by
the gateway — carries `context_calibrated: true` and `calibrated_context_length`.

### Estimating context fit

`abstractcore.utils.context_estimate.estimate_context_fit()` answers "how large a context could
this model sustain on this host, and what would it cost in KV memory" without loading any weights:

```python
from abstractcore.utils.context_estimate import estimate_context_fit

result = estimate_context_fit("ollama", "qwen3:4b", context_length=32768)
print(result["confidence"], result.get("predicted_max_context"))
```

The same estimate is served over HTTP as
`GET /acore/models/context_estimate?provider=&model=&context_length=`.

Every answer is labeled with a `confidence`:

- `"calibrated"` — a calibration entry for this provider/model/hardware exists;
  `predicted_max_context` is the measured settled context, also reported as
  `calibrated_context_length`. Calibration always wins over estimation.
- `"estimated"` — computed from model geometry and the host's **memory budget**. The budget basis
  is a real ceiling, not a blanket fraction: on Metal/MPS it is `sysctl iogpu.wired_limit_mb`
  when set (> 0) — the wired-memory ceiling the OS actually enforces, which operators raise
  precisely so larger models fit — else Metal's own `max_recommended_working_set_size` (via mlx
  `device_info()`), else a stated fallback of 75% of the device/RAM total (labeled as fallback in
  the notes); currently-allocated device memory is deducted. On CUDA the basis is
  `torch.cuda.mem_get_info()` free bytes. A small reserve — `max(2 GiB, 5% of the basis)` — is
  subtracted, and the `notes` array states basis and reserve so the budget (`budget_bytes`) is
  auditable. `predicted_max_context` is the context that fits **beside the weights**:
  `(budget - est_weights_bytes) // kv_bytes_per_token`, clamped to the model's maximum context
  (`budget // kv_bytes_per_token` with a note when the weight size is unknown), and `null` when
  the weights alone exceed the budget. The KV math assumes an f16 KV cache.
- `"unknown"` — no geometry source and no calibration entry; `predicted_max_context` is `null`.

Two additive tri-state verdicts split weights-fit from context-fit:

- `fits_weights` — `true`/`false` when `est_weights_bytes` and the budget are both known
  (`est_weights_bytes <= budget_bytes`), else `null`. A 93 GB quant on a ~110 GB wired-limit
  ceiling reads `fits_weights: true` with a reduced `predicted_max_context` — never "doesn't fit".
- `fits_requested_context` — whether weights **plus** the KV cost of the requested
  `context_length` fit the budget; `null` when any component is unknown.

The estimate is **advisory only**: no load path in AbstractCore (or the runtime/gateway layers
above it) gates or blocks on it — a real load must still probe.

Geometry sources by provider (unknown members are omitted, never guessed, and the function never
raises):

- `huggingface` — a local GGUF file's header (direct path or hub cache, `repo:selector`
  references included) via `abstractcore.utils.model_cache.read_gguf_geometry()`, else the
  snapshot's `config.json` via `model_geometry_for()`;
- `mlx` — the snapshot or local-directory `config.json`;
- `ollama` — the server's `/api/show` model metadata (short timeout, best-effort);
- `lmstudio` — no geometry source; results stay `"unknown"` unless calibrated.

Results also include the observed `memory` view (`ram_available_bytes`, `device_total_bytes`,
`device_allocated_bytes`), `budget_bytes`, `geometry`, `kv_bytes_per_token`, `est_weights_bytes`
(the GGUF file size or the snapshot's weight-file sum), and `est_kv_bytes` for a requested
`context_length` — each present only when known.

## Modalities and host identity on residency records

Residency records served by the AbstractCore server carry two more pieces of truth:

- **`modalities`** — the registry-declared modality routes for the model (for example
  `["input.text", "input.image", "output.text"]`), from
  `abstractcore.providers.model_capabilities.modalities_for_model()`, stamped on
  text-generation and host-sweep records (capability-task rows such as image or TTS runtimes do
  not carry the field). A model the registry does
  not know is served **without** the field: absence means unknown, never "text-only". When a
  provider instance knows its vision lane is unusable (an MLX runtime without the vision stack),
  `input.image` is removed from that record and `modalities_note: "vision_unusable"` says why.
- **`host_id` / `host_name`** — the identity of the machine the record was observed on, from
  `abstractcore.utils.hostinfo.get_host_identity()` (`host_id` is a stable 12-hex hash of the
  hostname; the function is cached and never raises). Every row served by
  `GET /acore/models/loaded` carries them, and `get_memory_snapshot()` includes the same identity
  as its top-level `host` block, so listings gathered from multiple machines can be merged without
  ambiguity.

## HTTP surface

The AbstractCore server exposes the same visibility over HTTP (see [Server](server.md) for full
route documentation):

- `GET /acore/memory` — the host memory snapshot, including the `host` identity block and the
  process-local and cross-process accelerator-heap device figures.
- `GET /acore/models/loaded` — warm gateway runtimes merged with the host sweep; sweep-only rows
  carry `source: "provider_server"`. Rows carry lock state, `modalities` where declared (text and
  sweep rows), `host_id` / `host_name`, and the per-model memory figures
  (`size_bytes` / `size_vram_bytes` / `est_weights_bytes` / `cache_bytes`) where known.
- `POST /acore/models/lock` / `POST /acore/models/unlock` — lock or unlock a warm runtime, or adopt
  and lock a sweep-resident one; see [Locking a model in memory](#locking-a-model-in-memory).
- `GET /acore/models/context_estimate` — the context-fit estimate; see
  [Estimating context fit](#estimating-context-fit).
- `GET /acore/prompt_cache/stats` — per-key cache stats for one runtime, or an enumeration across
  every loaded runtime when called with no selector.
