# ADR 0011: One weights object per model, concurrent requests on it — a per-model slot scheduler in the process that holds the weights

Status: Accepted (2026-10-29). Operator ruling, verbatim: "1) yes we MUST share a pool of resources;
2) however, we must also allow concurrency. Make sure we are on track for that and we have the proper
tests to confirm it."

## Context

### What the audit measured (2026-10-29, probes and raw results in the gateway's `untracked/residency-probe/`)

A gateway process serves one model to several request paths: the pooled client of each account's
runtime (`MultiLocalAbstractCoreLLMClient`: session chat, automations), the gateway-side client
(`LocalAbstractCoreLLMClient` built by `model_host.local_client` for run summaries, Ask on a run, the
console sandbox, entity chat), Core's serving app (`/v1`), and the embedding managers.

- **MLX** (`mlx-community/Qwen1.5-0.5B-Chat-4bit`): one weights object for every path
  (`_SharedMLXModel`: `holders` 1 → 2 → 3 as the admin pool, the bob pool and the gateway-side client
  were built; one `weights_id`; MLX active memory flat at 0.26 GB). Sharing is solved on this engine.
- **GGUF / llama.cpp** (`unsloth/Qwen3.5-4B-GGUF`): every holder loads its own full copy. In process:
  RSS 11.8 GB after the admin pool, 23.3 GB after the bob pool, 34.9 GB after the gateway-side client
  (three `llm` objects, three `weights_id`s, `held_bytes` 34.1 GB for one 2.7 GB file because each copy
  reserves its own context). Through the model host child (ADR 0002 of the gateway): 12.15 GB after the
  pool warm-up, 24.21 GB after the first console-sandbox call; after `kill -9` and the warm rebuild the
  child built the model twice again. On the Linux release server (one 16 GB card) that is two to three
  copies of a 9B model, which is where `decode: failed to find a memory slot` came from.
- **Embeddings**: one `EmbeddingManager` per service and per route (`/v1/embeddings`, the gateway's
  embeddings route, the memory services), each with its own model. PR #28 covers the `/v1` part.

### What serialises requests today

- `HuggingFaceProvider._InstanceGenerationWorker` (`huggingface_provider.py`): every operation on one
  provider instance (generations, prompt-cache prefills, save/load, unload) runs on ONE worker thread,
  one at a time. That is correct for an instance that owns one llama.cpp context: two operations on the
  same context corrupt it. It is also the reason the operator's risk is real: share the instance and
  every request on that model, from every account, waits for the one worker.
- Core's serving app (`server/app.py`, `_GatewayLoadedRuntime.provider_executor`): one executor thread
  per loaded runtime for every provider that does not declare `supports_concurrent_generation()`.
- The MLX **mlx-lm lane** (`_SharedMLXModel`) has no lock at all. Two threads generating on one mlx-lm
  model is a Metal assertion (`-[_MTLCommandBuffer addCompletedHandler:]`, the 2026-08-02 gateway crash
  loop recorded in `mlx_provider.py`): the lane is safe today only because callers happen not to overlap.
- The MLX **native lane** (`mlx_batching=True`, mlx-vlm, `mlx_runtime.NativeRuntime`) already does what
  this ADR asks: one weights object per `NativeSession`, continuous batching over it (`max_batch_size` 4,
  `max_queue_size` 32, `queue_timeout_s` 120), per-job cancel events, typed `queue_full`,
  `queue_timeout`, `stream_backpressure` errors, and `stats()` with `queued` and `active`. It is the
  reference shape for the other lanes.

### What each engine can do (versions installed on the development machine)

- **llama-cpp-python 0.3.35**: the high-level `Llama` object is one context and one sequence; it sets
  `n_seq_max > 1` and `kv_unified` only in embedding mode. One `LlamaModel` (the weights) can back any
  number of `LlamaContext`s, each with its own KV cache; `llama_decode` releases the GIL, so contexts
  decode in parallel threads. Measured in the R1 prototype (`untracked/r1-proto/proto.py`, Metal,
  Qwen3.5-4B Q4_K_M, n_ctx 4096): RSS 2.92 GB with one context, 3.33 GB with three (each extra context
  costs its KV only), one model pointer, three requests 0.48 s one after another and 0.39 s in parallel
  threads, identical outputs at temperature 0, clean close in reverse order. True multi-sequence batching
  inside ONE context (several `seq_id`s in one `llama_batch`, llama-server's slot model) needs the
  low-level batch API, not the high-level class: a later item.
- **mlx-lm 0.31.3**: `BatchGenerator` (continuous batching: `insert`, `remove`, `next`, per-uid prompt
  caches, `stats`) exists. The provider's mlx-lm lane drives `stream_generate` / `generate_step` (one
  sequence) together with its prompt-cache store, hybrid snapshots and delta feed, none of which is
  written against `BatchGenerator` yet. MLX kernels serialise on the GPU; parallel threads on one model
  buy nothing and are unsafe (above).
- **mlx-vlm 0.7.2** (native lane): batched, see above.
- **transformers 5.17**: `generate` runs one batch; concurrent calls on one model on MPS are serialised
  by the process-wide `_MPS_GENERATION_LOCK`; on CUDA by the instance worker. Its continuous-batching
  manager is experimental and not a release gate for this framework.
- **Remote providers** (OpenAI-compatible, Anthropic, Ollama, LM Studio, endpoint profiles): the remote
  server schedules; this process has no weights and no constraint.

### Rules this decision sits under

ADR 0001 (no silent degradation), ADR 0008 (provider-owned residency truth), ADR 0010 (fit before load,
evict unlocked LRU, refuse with a typed error; point 8: a load reserves the maximum context explicitly,
a person's context is used exactly or refused, never shrunk), gateway ADR 0002 (inference runs in the
model host child; the gateway process holds no model), runtime ADR 0007 (the runtime relays Core's
residency truth) and 0008 (hosts reach Core only through runtime facades), framework ADR 0026 (nothing
is truncated silently) and 0043 (readable labels, never ids).

## Decision

### 1. One weights object per model per process, for every request path

Each in-process engine keeps a process-wide registry of loaded weights keyed by the resolved artifact
path plus the load parameters that change the weights in memory (llama.cpp: `n_gpu_layers`, `use_mmap`,
`use_mlock`; MLX: the snapshot path, as `_SharedMLXModel` does today; transformers: device, dtype,
quantisation, adapters). **The context size (`n_ctx`) is not part of the key**: it belongs to a slot
(§2). A provider instance that loads a model whose weights are resident adopts them (no load, no fit
check); holders are weak references, and the weights are freed when the last holder goes
(`_SharedMLXModel` semantics, extended to GGUF by R1 and to transformers afterwards). The gateway-side
clients, `/v1` and the embedding managers go through the same registry as the runtime pools: the audit's
probe, turned into a test (Enforcement, test a), is the proof.

Model-host restart (gateway ADR 0002): the registry is process state and dies with the child; the
replayed `set_*` configuration must rebuild ONE copy per model and warm it once. The audit's "two
builds after restart" is a defect under this ADR.

### 2. Concurrency: a per-model slot scheduler, owned by the process that holds the weights

A **slot** is one engine execution context on the shared weights: for llama.cpp a `Llama` context with
its own KV cache and its own worker thread; for the MLX native lane a row of the `NativeRuntime` batch;
for the mlx-lm and transformers lanes, this wave, the one execution thread. A generation **borrows a
slot for its duration and returns it**; holders own no slot. One scheduler per shared weights object
(`abstractcore.providers.model_scheduler`, the same run/stream interface as `_InstanceGenerationWorker`,
which becomes per-slot instead of per-instance). The native lane's `NativeRuntime` is that scheduler for
its lane and only gains the common state surface (§2.4).

The scheduler lives where the weights live: in the **model host child** in the gateway's `process` mode
(ADR 0002: it holds every provider, the pools, the gateway-side clients and the `/v1` app, so it sees
every request on the model and can order them), in the application process in `in-process` mode. The
runtime and the gateway add no lock, no worker and no second instance to make sharing safe: they pass
the cancel event and the progress callback through, and surface the state the scheduler reports.

#### 2.1 The bound

`max_concurrent` per (provider, model), the number of slots the scheduler may hold. Defaults by engine:

| engine | default `max_concurrent` | why |
| --- | --- | --- |
| llama.cpp, GPU or Metal | 4 | the prototype's parallel decode; each slot costs its KV |
| llama.cpp, CPU-only (`n_gpu_layers=0`) | 1 | `n_threads` is half the cores per context; two contexts oversubscribe |
| MLX, mlx-lm lane | 1 (this wave) | one model, one execution stream; `BatchGenerator` later (§3) |
| MLX, native lane (`mlx_batching=True`) | `mlx_max_batch_size` (4) | already batched |
| transformers | 1 | the MPS lock / the instance worker IS the slot |
| remote providers | unbounded | the remote server schedules; a bound is a rate limit a person sets |

A person sets a model's concurrency in the Core store (`config.model_concurrency.set_model_concurrency`,
the sibling of `set_model_context`), and it is used exactly or refused with the sentence
("MODEL cannot hold 4 slots at 131,072 tokens on this machine: 4 × 2.1 GB of KV need 8.4 GB, 3.1 GB is
free after the weights; set 1 slot, or a smaller context"), never lowered silently.

#### 2.2 Slots, memory and the ADR 0010 fit check — never a silent context reduction

- **Every slot of a pool has the pool's `n_ctx`**, the context ADR 0010 point 8 reserved for the model.
  A request must never get a smaller context because of which slot it landed on: a slot-dependent
  context is a silent context reduction. A holder pinned to a different context (a person's explicit
  context) gets its own slot pool on the same weights (no second copy).
- **The first slot is allocated with the weights** (today's load). **Extra slots are allocated lazily**:
  when a request arrives and every slot is busy and the pool is under its bound, the scheduler runs the
  ADR 0010 fit arithmetic for ONE slot (KV at `n_ctx` + the engine's per-context compute buffers; the
  weights count as already resident) against free memory now. A slot that fits is allocated and takes
  the request. A slot that does not fit is **not allocated and nothing is evicted**: a slot is capacity,
  not a load (eviction stays reserved for loads, ADR 0010 point 4); the request queues on the existing
  slots and the queued state carries the reason (`slot_note`: "a 2nd slot needs 2.1 GB of KV, 0.6 GB is
  free"). The context is never shrunk to make a slot fit.
- After the first extra slot is allocated, its **measured** footprint (`slot_bytes_measured`, the
  engine's own allocation figures, as `context_reservation` records the first) replaces the estimate for
  the next slots, labelled as such.
- Extra slots idle for `slot_idle_ttl_s` (default 300 s) are freed; the first slot stays with the
  weights. The residency row shows the pool: `slots` (allocated), `slots_max`, `slot_bytes`, `active`,
  `queued`, `slot_note`.
- ADR 0010 point 2 extends: a model with an active slot OR a non-empty queue is in use, hence locked
  against eviction; `last_used_at` is stamped per slot borrow and return.

#### 2.3 The queue

- **Fair FIFO by admission time**, one queue per slot pool, every principal in the same queue (a
  session chat, an automation, a `/v1` client, a summary are all requests). Per-principal weighted
  fairness is not in this decision; it becomes an item if the benchmark (Validation) shows one account
  starving another at the queue bound.
- **Bounded**: `max_queue` (default 32). Past it the request fails at once with the typed
  `ModelQueueFullError` (HTTP 503 with `Retry-After`, `error.code = "model_queue_full"`), never a silent
  wait. **Timed**: `queue_timeout_s` (default 120) per request, then `ModelQueueTimeoutError` (HTTP 504,
  `error.code = "model_queue_timeout"`). Both are request-local (never retried, never counted against
  model health), as the native lane's `queue_full` / `queue_timeout` already are.
- A slot is handed to the head of the queue the moment it is returned. A slot's live KV is an affinity
  hint only (a request whose session prefix is in a slot's KV prefers that slot when it is free); FIFO
  never waits for affinity, and correctness never depends on it: the prompt-cache store (llama.cpp
  `LlamaState`, the MLX prompt-cache store) is per MODEL, shared by every slot of the pool.

#### 2.4 The queued state is visible, everywhere, never silent

A request that is not running says so, in words, at every layer:

1. **Progress events** (`providers/generation_progress.py` contract): a new `phase: "queued"` with
   `queue_position` (1-based), `queue_ahead`, `slots`, `slots_max`, `slot_note`, emitted on admission to
   the queue and then at the emitter's cadence (0.5 s) until the request takes a slot, then the existing
   `prefill` / `generate` / `complete` phases. AbstractRuntime turns them into ledger events as it does
   the others; the replay view (runtime ADR 0010) shows "Queued behind N requests on MODEL" in place of
   "Thinking…"; the console and the thin clients render that sentence (framework ADR 0043: the model's
   name, never an id).
2. **Response metadata** on every generation: `execution: {slot, queued_s, slots_active, slots_max,
   mode}` (`mode` is `slot`, `batch` for the native lane, `remote`).
3. **`/v1`** (OpenAI-compatible, so no new body fields): the response header
   `X-AbstractCore-Queued: position=N; ahead=M; slots=A/B` on admission to the queue, and, for a
   streaming request, SSE comment lines (`: queued ahead=M`) at the same cadence until the first token;
   comment lines are legal SSE and ignored by OpenAI clients. The typed errors map to 503 / 504.
4. **Residency rows and claims** (`process_residency.resident_rows`, `claims_for`): the pool fields of
   §2.2, relayed by the runtime (ADR 0007) to the gateway's Models row and `/api/health`, and present in
   the diagnostics bundle.

#### 2.5 Streaming and cancellation

- **Streams interleave.** Each borrowed slot produces into its own bounded bridge (`async_stream` per
  request, as the native lane does today); two concurrent streams on one model both deliver tokens before
  either completes (the per-slot worker never blocks on another slot's consumer). The native lane's batch
  step yields one token per active row per step.
- **Cancellation is per request** (`providers/generation_cancel.py`: the host passes a
  `threading.Event`; an HTTP disconnect sets it through `run_sync_with_disconnect`). Queued: the request
  leaves the queue, raises `GenerationCancelledError`, touches no slot. Active: the slot stops at its
  next observable point (llama.cpp: between `n_batch` slices in prefill, per sampled token in decode;
  mlx-lm: per token, prefill chunks included; native: per job per step) and is returned; the other slots
  never notice. Cancelling one of N affects exactly one.

### 3. Per engine: what ships in this wave, what comes later, what a person sees

| engine | this wave (R1, R2, R3 + the scheduler) | later item | what a person sees now |
| --- | --- | --- | --- |
| llama.cpp (GGUF) | one `LlamaModel` per weights key; a slot pool of contexts on it (lazy, fit-checked, same `n_ctx`); the per-instance worker becomes per-slot; the queue of §2.3 and the state of §2.4 | one context with `n_seq_max` sequences and `kv_unified` through the low-level batch API (continuous batching, one KV budget shared by the sequences, prefix sharing); Core backlog 0871 | N requests on one model run in parallel up to the slots that fit; past that, "queued behind N" and a clean 503 at the queue bound; one copy of the weights however many accounts and paths |
| MLX, mlx-lm lane | shared weights (already); the scheduler with one slot: concurrent requests queue, visibly, cancellable from the queue; two requests never touch the weights at once (the Metal crash class is removed by construction) | the lane rewritten on mlx-lm's `BatchGenerator` (continuous batching, per-uid prompt caches mapped onto the prompt-cache store), `max_concurrent` 4 by default; Core backlog 0871 | one request at a time, in order, with the position shown; no crash under concurrent load |
| MLX, native lane | unchanged batching; its `stats()` and errors exposed through §2.4's surface | — | as today: concurrent requests batch; the queue is now visible in the UI |
| transformers | shared weights through `hf_residency`; the scheduler with one slot (the MPS lock stays and is that slot) | transformers' continuous batching once it is stable; not a release gate | one at a time, queued state visible |
| remote providers | no scheduler; an optional per-endpoint `max_concurrent` as a rate limit, with the same queued event when set | — | unchanged |
| embeddings (R3) | one `EmbeddingManager` per (provider, model) process-wide, on the same registry; its calls are batched by the engine (llama.cpp embedding mode uses `n_seq_max`), `max_concurrent` 1 | — | one embedding model in memory, not one per service |

### 4. What this does NOT decide

- How many slots a given machine should run: the defaults are starting points; the benchmark
  (Validation) and the release report set them per machine class.
- Per-principal fairness beyond FIFO (see §2.3).
- Continuous batching for llama.cpp and mlx-lm: designed above, delivered by backlog 0871.

## Enforcement

The tests below go red without the change they guard; a check whose absent-input case passes is not a
check (framework rule). Files are in AbstractCore unless marked; the gateway and runtime items (gateway
backlog 0978, runtime backlog 0866) carry their halves. Unit tests run in CI with fake engines; the
real-weights tests are opt-in (`ABSTRACT_REAL_WEIGHTS=1`) and run before a release on the development
machine (Metal) and the Linux test server (CUDA).

**(a) One weights instance per model across every path, before and after a model-host restart.**
`tests/providers/test_shared_weights_every_path_unit.py` (fake engine objects, counts constructor
calls) and gateway `tests/test_model_host_one_weights_copy.py` (the audit's `probe_gateway.py` turned
into a test: admin pool, bob pool, gateway-side `local_client`, `/v1/chat/completions`, `/v1/embeddings`
and the gateway embeddings route, then `kill -9` of the child): asserts `builds_of_model_total == 1`
per child generation, one `weights_id` in the residency rows, `holders` equal to the number of paths,
`held_bytes == weights_bytes + slots × slot_bytes`. Red without R1 (GGUF builds 2-3 copies), without
R2 (the gateway-side client builds its own), without R3 (two embedding managers), and without the
restart fix (two builds after restart). Real-weights twin: `tests/providers/test_gguf_shared_weights_real.py`
with `unsloth/Qwen3.5-4B-GGUF` (Q4_K_M) and `mlx-community/Qwen1.5-0.5B-Chat-4bit`.

**(b) N concurrent requests, no second copy, wall-clock.** `tests/providers/test_model_slot_scheduler_unit.py`
(a fake engine whose decode sleeps a configurable time): with N = 4 and `max_concurrent` 4 the total
wall-clock is under 1.5 × a single request; with N = `max_concurrent` + 2 the two extra requests carry
`phase: "queued"` events with `queue_ahead` 2 then 1, and the total is about 2 × a single request;
`max_queue` + 1 requests → exactly one `ModelQueueFullError`; a `queue_timeout_s` of 0.1 s with a busy
pool → `ModelQueueTimeoutError`; order of completion follows admission order. Real weights:
`tests/providers/test_gguf_concurrency_real.py` (Qwen3.5-4B-GGUF, Metal and CUDA): N = 4 concurrent
`generate` calls complete, one `LlamaModel` pointer, RSS after the run ≤ RSS after the first slot
allocation + 3 × `slot_bytes_measured` + 256 MB, and wall-clock < 0.7 × the serial time (the prototype's
0.39 s vs 0.48 s is the floor; the real figure goes in the release report). MLX mlx-lm lane
(`test_mlx_lm_concurrency_real.py`, Qwen1.5-0.5B-4bit): N = 4 complete serially, no crash, every request
after the first saw a `queued` event, wall-clock ≈ serial time. Native lane: the existing
`tests/providers/test_mlx_runtime_*` already prove batching; they gain the §2.4 state assertions.

**(c) Streaming interleaving.** `tests/providers/test_model_slot_scheduler_stream_unit.py`: two streams
on a pool of 2 slots; the test records the arrival order of chunks and asserts that both streams have
delivered at least one token before either delivers its last one. Red when one worker serves both
(the second stream's first token arrives after the first stream's last).

**(d) Cancellation of one of N.** Same file: four active requests, cancel the second's event mid-decode →
it raises `GenerationCancelledError` within one token, its slot is returned (the queue head takes it),
the other three complete with full output; cancel a QUEUED request → it leaves the queue without
touching a slot and the positions behind it decrease by one. Red when cancel stops the shared worker.

**(e) Memory flat, within the fit check.** `tests/providers/test_slot_fit_check_unit.py`: the fit check
is asked for exactly one slot's bytes with the weights marked resident; a slot that does not fit is not
allocated, nothing is evicted (the eviction hook is a spy that must not be called), the request queues
and the `slot_note` names the numbers; a person's `model_concurrency` that does not fit is refused with
the sentence, never lowered. Real weights (in `test_gguf_concurrency_real.py`): sample RSS (and
`torch.cuda.mem_get_info` / `mx.get_active_memory` where applicable) every 0.5 s during the N = 4 run:
max − min ≤ the slots' measured bytes + 256 MB; no sample exceeds the fit check's admitted total.

**(f) Benchmark.** Root backlog 1064's `scripts/perf/bench_concurrency.py` (gateway) gains a
real-model mode: matrix 1×1, 1×4, 4×1, 4×4 (users × sessions) on one GGUF model and one MLX model,
reporting per request p50/p95/p99 latency, time to first token, time queued, slots active, throughput
(tokens/s total), RSS and GPU memory min/max, and the number of weights copies (must be 1). The first
baseline is recorded in the release report; the gate is "copies == 1 and p95 queued time at 4×4 ≤ 3 ×
the single-request time", tightened from the baseline.

**Review** refuses (i) a shared instance whose generations all pass through one worker without the
scheduler; (ii) a second provider instance or second context created to obtain concurrency outside the
scheduler; (iii) any `n_ctx` lowered to make a slot fit; (iv) a queue wait with no `queued` event and
no bound.

## Validation

- Real-weights proofs, smallest models that prove them: GGUF `unsloth/Qwen3.5-4B-GGUF` Q4_K_M (the
  audit's model; 2.7 GB of weights, 0.2 GB per 4,096-token slot) on Metal and on the Linux test server's
  16 GB card; MLX `mlx-community/Qwen1.5-0.5B-Chat-4bit` (0.26 GB) for the mlx-lm lane; the native
  lane's existing `Qwen3.5-4B` pair. CI runs the unit tests only.
- The release report records, per engine: copies (1), slots allocated at the default context, the
  N = 4 wall-clock ratio, the RSS/GPU delta, the queued-event latency, and the 1064 matrix numbers.
- The operator's acceptance: on the Linux server, two accounts and a `/v1` client on one 9B GGUF model
  at the same time, one copy in memory, every request answered or visibly queued, nothing crashed.

## Backlog links

- AbstractCore 0870 (this wave: the scheduler, the shared GGUF weights, the test files above);
  0871 (later: continuous batching for llama.cpp `n_seq_max` and the mlx-lm `BatchGenerator` lane).
- AbstractRuntime 0866 (the `queued` phase in the ledger and the replay view; pools borrow, never
  build, for gateway-side generations; the `model_concurrency` facade).
- AbstractGateway 0978 (the Models row and `/api/health` pool fields, the `/v1` header and SSE
  comments, the model-host one-copy test, the 1064 benchmark extension).
- Root 1064 (the concurrency benchmark), root 1060 (resilience standing duty).

## Related

- ADR 0010 (fit and eviction; this ADR adds slots to its arithmetic and "queued" to its "in use").
- Gateway ADR 0002 (the model host child owns the scheduler in `process` mode).
- Runtime ADR 0007, 0008, 0010.
- `docs/native-mlx-runtime.md` (the native lane's scheduler), `docs/concurrency.md`,
  `docs/memory-management.md`.
