# Planned: MLX provider continuous batching

## Metadata
- Created: 2026-05-18
- Status: Planned
- Completed: N/A

## Context

AbstractCore already has three relevant pieces for MLX concurrency:

- `MLXProvider` for in-process Apple Silicon inference;
- `BaseProvider.agenerate(...)` for async entry points;
- `examples/performance/mlx_concurrency_benchmark.py`, which proves that the installed `mlx-lm`
  runtime can drive continuous batching through `BatchGenerator`.

Today these pieces are disconnected.

`MLXProvider` still executes one request at a time through direct `generate(...)` /
`stream_generate(...)` calls. Async usage works only through `asyncio.to_thread(...)`, which keeps
the event loop responsive but does not provide a batching scheduler. The benchmark script shows that
`mlx-lm` can do much better under concurrent load, but that path is not available through the normal
provider API or the OpenAI-compatible server.

There is a second architectural constraint: the OpenAI-compatible server currently creates a fresh
provider instance per request, while `abstractcore.endpoint.app` can serve through a singleton
provider instance. That means continuous batching cannot be implemented only as per-instance mutable
state inside `MLXProvider`; the batching runtime must be shared across requests that target the same
MLX model/configuration.

## Current Code Reality

- `MLXProvider` loads one model/tokenizer pair in-process and calls:
  - direct generation via `self.generate_fn(...)`;
  - direct streaming via `self.stream_generate_fn(...)`.
- `BaseProvider.agenerate(...)` falls back to `asyncio.to_thread(...)` for providers that do not
  implement native async execution.
- the OpenAI-compatible server request path calls `create_llm(...)` inside the endpoint handler, so
  provider instances are short-lived from that server's point of view.
- `abstractcore.endpoint.app` can instead hold a singleton provider instance for its whole process.
- the MLX benchmark uses `mlx_lm.generate.BatchGenerator`, which supports:
  - continuous batching;
  - per-request samplers;
  - per-request caches;
  - per-request logits processors;
  - per-request state machines.
- current MLX prompt-cache storage is provider-instance-local, which is useful for direct Python
  usage but not sufficient for server-wide cache reuse across independently created provider
  instances.
- structured-output handling does not only go through public `generate(...)` / `agenerate(...)`;
  `StructuredOutputHandler` calls `provider._generate_internal(...)` directly, and MLX native
  Outlines support lives inside that path.

## Problem

Without a shared batching runtime, AbstractCore leaves a large amount of Apple Silicon throughput on
the table and exposes a misleading async story for local MLX inference:

- concurrent MLX requests are not co-batched;
- server traffic cannot benefit from a per-model long-lived scheduler;
- naive concurrent usage risks worse latency and Metal/thread-safety issues;
- prompt-cache behavior and batching behavior live in separate, incompatible lifetimes;
- the benchmark path and the production provider path diverge.

If we add batching naively, we create other risks:

- per-request provider instances will each load their own model or their own scheduler;
- sync, async, and streaming APIs can diverge in behavior;
- cancellations and disconnects can leak queued work;
- long prompt prefills can starve active decode streams;
- structured output and stop handling can become inconsistent across requests.

## What We Want To Do

Implement continuous batching as a first-class execution mode of the MLX provider, backed by a
process-wide shared MLX runtime keyed by model/configuration, and make the OpenAI-compatible server
use the same path automatically when it creates MLX providers.

The MLX batching runtime should:

- own the model, tokenizer, and `BatchGenerator`;
- accept sync and async requests through one scheduler;
- stream per-request token outputs back to callers;
- preserve per-request generation parameters where supported;
- support explicit cancellation and bounded queueing;
- integrate with shared prompt-cache state where feasible;
- surface batching metadata for debugging and benchmarking.

## Why

- This is the only way to expose MLX continuous batching through the public Python and server APIs
  without duplicating inference paths.
- Apple Silicon throughput under load depends much more on iteration-level batching than on naive
  thread concurrency.
- A shared runtime can also become the correct home for MLX prompt-cache persistence across server
  requests.
- The installed `mlx-lm` surface is already strong enough to justify a practical first
  implementation based on `BatchGenerator`, rather than inventing a custom low-level scheduler on
  day one.

## Requirements

- The implementation must support one shared runtime per normalized MLX model/config key, not one
  runtime per provider instance.
- All MLX text execution entry points must be aligned when continuous batching is enabled,
  including `_generate_internal()` paths used by structured-output handlers.
- The feature must be configurable and safe to roll out gradually:
  - initial mode should be opt-in or explicit `auto` for MLX;
  - direct non-batched execution path must remain available as a fallback.
- Both non-streaming and streaming text generation must be supported in the batching path.
- Sync and async callers must see consistent semantics.
- Streaming must support per-request fan-out without cross-talk between responses.
- Queueing must be bounded:
  - max active concurrency;
  - max waiting queue size;
  - request queue timeout.
- Cancellation must work for:
  - async task cancellation;
  - sync iterator close/break;
  - HTTP client disconnect after disconnect signals are plumbed through the server path.
- Fairness must be explicit:
  - active decode streams should not be starved by a flood of new long-prefill requests;
  - the first implementation should prefer a decode-first or bounded-prefill policy if the
    `BatchGenerator` substrate permits it cleanly; otherwise the actual policy must be documented
    and benchmarked.
- Per-request parameters must be treated as new behavior to validate, not assumed parity with the
  current direct MLX path:
  - max output tokens;
  - stop sequences/state machines;
  - sampler settings;
  - logits processors when available.
- Prompt-cache interactions must be defined explicitly, not left accidental.
- Provider metadata should indicate whether a response used continuous batching and how long it
  waited in the queue.

## Proposed Architecture

### 1. Introduce a shared MLX runtime registry

Add a process-wide registry, for example:

- `abstractcore/providers/mlx_runtime.py`

Core concepts:

- `MLXRuntimeKey`
- `SharedMLXRuntime`
- `MLXRequest`
- `MLXStreamHandle`
- `MLXRuntimeRegistry`

The registry owns long-lived runtime instances keyed by normalized model load target and batching
configuration. `MLXProvider` becomes a lightweight front-end that resolves a runtime and submits
requests into it.

This is required because:

- Python users may create several `MLXProvider` instances for the same model;
- the server creates fresh providers per request;
- continuous batching only works if requests for the same model meet inside one scheduler.

### 1a. Define lifecycle and ownership explicitly

The runtime boundary is only safe if ownership is explicit.

Required rules:

- runtime key is the resolved local load target plus execution-affecting runtime config, not only
  the raw model string;
- runtime instances use reference or lease semantics so many provider instances can safely share one
  runtime;
- `MLXProvider.unload_model()` must no longer behave like unconditional model ownership; it must
  either release the caller's lease or explicitly request shared runtime eviction through a safe
  policy;
- teardown behavior must not allow one caller to destroy an active runtime still serving another
  request.

### 2. Use one scheduler thread per shared runtime

The shared runtime should own:

- the loaded model;
- the tokenizer;
- the `BatchGenerator`;
- a waiting queue;
- active request state;
- per-request output queues;
- shared prompt-cache state for that runtime.

The scheduler should run in one dedicated worker thread. That gives a single ownership point for
the MLX runtime and avoids relying on concurrent cross-thread calls into the same model object.

### 3. Build the first implementation on top of `BatchGenerator`

Use `BatchGenerator` as the batching substrate for v1.

Rationale:

- it already supports continuous insertion/removal;
- it already supports per-request `samplers`;
- it already supports per-request `caches`;
- it already supports per-request `logits_processors`;
- it already supports per-request `state_machines`.

That is enough to implement a useful provider path without immediately dropping to custom
`generate_step(...)` plumbing.

The design should still leave room to replace the scheduler core later if:

- `BatchGenerator` turns out insufficient for fairness;
- per-request cache extraction or stop/state-machine handling is not viable through the higher-level
  API;
- advanced structured output needs lower-level control;
- MTP/speculative interactions require tighter scheduling.

V1 should not immediately drop to custom `generate_step(...)`. If a requested feature cannot be
implemented safely through `BatchGenerator`, it should fall back to the direct path instead of
forcing a lower-level scheduler rewrite into the first release.

### 4. Make `_generate_internal()` the real dispatch boundary

The dispatch boundary is not only the public `generate(...)` / `agenerate(...)` methods. Structured
output code already calls `provider._generate_internal(...)` directly.

So the batching-aware routing decision must live at or below the `_generate_internal()` boundary for
plain text MLX execution. Otherwise structured callers will bypass batching entirely or diverge in
behavior.

### 5. Make the provider a request adapter, not the runtime owner

`MLXProvider` should still:

- build the final prompt/messages/tool prompt;
- apply multimodal prompt adaptation where already supported;
- normalize request kwargs;
- perform prompted tool post-processing after completion;
- attach telemetry metadata to responses.

But the actual token generation should happen in the shared runtime.

### 6. Add a small batching config surface

Suggested public/provider parameters:

- `continuous_batching`: `False | True | "auto"`
- `batching_max_concurrency`
- `batching_max_queue_size`
- `batching_queue_timeout_s`
- `batching_prefill_batch_size`
- `batching_prefill_step_size`

Suggested initial behavior:

- Python direct usage: default `False`
- server deployments: allow explicit env-driven default later, but do not silently enable it before
  it is benchmarked and documented
- unsupported batched features must explicitly route to the direct path rather than silently
  partially batching

### 7. Define fairness and backpressure explicitly

The scheduler should not be a black box.

First-pass policy:

- FIFO admission order for waiting requests;
- bounded queue length;
- decode-first or bounded-prefill scheduling to protect active streams;
- `prefill_batch_size` kept separate from decode concurrency;
- timed rejection when queue wait exceeds configured timeout.

This is intentionally simpler than a full priority scheduler, but it is materially better than
"admit everything immediately and hope."

### 8. Treat streaming as first-class

Each submitted request should have its own output channel.

For sync callers:

- blocking iterator over a thread-safe queue

For async callers:

- async iterator bridge over the same scheduler events

For non-streaming callers:

- accumulate output until finish, then return one `GenerateResponse`

The batching runtime should emit token chunks, but the provider-facing behavior must preserve the
existing streaming contract where applicable:

- TTFT metadata expectations;
- thinking-tag stripping;
- tool-call passthrough behavior used by the base streaming stack.

### 9. Integrate prompt-cache behavior conservatively

Prompt-cache semantics must be explicit because the current instance-local store is the wrong
lifetime for server batching, but the full prompt-cache control plane is larger than keyed reuse.

For v1:

- support only keyed prompt-cache reuse for plain-text batched requests;
- if a request specifies `prompt_cache_key`, resolve that cache in runtime-backed batching state;
- active requests must never mutate the same cache object concurrently; clone/fork semantics or
  explicit rejection are required;
- pass the resolved cache into `BatchGenerator.insert(...)`;
- when a request finishes, store the extracted cache back under that key if appropriate.

Prompt-cache control-plane operations beyond keyed reuse should stay on the direct path in v1:

- `update`
- `fork`
- `prepare_modules`
- `save`
- `load`
- `stats`

Do not attempt paged attention or arbitrary cache composition in v1.

### 10. Scope v1 explicitly

Minimal batched v1:

- plain-text non-streaming generation
- plain-text streaming generation
- normal provider entry through `create_llm("mlx", ...)`
- shared runtime reuse across many provider instances
- bounded queueing and cancellation
- keyed prompt-cache reuse for plain text only

Direct-path fallback in v1:

- all `response_model` flows
- native Outlines structured output
- prompted structured-output retry flows
- tools + structured hybrid behavior
- multimodal MLX requests
- any request feature not proven safely through `BatchGenerator`

### 11. Scope structured output conservatively

`BatchGenerator` can accept per-request logits processors and state machines, so there is a credible
path for constrained decoding. But the first rollout should keep scope tight.

Recommended v1 behavior:

- native Outlines structured output: direct path only
- prompted structured output: direct path only
- tool + structured hybrid behavior: direct path only
- state-machine or logits-processor batching: future follow-up after explicit validation

This avoids claiming more than we can test thoroughly.

## Suggested Implementation

1. Add a shared MLX runtime registry and request objects.
2. Define runtime lifecycle, lease/refcount semantics, and shared unload behavior.
3. Make plain-text MLX execution resolve a shared runtime when batching is enabled.
4. Implement non-streaming batched generation first.
5. Add streaming fan-out and cancellation.
6. Add queue telemetry and response metadata.
7. Add keyed prompt-cache reuse for plain-text batched requests only.
8. Add explicit fallback routing for unsupported batched features.
9. Integrate server-side disconnect cancellation only after the signal is available.
10. Update docs and performance tooling.

## Testing Strategy

### Unit tests

Use fake/stub scheduler components so the hard logic is testable without requiring Apple Silicon.

Required unit coverage:

- runtime registry keying and reuse;
- one shared runtime reused across many provider instances;
- queue admission and timeout behavior;
- non-streaming request completion;
- streaming fan-out to the correct request;
- sync and async wrapper behavior;
- cancellation of queued requests;
- cancellation of active streaming requests;
- differing per-request max token settings;
- runtime key normalization: alias model id vs resolved local path;
- shared-runtime unload semantics under multiple provider users;
- prompt-cache lookup/store behavior in shared runtime state;
- concurrent use of the same `prompt_cache_key` with clone/fork isolation or explicit rejection;
- direct fallback for structured-output and multimodal requests;
- fallback to direct path for unsupported batched features.

### Integration tests

Add gated real-MLX tests for Apple Silicon environments only.

Suggested gates:

- env flag similar to `ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1`
- skip cleanly on non-macOS or when MLX dependencies/models are unavailable

Required integration coverage:

- two or more concurrent requests share one runtime;
- batched MLX provider path returns correct text for simple prompts;
- streaming works for two concurrent requests without chunk cross-talk;
- cancellation of one request does not poison the other active requests;
- server path creates fresh providers but still reuses the same shared runtime;
- explicit `continuous_batching=False` preserves direct execution path;
- singleton-provider endpoint path remains correct.

### Server tests

Add focused server tests that:

- verify multiple requests to the MLX provider reuse the same shared runtime key;
- verify batching metadata appears in response metadata when enabled;
- verify per-request overrides do not leak across concurrent requests;
- add disconnect cleanup tests only after request disconnect signals are actually propagated.

### Benchmarking

Benchmarking is mandatory before enabling this broadly.

Keep two benchmark layers:

1. Raw MLX runtime benchmark
   - continue using `examples/performance/mlx_concurrency_benchmark.py`
   - this remains the substrate/best-case benchmark

2. Provider-path benchmark
   - add a new benchmark that runs through `create_llm("mlx", ..., continuous_batching=...)`
   - measure the real provider path, not only `BatchGenerator` directly

Benchmark matrix:

- at least one small MLX model
- at least one mid-size Qwen model
- concurrency sweep: `1 2 4 8 16 32` at minimum
- both streaming and non-streaming
- fixed prompt set for reproducibility
- mixed prompt workloads with short and long prefills to stress fairness claims

Metrics:

- total throughput tokens/sec
- average and p90 TTFT
- average and p90 per-query tokens/sec
- queue wait time
- peak memory
- request failure/cancellation count

Acceptance gates before promotion to default use:

- compare three paths on the same fixed prompt set and model:
  - raw `BatchGenerator` benchmark substrate
  - current provider baseline with `continuous_batching=False`
  - new provider path with `continuous_batching=True`
- at concurrency `>=4`, the batched provider path must materially outperform the current provider
  baseline on total output tokens/sec;
- at concurrency `1`, the batched path must not show a meaningful TTFT regression versus direct
  execution;
- scheduler must show no cross-request output corruption in streaming tests and preserve first-chunk
  timing metadata shape;
- repeated cancellation runs must leave no stuck queued requests and no poisoned shared runtime;
- provider-path performance should land reasonably close to the raw `BatchGenerator` substrate.

## Scope

- In-process continuous batching for `MLXProvider`
- Shared runtime registry
- Non-streaming and streaming text generation
- Server reuse through fresh provider instances
- Shared keyed prompt-cache reuse for plain text
- Benchmarks and docs

## Non-Goals

- Do not implement a general batching framework for every local provider in this item.
- Do not promise multimodal MLX batching in the first implementation.
- Do not promise structured output in batched mode in the first implementation.
- Do not implement paged attention or a custom KV memory manager in v1.
- Do not replace external MLX-serving backends; this item is about the in-process MLX provider.
- Do not silently enable batching for all MLX usage before benchmarks justify it.
- Do not promise full prompt-cache control-plane parity in the batched runtime in v1.

## Dependencies And Related Tasks

- `abstractcore/providers/mlx_provider.py`
- `abstractcore/providers/base.py`
- `abstractcore/core/factory.py`
- `abstractcore/server/app.py`
- `examples/performance/mlx_concurrency_benchmark.py`
- `docs/concurrency.md`
- `docs/backlog/proposed/2026-05-18_native-mtp-and-speculative-decoding-support.md`

Related future work:

- MLX speculative decoding through draft models
- Native MTP support where the runtime actually supports it
- broader local-provider batching abstractions if MLX proves the pattern out

## Expected Outcomes

- AbstractCore exposes real MLX continuous batching through the normal provider API.
- The server benefits from batching even though it creates fresh provider instances per request.
- Concurrent Apple Silicon workloads see materially better throughput than naive per-request
  execution.
- Batching behavior, queue delay, and fallback paths become observable and testable.
- Plain-text keyed prompt-cache reuse becomes more coherent for server-side MLX usage.

## Validation

- Unit tests for shared runtime logic and request isolation
- Gated real-MLX integration tests on Apple Silicon
- Provider-path benchmarks checked against a fixed prompt set
- Documentation updated to distinguish:
  - direct MLX execution
  - async thread fallback
  - continuous batching mode

## Progress Checklist

- [ ] Define shared MLX runtime registry and request objects.
- [ ] Define runtime ownership and unload semantics.
- [ ] Add batching config surface to MLX provider.
- [ ] Implement non-streaming batched generation.
- [ ] Implement streaming fan-out.
- [ ] Implement cancellation and queue timeout handling.
- [ ] Integrate keyed prompt-cache reuse for plain text.
- [ ] Add response metadata and telemetry.
- [ ] Add unit tests.
- [ ] Add gated real-MLX integration tests.
- [ ] Add provider-path benchmarks.
- [ ] Update docs.

## Guidance For The Implementing Agent

Do not start by trying to make the current per-instance `MLXProvider` secretly thread-safe. That
would miss the actual architecture problem. Start by introducing a shared runtime boundary and prove
that:

- many provider instances resolve to one runtime for the same model;
- the scheduler can serve concurrent requests safely;
- the provider and server paths remain behaviorally aligned.

Use `BatchGenerator` first because the installed runtime already supports the key per-request hooks.
Drop lower to custom step scheduling only if tests or benchmarks prove that `BatchGenerator` is not
enough for fairness, constrained decoding, or cache behavior.
