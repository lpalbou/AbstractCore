# Native MLX concurrency and prefix caching

The MLX provider serves concurrent Qwen3.8-27B and Qwen3.8-Flash-Next requests
inside your Python process with `mlx_batching=True`. Execution uses AbstractCore,
MLX, mlx-lm and mlx-vlm; it does not import or run oMLX. See
[speculative decoding](speculative-decoding.md) for checkpoints and MTP controls.
For component ownership and request flow, see
[Architecture](architecture.md#native-mlx-execution-ownership). The
[M5 Max measurements](native-mlx-benchmarks.md) compare both models with oMLX,
including vision and concurrent/staggered requests.

For a source checkout, install the Apple provider stack with
`pip install -e ".[mlx]"`. Released packages use `pip install -U "abstractcore[mlx]"`;
see the [changelog](../CHANGELOG.md) for release availability.
The native runtime requires MLX ≥0.32.2, mlx-lm ≥0.31.3 and mlx-vlm ≥0.7.1,<0.8.0.
Download target and head weights explicitly before loading; AbstractCore's model
cache is offline-first. See [model downloads](api.md#model-downloads-download_model-optional).

```python
from concurrent.futures import ThreadPoolExecutor
from abstractcore import create_llm

llm = create_llm(
    "mlx", model="mlx-community/Qwen3.8-27B-4bit",
    speculation={"mode": "native_mtp", "num_draft_tokens": 2,
                 "require_acceleration": True},
    mlx_batching=True, mlx_max_batch_size=4,
)

def answer(question):
    return llm.generate(question, thinking=False, temperature=0,
                        max_output_tokens=256)

with ThreadPoolExecutor(max_workers=2) as pool:
    answers = list(pool.map(answer, ["Explain TTL caches.", "Explain LRU caches."]))
for answer in answers:
    print(answer.content, answer.metadata["execution"])
llm.unload_model(llm.model)
```

For Flash-Next, use `model="Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"`; its MTP head
is embedded. Keep the default PLE offload on 128 GiB machines. Both large models
should not be loaded simultaneously without checking available memory.

## Vision with MTP off or on

Both documented checkpoints accept native image input with speculation disabled
or enabled. The same vision encoder supplies image embeddings to the target;
MTP changes token generation, not image delivery. The 27B target uses the matching
`mlx-community/Qwen3.8-27B-MTP-4bit` head; Flash-Next embeds its head.

```python
from abstractcore import create_llm

llm = create_llm(
    "mlx", model="mlx-community/Qwen3.8-27B-4bit", mlx_batching=True,
    speculation={"mode": "native_mtp", "num_draft_tokens": 2},
)
# The target/head remain resident for both requests.
for setting in (False, {"mode": "native_mtp", "num_draft_tokens": 2,
                        "require_acceleration": True}):
    response = llm.generate(
        "Describe the left and right halves of this image.",
        media=["halves.png"], speculation=setting,
        thinking=False, temperature=0, max_output_tokens=64,
    )
    print(response.content)
    print(response.metadata["speculation"]["used"])
llm.unload_model(llm.model)
```

Swapped red/blue image checks passed on both models with MTP off/on; see
[vision measurements](native-mlx-benchmarks.md#vision-and-functional-coverage).
These establish image delivery and behavior for those inputs, not general OCR
accuracy, image-history compatibility or concurrent multimodal throughput.

## Scheduling and request controls

| Request | Execution policy |
| --- | --- |
| Compatible greedy target-only requests (`speculation=False`) | Continuous admission into available batch slots |
| Compatible greedy MTP requests with the same depth | Fixed cohort; later requests wait for another cohort |
| Sampled or incompatible requests | Exclusive execution with their requested controls |

`num_draft_tokens` counts proposals, excluding the target seed token, on both
models. Change depth or disable MTP per request without reloading. MTP does not
support non-neutral logits penalties or logit bias: strict acceleration refuses
them; optional acceleration reports an explicit unaccelerated outcome. Stop
strings work with the scheduled runtime.

Benchmark MTP against `speculation=False` for your workload. Speculative cohorts
can improve a single request while reducing aggregate throughput under load;
target-only continuous batching can be the better choice for concurrent clients.
Change the request control without unloading the resident target/head pair.

Batch compatibility also depends on prefix-cache manager identity. Requests using
different cache managers are separate scheduling groups; concurrent submission
does not guarantee a shared batch across independently keyed application sessions.
Keep cache isolation intact and measure the session/key pattern your application uses.

### AbstractRuntime integration

Use the existing `mlx` provider with scheduler/head options at construction.
AbstractRuntime's local client consults the loaded instance's concurrency capability;
ordinary MLX instances retain serialized generation. Per-call `params.speculation`
accepts `False` or a native-MTP control object, and `params.thinking` accepts boolean
on/off or a reasoning level. The remote client forwards these controls to Core HTTP
and preserves actual execution/speculation metadata. Head loading and scheduler
configuration remain load-time choices; forwarding a request does not load a head.
These integration changes are unreleased; update both Core and Runtime together.
Applications can inherit the configured MTP policy or override it per run or node;
see the default policy below.

### MTP defaults and application overrides

A fresh Core configuration seeds native MTP at draft depth **2** for compatible
models. Existing configuration files are not reseeded. Depth 2 is a starting
policy, not a guarantee of best performance for every model or workload.

The policy is stored at `capability_defaults.routes.input.text.options.speculation`;
the `output.text` route exposes the same setting. Core's console-TUI and Gateway's
web/TUI capability-default editors edit this Core-owned configuration. The CLI
also preserves other route options when changing only MTP:

```bash
abstractcore config set-default output.text --speculation 2
abstractcore config set-default output.text --speculation off
abstractcore config set-default output.text --speculation inherit
```

`inherit` removes the configured policy; it does not restore a fresh-install seed.
With no policy or explicit constructor/request setting, speculation is off.
An inherited policy prepares an available matching cached head during model load;
it never downloads weights. Unsupported backends remain non-speculative. A model
already loaded without a head requires explicit provisioning/reloading before
an application can enable MTP.

Flow, Assistant and Code leave `speculation` unset when using the Gateway default.
Their Off choice sends `false`; an explicit depth sends a native-MTP object with
`num_draft_tokens`. Explicit call/node settings win over inherited run settings,
which win over the provider's default. An explicit constructor default wins over
the Core policy. Changes to depth or Off do not unload a prepared target/head.
Strict requests require MTP execution, not a measured speedup.

`get_execution_capabilities(model_name, provider=..., instance=...)` in
`abstractcore.providers.speculation`, and `GET /v1/models/execution-capabilities`,
describe backend support, instance readiness, available depth choices and the
effective default without loading a model. A missing readiness value is unknown,
not proof that the head is ready. Inspect each response's `speculation.used` for
what actually executed; unsupported explicit requests fail when strict or return
a named non-speculative outcome when fallback is allowed.

`metadata["execution"]` reports `mode`, observed `peak_batch_size`, queue time
and time to first backend token. Multi-request MTP counters are shared cohort
diagnostics, not independent per-request token totals; inspect `accounting` and
`counter_semantics` before aggregating them. Use response `usage` for actual
prompt/completion tokens. Client first-token latency also includes transport.

Defaults: four batch slots, 32 waiting requests, a 10 ms collection window,
a 120-second queue timeout and 256 buffered output records per request.
Controls: `mlx_max_batch_size`, `mlx_max_queue_size`, `mlx_batch_wait_ms`,
`mlx_queue_timeout_s`, and `mlx_output_queue_size`. Full queues, expired requests
and stalled stream consumers raise actionable errors. Close abandoned streams,
including `await stream.aclose()` for async streams. Cancellation suppresses
output immediately; GPU state is reclaimed at the next safe backend boundary,
which can follow an in-progress prefill.

Providers sharing a target/head pair share weights and one execution owner.
They must agree on scheduler/cache settings. Releasing an idle provider does
not unload another provider's model. Unloading an owner with active requests is
refused. Direct and scheduled execution cannot share a resident native session.
Garbage collection retires a scheduled provider's owner without blocking on an
in-flight request; final cleanup occurs at a safe worker boundary. Prefer explicit
unload after finishing or closing streams when you need deterministic SSD flush
and error reporting. A live-worker shutdown timeout keeps resources owned for a
later retry. A flush failure after the worker stops is reported, but does not
leave a closed model permanently resident.

## Bounded RAM and optional SSD prefix reuse

Use an explicit `prompt_cache_key` and the complete conversation in `messages`
(`messages=[]` for standalone prompts). Native prefix reuse does not append
hidden history. Requests without a key do not use the automatic cache.

```python
llm = create_llm(
    "mlx", model="mlx-community/Qwen3.8-27B-4bit",
    speculation={"mode": "native_mtp"}, mlx_batching=True,
    mlx_cache_disk_path="./private-mlx-cache",
    mlx_cache_disk_max_gb=4, mlx_cache_memory_max_gb=8,  # RAM cap is optional
)
result = llm.generate(long_prompt, messages=[], prompt_cache_key="conversation-1",
                      thinking=False, temperature=0, max_output_tokens=128)
print(result.usage["cached_input_tokens"])
print(llm.get_prompt_cache_stats())
llm.unload_model(llm.model)  # Flushes SSD storage for compatible future processes.
```

The RAM budget defaults to mlx-vlm's machine-relative sizing,
`min(8 GiB, recommended working set / 10)` (8 GiB on a 128 GB Mac, about
1.2 GiB on 16 GB), and it caps EACH retained snapshot: a snapshot larger than
the budget is not retained and the next turn re-prefills the whole
conversation, which the response reports as `prompt_cache.degraded_reason`
with the call's `prompt_cache.apc` store/skip counters. A hybrid model's
snapshot grows roughly 27-50 KB per prompt token, so set
`mlx_cache_memory_max_gb` only to a value larger than your longest
conversation's snapshot. Two recurrent-state checkpoints are kept. SSD storage
is opt-in, defaulting to a 4 GiB budget. One runtime shares budgets across keys and
depths. Cache identity includes weights, tokenizer/configuration, engine versions,
execution layout, MTP depth and image content; changes can prevent reuse.
A private namespace has one live writer; concurrent processes need separate
paths. `mlx_cache_scope` partitions local identity, not authenticated tenants.
Treat local checkpoint directories as immutable. Hub snapshots carry revision
identity; arbitrary local checkpoints use bounded file/header fingerprints,
not a full hash of every tensor value. Clear caches after manually editing weights.

`prompt_cache_clear()` invalidates owned RAM and SSD cache entries. Key-specific
clear also clears the whole shared native namespace and warns about its scope.
Weights and other namespaces are untouched. Native APC is separate from the
ordinary MLX durable append-cache artifact format: manual prefill, append, fork,
save and load are unsupported. See [prompt caching](prompt-caching.md) and
[model residency](memory-management.md).

## HTTP serving

`abstractcore.endpoint.app.create_app(provider_instance=llm)` serves the same
provider. The AbstractCore server accepts these constructor controls inside
`options` on `/acore/models/load`. Load before sending concurrent chat requests;
changing resident execution options requires unloading first. Both chat APIs
accept per-request `speculation` or `false`. Native chat responses expose
diagnostics under the additive `abstractcore` field, including terminal streaming
events. Cache and lifecycle controls remain serialized through the owner.

The single-model native endpoint accepts inline base64 PNG/JPEG/WebP/GIF
`image_url` parts in the final user message, up to 10 MiB per image, 32 MiB total
and 16 images. Remote/file image URLs, unsupported part types and image-bearing
history are rejected explicitly. Supply complete text history plus current-turn
images, or use the Python `media=` API for local files. Image detail metadata does
not override the native processor's resizing policy.

Use one server process per resident model session: web workers do not share
an in-process scheduler or weight allocation. These controls do not enable
batching or MTP for Hugging Face/GGUF providers.

## Live phase feedback (prefill vs generation)

A 6,900-token prompt on this lane spends ~1.6 s in prefill before the first
token, then decodes at ~150 tok/s. From outside, both look like "the call has
not returned yet", so a host renders one undifferentiated spinner over two very
different states. Pass a callback and the provider reports the boundary:

```python
def on_phase(event: dict) -> None:
    print(event["phase"], event.get("prompt_tokens"), event.get("generated_tokens"))

llm = create_llm("mlx", model="mlx-community/Qwen3.5-4B-4bit")
llm.generate("", messages=messages, system_prompt=system, tools=tools,
             prompt_cache_key="session:chat", on_progress=on_phase)
```

`on_progress=`, `progress_callback=` and `progress_event_callback=` are accepted
(the first callable wins) and receive plain JSON-safe dicts:

| key | meaning |
| --- | --- |
| `kind` | always `"llm"` — the discriminator that separates these from generated-media progress, which shares the event name downstream |
| `phase` | `"prefill"`, `"generate"` or `"complete"` |
| `prompt_tokens` | total prompt tokens for this call. **Only ever grows**: on a warm cache the backend reports the tokens IT processed (e.g. `1`), which is a narrower measurement of the same prompt, not news |
| `cached_tokens` / `fed_tokens` | prompt tokens restored from KV vs actually prefilled |
| `generated_tokens` | tokens sampled so far |
| `ttft_s` | time to first token, set on the first-token event and stable afterwards |
| `tokens_per_second` | generation rate; absent until a second token exists (one token divided by a near-zero interval is not a rate) |
| `prefill_processed_tokens` | `prefill` events only: prompt tokens already in the KV cache — restored tokens count at once, then fed tokens as each chunk is evaluated. Cumulative, monotone, never above `prompt_tokens`. **Absent** when the lane cannot observe its prefill |
| `prefill_tokens_per_second` | `prefill` events only: rate over the tokens NEWLY processed since the first mid-prefill observation |
| `prompt_tokens_per_second`, `elapsed_s`, `event_index`, `final`, `finish_reason` | as named |

Keys with no known value are absent rather than zero: on the native APC lane the
cached/fed split is only known once prefill has started on the worker, so it
appears from the first mid-prefill event on (or the first-token event when the
prompt is too short to observe), while the mlx-lm key-mode lane already knows it
and puts it on the `prefill` event.

**Mid-prefill progress.** After the start event, `prefill` events keep arriving
at the same cadence while the prompt is processed, each carrying
`prefill_processed_tokens` — what a UI renders as
`Prefill · 2,100 / 5,642 tokens (37%)`. Every lane reports from a real
per-chunk observation, never an estimate:

| lane | seam | granularity |
| --- | --- | --- |
| mlx-lm (`mlx_batching=False`, no speculation) | mlx-lm's `prompt_progress_callback(processed, total)` | per `prefill_step_size` chunk (2,048) |
| native runtime, batched (temperature 0) | upstream `PromptProcessingBatch` row counters, read after every `BatchGenerator.next()` | per 256-token chunk |
| native runtime, exclusive (sampling) and in-process mlx-vlm (MTP included) | mlx-vlm's chunked-prefill "Prefill" bar, observed per thread by `mlx_prefill_observer` | per chunk (256 native / 2,048 in-process) |
| HuggingFace transformers | the 2,048-token chunk loop (`ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP`), device-synchronized per chunk | per chunk; a prompt short enough for one forward reports no position |
| HuggingFace llama.cpp (control plane) | `n_batch` slices (512) | per slice; the `create_chat_completion` fallback reports no position |
| HTTP providers (LM Studio, Ollama, llama.cpp server, …) | none — the server does not expose it | no callback, no event |

`NativeRuntime.stream(request, on_prefill_progress=...)` delivers
`NativePrefillProgress` observations on the iterating thread and never yields
them, so a consumer that did not ask sees the stream it saw before. No
`prefill` event is emitted after the first token.

**Cost.** Time-limited, never count-capped (ADR-0026). `prefill`, the
first-token event and the terminal `complete` always fire; every other event
(mid-prefill or decode cadence) is dropped unless at least
`ABSTRACTCORE_PROGRESS_MIN_INTERVAL_S` (default 0.5) seconds have passed since
the previous one, so events flow for the whole call at about two per second.
A 3.7 s answer costs 9 events; a 6 s prefill of 21,066 tokens adds 11. Subscribing also routes the call through `stream_generate` instead of
`generate` — byte-identical output, since mlx-lm's `generate` is exactly
`"".join(r.text for r in stream_generate(...))` — and costs one extra tokenizer
encode to report the prompt size before the GPU work starts. With no subscriber
nothing is computed and nothing changes.

`BaseProvider.supports_text_progress_events()` gates this and defaults to
**False**. For a provider that answers False the host's callback is consumed at
the provider boundary and dropped — it never reaches provider kwargs, so no
strict SDK sees an unknown callable — and no phase is invented. AbstractRuntime
turns each event into one durable `abstract.progress` ledger record; see
AbstractRuntime's `docs/integrations/abstractcore.md`.

## Supported boundaries

- MTP is fixed-cohort scheduling, not late-join continuous batching. Sampled
  execution is exclusive; larger draft depths need workload-specific measurement.
- Native prefix reuse requires full history and an explicit key. Legacy manual
  append/prefill/fork/save/load artifacts are not supported by this native cache.
- Prompted structured output is supported and may validate/retry. The Outlines
  adapter is not supported by this native runtime.
- One process owns a resident session; additional web workers do not share its
  tensors. Native SSD namespaces are private and single-writer.
- Model load/unload controls do not provide automatic multi-model memory-pressure
  LRU eviction; managed model TTL is advisory. See [model residency](memory-management.md).
- Upstream private batching/cache interfaces are version-sensitive. Use the
  documented dependency range and revalidate before changing the execution stack.
