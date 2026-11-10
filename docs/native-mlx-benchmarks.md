# Native MLX benchmarks: Qwen3.8 on M5 Max

This profile compares AbstractCore's opt-in [native MLX runtime](native-mlx-runtime.md)
with installed oMLX 0.6.4 on two quantized Qwen checkpoints. Use it to select an
execution policy, not to rank model quality or predict every workload.
See [Architecture](architecture.md#native-mlx-execution-ownership) for ownership
and request flow, and [speculative decoding](speculative-decoding.md) for MTP setup.

## Hardware, artifacts and method

Measurements: 2026-09-20, MacBook Pro 14-inch, Apple M5 Max, 128 GiB unified
memory, macOS 26.3, AC power. Only one model/server was loaded at a time.

| Artifact | Revision | Quantization / MTP |
| --- | --- | --- |
| `mlx-community/Qwen3.8-27B-4bit` | `3e6447f082e89cc7f0bc6e5441afd38dfce760ff` | 4-bit, group 64; ordinary target weights |
| `mlx-community/Qwen3.8-27B-MTP-4bit` | `b643c01b6d3b094e325edb6ebd832e16c486c575` | Matching separate trained 4-bit head |
| `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` | `2615fc0e976e65c2f3b55daca3a948f1cdc5b9f8` | Mixed-bit oQ4e, embedded head, PLE offload enabled |

These are MLX artifacts, not GGUF `IQ4_XS`. Flash uses a 4-bit default with
higher-precision module overrides; it is not uniformly 4-bit.

| Stack | Measured versions |
| --- | --- |
| AbstractCore native | AbstractCore 2.13.41; MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.7.1, Transformers 5.17.0 |
| oMLX | 0.6.4; MLX 0.32.0, mlx-lm 0.31.3 (`ab1806e8`), mlx-vlm 0.6.3 (`78b96eb`), Transformers 5.12.1 |

This is an installed-stack comparison, not an isolated framework-overhead
experiment. oMLX is a separate comparator, not an AbstractCore dependency.

Main profile: concurrency 1/2/4, MTP off/depth 2, 256 output tokens per request,
temperature 0, top-p 1, seed 6330, thinking disabled and prefix caching disabled.
Each condition used a fresh server, excluded warmup and two measured bursts.
At concurrency one, there is one code and one prose observation, not repeated
trials of each prompt. GPU-idle intervals were 90 seconds after server unload
and 60 seconds after warmup and before each measured burst. Cooldowns reduce
thermal carryover but do not prove absence of hardware throttling.

**Throughput** is total actual output tokens divided by full HTTP burst wall
time, including prompt evaluation and transport. It is not decode-only tokens/s.
**TTFT** is time to first visible client text, including server chunk buffering.
Two bursts do not establish statistical significance or reliable p95 latency.
Per-condition values are available in the
[measurement CSV](assets/native_mlx/20260920-m5max.csv).

The 27B throughput profile preceded final lifecycle/async/Flash compatibility
changes; source-path inspection found no material change to its timed decoder
or admission path. It is not a byte-identical final-source timing. Later
functional and HTTP correctness coverage includes those changes.

## Main throughput comparison

Aggregate output tokens/s; C is simultaneous client requests.

| Model | Requested MTP | C | AbstractCore | oMLX | Actual execution |
| --- | --- | ---: | ---: | ---: | --- |
| 27B | Off | 1 | 30.99 | 31.33 | Target decoding |
| 27B | Depth 2 | 1 | 49.36 | 51.69 | Both use MTP |
| 27B | Off | 2 | 55.56 | 56.74 | Target batching |
| 27B | Depth 2 | 2 | 45.87 | 57.08 | Native MTP cohort; oMLX target-only fallback |
| 27B | Off | 4 | 81.83 | 84.11 | Target batching |
| 27B | Depth 2 | 4 | 42.38 | 83.83 | Native MTP cohort; oMLX target-only fallback |
| Flash | Off | 1 | 28.20 | 35.14 | Target decoding |
| Flash | Depth 2 | 1 | 42.19 | 42.09 | Both use MTP; fixed versus adaptive depth |
| Flash | Off | 2 | 51.82 | 51.14 | Target batching |
| Flash | Depth 2 | 2 | 52.83 | 51.42 | Native MTP cohort; oMLX target batching |
| Flash | Off | 4 | 76.93 | 81.61 | Target batching |
| Flash | Depth 2 | 4 | 69.68 | 80.67 | Native MTP cohort; oMLX target batching |

Native singleton MTP improves its own off throughput by about **1.59× on 27B**
and **1.50× on Flash**. Flash singleton MTP is essentially tied across stacks;
the code/prose ranking reverses. Small native C2 advantages are descriptive,
not established statistical wins. Target-only batching is the faster native
C4 policy on both models.

oMLX's concurrent 27B requests fall back on scheduler contention. Its default
Flash multi-row policy also uses ordinary target batching; the main-profile
singleton tails show zero draft cycles. Those enabled rows are not equivalent
to native batched MTP. Native response diagnostics confirm actual batch sizes
2/4 and MTP execution in the corresponding enabled rows.

For target-only C1/C2/C4, median TTFT was 417/550/815 ms on native 27B versus
489/661/1067 ms on oMLX. Flash was 1229/1134/1498 ms versus 936/1153/1600 ms.
Lower client TTFT in a condition does not imply a faster token-generation kernel.

## Explicit Flash MTP under concurrency

oMLX's optional `OMLX_MTP_ROWWISE_BATCH=1` policy was measured separately.
Each measured burst had real multi-row activation and positive per-depth draft
counts for every measured request; warmup and depth-zero probes were excluded.

| C | Native MTP cohort | oMLX forced row-wise MTP | oMLX default enabled setting, target batching |
| ---: | ---: | ---: | ---: |
| 2 | 52.83 | 39.20 | 51.42 |
| 4 | 69.68 | 39.02 | 80.67 |

Native is approximately **35% faster at C2 and 79% faster at C4** than the
forced row-wise MTP policy in this sample. Its compatible cohort shares batched
target verification; oMLX's opt-in implementation runs backbone work per row.
This architectural difference is relevant, but versions, generated tokens and
fixed/adaptive depth also differ, so it does not isolate the cause of the gap.
The result does not establish superiority over oMLX's faster default C4 policy.

## Staggered arrivals

Two 256-token requests, with the second submitted 250 ms later. One cooled
burst per condition; each TTFT is measured from that request's own submission.

| Model | MTP | Native tokens/s | oMLX tokens/s | Native TTFT first / second (s) | oMLX TTFT first / second (s) |
| --- | --- | ---: | ---: | --- | --- |
| 27B | Off | 55.16 | 55.44 | 0.650 / 0.332 | 0.483 / 0.689 |
| 27B | Depth 2 | 49.10 | 37.67 | 0.394 / 4.513 | 0.469 / 0.637 |
| Flash | Off | 51.13 | 52.84 | 1.161 / 0.809 | 1.075 / 0.823 |
| Flash | Depth 2 | 47.02 | 51.93 | 0.843 / 4.975 | 1.122 / 0.870 |

Native target-only requests join active decoding. Native MTP requests instead
run as successive singleton cohorts in this profile: the later request waits.
For 27B, oMLX uses MTP for the first request and target decoding for the second;
native's roughly 30% aggregate advantage comes with substantially worse second
TTFT. For Flash, oMLX target batching is faster and more responsive here.
Target-only continuous admission is the more responsive tested native policy
for staggered multi-client service.

## Vision and functional coverage

**AbstractCore vision worked with MTP off and on on both selected models.**
The HTTP profile swapped red/blue image halves and required the answer to
follow the pixels. Both color arrangements passed for every engine/model/MTP
combination; enabled requests had actual speculative execution evidence.

| Model | AbstractCore off | AbstractCore depth 2 | oMLX off | oMLX depth 2 |
| --- | --- | --- | --- | --- |
| 27B | Pass | Pass; actual MTP | Pass | Pass; actual MTP |
| Flash-Next | Pass | Pass; actual MTP | Pass | Pass; actual MTP |

All 16 image answers were correct across eight conditions. These are image
delivery/functionality checks, not OCR, general visual reasoning or concurrent
multimodal throughput measurements. Prompt construction, buffered short-output
timings and token accounting differ between stacks; no vision speed ranking is
derived from them.

The two-model native coverage additionally includes selectable depths 1/2/3,
four-row MTP with different completion lengths, sync/async/stream behavior,
stop strings, sampling controls, explicit tool dispatch, prompted structured
output, cancellation, shared-owner lifecycle and prefix-cache restart/clear.
The public concurrent arithmetic profile produced all 32 required answers
correctly across eight conditions, with actual four-row native batches.
Overall scope: 36 native functional phases and 50 comparison conditions; the
scoped CPU suite passed 610 tests with two opt-in skips. This is not a
full-project, all-provider or all-model certification.

## Memory, cache and serving policy

Native lifecycle probes observed about **15.18 GiB MLX active allocation for
27B** and **69.39 GiB for Flash** before final unload. Both returned to about
1 KiB active allocation and zero allocator cache after final-owner release.
These are workload-specific allocations, not maximum-context capacity estimates.
Flash's roughly 50–55 GiB process high-water RSS across both stacks is a
different measure, not its total unified-memory cost or a native RAM advantage.
Native 27B MTP-off retains the head; oMLX off does not attach it.

Both native models reused 732 of 733 prefix tokens after process restart;
changed-suffix requests reused 512 tokens and answered the changed question.
Explicit clear restored cold-cache behavior. Enable full-history keyed reuse
and optional private SSD storage as described in [native caching](native-mlx-runtime.md#bounded-ram-and-optional-ssd-prefix-reuse).

AbstractCore provides model load/unload controls, shared native weights and
bounded caches. These differ from oMLX's automatic memory-pressure multi-model
LRU eviction and idle-TTL unloading; AbstractCore's managed TTL is advisory.
The [historical upstream batching profile](concurrency.md) measures a different
model/hardware/API path and must not be pooled with these results.

## Choosing a configuration and interpreting limits

- For a lower-memory, lower-latency option on this machine, use 27B. This profile
  does not compare answer quality between 27B and Flash.
- Use depth-2 MTP for a single interactive request. Prefer target-only continuous
  batching for concurrent/staggered clients, then measure your workload.
- Keep Flash PLE offload enabled on a busy 128-GiB machine; do not load both
  models simultaneously without checking available memory.
- Native MTP cohorts cannot admit late requests. Sampled execution is exclusive.
  Unsupported MTP logits penalties are refused or explicitly fall back when
  acceleration is optional.
- Native Outlines and legacy manual cache artifact operations are unsupported.
  Prompted structured output may retry; the Flash structured check used three
  backend completions for one validated result.
- Native endpoint inline images are bounded and current-turn-only. Remote/file
  URLs and image-bearing history are not supported by that endpoint path.
- The shared speculation contract is portable; HF/GGUF MTP tensor execution is
  not provided by this upgrade. Upstream private interfaces remain version-sensitive.
- Long-context scaling, maximum concurrency, model-quality equivalence and
  sustained-load tail latency were not measured. Finite-precision kernels and
  generation policies can produce different text even at temperature zero.

Installed oMLX uses additional routed gate/up, GDN and hyper-connection kernel
optimizations on eligible target-only Flash paths. Native fusion at the actual
decode helper is a possible optimization direction, not a delivered or measured
speedup. Both stacks use PLE memory mapping; full PLE residency is not established
as a remedy for the target-only gap.
