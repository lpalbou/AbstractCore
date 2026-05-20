# Backlog Overview

## Summary

This backlog is the durable planning record for AbstractCore package work. It should reflect the
codebase as it exists today, not what earlier notes assumed.

Use this file as the entry point for planning status, recommended next work, and lifecycle counts.

## Counts

- Planned: 13
- Proposed: 8
- Completed: 7
- Deprecated: 0
- Recurrent: 0

## Next recommended work

1. `planned/2026-05-04_unified-lora-adapter-serving.md`
   The repository already exposes provider-specific vLLM adapter methods and even documents a
   routing story that the shared provider path does not actually implement. This is now the clearest
   abstraction gap between provider-specific capability and first-class Core behavior.
2. `planned/2026-05-04_vllm-base-model-swap-orchestration.md`
   Keep this behind adapter lifecycle work. Base-model swap is a separate admin problem and should
   not be mixed into first-pass adapter support.
3. `planned/2026-05-18_mlx-provider-continuous-batching.md`
   High-value throughput work, but orthogonal to the adapter gap and already well specified.

## Planned ledger

| Item | Notes |
| --- | --- |
| `planned/2026-03-30_llama-cpp-python_expose_chat_template_kwargs.md` | Upstream GGUF thinking-toggle prerequisite for clean Qwen reasoning control. |
| `planned/2026-03-30_qwen3-5_lfm2_nemotron_capabilities_and_thinking_controls.md` | Model-capability and reasoning-control coverage for newer model families. |
| `planned/2026-05-04_unified-lora-adapter-serving.md` | First-class text adapter lifecycle and hot-switch routing across providers. |
| `planned/2026-05-04_vllm-base-model-swap-orchestration.md` | Separate admin flow for switching the served vLLM base model. |
| `planned/2026-05-06_consensus-generate.md` | Consensus-generation orchestration work. |
| `planned/2026-05-06_robust-fallback-generate.md` | Stronger fallback behavior for generation paths. |
| `planned/2026-05-07_multimodal-generation-and-deterministic-inference-cache.md` | Multimodal generation plus deterministic cache behavior. |
| `planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md` | MLX bloc KV compiler/loader support. |
| `planned/2026-05-18_mlx-provider-continuous-batching.md` | Shared MLX runtime and batching scheduler. |
| `planned/788-response.md` | Responses-related planned work (legacy naming retained). |
| `planned/789_server-auth-rate-limits.md` | Server auth and rate-limit controls. |
| `planned/790_server-response-cache.md` | Server response-cache work. |
| `planned/791_server-metrics-otel.md` | Server metrics and OpenTelemetry work. |

## Proposed ledger

| Item | Promotion criteria |
| --- | --- |
| `proposed/2026-05-06_docs-site-publishing-strategy.md` | Promote when docs deployment scope and ownership are committed. |
| `proposed/2026-05-06_remote-prompt-cache-session-parity.md` | Promote when endpoint/session parity becomes blocking for runtime users. |
| `proposed/2026-05-07_runtime-ready-multimodal-generation-abstraction.md` | Promote when runtime/kernel integration work is scheduled. |
| `proposed/2026-05-08_dual-server-docker-image-profiles.md` | Promote when packaging/deployment scope becomes active. |
| `proposed/2026-05-18_native-mtp-and-speculative-decoding-support.md` | Promote when serving backends and benchmarks justify active implementation. |
| `proposed/2026-05-20_composable-prompt-cache-recipes-for-immutable-memory-clusters.md` | Promote when real workloads show stable cluster reuse beyond shelf/bloc caches and the work is narrowed to one deterministic exact-prefix recipe per target backend. |
| `proposed/2026-05-20_exact-bloc-and-shelf-cache-binding-for-external-clients.md` | Promote when MLX bloc-derived `prompt_cache_key` reuse needs an explicit public binding contract beyond current load-time validation. |
| `proposed/2026-05-20_transformers-and-gguf-prompt-cache-parity-for-exact-blocs-and-shelves.md` | Promote when exact bloc/shelf acceleration needs backend parity evidence outside MLX, or when another local backend appears strong enough to justify production support. |

## Completed ledger

| Original path | Final path | Completed | Outcome | Comment | Key validation |
| --- | --- | --- | --- | --- | --- |
| `planned/2026-05-06_unified-multimodal-generate-api.md` | `completed/2026-05-06_unified-multimodal-generate-api.md` | 2026-05-06 | Done | Unified multimodal generation API landed. | Backlog completion report + targeted tests recorded in item. |
| `planned/2026-05-07_public-output-selector-contract.md` | `completed/2026-05-07_public-output-selector-contract.md` | 2026-05-07 | Done | Public output-selector helpers exposed for runtimes. | `tests/test_output_specs.py` and related validation recorded in item. |
| `planned/2026-05-07_task-only-text-generation-output-normalization.md` | `completed/2026-05-07_task-only-text-generation-output-normalization.md` | 2026-05-07 | Done | Task-only text-generation selectors normalized correctly. | Item records targeted pytest, compile, lint, and package checks. |
| `planned/2026-05-08_capability_plugin_catalog_discovery_routes.md` | `completed/2026-05-08_capability_plugin_catalog_discovery_routes.md` | 2026-05-08 | Done | Capability plugin catalog routes added. | Completion report in item. |
| `planned/2026-05-08_core_install_profiles_and_gateway_config_boundary.md` | `completed/2026-05-08_core_install_profiles_and_gateway_config_boundary.md` | 2026-05-08 | Done | Install-profile and gateway-config boundaries clarified. | Completion report in item. |
| `planned/2026-05-19_generalize_acore_models_residency.md` | `completed/2026-05-19_generalize_acore_models_residency.md` | 2026-05-19 | Done | Task-aware `/acore/models/*` residency control plane landed. | Item records targeted residency test suites and docs updates. |
| `planned/2026-05-20_public_local_vision_cache_catalog_helper.md` | `completed/2026-05-20_public_local_vision_cache_catalog_helper.md` | 2026-05-20 | Done | Public non-server local vision cache catalog helper landed. | Item records focused Core/Runtime pytest and compile validation. |

## Adding or updating work

- Inspect code and current docs before writing or revising backlog text.
- Link governing ADRs explicitly for architecture-significant work.
- Prefer `proposed/` for uncertain ideas and `planned/` only for committed implementation work.
- When a planned item completes, move it to `completed/`, append a completion report, and update
  this overview in the same pass.
