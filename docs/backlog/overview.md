# Backlog Overview

## Summary

This backlog is the durable planning record for AbstractCore package work. It should reflect the
codebase as it exists today, not what earlier notes assumed.

Use this file as the entry point for planning status, recommended next work, and lifecycle counts.

## Counts

- Planned: 12
- Proposed: 9
- Completed: 11
- Deprecated: 1
- Recurrent: 0

## Next recommended work

1. `planned/2026-05-18_mlx-provider-continuous-batching.md`
   The durable bloc cache contract is now provider-wide for supported local backends. The next
   high-impact local-runtime work is batching/scheduler safety for MLX.
2. `planned/2026-03-30_llama-cpp-python_expose_chat_template_kwargs.md`
   GGUF durable bloc artifacts remain exact-renderer gated. Upstream template-kwargs support is
   still useful for clean reasoning/template control.
3. `planned/2026-03-30_qwen3-5_lfm2_nemotron_capabilities_and_thinking_controls.md`
   Keep newer model-family reasoning controls aligned with cache preparation and generation.

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
| `proposed/2026-05-20_composable-prompt-cache-recipes-for-immutable-memory-clusters.md` | Keep proposed until real workloads show stable superbloc reuse beyond single-bloc caches and the work is narrowed to one deterministic exact-prefix recipe per target backend. |
| `proposed/2026-05-20_audio-capability-matrix-consumption-from-abstractvoice.md` | Promote when Core needs truthful audio feature filtering, validation, or UI behavior beyond legacy `controls` booleans. |
| `proposed/0792_hf_transformers_metal_quantization_on_apple.md` | Promote after `Qwen/Qwen3.5-4B` proves MPS `MetalConfig(bits=4)` load/generation/cache correctness with optional `kernels`, clear failure modes, and measured memory/performance deltas. |
| `proposed/0793_public_prompt_cache_persistence_control_plane.md` | Keep proposed as an operator-only decision item. App-facing durable reuse should first ship through Runtime/Gateway exposure of `/acore/blocs/kv/*` and `prompt_cache_binding`; only promote this if hot local snapshot restore is still needed after that. |

## Completed ledger

| Original path | Final path | Completed | Outcome | Comment | Key validation |
| --- | --- | --- | --- | --- | --- |
| `planned/2026-05-06_unified-multimodal-generate-api.md` | `completed/2026-05-06_unified-multimodal-generate-api.md` | 2026-05-06 | Done | Unified multimodal generation API landed. | Backlog completion report + targeted tests recorded in item. |
| `planned/2026-05-07_public-output-selector-contract.md` | `completed/2026-05-07_public-output-selector-contract.md` | 2026-05-07 | Done | Public output-selector helpers exposed for runtimes. | `tests/test_output_specs.py` and related validation recorded in item. |
| `planned/2026-05-07_task-only-text-generation-output-normalization.md` | `completed/2026-05-07_task-only-text-generation-output-normalization.md` | 2026-05-07 | Done | Task-only text-generation selectors normalized correctly. | Item records targeted pytest, compile, lint, and package checks. |
| `planned/2026-05-08_capability_plugin_catalog_discovery_routes.md` | `completed/2026-05-08_capability_plugin_catalog_discovery_routes.md` | 2026-05-08 | Done | Capability plugin catalog routes added. | Completion report in item. |
| `planned/2026-05-08_core_install_profiles_and_gateway_config_boundary.md` | `completed/2026-05-08_core_install_profiles_and_gateway_config_boundary.md` | 2026-05-08 | Done | Install-profile and gateway-config boundaries clarified. | Completion report in item. |
| `planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md` | `completed/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md` | 2026-05-19 | Done | MLX exact bloc artifact compiler/loader is the completed baseline for provider parity work. | `tests/test_bloc_kv.py`, endpoint tests, and loaded-runtime bloc/cache tests. |
| `planned/2026-05-19_generalize_acore_models_residency.md` | `completed/2026-05-19_generalize_acore_models_residency.md` | 2026-05-19 | Done | Task-aware `/acore/models/*` residency control plane landed. | Item records targeted residency test suites and docs updates. |
| `planned/2026-05-20_public_local_vision_cache_catalog_helper.md` | `completed/2026-05-20_public_local_vision_cache_catalog_helper.md` | 2026-05-20 | Done | Public non-server local vision cache catalog helper landed. | Item records focused Core/Runtime pytest and compile validation. |
| `planned/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md` | `completed/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md` | 2026-05-20 | Done | Unified public durable bloc artifact API, strict optional binding, debug proof payloads, and ADR 0007 landed. | `pytest -q`; focused bloc/cache/server tests; real MLX, HF transformers, and HF GGUF smoke proofs. |
| `planned/2026-05-20_hf-transformers-bloc-kv-artifact-compiler-loader.md` | `completed/2026-05-20_hf-transformers-bloc-kv-artifact-compiler-loader.md` | 2026-05-20 | Done | HuggingFace transformers exact bloc artifacts now use the unified API and `.safetensors` provider artifact format. | Focused unit tests; real `Qwen/Qwen3.5-4B` proof with 2.54s -> 0.82s processing-phase speedup and correct cached answer. |
| `planned/2026-05-20_hf-gguf-bloc-kv-artifact-compiler-loader.md` | `completed/2026-05-20_hf-gguf-bloc-kv-artifact-compiler-loader.md` | 2026-05-20 | Done | HuggingFace GGUF exact-renderer paths now use the unified API and `.npz` artifact format. | Focused unit tests; real Qwen3-4B-Instruct GGUF proof with 1.37s -> 0.17s processing-phase speedup and correct cached answer. |

## Deprecated ledger

| Item | Reason |
| --- | --- |
| `deprecated/2026-05-20_transformers-and-gguf-prompt-cache-parity-for-exact-blocs-and-superblocs.md` | Superseded by completed unified API, transformers, and GGUF provider-specific items. |

## Adding or updating work

- Inspect code and current docs before writing or revising backlog text.
- Link governing ADRs explicitly for architecture-significant work.
- Prefer `proposed/` for uncertain ideas and `planned/` only for committed implementation work.
- When a planned item completes, move it to `completed/`, append a completion report, and update
  this overview in the same pass.
