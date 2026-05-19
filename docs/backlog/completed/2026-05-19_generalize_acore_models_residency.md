# Completed: Generalize `/acore/models/*` into task-based model residency

## Metadata
- Created: 2026-05-19
- Status: Completed
- Completed: 2026-05-19
- Priority: P1

## Context

AbstractCore Server already exposes:

- `GET /acore/models/loaded`
- `POST /acore/models/load`
- `POST /acore/models/unload`

The route names look modality-neutral, but the current implementation is scoped to local
LLM/provider runtimes:

- loaded entries are stored in `_GATEWAY_LOADED_RUNTIMES`;
- loading calls `create_llm(provider, model=...)`;
- reuse is wired into chat/completions, prompt-cache control-plane routes, and bloc-KV routes;
- the schema accepts `provider`, `model`, `base_url`, and `timeout_s`, with no modality/task field.

Vision and audio are handled through separate capability plugin paths and server endpoints. They have
their own internal caches, but they are not controlled by `/acore/models/*` today. That makes the
public API ambiguous: users can reasonably expect "models" to include image, TTS, and STT models,
while the current route only keeps text-generation providers warm.

## Problem

The current split causes avoidable confusion and cold-start behavior:

- local image generation can reload large image backends even though the selected model is known;
- TTS/STT engine caches exist, but there is no Core-level model residency API for them;
- Flow/Gateway have to reason about text, image, voice, and transcription readiness through
  different mechanisms;
- adding a second, separate warm-model endpoint family would duplicate the existing `/acore/models/*`
  control plane and make the API harder to explain.

## Design Direction

Extend the existing `/acore/models/*` routes instead of introducing a parallel endpoint family.
This is the cleanest API because clients already see these routes as the model residency/control
plane.

Keep the existing text behavior as the backward-compatible default:

```json
{
  "provider": "mlx",
  "model": "mlx-community/Qwen3.6-27B-4bit"
}
```

Add a task-aware request shape. `task` is the primary discriminator because `tts` and `stt` are
tasks, not media modalities:

```json
{
  "task": "text_generation | image_generation | tts | stt | music_generation",
  "provider": "mflux",
  "model": "qwen/qwen3.5-35b-a3b",
  "options": {
    "device": "auto",
    "dtype": null,
    "quantize": 8,
    "voice": null,
    "profile": null,
    "language": null
  },
  "pin": true,
  "ttl_s": 3600
}
```

For backward compatibility, omitted `task` means `text_generation`.

The residency key should include:

- task;
- provider;
- model;
- base URL/API-key fingerprint for remote providers;
- device/dtype/quantization and other backend-shaping options;
- voice/profile/language where they affect loaded TTS/STT state.

The response should describe actual residency, not just catalog availability:

```json
{
  "runtime_id": "...",
  "task": "image_generation",
  "provider": "mflux",
  "model": "qwen/qwen3.5-35b-a3b",
  "state": "configured | loading | resident | failed | unloading",
  "resident": true,
  "loaded_new": true,
  "loaded_at": 0,
  "last_used_at": 0,
  "request_count": 0,
  "pinned": true,
  "isolation": "in_process | worker | remote",
  "health": "ok",
  "error": null
}
```

Use `resident=false` with `state=configured` for remote providers where no local model is actually
loaded. Do not report a remote OpenAI-compatible endpoint as "warm" unless the remote endpoint gives
Core a real loaded-state signal.

## Capability Contract

Core should define a small optional residency protocol for providers and capability plugins:

```python
class ModelResidencyProvider(Protocol):
    def load_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def list_resident_models(self, filters: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]: ...
    def unload_resident_model(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
```

Keep this protocol dictionary-shaped at the plugin boundary to avoid forcing AbstractCore to import
AbstractVoice/AbstractVision types.

Text providers can keep using the current `create_llm(...)` path internally. Vision, voice, audio,
and music should route through `CapabilityRegistry` and plugin-provided residency hooks when present.
Existing provider `unload_model(model_name)` methods can be adapted behind this protocol rather than
exposed directly as the new generic contract.

Server generation routes should then use the same registry:

- chat/responses use warm `text` runtimes when present;
- vision routes use warm `image` backends when present;
- audio speech routes use warm `tts` engines when present;
- audio transcription routes use warm `stt` engines when present.

## Internal Shape

Implement Core residency as one process-local registry with adapters:

- `TextResidencyAdapter`: wraps the existing `_GATEWAY_LOADED_RUNTIMES` behavior.
- `VisionResidencyAdapter`: delegates to the selected vision capability/server backend resolver and
  its existing backend cache/preload behavior.
- `VoiceResidencyAdapter`: delegates to the selected voice/audio capability plugin.
- `MusicResidencyAdapter`: future adapter; not required for the first implementation.

The registry should own filtering, deterministic runtime ids, timestamps, errors, TTL/pin policy,
and JSON response shape. Modality packages should own model-specific loading and unloading.

For Core Server specifically, avoid duplicating current media caches:

- image residency should reuse the same backend instances used by image generation routes;
- audio residency should reuse the same capability-core/VoiceManager instances used by speech and
  transcription routes.

## Workflow Preflight

Gateway/Flow should be able to preflight a workflow by calling the same route once per selected
runtime:

```json
[
  {"task": "text_generation", "provider": "mlx", "model": "mlx-community/..."},
  {"task": "image_generation", "provider": "mflux", "model": "qwen/qwen3.5-35b-a3b"},
  {"task": "tts", "provider": "omnivoice", "model": "default", "options": {"voice": "default"}},
  {"task": "stt", "provider": "faster-whisper", "model": "base"}
]
```

No separate `touch` endpoint is required for v1. Generation/transcription/synthesis calls should
update `last_used_at`. A later lease API can be added only if TTL/pin is not enough.

## Non-Goals

- Do not create a separate `/api/gateway/models/warm` family when `/acore/models/*` can be extended.
- Do not infer provider/model from model ids when the caller selected them explicitly.
- Do not use dummy prompt/image/audio generation as the primary warm contract if a backend can load
  directly.
- Do not make Flow import AbstractVision or AbstractVoice directly for warm-model behavior.
- Do not conflate catalog availability, downloaded weights, and resident in-memory state.
- Do not report remote providers as locally resident without evidence from that remote service.

## Dependencies

- AbstractVision should expose capability-level image residency hooks.
- AbstractVoice should expose capability-level TTS/STT residency hooks.
- Runtime/Gateway can then use this Core contract for workflow preflight, leases, and worker-backed
  local image isolation.

## Success Criteria

- `/acore/models/load` can warm text, image, TTS, and STT models with one documented request shape.
- Existing text-only clients continue to work unchanged.
- `/acore/models/loaded` reports resident/configured state across tasks.
- `/acore/models/unload` can unload by `runtime_id` or by task/provider/model selector.
- Vision/audio server endpoints reuse warm entries instead of silently constructing independent
  instances.
- Gateway/Flow can preflight a workflow's warm-model requirements without modality-specific APIs.

## Validation Ideas

- Unit tests for residency key construction and backward-compatible text requests.
- Server tests that load an image model, call image generation, and observe the same warm entry.
- Server tests that load a TTS/STT engine and observe reuse through audio endpoints.
- Failure tests for unknown modality, unavailable plugin, and unload of missing residency ids.

## Completion Report - 2026-05-19

Implemented in AbstractCore:

- Extended the existing `/acore/models/load`, `/acore/models/loaded`, and
  `/acore/models/unload` routes into a task-aware residency control plane.
- Preserved backward compatibility for text-generation clients:
  - omitted `task` still means `text_generation`;
  - existing `provider`, `model`, `base_url`, and `timeout_s` requests keep using the current
    `_GATEWAY_LOADED_RUNTIMES` path;
  - chat/completion, prompt-cache, and bloc-KV routes continue to reuse loaded text runtimes.
- Added the shared task vocabulary and aliases for:
  - `text_generation`
  - `image_generation`
  - `tts`
  - `stt`
  - `music_generation` as a reserved/future task that currently returns a clear `501`.
- Standardized loaded runtime responses with residency fields including `task`, `state`,
  `resident`, `loaded`, `pinned`, `isolation`, `health`, and `error`.
- Kept `loaded_new` as a load-call event signal, not a resident-state alias:
  - text and image residency use explicit loader/cache results;
  - TTS/STT residency uses explicit backend event fields when present;
  - AbstractVoice-style `details.engine_cached=false` is treated as a clear signal that the call
    warmed a new local engine;
  - already-resident or ambiguous capability responses return `loaded_new=false`.
- Added task-aware unload behavior:
  - explicit non-text `task` routes directly to the capability residency adapter;
  - omitted `task` first preserves old text-runtime unload behavior, then falls back to
    image/TTS/STT matching by `runtime_id`, `provider`, or `model`.

Implemented Python capability support:

- Added optional residency methods to Core capability protocols:
  - `load_resident_model(request)`
  - `list_loaded_models(filters=None)`
  - `list_resident_models(filters=None)`
  - `unload_resident_model(request)`
- Added matching facade methods on:
  - `llm.vision`
  - `llm.voice`
  - `llm.audio`
- Kept the plugin boundary dictionary-shaped so Core does not import AbstractVision or
  AbstractVoice implementation types.
- Supported legacy/alternate plugin method names where useful:
  - `load_model`
  - `unload_model`
  - `list_loaded_models` falling back to `list_resident_models`.

Implemented server-side image residency:

- Added server vision residency helpers around the existing `/v1/images/*` backend cache:
  - `load_server_vision_resident_model(...)`
  - `list_server_vision_resident_models(...)`
  - `unload_server_vision_resident_model(...)`
- Reused the same `_BACKEND_CACHE` instances used by image generation and editing routes, avoiding
  a second independent image model cache.
- Called backend `preload()` when available.
- Called backend `unload()` on explicit unload and on best-effort LRU eviction.
- Cleared residency sidecar records when cached image backends are evicted.
- Reported OpenAI-compatible remote image providers as `state=configured`, `resident=false`, and
  `isolation=remote` instead of pretending Core has a local warm model.
- Removed the older public vision-specific load/unload endpoints from docs/OpenAPI because the
  public control plane is now `/acore/models/*`.

Implemented server-side audio/voice residency routing:

- Routed `task=tts` through the shared AbstractVoice-backed capability core used by the speech
  endpoints.
- Routed `task=stt` through the shared audio capability core used by transcription endpoints.
- Kept Core responsible only for the control-plane dispatch and error shape; model-specific warmup
  semantics remain owned by AbstractVoice.

Sibling package findings:

- AbstractVision exposes the expected residency protocol through its AbstractCore plugin:
  - `load_resident_model`
  - `list_loaded_models`
  - `list_resident_models`
  - `unload_resident_model`
  - compatibility aliases `load_model` and `unload_model`
- AbstractVoice exposes the same protocol shape, but current explicit residency behavior is narrow:
  - cloned TTS engine residency is implemented;
  - generic base TTS and STT residency currently return `not_implemented`;
  - the plugin still maintains shared `VoiceManager` instances, so ordinary TTS/STT calls reuse the
    long-lived capability core even where explicit preload/unload is not yet implemented.
- Core is now ready for broader AbstractVoice TTS/STT residency support without another Core API
  change.

Documentation updated:

- `docs/server.md`
- `abstractcore/server/README.md`
- `docs/memory-blocs.md`

Release preparation:

- Bumped `abstractcore/utils/version.py` to `2.13.18`.
- Added the `2.13.18` changelog entry covering task-aware residency, developer-message support,
  MLX Qwen no-thinking control, and removal of the old vision-specific model control endpoints.

Tests added or updated:

- `tests/capabilities/test_model_residency_facades.py`
- `tests/server/test_server_model_residency_control_plane.py`
- `tests/server/test_server_loaded_runtime_control_plane.py`
- `tests/server/test_server_openapi_docs.py`

Validation:

- `python -m py_compile abstractcore/capabilities/types.py abstractcore/capabilities/registry.py abstractcore/server/vision_endpoints.py abstractcore/server/app.py`
- `python -m pytest tests/capabilities/test_model_residency_facades.py tests/server/test_server_model_residency_control_plane.py tests/server/test_server_loaded_runtime_control_plane.py tests/server/test_server_openapi_docs.py`
  - Result: `16 passed`
- `python -m pytest tests/capabilities/test_model_residency_facades.py tests/server/test_server_model_residency_control_plane.py tests/server/test_server_loaded_runtime_control_plane.py tests/server/test_server_openapi_docs.py tests/server/test_server_vision_image_endpoints.py tests/server/test_server_developer_role.py tests/providers/test_system_prompt_alias.py tests/providers/test_thinking_mode_control_unit.py`
  - Result: `63 passed, 1 warning`
- `python -m pytest tests/test_capabilities_registry.py tests/test_capabilities_registry_preferred_backends_from_config.py tests/integration/test_capabilities_facades_with_fake_plugin.py tests/server/test_server_audio_endpoints.py tests/server/test_server_capability_catalog_routes.py`
  - Result: `65 passed`
- `ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=0 ABSTRACTCORE_RUN_LIVE_API_TESTS=0 ABSTRACTCORE_RUN_MLX_TESTS=0 ABSTRACTCORE_RUN_HUGGINGFACE_TESTS=0 ABSTRACTCORE_TEST_HERMETIC_MODEL_DISCOVERY=1 python -m pytest`
  - Result: `1394 passed, 243 skipped, 106 warnings`
- `python -m mkdocs build -q`
  - Result: passed
- `python -m build --outdir /tmp/abstractcore-dist-2.13.18`
  - Result: created `abstractcore-2.13.18.tar.gz` and `abstractcore-2.13.18-py3-none-any.whl`
- `python -m twine check /tmp/abstractcore-dist-2.13.18/*`
  - Result: passed for both artifacts
- `git diff --check`
  - Result: clean

Known boundaries and follow-up:

- `music_generation` remains reserved and returns `501` until a music residency-capable plugin
  contract exists.
- Core does not fabricate local residency for remote providers. Remote providers are reported as
  configured unless the upstream exposes a real loaded-state signal.
- Broader explicit preload/unload for non-cloned TTS and STT should be implemented in AbstractVoice;
  Core already dispatches to those hooks when they become available.
- TTL policy remains advisory in the request shape. Pin/TTL leasing can be added later without
  changing the public `/acore/models/*` route family.
