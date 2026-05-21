# Planned: Generic capability plugin contract

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-21

## ADR status
- Governing ADRs: ADR-0002, ADR-0003, ADR-0005
- ADR impact: May revise ADR-0003 during implementation if the contract becomes a stable public extension API

## Context
AbstractCore already discovers optional modality packages through the
`abstractcore.capabilities_plugins` entry point group. This keeps Core dependency-light while
allowing packages such as `abstractvoice`, `abstractvision`, and `abstractmusic` to surface
voice/audio, image/video, and music generation capabilities.

The current direction is right: capability packages own their heavy dependencies and
modality-specific behavior, while AbstractCore owns the common routing, discovery, artifact, and
server integration layer. The next gap is consistency. Each plugin currently chooses its own method
names, catalog shapes, provider/model projections, and residency conventions, so every new
capability risks adding another bespoke adapter in Core and Server.

This item creates a small generic capability plugin interface that existing and future plugins can
implement while preserving their typed modality APIs.

## Current code reality
- `abstractcore/capabilities/registry.py` already provides entry-point loading,
  `register_backend(...)`, and convenience registration helpers for `voice`, `audio`, `vision`, and
  `music`.
- `abstractcore/capabilities/types.py` has separate `VoiceCapability`, `AudioCapability`,
  `VisionCapability`, and `MusicCapability` protocols plus shared provider/model/operation/result
  records for common discovery and artifact-compatible results.
- The Core-side foundation now includes shared capability provider/model/operation/result types,
  generic registry discovery helpers, and a plugin-safe host text service. Plugin package adoption
  is still the remaining work.
- `abstractvoice` exposes a rich plugin surface in
  `../abstractvoice/abstractvoice/integrations/abstractcore_plugin.py`: `available_providers()`,
  `list_models(kind=..., provider=...)`, `voice_catalog()`, `compatibility_catalog()`, TTS/STT/clone
  operations, and partial residency. This is useful but voice-specific and JSON shapes are evolved
  inside the package.
- `abstractvision` exposes `available_providers(task=...)`, `list_provider_models(task=...)`,
  image/video generation methods, and local residency in
  `../abstractvision/src/abstractvision/integrations/abstractcore_plugin.py`. Its model records are
  closer to provider-model metadata but differ from AbstractVoice's catalogs.
- `abstractmusic` already exists locally at `../abstractmusic` and registers three music backends
  through `abstractcore.capabilities_plugins`; the minimal backend method is `t2m`, with Core-owned
  `llm.music.generate(...)` and `llm.generate(..., output="music")` adapters layered above it.
  Its lower-level backends and assets already contain useful `get_capabilities()` and
  `list_provider_models(...)` signals that Core cannot consume consistently yet.
- `abstractmusic` also has a dependency-free prompt-planning contract and already accepts injected
  host planners through `music_text_planner`, `music_text_planner_instance`, and
  `music_text_planner_factory`. This proves plugins may need host text-generation help without
  hard-depending on AbstractCore.
- `AbstractCoreInterface.generate(...)` already supports provider-agnostic text generation,
  structured output through `response_model`, tools, thinking controls, and media inputs. Passing
  the raw provider object directly into plugins is too broad for a public plugin contract, because
  a plugin could accidentally re-enter capability generation or depend on provider internals.
- AbstractCore Server already exposes modality-specific routes such as `/v1/vision/providers/`,
  `/v1/audio/speech/models`, `/v1/audio/transcriptions/models`, `/v1/voice/clone/providers`, and
  `/v1/audio/music`.
- AbstractCore Server now also exposes generic capability discovery routes and music
  provider/model discovery routes. The next step is making AbstractMusic, AbstractVoice, and
  AbstractVision produce richer normalized records source-first instead of relying on compatibility
  normalization.

## Problem
Capability plugins are becoming a framework extension point, but the extension contract is still
implicit. That creates avoidable drift:

- Core facades need capability-specific fallback logic instead of one shared discovery path.
- Server endpoints need bespoke provider/model projection code for each modality.
- New packages such as AbstractMusic and future video plugins have no clear minimum surface to
  implement.
- UI/runtime clients cannot ask one generic question such as "which providers and models can serve
  this capability/task?" without knowing package-specific catalog names.
- Residency and artifact handling are similar across plugins but not shaped by one contract.
- Some plugins need optional text intelligence when hosted by AbstractCore, for example prompt
  expansion, lyrics, captions, metadata extraction, safety rewrites, or structured planning. There
  is no narrow contract for "use the host LLM for text only" that avoids plugin -> AbstractCore
  dependency cycles and accidental capability recursion.

## What we want to do
Define and implement a small, versioned capability plugin contract in AbstractCore. Existing
plugins should be able to adopt it incrementally without breaking their current APIs.

The v1 contract should standardize common capability behavior:

- plugin/backend identity and metadata
- lightweight provider availability
- provider-filtered model discovery
- operation/task capability metadata
- artifact result semantics
- optional local residency hooks
- optional host text-generation service available to plugins when AbstractCore is the host
- optional generic invocation metadata for adapters; typed methods such as `tts`, `transcribe`,
  `t2i`, `i2i`, `t2v`, `i2v`, and `t2m` remain the minimal plugin contract, while Core-owned
  facades may expose ergonomic `generate(...)` aliases and `llm.generate(..., output=...)`
  routing for common tasks

## Why
This makes AbstractCore a cleaner orchestration layer for the full AbstractFramework ecosystem. A
future music or video package should not require Core to learn a fresh ad hoc catalog shape before
it can expose providers, models, capabilities, artifacts, and server routes.

The interface also gives plugin authors a concrete target: implement the shared contract for
discovery and lifecycle, then add modality-specific request parameters where they belong.

## Requirements
- Keep plugin imports lightweight. Entry-point loading and provider/model discovery must not load
  multi-GB models or optional heavy runtimes.
- Preserve typed modality APIs. Do not force all plugin implementations through one weakly typed
  `generate(**kwargs)` path; Core may provide `generate(...)` adapters over typed plugin methods.
- Provide a generic discovery layer that works across `voice`, `audio`, `vision`, `music`, and
  future `video`.
- Standardize provider records with required fields `provider_id`, `display_name`, `capability`,
  `tasks`, `local`, `remote`, and `status`; optional fields include `backend_id`, `installed`,
  `configured`, `reachable`, `selected`, `install_hint`, `config_hint`, and bounded `metadata`.
- Standardize model records with required fields `model_id`, `provider_id`, `capability`, `tasks`,
  `modalities`, `local`, `remote`, and `status`; optional fields include `backend_id`,
  `routed_model`, `formats`, `source`, `recommended`, `license`, `commercial_allowed`, and
  bounded `raw_metadata`. Core must not infer license or commercial status when plugins do not
  provide it.
- Standardize operation records with fields such as `operation_id`, `capability`, `task`,
  `input_modalities`, `output_modalities`, optional plugin-authored `parameter_schema`,
  `required_parameters`, and `artifact_output`.
- Keep artifact behavior aligned with current Core conventions: library mode may return bytes,
  artifact refs, or a precise `CapabilityInvokeResult`; framework/server mode can pass
  `artifact_store`, `run_id`, `tags`, and `metadata` and receive a structured artifact ref.
- Keep residency optional and explicit. Remote HTTP providers should not pretend to be unloadable
  local runtimes.
- Keep existing OpenAI-compatible routes and modality routes stable. Generic routes should be
  additive or implemented as shared internal helpers underneath the existing routes.
- Keep generic execution optional in v1. If implemented, it must accept a known `operation_id` and
  validated `parameters`, reject unknown top-level fields, and delegate to plugin-owned typed
  request schemas.
- Provide a host text-generation service contract for plugin use. This must be opt-in, text-only,
  and narrow: plugins can request text or structured text from the host LLM, but cannot receive the
  raw provider object as their public dependency.
- Prevent dependency cycles. Plugin base packages must remain usable without AbstractCore; any
  AbstractCore-specific imports belong only in optional `integrations/abstractcore*.py` modules or
  should be avoided through structural typing.
- Prevent capability recursion. Host text generation for plugins must call the text provider with
  non-text `output` disabled and must not route back through `voice`, `vision`, `music`, or future
  generated-media capabilities.

## Contract v1 decisions
- Minimum plugin author contract:
  - register through `abstractcore.capabilities_plugins` or Core's existing registration helpers
  - expose lightweight `available_providers(task=None)`, `list_models(task=None, provider_id=None)`,
    and `list_operations(task=None)` or equivalent methods Core can normalize
  - keep typed generation methods for claimed operations, such as `tts`, `transcribe`, `t2i`,
    `i2i`, `t2v`, `i2v`, and `t2m`
  - return JSON-safe normalized records; plugin-specific fields must live under bounded
    `metadata` or `raw_metadata`
- Optional extensions:
  - `invoke(...)` for adapters and generic clients
  - local residency hooks
  - health checks
  - plugin-authored parameter JSON Schema
  - progress callbacks
  - rich plugin-specific catalogs
  - host text-generation service for prompt planning, captioning, metadata extraction, and other
    plugin-owned text tasks
- Discovery depth:
  - `list_backend_infos(capability=None)` must not instantiate plugin factories
  - `available_providers(...)` must not load models or call remote `/models` endpoints
  - `list_models(...)` may perform configured provider discovery only when explicitly requested,
    with timeout, stale/error metadata, and no process-wide side effects
  - all-backend aggregation must be opt-in and proven import-light; selected-backend discovery
    remains the default execution-oriented path
- Canonical task IDs:

| Capability | Canonical tasks | Legacy aliases accepted at adapters |
| --- | --- | --- |
| `voice` | `text_to_speech`, `voice_clone` | `tts`, `cloning` |
| `audio` | `speech_to_text`, `transcription` | `stt` |
| `vision` | `text_to_image`, `image_to_image`, `text_to_video`, `image_to_video` | `t2i`, `i2i`, `t2v`, `i2v` |
| `music` | `text_to_music`, `lyrics_to_music`, `text_to_audio` | `t2m`, `text2music` |
| `video` | `text_to_video`, `image_to_video`, `video_to_video` | `t2v`, `i2v`, `v2v` |

`capability` is the package/domain bucket. `task` is the operation family. Core should normalize
aliases at the boundary but expose canonical IDs in new generic responses.

## Host text service for plugins
Add a narrow host-service protocol alongside the discovery/result protocols:

- `CoreTextGenerationService.generate_text(prompt, *, messages=None, system_prompt=None,
  max_output_tokens=None, temperature=None, thinking=None, purpose=None, metadata=None) ->
  CoreTextResult`
- `CoreTextGenerationService.generate_structured(prompt, *, response_model=None,
  json_schema=None, messages=None, system_prompt=None, purpose=None, metadata=None) -> object`

The service is supplied by AbstractCore when it hosts a plugin. It wraps the selected text provider
and forces text-only generation. It must not expose arbitrary provider methods, capability facades,
server request objects, or mutable global state.

Clean dependency rule:

- AbstractCore may pass a `CapabilityHostContext` or equivalent object into plugin factories.
- Plugins may accept that object structurally, or use it only inside their optional AbstractCore
  integration module.
- Plugin base packages must not import AbstractCore just to define their core request/response
  models.
- A plugin should also keep deterministic fallback behavior for no-host/no-LLM operation unless
  the caller explicitly sets a required planner mode.

Safety rule:

- The service must be explicit and observable. Host config should decide whether a plugin may use
  text generation, and plugin results should report provenance such as `planner_backend`,
  `planner_model`, generated fields, warnings, and failures.
- The service must strip or reject non-text `output`, streaming media outputs, and capability
  requests before delegating to `generate(...)`.
- Structured calls should use AbstractCore's existing `response_model` path when a Pydantic model
  is supplied, or a JSON-schema path only if Core already supports it safely for the selected
  provider.

## Suggested implementation
1. Add a small public module such as `abstractcore.capabilities.interfaces` or extend
   `abstractcore.capabilities.types` with shared dataclasses/protocols:
   `CapabilityProviderInfo`, `CapabilityModelInfo`, `CapabilityOperationInfo`,
   `CapabilityArtifactRef`, `CapabilityInvokeResult`, `CapabilityHostContext`,
   `CoreTextGenerationService`, and `CoreTextResult`.
2. Keep structural typing as the compatibility layer. Plugins may inherit from a base class/mixin,
   but Core should accept objects that satisfy the protocol.
3. Split the shared protocol into small optional protocols instead of one monolithic
   `CapabilityBackend`: discovery, operation metadata, artifact/result handling, and residency.
4. Add shared registry/facade methods:
   `list_backend_infos(capability=None)`, `list_capabilities()`,
   `available_providers(capability, task=None)`, `list_models(capability, task=None,
   provider=None)`, and `capability_catalog(capability, ...)`. Keep `invoke(capability,
   operation, request)` optional and secondary.
5. Add a safe host text-service wrapper that uses the current text provider for text-only
   generation and structured text output. The wrapper must not expose provider internals or
   generated-media capability facades to plugins.
6. Adapt `VoiceCapability`, `AudioCapability`, `VisionCapability`, and `MusicCapability` to satisfy
   the shared discovery/result protocols while keeping their existing typed methods. Do not require
   runtime inheritance from a common base class.
7. Use `abstractmusic` as the first validation target: expose music provider availability,
   model discovery, capability metadata, and `t2m` operation metadata from its packaged
   `music_model_capabilities.json` and backend `get_capabilities()`/`list_provider_models(...)`
   APIs.
8. Use AbstractMusic's text-planner injection as the first host-text-service validation target:
   Core should be able to supply an opt-in planner backed by `CoreTextGenerationService` without
   AbstractMusic importing AbstractCore in its base package.
9. Prefer plugin-owned normalized producers for `abstractvoice` and `abstractvision`. Transitional
   Core normalizers may adapt their existing catalogs, but must be bounded, version-gated, and
   removable once plugin floors expose the normalized contract.
10. Add server-facing helpers and routes only where they reduce duplication. Candidate additive
   routes:
   - `GET /v1/capabilities`
   - `GET /v1/capabilities/{capability}/providers`
   - `GET /v1/capabilities/{capability}/models`
   - `GET /v1/audio/music/providers`
   - `GET /v1/audio/music/models`
11. Preserve existing route/error behavior for `/v1/audio/*`, `/v1/voice/*`, `/v1/vision/*`,
   `/v1/images/*`, and `/v1/models`. New aggregate routes must not replace or reshape existing
   catalog routes.
12. Update docs to explain the plugin author contract, host-service contract, and client discovery
    flow.

## Scope
- Core protocol/dataclass definitions for shared capability plugin behavior.
- Registry/facade helpers that expose common provider/model/catalog discovery.
- Optional host-service protocol for plugin-safe text and structured-text generation.
- Source-first plugin adoption for AbstractMusic plus bounded compatibility normalizers for the
  existing AbstractVoice and AbstractVision plugin surfaces.
- AbstractMusic plugin adoption as the proof target.
- AbstractMusic host-text-planner injection as the proof target for plugin-safe LLM reuse.
- Server and docs updates for common discovery, especially music provider/model discovery.
- Tests with fake plugins plus real import-light checks against AbstractVoice, AbstractVision, and
  AbstractMusic when available locally.

## Non-goals
- Do not move heavy voice, vision, music, or video runtime dependencies into AbstractCore.
- Do not replace `BaseProvider` or the LLM provider abstraction.
- Do not merge generated-media model catalogs into `/v1/models`; keep LLM/embedding model
  discovery separate from generated-media capability discovery.
- Do not require every plugin to implement local residency.
- Do not design a full plugin marketplace, dependency resolver, UI schema language, or remote
  execution protocol in this item.
- Do not make Core interpret plugin parameter schemas as a UI schema language. Core may validate
  envelope fields and expose plugin-authored schemas as metadata.
- Do not let plugins create their own AbstractCore LLM provider by default. Host text generation
  should be supplied by the host application/runtime as an explicit service, not discovered
  implicitly by plugins.
- Do not let host text services become a general agent/runtime API. This item covers text and
  structured-text helper calls for plugin-owned planning only.
- Do not block plugin-specific rich catalogs. The shared contract should provide a normalized
  baseline and allow bounded plugin-specific metadata.

## Dependencies and related tasks
- ADR-0003: provider, capability, and output boundaries.
- ADR-0002: validation and evidence requirements.
- ADR-0005: source-first quality fixes.
- `docs/backlog/completed/2026-05-08_capability_plugin_catalog_discovery_routes.md`
- `docs/backlog/completed/2026-05-19_generalize_acore_models_residency.md`
- `docs/backlog/planned/2026-05-07_multimodal-generation-and-deterministic-inference-cache.md`
- `abstractcore/capabilities/registry.py`
- `abstractcore/capabilities/types.py`
- `abstractcore/server/audio_endpoints.py`
- `abstractcore/server/vision_endpoints.py`
- `../abstractvoice/abstractvoice/integrations/abstractcore_plugin.py`
- `../abstractvision/src/abstractvision/integrations/abstractcore_plugin.py`
- `../abstractmusic/src/abstractmusic/integrations/abstractcore_plugin.py`
- `../abstractmusic/docs/backlog/proposed/0080_text_planning_provider_contract_for_music.md`
- `../abstractmusic/src/abstractmusic/prompt_planner.py`

## Expected outcomes
- Plugin authors have one documented minimum contract for provider/model/capability discovery and
  artifact-compatible typed generation.
- AbstractCore can list providers and models for `voice`, `audio`, `vision`, `music`, and later
  `video` through one common Python surface.
- AbstractCore Server can expose consistent provider/model discovery for generated-media
  capabilities without duplicating bespoke catalog transformation logic per package.
- AbstractMusic can be used as the first clean adopter before publication.
- AbstractMusic can optionally use a host-provided AbstractCore text planner for prompt/lyrics
  planning without adding a hard AbstractCore dependency or making surprise network calls.
- Existing AbstractVoice and AbstractVision integrations continue to work through their current
  typed APIs and server routes.
- Generic `invoke` is either deferred or implemented as an optional adapter layer with explicit
  parameter validation, not as a required kwargs tunnel.

## Validation
- Unit tests for the shared protocol/dataclass JSON serialization and normalization behavior.
- Registry tests with fake plugins covering provider discovery, model discovery, operation metadata,
  backend selection, missing-method errors, metadata truncation, `status()` compatibility, and
  no-heavy-import behavior.
- Registry tests proving `list_backend_infos(capability=None)` does not instantiate backend
  factories, and selected-backend discovery remains separate from opt-in all-backend aggregation.
- Compatibility tests that adapt existing AbstractVoice and AbstractVision catalog surfaces into the
  shared normalized shape.
- AbstractMusic plugin tests proving `available_providers`, `list_models`, `capability_catalog`,
  and `t2m` all work without loading a model during discovery.
- Host text-service tests proving plugins can call text and structured-text generation through a
  narrow service object, explicit config controls whether it is available, and non-text
  `output`/capability recursion is rejected.
- AbstractMusic integration tests proving `music_text_planner_factory` can receive the host text
  service, planner provenance is reported, required/off/auto modes behave correctly, and
  AbstractMusic's base import still does not require AbstractCore.
- Server tests for any added generic capability routes and music provider/model discovery routes.
- Server compatibility tests proving existing `/v1/audio/*`, `/v1/voice/*`, `/v1/vision/*`,
  `/v1/images/*`, and `/v1/models` response shapes and missing-plugin `501` behavior remain
  unchanged.
- OpenAPI tests proving added routes are documented without removing existing modality routes.
- Docs build.
- Optional local smoke: install AbstractMusic in editable mode, query music providers/models, then
  generate a short WAV through `/v1/audio/music` using one local model at a time.

## Progress checklist
- [x] Audit current plugin surfaces and write the exact shared contract.
- [x] Decide whether ADR-0003 needs a revision or whether the backlog item plus docs are sufficient.
- [x] Add shared capability interface types and normalization helpers.
- [x] Extend `CapabilityRegistry` and facades with generic provider/model/catalog methods.
- [x] Add plugin-safe host text-generation and structured-text service wrappers.
- [x] Adapt AbstractMusic as the first proof package.
- [x] Wire AbstractMusic text planning to the host text service through explicit injection.
- [x] Backfill AbstractVoice and AbstractVision compatibility adapters.
- [x] Add server discovery routes or shared server helpers.
- [x] Update docs and plugin-author guidance.
- [x] Add validation tests for the Core-side foundation.

## Guidance for the implementing agent
Keep the abstraction intentionally small. The valuable common layer is discovery, metadata,
artifact/result handling, lifecycle, and explicitly injected host services. The actual generation
methods should remain typed and modality-owned so voice, image, music, and video can expose the
parameters that matter without turning AbstractCore into a generic kwargs tunnel. If a plugin needs
LLM help, pass a narrow text service from the host; do not make plugins instantiate AbstractCore or
reach through the raw provider object.

## Completion report

### Summary (2026-05-21)

This item is complete from AbstractCore’s perspective: Core now exposes a small shared capability
plugin contract for discovery + optional residency + optional host text planning, without
introducing dependency cycles or requiring plugins to import AbstractCore in their base packages.

What Core expects from capability backends (v1, structural typing):

- Discovery (import-light): `available_providers(task=None)` and `list_models(task=None, provider=None|provider_id=None)` (or equivalent aliases such as `list_provider_models`).
- Optional operation metadata: `list_operations(task=None)` (or `get_capabilities()`).
- Optional residency control: `load_resident_model`, `list_loaded_models`/`list_resident_models`, `unload_resident_model`.
- Typed modality methods remain plugin-owned (`tts`, `transcribe`, `t2i`, `t2m`, etc.).

What Core exposes to plugins (host text only, no recursion):

- `owner.capability_host_context.text.generate_text(...)`
- `owner.capability_host_context.text.generate_structured(..., response_model=...)`

This wrapper forces `output=None`, disallows media, uses `stream=False`, and guards against
capability re-entry.

Plugin status checked (latest available at time of completion):

- `abstractmusic==0.1.8`: implements the shared discovery surface (`available_providers`, `list_models`, `list_operations`) and can optionally consume the host text service when co-installed with Core.
- `abstractvision==0.3.8`: implements `available_providers` and `list_provider_models` and works with Core’s shared discovery normalizers.
- `abstractvoice==0.10.12`: voice backend implements discovery; its audio/STT backend does not expose discovery methods yet, so `llm.capabilities.available_providers("audio")` and `llm.capabilities.list_models("audio")` are not currently supported through the generic surface (but `llm.audio.transcribe(...)` still works through the typed facade).

Follow-up (plugin-owned, not required for Core correctness): AbstractVoice can add generic discovery
methods to its audio/STT backend if/when consistency across all capabilities becomes important.

### Validation

- `pytest -q tests/test_capabilities_registry.py tests/server/test_server_capability_routes.py tests/server/test_server_openapi_docs.py`
