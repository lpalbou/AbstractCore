# Proposed: Capability Plugin Catalog Discovery Routes

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08

## Context

AbstractVision and AbstractVoice now expose package-owned discovery for remote OpenAI and
OpenAI-compatible capability catalogs:

- Vision can query provider model catalogs through the OpenAI-compatible `/models` shape.
- Voice can discover TTS model ids and voice/profile catalogs from OpenAI/OpenAI-compatible remote
  endpoints, while keeping built-in OpenAI voices as fallback data.

Core is the lower-level framework entry point and Core Server is the production HTTP boundary for
these capability packages. Core should expose this discovery through explicit server routes rather
than requiring clients to import AbstractVision or AbstractVoice directly.

2026-05-08 update: AbstractVoice's pending `0.9.1` release makes direct `VoiceManager()` and the
AbstractCore plugin remote-first by default, and adds `list_profiles(...)`, `list_tts_models()`,
and `voice_catalog()` on the plugin capability. AbstractVision's current integration exposes
`list_provider_models(...)` through its AbstractCore plugin capability. Core should now treat those
plugin methods as the intended package boundary, not as an unresolved lower-package design gap.

## Current Code Reality

- Core Server `/v1/models` lists LLM and embedding provider models.
- Core Server `/v1/vision/models` lists cached local AbstractVision registry models, not remote
  provider catalogs.
- Core Server audio routes expose TTS/STT/clone execution, but not voice profile/model discovery.
- `CapabilityRegistry.status()` reports registered plugin backends but does not instantiate them or
  expose deep catalogs.
- Core capability facades expose generation/transcription helpers but do not yet pass through the
  new plugin catalog methods.
- Core package pins needed to move to released package versions that contain the catalog methods.
  PyPI now has `abstractvoice==0.9.1` and `abstractvision==0.3.2`; both were verified locally
  through the AbstractCore plugin boundary. Core now uses `abstractvoice>=0.9.1` and
  `abstractvision>=0.3.2`.

## Problem

Thin clients need to know which image models, TTS models, and voice profiles are available before
issuing generation requests. This matters for Core Server and then cascades to Gateway, because
Gateway should not duplicate lower-level capability discovery.

The existing Core `/v1/models` endpoint is not enough:

- it is centered on LLM and embedding providers;
- its output capability enum currently covers text and embeddings, not image/audio generation;
- it does not query capability plugin catalogs;
- overloading it with plugin-specific media catalogs would blur Core provider models and capability
  backend models.

## Proposed Direction

Add explicit Core capability catalog routes and matching optional capability methods.

Suggested routes:

- `GET /v1/vision/provider_models`
  - calls the selected Vision capability/backend discovery path;
  - supports `task=text_to_image|image_to_image|text_to_video|image_to_video`;
  - returns provider model entries with raw provider metadata where available.
- `GET /v1/audio/voices`
  - returns the selected Voice capability's `voice_catalog()` shape when available;
  - includes `available`, `engine_id`, `active_profile`, `active_model`, `profiles`,
    `tts_models`, `source`, `stale`, and bounded `error` fields;
  - includes built-in, env-configured, provider-discovered, and custom profiles when available.
- `GET /v1/audio/speech/models`
  - returns TTS model ids from the selected Voice capability/backend;
  - may be implemented as a narrow projection of `/v1/audio/voices` to avoid duplicate discovery.

Core should keep `/v1/models` focused on LLM/embedding providers unless a separate ADR expands the
model taxonomy to include capability-generated media models.

## Capability Interface Direction

Prefer optional duck-typed methods on capability plugins:

- Vision capability:
  - `list_provider_models(task: str | None = None) -> list[dict]`
- Voice capability:
  - `list_profiles(kind: str = "tts") -> list[dict]`
  - `list_tts_models() -> list[str]`
  - `voice_catalog() -> dict`

Core can then expose these through `CapabilityRegistry` facades without importing package internals.

Required Core facade methods:

- `llm.vision.list_provider_models(task=None)`
- `llm.voice.list_profiles(kind="tts")`
- `llm.voice.list_tts_models()`
- `llm.voice.voice_catalog()`

Do not add a private Core adapter for current Voice/Vision releases. If the selected backend lacks a
catalog method, return a clear capability/configuration error. Private reach-throughs such as
`_get_vm()` are acceptable only for legacy compatibility paths that are already present for
execution, not for the new catalog contract.

## Security And Latency Rules

- General capability discovery must remain shallow and non-instantiating.
- Deep provider catalog routes may instantiate the selected capability backend and may make outbound
  provider calls.
- Deep catalog routes should use bounded timeouts and return explicit `available`, `source`,
  `stale`, and `error` fields where possible.
- Deep discovery should reuse the same config/env precedence as generation. For Voice this means
  Core Server maps `ABSTRACTVOICE_*` and `voice_*` config into the plugin host. For Vision this means
  adding equivalent plugin-host config mapping for `vision_backend`, `vision_base_url`,
  `vision_api_key`, `vision_model_id`, `vision_models_path`, and `vision_timeout_s`.
- Provider keys should follow the existing Core server credential boundary:
  - server-held provider keys require Core server auth for remote discovery;
  - request-level provider key overrides must use provider-key headers, not query strings;
  - request-level base URL overrides must use the existing allowlist/loopback guard.

## Non-Goals

- Do not make Gateway import AbstractVoice/AbstractVision directly for these catalogs when Core can
  provide the boundary.
- Do not make `/v1/models` silently mix LLM, embedding, image-generation, and TTS models without a
  broader model-taxonomy decision.
- Do not make shallow `/discovery/capabilities` instantiate heavy local engines or download models.
- Do not auto-select "latest" provider models from catalog listing. Catalog listing is inspection,
  not automatic model choice.

## Dependencies And Related Work

- AbstractVoice release dependency:
  - publish the remote-first/catalog-capable release before Core pins it;
  - expected minimum: `abstractvoice>=0.9.1`;
  - direct `VoiceManager()` and plugin mode are remote-first, so Core does not need to patch
    Voice env vars to get a lightweight server behavior.
- AbstractVision release dependency:
  - pin to `abstractvision>=0.3.2`; the PyPI `0.3.2` wheel contains the current plugin catalog
    method.
- Gateway proposed voice profile item:
  - `abstractgateway/docs/backlog/proposed/2026-05-08_voice_profile_discovery_endpoint.md`

## Promotion Criteria

Promote after the required Voice/Vision package versions are published and Core package pins can be
updated. Gateway/Flow clients need dynamic provider model and voice profile dropdowns that reflect
the server's actual configured capabilities, so this should be implemented before Gateway duplicates
any Voice/Vision catalog logic.

## Validation Ideas

- Core route tests with fake Vision and Voice capability plugins.
- Core facade tests for `llm.vision.list_provider_models(...)`, `llm.voice.list_profiles(...)`,
  `llm.voice.list_tts_models()`, and `llm.voice.voice_catalog()`.
- Route tests proving missing plugins return clear 501/install hints.
- Route tests proving unconfigured plugins return clear config hints.
- Provider-key/base-url security tests matching existing `/v1/models` and media generation rules.
- Timeout/error-shape tests for remote provider catalog failures.
- Packaging tests that prove `abstractcore[voice]` and `abstractcore[audio]` pull the catalog-capable
  AbstractVoice release, and `abstractcore[vision]` pulls the catalog-capable AbstractVision release.
- Gateway discovery tests can later proxy/consume these Core routes rather than duplicating package
  internals.

## Guidance For Implementing Agents

Keep the discovery contract explicit and typed. Start with plugin facade methods and server route
tests. Do not conflate Core's LLM model listing with capability-specific media catalogs unless a
separate ADR broadens that taxonomy.

## Completion Report - 2026-05-08

Implemented in AbstractCore:

- Added Core facade methods:
  - `llm.vision.list_provider_models(task=None)`
  - `llm.voice.list_profiles(kind="tts")`
  - `llm.voice.list_tts_models()`
  - `llm.voice.voice_catalog()`
- Added server catalog routes:
  - `GET /v1/vision/provider_models`
  - `GET /v1/audio/voices`
  - `GET /v1/audio/speech/models`
- Kept `/v1/models` focused on LLM/embedding discovery.
- Added provider-key/base-url handling for deep media catalogs under the same server credential
  boundary as media generation.
- Updated optional plugin floors to `abstractvision>=0.3.2` and `abstractvoice>=0.9.1`.
- Added unit/server tests for facades and catalog routes.
- Updated README, server docs, capabilities docs, architecture notes, and changelog.

Remaining downstream work:

- Gateway should consume these Core routes instead of importing AbstractVision/AbstractVoice.
- Runtime artifact/effect support remains tracked by the runtime-ready multimodal proposal.
