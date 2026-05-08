# Proposed: Core Install Profiles And Gateway Configuration Boundary

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08

## Context

AbstractCore is the lower-level entry point for developers who want provider/model access without
running durable agents through Gateway. It owns LLM provider execution, tools, media processing,
capability plugin discovery, server endpoints, provider prompt-cache behavior, and standalone
provider configuration.

Gateway is a higher-level entry point. It should use Core configuration and capabilities, but it
should own deployment/run configuration.

Core also owns the inbound security boundary for the standalone Core server. Gateway owns its own
inbound security boundary; neither token/origin policy should silently become the other.

## Current Code Reality

- Core base install is light: `pydantic` and `httpx`.
- Hosted/provider extras are explicit: `remote`, `openai`, `anthropic`, `openrouter`, `portkey`,
  `openai-compatible`, `ollama`, and `lmstudio`.
- Local engine extras are explicit: `huggingface`, `mlx`, `vllm`, `embeddings`, and local vision
  extras.
- Hardware-profile local LLM aliases are explicit: `apple` aliases the native Apple/MLX stack and
  `gpu` aliases the local GPU/vLLM stack.
- Aggregate profiles already exist: `all-apple`, `all-gpu`, and broad legacy `all`.
- Capability plugins are discovered through `abstractcore.capabilities_plugins`.
- AbstractVoice's pending `0.9.1` release makes the package and plugin remote-first by default and
  moves the full local voice stack behind `abstractvoice[local]`.
- AbstractVision's package is already remote/OpenAI-compatible light by default, with local
  Diffusers/stable-diffusion.cpp behind explicit extras.
- `abstractcore-config` persists provider keys, model defaults, media fallback settings, server
  settings, cache/logging settings, and more.
- Core server auth is already separate from provider API keys: server auth uses
  `ABSTRACTCORE_SERVER_API_KEY`, while per-request upstream provider overrides use the dedicated
  provider-key override path.
- Core server CORS/origin policy still needs first-class config hardening. The current server
  behavior should not be treated as the final production model.

## Problem

Core is already close to the desired shape, but its role in a Gateway-first stack needs clearer
rules:

- "default lightweight" can mean either minimal HTTP-compatible Core or hosted provider SDKs
  included by default.
- `abstractcore-config` currently mixes provider configuration with local install/model setup.
- Gateway needs to take precedence for deployment settings without duplicating every Core setting.
- Capability readiness needs to distinguish a registered plugin from a usable configured backend.

## Proposed Direction

Keep Core as the provider/capability foundation.

Profile vocabulary:

- `abstractcore`: light import-safe base. Decide explicitly whether to keep OpenAI/Anthropic SDKs
  in `remote` or promote them into base for a no-extra hosted-provider experience.
- `abstractcore[remote]`: official hosted provider SDKs and remote-provider convenience.
- `abstractcore[server]`: standalone Core server without local model engines.
- `abstractcore[vision]`: remote-light generative image capability plugin intent. It should depend
  on `abstractvision>=0.3.2` but not install Diffusers/Torch by default.
- `abstractcore[voice]`, `abstractcore[audio]`: remote-light TTS/STT capability plugin intent. They
  should depend on `abstractvoice>=0.9.1` (or the first published remote-first/catalog-capable
  release) and must not install Piper, faster-whisper, PortAudio, local cloning, or Torch by default.
- `abstractcore[voice-local]` or equivalent future extra: optional explicit bridge to
  `abstractvoice[local]` for users who want Core to launch local voice/STT engines in-process.
- `abstractcore[apple]`: native Apple local LLM stack; currently an alias of `mlx`.
- `abstractcore[gpu]`: local GPU LLM stack; currently an alias of `vllm`.
- `abstractcore[all-apple]`: aggregate native Apple stack.
- `abstractcore[all-gpu]`: aggregate local GPU stack.
- `abstractcore[all]`: legacy broad bundle, not recommended for cross-platform users.

Preferred recommendation: keep no local engines in the default path. If maintainers want
`pip install abstractcore` to support official OpenAI/Anthropic immediately, moving those SDKs into
base is acceptable because they are lightweight, but the decision must be explicit and tested.

## Configuration Boundary

`abstractcore-config` should own:

- provider API keys;
- provider base URL defaults where supported, plus clear env/runtime guidance where base URLs are
  not yet persisted as first-class config;
- Core default provider/model for direct Core users;
- media fallback policies;
- standalone Core server auth, allowed origins, base URL allowlists, media/fetch/local-file policy,
  and other server exposure settings;
- local model cache paths;
- local engine setup when explicitly selected.

For capability packages, Core config should record intent and defaults, not duplicate package
internals:

- Voice:
  - `voice_tts_engine`, `voice_stt_engine`, `voice_remote_base_url`, `voice_remote_api_key`,
    `voice_tts_model`, `voice_stt_model`, and `voice_allow_downloads` are enough for Core to host
    AbstractVoice through the plugin boundary.
  - Default server profile should be remote OpenAI/OpenAI-compatible, inherited from
    AbstractVoice's remote-first default; local `piper` / `faster_whisper` must be explicit.
- Vision:
  - `vision_backend`, `vision_base_url`, `vision_api_key`, `vision_model_id`,
    `vision_models_path`, and `vision_timeout_s` are enough for Core to host AbstractVision through
    the plugin boundary.
  - Default server profile should be OpenAI/OpenAI-compatible remote image generation; local
    Diffusers/stable-diffusion.cpp must be explicit.

Core should expose catalog/readiness APIs for these settings so direct Core users and Gateway can
preflight the actual configured capability surface before generation.

`abstractgateway-config` should own:

- Gateway auth and security;
- data/store/runner/workflow/workspace policy;
- Gateway run default provider/model;
- tool approval mode;
- embedding provider/model for Gateway memory;
- capability readiness preflight.

Gateway can read Core defaults and can invoke Core config during a setup wizard, but Gateway should
not mutate Core package defaults during individual requests.

## Server And Credential Boundary

Separate three concerns that are easy to conflate:

- Gateway client auth: protects Gateway routes and thin-client access.
- Core server auth: protects the standalone Core server when it is served directly or called by
  Gateway as a separate service.
- Provider credentials: authenticate outbound calls from Core or capability packages to OpenAI,
  Anthropic, OpenRouter, Portkey, OpenAI-compatible servers, and modality backends.

If Gateway embeds Core in-process, Core server auth/CORS settings are not part of the exposed HTTP
surface. If Gateway calls a standalone Core server, Gateway should use the Core server's URL and
server auth token explicitly. It should not reinterpret `ABSTRACTGATEWAY_AUTH_TOKEN` as
`ABSTRACTCORE_SERVER_API_KEY`, and it should not copy Gateway browser origins into Core unless the
deployment explicitly exposes Core to browsers.

Core server `Authorization` should remain the server-auth channel when server auth is configured.
Per-request upstream provider-key overrides should remain on the dedicated Core provider-key
override mechanism. Request-body or query-string API keys should stay disabled.

## Pending Changes Guidance

Core working tree is clean and should remain the source of truth for:

- light base versus explicit local engine extras;
- `all-apple` and `all-gpu` aggregates;
- capability plugin discovery.

Revise related pending changes outside Core:

- root `abstractframework` pins and default extras should align with current Core minimums;
- Gateway should depend on `abstractcore[remote,media,tools,tokens,compression,vision,voice,audio]`
  for server deployments, not invent parallel provider behavior;
- Core config docs should say when values are process env, persisted config, or only runtime
  in-memory provider config.
- Core server security backlog should focus on the remaining gaps: configured allowed origins,
  shared security helpers across routers, tenant keys/quotas/rate limits, and consistent base URL
  exfiltration guards for chat, embeddings, vision, audio, and future routers.

Immediate Core package changes after Voice/Vision releases:

- Update `abstractcore[voice]` and `abstractcore[audio]` pins from `abstractvoice>=0.9.0` to the
  first published release with remote-first defaults and plugin catalog methods, expected
  `abstractvoice>=0.9.1`.
- Update `abstractcore[vision]` pins to the first published AbstractVision release with
  `llm.vision.list_provider_models(...)`; `abstractvision==0.3.2` on PyPI was verified to contain
  that method, so the expected Core minimum is `abstractvision>=0.3.2`.
- Add a local voice extra only if Core maintainers want a one-command Core install for local
  Piper/faster-whisper/voice-cloning. Otherwise document `pip install "abstractvoice[local]"`
  alongside `abstractcore[voice]`.
- Keep `abstractcore[server]` dependency-light. Gateway server images can compose
  `server,remote,media,tokens,compression,vision,voice,audio`; local GPU/Apple images should add
  explicit local extras.

## Promotion Criteria

Promote when maintainers decide whether official remote SDKs belong in Core base or remain in
`abstractcore[remote]`, and after Core pins the released remote-first/catalog-capable Voice/Vision
versions.

## Validation Ideas

- Packaging tests for no-extra, `remote`, `server`, `apple`, `gpu`, `all-apple`, and `all-gpu`.
- Import-light tests proving no Torch/MLX/vLLM/Diffusers local stack is imported by base or remote.
- Config tests for env precedence, persisted keys, provider base URLs, and Gateway fallback to Core
  defaults.
- Capability registry tests for installed/registered/configured/ready states.
- Capability package pin tests proving default Core server/media profiles stay remote-light.
- Config tests proving Gateway-supplied runtime defaults can override Core defaults without mutating
  persisted Core package config.

## Completion Report - 2026-05-08

Implemented in AbstractCore:

- Kept the base install and `abstractcore[server]` dependency-light.
- Kept official hosted SDKs in `abstractcore[remote]` instead of moving them into base.
- Updated remote-light plugin floors:
  - `abstractcore[voice]` / `abstractcore[audio]`: `abstractvoice>=0.9.1`
  - `abstractcore[vision]`: `abstractvision>=0.3.2`
  - local vision extras now also target `abstractvision>=0.3.2`
- Preserved explicit local-engine extras and aggregate profiles:
  - `apple` aliasing `mlx`, and `gpu` aliasing `vllm`
  - `vision-diffusers`, `vision-sdcpp`, `vision-local`
  - `all-apple`, `all-gpu`, legacy `all`
- Added packaging tests that assert `abstractcore[server]` does not pull AbstractVoice or
  AbstractVision while the explicit media extras do.
- Documented the Core/Gateway boundary in README/server/capabilities docs:
  Core owns provider/capability configuration and media catalogs; Gateway should own deployment,
  workflow, tenant, and inbound Gateway auth policy.

Deferred/non-Core work:

- Gateway should choose its own composed dependency profile for server deployments.
- Tenant keys, quotas, rate limits, and stricter CORS/origin hardening remain part of the server
  security backlog, not this install-profile cleanup.
