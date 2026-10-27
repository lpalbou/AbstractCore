# 0857 — Voice library: clone language, deploy to a library, and per-request voice stores on the server

**Status**: planned · **Priority**: P2 · **Created**: 2026-10-27
**Cross-package**: AbstractVoice 0.19.0 (backlogs 0063, 0065: shipped), AbstractRuntime 0.13.0 (0851: shipped),
AbstractGateway backlog 0942 (personal voices per account).

## Why
AbstractVoice 0.19.0 records a cloned voice's language and gives each person a personal voice store in front of read-only
libraries; AbstractRuntime 0.13.0 applies a runtime's stores in process. AbstractCore 2.30.0 passes plugin settings and
`clone(..., **kwargs)` through, so in-process use works, but three things only AbstractCore can add are missing:

1. `POST /v1/voice/clone` has no `language` form field (AbstractVoice `clone(..., language=)`): over HTTP a recorded
   voice's language can only be identified from its transcript.
2. No facade method and no route deploys a voice to a library (AbstractVoice plugin `deploy_voice_to_library(voice_id)`):
   `_VoiceFacade` has no `deploy_voice_to_library`, so AbstractRuntime calls the backend through
   `CapabilityRegistry.get_voice()` in process and answers 501 remotely.
3. A Core server has one voice store (its own plugin configuration): every client of the server shares it, so a
   multi-account host that uses a remote Core server cannot isolate voices per account.

## Current code reality (2026-10-27, 2.30.0)
- `abstractcore/server/audio_endpoints.py` `voice_clone(...)`: form fields `file, model, provider, name, reference_text,
  validate, base_url, clone_path, file_field, consent`; the local branch calls `core.voice.clone(...)` without `language`.
- `abstractcore/capabilities/registry.py` `_VoiceFacade`: `list_languages`, `list_voices`, `list_cloned_voices`,
  `get_cloned_voice`, `rename_cloned_voice`, `set_cloned_voice_reference_text`, `delete_cloned_voice`, `design_voice`,
  `compatibility_catalog`; no deploy.
- Plugin config reaches the plugin from the owner's config (`voice_store_dir`, `voice_library_dirs` work in process).

## Scope
- In: `language` form field on `/v1/voice/clone` (and `/v1/{provider}/voice/clone`) forwarded to `voice.clone`;
  `_VoiceFacade.deploy_voice_to_library(voice_id)` (CapabilityUnavailableError when the backend lacks it) and
  `POST /v1/voice/clones/{voice_id}/deploy` (operator only); a decision on per-request stores for the server (for example
  a server-side mapping from an authenticated principal to a `voice_store_dir`, never a client-chosen path).
- Out: the stores themselves (AbstractVoice), the gateway policy (AbstractGateway 0942).

## Validation
A French recording cloned over HTTP with `language=fr` is listed with `language: fr`; deploy over HTTP refuses a
non-operator (401) and copies the voice for an operator; a remote AbstractRuntime voice facade's `deploy_voice_to_library`
answers instead of 501 once the route exists.
