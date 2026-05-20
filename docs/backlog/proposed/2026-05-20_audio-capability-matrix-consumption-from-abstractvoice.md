# Proposed: Consume AbstractVoice capability matrices instead of only legacy voice controls

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None in AbstractCore yet
- ADR impact: Promote only if Core begins enforcing capability semantics or exposing a new public
  capability-query surface beyond simple pass-through catalog data

## Context

AbstractVoice now exposes a richer package-owned audio capability surface than the older
`voice_catalog()["controls"]` booleans. The new payload includes:

- `speech_request_contract`
- `tts_capabilities`
- `compatibility_catalog`

That is the right long-term seam for truthful provider/model feature discovery:

- `instructions` can be modeled as supported for some models and not others
- future fields such as `scene_context`, `pace`, `ambient_audio`, or `background_sfx` can be
  represented without overloading `voice`, `profile`, or `model`
- support states can distinguish `native`, `emulated`, `conditional`, and `unsupported`

Today AbstractCore mostly passes the AbstractVoice catalog through, which preserves compatibility,
but it does not yet use the richer matrix to improve its own filtering, validation, or request UI.

## Current code reality

- [`abstractcore/server/audio_endpoints.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/server/audio_endpoints.py:1152) returns `core.voice.voice_catalog()` largely as-is.
- [`abstractcore/server/audio_endpoints.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/server/audio_endpoints.py:1219) still defaults legacy `controls` when a backend does not provide them.
- [`abstractcore/server/audio_endpoints.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/server/audio_endpoints.py:1605) already accepts and forwards request fields such as `instructions`, `speed`, `profile`, `provider`, and `model`.
- [`abstractcore/capabilities/types.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/capabilities/types.py:49) already includes `instructions` in the `VoiceCapability.tts()` protocol, so Core does not need a breaking transport change to carry richer TTS requests.
- Search across `abstractcore/` on 2026-05-20 found no current consumer of `compatibility_catalog`, `tts_capabilities`, or `speech_request_contract`; the richer fields are present only if the voice backend emits them.
- The current filtering helpers in [`abstractcore/server/audio_endpoints.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/server/audio_endpoints.py:881) still reason mostly from provider/model lists rather than explicit per-feature compatibility data.

## Problem or opportunity

Core can already expose richer audio capabilities without owning the semantics itself. The gap is
that it still behaves like all TTS providers share one flat control set.

That becomes increasingly inaccurate as:

- `instructions` is model-specific rather than universally true
- some support is `conditional` instead of guaranteed
- future engines such as Scenema-class or DramaBox-class runtimes expose richer directed-speech
  features only on certain providers/models/surfaces

If Core keeps relying only on old `controls`, higher-level apps will either over-offer unsupported
features or hardcode provider-specific heuristics outside the package that actually owns the audio
semantics.

## Proposed direction

Keep this work proposed for now, but preserve the integration path:

1. Treat `voice_catalog()["compatibility_catalog"]` as the preferred truth source when present.
2. Keep `controls` as a legacy compatibility fallback, not as the authoritative feature model.
3. Add Core-side helpers for two query directions:
   - “Can this `(kind, provider, model, surface)` use feature `X`?”
   - “Which `(provider, model)` pairs support feature `X` on surface `Y`?”
4. Use the capability matrix to improve:
   - provider/model filtering in catalog routes
   - request validation and better error messages
   - UI/schema hints for higher-level clients
5. Keep ownership boundaries clean:
   - AbstractVoice owns audio semantics and capability facts
   - AbstractCore serializes, filters, and exposes them
   - Core should not invent a parallel audio capability vocabulary

## Why it might matter

- It lets higher-level apps stop guessing which voice providers/models can honor `instructions`.
- It creates a durable path for future directed-speech fields without breaking the current API.
- It keeps AbstractCore aligned with the pattern already used elsewhere in the stack: capability
  assets and query helpers instead of provider-name heuristics.

## Promotion criteria

Promote this item only when one of these becomes true:

- a Core UI or client needs truthful per-feature filtering for current audio providers
- request validation bugs appear because `controls` is too coarse
- AbstractVoice begins exposing richer public directed-speech fields beyond today’s stable subset
- multiple higher-level apps are duplicating provider/model capability heuristics that should live
  in one Core helper

Promotion should stay additive:

- no removal of legacy `controls`
- no break to existing `/audio/voices` payloads
- no reinterpretation of AbstractVoice-owned semantics inside Core

## Validation ideas

- Add endpoint tests that verify `/audio/voices` pass-through still works while `compatibility_catalog`
  is present.
- Add focused helpers/tests that filter models by feature support without regressing legacy
  provider/model listings.
- Exercise at least one `native`, one `conditional`, and one `unsupported` audio feature case.
- Verify that older voice backends without `compatibility_catalog` still behave exactly as before.

## Non-goals

- Do not make AbstractCore the owner of audio capability semantics.
- Do not force every voice backend to implement the richer matrix immediately.
- Do not remove `controls` before downstream clients are migrated.
- Do not couple this work to Scenema or DramaBox runtime implementation.

## Related

- [`../abstractvoice/abstractvoice/assets/voice_model_capabilities.json`](/Users/albou/tmp/abstractframework/abstractvoice/abstractvoice/assets/voice_model_capabilities.json:1)
- [`../abstractvoice/abstractvoice/compatibility.py`](/Users/albou/tmp/abstractframework/abstractvoice/abstractvoice/compatibility.py:1)
- [`../abstractvoice/docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`](/Users/albou/tmp/abstractframework/abstractvoice/docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md:1)
- [`../abstractvoice/docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md`](/Users/albou/tmp/abstractframework/abstractvoice/docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md:1)
