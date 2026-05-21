# Planned: Request-scoped music backend selection and truthful reporting

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context
Gateway now exposes durable direct music generation through Runtime, and higher apps can ask for
specific local music providers and models. That only works if AbstractCore's local
`llm.generate(..., output={"modality": "music", ...})` path actually honors the requested music
backend and reports the backend it really used.

Live reproduction on 2026-05-21 showed that this is currently not true:

- Gateway requests asking for `ace-step` with
  `ACE-Step/acestep-v15-xl-turbo-diffusers` completed, but the resulting generated item came from
  `abstractmusic:acemusic`.
- Gateway requests asking for `stability-ai` with
  `stabilityai/stable-audio-open-small` also completed through `abstractmusic:acemusic`.
- Direct local `create_llm(...).generate(output={"modality": "music", ...})` reproduction showed
  the same behavior, including wrong output duration for a request that explicitly set
  `duration_s=10`.
- Direct Core server reproduction with `/v1/audio/music` and
  `backend="stable-audio"` attempted to load the ACE-Step Diffusers path with
  `model="stabilityai/stable-audio-open-small"`, proving selector drift below Gateway.

This item exists because the bug is now concrete, reproducible, and user-visible through the
Gateway music contract.

## Current code reality
- `abstractcore/providers/base.py` routes music outputs through `_run_music_output(...)`, which
  forwards request fields such as `provider`, `backend`, `music_backend`, and `model` as kwargs to
  `self.music.generate(...)`.
- `abstractcore/capabilities/registry.py` implements `MusicCapability.generate(...)` by calling
  `self._registry.get_music()` and then invoking the already-selected backend. There is no
  request-scoped backend resolution in that path.
- `abstractcore/core/interface.py` only maps instance-level `music_backend` config aliases for
  `acemusic`, `diffusers`, `acestep`, and `acestep-v15`.
- `abstractcore/server/audio_endpoints.py` has separate request-scoped selector logic for server
  routes, but its alias map still does not cover `stable-audio` and its local-path behavior is not
  shared with `llm.generate(..., output=music)`.
- `abstractcore/providers/base.py` currently reports `provider` and `model` from the request spec
  when building `GeneratedItem`, even if the actual backend used was different.
- Existing tests cover forwarding and fake plugin behavior:
  `tests/test_multimodal_generate_output.py`,
  `tests/server/test_server_music_endpoints.py`, and
  `tests/test_capabilities_registry_preferred_backends_from_config.py`.
  They do not prove that request-scoped backend/model selection works for the local output path or
  that reported backend/provider/model truth matches the invoked backend.

## Problem
AbstractCore's local music output path can silently ignore request-scoped backend selection while
still echoing the requested provider/model back to the caller. That creates three correctness
problems:

- callers cannot force a specific local backend per request;
- reported `provider`, `model`, and `backend_id` can disagree with the real backend;
- unsupported selectors can drift into a wrong backend instead of failing clearly.

This is the exact kind of abstraction leak that makes Runtime and Gateway look correct while the
lower layer is doing something else.

## What we want to do
Make request-scoped music backend selection explicit and truthful across local library mode and
server mode.

`llm.generate(..., output={"modality": "music", ...})`, `llm.music.generate(...)`, and
`POST /v1/audio/music` should either:

- run the requested backend/model combination and report the backend actually used, or
- fail with a clear selector/configuration error.

## Why
Higher layers already expose provider/model selection for music. If Core does not make that
selection real, users cannot trust discovery, test results, or produced artifacts.

Truthful reporting also matters for debugging. A generated WAV that claims it came from Stable
Audio but actually came from ACE Music is worse than an explicit failure.

## Requirements
- Add request-scoped local music backend resolution for `provider`, `backend`, `music_backend`,
  `model`, and server path-provider selectors.
- Do not silently fall back to the default selected backend when the caller explicitly requested a
  different backend or a backend/model combination that cannot be satisfied.
- Make `GeneratedItem.backend_id` reflect the backend that actually produced the artifact.
- Make reported `provider` and `model` come from the invoked backend/result metadata when
  available, not from a blind echo of the request payload.
- Keep server route alias handling and local library-mode alias handling on one shared selector
  path instead of two drifting implementations.
- Unknown backend aliases and model/backend mismatches must raise a clear user-facing error.
- Preserve current typed music plugin behavior; this item is about routing and truth, not about
  replacing the plugin contract.

## Suggested implementation
1. Add a small request-scoped music backend resolver in Core's capability layer, for example on
   `CapabilityRegistry` or a nearby helper module. It should accept backend/model/provider hints
   and return the concrete music capability backend to invoke.
2. Use that resolver from `MusicCapability.generate(...)` so local
   `llm.generate(..., output=music)` no longer depends on the process-wide selected backend.
3. Reuse the same resolver from the server audio route path, or make the server helper call into
   the same selection code, so alias behavior cannot drift.
4. Tighten generated item reporting so the actual backend/provider/model are surfaced from the
   invoked backend result. Keep requested values only as trace metadata when useful.
5. Add regression tests with multiple fake music backends proving:
   - explicit request-level backend selection works;
   - unsupported selectors fail explicitly;
   - reported backend/provider/model match the invoked backend;
   - `duration_s` and other request fields are preserved once the correct backend is selected.

## Scope
- Local `llm.generate(..., output={"modality": "music", ...})` routing.
- `llm.music.generate(...)` routing behavior where it shares the same selection path.
- Core server `/v1/audio/music` and provider-scoped music route selector behavior.
- Regression tests and docs needed to keep the selector contract truthful.

## Non-goals
- Do not add new music backends in this item.
- Do not change Gateway or Runtime contracts here except by making Core behave correctly under the
  existing contract.
- Do not broaden this into a generic cross-capability selector rewrite unless that is required to
  land the music fix cleanly.

## Dependencies and related tasks
- `docs/backlog/completed/0794_generic_capability_plugin_contract.md`
- `../abstractmusic/docs/backlog/planned/0085_truthful_stable_audio_capability_registration_and_music_routing.md`
- `../abstractgateway/src/abstractgateway/routes/gateway.py`
- `../abstractruntime/src/abstractruntime/integrations/abstractcore/run_facade.py`
- `tests/test_multimodal_generate_output.py`
- `tests/server/test_server_music_endpoints.py`

## Expected outcomes
- Request-scoped music backend selection is real in local library mode and server mode.
- Core reports the backend/provider/model it actually used.
- Wrong selectors fail explicitly instead of drifting into the default backend.
- Runtime and Gateway can rely on Core's music selection contract without compensating for it.

## Validation
- `python -m pytest -q tests/test_multimodal_generate_output.py tests/server/test_server_music_endpoints.py tests/test_capabilities_registry_preferred_backends_from_config.py`
- Add a focused regression test that reproduces the current failure with multiple fake music
  backends and asserts the invoked backend, `backend_id`, provider, model, and forwarded
  `duration_s`.
- When environment credentials and caches are available, rerun the local reproduction with:
  - `backend=acestep`, `model=ACE-Step/acestep-v15-xl-turbo-diffusers`
  - `backend=stable-audio`, `model=stabilityai/stable-audio-open-small`
  and confirm the returned backend is no longer `abstractmusic:acemusic`.

## Progress checklist
- [x] Add request-scoped music backend resolution for local output paths.
- [x] Reuse one selector path across local and server music routing.
- [x] Make generated music reporting reflect the actual invoked backend.
- [x] Add regression coverage for backend truth and selector failures.
- [x] Reproduce the original failure and record the corrected result.

## Completion report (2026-05-21)
### What changed in AbstractCore
- Added `abstractcore/capabilities/music_selectors.py` to normalize user-facing selectors (`acemusic`, `ace-step`, `stable-audio-3`, …) into concrete AbstractMusic `backend_id` strings.
- Implemented request-scoped backend selection inside `abstractcore/capabilities/registry.py` (`_MusicFacade.generate(...)`) so `llm.generate(output={"modality":"music", ...})` and `llm.music.generate(...)` honor per-request `backend` / `music_backend`, and can also use `provider` as a backend alias when it matches known aliases.
- Updated server `/v1/audio/music` routing to reuse the same selector normalization (via `music_selectors`) when constructing a request-scoped capability core.
- Tightened truth in `GeneratedItem` reporting for music in `abstractcore/providers/base.py`: `backend_id`/`provider`/`model` are sourced from the invoked backend result metadata when available (not blindly echoed from the request).
- Added an explicit stable-audio backend/model mismatch guard to fail early with a clear error (instead of backend-internal tracebacks).

### Regression coverage
- Added/extended tests proving:
  - request-scoped backend selection works even when the default backend is different;
  - unknown selectors fail without silently falling back;
  - reporting reflects the backend that actually ran;
  - stable-audio backend/model mismatch errors are raised early.

### Live verification against AbstractMusic 0.1.8
Ran local library-mode smoke checks (and one server-route check) with real `abstractmusic`:
- Remote `abstractmusic:acemusic`: generation succeeded; WAV duration matched `duration_s`.
- Local `abstractmusic:acestep-diffusers`: generation succeeded; WAV duration matched `duration_s`.
- Local `abstractmusic:stable-audio-3` with `stabilityai/stable-audio-3-small-music`: generation succeeded; WAV duration matched `duration_s`.
Notes:
- `stabilityai/stable-audio-open-small` is gated on HuggingFace in this environment (401); stable-audio-3 models were accessible and used for the local Stable Audio validation.

## Guidance for the implementing agent
Do not patch this by changing only the displayed provider/model fields. The real bug is that Core
can invoke one backend while claiming another. Fix selection first, then make reporting truthful.
