# Planned: Generate request object and output contract

## Metadata
- Created: 2026-06-14
- Status: In progress
- Completed: N/A

## ADR status
- Governing ADRs: root `docs/adr/0033-install-profiles-config-entrypoints-and-server-boundaries.md`, root `docs/adr/0035-capability-routing-defaults.md`
- ADR impact: No new Core-only ADR is required if this work stays within the accepted boundary:
  Core owns the lightweight request/output semantics, while Gateway and Runtime retain durable
  policy, replay, and workflow/action projection. Revisit ADR text only if request objects start
  absorbing routing authority or durability semantics.

## Context
AbstractCore already ships the public multimodal baseline:

- `generate(..., output=...)` and `agenerate(..., output=...)`;
- public output-selector normalization;
- route defaults for `input.*` and `output.*`;
- plugin-backed image, video, voice, transcription, music, and sound generation paths.

The remaining architectural gap is the input side. Direct Core callers still see a prompt-first API
surface, while multimodal routing and inference remain scattered through provider dispatch and
transport adapters. The intended end-state is a simple lower-level abstraction:

```python
result = llm.generate(request=..., output=...)
```

with route resolution handled through one shared internal route object, and a staged path toward
the even shorter `generate(request, output)` form if compatibility evidence justifies it later.

## Current code reality
- `AbstractCoreInterface.generate(...)` already treats `output=` as a real multimodal public
  contract while preserving text-only compatibility.
- Sync and async provider entrypoints duplicate important normalization and multimodal dispatch
  logic.
- Structural inference rules already exist, but they are buried inside provider dispatch:
  image edit vs generation, image-to-video, voice clone vs TTS, and transcription inference.
- Output routing and output aliases still reflect some implementation details such as sound/music
  bridging.
- Route defaults already distinguish `input.voice`, `input.sound`, `input.music`,
  `output.voice`, `output.sound`, and `output.music`, but direct Core request shape has not caught
  up.
- Default models and capability defaults are still ambient execution context, not one canonical
  call-scoped route object. Direct Core reads config or instance-scoped defaults ad hoc inside
  dispatch, while Runtime can pre-decorate some output specs before calling Core.
- Session wrappers and cached prompt flows are still prompt/history oriented, so positional
  request-first syntax needs a staged compatibility path rather than an abrupt overload.
- Route resolution and override behavior are not topology-parity-safe yet; the same requested
  generation can resolve differently in direct Core, local Runtime, remote Runtime, and server
  execution unless Core and Runtime share one normalized route-resolution path.

## Problem
Without a first-class request object and one normalization path:

- Core routing logic remains duplicated and drift-prone;
- direct callers do not have one obvious way to express multimodal input;
- the broader framework cannot point cleanly to Core as the lower-level semantic contract beneath
  Gateway actions and Runtime lowering;
- docs keep drifting between "prompt/media kwargs" and the intended `request/output` story;
- partial defaults, per-modality overrides, and topology-specific overrides can remain
  silent-fallback behavior because there is no single resolved route/plan object passed through the
  generate path.

## What we want to do
Add a lightweight request object/normalizer and central Core-owned request/output normalization
path plus a Core-owned internal default-route object so direct Core usage converges on
`generate(request, output)` semantics without copying Gateway durability or workflow concerns.

## Requirements
- Add a public keyword `request=` form to `generate()` and `agenerate()`.
- Keep the current first positional argument as the prompt string in this item. Do not overload it
  immediately with request objects.
- Keep the first stable canonical request scope small:
  - `text`
  - `messages`
  - `media`
- Preserve `output=` as the public output-selector contract. Do not hide output inside `request`.
- Keep provider/model/base URL/backend selection outside `request`; those remain explicit kwargs and
  route-default concerns.
- Centralize request normalization and structural task inference before provider/capability
  dispatch.
- Materialize one call-scoped internal route/plan object before dispatch. That object should carry:
  - the normalized request;
  - the normalized output request;
  - the Core-resolved text/input/output routes;
  - route provenance, including which values came from explicit call overrides versus inherited
    defaults;
  - an optional reasoning field when the resolved route targets a reasoning-capable model;
  - explicit override-denial or allow-with-record decisions.
- Defaults may originate from local Core config or from higher layers that pre-supply defaults, but
  Core still owns the merged call-scoped route object.
- Make request-to-output inference structural rather than prompt-semantic, including at least:
  - text request + `output=text` -> text generation;
  - text request + `output=image` -> text-to-image;
  - text request + `output=video` -> text-to-video;
  - text request + `output=music` -> text-to-music;
  - text request + `output=sound` -> text-to-audio/SFX;
  - text request + `output=voice` -> TTS;
  - image request + `output=image` -> image generation/edit/upscale according to roles/task;
  - image request + `output=video` -> image-to-video;
  - audio request + `output=text` -> transcription when structurally unambiguous.
- Preserve the current return-type split:
  - text-only compatibility calls can keep `GenerateResponse`;
  - multimodal outputs keep `MultimodalGenerateResponse`.
- Preserve current media transport plumbing. If more audio intent is needed, prefer a small media
  hint such as `kind: "voice" | "sound" | "music"` over adding many top-level kwargs.
- Use one normalized route-resolution path with Runtime so direct Core, local Runtime, remote
  Runtime, and server execution do not silently diverge on provider/model/base URL selection.
- Ensure output handlers and modality dispatch consume the same resolved defaults/route object
  rather than re-reading ambient defaults inconsistently later in the call.
- Keep the public override surface narrow. Compatibility selectors such as output-spec
  `provider/model` fields and input-specific override kwargs should normalize into the same
  internal route object rather than remaining separate side doors.
  The intended shape is “inherit defaults, override one route if needed,” not “rebuild the whole
  route table on every call.”
- Do not standardize a public structured route-overrides API here; only normalize existing
  compatibility selectors into the shared internal route object.
- Do not add a public route object inside `request` or `output` in this item. If a richer advanced
  override surface is later justified, track it separately from the core request/output contract.
- Treat partial defaults as configuration state, not executable routing truth.
- Do not let one output route silently rewrite the global text route. Text/input routing and output
  routing need distinct resolved fields in the internal plan.
- Add one canonical documentation page for the request/output abstraction and link existing docs to
  it instead of duplicating long example sets.

## Suggested implementation
1. Add a small `GenerateRequest` model plus `normalize_generate_request(...)` helper under
   `abstractcore/core/`.
2. Add a routing helper that resolves normalized request + output plus current defaults/overrides
   into a small internal plan before provider dispatch.
   The plan should include resolved per-modality routes and provenance rather than relying on
   ad hoc config reads or output-spec mutation later in the call.
3. Refactor sync and async provider multimodal entrypoints to consume the same normalized request,
   resolved plan, and route-resolution guardrails.
4. Keep current `generate(..., output=...)` kwargs working by normalizing them into the same
   request/output path.
5. Keep any internal `resolved_route` / `generation_plan` carrier private-first; document the
   public `request=` and `output=` contract before exposing more knobs.
6. Update docs/examples to teach:
   - text -> text;
   - text -> image;
   - text -> video;
   - text -> voice;
   - text -> music;
   - text -> sound;
   - image -> image edit;
   - image -> video;
   - audio -> transcription.

## Scope
- Core request object and normalizer.
- Central multimodal routing/inference helper.
- Sync/async parity cleanup.
- Route-parity guardrails for direct Core execution.
- Docs/examples/backlog cleanup for the request/output story.

## Non-goals
- Do not copy Gateway action/workflow descriptors or replay semantics into Core.
- Do not change `BasicSession.generate()` or cached-session signatures in this item.
- Do not add top-level `image=`, `video=`, `voice=`, `sound=`, or `music=` kwargs.
- Do not redesign streaming or structured-output behavior for non-text outputs here.
- Do not force positional `generate(request, output)` in the first implementation pass.
- Do not pass raw config payloads or Gateway discovery objects directly into `generate(...)`.
- Do not add a public route object or freeze an advanced override kwarg name in this item; the
  important contract here is the existence of one normalized call-scoped route object.

## Dependencies and related tasks
- Root `docs/backlog/planned/multimodal-capability-projection/0210_core_request_output_contract_and_gateway_projection_alignment.md`
- Root proposed item `docs/backlog/proposed/0211_public_generate_route_override_surface.md`
- Root completed item `docs/backlog/completed/0139_unified_framework_capability_defaults.md`
- Root completed item `docs/backlog/completed/0172_explicit_multimodal_default_fallback_routing.md`
- This package completed items:
  - `completed/2026-05-06_unified-multimodal-generate-api.md`
  - `completed/2026-05-07_public-output-selector-contract.md`
  - `completed/0807_task_specific_multimodal_default_routes.md`

## Expected outcomes
- Direct Core callers get one clear lower-level abstraction for multimodal input plus generated
  output selection.
- Sync and async multimodal behavior stop drifting.
- Per-modality default inheritance and compatibility override hints become explicit and
  topology-parity-safe instead of ambient behavior.
- The rest of the framework can rely on Core as the canonical lower-level request/output vocabulary.
- Existing text-first and `generate(..., output=...)` callers remain compatible.
- Silent route drift across topologies becomes a contract bug instead of a hidden implementation
  detail.

## Validation
- Focused tests prove `request=` and legacy `prompt/messages/media` kwargs normalize to equivalent
  internal plans.
- Focused tests cover structural inference for image edit, image-to-video, TTS, voice cloning, and
  transcription.
- Focused tests prove consistent route resolution or consistent rejection across direct Core,
  local Runtime, remote Runtime, and Core server execution for the same normalized request/output.
- Focused tests prove compatibility override hints normalize to the same resolved route object and
  effective execution across those topologies, or fail consistently when policy denies them.
- Docs and examples clearly distinguish the compatibility baseline from the request/output target.

## Implementation note - 2026-06-15

The first non-breaking Core pass is now landed:

- `GenerateRequest` and `normalize_generate_request(...)` exist under `abstractcore.core`.
- `generate()` and `agenerate()` accept `request=` while preserving prompt-first compatibility.
- sync and async generation now build the same normalized request and resolved-route summary before
  dispatch.
- multimodal generation responses and text responses can expose bounded
  `_resolved_generate_route` metadata for inspection/debug.

Remaining work is the broader parity wave: server entrypoints, stricter override-denial semantics,
and wider docs/examples beyond the initial request/output guide.

## Progress checklist
- [x] Define `GenerateRequest` and normalization helper.
- [x] Add central request/output routing helper.
- [x] Define the resolved-route object plus compatibility-override normalization.
- [ ] Add route-parity and override-denial guardrails.
- [x] Refactor sync and async multimodal dispatch to one normalization/resolution path before the
      existing multimodal dispatcher.
- [x] Add regression tests for request normalization parity, structural inference, and topology
      parity.
- [x] Publish the canonical request/output docs/examples and link older docs to them.

## Guidance for the implementing agent
Keep this item narrow and reversible. The goal is one lightweight semantic contract for direct Core
use, not a second durable action plane. Centralize normalization and structural inference first;
only revisit positional `generate(request, output)` after the keyword form, docs, and compatibility
tests prove stable. Fail closed on partial routes and ambiguous override behavior.
If a caller needs “default everything except this one modality,” normalize the existing explicit
override hints into the shared resolved-route object instead of mutating process-global config or
relying on hidden topology-specific kwargs. A future public structured override surface, if any,
belongs to the separate proposed follow-up.
