# Planned: Core resolved generate route/default object contract

## Metadata
- Created: 2026-06-15
- Status: In progress
- Completed: N/A

## ADR status
- Governing ADRs: root `docs/adr/0033-install-profiles-config-entrypoints-and-server-boundaries.md`, root `docs/adr/0035-capability-routing-defaults.md`
- ADR impact: No new Core-only ADR is required if this stays a lower-level route-resolution contract beneath Gateway descriptors. Revisit root `0210` and the related ADR work only if this item starts moving durable policy or replay authority into Core.

## Context
AbstractCore already has the raw ingredients for good multimodal routing:

- route-keyed defaults such as `input.text`, `input.voice`, `output.image.text_to_image`,
  `output.voice`, `output.sound`, and `output.music`;
- public `generate(..., output=...)` multimodal dispatch;
- per-output `provider` / `model` selectors for some generated-media paths;
- input-specific compatibility aliases such as STT provider/model kwargs; and
- Runtime/Gateway layers that already need the same lower-level route truth.

What is still missing is the one thing the stack actually needs when defaults exist but one call
should resolve differently: a single call-scoped route object created from defaults plus explicit
compatibility override hints and then consumed by the whole generate path.

Without that object, direct Core, local Runtime, remote Runtime, and server execution keep
resolving the same intent through slightly different ad hoc fields and topology-specific mutations.

## Current code reality
- `abstractcore.config.capability_defaults` and `abstractcore.config.manager` persist small
  capability-route rows, but they do not define a call-scoped merged route object.
- `abstractcore.providers.base.BaseProvider.generate_with_telemetry(...)` and `agenerate(...)`
  still resolve some defaults inline, including text/input fallback behavior and modality-specific
  compatibility aliases.
- `abstractruntime.integrations.abstractcore.llm_client` can pre-decorate generated-media output
  specs from capability defaults via `_with_capability_default_route(...)`, while text routing
  still uses separate `_provider` / `_model` reserved params.
- `abstractruntime.integrations.abstractcore.effect_handlers` can derive a call-global provider
  from the first output provider/profile hint, which proves the current route logic is still too
  topology-sensitive.
- Gateway text helper paths use a different `ProviderModelResolution` object for the text route,
  while generated-media descriptors and direct media routes use output-spec fragments and scoped
  capability defaults.
- Current docs tell users about capability defaults and output selectors, but not about one
  canonical internal route object. That leaves advanced users to guess whether they should set
  `provider`, `model`, `backend`, `stt_provider`, or some other field for a one-off request, and
  it leaves Core/Runtime to reinterpret those knobs differently by topology.

## Problem
The framework does not yet give `generate(...)` one explicit lower-level route contract for:

- inheriting capability defaults;
- normalizing explicit compatibility override hints for one call;
- recording which route actually won; and
- keeping that answer consistent across direct Core, Runtime, Gateway, and server execution.

That creates three concrete risks:

- override behavior is inconsistent and jargon-heavy;
- the same request can route differently by topology; and
- replay/debug surfaces cannot reliably say which route was actually used and why.

## What we want to do
Define one lower-level route contract for generation:

- one internal resolved route object that every generate path consumes; and
- one compatibility-mapping layer that turns existing explicit override fields into that
  Core-owned object without introducing a broad public override API.

This should make “use the defaults, but route just this modality differently for this call” resolve
through one shared substrate instead of remaining an emergent side effect of ad hoc kwargs.

## Why
- This is the cleanest way to make default-model inheritance and one-off overrides understandable
  before deciding whether a public structured override surface is worth exposing.
- It gives Runtime and Gateway a stable lower-level truth to project, persist, and replay.
- It prevents Core from needing separate topology-specific override logic for local, remote, and
  server paths.

## Requirements
- Define one canonical internal route object, for example `ResolvedGenerateRoute`, with distinct
  fields for:
  - the text/input route;
  - any input fallback routes used by the request;
  - each output route selected by the request;
  - an optional reasoning field for reasoning-capable models/routes only;
  - reasoning provenance/source when present;
  - provenance/source for every field; and
  - explicit denial or degradation reasons when resolution fails or is constrained.
- Normalize existing compatibility forms into that route object, including at least:
  - output-spec `provider` / `model` / `base_url` fields;
  - STT-specific provider/model aliases;
  - current text-route override fields used by Runtime/Gateway;
  - current backend/provider aliasing for music and sound generation.
- Keep any future public structured override surface out of this item. Simple callers should still
  use plain `request=` / `output=` and let defaults resolve automatically; advanced public
  overrides are deferred to root proposed item `0211`.
- Keep provider/model/base URL authority outside `request`. The request stays semantic; route
  authority remains operational and explicit.
- Gateway may supply default values or policy ceilings for a call path, but it must not own or
  redefine the resolved route/default object contract.
- Use the same route object across direct Core, local Runtime, remote Runtime, and server
  execution, or reject the request consistently when parity is impossible.
- Treat partial defaults as configuration state, not executable routing truth.
- Keep text/input routing separate from output routing. Output-route choices must not silently
  rewrite the global text route.
- Record enough provenance that debug/replay surfaces can explain:
  - which fields came from defaults;
  - which came from compatibility aliases;
  - which were denied by topology or policy ceilings.
- Emit a redacted stable summary form fit for Runtime/Gateway persistence and replay, while keeping
  the richer internal route/default object private-first inside Core.
- Surface the resolved route object to callers only in bounded metadata/debug forms where useful,
  without forcing it into the simple happy path.
- Do not make this item a commitment to any named public override parameter.

## Suggested implementation
1. Add a small route model/helper under `abstractcore/core/` or another clearly lower-level Core
   module, separate from provider implementations.
2. Implement one merge function that accepts:
   - normalized request/output intent;
   - scoped capability defaults;
   - compatibility alias fields; and
   - topology/policy constraints.
3. Refactor provider multimodal dispatch to consume the resolved route object instead of re-reading
   config or inferring overrides deep in modality-specific code.
4. Refactor Runtime and server entrypoints to build the same compatibility-override inputs and
   consume the same resolved route semantics.
5. Add focused docs/examples for “default inheritance + compatibility override normalization”
   without making that the first thing simple users see.

## Scope
- Core lower-level route-object design and implementation plan.
- Compatibility mapping from current override kwargs into the new contract.
- Cross-topology parity expectations for direct Core, Runtime, and server use.
- Backlog/docs alignment for resolved-route behavior and future override follow-up boundaries.

## Non-goals
- Do not move durable policy or replay ownership into Core.
- Do not copy Gateway action/workflow descriptors into Core.
- Do not force every caller to pass route overrides or understand route keys.
- Do not make ambient default mutation the recommended way to change one call.
- Do not design or freeze a public structured override API in this item; any later public surface
  must be separately justified, minimal, and layered on top of the internal Core contract.

## Dependencies and related tasks
- `0809_generate_request_object_and_output_contract.md`
- root `docs/backlog/planned/multimodal-capability-projection/0210_core_request_output_contract_and_gateway_projection_alignment.md`
- root `docs/backlog/planned/multimodal-capability-projection/0204_runtime_capability_intent_resolution_and_policy.md`
- root `docs/backlog/planned/multimodal-capability-projection/0202_gateway_replayable_session_and_action_envelope.md`
- root `docs/backlog/proposed/0211_public_generate_route_override_surface.md`
- root completed `docs/backlog/completed/0139_unified_framework_capability_defaults.md`
- root completed `docs/backlog/completed/0172_explicit_multimodal_default_fallback_routing.md`
- completed `2026-05-21_request_scoped_music_backend_selection_and_truthful_reporting.md`
- completed `0807_task_specific_multimodal_default_routes.md`

## Expected outcomes
- Compatibility override hints become explicit, legible, and topology-parity-safe once normalized
  into the shared route object.
- Core, Runtime, Gateway, and server paths can share one resolved route truth.
- Future durable replay can cite the same route object instead of reverse-engineering intent from
  scattered params.
- Docs can explain one lower-level resolved-route substrate instead of many special-case knobs.

## Validation
- Focused tests prove the same default-plus-override input resolves to the same route object across
  direct Core, local Runtime, remote Runtime, and Core server execution, or fails consistently.
- Focused tests prove output-spec/provider compatibility fields and STT/music alias fields normalize
  into the same resolved-route contract.
- Docs and backlog clearly distinguish ambient defaults, compatibility override hints, the internal
  resolved-route record, and the separate proposed public override follow-up.

## Implementation note - 2026-06-15

The shared Core substrate is partially implemented:

- `ResolvedGenerateRoute` and `ResolvedGenerateRouteEntry` now exist under `abstractcore.core`.
- Core resolves call-scoped text/input/output routes from normalized request/output plus explicit
  compatibility selectors and capability defaults.
- route summaries now preserve per-field provenance and optional reasoning defaults.
- Runtime local Core integration now consumes that Core-owned route summary instead of mutating
  output specs through a separate Runtime-only defaulting path.

The remaining gap is full topology parity and broader policy-denial semantics across direct Core,
Runtime remote paths, and Core server execution.

## Progress checklist
- [x] Define the resolved route object fields and provenance semantics.
- [x] Define the compatibility mappings that feed the resolved route object.
- [x] Add shared route merge/resolution helpers.
- [ ] Align direct Core, Runtime, and server entrypoints on the same semantics.
- [ ] Add parity and denial-path tests.
- [x] Publish clear resolved-route and compatibility-override examples.

## Guidance for the implementing agent
Keep this contract smaller than the durable Gateway action plane. The goal is not to expose all
control-plane semantics to direct Core callers. It is to replace hidden ambient route mutation with
one shared resolved route object and one explicit compatibility-mapping layer. Defer any public
advanced override surface until the proposed root follow-up has real evidence.
