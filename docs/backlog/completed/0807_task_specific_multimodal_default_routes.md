# Planned: Task-specific multimodal default routes

## Metadata
- Created: 2026-06-08
- Status: Completed
- Completed: 2026-06-13

## ADR status
- Governing ADRs: ADR 0003, ADR 0004
- ADR impact: No ADR text change required. The implementation follows the accepted Core boundaries:
  Core owns persisted capability-default grammar and server control-plane routes, while static model
  capability metadata remains broad and backend-owned.

## Context
Gateway Console exposes multimodal defaults, but image and video generation now have distinct
provider/model needs. Text-to-image, image edit, image restoration/upscale, text-to-video, and
image-to-video should not be forced to share one broad `output.image` or `output.video` default.

## Current code reality
- AbstractCore owns capability-default parsing in `abstractcore.config.capability_defaults` and
  persistence in `abstractcore.config.manager`.
- AbstractCore and Gateway already expose capability catalogs for
  `text_to_image`, `image_to_image`, `image_upscale`, `text_to_video`, and `image_to_video`.
- AbstractRuntime and VisualFlow already compile these tasks into distinct media output specs.
- Gateway Console previously configured only broad `output.image` and `output.video` rows.

## Problem
Operators cannot choose different default provider/model pairs for image generation, image edit,
image upscale, text-to-video, and image-to-video. Using broad route defaults hides important task
differences and can route workflows to the wrong model.

## What we want to do
Add task-specific generated-media default routes:

- `output.image.text_to_image`
- `output.image.image_to_image`
- `output.image.image_upscale`
- `output.video.text_to_video`
- `output.video.image_to_video`

Keep `output.image` and `output.video` as compatibility fallbacks in Core/Runtime, but hide them
from Gateway Console so task-specific rows are the operator-facing UI and Runtime lookup path.

## Why
The user workflow is selecting a default provider/model for a concrete generative capability. Image
edit, upscale, and image generation often use different model families, and video text-to-video and
image-to-video can also differ. The control plane must make that distinction visible and durable.

## Requirements
- Core schema must parse and persist two- and three-part capability-default keys.
- Core server must expose GET plus two- and three-part PUT/DELETE routes.
- Runtime must apply task-specific defaults to generated-media specs when the node/request does not
  explicitly specify provider/model/base URL.
- Gateway must proxy scoped task defaults and keep principal isolation.
- Gateway Console must fetch the provider/model catalog for the selected task and persist the
  three-segment route.
- Static model capability metadata remains broad and must not become a task-default store.

## Suggested implementation
Extend the default-route grammar to `<kind>.<modality>[.<task>]` for defaults only. Use task keys
only for generated-media defaults where the provider catalogs already expose task filters. Add
focused tests at Core config/server, Runtime routing, Gateway API, and Console JavaScript levels.

## Scope
- `abstractcore` default schema, config manager, server config endpoints, and tests.
- `abstractruntime` generated-media default resolution.
- `abstractgateway` default proxy routes, scoped config overlays, Console UI, and tests.
- ADR/backlog updates.

## Non-goals
- Do not add task suffixes to `model_capabilities.json`.
- Do not treat a configured default as proof a model is downloaded or resident.
- Do not add new Gateway-owned default files.
- Do not change the executable output-spec task names beyond existing normalization.

## Dependencies and related tasks
- ADR 0003 provider/capability/output boundaries.
- ADR 0004 operator control and server trust boundary.
- AbstractCore vision provider catalogs.
- AbstractRuntime media output selector contract.
- Gateway Console multimodal capability UI.

## Expected outcomes
- Users can configure image generation, image edit, image upscale, text-to-video, and image-to-video
  independently from Gateway Console.
- Runtime receives these defaults and applies them to Auto media nodes without overwriting explicit
  provider/model choices.
- Split deployments can proxy the same task route grammar to AbstractCore server.

## Validation
- `PYTHONPATH=abstractcore pytest -q abstractcore/tests/config/test_capability_defaults_config.py abstractcore/tests/config/test_capability_defaults_server.py`
- `PYTHONPATH=abstractruntime/src:abstractcore pytest -q abstractruntime/tests/test_provider_endpoint_profile_resolution.py`
- `PYTHONPATH=abstractgateway/src:abstractruntime/src:abstractcore pytest -q abstractgateway/tests/test_gateway_console.py abstractgateway/tests/test_gateway_principal_auth.py -k 'capability_defaults or gateway_console'`
- Headless browser/JS validation against the Gateway Console for row rendering and task-specific
  configure/save flows.

## Progress checklist
- [x] Audit Core, Runtime, Gateway, and Console current behavior.
- [x] Revise ADR-0035 route-default policy.
- [x] Add Core schema and server route support.
- [x] Add Runtime task-default resolution.
- [x] Add Gateway proxy and Console task catalog support.
- [x] Add focused regression tests.
- [x] Run headless browser validation against a live local Gateway Console.

## Guidance for the implementing agent
Treat Core as the schema owner and Gateway as the control plane. Keep model capabilities broad,
apply task defaults only when provider/model/base URL are absent, and preserve explicit workflow
overrides.

## Completion report

### Summary

AbstractCore now accepts and persists task-specific generated-media default routes:

- `output.image.text_to_image`
- `output.image.image_to_image`
- `output.image.image_upscale`
- `output.video.text_to_video`
- `output.video.image_to_video`

The config manager, server control-plane routes, docs, and focused regression
tests all now treat these as first-class operator-facing defaults without
turning static model capability metadata into a task-default store.

### Validation

- `pytest -q abstractcore/tests/config/test_capability_defaults_config.py abstractcore/tests/config/test_capability_defaults_server.py`
  - `18 passed`
- `python -m py_compile abstractcore/abstractcore/config/capability_defaults.py abstractcore/abstractcore/config/manager.py abstractcore/abstractcore/server/app.py`
