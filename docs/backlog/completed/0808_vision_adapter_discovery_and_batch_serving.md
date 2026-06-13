# Completed: Vision adapter discovery and batch serving through AbstractCore

## Metadata
- Created: 2026-06-13
- Status: Completed
- Completed: 2026-06-13

## ADR status
- Governing ADRs: ADR 0003, ADR 0005, ADR 0008
- ADR impact: No ADR change was required. The implemented boundary keeps AbstractVision as the owner of adapter/model/task truth while AbstractCore remains a thin hosting and server adapter.

## Context
AbstractVision already owned the real task-specific image/video request types, stacked LoRA adapter support, installed adapter discovery, and batch orchestration with deterministic seed planning. AbstractCore exposed the high-level vision tasks, but it still hid part of that contract from Python callers and from the OpenAI-compatible server boundary.

## Problem
Before this work:
- Core callers could not discover installed compatible adapters through `llm.vision`.
- Core did not expose first-class batch generation for `t2i`, `i2i`, `t2v`, and `i2v`.
- Sync and async server routes accepted partial batch-like parameters but did not delegate faithfully to the Vision contract.
- Typed stacked LoRA adapters and task-specific video controls were not preserved consistently across the Core bridge.

## What changed
- Extended the AbstractVision Core plugin to expose:
  - `list_provider_adapters(...)`
  - `t2i_batch(...)`
  - `i2i_batch(...)`
  - `t2v_batch(...)`
  - `i2v_batch(...)`
- Extended AbstractCore vision capability contracts and the `_VisionFacade` with the same methods.
- Updated `BaseProvider` generated-media dispatch so image/video output specs can request:
  - `count` / `n`
  - explicit `seeds`
  - stacked `lora_adapters`
  - typed video controls such as `guidance_2` and `flow_shift`
- Updated the server-local capability bridge and `/v1/images/*`, `/v1/videos/*`, and `/v1/vision/jobs/*` routes so they preserve the typed Vision contract instead of flattening it into generic extras or repeating singular calls.
- Added `GET /v1/vision/adapters` for installed compatible adapter discovery.
- Raised the AbstractCore dependency floor to `abstractvision>=0.3.26`.

## Why this shape
The key design choice was to avoid inventing a Core-owned adapter registry. Adapter compatibility and installed-adapter discovery remain backend-owned in AbstractVision. Core now exposes that truth faithfully to Python, server, Runtime, Gateway, and Flow without duplicating compatibility logic.

## Scope completed
- `abstractvision/src/abstractvision/integrations/abstractcore_plugin.py`
- `abstractcore/abstractcore/capabilities/types.py`
- `abstractcore/abstractcore/capabilities/registry.py`
- `abstractcore/abstractcore/providers/base.py`
- `abstractcore/abstractcore/server/capability_generation.py`
- `abstractcore/abstractcore/server/vision_endpoints.py`
- focused tests, docs, changelogs, and version-floor updates

## Expected outcomes
- Core Python callers can discover installed compatible adapters for a selected route.
- Core library and server callers can request multiple images/videos with different seeds in one call.
- Stacked LoRA adapters and typed video controls survive the full Core boundary.
- Downstream packages can rely on Core as a truthful host for the current Vision contract.

## Validation
- `pytest -q abstractvision/tests/test_abstractcore_plugin.py -q`
- `pytest -q abstractcore/tests/test_multimodal_generate_output.py abstractcore/tests/test_capabilities_registry.py abstractcore/tests/server/test_server_capability_catalog_routes.py abstractcore/tests/server/test_server_vision_image_endpoints.py -q`

## Progress checklist
- [x] Extend the AbstractVision Core plugin with adapter discovery and batch methods.
- [x] Extend Core vision capability/facade contracts.
- [x] Route batch image/video generation through Core output dispatch.
- [x] Preserve typed LoRA and video controls through the Core HTTP bridge.
- [x] Add adapter discovery and multi-output server routes/tests.
- [x] Update docs, dependency floors, changelogs, and release notes.
- [x] Release `abstractvision`, then release `abstractcore`.

## Completion report

Completed on 2026-06-13.

### What changed
- AbstractVision’s Core plugin now forwards installed-adapter discovery and the four batch generation helpers directly from `VisionManager`.
- Core’s public vision capability exposes the same methods, with actionable errors when a backend does not implement them.
- Core multimodal output dispatch now batches image/video tasks through the capability layer instead of looping one singular request path.
- Sync and async server routes now honor explicit seed lists, stacked LoRA adapter arrays, and typed `flow_shift` / `guidance_2` video controls.
- Async server job routes now reuse `ServerVisionFacade` for batch seed planning, backend request construction, and progress-method dispatch instead of duplicating those decisions in `vision_endpoints.py`.
- Async image and video jobs now preserve backend-reported denoise totals when the caller omits explicit `steps`, so polling clients see truthful step-based progress instead of frame-count or placeholder totals.
- New server route `GET /v1/vision/adapters` surfaces installed compatible adapters without re-owning compatibility truth in Core.

### Validation summary
- AbstractVision focused plugin tests: passed.
- AbstractCore focused capability/output/server suites: passed.
- Added regression coverage for:
  - server adapter discovery
  - sync image/video batch forwarding
  - batch seed expansion and LoRA forwarding
  - async progress aggregation when totals come only from backend progress callbacks/events

### Residual risks
- Adapter compatibility quality still depends on backend-owned discovery in AbstractVision. That is intentional, but it means future adapter/model truth updates must be made in Vision first.
- Gateway/Runtime/Flow integration work should continue to reuse this Core surface instead of adding package-local adapter logic.
