# Planned: Public local vision cache catalog helper

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20

## ADR status
- Governing ADRs: None
- ADR impact: None

## Context

AbstractCore already exposes public capability catalog routes and capability-facade methods for
voice and vision discovery. That work landed in:

- `docs/backlog/completed/2026-05-08_capability_plugin_catalog_discovery_routes.md`

Core Server still owns one additional local-only snapshot path for vision caches:

- `GET /v1/vision/models`

That route currently depends on a private helper in:

- `abstractcore/server/vision_endpoints.py::_local_vision_model_catalog`

Sibling packages now need the same local cached-vision snapshot without importing Core server
internals. AbstractRuntime's new discovery facade currently has to import that private helper
directly, and Runtime's `abstractcore` extra only depends on the base Core package rather than the
server extra.

## Current code reality

- `abstractcore/server/vision_endpoints.py::_local_vision_model_catalog()` performs the real
  cache scan and model snapshot shaping.
- `abstractcore/server/vision_endpoints.py::list_cached_vision_models()` uses that helper for the
  HTTP route.
- `abstractcore/server/vision_endpoints.py::_vision_provider_catalog()` also calls the same helper,
  so `/v1/vision/models` currently scans the local cache twice per request.
- The helper lives in the server module, so importing it also pulls server-layer dependencies and
  warnings that are irrelevant to in-process library consumers.
- AbstractRuntime currently imports the private helper from outside Core to implement local
  discovery snapshots:
  - `../abstractruntime/src/abstractruntime/integrations/abstractcore/discovery_queries.py`
- `abstractcore/server/vision_endpoints.py` imports FastAPI at module import time, while Core's
  base dependency set does not include FastAPI. That means the current Runtime import reaches across
  both a private boundary and a package-extra boundary.
- `abstractcore/utils/model_cache.py` already exposes public Hugging Face and LM Studio cache-path
  helpers that overlap with part of the server-local scan.

## Problem

Core already has the right behavior, but not yet the right public seam.

Without a public helper:

- sibling packages must import a private `abstractcore.server.*` symbol;
- non-server consumers inherit server-module dependencies and import side effects;
- cache-catalog logic cannot evolve cleanly without risking external breakage.
- Runtime can fail this local discovery path in valid installs that do not include the Core server
  extra.

## What we want to do

Extract the local cached-vision scan into a public, server-agnostic helper in the Core library,
make the server route delegate to that helper, and switch Runtime to the public import path.

## Why

- It removes the last known Runtime import of private Core server internals for discovery.
- It keeps valid base-Core installs from accidentally depending on FastAPI just to inspect a local
  vision cache snapshot.
- It gives Gateway/Runtime/other in-process hosts one stable Core library boundary for local cached
  vision model inspection.
- It lets Core evolve server route code separately from reusable library helpers.
- It removes an avoidable duplicate local-cache scan inside `/v1/vision/models`.

## Requirements

- Add a public helper in a non-server module such as `abstractcore.capabilities.vision_catalog`.
- Expose a synchronous library API, e.g. `get_local_vision_cache_catalog(...) -> dict`.
- Keep FastAPI, router, request, auth, and HTTP exception concerns out of the helper.
- Preserve the current local snapshot fields needed by consumers:
  `models`, `registry_available`, `registry_total`, `cached_total`, `cache_dirs`, and bounded
  `error` behavior.
- Keep `/v1/vision/models` response compatibility, including any route-level augmentation such as
  active-backend state.
- Update Runtime to import the public helper instead of `abstractcore.server.vision_endpoints`.
- Avoid scanning the local cache twice inside the `/v1/vision/models` route path.

## Suggested implementation

- Add `abstractcore/capabilities/vision_catalog.py` as the public home for the local cached-vision
  snapshot helper.
- Reuse `abstractcore/utils/model_cache.py` where it already provides dependency-light cache-path
  helpers, and move only the vision-specific diffusers/local-directory scan logic that does not
  belong in the server module.
- Keep `abstractcore/server/vision_endpoints.py::_local_vision_model_catalog()` as a thin
  server-only wrapper if that minimizes route churn, but make it delegate to the public helper.
- Thread the route's first local catalog result through `_vision_provider_catalog(...)` so
  `/v1/vision/models` does not perform the same scan twice.
- Update Runtime discovery queries and tests to import the new public helper directly.

## Scope

- Public Core helper for local cached-vision snapshot inspection.
- Server delegation updates for `/v1/vision/models` and any internal server reuse of the same scan.
- Runtime import cleanup for local cached-vision discovery.
- Targeted Core and Runtime regression tests.
- Core version and changelog updates tied to the shipped boundary fix.

## Non-goals

- Do not redesign vision provider-model discovery broadly; that already has public routes/facades.
- Do not make this helper an HTTP concern.
- Do not change async offloading policy. Whether callers run the synchronous helper on the main
  thread or offload it remains a host concern.
- Do not broaden this into a general model-cache architecture rewrite unless current code forces a
  small utility extraction.

## Dependencies and related tasks

- `docs/backlog/completed/2026-05-08_capability_plugin_catalog_discovery_routes.md`
- `abstractcore/server/vision_endpoints.py`
- `abstractcore/capabilities/registry.py`
- `abstractcore/utils/model_cache.py`
- `../abstractruntime/src/abstractruntime/integrations/abstractcore/discovery_queries.py`
- `../abstractruntime/tests/test_abstractcore_discovery_facade.py`

## Expected outcomes

- Core exposes a documented public library helper for local cached-vision catalog snapshots.
- Runtime no longer imports `abstractcore.server.*` for this query.
- `/v1/vision/models` still returns the expected payload shape but no longer performs a duplicate
  local scan.
- Targeted tests prove the helper is import-safe for non-server consumers and that the route
  delegates to it.

## Validation

- Unit test the new public helper without importing FastAPI/router objects.
- Update `/v1/vision/models` tests to prove route delegation and single-scan behavior.
- Add a Runtime regression test proving local cached-vision discovery uses the public helper even
  when `abstractcore.server.vision_endpoints` is unavailable.
- Run focused pytest for the new helper, the server catalog routes, the capabilities facade tests,
  Runtime discovery tests, and import-safety coverage.

## Progress checklist
- [x] Promote the proposal to a planned item with current code reality.
- [x] Add a public non-server local vision cache catalog helper in Core.
- [x] Delegate server local vision catalog paths to the public helper.
- [x] Remove Runtime's private `abstractcore.server.*` import for cached local vision discovery.
- [x] Add focused Core and Runtime regression tests.
- [x] Update changelog and version metadata.
- [x] Append a completion report and move the item to `completed/`.

## Guidance for the implementing agent

Keep this narrow. Publish the existing local cached-vision snapshot logic behind a clean library
seam, preserve the current route behavior, and prefer explicit proof over broad refactors.

## Completion report

### Date

2026-05-20

### Summary

AbstractCore now exposes a public non-server helper for the local cached-vision snapshot:
`abstractcore.capabilities.vision_catalog.get_local_vision_cache_catalog()`, re-exported from
`abstractcore.capabilities`. Core Server delegates its local `/v1/vision/models` snapshot to that
helper, keeps `active` as route-owned server state, and avoids scanning the local cache twice on
the same request path. AbstractRuntime's local discovery query now imports the public Core helper
instead of `abstractcore.server.vision_endpoints`.

### Behavior changes

- Added a public dependency-light library seam for local cached-vision inspection.
- Kept server-only route augmentation out of the public helper.
- Removed the last known Runtime import of private `abstractcore.server.*` discovery code.
- Preserved the existing local snapshot shape used by current consumers while making the route path
  single-scan instead of double-scan.
- Raised the optional AbstractVision install floor to `abstractvision>=0.3.8` across current
  vision-enabled Core profiles after validating the helper against the local 0.3.8 source tree.

### Files and symbols touched

- `abstractcore/capabilities/vision_catalog.py`
- `abstractcore/capabilities/__init__.py`
- `abstractcore/server/vision_endpoints.py`
- `tests/capabilities/test_vision_catalog_helper.py`
- `tests/server/test_server_capability_catalog_routes.py`
- `tests/test_import_safety.py`
- `../abstractruntime/src/abstractruntime/integrations/abstractcore/discovery_queries.py`
- `../abstractruntime/tests/test_abstractcore_discovery_facade.py`
- `pyproject.toml`
- `tests/test_packaging_extras.py`
- `abstractcore/utils/version.py`
- `CHANGELOG.md`

### Validation

- `pytest -q tests/capabilities/test_vision_catalog_helper.py tests/server/test_server_capability_catalog_routes.py tests/test_import_safety.py tests/test_packaging_extras.py`
  - Result: `48 passed`
- `pytest -q ../abstractruntime/tests/test_abstractcore_discovery_facade.py`
  - Result: `17 passed`
- `python -m compileall abstractcore/capabilities/vision_catalog.py abstractcore/server/vision_endpoints.py tests/capabilities/test_vision_catalog_helper.py tests/server/test_server_capability_catalog_routes.py tests/test_import_safety.py`
  - Result: clean
- `python -m compileall ../abstractruntime/src/abstractruntime/integrations/abstractcore/discovery_queries.py ../abstractruntime/tests/test_abstractcore_discovery_facade.py`
  - Result: clean
- `PYTHONPATH=../abstractvision/src python - <<'PY' ...`
  - Result: local `abstractvision` source reports `0.3.8`, exports `VisionModelCapabilitiesRegistry`, and instantiates the registry successfully.

### Residual risks

- The new helper intentionally preserves the current local cache heuristics. If Core later wants to
  consolidate shared cache-path utilities, that should be a separate follow-up rather than being
  hidden inside this boundary cleanup.
- The helper remains synchronous by design. Async hosts should continue to offload it when they do
  not want local filesystem inspection on the main thread.

### Follow-ups

- No immediate follow-up is required for Core. Downstream hosts can now adopt the public helper
  without reaching into `abstractcore.server.*`.
