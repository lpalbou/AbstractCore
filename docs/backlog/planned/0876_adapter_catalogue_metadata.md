# 0876-abstractcore: [IMPROVEMENT] Adapter catalogue metadata

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: planned
> Priority: P1
> Labels: catalog, ux, seat-framework
> Work-item id: abstractcore-0876
> Owner: framework
> Thread: unavailable (Agora connection refused)
> Hub refs: none

## Current code reality and scope

0875/1000 made all cache assets first-class catalogue rows, but ordinary browsing
and Downloaded currently include adapters as peer model cards. Operator feedback
requires full models by default; advanced users explicitly select other asset types.

Normalize adapter type, declared base checkpoints, task, triggers and tuning metadata from cached adapter configs/cards into a schema-backed catalogue descriptor. Keep adapters distinct from standalone models and preserve exact identity. The equivalent inference registry already exists in AbstractVision (vision_adapter_capabilities.json); do not duplicate or relocate its runtime recommendations. Expose provenance and unknown fields without inferring compatibility from names. Add the descriptor/schema and document the boundary.

## Validation

Offline local-cache fixtures and schema validation; default/Downloaded/full-model
filter tests, advanced facets/counts and URL restoration, mixed inventories,
keyboard and light/dark/mobile browser checks. Scoped commits and authorized
local redeploy with actual Core worktree, healthy graceful restart and live
verification. No model writes or new persistent service installation.

- [ ] Implement and validate the correction.
- [ ] Commit, redeploy and retain review evidence.
