# 0876-abstractcore: [IMPROVEMENT] Adapter catalogue metadata

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: completed
> Completed: 2026-10-30
> Release: unreleased (local deployment verified)
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

- [x] Implement and validate the correction.
- [x] Commit, redeploy and retain review evidence.

## Completion report

Core `0062fa69` adds `adapter_metadata.schema.json` and a normalized adapter
descriptor from local configs/cards, with declared bases, task, trigger words,
target modules, rank and field provenance. Explicit adapter relations/configs
remain separate from full fine-tuned checkpoints; Hub file packaging identifies
remote adapters too. The existing AbstractVision `vision_adapter_capabilities.json`
and `adapter_capabilities.py` already own inference profiles, so they were
preserved rather than duplicated. Assets README documents that boundary.

168 Core checks passed in source and in the real deployment worktree (selective
cherry-pick `449aefce`). Gateway 38 unit checks / 248 browser assertions passed.
Actual local inventory exposes 44 adapter descriptors, 41 with declared bases;
no trigger words are invented where cards omit them. Healthy graceful redeploy,
URL reload, dark/light/390px/320px/touch checks pass with no model writes.
[Review and evidence](../../../../abstractgateway/design/model-types-1001/REVIEW.md).

This corrects the ordinary-list scope from [0875](0875_discovered_cached_assets_in_catalog.md)
while keeping complete inventory available. No new ADR impact or publication;
metadata declarations do not establish runtime compatibility. Existing unknowns
remain explicit; no required implementation work remains. Agora unavailable,
receipts recorded locally. Unrelated changes preserved.
