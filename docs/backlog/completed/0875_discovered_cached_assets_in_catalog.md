# 0875-abstractcore: [IMPROVEMENT] Discovered cached assets in the catalogue

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: completed
> Completed: 2026-10-30
> Release: unreleased (local deployment verified)
> Priority: P1
> Labels: catalog, ux, seat-framework
> Work-item id: abstractcore-0875
> Owner: framework
> Thread: unavailable (Agora connection refused)
> Hub refs: none

## Current code reality and scope

The seed/registry join leaves 124 observed local cached repositories outside catalog(), despite list_installed exposing them. Include unmatched installed/cached assets as first-class discovered rows, preserving exact identities, bytes, kind/tasks/provenance and normal checkpoint probes. Distinguish adapters/components/unknown from standalone models, avoid fitting or recommending incomplete/supporting assets as runnable models, and retain local file management. All inventories must be accounted for without claiming unknown capabilities. Preserve query/engine/tag/fit behavior and avoid duplicate curated aliases or redundant Hub enrichment; no runtime downloads or inference.

## Validation

Offline real-cache fixtures for full/partial models, LoRA, ControlNet, encoders,
unknown files and duplicate aliases; actual local inventory coverage and timing;
light/dark/mobile browser checks for typed cards, filtering and safe operations.
Scoped commits, apply only task patches to the real Core deployment checkout,
authorized graceful restart, live verification and durable evidence. No new
package publication or service persistence. Existing Core/UI ownership applies.

- [x] Implement and verify the catalogue correction.
- [x] Commit, deploy locally and record completion.

## Completion report

Core `17662521` implements first-class discovered catalogue rows, local kind/task
provenance, safe cache completion metadata and shard-index checks. The actual
Core deployment worktree received only this task patch as `8f2fd00b`. Curated
identities are preserved and exact local references remain manageable. This
closes the HF cache gap left by [0873](0873_installed_model_catalog_identity.md)
and [0874](0874_local_model_registry_completion.md).

161 Core tests passed both in source and in the deployed worktree; 35 Gateway
unit checks and 240 browser assertions passed. Actual local inventory: all 162
entries covered, zero unmatched; base catalogue 224 rows including 122
discovered assets, built in 0.855 seconds. Healthy graceful redeploy verified
in dark/light/390px/320px/touch, with no live model-management writes.
[Review and evidence](../../../../abstractgateway/design/catalog-assets-1000/REVIEW.md).

File/index checks identify 26 incomplete caches. Local metadata leaves 26 assets
unclassified; their capabilities/memory needs are not guessed. Existing unknown
metadata remains explicit, with inspection and dry-run deletion preserved.
No new ADR impact, registry publication, model loading/downloads or persistence.
No residual implementation work is required; richer metadata can improve
classification when available. Agora was unavailable, so receipts remain local.
