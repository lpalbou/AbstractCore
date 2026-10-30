# 0875-abstractcore: [IMPROVEMENT] Discovered cached assets in the catalogue

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: planned
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

- [ ] Implement and verify the catalogue correction.
- [ ] Commit, deploy locally and record completion.
