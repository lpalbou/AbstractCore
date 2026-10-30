# 0874-abstractcore: [IMPROVEMENT] Complete local model capabilities and architecture metadata

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: planned
> Priority: P1
> Labels: ux, catalog, seat-framework
> Work-item id: abstractcore-0874
> Owner: framework
> Thread: unavailable (Agora connection refused)
> Hub refs: none

## Current code reality and scope

Audit the reported installed Spark X2.6, LFM3, Bonsai, Gemma QAT, Qwen, LFM2 and embedding variants against capabilities, architecture and downloads registries. Existing Qwen/LFM2/Gemma entries must be reused with accurate aliases; genuinely new families and fine-tunes need explicit entries and primary-source provenance. Inspect official model cards/config/templates before adding facts; no guessed tool/vision/reasoning support. Validate schema, lookup, format and catalogue coverage. Core 0873 consumes these identities.

No registry publication or model downloads/inference required. Existing shared
ownership applies; record ADR impact at completion. Commit and authorized
local redeploy, retain validation and review evidence.

- [ ] Implement and verify the requested behavior.
- [ ] Commit, deploy locally and record completion evidence.
