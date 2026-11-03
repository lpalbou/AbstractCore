# 0874-abstractcore: [IMPROVEMENT] Complete local model capabilities and architecture metadata

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: completed
> Completed: 2026-10-30
> Release: unreleased (local deployment verified)
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

- [x] Implement and verify the requested behavior.
- [x] Commit, deploy locally and record completion evidence.

## Completion report

Implemented `62581e63`: five capability entries (Spark X2.6, LFM3, Bonsai,
HauhauCS Aggressive and MiniLM), four architecture/format entries and six download
rows including the previously missing Gemma E2B/12B variants. Reuse known Qwen,
Liquid and embedding identities; add explicit Gemma instruction/QAT aliases.
Primary cards/config/templates recorded in assets and the linked review. Distinct
fine-tunes stay separate; format parsers and non-generative embeddings checked;
MTP weights or hard completion caps are not guessed from configuration/examples.
Schema/registry/materializer/catalogue gates pass (132 in the actual deployment
checkout); live catalogue has 102 rows and covers all 23 LM Studio artifacts.
Core deployment `e42d5146` / `a2fd2b10`, Gateway `334dd3bb`, healthy restart and
live dark/light/mobile checks verified without model operations.
[Sources and review](../../../../abstractgateway/design/model-library-catalog-0998/REVIEW.md),
[validation](../../../../abstractgateway/design/model-library-catalog-0998/validation.json).
No ADR impact, registry publication or residual work within the requested set.
Agora unavailable; unrelated HF inventory remains explicitly unclassified.
