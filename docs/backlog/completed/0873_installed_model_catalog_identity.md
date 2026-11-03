# 0873-abstractcore: [IMPROVEMENT] Robust installed-model identity and catalogue integration

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: completed
> Completed: 2026-10-30
> Release: unreleased (local deployment verified)
> Priority: P1
> Labels: ux, catalog, seat-framework
> Work-item id: abstractcore-0873
> Owner: framework
> Thread: unavailable (Agora connection refused)
> Hub refs: none

## Current code reality and scope

catalog_id_for only indexes curated provider/artifact pairs and provider-specific suffix matching. Known registry models served by LM Studio are left outside the catalogue when the seed lacks that engine or precision. Resolve canonical identity through explicit registry aliases and authoritative engine metadata; append exact installed variants with accurate presence, size, quantization and safe operation identifiers. Preserve fine-tune, QAT/base and precision distinctions. Test positive and adversarial identity cases, actual local inventory and performance.

No registry publication or model downloads/inference required. Existing shared
ownership applies; record ADR impact at completion. Commit and authorized
local redeploy, retain validation and review evidence.

- [x] Implement and verify the requested behavior.
- [x] Commit, deploy locally and record completion evidence.

## Completion report

Implemented `62581e63` / `8afe7771`: exact registry-backed identities, engine-path
provenance, installed variants under canonical cards and explicit precision
separation. No partial-match base-model grouping of unknown fine-tunes. HF caches
retain checkpoint probes; actual engine identifiers remain operation targets.
All 23 observed LM Studio artifacts map and are reachable in live cards, with no
local non-catalogue extras. Source Core gate 165 plus follow-up identity gate 35;
actual deployment gate 132. Gateway 36 tests and 214 browser assertions pass.
Deployed only these patches into the runtime Core checkout as `e42d5146` /
`a2fd2b10`; healthy graceful restart and live dark/light/320px checks pass.
[Cross-repo review](../../../../abstractgateway/design/model-library-catalog-0998/REVIEW.md),
[validation](../../../../abstractgateway/design/model-library-catalog-0998/validation.json).
No model operations or registry publication. No ADR change: existing Core ownership
is preserved. No residual within the reported LM Studio inventory; unrelated HF
adapters/caches remain conservative unknowns. Agora unavailable.
