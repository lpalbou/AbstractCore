# 0873-abstractcore: [IMPROVEMENT] Robust installed-model identity and catalogue integration

> Package: abstractcore
> Type: improvement
> Created: 2026-10-30
> Status: planned
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

- [ ] Implement and verify the requested behavior.
- [ ] Commit, deploy locally and record completion evidence.
