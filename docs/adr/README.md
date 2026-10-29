# ADR Index

This directory records durable architecture decisions for AbstractCore.

Accepted ADRs are live project policy. Backlog items describe work to do; ADRs describe the rules
that future work must respect.

## Current ADRs

| ID | Status | Title | Purpose |
| --- | --- | --- | --- |
| [0001](0001-engineering-guardrails-and-no-silent-degradation.md) | Accepted | Engineering guardrails and no silent degradation | Makes fallback, truncation, timeout, and degraded behavior explicit. |
| [0002](0002-validation-and-evidence.md) | Accepted | Validation and evidence | Requires risk-proportional tests, docs, and completion evidence. |
| [0003](0003-provider-capability-and-output-boundaries.md) | Accepted | Provider, capability, and output boundaries | Defines the source of truth for providers, capability plugins, and output selectors. |
| [0004](0004-operator-control-and-server-trust-boundary.md) | Accepted | Operator control and server trust boundary | Protects server-held credentials, routing overrides, and local/remote access controls. |
| [0005](0005-source-first-quality-fixes.md) | Accepted | Source-first quality fixes | Requires fixes at the producer or contract boundary before downstream cleanup. |
| [0006](0006-text-generation-adapter-lifecycle.md) | Accepted | Text-generation adapter lifecycle | Defines how LoRA-style text adapters should become first-class Core behavior. |
| [0007](0007-durable-memory-bloc-cache-binding.md) | Accepted | Durable memory bloc cache binding | Defines exact local bloc artifact binding across Python, endpoint, gateway, and provider hooks. |
| [0008](0008-provider-owned-model-residency-truth.md) | Accepted | Provider-owned model residency truth | Requires model loaded-state truth to come from provider/capability contracts. |
| [0009](0009-model-handle-fidelity.md) | Accepted | Model handle fidelity | Forbids loading a different model artifact than the handle names; amends 0001. |
| [0010](0010-model-residency-fit-and-eviction.md) | Accepted | Model residency fit and eviction | A load checks weights + KV against free memory, evicts unlocked LRU models, else refuses with a typed error and smaller suggestions. |
| [0011](0011-shared-weights-and-concurrent-requests.md) | Proposed | Shared weights and concurrent requests | One weights object per model per process for every request path; a per-model slot scheduler (bounded, fair FIFO, visible queued state, per-request cancel) in the process that holds the weights; slots fit-checked under 0010, never a silent context reduction. |

## How To Use This Set

- Cite ADRs in backlog items and implementation notes when changing a governed boundary.
- Update the relevant ADR when the accepted boundary changes; do not leave drift implicit.
- Keep ordinary sequencing, task checklists, and completion logs in `docs/backlog/`.
- Add a new ADR only when the decision should constrain more than one implementation task.
