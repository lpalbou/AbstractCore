# 0853-abstractcore: [FEATURE] Surface instance concurrency and model MTP controls across clients

> Created: 2026-09-20
> Status: Planned
> Type: feature
> Priority: P2
> Labels: native-inference, capabilities, consoles

## Summary

Expose truthful model/backend/instance execution capabilities and use them for
Core/Gateway console controls and Flow/Code/Assistant request overrides.

## Current code reality

Audited 2026-09-20: existing MLX instances expose `supports_concurrent_generation`,
`speculation_status` and per-response execution/speculation metadata. Core HTTP
accepts per-request speculation; chat CLI has startup depth and on/off controls.
Configuration consoles have no first-class MTP/concurrency controls; their reasoning
choices are static, so do not copy those choices as an allegedly model-derived schema.
Runtime forwarding/lock repairs are the separate immediate item abstractruntime-0846.

## Scope and non-goals

Separate artifact evidence, backend implementation, loaded-instance readiness and
actual request outcomes. Describe continuous/cohort/exclusive scheduling and limits;
MTP supported/default/validated depths, head state, restrictions and reload needs.
Reuse Core config authority through Gateway. Preserve precedence and explicit Off.
Provide capability-driven selectors in consoles/applications and end-to-end tests.
No new provider, duplicate configuration store, automatic head downloads, hard-coded
claim that depths 2–5 work universally, or UI-only completion claims.

## Acceptance criteria

- [ ] Same model on different runtimes can report different actual capabilities.
- [ ] Depth/off changes round-trip through configuration and request overrides without truthiness loss.
- [ ] Load-time scheduler/head settings are distinct from per-call controls and flag required reloads.
- [ ] Unsupported choices explain why; diagnostics show request actuals, not a racy global last-call status.
- [ ] Core/Gateway consoles and each participating application are tested at their real transport boundary.

## Testing

Run `python -m pytest tests/providers/test_mtp_adv_capability_wire_contract_unit.py`
and each touched console's `cargo test`; add Gateway/Runtime/app round-trip tests,
including absence, false, depth changes and unsupported backend cases.

## Dependencies and ADR status

Related 0849, abstractruntime-0846, and existing provider/model capability registries.
ADR impact: review existing capability/residency ownership ADRs before engraving a
new persisted capability schema. Expected outcome: one Core-owned description
drives usable controls, with no unsupported acceleration or concurrency claims.
