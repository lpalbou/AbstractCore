# 0849-abstractcore: [IMPROVEMENT] Enforce shared speculation negotiation across providers

> Created: 2026-09-20
> Status: Planned
> Type: improvement
> Priority: P1
> Labels: native-inference, speculation, capability-contract

## Summary

Make the existing provider-neutral speculation request/outcome contract enforceable
for all local providers, while keeping backend-specific execution in adapters.

## Current code reality

Audited 2026-09-20: `providers/speculation.py` normalizes off/native_mtp, drafter,
depth and strict execution requirements and defines outcomes. MLX consumes it.
BaseProvider preserves the typed unavailable exception but does not uniformly
negotiate requests. HF/Transformers and in-process GGUF have no native MTP adapter.
A registry entry or config `mtp` key is not evidence of executable head weights.

## Scope and non-goals

Centralize request validation, provider/artifact/instance negotiation and honest
unsupported results, using the existing vocabulary. Keep tensor operations,
head binding, draft/verify and cache rollback in backend adapters. Distinguish
absent/inherit from explicit off; preserve per-call isolation. No universal tensor
loop in BaseProvider, renamed controls, remote-server requirement or claim that
executing speculation guarantees a measured speedup.

## Acceptance criteria

- [ ] Every local provider handles explicit speculation rather than losing it in kwargs.
- [ ] Strict unsupported requests raise typed actionable errors; optional fallback is named in response metadata.
- [ ] Capability reports distinguish weights present, implementation supported, instance ready and request actually used.
- [ ] Invalid depth/options fail before expensive generation; off remains a real override.
- [ ] MLX execution and metadata remain backward compatible.

## Testing

Run `python -m pytest tests/providers/test_mtp_adv_capability_wire_contract_unit.py tests/providers/test_mtp_native_request_adversarial_unit.py`.
Add a parametrized local-provider negotiation suite covering unsupported adapters,
constructor defaults, bool/dict controls, strictness, streaming and concurrent isolation.

## Dependencies and ADR status

Unblocks 0850/0851 and informs 0853; Runtime transport fix is abstractruntime-0846.
ADR impact: review existing provider/capability ownership ADRs and extend the
appropriate ADR before engraving a new public adapter protocol. Expected outcome:
one request contract, honest support, independently testable backend drivers.
