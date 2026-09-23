# 0852-abstractcore: [FEATURE] Add optional memory-pressure eviction respecting model locks

> Created: 2026-09-20
> Status: Planned
> Type: feature
> Priority: P2
> Labels: native-inference, residency, memory

## Summary

Add an off-by-default memory-pressure eviction toggle for Core-owned model
residency, reusing existing protection rather than inventing another pin mode.

## Current code reality

Audited 2026-09-20: server `/acore/models/lock` and load `lock=true` protect managed
models; ordinary unload returns 409, explicit `force=true` may override, and
unload-after cleanup skips locks. `pin=true` is a best-effort provider hint, not
the managed protection flag. Managed `ttl_s` remains advisory. Cache LRU/TTL is
not model-weight eviction. Runtime pools also respect locked models during eviction.

## Scope and non-goals

Implement the operator's narrow policy: configurable disabled-by-default pressure
eviction, selecting only idle unlocked owned models. The actual model host owns
the decision; Gateway delegates instead of competing. Protect active/queued work
with leases, account for shared weights only once, reserve transient/OS headroom,
and use hysteresis to avoid thrashing. Never use force to implement automatic
eviction. If all models are protected/busy, queue or fail admission explicitly.
No automatic idle TTL, new protection vocabulary, deleting checkpoint files,
OS-wide process management, mandatory persistence or eviction of unowned processes.

## Acceptance criteria

- [ ] Disabled toggle preserves current behavior; enabled toggle applies bounded admission policy.
- [ ] Locked and leased models cannot be automatically evicted, including racing load/unload/request cases.
- [ ] Unlock restores eligibility; explicit operator force behavior remains unchanged.
- [ ] Reasons, victim identity and reclaimed/estimated memory are observable without claiming uncertain RSS as exact.
- [ ] All-protected and insufficient-memory cases are safe and actionable.

## Testing

Run `python -m pytest tests/server/test_server_model_residency_control_plane.py`.
Add deterministic budget/LRU/lease/race tests and an opt-in two-model memory-pressure
test. Do not touch the operator's running servers or pinned models during tests.

## Dependencies and ADR status

Related shared model-pool item 0847 and existing residency ownership ADR 0008.
ADR impact: extend residency policy documentation before closure; retain Core-host
authority and existing lock semantics. Expected outcome: pressure relief with an
explicit opt-in and no surprise removal of protected models.
