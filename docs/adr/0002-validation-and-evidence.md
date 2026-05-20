# ADR 0002: Validation and evidence

Status: Accepted.

## Context

AbstractCore is infrastructure code. A small change can affect Python callers, OpenAI-compatible
HTTP routes, provider implementations, capability plugins, docs, and examples. The repository also
uses backlog files as planning memory, so there is a real risk that future work implements what an
old note says instead of what the code now does.

The repo already contains good evidence patterns: completed backlog items record targeted tests,
server docs describe trust boundaries, and several tests cover provider registry, output selectors,
capability residency, and server hardening. This ADR makes that standard explicit for future work.

## Decision

Every meaningful change must carry validation proportional to its risk and surface area.

The expected validation can be small, but it must be deliberate:

- Local implementation changes need focused unit tests or a clear reason tests are not practical.
- Shared provider, server, capability, config, or public-contract changes need targeted tests across
  the touched boundary.
- Documentation-only changes that assert behavior must be checked against code or tests.
- Completion notes for architecture-significant work must say what was validated and what remains
  unverified.
- Backlog items are planning memory. Code and current docs remain the operational source of truth.

## Consequences

### Positive

- Future maintainers can audit why a change was considered safe.
- Docs and backlog items become less likely to drift away from implementation.
- Riskier work gets broader validation without forcing heavy test runs for every small edit.

### Negative

- Architecture-significant changes carry a modest reporting burden.
- Some backlog items may need cleanup before they can be implemented safely.

### Neutral

- This ADR does not require one universal test command for every task.
- This ADR allows explicit residual risk when a live provider, GPU runtime, or optional plugin is
  unavailable.

## Enforcement

- Planned work that changes durable behavior should link the governing ADR or state that no ADR is
  needed.
- Reviews should ask for targeted validation when a change touches shared contracts.
- Completion reports should include commands run, skipped checks, and residual risk.
- Stale docs or backlog statements found during implementation should be fixed or called out.

## Validation

- Audit completed backlog items for validation sections before treating them as implementation
  evidence.
- Prefer targeted `pytest` runs for changed code paths.
- Use `python -m compileall` or import smoke checks for broad docs/interface edits when full test
  suites are impractical.

## Related

- `docs/backlog/README.md`
- `docs/backlog/overview.md`
- `docs/backlog/completed/2026-05-19_generalize_acore_models_residency.md`
- `tests/server/test_server_loaded_runtime_control_plane.py`
- `tests/test_output_specs.py`
