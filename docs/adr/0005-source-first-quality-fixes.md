# ADR 0005: Source-first quality fixes

Status: Accepted.

## Context

AbstractCore sits between users, providers, local runtimes, capability plugins, and HTTP clients.
When behavior is wrong, it can be tempting to patch the final response, add a fallback in the
server, or normalize downstream until tests pass. That can hide the true contract violation and
leave the next caller with the same broken source state.

This repo already has several places where source ownership matters: output selector semantics
belong to `abstractcore.core.output_specs`, provider construction belongs to the registry/factory,
capability residency belongs to plugin facades, and server credential policy belongs to the server
hardening layer.

## Decision

Quality, correctness, routing, parsing, and generated-output problems should be fixed at the
producer or contract boundary first.

Use downstream cleanup only when the source is external, untrusted, provider-specific, or not fully
controllable. When downstream cleanup is necessary, it must be explicit and bounded.

Examples of source-first ownership:

- Output-selector drift is fixed in `abstractcore.core.output_specs`, not by duplicating private
  provider helpers.
- Provider metadata drift is fixed in `ProviderRegistry` or provider implementations, not by adding
  server-only tables.
- Capability residency drift is fixed in capability facades or plugin contracts, not by making
  `/acore/models/*` guess plugin internals.
- Server trust violations are fixed in shared server guard helpers, not by masking one route.

## Consequences

### Positive

- Contract bugs stay fixed for every caller, not just one endpoint.
- Tests can assert the real invariant instead of one downstream normalization.
- Future work has clearer ownership when behavior crosses layers.

### Negative

- Some fixes require touching a deeper shared layer and therefore need broader tests.
- External provider quirks still require bounded adapters where the source cannot be changed.

### Neutral

- This ADR does not forbid response normalization.
- This ADR does not require rewriting external provider behavior.

## Enforcement

- Reviews should ask where the contract is produced before accepting downstream cleanup.
- New compatibility shims must name the external or provider-specific reason they cannot be fixed at
  the source.
- Shared helper modules should be preferred over repeated route/provider-local cleanup for the same
  contract.
- Backlog items that intentionally defer a source fix should record the residual risk.

## Validation

- Tests should cover the producer or shared contract when possible.
- Downstream-only tests are acceptable when the source is external, but they should assert explicit
  bounded behavior.
- Regression tests for source-first fixes should fail if a future change reintroduces a duplicate
  source of truth.

## Related

- `abstractcore/core/output_specs.py`
- `abstractcore/providers/registry.py`
- `abstractcore/capabilities/registry.py`
- `abstractcore/server/app.py`
- `docs/backlog/completed/2026-05-07_public-output-selector-contract.md`
- `docs/backlog/completed/2026-05-08_capability_plugin_catalog_discovery_routes.md`
