# ADR 0003: Provider, capability, and output boundaries

Status: Accepted.

## Context

AbstractCore has multiple abstraction layers that are easy to blur:

- text providers created by `create_llm(...)` and the provider registry;
- OpenAI-compatible protocol families implemented through a shared provider base;
- optional generated-media capabilities discovered through `CapabilityRegistry`;
- public output selectors normalized by `abstractcore.core.output_specs`;
- server runtime control through `/acore/models/*`.

The current code is strongest when each layer owns one contract. Provider construction flows through
`ProviderRegistry`, OpenAI-compatible request logic is centralized in
`OpenAICompatibleProvider`, capability plugins expose dict-shaped residency hooks, and output
selector helpers live in Core rather than private provider code.

## Decision

AbstractCore will keep explicit ownership boundaries for provider, capability, and output contracts.

The accepted boundaries are:

- `create_llm(...)` and `ProviderRegistry` are the canonical provider construction and metadata
  entry points.
- Shared wire-protocol behavior belongs in shared provider bases before adding thin subclasses.
- Provider-specific extensions remain in provider subclasses or provider-owned hooks.
- Optional voice, audio, vision, and music behavior remains behind capability plugins.
- Capability residency and plugin control surfaces use mapping/dict-shaped payloads to avoid hard
  dependencies on plugin implementation packages.
- `abstractcore.core.output_specs` owns public output-selector normalization and dispatch semantics.
- Server routes adapt over Core/provider/capability contracts instead of creating parallel sources
  of truth.

## Consequences

### Positive

- New providers and plugins have a clear integration path.
- Protocol-family duplication is easier to catch.
- Downstream runtimes can depend on public Core helpers instead of private provider details.

### Negative

- Some provider-specific features require explicit exceptions instead of quick shared-base patches.
- Capability APIs remain intentionally asymmetric where plugin packages expose different semantics.

### Neutral

- This ADR does not require every provider to inherit from the same immediate class.
- This ADR does not make optional capability plugins mandatory dependencies.

## Enforcement

- New providers should register through `ProviderRegistry`.
- New OpenAI-compatible providers should justify duplicated protocol logic.
- New output selectors or task aliases should start in `abstractcore.core.output_specs`.
- New capability residency routes should extend the task-aware `/acore/models/*` family unless a
  separate ADR accepts a different control plane.
- Reviews should reject server-only provider/capability tables that drift from Core metadata.

## Validation

- Provider registry and factory tests should cover new provider entries.
- Output selector tests should cover new selector aliases or dispatch behavior.
- Capability residency tests should cover any new plugin control-plane shape.
- Server tests should verify that public routes call the Core/provider/plugin contract they claim to
  expose.

## Related

- `abstractcore/core/factory.py`
- `abstractcore/providers/registry.py`
- `abstractcore/providers/openai_compatible_provider.py`
- `abstractcore/core/output_specs.py`
- `abstractcore/capabilities/registry.py`
- `abstractcore/capabilities/types.py`
- `abstractcore/server/app.py`
- `docs/backlog/completed/2026-05-07_public-output-selector-contract.md`
- `docs/backlog/completed/2026-05-19_generalize_acore_models_residency.md`
