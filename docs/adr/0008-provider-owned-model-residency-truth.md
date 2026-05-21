# ADR 0008: Provider-owned model residency truth

Status: Accepted.

## Context

Model residency is a provider-owned fact. Core can know it when a provider or
capability implementation can verify the backing runtime state; higher layers
cannot derive it safely from configuration or transport shape.

The failure mode is concrete: an LM Studio default model can be configured and a
client can exist while LM Studio has no loaded model instance. Reporting that as
`loaded=true` misleads Runtime, Gateway, Flow, and operators.

## Decision

AbstractCore owns provider and capability model-residency truth.

The accepted rules are:

- Text providers expose verified provider residency through a public
  provider-owned contract, currently `get_model_residency(...)`.
- Provider-native APIs, load/warm hooks, loaded-instance probes, and unload
  semantics stay inside provider implementations or capability plugins.
- `/acore/models/*` adapts provider/capability residency contracts; it must not
  treat gateway client cache, provider availability, model catalogs, provider
  names, ports, or `base_url` values as loaded-model proof.
- `/acore/models/load` may call a provider-owned `load_model(...)` hook when a
  provider can actively warm remote/server-side residency, but the response is
  still verified through `get_model_residency(...)`.
- A provider that cannot verify residency returns an explicit unknown result:
  `provider_residency_verified=false`, `provider_resident=null`, and
  `loaded=false`.
- Capability residency for image, TTS, and STT remains capability-owned and is
  relayed by the server routes.

## Consequences

### Positive

- Runtime and hosts can consume one Core-owned truth contract.
- Provider-specific behavior stays near provider code where it can be tested.
- Unknown provider residency no longer appears as a loaded model.

### Negative

- Providers without residency support cannot produce positive loaded state until
  they implement the contract.
- Server and Runtime responses need to carry both cache state and provider
  residency state.

### Neutral

- This ADR does not require every provider to support positive residency truth.
- This ADR does not make model catalog availability a residency signal.

## Enforcement

- Reviews should reject Runtime, Gateway, or Flow implementations that query
  provider-native residency endpoints directly.
- Reviews should reject provider-name or `base_url` heuristics as loaded-state
  evidence.
- New provider-specific residency probes belong in provider implementations or
  capability plugins behind the public contract.

## Validation

- Provider tests should cover loaded, not-loaded, and unknown residency results.
- Provider tests should cover native load/warm behavior when a provider exposes
  it.
- Server tests should prove `/acore/models/*` reports cache state separately
  from provider residency.
- Runtime-facing tests should prove consumers get only Core-owned provider truth.

## Related

- `abstractcore/core/interface.py`
- `abstractcore/providers/lmstudio_provider.py`
- `abstractcore/providers/ollama_provider.py`
- `abstractcore/server/app.py`
- `docs/adr/0003-provider-capability-and-output-boundaries.md`
- `docs/adr/0005-source-first-quality-fixes.md`
- `docs/backlog/completed/0798_provider_owned_text_residency_truth_contract.md`
