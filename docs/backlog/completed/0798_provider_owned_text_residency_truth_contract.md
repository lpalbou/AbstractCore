# Proposed: Provider-owned text residency truth contract

## Metadata
- Created: 2026-05-21
- Status: Completed
- Completed: 2026-05-21

## ADR status
- Governing ADRs:
  - `docs/adr/0003-provider-capability-and-output-boundaries.md`
  - `docs/adr/0005-source-first-quality-fixes.md`
  - `docs/adr/0008-provider-owned-model-residency-truth.md`
- ADR impact: Created ADR 0008 because provider-owned residency truth is now a durable Core boundary rule.

## Context

Runtime and host UIs need to distinguish three different states for text-generation models:

- a configured default provider/model;
- a cached Runtime or gateway provider client;
- a model that is actually resident in the backing provider runtime.

For in-process providers, a cached provider object can be meaningful residency evidence. For
provider-server-backed implementations, including OpenAI-compatible local servers, Runtime cannot
derive loaded-instance truth without crossing the provider boundary. That truth belongs in
AbstractCore provider code and should be surfaced through an AbstractCore-owned contract.

## Current code reality

Inspected on 2026-05-21:

- `abstractcore/core/interface.py`
  - exposes `get_model_residency(...)`;
  - the default implementation is explicit unknown/fail-closed, not loaded.
- `abstractcore/providers/registry.py`
  - exposes `ProviderRegistry`, `ProviderInfo`, and metadata such as `local_provider`;
  - metadata is not residency truth and must not be consumed as loaded-state evidence.
- `abstractcore/server/app.py`
  - exposes `/acore/models/load`, `/acore/models/loaded`, and `/acore/models/unload`;
  - text-generation residency now reports gateway client cache state separately from provider-owned residency truth;
  - non-text residency delegates to capability-owned hooks where available.
- `abstractcore/providers/lmstudio_provider.py`
  - owns LM Studio native REST knowledge for load, unload, and loaded-instance resolution;
  - exposes loaded-instance truth through `get_model_residency(...)`.
- `abstractcore/providers/mlx_provider.py` and `abstractcore/providers/huggingface_provider.py`
  - expose in-process loaded state through `get_model_residency(...)`.
- `abstractcore/providers/ollama_provider.py`
  - owns Ollama server API access for model discovery, native `keep_alive` load/unload, and
    `/api/ps` running-model truth;
  - exposes loaded-instance truth through `get_model_residency(...)`.
- `abstractcore/providers/openai_compatible_provider.py`
  - owns generic OpenAI-compatible HTTP behavior and model discovery;
  - does not have a generic way to tell callers whether the backing service has loaded the requested model.
- `../abstractruntime/src/abstractruntime/integrations/abstractcore/llm_client.py`
  - must not call provider-specific HTTP APIs directly;
  - consumes the public `get_model_residency(...)` provider contract when a local Core provider object is cached;
  - reports cached clients without verified Core provider truth as `loaded=false`.

## Problem or opportunity

Runtime, Gateway, Flow, and similar hosts need truthful loaded-model state without duplicating
provider-specific knowledge. If each host decides by provider name, by default port, or by scraping
native provider APIs, the system gets multiple sources of truth and provider behavior drifts.

The missing piece is a Core-owned text provider residency contract that provider implementations can
support when they have reliable loaded-instance truth and can decline when they do not.

## Proposed direction

Broaden and stabilize the narrow provider-owned contract for text-generation residency, consumed by
`/acore/models/*` and higher layers.

Candidate shape:

- the provider method or shared helper returns a mapping with stable fields such as:
  - `provider_residency_verified`
  - `provider_resident`
  - `provider_instance_ids`
  - `provider_residency_source`
  - `state`
  - `warnings`
- a provider capability flag or registry metadata field that tells hosts whether provider-side
  residency truth is supported, unsupported, or not applicable;
- gateway `/acore/models/loaded` support that merges gateway client cache state with provider-owned
  loaded-instance truth without overwriting either state;
- gateway `/acore/models/load` support that calls provider-owned load/warm hooks when available,
  then verifies the result through the same residency contract;
- conservative default behavior: if a provider does not implement this contract, callers get
  unverified cache/configuration state, not `loaded=true`.

The contract should be provider-generic from the caller perspective. LM Studio, Ollama, vLLM, and
other provider-server-backed implementations can hide their native details behind provider-owned
methods.

## Non-goals

- Do not make Runtime, Gateway, or Flow query provider-specific native endpoints directly.
- Do not build provider-name lists in Runtime or host UIs to infer residency semantics.
- Do not treat provider availability or model catalog membership as proof that a model is loaded.
- Do not force every provider to implement positive loaded-instance truth before exposing a
  conservative unsupported/unverified result.

## Promotion criteria

Promote when hosts need additional provider coverage, capability flags, or stronger response-shape
guarantees beyond the initial `get_model_residency(...)` hook.

This should also be promoted before adding any new Runtime or Gateway behavior that would otherwise
need to infer provider residency from provider names, ports, or private provider implementation
details.

## Validation ideas

- Provider unit tests for an implementation that can report loaded and not-loaded states without
  requiring a real server.
- Server `/acore/models/loaded` tests showing both gateway client cache state and provider-side
  residency truth remain visible and distinct.
- Runtime-facing contract tests proving hosts consume only the public AbstractCore fields.
- Negative tests proving model catalog availability does not imply `provider_resident=true`.

## Guidance for future agents

Start in provider code and the registry/server contract, not in Runtime. Preserve the boundary:
provider-specific probes belong to provider implementations; Runtime and host UIs should only
consume AbstractCore-owned fields.

## Completion report (2026-05-21)

This item is completed as an implementation record: the provider-owned text residency contract and
server control plane described above landed in AbstractCore.

What shipped (key points):
- Provider-owned `get_model_residency(...)` contract is present and fail-closed by default
  (`provider_residency_verified=false`, `provider_resident=null`, `loaded=false`).
- Verified implementations exist for:
  - LM Studio (native REST loaded-instances; native load/unload);
  - Ollama (`/api/ps` running-model truth; native keep-alive load/unload);
  - MLX and HuggingFace in-process backends (object-resident truth);
  - HuggingFace unload correctness fixed so residency flips to not-loaded after unload.
- Server `/acore/models/*` now separates gateway/runtime cache state from provider-resident truth
  and uses provider-owned load hooks when available, verifying residency via the same contract.

Evidence / validation:
- Provider + server unit coverage for residency/load/unload behavior.
- Full hermetic test suite run after the changes: `1488 passed, 243 skipped` (2026-05-21).

Residual gap (intentional):
- `openai_compatible_provider` cannot generically verify provider-side residency; it continues to
  report fail-closed/unknown rather than guessing.
