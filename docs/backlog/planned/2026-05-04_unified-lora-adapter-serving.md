# Planned: Unified text-generation adapter lifecycle and hot-switch selection

## Metadata
- Created: 2026-05-04
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0006-text-generation-adapter-lifecycle.md`
- ADR impact: Governed by accepted adapter lifecycle policy.

## Context

AbstractCore already abstracts providers, models, structured output, tool calling, and a shared
server/runtime control plane. Adapter-backed text generation is the clear missing layer in that
abstraction.

The desired user story is still the same:

- keep a base text model warm once;
- load one or more LoRA-style adapters by name;
- select a specific adapter for one generation request without replacing the base runtime;
- unload adapters when they are no longer needed;
- expose capability and state clearly so callers know whether adapter lifecycle and request-time
  selection are supported.

## Current code reality

- `abstractcore/providers/vllm_provider.py` implements provider-specific `load_adapter`,
  `unload_adapter`, and `list_adapters` methods, but they return simple strings/lists and are not
  part of the shared provider contract.
- `abstractcore/core/interface.py` and `abstractcore/providers/base.py` expose no adapter lifecycle
  or request-time adapter selection API. The only canonical lifecycle hook today is
  `unload_model(model_name)`.
- `abstractcore/providers/openai_compatible_provider.py` always sends `{"model": self.model}` for
  chat-completions requests. The shared provider path does not honor a per-call `model` or
  `adapter` override.
- `abstractcore/server/app.py` keeps loaded text runtimes in `_GATEWAY_LOADED_RUNTIMES` keyed by
  provider + base model + base URL and exposes `/acore/models/*` only for base model residency.
  There is no text adapter subresource or text adapter residency state.
- `docs/prerequisites.md` documented `llm.generate(..., model="sql-expert")` after
  `load_adapter(...)`, but that is not implemented as a portable Core behavior and was stale at the
  time of this audit.

## Problem

The repository currently has a partial, provider-specific adapter story instead of a first-class
Core capability:

- vLLM lifecycle support exists but is stranded behind provider-specific methods;
- there is no portable request-time hot-switch contract for text adapters;
- the server's existing residency/control plane cannot manage text adapters;
- capability metadata does not tell callers whether lifecycle or hot-switch selection is supported;
- documentation can imply support that the shared provider path does not actually guarantee.

## What we want to do

Make text-generation adapter lifecycle and request-time adapter selection first-class Core behavior,
starting with vLLM and phrasing the surface so future PEFT/Hugging Face, MLX, or LoRAX-style
backends can implement the same contract.

## Why

- LoRA adapters are the natural middle ground between one fixed base model and full base-model
  process churn.
- The repo already has the architecture pieces needed for a clean design:
  - a shared provider contract;
  - a text runtime registry (`/acore/models/*`);
  - provider metadata discovery;
  - request-time generation kwargs passed through Core providers.
- Without a first-class contract, users either stay locked to provider-specific methods or depend on
  brittle model-id tricks that do not match how the shared provider path works today.

## Requirements

- The shared Python contract must expose adapter lifecycle support without breaking every provider
  subclass. Optional concrete methods or a protocol-style surface are acceptable; new abstract
  methods on every provider are not.
- Unsupported providers must fail explicitly instead of silently ignoring adapter requests.
- Request-time adapter selection must use an explicit field or kwarg such as `adapter`; it must not
  rely on `model="adapter-name"` as the portable contract.
- Adapter lifecycle results must be structured and stable enough for both Python callers and server
  responses.
- The server must expose trusted control-plane operations for text adapters tied to a base text
  runtime, not pretend that an adapter is a separate base model.
- Provider and server metadata must advertise at least:
  - whether adapter lifecycle is supported;
  - whether request-time hot-switch selection is supported;
  - adapter kind(s), starting with LoRA.
- Existing behavior must remain unchanged unless adapter lifecycle or request-time selection is
  explicitly used.

## Suggested implementation

### 1. Add shared adapter types and default behavior

Introduce small shared types such as:

- `AdapterInfo`
- `AdapterOperationResult`
- `AdapterCapabilities`

Then add optional default methods on the shared text-provider surface, for example:

- `get_adapter_capabilities()`
- `list_adapters()`
- `load_adapter(...)`
- `unload_adapter(...)`

Default behavior should raise an explicit unsupported error or return a structured unsupported
result, depending on the surface chosen during implementation.

### 2. Add explicit request-time selection

Add an explicit request kwarg/field such as `adapter` for text generation.

Provider implementations may translate that to backend-native behavior:

- vLLM may route it through request model selection or another backend-native payload shape;
- future in-process backends may activate adapters in memory before generation.

The shared contract should not depend on `/v1/models` surfacing pseudo-model ids for adapters.

### 3. Extend the text control plane

Extend the existing model-residency/control-plane surface with a text adapter subresource, for
example:

- `GET /acore/models/adapters`
- `POST /acore/models/adapters/load`
- `POST /acore/models/adapters/unload`

Selectors should reference a loaded text runtime by `runtime_id` or the existing
`provider` + `model` + optional `base_url` shape.

### 4. Implement vLLM first

Normalize the existing vLLM helper methods around the shared contract:

- keep using the vLLM management endpoints for lifecycle;
- add request-time adapter selection support through the shared text generation path;
- return structured results instead of plain success strings.

### 5. Surface capabilities and fix docs

- update provider registry metadata and `/providers`;
- expose adapter capability information on relevant control-plane responses;
- correct docs that currently imply implemented hot-switch behavior through `model=...`.

## Scope

- Shared text-provider adapter lifecycle contract.
- Request-time adapter selection for text generation.
- vLLM first implementation.
- Trusted server control-plane support for text adapters.
- Capability metadata, docs, and tests needed to make the behavior auditable.

## Non-goals

- Training or fine-tuning adapters.
- Arbitrary remote adapter downloads enabled by default.
- Production multi-tenant adapter management without RBAC/audit controls.
- Base model hot-swap. That remains separate work in
  `docs/backlog/planned/2026-05-04_vllm-base-model-swap-orchestration.md`.
- Generalizing to non-text generation capabilities in the first pass.

## Dependencies and related tasks

- `docs/adr/0006-text-generation-adapter-lifecycle.md`
- `docs/backlog/planned/2026-05-04_vllm-base-model-swap-orchestration.md`
- `abstractcore/core/interface.py`
- `abstractcore/providers/base.py`
- `abstractcore/providers/openai_compatible_provider.py`
- `abstractcore/providers/vllm_provider.py`
- `abstractcore/server/app.py`

## Expected outcomes

- Core exposes a documented shared adapter lifecycle surface for text providers.
- vLLM no longer requires callers to drop to provider-specific methods for the supported path.
- Text generation can select a loaded adapter explicitly per request.
- The server can manage text adapters through a trusted control-plane API tied to loaded text
  runtimes.
- Docs and examples stop implying behavior that the shared code path does not implement.

## Validation

- Unit tests for default unsupported adapter behavior and structured adapter result types.
- Provider tests for vLLM adapter lifecycle request shapes and error normalization.
- Server tests for adapter control-plane routing against loaded text runtimes.
- Request-path tests showing explicit adapter selection through generation requests.
- Docs audit ensuring no shared examples still claim `model="adapter-name"` as the portable
  adapter-switch mechanism.

## Progress checklist

- [ ] Add shared adapter types and default provider/interface behavior.
- [ ] Add explicit request-time `adapter` selection for text generation.
- [ ] Normalize `VLLMProvider` adapter lifecycle methods around the shared contract.
- [ ] Extend the trusted server control plane for text adapter lifecycle.
- [ ] Surface adapter capabilities in provider/runtime metadata.
- [ ] Add provider, server, and request-path tests.
- [ ] Update docs and examples to match implemented behavior.

## Guidance for the implementing agent

- Re-check the current `openai_compatible_provider.py` request path before coding. Today it always
  posts `self.model`, and that is the main hot-switch gap.
- Keep the base text runtime and the adapter overlay conceptually separate; do not collapse them
  into one synthetic model identity unless a backend truly requires that internally.
- Preserve backward compatibility for providers and callers that never mention adapters.
- Treat doc drift as a real bug. If examples claim portable adapter behavior before the code lands,
  fix the docs in the same pass.
