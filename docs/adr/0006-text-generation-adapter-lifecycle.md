# ADR 0006: Text-generation adapter lifecycle

Status: Accepted.

## Context

AbstractCore abstracts text generation across multiple providers, but adapter-backed text
generation is only partially represented today.

The concrete code gap is narrow and important:

- `VLLMProvider` has provider-specific `load_adapter(...)`, `unload_adapter(...)`, and
  `list_adapters(...)` helpers.
- The shared provider contract does not expose adapter lifecycle or request-time adapter selection.
- `OpenAICompatibleProvider` sends the provider instance model in chat-completions payloads, so
  hot-switching an adapter per request is not a shared Core behavior.
- The gateway runtime registry and `/acore/models/*` model residency control plane track base text
  runtimes, not text adapter overlays.

This should be fixed for text generation in Core. It should not be generalized to voice or vision
inside Core: those capabilities are plugin-owned and have different concepts such as vision LoRA
overlays, voice profiles, and cloned voices.

## Decision

Text-generation adapters are a first-class Core concern, but only for Core-owned text providers.

The portable text contract will separate adapter lifecycle from request selection:

- Lifecycle is trusted control-plane behavior: load, list, inspect, and unload.
- Selection is inference behavior: one request may choose a loaded adapter explicitly.

The shared request contract should use an explicit field such as `adapter`, not a portable
`model="adapter-name"` convention.

Provider implementations may translate that portable field into backend-specific behavior, including
vLLM dynamic LoRA endpoints or future in-process PEFT/MLX-style activation. Callers should not need
to know those backend details.

Server adapter operations should target a loaded base text runtime by `runtime_id` or the existing
provider/model/base-url selector. Adapter mutation remains a trusted control-plane operation.

## Consequences

### Positive

- vLLM adapter support can move from provider-specific convenience methods into the shared text
  abstraction.
- Warm base text runtimes can be reused while selecting different adapters explicitly.
- Docs and server control surfaces can stop relying on backend-specific model-id tricks.

### Negative

- The text-provider contract needs new structured result types or protocol-style optional methods.
- vLLM will likely be the first complete implementation, with other backends following later.
- The server needs new trusted adapter routes or subresources tied to loaded text runtimes.

### Neutral

- This ADR does not authorize training or fine-tuning adapters.
- This ADR does not define production multi-tenant adapter safety by itself.
- This ADR does not govern `abstractvision` or `abstractvoice` adapter/profile semantics.

## Enforcement

- Shared docs must not claim portable adapter selection through `model=` unless that behavior is
  explicitly implemented and documented as provider-specific.
- Providers that do not support text adapters must fail explicitly when adapter lifecycle or
  selection is requested.
- Server adapter mutation must stay on trusted/admin control-plane paths.
- Backlog items changing text adapter behavior should cite this ADR.

## Validation

- Unit tests for default unsupported adapter behavior and structured adapter results.
- Provider tests for vLLM adapter lifecycle request shapes and error normalization.
- Request-path tests proving explicit adapter selection does not require changing the base provider
  instance model.
- Server tests for adapter lifecycle operations against a loaded text runtime.
- Docs audit for examples that imply unimplemented adapter hot-switch behavior.

## Related

- `abstractcore/providers/vllm_provider.py`
- `abstractcore/providers/openai_compatible_provider.py`
- `abstractcore/core/interface.py`
- `abstractcore/providers/base.py`
- `abstractcore/server/app.py`
- `docs/backlog/planned/2026-05-04_unified-lora-adapter-serving.md`
- `docs/backlog/planned/2026-05-04_vllm-base-model-swap-orchestration.md`
