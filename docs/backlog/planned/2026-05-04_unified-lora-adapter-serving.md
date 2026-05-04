# 2026-05-04 — Unified LoRA adapter serving controls

## Goal

Add a provider-level adapter control surface so AbstractCore can serve open-source base models plus LoRA adapters through one consistent API.

The target user story:

- start a local/self-hosted base model once
- load one or more LoRA adapters by name
- route requests to either the base model or an adapter model id
- unload adapters when they are no longer needed
- expose capabilities clearly so callers know whether runtime adapter control is supported

## Why

LoRA adapters are a high-value middle ground between one fixed open-source model and fully separate fine-tuned model deployments. They let a user keep a base model resident while adding domain/task-specific behavior with less memory and startup cost than separate full model processes.

vLLM already exposes runtime LoRA endpoints, and `VLLMProvider` already has provider-specific `load_adapter`, `unload_adapter`, and `list_adapters` methods. This should become a first-class AbstractCore capability instead of a vLLM-only convenience.

## Scope

### In scope

- Add optional base provider methods:
  - `supports_adapters() -> bool`
  - `list_adapters() -> list[str]`
  - `load_adapter(name: str, path: str, **kwargs) -> AdapterStatus`
  - `unload_adapter(name: str, **kwargs) -> AdapterStatus`
- Add a small structured result type for adapter operations.
- Add provider capability metadata, e.g. `adapter_management`, `lora_adapters`, `runtime_lora`.
- Implement the interface for `VLLMProvider` first.
- Document how an adapter name maps to request `model` values.
- Keep dynamic adapter loading behind explicit admin/trusted controls in server mode.

### Out of scope

- Training/fine-tuning adapters.
- Arbitrary remote adapter downloads enabled by default.
- Production multi-tenant adapter management without RBAC/audit controls.
- Base model hot-swap. That is a separate task.

## Design notes

Provider implementations should distinguish:

- `static_adapters`: adapters configured at backend startup
- `runtime_adapters`: adapters loaded/unloaded while the backend is running
- `adapter_model_ids`: model names that callers can use in requests once an adapter is loaded

For vLLM, the first implementation can wrap:

- `POST /v1/load_lora_adapter`
- `POST /v1/unload_lora_adapter`
- adapter/model discovery via the vLLM model list or existing adapter endpoint where available

The server should not expose adapter mutation to ordinary inference clients. If exposed over HTTP, it should be behind an admin key or local-only control path.

## Acceptance criteria

- `BaseProvider` exposes optional adapter methods with default "unsupported" behavior.
- `VLLMProvider` implements adapter load/unload/list through the unified interface.
- Server docs explain that runtime LoRA loading is for trusted environments only.
- Unit tests cover:
  - unsupported provider behavior
  - vLLM request payloads for load/unload
  - normalized adapter operation results
- No existing provider behavior changes unless the new methods are called.
