# 2026-05-04 — vLLM base model swap orchestration

## Goal

Explore a controlled way for AbstractCore to manage a locally served vLLM base model process, including starting, stopping, restarting, and switching the served base model from a predefined allowlist.

This is a convenience/admin workflow, not a promise that base models can be swapped cheaply or safely on every request.

## Why

For local NVIDIA inference, vLLM is often the right serving backend, but changing the base model usually means restarting the vLLM server with different launch arguments. A thin AbstractCore controller could make this easier for local labs, demos, and single-tenant deployments:

- keep model profiles in config
- launch vLLM with known-safe arguments
- wait for readiness
- route AbstractCore server traffic to the active backend
- capture status/logs/errors in a predictable place

## Scope

### In scope

- Define a model profile schema:
  - profile name
  - Hugging Face model id/local path
  - served model name
  - vLLM image/tag or local executable
  - GPU/memory hints
  - max model length, tensor parallel size, quantization, chat template, reasoning parser
  - optional API key
- Add a process/container controller abstraction:
  - start profile
  - stop active profile
  - restart profile
  - status/readiness
  - recent logs
- Support a safe cutover flow:
  - reject or drain new requests
  - stop old backend
  - start new backend
  - wait for `/health` and `/v1/models`
  - update gateway route/base URL
- Make failure modes explicit and recoverable.

### Out of scope

- Per-request base model switching.
- Multi-tenant scheduling across many users.
- GPU cluster orchestration.
- Owning/maintaining a custom vLLM CUDA image unless there is a strong reason later.
- Replacing official vLLM Docker images.

## Design notes

This should probably start as a local/admin feature rather than a public server endpoint. If HTTP control endpoints are added later, they must require admin auth, audit logging, and a clear separation from inference auth.

The first implementation should prefer orchestrating the official vLLM image or a local `vllm serve` executable instead of baking vLLM into the AbstractCore image.

Base model swap is different from LoRA adapter loading:

- LoRA adapter loading can keep the base model resident.
- Base model swap generally requires unloading the old model and loading a new one, with GPU memory churn and downtime.

## Acceptance criteria

- A design document chooses the first controller target: local process, Docker, or both.
- Model profiles are validated before launch.
- A dry-run mode prints the exact vLLM command/container config.
- A fake-process test covers success, startup failure, timeout, and rollback behavior.
- Docs clearly state that base model swap is admin-only and not intended for high-frequency runtime routing.
