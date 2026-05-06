# Planned: Multimodal Generation And Deterministic Inference Cache

## Metadata
- Created: 2026-05-07
- Status: Planned
- Completed: N/A

## Context
AbstractCore already has prompt/session caching primitives for some text provider paths, but server-side generation endpoints for images, TTS, STT, and voice cloning do not currently expose a unified content-addressed cache. There is also no general deterministic response cache for repeated LLM calls where the request is intentionally reproducible, such as the same prompt/messages, provider, model, generation parameters, `temperature=0`, and `seed`.

This matters more now that AbstractCore Server exposes OpenAI-compatible routes for:
- `/v1/images/generations`
- `/v1/images/edits`
- `/v1/audio/speech`
- `/v1/audio/transcriptions`
- `/v1/voice/clone`
- `/v1/chat/completions`
- `/v1/responses`

## Current code reality
- Image generation/edit routes live in `abstractcore/server/vision_endpoints.py`.
- Audio/STT/TTS/voice-clone routes live in `abstractcore/server/audio_endpoints.py`.
- Chat/responses routes live in `abstractcore/server/app.py`.
- Prompt-cache control-plane proxy routes exist under `/acore/prompt_cache/*`, but these target upstream KV/prefix-cache behavior rather than a durable AbstractCore response cache.
- Existing caches should not be confused with semantic response reuse. Provider-side caches can reduce latency/cost but do not guarantee byte-identical reuse of generated outputs.

## Problem
Repeated deterministic or expensive generation requests can waste time, money, and GPU/provider quota:
- image generation may be expensive and slow;
- TTS can repeatedly synthesize identical audio;
- STT can repeatedly transcribe identical uploaded files;
- voice cloning can repeatedly create the same custom voice from the same reference sample;
- text inference with deterministic settings can repeatedly call providers for identical outputs.

At the same time, unsafe caching would be worse than no caching. Cached generations can leak user data, accidentally reuse non-deterministic outputs, ignore provider/model version drift, or bypass authorization boundaries.

## What We Want To Do
Design and implement an explicit, opt-in cache layer for deterministic generation and inference outputs. The cache should be safe by default, transparent to callers, and controllable per request and per deployment.

## Why
- Reduce repeated provider cost for stable prompts/assets.
- Improve latency for repeated workflows and demos.
- Make server behavior more predictable for deterministic test cases.
- Avoid ad hoc per-endpoint cache implementations.

## Requirements
- Caching must be disabled by default unless the deployment opts in.
- Cache keys must include all behavior-changing inputs:
  - route name and API version;
  - provider/model/base URL;
  - normalized request body/form fields;
  - uploaded file content hashes;
  - generation parameters such as `temperature`, `seed`, `steps`, `guidance_scale`, `voice`, `response_format`, and provider-specific `extra`;
  - relevant server/plugin version identifiers where available.
- Auth and tenant boundaries must be part of the cache namespace when server auth or multi-tenant deployment is enabled.
- Non-deterministic LLM calls should not be cached by default. Candidate default policy: allow text response caching only when `temperature=0` and either `seed` is set or the provider/model is explicitly marked deterministic for that cache profile.
- Binary outputs must preserve content type and response shape exactly.
- Errors should not be cached by default.
- The API should expose clear controls, probably:
  - request field or header: `cache_control`;
  - env config: `ABSTRACTCORE_SERVER_RESPONSE_CACHE=0|1`;
  - TTL and max-size settings;
  - cache bypass and cache refresh controls.
- The cache backend should start with a simple local filesystem or SQLite implementation and leave room for Redis/object-store backends later.

## Suggested Implementation
1. Add a small cache abstraction in a shared server module, not inside individual endpoints.
2. Define canonical request normalization helpers for JSON bodies, multipart fields, and uploaded bytes.
3. Add cache-key builders for image, audio, clone, chat, and responses routes.
4. Add opt-in middleware/helper calls around expensive endpoint execution.
5. Persist binary payloads with metadata (`content_type`, `created`, route, model, hash inputs).
6. Add cache-hit metadata in response headers, not in OpenAI-compatible response bodies unless the route already supports metadata.
7. Add tests for hit/miss/bypass/TTL/key isolation and deterministic LLM policy.

## Scope
- Server-side response/output caching for image/audio/voice/text routes.
- Documentation of policy, security implications, and controls.
- Unit tests and small integration tests with fake providers.

## Non-Goals
- Do not replace provider-side KV/prefix caches.
- Do not cache streaming chunks in v1 unless the whole response can be reconstructed safely.
- Do not enable caching for private/multi-tenant deployments without explicit configuration.
- Do not cache arbitrary tool outputs here; tool execution has different safety and idempotency constraints.

## Dependencies And Related Tasks
- Existing `/acore/prompt_cache/*` docs and implementation.
- `docs/backlog/planned/790_server-response-cache.md`, which should be reviewed and either merged with this item or narrowed to HTTP response-cache mechanics.
- Provider metadata for deterministic support may need extension.

## Expected Outcomes
- Operators can enable a conservative response cache for repeated deterministic calls.
- Image/audio/STT/voice-clone endpoints can avoid repeated work for identical inputs.
- Text inference cache can be enabled only under explicit deterministic policy.
- Cache behavior is observable through headers/logs and does not alter OpenAI-compatible response bodies.

## Validation
- Unit tests for key construction and canonicalization.
- Endpoint tests proving identical requests hit cache and changed parameters miss cache.
- Tests proving uploaded file hash changes invalidate cache.
- Tests proving auth/tenant namespaces isolate cache entries.
- Tests proving non-deterministic text requests are not cached by default.
- Manual server smoke test with repeated image/TTS requests and cache-hit headers.

## Progress Checklist
- [ ] Reconcile with `790_server-response-cache.md`.
- [ ] Design cache key schema.
- [ ] Implement cache abstraction.
- [ ] Wire image/audio/voice/text routes.
- [ ] Add docs and configuration reference.
- [ ] Add tests and smoke scripts.

## Guidance For The Implementing Agent
Reassess current server code before implementation. Keep the cache opt-in, deterministic, and easy to disable. Favor one shared abstraction over per-route cache branches. Treat privacy and tenant isolation as correctness requirements, not deployment notes.
