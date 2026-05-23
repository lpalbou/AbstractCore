# Proposed: Generate() concurrency + async streaming correctness audit

## Metadata
- Created: 2026-05-22
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: ADR 0001, ADR 0002, ADR 0003
- ADR impact: None

## Context
AbstractCore increasingly positions `generate(...)` / `agenerate(...)` as the unified “one call”
surface for model inference (text + multimodal output selectors), and encourages async usage for
concurrency in FastAPI-style servers and batch workloads.

To keep AbstractCore user-first, we should be explicit about what “concurrent” means for:
- remote HTTP providers (I/O bound; concurrency is mostly client-side);
- in-process local inference providers (GPU/CPU bound; concurrency may reduce throughput unless the
  backend supports continuous batching);
- streaming vs non-streaming responses.

This becomes more important if we later route additional stateless compute tasks (embeddings,
rerank) through unified entrypoints.

## Current code reality
- `abstractcore/providers/base.py` implements `BaseProvider.agenerate(...)` and uses:
  - native async overrides for some providers; otherwise
  - `asyncio.to_thread()` fallback to avoid blocking the event loop.
- For `stream=True` in the default implementation, `_agenerate_internal(...)` returns
  `_async_stream_generate(...)`, which:
  - obtains a **sync** streaming generator via `asyncio.to_thread(get_sync_stream)`, then
  - iterates the returned generator in the async context (`for chunk in sync_gen: ...`).
  This is likely to block the event loop when the sync generator blocks on network I/O or local
  compute per chunk.
- `abstractcore/core/session.py` sessions mutate `self.messages` and are not designed for concurrent
  use of the *same* session instance (stateful history).

## Problem or opportunity
We may be overstating the correctness/performance of async streaming in cases where the default
fallback path is used, and we may not have tests that prevent regressions.

Even if most production streaming uses native-async providers, the fallback should be correct (or
explicitly documented as limited) because:
- it is used by local-first stacks;
- it affects FastAPI/SSE integrations;
- it sets expectations for future “compute tasks” (embeddings/rerank) that may be batch-shaped and
  concurrently invoked.

## Proposed direction
1. Define and document concurrency expectations:
   - which providers are safe for concurrent calls on the same provider instance;
   - which providers intentionally serialize (e.g., device locks);
   - what “async streaming” guarantees for native vs fallback.
2. Add a correctness test for fallback async streaming:
   - construct a sync generator that blocks per-yield (simulating chunked I/O);
   - assert the event loop stays responsive (heartbeat task keeps running) while streaming.
3. If the fallback path blocks, fix it by moving sync iteration into a background producer:
   - run the sync generator in a thread;
   - push chunks into an `asyncio.Queue`;
   - expose an async iterator that awaits queue items;
   - support early cancellation/cleanup.
4. Optionally, add lightweight concurrency controls:
   - per-provider “max in-flight requests” (semaphores) for local inference to prevent accidental
     overload;
   - a clear escape hatch for advanced users to disable/enforce concurrency.

## Why it might matter
- Prevents “async but blocking” surprises for users building web servers or running batch jobs.
- Gives AbstractCore a clear, testable concurrency story before adding more task-specific compute
  surfaces (e.g., rerank).
- Makes behavior predictable across providers and avoids silent performance cliffs.

## Promotion criteria
- Reproduce event-loop blocking (or other concurrency bugs) in the fallback streaming path with a
  minimal test, **or**
- receive a user report that `agenerate(stream=True)` blocks other async tasks for at least one
  provider path, **or**
- implementers are about to add new async task surfaces (e.g., `RerankManager.arerank(...)`) that
  would reuse the same streaming/concurrency patterns.

## Validation ideas
- Unit test: fallback async streaming does not stall a periodic async heartbeat task.
- Unit test: early consumer break/cancel closes the producer thread and does not leak resources.
- Smoke: minimal FastAPI StreamingResponse example stays responsive under concurrent streams.

## Non-goals
- Not a full rewrite to make every provider natively async.
- Not “guarantee faster local inference with more concurrency” (throughput depends on the backend).
- Not a replacement for continuous batching work (tracked elsewhere).

## Guidance for future agents
Treat this as a “truth in interfaces” item: either make async streaming genuinely non-blocking in
fallback mode, or update docs and contracts so users are not misled.
