# Planned: Consensus Generate

## Metadata
- Created: 2026-05-06
- Status: Planned
- Completed: N/A

## Context
AbstractCore already provides one provider-agnostic generation interface through
`create_llm(provider, model=...)`, `BaseProvider.generate(...)`, and
`BaseProvider.agenerate(...)`. Users can manually create several providers and run them with
`asyncio.gather(...)`, and `docs/async-guide.md` even shows a small "Provider Consensus" example.

That example is useful, but it leaves every application to reimplement:
- provider/model spec parsing;
- concurrent execution and timeout handling;
- partial failure capture;
- output normalization;
- review/aggregation prompting;
- trace metadata for which models participated.

## Current Code Reality
- Provider construction is centralized in `abstractcore/core/factory.py` and
  `abstractcore/providers/registry.py`.
- The canonical provider boundary is `AbstractCoreInterface.generate(...)` and
  `AbstractCoreInterface.agenerate(...)` in `abstractcore/core/interface.py`.
- `BaseProvider.generate(...)` delegates to `generate_with_telemetry(...)` in
  `abstractcore/providers/base.py`; async generation exists through `agenerate(...)`.
- `GenerateResponse` in `abstractcore/core/types.py` has `content`, `usage`, `tool_calls`,
  `metadata`, and timing fields that can carry aggregation traces without changing provider
  response schemas.
- `docs/async-guide.md` documents manual concurrent multi-provider calls and a toy consensus
  example, but there is no first-class `consensus_generate` helper or reusable response type.
- `BasicJudge` in `abstractcore/processing/basic_judge.py` provides an LLM-as-judge pattern that
  may be useful inspiration for the optional review step, but it is not a multi-model generation
  orchestrator.

## Problem
Users who want to compare multiple models or ask for a consensus answer must write orchestration
code themselves. That creates repeated, brittle implementations that may:
- run providers sequentially when concurrent execution is expected;
- fail the whole call when one provider is unavailable;
- lose raw model outputs after aggregation;
- hide which model produced which answer;
- conflate majority vote, judge review, and synthesis into one opaque result;
- make debugging, audit, and cost attribution hard.

## What We Want To Do
Create a first-class consensus generation API that accepts a set of provider/model candidates,
runs inference across them, returns every candidate result, and optionally produces a consensus
answer by reviewing and aggregating the candidate inferences.

The API should make the raw individual outputs the primary evidence and treat the consensus answer
as an additional, traceable synthesis.

## Why
- Model comparison is a common workflow for evaluation, robustness, and prompt development.
- Consensus can reduce single-model brittleness for high-stakes or ambiguous tasks.
- Applications can expose richer results without building provider orchestration each time.
- A shared implementation can make failure handling, latency, and provenance consistent.

## Requirements
- Accept multiple provider/model specs. Supported input forms should be explicit and easy to parse,
  for example:
  - `("openai", "gpt-4o-mini")`
  - `{"provider": "anthropic", "model": "claude-haiku-4-5"}`
  - `"ollama/qwen3:4b"` or another provider/model string split on the first slash.
- Preserve nested model ids such as `openrouter/anthropic/claude-...` by splitting only the provider
  prefix.
- Run candidate generation concurrently for async execution.
- Provide a sync convenience wrapper that is safe for normal scripts.
- Return all candidate outcomes, including failed candidates, with bounded error details.
- Do not fail the entire call unless every candidate fails or the caller opts into strict behavior.
- Record per-candidate metadata:
  - provider;
  - model;
  - status;
  - content or structured result reference;
  - error type/message when failed;
  - latency;
  - usage when available.
- Support optional consensus aggregation:
  - disabled by default or explicitly requested;
  - reviewer provider/model configurable;
  - reviewer may be one of the candidates or a separate provider/model;
  - review prompt must include all successful candidate answers with provider/model provenance;
  - consensus output must include the selected aggregation method and source candidates.
- Support at least one deterministic non-LLM aggregation mode for simple cases, such as exact-answer
  majority voting when callers provide a normalizer or choices.
- Make partial failure visible in the returned metadata and logs.
- Avoid silent truncation when candidate answers are too large for the reviewer context. If truncation
  is needed, use explicit metadata and searchable `#TRUNCATION:` comments near the implementation.
- Expose clear timeout controls for the whole consensus call and per-candidate calls.
- Keep provider kwargs consistent with normal `generate(...)` calls.

## Suggested Implementation
1. Add small dataclasses in a core module, for example `abstractcore/core/consensus.py`:
   - `ProviderModelSpec`
   - `CandidateGeneration`
   - `ConsensusGenerateResponse`
   - `ConsensusOptions`
2. Add parser helpers that normalize tuples, dicts, and provider/model strings without breaking
   nested model ids.
3. Implement `aconsensus_generate(...)` using `create_llm(...)` and `agenerate(...)` with
   `asyncio.gather(..., return_exceptions=True)`.
4. Implement `consensus_generate(...)` as a sync wrapper around the async path, with care for
   already-running event loops.
5. Add optional aggregation modes:
   - `aggregate=False`: return candidates only;
   - `aggregate="majority"`: deterministic vote over normalized short answers;
   - `aggregate="review"`: call a reviewer model to synthesize a consensus.
6. Store candidate traces in `ConsensusGenerateResponse.metadata` and, when a consensus answer is a
   `GenerateResponse`, include a compact copy in `GenerateResponse.metadata["consensus"]`.
7. Export the public helpers from `abstractcore/__init__.py` only after the API is stable enough.
8. Document the workflow in `docs/async-guide.md` and `docs/api.md`.

## Scope
- Text generation consensus for non-streaming calls.
- Provider/model spec normalization.
- Concurrent async implementation and sync convenience wrapper.
- Optional reviewer-based aggregation.
- Unit tests with fake providers and fake reviewer behavior.
- Documentation examples for candidate-only and reviewed consensus flows.

## Non-Goals
- Do not make consensus imply correctness.
- Do not hide individual model outputs behind only the aggregated answer.
- Do not support streaming consensus in the first implementation.
- Do not implement model ranking, benchmark dashboards, or long-running evaluation suites here.
- Do not require `BasicJudge`; reuse its ideas only if that keeps the implementation simple.

## Dependencies And Related Tasks
- `abstractcore/core/factory.py`
- `abstractcore/providers/registry.py`
- `abstractcore/core/interface.py`
- `abstractcore/core/types.py`
- `abstractcore/providers/base.py`
- `docs/async-guide.md`
- `docs/api.md`
- `abstractcore/processing/basic_judge.py`
- Related future item: robust fallback generation should share provider/model spec parsing.

## Expected Outcomes
- Users can call one AbstractCore helper to query several models at once.
- Applications can display all model answers plus an optional consensus answer.
- Partial failures are visible and do not erase successful candidate answers.
- Review-based consensus includes enough provenance to audit how it was produced.
- Tests cover successful, partially failed, and all-failed consensus runs.

## Validation
- Unit tests for provider/model spec normalization, including nested OpenRouter-style model ids.
- Unit tests with fake async providers proving candidate calls run concurrently.
- Unit tests proving partial failures are captured and all-failed calls raise or return strict errors
  according to options.
- Unit tests for deterministic majority aggregation.
- Unit tests for reviewer aggregation prompt construction and returned metadata.
- Documentation examples checked for import paths and expected response fields.

## Progress Checklist
- [ ] Define public API shape and response dataclasses.
- [ ] Implement provider/model spec parsing.
- [ ] Implement async candidate execution.
- [ ] Implement sync convenience wrapper.
- [ ] Add optional majority aggregation.
- [ ] Add optional reviewer aggregation.
- [ ] Add metadata, logging, and truncation visibility.
- [ ] Add tests.
- [ ] Update docs and examples.

## Guidance For The Implementing Agent
Reassess current provider and async code before implementation. Keep the first API small,
evidence-preserving, and easy to test with fake providers. Favor explicit provenance over magical
"best answer" behavior, and share parsing utilities with any robust fallback generate work if both
items are implemented together.
