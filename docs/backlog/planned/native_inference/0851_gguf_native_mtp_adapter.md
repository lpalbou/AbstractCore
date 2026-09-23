# 0851-abstractcore: [FEATURE] Connect an in-process GGUF native MTP driver

> Created: 2026-09-20
> Status: Planned
> Type: feature
> Priority: P2
> Labels: native-inference, gguf, mtp

## Summary

Execute native MTP through the existing HuggingFace provider's GGUF backend,
without confusing llama.cpp server support with Python-binding support.

## Current code reality

Audited 2026-09-20: the provider inspects GGUF nextn metadata and warns that its
llama-cpp-python path cannot execute the native driver. Current high-level
`Llama.__init__` has generic `draft_model` but no native MTP load/context controls.
The warning identifies the common-library driver/binding gap. Recheck actual
installed/upstream ABI before choosing an implementation; this is version-sensitive.

## Scope and non-goals

Assess upstream binding support first; otherwise design the smallest maintainable
binding to the actual native driver. Carry the shared contract through head load,
verification, KV/recurrent-state recovery and cleanup. No mandatory llama-server,
LM Studio or oMLX process, no second GGUF provider, and no pretending that metadata
or a generic draft-model argument proves native MTP works.

## Acceptance criteria

- [ ] A pinned GGUF with real nextn tensors executes native MTP in process.
- [ ] Missing symbols/ABI or architecture support produces a named strict failure.
- [ ] Off/depth, stop/EOS, cancellation, streaming and hybrid rollback are tested.
- [ ] Memory/resource cleanup and actual outcome reporting are validated.
- [ ] Packaging/license/platform burden is documented before adding a binding dependency.

## Testing

Run `python -m pytest tests/providers -k 'gguf or speculation or mtp'`.
Add native-symbol probing tests and a gated real-GGUF comparison at supported
depths; use 60–120 second cooldowns between Apple GPU blocks.

## Dependencies and ADR status

Depends on 0849; related hybrid-cache recovery item 0844.
ADR impact: review needed if shipping a new native binding/library; do not decide
that durable packaging boundary solely in this backlog record.
Expected outcome: honest, native GGUF MTP for a pinned supported runtime.
