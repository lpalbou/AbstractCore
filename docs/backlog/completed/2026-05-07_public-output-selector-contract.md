# Completed: Public output selector contract for runtimes

## Metadata
- Created: 2026-05-07
- Status: Completed
- Completed: 2026-05-07
- Related: `2026-05-07_runtime-ready-multimodal-generation-abstraction.md`

## Context

AbstractRuntime must decide some safety and routing behavior before calling an AbstractCore
provider:

- whether `output=...` is an AbstractCore multimodal output request;
- whether the request can produce generated binary media and therefore requires artifact storage in
  a durable runtime;
- whether turn-grounding should be skipped because the request is image/TTS/STT rather than a normal
  chat turn;
- whether remote execution should use `/v1/chat/completions`, `/v1/images/generations`,
  `/v1/audio/speech`, or `/v1/audio/transcriptions`.

Today the canonical selector logic lives on a private provider helper:
`BaseProvider._is_acore_output_request(...)`. Runtime should not depend on private provider methods,
but mirroring the helper in Runtime creates drift risk.

## Proposal

Expose a small public helper module in AbstractCore, for example
`abstractcore.core.output_specs`:

```python
def is_output_request(value: object) -> bool: ...
def normalize_output_spec(value: object) -> GenerationOutputSpec: ...
def normalize_output_specs(value: object) -> list[GenerationOutputSpec]: ...
def output_has_generated_media(value: object) -> bool: ...
def output_requires_non_chat_dispatch(value: object) -> bool: ...
```

The public helpers should use the same alias vocabulary as provider dispatch. Provider internals can
call the public helpers too, so there is only one selector implementation.

## Acceptance Criteria

- The public helper returns the same results as current provider dispatch for existing accepted
  shapes, including string `"audio"` and dict/list selector behavior.
- Runtime can import only the public helper module and no private provider classes.
- The helper distinguishes generated binary media from text/STT outputs.
- The helper strips or ignores runtime-only metadata such as `run_id`, `tags`, and `artifact_id`
  when producing provider kwargs.
- Tests cover strings, dicts, lists, unsupported values, alias normalization, and text
  transcription.

## Why This Matters

This keeps AbstractCore as the owner of output semantics while letting durable runtimes enforce
artifact and routing guardrails before an expensive or unsafe provider call starts.

## Implementation Report

- Date: 2026-05-07
- Summary: Added `abstractcore.core.output_specs` as the public selector contract and delegated
  existing `BaseProvider` private wrappers to it.
- Files touched:
  - `abstractcore/core/output_specs.py`
  - `abstractcore/core/__init__.py`
  - `abstractcore/providers/base.py`
  - `tests/test_output_specs.py`
- Validation:
  - `python -m pytest tests/test_output_specs.py tests/test_multimodal_generate_output.py tests/test_generate_with_outputs.py -q`
  - `python -m compileall -q abstractcore/core/output_specs.py abstractcore/providers/base.py abstractcore/core/__init__.py`
  - `git diff --check`
  - `python -m pytest -q`
- Behavior notes:
  - Existing selector quirks are preserved intentionally, including string `"audio"` being selected
    while dict `"audio"` aliases normalize but are not selected by `is_output_request(...)`.
  - `output_has_generated_media(...)` is conservative for ambiguous `voice` selectors but does not
    classify explicit `voice_clone` requests as generated binary media.
  - Runtime metadata can be stripped through `strip_runtime_output_metadata(...)` or
    `output_plugin_kwargs(..., strip_runtime_metadata=True)` without changing current provider
    backend-kwarg forwarding behavior.
- Follow-up: AbstractRuntime can replace its local mirror with imports from
  `abstractcore.core.output_specs`.
