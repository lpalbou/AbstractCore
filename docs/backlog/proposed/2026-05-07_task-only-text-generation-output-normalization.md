# Proposed: Normalize task-only text generation output selectors

## Metadata
- Created: 2026-05-07
- Status: Proposed
- Completed: N/A
- Related: `../completed/2026-05-07_public-output-selector-contract.md`

## Context

AbstractCore now exposes `abstractcore.core.output_specs` as the public output selector contract for
runtimes. AbstractRuntime delegates generated-media and non-chat dispatch checks to that module so
Core remains the owner of output semantics.

One task-only selector still has surprising behavior:

```python
from abstractcore.core.output_specs import (
    is_output_request,
    normalize_output_spec,
    output_has_generated_media,
    output_requires_non_chat_dispatch,
)

selector = {"task": "text_generation"}

print(is_output_request(selector))
print(normalize_output_spec(selector))
print(output_has_generated_media(selector))
print(output_requires_non_chat_dispatch(selector))
```

Expected output, if task-only text generation is supported:

```text
True
{'task': 'text_generation', 'modality': 'text'}
False
False
```

Current output:

```text
True
{'task': 'text_generation', 'modality': ''}
True
True
```

## Problem

`{"task": "text_generation"}` is accepted by `is_output_request(...)`, but
`normalize_output_spec(...)` does not infer `modality="text"` from that task. Downstream helpers then
see an empty modality as non-text:

- `output_has_generated_media(...)` returns `True`.
- `output_requires_non_chat_dispatch(...)` returns `True`.

This can make a durable runtime treat an explicit text-generation request like generated binary
media. In AbstractRuntime, that means it may require an artifact store or skip the normal chat/text
path even though the requested output is text.

## Proposal

Decide whether `task="text_generation"` is part of the supported public selector vocabulary.

If it is supported, normalize it to text:

```python
if not modality:
    if task == "text_generation":
        modality = "text"
```

If task-only text generation should not be supported, remove it from the accepted dict selector
values or document that callers must pass `{"modality": "text", "task": "text_generation"}`.

## Acceptance Criteria

- `normalize_output_spec({"task": "text_generation"})` either returns
  `{"task": "text_generation", "modality": "text"}` or the selector is no longer accepted.
- `output_has_generated_media({"task": "text_generation"})` is `False` when the selector is
  accepted.
- `output_requires_non_chat_dispatch({"task": "text_generation"})` is `False` when the selector is
  accepted.
- Tests cover task-only text generation and the explicit
  `{"modality": "text", "task": "text_generation"}` shape.

## Why This Matters

Runtime safety checks rely on Core's public contract. Normalizing accepted text selectors prevents
durable runtimes from applying generated-media guardrails to a plain text request.
