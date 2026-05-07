# Completed: Normalize task-only text generation output selectors

## Metadata
- Created: 2026-05-07
- Status: Completed
- Completed: 2026-05-07
- Related: `2026-05-07_public-output-selector-contract.md`

## Context

AbstractCore exposes `abstractcore.core.output_specs` as the public output selector contract for
runtimes. AbstractRuntime delegates generated-media and non-chat dispatch checks to that module so
Core remains the owner of output semantics.

The selector `{"task": "text_generation"}` was accepted by `is_output_request(...)`, but
`normalize_output_spec(...)` did not infer `modality="text"` from that task. The normalized selector
kept an empty modality, which caused downstream helpers to classify the request as generated media
and non-chat dispatch.

That affected Core directly too: `llm.generate("hello", output={"task": "text_generation"})` could
raise `ValueError: Unsupported multimodal output modality: ''`, even though
`output={"modality": "text", "task": "text_generation"}` and `output="text"` followed the normal
text path.

## Decision

Keep `task="text_generation"` as a supported public selector value and normalize it to
`modality="text"`.

This keeps AbstractCore's two text-producing tasks explicit:

- `text` + `text_generation`: normal LLM/chat generation.
- `text` + `transcription`: speech-to-text, still routed through non-chat STT dispatch.

## Implementation Report

- Added `OUTPUT_TASK_MODALITIES` in `abstractcore/core/output_specs.py` as the canonical
  task-to-modality mapping for normalized task values.
- Changed `normalize_output_spec(...)` to infer a missing modality from that canonical task map
  after task aliases are normalized.
- Added `text_generation` to modality aliases so accepted dict selectors such as
  `{"output": "text_generation"}` also normalize to `{"modality": "text", "task": "text_generation"}`.
- Preserved the existing transcription behavior: `{"task": "transcription"}` normalizes to text,
  reports no generated media, and still requires non-chat dispatch because it is STT.
- Updated selector tests that previously locked in the broken empty-modality result.
- Added helper coverage for both `{"task": "text_generation"}` and
  `{"modality": "text", "task": "text_generation"}`.
- Added provider-dispatch coverage showing text-generation selectors use the normal text path,
  do not pass a multimodal `output` kwarg to `_generate_internal(...)`, and do not call generated
  media plugins.
- Matched native async dispatch to sync dispatch so
  `agenerate(..., output={"task": "text_generation"})` also strips the normalized text selector
  before `_agenerate_internal(...)`.
- Added native async regression coverage for task-only and explicit text-generation selectors.

## Files Touched

- `abstractcore/core/output_specs.py`
- `tests/test_output_specs.py`
- `tests/test_multimodal_generate_output.py`
- `CHANGELOG.md`
- `abstractcore/utils/version.py`
- `docs/server.md`
- `abstractcore/server/README.md`

## Acceptance Criteria

- `normalize_output_spec({"task": "text_generation"})` returns
  `{"task": "text_generation", "modality": "text"}`.
- `output_has_generated_media({"task": "text_generation"})` returns `False`.
- `output_requires_non_chat_dispatch({"task": "text_generation"})` returns `False`.
- Tests cover task-only text generation and explicit
  `{"modality": "text", "task": "text_generation"}` selectors.
- Provider dispatch treats text-generation selectors as ordinary text generation rather than
  generated-media dispatch.
- Native async provider dispatch has the same selector-stripping behavior as sync dispatch.

## Validation

- `python -m pytest tests/test_output_specs.py tests/test_multimodal_generate_output.py -q`
  passed: 77 tests.
- `python -m compileall -q abstractcore/core/output_specs.py abstractcore/providers/base.py`
  passed.
- `/Users/albou/.pyenv/versions/3.9.25/bin/python -m py_compile abstractcore/core/output_specs.py abstractcore/providers/base.py`
  passed, confirming the touched modules still parse under the package's Python 3.9 floor.
- `python -m black --check abstractcore/core/output_specs.py tests/test_output_specs.py tests/test_multimodal_generate_output.py`
  passed.
- `python -m ruff check abstractcore/core/output_specs.py tests/test_output_specs.py tests/test_multimodal_generate_output.py`
  passed.
- `python -m ruff check --select F821 abstractcore` passed.
- `git diff --check` passed.
- `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY python -m pytest -q` passed: 1282 tests,
  258 skipped, 86 warnings.
- `python -m pytest -q` with the local live API environment reached Anthropic and failed one
  external account check with `credit balance is too low`; this was not related to the selector
  change. The full suite was rerun with live OpenAI/Anthropic keys unset so those tests skipped.
- `python -c "import abstractcore; print(abstractcore.__version__)"` returned `2.13.10`.
- `python -m build` passed and created `dist/abstractcore-2.13.10.tar.gz` and
  `dist/abstractcore-2.13.10-py3-none-any.whl`.
- `python -m twine check dist/abstractcore-2.13.10*` passed for both artifacts.
- `python -m mkdocs build` passed. `python -m mkdocs build --strict` still aborts on 13 existing
  documentation nav/link warnings outside this change.

## Release Notes

Prepared for AbstractCore `2.13.10`.

The patch release fixes task-only text-generation output selector normalization and documents the
completed backlog item. No remote push has been performed.
