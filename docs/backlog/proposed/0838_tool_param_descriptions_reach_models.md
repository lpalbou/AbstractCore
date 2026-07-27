# 0838 — Tool parameter descriptions never reach models (native schema + prompted format)

- Status: proposed
- Date: 2026-07-26
- Source: edit_file line-range failure investigation (operator screenshots, two fable5
  adversaries); the tool-lane adversary's out-of-lane finding, independently corroborated by the
  hints-lane audit.
- Owner: core (tools/core.py, tools/parser.py)

## Problem

A live trace showed a model calling `edit_file(start_line=0, end_line=0, ...)` — a 0-based
habit. The docstring says "(1-indexed)" but NO MODEL EVER SEES IT:

1. **Native lane**: `ToolDefinition.from_function` (`tools/core.py:175-187`) emits each
   parameter as `{type, default}` only. No `description` field is populated from the docstring
   Args section, so native tool payloads (OpenAI/Anthropic-style `tools=[...]`) carry zero
   per-parameter teaching for every builtin tool.
2. **Prompted lane**: `_format_parameters_compact` (`tools/parser.py:1223-1263`) renders
   `name: type (optional)` and drops `description` even when present — so prompted-mode models
   (fenced/XML conventions) never see parameter teaching either.

Every carefully-written Args teaching in every tool docstring is dead weight at call time. The
edit_file fix (2026-07-26) closed this for ONE tool by post-definition schema injection; the
class remains open for the other ~26 builtins (e.g. read_file's `start_char` byte-offset
semantics, skim_websearch's `require_in` values, search_files' `output_mode`).

## Direction

1. Parse the docstring Args section in `ToolDefinition.from_function` (or accept an explicit
   `param_descriptions` kwarg on `@tool`, with docstring parsing as the default source) and emit
   JSON-Schema `description` per property. Verify `handler.py` passes properties through to
   native payloads (it does today — the edit_file injection rides it).
2. Prompted lane: append a truncated description in `_format_parameters_compact`
   (`name: type (required) — first sentence, budget-capped`), so fenced-convention models get
   the same teaching.
3. Token discipline: descriptions ride EVERY request carrying the tool. Cap per-param
   description length (first sentence or ~120 chars, `#TRUNCATION`-labeled if cut at an odd
   boundary); measure the aggregate cost across the default toolset before/after and record it.
4. Tests: schema emission pinned per tool (spot-pin the semantically load-bearing ones:
   edit_file range params, read_file `start_char`, skim_websearch `require_in`); prompted-format
   render pinned; a budget test asserting the default toolset's schema growth stays within the
   measured envelope.

## Why it matters

This is the actual root cause behind the "model passes wrong parameter shapes" failure class:
the model is not hallucinating against teaching we gave it — we never gave it. Error-message
quality (fixed for edit_file) is the recovery path; schema teaching is the prevention path.
