# Proposed: `detect_tool_calls` disagrees with `parse_tool_calls` for every format-specific family

## Metadata
- Created: 2026-10-21
- Status: Proposed
- Completed: N/A
- Area: `abstractcore/tools/parser.py` (public API re-exported by `abstractcore/tools/__init__.py`)
- Found by: the adversarial review of the Hy-Memory-30B-A3B support work (nit N6 in
  `untracked/hy-memory/REVIEW.md`, 2026-10-21). Pre-existing defect; the Hy-Memory work only
  moved that model from the `generic` branch, which is correct, into an affected branch.

## ADR status
- Governing ADRs: `0001` (no silent degradation).
- ADR impact: none expected. The fix restores one invariant that the docstrings already imply
  ("True if tool calls detected"); it does not create a new cross-task rule.

## Plain description

`detect_tool_calls(text, model_name)` is the public yes/no question "does this response contain a
tool call?". `parse_tool_calls(text, model_name)` is the public extractor. For every model whose
architecture declares a specific `tool_format` (`special_token`, `xml`, `glm_xml`, `json`,
`tool_code`, ...), the detector answers **False** for payload shapes the parser then extracts
successfully. Only `model_name=None` (the "try everything" branch) agrees with the parser.

## Current code reality

`abstractcore/tools/parser.py`:

- `detect_tool_calls` (line 74 onward) resolves `tool_format = _get_tool_format(model_name)` and
  then returns from one **format-specific substring branch**: `TOOL_CODE` looks only for
  ```` ```tool_code ````/```` ```tool_call ````; `SPECIAL_TOKEN` only for `<|tool_call|>`,
  `<|tool_call>`, `<|tool_call_start|>`; `FUNCTION_CALL` only for `<function_call` or a JSON tool
  pattern; `XML_WRAPPED` only for `<tool_call>`. Only the `else` branch (no/unknown format) ORs
  every pattern together.
- `parse_tool_calls` (line 161 onward) tries the format-specific parser first and then falls
  back to `_parse_any_format` (line 1335 onward), which accepts raw JSON, `<|tool_call|>`,
  `<function_call>`, `<tool_call>` JSON, Gemma4 `<|tool_call>` and the bracket/harmony prefixes
  regardless of the declared format.
- The detector never falls back to `_parse_any_format`, so the two public functions diverge
  exactly where the parser's fallback fires.
- Callers: `abstractcore/tools/handler.py:14` imports `detect_tool_calls` but does not call it;
  nothing else inside the package calls it. Tests that use it: `tests/tools/test_tool_parser_harmony.py`
  (gpt-oss harmony, passes) and `tests/tools/test_tool_calling.py:106` (live-provider test, skipped
  by default). External users of the public API get the wrong answer.

## Reproduction (verified 2026-10-21 on commit b8c862c1)

```python
from abstractcore.tools.parser import detect_tool_calls, parse_tool_calls

cases = {
    "raw_json":      '{"name": "get_weather", "arguments": {"city": "Paris"}}',
    "special_token": '<|tool_call|>{"name": "get_weather", "arguments": {"city": "Paris"}}</|tool_call|>',
    "function_call": '<function_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</function_call>',
    "xml_json":      '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    "gemma4":        '<|tool_call>call:get_weather{city:"Paris"}<tool_call|>',
}
for model in ["tencent/Hy-Memory-30B-A3B", "Qwen/Qwen3.6-35B-A3B", "poolside/Laguna-S-2.1", "Qwen/Qwen3-8B", None]:
    for name, text in cases.items():
        d, p = detect_tool_calls(text, model), parse_tool_calls(text, model)
        print("MISMATCH" if d != bool(p) else "ok", d, len(p), model, name)
```

Observed (every `parse_tool_calls` call returns exactly 1 call):

| model (declared format) | raw_json | special_token | function_call | xml_json | gemma4 |
|---|---|---|---|---|---|
| `tencent/Hy-Memory-30B-A3B` (`glm_xml`) | **False** | **False** | **False** | True | **False** |
| `Qwen/Qwen3.6-35B-A3B` (`xml`) | **False** | **False** | **False** | True | **False** |
| `poolside/Laguna-S-2.1` (`glm_xml`) | **False** | **False** | **False** | True | **False** |
| `Qwen/Qwen3-8B` (`special_token`) | **False** | True | **False** | **False** | True |
| `None` | True | True | True | True | True |

The `Qwen/Qwen3-8B` × `xml_json` cell matters most in practice: Qwen3 models routinely emit
`<tool_call>{json}</tool_call>` even though the registry declares `special_token`, the parser
extracts it, and the detector says there is no tool call.

## Problem

- A caller that gates on `detect_tool_calls` before calling `parse_tool_calls` (the documented
  shape of the API; `tests/tools/test_tool_calling.py:106` does exactly this) silently skips real
  tool calls for most registered families.
- The disagreement is per-family and per-shape, so it is not discoverable from either docstring.

## Proposed fix

Make the detector exactly as permissive as the parser, at the same place the parser becomes
permissive: end each format-specific branch of `detect_tool_calls` with a fallback
`or bool(_parse_any_format(response))` (or, simpler and strictly equivalent, define
`detect_tool_calls` as the fast-path substring checks followed by
`bool(parse_tool_calls(response, model_name))`). Keep the fast paths so the common case stays a
substring test.

Non-goals:
- Do not change `parse_tool_calls` or any provider lane; they are correct.
- Do not make the detector *more* permissive than the parser (no "looks like JSON" heuristics
  beyond what `_parse_any_format` accepts).

## Validation expectations

- A parametrized unit test in `tests/tools/` asserting `detect_tool_calls(text, model) ==
  bool(parse_tool_calls(text, model))` over the five payload shapes above × the registered
  families with a declared `tool_format` (`special_token`, `xml`, `glm_xml`, `json`, `tool_code`,
  `prompted`, `native`, `pythonic`, `harmony`) × `None`; the test must be RED on the current code
  for the table cells marked **False**.
- `tests/tools/test_tool_parser_harmony.py` and
  `tests/tools/test_tool_parsing_cross_family_regressions.py` stay green (no change to parse
  results).
- A CHANGELOG `Fixed` line under `[Unreleased]`: "`detect_tool_calls` now agrees with
  `parse_tool_calls` for every declared tool format (it returned False for payload shapes the parser
  extracts)".

## Decision boundaries

- Public API behaviour changes for every family (False → True on the cells above). That is why
  this was not folded into the Hy-Memory change: it deserves its own entry and its own review.
- If a caller is found that relies on the narrow per-format answer (none exists in-tree), add a
  `strict: bool = False` keyword rather than keeping the divergence.

## References

- `abstractcore/tools/parser.py` (`detect_tool_calls`, `parse_tool_calls`, `_parse_any_format`)
- `untracked/hy-memory/REVIEW.md` N6 and `untracked/hy-memory/REVIEW-FIXES.md` "Not done"
  (local scratch; the reproduction above is self-contained)
- Related: `proposed/0823_native_tool_declaration_for_capable_models.md` (tool-format truth per model)
