# Completed: web_search DDGS numeric argument boundary

## Metadata
- Created: 2026-06-21
- Status: Completed
- Completed: 2026-06-21

## ADR status
- Governing ADRs: `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md`, `docs/adr/0005-source-first-quality-fixes.md`
- ADR impact: No new ADR required. This item applies existing no-silent-degradation and source-first-boundary policy to the web-search tool layer.

## Context
`web_search(...)` prefers `ddgs.text` and falls back to DuckDuckGo HTML scraping when the primary backend fails. A bug report from AbstractAssistant showed a persistent degradation pattern:

- primary backend reported `unsupported operand type(s) for /: 'str' and 'int'`;
- fallback backend `duckduckgo.html` still returned usable results;
- the visible symptom was degraded-but-working search with `backend_used="duckduckgo.html"`.

Local inspection showed two separate boundary problems:

- `web_search(...)` forwarded `num_results` to `ddgs` unchanged;
- `skim_websearch(...)` silently defaulted invalid `num_results` to `5` and silently capped large requests to `15`.

## Current code reality
- `abstractcore/tools/common_tools.py`
  - `web_search(...)` built `search_params["max_results"] = num_results` directly.
  - the HTML fallback loop also used `int(num_results or 0)` inline.
  - payload metadata returned the original `num_results` value, even if it was string-like.
  - `web_search(...)` imported `ddgs` only, even though the `<3.10` install profile declared `duckduckgo-search` as the compatibility dependency.
  - `skim_websearch(...)` defaulted bad `num_results` to `5` and capped large values to `15` without surfacing that truncation.
- `tests/tools/test_common_tools_skim_url_and_websearch.py`
  - already covered the modern `DDGS.text(query=..., max_results=...)` contract;
  - did not cover string-valued numeric inputs reaching `web_search(...)`.
- `abstractcore/tools/core.py`
  - tool schemas describe parameter types, but do not coerce arguments into Python-native runtime values.
- `abstractcore/tools/common_tools.py`
  - other tools already defensively coerce booleans and numeric timeouts because some hosts send tool arguments as strings.

## Problem
`ddgs` expects `max_results` to be an integer. In `ddgs 9.14.2`, the internal `_search_sync(...)` path computes `ceil(max_results / 10) + 1`, so `max_results="5"` raises exactly:

`TypeError: unsupported operand type(s) for /: 'str' and 'int'`

That meant:

- normal direct Python callers using `int` worked;
- JSON/function-calling hosts that passed `"5"` instead of `5` degraded into the fallback backend;
- the failure looked like an upstream DDGS outage even though the real bug was our boundary typing.

## What we wanted to do
Normalize numeric-like `num_results` inputs at the search-tool boundary so both `web_search(...)` and `skim_websearch(...)` receive truthful integer counts, invalid values fail explicitly instead of being silently rewritten, and the Python 3.9 compatibility dependency path matches runtime import behavior.

## Why
- Fixes the real degradation reported by AbstractAssistant.
- Keeps `ddgs.text` as the primary backend when the only issue is JSON-style string arguments.
- Matches the repo’s existing defensive pattern for tool arguments coming from heterogeneous runtimes.
- Removes a packaging/runtime mismatch where Python 3.9 installs could declare a preferred backend dependency that runtime code never imported.

## Requirements
- Accept string-valued `num_results` such as `"5"` without degrading to fallback.
- Use the normalized integer consistently in:
  - DDGS `max_results`;
  - fallback result truncation;
  - returned metadata.
- Make invalid `num_results` fail explicitly in both `web_search(...)` and `skim_websearch(...)`.
- Surface `skim_websearch(...)` compact-mode capping instead of hiding it.
- Honor the declared Python `<3.10` compatibility dependency path when `ddgs` is unavailable.
- Add regression coverage for the exact bug trigger.
- Preserve the existing fallback contract for genuine DDGS failures.

## Scope
- `web_search(...)` input normalization.
- Regression tests for the DDGS numeric boundary.
- Backlog/docs updates.

## Non-goals
- Do not redesign the broader web-search backend strategy.
- Do not remove the HTML fallback backend.
- Do not change search ranking, snippets, or filtering semantics beyond fixing the argument-type bug.

## Dependencies and related tasks
- `proposed/0811_optional_trafilatura_html_extractor_for_web_tools.md`
- `completed/0812_fetch_url_pdf_router_and_native_fallback_contract.md`

## Expected outcomes
- `web_search(..., num_results="5")` uses `ddgs.text` successfully.
- Returned payload metadata reports `params.num_results` as an integer.
- The old `'str' and 'int'` DDGS failure no longer appears for stringified numeric tool arguments.
- `skim_websearch(..., num_results="abc")` fails explicitly instead of silently defaulting to `5`.
- `skim_websearch(..., num_results=20)` surfaces its compact cap at `15`.
- Python 3.9-style `duckduckgo_search.DDGS` imports remain usable as the primary backend path.

## Validation
- Focused pytest for `web_search` / `skim_websearch`.
- Direct live repro against the current installed `ddgs` version with string-valued `num_results`.
- Local source inspection of installed `ddgs` to confirm the exact failing line.

## Progress checklist
- [x] Reproduce the bug with `web_search(..., num_results="5")`.
- [x] Confirm the exact DDGS failure path in installed package code.
- [x] Normalize numeric-like inputs at the `web_search(...)` boundary.
- [x] Add regression coverage for string-valued `num_results`.
- [x] Re-run live searches to prove `ddgs.text` stays primary.

## Completion report

Completed on 2026-06-21.

### What changed
- `abstractcore/tools/common_tools.py`
  - added shared positive-integer normalization for search tool arguments;
  - normalized `num_results` before the DDGS call and reused the normalized integer in fallback truncation and returned metadata;
  - made invalid values fail with a structured `num_results must be a positive integer` error instead of silently defaulting;
  - added an import helper that uses `ddgs` first and `duckduckgo_search` second, matching the declared `<3.10` compatibility extra;
  - made `skim_websearch(...)` fail explicitly on invalid `num_results` and surface the compact `15`-result cap in warnings/limitations instead of hiding it;
  - documented why the normalization exists: some tool hosts pass JSON-style strings.
- `tests/tools/test_common_tools_skim_url_and_websearch.py`
  - added a regression proving `web_search("pets", num_results="5")` calls `ddgs.text(..., max_results=5)` with an actual integer and remains on `ddgs.text`.
  - added a regression proving invalid values such as `"abc"` fail explicitly instead of silently degrading or defaulting.
  - added a regression proving the legacy `duckduckgo_search.DDGS` import path still works when `ddgs` is unavailable.
  - added a regression proving `skim_websearch(...)` surfaces its compact cap.

### Root cause
The bug was not DDGS response parsing. It was our caller boundary:

- `DDGS.text(query=..., max_results="5")` fails in installed `ddgs 9.14.2`;
- `DDGS.text(query=..., max_results=5)` succeeds;
- `skim_websearch(...)` already coerced its own count, which is why it often hid the issue.

### Validation
- Installed package inspection:
  - `ddgs` version: `9.14.2`
  - `DDGS.text` signature: `(self, query: str, **kwargs: Any)`
  - internal failing line in installed `ddgs/ddgs.py`:
    - `ceil(max_results / 10) + 1`
- `python -m pytest -q tests/tools/test_common_tools_skim_url_and_websearch.py`
  - Result: `12 passed`
- Live repro after the fix:
  - `web_search("abstractframework.ai", num_results="5")`
    - `backend_used=ddgs.text`
    - `degraded=false`
    - `params.num_results=5`
  - `web_search("Genentech ML Engineering Director AI4DD", num_results="10")`
    - `backend_used=ddgs.text`
    - `degraded=false`
  - `web_search("Roche Basel Machine Learning Scientist", num_results="3")`
    - `backend_used=ddgs.text`
    - `degraded=false`
  - `web_search("pets", num_results="abc")`
    - `success=false`
    - `status_hint=error`
    - `error="num_results must be a positive integer"`
  - `skim_websearch("pets", num_results="abc")`
    - `success=false`
    - `status_hint=error`
    - `error="num_results must be a positive integer"`
  - `skim_websearch("pets", num_results=20)`
    - `warnings` includes `num_results was capped at 15 for compact skim output.`
    - `limitations` includes `num_results_capped_at_15`

### Residual risks
- Other tool boundaries may still forward stringified numerics or booleans into strict third-party libraries if they have not been hardened similarly.
- Genuine DDGS outages, backend removals, or rate limits still require the HTML fallback path.
- The legacy `duckduckgo_search` compatibility path is unit-tested, but I did not run a real Python 3.9 environment in this workspace.
