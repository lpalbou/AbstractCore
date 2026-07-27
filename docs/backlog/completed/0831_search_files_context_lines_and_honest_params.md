# Completed: search_files context lines + honest parameter handling

## Metadata
- Created: 2026-07-25
- Status: Completed
- Completed: 2026-07-25
- Origin: tools quality audit 2026-07-25 (SOTA adversary + correctness pass), verified.

## Problem
`search_files` accepted several parameters in its signature and then silently discarded them
(`_ = (output_mode, context_lines, case_sensitive, ignore_dirs)`) — the exact "silent default"
behaviour the suite's own `arg_coercion` philosophy forbids. Plus: the multiline branch read
whole files with no cap, and the line-number model split (`re.finditer("\n")` for numbers vs
`content.splitlines()` for excerpts) could drift on form-feed / `\v` / U+2028-9.

## Approach
1. Implement `context_lines` (ring buffer, `-` for context vs `:` for match, `--` between groups, capped).
2. Stop silently ignoring `case_sensitive` / `output_mode` / `ignore_dirs`.
3. Per-file byte cap in the multiline branch.
4. One `\n`-based line model for numbering and excerpts.

## Completion report
- Date: 2026-07-25.
- Implemented (`abstractcore/tools/common_tools.py`, `search_files`): `context_lines` (0–10,
  ring buffer, `-`/`:` separators, `--` between non-adjacent groups, early break once the match
  cap is hit); `case_sensitive` toggles `re.IGNORECASE`; `output_mode` in
  content|files_with_matches|count with an explicit error for unknown values (never silent) and
  labeled `#TRUNCATION` when the multiline cap trims a scanned file; `ignore_dirs` extends the
  skip set and accepts BOTH a comma-string and a list; multiline uses `content.split("\n")` for
  line-model consistency and is bounded by the module constant `_SEARCH_MAX_MULTILINE_BYTES`
  (monkeypatchable). `context_lines=0` output is byte-identical to the previous format (verified
  against `git show HEAD:` in review). Params removed from `hide_args`; decorator + docstring
  updated.
- Tests: 9 regression tests in `tests/tools/test_common_tools_file_tool_ux.py` (context lines,
  byte-identical default, case sensitivity, both output modes, invalid-mode error, ignore_dirs
  string + list, multiline-truncation label + no-false-positive). Green; full tools suite green.
- Review folds: the alt-mode (`files_with_matches`/`count`) multiline truncation was initially
  unlabeled (undercount / false "No matches") — fixed with the same `+1`-detect + label; the
  cap hoisted to a module constant for testability; the perf early-break added.
- Docs: CHANGELOG (Fixed).
- Residual risk: `multiline=True` does not add context lines (documented in the docstring).
  ADR impact: None.
