# Completed: edit_file exact-match-first escape handling

## Metadata
- Created: 2026-07-25
- Status: COMPLETED 2026-07-25 (implemented + tested + fable5-reviewed; operator-approved after
  the defect was independently confirmed with a live reproduction).
- Origin: tools quality audit 2026-07-25; the operator was initially unconvinced, so an
  adversary re-derived it end-to-end (live repro) before implementation.

## Problem (confirmed real, worse than first framed)
`edit_file` unconditionally rewrote literal escape sequences in BOTH `pattern` and
`replacement` before any match (`_normalize_escape_sequences` at both call sites), converting
`\n`/`\t`/`\r` (two-char forms) into real control characters. Consequences (reproduced live):
- A replacement containing a literal escape (e.g. writing `sep = "\n"` into JS source, or
  `re.split("\\n", x)` into Python) got a real line break written. On unguarded types
  (.js/.c/.md/.txt) the corruption wrote through silently; on guarded types (.py/.json/.yaml)
  the parse guard refused "would introduce a syntax error" WITHOUT revealing the tool's rewrite —
  an unfixable retry loop (no encoding, not even double-escaping, could get a literal `\n` through).
- Regex patterns/templates using `\n`/`\t` were corrupted before `re.compile`/`re.sub`.

## Implemented fix
- Removed the up-front `_normalize_escape_sequences` on `pattern`/`replacement`; both are kept
  VERBATIM (`common_tools.py`, edit_file body).
- A PROBE runs ONLY for literal find/replace (never `use_regex`, never range-replace): if the RAW
  pattern matches nothing but the escape-normalized pattern does, it swaps to the normalized
  pattern and records `escape_note` (the weak-model over-escape affordance, now a labeled
  fallback). Existence uses `search_content.count` + `_flexible_whitespace_match`, mirroring the
  real match logic.
- The `replacement` is NEVER normalized (writing a literal escape is possible; a weak model that
  over-escapes only the replacement writes a visible, correctable literal instead of silent
  corruption).
- Regex mode is never normalized (regex assigns its own escape semantics).
- `escape_note` is surfaced on EVERY post-swap surface — the success render, both guard-refusal
  paths (python-syntax + generic), the ambiguity refusal, and the "no changes applied" path — so
  a refusal or no-op can never hide a rewrite.

## fable5 depth/quality review: SOLID (no blocking defect)
Two low-severity polish items folded: (a) the ambiguity + no-op surfaces omitted the escape note
(now disclosed); (b) two regex tests were non-discriminating (passed under the old code too) —
replaced with discriminating shapes (regex `a\\nb` matching a literal backslash-n in content; a
re.sub template emitting a literal backslash-n) that fail on the old normalization.

## Validation (all green)
- 68 existing edit_file tests + 10 new regression tests (`tests/tools/test_common_tools_edit_file_literal_escapes.py`):
  literal backslash-n written verbatim (JS + range-replace), literal backslash-t, match source
  containing a literal escape, over-escaped-pattern fallback with disclosure, raw-pattern-wins
  (no spurious note), regex `\n` template uncorrupted, regex `\s`/`\d` pattern intact, and the two
  discriminating regex tests.
- Live probes across the headline case, fallback, regex integrity, range-replace.
- Full tools suite green (494 non-browser/shell + 52 browser/shell).
- No existing test asserted the old normalization behavior (verified) — the fix contradicts nothing.
