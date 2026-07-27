# Completed: fix skim_files dead structure-detection regexes

## Metadata
- Created: 2026-07-25
- Status: Completed
- Completed: 2026-07-25
- Origin: tools quality audit 2026-07-25 (correctness + SOTA adversary passes), verified first-hand.

## Problem
`skim_files` advertises structure-aware "lecture diagonale" sampling, but its five
structure/heading regexes are raw strings with doubled backslashes, so they match a
literal backslash instead of the intended character class:
- `common_tools.py:3401` `re.match(r"^#{1,6}\\s+\\S", s)` — markdown heading
- `common_tools.py:3373` `re.match(r"^[-=]{3,}\\s*$", stripped)` — setext underline / HR
- `common_tools.py:3377` list/checkbox detection
- `common_tools.py:3385` `re.match(r"^(class|def)\\s+\\w+", stripped)` — code structure
- `common_tools.py:3403` `_SENTENCE_END_RE = re.compile(r"([.!?])(\\s+|$)")` — sentence split

Probe-verified: `## Title`, `- item`, `def foo():` all return `False`; the sentence
splitter only ever matches at end-of-string (its `$` branch), so `_first_sentence`
degrades to the whole line truncated at 240 chars.

## Impact
Not a crash — bookends, paragraph-start sampling, and the non-regex markers (`|`, `>`,
fences, `@`, ALL-CAPS, trailing colon) still carry the tool. But heading prioritisation
in the middle sample, `heading_followup` (include the line after a heading), and
`class`/`def` anchoring are all no-ops. Since several tools' `when_to_use` point at
`skim_files` as the pre-read step, every skim-first workflow silently pays the quality tax.

## Approach
1. Remove the doubled backslashes in the six patterns above (`\\s`→`\s`, `\\S`→`\S`,
   `\\d`→`\d`, `\\w`→`\w`).
2. Hoist the five regexes to module scope (compiled once) so a future formatter/refactor
   pass cannot silently re-break them inside the closure — the bug pattern reads
   plausibly on review, so the structural guard matters more than the one-line fix.

## Completion report
- Date: 2026-07-25.
- Implemented: the five patterns were un-doubled and HOISTED to module-level compiled
  constants `_SKIM_HEADING_RE` / `_SKIM_HR_RE` / `_SKIM_LIST_RE` / `_SKIM_CODE_DECL_RE` /
  `_SKIM_SENTENCE_END_RE` in `abstractcore/tools/common_tools.py` (just above the
  `skim_files` @tool decorator); the nested `_is_structure_marker` / `_is_heading_line` /
  `_first_sentence` helpers now reference them.
- Tests: `tests/tools/test_common_tools_skim_files_basic.py` gains
  `test_skim_structure_patterns_are_live_regression` (isolation test asserting each pattern
  fires on real markdown/code and rejects non-matches — would have caught the dead regexes)
  and `test_skim_files_prioritizes_midfile_heading` (end-to-end, asserts a mid-file heading
  AND its follow-up line surface — the discriminating assertion, since the follow-up appears
  only when the heading rule is live). 9 skim tests green; full tools suite green.
- Docs: CHANGELOG (Fixed).
- Behavior change: structure-aware sampling (heading prioritisation, `heading_followup`,
  `def`/`class` anchoring, sentence splitting) is now live; default sampling for files with
  no detectable structure is unchanged.
- Residual risk: none. ADR impact: None (a bug fix, no new durable rule).
