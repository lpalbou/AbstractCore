# Completed: read_file byte-true char-offset continuation (non-ASCII correctness)

## Metadata
- Created: 2026-07-25
- Status: Completed
- Completed: 2026-07-25
- Origin: tools quality audit 2026-07-25 (both adversary passes), probe-verified.

## Problem
`read_file`'s oversized/minified continuation lane mixes byte and character units.
`_partial_chunk` opened the file in **text** mode and called `fh.seek(offset)` — for a
`TextIOWrapper`, a non-zero seek argument is a **byte** offset (only `0` or a value returned
by `tell()` is a valid text cookie). But the continuation value handed back in the
`#TRUNCATION` footer was `start_char = offset + len(chunk)` — computed in **characters** —
and `total` was `st_size` in **bytes**. The range-mode overflow path had the same defect.

Probe-verified: on a 2-byte-per-char file, `seek(N)` lands at char `N/2`, so the "next part"
call re-reads overlapping content, and an odd byte offset lands mid-UTF-8-sequence, producing
a U+FFFD at the chunk head (masked by `errors="replace"`). On pure ASCII bytes == chars,
which is why it passed casual testing and only bit i18n / emoji / accented content — the
failure was invisible to the model, the worst kind.

## Approach
- Open the file binary in `_partial_chunk`; `seek(byte_offset)`; read a byte budget; decode
  on a codepoint boundary. Report byte offsets and byte totals consistently.
- Set the continuation offset to the byte position where the decoded chunk ended (a codepoint
  boundary), so the next chunk neither overlaps nor starts mid-sequence.
- Document `start_char` as a byte offset. Fix the range-mode overflow path the same way.

## Completion report
- Date: 2026-07-25.
- Implemented (`abstractcore/tools/common_tools.py`, `read_file._partial_chunk` + the
  range-overflow offset): binary read + byte seek; codepoint-boundary trim (drop up to 3
  trailing bytes to the last complete UTF-8 sequence when not at EOF, so the seam never splits
  a char; the trimmed bytes are re-read cleanly by the next chunk); byte-true continuation
  offset; a genuinely-corrupt-mid-chunk region falls back to `errors="replace"` consuming the
  whole budget; the range-overflow offset counts bytes (binary read) not chars; notices/pct
  are byte/byte; the `start_char` docstring now says byte offset (copied verbatim from the
  footer).
- Tests: `tests/tools/test_common_tools_file_tool_ux.py` gains a continuation-chain harness +
  `test_read_file_partial_continuation_is_byte_true_on_multibyte` (200k-char multibyte file
  reassembles exactly across ≥2 chunks, zero U+FFFD, strictly increasing offsets) and
  `test_read_file_partial_continuation_exact_on_ascii` (ASCII parity). Green; full tools suite green.
- Docs: CHANGELOG (Fixed).
- Behavior change: continuation offsets are byte-true; ASCII behavior unchanged.
- Residual risk: lone-`\r` (classic-Mac) files diverge between text/binary line counting in the
  range-overflow path — degenerate, graceful output (noted by the review; not worth a fix).
  ADR impact: None.
