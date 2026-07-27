# Proposed: read-boundary hygiene for edit_file / analyze_code — REFRAMED (the original premise was wrong)

## Metadata
- Created: 2026-07-25
- Status: Proposed — REFRAMED 2026-07-25 after the operator challenged the premise and a
  fable5 adversary confirmed the challenge. The original item conflated TOOL-PROCESS MEMORY
  with MODEL TOKENS; that framing is retracted below and replaced with the real, narrower work.
- Origin: tools quality audit 2026-07-25; operator correction ("I don't see where the model
  would EVER read the whole file"; "analyze_code should work on the full file independently of
  its size"); verified by adversary.

## What the original item got WRONG (retracted)
The original hook was "the suite's discipline is every read bounded + labeled, and these two
bypass it." That is a category error: `read_file`'s caps and `#TRUNCATION` labels govern
MODEL-FACING prompt currency. Neither `edit_file` nor `analyze_code` sends the file to the model.
- `edit_file` returns a compact unified diff (n=1 context — output proportional to CHANGED lines,
  `_render_edit_file_diff`) + a post-edit excerpt that is OMITTED entirely when the modified
  region exceeds 220 lines (`common_tools.py:8184-8195`) + a ≤20-message lint notice. A regex
  replace-all across a huge file returns exactly the changed sections — the operator's own
  description of correct behavior. This is already the shipped design.
- `analyze_code` returns an OUTLINE (one line per definition, bodies never included); the engine
  lane caps every section at `MAX_SECTION_ENTRIES=50`. Bounded output regardless of input size.
- The whole-file `f.read()` in `edit_file` is REQUIRED for correctness: the uniqueness guarantee
  (count all matches or refuse), `re.MULTILINE|re.DOTALL` matching, the Python/JSON/YAML parse
  guards, dominant-newline-style detection, and the faithful full-file diff all need the whole
  content. A "read only relevant parts" edit_file would be a weaker, unsafe tool. DROP the
  proposed 10-20 MB refusal — it would break legitimate edits to large generated/data files the
  tool handles correctly today.

## The REAL work (narrow, verified)
### R1 — analyze_code deep-lane OUTPUT caps (the actual gap, opposite of the original proposal)
The python deep lane's imports/classes/functions sections extend WITHOUT a limit
(`common_tools.py:1298-1316`); only relationships are capped `[:50]`. A pathological generated
`.py` with tens of thousands of defs floods the model through the OUTPUT, not the read. Give the
deep lanes `MAX_SECTION_ENTRIES` parity with the engine lane (cap + a "N more — narrow with …"
recovery hint). This is the model-facing bound that genuinely matters.

### R2 — one shared absurd-input OOM guard (stat-based, both tools)
The only real memory concern is a genuinely pathological input (hundreds of MB+, text suffix,
not `.abstractignore`'d): `edit_file` allocates ~5-8× file size (raw + LF copy + updated copy +
difflib line lists) and `analyze_code` runs `ast.parse` (~10-30× source). Add ONE shared
`stat()`-based guard at a TRULY absurd, configurable threshold (order ~100-256 MB), framed
purely as host-OOM protection with an honest refusal ("use write_file for full rewrites of
generated artifacts"). Not the suite's read-caps — a separate, much higher ceiling.

### R3 — fix the engine cap that defeats its own purpose + reconsider its value
`code_analysis.read_text_bounded` does `raw = path.read_bytes()` (`:680`) — reads the ENTIRE
file, THEN slices to 4 MB (`:684-686`). So the 4 MB cap delivers TRUNCATED analysis (declarations
past 4 MB vanish from the outline of exactly the files where an outline beats raw reading) WITHOUT
delivering OOM protection (the full read already happened). Two fixes: (a) read only up to the
guard threshold from R2 (stop defeating memory protection); (b) reconsider whether the 4 MB
ANALYSIS cap should exist at all — the operator's intent is full-file structure; the engine is
O(n) line-anchored regex with a 5000-char minified-line short-circuit already, so raising/removing
the analysis cap (keeping only R2's absurd-input floor) matches the intent.

### R4 — relationship-pass optimization (real, keep from the original)
`analyze_code`'s python relationship pass re-walks the full AST once per method
(`common_tools.py:1219-1230`) — O(methods × nodes). Replace with a single indexed pass. Pure CPU
win on large files; no behavior change.

## Validation
- A generated `.py` with thousands of defs yields a CAPPED outline (R1), not a flooded context.
- A large-but-normal source file analyzes fully (R3) — no "first 4 MB only" truncation.
- A ~500 MB text-suffixed file refuses with an OOM-framed message (R2), not an OOM.
- Relationship-section output unchanged on normal files; CPU linear in file size (R4).
