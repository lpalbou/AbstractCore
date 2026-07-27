# Proposed: batch multi-edit and multi-file atomic edit (SOTA parity study)

## Metadata
- Created: 2026-07-25
- Status: Proposed (research-backed; not a commitment)
- Origin: operator research question 2026-07-25 ("do we enable multiple file edits in a single
  call? is it SOTA?"), with a web-search comparison of Claude Code / Cursor / aider / Codex CLI.

## Current code reality (verified 2026-07-25)
- `edit_file` (`abstractcore/tools/common_tools.py`) is the ONLY editing tool besides `write_file`.
  It edits ONE file per call, but WITHIN that file it already supports MULTIPLE edits in a single
  call two ways: (a) unified-diff mode applies MULTIPLE HUNKS atomically in one call
  (probe-verified: a 2-hunk patch changed lines 1 and 4 in one call, one write); (b)
  `max_replacements=-1`/N replaces many occurrences of one pattern. It does NOT accept a
  Claude-Code-style `edits` array of distinct find/replace PAIRS, and it cannot edit MULTIPLE
  FILES in one call.
- So we are at rough parity with single-file multi-edit; the genuine gap is a single atomic call
  that spans MULTIPLE files.

## SOTA (2026, web-searched)
- Claude Code ships a built-in `MultiEdit` for a SINGLE file (an `edits` array applied
  sequentially + atomically). Multi-FILE atomic edits are NOT a built-in terminal-agent feature —
  they are provided by an MCP extension (`mcp-multi-edit`: `multi_edit` + `multi_edit_files` with
  all-or-nothing rollback, dry-run, backups, structured error codes) or by an IDE (Cursor
  Composer coordinates cross-file edits with live static analysis + conflict rollback).
- aider emits multiple SEARCH/REPLACE blocks per model response (can span files), but that is the
  MODEL emitting N blocks each applied individually — not one atomic tool call. Codex CLI is
  single-turn.
- Net: single-file multi-edit is table stakes (we have it via diff mode); multi-file ATOMIC edit
  in one tool call is an MCP/IDE-tier capability, not a terminal-agent built-in.

## SOTA survey findings (folded from code-tui c5588 + runtime c5589, 2026-07-25, file:line in their review)
A three-framework survey (codex / opencode / pi) commissioned during the tool-batching incident,
folded here per the durable-home pointer:
- **No SOTA framework bans batching writes in its PROMPT.** Same-file safety is solved in the
  HARNESS: codex serializes non-parallel-safe tools with an exclusive write-lock; pi uses a
  per-file mutation queue (different files concurrent); opencode a per-file semaphore. (Our
  executor already serializes ALL batched mutations in payload order — runtime 0214 + the c5584
  pin — which is STRICTER than pi/opencode's per-file concurrency, so cross-file batching is
  correctness-safe today with zero runtime change; a per-file queue with cross-file concurrency
  is a possible future 0214 optimization, measured need first.)
- **Round-trip economy is solved with RICHER SINGLE CALLS**, exactly the D direction here:
  codex `apply_patch` = one call, N files × N hunks, context-anchored; pi `edit` = one call, one
  file × N `edits[]` all anchored to the ORIGINAL text with overlapping edits rejected. pi ships
  the exact single-file teaching line our spec lacks: "use one edit call with multiple entries in
  edits[] instead of multiple edit calls" (`edit.ts:301`).
- **CRUCIAL design constraint — richer patch formats must be MODEL-GATED.** Patch-format fluency
  is a per-model TRAINING fact: opencode gates its patch tool to `gpt-* && !oss && !gpt-4`; codex
  carries a per-model `apply_patch_tool_type` + per-model prompts. So a `apply_patch`-style
  one-call-many-files shape must be gated to the model families trained on it (GPT-5-class);
  qwen/oss-class models must keep the simple find/replace + unified-diff shapes. Any 0836 build
  MUST condition the richer shape on a model-capability signal (candidate: a
  `patch_format`/`apply_patch` capability in `model_capabilities.json`), never expose it unconditionally.

## Why it could be valuable (the case for building)
- Cross-file refactors (rename a symbol across N files, migrate an import) currently cost N
  `edit_file` calls — N× context tokens + latency + no atomicity (a mid-sequence failure leaves a
  half-applied refactor). One atomic `edit_files` with rollback would cut calls and remove the
  partial-state risk.

## Approach options (if pursued)
1. `edit_files(edits=[{file_path, pattern, replacement, use_regex, ...}, ...], atomic=True)` — a
   thin batch over the existing `edit_file` engine: validate all edits (dry-run) first, apply all
   or none (write to temp + `os.replace` per file, roll back on any failure), return a combined
   diff + per-file status. Reuses `edit_file`'s uniqueness check, parse guards, exact-match-first
   escape handling (0829), and atomic write (proposed 0833) per file.
2. Alternatively a single-file `edits`-array variant to mirror Claude Code MultiEdit / pi
   `edits[]` ergonomics — lower value since diff mode already covers the single-file case.
3. `apply_patch`-style ONE call = N files × N hunks (codex's shape, the "D" step of the fix
   menu) — the richest economy, but MUST be MODEL-GATED (see the SOTA constraint above): expose
   it only to families trained on the format (GPT-5-class) via a `model_capabilities.json`
   signal; qwen/oss-class keep find/replace + unified diff. Deferred pending a decision.
- Depends on 0833 (atomic writes) for the rollback primitive.

## Non-goals / open questions
- Not a commitment — filed for memory pending operator direction. Do not build without a decision.
- Cross-file atomicity is only meaningful with per-file atomic writes (0833); sequence those.
- Whether the batch should share one uniqueness/guard policy across files or per-file.

## Validation (if built)
- A 3-file rename applies in one call; injecting a failure on file 2 rolls back files 1–3 (no
  partial state); dry-run/preview shows the combined diff without writing.
