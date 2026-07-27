# Proposed: atomic writes + overwrite disclosure (write_file, edit_file)

## Metadata
- Created: 2026-07-25
- Status: Proposed
- Origin: tools quality audit 2026-07-25 (SOTA/design adversary), verified.

## Problem
Both writers modify the target in place:
- `write_file` opens `open(path, mode)` directly (`common_tools.py:3742`).
- `edit_file` writes back in place (`:9009-9010`).

A crash / kill / disk-full mid-write truncates the target — the exact failure the rest of
the framework already learned (stores use `tmp.replace()` everywhere for this reason).
`edit_file`'s pre-write parse guards protect against bad CONTENT but nothing protects
against bad TIMING.

Separately, `write_file` with `mode="w"` overwrites silently and the success message does
not even say the file previously existed — a wrong-path write destroys content with no
signal. And `mode` is passed raw to `open()` (`:3742`): `mode="r"` surfaces as "Unexpected
error", `mode="wb"` fails on str content.

## Approach (small, contained)
- Write to a sibling temp file in the same directory, copy the source mode bits, then
  `os.replace()` (atomic on POSIX; same-directory keeps it on one filesystem).
- `write_file`: add "(overwrote existing file, was N bytes)" to the success line when the
  target existed — the cheapest possible clobber mitigation short of read-before-write
  state (which needs host cooperation; out of scope here).
- `write_file`: validate `mode ∈ {"w","a","x"}` with a clear error instead of leaking an
  `open()` exception.

## Non-goals
- Read-before-edit / mtime-staleness handshake (Claude Code tracks read-state) — needs
  host/conversation-state cooperation; a framework seam, not a tool patch.

## Validation
- A simulated failure between temp-write and replace leaves the original intact.
- Overwriting an existing file reports the prior size.
- `mode="r"`/`"wb"` return a clear validation error.
