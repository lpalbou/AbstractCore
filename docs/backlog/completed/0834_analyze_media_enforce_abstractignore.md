# Completed: analyze_media enforce .abstractignore (data boundary)

## Metadata
- Created: 2026-07-25
- Status: Completed
- Completed: 2026-07-25
- Origin: tools quality audit 2026-07-25 (SOTA adversary), verified.

## Problem
Every file-touching builtin checks `.abstractignore` before acting — except `analyze_media`,
which is the one tool that ships file BYTES to a possibly-remote vision provider. It gated on
exists / is_file / suffix / PIL-decodes and then dispatched to the configured vision route with
no ignore-policy check. Backwards: the tool most in need of the boundary was the one missing it.
An operator who ignored a directory to keep artifacts/secrets out of tool reach reasonably
expects that to include the tool that exfiltrates content off-host.

## Approach
Add the same ignore-policy check the sibling file tools use, before dispatch; on an ignored
path return the standard actionable refusal (name the path, name `.abstractignore`).

## Completion report
- Date: 2026-07-25.
- Implemented (`abstractcore/tools/common_tools.py`, `analyze_media`): `AbstractIgnore.for_path`
  + `is_ignored` check inserted after the exists/is_file checks and BEFORE the suffix/PIL-decode
  checks and before the vision-handler import — so an ignored path is refused before any bytes
  can leave the host. Refusal names the path and explains the tool sends bytes to a possibly
  remote route.
- Tests: `tests/tools/test_analyze_media_tool.py` gains
  `test_ignored_path_refused_before_any_vision_dispatch` — a tripwire monkeypatch on
  `VisionFallbackHandler.create_description_with_trace` asserts it is NEVER reached for an ignored
  path. 14 analyze_media tests green; full tools suite green.
- Docs: CHANGELOG (Fixed).
- Residual risk: the ignore check runs after exists/is_file (so the error can disclose whether an
  ignored path exists) — cosmetic; read_file's ignore-first ordering is marginally cleaner if this
  is ever touched again. ADR impact: None (a data-boundary consistency fix).
