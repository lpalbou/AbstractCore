# Proposed: analyze_media tool — agents see the media they capture (delegated sight)

## Metadata
- Created: 2026-07-21
- Status: Completed
- Origin: laurent's directive via camera seat (commons c3969: "give agent the capability
  to also see the image or video they took"); core accepted ownership of shape (B) at c3977.

## Problem
An agent can CAPTURE media mid-loop (camera tools return paths/`$artifact` refs) but
cannot SEE it: media enters a run only at run start, and no default tool lets a
text-only main model delegate sight mid-loop. The agent shoots blind.

## Shape (B of camera's c3969 split; (A) tool-result media refs is runtime+agent's lane)
`analyze_media(path_or_artifact, question)` in the default toolset: runs the
CONFIGURED vision route over the media and returns bounded text.

## Constraints (ruled at c3977)
- Rides the EXISTING vision-fallback / `input.image` config route — never a second
  model knob; loud actionable error when no route is configured (501-with-hint
  pattern, never silent).
- Bounded text output (caption/answer), never image tokens into the caller context.
- Classified read-only-with-model-cost so hosts budget/approve it distinctly.
- One attempt, no retry stacking; inherits `retry_wall_clock_budget_s`.

## Precedent
Vision fallback already runs a second model inside `generate()` (two-stage caption →
inject); processing components construct LLMs inside utilities. This is a tool wrapper
over shipped machinery: wrapper + classification + tests.

## Validation
- Text-only main + configured vision route: returns grounded answer about a real image.
- No vision route configured: loud actionable refusal.
- Classification visible in tool inventory (approval/budget surfaces).

## Completion report (2026-11-07)

- Summary: shipped. `analyze_media` reads the host-injected `_session_route` stamp and uses the session's own
  provider and model first (`abstractcore/tools/common_tools.py` `_analyze_media_session_route`, ~14083;
  malformed stamps are reported with a `#FALLBACK` marker, never guessed).
- Related: `0837` option B (the Core half) is the same mechanism; option A (tool-result media refs) stays with
  the gateway and runtime.
- Closed by the 2026-11-07 hygiene pass after a code audit.
