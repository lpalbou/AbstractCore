# Planned: assert media delivery positively; close the structured-output and streaming honesty gaps

## Metadata
- Created: 2026-08-21
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: `0001` (no silent degradation — directly authoritative), `0002` (validation and
  evidence)
- ADR impact: None expected. This item implements ADR `0001`'s existing rule ("fail closed on
  correctness-critical paths where degraded output would be misleading") on three paths that
  currently escape it.

## Context

The 2026-08-21 delegated-sight work made the MLX provider honest about images it could not carry:
it records `dropped_media`, surfaces `response.metadata["media_dropped"]`, and warns. That work is
correct and should stay. It has three holes, all found while designing MLX native vision
(`0840`), all present today and independent of it.

The holes share one root cause: **the contract is a negative-absence claim.** "The image was
delivered" is expressed as "no `media_dropped` key". Anything that skips the honesty code
therefore reads as success.

## Current code reality

Verified in `abstractcore/providers/mlx_provider.py::_generate_internal`:

1. **`response_model=` drops media silently.** The Outlines structured-output branch returns
   before the media block is reached — the branch's success and error returns both precede the
   `# Handle media content first if present` comment. So
   `generate("what is in this image?", media=[img], response_model=X)` on MLX ignores the image and
   attaches **no** `media_dropped` marker at all.
2. **Streaming has no honesty channel.** `media_dropped` and the enrichment metadata are attached
   only on the synchronous branch; the streaming branch returns its generator earlier, with
   neither.
3. **The consumer gate reads absence as success.**
   `abstractcore/media/vision_fallback.py::VisionFallbackHandler._generate_description` raises
   `VisionGenerationError` only `if dropped:`. An absent key means delivered.

Structured output additionally cannot carry a vision lane at all: the Outlines MLX adapter accepts
only `str`/`Chat` input and re-encodes the prompt with the plain tokenizer, so an image's expanded
placeholder tokens cannot survive that path.

## Problem

A request that silently loses its image is indistinguishable from one that delivered it — to the
one code path built specifically to catch that. `analyze_media` will then stamp the model's prose
as an observation with provenance. That is the exact regression the delegated-sight work fixed,
reachable today through two ordinary API shapes (`response_model=`, `stream=True`).

It gets worse once `0840` lands, because the vision lane's success contract is *also*
`dropped_media == []` — the absence of a negative — and several of its failure paths reach that
state.

## What we want to do

Replace the negative-absence claim with a **positive delivery record**, written only after media
has actually entered the forward pass, and make consumers gate on its presence.

Close the two paths that bypass honesty entirely.

## Why

ADR `0001` requires that best-effort behaviour be observable and that correctness-critical paths
fail closed. A modality claim is correctness-critical: downstream tooling attributes model prose to
an image on the strength of it.

Doing this before `0840` also means the vision lane is born with a contract that cannot silently
regress — rather than retrofitting one onto a lane that already shipped.

## Requirements

- A positive per-request record of what media actually reached the model — enough to distinguish
  "delivered" from "not attempted" from "attempted and failed". Something like
  `media_delivered: [{index, kind, sha256, tokens}]`, written **after** the merge/encode returned.
- The record is attached on **both** the synchronous and streaming paths.
- `response_model=` + `media=` either routes through a path that can carry media, or refuses with a
  named error. It must not continue silently.
- `media/vision_fallback.py` and `analyze_media` gate on the **presence of the positive record**,
  not the absence of `media_dropped`.
- `dropped_media` and its reason strings stay. They are the correct report for a text-only
  checkpoint and for every degraded path; the two channels are complementary.
- Every failure reason is a named literal, not free prose: at minimum
  `media_processing_unavailable`, `media_processing_failed`, `structured_output_unsupported`,
  and (once `0840` lands) `mlx_vlm_not_installed`, `vision_family_unsupported`,
  `vision_encode_failed`.

## Suggested implementation

1. Move the media block **above** the Outlines branch, or gate the Outlines branch on
   `not media`. The second is smaller and more obviously correct; prefer it unless moving the block
   turns out to be clean.
2. Attach `media_delivered` and the enrichment metadata in one shared helper called from both the
   sync and streaming return paths, so a future third path cannot miss it by omission.
3. Flip the two consumer gates to require the positive record.
4. Audit the other providers for the same two shapes (structured-output-before-media, streaming
   without metadata) — this is a provider-shaped bug, and MLX is unlikely to be the only one.

## Scope

- `providers/mlx_provider.py::_generate_internal` (both return paths and the Outlines branch).
- `media/vision_fallback.py` consumer gate.
- `tools/common_tools.py::analyze_media` provenance gate.
- A read-only audit of the other providers, filed as follow-ups rather than fixed here if the shape
  differs.

## Non-goals

- Implementing vision transport (`0840`).
- Fixing capability *advertisement* (`0841`) — this item is about what is reported after a request,
  not what is offered before one.
- Making structured output work **with** images. Refusing loudly is sufficient and correct for now;
  the Outlines MLX adapter cannot carry expanded placeholder tokens.

## Dependencies and related tasks

- `0840` — MLX native vision. **This item should land first or together**; `0840`'s failure paths
  depend on the positive assertion existing.
- `0841` — transport-actual capability. Complementary, not blocking.
- The 2026-08-21 delegated-sight honesty work, which this extends rather than replaces.

## Expected outcomes

- No API shape silently loses an image on MLX.
- `analyze_media` cannot stamp a text-only answer with sight provenance, on any path.
- A single machine-readable answer to "did the model actually see it", usable by callers.

## Validation

- `generate(..., media=[img], response_model=X)` on a text-only MLX checkpoint: asserts a named
  refusal or a `media_dropped` marker. **Must fail against today's code** — write this test first
  and watch it fail.
- `generate(..., media=[img], stream=True)`: the streamed response carries the same media metadata
  as the equivalent non-streamed call.
- `VisionFallbackHandler` raises when the positive record is absent, including when
  `media_dropped` is also absent.
- A test that a delivered image produces a `media_delivered` entry whose token count is greater
  than one — i.e. the placeholder actually expanded — so the record cannot be faked by a code path
  that merely intended to deliver.
- Existing media and tool suites unchanged, in particular
  `tests/tools/test_analyze_media_refuses_undelivered_image.py`.

## Progress checklist

- [ ] Write the three failing tests first
- [ ] Gate or reorder the Outlines branch so media is never skipped silently
- [ ] Shared metadata-attachment helper used by both return paths
- [ ] `media_delivered` record with per-item token counts
- [ ] Flip the `vision_fallback` and `analyze_media` gates to the positive
- [ ] Named reason literals documented in one place
- [ ] Audit the other providers; file follow-ups

## Guidance for the implementing agent

**Write the failing tests before the fix.** All three defects are invisible by construction — they
produce ordinary successful responses. A test suite that passes before and after proves nothing
here, and ADR `0002` asks for evidence rather than assertion.

**Do not delete `dropped_media`.** It is the right answer for text-only checkpoints and for every
degraded path, and callers already read it. The change is to add a positive channel and move the
*gates* onto it — not to replace the negative one.

**Resist making structured output carry images.** It looks like the natural fix and it is a trap:
the Outlines MLX adapter re-encodes a prompt string with the plain tokenizer, so image placeholder
expansion cannot survive. A named refusal is the honest outcome.
