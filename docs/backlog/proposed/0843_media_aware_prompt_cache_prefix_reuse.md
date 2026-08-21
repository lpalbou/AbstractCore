# Proposed: media-aware prompt-cache prefix reuse for the MLX vision lane

## Metadata
- Created: 2026-08-21
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `0007` (durable memory bloc/cache binding), `0001` (no silent degradation)
- ADR impact: If adopted, the rule "a KV prefix containing media tokens is reusable only when the
  media content hash also matches" is a durable cache-correctness invariant and belongs in ADR
  `0007` or a sibling.

## Context

`0840` deliberately bypasses prompt-cache reuse on any turn that carries media, and refuses to
persist a KV artifact whose cache contains image tokens. That is the correct v1 decision: it is
four lines, obviously correct, and it removes an entire class of silent-wrongness. This item
records what it costs and what adopting it back would require, so the deferral is a decision on the
record rather than an omission.

## Current code reality

`MLXProvider._prepare_cache_delta_feed` matches the current turn's token ids against the recorded
`fed_token_ids` for a cache key by longest common prefix. The word "image" does not appear anywhere
in that subsystem. Neither does any notion of media content.

`mlx_vlm` solved the same problem and its vocabulary is the reference: `hash_image_payload`,
`multimodal_token_ids_from_config`, `media_token_spans`, `media_safe_prefix_min`,
`prefix_contains_media_tokens`, `adjust_prefix_to_text_suffix_boundary` — with the image hash
folded into every block key.

## Problem

Three distinct defects, all silent, if the vision lane were wired into the cache path naively:

1. **Same-shape different-image poisoning.** Two requests with identical text and two *different*
   images **of the same pixel dimensions** produce byte-identical token id sequences, because the
   processor expands the placeholder by grid size. The LCP check reports a full hit and the model
   answers about the **previous** image, with a cache-hit recorded. Note the asymmetry: a
   different-sized image changes the token count and the cache correctly rebuilds — so the failure
   fires precisely in the batch/pipeline case (screenshots, scanned pages, fixed-size frames) that
   a cache key exists for.
2. **Two tokenizers, one arithmetic.** `_encode_prompt_token_ids` uses the mlx-lm tokenizer, where
   an image is **one** token. The KV actually contains the processor's **N** expanded placeholders.
   For trimmable families the trim length is then computed against the wrong sequence and cuts into
   the middle of an image block. For hybrid families the lengths never agree and the cache silently
   never reuses.
3. **Prefill without embeddings.** The hybrid-snapshot path prefills token ids through a helper
   that has no `input_embeddings` parameter, so image placeholder ids would be embedded as ordinary
   vocabulary tokens (the placeholder id is *within* vocab, so no `IndexError`) — filling the KV
   with hundreds of copies of an untrained embedding.

The persisted-artifact fingerprints compound this: a saved cache validates on engine, tokenizer,
model-config and weights axes and carries **no record of which image produced it**.

## What we want to do

Nothing, for now. Record the deferral, and specify what adoption would require if the cost of the
bypass becomes visible.

## Why

Because the bypass costs less than it appears to. It fires **only on turns that carry a new
image**. A multi-turn conversation about one image pays the image prefill once, on the turn the
image arrives; subsequent text-only turns take the normal delta-feed path with the image already in
the KV. The widely-cited "~25× faster follow-up" benefit of unified text+vision engines is a claim
about **text-only turns on a VLM**, and those are unaffected — they remain today's mlx-lm path with
today's full prompt cache.

The pathological case is an image on every turn, which is exactly the case where the prefix
genuinely cannot be reused, because the pixels changed.

And the cost of getting it wrong is the worst failure mode in this whole area: a confident answer
about a *different image*, with a cache hit in telemetry and no error anywhere.

## Requirements (if adopted)

- A media content hash (over `pixel_values`, **never** the file path) folded into the fed-token-id
  record and into the cache key metadata.
- A media-span predicate: reuse a prefix only if it leaves a text-only suffix, or if every media
  span in the shared prefix has a matching content hash.
- One id sequence: the processor's expanded ids are the sole source for both the KV and the
  fed-token record.
- Prefill helpers accept an embeddings slice in lockstep with the token slice, or refuse
  media-bearing prefill.
- A sixth fingerprint axis covering the vision encoder version, or a refusal to persist any KV
  containing media tokens.

## Suggested implementation

Cheapest useful increment first — this does **not** need the full apparatus:

1. **Hash-gated exact reuse (~10 lines).** Allow reuse when the recorded fed ids match **and** a
   sha256 of the concatenated image bytes matches. Covers the common "same image, follow-up
   question" case and nothing else.
2. Only if measurement justifies it: media spans, safe-prefix computation, and per-span hashing,
   following mlx-vlm's predicate set.

Do not port the whole subsystem speculatively. For scale: `lmstudio-ai/mlx-engine` needed a
multi-file prompt-cache package (image spans, restore planner, records, blob store) to do this
safely for a batching server. AbstractCore's MLX lane serves one request at a time.

## Scope

- `MLXProvider` prompt-cache delta feed, snapshot lane, and artifact persistence, for
  media-bearing turns only.

## Non-goals

- Any change to text-only cache behaviour. It is correct and heavily invested in; this item must
  not touch it.
- Cross-process or disk-backed vision feature caching.
- Multi-image and video caching — unsolved upstream (Blaizzy/mlx-vlm#832,
  lmstudio-ai/mlx-engine#287).

## Dependencies and related tasks

- `0840` — implements the bypass this item would relax. Must land first.
- `0842` — the positive delivery assertion makes a cache-poisoning regression detectable.

## Expected outcomes

- Either a measured case for hash-gated reuse, or a recorded decision that the bypass is
  permanent and cheap.

## Validation

- **The poisoning test comes first, and must fail before any reuse is enabled**: same text, two
  different images of identical dimensions, asserting the second answer describes the second image.
- A test that the fed-token record for a media turn is not a prefix-match for a different image.
- Text-only cache behaviour byte-identical before and after.
- A benchmark quantifying what the bypass actually costs on a realistic image conversation, since
  that number is what justifies adopting this at all.

## Progress checklist

- [ ] Measure the real cost of the bypass on a multi-turn image conversation
- [ ] If material: write the poisoning test, watch it fail
- [ ] Hash-gated exact reuse
- [ ] Decide whether spans are needed; record the decision
- [ ] ADR for the cache-reuse invariant if adopted

## Guidance for the implementing agent

**Do not start this because it looks like the "proper" solution.** The bypass in `0840` is not a
shortcut — it is the correct default, and this item exists so that reversing it requires evidence.
Measure the cost before writing the fix.

**If you do enable reuse, the poisoning test is not optional and must be written first.** A guard
that is subtly wrong here produces an answer about the previous image with no error, which is
strictly worse than a slow turn.
