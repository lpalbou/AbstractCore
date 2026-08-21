# Proposed: clarify model capability vs provider capability as distinct concepts

## Metadata
- Created: 2026-08-21
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `0003` (provider/capability boundaries), `0008` (provider-owned residency truth)
- ADR impact: If this investigation concludes that provider capability is a first-class fact, the
  outcome belongs in an ADR alongside `0008` rather than in provider prose. If it concludes the
  two should stay separate and unlinked, that is also worth recording.

## Context

`assets/model_capabilities.json` describes **model** capability: what these weights can do. That
is correct and useful — `qwen3.8-27b` genuinely has a vision tower, and the entry is right for the
`huggingface` provider.

Separately, a **provider** has capability: what a given transport can actually carry. Before
`0840`, the MLX provider could not carry an image at all, regardless of the weights. After `0840`
it can, for some families and not others. Ollama and LM Studio have the same distinction, keyed on
whether an `mmproj` was pulled alongside the GGUF.

These are genuinely different facts, and the current code does not distinguish them. Discovery
surfaces read the model fact and present it as if it were the effective one — most visibly
`/v1/models?input_type=image`, which lists models whose transport may not accept an image.

**This item is an investigation, not a commitment to a specific fix.** An earlier draft assumed the
answer (a provider-owned predicate consulted by the registry) and was filed as planned work. That
presumed too much: it is not yet established that the two facts should be merged into one query at
all, nor that discovery should reflect the intersection.

## Current code reality

Every capability entry point takes a model **name** and nothing else:

- `architectures/detection.py::supports_vision(model_name)`.
- `providers/model_capabilities.py` — all nine public functions are model-name-only, including
  `filter_models_by_capabilities`, which is what the server's `input_type` filter uses.
- `MLXProvider.list_available_models` is a **`@classmethod`**, so no provider instance exists on
  the discovery path even in principle.
- `media/capabilities.py::MediaCapabilities._apply_provider_adjustments` **has** a `provider`
  argument and uses it only for image size limits and streaming flags — never for
  `vision_support`.
- `media/capabilities.py::_apply_model_adjustments` sets `vision_support = True` for any model name
  containing `vision`, `vl`, or `visual`, unconditionally. This one is indefensible on any design
  and could be removed independently of the rest.
- `media/auto_handler.py::_check_vision_support(model)` gates glyph PDF→image compression on the
  same model-name fact.

## Problem

Three questions, currently unanswered:

1. **Should discovery reflect the intersection?** `/v1/models?input_type=image` today answers
   "which models can take an image". Should it answer "which models can take an image *through the
   provider you asked about*"? There is a real argument for the current behaviour: the catalogue
   describes models, and the caller chooses the transport.
2. **Where would provider capability live?** A static hook on the provider class (reachable from
   the classmethod path), a resolver object, or an instance-only fact that discovery deliberately
   ignores.
3. **What is the honest default when a provider cannot answer?** ADR `0008` chose an explicit
   unknown for residency. Whether capability deserves the same three-state treatment, or whether
   unknown-means-model-fact is acceptable, is exactly what this item should decide.

## What we want to do

Investigate and record a decision. Produce a short written answer to the three questions above,
with the code seams each answer implies, before writing any capability-routing code.

## Why

Because the failure this would prevent is real but narrow, and the fix touches a surface that four
providers and the server all read. Getting the *concept* wrong here is more expensive than the bug
it fixes. After `0840` lands, the residual inaccuracy is small — MLX will genuinely carry images
for the families it supports — which makes this a good time to think rather than a crisis.

## Scope

- The capability query path and the two gates named above.
- A written decision. Code only if the decision calls for it.

## Non-goals

- Changing `assets/model_capabilities.json`. It is correct: it describes weights.
- Implementing vision transport (`0840`).
- Response-level delivery honesty (`0842`) — that is about what we report *after* a request and is
  separately committed.

## Dependencies and related tasks

- `0840` — MLX native vision. Shrinks the inaccuracy this item is about; worth doing first so the
  investigation reasons about the post-`0840` world.
- `0842` — positive media-delivery assertion. Covers the response-level half of honesty, so this
  item can focus purely on the discovery-level question.
- `docs/adr/0008-provider-owned-model-residency-truth.md` — the closest existing precedent, for a
  different fact.

## Expected outcomes

- A recorded answer to "are model capability and provider capability one query or two", with
  rationale.
- Either a scoped implementation item, or a recorded decision that the current split is correct and
  only the name-substring heuristic needs removing.

## Validation

- The decision is written down somewhere durable (ADR preferred) and cites the seams.
- If code follows: a server test showing the chosen behaviour for a provider that can answer and
  one that cannot.

## Progress checklist

- [ ] Answer question 1: should discovery intersect?
- [ ] Answer question 2: where does provider capability live?
- [ ] Answer question 3: what is the unknown default?
- [ ] Record in an ADR
- [ ] Remove the name-substring heuristic in `media/capabilities.py` (independent, do this anyway)
- [ ] File a scoped implementation item if warranted

## Guidance for the implementing agent

**Do not start by writing a resolver.** The first deliverable is prose: which fact belongs to whom,
and whether a caller asking the catalogue a question should get a transport-qualified answer. An
earlier draft of this item skipped that step and proposed a mechanism; the mechanism may well be
right, but it was not argued.

**One change is safe to make regardless of the outcome:** `media/capabilities.py` inferring
`vision_support = True` from the substrings `vision`/`vl`/`visual` in a model name is wrong under
every possible answer to the three questions.
