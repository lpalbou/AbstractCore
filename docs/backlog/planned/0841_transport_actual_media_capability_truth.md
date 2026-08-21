# Planned: make media capability a `(model × provider × runtime)` fact, not a model-name fact

## Metadata
- Created: 2026-08-21
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: `0008` (provider-owned model residency truth), `0003` (provider/capability
  boundaries), `0001` (no silent degradation)
- ADR impact: **Extend ADR `0008` rather than write a new one.** `0008` already decided this
  question for a different fact. Amend it (or add a sibling ADR that cites it) to cover capability
  as well as residency, including the explicit-unknown rule.

## Context

`assets/model_capabilities.json` marks 133 entries `vision_support: true`. Nothing downstream can
express that a given **transport** cannot honour that claim, so an AbstractCore server fronting
MLX lists `qwen3.8-27b` at `/v1/models?input_type=image`, a client sends an image, gets a fluent
answer, and never learns the model was blind.

This is not a new policy question. `docs/adr/0008-provider-owned-model-residency-truth.md`
(Accepted) already decided it for residency:

> "Model residency is a provider-owned fact. Core can know it when a provider or capability
> implementation can verify the backing runtime state; **higher layers cannot derive it safely from
> configuration or transport shape**."

and set the fail-closed shape:

> "A provider that cannot verify residency returns an explicit unknown result:
> `provider_residency_verified=false`, `provider_resident=null`, and `loaded=false`."

Vision capability has the identical shape and is currently derived exactly the way `0008`
forbids.

## Current code reality

Every capability entry point takes a model **name** and nothing else:

- `architectures/detection.py::supports_vision(model_name)` → `capabilities.get("vision_support")`.
- `providers/model_capabilities.py` — all nine public functions are model-name-only
  (`get_model_capability_routes`, `model_supports_capability_route`,
  `model_matches_capability_routes`, `get_model_input_capabilities`,
  `get_model_output_capabilities`, `model_matches_input_capabilities`,
  `model_matches_output_capabilities`, `filter_models_by_capabilities`,
  `get_capability_summary`).
- `providers/model_capabilities.py` emits the `input.image` route from `vision_support`; that is
  what the server's `input_type=image` filter consumes.
- `media/capabilities.py::MediaCapabilities._apply_provider_adjustments` **has** a `provider`
  argument and uses it only to tune image size limits and streaming flags. It never touches
  `vision_support`.
- `media/capabilities.py::_apply_model_adjustments` sets `vision_support = True` for any model name
  containing `vision`, `vl`, or `visual` — **unconditionally, with no provider input at all**.
- `media/handlers/local_handler.py::create_multimodal_message` names its variable
  `provider_vision_support` but reads `self.capabilities.vision_support` — a model fact wearing a
  provider label — so the `and` that follows compares a model fact with itself.

### The seam that a per-instance fix does not reach

An earlier draft of this work claimed that downgrading the per-instance `self.model_capabilities`
in `_load_model` "alone stops the server lying". **It does not.** Verified:

- `MLXProvider.list_available_models` is a **`@classmethod`**. `/v1/models?input_type=image`
  resolves through it into `filter_models_by_capabilities`, which reads the name-keyed registry.
  There is no provider instance anywhere on that path.
- `analyze_media`'s route gate reads `get_media_capabilities(model, provider)` and is documented
  as a local registry read that must never construct a provider client.
- `media/auto_handler.py::_check_vision_support(model)` gates glyph PDF→image compression and is
  also model-name based — so a blind transport can have a PDF compressed into an image and then
  dropped.

## Problem

Two separable defects:

1. **The lie.** Discovery surfaces advertise a modality the transport cannot carry. Under ADR
   `0001` that is a silent degradation on a correctness-critical path.
2. **The shape.** Capability is keyed on the wrong tuple, so there is nowhere to *put* the truth
   even once it is known. This is why the same lie exists for `ollama` and `lmstudio` (whether a
   GGUF has an `mmproj` alongside it is a runtime fact) and for the `huggingface` GGUF lane.

## What we want to do

Introduce a provider-owned capability predicate — the capability analogue of
`get_model_residency(...)` — with three states (`supported` / `unsupported` / `unknown`), and route
the registry-facing capability functions through it when a provider context is available.

The registry keeps stating the **ceiling** ("these weights contain a vision tower"). The provider
states the **floor** ("this transport can carry pixels"). The effective capability is the minimum,
computed at query time.

## Why

Because the fix must reach the *static* seams, not just the instance. And because a per-instance
downgrade has a second, real benefit that should be kept even though it is not the headline one:
`self.model_capabilities` is fed into `LocalMediaHandler`, and downgrading it makes
`create_multimodal_message` take the text-embedded path, which runs the existing vision fallback
and returns **an actual caption** instead of a dropped image. That is a visible user improvement
independent of the discovery fix.

## Requirements

- A provider-owned predicate answering "can this provider+runtime carry modality M for model X",
  with an explicit `unknown` that is never silently coerced to `true`.
- `/v1/models?input_type=image` must not list models whose transport cannot accept an image, for
  providers that can answer. Providers that cannot answer report `unknown` and are handled by a
  stated policy, not by an implicit default.
- `media/capabilities.py`'s name-substring vision heuristic is deleted or demoted behind an
  explicit registry check.
- The `analyze_media` route gate and the glyph compression gate consult the same predicate.
- No provider-client construction on discovery paths that are documented as registry-only reads.
- Applies to `ollama`, `lmstudio`, `huggingface` (GGUF) and `mlx` — not MLX alone.

## Suggested implementation

Options, cheapest first; the implementation should pick and record one:

1. **Static provider hook.** A classmethod-friendly
   `supports_media_input(cls, model, modality) -> True | False | None` on the provider base,
   defaulting to `None` (unknown). `filter_models_by_capabilities` consults it when a provider is
   in scope. Cheapest; reaches the classmethod seam.
2. **Capability resolver object.** A small `(model, provider, runtime) → routes` resolver that the
   nine registry functions delegate to, with the current behaviour as the `unknown` fallback.
   Cleaner; larger blast radius.
3. **Per-instance only** (the earlier draft). Rejected as insufficient on its own — keep the
   instance downgrade for the `LocalMediaHandler` benefit, but it does not close the discovery lie.

Note the MLX predicate is cheap and offline: a `config.json` read for a non-empty `vision_config`,
plus whether the optional vision extra is importable. It needs no weights and no model load.

## Scope

- The capability query path: `providers/model_capabilities.py`, `architectures/detection.py`,
  `media/capabilities.py`, and the two gates named above.
- Provider-side predicates for `mlx`, `ollama`, `lmstudio`, `huggingface`.

## Non-goals

- Changing the shape or contents of `assets/model_capabilities.json`. It is correct: it describes
  the weights.
- Implementing vision transport for any provider — that is `0840` for MLX.
- Reworking output-side capability (image/audio generation), which already flows through the
  capability-plugin residency contracts ADR `0003` describes.

## Dependencies and related tasks

- `0840` — MLX vision transport. Independent: this item makes the declaration honest whether or
  not the transport gains sight, and shrinks to a narrow residue once it does.
- `0842` — response-level delivery honesty. Complementary: this item fixes what we *advertise*,
  `0842` fixes what we *report after the fact*.
- `docs/adr/0008-provider-owned-model-residency-truth.md` — the pattern to follow and amend.

## Expected outcomes

- Discovery surfaces stop advertising modalities the transport cannot carry, for every local
  provider.
- A blind-transport image request returns a caption with provenance rather than a silently dropped
  image.
- A recorded answer to "where does capability truth live", consistent with residency truth.

## Validation

- A server test asserting `/v1/models?input_type=image` omits a vision-capable model served by a
  transport that reports `unsupported`, and includes it when the transport reports `supported`.
  Extend `tests/server/test_model_capability_filtering.py`.
- A test that `unknown` is never coerced to `true` anywhere on the path.
- A test that the name-substring heuristic no longer promotes an arbitrary `*-vl-*` model name to
  vision-capable on a blind transport.
- A test that the `analyze_media` route gate refuses a blind transport, and that it does so from a
  registry read without constructing a provider client.
- Regression: the four existing local providers still list their genuinely-capable models.

## Progress checklist

- [ ] Choose option 1 or 2 and record the decision
- [ ] Amend ADR `0008` (or add a citing sibling) to cover capability
- [ ] Provider predicate on the base class + `mlx` implementation
- [ ] Route `filter_models_by_capabilities` and the two gates through it
- [ ] `ollama` / `lmstudio` / `huggingface` predicates (mmproj presence, etc.)
- [ ] Delete or demote the name-substring heuristic
- [ ] Keep the per-instance downgrade for the `LocalMediaHandler` caption benefit
- [ ] Tests above

## Guidance for the implementing agent

**Read ADR `0008` before writing any code.** This item is that decision applied to a second fact;
its vocabulary (`*_verified`, explicit unknown, provider-owned) should be reused rather than
reinvented, and the ADR amendment is part of the work, not a follow-up.

**Do not fix this by editing `assets/model_capabilities.json`.** Marking `qwen3.8-27b` as
non-vision would be false — the weights are there, and the same entry is correct for the
`huggingface` provider. The registry is not wrong; the query is under-specified.

**The classmethod seam is the test of whether the fix is real.** If `/v1/models?input_type=image`
still lists a blind model after the change, the fix landed on the instance path only.
