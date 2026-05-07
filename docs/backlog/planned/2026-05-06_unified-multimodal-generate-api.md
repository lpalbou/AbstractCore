# Planned: Unified Multimodal Generate API

## Metadata
- Created: 2026-05-06
- Status: Planned
- Completed: N/A

## Context
AbstractCore's main public generation surface is `generate(...)`, but that surface is still mostly
text-generation shaped. The framework now has more than text available:

- text generation through provider implementations;
- multimodal input through `media=...`, including image/audio/video/document policies and
  fallbacks;
- optional capability plugins for deterministic output modalities:
  - `llm.voice` / `llm.audio` through `abstractvoice` for TTS/STT and some voice operations;
  - `llm.vision` through `abstractvision` for text-to-image, image-to-image, and related generative
    vision operations;
  - `llm.music` through `abstractmusic`, currently experimental/not yet reliable enough to treat as
    a stable dependency.

The user-facing goal is to let callers ask AbstractCore to generate text, image, voice/speech, and
music from one coherent API instead of making them stitch together `generate(...)`, `llm.vision.*`,
`llm.voice.*`, and `llm.music.*` manually.

## Current Code Reality
- `AbstractCoreInterface.generate(...)` in `abstractcore/core/interface.py` accepts text/chat inputs,
  tools, media inputs, streaming, and thinking controls. It returns `GenerateResponse` or an
  iterator of `GenerateResponse`.
- `GenerateResponse` in `abstractcore/core/types.py` is text-response shaped: `content`,
  `raw_response`, `model`, `finish_reason`, `usage`, `tool_calls`, `metadata`, and `gen_time`.
- `AbstractCoreInterface` already exposes capability facades: `voice`, `audio`, `vision`, and
  `music`.
- `AbstractCoreInterface.generate_with_outputs(...)` is a useful v0 convenience wrapper. It runs
  `generate(...)` first, then optionally calls:
  - `self.voice.tts(...)` for `outputs={"tts": {...}}`;
  - `self.vision.t2i(...)` for `outputs={"t2i": {...}}`;
  - `self.music.t2m(...)` for `outputs={"t2m": {...}}`.
- `generate_with_outputs(...)` does not support `stream=True`, has no async equivalent, always
  starts with text generation, uses operation keys (`tts`, `t2i`, `t2m`) instead of user-facing
  modalities (`voice`, `image`, `music`), and returns a minimal
  `GenerateWithOutputsResult(response, outputs)` object.
- `abstractcore/capabilities/registry.py` supports plugin entry points for voice, audio, vision, and
  music. Missing plugins raise `CapabilityUnavailableError` with install hints.
- Server routes already expose separate modality endpoints:
  - `/v1/images/generations` and `/v1/images/edits` in `abstractcore/server/vision_endpoints.py`;
  - `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/voice/clone`, and `/v1/audio/music` in
    `abstractcore/server/audio_endpoints.py`.
- `docs/capabilities.md`, `docs/vision-capabilities.md`, and `docs/server.md` document separate
  capability and endpoint surfaces. They do not document a single multimodal `generate(...)`
  contract.
- Tests already cover the capability registry and the current `generate_with_outputs(...)` v0 path
  with fake plugins.

## Problem
The public API shape now undersells what AbstractCore can orchestrate. Users who reasonably expect
`generate(...)` to cover more than text have to know multiple lower-level surfaces and manually
sequence them.

Current rough edges:

- `generate(...)` has no first-class output modality selector.
- The existing `generate_with_outputs(...)` path is discoverable only if users already know it exists.
- Operation names (`tts`, `t2i`, `t2m`) leak backend/task terminology instead of presenting a
  user-facing modality contract.
- The v0 wrapper always generates text first, even when the caller only wants an image, voice clip,
  or music output from the original prompt.
- There is no structured artifact result model with content type, bytes/artifact reference, format,
  backend, model, and generation metadata.
- There is no async multimodal generation surface.
- Missing capability behavior, partial output behavior, and mixed success/failure behavior need an
  explicit policy.
- Music support must be represented as experimental without blocking the stable text/image/voice
  design.

## What We Want To Do
Design and implement a unified multimodal generation API that lets callers request one or more output
modalities from the normal generation surface while preserving backward compatibility for existing
text-only calls.

Example target shape, subject to design validation:

```python
result = llm.generate(
    "Create a bedtime story and a cover image, then narrate it.",
    output=["text", "image", "voice"],
    output_options={
        "image": {"width": 1024, "height": 1024},
        "voice": {"voice": "alloy", "format": "wav"},
    },
)

print(result.text.content)
image = result.outputs["image"][0]
voice = result.outputs["voice"][0]
```

The exact names can change during implementation, but the API should clearly support:

- text only;
- image only;
- voice/speech only;
- music only when a music backend is installed;
- text plus one or more generated artifacts;
- explicit modality options without requiring provider-specific details at the call site.

## Why
- AbstractCore already has optional output capabilities; the main API should make them coherent.
- Apps can ask for a result instead of manually chaining text generation, image generation, and TTS.
- A controlled modality vocabulary prevents ad hoc per-app wrappers from spreading.
- The implementation can preserve dependency-light defaults while giving installed plugins a better
  orchestration surface.
- Music can be designed into the contract now without pretending `abstractmusic` is production-ready.

## Requirements
- Preserve backward compatibility:
  - `llm.generate(prompt)` must keep returning the existing `GenerateResponse` for text-only calls.
  - Existing provider implementations and sessions must not break when no multimodal output is
    requested.
- Define a small controlled output vocabulary. Candidate canonical names:
  - `text`
  - `image`
  - `voice` or `speech` (decide and document the alias policy)
  - `music`
- Support either/or and combined outputs:
  - no required text generation when the caller only asks for `image`, `voice`, or `music` from the
    input prompt;
  - optional text-first chaining when a generated text response should become the prompt/source for
    image, voice, or music.
- Define source semantics explicitly, for example:
  - source is the original prompt;
  - source is the generated text response;
  - source is an explicit per-modality prompt/text.
- Add a structured multimodal result type with:
  - optional text `GenerateResponse`;
  - generated artifacts grouped by modality;
  - artifact bytes or `ArtifactRef`;
  - content type / format;
  - backend id, model id when known, and generation parameters;
  - bounded warnings/errors for partial failures.
- Keep missing capability behavior explicit. Do not silently drop requested modalities.
- Preserve "no silent fallback" behavior when a capability backend is unavailable or when routing
  changes provider/model/backend.
- Add async support, likely through `agenerate(...)` or a paired helper.
- Decide how streaming works:
  - v1 may reject `stream=True` for non-text/mixed outputs;
  - if text streams and artifacts are produced after text completion, document that ordering.
- Keep artifacts dependency-light:
  - return raw bytes in library mode when no artifact store is provided;
  - support `ArtifactStoreLike` for durable artifact references.
- Treat music as experimental:
  - expose the planned contract;
  - return actionable `CapabilityUnavailableError`/install hints when no backend exists;
  - avoid examples that imply `abstractmusic` is stable until it is.
- Add docs that clearly distinguish:
  - multimodal input to text generation (`media=...`);
  - deterministic output capabilities (`llm.voice`, `llm.vision`, `llm.music`);
  - unified multimodal output generation through `generate(...)`.

## Suggested Implementation
1. Write or update an ADR for the output modality vocabulary and result schema before the names
   spread across docs and tests.
2. Add core types in a small module, for example `abstractcore/core/multimodal_generation.py`:
   - `OutputModality`
   - `OutputRequest`
   - `GeneratedArtifact`
   - `MultimodalGenerateResponse`
   - possibly `PartialOutputError` or a bounded warning record.
3. Decide whether the public entry point is:
   - `generate(..., output=..., output_options=...)` with text-only backward compatibility; or
   - `generate_multimodal(...)` plus a documented migration path from `generate_with_outputs(...)`.
   The user-facing goal favors `generate(...)`, but implementation should avoid surprising existing
   provider subclasses.
4. Refactor `generate_with_outputs(...)` to become a compatibility wrapper over the new
   orchestrator, or deprecate it after the new API is documented.
5. Implement modality orchestration centrally instead of adding per-provider branches:
   - text: normal `generate(...)`;
   - image: `self.vision.t2i(...)` for v1;
   - voice/speech: `self.voice.tts(...)` for v1;
   - music: `self.music.t2m(...)` when available.
6. Add source routing:
   - original prompt;
   - generated text;
   - explicit per-output prompt/text.
7. Add async orchestration. If capability plugins are sync-only, use a clear sync bridge policy and
   document it.
8. Update `BasicSession`/session behavior only after the base interface is stable.
9. Update docs and examples:
   - `docs/capabilities.md`;
   - `docs/api.md`;
   - `docs/getting-started.md`;
   - `examples/` with one text+image+voice script and one image-only script.
10. Add tests using fake capability plugins before touching real optional dependencies.

## Scope
- Library-level multimodal generation API and result types.
- Integration with existing capability plugin facades.
- Backward-compatible handling for existing text-only `generate(...)`.
- Tests with fake plugins for text, image, voice, music, missing capabilities, and partial failures.
- Documentation and examples for the stable text/image/voice subset, with music clearly marked
  experimental.

## Non-Goals
- Do not make every provider natively support image, voice, or music output.
- Do not make `abstractmusic` a required dependency or imply it is stable.
- Do not implement the deterministic output cache in this item; that belongs to
  `docs/backlog/planned/2026-05-07_multimodal-generation-and-deterministic-inference-cache.md`.
- Do not replace the lower-level `llm.voice`, `llm.audio`, `llm.vision`, or `llm.music` facades.
- Do not silently synthesize missing modalities or silently skip requested outputs.
- Do not redesign OpenAI-compatible server endpoints unless the library API reveals a clear server
  contract that should be added later.
- Do not require an artifact store for simple library-mode use.

## Dependencies And Related Tasks
- `abstractcore/core/interface.py`
- `abstractcore/core/types.py`
- `abstractcore/capabilities/types.py`
- `abstractcore/capabilities/registry.py`
- `abstractcore/core/session.py`
- `abstractcore/server/vision_endpoints.py`
- `abstractcore/server/audio_endpoints.py`
- `docs/capabilities.md`
- `docs/vision-capabilities.md`
- `docs/server.md`
- `docs/backlog/planned/788-response.md`
- `docs/backlog/planned/2026-05-07_multimodal-generation-and-deterministic-inference-cache.md`
- `docs/backlog/planned/2026-05-06_robust-fallback-generate.md`
- `docs/backlog/planned/2026-05-06_consensus-generate.md`

## Expected Outcomes
- Users can request multimodal outputs through one coherent generation API.
- Text-only calls keep their current behavior and return type.
- Generated image/voice/music artifacts have a structured, inspectable result shape.
- Missing or experimental capability behavior is visible and actionable.
- The existing `generate_with_outputs(...)` v0 path is either folded into the new implementation or
  clearly documented as legacy.
- Docs make the distinction between multimodal inputs and multimodal outputs obvious.

## Validation
- A-level unit tests:
  - output vocabulary parsing and alias handling;
  - structured result serialization;
  - source selection from original prompt, generated text, and explicit per-modality input;
  - missing capability errors;
  - partial failure policy.
- B-level integration tests with fake plugins:
  - text only returns existing `GenerateResponse`;
  - image only does not call text generation unless configured to do so;
  - text plus image calls text then image with the expected source;
  - text plus voice calls TTS with the expected source;
  - music request reports experimental/missing backend cleanly when no plugin is installed;
  - async path follows the same behavior.
- C-level smoke tests when optional packages are available:
  - `abstractvoice` TTS output;
  - `abstractvision` text-to-image output;
  - `abstractmusic` text-to-music only behind an explicit opt-in env flag.
- Documentation checks for import paths and examples.

## Progress Checklist
- [ ] Reconfirm current provider/session call paths before implementation.
- [ ] Decide and record the modality vocabulary and result schema.
- [ ] Decide whether the public entry point is `generate(..., output=...)`,
      `generate_multimodal(...)`, or both.
- [ ] Implement core multimodal request/result types.
- [ ] Implement centralized orchestration over capability facades.
- [ ] Preserve and/or wrap `generate_with_outputs(...)`.
- [ ] Add async behavior.
- [ ] Add fake-plugin tests.
- [ ] Add docs and examples.
- [ ] Revisit whether a server-level unified endpoint should become a separate backlog item.

## Guidance For The Implementing Agent
Start from the current code, not from this backlog item alone. In particular, inspect
`generate_with_outputs(...)`, capability plugin protocols, and session behavior before designing the
public API. Keep the first stable version small, explicit, and dependency-light. Treat modality
names, result shape, missing capability behavior, and partial failure policy as API design decisions,
not implementation details.
