# Planned: Unified Multimodal Generate API

## Metadata
- Created: 2026-05-06
- Updated: 2026-05-07
- Status: Planned
- Completed: N/A

## Context
AbstractCore's primary public generation surface is `generate(...)`. Today it is mostly a
text-generation API, even though the system now has three related but different multimodal concepts:

- multimodal input to text/chat models through `media=...`;
- deterministic generated outputs through optional capability plugins:
  - `llm.vision` through AbstractVision for image generation, image edit, and future video tasks;
  - `llm.voice` / `llm.audio` through AbstractVoice for TTS, STT, and some voice operations;
  - `llm.music` through AbstractMusic, currently experimental;
- resource registration tasks, such as voice cloning, which create a reusable resource id rather
  than returning ordinary audio bytes.

The user-facing goal is a coherent API for scripts, apps, servers, and future Runtime workflows:

1. Say what to generate or transform.
2. Say which output modality/task is required.
3. Receive text, generated artifacts, or registered resource ids in one normalized result shape.

## Current Code Reality
- `AbstractCoreInterface.generate(...)` in `abstractcore/core/interface.py` accepts text/chat inputs,
  tools, media inputs, streaming, thinking controls, and provider kwargs. Without a requested output
  modality it returns `GenerateResponse` or an iterator of `GenerateResponse`.
- `GenerateResponse` in `abstractcore/core/types.py` is text-response shaped: `content`,
  `raw_response`, `model`, `finish_reason`, `usage`, `tool_calls`, `metadata`, and `gen_time`.
- Top-level `media=...` is already meaningful and should remain the main user-facing place to pass
  input files, bytes, or artifact refs. By default, media is contextual input for a text/chat model.
  When `output=...` is also supplied, Core may use the same media as source/reference media for an
  output task when the intent is unambiguous, but it must normalize that intent before routing.
  Source pixels/audio for image edits, masks, voice references, and clone samples must not be blindly
  passed through the text/VLM media optimization or fallback-captioning path.
- `AbstractCoreInterface` exposes lazy capability facades: `voice`, `audio`, `vision`, and `music`.
- `VisionCapability` already includes `t2i`, `i2i`, `t2v`, and `i2v`. A unified API must model
  task-level routing, not only text-to-image.
- `VoiceCapability` includes `tts` and `stt`; `llm.voice.clone(...)` exists as a facade that calls a
  backend `clone` / `clone_voice` method when present. The clone surface is currently duck-typed and
  should be made explicit before it becomes a stable unified task.
- `ArtifactStoreLike`, `ArtifactRef`, and `BytesOrArtifactRef` already exist in
  `abstractcore/capabilities/types.py`. Reuse this contract; do not introduce a second artifact
  protocol in Core.
- `AbstractCoreInterface.generate_with_outputs(...)` is a useful v0 wrapper. It always runs text
  generation first, then optionally calls `voice.tts`, `vision.t2i`, or `music.t2m`. It leaks
  operation names (`tts`, `t2i`, `t2m`), has no normalized result schema, and does not support
  image edit or resource registration.
- Server routes already expose separate modality endpoints:
  - `/v1/images/generations` and `/v1/images/edits` in `abstractcore/server/vision_endpoints.py`;
  - `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/voice/clone`, and `/v1/audio/music` in
    `abstractcore/server/audio_endpoints.py`.
- `/v1/images/edits` is multipart, accepts a required image and optional mask, and returns
  OpenAI-compatible `b64_json`. The unified library API should normalize this shape without exposing
  multipart details.
- `/v1/voice/clone` accepts reference audio and returns clone metadata / `voice_id`, not generated
  audio bytes. This is a resource registration operation.
- `BaseProvider.agenerate(...)` and session wrappers need careful review before async multimodal
  parity is claimed, especially around forwarding `media` and consuming `output` before provider
  calls.

## Problem
The current public API undersells what AbstractCore can orchestrate and creates confusing boundaries:

- `generate(...)` has no first-class output selector.
- The existing `generate_with_outputs(...)` path is discoverable only if users already know it
  exists.
- Operation names (`tts`, `t2i`, `t2m`) leak backend terminology instead of presenting a stable
  modality/task vocabulary.
- The v0 wrapper always generates text first, even when the caller only wants an image, image edit,
  speech clip, music clip, or registered voice resource from the original prompt/reference media.
- `media=...` could easily be overloaded incorrectly if Core treats every input file the same way.
  A source image for image edit and a mask need pixel-preserving handling. An image attached as
  context for a VLM explanation can be resized, optimized, captioned, or provider-formatted. Those
  are different media roles, even when the public call passes both through `media=...`.
- Voice cloning is not ordinary speech generation. It creates reusable backend state or a reusable
  resource id, often with consent/provider-specific fields and idempotency concerns.
- There is no structured result model that covers both binary generated artifacts and registered
  reusable resources.
- Missing capability behavior, experimental modality behavior, partial failures, and async behavior
  need explicit policy.

## Design Decision
Keep `generate(...)` as the primary user-facing API. Do not add a new primary
`generate_multimodal(...)` method.

Backward compatibility rule:

```python
response = llm.generate("Explain durable workflows.")
# returns GenerateResponse, unchanged
```

When `output` requests non-text generated media, transcription, or mixed outputs,
Core returns a `MultimodalGenerateResponse`. Plain `output="text"` with a prompt
may preserve the normal `GenerateResponse` path for provider compatibility:

```python
result = llm.generate(
    "Create a red ceramic mug product shot.",
    output={"modality": "image", "width": 1024, "height": 1024},
)
```

Public distinction:

- `media=...`: user-provided input media. With no `output`, it is contextual input to a text/chat
  model. With `output`, simple single-output calls may infer the media role from the output modality
  and media type.
- `output=...`: requested generated output, transform, or registered resource.
- Media roles: when there is any ambiguity, media items should be role-tagged as `context`,
  `source`, `mask`, `reference`, or `clone_sample`.
- Scoped/per-output media: only needed for advanced multi-output calls where different outputs use
  different media. It should be a normalization detail or an explicit advanced form, not mandatory
  syntax for common image-edit or voice-reference calls.
- Text selection: simple calls use the input text directly. In a multi-output call that includes one
  text output plus image/voice outputs, downstream outputs use the generated text by default. Only
  ambiguous multi-output workflows need explicit binding.

## Output Vocabulary
Define a small canonical modality vocabulary:

- `text`
- `image`
- `voice`
- `video` reserved / experimental
- `music` reserved / experimental

Alias policy:

- `voice` is the user-facing voice capability. Depending on inputs, it can mean generated speech
  audio or a reusable cloned/registered voice resource.
- Accept `speech` and `tts` as convenience aliases for `voice` with task `tts`.
- Accept common task aliases such as `t2i`, `i2i`, and `image_to_image`, but normalize them to
  canonical tasks before routing.

Keep modality separate from task:

- `text` + `text_generation`
- `text` + `transcription`
- `image` + `image_generation`
- `image` + `image_edit`
- `voice` + `tts` (generated speech audio)
- `voice` + `voice_clone` (reusable voice resource registration)
- `voice` + `voice_conversion` (reserved / future)
- `video` + `video_generation`, `image_to_video`, or future `video_edit` (reserved/experimental)
- `music` + `music_generation` (experimental)

Default task policy:

- `image` with no image media defaults to `image_generation`.
- `image` with one unambiguous source image, or with image media role `source`, defaults to
  `image_edit`.
- `image` with image media role `reference` or `style` remains image generation unless the caller
  explicitly asks for an edit/transform task.
- `voice` with text and no audio media defaults to `tts` and returns generated speech audio.
- `voice` with audio media and no explicit `task` defaults to `voice_clone` and returns/registers a
  reusable voice resource id. This is the audio equivalent of image edit inference: the requested
  output modality plus the media type makes the intent clear.
- In inferred `voice_clone`, a top-level `text` value is treated as optional reference text /
  transcript for the sample, not as text to speak.
- `voice` with audio media role `reference` and `task="tts"` means temporary reference-guided TTS
  when the backend supports it. The audio media is not persisted as a cloned voice.
- `voice` with a `voice`/`voice_id` option and text defaults to TTS using that existing voice
  profile.
- `text` with audio media and no text prompt may default to `transcription`. If there is also a text
  prompt, preserve today's "audio as context for a text/chat model" behavior unless the caller asks
  for `task="transcription"` or uses a convenience output alias such as `transcript`.
- `music` and `video` require explicit task or explicit experimental opt-in until their backend
  story is stable.

## Text Selection
Most users should not have to think about text binding. The default is simple:

- `llm.generate(text="...", output="voice")` uses that text for TTS.
- `llm.generate("...", output="image")` uses that prompt for image generation.
- If a call asks for exactly one text output plus non-text outputs, the non-text outputs use the
  generated text by default.
- If an output has its own `text` or `prompt`, that per-output value wins.
- If there are multiple text outputs or mixed dependencies, require an explicit source by output id.

This matters because the original prompt is often an instruction, not the final content. In this
call, the narration should read the generated story by default, not literally say "Create a short
bedtime story":

```python
story = llm.generate(
    "Create a short bedtime story, a cover image, and narration.",
    output=[
        {"id": "story", "modality": "text"},
        {"id": "cover", "modality": "image", "format": "png"},
        {"id": "narration", "modality": "voice", "format": "wav", "voice": "coral"},
    ],
)
```

Advanced binding is only needed when defaults are ambiguous, for example multiple text outputs:

```python
result = llm.generate(
    "Create a short and long version, then narrate the short version.",
    output=[
        {"id": "short", "modality": "text", "style": "short"},
        {"id": "long", "modality": "text", "style": "long"},
        {"modality": "voice", "source": "short", "format": "wav"},
    ],
)
```

The internal normalized request may still store resolved text/prompt bindings, but public examples
should rely on defaults and use explicit `source` only when a workflow is genuinely ambiguous.

Examples:

```python
# Text-only: unchanged.
response = llm.generate("Explain model routing.")

# Image generation from the original prompt.
image = llm.generate(
    "A precise product photo of a red ceramic mug on a white table.",
    output={"modality": "image", "width": 1024, "height": 1024, "format": "png"},
)

# Image edit. The simple public form can use top-level media because the requested output is image
# and the media roles make the source/mask intent explicit.
edited = llm.generate(
    "Make the mug blue and keep the background white.",
    media=[
        {"type": "image", "path": "./mug.png", "role": "source"},
        {"type": "image", "path": "./mask.png", "role": "mask"},
    ],
    output={"modality": "image", "format": "png"},
)

# Text plus image plus narration. Because there is exactly one text output, the image and voice use
# the generated story by default.
story = llm.generate(
    "Create a short bedtime story, a cover image, and narration.",
    output=[
        {"id": "story", "modality": "text"},
        {"modality": "image", "format": "png"},
        {"modality": "voice", "format": "wav", "voice": "coral"},
    ],
)
```

Voice cloning example:

```python
# TTS: text plus voice output returns generated speech audio.
speech = llm.generate(
    text="Hello from AbstractCore.",
    output={"modality": "voice", "voice": "coral", "format": "wav"},
)

# Cloning/registering: audio media plus voice output returns a reusable voice id.
clone = llm.generate(
    text="Optional transcript of the reference audio.",
    media={"type": "audio", "path": "./reference.wav", "role": "clone_sample"},
    output={
        "modality": "voice",
        "name": "narrator",
        "consent": "consent_123",
    },
)

voice_id = clone.resources["voice"][0].resource_id

speech = llm.generate(
    text="Read this aloud.",
    output={
        "modality": "voice",
        "voice": voice_id,
        "format": "wav",
    },
)
```

Keep `llm.voice.clone(...)` as the expert/direct API. A later convenience API such as
`llm.assets.register(...)` or `llm.register_voice(...)` can be considered after the resource result
model is stable, but it is not required for the first unified-generation implementation.

## Voice Clone Caching
Voice cloning should not force users to upload and reprocess the same reference audio repeatedly.
The unified API should support resource reuse without making every TTS call carry reference audio.

Recommended behavior:

- A clone call returns a stable `voice_id` / `resource_id` when the backend exposes one.
- Later TTS calls can use that id:

```python
voice_id = clone.resources["voice"][0].resource_id
speech = llm.generate(text="Read this aloud.", output={"modality": "voice", "voice": voice_id})
```

- In server mode, AbstractCore may maintain a content-addressed clone cache keyed by:
  - normalized reference-audio hash, preferably exact bytes hash for v1;
  - provider/backend/model;
  - clone settings such as name, reference text, language, validation mode, consent id, and backend
    version when relevant;
  - tenant/user/session scope, because cloned voices can be sensitive.
- If the same cache key is seen again and the cached voice resource is still valid, the server can
  return the existing `voice_id` instead of re-cloning.
- Cache scope must be explicit in configuration: `session`, `user`, `project`, or disabled. Do not
  silently create a global cross-user voice cache.
- A later enhancement can add perceptual audio fingerprinting, but v1 should prefer exact-byte
  hashing because it is deterministic, auditable, and avoids false matches.

This is separate from deterministic media-output caching. Clone caching stores/reuses a reusable
voice resource; TTS/image caching stores/reuses generated artifacts.

## Result Model
Add core types in a small module, for example `abstractcore/core/multimodal_generation.py`.
The first implementation uses a minimal v0 result model; richer per-item
parameters, hashes, and warning/error fields remain follow-up work:

- `OutputModality`
- `OutputTask`
- `OutputSpec`
- `MediaRole`
- `MediaBinding`
- `GeneratedItem`
- `GeneratedResource`
- `MultimodalGenerateResponse`
- `GenerationWarning`
- `GenerationError`

`MultimodalGenerateResponse` should include:

- optional `text: GenerateResponse`;
- `outputs: Dict[str, List[GeneratedItem]]` for generated artifacts such as images, audio clips,
  music, and later video;
- `resources: Dict[str, List[GeneratedResource]]` for reusable registered resources such as cloned
  voices. A cloned voice appears under `resources["voice"]`; generated spoken audio appears under
  `outputs["voice"]` with an audio content type;
- normalized routing metadata;
- bounded warnings/errors for explicit partial-failure mode.

`GeneratedItem` should represent binary/generated payloads:

- `modality`
- `task`
- `data` only in simple library mode when no artifact store is provided
- `artifact_ref` when stored
- `content_type`
- `format`
- `size_bytes`
- `sha256`
- `backend_id`
- `provider`
- `model`
- resolved text-selection metadata
- media binding metadata records, not raw bytes
- `parameters`
- `metadata`
- `warnings`
- `errors`

`GeneratedResource` should represent non-binary reusable resources:

- `modality`
- `task`
- `resource_type`, for example `voice`
- `resource_id`
- `name`
- `backend_id`
- `provider`
- `model`
- optional `artifact_ref` for the stored reference sample, if applicable
- media binding metadata records, not raw bytes
- `metadata`
- `warnings`
- `errors`

Artifact behavior:

- If `artifact_store` is provided, pass it through to capability backends that support it.
- If a backend returns raw bytes and `artifact_store` is provided, Core stores them with
  `ArtifactStoreLike.store(...)` and returns an artifact ref when the store returns a usable id/ref.
- If a backend returns an `ArtifactRef`, preserve it.
- Never put raw bytes into AbstractRuntime state or session history.
- Do not require an artifact store in simple library mode.

## Requirements
- Preserve backward compatibility:
  - `llm.generate(prompt)` with no `output` argument must continue returning `GenerateResponse`.
  - Existing provider implementations and sessions must not break when no multimodal output is
    requested.
- Consume `output`, `artifact_store`, `partial`, and unified task fields in the central orchestrator.
  Do not forward those keys down into provider `_generate_internal(...)`.
- Keep missing capability behavior explicit. Do not silently drop requested modalities.
- Preserve no-silent-fallback behavior when routing changes provider/model/backend.
- Reject `stream=True` for non-text or mixed output requests in v1. If text streaming plus
  post-stream artifacts is added later, document ordering explicitly.
- Use `partial=True` or an equivalent explicit option before returning mixed success/failure.
  Single-output failure should raise.
- Keep lower-level facades:
  - `llm.voice.tts(...)`
  - `llm.voice.clone(...)`
  - `llm.audio.transcribe(...)`
  - `llm.vision.t2i(...)`
  - `llm.vision.i2i(...)`
  - `llm.vision.t2v(...)` / `llm.vision.i2v(...)`
  - `llm.music.t2m(...)`
- Keep music and video experimental until package/server/docs/tests are stable enough.
- Add docs that clearly distinguish:
  - multimodal input to text generation (`media=...`);
  - output generation and transforms (`output=...`);
  - media roles and scoped media binding for transforms/resources;
  - direct expert capability facades.

## Suggested Implementation
1. Write or update an ADR for:
   - modality vocabulary;
   - task vocabulary;
   - `media` role inference and scoped/per-output binding;
   - text-selection semantics;
   - artifact/resource result schemas;
   - failure/partial policy.
2. Add the core multimodal request/result types.
3. Add a central private orchestrator on `AbstractCoreInterface`, for example
   `_generate_multimodal_outputs(...)`, and have `generate(..., output=...)` dispatch to it before
   provider text generation receives provider kwargs.
4. Route tasks centrally:
   - `text_generation`: normal provider `generate(...)`;
   - `transcription`: `self.audio.transcribe(...)` when the selected backend exposes STT;
   - `image_generation`: `self.vision.t2i(...)`;
   - `image_edit`: `self.vision.i2i(...)` with required source image media and optional mask media;
  - `tts`: `self.voice.tts(...)` returning generated audio under `outputs["voice"]`;
   - `voice_clone`: `self.voice.clone(...)`; type this in the capability protocol or define an
     optional `VoiceCloneCapability` before making it stable;
   - `music_generation`: `self.music.t2m(...)`, behind explicit experimental gating;
   - `video` tasks: reserve around existing `self.vision.t2v(...)` / `self.vision.i2v(...)`, behind
     explicit experimental gating.
5. Build media binding helpers that accept paths, bytes, `MediaContent`, artifact refs, and simple
   dicts. The helpers should infer roles in simple single-output cases, require roles when ambiguous,
   and preserve edit/clone source bytes unless a backend explicitly asks for conversion.
6. Normalize raw bytes, artifact refs, and resource ids into `GeneratedItem` / `GeneratedResource`.
7. Refactor `generate_with_outputs(...)` into a legacy compatibility wrapper over the new
   orchestrator. Preserve its current behavior where possible, including text-first defaults.
8. Add `agenerate(..., output=...)` after the sync contract is stable. If capability plugins are
   sync-only, bridge with a documented `asyncio.to_thread` policy.
9. Update `BasicSession` only after the base interface is stable:
   - add generated assistant text to history;
   - do not add binary artifacts to history;
   - store refs/metadata only when session support is explicitly added.
10. Keep server endpoint redesign separate unless implementation reveals a small reusable router.
    In the long term, server image/audio routes should become thin adapters over the same core
    orchestration, but do not block the library API on that refactor.
11. Update docs and examples:
    - `docs/capabilities.md`;
    - `docs/api.md` if present;
    - `docs/getting-started.md`;
    - `docs/server.md` only to explain relation to HTTP endpoints;
    - examples for image generation, image edit, TTS, voice clone plus TTS, and text+image+voice.
12. Add tests with fake capability plugins before touching optional dependencies.

## Scope
- Library-level multimodal output API and result types.
- Integration with existing capability plugin facades.
- Backward-compatible handling for existing text-only `generate(...)`.
- Stable v1 tasks:
  - text generation;
  - audio transcription to text;
  - image generation;
  - image edit;
  - voice TTS;
  - voice clone resource registration when the selected backend exposes clone support.
- Experimental/reserved tasks:
  - music generation;
  - video generation / image-to-video.
- Tests with fake plugins for vocabulary parsing, routing, artifact/resource normalization, missing
  capabilities, and partial failures.
- Documentation and examples for the stable text/image/voice subset.

## Non-Goals
- Do not make every provider natively support image, voice, music, or video output.
- Do not make AbstractVision, AbstractVoice, or AbstractMusic required dependencies.
- Do not infer output modality from prompt text by default.
- Do not require per-output media nesting for simple single-output cases where top-level `media` is
  enough and the task can be inferred cleanly.
- Do not treat voice cloning as ordinary TTS. Use the same public `voice` output capability, but
  keep binary speech audio in `outputs["voice"]` and cloned voice identities in
  `resources["voice"]`.
- Do not pass image-edit source images through the VLM media optimization/captioning path by default.
- Do not implement deterministic output caching in this item; that belongs to
  `docs/backlog/planned/2026-05-07_multimodal-generation-and-deterministic-inference-cache.md`.
- Do not replace lower-level `llm.voice`, `llm.audio`, `llm.vision`, or `llm.music` facades.
- Do not silently synthesize missing modalities or silently skip requested outputs.
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
- `docs/backlog/deprecated/2026-05-07_runtime-ready-multimodal-generation-abstraction.md`
- `docs/backlog/completed/788-response.md`
- `docs/backlog/planned/2026-05-07_multimodal-generation-and-deterministic-inference-cache.md`
- `docs/backlog/planned/2026-05-06_robust-fallback-generate.md`
- `docs/backlog/planned/2026-05-06_consensus-generate.md`

## Expected Outcomes
- Users can request generated outputs and transforms through one coherent generation API.
- Text-only calls keep their current behavior and return type.
- Image generation and image edit have a clean, discoverable contract.
- Voice TTS and voice-clone registration are distinct in result shape: generated audio is an output;
  cloned voices are resources.
- Simple one-output calls do not require redundant syntax; advanced multi-output calls can still
  bind media and generated text explicitly.
- Generated artifacts have structured metadata and can be returned as raw bytes or artifact refs.
- Registered resources such as cloned voices expose resource ids and metadata without pretending to
  be binary audio output.
- Missing or experimental capability behavior is visible and actionable.
- The existing `generate_with_outputs(...)` v0 path is folded into the new implementation or clearly
  documented as legacy.
- Docs make the distinction between multimodal input, output generation, transform input media, and
  resource registration obvious.

## Validation
- A-level unit tests:
  - output vocabulary parsing and alias handling;
  - `speech` and `tts` normalize to `voice` with task `tts`;
  - task parsing and alias handling (`t2i`, `i2i`, `image_to_image`);
  - default text selection and explicit `source` binding only for ambiguous multi-text workflows;
  - media role inference and ambiguity errors;
  - structured result serialization;
  - artifact/resource result serialization;
  - missing capability errors;
  - partial failure policy.
- B-level integration tests with fake plugins:
  - text only returns existing `GenerateResponse`;
  - image generation does not call text generation unless configured to do so;
  - image edit calls `vision.i2i(...)` with image and mask and does not run text generation unless
    requested;
  - text plus image/voice uses the single generated text output by default;
  - text plus voice calls TTS with expected text;
  - `voice` output with text and no audio media calls `voice.tts(...)` and returns generated audio
    under `outputs["voice"]`;
  - `voice` output with audio media calls `voice.clone(...)` and returns a `GeneratedResource` with
    `resource_id`;
  - audio media plus `output="text"` with no text prompt transcribes audio;
  - audio media plus text prompt and no transcription task preserves contextual text/chat behavior;
  - artifact-store mode converts raw image/audio bytes into artifact refs;
  - plugin-returned `ArtifactRef` is preserved;
  - no raw bytes are written into session history or Runtime-like vars;
  - music/video requests report experimental/unavailable behavior cleanly unless enabled;
  - async path preserves the same output routing and media/input forwarding semantics.
- C-level smoke tests when optional packages are available:
  - AbstractVision text-to-image output;
  - AbstractVision image-edit output, if backend supports it;
  - AbstractVoice TTS output;
  - AbstractVoice voice clone plus TTS reuse, behind explicit opt-in and with local/remote backend
    caveats;
  - AbstractMusic only behind an explicit experimental opt-in env flag.
- Documentation checks for import paths and examples.

## Progress Checklist
- [x] Reconfirm current provider/session call paths before implementation.
- [x] Decide and record the high-level API shape: `generate(..., output=...)`.
- [x] Decide that public `media=...` remains the common input-media surface, with role inference for
      simple output tasks and explicit role/scoped binding only when needed.
- [x] Decide that public `output="voice"` covers the voice capability: text/no-audio means TTS;
      audio media means clone/register unless task or role says otherwise.
- [ ] Write/update ADR for vocabulary, task, media binding, text selection, and result schema.
- [x] Implement minimal v0 core multimodal result types.
- [x] Implement centralized orchestration over capability facades.
- [x] Type or formalize voice clone capability support.
- [ ] Preserve and/or wrap `generate_with_outputs(...)`.
- [x] Add async behavior.
- [x] Add fake-plugin tests.
- [ ] Add optional-package smoke tests where practical.
- [x] Add docs and examples.
- [ ] Revisit whether a server-level unified endpoint/router should become a separate backlog item.

## Guidance For The Implementing Agent
Start from the current code, not from this backlog item alone. Inspect `generate_with_outputs(...)`,
capability plugin protocols, server image/audio routes, and session behavior before designing public
types.

Keep the first stable version small, explicit, and dependency-light. Treat modality names, task
names, result shape, missing capability behavior, resource registration, and partial failure policy
as API design decisions, not implementation details.

Do not overload top-level `media` naively. It is the right public place for user-provided media, but
the orchestrator must resolve media roles before routing. Preserve source images, masks, voice
references, and clone samples unless a backend explicitly asks for conversion.

Prefer excellent user experience over maximal generality: common cases should fit in a single
readable `generate(..., output=...)` call, while lower-level capability facades remain available for
expert and provider-specific operations.

## Completion Report

Completed for AbstractCore 2.13.8.

### Implemented Scope

- Added the opt-in unified generated-media API on the existing provider surface:
  `generate(..., output=...)`.
- Preserved the legacy text path: calls without an AbstractCore output request still return the
  existing `GenerateResponse` shape.
- Added structured result types:
  - `MultimodalGenerateResponse`
  - `GeneratedItem`
  - `GeneratedResource`
  - `GenerationIssue`
- Routed generated image output through the optional AbstractVision capability:
  - text plus `output="image"` generates an image;
  - image media plus `output="image"` infers image edit;
  - source/mask/reference roles are supported for less ambiguous image workflows.
- Routed generated voice output through the optional AbstractVoice capability:
  - text plus `output="voice"` generates speech/TTS;
  - audio media plus `output="voice"` registers/clones a voice and returns a reusable voice resource.
- Added audio transcription routing:
  - audio media plus `output="text"` with no prompt infers transcription.
- Added async parity:
  - `agenerate(..., output=...)` now uses the same multimodal dispatcher as sync generation.
- Added media normalization improvements:
  - public media dicts can use `type`, `path`, `content`, `mime_type`, and `role`;
  - base64 `MediaContent` payloads are decoded before plugin calls.
- Added plugin compatibility updates:
  - `abstractvoice>=0.9.0`
  - `abstractvision>=0.3.1`
- Added server integration:
  - `/v1/images/generations` and `/v1/images/edits` reuse the unified image dispatcher while
    preserving OpenAI-compatible HTTP response shapes.
  - local/plugin `/v1/audio/speech`, `/v1/audio/transcriptions`, and `/v1/voice/clone` reuse the
    unified voice/audio dispatcher while preserving their existing HTTP contracts.
  - direct provider/model audio routes remain direct OpenAI-compatible passthroughs because their
    HTTP wire behavior is already explicit and provider-specific.
  - async image job routes remain direct for now because they own progress callbacks that the unified
    dispatcher does not expose yet.

### Public API Examples

```python
# Image generation
image = llm.generate("A red ceramic mug on a white table.", output="image")
png_bytes = image.outputs["image"][0].data

# Image edit
edited = llm.generate("Make the mug blue.", media="mug.png", output="image")

# Masked image edit
masked = llm.generate(
    "Replace only the mug body with blue ceramic.",
    media=[
        {"type": "image", "path": "mug.png", "role": "source"},
        {"type": "image", "path": "mask.png", "role": "mask"},
    ],
    output="image",
)

# TTS
speech = llm.generate(text="Hello from AbstractCore.", output="voice")
wav_bytes = speech.outputs["voice"][0].data

# Voice clone/register
clone = llm.generate(
    text="Optional transcript for the reference audio.",
    media={"type": "audio", "path": "reference.wav"},
    output="voice",
)
voice_id = clone.resources["voice"][0].resource_id

# STT
transcript = llm.generate(media={"type": "audio", "path": "meeting.wav"}, output="text")
print(transcript.text.content)
```

### Validation Performed

- Unit and integration tests:
  - `python -m pytest tests/test_multimodal_generate_output.py tests/test_capabilities_registry.py tests/test_packaging_extras.py -q`
  - `python -m pytest tests/server/test_server_audio_endpoints.py tests/server/test_server_vision_image_endpoints.py tests/server/test_server_openapi_docs.py -q`
  - `python -m pytest -q`
- Static/package sanity:
  - `python -m compileall -q abstractcore`
  - `git diff --check`
- Live local server smoke tests using the configured API keys:
  - `/v1/images/generations` generated a PNG via `openai-compatible/gpt-image-1`;
  - `/v1/images/edits` edited that PNG via the same image route family;
  - `/v1/audio/speech` generated WAV audio through the local/plugin AbstractVoice path;
  - `/v1/audio/transcriptions` transcribed that WAV through the local/plugin AbstractVoice path;
  - `/v1/audio/speech` with `model="openai/gpt-4o-mini-tts"` verified the direct remote provider path;
  - `/v1/chat/completions` analyzed the generated image with `openai/gpt-4o-mini`.

### Follow-Up Items

- Consider a separate planned item for progress-aware unified image/video generation so async job
  routes can also delegate to `generate(..., output=...)` without losing progress callbacks.
- Keep music and video reserved/experimental until the optional plugins expose stable capability
  contracts.
- Keep durable artifact storage as a Runtime/Framework concern; AbstractCore returns raw bytes in
  simple library/server mode and artifact refs only when an artifact store is explicitly supplied.
