# Proposed: Runtime-ready multimodal generation abstraction

## Metadata
- Created: 2026-05-07
- Status: Deprecated
- Completed: N/A
- Deprecated: 2026-05-21

## Context

AbstractCore now has stronger integration points for deterministic output capabilities through
AbstractVision and AbstractVoice. Users should be able to generate text, images, and voice/audio
without manually stitching together the text `generate(...)` API, `llm.vision.*`, `llm.voice.*`,
server image routes, server audio routes, and runtime artifact handling.

2026-05-08 update: AbstractVoice is now moving to a remote-first base/default and exposes voice
profile/model discovery through the AbstractCore plugin boundary. AbstractVision exposes remote
provider model discovery through its AbstractCore plugin boundary. That makes Core the right
abstraction layer for both execution and capability catalogs: clients should be able to ask Core
"what can this configured server generate?" and then call the same Core generation contract.

AbstractRuntime has a different responsibility. It should not know how AbstractVision or
AbstractVoice work internally. It should know how to execute a durable workflow effect, persist
large outputs by reference, and resume workflows safely.

The desired system boundary is:

- AbstractCore owns generation semantics, capability routing, provider/model metadata, and
  multimodal result normalization.
- AbstractRuntime owns durable execution, workflow state, retries/waits where applicable, and
  artifact references.
- User-facing APIs stay small, explicit, and predictable.

2026-05-08 Core update: the Core-side catalog/readiness slice is now implemented. AbstractCore
facades expose `llm.vision.list_provider_models(...)`, `llm.voice.voice_catalog()`,
`llm.voice.list_profiles(...)`, and `llm.voice.list_tts_models()`, and the server exposes
`GET /v1/vision/provider_models`, `GET /v1/audio/voices`, and
`GET /v1/audio/speech/models`. The remaining open part of this proposal is the durable
AbstractRuntime effect/artifact-store integration, which belongs outside the AbstractCore
workspace.

This proposal complements the planned unified multimodal generation API item by focusing on the
Core/Runtime boundary and on the simplest abstraction that can serve scripts, apps, server calls,
and durable workflows.

## Problem

There are several viable implementation paths, but some would create long-term complexity:

- If AbstractRuntime calls AbstractVision or AbstractVoice directly, runtime becomes coupled to
  modality-specific packages and duplicates AbstractCore capability routing.
- If AbstractCore exposes only separate image and voice helper methods, users must build their own
  orchestration for common "text plus image plus narration" flows.
- If modality selection is inferred only from prompt text, workflows become hard to audit and hard
  to reproduce.
- If generated images/audio are returned as raw bytes in runtime state, run state becomes large,
  fragile, and less portable.
- If missing capabilities silently fall back or silently skip requested outputs, user workflows can
  appear successful while producing incomplete or wrong results.

The abstraction should be broad enough for text/image/voice today and music later, but narrow enough
that the first implementation is easy to explain and test.

## Position

### AbstractCore perspective

AbstractCore should provide one coherent generation contract for all stable output modalities. That
contract should preserve the existing text-only behavior:

```python
response = llm.generate("Explain durable workflows.")
```

Text-only calls should keep returning the existing text-shaped `GenerateResponse` unless the caller
explicitly opts into multimodal output.

When the caller asks for generated output, the same public entry point can become multimodal:

```python
result = llm.generate(
    "A clean product render of a modular runtime engine",
    output={"modality": "image", "format": "png", "width": 1024, "height": 1024},
)
```

For voice:

```python
result = llm.generate(
    "Read this sentence aloud.",
    output={"modality": "voice", "voice": "alloy", "format": "wav"},
)
```

For combined generation:

```python
result = llm.generate(
    "Create a short explanation, a cover image, and narration.",
    output=[
        {"modality": "text"},
        {"modality": "image", "source": "generated_text", "format": "png"},
        {"modality": "voice", "source": "generated_text", "format": "wav"},
    ],
)
```

The important choice is that `output.modality` is explicit. An `auto` mode can exist later, but it
should be opt-in and should return clear routing metadata. Production workflows should not depend
on prompt inference when an explicit output modality is known.

### AbstractRuntime perspective

AbstractRuntime should add one durable generation effect after AbstractCore exposes a stable
contract. The runtime effect should mirror Core's request shape and keep results JSON-safe:

```python
Effect(
    type=EffectType.GENERATION_CALL,
    payload={
        "prompt": "Generate a product image.",
        "output": {"modality": "image", "format": "png"},
        "params": {"model": "example/image-model"},
    },
    result_key="image",
)
```

The runtime handler should:

- call AbstractCore locally, remotely, or through the existing hybrid configuration model;
- pass the runtime artifact store into AbstractCore when available;
- store image/audio bytes as artifacts;
- write only artifact refs, content types, provider/model metadata, and bounded warnings/errors into
  `RunState.vars`;
- leave `EffectType.LLM_CALL` in place for existing text workflows.

Runtime should not import AbstractVision or AbstractVoice directly for this feature. If Core's
contract is right, Runtime only needs to understand one generated result shape.

### User perspective

Users should not need to learn three different systems to generate multimodal outputs. The normal
mental model should be:

1. Say what to generate.
2. Say what output modality is required.
3. Receive text or durable artifact refs with metadata.

The public vocabulary should be small:

- `text`
- `image`
- `voice`
- `music` reserved as experimental

Internally, Core may map `voice` to audio/TTS backends and MIME types such as `audio/wav`, but the
user-facing request can use `voice` because that is the product intent. The result should still
record the technical content type.

## Proposal

### 0. Make capability readiness and catalogs first-class

Before broadening generation orchestration, Core should expose stable capability introspection:

- `llm.capabilities.status()` remains shallow and non-instantiating.
- `llm.vision.list_provider_models(...)` returns deep image provider catalogs when requested.
- `llm.voice.voice_catalog()` returns deep TTS profile/model catalogs when requested.
- Core Server exposes these through explicit media catalog routes, not through `/v1/models`.

This is important for Core as an entry point. A direct Core user, Gateway, Flow, or any thin client
must be able to preflight the configured generative media surface without importing AbstractVoice or
AbstractVision directly. Readiness should distinguish:

- plugin package installed;
- backend registered;
- backend configured;
- remote provider reachable / local engine loadable;
- generation route actually usable.

The generation contract below should then record the selected backend/model/profile in every
generated item.

### 1. Define a small generation request contract

Add a typed request/result layer in AbstractCore, even if the public API accepts plain dictionaries
for convenience.

Suggested concepts:

- `GenerationRequest`
- `GenerationOutputSpec`
- `GeneratedItem`
- `MultimodalGenerateResponse`
- `GenerationWarning`
- `GenerationError`

Minimum request fields:

```python
{
    "prompt": "...",
    "messages": [...],
    "media": [...],
    "output": {"modality": "image", "format": "png"},
    "params": {"temperature": 0.0, "seed": 123},
}
```

`prompt` and `messages` should follow existing AbstractCore semantics. `media` remains multimodal
input. `output` is the requested output modality or modalities.

### 2. Keep modality and task separate

Use `modality` for what comes out:

- `text`
- `image`
- `voice`
- later, `music`

Use `task` only when the same modality has multiple operations:

- `text_generation`
- `image_generation`
- `image_edit`
- `tts`
- `voice_clone`
- `speech_to_speech`
- `music_generation`

For v1, stable generation should focus on:

- `text` / `text_generation`
- `image` / `image_generation`
- `voice` / `tts`

This avoids overloading `voice` to mean every audio operation while still giving users a simple
word for the common "generate spoken audio" case.

### 3. Define source semantics explicitly

Mixed outputs need a clear source policy. For example:

```python
output=[
    {"modality": "text"},
    {"modality": "image", "source": "original_prompt"},
    {"modality": "voice", "source": "generated_text"},
]
```

Supported source values should start small:

- `original_prompt`
- `generated_text`
- `explicit`

For `explicit`, the output spec carries its own prompt/text:

```python
{"modality": "voice", "source": "explicit", "text": "Read exactly this."}
```

This removes ambiguity from workflows and avoids accidental text-first generation when the caller
only wanted an image or voice clip from the original prompt.

### 4. Return structured results with artifact support

The multimodal result should be inspectable and JSON-safe when an artifact store is provided:

```json
{
  "text": {
    "content": "Short explanation...",
    "model": "provider/text-model",
    "usage": {}
  },
  "outputs": {
    "image": [
      {
        "artifact": {"$artifact": "abc123"},
        "content_type": "image/png",
        "format": "png",
        "backend": "abstractvision:...",
        "model": "provider/image-model",
        "metadata": {}
      }
    ],
    "voice": [
      {
        "artifact": {"$artifact": "def456"},
        "content_type": "audio/wav",
        "format": "wav",
        "backend": "abstractvoice:...",
        "model": "provider/tts-model",
        "metadata": {}
      }
    ]
  },
  "warnings": []
}
```

In simple library mode, Core can return raw bytes when no artifact store is supplied. In framework
mode, Core should prefer artifact refs. Runtime should always use artifact refs for binary outputs.

For generated media, include enough routing metadata for auditability:

- capability backend id, e.g. `abstractvision:openai` or `abstractvoice:default`;
- provider/model id where known;
- profile/voice id for TTS;
- whether the output came from a remote provider, local engine, or compatible endpoint;
- bounded warnings/errors with `#FALLBACK` / `#TRUNCATION` labels where applicable.

### 5. Keep missing capabilities explicit

If a user asks for `image` and AbstractVision is not installed or no backend is configured, Core
should raise a structured capability error or return a failed output item according to an explicit
partial-failure policy. It should not silently drop the image output.

Recommended default:

- Single requested modality failure raises.
- Multiple requested modalities may return successful outputs plus bounded errors only if the user
  requested `partial=True` or equivalent.
- All routing changes and fallback choices are recorded in metadata.

### 6. Add one Runtime generation effect after Core stabilizes

Once Core exposes the unified contract, add one runtime integration effect:

```python
EffectType.GENERATION_CALL = "generation_call"
```

The effect payload should be intentionally close to Core's request shape:

```json
{
  "prompt": "...",
  "messages": [],
  "media": [],
  "output": [
    {"modality": "text"},
    {"modality": "image", "source": "generated_text"}
  ],
  "params": {
    "provider": "openai",
    "model": "..."
  }
}
```

The effect result should be normalized:

```json
{
  "text": {"content": "...", "model": "..."},
  "outputs": {
    "image": [
      {
        "artifact": {"$artifact": "..."},
        "content_type": "image/png",
        "metadata": {}
      }
    ]
  },
  "routing": {},
  "warnings": []
}
```

This keeps the durable workflow layer simple. Runtime does not need separate `IMAGE_GENERATE` and
`VOICE_GENERATE` effects unless real usage proves that separate effect types are easier to govern.

### 7. Keep existing APIs as compatibility layers

Do not remove lower-level APIs:

- `llm.generate(...)` for text-only generation remains stable.
- `llm.vision.t2i(...)` remains available for direct capability calls.
- `llm.voice.tts(...)` remains available for direct capability calls.
- `generate_with_outputs(...)` can become a compatibility wrapper over the new orchestrator.
- Runtime `LLM_CALL` remains available for existing workflow specs.

The new abstraction should reduce what users need to know, not remove escape hatches.

### 8. Keep Core as the entry point, not a package-specific proxy

Core should not simply proxy package-specific method names over HTTP. It should define the stable
framework vocabulary and map packages into it:

- Vision package vocabulary stays inside AbstractVision.
- Voice/profile package vocabulary stays inside AbstractVoice.
- Core exposes stable "image generation", "image edit", "voice/TTS", "speech transcription",
  "catalog", and "readiness" concepts.
- Gateway consumes Core's stable concepts, then adds deployment policy, workflows, memory, agents,
  and auth for thin clients.

This lets AbstractVoice/AbstractVision evolve internally while keeping Core as a coherent foundation
for scripts, standalone server users, Gateway, and Runtime effects.

## Why this shape

This design keeps the system easy to reason about:

- Users get one public generation concept.
- Core keeps ownership of provider/model/capability logic.
- Runtime stays a durable orchestrator, not a modality framework.
- Artifact refs prevent large binary payloads from polluting workflow state.
- Explicit modality selection keeps workflows reproducible.
- Music can be represented later without making it look stable today.

It also keeps the initial implementation small. The first useful release can support:

- text only with existing behavior;
- image only through AbstractVision;
- voice/TTS only through AbstractVoice;

## Deprecation report (2026-05-21)

This proposal is deprecated in the AbstractCore backlog because the Core-side work it describes is
already implemented and tracked in completed items (notably
`completed/2026-05-06_unified-multimodal-generate-api.md`, plus later capability catalog and music
selection fixes).

The remaining “durable AbstractRuntime effect/artifact-store integration” work belongs in
`../abstractruntime/` (not in this repository). Keeping this item in AbstractCore `proposed/`
creates duplicate planning memory and makes it look like Core still needs a unified multimodal
contract, which is no longer true.
- text plus image;
- text plus voice;
- artifact-backed runtime outputs.

## Evidence needed before promotion

Promote this to `planned/` when these are true:

- The team agrees on the canonical modality vocabulary and alias policy.
- AbstractVision and AbstractVoice releases expose enough stable artifact-backed capability methods
  and catalog methods for Core to wrap.
- Core package pins target those released versions (`abstractvoice>=0.9.1` expected for Voice;
  `abstractvision>=0.3.2` for Vision, verified from the PyPI wheel).
- The existing `generate_with_outputs(...)` behavior has been reviewed and either folded into the
  new orchestrator or marked as legacy.
- Runtime maintainers agree that one generation effect is preferable to separate modality-specific
  effects for v1.

## Suggested implementation

1. Write a short ADR for vocabulary:
   - `text`
   - `image`
   - `voice`
   - `music` experimental/reserved
2. Add Core dataclasses for output specs and generated artifacts.
3. Implement a central Core orchestrator that can call:
   - existing text `generate(...)`;
   - `self.vision.t2i(...)`;
   - `self.voice.tts(...)`;
   - music only behind explicit experimental gating.
4. Add Core facade methods for generative capability catalogs before or alongside the orchestrator:
   - `self.vision.list_provider_models(...)`;
   - `self.voice.voice_catalog()`;
   - route-level readiness/error normalization.
5. Preserve backward compatibility by returning `GenerateResponse` for text-only calls without
   `output=...`.
6. Add artifact-store support to the orchestrator and normalize raw plugin outputs into one result
   shape.
7. Add sync tests with fake text, vision, and voice capabilities.
8. Add async support after the sync contract is stable.
9. Add an AbstractRuntime `GENERATION_CALL` handler that invokes the Core contract and stores binary
   outputs as artifact refs.
10. Add runtime tests with a fake Core client returning image/audio bytes and artifact refs.
11. Add docs and examples showing:
    - image-only generation;
    - voice-only generation;
    - text plus image plus voice;
    - runtime workflow generation with artifact refs.

## Non-goals

- Do not make AbstractRuntime depend directly on AbstractVision or AbstractVoice.
- Do not require image, voice, or music dependencies for text-only AbstractCore users.
- Do not infer output modality from prompt text by default.
- Do not make `music` stable until the backend story is reliable.
- Do not silently skip failed requested outputs.
- Do not store generated binary outputs directly in Runtime run variables.
- Do not replace lower-level capability APIs; keep them as expert escape hatches.

## Validation ideas

Core tests:

- `llm.generate(prompt)` still returns the existing `GenerateResponse`.
- `llm.generate(prompt, output={"modality": "image"})` calls the vision capability and does not
  perform unnecessary text generation unless requested.
- `llm.generate(prompt, output={"modality": "voice"})` calls TTS and returns audio metadata.
- Combined text/image/voice generation respects `source`.
- Missing AbstractVision or AbstractVoice produces explicit capability errors.
- Artifact-store mode returns refs, not raw bytes.
- Partial-failure behavior is opt-in and bounded.
- Catalog/readiness calls do not trigger generation, do not auto-select "latest" models, and do not
  instantiate heavy local engines during shallow discovery.

Runtime tests:

- `GENERATION_CALL` stores generated image/audio bytes in `ArtifactStore`.
- Runtime vars contain only refs and metadata.
- The ledger result remains JSON-safe.
- Remote Core responses and local Core responses normalize to the same shape.
- Existing `LLM_CALL` workflows remain unchanged.

Documentation checks:

- Examples distinguish multimodal input from multimodal output.
- Examples show how to open/load generated artifacts.
- Music is documented as reserved or experimental, not as a stable promise.

## Guidance for future agents

Start in AbstractCore, not AbstractRuntime. First make the generation result contract boring and
predictable. Then add the runtime effect as a thin durable adapter over that contract.

Prefer explicit `output.modality` over prompt inference. If `auto` routing is added, make it an
opt-in mode that returns the selected modality, backend, model, and reason. Treat capability
visibility and artifact safety as correctness requirements.
