# Request And Output

AbstractCore still supports the familiar prompt-first forms:

```python
resp = llm.generate("Summarize this.")
resp = llm.generate("Red cube", output="image")
resp = llm.generate(media={"type": "audio", "path": "meeting.wav"}, output="text")
```

The lower-level canonical shape is now the keyword form:

```python
resp = llm.generate(
    request={
        "text": "Turn this into a calm synth loop.",
        "media": [],
    },
    output={"modality": "music", "format": "wav"},
)
```

`request=` and the legacy `prompt` / `text` / `messages` / `media` kwargs normalize to the same
internal contract. This keeps the public API compatible while giving Core one stable semantic
request shape under the hood.

## Request shape

The stable request payload is intentionally small:

- `text`: plain request text
- `messages`: optional chat history
- `media`: optional image/audio/video/document inputs

Routing fields do not live inside `request`. Provider/model/base URL selection still comes from:

- explicit call arguments and output-spec fields when you intentionally pin a route;
- capability defaults (`input.*`, `output.*`) when you do not; or
- the provider instance you created with `create_llm(...)`.

## Output shape

`output=` stays separate from `request` and remains the public generated-output selector:

```python
output="text"
output={"modality": "image"}
output={"modality": "video", "task": "image_to_video"}
output={"task": "tts"}
```

Core infers the concrete task structurally when it can:

- text request + `output=image` -> text-to-image
- one source image + `output=image` -> image edit
- one source image + `output=video` -> image-to-video
- audio media + `output=text` with no text prompt -> transcription
- audio media + `output=voice` with no explicit voice id -> voice clone

## Defaults and explicit overrides

The request/output contract does not remove explicit control. Manual routing still works when you
need it:

```python
resp = llm.generate(
    request={"text": "Slow dolly shot over a misty valley."},
    output={
        "modality": "video",
        "provider": "mlx-gen",
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "num_frames": 41,
    },
)
```

When you do not pin a route, capability defaults can fill in provider/model/base URL and, for
reasoning-capable text routes, a default `reasoning` level.

## Route inspection

For debugging and replay-oriented integrations, responses now include a bounded normalized route
summary in metadata:

```python
resp = llm.generate(request={"text": "hello"}, output="text")
route = resp.metadata.get("_resolved_generate_route")
```

That summary is meant for inspection and integration, not for ordinary application logic. It
records the normalized request, normalized outputs, resolved text/input/output routes, and the
effective reasoning default when one applied.

## Related

- [Getting Started](getting-started.md)
- [Centralized Config](centralized-config.md)
- [Capabilities](capabilities.md)
- [Server](server.md)
