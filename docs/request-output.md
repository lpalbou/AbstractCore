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

A sound effect uses the `sound` output and takes its length in `seconds` (or `duration_s`); the
route's `model` and the task reach the music plugin, and a sound effect without a length is 5
seconds:

```python
resp = llm.generate(
    text="laser gunshot",
    output={"modality": "sound", "task": "text_to_audio", "seconds": 3, "format": "wav"},
)
clip = resp.outputs["sound"][0]
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

When you do not pin a route, capability defaults fill in provider/model/base URL and, for the
text-generation route, a default reasoning effort.

### The reasoning effort

The text-generation route carries an optional `reasoning` field alongside its provider and model.
It is the execution host's default reasoning effort, and it follows the same precedence as the
provider and the model:

1. **an explicit `thinking=` on the call**, including `thinking=False`, which turns reasoning off
   for that call, and `thinking="auto"`, which asks for no control at all;
2. **a provider-native reasoning field on the call** (`reasoning_effort=`, `reasoning=`, `think=`),
   sent as given;
3. **the effort configured on the text route**, applied when the call names none of the above and
   the route describes the instance serving the call: same provider (or endpoint profile) and same
   model. A route that names only a reasoning effort applies to every instance. A route configured
   for another provider or model does not set this instance's reasoning;
4. **nothing** — no reasoning parameter is sent and the model behaves as it does without one.

The full rule, with the metadata that reports which one applied, is in
[Reasoning Control](reasoning-control.md#precedence).

```bash
abstractcore config set-default output.text --provider lmstudio --model qwen3-30b --reasoning high
```

The same field is editable from the AbstractGateway console, and both entry points write the same
row. See [Centralized Config](centralized-config.md) for the full route grid.

## Capability default routes: one store, two entry points

There are two entry points to the framework — AbstractCore (low level) and AbstractGateway (high
level) — and **one** store of per-modality defaults. AbstractCore owns it
(`~/.abstractcore/config/abstractcore.json`, key `capability_defaults.routes`). AbstractGateway is a
full CRUD surface over that store, not a second copy: every gateway read hits Core's manager and
every gateway write goes through Core's setter. Setting a default in the Gateway console configures
Core's default, and both entry points then read the same value.

Route keys are a `kind.modality[.task]` grid. `kind` is the direction of the **media** the route
handles, not the direction of your intent — that is the only reading under which every modality
lands in exactly one cell (speech-to-text provisions the *audio* side, so it is an `input` route).

| Generation task | Route key | Broad fallback |
| --- | --- | --- |
| `text_generation` | `output.text` (derived, read-only — canonical storage is `input.text`) | — |
| `image_generation` / `text_to_image` | `output.image.text_to_image` | `output.image` |
| `image_to_image` / `image_edit` | `output.image.image_to_image` | `output.image` |
| `image_upscale` | `output.image.image_upscale` | `output.image` |
| `text_to_video` | `output.video.text_to_video` | `output.video` |
| `image_to_video` | `output.video.image_to_video` | `output.video` |
| `tts` | `output.voice` | — |
| `voice_clone` (voice output + reference audio) | `output.voice` | — |
| `stt` / `transcription` | `input.voice` | — |
| `music_generation` | `output.music` | — |
| `sound_generation` | `output.sound` | — |
| `text_to_scene3d` | `output.scene3d.text_to_scene3d` | `output.scene3d` |
| `image_to_scene3d` | `output.scene3d.image_to_scene3d` | `output.scene3d` |
| image / video / sound / music **understanding** | `input.image` / `input.video` / `input.sound` / `input.music` | covered by `input.text` when that model is multimodal |

This mapping is stated once in code, in
`abstractcore/config/capability_defaults.py::_OUTPUT_ROUTE_TABLE`
(`capability_route_key_for_output`). Core's execution path and AbstractRuntime's media path both
delegate to it.

Two rules follow from the table:

- **A `.task` suffix is only legal for the tasks the store persists** — the seven in
  `CAPABILITY_ROUTE_TASKS`. `tts`, `stt`, `music_generation` and `sound_generation` have no
  sub-route; they resolve at the modality cell. Writing `output.voice.tts` is rejected by
  `set_capability_default` and dropped on load.
- **A bare request with a source image attached** resolves to the edit variant
  (`image_to_image` / `image_to_video` / `image_to_scene3d`) rather than the text-to-X one.

**A pin always wins.** A default only ever fills a field the request left absent. Naming
`provider`, `model` *or* `base_url` on an output spec makes that spec the author's: the route
contributes nothing further to it (an explicit provider that contradicts the configured row also
drops the row's options — see `_route_row_contribution`). So a default is safe to set: it changes
what unpinned calls do, never what pinned ones do.

### Setting one from the Core entry point

```bash
abstractcore config set-default output.voice --provider supertonic --model supertonic-3
abstractcore config set-default output.image.text_to_image --provider mlx-gen --model <model-id>
abstractcore config set-default output.text            # shows the row
abstractcore config clear-default output.music
```

AbstractCore's console-TUI (`abstractcore --config`) drives the same command. In Python:
`get_capability_default` / `set_capability_default` / `clear_capability_default` on the
configuration manager; `list_capability_defaults()` returns the whole grid, each row naming its
source (`abstractcore.capability_defaults`, `not_configured`, or a derived/covered marker).

Either spelling writes the one store, so a running AbstractGateway picks it up too: the Gateway host
fingerprints the config file and re-publishes the defaults to its live runtime on the next run (see
`abstractgateway/docs/configuration.md`).

## Route inspection

For debugging and replay-oriented integrations, responses carry a bounded normalized route
summary in metadata:

```python
resp = llm.generate(request={"text": "hello"}, output="text")
route = resp.metadata.get("_resolved_generate_route")
```

That summary is meant for inspection and integration, not for ordinary application logic. It
records the normalized request, normalized outputs, resolved text/input/output routes, and the
effective reasoning default when one applied. When a configured effort was not applied,
`reasoning` is `None` and `reasoning_source` says why (`not_applied:explicit_reasoning_effort`,
`not_applied:route_provider_differs`, `not_applied:route_model_differs`).

## Related

- [Getting Started](getting-started.md)
- [Centralized Config](centralized-config.md)
- [Capabilities](capabilities.md)
- [Server](server.md)
