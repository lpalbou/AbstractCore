# Vision in AbstractCore (Image/Video Input)

This document describes **vision as an input modality** in AbstractCore (images and video-understanding), and clarifies how it relates to:
- **vision fallback** (caption → inject short observations), and
- **generative vision** (image/video creation), which lives in `abstractvision`.

## Quick requirements

- **Images**: install `pip install "abstractcore[media]"` and use either:
  - a **vision-capable model** (VLM/VL), or
  - a text-only model with **vision fallback** configured (`abstractcore --set-vision-provider PROVIDER MODEL`).
- **Video**: native video input is model/provider dependent. For the portable frame-sampling path (`video_policy="frames_caption"` / `"auto"` fallback), you need:
  - `ffmpeg`/`ffprobe` available on `PATH`, and
  - image/vision handling (a vision-capable model or configured vision fallback).

## 1) Image/video input modalities (owned by AbstractCore)

Attach media to an LLM call using `media=[...]`:

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")  # example; pick a vision-capable model you have access to
resp = llm.generate("What is in this image?", media=["photo.jpg"])
print(resp.content)
```

Support depends on the selected provider/model and is normalized via:
- `abstractcore/assets/model_capabilities.json`

Video attachments use the same `media=[...]` surface and are controlled by `video_policy` (see `abstractcore/providers/base.py`):

```python
resp = llm.generate(
    "Summarize what happens in this clip.",
    media=["clip.mp4"],
    video_policy="auto",  # native when supported; otherwise sample frames
)
```

You can tune frame sampling defaults via the config CLI:

```bash
abstractcore --set-video-strategy auto
abstractcore --set-video-max-frames 6
abstractcore --set-video-sampling-strategy keyframes
```

## 2) Vision fallback for text-only models (optional; config-driven)

When a user attaches an image to a text-only model, AbstractCore can optionally run a **two-stage fallback**:
1) run a configured vision-capable backend to produce **short grounded observations**, then
2) inject those observations into the main request.

This is:
- **explicit** (config-driven; not a silent default), and
- **transparent** via response metadata (`metadata.media_enrichment[]`).

Code pointers:
- Fallback handler: `abstractcore/media/vision_fallback.py`
- Enrichment metadata: `abstractcore/media/enrichment.py`

Configure vision fallback via the config CLI:

```bash
abstractcore --set-vision-provider lmstudio qwen/qwen3-vl-4b
abstractcore --add-vision-fallback huggingface Salesforce/blip-image-captioning-base
```

## 3) Generative vision output is not part of AbstractCore’s default install

Creating/editing images and videos is a **deterministic capability** that lives in `abstractvision` and can be integrated in two ways:

1) **Capability plugin (library mode)**: install `abstractvision` and use `llm.vision.*` (e.g. `t2i`, `i2i`).  
   See: `abstractvision/docs/reference/abstractcore-integration.md`

2) **AbstractCore Server (HTTP interop)**: run the optional server and enable `/v1/images/*` endpoints delegated to `abstractvision`.  
   See: `docs/server.md`

This separation keeps the default `abstractcore` install dependency-light: input handling lives in AbstractCore, and generative vision outputs live in `abstractvision`.

## Troubleshooting (common)

- **“Image input is not supported by model …”**: choose a vision-capable model, or configure vision fallback.
- **Vision fallback errors**: confirm your AbstractCore config enables it and that the configured backend is reachable/works.
- **Video frame fallback issues**: frame extraction relies on `ffmpeg`/`ffprobe` availability in the runtime environment, and requires image/vision handling (vision-capable model or configured vision fallback).

## Related
- Media pipeline overview: `docs/media-handling-system.md`
- Server endpoints: `docs/server.md`
- Capability plugins (voice/audio/vision): `docs/capabilities.md`
- Architecture overview: `docs/architecture.md`
