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
- `abstractcore/assets/model_capabilities.json` (source of truth; update when new vision-capable models ship)

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

## 3) Generative vision output is dependency-light by default

Creating/editing images and videos is a **deterministic capability** that can be integrated in two ways:

1) **Capability plugin (library mode)**: install `abstractvision` and use `llm.vision.*` (e.g. `t2i`, `i2i`, `upscale_image`, `t2v`, `i2v`) or the unified `llm.generate(..., output=...)` surface. Configure the AbstractVision backend/default for your environment; local Diffusers remains cache-only unless downloads are explicitly enabled, and MLX-Gen local models are selected by exact repo id. Install `abstractvision[mlx-gen]` when you need the local MLX-Gen runtime.
   See: `abstractvision/docs/reference/abstractcore-integration.md`

2) **AbstractCore Server (HTTP interop)**: run the optional server and use `/v1/images/*` and `/v1/videos/*` as OpenAI-compatible media routes. Local Diffusers/sdcpp/MLX-Gen backends remain available when `abstractvision` and the needed backend runtime extra are installed in the server environment; `abstractcore[server,vision]` installs the plugin API surface, while `abstractvision[mlx-gen]` or aggregate profiles such as `abstractcore[all-apple]` provide local MLX-Gen execution. Omit `model` only when the server has a configured default, or use provider/model ids such as `model="diffusers/default"`, `model="diffusers/<huggingface-repo>"`, `model="mlx-gen/AbstractFramework/qwen-image-2512-4bit"`, `model="mlx-gen/AbstractFramework/seedvr2-3b-8bit"`, `model="mlx-gen/AbstractFramework/seedvr2-7b-4bit"`, `model="mlx-gen/AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"`, `model="sdcpp/default"`, or `model="openai-compatible/gpt-image-2"` with a configured upstream media endpoint.
   See: `docs/server.md`

AbstractVision remains the truth owner for route-specific model and adapter
compatibility. AbstractCore hosts that truth and exposes it through:

- `llm.vision.list_provider_models(task=...)`
- `llm.vision.list_provider_adapters(provider=..., model=..., task=...)`
- `GET /v1/vision/models`
- `GET /v1/vision/adapters`

Python progress callbacks can be supplied on the unified call for generated image/video outputs:

```python
def on_progress(event):
    print(event)

upscaled_direct = llm.vision.upscale_image(
    "input.png",
    provider="mlx-gen",
    model="AbstractFramework/seedvr2-3b-8bit",
    resolution="2x",
    softness=0.25,
    on_progress=on_progress,
)

upscaled = llm.generate(
    media={"type": "image", "path": "input.png", "role": "source"},
    on_progress=on_progress,
    output={
        "task": "image_upscale",
        "provider": "mlx-gen",
        "model": "AbstractFramework/seedvr2-3b-8bit",
        "resolution": "2x",
        "softness": 0.25,
    },
)
png = upscaled.outputs["image"][0].data

resp = llm.generate(
    "A slow camera move through a luminous data center.",
    on_progress=on_progress,
    output={
        "task": "text_to_video",
        "provider": "mlx-gen",
        "model": "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
        "width": 432,
        "height": 240,
        "num_frames": 41,
        "fps": 24,
        "steps": 20,
        "guidance_scale": 4.0,
        "guidance_2": 3.0,
        "extra": {"max_sequence_length": 256},
    },
)
mp4 = resp.outputs["video"][0].data
```

For image-to-video, pass one source image and set `task="image_to_video"`:

```python
resp = llm.generate(
    "Slow camera push-in.",
    media={"type": "image", "path": "first-frame.png", "role": "source"},
    output={
        "task": "image_to_video",
        "provider": "mlx-gen",
        "model": "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
        "width": 432,
        "height": 240,
        "num_frames": 41,
        "fps": 24,
        "steps": 20,
        "guidance_scale": 3.5,
        "guidance_2": 3.5,
        "extra": {"max_sequence_length": 256},
    },
)
```

Batch generation uses the same Core surface. Set `count` / `n` plus either a
base `seed` or an explicit `seeds=[...]` list. Core delegates the planning to
AbstractVision instead of inventing a separate seed policy:

```python
resp = llm.generate(
    "An isometric research outpost on an icy exoplanet at blue hour.",
    output={
        "task": "text_to_image",
        "provider": "mlx-gen",
        "model": "AbstractFramework/qwen-image-2512-8bit",
        "count": 2,
        "seeds": [2512, 2513],
        "lora_adapters": [
            {
                "source": "prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA:Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors",
                "scale": 1.0,
            }
        ],
    },
)
assert len(resp.outputs["image"]) == 2
```

The same typed `lora_adapters=[...]` contract works for `text_to_image`,
`image_to_image`, `text_to_video`, and `image_to_video`. Video routes also keep
typed `guidance_2` and `flow_shift` fields instead of burying them inside
generic `extra`.

Image edits can also pass additional media items with `role="reference"` or
`role="style"`; Core forwards those as AbstractVision `reference_images` for
backends that support multi-image composition. Async HTTP routes under
`/v1/vision/jobs/images/*` and `/v1/vision/jobs/videos/*` expose
`progress.last_event` when the selected backend reports richer progress events.
For MLX-Gen, `progress` is denoise-step progress; video frame context is exposed
separately as `frame`, `total_frames`, and `frame_progress`.

For task-specific Wan A14B video models, pass `guidance_2` as a normal output
field when you need the second-stage/low-noise guidance control. Keep
backend-specific fields such as `max_sequence_length` in `extra`.

This separation keeps the default `abstractcore` install dependency-light: remote media proxying lives in the server, while local generative vision runtimes remain opt-in through `abstractvision`. Quantized MLX-Gen generation/edit/video models are selected by their published repo id. SeedVR2 upscaling follows the same rule for the canonical packages: use `AbstractFramework/seedvr2-3b-8bit` by default, `AbstractFramework/seedvr2-7b-8bit` when memory allows, or the matching q4 package when memory is tight. The default upscaler request uses `resolution="2x"` and `softness=0.25`. Core forwards the runtime `quantize` request field only for official/source SeedVR2 loads that need runtime quantization.

## Troubleshooting (common)

- **“Image input is not supported by model …”**: choose a vision-capable model, or configure vision fallback.
- **Vision fallback errors**: confirm your AbstractCore config enables it and that the configured backend is reachable/works.
- **Video frame fallback issues**: frame extraction relies on `ffmpeg`/`ffprobe` availability in the runtime environment, and requires image/vision handling (vision-capable model or configured vision fallback).

## Related
- Media pipeline overview: `docs/media-handling-system.md`
- Server endpoints: `docs/server.md`
- Capability plugins (voice/audio/vision): `docs/capabilities.md`
- Architecture overview: `docs/architecture.md`
