# Vision in AbstractCore (Image/Video Input)

This document describes **vision as an input modality** in AbstractCore (images and video-understanding), and clarifies how it relates to:
- **vision fallback** (caption → inject short observations), and
- **generative vision** (image/video creation), which lives in `abstractvision`.

## Quick requirements

- **Images**: install `pip install "abstractcore[media]"` and use either:
  - a **vision-capable model** (VLM/VL), or
  - a text-only model with **vision fallback** configured (`abstractcore --set-vision-provider PROVIDER MODEL`).
  - On Apple silicon, native image input on MLX checkpoints needs no extra step: `mlx-vlm`
    ships with the MLX provider in `abstractcore[mlx]`, `[apple]`, `[all]`, `[all-apple]` and
    `[full-dev]` (see [Native image input on the MLX provider](#1b-native-image-input-on-the-mlx-provider-apple-silicon)).
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

## 1b) Native image input on the MLX provider (Apple silicon)

The MLX provider reads images natively for vision-capable MLX checkpoints. Nothing extra to
install — `mlx-vlm` is part of the MLX provider's dependency set, so any profile that gives you
`mlx-lm` also gives you image input. Pass `media=[...]` as usual:

```bash
pip install "abstractcore[apple]"
```

If a sighted checkpoint still drops images with `mlx_vlm_not_installed`, the environment is
missing half of a package set that is meant to arrive together — most often because the
interpreter running the model is not the one that was installed into. Check with
`<the-python-that-runs-the-model> -c "import mlx_vlm"` and repair with
`pip install "abstractcore[mlx]"`.

```python
from abstractcore import create_llm

llm = create_llm("mlx", model="mlx-community/Qwen3.5-4B-MLX-4bit")
resp = llm.generate("Describe this image in one sentence.", media=["photo.png"])
print(resp.content)
```

The extra is separate from `abstractcore[mlx]` and `abstractcore[apple]` on purpose: it pulls a
web framework, OpenCV, a datasets stack, and raises the `transformers` floor, none of which belong
in a text-only local LLM install.

### Supported checkpoints

Support is decided from the checkpoint on disk and from the capability registry, not from the
model name. A checkpoint is served when:

- the capability registry declares it vision-capable (`abstractcore/assets/model_capabilities.json`);
- its `config.json` declares a `vision_config` and the vision weights are present;
- it declares an image token so the image position can be rendered into the prompt; and
- the encoder and decoder agree on the embedding convention (probed at load, and reconciled
  automatically where the difference is a uniform scale).

Verified families include `qwen3_5`, `qwen3_5_moe`, `qwen3_vl` and `gemma4`. Text-only checkpoints
are unaffected and never load the vision stack — `mlx_vlm` is imported only when an image is
actually attached.

### What you get back

A delivered image is reported positively in `response.metadata`:

```python
resp.metadata["media_delivered"]
# [{"index": 0, "kind": "image", "sha256": "3da0b68c…",
#   "tokens": 1024, "transport": "mlx_vision_addon"}]
```

- `tokens` is the measured number of image tokens the model actually consumed.
- `sha256` identifies which image was delivered.
- `fidelity`, when present, names any known precision trade-off for that checkpoint. For example
  `rope_1d_substituted` appears on models whose positional encoding is approximated, which can
  cost fine character-level detail in dense text.

If the image could not be carried, `media_delivered` is absent and `response.metadata["media_dropped"]`
names the reason. Use `abstractcore.media.delivery.media_delivery_verdict(response, provider="mlx")`
when you need a single answer:

```python
from abstractcore.media.delivery import media_delivery_verdict

verdict = media_delivery_verdict(resp, provider="mlx")
verdict.state  # "delivered" | "not_delivered" | "unverified"
```

`unverified` means the provider does not participate in the delivery contract, not that the image
was lost.

### Current limits

- **One image per request.** Multiple images in a single call are refused with
  `vision_multi_image_unsupported`; the request still answers from text.
- **Structured output does not carry images.** `response_model=` together with `media=[...]`
  raises rather than dropping the image silently, because a validated model carries no metadata on
  which the drop could be reported. Parse the text instead, or caption the image first.
- **Prompt-cache reuse is skipped on turns that carry an image.** Text-only turns in the same
  session keep the full prompt cache; see [Prompt caching](prompt-caching.md).
- Checkpoints the lane cannot serve fall through to the vision fallback below, so an image still
  produces a caption when one is configured.

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
