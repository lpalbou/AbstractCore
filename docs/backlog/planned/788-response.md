# 788 — Response: vision image generation endpoint mismatch & routing (v0)

## Context (how AbstractCore handles “generative vision”)

- **Generative vision (text→image / image→image)** is not part of AbstractCore’s default install; it comes from the optional `abstractvision` package.
- AbstractCore’s **gateway server** (FastAPI) exposes **OpenAI-compatible** image endpoints:
  - `POST /v1/images/generations`
  - `POST /v1/images/edits`
  - Implementation: `abstractcore/server/vision_endpoints.py`
- These endpoints delegate to **AbstractVision backends** (Diffusers / stable-diffusion.cpp / upstream OpenAI-compatible proxy) and return OpenAI-shaped responses (`data[].b64_json`).

## What the “mismatch” is

Across the framework, model ids are typically expressed as:

- `provider/model` (e.g. `openai/gpt-4o-mini`, `openrouter/anthropic/claude-...`, `huggingface/Qwen/Qwen-Image-2512`)

Historically, the `/v1/images/*` endpoints treated the `model` field as an **AbstractVision model id** (e.g. `Qwen/Qwen-Image-2512`) or a local path — so passing an AbstractCore-style id like:

- `openai/gpt-4o-mini` (a chat model id), or
- `huggingface/Qwen/Qwen-Image-2512` (a provider-prefixed vision model id)

could be mis-routed (e.g. interpreted as a Diffusers repo id) and fail in confusing ways.

## Current status in AbstractCore

This repo now normalizes AbstractCore-style ids for `/v1/images/*` routing:

### A) Local Diffusers models (recommended default)

You can use **either** form:

- `model="Qwen/Qwen-Image-2512"` (raw AbstractVision/HF id)
- `model="huggingface/Qwen/Qwen-Image-2512"` (AbstractCore-style; the `huggingface/` prefix is stripped)
- `model="mlx/Qwen/Qwen-Image-2512"` (AbstractCore-style; treated the same way for routing; device selection remains env-driven)

### B) “Chat model ids” fall back to your configured vision default

If a caller passes a chat model id like `openai/gpt-4o-mini` and you **don’t** have an upstream images proxy configured, the server treats it as “no explicit vision model” and uses your configured vision backend (env-driven).

That means other apps can keep sending their usual `model="provider/model"` and still get image generation — as long as the server has a default vision model configured.

### C) Upstream proxy routing (optional)

If `ABSTRACTCORE_VISION_UPSTREAM_BASE_URL` is set, provider-prefixed ids like `openrouter/...` will be routed through the upstream OpenAI-compatible images endpoint (the provider prefix is stripped before forwarding).

## How to enable a framework-default vision generator (Qwen/Qwen-Image-2512)

Run the AbstractCore server with `abstractvision` installed, and configure a default local Diffusers model:

```bash
pip install "abstractcore[server]"
pip install abstractvision

# Recommended default (local Diffusers)
export ABSTRACTCORE_VISION_BACKEND=diffusers
export ABSTRACTCORE_VISION_MODEL_ID=Qwen/Qwen-Image-2512

# Optional tuning
export ABSTRACTCORE_VISION_DEVICE=auto          # or: mps/cuda/cpu
export ABSTRACTCORE_VISION_TORCH_DTYPE=float16  # common for large models
export ABSTRACTCORE_VISION_ALLOW_DOWNLOAD=1     # set 0 for cache-only/offline

python -m abstractcore.server.app
```

### Call it (explicit vision model id)

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a red fox in snow, cinematic lighting",
    "model": "huggingface/Qwen/Qwen-Image-2512",
    "response_format": "b64_json"
  }'
```

### Call it (app passes a chat model id; server falls back to the configured vision default)

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a red fox in snow, cinematic lighting",
    "model": "openai/gpt-4o-mini",
    "response_format": "b64_json"
  }'
```

## Example: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)

This also works with classic Diffusers models like `runwayml/stable-diffusion-v1-5`:

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute corgi puppy wearing sunglasses, bright daylight, high quality photo",
    "negative_prompt": "nsfw, nude, naked",
    "model": "runwayml/stable-diffusion-v1-5",
    "response_format": "b64_json",
    "width": 256,
    "height": 256,
    "steps": 8,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

Notes:
- For some older SD1.x repos, a `variant=fp16` may not exist. If you see a `variant=fp16` error on Apple Silicon, set `ABSTRACTCORE_VISION_TORCH_DTYPE=float32` (or run on CPU).
- Very low step counts can trigger the safety checker and yield all-black outputs; use a reasonable `steps` value (e.g. `8+` for smoke tests).

## Notes / expectations

- If `abstractvision` is not installed in the server environment, `/v1/images/*` returns `501` with install hints.
- If the server has **no vision backend configured** (no request model that can be routed, and no relevant `ABSTRACTCORE_VISION_*` env vars), it returns `501` with an actionable configuration message.
- The default model choice (e.g. `Qwen/Qwen-Image-2512`) is a **deployment default**: configure it once on the server, and apps can stay simple.
