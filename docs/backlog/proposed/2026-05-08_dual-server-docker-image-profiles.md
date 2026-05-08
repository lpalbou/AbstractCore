# Proposed: Dual server Docker image profiles

## Metadata
- Created: 2026-05-08
- Status: Proposed
- Completed: N/A

## Context

As of AbstractCore 2.13.11, release automation publishes one GitHub Container
Registry image:

```text
ghcr.io/lpalbou/abstractcore-server:<version>
ghcr.io/lpalbou/abstractcore-server:latest
```

The current Dockerfile is `docker/abstractcore-server/Dockerfile`. It is based
on `python:<version>-slim`, installs AbstractCore from the matching PyPI wheel,
and currently installs:

```text
abstractcore[server,remote,media,tokens,compression]==<version>
```

This is the right default for a dependency-light OpenAI-compatible gateway. It
is not a local inference image. It intentionally avoids local LLM engines,
local embedding engines, Diffusers, vLLM, MLX, and the heavy native audio/image
plugin stacks.

The install-profile ADR already defines:

- `abstractcore[apple]`: native Apple local LLM stack alias for MLX.
- `abstractcore[gpu]`: local GPU LLM stack alias for vLLM.
- `abstractcore[all-apple]`: aggregate Apple local-development stack.
- `abstractcore[all-gpu]`: aggregate NVIDIA/GPU local-development stack.

`abstractcore[all-gpu]` currently includes Core server dependencies, remote
provider SDKs, Hugging Face/Torch local LLM dependencies, vLLM, local embedding
dependencies, media parsing, tools, AbstractVoice base plugin package, and
AbstractVision base plugin package. It does not automatically include every
heavy local AbstractVoice or AbstractVision optional runtime unless those plugin
extras are requested too.

## Question

Should AbstractCore publish one Docker image or two?

## Recommendation

Publish two official server image profiles after each PyPI release:

1. `ghcr.io/lpalbou/abstractcore-server:<version>`
   - Default lightweight server image.
   - Remote-first, multi-architecture, low operational surprise.
   - This image should remain the recommended image for hosted provider access,
     remote OpenAI-compatible endpoints, API gateway use, and most thin-client
     deployments.
2. `ghcr.io/lpalbou/abstractcore-server-gpu:<version>`
   - NVIDIA/CUDA server image.
   - Linux amd64 only.
   - Intended for users who want one container that can serve Core as an
     OpenAI-compatible server and also use local GPU-backed generation where the
     installed providers/plugins support it.

Do not publish an official `all-apple` Docker image for now. MLX depends on
Apple's native Metal runtime and is not a good fit for normal Linux Docker
deployment. `abstractcore[all-apple]` should remain a pip install profile for
native macOS environments.

## Image 1: lightweight server

### Name

```text
ghcr.io/lpalbou/abstractcore-server:<version>
ghcr.io/lpalbou/abstractcore-server:latest
```

### Install command

```bash
python -m pip install \
  "abstractcore[server,remote,media,tokens,compression]==<version>"
```

### Installed dependency groups

- Server:
  - `fastapi`
  - `uvicorn[standard]`
  - `python-multipart`
  - `sse-starlette`
- Remote provider SDKs:
  - `openai`
  - `anthropic`
  - OpenRouter, Portkey, Ollama, LM Studio, and generic OpenAI-compatible
    providers use Core HTTP dependencies.
- Media/document parsing helpers:
  - `Pillow`
  - `pymupdf4llm`
  - `pymupdf-layout`
  - `unstructured[docx,pptx,xlsx,odt,rtf]`
  - `pandas`
- Token/counting helpers:
  - `tiktoken`
- Compression helpers:
  - `Pillow`

### Not installed

- `vllm`
- `mlx` / `mlx-lm`
- `transformers`, `torch`, `torchvision`, `torchaudio`
- `sentence-transformers`
- `abstractvoice`
- `abstractvision`
- Diffusers or stable-diffusion.cpp
- local AbstractVoice engines such as Piper, Faster Whisper, F5-TTS, Chroma,
  OmniVoice, or AEC dependencies

### Capabilities

- Remote chat/responses through OpenAI, Anthropic, OpenRouter, Portkey, and
  OpenAI-compatible endpoints.
- Remote embeddings where provider support exists.
- Remote OpenAI-compatible audio and image proxy routes.
- Document/media input parsing.
- Swagger/OpenAPI server UX.

### Non-goals

- No local model weights.
- No local image generation.
- No local voice synthesis/transcription/cloning.
- No local vLLM server embedded in the image.

## Image 2: NVIDIA all-gpu server

### Name

Preferred:

```text
ghcr.io/lpalbou/abstractcore-server-gpu:<version>
ghcr.io/lpalbou/abstractcore-server-gpu:latest
```

Alternative if we want the name to mirror the extra exactly:

```text
ghcr.io/lpalbou/abstractcore-server-all-gpu:<version>
ghcr.io/lpalbou/abstractcore-server-all-gpu:latest
```

The shorter `abstractcore-server-gpu` name is clearer for users and avoids
confusing the image name with a Python extra syntax. The docs can still say that
the image is based on the `abstractcore[all-gpu]` install profile.

### Platform

```text
linux/amd64 only
```

The image should not be built for `linux/arm64` until there is a tested
NVIDIA/CUDA arm64 path for every advertised local capability. Users should run
it with the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all ...
```

### Base image

Use a CUDA/PyTorch runtime base rather than `python:slim`.

The exact base should be selected by build validation, but the direction should
be one of:

- `pytorch/pytorch:<torch-version>-cuda<cuda-version>-cudnn<version>-runtime`
- `nvidia/cuda:<cuda-version>-cudnn-runtime-ubuntu<version>` plus a pinned
  Python install

The selected base must be pinned. Do not float CUDA or PyTorch base tags.

### Minimum install command

If the intent is "Core all-gpu server, but local voice/image engines remain
plugin-runtime opt-ins":

```bash
python -m pip install "abstractcore[all-gpu]==<version>"
```

This installs:

- everything from the lightweight server path;
- `openai`, `anthropic`;
- `transformers`, `torch`, `torchvision`, `torchaudio`;
- `llama-cpp-python`;
- `outlines`;
- `vllm`;
- `sentence-transformers`;
- `numpy`;
- `requests`, `beautifulsoup4`, `lxml`, `ddgs` or `duckduckgo-search`,
  `psutil`;
- `abstractvoice` base package;
- `abstractvision` base package.

This is useful, but the name "all capabilities" would be too strong because the
base AbstractVoice and AbstractVision packages deliberately do not install their
heavy local runtime extras.

### Recommended all-capability install command

If the image is marketed as a "full NVIDIA all capabilities" image, it should
install `abstractcore[all-gpu]` plus the local media plugin runtimes:

```bash
python -m pip install \
  "abstractcore[all-gpu,vision-local]==<version>" \
  "abstractvoice[local]>=0.9.1"
```

This keeps AbstractCore as the source of truth for the Core profile while
explicitly opting into the plugin-owned local voice/image stacks.

Longer term, AbstractCore could add `voice-local` / `audio-local` extras so the
Dockerfile can avoid naming `abstractvoice[local]` directly:

```bash
python -m pip install \
  "abstractcore[all-gpu,vision-local,voice-local]==<version>"
```

That should be considered only if we want Core to expose a first-class local
voice install profile. It is not required to publish the Docker image.

### Additional installed local image dependencies

From `abstractvision[local]` / `abstractcore[vision-local]`:

- `diffusers`
- `torch`
- `transformers`
- `accelerate`
- `safetensors`
- `sentencepiece`
- `protobuf`
- `einops`
- `peft`
- `stable-diffusion-cpp-python`
- `Pillow`

### Additional installed local voice dependencies

From `abstractvoice[local]`:

- `piper-tts`
- `faster-whisper`
- `sounddevice`
- `webrtcvad`
- `soundfile`
- `librosa`
- `huggingface_hub`
- `f5-tts` on Python 3.10+
- `torch`
- `torchaudio` on Python 3.10+
- `torchvision` on Python 3.10+
- `transformers`
- `accelerate` on Python 3.10+
- `av` on Python 3.10+
- `audioread` on Python 3.10+
- `pillow` on Python 3.10+
- `safetensors`
- `einops`
- `sentencepiece`
- `omnivoice` on Python 3.10+
- `aec-audio-processing` on Python 3.11+

The image should use Python 3.11 or 3.12, not Python 3.9, because multiple
local voice runtimes intentionally have Python 3.10+ or Python 3.11+ markers.

### Capabilities

- All lightweight server capabilities.
- Local vLLM-backed text generation where configured and supported.
- Local Hugging Face/Torch-backed provider paths where configured and supported.
- Local embeddings through `sentence-transformers`.
- AbstractVision plugin registration and provider catalog discovery.
- Local image generation through AbstractVision local runtimes when models are
  available and configured.
- AbstractVoice plugin registration and voice catalog discovery.
- Local TTS/STT/voice cloning through AbstractVoice local runtimes when models
  are available and configured.
- Remote audio/image routes still remain available.

### What should not be baked into the image

- Model weights.
- User API keys.
- Hugging Face tokens.
- User voice profiles.
- User media outputs.
- Provider configuration files containing secrets.

Models and caches should be mounted as volumes, for example:

```bash
docker run --rm --gpus all \
  -p 127.0.0.1:8000:8000 \
  --env-file .env \
  -v "$HOME/.cache/huggingface:/home/abstractcore/.cache/huggingface" \
  -v "$PWD/abstractcore-cache:/home/abstractcore/.cache/abstractcore" \
  ghcr.io/lpalbou/abstractcore-server-gpu:<version>
```

Relevant environment variables should remain the same as the lightweight server:

- `ABSTRACTCORE_SERVER_API_KEY`
- `ABSTRACTCORE_SERVER_PROTECT_DOCS`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `PORTKEY_API_KEY`
- `PORTKEY_CONFIG`
- `OPENAI_COMPATIBLE_BASE_URL`
- `OPENAI_COMPATIBLE_API_KEY`
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for gated models when needed
- model/provider-specific AbstractCore, AbstractVision, and AbstractVoice config
  variables documented by those packages

## Why two images

One image cannot serve both audiences well:

- The default server user wants a small, fast, secure gateway that starts
  quickly and only needs provider keys.
- The local GPU user accepts a large image, CUDA coupling, slower pulls, model
  cache volumes, and machine-specific deployment requirements.

Publishing only a heavy GPU image would make the common remote gateway path
worse. Publishing only the lightweight image leaves NVIDIA users without a
turnkey server deployment path for local inference.

## Release workflow proposal

On every release tag, after PyPI publication succeeds:

1. Build and publish the lightweight image:
   - `ghcr.io/lpalbou/abstractcore-server:<version>`
   - `ghcr.io/lpalbou/abstractcore-server:latest`
   - platforms: `linux/amd64,linux/arm64`
2. Build and publish the GPU image:
   - `ghcr.io/lpalbou/abstractcore-server-gpu:<version>`
   - `ghcr.io/lpalbou/abstractcore-server-gpu:latest`
   - platforms: `linux/amd64`
3. Create the GitHub release only after both image jobs pass.

If the GPU image build is too slow or too flaky for the main release workflow,
make it a separate workflow that is triggered by the release workflow and marks
the GitHub release notes with the GPU image status.

## Required validation before promotion

Promote this to `planned/` only after deciding:

- final GPU image name;
- whether the GPU image is `abstractcore[all-gpu]` only or true full local
  media by also installing `abstractcore[vision-local]` and `abstractvoice[local]`;
- CUDA/PyTorch base image and Python version;
- whether `latest` should be published for the GPU image or only versioned tags;
- whether model-cache volume conventions should be standardized.

Implementation validation should include:

- Build lightweight image locally or in CI.
- Build GPU image in CI on a runner capable of at least resolving/installing the
  full dependency stack.
- Smoke test lightweight image:
  - `/health`
  - `/v1/models`
  - `/v1/chat/completions` remote provider
  - `/v1/embeddings` remote provider
  - `/v1/images/generations` remote OpenAI-compatible proxy
  - `/v1/audio/speech` remote OpenAI-compatible proxy
- Smoke test GPU image on an NVIDIA runner:
  - `nvidia-smi` visible inside the container
  - `/health`
  - vLLM-backed text generation or model-load dry run
  - local embedding generation
  - AbstractVision provider catalog route
  - AbstractVoice voice catalog route
  - local image generation with a small/default configured model when feasible
  - local STT/TTS smoke tests when model artifacts are available
- Confirm no API keys or model weights are baked into either image.
- Confirm server auth works identically in both images.
- Confirm docs explain when to choose each image.

## Non-goals

- Do not replace PyPI as the canonical Python package release.
- Do not publish an Apple/MLX Docker image unless Docker can realistically use
  the native Apple acceleration stack.
- Do not bake model weights into official images.
- Do not make the lightweight server image depend on CUDA, Torch, Diffusers,
  AbstractVoice, or AbstractVision.
- Do not guarantee every local model works in one image; model compatibility
  remains provider/plugin-specific.

## Documentation updates if promoted

Update:

- `docs/server.md`
- `README.md`
- `docs/getting-started.md`
- `docs/prerequisites.md`
- `llms.txt`
- `llms-full.txt`
- release notes

Docs should include a short selector:

```text
Use abstractcore-server when you want a lightweight hosted-provider gateway.
Use abstractcore-server-gpu when you want a CUDA/NVIDIA local inference server.
Use pip install abstractcore[all-apple] directly on macOS for Apple Silicon.
```

## Guidance for future agents

Do not implement this by making the existing `abstractcore-server` image heavy.
Keep the remote gateway and NVIDIA local-inference server as separate deployment
products. Before coding, re-check the current AbstractCore, AbstractVoice, and
AbstractVision extras because plugin dependency surfaces are expected to evolve.
