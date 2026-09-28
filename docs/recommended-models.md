# Recommended models

AbstractCore recommends one model for each capability of the framework: text and chat, image
input (vision), speech output, speech input, image generation, video generation and music. The
recommendation depends on the machine: its platform, its GPU, and on Apple silicon its unified
memory. This page lists every recommendation for every kind of machine, explains how each one is
chosen, and shows how to read the same data from the CLI, from Python and as JSON.

Related pages: [Local Models](models.md) (the download catalog, fit verdicts and downloads),
[Centralized Config](centralized-config.md) (capability routes, `apply-recommended`), and
[Local Engines](engines.md) (installing LM Studio, Ollama and MLX).

## Read the recommendations

| What | CLI | Python |
|---|---|---|
| Every machine class, every capability | `abstractcore models recommendations [--json \| --markdown]` | `abstractcore.config.recommendations.recommendation_matrix()` |
| This machine | `abstractcore models recommendations --host [--json]` | `abstractcore.config.recommendations.recommended_models()` |

The command only reads: it evaluates the curated catalog on reference machines and never probes
a model hub or downloads anything. `--host` runs the full host probe of this machine (GPU memory
limit included), so its fit verdicts are this machine's own. `--output PATH` writes the result to
a file instead of standard output.

## How the recommendation is chosen

One table in AbstractCore holds the recommended model per capability,
`abstractcore.config.capability_defaults.RECOMMENDED_MODELS`, and one function answers per
machine, `recommended_models(host)`. Every surface reads them: the fresh-install defaults,
`abstractcore config apply-recommended`, `abstractcore models download --recommended`, the model
catalog's `starter` flags, the Gateway's first-run guide, this page and the AbstractFramework
website.

- **Text** follows `recommended_text_model()`. On Apple silicon it is an MLX build chosen by
  unified memory, and each tier starts where its model fits: below 16 GiB Qwen3 1.7B (8-bit), 16
  to below 32 GiB Qwen3.5 9B, 32 to below 128 GiB Qwen3.8 27B, 128 GiB and above Qwen3.8
  Flash-Next (on a 128 GB Mac after raising the GPU memory limit). Other computers use the LM
  Studio build `qwen/qwen3.5-9b@4bit`, or the same model's Ollama build where LM Studio has no
  build (Intel Macs).
- **Image input** is read by the recommended text model where it accepts images, so `input.image`
  is covered by `input.text` and needs no second model. Every tier from 16 GB up reads images;
  the 8 GB tier (Qwen3 1.7B) does not, because no vision-capable catalog model fits 8 GB.
- **Speech output** is Supertonic 3 on ONNX Runtime, on the processor, on every desktop platform.
- **Speech input** is Whisper base on AbstractVoice's faster-whisper engine (CTranslate2). It uses
  CUDA on an NVIDIA GPU and the processor elsewhere, including Apple silicon.
- **Image generation** is FLUX.2 klein 4B (8-bit) on MLX-Gen, which runs on Apple silicon only.
- **Video generation** is Wan2.2 TI2V 5B (8-bit) on MLX-Gen: one checkpoint for text-to-video
  and image-to-video, on Apple silicon only. Its memory need depends on the output size, not the
  frame count: about 60.5 GiB at AbstractVision's default 1280x704 canvas and 32.7 GiB at 832x480,
  the smallest size it accepts (both measured). The recommendation follows the default canvas;
  where only 832x480 fits (64 GB Macs) the table says so and you set the route yourself.
- **Music** is ACE-Step 1.5 XL turbo on AbstractMusic's `acestep` backend (Diffusers on PyTorch:
  CUDA, Apple MPS in bfloat16, or the processor in float32).

A recommendation must fit the machine. A capability is **not available** on a machine when its
engine has no build for the platform, or, for image, video and music, when the model does not fit
the machine's memory; the entry then says why and what to use instead. Text always has an entry:
each Apple silicon tier starts where its model fits, so the text model always fits (on a 128 GB
Mac, after raising the GPU memory limit).

### Starter set and opt-in recommendations

Text, speech output, image and video form the **starter set**: a fresh install writes them as
capability defaults where the machine can run them, and `apply-recommended` and `models download
--recommended` act on them. Speech input and music are recommendations you apply yourself:

```bash
abstractcore config set-default input.voice --provider faster-whisper --model base
abstractcore config set-default output.music --provider acestep --model ACE-Step/acestep-v15-xl-turbo-diffusers
abstractcore models download diffusers ACE-Step/acestep-v15-xl-turbo-diffusers
```

faster-whisper downloads Whisper base on first use. AbstractMusic loads ACE-Step from the local
Hugging Face cache only, so download it first. The engines install with
`pip install "abstractcore[voice]"` and `pip install "abstractcore[music]"`.

### Fit verdicts on this page

Download sizes and memory needs come from the catalog (see [Local Models](models.md#fit-verdicts)).
Memory needs are estimates from the model files, except where the catalog records a measured
peak (video). On Apple silicon the verdicts assume macOS's default GPU memory limit, 75% of unified
memory; "fits after raising the GPU memory limit" shows the `sysctl` command that makes the model
fit (it asks for your password and lasts until the Mac restarts). The other machine classes are
evaluated on the reference machine each row names. Run `abstractcore models recommendations
--host` for your own machine's verdicts.

Reference machines:

- Apple silicon: every unified memory size Apple ships (8 to 512 GB). Sizes with identical answers
  share one row.
- Linux or Windows with an NVIDIA GPU: x86_64, a GPU with 24 GB of memory, 64 GB of RAM.
- Linux or Windows, processor only: x86_64 (Linux arm64 answers the same), 16 GB of RAM.
- Intel Mac: 16 GB of RAM.

## Recommended models by machine

<!-- BEGIN GENERATED: recommended-models (scripts/update_recommended_models_doc.py) -->

### Text and chat

Route `input.text` (text generation).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | `mlx-community/Qwen3-1.7B-8bit` | MLX (AbstractCore), Apple GPU (Metal) | 1.7 GiB | 3.1 GiB | fits |
| Apple silicon Mac, 16 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 18 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 24 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 32 GB, 36 GB or 48 GB | `mlx-community/Qwen3.8-27B-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 64 GB | `mlx-community/Qwen3.8-27B-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 96 GB | `mlx-community/Qwen3.8-27B-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 128 GB | `mlx-community/Qwen3.8-Flash-Next-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits after raising the GPU memory limit: `sudo sysctl iogpu.wired_limit_mb=117760` |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `mlx-community/Qwen3.8-Flash-Next-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits |
| Linux or Windows with an NVIDIA GPU | `qwen/qwen3.5-9b` (download `qwen/qwen3.5-9b@4bit`) | LM Studio, NVIDIA GPU (CUDA) | unknown | 5.6 GiB | fits |
| Linux or Windows, processor only | `qwen/qwen3.5-9b` (download `qwen/qwen3.5-9b@4bit`) | LM Studio, processor | unknown | 5.6 GiB | fits |
| Intel Mac | `qwen3.5:9b` | Ollama, processor | unknown | 6.0 GiB | fits |

### Image input (vision)

Route `input.image` (image understanding).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | Not available: the recommended text model (mlx-community/Qwen3-1.7B-8bit) does not read images; set input.image to a vision-capable model on another machine or a cloud provider | | | | |
| Apple silicon Mac, 16 GB | the text model (`mlx-community/Qwen3.5-9B-MLX-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 18 GB | the text model (`mlx-community/Qwen3.5-9B-MLX-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 24 GB | the text model (`mlx-community/Qwen3.5-9B-MLX-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 32 GB, 36 GB or 48 GB | the text model (`mlx-community/Qwen3.8-27B-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 64 GB | the text model (`mlx-community/Qwen3.8-27B-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 96 GB | the text model (`mlx-community/Qwen3.8-27B-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 128 GB | the text model (`mlx-community/Qwen3.8-Flash-Next-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits after raising the GPU memory limit: `sudo sysctl iogpu.wired_limit_mb=117760` |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | the text model (`mlx-community/Qwen3.8-Flash-Next-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits |
| Linux or Windows with an NVIDIA GPU | the text model (`qwen/qwen3.5-9b`) | LM Studio, NVIDIA GPU (CUDA) | unknown | 5.6 GiB | fits |
| Linux or Windows, processor only | the text model (`qwen/qwen3.5-9b`) | LM Studio, processor | unknown | 5.6 GiB | fits |
| Intel Mac | the text model (`qwen3.5:9b`) | Ollama, processor | unknown | 6.0 GiB | fits |

### Speech output (text to speech)

Route `output.voice` (text to speech).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 16 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 18 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 24 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 32 GB, 36 GB or 48 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 64 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 96 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 128 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Linux or Windows with an NVIDIA GPU | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Linux or Windows, processor only | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Intel Mac | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |

### Speech input (speech to text)

Route `input.voice` (speech to text).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 16 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 18 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 24 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 32 GB, 36 GB or 48 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 64 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 96 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 128 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Linux or Windows with an NVIDIA GPU | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), NVIDIA GPU (CUDA) | 141 MiB | 653 MiB | fits |
| Linux or Windows, processor only | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Intel Mac | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |

### Image generation

Route `output.image` (text to image).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | Not available: FLUX.2 [klein] 4B (8-bit) needs about 8.5 GiB of memory while it generates (estimated), and this computer can give a model about 4.0 GiB; use a Mac with more unified memory, or a cloud image provider | | | | |
| Apple silicon Mac, 16 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits, tightly |
| Apple silicon Mac, 18 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 24 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 32 GB, 36 GB or 48 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 64 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 96 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 128 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Linux or Windows with an NVIDIA GPU | Not available: MLX-Gen image generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); set output.image to an image engine this host runs: diffusers (install profile gpu), sdcpp (stable-diffusion.cpp, optional extra) or a cloud image provider | | | | |
| Linux or Windows, processor only | Not available: MLX-Gen image generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); set output.image to an image engine this host runs: diffusers (install profile gpu), sdcpp (stable-diffusion.cpp, optional extra) or a cloud image provider | | | | |
| Intel Mac | Not available: MLX-Gen image generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); set output.image to an image engine this host runs: diffusers (install profile gpu), sdcpp (stable-diffusion.cpp, optional extra) or a cloud image provider | | | | |

### Video generation

Route `output.video` (text to video, image to video).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 63.5 GiB of memory while it generates at its default canvas (measured), and this computer can give a model about 4.0 GiB; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 16 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 63.5 GiB of memory while it generates at its default canvas (measured), and this computer can give a model about 10.0 GiB; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 18 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 63.5 GiB of memory while it generates at its default canvas (measured), and this computer can give a model about 11.5 GiB; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 24 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 63.5 GiB of memory while it generates at its default canvas (measured), and this computer can give a model about 16.0 GiB; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 32 GB, 36 GB or 48 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 63.5 GiB of memory while it generates at its default canvas (measured), and this computer can give a model about 22.0 GiB; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 64 GB | At the default canvas: not available, it needs more memory. At 832x480 (121 frames): `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`, set it yourself | MLX-Gen (AbstractVision), Apple GPU (Metal) | 16.9 GiB | 34.3 GiB at 832x480 (measured) | fits at 832x480 |
| Apple silicon Mac, 96 GB | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 16.9 GiB | 63.5 GiB (measured) | fits, tightly |
| Apple silicon Mac, 128 GB | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 16.9 GiB | 63.5 GiB (measured) | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 16.9 GiB | 63.5 GiB (measured) | fits |
| Linux or Windows with an NVIDIA GPU | Not available: MLX-Gen video generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); no other local engine in AbstractFramework generates video today (abstractvision's Diffusers video path is disabled, stable-diffusion.cpp has none); the remaining option is an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Linux or Windows, processor only | Not available: MLX-Gen video generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); no other local engine in AbstractFramework generates video today (abstractvision's Diffusers video path is disabled, stable-diffusion.cpp has none); the remaining option is an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Intel Mac | Not available: MLX-Gen video generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); no other local engine in AbstractFramework generates video today (abstractvision's Diffusers video path is disabled, stable-diffusion.cpp has none); the remaining option is an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |

### Music generation

Route `output.music` (text to music).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | Not available: ACE-Step 1.5 XL turbo (music) needs about 10.9 GiB of memory while it generates (estimated), and this computer can give a model about 4.0 GiB; use a computer with more memory, or a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |
| Apple silicon Mac, 16 GB | Not available: ACE-Step 1.5 XL turbo (music) needs about 10.9 GiB of memory while it generates (estimated), and this computer can give a model about 10.0 GiB; use a computer with more memory, or a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |
| Apple silicon Mac, 18 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits, tightly |
| Apple silicon Mac, 24 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 32 GB, 36 GB or 48 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 64 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 96 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 128 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Linux or Windows with an NVIDIA GPU | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), NVIDIA GPU (CUDA) | 10.3 GiB | 10.9 GiB | fits |
| Linux or Windows, processor only | Not available: ACE-Step 1.5 XL turbo (music) needs about 10.9 GiB of memory while it generates (estimated), and this computer can give a model about 10.0 GiB; use a computer with more memory, or a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |
| Intel Mac | Not available: ACE-Step music generation runs on PyTorch, and current PyTorch builds for macOS require Apple Silicon (arm64); the last Intel-Mac build is 2.2.2; set output.music to a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |

<!-- END GENERATED: recommended-models -->

## The JSON export

`abstractcore models recommendations --json` emits `model_recommendations_v1`:

- `capabilities[]`: `{id, route, label, tasks}` in display order. The ids are `text`, `vision`,
  `speech_output`, `speech_input`, `image`, `video` and `music`.
- `classes[]`: `{id, family, label, reference, platform: {os, arch, accelerator}, entries}`.
  Apple silicon bands also carry `memory_gib` (the sizes the row covers), `memory_gib_min` and
  `memory_gib_below` (the next row's first size, `null` for the last). The other classes carry
  `variants`, the platforms that answer the same as the reference.
- `entries.<capability>`: `capability` (the id), `label`, `route` and `tasks` (the
  capability's `capabilities[]` fields, repeated so an entry reads on its own), `status`
  (`recommended`, `covered` or `unavailable`), `starter`, `provider`, `engine`, `device`, `model`
  (what the route stores), `artifact` and `download_provider` (what `abstractcore models
  download` fetches), `catalog_id`, `display_name`, `download_bytes` (`null` when the catalog has
  no exact size), `memory_need_bytes`, `memory_need_source` (`measured` or `estimated`), `fit`,
  `gpu_limit_command`, `covered_by`, `smaller_canvas` (`null`, or `{canvas, memory_need_bytes,
  memory_need_source, fit}`: the largest measured smaller output size, `WIDTHxHEIGHTxFRAMES`, at
  which an `unavailable` video model still fits), `reason` (for `unavailable`), `warning` (the
  sentence surfaces show for a doubtful fit) and `notes`. The text entry also carries `basis` and
  `tier` from `recommended_text_model()`.

`--host --json` emits the same schema with `host: {os, arch, accelerator, memory_gib,
ceiling_bytes, ceiling_source}` and one `entries` object.
