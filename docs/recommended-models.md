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
  unified memory: below 24 GiB Qwen3.5 9B (4-bit; it fits from 16 GB, and on an 8 GB Mac it is
  tight: it runs with a small context; close other apps first), 24 to below 128 GiB Qwen3.8 27B
  (4-bit; on a 24 GB Mac it runs with a small context by default, and with about 30k tokens after
  `sudo sysctl iogpu.wired_limit_mb=20480`, measured), 128 GiB
  and above Qwen3.8 Flash-Next (on a 128 GB Mac after raising the GPU memory limit). Other
  computers use the LM Studio build `qwen/qwen3.5-9b@q4_k_m`, or the same model's Ollama build where
  LM Studio has no build (Intel Macs).
- **Image input** is read by the recommended text model where it accepts images, so `input.image`
  is covered by `input.text` and needs no second model. Every recommended text model reads
  images; where it does not (or no text engine runs), `input.image` is reported unavailable with
  the reason.
- **Speech output** is Supertonic 3 on ONNX Runtime, on the processor, on every desktop platform.
- **Speech input** is Whisper base on AbstractVoice's faster-whisper engine (CTranslate2). It uses
  CUDA on an NVIDIA GPU and the processor elsewhere, including Apple silicon.
- **Image generation** is FLUX.2 klein 4B. On Apple silicon it is the 8-bit build on MLX-Gen
  (Apple silicon only). On an NVIDIA GPU it is the Diffusers repo `black-forest-labs/FLUX.2-klein-4B`
  on AbstractVision's `diffusers` backend (CUDA, float16). Its 14.9 GiB of float16 weights do not
  fit a 16 GB card whole, so AbstractVision turns on model CPU offload by itself there: measured on
  a 16 GB Quadro RTX 5000, 768x768 in about 17 s with a GPU peak of about 8.3 GiB, while about
  15 GiB of system RAM holds the idle weights. Processor-only computers get no image
  recommendation.
- **Video generation** is Wan2.2 TI2V 5B (8-bit) on MLX-Gen: one checkpoint for text-to-video
  and image-to-video, on Apple silicon only. The memory figures are AbstractVision/mlx-gen's,
  measured at 1280x704x121 (about 60.5 GiB) and at 832x480x121 (32.7 GiB), with the text encoder
  and VAE kept in memory. They are this engine's figures, not the model's own requirement:
  runtimes that offload those parts need far less (Wan 2.2 runs TI2V-5B at 720p on one 24 GB GPU
  with offloading). The recommendation follows AbstractVision's default canvas; where only
  832x480 fits (64 GB Macs) the table says so and you set the route yourself.
- **Music** is ACE-Step 1.5 XL turbo on AbstractMusic's `acestep` backend (Diffusers on PyTorch:
  CUDA, Apple MPS in bfloat16, or the processor in float32).

A capability is **not available** on a machine when its engine has no build for the platform, or,
for image, video and music, when the model does not fit the machine's memory; the entry then says
why and what to use instead. Text always has an entry: a tier whose estimate doubts it (8 GB) is
still the tier, with its warning.

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
Hugging Face cache only, so download it first. Both engines are local: they install with
`pip install "abstractcore[apple]"` (Apple silicon) or `pip install "abstractcore[gpu]"` (NVIDIA / AMD).

### Fit verdicts on this page

Download sizes and memory needs come from the catalog (see [Local Models](models.md#fit-verdicts)).
Memory needs are estimates from the model files, except where the catalog records a peak measured
with AbstractVision/mlx-gen (video). On Apple silicon the verdicts assume macOS's default GPU
memory limit of 75% of unified memory, AbstractCore's fallback when it cannot read your Mac's own
(see [GPU memory on Apple silicon](#gpu-memory-on-apple-silicon)). "fits after raising the GPU
memory limit" shows the `sysctl` command that makes the model fit, and "tight: runs with a small
context by default" shows the command that gives it more context (both ask for your password and
last until the Mac restarts). No table states an estimated context size; the one token count shown
was measured. The other machine classes are
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
| Apple silicon Mac, 8 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | tight: runs with a small context by default |
| Apple silicon Mac, 16 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 18 GB | `mlx-community/Qwen3.5-9B-MLX-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 24 GB | `mlx-community/Qwen3.8-27B-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | tight: runs with a small context by default; about 30k tokens after `sudo sysctl iogpu.wired_limit_mb=20480` (measured on a 24 GB Mac mini) |
| Apple silicon Mac, 32 GB, 36 GB, 48 GB, 64 GB or 96 GB | `mlx-community/Qwen3.8-27B-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 128 GB | `mlx-community/Qwen3.8-Flash-Next-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits after raising the GPU memory limit: `sudo sysctl iogpu.wired_limit_mb=114688` |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `mlx-community/Qwen3.8-Flash-Next-4bit` | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits |
| Linux or Windows with an NVIDIA GPU | `qwen/qwen3.5-9b` (download `qwen/qwen3.5-9b@q4_k_m`) | LM Studio, NVIDIA GPU (CUDA) | unknown | 6.0 GiB | fits |
| Linux or Windows, processor only | `qwen/qwen3.5-9b` (download `qwen/qwen3.5-9b@q4_k_m`) | LM Studio, processor | unknown | 6.0 GiB | fits |
| Intel Mac | `qwen3.5:9b` | Ollama, processor | unknown | 6.0 GiB | fits |

### Image input (vision)

Route `input.image` (image understanding).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | the text model (`mlx-community/Qwen3.5-9B-MLX-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | tight: runs with a small context by default |
| Apple silicon Mac, 16 GB | the text model (`mlx-community/Qwen3.5-9B-MLX-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 18 GB | the text model (`mlx-community/Qwen3.5-9B-MLX-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 5.7 GiB | 6.4 GiB | fits |
| Apple silicon Mac, 24 GB | the text model (`mlx-community/Qwen3.8-27B-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | tight: runs with a small context by default; about 30k tokens after `sudo sysctl iogpu.wired_limit_mb=20480` (measured on a 24 GB Mac mini) |
| Apple silicon Mac, 32 GB, 36 GB, 48 GB, 64 GB or 96 GB | the text model (`mlx-community/Qwen3.8-27B-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 15.2 GiB | 16.4 GiB | fits |
| Apple silicon Mac, 128 GB | the text model (`mlx-community/Qwen3.8-Flash-Next-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits after raising the GPU memory limit: `sudo sysctl iogpu.wired_limit_mb=114688` |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | the text model (`mlx-community/Qwen3.8-Flash-Next-4bit`) | MLX (AbstractCore), Apple GPU (Metal) | 103.9 GiB | 109.2 GiB | fits |
| Linux or Windows with an NVIDIA GPU | the text model (`qwen/qwen3.5-9b`) | LM Studio, NVIDIA GPU (CUDA) | unknown | 6.0 GiB | fits |
| Linux or Windows, processor only | the text model (`qwen/qwen3.5-9b`) | LM Studio, processor | unknown | 6.0 GiB | fits |
| Intel Mac | the text model (`qwen3.5:9b`) | Ollama, processor | unknown | 6.0 GiB | fits |

### Speech output (text to speech)

Route `output.voice` (text to speech).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 16 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 18 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 24 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
| Apple silicon Mac, 32 GB, 36 GB, 48 GB, 64 GB or 96 GB | `supertonic-3` | Supertonic on ONNX Runtime (AbstractVoice), processor | 383 MiB | 895 MiB | fits |
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
| Apple silicon Mac, 32 GB, 36 GB, 48 GB, 64 GB or 96 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 128 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Linux or Windows with an NVIDIA GPU | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), NVIDIA GPU (CUDA) | 141 MiB | 653 MiB | fits |
| Linux or Windows, processor only | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |
| Intel Mac | `base` (download `Systran/faster-whisper-base`) | faster-whisper on CTranslate2 (AbstractVoice), processor | 141 MiB | 653 MiB | fits |

### Image generation

Route `output.image` (text to image).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | Not available: FLUX.2 [klein] 4B needs about 8.5 GiB of memory while it generates (estimated), and macOS's GPU memory limit on this Mac is about 6.0 GiB, about 4.0 GiB of it left for a model after working buffers; use a Mac with more unified memory, or a cloud image provider | | | | |
| Apple silicon Mac, 16 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits, tightly |
| Apple silicon Mac, 18 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 24 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 32 GB, 36 GB, 48 GB, 64 GB or 96 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 128 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `AbstractFramework/flux.2-klein-4b-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 8.0 GiB | 8.5 GiB | fits |
| Linux or Windows with an NVIDIA GPU | `black-forest-labs/FLUX.2-klein-4B` | Diffusers on PyTorch (AbstractVision), NVIDIA GPU (CUDA) | 22.1 GiB | 8.3 GiB (measured with AbstractVision/diffusers with model CPU offload) | fits |
| Linux or Windows, processor only | Not available: MLX-Gen image generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); set output.image to a local image engine: diffusers, included with abstractcore[gpu] on Linux and Windows, or sdcpp (stable-diffusion.cpp), included with abstractcore[gpu] on Linux (on an Intel Mac or Windows on ARM they are not available with AbstractCore's install settings), or to a cloud image provider | | | | |
| Intel Mac | Not available: MLX-Gen image generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); set output.image to a local image engine: diffusers, included with abstractcore[gpu] on Linux and Windows, or sdcpp (stable-diffusion.cpp), included with abstractcore[gpu] on Linux (on an Intel Mac or Windows on ARM they are not available with AbstractCore's install settings), or to a cloud image provider | | | | |

- Linux or Windows with an NVIDIA GPU: AbstractVision runs it in float16 and turns on model CPU offload by itself when its 14.9 GiB of weights do not fit the GPU's free memory (a 16 GB card): the GPU then peaks at about 8.3 GiB (measured) and about 15 GiB of system RAM holds the idle weights. A GPU with room loads it whole.

### Video generation

Route `output.video` (text to video, image to video).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 16.6 GiB of memory while it generates (measured with AbstractVision/mlx-gen at 832x480x121 for image-to-video, its larger task, text-to-video needing 16.3 GiB; the engine keeps the text encoder and VAE in memory; this engine's figure, not the model's minimum), and macOS's GPU memory limit on this Mac is about 6.0 GiB, about 4.0 GiB of it left for a model after working buffers; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 16 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 16.6 GiB of memory while it generates (measured with AbstractVision/mlx-gen at 832x480x121 for image-to-video, its larger task, text-to-video needing 16.3 GiB; the engine keeps the text encoder and VAE in memory; this engine's figure, not the model's minimum), and macOS's GPU memory limit on this Mac is about 12.0 GiB, about 10.0 GiB of it left for a model after working buffers; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 18 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 16.6 GiB of memory while it generates (measured with AbstractVision/mlx-gen at 832x480x121 for image-to-video, its larger task, text-to-video needing 16.3 GiB; the engine keeps the text encoder and VAE in memory; this engine's figure, not the model's minimum), and macOS's GPU memory limit on this Mac is about 13.5 GiB, about 11.5 GiB of it left for a model after working buffers; use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Apple silicon Mac, 24 GB | Not available: Wan2.2 TI2V 5B (text/image to video) needs about 16.6 GiB of memory while it generates (measured with AbstractVision/mlx-gen at 832x480x121 for image-to-video, its larger task, text-to-video needing 16.3 GiB; the engine keeps the text encoder and VAE in memory; this engine's figure, not the model's minimum), and macOS's GPU memory limit on this Mac is about 18.0 GiB, about 16.0 GiB of it left for a model after working buffers. It fits once macOS lets the GPU use 20 GiB: run `sudo sysctl iogpu.wired_limit_mb=20480` in a terminal (asks for your password; lasts until the Mac restarts), then load it. | | | | |
| Apple silicon Mac, 32 GB, 36 GB, 48 GB, 64 GB or 96 GB | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 16.9 GiB | 16.6 GiB (measured with AbstractVision/mlx-gen) | fits |
| Apple silicon Mac, 128 GB | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 16.9 GiB | 16.6 GiB (measured with AbstractVision/mlx-gen) | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` | MLX-Gen (AbstractVision), Apple GPU (Metal) | 16.9 GiB | 16.6 GiB (measured with AbstractVision/mlx-gen) | fits |
| Linux or Windows with an NVIDIA GPU | Not available: MLX-Gen video generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); no other local engine in AbstractFramework generates video today (abstractvision's Diffusers video path is disabled, stable-diffusion.cpp has none); the remaining option is an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Linux or Windows, processor only | Not available: MLX-Gen video generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); no other local engine in AbstractFramework generates video today (abstractvision's Diffusers video path is disabled, stable-diffusion.cpp has none); the remaining option is an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |
| Intel Mac | Not available: MLX-Gen video generation needs MLX, and MLX runs only on Apple Silicon Macs (macOS, arm64); no other local engine in AbstractFramework generates video today (abstractvision's Diffusers video path is disabled, stable-diffusion.cpp has none); the remaining option is an OpenAI-compatible video endpoint (abstractvision openai-compatible backend) | | | | |

### Music generation

Route `output.music` (text to music).

| Machine | Recommended model | Engine, device | Download | Memory need | Fit |
|---|---|---|---|---|---|
| Apple silicon Mac, 8 GB | Not available: ACE-Step 1.5 XL turbo (music) needs about 10.9 GiB of memory while it generates (estimated), and macOS's GPU memory limit on this Mac is about 6.0 GiB, about 4.0 GiB of it left for a model after working buffers; use a computer with more memory, or a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |
| Apple silicon Mac, 16 GB | Not available: ACE-Step 1.5 XL turbo (music) needs about 10.9 GiB of memory while it generates (estimated), and macOS's GPU memory limit on this Mac is about 12.0 GiB, about 10.0 GiB of it left for a model after working buffers; use a computer with more memory, or a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |
| Apple silicon Mac, 18 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits, tightly |
| Apple silicon Mac, 24 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 32 GB, 36 GB, 48 GB, 64 GB or 96 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 128 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Apple silicon Mac, 192 GB, 256 GB or 512 GB | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), Apple GPU (MPS, bfloat16) | 10.3 GiB | 10.9 GiB | fits |
| Linux or Windows with an NVIDIA GPU | `ACE-Step/acestep-v15-xl-turbo-diffusers` | ACE-Step on Diffusers and PyTorch (AbstractMusic), NVIDIA GPU (CUDA) | 10.3 GiB | 10.9 GiB | fits |
| Linux or Windows, processor only | Not available: ACE-Step 1.5 XL turbo (music) needs about 10.9 GiB of memory while it generates (estimated), and this computer can give a model about 10.0 GiB; use a computer with more memory, or a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |
| Intel Mac | Not available: ACE-Step music generation runs on PyTorch, and current PyTorch builds for macOS require Apple Silicon (arm64); the last Intel-Mac build is 2.2.2; set output.music to a cloud music backend (acemusic or elevenlabs-music, with its API key) | | | | |

<!-- END GENERATED: recommended-models -->

## GPU memory on Apple silicon

macOS lets the GPU use only part of unified memory. On a 24 GB Mac mini that limit was measured at
17.8 GB (16.6 GiB, about 69% of its memory). AbstractCore reads the real limit on your Mac
(`abstractcore models recommendations --host`); when it cannot, and in the tables above, it
assumes 75% of unified memory.

A model runs when its weights fit under that limit; what is left holds the context (KV cache) and
the working memory MLX needs beside it (about 2 GiB). On a 24 GB Mac, Qwen3.8 27B 4-bit (about
16.2 GB of weights) therefore runs out of the box with a small context. Raising the limit gives it
more:

```bash
sudo sysctl iogpu.wired_limit_mb=20480
```

That value is safe on a 24 GB Mac and gives about 30k tokens of context (measured). Do not go
beyond 21504 MB on a 24 GB Mac: apps may crash. The one value AbstractCore suggests, whenever a
model needs more GPU memory or more context, leaves macOS max(4 GiB, 12.5% of RAM): 20480 MB on a
24 GB Mac, 28672 MB on 32 GB, 114688 MB on 128 GB. AbstractCore only prints the command; it never
runs `sudo`. The setting lasts until the Mac restarts, and
`sudo sysctl iogpu.wired_limit_mb=0` returns to the default.

### Keeping the limit across restarts

To keep it, an administrator can add a LaunchDaemon that runs the command at boot. AbstractCore
never installs this; it is your machine's configuration. Save as
`/Library/LaunchDaemons/local.iogpu.wired-limit.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.iogpu.wired-limit</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/sbin/sysctl</string>
    <string>iogpu.wired_limit_mb=20480</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
```

Then load it:

```bash
sudo chown root:wheel /Library/LaunchDaemons/local.iogpu.wired-limit.plist
sudo chmod 644 /Library/LaunchDaemons/local.iogpu.wired-limit.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/local.iogpu.wired-limit.plist
```

Use 20480 on a 24 GB Mac. Do not use 21504 unless you know what you're doing. To remove it:

```bash
sudo launchctl bootout system /Library/LaunchDaemons/local.iogpu.wired-limit.plist
sudo rm /Library/LaunchDaemons/local.iogpu.wired-limit.plist
sudo sysctl iogpu.wired_limit_mb=0
```

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
  `gpu_limit_command` (for `needs_gpu_limit`, and for a text fit that is `tight` on Apple
  silicon: the command for more context), `context` (`null`, or `{small, measured}` for such a
  text fit: whether it runs only with a small context by default, and the measured context after
  the command where one was measured, else `null`; no estimated token count), `covered_by`, `smaller_canvas` (`null`, or
  `{canvas, memory_need_bytes, memory_need_source, fit}`: the largest measured smaller output
  size, `WIDTHxHEIGHTxFRAMES`, at which an `unavailable` video model still fits), `reason` (for
  `unavailable`), `warning` (the sentence surfaces show for a doubtful fit) and `notes`. The text
  entry also carries `basis` and `tier` from `recommended_text_model()`.

`--host --json` emits the same schema with `host: {os, arch, accelerator, memory_gib,
ceiling_bytes, ceiling_source}` and one `entries` object.
