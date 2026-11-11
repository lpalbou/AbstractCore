# Installation

AbstractCore has three install settings. Pick the one that matches where your models run:

| Setting | Command | What you get |
|---|---|---|
| **Light** (default) | `pip install abstractcore` | Every remote provider, the built-in tools, media inputs, the HTTP server and the capability plugins |
| **Apple** | `pip install "abstractcore[apple]"` | Light + every local engine Apple silicon can run |
| **GPU** | `pip install "abstractcore[gpu]"` | Light + every local engine NVIDIA and AMD machines can run |

`apple` and `gpu` each include everything in light. There is nothing else to add or combine:
install once, then configure providers and models (see [Getting Started](getting-started.md) and
[Prerequisites & Setup](prerequisites.md)).

## One-line install

No admin rights and no Python needed: the installer gets [uv](https://docs.astral.sh/uv) when it is
missing, Python 3.12 through uv, and installs AbstractCore alone (no gateway, no services) in its
own environment, with its commands in `~/.local/bin`: `abstractcore` (configuration, `models`,
`engines`, `serve`), `abstractcore-config`, `abstractcore-chat`, `abstractcore-endpoint` and the
apps `summarizer`, `extractor`, `judge`, `intent` and `deepsearch`.

```bash
# macOS / Linux
curl -LsSf https://abstractframework.ai/core/install.sh | sh
```

```powershell
# Windows 10 / 11
powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/core/install.ps1 | iex"
```

```bash
# With pip, in your own Python environment
pip install abstractcore              # light
pip install "abstractcore[apple]"     # Apple Silicon Mac
pip install "abstractcore[gpu]"       # NVIDIA or AMD GPU
```

It picks the setting from the machine (`--profile` chooses another one):

| Machine | Profile | What it installs |
|---|---|---|
| Apple Silicon Mac, macOS 14 or later | `apple` | `abstractcore[apple]` (below), with llama.cpp's prebuilt Metal build |
| NVIDIA card (`nvidia-smi -L` works) or, on Linux, ROCm (`rocminfo`) | `gpu` | `abstractcore[gpu]` (below); with an NVIDIA card, llama.cpp's prebuilt CUDA build (below); on Windows, PyTorch's CUDA build for the driver (`--torch-backend cu130` or `cu126`) |
| Anything else (an Intel Mac, an older macOS, no GPU) | `light` | `abstractcore` with no extra: every remote provider, the built-in tools, media inputs and the HTTP server |

`--profile gpu` on a machine without a card installs the same stack, whose engines then run on the
processor. Every step prints its command, and so does the summary: the version, the profile, the
llama.cpp result, where the commands are and what to try first:

```bash
abstractcore --help
abstractcore config defaults   # the capability routes this install resolves to
```

| macOS / Linux | Windows | |
|---|---|---|
| `--print` (or `--dry-run`, `-n`) | `-Print` (or `-DryRun`) | show the plan and every command; change nothing |
| `--profile auto\|light\|apple\|gpu` | `-Profile auto\|light\|gpu` | the profile (default `auto`; `AC_PROFILE`) |
| `--pin VERSION\|latest` | `-Pin VERSION\|latest` | the version: by default the release the script ships with, `latest` = the newest on PyPI (`AC_PIN`) |
| `--from PATH` | `-From PATH` | install from a checkout, sdist or wheel, for testing (`AC_FROM`) |
| `--no-modify-path` | `-NoModifyPath` | do not run `uv tool update-shell` |
| `-v`, `--verbose` | `-Verbose` | show the full output of every command |

Options go after `sh -s --` (`curl -LsSf https://abstractframework.ai/core/install.sh | sh -s -- --print`)
or into a script block on Windows
(`& ([scriptblock]::Create((irm https://abstractframework.ai/core/install.ps1))) -Print`).

**Upgrade**: run the same line again. It finds the install, says
`AbstractCore 2.34.0 found: upgrading to AbstractCore 2.35.0` (or `already up to date`), keeps the
profile unless `--profile` says otherwise, ends with `abstractcore 2.34.0 -> 2.35.0`, and
reinstalls in place when the command no longer starts. Remove it with
`uv tool uninstall abstractcore`. AbstractFramework's installer already includes AbstractCore and
its commands: on such a machine this script stops and says so, and nothing is changed. A standalone
AbstractCore beside the framework's gateway tool is not supported (there is no `--force`: both
would own the same commands); upgrade AbstractFramework instead.

**Prebuilt wheels only**: the installer never starts a compiler (no Xcode tools, no build
toolchain). Three packages of the `apple` and `gpu` trees publish no usable wheel on PyPI, so on
macOS and Linux:

- **llama.cpp** (GGUF models; `llama-cpp-python`, a source build on PyPI) comes from upstream's
  prebuilt wheels instead: the Metal build on Apple Silicon, the CPU build on Linux (x86_64 or
  aarch64, glibc or musl). When it does not install, it is left out and the summary says so.
  With an NVIDIA card (`gpu` profile) the installer then swaps in llama.cpp's CUDA build matching
  PyTorch's CUDA (`cu130` for CUDA 13 with driver 580 or newer, `cu125` for CUDA 12 with driver 525
  or newer, x86_64 only; on Windows `cu130`, then `cu132`, or `cu125`) and keeps it only when it
  loads and `llama_supports_gpu_offload()` is True; otherwise, and with ROCm (no prebuilt ROCm
  build) or on aarch64, the CPU build stays and the summary says why. After
  `uv tool upgrade --reinstall` (or when the first Windows build did not install), run the
  installer again to get the CUDA build back.
  **Plain `pip install "abstractcore[gpu]"` on Linux** gets the same result without the installer
  (2.35.1): pip can only resolve the processor build, so the first time AbstractCore is about to load
  a GGUF model on an x86_64 host where PyTorch is a CUDA build and `nvidia-smi` answers with a driver
  new enough (580 for CUDA 13, 525 for CUDA 12), it runs that swap itself, once per environment
  (`pip` or `uv pip`, same index, same pinned version), prints one line on the console before and one after
  (the failure reason too, whatever the log level), and records
  the outcome next to the package (`llama_cpp/.abstractcore-cuda-swap.json`). A host set up by the
  installer already has the CUDA build and nothing runs. If the swap cannot happen (offline,
  read-only site-packages, no driver), GGUF models stay on the processor and the provider's runtime
  info (`llama_cuda_swap`) and a warning say why; `ABSTRACTCORE_LLAMA_CUDA_SWAP=0` keeps the
  processor build on purpose.
- **Echo cancellation** (`aec-audio-processing`) and **stable-diffusion.cpp**
  (`stable-diffusion-cpp-python`), both optional, are left out by the overrides file and the
  summary says so (Diffusers / MLX-Gen cover image generation; with a C compiler you can add them
  to the tool's environment yourself).

On Windows x64 every package of these settings has a wheel, so nothing is left out beyond what
[GPU on Windows](#gpu-on-windows) describes. Windows on ARM is not supported by the installer:
tiktoken, lxml and pandas publish no Windows ARM64 wheels for the versions AbstractCore needs.

## Light: `pip install abstractcore`

The light install runs everything through remote inferencers and servers you already have. It
works on macOS, Linux and Windows with Python 3.9 or newer.

- **Providers**: OpenAI, Anthropic, OpenRouter, Portkey, any OpenAI-compatible `/v1` endpoint,
  LM Studio, Ollama, and a vLLM server. No provider needs another install step; set the API key
  or `base_url` and call `create_llm(...)`.
- **Tools**: the built-in web and system tools (`web_search`, `skim_websearch`, `skim_url`,
  `fetch_url`, ...). The headless-browser probe (`browser_probe`, JavaScript rendering in
  `fetch_url`) needs a browser, which is never part of the light install: `abstractcore[apple]`
  and `abstractcore[gpu]` carry Playwright; then download the browser once with
  `python -m playwright install --only-shell chromium`.
- **Media inputs**: images (Pillow), PDFs (pypdf), Office documents and spreadsheets
  (unstructured, pandas), Glyph visual-text compression, and precise token counting (tiktoken).
- **HTTP server**: the OpenAI-compatible server and console (`abstractcore serve`) and the
  single-model endpoint (`abstractcore-endpoint`).
- **Capability plugins**: AbstractVoice (`llm.voice`, `llm.audio`), AbstractVision (`llm.vision`),
  AbstractMusic (`llm.music`) and Abstract3D (`llm.scene3d`), with their remote backends. Music
  and 3D need Python 3.10 or newer.

For a local model on a machine without a local engine setting (an Intel Mac, Windows on ARM), run
it in Ollama or LM Studio and use the light install.

## Apple: `pip install "abstractcore[apple]"`

For Apple silicon Macs (macOS 14 or newer, Python 3.10 or newer; 3.11 or newer enables every
local voice engine). Adds, on top of light:

- **MLX provider**: text and image input for MLX checkpoints (mlx, mlx-lm, mlx-vlm), with
  native structured output (outlines).
- **HuggingFace provider**: transformers and GGUF models through llama.cpp, and local embedding
  models (sentence-transformers) for `EmbeddingManager`.
- **Local voice**: speech output and speech input engines (AbstractVoice's Apple engines,
  OmniVoice).
- **Local image and video**: AbstractVision's Apple engines (MLX-Gen, Diffusers,
  stable-diffusion.cpp).
- **Local music**: AbstractMusic's Apple engines.
- **Headless browser**: Playwright for `browser_probe`; download the browser once with
  `python -m playwright install --only-shell chromium`.

## GPU: `pip install "abstractcore[gpu]"`

For Linux machines with an NVIDIA (CUDA) or AMD (ROCm) GPU, and Windows x86_64 machines with an
NVIDIA GPU, Python 3.10 or newer. Adds, on top of light:

- **vLLM engine** (Linux): run `vllm serve` on the machine and reach it with the `vllm` provider.
- **HuggingFace provider**: transformers and GGUF models through llama.cpp, and local embedding
  models (sentence-transformers) for `EmbeddingManager`.
- **Local voice, image, video and music engines**: AbstractVoice, OmniVoice, AbstractVision and
  AbstractMusic GPU engines.
- **Headless browser**: Playwright for `browser_probe`; download the browser once with
  `python -m playwright install --with-deps chromium`.

The resolver targets x86_64 Linux with glibc 2.35 or newer (for example Ubuntu 22.04 or newer).
For AMD GPUs, install the ROCm builds of PyTorch and vLLM first, following their install guides.

### GPU on Windows

`abstractcore[gpu]` installs on Windows x86_64 with prebuilt wheels only. Three engines differ
from Linux:

- **vLLM** is not part of it: upstream supports Linux only (use WSL for vLLM).
- **llama.cpp** (GGUF models): on Windows the AbstractFramework installer adds llama.cpp's
  prebuilt GPU build (CUDA, else Vulkan, else CPU); with plain pip, llama.cpp is not part of
  abstractcore[gpu] on Windows, because PyPI ships it as a source build only. AbstractCore puts
  PyTorch's `torch\lib` folder (which carries cuBLAS and the CUDA runtime) on the DLL search
  before it loads llama.cpp, so the CUDA build can find those libraries without the CUDA toolkit
  (not yet validated on Windows hardware).
- **stable-diffusion.cpp** is not part of it (source build only, which needs MSVC); Diffusers on
  PyTorch covers image generation.

Plain pip installs PyTorch's CPU build on Windows. The AbstractFramework installer
(`install.ps1`, gpu setting) picks PyTorch's CUDA build for the NVIDIA driver it finds
(CUDA 13 for driver 580 or newer, CUDA 12 for driver 525 or newer) and reports what runs on the
GPU at the end.

## Upgrading

```bash
pip install -U abstractcore            # light
pip install -U "abstractcore[apple]"   # Apple silicon
pip install -U "abstractcore[gpu]"     # NVIDIA / AMD
```

In zsh, keep the quotes around `abstractcore[apple]` and `abstractcore[gpu]`.

## Optional PDF extraction with PyMuPDF

The light install extracts PDFs with pypdf (permissive licence). If you have reviewed
PyMuPDF-family licensing (AGPL or a commercial licence) and prefer its layout-aware extraction,
install it directly with `pip install pymupdf4llm pymupdf-layout` and select it with the
`pdf_backend="pymupdf4llm"` option (see [Media Handling](media-handling-system.md)).

## Deprecated aliases

Older AbstractCore releases had one extra per provider or feature. Those names still install, so
existing requirements and dependent packages keep working, but they are deprecated aliases kept
for compatibility: use light (no extra), `apple` or `gpu`.

| Alias | Installs |
|---|---|
| `remote`, `openai`, `anthropic`, `openrouter`, `portkey`, `openai-compatible`, `ollama`, `lmstudio` | light |
| `tools`, `tool`, `tokens`, `media`, `compression`, `server` | light |
| `voice`, `audio`, `vision`, `music`, `scene3d`, `3d` | light |
| `all-apple`, `mlx`, `mlx-vision` | apple |
| `all-gpu`, `vllm` | gpu |
| `huggingface`, `embeddings`, `browser`, `vision-diffusers`, `vision-sdcpp`, `vision-local`, `pdf-pymupdf-commercial`, `all`, `all-non-mlx`, `mlx-bench`, `full-dev` | unchanged: the same packages as before, on top of light |

Contributor tooling (`dev`, `test`, `docs`) is described in [Contributing](../CONTRIBUTING.md); it
is not an install setting.

## Related docs

- [Getting Started](getting-started.md): first calls after installing.
- [Prerequisites & Setup](prerequisites.md): API keys, local servers and provider configuration.
- [Troubleshooting](troubleshooting.md): install and import errors.
