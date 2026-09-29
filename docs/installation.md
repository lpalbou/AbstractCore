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

## Light: `pip install abstractcore`

The light install runs everything through remote inferencers and servers you already have. It
works on macOS, Linux and Windows with Python 3.9 or newer.

- **Providers**: OpenAI, Anthropic, OpenRouter, Portkey, any OpenAI-compatible `/v1` endpoint,
  LM Studio, Ollama, and a vLLM server. No provider needs another install step; set the API key
  or `base_url` and call `create_llm(...)`.
- **Tools**: the built-in web and system tools (`web_search`, `skim_websearch`, `skim_url`,
  `fetch_url`, ...). The headless-browser probe (`browser_probe`, JavaScript rendering in
  `fetch_url`) needs a browser, which is never part of the light install: add it with
  `pip install playwright`, then `python -m playwright install --only-shell chromium`.
- **Media inputs**: images (Pillow), PDFs (pypdf), Office documents and spreadsheets
  (unstructured, pandas), Glyph visual-text compression, and precise token counting (tiktoken).
- **HTTP server**: the OpenAI-compatible server and console (`abstractcore serve`) and the
  single-model endpoint (`abstractcore-endpoint`).
- **Capability plugins**: AbstractVoice (`llm.voice`, `llm.audio`), AbstractVision (`llm.vision`),
  AbstractMusic (`llm.music`) and Abstract3D (`llm.scene3d`), with their remote backends. Music
  and 3D need Python 3.10 or newer.

For a local model on a machine without a local engine setting (an Intel Mac, Windows), run it in
Ollama or LM Studio and use the light install.

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

For Linux machines with an NVIDIA (CUDA) or AMD (ROCm) GPU, Python 3.10 or newer. Adds, on top
of light:

- **vLLM engine**: run `vllm serve` on the machine and reach it with the `vllm` provider.
- **HuggingFace provider**: transformers and GGUF models through llama.cpp, and local embedding
  models (sentence-transformers) for `EmbeddingManager`.
- **Local voice, image, video and music engines**: AbstractVoice, OmniVoice, AbstractVision and
  AbstractMusic GPU engines.
- **Headless browser**: Playwright for `browser_probe`; download the browser once with
  `python -m playwright install --with-deps chromium`.

The resolver targets x86_64 Linux with glibc 2.35 or newer (for example Ubuntu 22.04 or newer).
For AMD GPUs, install the ROCm builds of PyTorch and vLLM first, following their install guides.

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
