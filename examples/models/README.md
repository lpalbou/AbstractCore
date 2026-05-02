# Model Discovery / Selection Examples

## What This Folder Teaches

How to answer practical questions like:
- “What models are available through my current backend?”
- “Which of these are embeddings vs text-generation?”
- “How do I pick a reasonable default model for my task?”

## Prereqs

- `model_filtering_demo.py` expects an AbstractCore server running (see `docs/server.md`).

## Downloading Your First Local Model (HF / GGUF / MLX)

AbstractCore is offline-first by default: local providers do not auto-download weights.

To get a model onto your machine, download it once (then `create_llm(...)` will load from disk):

### Option A: Hugging Face CLI (recommended)

Install the CLI if you don't already have it:
```bash
pip install -U huggingface_hub
```

Then download a model into your Hugging Face cache:
```bash
# Transformers (text generation)
huggingface-cli download microsoft/phi-2

# GGUF (llama.cpp backend via HuggingFaceProvider)
huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF --include "*.gguf"

# MLX (Apple Silicon, MLXProvider)
huggingface-cli download mlx-community/Qwen3-4B-Instruct-2507-4bit
```

### Option B: Python (no CLI)

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download("microsoft/phi-2")  # downloads into the HF cache
print("done")
PY
```

### Loading After Download

```python
from abstractcore import create_llm

llm = create_llm("huggingface", model="microsoft/phi-2")  # transformers
llm = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")  # gguf
llm = create_llm("mlx", model="mlx-community/Qwen3-4B-Instruct-2507-4bit")  # mlx
```

## Key AbstractCore Concepts

- Server model discovery: `/v1/models` (OpenAI-style) lists models, with optional filtering.
- Model selection is deliberately *separate* from calling: you can discover/choose first, then `create_llm(...)`.
- Embeddings vs generation: AbstractCore treats these as different capabilities.

## How The Examples Work

- `model_filtering_demo.py` shows the server contract: it calls `/v1/models` and filters by provider/type.
- The other scripts are “selection helpers”: they demonstrate how you might curate a set of models for a product (defaults + fallbacks).

## Scripts

- `model_filtering_demo.py`
  - Demonstrates: querying `/v1/models` and filtering with `provider=` and `type=`.
  - Takeaway: model discovery can be an API step in your app (not a hardcoded list).

- `favored_models_demo.py`
  - Demonstrates: listing the “favored” embedding aliases and their configs; quick smoke tests per category.
  - Takeaway: wrap raw HF IDs behind stable aliases so apps can upgrade models without code churn.

- `real_models_showcase.py` (dev/showcase oriented)
  - Demonstrates: larger “real model” runs to illustrate multilingual + matryoshka + performance tradeoffs.
  - Takeaway: use this as an exploratory script; for production pick a subset and pin versions.

## Key Takeaways

- Treat “model selection” as a first-class concern: make it visible/configurable in your app.
- Prefer stable aliases (or your own model registry) over hardcoding raw model IDs everywhere.
