# Feature Request: Model Download API

## Context

Digital Article is a computational notebook application that uses AbstractCore for LLM interactions. When deployed via Docker, users need to download models (Ollama, HuggingFace) through the web UI without leaving the application.

## Problem

Currently, to download a model programmatically, we must implement provider-specific logic in our application:

```python
# Our current implementation in Digital Article
from huggingface_hub import snapshot_download, HfFileSystem
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

# Provider-specific knowledge we shouldn't need to know
async def download_huggingface_model(model: str, token: str = None):
    fs = HfFileSystem(token=token)
    files = fs.ls(model, detail=True)  # Check if exists
    snapshot_download(repo_id=model, token=token)  # Download
```

```python
# For Ollama, different API entirely
async with client.stream("POST", f"{base_url}/api/pull", json={"name": model}):
    # Parse NDJSON progress...
```

This duplicates provider knowledge that AbstractCore already possesses.

## Request

A provider-agnostic API for downloading models with progress reporting.

## Use Case

```python
from abstractcore import download_model

# User clicks "Download Model" in our web UI
# We stream progress back via SSE

async for progress in download_model("huggingface", "meta-llama/Llama-2-7b", token="hf_..."):
    # progress could be a dict or dataclass with: status, percent, message, etc.
    yield f"data: {json.dumps(progress)}\n\n"
```

The application controls *how* to surface progress (SSE, CLI progress bar, logs). AbstractCore handles *what* to do for each provider.

## Why AbstractCore?

1. **AbstractCore already knows providers** - model formats, cache locations, auth mechanisms
2. **Avoids duplication** - next project using AbstractCore won't re-implement this
3. **Consistency** - same API regardless of provider

## Notes

- Ollama has `/api/pull` with streaming NDJSON
- HuggingFace uses `huggingface_hub.snapshot_download` (respects `HF_HOME`)
- LMStudio has no download API (CLI/GUI only) - returning an error or skip is fine
- OpenAI/Anthropic are cloud-only, no download needed

---

*Submitted by: Digital Article project*
*Date: 2025-12-01*

