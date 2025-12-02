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

---

## Implementation Report (Completed 2025-12-01)

**Status**: COMPLETED
**Implementation Time**: ~4 hours
**Tests**: 11 passed (100% success rate)

### What Was Implemented

1. **Download Module** (`abstractcore/download.py`):
   - Created async `download_model()` function with progress reporting
   - Implemented `DownloadProgress` dataclass with status, message, percent, bytes
   - Implemented `DownloadStatus` enum (STARTING, DOWNLOADING, VERIFYING, COMPLETE, ERROR)
   - Provider-specific implementations for Ollama and HuggingFace/MLX
   - ~240 lines of clean, well-documented code

2. **Ollama Download** (`_download_ollama()`):
   - Uses `/api/pull` endpoint with streaming NDJSON
   - Parses progress from Ollama response format
   - Full progress reporting with percent and bytes
   - Error handling for connection failures and server errors

3. **HuggingFace/MLX Download** (`_download_huggingface()`):
   - Uses `huggingface_hub.snapshot_download` via `asyncio.to_thread`
   - Handles gated models with token parameter
   - Error handling for RepositoryNotFoundError and GatedRepoError
   - Same implementation for both HuggingFace and MLX providers

4. **Package Exports** (`abstractcore/__init__.py`):
   - Exported `download_model`, `DownloadProgress`, `DownloadStatus`
   - Available via `from abstractcore import download_model`

5. **Test Suite** (`tests/download/test_model_download.py`):
   - 11 comprehensive tests covering:
     * Ollama small model download
     * Custom base URL configuration
     * Non-existent model error handling
     * HuggingFace small model download
     * MLX provider name (same as HuggingFace)
     * Unsupported providers (OpenAI, Anthropic, LMStudio)
     * DownloadProgress dataclass creation
   - All tests use real implementations (no mocking)
   - 100% passing rate

### Usage Examples

**Ollama Model:**
```python
from abstractcore import download_model

async for progress in download_model("ollama", "llama3:8b"):
    print(f"{progress.status.value}: {progress.message}")
    if progress.percent:
        print(f"  Progress: {progress.percent:.1f}%")
```

**HuggingFace Model:**
```python
async for progress in download_model("huggingface", "meta-llama/Llama-2-7b"):
    print(progress.message)
```

**Gated Model with Token:**
```python
async for progress in download_model(
    "huggingface",
    "meta-llama/Llama-2-7b",
    token="hf_..."
):
    print(progress.message)
```

**MLX Model:**
```python
async for progress in download_model("mlx", "mlx-community/Qwen3-4B-4bit"):
    print(progress.message)
```

### Provider Support Matrix

| Provider | Support | Method | Notes |
|----------|---------|--------|-------|
| **Ollama** | ✅ | `/api/pull` streaming NDJSON | Full progress with percent and bytes |
| **HuggingFace** | ✅ | `huggingface_hub.snapshot_download` | Start and completion messages |
| **MLX** | ✅ | Same as HuggingFace | Uses HF Hub internally |
| **LMStudio** | ❌ | N/A | No download API (CLI/GUI only) |
| **OpenAI** | ❌ | N/A | Cloud-only |
| **Anthropic** | ❌ | N/A | Cloud-only |

### Files Created/Modified

**Created:**
1. `abstractcore/download.py` - Main download module (240 lines)
2. `tests/download/__init__.py` - Test package init
3. `tests/download/test_model_download.py` - Test suite (161 lines, 11 tests)

**Modified:**
1. `abstractcore/__init__.py` - Added exports for download API
2. `llms.txt` - Added Model Downloads feature line
3. `llms-full.txt` - Added comprehensive documentation section with examples

### Test Results

```bash
python -m pytest tests/download/ -v --tb=short

======================== 11 passed in 5.11s =========================

TestOllamaDownload:
  ✅ test_download_small_model
  ✅ test_download_with_custom_base_url
  ✅ test_download_nonexistent_model

TestHuggingFaceDownload:
  ✅ test_download_small_model
  ✅ test_download_with_mlx_provider_name
  ✅ test_download_nonexistent_model

TestUnsupportedProvider:
  ✅ test_unsupported_provider_raises
  ✅ test_lmstudio_unsupported
  ✅ test_anthropic_unsupported

TestDownloadProgress:
  ✅ test_download_progress_creation
  ✅ test_download_progress_minimal
```

### Benefits

1. **Provider-Agnostic**: Single API for all supported providers
2. **Progress Reporting**: Real-time progress updates for UIs
3. **Async-Native**: Natural async generator pattern for streaming
4. **Error Handling**: Clear error messages for all failure modes
5. **Zero Breaking Changes**: New functionality, no API changes
6. **Production Ready**: Tested with real implementations, no mocking
7. **Well Documented**: Examples in llms-full.txt and comprehensive docstrings

### Design Decisions

1. **Async-Only**: Matches streaming nature of progress reporting
2. **Top-Level Function**: Simple import (`from abstractcore import download_model`)
3. **Dataclass Progress**: Structured, typed progress information
4. **Enum Status**: Type-safe status tracking
5. **No Progress Callbacks for HF**: huggingface_hub.snapshot_download doesn't support detailed progress callbacks, so we provide start/complete messages
6. **Optional huggingface_hub**: Import inside function to keep it optional

### Verification

```bash
# Test Ollama download
python -c "
import asyncio
from abstractcore import download_model

async def test():
    async for p in download_model('ollama', 'gemma3:1b'):
        print(f'{p.status.value}: {p.message}')

asyncio.run(test())
"

# Test HuggingFace download
python -c "
import asyncio
from abstractcore import download_model

async def test():
    async for p in download_model('huggingface', 'hf-internal-testing/tiny-random-gpt2'):
        print(p.message)

asyncio.run(test())
"
```

### Conclusion

Successfully implemented async model download API with progress reporting for Ollama, HuggingFace, and MLX providers. All 11 tests passing with real implementations. Feature is production-ready and fulfills all requirements from the original feature request.

