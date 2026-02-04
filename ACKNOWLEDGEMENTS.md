# Acknowledgements

AbstractCore is possible thanks to the work of many open-source maintainers and research communities. We’re grateful to everyone who builds and maintains the ecosystem this project depends on.

This list is not exhaustive. The source of truth for install-time dependencies is `pyproject.toml`.

## Core dependencies (default install)

- **Python** and the Python packaging ecosystem
- **Pydantic** (data validation / typed schemas)
- **httpx** (HTTP client)

## Optional dependencies (installed via extras)

Provider SDKs and runtimes:
- **openai** (OpenAI SDK)
- **anthropic** (Anthropic SDK)
- **transformers**, **torch**, **torchvision**, **torchaudio** (HuggingFace runtime)
- **mlx**, **mlx-lm** (Apple Silicon local inference)
- **llama-cpp-python** (GGUF inference)
- **vllm** (GPU inference server integration)
- **outlines** (optional constrained decoding / structured output support for some backends)

Feature extras:
- **sentence-transformers** and **numpy** (local embeddings)
- **tiktoken** (optional precise token counting)
- **Pillow**, **pymupdf4llm**, **pymupdf-layout**, **unstructured**, **pandas** (media/PDF/Office processing)
- **fastapi**, **uvicorn**, **python-multipart**, **sse-starlette** (optional HTTP server)
- **abstractvision** (optional vision generation delegated to AbstractVision)

## Tooling (development)

- **pytest** (test runner)
- **ruff** / **black** (linting and formatting)

## Ecosystem and communities

We’re also grateful to the broader communities around local inference (Ollama, LM Studio, llama.cpp, MLX) and open model development, which make “cloud + local” workflows possible.

