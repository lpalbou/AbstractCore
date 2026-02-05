# Acknowledgements

AbstractCore is possible thanks to the work of many open-source maintainers and research communities. We’re grateful to everyone who builds and maintains the ecosystem this project depends on.

This list is not exhaustive. The source of truth for install-time dependencies is `pyproject.toml`.

## Core dependencies (default install)

- **Python** and the Python packaging ecosystem
- **Pydantic** (data validation / typed schemas)
- **httpx** (HTTP client)

## Optional dependencies (installed via extras)

Providers and runtimes:
- **openai** (OpenAI SDK)
- **anthropic** (Anthropic SDK)
- **transformers**, **torch**, **torchvision**, **torchaudio** (HuggingFace runtime)
- **mlx**, **mlx-lm** (Apple Silicon local inference)
- **llama-cpp-python** (GGUF inference)
- **vllm** (GPU inference server integration)
- **outlines** (optional constrained decoding / structured output support for some backends)

Features:
- **Tools / web**: **requests**, **beautifulsoup4**, **lxml**, **ddgs** / **duckduckgo-search**, **psutil**
- **Embeddings**: **sentence-transformers**, **numpy**
- **Tokens**: **tiktoken**
- **Media / documents**: **Pillow**, **pymupdf4llm**, **pymupdf-layout**, **unstructured**, **pandas**
- **Compression**: **Pillow** (glyph rendering)
- **Server**: **fastapi**, **uvicorn**, **python-multipart**, **sse-starlette**
- **Vision plugin integration**: **abstractvision**
- **Docs**: **mkdocs**, **mkdocs-material**, **mkdocstrings[python]**, **mkdocs-autorefs**
- **Benchmarks**: **matplotlib** (MLX benchmark plots)

## Tooling (development)

- **pytest** (and plugins like **pytest-asyncio**, **pytest-mock**, **pytest-cov**)
- **responses** (HTTP mocking)
- **ruff**, **black**, **isort**, **mypy**, **pre-commit**

## Ecosystem and communities

We’re also grateful to the broader communities around local inference (Ollama, LM Studio, llama.cpp, MLX) and open model development, which make “cloud + local” workflows possible.
