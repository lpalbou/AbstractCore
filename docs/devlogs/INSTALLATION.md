# AbstractLLM Installation Guide

AbstractLLM provides a modular installation system that allows you to install only the dependencies you need for your specific use case.

## Quick Start

### Core Installation (Minimal)
```bash
pip install abstractllm
```

This installs the core package with minimal dependencies:
- `pydantic` - Data validation
- `httpx` - HTTP client
- `tiktoken` - Token counting

**Supported providers with core only:** Mock, Ollama, LMStudio

### Provider-Specific Installation

Install AbstractLLM with specific provider support:

```bash
# OpenAI (GPT models)
pip install abstractllm[openai]

# Anthropic (Claude models)
pip install abstractllm[anthropic]

# HuggingFace (Transformers, local models, GGUF)
pip install abstractllm[huggingface]

# MLX (Apple Silicon optimized)
pip install abstractllm[mlx]

# Multiple providers
pip install abstractllm[openai,anthropic,mlx]
```

### Convenience Groups

```bash
# API-based providers (lightweight)
pip install abstractllm[api-providers]
# Includes: openai, anthropic

# Local providers
pip install abstractllm[local-providers]
# Includes: ollama, lmstudio, mlx

# Heavy ML providers
pip install abstractllm[heavy-providers]
# Includes: huggingface (with PyTorch)

# All providers
pip install abstractllm[all-providers]

# Lightweight (no PyTorch/Transformers)
pip install abstractllm[lightweight]
```

### Development Installation

```bash
# Full development environment
pip install abstractllm[full-dev]

# Just development tools
pip install abstractllm[dev]

# Testing only
pip install abstractllm[test]
```

## Provider Dependencies

### Core Dependencies (Always Installed)
- **pydantic** (>=2.0.0) - Data validation and serialization
- **httpx** (>=0.24.0) - HTTP client for API calls
- **tiktoken** (>=0.5.0) - Token counting and encoding

### Provider-Specific Dependencies

#### OpenAI (`abstractllm[openai]`)
- **openai** (>=1.0.0) - Official OpenAI Python client

**Supports:** GPT-3.5, GPT-4, GPT-4o, o1, future models

#### Anthropic (`abstractllm[anthropic]`)
- **anthropic** (>=0.25.0) - Official Anthropic Python client

**Supports:** Claude 3 (Haiku, Sonnet, Opus), Claude 3.5, future models

#### Ollama (No Extra Dependencies)
Uses core `httpx` for local API communication.

**Supports:** All Ollama-compatible models running locally

#### LMStudio (No Extra Dependencies)
Uses core `httpx` for OpenAI-compatible API.

**Supports:** All LMStudio models via OpenAI API compatibility

#### HuggingFace (`abstractllm[huggingface]`)
- **transformers** (>=4.30.0) - HuggingFace Transformers library
- **torch** (>=1.12.0) - PyTorch for model inference
- **llama-cpp-python** (>=0.2.0) - GGUF model support

**Supports:**
- Standard HuggingFace models
- GGUF quantized models
- Local inference
- GPU acceleration

#### MLX (`abstractllm[mlx]`)
- **mlx** (>=0.15.0) - Apple's MLX framework
- **mlx-lm** (>=0.15.0) - MLX language model support

**Supports:** Apple Silicon optimized models, quantized inference

## Installation Examples

### API-Only Usage (Lightweight)
```bash
pip install abstractllm[openai,anthropic]
```
**Size:** ~50MB | **Use case:** Cloud API access only

### Local Development (Medium)
```bash
pip install abstractllm[openai,anthropic,ollama,mlx]
```
**Size:** ~500MB | **Use case:** Mix of API and local models

### Full ML Research (Heavy)
```bash
pip install abstractllm[all-providers]
```
**Size:** ~3GB+ | **Use case:** Research, fine-tuning, all model types

### Production API Service
```bash
pip install abstractllm[api-providers]
```
**Size:** ~50MB | **Use case:** API service deployment

## Verification

Test your installation:

```bash
python -c "
import abstractllm
llm = abstractllm.create_llm('mock')
print(f'✅ AbstractLLM {abstractllm.__version__} installed successfully')
"
```

Or run the comprehensive test:

```bash
python -c "
import subprocess
import sys
subprocess.run([sys.executable, '-m', 'test_installation'])
"
```

## Troubleshooting

### Import Errors

If you get import errors for a specific provider:

```python
ImportError: OpenAI dependencies not installed. Install with: pip install abstractllm[openai]
```

Install the required provider extras as shown in the error message.

### Version Conflicts

If you encounter dependency conflicts:

```bash
# Create a fresh virtual environment
python -m venv abstractllm_env
source abstractllm_env/bin/activate  # On Windows: abstractllm_env\Scripts\activate
pip install abstractllm[your-providers]
```

### Apple Silicon (M1/M2/M3)

For optimal performance on Apple Silicon:

```bash
pip install abstractllm[mlx,openai,anthropic]
```

This gives you both cloud APIs and local Apple Silicon optimized inference.

### GPU Support

For NVIDIA GPU support with HuggingFace:

```bash
# Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install AbstractLLM
pip install abstractllm[huggingface]
```

## Minimal Example

```python
import abstractllm

# Core only (works with any installation)
mock_llm = abstractllm.create_llm('mock')
response = mock_llm.generate("Hello!")
print(response.content)

# With OpenAI (requires abstractllm[openai])
try:
    openai_llm = abstractllm.create_llm('openai', model='gpt-4')
    response = openai_llm.generate("Hello!")
    print(response.content)
except ImportError:
    print("Install with: pip install abstractllm[openai]")
```

## Migration from v0.4.x

If upgrading from previous versions:

```bash
pip uninstall abstractllm
pip install abstractllm[your-providers]
```

The API remains backward compatible, but installation now requires explicit provider selection.