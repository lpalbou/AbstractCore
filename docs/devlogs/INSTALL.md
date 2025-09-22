# AbstractLLM Core Installation Guide

## Quick Start (Development)

Since this project is not yet published to PyPI, install it in development mode:

```bash
# Clone and install in development mode
cd /path/to/abstractllm_core
pip install -e .

# Install specific provider dependencies as needed
pip install -e ".[openai]"      # For OpenAI
pip install -e ".[anthropic]"   # For Anthropic
pip install -e ".[ollama]"      # For Ollama
pip install -e ".[huggingface]" # For HuggingFace + GGUF
pip install -e ".[mlx]"         # For MLX (Apple Silicon)
pip install -e ".[all]"         # Install all providers
```

## Fresh Virtual Environment Setup

```bash
# Create new virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install AbstractLLM Core in development mode
pip install -e .

# Install providers you need
pip install -e ".[openai,anthropic,ollama]"  # Common providers
pip install -e ".[huggingface]"             # For local GGUF models
pip install -e ".[mlx]"                     # For Apple Silicon
```

## Provider-Specific Notes

### HuggingFace + GGUF Models
- Installs `llama-cpp-python` which requires compilation
- May take several minutes to install
- Requires models to be pre-downloaded to HuggingFace cache

### MLX (Apple Silicon Only)
- Only works on Apple Silicon Macs (M1, M2, M3, M4)
- Requires macOS 13.3 or later

### Dependencies Issues
If you encounter dependency conflicts:

```bash
# Install with no-deps and manually install core requirements
pip install -e . --no-deps
pip install pydantic httpx tiktoken

# Then install specific providers
pip install openai anthropic ollama
```

## Verification

Test your installation:

```python
from abstractllm import create_llm

# Test available providers
provider = create_llm("ollama", model="qwen3-coder:30b")
print("Installation successful!")
```