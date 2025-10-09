# Prerequisites & Setup Guide

This guide walks you through setting up AbstractCore with different LLM providers. Choose the provider(s) that best fit your needs - you can use multiple providers in the same application.

## Quick Decision Guide

**Want to get started immediately?** → [OpenAI Setup](#openai-setup) (requires API key, costs ~$0.001-0.01 per request)

**Want free local models?** → [Ollama Setup](#ollama-setup) (free, runs on your machine)

**Have Apple Silicon Mac?** → [MLX Setup](#mlx-setup) (optimized for M1/M2/M3 chips)

**Want a GUI for local models?** → [LMStudio Setup](#lmstudio-setup) (easiest local setup)

## Core Installation

First, install AbstractCore with your preferred providers:

```bash
# Option 1: Start with cloud providers (fastest setup)
pip install abstractcore[openai,anthropic]

# Option 2: Local models only (no API keys needed)
pip install abstractcore[ollama,mlx]

# Option 3: Everything (recommended for development)
pip install abstractcore[all]

# Option 4: Minimal core only
pip install abstractcore
```

## Cloud Provider Setup

### OpenAI Setup

**Best for**: Production applications, latest models (GPT-4o, GPT-4o-mini), fast inference

**Cost**: ~$0.001-0.01 per request depending on model

#### 1. Get API Key

1. Go to [OpenAI API Dashboard](https://platform.openai.com/api-keys)
2. Create account or sign in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

#### 2. Set Environment Variable

```bash
# Option 1: Export in terminal (temporary)
export OPENAI_API_KEY="sk-your-actual-api-key-here"

# Option 2: Add to ~/.bashrc or ~/.zshrc (permanent)
echo 'export OPENAI_API_KEY="sk-your-actual-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Option 3: Create .env file in your project
echo 'OPENAI_API_KEY=sk-your-actual-api-key-here' > .env
```

#### 3. Test Setup

```python
from abstractllm import create_llm

# Test with GPT-4o-mini (fastest, cheapest)
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("Say hello in French")
print(response.content)  # Should output: "Bonjour!"
```

**Available Models**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

### Anthropic Setup

**Best for**: Long context tasks, Claude models, high-quality reasoning

**Cost**: ~$0.001-0.02 per request depending on model

#### 1. Get API Key

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create account or sign in
3. Go to "API Keys" section
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-`)

#### 2. Set Environment Variable

```bash
# Option 1: Export in terminal (temporary)
export ANTHROPIC_API_KEY="sk-ant-your-actual-api-key-here"

# Option 2: Add to shell profile (permanent)
echo 'export ANTHROPIC_API_KEY="sk-ant-your-actual-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Option 3: Create .env file
echo 'ANTHROPIC_API_KEY=sk-ant-your-actual-api-key-here' > .env
```

#### 3. Test Setup

```python
from abstractllm import create_llm

# Test with Claude 3.5 Haiku (fast, cost-effective)
llm = create_llm("anthropic", model="claude-3-5-haiku-20241022")
response = llm.generate("Explain Python in one sentence")
print(response.content)
```

**Available Models**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`

## Local Provider Setup

### Ollama Setup

**Best for**: Privacy, no costs, offline usage, customization

**Requirements**: 8GB+ RAM, works on Mac/Linux/Windows

#### 1. Install Ollama

**macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
# OR download from https://ollama.com/download
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
1. Download installer from [ollama.com/download](https://ollama.com/download)
2. Run the installer
3. Restart terminal

#### 2. Start Ollama Service

```bash
# Start Ollama server (runs in background)
ollama serve
```

#### 3. Download Models

```bash
# Recommended starter models (verified available)
ollama pull qwen3-coder:30b      # 18GB - Excellent for code, works great with AbstractCore
ollama pull qwen3:4b-instruct-2507-q4_K_M     # 4GB - AbstractCore default, balanced performance
ollama pull gemma3:1b            # 815MB - Very fast, good quality
ollama pull cogito:3b            # 2.2GB - Good general purpose

# Smaller models for resource-constrained systems
ollama pull gemma3:270m-it-qat   # 241MB - Ultra-fast for testing
ollama pull gemma3:270m          # 291MB - Minimal resource usage

# Specialized models
ollama pull nomic-embed-text     # 274MB - For embeddings
ollama pull granite3.3:2b        # 1.5GB - Good general purpose
```

#### 4. Test Setup

```python
from abstractllm import create_llm

# Test with a fast model
llm = create_llm("ollama", model="gemma3:1b")
response = llm.generate("What is Python?")
print(response.content)
```

#### Model Selection Guide

| Model | Size | RAM Needed | Speed | Best For |
|-------|------|------------|-------|----------|
| `gemma3:270m-it-qat` | 241MB | 2GB | ⚡⚡⚡ | Ultra-fast testing |
| `qwen3:4b-instruct-2507-q4_K_M` | 4GB | 8GB | ⚡⚡ | AbstractCore default, balanced performance |
| `gemma3:1b` | 815MB | 4GB | ⚡⚡⚡ | Fast general purpose |
| `cogito:3b` | 2.2GB | 6GB | ⚡⚡ | Balanced quality/speed |
| `granite3.3:2b` | 1.5GB | 6GB | ⚡⚡ | Good reasoning |
| `qwen3-coder:30b` | 18GB | 32GB | ⚡ | Code generation, complex tasks |

### MLX Setup (Apple Silicon)

**Best for**: M1/M2/M3/M4 Macs, optimized inference, good speed

**Requirements**: Apple Silicon Mac (M1/M2/M3/M4)

#### 1. Install MLX Dependencies

```bash
# MLX is automatically installed with AbstractCore
pip install abstractcore[mlx]
```

#### 2. Download Models

MLX models are automatically downloaded when first used. Popular options:

```python
from abstractllm import create_llm

# Models are auto-downloaded on first use
llm = create_llm("mlx", model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")  # 4.2GB
# OR
llm = create_llm("mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")      # 1.8GB
```

#### 3. Test Setup

```python
from abstractllm import create_llm

# Test with a good balance model
llm = create_llm("mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")
response = llm.generate("Explain machine learning briefly")
print(response.content)
```

**Popular MLX Models**:
- `mlx-community/Llama-3.2-3B-Instruct-4bit` - 1.8GB, fast
- `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` - 4.2GB, excellent for code
- `mlx-community/Llama-3.1-8B-Instruct-4bit` - 4.7GB, high quality

### LMStudio Setup

**Best for**: Easy GUI management, Windows users, non-technical users

**Requirements**: 8GB+ RAM, works on Mac/Linux/Windows

#### 1. Install LMStudio

1. Download from [lmstudio.ai](https://lmstudio.ai/)
2. Install the application
3. Launch LMStudio

#### 2. Download Models

1. Open LMStudio
2. Go to "Discover" tab
3. Search for recommended models:
   - `microsoft/Phi-3-mini-4k-instruct-gguf` (small, fast)
   - `microsoft/Phi-3-medium-4k-instruct-gguf` (medium quality)
   - `meta-llama/Llama-2-7b-chat-gguf` (good general purpose)
4. Click download for your preferred model

#### 3. Start Local Server

1. Go to "Local Server" tab in LMStudio
2. Select your downloaded model
3. Click "Start Server"
4. Note the port (usually 1234)

#### 4. Test Setup

```python
from abstractllm import create_llm

# LMStudio runs on localhost:1234 by default
llm = create_llm("lmstudio", base_url="http://localhost:1234")
response = llm.generate("Hello, how are you?")
print(response.content)
```

### HuggingFace Setup

**Best for**: Latest research models, custom models, GGUF files

**Requirements**: 8GB+ RAM, Python environment

#### 1. Install Dependencies

```bash
pip install abstractcore[huggingface]
```

#### 2. Optional: Get HuggingFace Token

For private models or higher rate limits:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a "Read" token
3. Set environment variable:

```bash
export HUGGINGFACE_TOKEN="hf_your-token-here"
```

#### 3. Test Setup

```python
from abstractllm import create_llm

# Use a small model for testing (auto-downloads)
llm = create_llm("huggingface", model="microsoft/DialoGPT-medium")
response = llm.generate("Hello there!")
print(response.content)
```

**Popular HuggingFace Models**:
- `microsoft/DialoGPT-medium` - Good for conversation
- `facebook/blenderbot-400M-distill` - Conversational AI
- `microsoft/CodeBERT-base` - Code understanding

## Troubleshooting

### Common Issues

#### "No module named 'abstractllm'"
```bash
# Make sure you installed AbstractCore
pip install abstractcore
```

#### "OpenAI API key not found"
```bash
# Check if environment variable is set
echo $OPENAI_API_KEY

# If empty, set it:
export OPENAI_API_KEY="sk-your-key-here"
```

#### "Connection error to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Check if models are available
ollama list

# Pull a model if none available
ollama pull gemma3:1b
```

#### "Model not found in MLX"
```python
# Use exact model names from HuggingFace MLX community
llm = create_llm("mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")
```

#### "LMStudio connection refused"
```bash
# Make sure LMStudio server is running on correct port
# Check LMStudio logs for the exact port and URL
```

### Memory Issues

#### "Out of memory" with local models
```bash
# Try smaller models
ollama pull gemma3:1b        # Only 1.3GB
ollama pull tinyllama        # Only 637MB

# Or increase swap space on Linux
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### MLX models too slow
```python
# Use 4-bit quantized models for faster inference
llm = create_llm("mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")
```

### API Key Issues

#### OpenAI billing issues
1. Check your [billing dashboard](https://platform.openai.com/account/billing)
2. Add payment method if needed
3. Check usage limits

#### Anthropic rate limits
1. Check your [console](https://console.anthropic.com/)
2. Upgrade to higher tier if needed
3. Implement retry logic in your code

## Testing Your Setup

### Universal Test Script

Save this as `test_setup.py` and run it to test all your providers:

```python
#!/usr/bin/env python3
"""Test script for AbstractCore providers"""

import os
from abstractllm import create_llm

def test_provider(provider_name, model, **kwargs):
    """Test a specific provider"""
    try:
        print(f"\n🧪 Testing {provider_name} with {model}...")
        llm = create_llm(provider_name, model=model, **kwargs)
        response = llm.generate("Say 'Hello from AbstractCore!'")
        print(f"✅ {provider_name}: {response.content}")
        return True
    except Exception as e:
        print(f"❌ {provider_name}: {e}")
        return False

def main():
    print("AbstractCore Provider Test Suite")
    print("=" * 40)

    results = {}

    # Test cloud providers (if API keys available)
    if os.getenv("OPENAI_API_KEY"):
        results["OpenAI"] = test_provider("openai", "gpt-4o-mini")
    else:
        print("\n⚠️  Skipping OpenAI (no OPENAI_API_KEY)")

    if os.getenv("ANTHROPIC_API_KEY"):
        results["Anthropic"] = test_provider("anthropic", "claude-3-5-haiku-20241022")
    else:
        print("\n⚠️  Skipping Anthropic (no ANTHROPIC_API_KEY)")

    # Test local providers
    results["Ollama"] = test_provider("ollama", "gemma3:1b")

    try:
        results["MLX"] = test_provider("mlx", "mlx-community/Llama-3.2-3B-Instruct-4bit")
    except:
        print("\n⚠️  Skipping MLX (not on Apple Silicon or model not available)")

    try:
        results["LMStudio"] = test_provider("lmstudio", base_url="http://localhost:1234")
    except:
        print("\n⚠️  Skipping LMStudio (server not running on localhost:1234)")

    # Summary
    print("\n" + "=" * 40)
    print("Test Results:")
    working = [name for name, success in results.items() if success]
    if working:
        print(f"✅ Working providers: {', '.join(working)}")
    else:
        print("❌ No providers working")

    print("\n💡 Next steps:")
    print("- Add API keys for cloud providers")
    print("- Install Ollama and download models")
    print("- Start LMStudio local server")
    print("- See docs/prerequisites.md for detailed setup")

if __name__ == "__main__":
    main()
```

Run the test:
```bash
python test_setup.py
```

## Performance Recommendations

### For Development
- **Local**: `ollama` with `gemma3:1b` (fast, free)
- **Cloud**: `openai` with `gpt-4o-mini` (fast, cheap)

### For Production
- **High Quality**: `openai` with `gpt-4o` or `anthropic` with `claude-3-5-sonnet`
- **Cost-Effective**: `openai` with `gpt-4o-mini`
- **Privacy**: `ollama` with `qwen2.5:14b` (if you have the hardware)

### For Specific Tasks
- **Code Generation**: `ollama` with `qwen2.5-coder:7b`
- **Long Context**: `anthropic` with Claude models (up to 200K tokens)
- **Fast Responses**: `ollama` with `gemma3:1b` or `openai` with `gpt-4o-mini`
- **Offline/Air-gapped**: `ollama` or `mlx` with downloaded models

## Security Notes

### API Keys
- Never commit API keys to version control
- Use environment variables or `.env` files
- Rotate keys periodically
- Monitor usage for unexpected spikes

### Local Models
- Local models keep data on your machine
- No internet required after initial download
- Models can be large (1GB-20GB+)
- Some models may have usage restrictions

### Network Security
- LMStudio and Ollama servers run locally by default
- Be careful exposing servers to network (use authentication)
- Consider firewall rules for production deployments

This setup guide should get you running with any AbstractCore provider. Choose what works best for your use case - you can always add more providers later!