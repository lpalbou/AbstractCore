# Provider Guide

This guide covers all supported LLM providers in AbstractLLM and how to use them.

## Quick Start

```python
from abstractllm import create_llm

# Create a provider
llm = create_llm("openai", model="gpt-3.5-turbo")

# Generate a response
response = llm.generate("Hello, world!")
print(response.content)
```

## Supported Providers

### OpenAI

**Installation**:
```bash
pip install abstractllm[openai]
```

**Configuration**:
```python
# Using environment variable
export OPENAI_API_KEY=your_api_key

# Or pass directly
llm = create_llm("openai",
                 model="gpt-4",
                 api_key="your_api_key")
```

**Supported Models**:
- GPT-3.5: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`
- GPT-4: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`

**Features**:
- ✅ Tool calling
- ✅ Vision (GPT-4V models)
- ✅ Streaming
- ✅ JSON mode

### Anthropic (Claude)

**Installation**:
```bash
pip install abstractllm[anthropic]
```

**Configuration**:
```python
# Using environment variable
export ANTHROPIC_API_KEY=your_api_key

# Or pass directly
llm = create_llm("anthropic",
                 model="claude-3-haiku-20240307",
                 api_key="your_api_key")
```

**Supported Models**:
- Claude 3: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- Claude 3.5: `claude-3-5-sonnet-20240620`, `claude-3-5-sonnet-20241022`
- Claude 2: `claude-2.1`, `claude-2.0`

**Features**:
- ✅ Tool calling (XML format)
- ✅ Vision (all Claude 3 models)
- ✅ Streaming
- ✅ 200k context window

### Ollama

**Installation**:
```bash
# Install Ollama first
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull qwen3-coder:30b
```

**Configuration**:
```python
llm = create_llm("ollama",
                 model="qwen3-coder:30b",
                 base_url="http://localhost:11434")
```

**Popular Models**:
- Qwen: `qwen3-coder:30b`, `qwen3:8b`, `qwen3:4b`
- Llama: `llama3.1:8b`, `llama3:70b`
- Mistral: `mistral:7b`, `mixtral:8x7b`
- Gemma: `gemma3:27b`, `gemma3:9b`

**Features**:
- ✅ Local execution
- ✅ No API costs
- ✅ Privacy-first
- ⚠️ Tool calling (architecture-specific)

### MLX (Apple Silicon)

**Installation**:
```bash
pip install abstractllm[mlx]
```

**Configuration**:
```python
llm = create_llm("mlx",
                 model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
```

**Popular Models**:
- Qwen: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`
- Mistral: `mlx-community/Mistral-7B-Instruct-v0.1-4bit`
- Llama: `mlx-community/Meta-Llama-3-8B-Instruct-4bit`

**Features**:
- ✅ Optimized for Apple Silicon
- ✅ 4-bit quantization
- ✅ Fast inference
- ❌ No native streaming

### LM Studio

**Installation**:
```bash
# Download LM Studio from https://lmstudio.ai/
# Load a model in the UI
# Start the server
```

**Configuration**:
```python
llm = create_llm("lmstudio",
                 model="local-model",
                 base_url="http://localhost:1234")
```

**Features**:
- ✅ OpenAI-compatible API
- ✅ GUI for model management
- ✅ Multiple model formats
- ✅ Built-in quantization

## Provider Comparison

| Provider | Speed | Cost | Privacy | Tool Support | Context |
|----------|-------|------|---------|--------------|---------|
| OpenAI | Fast | $$$ | Cloud | ✅ Native | 128k |
| Anthropic | Fast | $$$ | Cloud | ✅ Native | 200k |
| Ollama | Medium | Free | Local | ⚠️ Limited | Varies |
| MLX | Fast | Free | Local | ❌ | Varies |
| LM Studio | Medium | Free | Local | ⚠️ Limited | Varies |

## Common Parameters

All providers support these parameters:

```python
response = llm.generate(
    prompt="Your prompt",
    temperature=0.7,      # Creativity (0-2)
    max_tokens=1000,      # Max response length
    top_p=0.9,           # Nucleus sampling
    stream=False         # Streaming response
)
```

## Tool Calling

Different providers handle tools differently:

```python
# Define a tool
tools = [{
    "name": "get_weather",
    "description": "Get current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"]
    }
}]

# OpenAI/Anthropic handle this natively
response = llm.generate("What's the weather in NYC?", tools=tools)

if response.has_tool_calls():
    for call in response.tool_calls:
        print(f"Calling {call['name']} with {call['arguments']}")
```

## Streaming

Stream responses for better user experience:

```python
# Enable streaming
for chunk in llm.generate("Tell me a story", stream=True):
    print(chunk.content, end="", flush=True)
```

## Error Handling

```python
from abstractllm.exceptions import ProviderAPIError, RateLimitError

try:
    response = llm.generate("Hello")
except RateLimitError:
    # Wait and retry
    time.sleep(60)
except ProviderAPIError as e:
    print(f"API error: {e}")
```

## Best Practices

1. **Choose the right provider**:
   - Production: OpenAI/Anthropic for reliability
   - Development: Ollama/MLX for cost savings
   - Privacy-sensitive: Local providers only

2. **Optimize for your hardware**:
   - Apple Silicon: Use MLX
   - NVIDIA GPU: Use Ollama with CUDA
   - CPU only: Use smaller models

3. **Handle provider differences**:
   - Test across multiple providers
   - Have fallback providers
   - Abstract provider-specific logic

4. **Monitor usage**:
   - Track token consumption
   - Log response times
   - Monitor error rates

## Testing

Use the mock provider for testing:

```python
# No API calls made
llm = create_llm("mock")
response = llm.generate("Test prompt")
assert response.content == "Mock response for: Test prompt"
```