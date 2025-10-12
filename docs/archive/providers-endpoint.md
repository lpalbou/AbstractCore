# Providers Endpoint Documentation

## Overview

The `/providers` endpoint returns information about all available AbstractCore providers, including their capabilities, available models, and current status. This endpoint is useful for discovering what providers are accessible and building dynamic provider selection interfaces.

## Endpoint

```
GET /providers
```

## Request

No parameters required. This is a simple GET request.

```bash
curl http://localhost:8000/providers
```

## Response Format

Returns a JSON object containing a list of provider information.

### Response Structure

```json
{
  "providers": [
    {
      "name": "anthropic",
      "type": "llm",
      "model_count": 5,
      "status": "available",
      "description": "Anthropic provider with 5 available models"
    },
    {
      "name": "ollama",
      "type": "llm",
      "model_count": 12,
      "status": "available",
      "description": "Ollama provider with 12 available models"
    }
  ]
}
```

### Response Fields

#### `providers` (array)
List of provider objects. Only includes providers that have available models.

**Provider Object:**
- `name` (string): Provider identifier (lowercase)
- `type` (string): Provider type (currently always `"llm"`, may include `"embedding"` in the future)
- `model_count` (integer): Number of models available from this provider
- `status` (string): Provider availability status (currently always `"available"`)
- `description` (string): Human-readable description of the provider and model count

## Supported Providers

### Commercial API Providers

#### OpenAI
- **Name**: `openai`
- **Type**: Commercial API
- **Models**: GPT-4, GPT-3.5-turbo, text-embedding-ada-002, and more
- **Authentication**: Requires `OPENAI_API_KEY` environment variable
- **Cost**: Pay-per-use based on tokens
- **Best For**: Production applications, highest quality outputs

**Key Features:**
- State-of-the-art language models
- Reliable uptime and performance
- Extensive documentation and support
- Native function calling support

#### Anthropic
- **Name**: `anthropic`
- **Type**: Commercial API
- **Models**: Claude 3 family (Opus, Sonnet, Haiku)
- **Authentication**: Requires `ANTHROPIC_API_KEY` environment variable
- **Cost**: Pay-per-use based on tokens
- **Best For**: Long context tasks, reasoning, analysis

**Key Features:**
- Extended context windows (up to 200K tokens)
- Strong reasoning capabilities
- Constitutional AI safety measures
- Excellent at following instructions

### Local/Self-Hosted Providers

#### Ollama
- **Name**: `ollama`
- **Type**: Local LLM server
- **Models**: LLaMA, Mistral, Qwen, Phi, CodeLLaMA, and many more
- **Authentication**: None (local access)
- **Cost**: Free (uses local compute)
- **Best For**: Privacy, offline use, experimentation

**Key Features:**
- Run models locally without internet
- No API costs or rate limits
- Full data privacy
- Easy model management
- Support for custom models

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3

# Start Ollama server (runs on http://localhost:11434)
ollama serve
```

#### LMStudio
- **Name**: `lmstudio`
- **Type**: Local development platform
- **Models**: Any GGUF format model from HuggingFace
- **Authentication**: None (local access)
- **Cost**: Free (uses local compute)
- **Best For**: Model testing, development, fine-tuning

**Key Features:**
- User-friendly GUI for model management
- OpenAI-compatible API
- Model quantization support
- Easy model switching
- Performance monitoring

**Setup:**
1. Download LMStudio from [lmstudio.ai](https://lmstudio.ai)
2. Load a model through the GUI
3. Start the local server (runs on http://localhost:1234)

#### MLX
- **Name**: `mlx`
- **Type**: Apple Silicon optimized inference
- **Models**: MLX-optimized models from HuggingFace
- **Authentication**: None (local)
- **Cost**: Free (uses local compute)
- **Best For**: Mac users with Apple Silicon (M1/M2/M3)

**Key Features:**
- Optimized for Apple Silicon
- Fast inference on Mac
- Unified memory architecture benefits
- Open-source framework

#### HuggingFace
- **Name**: `huggingface`
- **Type**: Transformers library / local models
- **Models**: Access to 100,000+ models on HuggingFace Hub
- **Authentication**: Optional (for private models)
- **Cost**: Free (uses local compute) or HF Inference API costs
- **Best For**: Research, custom models, embeddings

**Key Features:**
- Massive model selection
- Direct model loading from HuggingFace Hub
- Support for custom/fine-tuned models
- Excellent for embeddings
- Active community

### Testing Provider

#### Mock
- **Name**: `mock`
- **Type**: Testing/development provider
- **Models**: Simulated models for testing
- **Authentication**: None
- **Cost**: Free
- **Best For**: Testing, CI/CD, development without external dependencies

**Key Features:**
- No external dependencies
- Predictable responses for testing
- Fast response times
- No rate limits

## Example Requests

### Basic Request

```bash
curl http://localhost:8000/providers
```

### Response Example

```json
{
  "providers": [
    {
      "name": "anthropic",
      "type": "llm",
      "model_count": 5,
      "status": "available",
      "description": "Anthropic provider with 5 available models"
    },
    {
      "name": "huggingface",
      "type": "llm",
      "model_count": 3,
      "status": "available",
      "description": "Huggingface provider with 3 available models"
    },
    {
      "name": "lmstudio",
      "type": "llm",
      "model_count": 2,
      "status": "available",
      "description": "Lmstudio provider with 2 available models"
    },
    {
      "name": "ollama",
      "type": "llm",
      "model_count": 15,
      "status": "available",
      "description": "Ollama provider with 15 available models"
    },
    {
      "name": "openai",
      "type": "llm",
      "model_count": 8,
      "status": "available",
      "description": "Openai provider with 8 available models"
    }
  ]
}
```

## Use Cases

### 1. Dynamic Provider Selection UI

```python
import requests

def build_provider_menu():
    """Build a dynamic provider selection menu."""
    response = requests.get("http://localhost:8000/providers")
    providers = response.json()["providers"]
    
    print("Available Providers:")
    print("-" * 50)
    
    for i, provider in enumerate(providers, 1):
        print(f"{i}. {provider['name'].title()}")
        print(f"   Models: {provider['model_count']}")
        print(f"   {provider['description']}")
        print()
    
    return providers

# Usage
providers = build_provider_menu()
```

### 2. Check Provider Availability

```python
import requests

def is_provider_available(provider_name: str) -> bool:
    """Check if a specific provider is available."""
    response = requests.get("http://localhost:8000/providers")
    providers = response.json()["providers"]
    
    return any(p["name"] == provider_name for p in providers)

# Usage
if is_provider_available("ollama"):
    print("✓ Ollama is available")
    # Use Ollama for requests
else:
    print("✗ Ollama is not available, falling back to OpenAI")
    # Use OpenAI instead
```

### 3. Find Provider with Most Models

```python
import requests

def get_provider_with_most_models():
    """Find the provider with the most available models."""
    response = requests.get("http://localhost:8000/providers")
    providers = response.json()["providers"]
    
    if not providers:
        return None
    
    best_provider = max(providers, key=lambda p: p["model_count"])
    return best_provider

# Usage
best = get_provider_with_most_models()
if best:
    print(f"Provider with most models: {best['name']} ({best['model_count']} models)")
```

### 4. List Models for Each Provider

```python
import requests

def list_all_provider_models():
    """Get detailed model information for all providers."""
    # Get providers
    providers_response = requests.get("http://localhost:8000/providers")
    providers = providers_response.json()["providers"]
    
    for provider in providers:
        print(f"\n{provider['name'].upper()}")
        print("=" * 50)
        
        # Get models for this provider
        models_response = requests.get(
            f"http://localhost:8000/v1/models?provider={provider['name']}"
        )
        models = models_response.json()["data"]
        
        for model in models:
            print(f"  • {model['id']}")
        
        print(f"\nTotal: {len(models)} models")

# Usage
list_all_provider_models()
```

### 5. Smart Provider Fallback

```python
import requests

def chat_with_fallback(message: str, preferred_providers: list):
    """
    Try to chat with preferred providers, falling back to alternatives.
    """
    # Get available providers
    response = requests.get("http://localhost:8000/providers")
    available = {p["name"] for p in response.json()["providers"]}
    
    # Try preferred providers in order
    for provider in preferred_providers:
        if provider in available:
            try:
                # Get first model from provider
                models_response = requests.get(
                    f"http://localhost:8000/v1/models?provider={provider}"
                )
                models = models_response.json()["data"]
                
                if models:
                    model_id = models[0]["id"]
                    
                    # Make chat request
                    chat_response = requests.post(
                        "http://localhost:8000/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": message}]
                        }
                    )
                    
                    if chat_response.status_code == 200:
                        print(f"✓ Using provider: {provider}")
                        return chat_response.json()
            
            except Exception as e:
                print(f"✗ Provider {provider} failed: {e}")
                continue
    
    raise Exception("No available providers could handle the request")

# Usage
response = chat_with_fallback(
    "Hello!",
    preferred_providers=["ollama", "openai", "anthropic"]
)
print(response["choices"][0]["message"]["content"])
```

### 6. Cost Optimization

```python
import requests

def choose_cost_effective_provider(task_complexity: str):
    """
    Choose provider based on task complexity and cost.
    
    Args:
        task_complexity: 'simple', 'medium', or 'complex'
    """
    response = requests.get("http://localhost:8000/providers")
    available = {p["name"] for p in response.json()["providers"]}
    
    # Priority based on cost (free first, then paid)
    if task_complexity == "simple":
        # Simple tasks: prefer free local models
        priorities = ["ollama", "lmstudio", "mlx", "openai"]
    elif task_complexity == "medium":
        # Medium tasks: balance quality and cost
        priorities = ["ollama", "openai", "anthropic"]
    else:  # complex
        # Complex tasks: prefer highest quality
        priorities = ["anthropic", "openai", "ollama"]
    
    for provider in priorities:
        if provider in available:
            return provider
    
    return None

# Usage
provider = choose_cost_effective_provider("simple")
print(f"Using {provider} for cost-effective processing")
```

## Provider Comparison

| Provider | Type | Cost | Privacy | Quality | Context | Use Case |
|----------|------|------|---------|---------|---------|----------|
| **OpenAI** | API | $$$ | Cloud | Excellent | 128K | Production apps |
| **Anthropic** | API | $$$ | Cloud | Excellent | 200K | Long context, reasoning |
| **Ollama** | Local | Free | Private | Good | Varies | Privacy, offline |
| **LMStudio** | Local | Free | Private | Good | Varies | Development, testing |
| **MLX** | Local | Free | Private | Good | Varies | Mac optimization |
| **HuggingFace** | Local/API | Free/$ | Private/Cloud | Varies | Varies | Research, custom models |

## Best Practices

### 1. Check Availability First

Always check provider availability before making requests:

```python
providers = requests.get("http://localhost:8000/providers").json()["providers"]
provider_names = {p["name"] for p in providers}

if "ollama" in provider_names:
    # Use Ollama
    pass
```

### 2. Implement Fallback Logic

Have backup providers in case primary is unavailable:

```python
provider_priority = ["ollama", "openai", "anthropic"]
for provider in provider_priority:
    if check_provider_available(provider):
        break
```

### 3. Cache Provider List

Cache the provider list to reduce API calls:

```python
import time

class ProviderCache:
    def __init__(self, ttl=300):  # 5 minutes
        self.cache = None
        self.last_update = 0
        self.ttl = ttl
    
    def get_providers(self):
        now = time.time()
        if not self.cache or (now - self.last_update) > self.ttl:
            response = requests.get("http://localhost:8000/providers")
            self.cache = response.json()["providers"]
            self.last_update = now
        return self.cache

cache = ProviderCache()
providers = cache.get_providers()
```

### 4. Monitor Provider Status

Regularly check provider status for monitoring:

```python
import requests
from datetime import datetime

def monitor_providers():
    """Monitor provider availability."""
    response = requests.get("http://localhost:8000/providers")
    providers = response.json()["providers"]
    
    print(f"Provider Status at {datetime.now()}")
    print("-" * 60)
    
    for provider in providers:
        status_icon = "✓" if provider["status"] == "available" else "✗"
        print(f"{status_icon} {provider['name']}: {provider['model_count']} models")
```

### 5. Build Provider-Aware Applications

Design applications that adapt to available providers:

```python
class AdaptiveChat:
    def __init__(self):
        self.update_providers()
    
    def update_providers(self):
        """Update available providers list."""
        response = requests.get("http://localhost:8000/providers")
        self.providers = response.json()["providers"]
    
    def chat(self, message, preferred_provider=None):
        """Chat with fallback to available providers."""
        if preferred_provider and self.is_available(preferred_provider):
            return self._chat_with_provider(message, preferred_provider)
        else:
            # Use first available provider
            return self._chat_with_provider(message, self.providers[0]["name"])
    
    def is_available(self, provider_name):
        return any(p["name"] == provider_name for p in self.providers)
```

## Related Endpoints

- **List Models**: `GET /v1/models` - List all models from all providers
- **List Provider Models**: `GET /v1/models?provider={name}` - List models from specific provider
- **List by Type**: `GET /v1/models?type=text-generation` - Filter models by type
- **Chat Completions**: `POST /v1/chat/completions` - Create chat completions
- **Embeddings**: `POST /v1/embeddings` - Create embeddings

## Error Handling

The endpoint is designed to be robust and will return an empty providers list if there's an error:

```json
{
  "providers": []
}
```

This ensures the endpoint always returns a valid response, even if all providers are unavailable.

## Future Enhancements

Potential future additions to the providers endpoint:

1. **Provider Types**: Distinguish between `llm` and `embedding` provider types
2. **Health Status**: Real-time health checks for each provider
3. **Capabilities**: Detailed capability information (streaming, tools, vision, etc.)
4. **Performance Metrics**: Average response times, token costs
5. **Rate Limits**: Current rate limit status for API providers
6. **Model Categories**: Breakdown of text-generation vs embedding model counts

---

**AbstractCore-Specific**: This endpoint is unique to AbstractCore and provides provider discovery capabilities not available in standard OpenAI-compatible APIs. Use it to build intelligent, adaptive applications that work across multiple LLM providers.

