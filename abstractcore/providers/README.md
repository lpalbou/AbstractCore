# Provider System

Unified LLM provider interface with support for commercial APIs (OpenAI, Anthropic) and local models (Ollama, LMStudio, HuggingFace, MLX). This module provides consistent abstractions for generation, streaming, tool execution, and structured outputs across all providers.

## Quick Reference

### Provider Selection Guide

| Use Case | Recommended Provider | Why |
|----------|---------------------|-----|
| Production APIs | OpenAI, Anthropic | Latest features, reliable uptime, enterprise support |
| Gateway / aggregator APIs | OpenRouter | OpenAI-compatible gateway across many model vendors |
| Cost optimization | Ollama, LMStudio | Free local inference, no API costs |
| Privacy/offline | HuggingFace, MLX | Complete data control, air-gapped deployments |
| Apple Silicon | MLX | Metal GPU acceleration, optimized performance |
| Custom models | HuggingFace | Fine-tuned models, GGUF support |
| Model testing | LMStudio | Easy model switching, UI-based management |
| Custom OpenAI-compatible endpoint | OpenAI-compatible | Point to any `/v1` server or proxy |

### Feature Comparison at a Glance

| Feature | OpenAI | Anthropic | Ollama | LMStudio | HuggingFace | MLX |
|---------|:------:|:---------:|:------:|:--------:|:-----------:|:---:|
| Streaming | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| Native Tools | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Vision | ✅ | ✅ | ⚠️ | ⚠️ | ✅ | ❌ |
| Structured Output | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Cost | $$$ | $$$ | Free | Free | Free | Free |

**Also supported**: `openai-compatible` (generic OpenAI-compatible `/v1`), `vllm` (OpenAI-compatible + extras), and `openrouter` (OpenAI-compatible gateway API).

## Common Tasks

- **How do I create a provider?** → See [Creating Providers](#creating-providers)
- **How do I list available models?** → See [Model Selection](#model-selection)
- **How do I use vision models?** → See [Generation Patterns](#generation-patterns) with media
- **How do I handle errors?** → See [Error Handling](#error-handling)
- **How do I check provider health?** → See [Public Interface](#public-interface) for `health()` method
- **How do I manage memory?** → See [Memory Management](#memory-management)
- **How do I use structured outputs?** → See [Generation Patterns](#generation-patterns) with response_model
- **How do I switch providers?** → Use `create_llm(provider_name, ...)` - see [Usage Patterns](#usage-patterns)

## Architecture Position

**Layer**: Provider abstraction layer
**Dependencies**: Core types, tools, media handlers, events, exceptions, architectures
**Used by**: Factory (`core/factory.py`), Server (`server/app.py`), CLI

This module sits between the core API and underlying LLM implementations, providing a unified interface that abstracts provider-specific details.

## Component Structure

```
providers/
├── base.py                    # BaseProvider abstract class with telemetry
├── registry.py                # Provider discovery and centralized factory
├── streaming.py               # Unified streaming with tool detection
├── openai_provider.py         # OpenAI (official SDK)
├── openai_compatible_provider.py  # Shared HTTP for OpenAI-compatible /v1 endpoints
├── openrouter_provider.py     # OpenRouter (OpenAI-compatible gateway)
├── anthropic_provider.py      # Anthropic Claude
├── ollama_provider.py         # Ollama local server
├── lmstudio_provider.py       # LMStudio local server
├── vllm_provider.py           # vLLM OpenAI-compatible server (+ guided decoding, LoRA, etc.)
├── huggingface_provider.py    # HuggingFace transformers + GGUF
└── mlx_provider.py            # Apple Silicon MLX
```

## Detailed Components

### base.py - BaseProvider Abstract Class

Comprehensive base class with integrated telemetry, events, error handling, and tool execution.

**Key Features**:
- **Telemetry tracking**: Automatic generation time, token usage, cost estimation
- **Event system integration**: GENERATION_STARTED, GENERATION_COMPLETED, TOOL_STARTED, TOOL_COMPLETED, ERROR
- **Retry management**: Circuit breaker with exponential backoff
- **Tool execution**: Unified tool handling with prompted/native support
- **Media processing**: Multi-modal content handling (images, PDFs, audio)
- **Structured outputs**: Pydantic model support with validation
- **Token management**: Unified max_tokens/max_output_tokens handling
- **Health checks**: Provider availability monitoring via `health()` method
- **Memory management**: `unload()` for explicit memory cleanup

**Abstract Methods**:
```python
def _generate_internal(self, prompt, messages, system_prompt, tools, media, stream, response_model, **kwargs)
def list_available_models(self, **kwargs) -> List[str]
```

**Public Interface**:
```python
# Generation
response = provider.generate(prompt="Hello", system_prompt="You are helpful", stream=False)

# Structured output
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int
validated_person = provider.generate(prompt="Extract: John, 30", response_model=Person)

# Tool execution
tools = [{"name": "search", "description": "Search the web", "parameters": {...}}]
response = provider.generate(prompt="Search for cats", tools=tools, execute_tools=True)

# Glyph compression (EXPERIMENTAL - vision models only)
# Compresses text into optimized images for 3-4x token savings
response = provider.generate(
    prompt="Summarize this document",
    media=["long_document.txt"],
    glyph_compression="auto"  # "auto" | "always" | "never"
)

# Health check
health = provider.health(timeout=5.0)  # {"status": True, "model_count": 10, ...}

# Memory cleanup
provider.unload()  # Free model memory
```

#### glyph_compression Parameter (⚠️ EXPERIMENTAL)

Controls visual-text compression for large documents (vision models only):

| Value | Behavior | Use Case |
|-------|----------|----------|
| `"auto"` (default) | Automatically compress if beneficial | Recommended for most cases |
| `"always"` | Force compression (raises error if model lacks vision) | When you know compression is needed |
| `"never"` | Disable compression | When you want raw text processing |

**Vision Model Requirement**: This feature **ONLY works with vision-capable models** (e.g., gpt-4o, claude-haiku-4-5, llama3.2-vision).

**Error Handling**:
- `glyph_compression="always"` with non-vision model → `UnsupportedFeatureError`
- `glyph_compression="auto"` with non-vision model → Warning logged, falls back to text processing

See [Compression Module](../compression/README.md) for detailed documentation.

```

### registry.py - Provider Registry & Discovery

Centralized registry providing single source of truth for provider metadata, discovery, and instantiation.

**Key Features**:
- **Lazy loading**: Providers loaded on-demand
- **Metadata management**: Features, installation instructions, authentication requirements
- **Model discovery**: Automatic enumeration via `list_available_models()`
- **Health status**: Provider availability checking
- **Factory pattern**: Centralized `create_provider()` with helpful error messages

**Provider Metadata**:
```python
ProviderInfo(
    name="openai",
    display_name="OpenAI",
    description="Commercial API with GPT-4, GPT-3.5, and embedding models",
    default_model="gpt-5-nano-2025-08-07",
    supported_features=["chat", "completion", "embeddings", "native_tools", "streaming", "structured_output"],
    authentication_required=True,
    local_provider=False,
    installation_extras="openai"
)
```

**Usage**:
```python
from abstractcore.providers import (
    create_provider,
    list_available_providers,
    get_all_providers_with_models,
    get_provider_info
)

# List all providers
providers = list_available_providers()  # ['openai', 'anthropic', 'ollama', ...]

# Get provider details
info = get_provider_info("openai")
print(f"Features: {info.supported_features}")
print(f"Install: pip install abstractcore[{info.installation_extras}]")

# Discover all providers with models
providers_with_models = get_all_providers_with_models()
for provider in providers_with_models:
    print(f"{provider['display_name']}: {provider['model_count']} models")

# Create provider instance
provider = create_provider("openai", model="gpt-4o", api_key="sk-...")
```

### streaming.py - Unified Streaming Processor

Real-time streaming with incremental tool detection, tag rewriting, and format conversion.

**Key Features**:
- **Incremental tool detection**: Detects tool calls character-by-character without buffering entire response
- **Tag rewriting**: Convert between formats (Qwen3 ↔ LLaMA ↔ XML ↔ OpenAI JSON)
- **Smart buffering**: Prevents partial tag streaming (e.g., `<fu` before `nction_call>`)
- **Format conversion**: Text-based → OpenAI JSON structured format
- **Pass-through mode**: Tools detected but not executed (for agentic workflows)

**Supported Formats**:
- **Qwen3**: `<|tool_call|>{"name":"search","arguments":{}}</|tool_call|>`
- **LLaMA**: `<function_call>{"name":"search","arguments":{}}</function_call>`
- **XML**: `<tool_call>{"name":"search","arguments":{}}</tool_call>`
- **Gemma**: `` ```tool_code...``` ``
- **OpenAI**: `{"id":"call_abc","type":"function","function":{...}}`

**Usage**:
```python
from abstractcore.providers.streaming import UnifiedStreamProcessor

processor = UnifiedStreamProcessor(
    model_name="qwen3:4b",
    execute_tools=False,  # Pass-through mode (default)
    tool_call_tags="llama3",  # Rewrite to LLaMA format
    default_target_format="qwen3"
)

for chunk in processor.process_stream(response_stream, tools):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.tool_calls:
        print(f"\nTool calls: {chunk.tool_calls}")
```

## Individual Providers

### OpenAI Provider

**Capabilities**: GPT-3.5, GPT-4, GPT-5, native tools, structured outputs, vision (4o/4-turbo), embeddings
**Authentication**: API key required
**Installation**: `pip install abstractcore[openai]`

**Setup**:
```bash
export OPENAI_API_KEY="sk-..."
```

**Usage**:
```python
from abstractcore import create_llm

# Basic generation
llm = create_llm("openai", model="gpt-5-mini")
response = llm.generate("Explain quantum computing")

# Vision
response = llm.generate(
    prompt="What's in this image?",
    media=["image.jpg"]
)

# Native structured output
from pydantic import BaseModel
class Analysis(BaseModel):
    summary: str
    sentiment: str
result = llm.generate("Analyze: Great product!", response_model=Analysis)

# Streaming with tools
tools = [{"name": "search", ...}]
for chunk in llm.generate("Search for AI news", tools=tools, stream=True):
    print(chunk.content, end="")
```

**Special Features**:
- Reasoning models (o1, gpt-5): Limited parameter support, use `max_completion_tokens`
- Strict mode schemas: Automatic `additionalProperties: false` enforcement
- Cached tokens: Prompt caching for repeated contexts
- Seed support: Deterministic outputs via `seed` parameter

### OpenAI-Compatible Provider (Generic)

**Capabilities**: Any OpenAI-compatible `/v1` endpoint (chat, streaming, tools if supported, embeddings if supported)
**Authentication**: Optional (depends on your server/proxy)
**Installation**: `pip install abstractcore` (no extra deps)

**Setup**:
- Base URL: `base_url=...` or `OPENAI_COMPATIBLE_BASE_URL`
- Optional key: `api_key=...` or `OPENAI_COMPATIBLE_API_KEY`

**Usage**:
```python
llm = create_llm(
    "openai-compatible",
    base_url="http://localhost:1234/v1",
    model="local-model",
)
response = llm.generate("Hello from a custom OpenAI-compatible endpoint")
```

**Notes**:
- `OpenAICompatibleProvider` is the shared HTTP implementation used by `lmstudio`, `vllm`, and `openrouter`.
- Prefer the specific provider (`lmstudio`/`vllm`/`openrouter`) when available for clearer intent + better defaults.

### OpenRouter Provider

**Capabilities**: OpenAI-compatible gateway API (multi-provider routing + unified billing)
**Authentication**: API key required
**Installation**: `pip install abstractcore` (no extra deps)

**Setup**:
```bash
export OPENROUTER_API_KEY="sk-or-..."
# Optional metadata headers:
export OPENROUTER_SITE_URL="https://your-app.example"
export OPENROUTER_APP_NAME="YourApp"
```

**Usage**:
```python
llm = create_llm("openrouter", model="openai/gpt-4o-mini")
response = llm.generate("Hello from OpenRouter")
```

**Notes**:
- Default endpoint is `https://openrouter.ai/api/v1` (override via `base_url=` or `OPENROUTER_BASE_URL`).

### Anthropic Provider

**Capabilities**: Claude (Haiku, Sonnet, Opus), native tools, structured outputs (via tool trick), vision
**Authentication**: API key required
**Installation**: `pip install abstractcore[anthropic]`

**Setup**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Usage**:
```python
llm = create_llm("anthropic", model="claude-haiku-4-5")
response = llm.generate("Write a story", max_output_tokens=2048)

# Vision
response = llm.generate("Describe this chart", media=["chart.png"])

# Structured output (synthetic tool approach)
class Report(BaseModel):
    findings: List[str]
    conclusion: str
report = llm.generate("Analyze sales data", response_model=Report)
```

**Special Features**:
- Structured outputs via "tool trick": Uses native tools under the hood for guaranteed schema compliance
- No seed support: Use `temperature=0` for more consistent outputs (warning issued)
- Extended thinking: Supports extended reasoning with `thinking` parameter

### Ollama Provider

**Capabilities**: Local models (Llama, Qwen, Mistral, etc.), prompted tools, streaming, structured outputs (native JSON schema), embeddings
**Authentication**: None (local server)
**Installation**: `pip install abstractcore[ollama]` + [Ollama installation](https://ollama.com)

**Setup**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen3:4b-instruct-2507-q4_K_M

# Start server (runs on http://localhost:11434 by default)
ollama serve
```

**Usage**:
```python
llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
response = llm.generate("Hello!")

# Custom base URL
llm = create_llm("ollama", model="llama3.2", base_url="http://192.168.1.100:11434")

# Structured output with native JSON schema
class Person(BaseModel):
    name: str
    age: int
person = llm.generate("Extract: Alice, 28", response_model=Person)

# Embeddings
embedding_response = llm.embed("Hello world")
```

**Special Features**:
- Native structured outputs: Server-side JSON schema validation (100% compliance)
- Message conversion: Automatically converts unsupported `tool` role to `user` role with markers
- Model unloading: `llm.unload()` with `keep_alive=0` to free server memory

### LMStudio Provider

**Capabilities**: Local models via OpenAI-compatible API, prompted tools, streaming, structured outputs (native), embeddings
**Authentication**: None (local server)
**Installation**: `pip install abstractcore` + [LM Studio download](https://lmstudio.ai)

**Setup**:
1. Download and open LM Studio
2. Load a model (e.g., `qwen/qwen3-4b-2507`)
3. Start local server (default: `http://localhost:1234/v1`)

**Usage**:
```python
llm = create_llm("lmstudio", model="qwen/qwen3-4b-2507")
response = llm.generate("What is AI?")

# Custom base URL
llm = create_llm("lmstudio", model="local-model", base_url="http://localhost:1234/v1")

# Vision (if model supports it)
response = llm.generate("Describe image", media=["photo.jpg"])

# Native structured output
class Task(BaseModel):
    title: str
    priority: int
task = llm.generate("Create task: Fix bug, priority 1", response_model=Task)
```

**Special Features**:
- OpenAI-compatible API: Same interface as OpenAI provider
- Vision model auto-detection: Automatically uses appropriate media handler based on model capabilities
- Native structured outputs: Server-side JSON schema enforcement
- Model normalization: Handles various model name formats (lmstudio/, qwen/, etc.)

### vLLM Provider

**Capabilities**: OpenAI-compatible API + guided decoding, beam search, Multi-LoRA management
**Authentication**: Optional (depends on your deployment)
**Installation**: `pip install abstractcore[vllm]` (plus a running vLLM server)

**Setup**:
- Base URL: `base_url=...` or `VLLM_BASE_URL` (default: `http://localhost:8000/v1`)
- Optional key: `api_key=...` or `VLLM_API_KEY`

**Usage**:
```python
llm = create_llm("vllm", model="Qwen/Qwen3-Coder-30B-A3B-Instruct", base_url="http://localhost:8000/v1")

# Guided decoding (vLLM extension)
response = llm.generate(
    "Return a JSON object with {name: string, age: number}",
    guided_json={"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}},
)
```

### HuggingFace Provider

**Capabilities**: Transformers models, GGUF models (via llama-cpp-python), vision models, prompted tools, structured outputs (native for GGUF, optional Outlines for transformers)
**Authentication**: Optional (public models work without API key)
**Installation**:
- Transformers: `pip install abstractcore[huggingface]`
- GGUF: `pip install abstractcore[huggingface] llama-cpp-python`
- Outlines (optional): `pip install outlines>=0.1.0`

**Setup**:
```bash
# Optional: Set HuggingFace token for private models
export HUGGINGFACE_TOKEN="hf_..."

# Download models (cache-only mode - no auto-download)
huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF
huggingface-cli download mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

**Usage**:
```python
# GGUF model (recommended for local inference)
llm = create_llm("huggingface", model="unsloth/Qwen3-4B-Instruct-2507-GGUF")
response = llm.generate("Hello!")

# Transformers model
llm = create_llm("huggingface", model="microsoft/phi-2")
response = llm.generate("Write code", max_tokens=512)

# Vision model (Glyph, GLM-4.1V)
llm = create_llm("huggingface", model="zai-org/Glyph")
response = llm.generate("Extract text from image", media=["document.jpg"])

# GGUF with native structured output
class Output(BaseModel):
    answer: str
    confidence: float
result = llm.generate("Is this spam?", response_model=Output)

# Transformers with Outlines (optional)
llm = create_llm("huggingface", model="phi-2", structured_output_method="native_outlines")
result = llm.generate("Extract data", response_model=Output)

# GPU offloading for GGUF
llm = create_llm("huggingface", model="GGUF-model", n_gpu_layers=-1)  # All layers on GPU
```

**Special Features**:
- **Dual backend**: Transformers (standard) + llama-cpp-python (GGUF)
- **Vision model support**: AutoModelForImageTextToText for Glyph, GLM-4.1V, Qwen2-VL, LLaVA
- **Custom models**: Supports DeepSeek-OCR and other specialized architectures
- **Structured outputs**:
  - GGUF: Native via llama-cpp-python (100% compliance, 0 retries)
  - Transformers: Optional Outlines integration (slower but guaranteed), or prompted fallback (faster, still 100% success)
- **Offline-first**: Cache-only mode, never auto-downloads
- **Apple Silicon**: MPS fallback for unsupported operations
- **Model type detection**: Auto-detects GGUF vs transformers based on filename/repo name

### MLX Provider

**Capabilities**: Apple Silicon optimized models, prompted tools, streaming, structured outputs (optional Outlines)
**Authentication**: None (local models)
**Installation**: `pip install abstractcore[mlx]` (macOS with Apple Silicon only)

**Setup**:
```bash
# Download MLX-optimized models
huggingface-cli download mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
huggingface-cli download mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

**Usage**:
```python
llm = create_llm("mlx", model="mlx-community/Qwen3-4B")
response = llm.generate("Explain machine learning")

# Streaming
for chunk in llm.generate("Write a poem", stream=True):
    print(chunk.content, end="")

# Structured output with Outlines (optional)
llm = create_llm("mlx", model="mlx-community/Qwen3-4B", structured_output_method="native_outlines")
class Summary(BaseModel):
    title: str
    points: List[str]
summary = llm.generate("Summarize article", response_model=Summary)

# Seed for deterministic output
response = llm.generate("Random number", seed=42)
```

**Special Features**:
- **Apple Silicon optimization**: Metal GPU acceleration via MLX framework
- **Real streaming**: Character-by-character streaming (not simulated)
- **Qwen template**: Automatic chat template formatting for Qwen models
- **Structured outputs**: Optional Outlines integration for constrained generation
- **Memory efficiency**: Optimized for on-device inference

## Provider Comparison

| Feature | OpenAI | Anthropic | Ollama | LMStudio | HuggingFace | MLX |
|---------|--------|-----------|--------|----------|-------------|-----|
| **Type** | Cloud API | Cloud API | Local Server | Local Server | Local Library | Local Library |
| **Authentication** | Required | Required | None | None | Optional | None |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ (simulated for transformers) | ✅ (native) |
| **Native Tools** | ✅ | ✅ | ❌ | ❌ | ❌ (GGUF supports limited) | ❌ |
| **Prompted Tools** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Structured Output** | ✅ Native | ✅ Via tool trick | ✅ Native JSON schema | ✅ Native | ✅ Native (GGUF), Optional Outlines (transformers) | Optional Outlines |
| **Vision** | ✅ (4o, 4-turbo) | ✅ (All Claude 3+) | ❌ | ✅ (model-dependent) | ✅ (vision models) | ❌ |
| **Embeddings** | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Seed Support** | ✅ | ❌ (warning) | ✅ | ✅ | ✅ (GGUF), ✅ (transformers) | ✅ |
| **Timeout Support** | ✅ | ✅ | ✅ | ✅ | ❌ (warning) | ❌ (warning) |
| **Model Unloading** | N/A | N/A | ✅ | Automatic TTL | ✅ | ✅ |
| **Health Check** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Best For** | Production, latest models | Long context, reasoning | Local experimentation | Model development | Custom/fine-tuned models | Apple Silicon devices |

## Usage Patterns

### Creating Providers

```python
from abstractcore import create_llm

# Using factory (recommended)
llm = create_llm("openai", model="gpt-4o", api_key="sk-...")

# Using registry directly
from abstractcore.providers import create_provider
provider = create_provider("anthropic", model="claude-3-5-sonnet-latest", api_key="sk-ant-...")

# Direct instantiation (advanced)
from abstractcore.providers.ollama_provider import OllamaProvider
provider = OllamaProvider(model="llama3.2", base_url="http://localhost:11434")
```

### Model Selection

```python
# List available models for a provider
from abstractcore.providers import get_provider_registry
registry = get_provider_registry()
models = registry.get_available_models("ollama")
print(f"Available: {models}")

# Check if model exists before creating provider
if "gpt-4o" in models:
    llm = create_llm("openai", model="gpt-4o")
```

### Generation Patterns

```python
# Simple generation
response = llm.generate("Hello!")
print(response.content)

# With conversation history
messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."},
]
response = llm.generate("Tell me more", messages=messages)

# With system prompt
response = llm.generate(
    "Write code",
    system_prompt="You are an expert Python developer"
)

# Streaming
for chunk in llm.generate("Long story", stream=True):
    print(chunk.content, end="", flush=True)

# With media
response = llm.generate(
    "Describe this image",
    media=["image.jpg", "document.pdf"]
)

# Structured output
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int
    occupation: str
person = llm.generate("Extract: John Doe, 35, Engineer", response_model=Person)
print(f"Name: {person.name}, Age: {person.age}")

# Tools (pass-through mode - default)
tools = [
    {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
]
response = llm.generate("Search for cats", tools=tools, execute_tools=False)
# Tool calls detected but not executed (for agentic workflows)

# Tools (execution mode)
response = llm.generate("Search for dogs", tools=tools, execute_tools=True)
# Tools executed automatically, results appended to response
```

## Integration Points

### Core Factory

The factory (`core/factory.py`) uses the registry to create providers:

```python
from abstractcore import create_llm
llm = create_llm("openai", model="gpt-4o")
# Internally calls: create_provider("openai", model="gpt-4o")
```

### Tools System

Providers integrate with the universal tool system:

```python
from abstractcore.tools import UniversalToolHandler

# Automatic detection in BaseProvider
self.tool_handler = UniversalToolHandler(model_name)

# Capabilities
if self.tool_handler.supports_native:
    # Use provider's native tools (OpenAI, Anthropic)
    call_params["tools"] = self.tool_handler.prepare_tools_for_native(tools)
elif self.tool_handler.supports_prompted:
    # Use prompted format (Ollama, LMStudio, HuggingFace, MLX)
    tool_prompt = self.tool_handler.format_tools_prompt(tools)
```

### Media Handlers

Providers use media handlers for vision/multimodal support:

```python
from abstractcore.media.handlers import OpenAIMediaHandler, AnthropicMediaHandler, LocalMediaHandler

# Cloud providers (OpenAI, Anthropic, LMStudio with vision)
media_handler = OpenAIMediaHandler(model_capabilities)
multimodal_message = media_handler.create_multimodal_message(prompt, media)

# Local providers (Ollama, HuggingFace, MLX)
media_handler = LocalMediaHandler("ollama", model_capabilities, model_name=model)
multimodal_message = media_handler.create_multimodal_message(prompt, media)
```

### Event System

All providers emit events for observability.

→ See [events/](../events/README.md) for complete event system documentation and subscription patterns

### Structured Output Handler

Providers use the structured output handler for validation:

```python
from abstractcore.structured import StructuredOutputHandler

# In BaseProvider.generate_with_telemetry()
if response_model:
    handler = StructuredOutputHandler(retry_strategy=retry_strategy)
    return handler.generate_structured(
        provider=self,
        prompt=prompt,
        response_model=response_model,
        ...
    )
```

## Best Practices

### Model Selection

**Cloud APIs (Production)**:
```python
# OpenAI: Best for latest features, widest model selection
llm = create_llm("openai", model="gpt-4o")  # Vision + tools + structured

# Anthropic: Best for long context, reasoning, safety
llm = create_llm("anthropic", model="claude-3-5-sonnet-latest")  # 200K context
```

**Local Servers (Development/Privacy)**:
```python
# Ollama: Best for experimentation, easy setup
llm = create_llm("ollama", model="qwen3:4b")  # Quick start

# LMStudio: Best for development, model comparison
llm = create_llm("lmstudio", model="qwen3-4b-2507")  # UI-based model management
```

**Local Libraries (Offline/Custom)**:
```python
# HuggingFace: Best for custom models, fine-tuning, GGUF
llm = create_llm("huggingface", model="unsloth/Qwen3-4B-GGUF")  # Maximum flexibility

# MLX: Best for Apple Silicon, optimized performance
llm = create_llm("mlx", model="mlx-community/Qwen3-4B")  # M1/M2/M3/M4 optimized
```

### API Key Management

```python
# Method 1: Environment variables (recommended)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
llm = create_llm("openai", model="gpt-4o")

# Method 2: Direct parameter
llm = create_llm("openai", model="gpt-4o", api_key="sk-...")

# Method 3: Configuration file (centralized config)
from abstractcore.config import get_config_manager
cfg = get_config_manager()
cfg.set_api_key("openai", "sk-...")
llm = create_llm("openai", model="gpt-4o")
```

### Error Handling

```python
from abstractcore.exceptions import (
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError,
    ProviderAPIError
)

try:
    llm = create_llm("openai", model="gpt-4o")
    response = llm.generate("Hello")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
    # Suggestion: Use llm.list_available_models() to see options
except AuthenticationError as e:
    print(f"Auth failed: {e}")
    # Check API key
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Implement retry with backoff
except ProviderAPIError as e:
    print(f"API error: {e}")
    # Handle generic provider error
```

### Memory Management

```python
# Local models (HuggingFace, MLX)
llm = create_llm("mlx", model="mlx-community/Qwen3-4B")
response = llm.generate("Hello")
llm.unload()  # Free model from memory immediately
del llm       # Remove reference

# Ollama (server-side)
llm = create_llm("ollama", model="qwen3:4b")
response = llm.generate("Hello")
llm.unload()  # Request server to unload model (keep_alive=0)

# Cloud APIs (no memory management needed)
llm = create_llm("openai", model="gpt-4o")
response = llm.generate("Hello")
# No unload() needed - server manages memory
```

### Performance Optimization

```python
# Use smaller models for simple tasks
simple_llm = create_llm("ollama", model="qwen3:4b")  # Fast, low memory

# Use larger models for complex tasks
complex_llm = create_llm("ollama", model="qwen3:70b")  # Slower, better quality

# Enable GPU acceleration (GGUF models)
gpu_llm = create_llm("huggingface", model="GGUF-model", n_gpu_layers=-1)

# Batch processing
prompts = ["Question 1", "Question 2", "Question 3"]
responses = [llm.generate(p) for p in prompts]

# Streaming for long outputs
for chunk in llm.generate("Write essay", stream=True):
    print(chunk.content, end="", flush=True)
```

## Common Pitfalls

### Authentication Issues

❌ **Wrong**:
```python
llm = create_llm("openai", model="gpt-4o")  # Missing API key
```

✅ **Correct**:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
llm = create_llm("openai", model="gpt-4o")
```

### Rate Limits

❌ **Wrong**:
```python
for i in range(1000):
    response = llm.generate(f"Task {i}")  # No rate limit handling
```

✅ **Correct**:
```python
from abstractcore.core.retry import RetryConfig
retry_config = RetryConfig(max_retries=3, initial_delay=1.0)
llm = create_llm("openai", model="gpt-4o", retry_config=retry_config)

for i in range(1000):
    try:
        response = llm.generate(f"Task {i}")
    except RateLimitError:
        time.sleep(60)  # Wait before retry
```

### Model Compatibility

❌ **Wrong**:
```python
# Using vision prompt with non-vision model
llm = create_llm("openai", model="gpt-3.5-turbo")
response = llm.generate("Describe image", media=["image.jpg"])  # Fails
```

✅ **Correct**:
```python
# Check model capabilities first
from abstractcore.architectures import supports_vision
if supports_vision("gpt-3.5-turbo"):
    response = llm.generate("Describe", media=["image.jpg"])
else:
    print("Model doesn't support vision")

# Or use vision-capable model
llm = create_llm("openai", model="gpt-4o")
response = llm.generate("Describe", media=["image.jpg"])
```

### Token Limits

❌ **Wrong**:
```python
llm = create_llm("openai", model="gpt-4o")
response = llm.generate("Write 10000 word essay")  # May exceed limits
```

✅ **Correct**:
```python
from abstractcore.architectures import get_context_limits
limits = get_context_limits("gpt-4o")
print(f"Max tokens: {limits['max_tokens']}")
print(f"Max output: {limits['max_output_tokens']}")

llm = create_llm("openai", model="gpt-4o", max_output_tokens=4096)
response = llm.generate("Write essay", max_output_tokens=4096)
```

### Timeout Issues

❌ **Wrong**:
```python
# Local provider with timeout (warning issued)
llm = create_llm("huggingface", model="GGUF-model", timeout=30)
```

✅ **Correct**:
```python
# Cloud provider with timeout
llm = create_llm("openai", model="gpt-4o", timeout=30)

# Local provider without timeout (correct)
llm = create_llm("huggingface", model="GGUF-model")  # No timeout needed
```

## Testing Strategy

### Unit Tests

```python
# Test provider initialization
def test_provider_init():
    llm = create_llm("openai", model="gpt-4o", api_key="test-key")
    assert llm.model == "gpt-4o"
    assert llm.provider == "openai"

# Test model listing
def test_list_models():
    from abstractcore.providers import get_provider_registry
    registry = get_provider_registry()
    models = registry.get_available_models("ollama")
    assert isinstance(models, list)

# Test generation
def test_generate():
    llm = create_llm("openai", model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    response = llm.generate("Say hello")
    assert response.content
    assert response.model == "gpt-4o"
```

### Integration Tests

```python
# Test with real providers
@pytest.mark.integration
def test_ollama_integration():
    llm = create_llm("ollama", model="qwen3:4b")
    response = llm.generate("What is 2+2?")
    assert "4" in response.content

@pytest.mark.integration
def test_streaming():
    llm = create_llm("ollama", model="qwen3:4b")
    chunks = list(llm.generate("Count to 5", stream=True))
    assert len(chunks) > 0
    full_content = "".join(c.content for c in chunks)
    assert len(full_content) > 0
```

### Provider Health Tests

```python
def test_provider_health():
    providers = ["ollama", "lmstudio"]
    for provider_name in providers:
        try:
            llm = create_llm(provider_name, model="test-model")
            health = llm.health(timeout=5.0)
            print(f"{provider_name}: {health['status']}")
            if health["status"]:
                print(f"  Models: {health['model_count']}")
        except Exception as e:
            print(f"{provider_name}: Offline ({e})")
```

## Public API

**Recommended Imports**:
```python
# Factory (primary interface)
from abstractcore import create_llm

# Registry functions
from abstractcore.providers import (
    create_provider,
    list_available_providers,
    get_provider_info,
    get_all_providers_with_models,
    get_provider_registry
)

# Direct provider access (advanced)
from abstractcore.providers.openai_provider import OpenAIProvider
from abstractcore.providers.anthropic_provider import AnthropicProvider
from abstractcore.providers.ollama_provider import OllamaProvider
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.providers.mlx_provider import MLXProvider

# Streaming (advanced)
from abstractcore.providers.streaming import UnifiedStreamProcessor

# Exceptions
from abstractcore.exceptions import (
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError,
    ProviderAPIError
)
```

**Core Types**:
```python
from abstractcore.core.types import GenerateResponse
from abstractcore.core.retry import RetryConfig, RetryManager
from pydantic import BaseModel  # For structured outputs
```

---

## Summary

The provider system provides a unified, production-ready interface for interacting with diverse LLM backends. Key design principles:

1. **Consistency**: Same interface works across all providers
2. **Observability**: Integrated telemetry, events, and logging
3. **Reliability**: Retry logic, circuit breakers, health checks
4. **Flexibility**: Supports streaming, tools, structured outputs, vision
5. **Developer-friendly**: Clear error messages, helpful warnings, comprehensive documentation

For questions or issues, refer to:
- **Error messages**: Include suggested fixes and available alternatives
- **Health checks**: Use `provider.health()` for diagnostics
- **Registry**: Use `get_all_providers_with_models()` for discovery
- **Examples**: See individual provider sections above

## Related Modules

**Direct dependencies**:
- [`core/`](../core/README.md) - Base provider abstractions, types
- [`exceptions/`](../exceptions/README.md) - Provider-specific errors
- [`events/`](../events/README.md) - Generation and tool execution events
- [`tools/`](../tools/README.md) - Tool execution framework
- [`media/`](../media/README.md) - Media processing and formatting
- [`structured/`](../structured/README.md) - Response model handling
- [`architectures/`](../architectures/README.md) - Model capability detection
- [`utils/`](../utils/README.md) - Token estimation, logging

**Used by**:
- [`core/`](../core/README.md) - Factory creates provider instances
- [`processing/`](../processing/README.md) - High-level processors
- [`apps/`](../apps/README.md) - Application integrations
- [`server/`](../server/README.md) - API endpoints
- [`embeddings/`](../embeddings/README.md) - Embedding providers

**Configuration**:
- [`config/`](../config/README.md) - Provider API keys and defaults
- [`assets/`](../assets/README.md) - Model capabilities data
