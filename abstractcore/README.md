# AbstractCore Framework

**The Unified Python Framework for Provider-Agnostic LLM Applications**

Version: 2.11.0 | Architecture: Layered, Modular | Philosophy: Offline-First, Production-Ready

---

## Overview

AbstractCore is a Python framework that provides a unified, provider-agnostic interface for building LLM applications. It supports cloud and local providers including OpenAI, Anthropic, OpenRouter, Ollama, LMStudio, MLX, HuggingFace, vLLM, and generic OpenAI-compatible endpoints.

**Key Philosophy**: Built primarily for open-source LLMs with complete offline capability while maintaining seamless cloud provider integration when needed.

---

## Quick Start

### Installation

```bash
# Core (small default)
pip install abstractcore

# Turnkey "everything" installs (pick one)
pip install "abstractcore[all-apple]"    # macOS/Apple Silicon (includes MLX, excludes vLLM)
pip install "abstractcore[all-non-mlx]"  # Linux/Windows/Intel Mac (excludes MLX and vLLM)
pip install "abstractcore[all-gpu]"      # Linux NVIDIA GPU (includes vLLM, excludes MLX)
```

### Hello World

```python
from abstractcore import create_llm

# Works with any provider - just change the provider name
llm = create_llm("ollama", model="qwen3:4b")
response = llm.generate("What is the capital of France?")
print(response.content)
```

### With Vision

```python
# Requires `pip install "abstractcore[media]"`
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "Describe this image",
    media=["photo.jpg"]
)
```

### With Tools

```python
from abstractcore import create_llm, tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return f"Weather in {city}: Sunny, 72°F"

llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)
```

---

## Architecture Overview

AbstractCore is organized into **16 specialized modules** across **5 architectural layers**, providing a clear separation of concerns and modular design:

```
┌─────────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER (3)                          │
│  User-Facing Interfaces, APIs, and Tools                        │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│ apps/        │ server/      │ processing/   │                   │
│ CLI tools    │ REST API     │ NLP workflows │                   │
└──────────────┴──────────────┴───────────────┴───────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE LAYER (4)                             │
│  High-Level Capabilities and Integrations                       │
├──────────────┬──────────────┬───────────────┬───────────────────┤
│ structured/  │ embeddings/  │ config/       │ tools/            │
│ Response     │ Vector       │ Settings      │ Function calling  │
│ validation   │ embeddings   │ management    │ & execution       │
└──────────────┴──────────────┴───────────────┴───────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PROVIDER LAYER (1)                            │
│  LLM Provider Implementations and Registry                      │
├─────────────────────────────────────────────────────────────────┤
│ providers/                                                       │
│ OpenAI, Anthropic, OpenRouter, Ollama, LMStudio, MLX, HF, vLLM │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     ADVANCED LAYER (2)                           │
│  Specialized Processing and Media Handling                      │
├──────────────┬──────────────────────────────────────────────────┤
│ compression/ │ media/                                            │
│ Glyph visual │ Multimodal content processing                    │
│ compression  │ (images, PDFs, documents, CSV, etc.)             │
└──────────────┴───────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      CORE LAYER (3)                              │
│  Foundation Abstractions and Base Interfaces                    │
├──────────────┬──────────────┬───────────────────────────────────┤
│ core/        │ architectures/│ utils/                           │
│ Base LLM,    │ Model         │ Token counting, logging,         │
│ factory,     │ capability    │ validation, web utilities        │
│ session      │ detection     │                                  │
└──────────────┴──────────────┴───────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   FOUNDATION LAYER (3)                           │
│  Infrastructure and Cross-Cutting Concerns                      │
├──────────────┬──────────────┬───────────────────────────────────┤
│ exceptions/  │ events/      │ assets/                           │
│ Error        │ Observability│ Static resources                 │
│ hierarchy    │ system       │ (model capabilities, schemas)    │
└──────────────┴──────────────┴───────────────────────────────────┘
```

---

## Core Concepts

### 1. Provider Abstraction

**One interface, all providers**. AbstractCore provides a unified API that works identically across all LLM providers:

```python
# Same code works with any provider
providers = ["openai", "anthropic", "ollama", "lmstudio", "huggingface", "mlx"]

for provider in providers:
    llm = create_llm(provider, model="auto")  # Auto-selects appropriate model
    response = llm.generate("Hello!")
    print(f"{provider}: {response.content}")
```

### 2. Structured Outputs

**Type-safe responses** with automatic validation using Pydantic models:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

llm = create_llm("ollama", model="qwen3:4b")
person = llm.generate(
    "Extract: John Doe, 35, Software Engineer",
    response_model=Person
)
# Returns validated Person instance
```

### 3. Universal Tool Calling

**Function calling that works everywhere**, with native API support where available and intelligent prompting otherwise:

```python
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Works with OpenAI (native), Ollama (prompted), etc.
response = llm.generate("What is 5 + 3?", tools=[calculate_sum])
```

### 4. Media Processing

**Unified multimodal support** across all providers with automatic format detection:

```python
# Images, PDFs, CSV, Excel, JSON - all work the same way
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "Analyze these documents",
    media=["report.pdf", "chart.png", "data.xlsx"]
)
```

### 5. Session Management

**Persistent conversations** with automatic history tracking, compaction, and serialization:

```python
from abstractcore import BasicSession

session = BasicSession(llm, system_prompt="You are helpful")
session.generate("Hello!")
session.generate("Tell me a joke")
session.save("conversation.json")  # Complete history preserved
```

---

## Module Guide

### Foundation Layer

#### [exceptions/](exceptions/README.md) - Error Handling

**Purpose**: Structured exception hierarchy for precise error handling

**Key Classes**:
- `AbstractCoreError` - Base exception for all framework errors
- `ProviderError`, `ModelNotFoundError`, `AuthenticationError`, `RateLimitError`
- `UnsupportedFeatureError`, `FileProcessingError`, `ToolExecutionError`

**Quick Example**:
```python
from abstractcore.exceptions import ModelNotFoundError, RateLimitError

try:
    llm = create_llm("openai", model="gpt-4o")
    response = llm.generate("Hello")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except RateLimitError as e:
    print(f"Rate limited: {e}")
```

→ [Detailed Documentation](exceptions/README.md)

---

#### [events/](events/README.md) - Observability System

**Purpose**: Event-driven observability for monitoring and debugging

**Key Features**:
- Event types: GENERATION_STARTED, GENERATION_COMPLETED, TOOL_STARTED, TOOL_COMPLETED, ERROR
- Global and scoped event subscriptions
- Structured event data with metadata
- Real-time monitoring capabilities

**Quick Example**:
```python
from abstractcore.events import subscribe, EventType

def log_generation(event):
    print(f"Generation: {event['data']['model']} - {event['data']['gen_time']}ms")

subscribe(EventType.GENERATION_COMPLETED, log_generation)
```

→ [Detailed Documentation](events/README.md)

---

#### [assets/](assets/README.md) - Static Resources

**Purpose**: Central repository for model capabilities, schemas, and configuration data

**Key Assets**:
- `model_capabilities.json` - Comprehensive model feature matrix (137+ models)
- JSON schemas for structured validation
- Provider metadata and defaults

**Quick Example**:
```python
from abstractcore.assets import get_model_capabilities

capabilities = get_model_capabilities("gpt-4o")
print(capabilities["vision"])  # True
print(capabilities["max_tokens"])  # 128000
```

→ [Detailed Documentation](assets/README.md)

---

### Core Layer

#### [core/](core/README.md) - Base Abstractions

**Purpose**: Foundational abstractions for LLM interaction and conversation management

**Key Components**:
- `AbstractCoreInterface` - Base class for all providers
- `create_llm()` - Factory function for provider instantiation
- `BasicSession` - Conversation management with auto-compaction
- `RetryManager` - Production-ready retry with circuit breakers
- `Message`, `GenerateResponse` - Core data structures

**Quick Example**:
```python
from abstractcore import create_llm, BasicSession

llm = create_llm("ollama", model="qwen3:4b", max_tokens=32000)

# Simple generation
response = llm.generate("Hello!")

# Session-based conversation
session = BasicSession(llm, auto_compact=True)
session.generate("What is Python?")
session.generate("Tell me more")
session.save("conversation.json")
```

→ [Detailed Documentation](core/README.md)

---

#### [architectures/](architectures/README.md) - Model Detection

**Purpose**: Automatic model architecture and capability detection

**Key Features**:
- Architecture detection from model names (qwen3, llama3, gemma, etc.)
- Capability queries (vision, tools, streaming support)
- Context limit detection
- Tool format detection

**Quick Example**:
```python
from abstractcore.architectures import (
    detect_architecture,
    supports_vision,
    get_context_limits
)

arch = detect_architecture("qwen3:4b-instruct-2507-q4_K_M")
print(arch)  # "qwen3"

vision = supports_vision("gpt-4o")
print(vision)  # True

limits = get_context_limits("claude-3-5-sonnet")
print(limits["max_tokens"])  # 200000
```

→ [Detailed Documentation](architectures/README.md)

---

#### [utils/](utils/README.md) - Cross-Cutting Utilities

**Purpose**: Shared utilities for token counting, logging, validation, and web operations

**Key Components**:
- `TokenUtils` - Universal token estimation across providers
- `structured_logging` - Centralized logging configuration
- `json_utils` - JSON repair and validation
- `web_utils` - HTTP utilities and content fetching
- `version` - Version management

**Quick Example**:
```python
from abstractcore.utils.token_utils import TokenUtils
from abstractcore.utils.structured_logging import configure_logging

# Token estimation
tokens = TokenUtils.estimate_tokens("Hello world", model_name="gpt-4o")
print(tokens)  # Accurate token count

# Logging configuration
import logging
configure_logging(
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    log_dir="logs"
)
```

→ [Detailed Documentation](utils/README.md)

---

### Provider Layer

#### [providers/](providers/README.md) - LLM Integrations

**Purpose**: Unified interface to all LLM providers with consistent capabilities

**Supported Providers**:
- **OpenAI**: GPT-3.5, GPT-4, GPT-5, vision, native tools
- **Anthropic**: Claude 3/3.5, vision, structured outputs
- **Ollama**: Local models, structured outputs (native), embeddings
- **LMStudio**: Local server, OpenAI-compatible API
- **HuggingFace**: Transformers + GGUF, vision models, optional Outlines
- **MLX**: Apple Silicon optimized, optional Outlines

**Quick Example**:
```python
from abstractcore import create_llm
from abstractcore.providers import get_all_providers_with_models

# List all available providers and models
providers = get_all_providers_with_models()
print(f"Total models: {sum(p['model_count'] for p in providers)}")

# Create provider instances
openai_llm = create_llm("openai", model="gpt-4o-mini")
local_llm = create_llm("ollama", model="qwen3:4b")
mlx_llm = create_llm("mlx", model="mlx-community/Qwen3-4B")

# Same interface for all
for llm in [openai_llm, local_llm, mlx_llm]:
    response = llm.generate("Hello!")
```

→ [Detailed Documentation](providers/README.md)

---

### Advanced Layer

#### [compression/](compression/README.md) - Glyph Visual-Text Compression

**Purpose**: Revolutionary 3-4x token compression through visual text rendering

**Key Features**:
- Lower effective token usage for long documents (best-effort; validate for your domain)
- Best-effort quality validation with automatic fallback
- Transparent operation with automatic mode
- Configurable compression policies

**Quick Example**:
```python
from abstractcore import create_llm

llm = create_llm("ollama", model="llama3.2-vision:11b")

# Automatic compression for large documents
response = llm.generate(
    "Analyze this large document",
    media=["large_report.pdf"],  # Automatically compressed if beneficial
    glyph_compression="auto"     # "auto", "always", "never"
)

# Check compression stats
if response.metadata.get('compression_used'):
    stats = response.metadata['compression_stats']
    print(f"Compression ratio: {stats['compression_ratio']}x")
```

→ [Detailed Documentation](compression/README.md)

---

#### [media/](media/README.md) - Multimodal Processing

**Purpose**: Unified media handling for images, documents, and structured data

**Supported Formats**:
- **Images**: PNG, JPEG, GIF, WEBP, BMP, TIFF (up to 10MB)
- **Documents**: PDF, DOCX, XLSX, PPTX (text extraction)
- **Data**: CSV, TSV, JSON, XML (rendering and analysis)
- **Text**: TXT, MD, HTML (direct processing)

**Quick Example**:
```python
llm = create_llm("openai", model="gpt-4o")

# Single media
response = llm.generate("Describe this", media=["chart.png"])

# Multiple media types
response = llm.generate(
    "Analyze these business documents",
    media=["report.pdf", "sales.xlsx", "chart.png", "data.csv"]
)

# Vision automatically optimized for model
local_llm = create_llm("ollama", model="qwen2.5vl:7b")
response = local_llm.generate(
    "Extract text from image",
    media=["screenshot.png"]  # Automatically optimized for qwen2.5vl
)
```

→ [Detailed Documentation](media/README.md)

---

### Feature Layer

#### [structured/](structured/README.md) - Response Validation

**Purpose**: Type-safe structured outputs with automatic validation and retry

**Key Features**:
- Native support detection (Ollama, LMStudio, HuggingFace GGUF, optional Outlines)
- Prompted fallback with validation retry
- 100% success rate (tested across providers)
- Enum simplification for LLM clarity
- JSON self-healing

**Quick Example**:
```python
from pydantic import BaseModel
from typing import List
from enum import Enum

class Status(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class Task(BaseModel):
    title: str
    status: Status
    priority: int

class Project(BaseModel):
    name: str
    tasks: List[Task]

llm = create_llm("ollama", model="qwen3:4b")
project = llm.generate(
    "Extract project: Website Redesign with tasks...",
    response_model=Project
)
# Returns validated Project instance with type safety
```

→ [Detailed Documentation](structured/README.md)

---

#### [embeddings/](embeddings/README.md) - Vector Embeddings

**Purpose**: Text embeddings for semantic search and RAG applications

**Key Features**:
- Multiple embedding models (SentenceTransformers, OpenAI, HuggingFace)
- Batch processing support
- Similarity computation (cosine, dot product, euclidean)
- Provider-agnostic interface

**Quick Example**:
```python
from abstractcore.embeddings import EmbeddingManager

embedder = EmbeddingManager(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Single embedding
embedding = embedder.embed("Hello world")

# Batch embeddings
embeddings = embedder.embed_batch([
    "Python is great",
    "JavaScript is popular",
    "Rust is fast"
])

# Similarity computation
similarity = embedder.compute_similarity(
    embeddings[0],
    embeddings[1]
)
```

→ [Detailed Documentation](embeddings/README.md)

---

#### [config/](config/README.md) - Configuration Management

**Purpose**: Centralized configuration for defaults, API keys, and settings

**Key Features**:
- Global and app-specific defaults
- API key management
- Logging configuration
- Vision provider fallback
- Interactive setup

**Quick Example**:
```bash
# CLI configuration
abstractcore --status  # View current config
abstractcore --set-global-default ollama/qwen3:4b
abstractcore --set-api-key openai sk-...
abstractcore --enable-file-logging
```

```python
# Programmatic configuration
from abstractcore.config import get_config_manager

config = get_config_manager()
config.set("providers.openai.api_key", "sk-...")
config.set("defaults.global.model", "ollama/qwen3:4b")
```

→ [Detailed Documentation](config/README.md)

---

#### [tools/](tools/README.md) - Universal Tool System

**Purpose**: Provider-agnostic tool calling with native API support and prompted fallback

**Key Features**:
- 5+ tool formats (OpenAI, Qwen3, LLaMA3, XML, Gemma)
- Automatic format detection
- Built-in tools (file ops, web search, commands)
- Tag rewriting for agentic CLIs
- Streaming tool detection

**Quick Example**:
```python
from abstractcore import create_llm, tool
from abstractcore.tools import (
    list_files, search_files, read_file, write_file,
    web_search, fetch_url, execute_command
)

# Custom tool
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Built-in tools
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "Search for Python documentation and list files in current directory",
    tools=[web_search, list_files]
)

# Works with prompted models too
local_llm = create_llm("ollama", model="qwen3:4b")
response = local_llm.generate(
    "Calculate 5 + 3",
    tools=[calculate_sum]
)
```

→ [Detailed Documentation](tools/README.md)

---

### Application Layer

#### [processing/](processing/README.md) - NLP Workflows

**Purpose**: High-performance text processing with structured outputs and automatic chunking

**Available Processors**:
- `BasicSummarizer` - Document summarization (6 styles, 4 lengths)
- `BasicExtractor` - Knowledge extraction (JSON-LD, 9 entity types)
- `BasicJudge` - Quality evaluation (9 criteria, 1-5 scoring)
- `BasicIntentAnalyzer` - Intent analysis (17 types, deception detection)
- `BasicDeepSearch` - Autonomous web research (4-stage pipeline)

**Quick Example**:
```python
from abstractcore.processing import (
    BasicSummarizer, BasicExtractor, BasicJudge,
    BasicIntentAnalyzer, BasicDeepSearch
)

# Summarization
summarizer = BasicSummarizer()
result = summarizer.summarize(
    text="Long document...",
    style="executive",
    length="detailed"
)

# Knowledge extraction
extractor = BasicExtractor()
knowledge = extractor.extract(
    text="OpenAI created GPT-4",
    output_format="jsonld"
)

# Quality evaluation
judge = BasicJudge()
assessment = judge.evaluate(
    content="Code review...",
    context="code review"
)

# Intent analysis
analyzer = BasicIntentAnalyzer()
intent = analyzer.analyze_intent(
    text="I was wondering if...",
    depth="underlying"
)

# Deep research
searcher = BasicDeepSearch()
report = searcher.research(
    query="Latest AI developments",
    max_sources=15
)
```

→ [Detailed Documentation](processing/README.md)

---

#### [server/](server/README.md) - REST API

**Purpose**: OpenAI-compatible HTTP API server for universal LLM access

**Key Endpoints**:
- `POST /v1/chat/completions` - Chat completions (text, vision, tools)
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/responses` - OpenAI Responses API
- `GET /v1/models` - List available models
- `GET /providers` - Provider metadata

**Quick Example**:
```bash
# Start server
python -m abstractcore.server --debug

# Use with OpenAI client
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

→ [Detailed Documentation](server/README.md)

---

#### [apps/](apps/README.md) - CLI Applications

**Purpose**: Production-ready command-line tools for common LLM tasks

**Available Apps**:
- `summarizer` - Document summarization
- `extractor` - Knowledge extraction
- `judge` - Text evaluation
- `intent` - Intent analysis
- `deepsearch` - Autonomous research

**Quick Example**:
```bash
# Document summarization
summarizer document.pdf --style executive --length brief

# Entity extraction
extractor research_paper.pdf --format json-ld --focus technology

# Text evaluation
judge essay.txt --criteria clarity,soundness --context "academic writing"

# Intent analysis
intent conversation.txt --focus-participant user --depth comprehensive

# Deep research
deepsearch "What are the latest advances in quantum computing?" --max-sources 20
```

→ [Detailed Documentation](apps/README.md)

---

## Common Workflows

### 1. Local Development, Cloud Production

```python
from abstractcore import create_llm

# Development (free, local)
dev_llm = create_llm("ollama", model="qwen3:4b")

# Production (high quality, cloud)
prod_llm = create_llm("openai", model="gpt-4o-mini")

# Same code for both
def process_request(llm, text):
    return llm.generate(f"Summarize: {text}")

result_dev = process_request(dev_llm, "...")
result_prod = process_request(prod_llm, "...")
```

### 2. Vision Analysis Pipeline

```python
llm = create_llm("openai", model="gpt-4o")

# Step 1: Describe images
description = llm.generate(
    "Describe these product images",
    media=["product1.jpg", "product2.jpg"]
)

# Step 2: Extract structured data
from pydantic import BaseModel
class Product(BaseModel):
    name: str
    features: list[str]
    price_estimate: float

products = llm.generate(
    f"Extract product info from: {description.content}",
    response_model=list[Product]
)

# Step 3: Generate recommendations
recommendations = llm.generate(
    f"Suggest improvements for: {products}"
)
```

### 3. Document Analysis with Extraction and Judgment

```python
from abstractcore.processing import BasicExtractor, BasicJudge

# Extract knowledge
extractor = BasicExtractor()
knowledge = extractor.extract(
    text=document_content,
    output_format="jsonld"
)

# Evaluate quality
judge = BasicJudge()
assessment = judge.evaluate(
    content=str(knowledge),
    context="knowledge graph quality",
    focus="completeness, accuracy"
)

# Refine if needed
if assessment.overall_score >= 4:
    refined = extractor.refine_extraction(
        text=document_content,
        previous_extraction=knowledge,
        length="detailed"
    )
```

### 4. Multi-Provider Fallback

```python
from abstractcore import create_llm
from abstractcore.exceptions import ProviderAPIError

providers = [
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-haiku"),
    ("ollama", "qwen3:4b")
]

for provider, model in providers:
    try:
        llm = create_llm(provider, model=model)
        response = llm.generate("Analyze this text...")
        break  # Success
    except ProviderAPIError as e:
        print(f"{provider} failed: {e}")
        continue  # Try next provider
```

### 5. RAG Pipeline

```python
from abstractcore import create_llm
from abstractcore.embeddings import EmbeddingManager

# Create embeddings for knowledge base
embedder = EmbeddingManager()
kb_texts = ["Document 1...", "Document 2...", "Document 3..."]
kb_embeddings = embedder.embed_batch(kb_texts)

# Query
query = "What is the process for...?"
query_embedding = embedder.embed(query)

# Find most relevant documents
similarities = [
    embedder.compute_similarity(query_embedding, emb)
    for emb in kb_embeddings
]
top_docs = sorted(zip(kb_texts, similarities), key=lambda x: x[1], reverse=True)[:3]

# Generate answer with context
llm = create_llm("openai", model="gpt-4o-mini")
context = "\n\n".join([doc for doc, _ in top_docs])
answer = llm.generate(
    f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on context:"
)
```

---

## Navigation Index

### By Functionality

**LLM Interaction**:
- [core/](core/README.md) - Base abstractions, factory, session management
- [providers/](providers/README.md) - Provider implementations and registry

**Media & Compression**:
- [media/](media/README.md) - Images, documents, CSV processing
- [compression/](compression/README.md) - Glyph visual-text compression

**Structured Data**:
- [structured/](structured/README.md) - Response validation with Pydantic
- [embeddings/](embeddings/README.md) - Vector embeddings for search

**Tool Calling**:
- [tools/](tools/README.md) - Universal tool system, built-in tools

**Text Processing**:
- [processing/](processing/README.md) - Summarization, extraction, evaluation, research

**Infrastructure**:
- [exceptions/](exceptions/README.md) - Error handling
- [events/](events/README.md) - Observability and monitoring
- [config/](config/README.md) - Configuration management
- [utils/](utils/README.md) - Token counting, logging, validation

**Detection & Capabilities**:
- [architectures/](architectures/README.md) - Model architecture detection
- [assets/](assets/README.md) - Model capabilities database

**Applications**:
- [apps/](apps/README.md) - CLI tools
- [server/](server/README.md) - REST API server

---

### By Use Case

**Getting Started**:
1. [core/](core/README.md) - Learn `create_llm()` and basic generation
2. [providers/](providers/README.md) - Choose and configure providers
3. [config/](config/README.md) - Set up defaults and API keys

**Building Applications**:
1. [processing/](processing/README.md) - High-level text processing
2. [structured/](structured/README.md) - Type-safe responses
3. [tools/](tools/README.md) - Add function calling
4. [server/](server/README.md) - HTTP API integration

**Advanced Features**:
1. [media/](media/README.md) - Multimodal processing
2. [compression/](compression/README.md) - Token optimization
3. [embeddings/](embeddings/README.md) - Semantic search
4. [events/](events/README.md) - Monitoring and debugging

**Troubleshooting**:
1. [exceptions/](exceptions/README.md) - Error types and handling
2. [utils/](utils/README.md) - Debugging utilities
3. [architectures/](architectures/README.md) - Capability detection

---

## Design Principles

### 1. Provider Agnostic

Write once, run anywhere. AbstractCore abstracts provider differences behind a unified interface.

### 2. Offline First

Built primarily for open-source LLMs with complete offline capability. Cloud providers are first-class but optional.

### 3. Production Ready

Comprehensive error handling, retry strategies, circuit breakers, structured logging, and type safety throughout.

### 4. Modular Architecture

Clear separation of concerns across 16 specialized modules in 5 architectural layers.

### 5. Zero Configuration

Sensible defaults enable immediate use. Deep configuration available when needed.

### 6. Type Safe

Full Pydantic integration for structured outputs, comprehensive type hints, runtime validation.

### 7. Observable

Event-driven architecture with structured logging for complete visibility into operations.

---

## Best Practices

### DO

1. **Use the factory**: `create_llm()` instead of direct provider imports
2. **Configure defaults**: Use `abstractcore --set-global-default` for convenience
3. **Handle exceptions**: Catch specific exceptions for targeted error handling
4. **Leverage structured outputs**: Use Pydantic models instead of parsing text
5. **Check capabilities**: Use `supports_vision()` before sending images
6. **Monitor events**: Subscribe to events for observability
7. **Validate configuration**: Call `validate_token_constraints()` after creation
8. **Use sessions for conversations**: `BasicSession` for automatic history management

### DON'T

1. **Instantiate providers directly**: Bypasses registry and validation
2. **Ignore token limits**: Always set `max_tokens` and `max_output_tokens`
3. **Skip error handling**: Always wrap provider calls in try/except
4. **Mix manual and session history**: Use `session.add_message()` consistently
5. **Ignore confidence scores**: Check confidence in processor outputs
6. **Hardcode provider names**: Use configuration system for flexibility
7. **Skip fallbacks**: Always have a backup plan for provider failures

---

## Performance Considerations

### Token Management

```python
# Budget-conscious approach
llm = create_llm(
    "openai",
    model="gpt-4o-mini",
    max_tokens=8000,        # Total budget
    max_output_tokens=2000  # Reserve for output
)
```

### Chunking Strategy

```python
# For large documents with large-context models
processor = BasicSummarizer(
    max_chunk_size=15000,
    max_tokens=100000
)

# For standard models
processor = BasicSummarizer(
    max_chunk_size=6000,
    max_tokens=32000
)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

llm = create_llm("ollama", model="qwen3:4b")

def process_doc(text):
    return llm.generate(f"Summarize: {text}")

texts = ["Doc1...", "Doc2...", "Doc3..."]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_doc, texts))
```

---

## Testing Strategy

AbstractCore includes comprehensive test coverage across all modules:

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/core/
pytest tests/providers/
pytest tests/structured/

# Run with coverage
pytest --cov=abstractcore --cov-report=html
```

**Test Categories**:
- **Unit tests**: Individual function and class testing
- **Integration tests**: Provider and module integration
- **Performance tests**: Benchmark critical paths
- **Regression tests**: Ensure backward compatibility

---

## Migration and Compatibility

### From AbstractLLM

AbstractCore was previously known as AbstractLLM (versions < 2.4.0):

```python
# Old (AbstractLLM)
from abstractllm import create_llm

# New (AbstractCore)
from abstractcore import create_llm

# API is identical - just rename package
```

### Version Compatibility

- **Python**: 3.9+ (tested on 3.9, 3.10, 3.11, 3.12)
- **Pydantic**: 2.0+
- **Provider SDKs**: Latest stable versions

---

## Contributing

We welcome contributions! Key areas:

1. **New providers**: Implement `AbstractCoreInterface`
2. **New processors**: Add to `processing/`
3. **Tool integrations**: Extend `tools/common_tools.py`
4. **Documentation**: Improve module READMEs
5. **Tests**: Increase coverage

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## Getting Help

### Documentation

- **Module READMEs**: Detailed documentation for each module (linked above)
- **Main README**: `/Users/albou/projects/abstractcore/README.md` - User-facing guide
- **API Reference**: `docs/api-reference.md` - Complete API documentation
- **Examples**: `examples/` - Working code samples

### Support

- **Issues**: https://github.com/lpalbou/AbstractCore/issues
- **Discussions**: https://github.com/lpalbou/AbstractCore/discussions
- **Email**: contact@abstractcore.ai

---

## License

MIT License - see [LICENSE](../LICENSE) file for details.

---

## Summary

AbstractCore provides a **comprehensive, production-ready framework** for building LLM applications with:

- **16 specialized modules** across 5 architectural layers
- **6 provider integrations** covering 137+ models
- **Universal interfaces** for generation, tools, media, and structured outputs
- **Rich ecosystem** of processors for common NLP tasks
- **Production-grade** error handling, retry, and observability
- **Complete offline capability** with seamless cloud integration

**Start building**:

```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3:4b")
response = llm.generate("Hello, AbstractCore!")
```

**Explore the modules** using the navigation index above, or dive into the [core documentation](core/README.md) to get started.

---

**AbstractCore** - One framework, all LLMs. Focus on building, not managing API differences.

Note: Model count includes capabilities database entries and may include models across multiple providers and quantization variants.
