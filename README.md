# AbstractCore

[![PyPI version](https://img.shields.io/pypi/v/abstractcore.svg)](https://pypi.org/project/abstractcore/)
[![Python Version](https://img.shields.io/pypi/pyversions/abstractcore)](https://pypi.org/project/abstractcore/)
[![license](https://img.shields.io/github/license/lpalbou/abstractcore)](https://github.com/lpalbou/abstractcore/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/lpalbou/abstractcore?style=social)](https://github.com/lpalbou/abstractcore/stargazers)

A unified Python library for interaction with multiple Large Language Model (LLM) providers.

**Write once, run everywhere.**

## Quick Start

### Installation

```bash
# macOS/Apple Silicon (includes MLX)
pip install abstractcore[all]

# Linux/Windows (excludes MLX)
pip install abstractcore[all-non-mlx]
```

### Basic Usage

```python
from abstractcore import create_llm

# Works with any provider - just change the provider name
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
response = llm.generate("What is the capital of France?")
print(response.content)
```

### Deterministic Generation

```python
from abstractcore import create_llm

# Deterministic outputs with seed + temperature=0
llm = create_llm("openai", model="gpt-3.5-turbo", seed=42, temperature=0.0)

# These will produce identical outputs
response1 = llm.generate("Write exactly 3 words about coding")
response2 = llm.generate("Write exactly 3 words about coding")
print(f"Response 1: {response1.content}")  # "Innovative, challenging, rewarding."
print(f"Response 2: {response2.content}")  # "Innovative, challenging, rewarding."
```

### Tool Calling

```python
from abstractcore import create_llm, tool

@tool
def get_current_weather(city: str):
    """Fetch current weather for a given city."""
    return f"Weather in {city}: 72°F, Sunny"

llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate(
    "What's the weather like in San Francisco?",
    tools=[get_current_weather]
)
print(response.content)
```

### Tool Execution Modes

AbstractCore supports two tool execution modes:

**Mode 1: Passthrough (Default)** - Returns raw tool call tags for downstream processing

```python
from abstractcore import create_llm
from abstractcore.tools import tool

@tool(name="get_weather", description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 22°C"

llm = create_llm("ollama", model="qwen3:4b")  # execute_tools=False by default
response = llm.generate("What's the weather in Paris?", tools=[get_weather])
# response.content contains raw tool call tags: <|tool_call|>...
# Downstream runtime (AbstractRuntime, Codex, Claude Code) parses and executes
```

**Use case**: Agent loops, AbstractRuntime, Codex, Claude Code, custom orchestration

**Mode 2: Direct Execution** - AbstractCore executes tools and returns results

```python
from abstractcore import create_llm
from abstractcore.tools import tool
from abstractcore.tools.registry import register_tool

@tool(name="get_weather", description="Get weather for a city")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 22°C"

register_tool(get_weather)  # Required for direct execution

llm = create_llm("ollama", model="qwen3:4b", execute_tools=True)
response = llm.generate("What's the weather in Paris?", tools=[get_weather])
# response.content contains executed tool results
```

**Use case**: Simple scripts, single-turn tool use

> **Note**: The `@tool` decorator creates metadata but does NOT register globally. Tools are passed explicitly to `generate()`. Use `register_tool()` only when using direct execution mode.

### Response Object (GenerateResponse)

Every LLM generation returns a **GenerateResponse** object with consistent structure across all providers:

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("Explain quantum computing in simple terms")

# Core response data
print(f"Content: {response.content}")               # Generated text
print(f"Model: {response.model}")                   # Model used
print(f"Finish reason: {response.finish_reason}")   # Why generation stopped

# Consistent token access across ALL providers (NEW in v2.4.7)
print(f"Input tokens: {response.input_tokens}")     # Always available
print(f"Output tokens: {response.output_tokens}")   # Always available  
print(f"Total tokens: {response.total_tokens}")     # Always available

# Generation time tracking (NEW in v2.4.7)
print(f"Generation time: {response.gen_time}ms")    # Always available (rounded to 1 decimal)

# Advanced access
print(f"Tool calls: {response.tool_calls}")         # Tools executed (if any)
print(f"Raw usage: {response.usage}")               # Provider-specific token data
print(f"Metadata: {response.metadata}")             # Additional context

# Comprehensive summary
print(f"Summary: {response.get_summary()}")         # "Model: gpt-4o-mini | Tokens: 117 | Time: 1234.5ms"
```

**Token Count Sources:**
- **Provider APIs**: OpenAI, Anthropic, LMStudio (native API token counts)
- **AbstractCore Calculation**: MLX, HuggingFace (using `token_utils.py`)
- **Mixed Sources**: Ollama (combination of provider and calculated tokens)

**Backward Compatibility**: Legacy `prompt_tokens` and `completion_tokens` keys remain available in `response.usage` dictionary.

### Built-in Tools

AbstractCore includes a comprehensive set of ready-to-use tools for common tasks:

> Note: `abstractcore.tools.common_tools` requires `abstractcore[tools]` (BeautifulSoup, lxml, web search backends, etc.).

```python
from abstractcore.tools.common_tools import fetch_url, search_files, read_file

# Intelligent web content fetching with automatic parsing
result = fetch_url("https://api.github.com/repos/python/cpython")
# Automatically detects JSON, HTML, images, PDFs, etc. and provides structured analysis

# File system operations
files = search_files("def.*fetch", ".", file_pattern="*.py")  # Find function definitions
content = read_file("config.json")  # Read file contents

# Use with any LLM
llm = create_llm("anthropic", model="claude-3-5-haiku-latest")
response = llm.generate(
    "Analyze this API response and summarize the key information",
    tools=[fetch_url]
)
```

**Available Tools:**
- `fetch_url` - Intelligent web content fetching with automatic content type detection and parsing
- `search_files` - Search for text patterns inside files using regex
- `list_files` - Find and list files by names/paths using glob patterns  
- `read_file` - Read file contents with optional line range selection
- `write_file` - Write content to files with directory creation
- `edit_file` - Edit files using pattern matching and replacement
- `web_search` - Search the web using DuckDuckGo
- `execute_command` - Execute shell commands safely with security controls

### Session Management

```python
from abstractcore import BasicSession, create_llm

# Create a persistent conversation session
llm = create_llm("openai", model="gpt-4o-mini")
session = BasicSession(llm, system_prompt="You are a helpful assistant.")

# Add messages with metadata
session.add_message('user', 'Hello!', name='alice', location='Paris')
response = session.generate('What is Python?', name='bob')

# Save complete conversation with optional analytics
session.save('conversation.json')  # Basic save
session.save('analyzed.json', summary=True, assessment=True, facts=True)  # With analytics

# Load and continue conversation
loaded_session = BasicSession.load('conversation.json', provider=llm)
```

[Learn more about Session](docs/session.md)

### Interaction Tracing (Observability)

Enable complete observability of LLM interactions for debugging, compliance, and transparency:

```python
from abstractcore import create_llm
from abstractcore.core.session import BasicSession
from abstractcore.utils import export_traces

# Enable tracing on provider
llm = create_llm('openai', model='gpt-4o-mini', enable_tracing=True, max_traces=100)

# Or on session for automatic correlation
session = BasicSession(provider=llm, enable_tracing=True)

# Generate with custom metadata
response = session.generate(
    "Write Python code",
    step_type='code_generation',
    attempt_number=1
)

# Access complete trace
trace_id = response.metadata['trace_id']
trace = llm.get_traces(trace_id=trace_id)

# Full interaction context
print(f"Prompt: {trace['prompt']}")
print(f"Response: {trace['response']['content']}")
print(f"Tokens: {trace['response']['usage']['total_tokens']}")
print(f"Time: {trace['response']['generation_time_ms']}ms")
print(f"Custom metadata: {trace['metadata']}")

# Get all session traces
traces = session.get_interaction_history()

# Export to JSONL, JSON, or Markdown
export_traces(traces, format='markdown', file_path='workflow_trace.md')
```

**What's captured:**
- All prompts, system prompts, and conversation history
- Complete responses with token usage and timing
- Generation parameters (temperature, tokens, seed, etc.)
- Custom metadata for workflow tracking
- Tool calls and results

[Learn more about Interaction Tracing](docs/interaction-tracing.md)

### Async/Await Support

Execute concurrent LLM requests for batch operations, multi-provider comparisons, or non-blocking web applications. **Production-ready with validated 6-7.5x performance improvement** for concurrent requests.

```python
import asyncio
from abstractcore import create_llm

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")

    # Execute 3 requests concurrently (6-7x faster!)
    tasks = [
        llm.agenerate(f"Summarize {topic}")
        for topic in ["Python", "JavaScript", "Rust"]
    ]
    responses = await asyncio.gather(*tasks)

    for response in responses:
        print(response.content)

asyncio.run(main())
```

**Performance (Validated with Real Testing):**
- **Ollama**: 7.5x faster for concurrent requests
- **LMStudio**: 6.5x faster for concurrent requests
- **OpenAI**: 6.0x faster for concurrent requests
- **Anthropic**: 7.4x faster for concurrent requests
- **Average**: ~7x speedup across all providers

**Native Async vs Fallback:**
- **Native async** (httpx.AsyncClient): Ollama, LMStudio, OpenAI, Anthropic
- **Fallback** (asyncio.to_thread): MLX, HuggingFace
- All providers work seamlessly - fallback keeps event loop responsive

**Use Cases:**
- Batch operations with 6-7x speedup via parallel execution
- Multi-provider comparisons (query OpenAI and Anthropic simultaneously)
- FastAPI/async web frameworks integration
- Session async for conversation management

**Works with:**
- All 6 providers (OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace)
- Streaming via `async for chunk in llm.agenerate(..., stream=True)`
- Sessions via `await session.agenerate(...)`
- Zero breaking changes to sync API

**Learn async patterns:**

AbstractCore includes an educational [async CLI demo](examples/async_cli_demo.py) that demonstrates 8 core async/await patterns:
- Event-driven progress with GlobalEventBus
- Parallel tool execution with asyncio.gather()
- Proper async streaming pattern (await first, then async for)
- Non-blocking animations and user input

```bash
# Try the educational async demo
python examples/async_cli_demo.py --provider ollama --model qwen3:4b --stream
```

[Learn more in CLI docs](docs/acore-cli.md#async-cli-demo-educational-reference)

### Media Handling

AbstractCore provides unified media handling across all providers with automatic resolution optimization. Upload images, PDFs, and documents using the same simple API regardless of your provider.

```python
from abstractcore import create_llm

# Vision analysis - works with any vision model
# Images automatically processed at maximum supported resolution
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "What's in this image?",
    media=["photo.jpg"]  # Auto-resized to model's maximum capability
)

# Document analysis - works with any model
llm = create_llm("anthropic", model="claude-3.5-sonnet")
response = llm.generate(
    "Summarize this research paper",
    media=["research_paper.pdf"]
)

# Multiple files - mix images, PDFs, spreadsheets
response = llm.generate(
    "Analyze these business documents",
    media=["report.pdf", "chart.png", "data.xlsx"]
)

# Same code works with local models
llm = create_llm("ollama", model="qwen3-vl:8b")
response = llm.generate(
    "Describe this screenshot",
    media=["screenshot.png"]  # Auto-optimized for qwen3-vl
)
```

**Key Features:**
- **Smart Resolution**: Automatically uses maximum resolution supported by each model
- **Format Support**: PNG, JPEG, GIF, WEBP, BMP, TIFF images; PDF, TXT, MD, CSV, TSV, JSON documents
- **Office Documents**: DOCX, XLSX, PPT (with `pip install abstractcore[all]`)
- **Vision Optimization**: Model-specific image processing for vision results

**Provider compatibility:**
- **High-resolution vision**: GPT-4o (up to 4096x4096), Claude 3.5 Sonnet (up to 1568x1568)
- **Local models**: qwen3-vl (up to 3584x3584), gemma3:4b, llama3.2-vision
- **All models**: Automatic text extraction for non-vision models

[Learn more about Media Handling](docs/media-handling-system.md)

### Glyph Visual-Text Compression (🧪 EXPERIMENTAL)

> ⚠️ **Vision Model Requirement**: This feature ONLY works with vision-capable models (e.g., gpt-4o, claude-3-5-sonnet, llama3.2-vision)

Achieve **3-4x token compression** and **faster inference** with Glyph's revolutionary visual-text compression:

```python
from abstractcore import create_llm

# IMPORTANT: Requires a vision-capable model
llm = create_llm("ollama", model="llama3.2-vision:11b")  # ✓ Vision model

# Large documents are automatically compressed for efficiency
response = llm.generate(
    "Analyze the key findings in this research paper",
    media=["large_research_paper.pdf"]  # Automatically compressed if beneficial
)

# Force compression (raises error if model lacks vision)
response = llm.generate(
    "Summarize this document",
    media=["document.pdf"],
    glyph_compression="always"  # "auto" | "always" | "never"
)

# Non-vision models will raise UnsupportedFeatureError
# llm_no_vision = create_llm("openai", model="gpt-4")  # ✗ No vision
# response = llm_no_vision.generate("...", glyph_compression="always")  # Error!

# Check compression stats
if response.metadata and response.metadata.get('compression_used'):
    stats = response.metadata.get('compression_stats', {})
    print(f"Compression ratio: {stats.get('compression_ratio')}x")
    print(f"Processing speedup: 14% faster, 79% less memory")
```

**Validated Performance:**
- **14% faster processing** with real-world documents
- **79% lower memory usage** during processing  
- **100% quality preservation** - no loss of analytical accuracy
- **Transparent operation** - works with existing code

[Learn more about Glyph Compression](docs/glyphs.md)

## Key Features

- **Offline-First Design**: Built primarily for open source LLMs with full offline capability. Download once, run forever without internet access
- **Provider Agnostic**: Seamlessly switch between OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace, vLLM, and any OpenAI-compatible endpoint
- **Async/Await Support** ⭐ NEW in v2.6.0: Native async support for concurrent requests with `asyncio.gather()` - works with all providers
- **Dynamic Endpoint Configuration** ⭐ NEW in v2.6.5: Pass `base_url` in POST requests to connect to custom OpenAI-compatible endpoints without environment variables
- **Interaction Tracing**: Complete LLM observability with programmatic access to prompts, responses, tokens, timing, and trace correlation for debugging, trust, and compliance
- **Glyph Visual-Text Compression**: Revolutionary compression system that renders text as optimized images for 3-4x token compression and faster inference
- **Centralized Configuration**: Global defaults and app-specific preferences at `~/.abstractcore/config/abstractcore.json`
- **Intelligent Media Handling**: Upload images, PDFs, and documents with automatic maximum resolution optimization
- **Vision Model Support**: Smart image processing at each model's maximum capability
- **Document Processing**: PDF extraction (PyMuPDF4LLM), Office documents (DOCX/XLSX/PPT), CSV/TSV analysis
- **Unified Tools**: Consistent tool calling across all providers
- **Session Management**: Persistent conversations with metadata, analytics, and complete serialization
- **Native Structured Output**: Server-side schema enforcement for Ollama and LMStudio (OpenAI and Anthropic also supported)
- **Streaming Support**: Real-time token generation for interactive experiences
- **Consistent Token Terminology**: Unified `input_tokens`, `output_tokens`, `total_tokens` across all providers
- **Embeddings**: Built-in support for semantic search and RAG applications
- **Universal Server**: Optional OpenAI-compatible API server with `/v1/responses` endpoint

## Supported Providers

| Provider | Status | SEED Support | Hardware | Setup |
|----------|--------|-------------|----------|-------|
| **OpenAI** | Full | Native | Any | [Get API key](docs/prerequisites.md#openai-setup) |
| **Anthropic** | Full | Warning* | Any | [Get API key](docs/prerequisites.md#anthropic-setup) |
| **Ollama** | Full | Native | Any | [Install guide](docs/prerequisites.md#ollama-setup) |
| **LMStudio** | Full | Native | Any | [Install guide](docs/prerequisites.md#lmstudio-setup) |
| **MLX** | Full | Native | **Apple Silicon only** | [Setup guide](docs/prerequisites.md#mlx-setup) |
| **HuggingFace** | Full | Native | Any | [Setup guide](docs/prerequisites.md#huggingface-setup) |
| **vLLM** | Full | Native | **NVIDIA CUDA only** | [Setup guide](docs/prerequisites.md#vllm-setup) |
| **OpenAI-Compatible** ⭐ NEW | Full | Native | Any | Works with llama.cpp, text-generation-webui, LocalAI, etc. |

*Anthropic doesn't support seed parameters but issues a warning when provided. Use `temperature=0.0` for more consistent outputs.

## Server Mode (Optional HTTP REST API)

AbstractCore is **primarily a Python library**. The server is an **optional component** that provides OpenAI-compatible HTTP endpoints:

```bash
# Install with server support
pip install abstractcore[server]

# Start the server
uvicorn abstractcore.server.app:app --host 0.0.0.0 --port 8000
```

Use with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="anthropic/claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Server Features:**
- OpenAI-compatible REST endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/responses`)
- **NEW in v2.5.0**: OpenAI Responses API (`/v1/responses`) with native `input_file` support
- Multi-provider support through one HTTP API
- Comprehensive media processing (images, PDFs, Office documents, CSV/TSV)
- Agentic CLI integration (Codex, Crush, Gemini CLI)
- Streaming responses with optional opt-in
- Tool call format conversion
- Enhanced debug logging with `--debug` flag
- Interactive API docs at `/docs` (Swagger UI)

**When to use the server:**
- Integrating with existing OpenAI-compatible tools
- Using agentic CLIs (Codex, Crush, Gemini CLI)
- Building web applications that need HTTP API
- Multi-language access (not just Python)

## AbstractCore CLI (Optional Interactive Testing Tool)

AbstractCore includes a **built-in CLI** for interactive testing, development, and conversation management. This is an internal testing tool, distinct from external agentic CLIs.

```bash
# Start interactive CLI
python -m abstractcore.utils.cli --provider ollama --model qwen3-coder:30b

# With streaming enabled
python -m abstractcore.utils.cli --provider openai --model gpt-4o-mini --stream

# Single prompt execution
python -m abstractcore.utils.cli --provider anthropic --model claude-3-5-haiku-latest --prompt "What is Python?"
```

**Key Features:**
- Interactive REPL with conversation history
- Chat history compaction and management
- Fact extraction from conversations
- Conversation quality evaluation (LLM-as-a-judge)
- Intent analysis and deception detection
- Tool call testing and debugging
- System prompt management
- Multiple provider support

**Popular Commands:**
- `/compact` - Compress chat history while preserving context
- `/facts [file]` - Extract structured facts from conversation
- `/judge` - Evaluate conversation quality with feedback
- `/intent [participant]` - Analyze psychological intents and detect deception
- `/history [n]` - View conversation history
- `/stream` - Toggle real-time streaming
- `/system [prompt]` - Show or change system prompt
- `/status` - Show current provider, model, and capabilities

**Full Documentation:** [AbstractCore CLI Guide](docs/acore-cli.md)

**When to use the CLI:**
- Interactive development and testing
- Debugging tool calls and provider behavior
- Conversation management experiments
- Quick prototyping with different models
- Learning AbstractCore capabilities

## Built-in Applications (Ready-to-Use CLI Tools)

AbstractCore includes **five specialized command-line applications** for common LLM tasks. These are production-ready tools that can be used directly from the terminal without any Python programming.

### Available Applications

| Application | Purpose | Direct Command |
|-------------|---------|----------------|
| **Summarizer** | Document summarization | `summarizer` |
| **Extractor** | Entity and relationship extraction | `extractor` |
| **Judge** | Text evaluation and scoring | `judge` |
| **Intent Analyzer** | Psychological intent analysis & deception detection | `intent` |
| **DeepSearch** | Autonomous multi-stage research with web search | `deepsearch` |

### Quick Usage Examples

```bash
# Document summarization with different styles and lengths
summarizer document.pdf --style executive --length brief
summarizer report.txt --focus "technical details" --output summary.txt
summarizer large_doc.txt --chunk-size 15000 --provider openai --model gpt-4o-mini

# Entity extraction with various formats and options
extractor research_paper.pdf --format json-ld --focus technology
extractor article.txt --entity-types person,organization,location --output entities.jsonld
extractor doc.txt --iterate 3 --mode thorough --verbose

# Text evaluation with custom criteria and contexts
judge essay.txt --criteria clarity,accuracy,coherence --context "academic writing"
judge code.py --context "code review" --format plain --verbose
judge proposal.md --custom-criteria has_examples,covers_risks --output assessment.json

# Intent analysis with psychological insights and deception detection
intent conversation.txt --focus-participant user --depth comprehensive
intent email.txt --format plain --context document --verbose
intent chat_log.json --conversation-mode --provider lmstudio --model qwen/qwen3-30b-a3b-2507

# Autonomous research with web search and reflexive refinement
deepsearch "What are the latest advances in quantum computing?" --depth comprehensive
deepsearch "AI impact on healthcare" --focus "diagnosis,treatment,ethics" --reflexive
deepsearch "sustainable energy 2025" --max-sources 25 --provider openai --model gpt-4o-mini
```

### Installation & Setup

Apps are automatically available after installing AbstractCore:

```bash
# Install with all features
pip install abstractcore[all]

# Apps are immediately available
summarizer --help
extractor --help
judge --help
intent --help
deepsearch --help
```

### Alternative Usage Methods

```bash
# Method 1: Direct commands (recommended)
summarizer document.txt
extractor report.pdf
judge essay.md
intent conversation.txt
deepsearch "your research query"

# Method 2: Via Python module
python -m abstractcore.apps summarizer document.txt
python -m abstractcore.apps extractor report.pdf
python -m abstractcore.apps judge essay.md
python -m abstractcore.apps intent conversation.txt
python -m abstractcore.apps deepsearch "your research query"
```

### Key Parameters

**Common Parameters (all apps):**
- `--provider` + `--model` - Use different LLM providers (OpenAI, Anthropic, Ollama, etc.)
- `--output` - Save results to file instead of console
- `--verbose` - Show detailed progress information
- `--timeout` - HTTP timeout for LLM requests (default: 300s)

**Summarizer Parameters:**
- `--style` - Summary style: `structured`, `narrative`, `objective`, `analytical`, `executive`, `conversational`
- `--length` - Summary length: `brief`, `standard`, `detailed`, `comprehensive`
- `--focus` - Specific focus area for summarization
- `--chunk-size` - Chunk size for large documents (1000-32000, default: 8000)

**Extractor Parameters:**
- `--format` - Output format: `json-ld`, `triples`, `json`, `yaml`
- `--entity-types` - Focus on specific entities: `person,organization,location,technology,etc.`
- `--mode` - Extraction mode: `fast`, `balanced`, `thorough`
- `--iterate` - Number of refinement iterations (1-10, default: 1)
- `--minified` - Output compact JSON without indentation

**Judge Parameters:**
- `--context` - Evaluation context (e.g., "code review", "academic writing")
- `--criteria` - Standard criteria: `clarity,soundness,effectiveness,etc.`
- `--custom-criteria` - Custom evaluation criteria
- `--format` - Output format: `json`, `plain`, `yaml`
- `--include-criteria` - Include detailed criteria explanations

### Key Features

- **Provider Agnostic**: Works with any configured LLM provider (OpenAI, Anthropic, Ollama, etc.)
- **Multiple Formats**: Support for PDF, TXT, MD, DOCX, and more
- **Flexible Output**: JSON, JSON-LD, YAML, plain text formats
- **Batch Processing**: Process multiple files at once
- **Configurable**: Custom prompts, criteria, and evaluation rubrics
- **Production Ready**: Robust error handling and logging

### Full Documentation

Each application has documentation with examples and usage information:

- **[Summarizer Guide](docs/apps/basic-summarizer.md)** - Document summarization with multiple strategies
- **[Extractor Guide](docs/apps/basic-extractor.md)** - Entity and relationship extraction
- **[Intent Analyzer Guide](docs/apps/basic-intent.md)** - Psychological intent analysis and deception detection
- **[Judge Guide](docs/apps/basic-judge.md)** - Text evaluation and scoring systems
- **[DeepSearch Guide](docs/apps/basic-deepsearch.md)** - Autonomous multi-stage research with web search

**When to use the apps:**
- Processing documents without writing code
- Batch text analysis workflows
- Quick prototyping of text processing pipelines
- Integration with shell scripts and automation
- Standardized text processing tasks

## Configuration

AbstractCore provides a **centralized configuration system** that manages default models, cache directories, and logging settings from a single location. This eliminates the need to specify `--provider` and `--model` parameters repeatedly.

### Quick Setup

```bash
# Check current configuration (shows how to change each setting)
abstractcore --status

# Set defaults for all applications
abstractcore --set-global-default ollama/llama3:8b

# Or configure specific applications (examples of customization)
abstractcore --set-app-default summarizer openai gpt-4o-mini
abstractcore --set-app-default extractor ollama qwen3:4b-instruct
abstractcore --set-app-default judge anthropic claude-3-5-haiku

# Configure logging (common examples)
abstractcore --set-console-log-level WARNING  # Reduce console output
abstractcore --set-console-log-level NONE     # Disable console logging
abstractcore --enable-file-logging            # Save logs to files
abstractcore --enable-debug-logging           # Full debug mode

# Configure vision for image analysis with text-only models
abstractcore --set-vision-provider ollama qwen2.5vl:7b
abstractcore --set-vision-provider lmstudio qwen/qwen3-vl-4b

# Set API keys as needed
abstractcore --set-api-key openai sk-your-key-here
abstractcore --set-api-key anthropic your-anthropic-key

# Verify configuration (includes change commands for each setting)
abstractcore --status
```

### Priority System

AbstractCore uses a clear priority system where explicit parameters always override defaults:

1. **Explicit parameters** (highest priority): `summarizer doc.txt --provider openai --model gpt-4o-mini`
2. **App-specific config**: `abstractcore --set-app-default summarizer openai gpt-4o-mini`
3. **Global config**: `abstractcore --set-global-default openai/gpt-4o-mini`
4. **Built-in defaults** (lowest priority): `huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF`

### Usage After Configuration

Once configured, apps use your defaults automatically:

```bash
# Before configuration (requires explicit parameters)
summarizer document.pdf --provider openai --model gpt-4o-mini

# After configuration (uses configured defaults)
summarizer document.pdf

# Explicit parameters still override when needed
summarizer document.pdf --provider anthropic --model claude-3-5-sonnet
```

### Configuration Features

- **Application defaults**: Different optimal models for each app
- **Cache directories**: Configurable cache locations for models and data
- **Logging control**: Package-wide logging levels and debug mode
- **API key management**: Centralized API key storage
- **Interactive setup**: `abstractcore --configure` for guided configuration

**Complete guide**: [Centralized Configuration](docs/centralized-config.md)

### Environment Variables

AbstractCore supports environment variables for provider base URLs, enabling remote servers, Docker deployments, and non-standard ports:

```bash
# Ollama on remote server
export OLLAMA_BASE_URL="http://192.168.1.100:11434"
# Alternative: OLLAMA_HOST is also supported
export OLLAMA_HOST="http://192.168.1.100:11434"

# LMStudio on non-standard port
export LMSTUDIO_BASE_URL="http://localhost:1235/v1"

# OpenAI-compatible proxy
export OPENAI_BASE_URL="https://api.portkey.ai/v1"

# Anthropic proxy
export ANTHROPIC_BASE_URL="https://api.portkey.ai/v1"
```

**Priority**: Programmatic `base_url` parameter > Runtime configuration > Environment variable > Default value

**Provider discovery**: `get_all_providers_with_models()` automatically respects these environment variables when checking provider availability.

### Programmatic Configuration

Configure provider settings at runtime without environment variables:

```python
from abstractcore.config import configure_provider, get_provider_config, clear_provider_config
from abstractcore import create_llm

# Set provider base URL programmatically
configure_provider('ollama', base_url='http://192.168.1.100:11434')

# All future create_llm() calls automatically use the configured URL
llm = create_llm('ollama', model='llama3:8b')  # Uses http://192.168.1.100:11434

# Query current configuration
config = get_provider_config('ollama')
print(config)  # {'base_url': 'http://192.168.1.100:11434'}

# Clear configuration (revert to env var / default)
configure_provider('ollama', base_url=None)
# Or clear all providers
clear_provider_config()
```

**Use Cases**:
- **Web UI Settings**: Configure providers through settings pages
- **Docker Startup**: Read from custom env vars and configure programmatically
- **Testing**: Set mock server URLs for integration tests
- **Multi-tenant**: Configure different base URLs per tenant

**Priority System**:
1. Constructor parameter (highest): `create_llm("ollama", base_url="...")`
2. Runtime configuration: `configure_provider('ollama', base_url="...")`
3. Environment variable: `OLLAMA_BASE_URL`
4. Default value (lowest): `http://localhost:11434`

## Documentation

**📚 Complete Documentation:** [docs/](docs/) - Full documentation index and navigation guide

### Getting Started
- **[Prerequisites & Setup](docs/prerequisites.md)** - Install and configure providers (OpenAI, Anthropic, Ollama, etc.)
- **[Getting Started Guide](docs/getting-started.md)** - 5-minute quick start with core concepts
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

### Core Library (Python)
- **[Python API Reference](docs/api-reference.md)** - Complete Python API documentation
- **[Media Handling System](docs/media-handling-system.md)** - Images, PDFs, and document processing across all providers
- **[Session Management](docs/session.md)** - Persistent conversations, serialization, and analytics
- **[Embeddings Guide](docs/embeddings.md)** - Semantic search, RAG, and vector embeddings
- **[Code Examples](examples/)** - Working examples for all features
- **[Capabilities](docs/capabilities.md)** - What AbstractCore can and cannot do

### Server (Optional HTTP REST API)
- **[Server Documentation](docs/server.md)** - Complete server setup, API reference, and deployment

### Architecture & Advanced
- **[Architecture](docs/architecture.md)** - System design and architecture overview
- **[Tool Calling](docs/tool-calling.md)** - Universal tool system and format conversion

## Use Cases

### 1. Provider Flexibility

```python
# Same code works with any provider
providers = ["openai", "anthropic", "ollama"]

for provider in providers:
    llm = create_llm(provider, model="gpt-4o-mini")  # Auto-selects appropriate model
    response = llm.generate("Hello!")
```

### 2. Vision Analysis Across Providers

```python
# Same image analysis works with any vision model
image_files = ["product_photo.jpg", "user_feedback.png"]
prompt = "Analyze these product images and suggest improvements"

# OpenAI GPT-4o
openai_llm = create_llm("openai", model="gpt-4o")
openai_analysis = openai_llm.generate(prompt, media=image_files)

# Anthropic Claude
claude_llm = create_llm("anthropic", model="claude-3.5-sonnet")
claude_analysis = claude_llm.generate(prompt, media=image_files)

# Local model (free)
local_llm = create_llm("ollama", model="qwen3-vl:8b")
local_analysis = local_llm.generate(prompt, media=image_files)
```

### 3. Document Processing Pipeline

```python
# Universal document analysis
documents = ["contract.pdf", "financial_data.xlsx", "presentation.ppt"]
analysis_prompt = "Extract key information and identify potential risks"

# Works with any provider
llm = create_llm("anthropic", model="claude-3.5-sonnet")
response = llm.generate(analysis_prompt, media=documents)

# Automatic format handling:
# - PDF: Text extraction with PyMuPDF4LLM
# - Excel: Table parsing with pandas
# - PowerPoint: Slide content extraction with unstructured
```

### 4. Local Development, Cloud Production

```python
# Development (free, local)
llm_dev = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")

# Production (high quality, cloud)
llm_prod = create_llm("openai", model="gpt-4o-mini")
```

### 5. Embeddings & RAG

```python
from abstractcore.embeddings import EmbeddingManager

# Create embeddings for semantic search
embedder = EmbeddingManager()
docs_embeddings = embedder.embed_batch([
    "Python is great for data science",
    "JavaScript powers the web",
    "Rust ensures memory safety"
])

# Find most similar document
query_embedding = embedder.embed("Tell me about web development")
similarity = embedder.compute_similarity(query, docs[0])
```

[Learn more about Embeddings](docs/embeddings.md)

### 6. Structured Output

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: int  # 1-5
    summary: str

llm = create_llm("openai", model="gpt-4o-mini")
review = llm.generate(
    "Review the movie Inception",
    response_model=MovieReview
)
print(f"{review.title}: {review.rating}/5")
```

[Learn more about Structured Output](docs/structured-output.md)

### 7. Universal API Server

```bash
# Start server once
uvicorn abstractcore.server.app:app --port 8000

# Use with any OpenAI client
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/qwen3-coder:30b",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

## Why AbstractCore?

- **Offline-First Philosophy**: Designed for open source LLMs with complete offline operation. No internet required after initial model download
- **Unified Interface**: One API for all LLM providers
- **Multimodal Support**: Upload images, PDFs, and documents across all providers
- **Vision Models**: Seamless integration with GPT-4o, Claude Vision, qwen3-vl, and more
- **Production Ready**: Robust error handling, retries, timeouts
- **Type Safe**: Full Pydantic integration for structured outputs
- **Local & Cloud**: Run models locally or use cloud APIs
- **Tool Calling**: Consistent function calling across providers
- **Streaming**: Real-time responses for interactive applications
- **Embeddings**: Built-in vector embeddings for RAG
- **Server Mode**: Optional OpenAI-compatible API server
- **Well Documented**: Comprehensive guides and examples  

## Installation Options

```bash
# Minimal core
pip install abstractcore

# With media handling (images, PDFs, documents)
pip install abstractcore[media]

# With specific providers
pip install abstractcore[openai]
pip install abstractcore[anthropic]
pip install abstractcore[ollama]
pip install abstractcore[lmstudio]
pip install abstractcore[huggingface]
pip install abstractcore[mlx]   # macOS/Apple Silicon only
pip install abstractcore[vllm]  # NVIDIA CUDA only (Linux)

# With server support
pip install abstractcore[server]

# With embeddings
pip install abstractcore[embeddings]

# With compression (Glyph visual-text compression)
pip install abstractcore[compression]

# Everything (recommended for Apple Silicon)
pip install abstractcore[all]

# Cross-platform (all except MLX/vLLM - for Linux/Windows/Intel Mac)
pip install abstractcore[all-non-mlx]

# Provider groups
pip install abstractcore[all-providers]          # All providers (includes MLX, excludes vLLM)
pip install abstractcore[all-providers-non-mlx]  # All providers except MLX (excludes vLLM)
pip install abstractcore[local-providers]        # Ollama, LMStudio, MLX
pip install abstractcore[local-providers-non-mlx]  # Ollama, LMStudio only
pip install abstractcore[api-providers]          # OpenAI, Anthropic
pip install abstractcore[gpu-providers]          # vLLM (NVIDIA CUDA only)
```

**Hardware-Specific Notes:**
- **MLX**: Requires Apple Silicon (M1/M2/M3/M4). Will not work on Intel Macs or other platforms.
- **vLLM**: Requires NVIDIA GPUs with CUDA support. Will not work on Apple Silicon, AMD GPUs, or Intel integrated graphics.
- **All other providers** (OpenAI, Anthropic, Ollama, LMStudio, HuggingFace): Work on any hardware.

**Media processing extras:**
```bash
# For PDF processing
pip install pymupdf4llm

# For Office documents (DOCX, XLSX, PPT)
pip install unstructured

# For image optimization
pip install pillow

# For data processing (CSV, Excel)
pip install pandas
```

## Testing Status

All tests passing as of October 12th, 2025.

**Test Environment:**
- Hardware: MacBook Pro (14-inch, Nov 2024)
- Chip: Apple M4 Max
- Memory: 128 GB
- Python: 3.12.2

## Quick Links

- **[📚 Documentation Index](docs/)** - Complete documentation navigation guide
- **[🔍 Interaction Tracing](docs/interaction-tracing.md)** - LLM observability and debugging ⭐ NEW
- **[Getting Started](docs/getting-started.md)** - 5-minute quick start
- **[⚙️ Prerequisites](docs/prerequisites.md)** - Provider setup (OpenAI, Anthropic, Ollama, etc.)
- **[📖 Python API](docs/api-reference.md)** - Complete Python API reference
- **[🌐 Server Guide](docs/server.md)** - HTTP API server setup
- **[🔧 Troubleshooting](docs/troubleshooting.md)** - Fix common issues
- **[💻 Examples](examples/)** - Working code examples
- **[🐛 Issues](https://github.com/lpalbou/AbstractCore/issues)** - Report bugs
- **[💬 Discussions](https://github.com/lpalbou/AbstractCore/discussions)** - Get help

## Contact
**Maintainer:** Laurent-Philippe Albou  
📧 Email: [contact@abstractcore.ai](mailto:contact@abstractcore.ai)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**AbstractCore** - One interface, all LLM providers. Focus on building, not managing API differences.

---

> **Migration Note**: This project was previously known as "AbstractLLM" and has been completely rebranded to "AbstractCore" as of version 2.4.0. See [CHANGELOG.md](CHANGELOG.md) for migration details.
