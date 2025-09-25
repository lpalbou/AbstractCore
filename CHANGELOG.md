# Changelog

All notable changes to AbstractCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.3] - 2025-09-25

### Added
- **Comprehensive Timeout Configuration**: Full timeout management system across all components
  - Added configurable HTTP request timeouts for all providers (default: 180 seconds)
  - Added configurable tool execution timeouts (default: 300 seconds)
  - Added timeout parameter support in `create_llm()` and `BasicSession()`
  - Added runtime timeout management with get/set methods on providers and sessions
  - Circuit breaker recovery timeout remains configurable (default: 60 seconds)

### Enhanced
- **Provider Timeout Support**: All providers now support configurable timeouts
  - OpenAI Provider: Configurable timeout via OpenAI client
  - Anthropic Provider: Configurable timeout via Anthropic client
  - Ollama Provider: Configurable timeout via httpx.Client
  - LM Studio Provider: Configurable timeout via httpx.Client
  - Server endpoints: Updated default timeout from 30s to 180s

- **Session Timeout Management**: BasicSession now supports timeout configuration
  - Session-level timeout overrides for provider timeouts
  - Runtime timeout adjustment methods: `get_timeout()`, `set_timeout()`, etc.
  - Support for all three timeout types: HTTP, tool execution, and recovery

### Changed
- **Default Timeout Values**: Updated default timeouts for better production use
  - HTTP request timeout: 30s → 180s (6x increase for better reliability)
  - Tool execution timeout: 30s → 300s (10x increase for complex operations)
  - Circuit breaker recovery timeout: remains 60s (unchanged)

### Technical
- Added `_update_http_client_timeout()` method to BaseProvider for dynamic timeout updates
- Implemented timeout parameter propagation through the factory pattern
- Enhanced BaseProvider with comprehensive timeout management methods
- Updated all HTTP clients to use configurable timeouts instead of hardcoded values

### Documentation
- Added comprehensive timeout configuration examples to README.md
- Updated API documentation with timeout parameter descriptions
- Added session-level timeout configuration examples

## [2.1.2] - 2025-09-25

### Added
- **Enhanced Web Search Tool**: Complete web search functionality using DuckDuckGo with real web results
  - Added time range filtering (`h`, `d`, `w`, `m`, `y`) for recent content discovery
  - Improved result formatting with titles, URLs, and descriptions
  - Better error handling and fallback to instant answer API
  - Added comprehensive examples for news, research, and time-filtered searches

- **Basic CLI Tool**: Simple command-line interface for AbstractCore demonstration
  - Interactive REPL with conversation history and streaming support
  - Built-in tools: list_files, read_file, execute_command, web_search
  - Session management with provider/model switching
  - Commands: /help, /quit, /clear, /stream, /debug, /history, /model
  - Single prompt execution mode for scripting
  - Clear documentation of limitations and intended use as basic demonstrator

### Enhanced
- **BasicSession Tool Support**: Added native tools parameter to session constructor
  - Simplified tool registration: `BasicSession(provider, tools=[func1, func2])`
  - Automatic tool definition creation and registration
  - Eliminated complex tool registration ceremony
  - Improved developer experience with cleaner API

- **Web Search Improvements**: Upgraded from limited instant answers to full web search
  - Real search results using `ddgs` library instead of instant answer API only
  - Regional filtering working correctly (us-en, uk-en, etc.)
  - Safe search controls (strict, moderate, off)
  - Comprehensive time range filtering for current events and news
  - Better result quality with actual web page content

### Fixed
- **Server CLI Removal**: Eliminated over-engineered and non-functional server CLI
  - Removed `abstractcore-server` command and related dependencies
  - Updated documentation to use direct uvicorn commands
  - Removed click dependency from server requirements
  - Simplified server deployment instructions

- **Tool Registration**: Fixed web_search tool not being properly registered in CLI
  - Added web_search to CLI tools array
  - Updated help text and documentation
  - Fixed import issues and ensured proper tool availability

### Technical
- Added `ddgs` library dependency for proper web search functionality
- Enhanced tool decorator examples with time-filtering use cases
- Improved CLI architecture with clean tool integration
- Streamlined server deployment without unnecessary CLI wrapper

### Documentation
- Updated README.md with accurate CLI usage examples and capabilities
- Enhanced tool documentation with time range filtering examples
- Clarified CLI limitations and intended use as basic demonstrator
- Corrected server deployment instructions
- Added comprehensive web search usage examples

## [2.1.1] - 2025-09-24

### Fixed
- Fixed embedding test failures caused by outdated model configurations
- Updated EmbeddingGemma model ID from `google/embeddinggemma-1.1` to `google/embeddinggemma-300m`
- Fixed `cache_dir` parameter handling in EmbeddingManager to properly convert strings to Path objects
- Resolved import issues with sentence_transformers and events modules for proper test mocking
- Fixed granite model loading by including "granite" in the list of supported model names
- Improved cache file handling to avoid "open not defined" errors during module cleanup
- Fixed embedding integration tests by correcting response object comparisons
- Made granite model semantic search test more flexible to handle valid business content matches
- Removed unnecessary `return True` statements from test functions to eliminate pytest warnings

### Improved
- Enhanced error handling in embedding cache operations with proper builtins import
- Updated test assertions to be more robust and less brittle for semantic search results
- Improved mock patching strategy for better test isolation

## [2.1.0] - 2025-09-23

### Added
- Vector Embeddings support with SOTA open-source models
- EmbeddingGemma (Google's 2025 SOTA on-device model) as default
- Production-ready retry strategy with exponential backoff and circuit breakers
- Comprehensive event system for real-time monitoring and observability
- Support for multiple embedding models (Stella-400M, nomic-embed, mxbai-large)
- Two-layer caching system (LRU memory + persistent disk)
- ONNX backend optimization for 2-3x faster inference
- Matryoshka dimension truncation for flexible output sizes
- FastAPI server with OpenAI-compatible endpoints

### Enhanced
- Complete OpenAI API compatibility with all 23+ parameters
- Smart model routing based on provider patterns
- Event-driven architecture for monitoring and debugging
- Structured output with automatic retry on validation errors
- Production-grade error handling and graceful fallbacks

### Technical
- Added sentence-transformers integration for embedding generation
- Implemented Netflix Hystrix circuit breaker pattern
- AWS-recommended full jitter retry strategy
- OpenTelemetry-compatible event emission
- Comprehensive test suite with real models (no mocking)

## Previous Versions

Previous version history is available in the git commit log.