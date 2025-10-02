# Changelog

All notable changes to AbstractCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.2.0] - 2025-10-01

### Added
- **Agentic CLI Compatibility** (In Progress): Initial support for Codex, Gemini CLI, and Crush
  - New `/v1/responses` endpoint matching OpenAI Responses API format
  - Adaptive message format conversion for local vs cloud providers
  - Native `role: "tool"` message support for compatible models
  - Multi-turn tool calling with proper conversation context
  - Note: Tested with Codex with partial success; may require more capable models for full functionality

- **Memory Management for Local Providers**: Explicit model unloading
  - New `unload()` method on all providers for freeing memory
  - Critical for test suites testing multiple models sequentially
  - Prevents OOM errors in memory-constrained environments (<32GB RAM)
  - Provider-specific implementations:
    - Ollama: Sends `keep_alive=0` to immediately unload from server
    - MLX: Clears model/tokenizer and forces garbage collection
    - HuggingFace: Closes llama.cpp resources (GGUF) or clears references
    - LMStudio: Closes HTTP connection (server auto-manages via TTL)
    - OpenAI/Anthropic: No-op (safe to call)

### Fixed
- **Server Endpoint Message Handling**: Corrected message array processing
  - `/v1/chat/completions`: Now passes messages arrays instead of converting to prompt strings
  - `/v1/messages`: Removed forced token boosting and role prefix repetition
  - Fixed "User: User: User:" infinite loops in responses
  - Ollama provider now handles empty prompts correctly (no 400 errors)

- **Test Suite Reliability**: Resolved 15+ test failures
  - Fixed tool handler NoneType errors when response is null
  - Updated embedding model names (granite → granite-278m)
  - Corrected default context_length assertions (4096 → 16384)
  - Fixed Anthropic tool detection to accept "Tool not found" as valid detection
  - Removed DialoGPT model references from all tests

- **Tool Message Compatibility**: Server adapts tool messages per model capabilities
  - Models with native tool support: Keeps `role: "tool"` format (OpenAI, Anthropic)
  - Models without tool support: Converts to `role: "user"` with markers (Ollama, LMStudio, MLX)
  - Strips OpenAI-specific fields (`tool_calls`) when sending to local models
  - Handles `content: null` in tool messages correctly

### Enhanced
- **Test Coverage**: Added comprehensive test suites
  - `tests/test_unload_memory.py`: 6 tests validating unload() across all providers
  - `tests/test_agentic_cli_compatibility.py`: Codex/Gemini CLI integration tests (skip when server not running)
  - Updated existing tests to use `unload()` for proper cleanup
  - All 346+ tests now pass in full environment (Ollama, LMStudio, HF cache, API keys)

### Documentation
- **Memory Management Guide**: Added to README.md, docs/providers.md, docs/architecture.md
  - Usage examples for test suites, sequential model loading, and constrained environments
  - Provider-specific behavior explanations
  - Best practices for pairing `unload()` with `del` + `gc.collect()`
  - Performance notes (unload: <100ms, reload: seconds)

- **Agentic CLI Documentation**: New `docs/compatibility-agentic-cli.md`
  - Implementation details for ongoing Codex/Gemini CLI/Crush compatibility work
  - Adaptive message conversion strategy explained
  - Multi-turn tool calling examples
  - Server setup instructions
  - Current status and limitations documented

### Technical
- **Adaptive Message Conversion**: Smart format adaptation based on model capabilities
  - `supports_native_tool_role()`: Detects if model supports `role: "tool"`
  - `convert_tool_messages_for_model()`: Adapts messages for local vs cloud models
  - Preserves tool context for models without native tool support using text markers
  - Enables Codex/Gemini CLI to work with local Ollama/MLX models

- **Endpoint Consolidation**: Three unified endpoints with consistent message handling
  - `/v1/responses`: OpenAI Responses API format (for Codex compatibility testing)
  - `/v1/chat/completions`: OpenAI Chat Completions format (standard)
  - `/v1/messages`: Anthropic Messages format (Claude-compatible)
  - All support tools, streaming, and structured output
  - Work in progress: Full agentic CLI integration still under development

### Changed
- **Default Model Capabilities**: Updated for better compatibility
  - Default context_length: 4096 → 16384 (16K total: 12K input + 4K output)
  - Default max_output_tokens: 2048 → 4096 (matches increased context)
  - Better alignment with modern model capabilities

### Removed
- **Development Artifacts**: Cleaned up 78K+ lines of archive documentation
  - Removed old devlog files from docs/archive/devlogs/
  - Removed obsolete implementation reports
  - Kept only production-ready documentation
  - Repository now 50% smaller and more maintainable

## [2.1.6] - 2025-09-26

### Added
- **Similarity Matrix & Clustering**: SOTA vectorized similarity computation and automatic text clustering
  - `compute_similarities_matrix()` for L×C matrix operations with memory-efficient chunking
  - `find_similar_clusters()` for automatic semantic grouping using similarity thresholds
  - Normalized embedding cache providing 2x speedup for repeated calculations
  - 140x performance improvement over manual loops using vectorized NumPy operations

### Changed
- **Benchmark-Optimized Default Model**: Updated to all-MiniLM-L6-v2 based on comprehensive testing
  - **Perfect clustering purity (1.000)** with 318K sentences/sec processing speed
  - Lightweight 90MB model optimized for speed while maintaining semantic accuracy
  - Scientific benchmarking on 50 sentences across 5 semantic categories proved superior performance
  - Alternative models available: granite-278m (multilingual), qwen3-embedding (high coverage)

### Enhanced
- **Production-Ready CLI Summarizer**: Complete command-line application for document summarization
  - Standalone console command: `summarizer <file> [options]` (no module path required)
  - Comprehensive parameter support: chunk size (1k-32k), provider/model selection, verbose mode
  - Smart context window management: auto-adjusts max_tokens when chunk_size exceeds limits
  - Robust validation with clear error messages for invalid parameters
  - Timing information in verbose mode showing summarization duration

### Enhanced
- **BasicSummarizer Improvements**: Enhanced core summarization engine
  - Updated default model to `gemma3:1b-it-qat` (instruction-tuned & quantized for better performance)
  - Improved retry strategy with explicit 3-attempt validation error handling
  - Better chunk size defaults (8k chars) with 16k token context window
  - Enhanced error handling prevents validation crashes in multi-chunk documents

- **CLI Flexibility**: Advanced customization options for production use
  - `--chunk-size` parameter (1,000-32,000 characters) with automatic context window adjustment
  - `--provider` and `--model` parameters for custom LLM selection (both required together)
  - `--verbose` mode with detailed progress information and timing
  - `--style`, `--length`, `--focus`, and `--output` parameters for fine-grained control
  - Quiet mode as default (no progress output unless --verbose specified)

- **CLI Tools Consistency**: Updated CLI utilities to use consistent model defaults
  - `/compact` command in CLI now uses `gemma3:1b-it-qat` for chat history compaction
  - Consistent model selection across all AbstractCore CLI tools
  - Updated help text and documentation to reflect new defaults

### Technical
- **Console Script Integration**: Professional deployment-ready CLI installation
  - Added `summarizer` console script entry point in pyproject.toml
  - Package installation creates system-wide `summarizer` command
  - Maintained backward compatibility with module execution method
  - Professional user experience with standard CLI conventions

- **Parameter Validation**: Robust input validation and error handling
  - Chunk size range validation (1k minimum, 32k maximum)
  - Provider/model pair validation (both required when specified)
  - Clear error messages guide users to correct usage
  - Graceful handling of file reading errors and encoding issues

### Fixed
- **Validation Error Recovery**: Resolved structured output validation crashes
  - Added explicit retry strategy to all BasicSummarizer generate() calls
  - Fixed ValidationError crashes during chunk processing
  - Improved error handling prevents partial failure cascades
  - Comprehensive retry logic ensures robust document processing

### Documentation
- **Usage Examples**: Complete CLI usage documentation with real-world examples
  - Basic usage patterns and advanced configuration examples
  - Performance optimization guidance with timing information
  - File type support documentation (txt, md, py, js, html, json, csv, etc.)
  - Provider/model selection guidance for different use cases

### Improved
- **Default Style**: Changed from "objective" to "structured" for better readability
- **Word Count Accuracy**: Removed unreliable LLM-generated word counts, now computed client-side
- **Error Handling**: Added helpful messages when default Ollama model is unavailable
  - Clear instructions for installing Ollama and downloading gemma3:1b-it-qat model
  - Alternative provider examples (OpenAI, Anthropic, other Ollama models)
  - Graceful failure with actionable guidance instead of cryptic errors

## [2.1.5] - 2025-09-26

### Fixed
- **Package Structure**: Made processing module a core dependency (no longer optional import)
  - Fixes installation issues where compaction failed with "BasicSummarizer not available"
  - Processing module is now always included with AbstractCore
  - Added `processing` extra for explicit dependency management

## [2.1.4] - 2025-09-26

### Added
- **CLI System Prompt Management**: New `/system [prompt]` command for controlling AI behavior
  - `/system` - Shows current system prompt and full LLM context
  - `/system <prompt>` - Changes system prompt while preserving tools and conversation
  - Full visibility into system messages including compaction summaries
  - Enables fine-grained control over AI behavior and tool usage

### Fixed
- **Critical Chat Compaction Bugs**: Resolved multiple issues in conversation summarization
  - Fixed duplicate system messages after compaction (was creating 2-3 identical system prompts)
  - Fixed recent messages incorrectly included in conversation summaries
  - Summary now contains only older message context, recent messages preserved separately
  - Clean session structure: 1 system prompt + 1 summary + N recent messages
- **CLI History Display**: Enhanced `/history` command to show conversation summaries
  - Displays compacted conversation summaries with clear section separation
  - Shows both historical context and recent preserved messages
  - Users can now see compaction actually worked (not just message deletion)
- **Compaction Consistency**: Standardized CLI to match documentation defaults
  - Consistent `preserve_recent=4` messages (2 interactions) across CLI and docs
  - Updated all usage examples and documentation

### Improved
- **Session Copying**: Eliminated duplicate system messages in session copy operations
- **BasicSummarizer**: Chat history summarization no longer duplicates recent exchanges
- **CLI Documentation**: Updated `docs/cli-usage.md` with complete `/system` and `/history` examples
- **Code Cleanliness**: Simplified session creation logic to prevent message duplication

### Added
- **BasicSummarizer**: Production-ready text summarization capability built on AbstractCore infrastructure
  - Zero-shot structured prompting with sophisticated parameter control
  - Multiple summary styles: Structured, Narrative, Objective, Analytical, Executive
  - Configurable length levels: Brief, Standard, Detailed, Comprehensive
  - Focus parameter for domain-specific summarization (e.g., "business implications", "technical details")
  - Automatic document chunking with map-reduce approach for unlimited length documents
  - Rich structured output with confidence scoring and focus alignment metrics
  - Word count tracking and compression ratio reporting

- **Processing Module**: New `abstractllm.processing` module demonstrating advanced AbstractCore usage
  - Clean API design showcasing structured output, retry mechanisms, and provider abstraction
  - Comprehensive documentation with real-world examples and best practices
  - Integration with AbstractCore's event system for full observability

- **Local Model Evaluation**: Comprehensive benchmarking of BasicSummarizer with local Ollama models
  - Performance evaluation of gemma3:1b, qwen3-coder:30b, granite3.3:2b, cogito:3b
  - Quality assessment including confidence scoring, focus alignment, and structure compliance
  - Speed benchmarking and cost analysis for production deployment decisions
  - Detailed reports saved in `untracked/summaries/` with model-specific performance data

### Enhanced
- **Documentation Updates**: Enhanced README.md and created comprehensive BasicSummarizer documentation
  - Added BasicSummarizer to feature list and 30-second example
  - Created `docs/basic-summarizer.md` with complete usage guide, examples, and best practices
  - Updated provider selection guidance with benchmarked local model recommendations
  - Added installation instructions for Ollama and recommended model setup

- **Model Recommendations**: Evidence-based recommendations for optimal BasicSummarizer performance
  - **Primary recommendation**: `gemma3:1b` for fast, cost-effective processing (29s, 95% confidence)
  - **Premium option**: `qwen3-coder:30b` for highest quality (119s, 98% confidence)
  - Clear guidance on speed vs quality trade-offs with benchmarked performance data
  - Updated all documentation examples to showcase recommended local model setup

### Technical
- **Structured Output Integration**: BasicSummarizer demonstrates advanced AbstractCore features
  - Seamless integration with Pydantic validation and automatic retry mechanisms
  - Provider-agnostic implementation working identically across OpenAI, Anthropic, and Ollama
  - Event emission for comprehensive monitoring and debugging capabilities
  - Production-grade error handling with graceful fallbacks and detailed diagnostics

- **Test Infrastructure**: Comprehensive test suite following AbstractCore's no-mocking philosophy
  - Real-world testing with actual README.md content (15,333 characters)
  - Integration tests with live local and cloud models
  - Performance benchmarking and quality assessment validation
  - Edge case handling including chunking behavior and error scenarios

### Documentation
- **Local Model Setup**: Clear instructions for cost-effective local processing
  - Ollama installation and model download instructions (`ollama pull gemma3:1b`)
  - Performance comparison tables with speed, quality, and cost metrics
  - Provider selection guidance based on empirical evaluation results
  - Cost optimization examples using free local models vs paid cloud APIs

- **Usage Examples**: Comprehensive examples covering all major use cases
  - Executive summaries for business applications
  - Technical documentation summarization
  - Research paper analysis with focus parameters
  - Batch processing patterns and error handling strategies

### Fixed
- **read_file Tool Logic**: Fixed inconsistent behavior in common_tools.py read_file function
  - Added automatic override of `should_read_entire_file` when line range parameters are provided
  - When `start_line_one_indexed != 1` or `end_line_one_indexed_inclusive` is specified, automatically sets `should_read_entire_file = False`
  - Prevents unexpected full-file reads when partial reads were explicitly requested
  - Updated documentation to explain the automatic override behavior
  - Ensures intuitive tool behavior for LLMs and users specifying line ranges

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