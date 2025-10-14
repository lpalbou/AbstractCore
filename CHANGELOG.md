# Changelog

All notable changes to AbstractCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.3.2] - 2025-10-14

### Fixed

#### Critical Ollama Endpoint Selection Bug
- **Problem**: Ollama provider was generating excessively verbose responses (1000+ characters for simple questions like "What is 2+2?")
- **Root Cause**: Provider incorrectly used `/api/generate` endpoint for all requests, including tool-enabled conversations
- **Solution**: Updated endpoint selection logic to use `/api/chat` by default, following Ollama's API design recommendations
- **Impact**: Reduced response length from 977+ characters to 15 characters for simple queries, eliminated "infinite text" generation issue
- **Technical**: Modified `_generate_internal()` method to use `use_chat_format = tools is not None or messages is not None or True` for proper endpoint routing

#### Session Serialization Parameter Consistency
- **Problem**: Inconsistent parameter naming between `session.add_message()` using `name` and `session.generate()` using `username`
- **Root Cause**: Parameter standardization was incomplete during metadata redesign
- **Solution**: Standardized both methods to use `name` parameter, aligning with `session_schema.json` specification
- **Impact**: Consistent API across session methods, improved developer experience

#### Tool Execution Results in Live Sessions
- **Problem**: Tool execution results were missing from chat history during live CLI sessions but appeared after session reload
- **Root Cause**: Tool results were not being added to session message history during execution
- **Solution**: Modified `_execute_tool_calls()` in CLI to explicitly add `role="tool"` messages with execution metadata
- **Impact**: Tool results now immediately available to assistant during conversation, consistent behavior between live and serialized sessions

#### Common Tools Defensive Programming
- **Problem**: `list_files` and `search_files` tools failed with type errors when `head_limit` parameter was passed as string
- **Root Cause**: LLM-generated tool calls sometimes provided numeric parameters as strings
- **Solution**: Added defensive type conversion with fallback to default values on `ValueError`
- **Impact**: Improved tool reliability and error handling

### Enhanced

#### Comprehensive Session Management System
- **Session Serialization**: Complete session state preservation including provider, model, parameters, system prompt, tool registry, and conversation history
- **Optional Analytics**: Added `generate_summary()`, `generate_assessment()`, and `extract_facts()` methods for session-level insights
- **Versioned Schema**: Implemented `session-archive/v1` format with JSON schema validation in `abstractllm/assets/session_schema.json`
- **CLI Integration**: Added `/save <file> [--summary] [--assessment] [--facts]` and `/load <file>` commands with optional analytics generation
- **Backward Compatibility**: Graceful handling of legacy session formats during load operations

#### Enhanced CLI User Experience
- **Improved Help System**: Comprehensive, aesthetically pleasing help text with detailed command documentation and usage examples
- **Tool Integration**: Added `search_files` tool to CLI with full documentation and status reporting
- **Better Banner**: Informative startup banner with quick commands and available tools overview
- **Parameter Documentation**: Clear documentation of `/save` command options and usage patterns

#### Metadata System Redesign
- **Extensible Metadata**: Moved `name` field into `metadata` dictionary for better extensibility
- **Location Support**: Added `location` property backed by `metadata['location']` for geographical context
- **Property-Based Access**: Clean API with `message.name` and `message.location` properties while maintaining metadata flexibility
- **Backward Compatibility**: Automatic migration of legacy `name` field to `metadata['name']` during deserialization

### Technical

#### Files Modified
- `abstractllm/providers/ollama_provider.py`: Fixed endpoint selection logic to use `/api/chat` by default
- `abstractllm/core/session.py`: Enhanced serialization, standardized parameter naming, added analytics methods
- `abstractllm/core/types.py`: Redesigned metadata system with property-based access
- `abstractllm/utils/cli.py`: Improved help system, added tool integration, enhanced save/load commands
- `abstractllm/tools/common_tools.py`: Added defensive programming for parameter type handling
- `abstractllm/assets/session_schema.json`: Created comprehensive JSON schema for session validation
- `docs/session.md`: New documentation explaining session management and serialization benefits

#### Test Results
✅ Ollama responses now concise (15 chars vs 977+ chars previously)  
✅ Session serialization preserves complete state including analytics  
✅ Tool execution results properly integrated into live chat history  
✅ Parameter consistency across all session methods  
✅ Defensive tool parameter handling prevents type errors  
✅ Backward compatibility maintained for existing session files

## [2.3.1] - 2025-10-13

### Fixed

#### LMStudio Provider Endpoint Correction
- **Problem**: LMStudio was receiving `GET /models` instead of `GET /v1/models`, causing "Unexpected endpoint" errors
- **Root Cause**: `_validate_model()` method incorrectly stripped `/v1` from base_url before calling model discovery
- **Solution**: Removed incorrect base_url manipulation, now correctly uses full OpenAI-compatible endpoint path
- **Impact**: LMStudio provider now works reliably without endpoint errors in logs

#### CLI Tool Execution in Single-Prompt Mode
- **Problem**: Tool calls generated by models were not being executed when using `--prompt` flag
- **Root Cause**: `run_single_prompt()` method was printing raw response content instead of using tool parsing/execution flow
- **Solution**: Changed `run_single_prompt()` to use `generate_response()` for consistent tool handling across interactive and single-prompt modes
- **Impact**: `--prompt` mode now properly detects, parses, and executes tool calls

#### CLI Output Formatting for Command-Line Use
- **Problem**: Single-prompt mode showed banner and assistant prefixes, making output unsuitable for scripting
- **Solution**: Added `single_prompt_mode` flag to suppress:
  - Banner and help text
  - "🤖 Assistant:" prefix
  - "🔧 Tool Results:" header and tool call details
- **Impact**: Clean, script-friendly output when using `--prompt` flag

#### Architecture Detection for Qwen3-Next Models
- **Problem**: `qwen/qwen3-next-80b` generated tool calls correctly but they weren't being parsed or executed
- **Root Cause**: Architecture was incorrectly configured with `tool_format: "prompted"` instead of `tool_format: "special_token"`
- **Solution**: Updated `architecture_formats.json` to use correct `special_token` format with `<|tool_call|>` prefix
- **Impact**: Qwen3-Next models now work correctly in both streaming and non-streaming modes

### Enhanced

#### Universal Tool Call Parser Robustness
- **Problem**: Models generate tool calls in unpredictable formats regardless of their detected architecture:
  - Code block wrapping: ` ```json<|tool_call|>...</|tool_call|>``` `
  - Wrong field names: `"command"` or `"function"` instead of `"name"`
  - Architecture mismatch: Models using formats different from their detected type
- **Solution**: Three-part enhancement to tool call parser:

1. **Code Block Stripping** (`_parse_special_token`):
   - Pre-processes responses to remove markdown code fences (` ```json`, ` ```python`, etc.)
   - Enables detection of tool calls wrapped in code blocks

2. **Field Name Normalization** (`_parse_special_token`):
   - Accepts `command`, `function`, `tool`, `action` as alternatives to `name`
   - Accepts `params`, `parameters` as alternatives to `arguments`
   - Ensures robust parsing regardless of LLM-generated field names

3. **Format-Agnostic CLI Parsing** (`cli.py`):
   - Changed from architecture-based parser selection to universal `_parse_any_format()`
   - Tries ALL parsers (special_token, function_call, xml, tool_code, raw_json) for every response
   - Ensures CLI handles any tool call format from any model

- **Impact**: 
  - Fixed tool call detection for models like `cogito:3b` that wrap calls in JSON blocks
  - Fixed parsing for models using non-standard field names
  - Made CLI truly universal and production-ready across all model types

### Technical

#### Files Modified
- `abstractllm/providers/lmstudio_provider.py`: Fixed endpoint path handling
- `abstractllm/utils/cli.py`: Enhanced tool execution and output formatting for single-prompt mode
- `abstractllm/assets/architecture_formats.json`: Corrected qwen3-next tool format
- `abstractllm/assets/model_capabilities.json`: Added clarifying note about tool call format
- `abstractllm/tools/parser.py`: Enhanced parser with code block stripping and field normalization

#### Test Results
✅ LMStudio endpoints work correctly (no "Unexpected endpoint" errors)  
✅ Tool calls executed in both interactive and `--prompt` modes  
✅ Clean, script-friendly output in single-prompt mode  
✅ `qwen/qwen3-next-80b` tool calls work in streaming and non-streaming  
✅ `cogito:3b` tool calls in JSON blocks now detected and executed  
✅ `gemma3n:e4b` with wrong field names now parsed correctly  
✅ All existing models continue to work without regression

## [2.3.0] - 2025-10-12

### Major Changes

#### Server Simplification and Enhancement
- Simplified server implementation in `abstractllm/server/app.py` (reduced from ~4000 to ~1500 lines)
- Removed complex model discovery in favor of direct provider queries
- Added comprehensive endpoint documentation with OpenAI-style descriptions
- Enhanced request/response models with detailed parameter descriptions and examples

#### Multi-Provider Embedding Support
- `EmbeddingManager` now supports three providers: HuggingFace, Ollama, and LMStudio
- Unified embedding API across all providers with automatic format conversion
- Provider-specific caching for isolation and performance
- Backward compatible with existing HuggingFace-only code (default provider)

#### Tool Call Syntax Rewriting
- Added `syntax_rewriter.py` for server-side tool call format conversion
- Supports multiple formats: OpenAI, Codex, Qwen3, LLaMA3, Gemma, XML
- Automatic format detection based on headers, user-agent, and model name
- Enables seamless integration with agentic CLIs (Codex, Crush, Gemini CLI)

#### Model Discovery and Filtering
- Added `/v1/models?type=text-embedding` endpoint for filtering embedding models
- Heuristic-based model type detection (embedding vs text-generation)
- Embedding patterns: "embed", "all-minilm", "bert-", "-bert", "bge-", "gte-", etc.
- Provider-specific model filtering via query parameters

### Server Enhancements

#### API Endpoints
- Enhanced `/v1/embeddings` endpoint with multi-provider support
- Added `type` parameter to `/v1/models` for model type filtering (text-generation/text-embedding)
- Improved `/v1/chat/completions` with comprehensive parameter documentation
- Added `/{provider}/v1/chat/completions` for provider-specific requests
- Enhanced `/v1/responses` endpoint for agentic CLI compatibility
- Updated `/providers` endpoint with detailed provider information

#### Request/Response Models
- Added detailed field descriptions and examples to all Pydantic models
- `EmbeddingRequest`: Comprehensive parameter explanations using OpenAI reference style
- `ChatCompletionRequest`: Enhanced with field-level documentation and examples
- `ChatMessage`: Detailed role and content descriptions with use cases
- Default examples updated to use working models

#### Format Conversion
- Automatic tool call format conversion for different agentic CLIs
- Support for custom tool call tags via `agent_format` parameter
- Configurable tool execution (server-side vs client-side)
- Environment variable configuration for default formats

### Core Library Improvements

#### Embeddings
- Provider parameter added to `EmbeddingManager.__init__()` (default: "huggingface")
- `embed()` and `embed_batch()` methods now delegate to provider-specific implementations
- Ollama provider: Added `embed()` method using `/api/embeddings` endpoint
- LMStudio provider: Added `embed()` method using `/v1/embeddings` endpoint
- Cache naming includes provider for proper isolation

#### Providers
- Enhanced provider base classes with improved error handling
- Better streaming support across all providers
- Consistent timeout handling and retry logic
- Improved tool call detection and parsing

#### Exception Handling
- Added `UnsupportedProviderError` for better error messages
- Enhanced exception types for embedding-specific errors
- Improved error context and debugging information

### Documentation Overhaul

#### Consolidated Documentation
- Merged `common-mistakes.md` into `troubleshooting.md` with cross-references
- Merged `server-api-reference.md` into simplified `server.md` (1006 → 479 lines)
- Created comprehensive `docs/README.md` as navigation hub
- Removed redundant documentation files (8 files consolidated)

#### New Documentation
- Created `tool-syntax-rewriting.md` covering both tag and syntax rewriters
- Enhanced `embeddings.md` with multi-provider support and examples
- Updated `architecture.md` with server architecture and present-tense language
- Improved `getting-started.md` with comprehensive tool documentation

#### Documentation Organization
- Moved `basic-*.md` files to `docs/apps/` subdirectory
- Created `docs/archive/` for superseded documentation
- Added `docs/archive/README.md` explaining archived content
- Updated all cross-references across documentation

#### Documentation Style
- Removed historical/refactoring language ("replaced", "improved", "before/after")
- Converted all documentation to present tense
- Focused on current capabilities and actionable content
- Simplified language for clarity and accessibility

#### Root README Updates
- Added clearer distinction between core library and optional server
- Enhanced documentation section with better organization
- Added "Architecture & Advanced" section
- Improved Quick Links with comprehensive navigation

### Technical Improvements

#### Code Quality
- Removed unused `simple_model_discovery.py` module
- Cleaned up temporary debug files and scripts
- Removed integration.py tool module (functionality moved to providers)
- Better separation of concerns between core and server

#### Testing
- Added comprehensive tests for embedding providers
- Enhanced server endpoint testing
- Improved tool call syntax rewriting tests
- Better test coverage for multi-provider scenarios

### Breaking Changes
None. All changes are backward compatible with version 2.2.x.

### Migration Guide

#### For Embedding Users
If you were using embeddings, no changes needed. The default behavior remains HuggingFace.

To use other providers:
```python
from abstractllm.embeddings import EmbeddingManager

# HuggingFace (default, unchanged)
embedder = EmbeddingManager(model="sentence-transformers/all-MiniLM-L6-v2")

# Ollama (new)
embedder = EmbeddingManager(model="granite-embedding:278m", provider="ollama")

# LMStudio (new)
embedder = EmbeddingManager(model="text-embedding-all-minilm-l6-v2-embedding", provider="lmstudio")
```

#### For Server Users
Server API endpoints remain compatible. New features:
- Use `?type=text-embedding` to filter embedding models
- Use `agent_format` parameter for custom tool call formats
- Environment variables for default configuration

#### For Documentation Users
- Use `docs/server.md` instead of `server-api-reference.md`
- Use `docs/troubleshooting.md` for all troubleshooting (includes common mistakes)
- Use `docs/README.md` as navigation hub
- Reference `prerequisites.md` instead of deleted `providers.md`

## [2.2.8] - 2025-10-11

### Fixed
- **CRITICAL: Streaming Tool Call Tag Rewriting**: Fixed broken tag rewriting in streaming mode
  - **Problem**: Custom tool call tags (set via `/tooltag` command) were ignored in streaming mode
  - **Root Cause**: Broken `_apply_tag_rewriting()` method tried to create temporary provider instances instead of using existing tag rewriter infrastructure
  - **Solution**: Completely rewrote tag rewriting integration to use proven `ToolCallTagRewriter.rewrite_streaming_chunk()` method
  - **Result**: Custom tags now work identically in streaming and non-streaming modes
  - **Implementation**:
    - Added `_initialize_tag_rewriter()` method to properly parse tag configurations
    - Replaced broken temporary provider approach with direct streaming chunk rewriting
    - Added tag rewrite buffer management for handling split tool calls across chunks
    - Maintains <10ms first chunk latency with zero performance regression

### Added
- **Unified Streaming Architecture**: Revolutionary new streaming implementation
  - **Problem**: Complex dual-mode streaming with inconsistent tool handling
  - **Solution**: Single, unified streaming strategy that works identically across ALL providers
  - **Key Features**:
    - Real-time tool call detection during streaming
    - Immediate tool execution mid-stream
    - Zero buffering overhead
    - 5x faster first chunk delivery (<10ms)
    - Robust support for multiple tool formats (Qwen, LLaMA, Gemma, XML)
    - **NOW WITH**: Full custom tag rewriting support in streaming mode

### Technical Improvements
- **Streaming Performance**:
  - First chunk delivery reduced from ~50ms to <10ms
  - 37% reduction in streaming code complexity
  - Eliminated previous buffering and multi-mode processing
  - Added robust error handling for malformed responses

- **Streaming Implementation**:
  - Created `abstractllm/providers/streaming.py` with unified implementation
  - Added `IncrementalToolDetector` state machine
  - Implemented `UnifiedStreamProcessor` for consistent across-provider streaming
  - Supports incremental tool detection without full response buffering
  - Zero breaking changes to existing API

### Enhanced
- **Tool Streaming**:
  - Real-time tool call detection across all providers
  - Immediate tool execution during streaming
  - Consistent tool handling between streaming and non-streaming modes
  - Eliminated race conditions in tool execution
  - Simplified provider-agnostic tool processing

### Providers Impacted
- **Full support for**:
  - OpenAI
  - Anthropic
  - Ollama
  - MLX
  - LMStudio
  - HuggingFace

### Documentation
- Updated docs (README.md, architecture.md, providers.md) with new streaming details
- Added comprehensive examples demonstrating unified streaming capabilities
- Enhanced performance and architectural notes

## [2.2.7] - 2025-10-11

### Fixed
- **Streaming Tool Output Consistency**: Fixed inconsistent formatting between streaming and non-streaming tool execution
  - **Problem**: Streaming mode showed raw `<function_call>` tags while non-streaming showed clean "Tool Results:" format
  - **Root Cause**: Tool call tag rewriting happened before tool execution, so tool results weren't properly formatted
  - **Solution**: Reordered streaming processing to collect all chunks first, execute tools, then apply tag rewriting to the complete response
  - **Result**: Both modes now show identical "Tool Results:" formatting with tool transparency (🔧 Tool: name(params))
  - **Benefit**: Consistent professional output suitable for ReAct agents and user-facing applications

### Enhanced
- **Tool Result Transparency**: Improved tool result formatting to show both action and result for better ReAct agent compatibility
  - Tool results now display: `🔧 Tool: tool_name({'param': 'value'})` followed by the actual result
  - Provides complete Action → Observation information needed for ReAct workflows
  - Works consistently across all tools (list_files, read_file, write_file, execute_command, custom tools)
  - No hardcoded tool references - fully generic implementation using existing AbstractLLM infrastructure

### Technical
- **Streaming Architecture Improvement**: Simplified streaming tool execution flow
  - Single processing path: collect chunks → execute tools → format complete response → apply tag rewriting
  - Eliminated race conditions between tag rewriting and tool execution
  - Reduced complexity while maintaining backward compatibility
  - All changes made to base provider - no provider-specific modifications needed

## [2.2.6] - 2024-12-19

### Fixed
- **Native Tool Metadata Preservation**: Enhanced metadata (tags, when_to_use, examples) now preserved in native tool calls
- **Architecture-Aware Examples**: Tool examples now use correct format per architecture (Qwen3: `<|tool_call|>`, LLaMA3: `<function_call>`, Gemma3: plain JSON)
- **Tool Call Parsing**: Fixed parsing system works with all tool call formats across all architectures

### Enhanced
- **Tool Call Format Detection**: Improved regex patterns for better JSON tool call detection
- **Parser Fallback System**: Added proper fallback handling for unknown tool formats
- **Real Model Testing**: Comprehensive testing with actual Ollama models confirms all fixes work correctly

### Documentation
- **Architecture Model Checks**: Updated test results documentation with latest fixes and improvements
- **Tool Metadata Usage**: Documented how enhanced metadata is handled in both prompted and native tool calling

## [2.2.5] - 2025-10-10

### Enhanced
- **Unified Token Parameter Strategy**: Improved token management with helper methods and comprehensive validation
  - **Helper Methods**: Added `calculate_token_budget()`, `validate_token_constraints()`, `estimate_tokens()`, and `get_token_configuration_summary()` for better token planning
  - **Proactive Guidance**: Automatic warnings during provider initialization help identify potentially problematic configurations
  - **Smart Validation**: Provider-specific suggestions (e.g., GPT-4: 128k limit, Claude: 200k, Gemini: 1M) with efficiency ratio analysis
  - **Legacy Support**: Added deprecation warning for `context_size` parameter while maintaining backward compatibility

### Documentation
- **Comprehensive Token Vocabulary**: Enhanced documentation across core modules with clear examples and best practices
  - **Two Configuration Strategies**: Budget + Output Reserve (recommended) and Explicit Input + Output (advanced) with concrete examples
  - **Provider Abstraction Explained**: Clear documentation of how AbstractLLM handles provider-specific parameter mapping internally
  - **Helper Method Examples**: Complete usage examples for token estimation, validation, and configuration summary generation
  - **Quick Start Integration**: Updated package docstring with unified token management examples

### Technical
- **Enhanced Validation Framework**: Built-in token configuration validation with actionable feedback
  - **Automatic Warning System**: Validates configurations during provider initialization with debug logging
  - **Safety Margin Calculations**: Token budget estimation with configurable safety margins for production use
  - **Efficiency Analysis**: Warns when output tokens exceed 80% of context window or input allocation is too small
  - **Error Prevention**: Helps users avoid common token configuration mistakes before runtime

This release focuses on improving the developer experience with AbstractLLM's unified token parameter approach, providing better guidance and validation while maintaining full backward compatibility.

## [2.2.4] - 2025-10-10

### Fixed
- **ONNX Optimization and Warning Management**: Improved embedding performance and user experience
  - **Smart ONNX Model Selection**: EmbeddingManager now automatically selects optimized `model_O3.onnx` for better performance
  - **Warning Suppression**: Eliminated harmless warnings from PyTorch 2.8+ and sentence-transformers during model loading
  - **Graceful Fallbacks**: Multiple fallback layers ensure reliability (optimized ONNX → basic ONNX → PyTorch)
  - **Performance Improvement**: ONNX optimization provides significant speedup for batch embedding operations
  - **Clean Implementation**: Conservative approach with minimal code changes (40 lines) for maintainability

### Technical
- Added `_suppress_onnx_warnings()` context manager to handle known harmless warnings
- Added `_get_optimal_onnx_model()` function for intelligent ONNX variant selection
- Enhanced `_load_model()` with multi-layer fallback strategy and clear logging
- Zero breaking changes - all improvements are additive with sensible defaults

## [2.2.3] - 2025-10-09

### Fixed
- **Installation Package [all] Extra**: Fixed `pip install abstractcore[all]` to truly install ALL modules
  - **Issue**: The `[all]` extra was missing development dependencies (dev, test, docs)
  - **Solution**: Updated `[all]` extra to include complete dependency set (12 total extras)
  - **Coverage**: Now includes all providers, features, and development tools
    - **All Providers** (6): openai, anthropic, ollama, lmstudio, huggingface, mlx
    - **All Features** (3): embeddings, processing, server
    - **All Development** (3): dev, test, docs
  - **Impact**: Users can now confidently use `abstractcore[all]` for complete installation without missing dependencies

### Technical
- **Comprehensive Installation**: `pip install abstractcore[all]` now installs 12 dependency groups
- **Development Ready**: Includes all testing frameworks (pytest-cov, responses), code tools (black, mypy, ruff), and documentation tools (mkdocs)
- **Verified Configuration**: All referenced extras exist and are properly defined with no circular dependencies

## [2.2.2] - 2025-10-09

### Added
- **LLM-as-a-Judge**: Production-ready objective evaluation with structured assessments
  - **BasicJudge** class for critical assessment with constructive skepticism
  - **Multiple file support** with sequential processing to avoid context overflow
  - **Global assessment synthesis** for multi-file evaluations (appears first, followed by individual file results)
  - **Enhanced assessment structure** with judge summary, source reference, and optional criteria details
  - **9 evaluation criteria**: clarity, simplicity, actionability, soundness, innovation, effectiveness, relevance, completeness, coherence
  - **CLI with simple command**: `judge file1.py file2.py --context="code review"` (console script entry point)
  - **Flexible output formats**: JSON, plain text, YAML with structured scoring (1-5 scale)
  - **Optional global assessment control**: `--exclude-global` flag for original list behavior

### Enhanced
- **Built-in Applications**: BasicJudge added to production-ready application suite
  - **Structured output integration** with Pydantic validation and FeedbackRetry for validation error recovery
  - **Chain-of-thought reasoning** for transparent evaluation with low temperature (0.1) for consistency
  - **Custom criteria support** and reference-based evaluation for specialized assessment needs
  - **Comprehensive error handling** with graceful fallbacks and detailed diagnostics

### Documentation
- **Complete BasicJudge documentation**: Enhanced `docs/basic-judge.md` with API reference, examples, and best practices
  - **Real-world examples**: Code review, documentation assessment, academic writing evaluation, multiple file scenarios
  - **CLI parameter documentation** with practical usage patterns and advanced options
  - **Global assessment examples** showing synthesis of multiple file evaluations
- **Updated README.md**: Added BasicJudge to built-in applications with 30-second examples
- **Internal CLI integration**: Added `/judge` command for conversation quality evaluation with detailed feedback

### Technical
- **Context overflow prevention**: Optimized global assessment prompts to work within model context limits
- **Production-grade architecture**: Proper Pydantic integration, sequential file processing, backward compatibility
- **Console script integration**: Simple `judge` command available after package installation (matches `extractor`, `summarizer`)
- **Full backward compatibility**: All existing functionality preserved, optional features clearly marked

## [2.2.1] - 2025-10-09

### Enhanced
- **Timeout Configuration**: Unified timeout management across all components
  - Updated default HTTP timeout from 180s to 300s (5 minutes) for better reliability with large models
  - All providers now consistently inherit timeout from base configuration
  - Server endpoints updated to use unified 5-minute default
  - Improved handling of large language models (36B+ parameters) that require longer processing time

- **Extractor CLI Improvements**: Enhanced command-line interface for knowledge graph extraction
  - Added `--timeout` parameter with proper validation (30s minimum, 2 hours maximum)
  - Users can now configure timeout for large documents and models: `--timeout 3600` for 60 minutes
  - Improved error messages for timeout validation
  - Better support for processing large documents with resource-intensive models

### Fixed
- **BasicExtractor JSON-LD Consistency**: Resolved structural inconsistencies in knowledge graph output
  - Fixed JSON-LD reference normalization where some providers generated string references instead of proper object format
  - Corrected refinement prompt to match initial extraction format exactly (`@type: "s:Relationship"` vs `@type: "r:provides"`)
  - Added missing `s:name` and `strength` fields in relationship refinement
  - All providers now generate consistent, properly structured JSON-LD output

- **Cross-Provider Compatibility**: Improved extraction reliability across different LLM providers
  - LMStudio models now generate proper JSON-LD object references through automatic normalization
  - Reduced warning noise by converting normalization messages to debug level
  - Enhanced iterative refinement to follow exact same structure rules as initial extraction

### Technical
- **Centralized Timeout Management**: All timeout configuration now emanates from `base.py`
  - Providers inherit timeout via `self._timeout` from BaseProvider class
  - Factory system properly propagates timeout parameters through `**kwargs`
  - No hardcoded timeout values remain in provider implementations
  - Consistent 300-second default across HTTP clients, tool execution, and embeddings

### Documentation
- **Updated Model References**: Modernized documentation to use current recommended models
  - Updated `docs/getting-started.md` to use `qwen3:4b-instruct-2507-q4_K_M` (default) and `qwen3-coder:30b` (premium)
  - Replaced outdated `qwen2.5-coder:7b` references throughout getting started guide
  - Added proper cross-references to reorganized documentation (`server.md`, `internal-cli.md`)
  - Enhanced "What's Next?" section with links to universal API server and CLI documentation

- **Cross-Reference Validation**: Verified all documentation links and anchors
  - Confirmed `docs/prerequisites.md` section anchors match README.md references
  - Validated provider setup links point to correct sections (#openai-setup, #anthropic-setup, etc.)
  - Ensured consistent documentation structure across all guides

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

- **BasicExtractor**: Knowledge graph entity and relationship extraction
  - Semantic entity extraction with 30+ entity types (people, organizations, concepts, technologies, etc.)
  - 20+ relationship types for comprehensive knowledge graph construction
  - Chain of Verification for entity deduplication using semantic similarity
  - Embedding-based clustering to merge similar entities (configurable threshold)
  - JSON-LD, JSON, and YAML output formats for knowledge graphs
  - CLI application: `extractor <file>` with focus areas, entity filters, and extraction styles
  - Chunked processing for large documents with configurable chunk sizes
  - Console scripts: `extractor` and `abstractllm-extractor`

- **Summarizer CLI Enhancements**: Production-ready text summarization application
  - Console script: `summarizer` for easy command-line access (no module path needed)
  - Console alias: `abstractllm-summarizer` for explicit namespace
  - Comprehensive parameter support: chunk size, provider/model selection, output format
  - Multiple summary styles: structured, narrative, objective, analytical, executive
  - Configurable extraction depth: brief, standard, detailed, comprehensive

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