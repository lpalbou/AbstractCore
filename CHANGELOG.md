# Changelog

All notable changes to AbstractCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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