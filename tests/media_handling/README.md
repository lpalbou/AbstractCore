# AbstractCore Media Handling Tests

Comprehensive test suite for the AbstractCore Media Handling system, covering all file types, providers, and error conditions.

## Test Structure

### Core Test Files

- **`test_media_processors.py`** - Tests for individual media processors (Image, Text, PDF, Office)
- **`test_provider_handlers.py`** - Tests for provider-specific media handlers (OpenAI, Anthropic, Local)
- **`test_provider_integration.py`** - End-to-end integration tests through generate() method
- **`test_error_handling.py`** - Error handling, edge cases, and dependency management tests
- **`conftest.py`** - Pytest configuration and shared fixtures

### Supported File Types

✅ **Images**: PNG, JPEG, GIF, WEBP, BMP, TIFF
✅ **Text**: TXT, MD, CSV, TSV, JSON
✅ **Documents**: PDF
✅ **Office**: DOCX, XLSX, PPT (with unstructured library)

### Tested Providers

✅ **OpenAI**: GPT-4o, GPT-4 Turbo with Vision
✅ **Anthropic**: Claude 3.5 Sonnet, Claude 4 series
✅ **Ollama**: qwen3-vl, gemma3:4b, other vision models
✅ **LMStudio**: qwen/qwen2.5-vl-7b, google/gemma-3n-e4b
✅ **HuggingFace**: Local transformers and GGUF models
✅ **MLX**: Apple Silicon optimized models

## Running Tests

### Quick Test Run
```bash
# Run all media handling tests
python -m pytest tests/media_handling/ -v

# Run specific test categories
python -m pytest tests/media_handling/test_media_processors.py -v
python -m pytest tests/media_handling/test_provider_handlers.py -v
```

### Integration Tests
```bash
# Run integration tests (may require mocking)
python -m pytest tests/media_handling/test_provider_integration.py -v

# Run with real model endpoints (requires API keys)
TEST_WITH_REAL_MODELS=1 python -m pytest tests/media_handling/test_provider_integration.py -v
```

### Dependency-Specific Tests
```bash
# Test with office document support
TEST_WITH_OFFICE_DOCS=1 python -m pytest tests/media_handling/ -v

# Skip tests requiring optional dependencies
python -m pytest tests/media_handling/ -v -m "not requires_deps"
```

### Test Coverage
```bash
# Run with coverage reporting
python -m pytest tests/media_handling/ --cov=abstractcore.media --cov-report=html
```

## Test Categories

### Unit Tests
- Individual processor functionality
- Provider-specific formatting
- Media type detection
- Capability validation

### Integration Tests
- End-to-end media processing
- Provider API integration
- Multimodal message creation
- Streaming with media

### Error Handling Tests
- Missing dependencies
- Invalid file formats
- Permission errors
- Network failures
- Edge cases

## Environment Variables

- `TEST_WITH_REAL_MODELS=1` - Enable tests with real model endpoints
- `TEST_WITH_OFFICE_DOCS=1` - Enable Office document tests (requires unstructured)
- `OPENAI_API_KEY` - For real OpenAI API tests
- `ANTHROPIC_API_KEY` - For real Anthropic API tests

## Test Requirements

### Required Dependencies
```bash
pip install pytest pillow pandas
```

### Optional Dependencies
```bash
# For PDF processing
pip install pymupdf4llm

# For Office documents
pip install unstructured

# For coverage reporting
pip install pytest-cov
```

## Expected Test Results

### Normal Run
- **~50+ tests** covering all functionality
- **All core tests should pass** with required dependencies
- **Optional tests may be skipped** if dependencies missing

### With All Dependencies
- **~70+ tests** including all optional features
- **All tests should pass** with proper setup
- **No skipped tests** except those requiring real API keys

## Troubleshooting

### Common Issues

1. **PIL/Pillow Import Errors**
   ```bash
   pip install pillow
   ```

2. **Pandas Import Errors**
   ```bash
   pip install pandas
   ```

3. **Missing Media Module**
   - Ensure you're running from AbstractCore root directory
   - Check that `abstractcore/media/` exists

4. **Test Timeouts**
   - Use `-x` flag to stop on first failure
   - Check network connectivity for integration tests

### Test Debugging

```bash
# Run with verbose output
python -m pytest tests/media_handling/ -v -s

# Run single test function
python -m pytest tests/media_handling/test_media_processors.py::TestImageProcessor::test_png_processing -v

# Run with debugging
python -m pytest tests/media_handling/ --pdb
```

## Contributing Tests

When adding new tests:

1. **Follow naming conventions**: `test_*.py` files, `test_*` functions
2. **Use fixtures**: Leverage shared fixtures in `conftest.py`
3. **Handle dependencies**: Use `pytest.skip` for missing optional deps
4. **Test error conditions**: Include negative test cases
5. **Mock external APIs**: Use `unittest.mock` for provider tests

## Real Model Testing

For testing with real models, ensure you have:

1. **API Keys**: Set environment variables for provider APIs
2. **Model Access**: Ensure models are available/activated
3. **Network**: Stable internet connection
4. **Timeouts**: Be patient with real API calls

Example real model test:
```bash
OPENAI_API_KEY=your_key_here TEST_WITH_REAL_MODELS=1 python -m pytest tests/media_handling/test_provider_integration.py::test_openai_media_integration -v
```

## Test Coverage Goals

- **✅ 90%+ coverage** for core media processing
- **✅ 85%+ coverage** for provider handlers
- **✅ 80%+ coverage** for error handling
- **✅ 100% coverage** for public API methods

The comprehensive test suite ensures robust, reliable media handling across all supported providers and file types.