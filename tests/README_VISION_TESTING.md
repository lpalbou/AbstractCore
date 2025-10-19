# Vision Testing Framework for AbstractCore

## Overview

This directory contains a comprehensive **pytest-based vision testing framework** that provides both:

1. **Single model comprehensive testing** - Test one specific model across all 5 test images
2. **All models over all images testing** - Matrix testing of all available models on all images

Both test types include **automatic skipping** when providers or models are not available, making them safe to run in any environment.

## Quick Start

### ✅ **What We Have Confirmed Working**

1. **1) Test comprehensively one model over the 5 examples**: ✅ **CONFIRMED**
2. **2) Automatically test ALL models over ALL images**: ✅ **CONFIRMED**
3. **Part of pytest**: ✅ **CONFIRMED**
4. **Proper skipping when provider/model unavailable**: ✅ **CONFIRMED**

### 🚀 **Basic Usage**

```bash
# Quick smoke test (fastest)
pytest tests/test_vision_single_model.py::TestQuickVisionSmoke -v

# Test specific model on all 5 images
VISION_MODEL="ollama/qwen2.5vl:7b" pytest tests/test_vision_single_model.py::TestSpecificModel::test_env_model_all_images -s

# Test ALL available models on ALL images (comprehensive matrix)
pytest tests/test_vision_comprehensive.py::TestAllModelsAllImages::test_all_available_models_all_images -s

# Show available vision models
python tests/run_vision_tests.py --available
```

## Testing Framework Structure

### **Test Files**

| File | Purpose | When to Use |
|------|---------|-------------|
| `test_vision_comprehensive.py` | Full matrix testing, model availability, quality benchmarks | When you want to test all available models comprehensively |
| `test_vision_single_model.py` | Single model focused testing across all images | When you want to test one specific model thoroughly |
| `conftest.py` | Pytest configuration, fixtures, availability checking | Automatic configuration for all tests |
| `run_vision_tests.py` | Convenient test runner with multiple options | Command-line interface for running tests |

### **Test Images and References**

Each test image has a corresponding JSON reference file for accurate evaluation:

| Image | Content | Reference File | Theme |
|-------|---------|----------------|--------|
| `mystery1_mp.jpg` | Mountain hiking trail | `mystery1_mp.json` | Outdoor recreation and scenic nature |
| `mystery2_sc.jpg` | Cat in space helmet | `mystery2_sc.json` | Pet photography with sci-fi humor |
| `mystery3_us.jpg` | Urban sunset scene | `mystery3_us.json` | Urban landscape during golden hour |
| `mystery4_wh.jpg` | Whale breaching | `mystery4_wh.json` | Marine wildlife photography |
| `mystery5_so.jpg` | Food dish | `mystery5_so.json` | Food photography and presentation |

## 1. Single Model Comprehensive Testing

### **Purpose**: Test one model thoroughly across all 5 test images with 3 query types each

```bash
# Test specific provider/model combination
pytest tests/test_vision_single_model.py::TestSpecificModel::test_specific_model_all_images[ollama-qwen2.5vl:7b] -s

# Test using environment variable
VISION_MODEL="lmstudio/qwen/qwen2.5-vl-7b" pytest tests/test_vision_single_model.py::TestSpecificModel::test_env_model_all_images -s

# Test the newer qwen3-vl-4b model (uses vision fallback)
VISION_MODEL="lmstudio/qwen/qwen3-vl-4b" pytest tests/test_vision_single_model.py::TestSpecificModel::test_env_model_all_images -s

# Test any available provider for a model
pytest tests/test_vision_single_model.py::TestSpecificModel::test_any_available_provider_for_model[qwen2.5vl:7b] -s
```

### **What Gets Tested**

For each image, the model is tested with:
- **Keywords extraction**: F1 score against reference keywords
- **Summary generation**: Coverage score of key elements
- **Structured analysis**: Field coverage of structured response

### **Example Output**

```
🎯 COMPREHENSIVE TEST: ollama/qwen2.5vl:7b
============================================================

📸 Testing mystery1_mp.jpg
   🔍 Query: keywords
      F1: 0.876
      ✅ 2.34s
   🔍 Query: summary
      Coverage: 0.923
      ✅ 3.12s
   🔍 Query: structured
      Structure: 0.857
      ✅ 4.45s

📊 FINAL RESULTS for ollama/qwen2.5vl:7b
============================================================
Images: 5/5 successful
Queries: 15/15 successful
Avg Response Time: 3.24s
Avg Keyword F1: 0.834
Avg Summary Coverage: 0.891
Avg Structured Coverage: 0.823
```

## 2. All Models Over All Images Testing

### **Purpose**: Matrix testing of ALL available models across ALL test images

```bash
# Full comprehensive matrix test
pytest tests/test_vision_comprehensive.py::TestAllModelsAllImages::test_all_available_models_all_images -s

# Test all models with quality benchmarks
pytest tests/test_vision_comprehensive.py::TestVisionQualityBenchmarks -s

# Test model availability detection
pytest tests/test_vision_comprehensive.py::TestVisionModelAvailability -s
```

### **What Gets Tested**

- **Model Discovery**: Automatic detection of available vision models
- **Provider Matrix**: Tests every available (provider, model) combination
- **Image Matrix**: Tests every available test image
- **Quality Benchmarks**: Object detection accuracy for each image
- **Performance Metrics**: Response times and success rates

### **Example Output**

```
🎯 COMPREHENSIVE MATRIX TEST
   Images: 5
   Providers: ['ollama', 'lmstudio', 'openai']

🔧 Testing OLLAMA models:
   📱 Model: qwen2.5vl:7b
      ✅ mystery1_mp.jpg: 2.34s
      ✅ mystery2_sc.jpg: 1.87s
      ✅ mystery3_us.jpg: 3.21s
      ✅ mystery4_wh.jpg: 2.95s
      ✅ mystery5_so.jpg: 2.12s
      📊 5/5 images successful

🎯 FINAL MATRIX TEST SUMMARY
   Total combinations tested: 15
   Successful combinations: 14
   Success rate: 93.3%
```

## Automatic Skipping

### **Provider Unavailable**

Tests automatically skip when providers are not available:

```
SKIPPED [1] Provider ollama not available: connection refused
SKIPPED [1] Provider openai not available: OPENAI_API_KEY not set
SKIPPED [1] Provider anthropic not available: ANTHROPIC_API_KEY not set
```

### **Model Unavailable**

Tests skip when specific models don't support vision:

```
SKIPPED [1] Model gpt-3.5-turbo does not support vision
SKIPPED [1] Vision not supported: Model claude-3-haiku-20240307 is text-only
```

### **Missing Test Files**

Tests skip gracefully when test files are missing:

```
SKIPPED [1] Vision examples directory not found
SKIPPED [1] No reference file found for mystery6.jpg
```

## Pytest Integration

### **Running Tests**

```bash
# All vision tests
pytest tests/test_vision_comprehensive.py tests/test_vision_single_model.py -v

# Skip slow tests
pytest tests/test_vision_*.py -m "not slow" -v

# Only comprehensive tests
pytest tests/test_vision_*.py -m "comprehensive" -v

# Specific provider tests
pytest tests/test_vision_*.py -k "ollama" -v

# Parallel execution (if pytest-xdist installed)
pytest tests/test_vision_*.py -n auto
```

### **Test Markers**

| Marker | Purpose | Usage |
|--------|---------|-------|
| `vision` | Vision capability tests | `-m vision` |
| `slow` | Slow-running tests | `-m "not slow"` to skip |
| `comprehensive` | Full matrix tests | `-m comprehensive` |
| `smoke` | Quick functionality tests | `-m smoke` |

### **Fixtures Available**

- `vision_test_images`: List of test image paths
- `vision_reference_files`: Reference JSON files mapping
- `available_vision_providers`: Dynamic provider/model discovery
- `create_vision_llm`: Factory for creating vision LLMs
- `skip_if_provider_unavailable`: Conditional skipping function

## Test Configuration

### **Environment Variables**

```bash
# API Keys (optional - tests skip if not available)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Specific model testing
export VISION_MODEL="ollama/qwen2.5vl:7b"

# Test configuration
export PYTEST_TIMEOUT=300  # Test timeout in seconds
```

### **Pytest Configuration**

The `pytest.ini` file configures:
- Test discovery patterns
- Marker definitions
- Output formatting
- Timeout settings
- Warning suppression

## Convenient Test Runner

### **Using run_vision_tests.py**

```bash
# Show all available models
python tests/run_vision_tests.py --available

# Quick smoke test
python tests/run_vision_tests.py --smoke

# Test specific model
python tests/run_vision_tests.py --single ollama qwen2.5vl:7b
python tests/run_vision_tests.py --single lmstudio qwen/qwen3-vl-4b  # Uses vision fallback

# Test specific provider
python tests/run_vision_tests.py --provider ollama

# Test specific image
python tests/run_vision_tests.py --image mystery1_mp.jpg

# Full comprehensive test
python tests/run_vision_tests.py --comprehensive

# Quality benchmarks
python tests/run_vision_tests.py --quality

# Run everything
python tests/run_vision_tests.py --all
```

## Integration with CI/CD

### **Basic CI Integration**

```yaml
# .github/workflows/vision_tests.yml
name: Vision Tests
on: [push, pull_request]

jobs:
  vision-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -e .[all]
          pip install pytest pytest-xdist

      - name: Run smoke tests
        run: pytest tests/test_vision_single_model.py::TestQuickVisionSmoke -v

      - name: Run availability tests
        run: pytest tests/test_vision_comprehensive.py::TestVisionModelAvailability -v

      # Only run full tests if API keys available
      - name: Run comprehensive tests
        if: env.OPENAI_API_KEY != ''
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/test_vision_comprehensive.py -m "not slow" -v
```

### **Local Development**

```bash
# Add to your Makefile
test-vision-smoke:
	pytest tests/test_vision_single_model.py::TestQuickVisionSmoke -v

test-vision-single:
	VISION_MODEL="ollama/qwen2.5vl:7b" pytest tests/test_vision_single_model.py::TestSpecificModel::test_env_model_all_images -s

test-vision-comprehensive:
	pytest tests/test_vision_comprehensive.py::TestAllModelsAllImages::test_all_available_models_all_images -s

test-vision-all:
	pytest tests/test_vision_*.py -v
```

## Quality Metrics

### **Success Criteria**

Tests pass when:
- **F1 Score > 0.1**: Keyword matching against reference
- **Coverage > 40%**: Summary coverage of key elements
- **Response Time < 30s**: Reasonable performance
- **Success Rate > 0%**: At least some tests succeed

### **Benchmarking**

Quality benchmarks test object detection accuracy:

```bash
pytest tests/test_vision_comprehensive.py::TestVisionQualityBenchmarks -v
```

Expected object detection rates:
- **mystery1_mp.jpg**: Mountain, trail, fence, sky (≥50% detection)
- **mystery2_sc.jpg**: Cat, helmet, transparent, dome (≥50% detection)
- **mystery3_us.jpg**: Street, sunset, trees, lights (≥50% detection)
- **mystery4_wh.jpg**: Whale, ocean, water, breaching (≥50% detection)
- **mystery5_so.jpg**: Food, bowl, salad, vegetables (≥50% detection)

## Troubleshooting

### **Common Issues**

**Tests skip with "Provider not available"**
```bash
# Check if providers are running
ollama serve  # For Ollama
# Start LMStudio manually

# Check API keys
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

**Tests skip with "No vision models available"**
```bash
# Show what's actually available
python tests/run_vision_tests.py --available

# Install vision models
ollama pull qwen2.5vl:7b
ollama pull llama3.2-vision:11b
```

**Tests fail with timeout**
```bash
# Increase timeout in pytest.ini
timeout = 600  # 10 minutes

# Or use environment variable
PYTEST_TIMEOUT=600 pytest tests/test_vision_*.py
```

**Reference files missing**
```bash
# Ensure all reference files exist
ls tests/vision_examples/*.json

# Should see:
# mystery1_mp.json mystery2_sc.json mystery3_us.json mystery4_wh.json mystery5_so.json
```

### **Debug Commands**

```bash
# Validate test setup
python tests/vision_comprehensive/test_updated_system.py

# Test reference loading
python -c "
from tests.vision_comprehensive.dynamic_reference_loader import DynamicReferenceLoader
loader = DynamicReferenceLoader()
print('Available references:', loader.list_available_references())
"

# Test provider availability
python -c "
from tests.conftest import check_provider_availability
print('Ollama:', check_provider_availability('ollama'))
print('OpenAI:', check_provider_availability('openai'))
"
```

## Future Enhancements

### **Planned Features**

1. **Performance Regression Testing**: Track response times over time
2. **Model Comparison Reports**: Automatic comparison across models
3. **Custom Test Images**: Easy addition of new test scenarios
4. **Parallel Execution**: Faster testing with pytest-xdist
5. **Result Persistence**: Store test results for analysis

### **Contributing**

To add new test scenarios:

1. Add image to `tests/vision_examples/`
2. Create corresponding `.json` reference file
3. Follow existing reference format
4. Tests will automatically discover new images

---

## Summary

✅ **CONFIRMED**: We have both requested testing capabilities:

1. **Single model comprehensive testing**: Test one model across all 5 examples with detailed evaluation
2. **All models over all images**: Matrix testing of all available vision models across all test images

✅ **Pytest Integration**: Full pytest integration with proper skipping, markers, and fixtures

✅ **Production Ready**: Robust error handling, automatic provider detection, and comprehensive documentation

The testing framework provides confidence in vision capabilities across all providers and models, with automatic adaptation to available infrastructure.