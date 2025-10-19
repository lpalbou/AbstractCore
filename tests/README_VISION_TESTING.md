# Vision Testing for AbstractCore

## Overview

AbstractCore provides comprehensive **pytest-based vision testing** that evaluates all available vision models across multiple test images with performance analysis, warnings capture, and detailed result logging.

## Quick Start

### ✅ **Enhanced Comprehensive Testing**

The vision testing system provides:

1. **Matrix Testing**: Test ALL available vision models on ALL test images
2. **Performance Analysis**: Speed and reliability rankings
3. **Quality Assessment**: Response content saved for manual review
4. **Warnings Capture**: All warnings logged with context
5. **File Persistence**: JSON and Markdown results saved automatically
6. **Automatic Skipping**: Safe to run when providers/models unavailable

### 🚀 **Basic Usage**

```bash
# Comprehensive test (all models, all images) with enhanced logging
pytest tests/test_vision_comprehensive.py::TestAllModelsAllImages::test_all_available_models_all_images -s

# Test specific capabilities
pytest tests/test_vision_comprehensive.py::TestVisionModelAvailability -v
pytest tests/test_vision_comprehensive.py::TestVisionQualityBenchmarks -v

# Skip slow comprehensive tests
pytest tests/test_vision_comprehensive.py -m "not slow" -v

# Run all vision tests
pytest tests/test_vision_comprehensive.py -v
```

## Enhanced Output Features

### **Console Output**
```
🎯 FINAL MATRIX TEST SUMMARY
   Total combinations tested: 9
   Successful combinations: 9
   Success rate: 100.0%

⚠️  WARNINGS CAPTURED (3 total):
   UserWarning: Slow tokenizer warning
   DeprecationWarning: Legacy API usage

🚀 PERFORMANCE ANALYSIS
   Fastest Model: anthropic/claude-3-haiku-20240307 (1.64s avg)
   Most Reliable: anthropic/claude-3-haiku-20240307 (100.0% success)

📊 PROVIDER AVERAGES
   OLLAMA: 10.25s avg, 100.0% success
   ANTHROPIC: 1.64s avg, 100.0% success

⚡ SPEED RANKINGS (Top 3)
   1. anthropic/claude-3-haiku-20240307: 1.64s
   2. lmstudio/google/gemma-3n-e4b: 2.61s
   3. openai/gpt-4o: 7.02s

📁 Results saved to:
   JSON: test_results/vision_comprehensive/vision_test_results_TIMESTAMP.json
   Summary: test_results/vision_comprehensive/vision_test_summary_TIMESTAMP.md
```

### **File Output**
- **JSON**: Complete raw data with all responses and performance metrics
- **Markdown**: Human-readable summary with rankings and quality analysis
- **Response Content**: Full model responses for quality assessment
- **Warnings**: All captured warnings with file/line context

## Test Images

The testing uses 5 carefully selected mystery images:
- `mystery1_mp.jpg` - Multi-person scene
- `mystery2_sc.jpg` - Single character
- `mystery3_us.jpg` - Urban scene
- `mystery4_wh.jpg` - Wide horizontal view
- `mystery5_so.jpg` - Single object focus

## Results Analysis

### **Performance Metrics**
- Average response time per model (speed ranking)
- Success rate across all images (reliability ranking)
- Provider averages for speed and reliability
- Fastest and most reliable model identification

### **Quality Assessment**
- Response content saved for all successful tests
- Word count and response length metrics
- Response previews in markdown summary
- Full responses in JSON for detailed analysis

### **Error Tracking**
- Complete error capture with context
- Warning categorization and logging
- Success/failure statistics per model
- Provider-level reliability analysis

## File Structure

```
tests/
├── test_vision_comprehensive.py          # Main comprehensive test suite
├── README_VISION_TESTING.md             # This documentation
├── conftest.py                          # Pytest configuration and fixtures
├── vision_examples/                     # Test images
│   ├── mystery1_mp.json                # Image metadata
│   ├── mystery2_sc.json
│   ├── mystery3_us.json
│   ├── mystery4_wh.json
│   └── mystery5_so.json
└── test_results/vision_comprehensive/   # Generated results
    ├── vision_test_results_TIMESTAMP.json
    └── vision_test_summary_TIMESTAMP.md
```

## Common Use Cases

### **Development & Testing**
```bash
# Quick smoke test to verify setup
pytest tests/test_vision_comprehensive.py::TestVisionModelAvailability -v

# Full comprehensive test for performance analysis
pytest tests/test_vision_comprehensive.py::TestAllModelsAllImages::test_all_available_models_all_images -s
```

### **Performance Analysis**
Check the generated files in `test_results/vision_comprehensive/` for:
- Speed comparisons between providers
- Reliability analysis across models
- Quality assessment of responses
- Error patterns and warnings

### **Quality Benchmarking**
```bash
# Test specific quality benchmarks
pytest tests/test_vision_comprehensive.py::TestVisionQualityBenchmarks -v
```

## Integration with CI/CD

The tests automatically skip unavailable providers/models, making them safe for CI environments:

```bash
# CI-friendly command (skips unavailable models)
pytest tests/test_vision_comprehensive.py -v --tb=short
```

## Configuration

Tests automatically detect available models through:
- Ollama installed models
- LMStudio running models
- Cloud API keys in configuration
- HuggingFace model availability

No manual configuration required - tests adapt to your environment.

## Troubleshooting

### **No Vision Models Available**
- Install Ollama with vision models: `ollama pull qwen2.5vl:7b`
- Configure API keys: `abstractcore --set-api-key openai sk-...`
- Start LMStudio with vision models

### **Test Images Missing**
- Test images should be in `tests/vision_examples/`
- Run from project root directory
- Check file permissions

### **Performance Issues**
- Cloud models are fastest (Anthropic Claude, OpenAI GPT-4V)
- Local models vary by hardware (GPU vs CPU)
- Check results files for detailed timing analysis

For detailed implementation information, see [ENHANCED_VISION_TESTING.md](../ENHANCED_VISION_TESTING.md).