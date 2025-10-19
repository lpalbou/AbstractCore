# AbstractCore Server Media Tests

Comprehensive test suite for AbstractCore's OpenAI-compatible server endpoints with media processing capabilities across all supported data modalities.

## 🎯 Test Coverage

### 📁 Data Modalities Tested
- **Images** (PNG, JPEG, GIF, WEBP, BMP, TIFF) - Vision processing with OpenAI Vision API format
- **Documents** (PDF, DOCX, XLSX, PPTX) - Document processing and content extraction
- **Data Files** (CSV, TSV, JSON, XML) - Structured data analysis
- **Text Files** (TXT, MD) - Plain text and markdown processing
- **Mixed Media** - Multiple file types in single requests

### 🤖 Providers & Models Tested
- **Ollama**: `qwen2.5vl:7b`, `llama3.2-vision:11b`, `gemma3:4b`, `llama3:8b`
- **LMStudio**: `qwen/qwen2.5-vl-7b`, `qwen/qwen3-next-80b`, `meta-llama/llama-3.2-8b-instruct`

### 📋 Test Categories

#### 🖼️ Vision Processing (`media-vision.py`)
- Single image analysis with OpenAI Vision API format
- Chart and complex image analysis
- Multiple images in single request
- AbstractCore @filename syntax compatibility
- Streaming responses with vision models
- Error handling for invalid base64/URLs

#### 📄 Document Processing (`media-documents.py`)
- PDF text extraction and analysis
- Word document content processing
- Excel spreadsheet data analysis
- PowerPoint presentation processing
- Streaming document analysis
- Error handling for missing/invalid files

#### 📊 Data Processing (`media-data.py`)
- CSV data analysis and statistical queries
- JSON structure analysis and financial data extraction
- TSV employee data processing
- XML inventory processing
- Text and Markdown content analysis
- Streaming data analysis

#### 🔀 Mixed Media (`media-mixed.py`)
- Image + CSV analysis correlation
- Multiple file types in single request
- OpenAI vs AbstractCore format consistency
- Streaming comprehensive analysis
- Large file handling and limits
- Error scenarios with mixed valid/invalid files

## 🚀 Quick Start

### Prerequisites

1. **Start the server:**
   ```bash
   uvicorn abstractcore.server.app:app --port 8000
   ```

2. **Install test dependencies:**
   ```bash
   pip install pillow reportlab openpyxl python-docx python-pptx pytest requests
   ```

3. **Ensure models are available:**
   ```bash
   # Ollama
   ollama pull qwen2.5vl:7b
   ollama pull llama3:8b

   # LMStudio - Load models through the GUI
   ```

### Running Tests

#### 🏃‍♂️ Quick Validation
```bash
python tests/server/run_media_tests.py --quick
```

#### 🧪 Full Test Suite
```bash
python tests/server/run_media_tests.py --verbose
```

#### 🎯 Provider-Specific Tests
```bash
python tests/server/run_media_tests.py --provider ollama
python tests/server/run_media_tests.py --provider lmstudio
```

#### 📋 Individual Test Modules
```bash
pytest tests/server/media-vision.py -v
pytest tests/server/media-documents.py -v
pytest tests/server/media-data.py -v
pytest tests/server/media-mixed.py -v
```

## 📊 Test Examples

### OpenAI Vision API Format
```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="ollama/qwen2.5vl:7b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }]
)
```

### AbstractCore @filename Format
```python
response = client.chat.completions.create(
    model="lmstudio/qwen3-next-80b",
    messages=[{
        "role": "user",
        "content": "Analyze @chart.png and summarize @report.pdf"
    }]
)
```

### Streaming with Media
```python
for chunk in client.chat.completions.create(
    model="ollama/qwen2.5vl:7b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in detail"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }],
    stream=True
):
    print(chunk.choices[0].delta.content, end="")
```

## 🔧 Test Configuration

### Model Requirements
- **Vision Models**: Required for image processing tests (e.g., `qwen2.5vl:7b`)
- **Text Models**: Sufficient for document/data processing tests
- **Mixed Models**: Both types enable comprehensive testing

### File Size Limits
- **Individual files**: 10MB maximum
- **Total request**: 32MB maximum
- **Supported formats**: Images, PDFs, Office docs, data files, text files

### Timeout Settings
- **Standard requests**: 60 seconds
- **Document processing**: 90 seconds
- **Mixed media**: 120 seconds

## 📈 Expected Results

### ✅ Successful Test Indicators
- All file formats processed successfully
- OpenAI client compatibility confirmed
- Streaming responses work with media
- Error handling works for invalid inputs
- Both OpenAI and AbstractCore formats supported

### 🔍 Common Issues & Solutions

**Server not responding:**
```bash
# Check if server is running
curl http://localhost:8000/health
```

**No models available:**
```bash
# Check model availability
curl http://localhost:8000/providers/ollama/models
curl http://localhost:8000/providers/lmstudio/models
```

**Missing dependencies:**
```bash
pip install pillow reportlab openpyxl python-docx python-pptx
```

**Large file errors:**
- Ensure files are under 10MB individually
- Check total request size under 32MB

## 📄 Test Report

The test runner generates a comprehensive report showing:
- Server health status
- Available models by provider
- Test results and coverage
- Supported file formats
- Usage examples
- Next steps and recommendations

Run with `--verbose` for detailed test output and failure analysis.

## 🎉 Success Criteria

Tests pass when:
- ✅ All data modalities process correctly
- ✅ OpenAI Vision API format compatibility confirmed
- ✅ AbstractCore @filename syntax works
- ✅ Streaming responses include media processing
- ✅ Error handling prevents crashes and provides clear messages
- ✅ Both Ollama and LMStudio models work correctly

This comprehensive test suite validates that AbstractCore's server successfully exposes its powerful media processing capabilities through standard OpenAI-compatible endpoints, enabling any OpenAI client to leverage AbstractCore's universal provider support with seamless media attachments.