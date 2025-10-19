# Server Media Integration Status & Design Document

**Date**: October 19, 2025
**Author**: Claude Code Analysis
**Status**: Design Phase - Media capabilities NOT yet integrated into server endpoints

## Executive Summary

AbstractCore has a **production-ready unified media handling system** that supports 12+ file formats across all LLM providers, but these capabilities are **not yet exposed through the OpenAI-compatible server endpoints**. This document analyzes the current state, researches OpenAI API standards, and proposes three elegant integration approaches that would enable file attachments in the `/v1/chat/completions` and related endpoints.

## Current Status Analysis

### ✅ What Works (AbstractCore Core)

AbstractCore's media system is **fully functional** in the Python API and CLI:

- **Universal API**: `media=[]` parameter works across all providers
- **12+ File Formats**: Images (PNG, JPEG, GIF, WEBP, BMP, TIFF), Documents (PDF, DOCX, XLSX, PPTX), Data (CSV, TSV, JSON), Text (TXT, MD)
- **CLI Integration**: `@filename` syntax in CLI for instant file attachment
- **7 Providers**: OpenAI, Anthropic, Ollama, LMStudio, MLX, HuggingFace, Mock
- **Vision Fallback**: Text-only models can process images via two-stage pipeline
- **Production Tested**: All functionality working and tested

### ❌ What's Missing (Server Endpoints)

The FastAPI server endpoints **do not expose media capabilities**:

- **No file attachment support** in `/v1/chat/completions`
- **No multipart/form-data handling** for file uploads
- **No OpenAI-compatible image_url content format**
- **No integration** with MessagePreprocessor for `@filename` syntax
- **No media parameter** in ChatCompletionRequest model

### 📊 Server Architecture Current State

**File**: `/Users/albou/projects/abstractcore/abstractcore/server/app.py` (1096 lines)

**OpenAI-Compatible Endpoints**:
```
✅ /v1/chat/completions          (POST) - Chat without media
✅ /v1/responses                 (POST) - Streaming without media
✅ /{provider}/v1/chat/completions (POST) - Provider-specific without media
✅ /v1/models                    (GET)  - Model discovery
✅ /v1/embeddings                (POST) - Text embeddings only
✅ /providers                    (GET)  - Provider metadata
✅ /health                       (GET)  - Health check
```

**Current Message Format**:
```python
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str]  # TEXT ONLY - No content array support
    tool_call_id: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    name: Optional[str]
```

## OpenAI API Standards Research (2025)

### Current OpenAI Vision API Format

**Standard OpenAI Message Content Structure**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,{base64_string}",
            "detail": "high"
          }
        }
      ]
    }
  ]
}
```

### OpenAI File Attachment Methods

**Method 1: Base64 Inline** (Current Standard)
- Images embedded as `data:image/jpeg;base64,{base64_string}`
- Supported by all OpenAI-compatible clients
- **Limit**: 10 images per request

**Method 2: Multipart Form Upload** (Emerging)
- Multipart/form-data for file uploads
- Files uploaded as separate form fields
- **Recent Change**: ChatCompletions API no longer supports Files as inputs (Sept 2025)

**Method 3: URL References** (For Public Files)
- Direct HTTP/HTTPS URLs to publicly accessible files
- `"url": "https://example.com/image.jpg"`

### Key OpenAI Compatibility Requirements

1. **Content Array Support**: `content` can be string OR array of content objects
2. **Type Differentiation**: Content objects must have `"type": "text"|"image_url"`
3. **Image URL Object**: `image_url` must be object with `url` and optional `detail`
4. **Base64 Format**: `data:image/jpeg;base64,{base64_string}` format required
5. **File Size Limits**: 100 pages, 32MB total per request (for PDFs)

## Integration Design Approaches

Based on the analysis, here are **three elegant approaches** to integrate media capabilities:

---

## 🎯 Approach 1: Enhanced Content Array (RECOMMENDED)

**Strategy**: Extend the existing `ChatMessage.content` field to support both string and array formats, maintaining full OpenAI compatibility.

### Implementation Plan

**1. Update Message Model**:
```python
from typing import Union, List, Dict, Any
from pydantic import BaseModel, Field

class ContentItem(BaseModel):
    """Individual content item within a message"""
    type: Literal["text", "image_url"] = Field(description="Content type")
    text: Optional[str] = Field(default=None, description="Text content")
    image_url: Optional[Dict[str, Any]] = Field(default=None, description="Image URL object")

class ChatMessage(BaseModel):
    """OpenAI-compatible message format with media support"""
    role: Literal["system", "user", "assistant", "tool"]

    # ENHANCED: Support both string and array formats
    content: Optional[Union[str, List[ContentItem]]] = Field(
        default=None,
        description="Message content - can be a string or array of content objects for multimodal messages"
    )

    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None
```

**2. Content Processing Pipeline**:
```python
def process_message_content(message: ChatMessage) -> Tuple[str, List[str]]:
    """
    Extract media files from message content and return clean text + media list.

    Supports both OpenAI formats:
    - content as string: "Analyze this @image.jpg"
    - content as array: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]
    """
    if isinstance(message.content, str):
        # Legacy format: extract @filename references
        clean_text, media_files = MessagePreprocessor.parse_file_attachments(message.content)
        return clean_text, media_files

    elif isinstance(message.content, list):
        # OpenAI array format: extract image_url objects
        text_parts = []
        media_files = []

        for item in message.content:
            if item.type == "text" and item.text:
                text_parts.append(item.text)
            elif item.type == "image_url" and item.image_url:
                media_file = process_image_url_object(item.image_url)
                if media_file:
                    media_files.append(media_file)

        return " ".join(text_parts), media_files

    return "", []

def process_image_url_object(image_url_obj: Dict[str, Any]) -> Optional[str]:
    """
    Process OpenAI image_url object and return local file path or base64 data.

    Supports:
    - Base64: "data:image/jpeg;base64,{base64_string}"
    - HTTP URLs: "https://example.com/image.jpg"
    - Local paths: "/path/to/image.jpg" (extension)
    """
    url = image_url_obj.get("url", "")

    if url.startswith("data:"):
        # Base64 encoded image - save temporarily or handle directly
        return handle_base64_image(url)
    elif url.startswith(("http://", "https://")):
        # Download from URL temporarily
        return download_image_temporarily(url)
    else:
        # Assume local file path
        return url if os.path.exists(url) else None
```

**3. Enhanced Chat Completions Handler**:
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """Enhanced chat completions with media support"""

    # Extract media from all messages
    all_media_files = []
    processed_messages = []

    for message in request.messages:
        clean_text, media_files = process_message_content(message)
        all_media_files.extend(media_files)

        # Create processed message with clean text
        processed_message = message.model_copy()
        processed_message.content = clean_text
        processed_messages.append(processed_message)

    # Convert to AbstractCore format
    ac_messages = convert_to_abstractcore_messages(processed_messages)

    # Create LLM with media support
    provider_name, model_name = parse_model(request.model)
    llm = create_llm(provider_name, model=model_name, **extract_llm_kwargs(request))

    # Generate with media
    response = llm.generate(
        prompt="",  # Using messages instead
        messages=ac_messages,
        media=all_media_files,  # Pass extracted media
        tools=request.tools,
        stream=request.stream,
        **extract_generation_kwargs(request)
    )

    return format_openai_response(response, request)
```

### Benefits of Approach 1

✅ **Full OpenAI Compatibility**: Supports both string and array content formats
✅ **Backward Compatible**: Existing string-based clients continue working
✅ **Standards Compliant**: Follows OpenAI Vision API exactly
✅ **Unified Processing**: Single pipeline handles all media types
✅ **Extensible**: Easy to add new content types (e.g., `"document_url"`)

---

## 🔄 Approach 2: Separate Media Parameter (ALTERNATIVE)

**Strategy**: Add a dedicated `media` parameter to the request body while keeping `content` as text-only.

### Implementation Plan

**1. Enhanced Request Model**:
```python
class ChatCompletionRequest(BaseModel):
    """Enhanced chat completion request with media parameter"""
    model: str
    messages: List[ChatMessage]  # content remains string-only

    # NEW: Dedicated media parameter
    media: Optional[List[Union[str, Dict[str, Any]]]] = Field(
        default=None,
        description="Media files to attach. Can be file paths, URLs, or base64 data objects."
    )

    # ... existing parameters ...
```

**2. Media Parameter Processing**:
```python
def process_media_parameter(media: Optional[List[Union[str, Dict]]]) -> List[str]:
    """
    Process media parameter and return list of file paths for AbstractCore.

    Supports:
    - File paths: ["image.jpg", "document.pdf"]
    - URLs: ["https://example.com/image.jpg"]
    - Base64 objects: [{"type": "image", "data": "base64...", "format": "jpeg"}]
    """
    if not media:
        return []

    processed_files = []
    for item in media:
        if isinstance(item, str):
            # File path or URL
            processed_files.append(item)
        elif isinstance(item, dict):
            # Base64 or structured media object
            file_path = handle_media_object(item)
            if file_path:
                processed_files.append(file_path)

    return processed_files
```

### Benefits of Approach 2

✅ **Clean Separation**: Media handling separate from message content
✅ **Simple Migration**: Easy to add media without changing message format
✅ **AbstractCore Native**: Directly maps to AbstractCore's `media=[]` parameter
✅ **Flexible**: Supports multiple media input formats

❌ **Non-Standard**: Not compatible with standard OpenAI Vision API clients
❌ **Client Changes**: Requires client modifications to use media

---

## 🚀 Approach 3: Hybrid Multipart Upload (ADVANCED)

**Strategy**: Support both OpenAI content arrays AND multipart form uploads for maximum compatibility.

### Implementation Plan

**1. Dual Endpoint Support**:
```python
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.datastructures import FormData

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Optional[ChatCompletionRequest] = None,
    files: List[UploadFile] = File(default=[]),
    json_data: Optional[str] = Form(default=None)
):
    """
    Enhanced chat completions supporting multiple input methods:
    1. JSON body with content arrays (OpenAI standard)
    2. Multipart form with file uploads + JSON
    3. Hybrid: JSON with file references + uploaded files
    """

    # Method 1: Standard JSON request
    if request and not files:
        return await handle_json_request(request)

    # Method 2: Multipart form request
    elif files and json_data:
        request_data = json.loads(json_data)
        return await handle_multipart_request(request_data, files)

    # Method 3: Hybrid request
    else:
        return await handle_hybrid_request(request, files)
```

**2. File Upload Handling**:
```python
async def handle_multipart_request(request_data: Dict, files: List[UploadFile]) -> Dict:
    """Handle multipart form uploads with JSON metadata"""

    # Save uploaded files temporarily
    temp_files = []
    for file in files:
        temp_path = save_upload_temporarily(file)
        temp_files.append(temp_path)

    # Create request object
    request = ChatCompletionRequest(**request_data)

    # Process with uploaded files
    response = await process_chat_completion(request, media_files=temp_files)

    # Cleanup temporary files
    cleanup_temp_files(temp_files)

    return response
```

### Benefits of Approach 3

✅ **Maximum Compatibility**: Supports all OpenAI formats + multipart uploads
✅ **File Upload Efficiency**: Large files uploaded directly without base64 bloat
✅ **Standard + Extension**: Fully OpenAI compatible plus additional capabilities
✅ **Future Proof**: Ready for evolving OpenAI file upload standards

❌ **Implementation Complexity**: More complex to implement and test
❌ **Multiple Code Paths**: Different processing logic for different input methods

---

## 🏆 Recommended Implementation: Approach 1

**Why Approach 1 is Best**:

1. **OpenAI Standard Compliance**: Follows OpenAI Vision API exactly
2. **Client Compatibility**: Works with existing OpenAI clients immediately
3. **Unified Architecture**: Single processing pipeline for all media
4. **AbstractCore Integration**: Leverages existing media system seamlessly
5. **Backward Compatible**: Existing string-based clients continue working

### Implementation Roadmap

**Phase 1: Core Message Enhancement (1-2 days)**
- Update `ChatMessage` model to support content arrays
- Implement `process_message_content()` function
- Add base64 image handling utilities

**Phase 2: Endpoint Integration (1 day)**
- Modify `/v1/chat/completions` handler to extract media
- Update `/v1/responses` streaming endpoint
- Add media support to provider-specific endpoints

**Phase 3: Testing & Validation (1 day)**
- Unit tests for content processing
- Integration tests with real files
- OpenAI client compatibility tests

**Phase 4: Documentation & Examples (0.5 days)**
- Update API documentation
- Add usage examples for different formats
- Create migration guide for existing users

## Code Integration Points

### Key Files to Modify

1. **`abstractcore/server/app.py`**:
   - Update `ChatMessage` model (lines ~160-180)
   - Enhance `chat_completions()` handler (lines ~500-600)
   - Add content processing utilities

2. **Dependencies to Import**:
   - `from ..utils.message_preprocessor import MessagePreprocessor`
   - `from ..media import process_file, MediaContent`

3. **New Utility Functions**:
   - `process_message_content(message: ChatMessage)`
   - `process_image_url_object(image_url_obj: Dict)`
   - `handle_base64_image(data_url: str)`

### AbstractCore Integration

The integration leverages **existing AbstractCore capabilities**:

```python
# This already works in AbstractCore Python API:
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    "What's in these files?",
    media=["image.jpg", "document.pdf", "data.csv"]  # ← This functionality exists!
)

# Server needs to expose this through OpenAI-compatible endpoints
```

## OpenAI Client Compatibility Examples

After implementation, these standard OpenAI client calls will work:

### Python OpenAI Client
```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1")

# Base64 image
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_string}"}}
        ]
    }]
)

# URL image
response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this chart"},
            {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}
        ]
    }]
)
```

### Legacy AbstractCore Style (Backward Compatible)
```python
# This continues to work via @filename extraction
response = client.chat.completions.create(
    model="ollama/qwen2.5vl:7b",
    messages=[{
        "role": "user",
        "content": "What's in this image @screenshot.png and @report.pdf?"
    }]
)
```

### cURL Examples
```bash
# Standard OpenAI format
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'

# Legacy AbstractCore format
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-3.5-sonnet",
    "messages": [{
      "role": "user",
      "content": "Analyze @chart.png and @data.csv"
    }]
  }'
```

## Error Handling & Edge Cases

### File Processing Errors
```python
try:
    media_files = process_message_content(message)
except MediaProcessingError as e:
    raise HTTPException(
        status_code=400,
        detail={"error": {"message": f"Media processing failed: {e}", "type": "media_error"}}
    )
```

### File Size Limits
```python
def validate_media_files(files: List[str]) -> None:
    """Validate file sizes and formats before processing"""
    total_size = 0
    for file_path in files:
        if not os.path.exists(file_path):
            raise HTTPException(400, detail=f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        total_size += file_size

        if total_size > 32 * 1024 * 1024:  # 32MB limit
            raise HTTPException(400, detail="Total file size exceeds 32MB limit")
```

### Provider Compatibility
```python
def check_media_support(provider: str, model: str, media_files: List[str]) -> None:
    """Check if provider/model supports the requested media types"""
    from abstractcore.media.capabilities import is_vision_model, supports_documents

    has_images = any(is_image_file(f) for f in media_files)
    has_documents = any(is_document_file(f) for f in media_files)

    if has_images and not is_vision_model(f"{provider}/{model}"):
        logger.warning(f"Model {provider}/{model} doesn't support vision - using fallback")

    if has_documents and not supports_documents(f"{provider}/{model}"):
        logger.info(f"Converting documents to text for {provider}/{model}")
```

## Performance Considerations

### Base64 Processing
- **Memory Usage**: Base64 images are ~33% larger than binary
- **Processing Time**: Decode/encode adds latency
- **Caching**: Cache processed images temporarily

### File Upload Limits
- **Request Size**: FastAPI default 16MB limit may need increase
- **Concurrent Uploads**: Consider rate limiting for file endpoints
- **Temporary Storage**: Clean up uploaded files after processing

### Streaming with Media
- **Initial Processing**: Media processing before streaming starts
- **Memory Management**: Stream text while keeping media in memory
- **Error Handling**: Handle media errors during streaming

## Security Considerations

### File Upload Security
```python
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff',
                     '.pdf', '.docx', '.xlsx', '.pptx', '.csv', '.tsv', '.txt', '.md'}

def validate_file_security(file_path: str) -> None:
    """Validate file for security issues"""
    # Check extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, detail=f"File type {ext} not allowed")

    # Check file content (magic numbers)
    if not is_valid_file_type(file_path):
        raise HTTPException(400, detail="File content doesn't match extension")

    # Size limits
    max_size = 32 * 1024 * 1024  # 32MB
    if os.path.getsize(file_path) > max_size:
        raise HTTPException(400, detail="File too large")
```

### URL Processing Security
```python
def validate_image_url(url: str) -> None:
    """Validate image URLs for security"""
    if not url.startswith(('https://', 'data:')):
        raise HTTPException(400, detail="Only HTTPS URLs and data URLs allowed")

    if url.startswith('https://'):
        # Validate domain, check against blocklist, etc.
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname in BLOCKED_DOMAINS:
            raise HTTPException(400, detail="Domain not allowed")
```

## Testing Strategy

### Unit Tests
```python
def test_content_array_processing():
    """Test OpenAI content array processing"""
    message = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    )

    clean_text, media_files = process_message_content(message)
    assert clean_text == "What's in this image?"
    assert len(media_files) == 1

def test_legacy_filename_processing():
    """Test @filename syntax processing"""
    message = ChatMessage(
        role="user",
        content="Analyze @image.jpg and @doc.pdf"
    )

    clean_text, media_files = process_message_content(message)
    assert clean_text == "Analyze  and"
    assert media_files == ["image.jpg", "doc.pdf"]
```

### Integration Tests
```python
def test_chat_completions_with_media():
    """Test full chat completions with media"""
    response = client.post("/v1/chat/completions", json={
        "model": "openai/gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{test_image_b64}"}}
            ]
        }]
    })

    assert response.status_code == 200
    assert "image" in response.json()["choices"][0]["message"]["content"].lower()
```

## Migration Strategy

### Existing Clients
1. **No Breaking Changes**: Current string-based clients continue working
2. **Progressive Enhancement**: Clients can gradually adopt content arrays
3. **Documentation**: Clear examples for both formats

### AbstractCore CLI
- **CLI Already Works**: No changes needed for CLI (`@filename` syntax)
- **Server Compatibility**: CLI can target enhanced server endpoints

### Provider Support
- **Automatic Detection**: Media capabilities detected per provider/model
- **Graceful Fallback**: Non-vision models use text extraction
- **Error Messages**: Clear guidance when media not supported

## Success Metrics

### Implementation Success
- ✅ OpenAI client compatibility tests pass
- ✅ All existing AbstractCore media functionality available via server
- ✅ Performance within 10% of direct AbstractCore usage
- ✅ Error handling provides clear, actionable messages

### User Experience
- ✅ Standard OpenAI Vision API calls work without modification
- ✅ Legacy `@filename` syntax continues working
- ✅ Clear documentation and examples available
- ✅ Consistent behavior across all supported providers

## Next Steps

1. **Implement Approach 1** following the roadmap above
2. **Add comprehensive testing** for all OpenAI client scenarios
3. **Update documentation** with media API examples
4. **Consider Approach 3** for advanced use cases if needed

---

## Conclusion

AbstractCore has all the underlying infrastructure needed to support OpenAI-compatible media attachments. The recommended **Approach 1 (Enhanced Content Array)** provides the most elegant and standard-compliant solution that:

- ✅ **Maintains full OpenAI API compatibility**
- ✅ **Leverages existing AbstractCore media system**
- ✅ **Requires minimal code changes** to server endpoints
- ✅ **Supports all 12+ file formats** across all 7 providers
- ✅ **Enables immediate use** with existing OpenAI clients

The implementation would unlock AbstractCore's powerful media capabilities through standard OpenAI endpoints, making it a true universal multimodal LLM gateway while maintaining the "write once, run everywhere" philosophy.

**Estimated Implementation Time**: 3-4 days for full implementation, testing, and documentation.