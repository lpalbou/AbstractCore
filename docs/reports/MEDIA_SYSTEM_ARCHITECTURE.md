# AbstractCore Media System Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER APPLICATION LAYER                             │
│  (CLI, Server, Session, Processing Apps - all use media seamlessly)        │
└────────────────────────────────────────────┬────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FACTORY LAYER (create_llm)                         │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ Create LLM with unified interface                                     │   │
│ │ Auto-detect provider, handle token parameters, setup media support   │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────┬────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROVIDER REGISTRY & INSTANTIATION                       │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ Centralized Provider Registry                                         │   │
│ │  • 137+ models across 7 providers                                    │   │
│ │  • Capability detection                                             │   │
│ │  • Model metadata management                                        │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────┬────────────────────────────────┘
                                             │
                         ┌───────────────────┼───────────────────┐
                         ▼                   ▼                   ▼
          ┌──────────────────────┐  ┌──────────────────────┐  ...etc
          │ OpenAI Provider      │  │ Anthropic Provider   │
          │ (gpt-4, gpt-4o)      │  │ (claude-3.5-sonnet)  │
          │                      │  │                      │
          │ AbstractCoreInterface│  │ AbstractCoreInterface│
          └──────────┬───────────┘  └──────────┬───────────┘
                     │                         │
                     └────────────┬────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MEDIA SYSTEM ORCHESTRATOR                            │
│                   (Coordinates all media handling)                          │
│                                                                             │
│  INPUT: generate(text, media=["file1.jpg", "file2.pdf", ...])              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Configuration System Integration                                   │   │
│  │ • Vision strategy (disabled | text_embedding | two_stage)         │   │
│  │ • Fallback chains                                                 │   │
│  │ • Local model paths                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Media Processing Pipeline                                          │   │
│  │                                                                     │   │
│  │  For each media file:                                             │   │
│  │    1. Detect media type (extension, MIME type)                    │   │
│  │    2. Select processor (AutoMediaHandler)                         │   │
│  │    3. Process file                                                │   │
│  │    4. Format for provider (provider-specific handler)             │   │
│  │    5. Integrate into message                                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
    ┌──────────────────────┐  ┌──────────────────────┐  ...
    │ MEDIA CAPABILITIES   │  │ AUTO MEDIA HANDLER   │
    │ Detection System     │  │ (Smart Router)       │
    │                      │  │                      │
    │ • Vision support     │  │ Detects media type   │
    │ • Image limits       │  │ Selects processor    │
    │ • Max file sizes     │  │ Lazy initialization  │
    │ • Formats support    │  │ Fallback handling    │
    │ • Text embedding pref│  │ Unified interface    │
    └──────────────────────┘  └──────────┬───────────┘
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
        ┌────────────────────────────────────────────────────────┐
        │                   PROCESSOR LAYER                       │
        │                                                         │
        │  ┌─────────────────┐  ┌──────────────────────────┐    │
        │  │ ImageProcessor  │  │ TextProcessor            │    │
        │  │ ───────────────  │  │ ──────────────          │    │
        │  │ • JPG, PNG, GIF │  │ • TXT, MD, CSV         │    │
        │  │ • WebP, TIFF    │  │ • JSON, YAML, XML      │    │
        │  │ • Auto-optimize │  │ • UTF-8 encoding       │    │
        │  │ • Smart caption │  │ • Character counting   │    │
        │  │   fallback      │  │ • Format preservation  │    │
        │  └─────────────────┘  └──────────────────────────┘    │
        │                                                         │
        │  ┌─────────────────┐  ┌──────────────────────────┐    │
        │  │ PDFProcessor    │  │ OfficeProcessor          │    │
        │  │ ───────────────  │  │ ──────────────          │    │
        │  │ • PDF extraction│  │ • DOCX, XLSX, PPTX    │    │
        │  │ • Table preserve│  │ • Structured extraction│    │
        │  │ • Layout aware  │  │ • Formatting preserved │    │
        │  │ • Multi-page    │  │ • Fallback to text     │    │
        │  │ • Fallback text │  │                         │    │
        │  └─────────────────┘  └──────────────────────────┘    │
        │                                                         │
        │  Requires optional libs: PIL, PyMuPDF4LLM,            │
        │                          unstructured, transformers    │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │              OUTPUT: MediaContent Object               │
        │  • media_type (IMAGE, DOCUMENT, TEXT, AUDIO, VIDEO)   │
        │  • content (base64, text, url, etc.)                  │
        │  • content_format (BASE64, TEXT, URL, FILE_PATH)      │
        │  • mime_type (auto-detected)                          │
        │  • file_path (original path)                          │
        │  • metadata (extraction info, processing stats)       │
        └────────────────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │         PROVIDER-SPECIFIC HANDLERS LAYER               │
        │                                                         │
        │  ┌──────────────────────────────────────────────────┐  │
        │  │ OpenAIMediaHandler                               │  │
        │  │ • Base64 encoding for images                    │  │
        │  │ • URL references for documents                 │  │
        │  │ • Multi-image support (up to 10)               │  │
        │  │ • 20MB file limit                              │  │
        │  │ • Formats: PNG, JPEG, JPG, GIF, WEBP          │  │
        │  └──────────────────────────────────────────────────┘  │
        │                                                         │
        │  ┌──────────────────────────────────────────────────┐  │
        │  │ AnthropicMediaHandler                            │  │
        │  │ • Base64 encoding with media type               │  │
        │  │ • Image stacking support (up to 20 images)      │  │
        │  │ • 5MB file limit                                │  │
        │  │ • Formats: PNG, JPEG, JPG, GIF, WEBP           │  │
        │  └──────────────────────────────────────────────────┘  │
        │                                                         │
        │  ┌──────────────────────────────────────────────────┐  │
        │  │ LocalMediaHandler                                │  │
        │  │ • Base64 encoding                               │  │
        │  │ • Text descriptions for fallback                │  │
        │  │ • Device-aware optimization                     │  │
        │  │ • 10MB file limit                               │  │
        │  └──────────────────────────────────────────────────┘  │
        │                                                         │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │         FORMAT FOR PROVIDER API (provider-specific)   │
        │                                                         │
        │  OpenAI Format:      Anthropic Format:                 │
        │  {                   {                                 │
        │    "type": "image",    "type": "image",               │
        │    "source": {         "source": {                     │
        │      "type": "base64",   "type": "base64",            │
        │      "media_type":       "media_type": "image/jpeg",  │
        │      "data": "..."       "data": "..."                │
        │    }                   }                               │
        │  }                   }                                 │
        └────────────────────┬───────────────────────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │  VISION FALLBACK SYSTEM (if needed)     │
        │  (For text-only models with images)     │
        │                                         │
        │  Two-Stage Pipeline:                   │
        │  1. Vision model → Description         │
        │  2. Text model ← Description           │
        │                                         │
        │  Fallback Chain:                       │
        │  • Primary provider (if configured)    │
        │  • Fallback chain (user-defined)       │
        │  • Local models (BLIP, ViT-GPT2, GIT) │
        │                                         │
        └────────────────────┬────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │         PROVIDER API CALL (with media content)        │
        │  • Message with properly formatted media              │
        │  • Provider-specific requirements met                 │
        │  • Size/format validation passed                      │
        │  • Streaming or non-streaming generation              │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │                  RESPONSE GENERATION                   │
        │  • Model processes message with media                 │
        │  • Returns analysis/response                          │
        │  • Event emission for observability                   │
        │  • Token tracking                                     │
        │  • Error handling with retry strategies               │
        └────────────────────────────────────────────────────────┘
```

---

## Media Processing Decision Tree

```
User Input
   │
   ├─ media parameter provided?
   │  │
   │  └─ YES
   │     │
   │     ├─ For each file:
   │     │  │
   │     │  ├─ File exists?
   │     │  │  ├─ NO → Error
   │     │  │  └─ YES
   │     │  │     │
   │     │  │     ├─ Size check
   │     │  │     │  ├─ Too large → Error
   │     │  │     │  └─ OK
   │     │  │     │     │
   │     │  │     │     ├─ Detect media type
   │     │  │     │     │  ├─ IMAGE
   │     │  │     │     │  │  ├─ PIL available?
   │     │  │     │     │  │  │  ├─ NO → Error
   │     │  │     │     │  │  │  └─ YES → ImageProcessor
   │     │  │     │     │  │  │             ├─ Optimize resolution
   │     │  │     │     │  │  │             ├─ Base64 encode
   │     │  │     │     │  │  │             └─ Extract metadata
   │     │  │     │     │  │  │
   │     │  │     │     │  ├─ DOCUMENT
   │     │  │     │     │  │  ├─ PDF?
   │     │  │     │     │  │  │  ├─ PyMuPDF4LLM available?
   │     │  │     │     │  │  │  │  ├─ NO → TextProcessor
   │     │  │     │     │  │  │  │  └─ YES → PDFProcessor
   │     │  │     │     │  │  │  │           ├─ Extract text
   │     │  │     │     │  │  │  │           ├─ Preserve tables
   │     │  │     │     │  │  │  │           └─ Handle multi-page
   │     │  │     │     │  │  │  │
   │     │  │     │     │  │  ├─ OFFICE?
   │     │  │     │     │  │  │  ├─ unstructured available?
   │     │  │     │     │  │  │  │  ├─ NO → TextProcessor
   │     │  │     │     │  │  │  │  └─ YES → OfficeProcessor
   │     │  │     │     │  │  │  │           ├─ Extract structured
   │     │  │     │     │  │  │  │           └─ Preserve formatting
   │     │  │     │     │  │  │  │
   │     │  │     │     │  │  └─ OTHER → TextProcessor
   │     │  │     │     │  │
   │     │  │     │     │  ├─ TEXT → TextProcessor
   │     │  │     │     │  │         ├─ UTF-8 decode
   │     │  │     │     │  │         └─ Metadata
   │     │  │     │     │  │
   │     │  │     │     │  └─ AUDIO/VIDEO → Not yet supported
   │     │  │     │     │
   │     │  │     │     └─ Get MediaContent
   │     │  │     │         ├─ media_type
   │     │  │     │         ├─ content
   │     │  │     │         ├─ content_format
   │     │  │     │         └─ metadata
   │     │  │     │
   │     │  │     └─ Check provider capabilities
   │     │  │        ├─ Supports media type?
   │     │  │        │  ├─ NO → Check vision fallback config
   │     │  │        │  │  ├─ Not configured → Error
   │     │  │        │  │  └─ Configured → VisionFallbackHandler
   │     │  │        │  │                   ├─ Description generation
   │     │  │        │  │                   └─ Replace with description
   │     │  │        │  │
   │     │  │        └─ YES → Format for provider
   │     │  │              ├─ OpenAI format
   │     │  │              ├─ Anthropic format
   │     │  │              └─ Local format
   │     │  │
   │     │  └─ Add to message
   │     │
   │     └─ Generate with media
   │
   └─ NO
      │
      └─ Text-only generation
```

---

## Component Interaction Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                    USER APPLICATION CODE                           │
│                                                                     │
│  llm.generate("Analyze this", media=["image.jpg", "doc.pdf"])    │
└────────────────────────────────────────────┬────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PROVIDER INTERFACE                               │
│                                                                     │
│  1. Validate media parameter                                       │
│  2. Create media handler for provider                              │
│  3. Process each media file                                        │
│  4. Build message with media content                               │
│  5. Send to provider API                                           │
└────────────────────────────────────────────┬────────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    │                                                 │
                    ▼                                                 ▼
        ┌──────────────────────────────┐              ┌──────────────────────────────┐
        │  MEDIA CAPABILITY CHECK       │              │  AUTO MEDIA HANDLER          │
        │                               │              │                              │
        │ for each media:              │              │ for each media:              │
        │ • get_media_capabilities()   │              │ • Detect media type          │
        │ • supports_media_type()      │              │ • Select processor           │
        │ • validate_media_content()   │              │ • Process file               │
        │                               │              │ • Return MediaContent        │
        └──────────────────────────────┘              └──────────────────────────────┘
                    │                                                 │
                    └────────────────────┬────────────────────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │  PROVIDER-SPECIFIC HANDLER         │
                    │                                    │
                    │ • Format media for provider API    │
                    │ • Validate format/size             │
                    │ • Apply provider constraints       │
                    └────────────────────┬───────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │  FINAL MESSAGE CONSTRUCTION        │
                    │                                    │
                    │ role: "user"                       │
                    │ content: [                         │
                    │   "Your prompt text",              │
                    │   { media content... }             │
                    │ ]                                  │
                    └────────────────────┬───────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │  PROVIDER API CALL                 │
                    │                                    │
                    │ • Send to provider                 │
                    │ • Streaming or standard            │
                    │ • Token tracking                   │
                    │ • Error handling                   │
                    └────────────────────┬───────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │  GENERATE RESPONSE                 │
                    │                                    │
                    │ return GenerateResponse            │
                    └────────────────────────────────────┘
```

---

## Processor Selection Algorithm

```
def select_processor(file_path: Path, media_type: MediaType):
    
    if media_type == IMAGE:
        if PIL_available:
            return ImageProcessor()
        else:
            raise UnsupportedMediaTypeError("PIL required for images")
    
    elif media_type == TEXT:
        return TextProcessor()  # Always available
    
    elif media_type == DOCUMENT:
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            if PyMuPDF4LLM_available:
                return PDFProcessor()
            else:
                return TextProcessor()  # Fallback
        
        elif extension in {'.docx', '.xlsx', '.pptx'}:
            if unstructured_available:
                return OfficeProcessor()
            else:
                return TextProcessor()  # Fallback
        
        else:  # txt, md, csv, etc.
            return TextProcessor()
    
    elif media_type in [AUDIO, VIDEO]:
        raise UnsupportedMediaTypeError("Not yet implemented")
    
    else:
        raise UnsupportedMediaTypeError(f"Unknown type: {media_type}")
```

---

## State Diagram: Media Processing Lifecycle

```
                    ┌─────────────────┐
                    │   File Received │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
              ┌────→│ File Exists?    │←───┐
              │     └────────┬────────┘    │
              │              │             │
        NO    │              │ YES         │ RETRY (limited)
              │              ▼             │
         ┌────┴──────┐   ┌─────────────────────┐
         │   ERROR   │   │ Size Check         │
         │ (not found)   │ < max_file_size?   │
         └───────────┘   └────────┬───────────┘
                                  │
                      ┌───────────┴───────────┐
                      │                       │
                 NO   │                       │ YES
                      │                       │
                      ▼                       ▼
              ┌──────────────────┐   ┌──────────────────┐
              │     ERROR        │   │ Detect Media Type│
              │ (size exceeded)  │   └────────┬─────────┘
              └──────────────────┘            │
                                              ▼
                                      ┌───────────────────┐
                                      │ Select Processor  │
                                      └────────┬──────────┘
                                               │
                                               ▼
                                      ┌───────────────────┐
                              ┌──────→│ Processor         │←────┐
                              │       │ Available?        │     │
                              │       └────────┬──────────┘     │
                              │                │                │
                              │            YES │                │ NO
                              │                │                │
                          FALLBACK         ▼                FALLBACK
                              │       ┌────────────────┐        │
                              │       │ Process File   │        │
                              │       │ (specific to   │        │
                              │       │  processor     │        │
                              │       └────────┬───────┘        │
                              │                │                │
                              │                ▼                │
                              │       ┌────────────────────┐    │
                              │       │ Check Fallback     │    │
                              │       │ Available?         │    │
                              │       └────────┬───────────┘    │
                              │                │                │
                              │            YES │                │
                              │                ▼                │
                              │       ┌────────────────────┐    │
                              │       │ Use Fallback       │    │
                              │       │ Processor          │    │
                              │       └────────┬───────────┘    │
                              │                │                │
                              │                │    NO           │
                              │                └────────────────┘
                              │                                 │
                              └─────────────────┬────────────────┘
                                                │
                                                ▼
                                      ┌───────────────────┐
                                      │ Create MediaContent│
                                      │ - media_type      │
                                      │ - content         │
                                      │ - format          │
                                      │ - metadata        │
                                      └────────┬──────────┘
                                               │
                                               ▼
                                      ┌───────────────────┐
                                      │ Integration into  │
                                      │ Message           │
                                      └────────┬──────────┘
                                               │
                                               ▼
                                      ┌───────────────────┐
                                      │   SUCCESS         │
                                      │  (Ready for       │
                                      │   Provider API)   │
                                      └───────────────────┘
```

---

## Architecture Dependencies

```
        ┌──────────────────────────────────────────────┐
        │    Pydantic (Data Validation)               │
        └──────────────────────────────────────────────┘
                          ▲
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
    ┌────────────────┐           ┌────────────────┐
    │ Core Types     │           │ Media Types    │
    │ (MediaContent) │           │ (MediaType)    │
    └────────────────┘           └────────────────┘
        ▲                                   ▲
        │                                   │
        └─────────────┬─────────────────────┘
                      │
                      ▼
        ┌──────────────────────────────┐
        │ Base Handlers               │
        │ (BaseMediaHandler)          │
        │ (BaseProviderMediaHandler)  │
        └──────────────────────────────┘
                      ▲
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
    ┌─────────────┐ ┌──────────────┐ ┌──────────┐
    │  Auto       │ │  Processors  │ │ Handlers │
    │  Handler    │ │ (Image, Text,│ │(OpenAI,  │
    │             │ │  PDF, Office)│ │Anthropic)│
    │             │ │              │ │          │
    └─────────────┘ └──────────────┘ └──────────┘
         ▲                ▲                ▲
         │                │                │
         │      ┌─────────┴────────┐      │
         │      │                  │      │
         │      ▼                  ▼      │
         │   ┌──────────────┐  ┌──────────┐
         │   │ Capabilities │  │ Utils    │
         │   │ Detection    │  │(Scaler) │
         │   └──────────────┘  └──────────┘
         │
         └──────┬───────────┐
                │           │
                ▼           ▼
         ┌────────────┐ ┌──────────────┐
         │ Factory    │ │ Config System│
         │ (create_llm)│ │              │
         └────────────┘ └──────────────┘
                │           │
                └─────┬─────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │ Provider Implementations │
         │ (OpenAI, Anthropic, etc) │
         └──────────────────────────┘
                      ▲
                      │
                ┌─────┴─────┐
                │           │
                ▼           ▼
         ┌────────────┐ ┌──────────┐
         │ Server     │ │ Session  │
         │ (FastAPI)  │ │          │
         └────────────┘ └──────────┘
                │           │
                └─────┬─────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │ User Application Layer   │
         └──────────────────────────┘
```

This comprehensive architecture diagram shows how the media system integrates with all components of AbstractCore, from user applications down through factories, processors, and ultimately to provider APIs. The modular design allows for flexible extension while maintaining clean separation of concerns.
