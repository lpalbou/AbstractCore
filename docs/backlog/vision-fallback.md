# Vision Caption Fallback Implementation Plan

## Overview
Implement offline-first vision captioning fallback for text-only models using two-stage pipeline approach.

## Current Issue
When users provide images to text-only models, AbstractCore only provides basic filename placeholder: `[Image 1: photo.jpg]`, which provides no useful context.

## Solution Architecture

### Two-Stage Pipeline Approach
```
User Request: "What's in this image?" + image.jpg
    ↓
Stage 1: Vision Model → "A mountain hiking trail with wooden fence..."
    ↓
Stage 2: Text-Only Model → Uses description to answer user question
```

## Implementation Strategy

### 1. Configuration System Design

#### CLI Commands
```bash
# Download default small vision model (recommended)
abstractcore --download-vision-model
abstractcore --download-vision  # shorthand

# Configure existing provider/model for vision captioning
abstractcore --set-vision-caption qwen2.5vl:7b
abstractcore --set-vision-provider ollama --model qwen2.5vl:7b
abstractcore --set-vision-provider openai --model gpt-4o

# Management commands
abstractcore --vision-status     # Show current setup
abstractcore --list-vision       # Show available options
abstractcore --configure vision  # Interactive configuration
```

#### Configuration File Structure
```python
# ~/.abstractcore/config/vision.json
{
    "strategy": "two_stage",  # "two_stage", "disabled", "basic_metadata"
    "caption_provider": "ollama",
    "caption_model": "qwen2.5vl:7b",
    "fallback_chain": [
        {"provider": "ollama", "model": "qwen2.5vl:7b"},
        {"provider": "local_model", "model": "blip-base-caption"},
        {"provider": "openai", "model": "gpt-4o"}
    ],
    "local_models_path": "~/.abstractcore/models/"
}
```

### 2. Core Implementation Components

#### A. Vision Fallback Handler
```python
# abstractcore/media/vision_fallback.py
class VisionFallbackHandler:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)

    def create_description(self, image_path: str, user_prompt: str = None) -> str:
        """Generate description using configured vision model."""
        if not self._has_vision_capability():
            return self._show_setup_instructions()

        return self._generate_with_fallback(image_path)

    def _generate_with_fallback(self, image_path: str) -> str:
        """Try vision models in fallback chain order."""
        for provider_config in self.config["fallback_chain"]:
            try:
                return self._generate_description(provider_config, image_path)
            except Exception as e:
                logger.debug(f"Vision provider {provider_config} failed: {e}")
                continue

        return self._show_setup_instructions()
```

#### B. Configuration Manager
```python
# abstractcore/cli/vision_config.py
class VisionConfigManager:
    def set_vision_provider(self, provider: str, model: str):
        """Set the primary vision provider and model."""

    def download_vision_model(self, model_name: str = "blip-base-caption"):
        """Download lightweight vision model for offline use."""

    def get_vision_status(self):
        """Show current vision configuration status."""

    def list_available_options(self):
        """List all available vision providers and models."""
```

#### C. Model Download System
```python
# abstractcore/cli/model_downloader.py
AVAILABLE_VISION_MODELS = {
    "blip-base-caption": {
        "url": "Salesforce/blip-image-captioning-base",
        "size": "990MB",
        "description": "Basic image captioning model",
        "offline": True
    },
    "git-base": {
        "url": "microsoft/git-base",
        "size": "400MB",
        "description": "Lightweight captioning model",
        "offline": True
    },
    "vit-gpt2": {
        "url": "nlpconnect/vit-gpt2-image-captioning",
        "size": "500MB",
        "description": "ViT + GPT-2 captioning model",
        "offline": True
    }
}
```

### 3. Integration Points

#### A. LocalMediaHandler Enhancement
```python
# abstractcore/media/handlers/local_handler.py
def _create_text_embedded_message(self, media_contents, prompt):
    if not self.capabilities.vision_support:
        # Use vision fallback instead of basic placeholder
        fallback_handler = VisionFallbackHandler()

        for media_content in media_contents:
            if media_content.media_type == MediaType.IMAGE:
                description = fallback_handler.create_description(
                    media_content.file_path,
                    prompt
                )
                message_parts.append(description)
```

#### B. CLI Integration
```python
# abstractcore/cli/__main__.py
def main():
    parser = argparse.ArgumentParser()

    # Vision configuration commands
    parser.add_argument("--configure", choices=["vision"])
    parser.add_argument("--set-vision-provider", nargs=2, metavar=("PROVIDER", "MODEL"))
    parser.add_argument("--set-vision-caption", metavar="MODEL")
    parser.add_argument("--download-vision-model", nargs="?", const="blip-base-caption")
    parser.add_argument("--vision-status", action="store_true")
    parser.add_argument("--list-vision", action="store_true")
```

### 4. User Experience Flow

#### First-Time User (No Vision Setup)
```python
user_code = """
from abstractcore import create_llm
llm = create_llm("openai", model="gpt-4")  # text-only model
response = llm.generate("What's in this image?", media=["photo.jpg"])
"""

# Output:
# ⚠️  Vision capability not configured for text-only models.
#
# To enable image analysis with text-only models:
# 1. Download a vision model: abstractcore --download-vision-model
# 2. Or configure existing provider: abstractcore --set-vision-caption qwen2.5vl:7b
#
# For more options: abstractcore --list-vision
```

#### Configured User
```python
# After: abstractcore --download-vision-model
response = llm.generate("What's in this image?", media=["photo.jpg"])

# Gets: "Based on image analysis: A mountain hiking trail with wooden fence...
# User question: What's in this image?"
```

### 5. Implementation Timeline

#### Phase 1: Core Infrastructure (Week 1)
- [ ] VisionFallbackHandler class
- [ ] Configuration system with JSON storage
- [ ] Basic CLI commands (--set-vision-caption, --vision-status)
- [ ] Integration with LocalMediaHandler

#### Phase 2: Model Download System (Week 2)
- [ ] ModelDownloader with AVAILABLE_VISION_MODELS
- [ ] --download-vision-model command
- [ ] Local model management and caching
- [ ] Error handling and progress bars

#### Phase 3: Advanced Features (Week 3)
- [ ] Interactive --configure vision mode
- [ ] Fallback chain with multiple providers
- [ ] Performance optimization and caching
- [ ] Comprehensive testing and documentation

### 6. Configuration Examples

#### Simple Setup
```bash
# Easiest - download small model
abstractcore --download-vision-model

# Use existing Ollama model
abstractcore --set-vision-caption qwen2.5vl:7b

# Use cloud API
abstractcore --set-vision-provider openai --model gpt-4o
```

#### Advanced Setup
```bash
# Interactive configuration
abstractcore --configure vision

# Multiple fallbacks
abstractcore --set-vision-provider ollama qwen2.5vl:7b
# Then: abstractcore --add-vision-fallback openai gpt-4o
```

#### Status Checking
```bash
abstractcore --vision-status
# Output:
# Vision Configuration Status:
# ✅ Primary: ollama/qwen2.5vl:7b (available)
# ✅ Fallback: local/blip-base-caption (downloaded)
# ⚠️  Fallback: openai/gpt-4o (no API key)
```

## Success Criteria

1. **Zero Dependencies by Default**: Works offline without forced downloads
2. **Clear Error Messages**: Users know exactly how to enable vision fallback
3. **Simple Configuration**: One command to get started
4. **Flexible Options**: Supports local models, cloud APIs, and mixed setups
5. **Graceful Degradation**: Always works, gets better with configuration

## Key Design Principles

- **Offline-First**: No internet required after initial setup
- **User Choice**: Multiple options for different preferences/constraints
- **Clear Instructions**: Failed attempts provide actionable guidance
- **Progressive Enhancement**: Basic → local models → cloud APIs
- **Minimal Friction**: One command to get started for most users