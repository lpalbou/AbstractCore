"""
AbstractCore Media Handler System.

This module provides unified media handling capabilities across all providers,
supporting various file types including images, documents, audio, and video.

The system follows AbstractCore's proven architectural patterns:
- Interface → Base → Provider-Specific implementations
- Capability detection using model_capabilities.json
- Unified API across all providers
"""

from __future__ import annotations

# NOTE: Keep this package import-safe for minimal installs.
# Many submodules have optional dependencies (Pillow, PyMuPDF4LLM, unstructured, ...).
# Import them lazily so `from abstractcore.media.capabilities import ...` works without extras.

from importlib import import_module
from typing import Any

# Core types and base classes (dependency-free at import time)
from .base import BaseMediaHandler, BaseProviderMediaHandler
from .types import MediaContent, MediaType, ContentFormat, MultimodalMessage

# Capability helpers (dependency-free at import time)
from .capabilities import (
    MediaCapabilities,
    get_media_capabilities,
    is_vision_model,
    is_multimodal_model,
    get_supported_media_types,
    supports_images,
    supports_documents,
    get_max_images,
    should_use_text_embedding,
)


def __getattr__(name: str) -> Any:
    """Lazy attribute loader for optional media components."""
    lazy_map = {
        # Handlers
        "OpenAIMediaHandler": ("abstractcore.media.handlers.openai_handler", "OpenAIMediaHandler"),
        "AnthropicMediaHandler": ("abstractcore.media.handlers.anthropic_handler", "AnthropicMediaHandler"),
        "LocalMediaHandler": ("abstractcore.media.handlers.local_handler", "LocalMediaHandler"),

        # Auto handler
        "AutoMediaHandler": ("abstractcore.media.auto_handler", "AutoMediaHandler"),

        # Processors
        "ImageProcessor": ("abstractcore.media.processors.image_processor", "ImageProcessor"),
        "TextProcessor": ("abstractcore.media.processors.text_processor", "TextProcessor"),
        "PDFProcessor": ("abstractcore.media.processors.pdf_processor", "PDFProcessor"),
        "OfficeProcessor": ("abstractcore.media.processors.office_processor", "OfficeProcessor"),
        "AudioProcessor": ("abstractcore.media.processors.audio_processor", "AudioProcessor"),
        "VideoProcessor": ("abstractcore.media.processors.video_processor", "VideoProcessor"),
    }

    if name == "MediaHandler":
        AutoMediaHandler = getattr(import_module("abstractcore.media.auto_handler"), "AutoMediaHandler")

        class MediaHandler(AutoMediaHandler):  # type: ignore[misc]
            """Default media handler (alias of AutoMediaHandler)."""

        globals()["MediaHandler"] = MediaHandler
        return MediaHandler

    target = lazy_map.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr = target
    mod = import_module(module_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value

# Convenience functions for common operations
def process_file(file_path: str) -> MediaContent:
    """
    Process a file using the automatic media handler.

    Args:
        file_path: Path to the file to process

    Returns:
        MediaContent object with processed content
    """
    from .auto_handler import AutoMediaHandler
    handler = AutoMediaHandler()
    result = handler.process_file(file_path)
    if result.success:
        return result.media_content
    else:
        from .base import MediaProcessingError
        raise MediaProcessingError(result.error_message)

def get_media_type_from_path(file_path: str) -> MediaType:
    """
    Determine media type from file path.

    Args:
        file_path: Path to the file

    Returns:
        MediaType enum value
    """
    from .types import detect_media_type
    from pathlib import Path
    return detect_media_type(Path(file_path))

# Export all public components
__all__ = [
    # Core types
    'MediaContent',
    'MediaType',
    'ContentFormat',
    'MultimodalMessage',

    # Base classes
    'BaseMediaHandler',
    'BaseProviderMediaHandler',
    'AutoMediaHandler',

    # Capability detection
    'MediaCapabilities',
    'get_media_capabilities',
    'is_vision_model',
    'is_multimodal_model',
    'get_supported_media_types',
    'supports_images',
    'supports_documents',
    'get_max_images',
    'should_use_text_embedding',

    # Processors
    'ImageProcessor',
    'TextProcessor',
    'PDFProcessor',
    'OfficeProcessor',
    'AudioProcessor',
    'VideoProcessor',

    # Handlers
    'OpenAIMediaHandler',
    'AnthropicMediaHandler',
    'LocalMediaHandler',

    # Legacy and convenience
    'MediaHandler',
    'process_file',
    'get_media_type_from_path'
]
