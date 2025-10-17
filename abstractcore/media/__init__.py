"""
AbstractCore Media Handler System.

This module provides unified media handling capabilities across all providers,
supporting various file types including images, documents, audio, and video.

The system follows AbstractCore's proven architectural patterns:
- Interface → Base → Provider-Specific implementations
- Capability detection using model_capabilities.json
- Unified API across all providers
"""

# Core types and base classes
from .base import BaseMediaHandler, BaseProviderMediaHandler
from .types import MediaContent, MediaType, ContentFormat, MultimodalMessage
from .auto_handler import AutoMediaHandler

# Media processing capabilities
from .capabilities import (
    MediaCapabilities,
    get_media_capabilities,
    is_vision_model,
    is_multimodal_model,
    get_supported_media_types,
    supports_images,
    supports_documents,
    get_max_images,
    should_use_text_embedding
)

# Processors for different file types
from .processors import ImageProcessor, TextProcessor, PDFProcessor, OfficeProcessor

# Provider-specific handlers
from .handlers import OpenAIMediaHandler, AnthropicMediaHandler, LocalMediaHandler

# Default media handler - automatically selects appropriate processor
class MediaHandler(AutoMediaHandler):
    """
    Default media handler that automatically selects the appropriate processor.

    This class provides automatic file type detection and processor selection,
    making it easy to process any supported media type with a single interface.
    """
    pass

# Convenience functions for common operations
def process_file(file_path: str) -> MediaContent:
    """
    Process a file using the automatic media handler.

    Args:
        file_path: Path to the file to process

    Returns:
        MediaContent object with processed content
    """
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

    # Handlers
    'OpenAIMediaHandler',
    'AnthropicMediaHandler',
    'LocalMediaHandler',

    # Legacy and convenience
    'MediaHandler',
    'process_file',
    'get_media_type_from_path'
]