"""
Media capability detection and management for AbstractCore.

This module provides comprehensive capability detection for multimodal models,
leveraging the existing model_capabilities.json infrastructure to determine
what media types and formats each model supports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

from ..architectures import get_model_capabilities
from .types import MediaType


@dataclass
class MediaCapabilities:
    """
    Comprehensive media capabilities for a specific model.

    This class aggregates all media-related capabilities from model_capabilities.json
    and provides convenient methods for checking media support.
    """
    model_name: str

    # Core media support
    vision_support: bool = False
    audio_support: bool = False
    video_support: bool = False

    # Image capabilities
    max_images_per_message: int = 1
    supported_image_formats: List[str] = field(default_factory=lambda: ['jpg', 'jpeg', 'png'])
    image_resolutions: List[str] = field(default_factory=list)
    max_image_size_bytes: int = 5 * 1024 * 1024  # 5MB default

    # Document capabilities
    document_support: bool = True  # Most models can handle text documents
    max_document_size_bytes: int = 50 * 1024 * 1024  # 50MB default

    # Provider-specific features
    multimodal_message_support: bool = True
    text_embedding_preferred: bool = False  # For local models
    streaming_media_support: bool = False

    # Advanced features
    parallel_media_processing: bool = False
    media_token_estimation: bool = True

    @classmethod
    def from_model_capabilities(cls, model: str, provider: str = None) -> 'MediaCapabilities':
        """
        Create MediaCapabilities from model_capabilities.json data.

        Args:
            model: Model name to look up capabilities for
            provider: Optional provider name for provider-specific adjustments

        Returns:
            MediaCapabilities instance with detected capabilities
        """
        caps = get_model_capabilities(model)
        if not caps:
            caps = {}

        # Base capabilities from JSON
        instance = cls(
            model_name=model,
            vision_support=caps.get('vision_support', False),
            audio_support=caps.get('audio_support', False),
            video_support=caps.get('video_support', False),
            image_resolutions=caps.get('image_resolutions', [])
        )

        # Provider-specific adjustments
        if provider:
            instance._apply_provider_adjustments(provider, caps)

        # Model-specific adjustments based on model name patterns
        instance._apply_model_adjustments(caps)

        return instance

    def _apply_provider_adjustments(self, provider: str, caps: Dict[str, Any]):
        """Apply provider-specific capability adjustments."""
        provider_lower = provider.lower()

        if provider_lower == "openai":
            self.max_images_per_message = 10 if "gpt-4o" in self.model_name.lower() else 1
            self.max_image_size_bytes = 20 * 1024 * 1024  # 20MB for OpenAI
            self.supported_image_formats = ['png', 'jpeg', 'jpg', 'gif', 'webp']
            self.streaming_media_support = True

        elif provider_lower == "anthropic":
            self.max_images_per_message = 20  # Claude supports up to 20 images
            self.max_image_size_bytes = 5 * 1024 * 1024  # 5MB for Anthropic
            self.supported_image_formats = ['png', 'jpeg', 'jpg', 'gif', 'webp']
            self.streaming_media_support = True

        elif provider_lower in ["ollama", "mlx", "lmstudio"]:
            self.text_embedding_preferred = True  # Local models often prefer text
            self.multimodal_message_support = True
            self.streaming_media_support = False
            self.max_image_size_bytes = 10 * 1024 * 1024  # 10MB for local

        elif provider_lower == "huggingface":
            self.streaming_media_support = False
            self.max_image_size_bytes = 15 * 1024 * 1024  # 15MB for HF

    def _apply_model_adjustments(self, caps: Dict[str, Any]):
        """Apply model-specific capability adjustments based on model patterns."""
        model_lower = self.model_name.lower()

        # Vision model patterns
        if any(pattern in model_lower for pattern in ['vision', 'vl', 'visual']):
            self.vision_support = True
            if 'qwen' in model_lower:
                self.max_images_per_message = 5  # Qwen-VL supports multiple images

        # Multimodal model patterns
        if any(pattern in model_lower for pattern in ['4o', 'multimodal', 'omni']):
            self.vision_support = True
            if 'audio' not in caps or caps.get('audio_support'):
                self.audio_support = True

        # Local model adjustments
        if any(pattern in model_lower for pattern in ['llama', 'qwen', 'phi', 'gemma']):
            self.text_embedding_preferred = True

    def supports_media_type(self, media_type: MediaType) -> bool:
        """Check if the model supports a specific media type."""
        if media_type == MediaType.IMAGE:
            return self.vision_support
        elif media_type == MediaType.AUDIO:
            return self.audio_support
        elif media_type == MediaType.VIDEO:
            return self.video_support
        elif media_type in [MediaType.DOCUMENT, MediaType.TEXT]:
            return self.document_support
        return False

    def get_image_limits(self) -> Dict[str, Any]:
        """Get image-specific limits and capabilities."""
        return {
            'max_images_per_message': self.max_images_per_message,
            'supported_formats': self.supported_image_formats,
            'max_size_bytes': self.max_image_size_bytes,
            'supported_resolutions': self.image_resolutions,
            'vision_support': self.vision_support
        }

    def get_document_limits(self) -> Dict[str, Any]:
        """Get document-specific limits and capabilities."""
        return {
            'max_size_bytes': self.max_document_size_bytes,
            'document_support': self.document_support,
            'text_embedding_preferred': self.text_embedding_preferred
        }

    def estimate_media_tokens(self, media_type: MediaType, content_size: int = 0) -> int:
        """
        Estimate token usage for media content.

        Args:
            media_type: Type of media
            content_size: Size of content in bytes (optional)

        Returns:
            Estimated token count
        """
        if not self.media_token_estimation:
            return 0

        if media_type == MediaType.IMAGE and self.vision_support:
            # Base token cost for images varies by model
            model_lower = self.model_name.lower()
            if 'gpt-4o' in model_lower:
                return 85 + (170 * 4)  # Simplified GPT-4o calculation
            elif 'claude' in model_lower:
                return 1600  # Anthropic standard
            else:
                return 512  # Conservative estimate for local models

        elif media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # Text content token estimation
            if content_size > 0:
                return content_size // 4  # ~4 chars per token
            return 100  # Default estimate

        return 0

    def validate_media_content(self, media_type: MediaType, file_size: int = 0,
                              format: str = None) -> tuple[bool, Optional[str]]:
        """
        Validate if media content meets model requirements.

        Args:
            media_type: Type of media
            file_size: Size of file in bytes
            format: File format/extension

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.supports_media_type(media_type):
            return False, f"Model {self.model_name} does not support {media_type.value} content"

        if media_type == MediaType.IMAGE:
            if file_size > self.max_image_size_bytes:
                return False, f"Image size ({file_size} bytes) exceeds limit ({self.max_image_size_bytes} bytes)"

            if format and format.lower() not in [f.lower() for f in self.supported_image_formats]:
                return False, f"Image format '{format}' not supported. Supported: {self.supported_image_formats}"

        elif media_type in [MediaType.DOCUMENT, MediaType.TEXT]:
            if file_size > self.max_document_size_bytes:
                return False, f"Document size ({file_size} bytes) exceeds limit ({self.max_document_size_bytes} bytes)"

        return True, None

    def get_processing_strategy(self, media_type: MediaType) -> str:
        """
        Get the recommended processing strategy for this media type.

        Returns:
            Processing strategy: 'multimodal', 'text_embedding', or 'unsupported'
        """
        if not self.supports_media_type(media_type):
            return 'unsupported'

        if media_type == MediaType.IMAGE and self.vision_support:
            if self.text_embedding_preferred:
                return 'text_embedding'  # Local models often prefer text description
            else:
                return 'multimodal'

        elif media_type in [MediaType.DOCUMENT, MediaType.TEXT]:
            return 'text_embedding'  # Always embed documents as text

        return 'unsupported'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'vision_support': self.vision_support,
            'audio_support': self.audio_support,
            'video_support': self.video_support,
            'max_images_per_message': self.max_images_per_message,
            'supported_image_formats': self.supported_image_formats,
            'image_resolutions': self.image_resolutions,
            'max_image_size_bytes': self.max_image_size_bytes,
            'document_support': self.document_support,
            'max_document_size_bytes': self.max_document_size_bytes,
            'multimodal_message_support': self.multimodal_message_support,
            'text_embedding_preferred': self.text_embedding_preferred,
            'streaming_media_support': self.streaming_media_support,
            'parallel_media_processing': self.parallel_media_processing,
            'media_token_estimation': self.media_token_estimation
        }


def get_media_capabilities(model: str, provider: str = None) -> MediaCapabilities:
    """
    Get comprehensive media capabilities for a model.

    Args:
        model: Model name
        provider: Optional provider name for provider-specific adjustments

    Returns:
        MediaCapabilities instance
    """
    return MediaCapabilities.from_model_capabilities(model, provider)


def is_vision_model(model: str) -> bool:
    """Quick check if a model supports vision."""
    caps = get_media_capabilities(model)
    return caps.vision_support


def is_multimodal_model(model: str) -> bool:
    """Quick check if a model supports any multimodal content."""
    caps = get_media_capabilities(model)
    return caps.vision_support or caps.audio_support or caps.video_support


def get_supported_media_types(model: str, provider: str = None) -> List[MediaType]:
    """
    Get list of supported media types for a model.

    Args:
        model: Model name
        provider: Optional provider name

    Returns:
        List of supported MediaType values
    """
    caps = get_media_capabilities(model, provider)
    supported = []

    if caps.vision_support:
        supported.append(MediaType.IMAGE)
    if caps.audio_support:
        supported.append(MediaType.AUDIO)
    if caps.video_support:
        supported.append(MediaType.VIDEO)
    if caps.document_support:
        supported.extend([MediaType.DOCUMENT, MediaType.TEXT])

    return supported


# Convenience functions for common capability checks
def supports_images(model: str, provider: str = None) -> bool:
    """Check if model supports image processing."""
    return get_media_capabilities(model, provider).vision_support


def supports_documents(model: str, provider: str = None) -> bool:
    """Check if model supports document processing."""
    return get_media_capabilities(model, provider).document_support


def get_max_images(model: str, provider: str = None) -> int:
    """Get maximum images per message for model."""
    return get_media_capabilities(model, provider).max_images_per_message


def should_use_text_embedding(model: str, provider: str = None) -> bool:
    """Check if model prefers text embedding over multimodal messages."""
    return get_media_capabilities(model, provider).text_embedding_preferred