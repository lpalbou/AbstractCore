"""
OpenAI-specific media handler.

This module provides media formatting capabilities specifically for OpenAI's API,
including support for GPT-4 Vision, audio models, and document processing.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..base import BaseProviderMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat


class OpenAIMediaHandler(BaseProviderMediaHandler):
    """
    Media handler for OpenAI API formatting.

    Formats media content according to OpenAI's API specifications for
    GPT-4 Vision, audio models, and other multimodal capabilities.
    """

    def __init__(self, model_capabilities: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize OpenAI media handler.

        Args:
            model_capabilities: Model capabilities from model_capabilities.json
            **kwargs: Additional configuration
        """
        super().__init__("openai", model_capabilities, **kwargs)

        # OpenAI-specific configuration
        self.max_image_size = kwargs.get('max_image_size', 20 * 1024 * 1024)  # 20MB
        self.supported_image_detail = kwargs.get('supported_image_detail', ['auto', 'low', 'high'])

        self.logger.debug(f"Initialized OpenAI media handler with capabilities: {self.capabilities}")

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Process file using appropriate processor and return OpenAI-formatted content.

        Args:
            file_path: Path to the file
            media_type: Type of media
            **kwargs: Processing options

        Returns:
            MediaContent formatted for OpenAI
        """
        # Use appropriate processor based on media type
        if media_type == MediaType.IMAGE:
            from ..processors import ImageProcessor
            # Ensure maximum resolution for best OpenAI vision results
            if 'prefer_max_resolution' not in kwargs:
                kwargs['prefer_max_resolution'] = True
            processor = ImageProcessor(**kwargs)
        elif media_type == MediaType.DOCUMENT:
            if file_path.suffix.lower() == '.pdf':
                from ..processors import PDFProcessor
                processor = PDFProcessor(**kwargs)
            else:
                from ..processors import TextProcessor
                processor = TextProcessor(**kwargs)
        else:
            from ..processors import TextProcessor
            processor = TextProcessor(**kwargs)

        # Process the file
        result = processor.process_file(file_path, **kwargs)

        if not result.success:
            raise MediaProcessingError(f"Failed to process {file_path}: {result.error_message}")

        return result.media_content

    def format_for_provider(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format media content for OpenAI API.

        Args:
            media_content: MediaContent to format

        Returns:
            Dictionary formatted for OpenAI API
        """
        if media_content.media_type == MediaType.IMAGE:
            return self._format_image_for_openai(media_content)
        elif media_content.media_type in [MediaType.DOCUMENT, MediaType.TEXT]:
            return self._format_text_for_openai(media_content)
        else:
            raise MediaProcessingError(f"Unsupported media type for OpenAI: {media_content.media_type}")

    def _format_image_for_openai(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format image content for OpenAI's image_url format.

        Args:
            media_content: Image MediaContent

        Returns:
            OpenAI-compatible image object
        """
        if media_content.content_format != ContentFormat.BASE64:
            raise MediaProcessingError("OpenAI image formatting requires base64 content")

        # Construct data URL
        data_url = f"data:{media_content.mime_type};base64,{media_content.content}"

        # Create OpenAI image object
        image_obj = {
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        }

        # Add detail level if supported by model
        if self.model_capabilities.get('vision_support'):
            detail_level = media_content.metadata.get('detail_level', 'auto')
            if detail_level in self.supported_image_detail:
                image_obj["image_url"]["detail"] = detail_level

        return image_obj

    def _format_text_for_openai(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format text/document content for OpenAI API.

        Args:
            media_content: Text/Document MediaContent

        Returns:
            OpenAI-compatible text object
        """
        if isinstance(media_content.content, bytes):
            content = media_content.content.decode('utf-8')
        else:
            content = str(media_content.content)

        return {
            "type": "text",
            "text": content
        }

    def create_multimodal_message(self, text: str, media_contents: List[MediaContent]) -> Dict[str, Any]:
        """
        Create a multimodal message for OpenAI API.

        Args:
            text: Text content
            media_contents: List of media contents

        Returns:
            OpenAI-compatible message object
        """
        content = []

        # Add text content
        if text.strip():
            content.append({
                "type": "text",
                "text": text
            })

        # Add media contents
        for media_content in media_contents:
            if self.can_handle_media(media_content):
                formatted_content = self.format_for_provider(media_content)
                content.append(formatted_content)
            else:
                self.logger.warning(f"Skipping unsupported media type: {media_content.media_type}")

        return {
            "role": "user",
            "content": content
        }

    def validate_media_for_model(self, media_content: MediaContent, model: str) -> bool:
        """
        Validate if media content is compatible with specific OpenAI model.

        Args:
            media_content: MediaContent to validate
            model: OpenAI model name

        Returns:
            True if compatible, False otherwise
        """
        model_lower = model.lower()

        # Vision model validation
        if media_content.media_type == MediaType.IMAGE:
            # Check if model supports vision
            if not self.model_capabilities.get('vision_support', False):
                return False

            # Model-specific checks
            if 'gpt-4' in model_lower and 'vision' in model_lower:
                return True
            elif 'gpt-4o' in model_lower:
                return True
            elif 'gpt-4' in model_lower:
                # Regular GPT-4 models don't support vision
                return False

        # Text/document validation
        elif media_content.media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # All OpenAI models support text
            return True

        # Audio validation (future support)
        elif media_content.media_type == MediaType.AUDIO:
            return self.model_capabilities.get('audio_support', False)

        return False

    def estimate_tokens_for_media(self, media_content: MediaContent) -> int:
        """
        Estimate token usage for media content.

        Args:
            media_content: MediaContent to estimate

        Returns:
            Estimated token count
        """
        if media_content.media_type == MediaType.IMAGE:
            # OpenAI image token estimation
            # Base cost varies by detail level and image size
            detail_level = media_content.metadata.get('detail_level', 'auto')

            if detail_level == 'low':
                return 85  # Low detail images use 85 tokens
            else:
                # High detail calculation based on image dimensions
                width = media_content.metadata.get('final_size', [512, 512])[0]
                height = media_content.metadata.get('final_size', [512, 512])[1]

                # OpenAI's tile-based calculation (simplified)
                tiles_width = (width + 511) // 512
                tiles_height = (height + 511) // 512
                total_tiles = tiles_width * tiles_height

                return 85 + (170 * total_tiles)

        elif media_content.media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # Rough estimation: 4 characters per token
            content_length = len(str(media_content.content))
            return content_length // 4

        return 0

    def get_model_media_limits(self, model: str) -> Dict[str, Any]:
        """
        Get media-specific limits for OpenAI model.

        Args:
            model: OpenAI model name

        Returns:
            Dictionary of limits
        """
        limits = {
            'max_images_per_message': 1,
            'max_image_size_bytes': self.max_image_size,
            'supported_image_formats': ['png', 'jpeg', 'jpg', 'gif', 'webp'],
            'max_detail_level': 'high'
        }

        model_lower = model.lower()

        # Model-specific adjustments
        if 'gpt-4o' in model_lower:
            limits.update({
                'max_images_per_message': 10,  # GPT-4o supports multiple images
                'supports_audio': self.model_capabilities.get('audio_support', False),
                'supports_video': False  # Not yet supported
            })

        return limits