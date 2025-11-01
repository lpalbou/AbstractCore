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

        # Store model name for Qwen-specific optimizations
        self.model_name = kwargs.get('model_name', '')

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
            self.logger.debug(f"OpenAI Handler - MediaContent metadata: {media_content.metadata}")
            self.logger.debug(f"OpenAI Handler - Found detail_level: {detail_level}")
            
            # Auto-adjust detail level for Qwen models to prevent context overflow
            if self._is_qwen_model() and detail_level == 'auto':
                detail_level = self._get_optimal_detail_for_qwen(media_content)
                self.logger.debug(f"OpenAI Handler - Qwen auto-adjusted detail_level: {detail_level}")
            
            if detail_level in self.supported_image_detail:
                image_obj["image_url"]["detail"] = detail_level
                self.logger.info(f"OpenAI Handler - Setting detail level to '{detail_level}' for image")
            else:
                self.logger.warning(f"OpenAI Handler - Invalid detail level '{detail_level}', supported: {self.supported_image_detail}")

        return image_obj

    def _is_qwen_model(self) -> bool:
        """Check if the current model is a Qwen vision model."""
        if not hasattr(self, 'model_name') or not self.model_name:
            return False
        
        model_name_lower = self.model_name.lower()
        return any(qwen_variant in model_name_lower for qwen_variant in [
            'qwen3-vl', 'qwen2.5-vl', 'qwen-vl', 'qwen/qwen3-vl', 'qwen/qwen2.5-vl'
        ])

    def _get_optimal_detail_for_qwen(self, media_content: MediaContent) -> str:
        """
        Determine optimal detail level for Qwen models based on context constraints.
        
        According to SiliconFlow documentation:
        - detail=low: 256 tokens per image (448x448 resize)
        - detail=high: Variable tokens based on resolution (can be 24,576+ tokens)
        
        For Qwen3-VL-30B with 131,072 token context limit, we should use detail=low
        when processing multiple images to avoid context overflow.
        """
        # Get model context limit
        max_tokens = self.model_capabilities.get('max_tokens', 32768)
        max_image_tokens = self.model_capabilities.get('max_image_tokens', 24576)
        
        # Estimate how many images we might be processing
        # This is a heuristic - in practice we'd need the full batch context
        estimated_images = getattr(self, '_estimated_image_count', 1)
        
        # Calculate potential token usage with high detail
        high_detail_tokens = estimated_images * max_image_tokens
        
        # Use low detail if high detail would consume >60% of context
        context_threshold = max_tokens * 0.6
        
        if high_detail_tokens > context_threshold:
            self.logger.info(f"Using detail=low for Qwen model: {estimated_images} images would consume "
                           f"{high_detail_tokens:,} tokens (>{context_threshold:,} threshold)")
            return 'low'
        else:
            return 'high'

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
            # Image token estimation varies by model
            detail_level = media_content.metadata.get('detail_level', 'auto')

            if detail_level == 'low':
                # Qwen models use 256 tokens for low detail, OpenAI uses 85
                if self._is_qwen_model():
                    return 256  # Qwen low detail token count
                else:
                    return 85   # OpenAI low detail token count
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