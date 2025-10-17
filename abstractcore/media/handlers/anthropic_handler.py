"""
Anthropic-specific media handler.

This module provides media formatting capabilities specifically for Anthropic's API,
including support for Claude Vision and document processing.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..base import BaseProviderMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat


class AnthropicMediaHandler(BaseProviderMediaHandler):
    """
    Media handler for Anthropic API formatting.

    Formats media content according to Anthropic's API specifications for
    Claude Vision and other multimodal capabilities.
    """

    def __init__(self, model_capabilities: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize Anthropic media handler.

        Args:
            model_capabilities: Model capabilities from model_capabilities.json
            **kwargs: Additional configuration
        """
        super().__init__("anthropic", model_capabilities, **kwargs)

        # Anthropic-specific configuration
        self.max_image_size = kwargs.get('max_image_size', 5 * 1024 * 1024)  # 5MB
        self.max_images_per_message = kwargs.get('max_images_per_message', 20)  # Claude supports up to 20 images

        self.logger.debug(f"Initialized Anthropic media handler with capabilities: {self.capabilities}")

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Process file using appropriate processor and return Anthropic-formatted content.

        Args:
            file_path: Path to the file
            media_type: Type of media
            **kwargs: Processing options

        Returns:
            MediaContent formatted for Anthropic
        """
        # Use appropriate processor based on media type
        if media_type == MediaType.IMAGE:
            from ..processors import ImageProcessor
            # Ensure maximum resolution for best Anthropic vision results
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
        Format media content for Anthropic API.

        Args:
            media_content: MediaContent to format

        Returns:
            Dictionary formatted for Anthropic API
        """
        if media_content.media_type == MediaType.IMAGE:
            return self._format_image_for_anthropic(media_content)
        elif media_content.media_type in [MediaType.DOCUMENT, MediaType.TEXT]:
            return self._format_text_for_anthropic(media_content)
        else:
            raise MediaProcessingError(f"Unsupported media type for Anthropic: {media_content.media_type}")

    def _format_image_for_anthropic(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format image content for Anthropic's image format.

        Args:
            media_content: Image MediaContent

        Returns:
            Anthropic-compatible image object
        """
        if media_content.content_format != ContentFormat.BASE64:
            raise MediaProcessingError("Anthropic image formatting requires base64 content")

        # Create Anthropic image object
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_content.mime_type,
                "data": media_content.content
            }
        }

    def _format_text_for_anthropic(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format text/document content for Anthropic API.

        Args:
            media_content: Text/Document MediaContent

        Returns:
            Anthropic-compatible text object
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
        Create a multimodal message for Anthropic API.

        Args:
            text: Text content
            media_contents: List of media contents

        Returns:
            Anthropic-compatible message object
        """
        content = []

        # Add text content first (Anthropic prefers text before images)
        if text.strip():
            content.append({
                "type": "text",
                "text": text
            })

        # Add media contents
        image_count = 0
        for media_content in media_contents:
            if self.can_handle_media(media_content):
                # Check image limits
                if media_content.media_type == MediaType.IMAGE:
                    if image_count >= self.max_images_per_message:
                        self.logger.warning(f"Skipping image - exceeded max images per message ({self.max_images_per_message})")
                        continue
                    image_count += 1

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
        Validate if media content is compatible with specific Anthropic model.

        Args:
            media_content: MediaContent to validate
            model: Anthropic model name

        Returns:
            True if compatible, False otherwise
        """
        model_lower = model.lower()

        # Vision model validation
        if media_content.media_type == MediaType.IMAGE:
            # Check if model supports vision
            if not self.model_capabilities.get('vision_support', False):
                return False

            # Check image size
            if hasattr(media_content, 'metadata'):
                file_size = media_content.metadata.get('file_size', 0)
                if file_size > self.max_image_size:
                    return False

            # Model-specific checks
            if 'claude-3' in model_lower:
                return True  # All Claude 3 models support vision
            elif 'claude-3.5' in model_lower:
                return True  # All Claude 3.5 models support vision
            elif 'claude-4' in model_lower:
                return True  # Future Claude 4 models

        # Text/document validation
        elif media_content.media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # All Anthropic models support text
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
            # Anthropic image token estimation
            # Roughly ~1600 tokens per image for most cases
            # This varies based on image content and complexity
            return 1600

        elif media_content.media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # Rough estimation: 3.5 characters per token (slightly better than GPT)
            content_length = len(str(media_content.content))
            return int(content_length / 3.5)

        return 0

    def get_model_media_limits(self, model: str) -> Dict[str, Any]:
        """
        Get media-specific limits for Anthropic model.

        Args:
            model: Anthropic model name

        Returns:
            Dictionary of limits
        """
        limits = {
            'max_images_per_message': self.max_images_per_message,
            'max_image_size_bytes': self.max_image_size,
            'supported_image_formats': ['png', 'jpeg', 'jpg', 'gif', 'webp'],
            'max_image_resolution': '1568x1568'  # Anthropic's documented limit
        }

        model_lower = model.lower()

        # Model-specific adjustments
        if 'claude-3.5' in model_lower:
            limits.update({
                'parallel_tool_use': self.model_capabilities.get('parallel_tools', False),
                'disable_parallel_tool_use_option': True
            })
        elif 'claude-3' in model_lower and 'opus' in model_lower:
            limits.update({
                'parallel_tool_use': False,  # Claude 3 Opus doesn't support parallel tools
                'max_tools_per_request': 1
            })

        return limits

    def create_document_analysis_prompt(self, media_contents: List[MediaContent],
                                      analysis_type: str = "general") -> str:
        """
        Create a specialized prompt for document analysis with Claude.

        Args:
            media_contents: List of media contents (documents/images)
            analysis_type: Type of analysis ('general', 'summary', 'extract', 'qa')

        Returns:
            Optimized prompt for document analysis
        """
        document_count = len([mc for mc in media_contents if mc.media_type in [MediaType.DOCUMENT, MediaType.TEXT]])
        image_count = len([mc for mc in media_contents if mc.media_type == MediaType.IMAGE])

        if analysis_type == "summary":
            return f"""Please analyze and summarize the provided {'documents' if document_count > 0 else 'images'}.
            Provide a clear, concise summary that captures the key information, main points, and important details.

            {"Documents provided: " + str(document_count) if document_count > 0 else ""}
            {"Images provided: " + str(image_count) if image_count > 0 else ""}"""

        elif analysis_type == "extract":
            return f"""Please extract all relevant information from the provided {'documents' if document_count > 0 else 'images'}.
            Focus on:
            - Key facts and data points
            - Important names, dates, and numbers
            - Structured information like tables or lists
            - Any actionable items or conclusions

            Present the information in a clear, organized format."""

        elif analysis_type == "qa":
            return f"""I will ask questions about the provided {'documents' if document_count > 0 else 'images'}.
            Please read through all the content carefully and be prepared to answer detailed questions about:
            - Specific information contained in the materials
            - Relationships between different pieces of information
            - Analysis and interpretation of the content

            Please confirm that you have reviewed all the provided materials."""

        else:  # general
            return f"""Please analyze the provided {'documents' if document_count > 0 else 'images'} and provide:
            1. A brief overview of what is contained in each item
            2. Key insights or important information
            3. Any notable patterns, relationships, or conclusions
            4. Suggestions for how this information might be used or what actions might be taken

            Be thorough but concise in your analysis."""