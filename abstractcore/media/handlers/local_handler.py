"""
Local provider media handler.

This module provides media formatting capabilities for local providers
like Ollama, MLX, LMStudio that handle media differently than cloud APIs.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..base import BaseProviderMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat

# Import vision detection from existing architecture system
try:
    from ...architectures.detection import supports_vision
    VISION_DETECTION_AVAILABLE = True
except ImportError:
    VISION_DETECTION_AVAILABLE = False
    supports_vision = None


class LocalMediaHandler(BaseProviderMediaHandler):
    """
    Media handler for local providers (Ollama, MLX, LMStudio).

    Formats media content for local model providers that may have different
    capabilities and formatting requirements than cloud APIs.
    """

    def __init__(self, provider_name: str, model_capabilities: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize local media handler.

        Args:
            provider_name: Name of the local provider ('ollama', 'mlx', 'lmstudio')
            model_capabilities: Model capabilities from model_capabilities.json
            **kwargs: Additional configuration including:
                - model_name: Name of the specific model (for vision detection)
        """
        super().__init__(provider_name, model_capabilities, **kwargs)

        # Store model name for vision capability detection
        self.model_name = kwargs.get('model_name', None)

        # Local provider configuration
        self.max_image_size = kwargs.get('max_image_size', 10 * 1024 * 1024)  # 10MB for local
        self.prefer_text_extraction = kwargs.get('prefer_text_extraction', True)
        self.embed_images_in_text = kwargs.get('embed_images_in_text', False)

        self.logger.debug(f"Initialized {provider_name} local media handler with model={self.model_name}, capabilities: {self.capabilities}")

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Process file using appropriate processor for local providers.

        Args:
            file_path: Path to the file
            media_type: Type of media
            **kwargs: Processing options

        Returns:
            MediaContent formatted for local providers
        """
        # Local providers often prefer text extraction over binary content
        processing_kwargs = kwargs.copy()

        if media_type == MediaType.IMAGE:
            if self.capabilities.vision_support:
                from ..processors import ImageProcessor
                # Pass model name for model-specific resolution optimization
                if self.model_name:
                    processing_kwargs['model_name'] = self.model_name
                    processing_kwargs['prefer_max_resolution'] = True
                processor = ImageProcessor(**processing_kwargs)
            else:
                # If no vision support, skip image processing
                raise MediaProcessingError(f"Provider {self.provider_name} does not support image processing")

        elif media_type == MediaType.DOCUMENT:
            if file_path.suffix.lower() == '.pdf':
                from ..processors import PDFProcessor
                # Configure PDF processor for text-optimized output
                processing_kwargs.update({
                    'extract_images': False,  # Local providers typically don't need embedded images
                    'markdown_output': True,  # Prefer markdown for better structure
                    'preserve_tables': True
                })
                processor = PDFProcessor(**processing_kwargs)
            else:
                from ..processors import TextProcessor
                processor = TextProcessor(**processing_kwargs)
        else:
            from ..processors import TextProcessor
            processor = TextProcessor(**processing_kwargs)

        # Process the file
        result = processor.process_file(file_path, **processing_kwargs)

        if not result.success:
            raise MediaProcessingError(f"Failed to process {file_path}: {result.error_message}")

        return result.media_content

    def format_for_provider(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format media content for local provider.

        Args:
            media_content: MediaContent to format

        Returns:
            Dictionary formatted for local provider
        """
        if self.provider_name == "ollama":
            return self._format_for_ollama(media_content)
        elif self.provider_name == "mlx":
            return self._format_for_mlx(media_content)
        elif self.provider_name == "lmstudio":
            return self._format_for_lmstudio(media_content)
        else:
            # Generic local provider format
            return self._format_generic_local(media_content)

    def _format_for_ollama(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format content for Ollama.

        Ollama supports vision models and can handle base64 images.
        """
        if media_content.media_type == MediaType.IMAGE:
            if not self.capabilities.vision_support:
                raise MediaProcessingError("Ollama model does not support vision")

            if media_content.content_format != ContentFormat.BASE64:
                raise MediaProcessingError("Ollama image formatting requires base64 content")

            # Ollama uses a simple base64 format
            return {
                "type": "image",
                "data": media_content.content,
                "mime_type": media_content.mime_type
            }

        else:
            # Text content for Ollama
            content = str(media_content.content)
            return {
                "type": "text",
                "content": content
            }

    def _format_for_mlx(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format content for MLX (Apple Silicon).

        MLX may have specific optimizations for Apple hardware.
        """
        if media_content.media_type == MediaType.IMAGE:
            if not self.capabilities.vision_support:
                raise MediaProcessingError("MLX model does not support vision")

            # MLX might prefer direct tensor conversion, but fall back to base64
            return {
                "type": "image_base64",
                "content": media_content.content,
                "mime_type": media_content.mime_type,
                "metadata": media_content.metadata
            }

        else:
            # Text content for MLX
            content = str(media_content.content)
            return {
                "type": "text",
                "content": content,
                "format": media_content.content_format.value
            }

    def _format_for_lmstudio(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format content for LMStudio.

        LMStudio typically follows OpenAI-compatible formats but may have limitations.
        """
        if media_content.media_type == MediaType.IMAGE:
            if not self.capabilities.vision_support:
                raise MediaProcessingError("LMStudio model does not support vision")

            # LMStudio may use OpenAI-compatible format
            data_url = f"data:{media_content.mime_type};base64,{media_content.content}"
            return {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }

        else:
            # Text content for LMStudio
            content = str(media_content.content)
            return {
                "type": "text",
                "text": content
            }

    def _format_generic_local(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Generic format for unknown local providers.
        """
        if media_content.media_type == MediaType.IMAGE:
            return {
                "type": "image",
                "content": media_content.content,
                "content_format": media_content.content_format.value,
                "mime_type": media_content.mime_type
            }
        else:
            return {
                "type": "text",
                "content": str(media_content.content)
            }

    def create_multimodal_message(self, text: str, media_contents: List[MediaContent]) -> Union[Dict[str, Any], str]:
        """
        Create a multimodal message for local provider with intelligent vision routing.

        Args:
            text: Text content
            media_contents: List of media contents

        Returns:
            Formatted message (structured dict for vision models, string for text-only)
        """
        # Check if we have images in the media contents
        has_images = any(mc.media_type == MediaType.IMAGE for mc in media_contents)

        if not has_images:
            # No images - use text embedding for efficiency
            self.logger.debug("No images detected, using text-embedded format")
            return self._create_text_embedded_message(text, media_contents)

        # We have images - check vision capabilities
        provider_vision_support = self.capabilities.vision_support if self.capabilities else False

        # Check model-level vision support using existing detection system
        model_vision_support = False
        if VISION_DETECTION_AVAILABLE and self.model_name and supports_vision:
            try:
                model_vision_support = supports_vision(self.model_name)
                self.logger.debug(f"Model '{self.model_name}' vision support: {model_vision_support}")
            except Exception as e:
                self.logger.warning(f"Failed to detect vision support for model '{self.model_name}': {e}")
                model_vision_support = False
        elif not VISION_DETECTION_AVAILABLE:
            self.logger.warning("Vision detection system not available - falling back to provider capabilities only")
        elif not self.model_name:
            self.logger.warning("No model name provided - cannot check model-specific vision capabilities")

        # Decision logic: require BOTH provider AND model support for structured format
        if provider_vision_support and model_vision_support:
            self.logger.debug(f"Using structured format for vision model '{self.model_name}' on provider '{self.provider_name}'")
            try:
                return self._create_structured_message(text, media_contents)
            except Exception as e:
                self.logger.error(f"Failed to create structured message for vision model: {e}")
                self.logger.warning("Falling back to text-embedded format")
                return self._create_text_embedded_message(text, media_contents)

        # Handle capability mismatches with detailed warnings
        if has_images and not model_vision_support and self.model_name:
            self.logger.info(
                f"Model '{self.model_name}' does not support vision. "
                f"Using vision fallback system for image analysis."
            )
        elif has_images and not provider_vision_support and not self.model_name:
            self.logger.info(
                f"No model-specific vision capabilities detected for provider '{self.provider_name}'. "
                f"Using vision fallback system for image analysis."
            )
        elif has_images and not self.model_name:
            self.logger.info(
                f"No model name available for vision detection. "
                f"Using vision fallback system for image analysis."
            )

        # Fallback to text-embedded format
        self.logger.debug("Using text-embedded format due to insufficient vision capabilities")
        return self._create_text_embedded_message(text, media_contents)

    def _create_text_embedded_message(self, text: str, media_contents: List[MediaContent]) -> str:
        """
        Create a message with media content embedded as text.

        This is often more reliable for local providers that don't have
        robust multimodal support. For images on text-only models, uses vision fallback.
        """
        message_parts = []

        # Add main text
        if text.strip():
            message_parts.append(text)

        # Add processed content from media
        for i, media_content in enumerate(media_contents):
            if media_content.media_type == MediaType.IMAGE:
                if self.capabilities.vision_support:
                    # For vision models, we'll still need to handle images specially
                    # This will be handled by the provider's generate method
                    message_parts.append(f"[Image {i+1}: {media_content.metadata.get('file_name', 'image')}]")
                else:
                    # Use vision fallback for text-only models
                    try:
                        from ..vision_fallback import VisionFallbackHandler, VisionNotConfiguredError
                        fallback_handler = VisionFallbackHandler()

                        # Get the actual file path from media_content object
                        file_path = media_content.file_path or media_content.metadata.get('file_path') or media_content.metadata.get('file_name', 'image')

                        # Generate description using vision fallback
                        description = fallback_handler.create_description(str(file_path), text)
                        # Remove the original question from message_parts if it exists
                        if message_parts and text.strip() in message_parts[0]:
                            message_parts.clear()
                        # Completely different approach: make model think it's continuing its own observation
                        # No questions, no external framing - just natural continuation
                        simple_prompt = f"{description}"
                        message_parts.append(simple_prompt)

                    except VisionNotConfiguredError as e:
                        # Vision not configured - show warning to USER, not model
                        self.logger.warning("Vision capability not configured for text-only models")
                        self.logger.warning("To enable image analysis with text-only models:")
                        self.logger.warning("🔸 EASIEST: Download BLIP vision model (990MB): abstractcore --download-vision-model")
                        self.logger.warning("🔸 Use existing Ollama model: abstractcore --set-vision-caption qwen2.5vl:7b")
                        self.logger.warning("🔸 Use cloud API: abstractcore --set-vision-provider openai --model gpt-4o")
                        self.logger.warning("🔸 Interactive setup: abstractcore --configure")
                        self.logger.warning("Current status: abstractcore --status")

                        # Provide minimal placeholder to model (not configuration instructions!)
                        file_name = media_content.metadata.get('file_name', 'image')
                        message_parts.append(f"[Image {i+1}: {file_name}]")

                    except Exception as e:
                        self.logger.warning(f"Vision fallback failed: {e}")
                        # Fallback to basic placeholder
                        file_name = media_content.metadata.get('file_name', 'image')
                        message_parts.append(f"[Image {i+1}: {file_name} - vision processing unavailable]")
            else:
                # Embed text/document content directly
                content = str(media_content.content)
                file_name = media_content.metadata.get('file_name', f'document_{i+1}')
                message_parts.append(f"\n\n--- Content from {file_name} ---\n{content}\n--- End of {file_name} ---")

        return "\n\n".join(message_parts)

    def _create_structured_message(self, text: str, media_contents: List[MediaContent]) -> Dict[str, Any]:
        """
        Create a structured message for local providers using provider-specific format.
        """
        if self.provider_name == "ollama":
            return self._create_ollama_message(text, media_contents)
        elif self.provider_name == "lmstudio":
            return self._create_lmstudio_message(text, media_contents)
        else:
            # Generic structured format for other providers
            return self._create_generic_structured_message(text, media_contents)

    def _create_ollama_message(self, text: str, media_contents: List[MediaContent]) -> Dict[str, Any]:
        """
        Create Ollama-specific multimodal message format.

        Ollama expects: {"role": "user", "content": "text", "images": ["base64..."]}
        """
        message = {
            "role": "user",
            "content": text.strip() if text.strip() else "What's in this image?"
        }

        # Extract base64 images for Ollama's images array
        images = []
        for media_content in media_contents:
            if media_content.media_type == MediaType.IMAGE and self.can_handle_media(media_content):
                if media_content.content_format == ContentFormat.BASE64:
                    # Ollama expects raw base64 without data URL prefix
                    images.append(media_content.content)
                else:
                    self.logger.warning(f"Ollama requires base64 image format, got {media_content.content_format}")

        if images:
            message["images"] = images

        return message

    def _create_lmstudio_message(self, text: str, media_contents: List[MediaContent]) -> Dict[str, Any]:
        """
        Create LMStudio-specific multimodal message format.

        LMStudio follows OpenAI-compatible format with structured content array.
        """
        content = []

        # Add text content
        if text.strip():
            content.append({
                "type": "text",
                "text": text
            })

        # Add images in OpenAI format
        for media_content in media_contents:
            if media_content.media_type == MediaType.IMAGE and self.can_handle_media(media_content):
                if media_content.content_format == ContentFormat.BASE64:
                    data_url = f"data:{media_content.mime_type};base64,{media_content.content}"
                    image_obj = {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                    
                    # Add detail level if specified in metadata (for Qwen models)
                    detail_level = media_content.metadata.get('detail_level', 'auto')
                    self.logger.debug(f"MediaContent metadata: {media_content.metadata}")
                    self.logger.debug(f"Found detail_level: {detail_level}")
                    if detail_level in ['low', 'high', 'auto']:
                        image_obj["image_url"]["detail"] = detail_level
                        self.logger.info(f"Setting detail level to '{detail_level}' for LMStudio image")
                    else:
                        self.logger.warning(f"Invalid detail level '{detail_level}', skipping")
                    
                    content.append(image_obj)
                else:
                    self.logger.warning(f"LMStudio requires base64 image format, got {media_content.content_format}")

        return {
            "role": "user",
            "content": content
        }

    def _create_generic_structured_message(self, text: str, media_contents: List[MediaContent]) -> Dict[str, Any]:
        """
        Create generic structured message for unknown local providers.
        """
        content = []

        # Add text content
        if text.strip():
            content.append({
                "type": "text",
                "content": text
            })

        # Add media contents using provider-specific formatting
        for media_content in media_contents:
            if self.can_handle_media(media_content):
                formatted_content = self.format_for_provider(media_content)
                content.append(formatted_content)

        return {
            "role": "user",
            "content": content
        }

    def validate_media_for_model(self, media_content: MediaContent, model: str) -> bool:
        """
        Validate if media content is compatible with local model.

        Args:
            media_content: MediaContent to validate
            model: Local model name

        Returns:
            True if compatible, False otherwise
        """
        model_lower = model.lower()

        # Image validation
        if media_content.media_type == MediaType.IMAGE:
            # Check if model supports vision
            vision_support = self.model_capabilities.get('vision_support', False)
            if not vision_support:
                return False

            # Local models are generally more permissive with image sizes
            # but still check reasonable limits
            file_size = media_content.metadata.get('file_size', 0)
            if file_size > self.max_image_size:
                return False

            # Model-specific checks for known vision models
            vision_keywords = ['vision', 'vl', 'multimodal', 'llava', 'qwen2-vl', 'qwen3-vl']
            if any(keyword in model_lower for keyword in vision_keywords):
                return True

            return vision_support

        # Text/document validation
        elif media_content.media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # All local models support text
            return True

        return False

    def estimate_tokens_for_media(self, media_content: MediaContent) -> int:
        """
        Estimate token usage for media content with local models.

        Args:
            media_content: MediaContent to estimate

        Returns:
            Estimated token count
        """
        if media_content.media_type == MediaType.IMAGE:
            # Local vision models typically use fewer tokens than cloud models
            # but this varies significantly by model architecture
            return 512  # Conservative estimate

        elif media_content.media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # Local models typically use similar tokenization to their base models
            content_length = len(str(media_content.content))
            return content_length // 4  # Rough estimate

        return 0

    def get_model_media_limits(self, model: str) -> Dict[str, Any]:
        """
        Get media-specific limits for local model.

        Args:
            model: Local model name

        Returns:
            Dictionary of limits
        """
        limits = {
            'max_images_per_message': 1,  # Most local models support only 1 image
            'max_image_size_bytes': self.max_image_size,
            'supported_image_formats': ['png', 'jpeg', 'jpg'],
            'prefers_text_extraction': self.prefer_text_extraction
        }

        model_lower = model.lower()

        # Adjust limits based on known model capabilities
        if 'qwen' in model_lower and ('vl' in model_lower or 'vision' in model_lower):
            limits.update({
                'max_images_per_message': 5,  # Qwen-VL models can handle multiple images
                'supported_image_formats': ['png', 'jpeg', 'jpg', 'gif', 'bmp']
            })

        return limits