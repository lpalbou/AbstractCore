"""
Base media handler following AbstractCore patterns.

This module defines the base class for all media processing operations,
providing a unified interface for handling different file types across providers.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Type

from .types import (
    MediaContent, MediaType, ContentFormat, MediaCapabilities,
    MediaProcessingResult, detect_media_type, FILE_TYPE_MAPPINGS
)
from ..utils.structured_logging import get_logger
from ..events import EventType, emit_global
from ..exceptions import AbstractCoreError


class MediaProcessingError(AbstractCoreError):
    """Exception raised when media processing fails."""
    pass


class UnsupportedMediaTypeError(MediaProcessingError):
    """Exception raised when a media type is not supported."""
    pass


class FileSizeExceededError(MediaProcessingError):
    """Exception raised when file size exceeds limits."""
    pass


class BaseMediaHandler(ABC):
    """
    Base class for media handling operations.

    This class provides the fundamental interface and shared functionality
    for all media processors, following AbstractCore's architecture patterns.
    """

    def __init__(self, **kwargs):
        """
        Initialize the base media handler.

        Args:
            **kwargs: Configuration parameters
        """
        # Setup structured logging
        self.logger = get_logger(self.__class__.__name__)

        # Configuration
        self.max_file_size = kwargs.get('max_file_size', 50 * 1024 * 1024)  # 50MB default
        self.supported_formats = kwargs.get('supported_formats', [])
        self.temp_dir = kwargs.get('temp_dir', None)
        self.enable_events = kwargs.get('enable_events', True)

        # Capabilities (to be set by subclasses)
        self.capabilities = MediaCapabilities()

        self.logger.debug(f"Initialized {self.__class__.__name__} with max_file_size={self.max_file_size}")

    def process_file(self, file_path: Union[str, Path], **kwargs) -> MediaProcessingResult:
        """
        Process a file and return media content.

        This is the main entry point for media processing, providing telemetry
        and error handling around the actual processing implementation.

        Args:
            file_path: Path to the file to process
            **kwargs: Additional processing parameters

        Returns:
            MediaProcessingResult with success/failure information
        """
        start_time = time.time()
        file_path = Path(file_path)

        # Emit processing started event
        if self.enable_events:
            self._emit_processing_event(
                EventType.GENERATION_STARTED,  # Reuse generation events for media
                file_path=str(file_path),
                media_type=detect_media_type(file_path).value,
                processor=self.__class__.__name__
            )

        try:
            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Validate file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise FileSizeExceededError(
                    f"File size {file_size} exceeds maximum {self.max_file_size} bytes"
                )

            # Detect media type
            media_type = detect_media_type(file_path)

            # Check if media type is supported
            if not self.supports_media_type(media_type):
                raise UnsupportedMediaTypeError(
                    f"Media type {media_type.value} not supported by {self.__class__.__name__}"
                )

            # Check format support
            format_ext = file_path.suffix.lower().lstrip('.')
            if not self.supports_format(media_type, format_ext):
                raise UnsupportedMediaTypeError(
                    f"Format .{format_ext} not supported for {media_type.value}"
                )

            # Call the actual processing implementation
            media_content = self._process_internal(file_path, media_type, **kwargs)

            # Create successful result
            processing_time = time.time() - start_time
            result = MediaProcessingResult(
                success=True,
                media_content=media_content,
                processing_time=processing_time,
                metadata={
                    'file_size': file_size,
                    'file_name': file_path.name,
                    'processor': self.__class__.__name__
                }
            )

            # Track successful processing
            self._track_processing(file_path, result, start_time, success=True)

            return result

        except Exception as e:
            # Create error result
            processing_time = time.time() - start_time
            result = MediaProcessingResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                metadata={
                    'processor': self.__class__.__name__,
                    'error_type': type(e).__name__
                }
            )

            # Track failed processing
            self._track_processing(file_path, result, start_time, success=False, error=e)

            return result

    def process_multiple_files(self, file_paths: List[Union[str, Path]], **kwargs) -> List[MediaProcessingResult]:
        """
        Process multiple files efficiently.

        Args:
            file_paths: List of file paths to process
            **kwargs: Additional processing parameters

        Returns:
            List of MediaProcessingResult objects
        """
        results = []

        for file_path in file_paths:
            try:
                result = self.process_file(file_path, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result for failed file
                error_result = MediaProcessingResult(
                    success=False,
                    error_message=str(e),
                    metadata={
                        'file_path': str(file_path),
                        'processor': self.__class__.__name__,
                        'error_type': type(e).__name__
                    }
                )
                results.append(error_result)

        return results

    def supports_media_type(self, media_type: MediaType) -> bool:
        """
        Check if this handler supports the given media type.

        Args:
            media_type: MediaType to check

        Returns:
            True if supported, False otherwise
        """
        return self.capabilities.supports_media_type(media_type)

    def supports_format(self, media_type: MediaType, format_ext: str) -> bool:
        """
        Check if this handler supports the specific format.

        Args:
            media_type: MediaType of the content
            format_ext: File extension (without dot)

        Returns:
            True if supported, False otherwise
        """
        return self.capabilities.supports_format(media_type, format_ext)

    def get_capabilities(self) -> MediaCapabilities:
        """
        Get the capabilities of this media handler.

        Returns:
            MediaCapabilities object
        """
        return self.capabilities

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get supported formats organized by media type.

        Returns:
            Dictionary mapping media type to list of supported extensions
        """
        result = {}
        for media_type in MediaType:
            if self.supports_media_type(media_type):
                formats = []
                for ext, mt in FILE_TYPE_MAPPINGS.items():
                    if mt == media_type and self.supports_format(media_type, ext):
                        formats.append(ext)
                if formats:
                    result[media_type.value] = formats
        return result

    @abstractmethod
    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Internal processing method to be implemented by subclasses.

        This method contains the actual processing logic for the specific
        media type or file format.

        Args:
            file_path: Path to the file to process
            media_type: Detected media type
            **kwargs: Additional processing parameters

        Returns:
            MediaContent object with processed content

        Raises:
            MediaProcessingError: If processing fails
        """
        pass

    def _track_processing(self, file_path: Path, result: MediaProcessingResult,
                         start_time: float, success: bool = True,
                         error: Optional[Exception] = None):
        """
        Track media processing with telemetry and events.

        Args:
            file_path: Path to the processed file
            result: Processing result
            start_time: Processing start time
            success: Whether processing succeeded
            error: Error if failed
        """
        duration_ms = (time.time() - start_time) * 1000

        # Emit processing completed event
        if self.enable_events:
            event_data = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "media_type": detect_media_type(file_path).value,
                "processor": self.__class__.__name__,
                "success": success,
                "duration_ms": duration_ms,
                "error": str(error) if error else None
            }

            if result.media_content:
                event_data.update({
                    "content_format": result.media_content.content_format.value,
                    "mime_type": result.media_content.mime_type,
                    "content_size": len(str(result.media_content.content))
                })

            self._emit_processing_event(
                EventType.GENERATION_COMPLETED,
                **event_data
            )

        # Log processing result
        if error:
            self.logger.error(
                f"Media processing failed for {file_path.name}: {error} "
                f"(duration: {duration_ms:.2f}ms)"
            )
        else:
            content_info = ""
            if result.media_content:
                content_size = len(str(result.media_content.content))
                content_info = f" (content size: {content_size} chars)"

            self.logger.info(
                f"Media processing completed for {file_path.name}: "
                f"{duration_ms:.2f}ms{content_info}"
            )

    def _emit_processing_event(self, event_type: EventType, **event_data):
        """
        Emit a media processing event.

        Args:
            event_type: Type of event to emit
            **event_data: Event data
        """
        if self.enable_events:
            emit_global(event_type, event_data, source=self.__class__.__name__)

    def _validate_content_size(self, content: Union[str, bytes], max_size: Optional[int] = None) -> None:
        """
        Validate that content size doesn't exceed limits.

        Args:
            content: Content to validate
            max_size: Maximum allowed size in bytes

        Raises:
            FileSizeExceededError: If content exceeds size limit
        """
        if max_size is None:
            max_size = self.max_file_size

        content_size = len(content.encode('utf-8') if isinstance(content, str) else content)
        if content_size > max_size:
            raise FileSizeExceededError(
                f"Processed content size {content_size} exceeds maximum {max_size} bytes"
            )

    def _create_media_content(self, content: Union[str, bytes], file_path: Path,
                            media_type: MediaType, content_format: ContentFormat,
                            mime_type: str = "auto", **metadata) -> MediaContent:
        """
        Create a MediaContent object with consistent metadata.

        Args:
            content: Processed content
            file_path: Original file path
            media_type: Type of media content
            content_format: Format of the content
            mime_type: MIME type of the content
            **metadata: Additional metadata

        Returns:
            MediaContent object
        """
        # Validate content size
        self._validate_content_size(content)

        # Create base metadata
        base_metadata = {
            'file_size': file_path.stat().st_size,
            'file_name': file_path.name,
            'file_extension': file_path.suffix,
            'processor': self.__class__.__name__,
            'processing_timestamp': time.time()
        }
        base_metadata.update(metadata)

        return MediaContent(
            media_type=media_type,
            content=content,
            content_format=content_format,
            mime_type=mime_type,
            file_path=str(file_path),
            metadata=base_metadata
        )


class BaseProviderMediaHandler(BaseMediaHandler):
    """
    Base class for provider-specific media handlers.

    This class extends BaseMediaHandler to provide provider-specific
    media formatting capabilities.
    """

    def __init__(self, provider_name: str, model_capabilities: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize provider media handler.

        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            model_capabilities: Model capabilities from model_capabilities.json
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        self.provider_name = provider_name
        self.model_capabilities = model_capabilities or {}

        # Set capabilities based on model capabilities
        self._initialize_capabilities_from_model()

        self.logger.debug(f"Initialized provider media handler for {provider_name}")

    def _initialize_capabilities_from_model(self):
        """Initialize capabilities based on model capabilities."""
        if self.model_capabilities:
            self.capabilities = MediaCapabilities(
                vision_support=self.model_capabilities.get('vision_support', False),
                audio_support=self.model_capabilities.get('audio_support', False),
                video_support=self.model_capabilities.get('video_support', False),
                document_support=True,  # Assume document support for all providers
                max_image_resolution=self.model_capabilities.get('image_resolutions', [None])[0],
                supported_image_formats=['jpg', 'png', 'gif', 'webp'] if self.model_capabilities.get('vision_support') else [],
                supported_document_formats=['pdf', 'txt', 'md', 'csv', 'tsv'],
                max_file_size=self.max_file_size
            )

    @abstractmethod
    def format_for_provider(self, media_content: MediaContent) -> Dict[str, Any]:
        """
        Format media content for the specific provider's API.

        Args:
            media_content: MediaContent to format

        Returns:
            Dictionary formatted for provider's API
        """
        pass

    def can_handle_media(self, media_content: MediaContent) -> bool:
        """
        Check if this provider can handle the given media content.

        Args:
            media_content: MediaContent to check

        Returns:
            True if provider can handle this content
        """
        return self.supports_media_type(media_content.media_type)