"""
Automatic media handler that selects the appropriate processor for each file type.

This module provides a unified interface that automatically chooses the best
processor (ImageProcessor, TextProcessor, PDFProcessor, or OfficeProcessor)
based on the file type and content.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BaseMediaHandler
from .types import MediaContent, MediaType, ContentFormat, detect_media_type
from .processors import ImageProcessor, TextProcessor, PDFProcessor, OfficeProcessor


class AutoMediaHandler(BaseMediaHandler):
    """
    Automatic media handler that delegates to specialized processors.

    This handler analyzes the input file and automatically selects the most
    appropriate processor to handle the content, providing a unified interface
    for all media types.
    """

    def __init__(self, **kwargs):
        """
        Initialize the auto media handler.

        Args:
            **kwargs: Configuration parameters passed to processors
        """
        super().__init__(**kwargs)

        # Configuration for processors
        self.processor_config = kwargs

        # Initialize processors lazily (only when needed)
        self._image_processor = None
        self._text_processor = None
        self._pdf_processor = None
        self._office_processor = None

        # Track which processors are available
        self._available_processors = self._check_processor_availability()

        self.logger.debug(f"AutoMediaHandler initialized with processors: {list(self._available_processors.keys())}")

    def _check_processor_availability(self) -> Dict[str, bool]:
        """Check which processors are available."""
        availability = {}

        # ImageProcessor (requires PIL)
        try:
            from PIL import Image
            availability['image'] = True
        except ImportError:
            availability['image'] = False

        # TextProcessor (always available - uses built-in libraries)
        availability['text'] = True

        # PDFProcessor (requires PyMuPDF4LLM)
        try:
            import pymupdf4llm
            availability['pdf'] = True
        except ImportError:
            availability['pdf'] = False

        # OfficeProcessor (requires unstructured)
        try:
            import unstructured
            availability['office'] = True
        except ImportError:
            availability['office'] = False

        return availability

    def _get_image_processor(self) -> ImageProcessor:
        """Get or create ImageProcessor instance."""
        if self._image_processor is None:
            self._image_processor = ImageProcessor(**self.processor_config)
        return self._image_processor

    def _get_text_processor(self) -> TextProcessor:
        """Get or create TextProcessor instance."""
        if self._text_processor is None:
            self._text_processor = TextProcessor(**self.processor_config)
        return self._text_processor

    def _get_pdf_processor(self) -> PDFProcessor:
        """Get or create PDFProcessor instance."""
        if self._pdf_processor is None:
            self._pdf_processor = PDFProcessor(**self.processor_config)
        return self._pdf_processor

    def _get_office_processor(self) -> OfficeProcessor:
        """Get or create OfficeProcessor instance."""
        if self._office_processor is None:
            self._office_processor = OfficeProcessor(**self.processor_config)
        return self._office_processor

    def _select_processor(self, file_path: Path, media_type: MediaType) -> Optional[BaseMediaHandler]:
        """
        Select the appropriate processor for the file.

        Args:
            file_path: Path to the file
            media_type: Detected media type

        Returns:
            Appropriate processor instance or None if unsupported
        """
        file_extension = file_path.suffix.lower()

        # Handle images
        if media_type == MediaType.IMAGE:
            if self._available_processors.get('image', False):
                return self._get_image_processor()
            else:
                self.logger.warning("Image processing requested but PIL not available")
                return None

        # Handle text files
        elif media_type == MediaType.TEXT:
            return self._get_text_processor()

        # Handle documents
        elif media_type == MediaType.DOCUMENT:
            # PDF files
            if file_extension == '.pdf':
                if self._available_processors.get('pdf', False):
                    return self._get_pdf_processor()
                else:
                    self.logger.warning("PDF processing requested but PyMuPDF4LLM not available")
                    # Fall back to text processor for basic extraction
                    return self._get_text_processor()

            # Office documents
            elif file_extension in {'.docx', '.xlsx', '.pptx'}:
                if self._available_processors.get('office', False):
                    return self._get_office_processor()
                else:
                    self.logger.warning(f"Office document processing requested but unstructured library not available for {file_extension}")
                    # Fall back to text processor (limited functionality)
                    return self._get_text_processor()

            # Text-based documents
            else:
                return self._get_text_processor()

        # Handle other media types (audio, video) - not yet implemented
        else:
            self.logger.warning(f"Media type {media_type.value} not yet supported")
            return None

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Internal processing that delegates to the appropriate processor.

        Args:
            file_path: Path to the file to process
            media_type: Detected media type
            **kwargs: Additional processing parameters

        Returns:
            MediaContent object with processed content
        """
        # Select the appropriate processor
        processor = self._select_processor(file_path, media_type)

        if processor is None:
            # Create a basic text representation as fallback
            return self._create_fallback_content(file_path, media_type)

        # Delegate to the selected processor
        try:
            return processor._process_internal(file_path, media_type, **kwargs)
        except Exception as e:
            self.logger.error(f"Processor {processor.__class__.__name__} failed for {file_path}: {e}")
            # Fall back to basic content creation
            return self._create_fallback_content(file_path, media_type)

    def _create_fallback_content(self, file_path: Path, media_type: MediaType) -> MediaContent:
        """
        Create fallback content when processors are not available.

        Args:
            file_path: Path to the file
            media_type: Media type

        Returns:
            Basic MediaContent object
        """
        file_extension = file_path.suffix.lower()

        # Try to read as text for document types
        if media_type == MediaType.DOCUMENT and file_extension in {'.txt', '.md', '.csv', '.tsv'}:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                content_format = ContentFormat.TEXT
            except Exception:
                content = f"[Unable to read file: {file_path.name}]"
                content_format = ContentFormat.TEXT
        else:
            # For other types, create a placeholder
            content = f"[{media_type.value.title()}: {file_path.name}]"
            content_format = ContentFormat.TEXT

        return self._create_media_content(
            content=content,
            file_path=file_path,
            media_type=media_type,
            content_format=content_format,
            mime_type="auto",
            fallback_processing=True,
            available_processors=list(self._available_processors.keys())
        )

    def supports_media_type(self, media_type: MediaType) -> bool:
        """
        Check if this handler supports the given media type.

        Args:
            media_type: MediaType to check

        Returns:
            True if any processor can handle this type
        """
        if media_type == MediaType.IMAGE:
            return self._available_processors.get('image', False)
        elif media_type == MediaType.TEXT:
            return True  # Always supported via text processor
        elif media_type == MediaType.DOCUMENT:
            return True  # Always supported via text processor at minimum
        elif media_type == MediaType.AUDIO:
            return False  # Not yet implemented
        elif media_type == MediaType.VIDEO:
            return False  # Not yet implemented
        return False

    def supports_format(self, media_type: MediaType, format_ext: str) -> bool:
        """
        Check if this handler supports the specific format.

        Args:
            media_type: MediaType of the content
            format_ext: File extension (without dot)

        Returns:
            True if supported
        """
        if media_type == MediaType.IMAGE:
            if not self._available_processors.get('image', False):
                return False
            image_formats = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
            return format_ext.lower() in image_formats

        elif media_type == MediaType.TEXT:
            # Text formats (always available)
            text_formats = {'txt', 'md', 'csv', 'tsv', 'json', 'yaml', 'yml'}
            return format_ext.lower() in text_formats

        elif media_type == MediaType.DOCUMENT:
            # PDF support
            if format_ext.lower() == 'pdf':
                return self._available_processors.get('pdf', False) or True  # Fallback to text

            # Office document support
            if format_ext.lower() in {'docx', 'xlsx', 'pptx'}:
                return self._available_processors.get('office', False) or True  # Fallback to text

            # Text document support (always available)
            text_formats = {'txt', 'md', 'csv', 'tsv', 'json', 'yaml', 'yml'}
            return format_ext.lower() in text_formats

        return False

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get supported formats organized by media type.

        Returns:
            Dictionary mapping media type to list of supported extensions
        """
        formats = {}

        # Image formats
        if self._available_processors.get('image', False):
            formats['image'] = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']

        # Document formats
        doc_formats = ['txt', 'md', 'csv', 'tsv', 'json', 'yaml', 'yml']

        if self._available_processors.get('pdf', False):
            doc_formats.append('pdf')

        if self._available_processors.get('office', False):
            doc_formats.extend(['docx', 'xlsx', 'pptx'])

        formats['document'] = doc_formats

        return formats

    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about available processors and their capabilities.

        Returns:
            Dictionary with processor information
        """
        info = {
            'handler_type': 'AutoMediaHandler',
            'available_processors': self._available_processors.copy(),
            'supported_formats': self.get_supported_formats(),
            'capabilities': {
                'images': self._available_processors.get('image', False),
                'pdf_documents': self._available_processors.get('pdf', False),
                'office_documents': self._available_processors.get('office', False),
                'text_documents': True,
                'automatic_selection': True,
                'fallback_processing': True
            }
        }

        # Add processor-specific information
        if self._available_processors.get('image', False):
            info['image_processor'] = self._get_image_processor().get_processing_info()

        if self._available_processors.get('pdf', False):
            info['pdf_processor'] = self._get_pdf_processor().get_processing_info()

        if self._available_processors.get('office', False):
            info['office_processor'] = self._get_office_processor().get_processing_info()

        info['text_processor'] = self._get_text_processor().get_processing_info()

        return info

    def estimate_processing_time(self, file_path: Path) -> float:
        """
        Estimate processing time for a file.

        Args:
            file_path: Path to the file

        Returns:
            Estimated processing time in seconds
        """
        if not file_path.exists():
            return 0.0

        media_type = detect_media_type(file_path)
        processor = self._select_processor(file_path, media_type)

        if processor and hasattr(processor, 'estimate_processing_time'):
            return processor.estimate_processing_time(file_path)
        else:
            # Basic estimation based on file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            return max(0.1, file_size_mb / 10.0)  # ~10MB/second processing rate