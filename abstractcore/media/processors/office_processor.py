"""
Office document processor using unstructured library for SOTA document processing.

This module provides comprehensive processing capabilities for Microsoft Office documents
(DOCX, XLSX, PPT) using the unstructured library, which is the SOTA solution for
document processing in 2025.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import json

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat, MediaProcessingResult
from ...utils.structured_logging import get_logger


class OfficeProcessor(BaseMediaHandler):
    """
    Office document processor using unstructured library.

    Supports processing of:
    - DOCX (Word documents)
    - XLSX (Excel spreadsheets)
    - PPTX (PowerPoint presentations)

    Uses the unstructured library for SOTA document processing with intelligent
    element detection, table extraction, and structure preservation.
    """

    def __init__(self, **kwargs):
        """
        Initialize Office processor.

        Args:
            **kwargs: Additional configuration options
        """
        super().__init__(**kwargs)
        self.logger = get_logger(__name__)

        # Configuration options
        self.extract_tables = kwargs.get('extract_tables', True)
        self.preserve_structure = kwargs.get('preserve_structure', True)
        self.extract_images = kwargs.get('extract_images', False)  # Images in Office docs
        self.markdown_output = kwargs.get('markdown_output', True)
        self.include_metadata = kwargs.get('include_metadata', True)

        # Chunking options for large documents
        self.chunk_size = kwargs.get('chunk_size', None)  # No chunking by default
        self.chunk_overlap = kwargs.get('chunk_overlap', 0)

        # Check if unstructured library is available
        self._check_dependencies()

        # Set capabilities for office processing
        from ..types import MediaCapabilities
        self.capabilities = MediaCapabilities(
            vision_support=False,
            audio_support=False,
            video_support=False,
            document_support=True,
            supported_document_formats=['docx', 'xlsx', 'pptx'],
            max_file_size=self.max_file_size
        )

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import unstructured
            from unstructured.partition.auto import partition
            from unstructured.partition.docx import partition_docx
            from unstructured.partition.xlsx import partition_xlsx
            from unstructured.partition.pptx import partition_pptx
            self._unstructured_available = True
            self.logger.debug("Unstructured library available for Office document processing")
        except ImportError as e:
            self._unstructured_available = False
            self.logger.warning(f"Unstructured library not available: {e}")

    def can_process(self, file_path: Path) -> bool:
        """
        Check if this processor can handle the file.

        Args:
            file_path: Path to the file

        Returns:
            True if file can be processed
        """
        if not self._unstructured_available:
            return False

        supported_extensions = {'.docx', '.xlsx', '.pptx'}
        return file_path.suffix.lower() in supported_extensions

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Internal processing method for Office documents.

        Args:
            file_path: Path to the Office document
            media_type: Detected media type (should be DOCUMENT)
            **kwargs: Additional processing options

        Returns:
            MediaContent with processed Office document content

        Raises:
            MediaProcessingError: If processing fails
        """
        if media_type != MediaType.DOCUMENT:
            raise MediaProcessingError(f"OfficeProcessor only handles document types, got {media_type}")

        if not self._unstructured_available:
            raise MediaProcessingError(
                "Unstructured library not available. Install with: pip install \"abstractcore[media]\""
            )

        try:
            # Extract content based on file type
            file_extension = file_path.suffix.lower()

            if file_extension == '.docx':
                content, metadata = self._process_docx(file_path, **kwargs)
            elif file_extension == '.xlsx':
                content, metadata = self._process_xlsx(file_path, **kwargs)
            elif file_extension == '.pptx':
                content, metadata = self._process_pptx(file_path, **kwargs)
            else:
                raise MediaProcessingError(f"Unsupported Office file type: {file_extension}")

            # Create MediaContent object
            return self._create_media_content(
                content=content,
                media_type=MediaType.DOCUMENT,
                content_format=ContentFormat.TEXT,
                mime_type=self._get_mime_type(file_extension),
                file_path=file_path,
                metadata=metadata
            )

        except Exception as e:
            raise MediaProcessingError(f"Office document processing failed: {str(e)}")

    def process_file(self, file_path: Path, **kwargs) -> MediaProcessingResult:
        """
        Process an Office document file.

        Args:
            file_path: Path to the Office document
            **kwargs: Additional processing options

        Returns:
            MediaProcessingResult with extracted content
        """
        if not self._unstructured_available:
            return MediaProcessingResult(
                success=False,
                error_message="Unstructured library not available. Install with: pip install \"abstractcore[media]\""
            )

        if not self.can_process(file_path):
            return MediaProcessingResult(
                success=False,
                error_message=f"Unsupported Office file type: {file_path.suffix}"
            )

        try:
            self.logger.info(f"Processing Office document: {file_path}")

            # Extract content based on file type
            file_extension = file_path.suffix.lower()

            if file_extension == '.docx':
                content, metadata = self._process_docx(file_path, **kwargs)
            elif file_extension == '.xlsx':
                content, metadata = self._process_xlsx(file_path, **kwargs)
            elif file_extension == '.pptx':
                content, metadata = self._process_pptx(file_path, **kwargs)
            else:
                return MediaProcessingResult(
                    success=False,
                    error_message=f"Unsupported file extension: {file_extension}"
                )

            # Create MediaContent
            media_content = MediaContent(
                media_type=MediaType.DOCUMENT,
                content=content,
                content_format=ContentFormat.TEXT,
                mime_type=self._get_mime_type(file_extension),
                file_path=str(file_path),
                metadata=metadata
            )

            return MediaProcessingResult(
                success=True,
                media_content=media_content,
                processing_time=0  # Would be calculated in real implementation
            )

        except Exception as e:
            self.logger.error(f"Error processing Office document {file_path}: {e}")
            return MediaProcessingResult(
                success=False,
                error_message=f"Office document processing failed: {str(e)}"
            )

    def _process_docx(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process a DOCX document using unstructured.

        Args:
            file_path: Path to DOCX file
            **kwargs: Processing options

        Returns:
            Tuple of (content, metadata)
        """
        from unstructured.partition.docx import partition_docx

        # Partition the document
        elements = partition_docx(
            filename=str(file_path),
            include_metadata=self.include_metadata,
            extract_image_block_types=["Image"] if self.extract_images else []
        )

        # Convert to structured format
        content_parts = []
        tables = []
        images = []

        for element in elements:
            # Get element type and text content directly from the element
            element_type = type(element).__name__
            text_content = str(element)

            if element_type == 'Table' and self.extract_tables:
                # Extract table content
                tables.append(text_content)
                if self.markdown_output:
                    content_parts.append(f"\n**Table:**\n{text_content}\n")
                else:
                    content_parts.append(f"\nTable: {text_content}\n")

            elif element_type == 'Image' and self.extract_images:
                images.append(text_content)
                content_parts.append(f"\n[Image: {text_content}]\n")

            elif text_content.strip():
                if self.markdown_output and element_type in ['Title', 'Header']:
                    # Format headers in markdown
                    content_parts.append(f"\n## {text_content}\n")
                else:
                    content_parts.append(text_content)

        # Combine content
        content = '\n'.join(content_parts) if content_parts else "No text content found"

        # Build metadata
        metadata = {
            'file_name': file_path.name,
            'file_type': 'docx',
            'file_size': file_path.stat().st_size,
            'element_count': len(elements),
            'table_count': len(tables),
            'image_count': len(images),
            'processing_method': 'unstructured-docx'
        }

        if self.include_metadata and elements:
            # Add document-level metadata from first element
            first_element = elements[0]
            if hasattr(first_element, 'metadata') and first_element.metadata:
                metadata.update({
                    'author': getattr(first_element.metadata, 'author', None),
                    'creation_date': getattr(first_element.metadata, 'creation_date', None),
                    'last_modified': getattr(first_element.metadata, 'last_modified', None)
                })

        return content, metadata

    def _process_xlsx(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process an XLSX spreadsheet using unstructured.

        Args:
            file_path: Path to XLSX file
            **kwargs: Processing options

        Returns:
            Tuple of (content, metadata)
        """
        from unstructured.partition.xlsx import partition_xlsx

        # Partition the spreadsheet
        elements = partition_xlsx(
            filename=str(file_path),
            include_metadata=self.include_metadata
        )

        content_parts = []
        sheet_data = {}

        current_sheet = None
        for element in elements:
            # Get element content directly
            text_content = str(element)

            # For XLSX, try to get sheet information from element if available
            sheet_name = 'Sheet1'  # Default sheet name
            if hasattr(element, 'metadata') and element.metadata:
                sheet_name = getattr(element.metadata, 'sheet_name', 'Sheet1')

            if sheet_name != current_sheet:
                if self.markdown_output:
                    content_parts.append(f"\n## Sheet: {sheet_name}\n")
                else:
                    content_parts.append(f"\nSheet: {sheet_name}\n")
                current_sheet = sheet_name
                sheet_data[sheet_name] = []

            if text_content.strip():
                sheet_data[sheet_name].append(text_content)
                content_parts.append(text_content)

        # Format as tables if structured output is requested
        if self.markdown_output and sheet_data:
            formatted_content = []
            for sheet_name, data in sheet_data.items():
                formatted_content.append(f"\n## {sheet_name}\n")

                # Try to format as table if data looks tabular
                if len(data) > 1:
                    # Simple table formatting - could be enhanced
                    formatted_content.append("| " + " | ".join(str(item) for item in data[:5]) + " |")
                    if len(data) > 1:
                        formatted_content.append("|" + "---|" * min(5, len(data)) + "|")
                        for row in data[1:6]:  # Limit to first few rows
                            formatted_content.append("| " + str(row) + " |")
                    if len(data) > 6:
                        formatted_content.append("... (additional rows truncated)")
                else:
                    formatted_content.extend(data)

            content = '\n'.join(formatted_content)
        else:
            content = '\n'.join(content_parts) if content_parts else "No data found"

        # Build metadata
        metadata = {
            'file_name': file_path.name,
            'file_type': 'xlsx',
            'file_size': file_path.stat().st_size,
            'sheet_count': len(sheet_data),
            'sheet_names': list(sheet_data.keys()),
            'total_cells': sum(len(data) for data in sheet_data.values()),
            'processing_method': 'unstructured-xlsx'
        }

        return content, metadata

    def _process_pptx(self, file_path: Path, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Process a PPTX presentation using unstructured.

        Args:
            file_path: Path to PPTX file
            **kwargs: Processing options

        Returns:
            Tuple of (content, metadata)
        """
        from unstructured.partition.pptx import partition_pptx

        # Partition the presentation
        elements = partition_pptx(
            filename=str(file_path),
            include_metadata=self.include_metadata
        )

        content_parts = []
        slide_count = 0
        current_slide = None

        for element in elements:
            # Get element content directly
            text_content = str(element)
            element_type = type(element).__name__

            # Track slide information - try to get from element metadata if available
            slide_number = None
            if hasattr(element, 'metadata') and element.metadata:
                slide_number = getattr(element.metadata, 'slide_number', None)

            if slide_number != current_slide:
                slide_count += 1
                if self.markdown_output:
                    content_parts.append(f"\n## Slide {slide_count}\n")
                else:
                    content_parts.append(f"\nSlide {slide_count}:\n")
                current_slide = slide_number

            if text_content.strip():
                if self.markdown_output and element_type == 'Title':
                    content_parts.append(f"### {text_content}\n")
                elif element_type == 'ListItem':
                    content_parts.append(f"- {text_content}")
                else:
                    content_parts.append(text_content)

        content = '\n'.join(content_parts) if content_parts else "No text content found"

        # Build metadata
        metadata = {
            'file_name': file_path.name,
            'file_type': 'pptx',
            'file_size': file_path.stat().st_size,
            'slide_count': slide_count,
            'element_count': len(elements),
            'processing_method': 'unstructured-pptx'
        }

        if self.include_metadata and elements:
            # Add presentation-level metadata
            first_element = elements[0]
            if hasattr(first_element, 'metadata') and first_element.metadata:
                metadata.update({
                    'author': getattr(first_element.metadata, 'author', None),
                    'creation_date': getattr(first_element.metadata, 'creation_date', None),
                    'last_modified': getattr(first_element.metadata, 'last_modified', None)
                })

        return content, metadata

    def _get_mime_type(self, file_extension: str) -> str:
        """Get MIME type for Office file extension."""
        mime_types = {
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        return mime_types.get(file_extension.lower(), 'application/octet-stream')

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        if self._unstructured_available:
            return ['docx', 'xlsx', 'pptx']
        return []

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

        # Rough estimation based on file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Office documents generally process at ~2MB/second with unstructured
        return max(1.0, file_size_mb / 2.0)

    def supports_chunking(self) -> bool:
        """Check if this processor supports document chunking."""
        return True

    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about this processor."""
        return {
            'name': 'OfficeProcessor',
            'supported_formats': self.get_supported_formats(),
            'library': 'unstructured',
            'library_available': self._unstructured_available,
            'features': {
                'table_extraction': self.extract_tables,
                'structure_preservation': self.preserve_structure,
                'image_extraction': self.extract_images,
                'markdown_output': self.markdown_output,
                'metadata_extraction': self.include_metadata,
                'chunking_support': self.supports_chunking()
            }
        }