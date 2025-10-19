"""
PDF processor using PyMuPDF4LLM for optimized LLM processing.

This module provides comprehensive PDF processing capabilities using PyMuPDF4LLM,
optimized for LLM consumption with excellent markdown output and structure preservation.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    pymupdf4llm = None

try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat


class PDFProcessor(BaseMediaHandler):
    """
    PDF processor using PyMuPDF4LLM for LLM-optimized document processing.

    Provides high-quality text extraction, structure preservation, table detection,
    and image extraction from PDF documents.
    """

    def __init__(self, **kwargs):
        """
        Initialize the PDF processor.

        Args:
            **kwargs: Configuration parameters including:
                - extract_images: Whether to extract embedded images
                - preserve_tables: Whether to preserve table formatting
                - markdown_output: Whether to output as markdown
                - page_range: Tuple of (start_page, end_page) or None for all pages
                - extract_metadata: Whether to extract PDF metadata
        """
        if not PYMUPDF4LLM_AVAILABLE:
            raise ImportError(
                "PyMuPDF4LLM is required for PDF processing. "
                "Install with: pip install \"abstractcore[media]\""
            )

        super().__init__(**kwargs)

        # PDF processing configuration
        self.extract_images = kwargs.get('extract_images', False)
        self.preserve_tables = kwargs.get('preserve_tables', True)
        self.markdown_output = kwargs.get('markdown_output', True)
        self.page_range = kwargs.get('page_range', None)
        self.extract_metadata = kwargs.get('extract_metadata', True)

        # Set capabilities for PDF processing
        from ..types import MediaCapabilities
        self.capabilities = MediaCapabilities(
            vision_support=self.extract_images,
            audio_support=False,
            video_support=False,
            document_support=True,
            supported_document_formats=['pdf'],
            max_file_size=self.max_file_size
        )

        self.logger.debug(
            f"Initialized PDFProcessor with extract_images={self.extract_images}, "
            f"preserve_tables={self.preserve_tables}, markdown_output={self.markdown_output}"
        )

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Process a PDF file and return optimized content for LLM consumption.

        Args:
            file_path: Path to the PDF file
            media_type: Detected media type (should be DOCUMENT)
            **kwargs: Additional processing parameters:
                - page_range: Override default page range
                - extract_images: Override default image extraction
                - output_format: 'markdown', 'text', or 'structured'
                - dpi: DPI for image extraction (default: 150)

        Returns:
            MediaContent with processed PDF content

        Raises:
            MediaProcessingError: If PDF processing fails
        """
        if media_type != MediaType.DOCUMENT:
            raise MediaProcessingError(f"PDFProcessor only handles document types, got {media_type}")

        try:
            # Override defaults with kwargs
            page_range = kwargs.get('page_range', self.page_range)
            extract_images = kwargs.get('extract_images', self.extract_images)
            output_format = kwargs.get('output_format', 'markdown' if self.markdown_output else 'text')
            dpi = kwargs.get('dpi', 150)

            # Process PDF with PyMuPDF4LLM
            content, metadata = self._extract_pdf_content(
                file_path, page_range, extract_images, output_format, dpi
            )

            # Determine content format and MIME type based on output format
            if output_format == 'markdown':
                mime_type = 'text/markdown'
            elif output_format == 'structured':
                mime_type = 'application/json'
            else:
                mime_type = 'text/plain'

            return self._create_media_content(
                content=content,
                file_path=file_path,
                media_type=MediaType.DOCUMENT,
                content_format=ContentFormat.TEXT,
                mime_type=mime_type,
                **metadata
            )

        except Exception as e:
            raise MediaProcessingError(f"Failed to process PDF {file_path}: {str(e)}") from e

    def _extract_pdf_content(self, file_path: Path, page_range: Optional[Tuple[int, int]],
                           extract_images: bool, output_format: str, dpi: int) -> Tuple[str, Dict[str, Any]]:
        """
        Extract content from PDF using PyMuPDF4LLM.

        Args:
            file_path: Path to the PDF file
            page_range: Optional page range to process
            extract_images: Whether to extract images
            output_format: Output format ('markdown', 'text', 'structured')
            dpi: DPI for image extraction

        Returns:
            Tuple of (content, metadata)
        """
        try:
            # Configure PyMuPDF4LLM options
            extraction_options = {
                'pages': page_range,
                'write_images': extract_images,
                'image_format': 'png',
                'dpi': dpi,
                'table_strategy': 'lines_strict' if self.preserve_tables else 'lines'
            }

            # Remove None values from options
            extraction_options = {k: v for k, v in extraction_options.items() if v is not None}

            if output_format == 'markdown':
                # Use PyMuPDF4LLM for markdown extraction
                md_text = pymupdf4llm.to_markdown(str(file_path), **extraction_options)
                content = md_text
            else:
                # Use regular PyMuPDF for text extraction if available
                if PYMUPDF_AVAILABLE:
                    content, metadata = self._extract_with_pymupdf(file_path, page_range, extract_images)
                else:
                    # Fallback to PyMuPDF4LLM text extraction
                    md_text = pymupdf4llm.to_markdown(str(file_path), **extraction_options)
                    # Convert markdown to plain text (basic conversion)
                    content = self._markdown_to_text(md_text)

            # Extract metadata
            metadata = self._extract_pdf_metadata(file_path)

            # Add processing metadata
            metadata.update({
                'extraction_method': 'pymupdf4llm',
                'output_format': output_format,
                'page_range': page_range,
                'images_extracted': extract_images,
                'tables_preserved': self.preserve_tables,
                'content_length': len(content)
            })

            return content, metadata

        except Exception as e:
            raise MediaProcessingError(f"PyMuPDF4LLM extraction failed: {str(e)}") from e

    def _extract_with_pymupdf(self, file_path: Path, page_range: Optional[Tuple[int, int]],
                            extract_images: bool) -> Tuple[str, Dict[str, Any]]:
        """
        Extract content using regular PyMuPDF for text-only extraction.

        Args:
            file_path: Path to the PDF file
            page_range: Optional page range to process
            extract_images: Whether to extract images

        Returns:
            Tuple of (content, metadata)
        """
        doc = fitz.open(str(file_path))
        content_parts = []
        images = []

        try:
            # Determine page range
            start_page = page_range[0] if page_range else 0
            end_page = page_range[1] if page_range else doc.page_count - 1
            end_page = min(end_page, doc.page_count - 1)

            for page_num in range(start_page, end_page + 1):
                page = doc[page_num]

                # Extract text
                page_text = page.get_text()
                if page_text.strip():
                    content_parts.append(f"# Page {page_num + 1}\n\n{page_text}\n")

                # Extract images if requested
                if extract_images:
                    page_images = self._extract_page_images(page, page_num)
                    images.extend(page_images)

            content = "\n".join(content_parts)

            metadata = {
                'page_count': doc.page_count,
                'processed_pages': end_page - start_page + 1,
                'images_found': len(images),
                'extraction_method': 'pymupdf'
            }

            if images:
                metadata['images'] = images

            return content, metadata

        finally:
            doc.close()

    def _extract_page_images(self, page, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF page.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            List of image metadata dictionaries
        """
        images = []

        try:
            # Get image list
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    # Convert to PNG bytes
                    img_data = pix.tobytes("png")

                    # Create image metadata
                    image_info = {
                        'page': page_num + 1,
                        'index': img_index,
                        'width': pix.width,
                        'height': pix.height,
                        'colorspace': pix.colorspace.name if pix.colorspace else 'Unknown',
                        'size_bytes': len(img_data),
                        'format': 'png'
                    }

                    images.append(image_info)

                pix = None  # Free memory

        except Exception as e:
            self.logger.warning(f"Failed to extract images from page {page_num}: {e}")

        return images

    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary of PDF metadata
        """
        metadata = {}

        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(str(file_path))
                try:
                    pdf_metadata = doc.metadata

                    # Extract useful metadata
                    metadata.update({
                        'title': pdf_metadata.get('title', ''),
                        'author': pdf_metadata.get('author', ''),
                        'subject': pdf_metadata.get('subject', ''),
                        'creator': pdf_metadata.get('creator', ''),
                        'producer': pdf_metadata.get('producer', ''),
                        'creation_date': pdf_metadata.get('creationDate', ''),
                        'modification_date': pdf_metadata.get('modDate', ''),
                        'page_count': doc.page_count,
                        'encrypted': doc.needs_pass,
                        'pdf_version': doc.pdf_version()
                    })

                    # Clean up empty values
                    metadata = {k: v for k, v in metadata.items() if v}

                finally:
                    doc.close()

        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata: {e}")
            metadata['metadata_extraction_error'] = str(e)

        return metadata

    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Convert markdown content to plain text (basic conversion).

        Args:
            markdown_content: Markdown content

        Returns:
            Plain text content
        """
        import re

        # Remove markdown formatting
        text = markdown_content

        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Remove inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)

        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)

        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def get_pdf_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about a PDF without full processing.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with PDF information
        """
        file_path = Path(file_path)

        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(str(file_path))
                try:
                    info = {
                        'filename': file_path.name,
                        'file_size': file_path.stat().st_size,
                        'page_count': doc.page_count,
                        'encrypted': doc.needs_pass,
                        'pdf_version': doc.pdf_version(),
                        'metadata': doc.metadata
                    }

                    # Get first page info
                    if doc.page_count > 0:
                        first_page = doc[0]
                        info['page_size'] = first_page.rect
                        info['first_page_text_length'] = len(first_page.get_text())

                    return info

                finally:
                    doc.close()
            else:
                # Basic file info only
                return {
                    'filename': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'pymupdf_not_available': True
                }

        except Exception as e:
            return {
                'filename': file_path.name,
                'error': str(e),
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            }

    def extract_text_from_pages(self, file_path: Union[str, Path],
                               start_page: int, end_page: int) -> str:
        """
        Extract text from specific pages of a PDF.

        Args:
            file_path: Path to the PDF file
            start_page: Starting page number (1-based)
            end_page: Ending page number (1-based)

        Returns:
            Extracted text from specified pages
        """
        file_path = Path(file_path)

        try:
            # Convert to 0-based indexing
            page_range = (start_page - 1, end_page - 1)

            # Use PyMuPDF4LLM for extraction
            extraction_options = {
                'pages': page_range,
                'write_images': False,
                'table_strategy': 'lines_strict' if self.preserve_tables else 'lines'
            }

            if self.markdown_output:
                content = pymupdf4llm.to_markdown(str(file_path), **extraction_options)
            else:
                # Extract as markdown then convert to text
                md_content = pymupdf4llm.to_markdown(str(file_path), **extraction_options)
                content = self._markdown_to_text(md_content)

            return content

        except Exception as e:
            raise MediaProcessingError(f"Failed to extract text from pages {start_page}-{end_page}: {str(e)}") from e

    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get information about the PDF processor capabilities.

        Returns:
            Dictionary with processor information
        """
        return {
            'processor_type': 'PDFProcessor',
            'supported_formats': ['pdf'],
            'capabilities': {
                'extract_images': self.extract_images,
                'preserve_tables': self.preserve_tables,
                'markdown_output': self.markdown_output,
                'page_range_support': True,
                'metadata_extraction': self.extract_metadata,
                'pymupdf4llm_integration': True,
                'text_extraction': True,
                'structure_preservation': True
            },
            'dependencies': {
                'pymupdf4llm': PYMUPDF4LLM_AVAILABLE,
                'pymupdf': PYMUPDF_AVAILABLE
            }
        }