"""
PDF processor with a permissive default backend.

The default path uses pypdf for text and metadata extraction so the standard
media extras stay suitable for permissive/commercial redistribution. The older
PyMuPDF4LLM path remains available only when explicitly requested through the
commercial opt-in extra.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import importlib.util
import json
import re

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat
from ...utils.token_utils import estimate_tokens


PYPDF_INSTALL_HINT = 'Install with: pip install "abstractcore[media]"'
PYMUPDF_INSTALL_HINT = (
    'Install the explicit opt-in backend only after license review: '
    'pip install "abstractcore[pdf-pymupdf-commercial]"'
)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_pypdf_reader():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MediaProcessingError(
            f"pypdf is required for default PDF processing. {PYPDF_INSTALL_HINT}"
        ) from exc
    return PdfReader


def _load_pymupdf4llm():
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise MediaProcessingError(
            "The optional PyMuPDF4LLM PDF backend is not installed. "
            f"{PYMUPDF_INSTALL_HINT}"
        ) from exc
    return pymupdf4llm


def _load_pymupdf():
    try:
        import pymupdf
    except ImportError as exc:
        raise MediaProcessingError(
            "The optional PyMuPDF backend is not installed. "
            f"{PYMUPDF_INSTALL_HINT}"
        ) from exc
    return pymupdf


def _safe_pdf_version(doc: Any) -> Optional[str]:
    """Best-effort PDF version across optional PyMuPDF variants."""
    try:
        pv = getattr(doc, "pdf_version", None)
        if pv is not None:
            out = pv() if callable(pv) else pv
            if out is not None:
                s = str(out).strip()
                if s and s.lower() != "none":
                    return s
    except Exception:
        pass

    # PyMuPDF 1.26+ exposes the PDF version via `doc.metadata["format"]` (e.g. "PDF 1.5").
    try:
        md = getattr(doc, "metadata", None)
        if isinstance(md, dict):
            fmt = md.get("format")
            if isinstance(fmt, str) and fmt.strip():
                m = re.search(r"(?i)pdf\s*[- ]?\s*([0-9]+(?:\.[0-9]+)?)", fmt.strip())
                if m:
                    return m.group(1)
    except Exception:
        pass

    return None


class PDFProcessor(BaseMediaHandler):
    """
    PDF processor for LLM-oriented document processing.

    The default backend is pypdf. It provides permissive text and metadata
    extraction. The optional PyMuPDF4LLM backend can be requested explicitly for
    higher-fidelity Markdown/layout extraction after license approval.
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
                - pdf_backend: 'pypdf' (default) or explicit 'pymupdf4llm'
        """
        super().__init__(**kwargs)

        # PDF processing configuration
        self.pdf_backend = self._normalize_pdf_backend(
            kwargs.get('pdf_backend') or kwargs.get('backend') or 'pypdf'
        )
        self.extract_images = kwargs.get('extract_images', False)
        self.preserve_tables = kwargs.get('preserve_tables', True)
        self.markdown_output = kwargs.get('markdown_output', True)
        self.page_range = kwargs.get('page_range', None)
        self.extract_metadata = kwargs.get('extract_metadata', True)

        # Set capabilities for PDF processing
        from ..types import MediaCapabilities
        # The permissive pypdf backend extracts text/metadata only. Do not report
        # image/vision support unless the explicit high-fidelity backend is selected.
        self.capabilities = MediaCapabilities(
            vision_support=self.extract_images and self.pdf_backend == 'pymupdf4llm',
            audio_support=False,
            video_support=False,
            document_support=True,
            supported_document_formats=['pdf'],
            max_file_size=self.max_file_size
        )

        self.logger.debug(
            f"Initialized PDFProcessor with backend={self.pdf_backend}, "
            f"extract_images={self.extract_images}, "
            f"preserve_tables={self.preserve_tables}, markdown_output={self.markdown_output}"
        )

    @staticmethod
    def _normalize_pdf_backend(value: Any) -> str:
        raw = str(value or 'pypdf').strip().lower().replace('_', '-')
        if raw in {'pypdf', 'default', 'permissive', 'basic'}:
            return 'pypdf'
        if raw in {'pymupdf', 'pymupdf4llm', 'commercial', 'high-fidelity', 'layout'}:
            return 'pymupdf4llm'
        raise MediaProcessingError(
            f"Unsupported PDF backend '{value}'. Use 'pypdf' or explicit 'pymupdf4llm'."
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

            # Process PDF with the configured backend.
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

            # Add token estimation to metadata (no truncation, just informational)
            metadata['estimated_tokens'] = estimate_tokens(content)
            metadata['content_length'] = len(content)

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
        Extract content from PDF using the configured backend.

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
            if self.pdf_backend == 'pymupdf4llm':
                content, metadata = self._extract_with_pymupdf4llm(
                    file_path, page_range, extract_images, output_format, dpi
                )
            else:
                content, metadata = self._extract_with_pypdf(file_path, page_range, output_format)

            # Add processing metadata
            metadata.update({
                'pdf_backend': self.pdf_backend,
                'output_format': output_format,
                'page_range': page_range,
                'images_extracted': bool(extract_images and self.pdf_backend == 'pymupdf4llm'),
                'content_length': len(content)
            })

            return content, metadata

        except Exception as e:
            raise MediaProcessingError(f"PDF extraction failed with {self.pdf_backend}: {str(e)}") from e

    def _resolve_page_window(self, total_pages: int, page_range: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if total_pages <= 0:
            return 0, -1
        start_page = int(page_range[0]) if page_range else 0
        end_page = int(page_range[1]) if page_range else total_pages - 1
        start_page = max(0, min(start_page, total_pages - 1))
        end_page = max(start_page, min(end_page, total_pages - 1))
        return start_page, end_page

    def _extract_with_pypdf(self, file_path: Path, page_range: Optional[Tuple[int, int]],
                           output_format: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata with pypdf, the default permissive backend.
        """
        PdfReader = _load_pypdf_reader()
        reader = PdfReader(str(file_path))
        total_pages = len(reader.pages)
        start_page, end_page = self._resolve_page_window(total_pages, page_range)
        pages: List[Dict[str, Any]] = []
        content_parts: List[str] = []
        warnings: List[str] = []

        for page_num in range(start_page, end_page + 1):
            page = reader.pages[page_num]
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                page_text = ""
                warnings.append(f"Page {page_num + 1} text extraction failed: {exc}")
            pages.append({"page": page_num + 1, "text": page_text})
            if output_format == 'markdown':
                content_parts.append(f"# Page {page_num + 1}\n\n{page_text.strip()}")
            else:
                content_parts.append(page_text)

        if output_format == 'structured':
            content = json.dumps(
                {
                    "pages": pages,
                    "page_count": total_pages,
                    "processed_pages": len(pages),
                },
                ensure_ascii=False,
                indent=2,
            )
        else:
            content = "\n\n".join(part for part in content_parts if part is not None).strip()

        metadata = self._extract_pdf_metadata(file_path)
        metadata.update({
            'page_count': total_pages,
            'processed_pages': len(pages),
            'extraction_method': 'pypdf',
            'tables_preserved': False,
            'images_found': 0,
        })

        if self.preserve_tables:
            warnings.append("pypdf extracts text but does not preserve table structure.")
        if self.extract_images:
            warnings.append("pypdf default backend does not extract embedded images.")
        if getattr(reader, "is_encrypted", False):
            warnings.append("PDF is encrypted; extracted text may be incomplete without decryption.")
        if warnings:
            metadata["warnings"] = warnings

        return content, metadata

    def _extract_with_pymupdf4llm(self, file_path: Path, page_range: Optional[Tuple[int, int]],
                                 extract_images: bool, output_format: str,
                                 dpi: int) -> Tuple[str, Dict[str, Any]]:
        """
        Extract content with the explicit commercial PyMuPDF4LLM backend.
        """
        pymupdf4llm = _load_pymupdf4llm()
        extraction_options = {
            'pages': page_range,
            'write_images': extract_images,
            'image_format': 'png',
            'dpi': dpi,
            'table_strategy': 'lines_strict' if self.preserve_tables else 'lines'
        }
        extraction_options = {k: v for k, v in extraction_options.items() if v is not None}

        md_text = pymupdf4llm.to_markdown(str(file_path), **extraction_options)
        if output_format == 'markdown':
            content = md_text
        elif output_format == 'structured' and hasattr(pymupdf4llm, 'to_json'):
            content = json.dumps(pymupdf4llm.to_json(str(file_path), **extraction_options), ensure_ascii=False)
        else:
            content = self._markdown_to_text(md_text)

        metadata = self._extract_pdf_metadata(file_path)
        metadata.update({
            'extraction_method': 'pymupdf4llm',
            'tables_preserved': self.preserve_tables,
        })
        return content, metadata

    def _extract_with_pymupdf(self, file_path: Path, page_range: Optional[Tuple[int, int]],
                            extract_images: bool) -> Tuple[str, Dict[str, Any]]:
        """
        Extract content using regular PyMuPDF for the explicit optional backend.

        Args:
            file_path: Path to the PDF file
            page_range: Optional page range to process
            extract_images: Whether to extract images

        Returns:
            Tuple of (content, metadata)
        """
        fitz = _load_pymupdf()
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
                    page_images = self._extract_page_images(page, page_num, fitz)
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

    def _extract_page_images(self, page, page_num: int, fitz_module) -> List[Dict[str, Any]]:
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
                pix = fitz_module.Pixmap(page.parent, xref)

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
            PdfReader = _load_pypdf_reader()
            reader = PdfReader(str(file_path))
            pdf_metadata = reader.metadata or {}

            # Extract useful metadata. pypdf metadata keys usually keep the PDF slash prefix.
            metadata.update({
                'title': pdf_metadata.get('/Title') or pdf_metadata.get('title') or '',
                'author': pdf_metadata.get('/Author') or pdf_metadata.get('author') or '',
                'subject': pdf_metadata.get('/Subject') or pdf_metadata.get('subject') or '',
                'creator': pdf_metadata.get('/Creator') or pdf_metadata.get('creator') or '',
                'producer': pdf_metadata.get('/Producer') or pdf_metadata.get('producer') or '',
                'creation_date': str(pdf_metadata.get('/CreationDate') or ''),
                'modification_date': str(pdf_metadata.get('/ModDate') or ''),
                'page_count': len(reader.pages),
                'encrypted': bool(getattr(reader, "is_encrypted", False)),
            })

            # Clean up empty values while preserving false booleans and zero counts.
            metadata = {k: v for k, v in metadata.items() if v not in ("", None)}

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
            PdfReader = _load_pypdf_reader()
            reader = PdfReader(str(file_path))
            info = {
                'filename': file_path.name,
                'file_size': file_path.stat().st_size,
                'page_count': len(reader.pages),
                'encrypted': bool(getattr(reader, "is_encrypted", False)),
                'metadata': dict(reader.metadata or {}),
                'extraction_method': 'pypdf',
            }

            if len(reader.pages) > 0:
                first_page_text = reader.pages[0].extract_text() or ""
                mediabox = reader.pages[0].mediabox
                info['page_size'] = {
                    'width': float(mediabox.width),
                    'height': float(mediabox.height),
                }
                info['first_page_text_length'] = len(first_page_text)

            return info

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

            output_format = 'markdown' if self.markdown_output else 'text'
            content, _metadata = self._extract_pdf_content(
                file_path, page_range, False, output_format, 150
            )

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
            'default_backend': self.pdf_backend,
            'capabilities': {
                'extract_images': self.extract_images,
                'preserve_tables': self.preserve_tables,
                'markdown_output': self.markdown_output,
                'page_range_support': True,
                'metadata_extraction': self.extract_metadata,
                'pymupdf4llm_integration': self.pdf_backend == 'pymupdf4llm',
                'text_extraction': True,
                'structure_preservation': self.pdf_backend == 'pymupdf4llm',
                'permissive_default_backend': self.pdf_backend == 'pypdf',
            },
            'dependencies': {
                'pypdf': _module_available('pypdf'),
                'pymupdf4llm': _module_available('pymupdf4llm'),
                'pymupdf': _module_available('pymupdf'),
            }
        }
