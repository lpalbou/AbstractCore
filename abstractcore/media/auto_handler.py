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
from ..exceptions import UnsupportedFeatureError

# Import Glyph compression support
try:
    from ..compression.orchestrator import CompressionOrchestrator
    from ..compression.config import GlyphConfig
    GLYPH_AVAILABLE = True
except ImportError:
    CompressionOrchestrator = None
    GlyphConfig = None
    GLYPH_AVAILABLE = False

# Import vision detection
try:
    from ..architectures.detection import supports_vision
    VISION_DETECTION_AVAILABLE = True
except ImportError:
    supports_vision = None
    VISION_DETECTION_AVAILABLE = False


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
        
        # Initialize Glyph compression support
        self._compression_orchestrator = None
        self.glyph_config = kwargs.get('glyph_config')
        self.enable_compression = kwargs.get('enable_glyph_compression', GLYPH_AVAILABLE)

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
        
        # GlyphProcessor (requires reportlab and pdf2image)
        glyph_deps_available = True
        if GLYPH_AVAILABLE and self.enable_compression:
            # Check actual dependencies
            try:
                import reportlab
                import pdf2image
            except ImportError:
                glyph_deps_available = False
        else:
            glyph_deps_available = False
            
        availability['glyph'] = glyph_deps_available

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
    
    def _get_compression_orchestrator(self) -> 'CompressionOrchestrator':
        """Get or create CompressionOrchestrator instance."""
        if self._compression_orchestrator is None and GLYPH_AVAILABLE:
            config = self.glyph_config or GlyphConfig.from_abstractcore_config()
            self._compression_orchestrator = CompressionOrchestrator(config)
        return self._compression_orchestrator

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
        # Check if Glyph compression should be applied
        provider = kwargs.get('provider')
        model = kwargs.get('model')
        glyph_compression = kwargs.get('glyph_compression', 'auto')
        
        if self._should_apply_compression(file_path, media_type, provider, model, glyph_compression):
            try:
                # Remove provider and model from kwargs to avoid duplicate arguments
                compression_kwargs = {k: v for k, v in kwargs.items() if k not in ['provider', 'model']}
                return self._apply_compression(file_path, provider, model, **compression_kwargs)
            except Exception as e:
                self.logger.warning(f"Glyph compression failed, falling back to standard processing: {e}")
                # Continue with standard processing
        
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
    
    def _should_apply_compression(self, file_path: Path, media_type: MediaType,
                                provider: str, model: str, glyph_compression: str) -> bool:
        """
        Check if Glyph compression should be applied.

        ⚠️ EXPERIMENTAL FEATURE: Glyph compression requires vision-capable models.

        Raises:
            UnsupportedFeatureError: When glyph_compression="always" but model lacks vision support
        """
        # Check if Glyph is available
        if not self._available_processors.get('glyph', False):
            if glyph_compression == "always":
                # User explicitly requested compression but it's not available
                self._log_compression_unavailable_warning()
            return False

        if glyph_compression == "never":
            return False

        # Check vision support for compression
        model_supports_vision = self._check_vision_support(model)

        if glyph_compression == "always":
            # Explicit compression request - enforce vision requirement
            if not model_supports_vision:
                raise UnsupportedFeatureError(
                    f"Glyph compression requires a vision-capable model. "
                    f"Model '{model}' does not support vision. "
                    f"Vision-capable models include: gpt-4o, gpt-4o-mini, claude-3-5-sonnet, "
                    f"llama3.2-vision, qwen2-vl, gemini-1.5-pro, gemini-1.5-flash, etc."
                )
            return True

        # Auto-decision logic
        if not provider or not model:
            return False

        # Only compress text-based content
        if media_type not in [MediaType.TEXT, MediaType.DOCUMENT]:
            return False

        # Auto mode: check vision support and warn if not supported
        if not model_supports_vision:
            self.logger.warning(
                f"Glyph compression skipped: model '{model}' does not support vision. "
                f"Use a vision-capable model to enable compression."
            )
            return False

        try:
            orchestrator = self._get_compression_orchestrator()
            if orchestrator:
                return orchestrator.should_compress(file_path, provider, model, glyph_compression)
        except Exception as e:
            self.logger.debug(f"Compression decision failed: {e}")

        return False

    def _check_vision_support(self, model: str) -> bool:
        """
        Check if the model supports vision capabilities.

        Args:
            model: Model name to check

        Returns:
            True if model supports vision, False otherwise
        """
        if not model or not VISION_DETECTION_AVAILABLE:
            # Conservative approach: assume no vision if detection unavailable
            return False

        try:
            return supports_vision(model)
        except Exception as e:
            self.logger.debug(f"Failed to check vision support for model '{model}': {e}")
            return False

    def _log_compression_unavailable_warning(self):
        """Log detailed warning about why Glyph compression is unavailable."""
        self.logger.warning("Glyph compression requested but not available")
        
        # Check specific reasons
        if not GLYPH_AVAILABLE:
            self.logger.warning("Glyph compression modules could not be imported")
            
        # Check dependencies
        missing_deps = []
        try:
            import reportlab
        except ImportError:
            missing_deps.append("reportlab")
            
        try:
            import pdf2image
        except ImportError:
            missing_deps.append("pdf2image")
            
        if missing_deps:
            deps_str = ", ".join(missing_deps)
            self.logger.warning(f"Missing Glyph dependencies: {deps_str}")
            self.logger.warning(f"Install with: pip install {' '.join(missing_deps)}")
        
        if not self.enable_compression:
            self.logger.warning("Glyph compression is disabled in AutoMediaHandler configuration")
    
    def _apply_compression(self, file_path: Path, provider: str, model: str, **kwargs) -> MediaContent:
        """Apply Glyph compression to the file."""
        media_type = detect_media_type(file_path)
        
        # For PDF files, use direct PDF-to-image conversion (no text extraction!)
        if media_type == MediaType.DOCUMENT and file_path.suffix.lower() == '.pdf':
            try:
                from .processors.direct_pdf_processor import DirectPDFProcessor
                
                # Configure for optimal compression (2 pages per image)
                direct_processor = DirectPDFProcessor(
                    pages_per_image=2,  # 16 pages → 8 images
                    dpi=150,  # Good quality for VLM processing
                    layout='horizontal',  # Side-by-side like open book
                    gap=20,  # Small gap between pages
                    **kwargs
                )
                
                # Get all combined images
                combined_images = direct_processor.get_combined_image_paths(file_path)
                
                # Get session info for metadata from DirectPDFProcessor
                from ..config import get_config_manager
                import hashlib
                config_manager = get_config_manager()
                glyph_cache_base = Path(config_manager.config.cache.glyph_cache_dir).expanduser()
                pdf_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                session_id = f"pdf_{pdf_hash}_{len(combined_images)}pages"
                
                # Create MediaContent objects for each combined image
                media_contents = []
                for i, img_path in enumerate(combined_images):
                    with open(img_path, 'rb') as f:
                        image_data = f.read()
                    
                    import base64
                    encoded_data = base64.b64encode(image_data).decode('utf-8')
                    
                    media_content = MediaContent(
                        media_type=MediaType.IMAGE,
                        content=encoded_data,
                        content_format=ContentFormat.BASE64,
                        mime_type="image/png",
                        metadata={
                            'compression_used': True,
                            'compression_method': 'direct_pdf_conversion',
                            'pages_per_image': 2,
                            'image_index': i,
                            'total_images': len(combined_images),
                            'original_file': str(file_path),
                            'glyph_session_id': session_id,
                            'glyph_cache_dir': str(glyph_cache_base / session_id),
                            'processing_method': 'direct_pdf_conversion'  # For compatibility with test script
                        }
                    )
                    media_contents.append(media_content)
                
                self.logger.info(f"Direct PDF conversion: {len(combined_images)} combined images created")
                
                # Return first image (in full implementation, would handle multiple)
                if media_contents:
                    return media_contents[0]
                else:
                    raise Exception("No combined images created")
                    
            except Exception as e:
                self.logger.warning(f"DirectPDFProcessor failed: {e}, falling back to text extraction")
                # Fall back to text extraction method
                pass
        
        # Fallback: text extraction method (for non-PDF or if direct method fails)
        orchestrator = self._get_compression_orchestrator()
        if not orchestrator:
            raise Exception("Compression orchestrator not available")
        
        if media_type == MediaType.DOCUMENT and file_path.suffix.lower() == '.pdf':
            processor = self._get_pdf_processor()
        elif media_type == MediaType.DOCUMENT:
            processor = self._get_office_processor()
        else:
            processor = self._get_text_processor()
        
        # Extract text content
        extracted_content = processor._process_internal(file_path, media_type, **kwargs)
        text_content = extracted_content.content
        
        # Compress the extracted text content
        glyph_compression = kwargs.get('glyph_compression', 'auto')
        compressed_content = orchestrator.compress_content(text_content, provider, model, glyph_compression)
        
        if compressed_content and len(compressed_content) > 0:
            # Return first compressed image as primary content
            # Additional images can be accessed through metadata
            primary_content = compressed_content[0]
            
            # Add information about additional images
            if len(compressed_content) > 1:
                primary_content.metadata['additional_images'] = len(compressed_content) - 1
                primary_content.metadata['total_compressed_images'] = len(compressed_content)
            
            # Add compression metadata
            primary_content.metadata['compression_used'] = True
            primary_content.metadata['original_file'] = str(file_path)
            
            return primary_content
        else:
            raise Exception("No compressed content generated")

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
            # TextProcessor can handle ANY text file through its plain text fallback
            # This is always available and supports all text-based files
            return True

        elif media_type == MediaType.DOCUMENT:
            # PDF support
            if format_ext.lower() == 'pdf':
                return self._available_processors.get('pdf', False) or True  # Fallback to text

            # Office document support
            if format_ext.lower() in {'docx', 'xlsx', 'pptx'}:
                return self._available_processors.get('office', False) or True  # Fallback to text

            # Any other document type can be handled by text processor as fallback
            # This allows processing of unknown document formats
            return True

        return False

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get supported formats organized by media type.

        Returns comprehensive list of all supported file extensions.
        Note: TEXT type supports ANY text-based file through content detection
        and fallback processing, not just the listed extensions.

        Returns:
            Dictionary mapping media type to list of supported extensions

        Example:
            >>> handler = AutoMediaHandler()
            >>> formats = handler.get_supported_formats()
            >>> len(formats['text'])  # 70+ text extensions
            70+
            >>> 'r' in formats['text']  # R scripts supported
            True
        """
        from .types import get_all_supported_extensions

        # Get comprehensive list from FILE_TYPE_MAPPINGS
        all_formats = get_all_supported_extensions()

        # Filter based on available processors
        result = {}

        # Image formats (requires PIL)
        if self._available_processors.get('image', False):
            result['image'] = all_formats.get('image', [])

        # Text formats (always available - TextProcessor has built-in fallback)
        # Note: This includes 70+ extensions + unknown text files via content detection
        result['text'] = all_formats.get('text', [])

        # Document formats (includes PDFs, Office docs, and text fallbacks)
        result['document'] = all_formats.get('document', [])

        # Audio/Video (not yet implemented but listed for completeness)
        if 'audio' in all_formats:
            result['audio'] = all_formats['audio']
        if 'video' in all_formats:
            result['video'] = all_formats['video']

        return result

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