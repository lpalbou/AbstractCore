"""
Media processors for different file types.

This module contains concrete implementations of media processors
for various file formats including images, documents, and text files.
"""

from .image_processor import ImageProcessor
from .text_processor import TextProcessor
from .pdf_processor import PDFProcessor
from .office_processor import OfficeProcessor

# Import Glyph processor if available
try:
    from ...compression.glyph_processor import GlyphProcessor
    GLYPH_AVAILABLE = True
except ImportError:
    GlyphProcessor = None
    GLYPH_AVAILABLE = False

__all__ = ['ImageProcessor', 'TextProcessor', 'PDFProcessor', 'OfficeProcessor']
if GLYPH_AVAILABLE:
    __all__.append('GlyphProcessor')