"""
Glyph visual-text compression system for AbstractCore.

This module provides visual-text compression capabilities that transform long textual
sequences into optimized images for processing by Vision-Language Models (VLMs),
achieving 3-4x token compression without accuracy loss.

Based on the Glyph framework by Z.ai/THU-COAI with AbstractCore-specific enhancements.
"""

from .glyph_processor import GlyphProcessor
from .orchestrator import CompressionOrchestrator
from .config import GlyphConfig, RenderingConfig
from .quality import QualityValidator, CompressionStats
from .cache import CompressionCache
from .exceptions import CompressionError, CompressionQualityError

__all__ = [
    'GlyphProcessor',
    'CompressionOrchestrator', 
    'GlyphConfig',
    'RenderingConfig',
    'QualityValidator',
    'CompressionStats',
    'CompressionCache',
    'CompressionError',
    'CompressionQualityError'
]

