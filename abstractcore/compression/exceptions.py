"""
Glyph compression exceptions.
"""

from ..exceptions import AbstractCoreError


class CompressionError(AbstractCoreError):
    """Base exception for compression-related errors."""
    pass


class CompressionQualityError(CompressionError):
    """Exception raised when compression quality is below threshold."""
    
    def __init__(self, message: str, quality_score: float = None, threshold: float = None):
        super().__init__(message)
        self.quality_score = quality_score
        self.threshold = threshold


class RenderingError(CompressionError):
    """Exception raised when text rendering fails."""
    pass


class CompressionCacheError(CompressionError):
    """Exception raised when compression cache operations fail."""
    pass

