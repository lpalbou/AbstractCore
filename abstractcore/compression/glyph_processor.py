"""
Glyph visual-text compression processor for AbstractCore.

Based on the actual Glyph implementation using reportlab for PDF generation
and pdf2image for conversion, with provider-specific optimization.
"""

import time
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from ..media.base import BaseMediaHandler
from ..media.types import MediaContent, MediaType, ContentFormat
from ..utils.structured_logging import get_logger
from ..utils.token_utils import TokenUtils

from .config import GlyphConfig, RenderingConfig
from .quality import QualityValidator, CompressionStats
from .cache import CompressionCache
# ReportLab renderers removed - now using PIL renderer exclusively
from .pil_text_renderer import PILTextRenderer
from .text_formatter import TextFormatter, FormattingConfig
from .exceptions import CompressionError, CompressionQualityError


class GlyphProcessor(BaseMediaHandler):
    """
    Glyph visual-text compression processor for AbstractCore.
    
    Transforms long textual sequences into optimized images for processing
    by Vision-Language Models (VLMs), achieving 3-4x token compression.
    """
    
    def __init__(self, config: Optional[GlyphConfig] = None, **kwargs):
        """
        Initialize Glyph processor.
        
        Args:
            config: Glyph configuration
            **kwargs: Additional configuration passed to base handler
        """
        super().__init__(**kwargs)
        
        self.config = config or GlyphConfig.from_abstractcore_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize components
        self.pil_text_renderer = None  # Lazy initialization
        self.quality_validator = QualityValidator()
        self.cache = CompressionCache(
            cache_dir=self.config.cache_directory,
            max_size_gb=self.config.cache_size_gb,
            ttl_days=self.config.cache_ttl_days
        )
        
        # Initialize text formatter
        self.text_formatter = TextFormatter(FormattingConfig())
        
        # Load provider profiles
        self.provider_profiles = self.config.provider_profiles
        
        self.logger.debug("GlyphProcessor initialized")
    
    def _get_pil_text_renderer(self) -> 'PILTextRenderer':
        """Get or create PIL text renderer instance (lazy initialization)."""
        if self.pil_text_renderer is None:
            self.pil_text_renderer = PILTextRenderer(self.config)
        return self.pil_text_renderer
    
    def can_process(self, content: str, provider: str, model: str) -> bool:
        """
        Determine if content should be compressed using Glyph.
        
        Args:
            content: Text content to evaluate
            provider: Provider name
            model: Model name
            
        Returns:
            True if compression should be applied
        """
        # Check if compression is enabled
        if not self.config.enabled:
            return False
        
        # Estimate token count
        token_count = TokenUtils.estimate_tokens(content, model)
        
        # Check minimum token threshold
        if token_count < self.config.min_token_threshold:
            return False
        
        # Check if provider supports vision
        try:
            # Lazy import to avoid circular dependency
            from ..media.capabilities import get_model_capabilities
            capabilities = get_model_capabilities(provider, model)
            if not capabilities.get('vision_support', False):
                return False
        except Exception:
            # Conservative approach if capabilities unknown
            return False
        
        return True
    
    def process_text(self, content: str, provider: str = None, model: str = None, user_preference: str = "auto") -> List[MediaContent]:
        """
        Process text content into compressed visual format.
        
        Args:
            content: Text content to compress
            provider: Provider name for optimization
            model: Model name for optimization
            
        Returns:
            List of MediaContent objects with compressed images
        """
        start_time = time.time()
        
        try:
            # Get provider-specific configuration
            render_config = self._get_provider_config(provider, model)
            
            # Apply text formatting if enabled
            processed_content = content
            text_segments = None
            if render_config.render_format:
                self.logger.debug("Applying text formatting")
                text_segments = self.text_formatter.format_text(content)
                # For now, convert back to string for compatibility with existing renderer
                processed_content = self.text_formatter.format_text_to_string(content)
                self.logger.debug("Text formatting applied", 
                                original_length=len(content),
                                formatted_length=len(processed_content),
                                segments_count=len(text_segments))
            else:
                self.logger.debug("Text formatting disabled, using raw content")
            
            # Check cache first (use processed content for cache key)
            cache_key = self._generate_cache_key(processed_content, render_config)
            if cached_result := self.cache.get(cache_key):
                self.logger.debug(f"Using cached compression for key {cache_key[:16]}...")
                return self._create_media_content_from_images(cached_result, processed_content, provider, render_config)
            
            # Always use PIL text renderer (ReportLab removed)
            self.logger.debug("Using PIL text renderer")
            pil_renderer = self._get_pil_text_renderer()
            
            if render_config.render_format and text_segments:
                # Use formatted text segments
                self.logger.debug("Rendering with text formatting")
                images = pil_renderer.segments_to_images(
                    segments=text_segments,
                    config=render_config,
                    output_dir=self.config.temp_dir,
                    unique_id=cache_key[:16]
                )
            else:
                # Convert plain text to segments for PIL renderer
                self.logger.debug("Rendering plain text (no formatting)")
                from .text_formatter import TextSegment
                plain_segments = [TextSegment(text=processed_content)]
                images = pil_renderer.segments_to_images(
                    segments=plain_segments,
                config=render_config,
                output_dir=self.config.temp_dir,
                unique_id=cache_key[:16]
            )
            
            # Quality validation (bypass if user explicitly wants compression)
            quality_score = self.quality_validator.assess(content, images, provider)
            min_threshold = self.quality_validator.get_provider_threshold(provider)
            
            if user_preference != "always" and quality_score < min_threshold:
                raise CompressionQualityError(
                    f"Compression quality {quality_score:.3f} below threshold {min_threshold:.3f} for {provider}",
                    quality_score=quality_score,
                    threshold=min_threshold
                )
            elif user_preference == "always" and quality_score < min_threshold:
                self.logger.warning(f"Compression quality {quality_score:.3f} below threshold {min_threshold:.3f} for {provider}, but proceeding due to 'always' preference")
            
            # Calculate compression statistics
            original_tokens = TokenUtils.estimate_tokens(processed_content, model)
            # Calculate accurate token count using proper VLM token calculation
            from ..utils.vlm_token_calculator import VLMTokenCalculator
            from pathlib import Path
            
            calculator = VLMTokenCalculator()
            try:
                # Extract image paths from MediaContent objects
                image_paths = []
                for img in images:
                    if hasattr(img, 'metadata') and img.metadata.get('temp_file_path'):
                        image_paths.append(Path(img.metadata['temp_file_path']))
                    elif hasattr(img, 'file_path') and img.file_path:
                        image_paths.append(Path(img.file_path))
                
                if image_paths:
                    token_analysis = calculator.calculate_tokens_for_images(
                        image_paths=image_paths,
                        provider=provider or 'openai',
                        model=model or ''
                    )
                    compressed_tokens = token_analysis['total_tokens']
                    self.logger.info(f"Accurate token calculation: {compressed_tokens} tokens for {len(image_paths)} images")
                else:
                    # Fallback calculation
                    base_tokens = calculator.PROVIDER_CONFIGS.get(provider or 'openai', {}).get('base_tokens', 512)
                    compressed_tokens = len(images) * base_tokens
                    self.logger.warning(f"Using fallback token estimation: {compressed_tokens} tokens")
                    
            except Exception as e:
                self.logger.warning(f"VLM token calculation failed, using fallback: {e}")
                compressed_tokens = len(images) * 1500  # Conservative fallback
            compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
            
            compression_stats = CompressionStats(
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                token_savings=original_tokens - compressed_tokens,
                processing_time=time.time() - start_time,
                provider_optimized=provider or "unknown",
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens
            )
            
            # Cache successful compression
            self.cache.set(cache_key, images, compression_stats.to_dict())
            
            # Create MediaContent objects
            media_contents = self._create_media_content_from_images(
                images, processed_content, provider, render_config, compression_stats
            )
            
            self.logger.info(
                f"Glyph compression completed: {compression_ratio:.1f}x ratio, "
                f"{quality_score:.1%} quality, {len(images)} images"
            )
            
            return media_contents
            
        except Exception as e:
            self.logger.error(f"Glyph compression failed: {e}")
            raise CompressionError(f"Compression failed: {e}") from e
    
    def _get_provider_config(self, provider: str, model: str) -> RenderingConfig:
        """Get provider-specific rendering configuration."""
        return self.config.get_provider_config(provider, model)
    
    def _generate_cache_key(self, content: str, config: RenderingConfig) -> str:
        """Generate cache key from content and configuration."""
        import hashlib
        
        # Create hash of content + configuration
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        config_hash = hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode('utf-8')
        ).hexdigest()[:8]
        
        return f"{content_hash}_{config_hash}"
    
    def _create_media_content_from_images(
        self, 
        images: List[Path], 
        original_content: str,
        provider: str,
        render_config: RenderingConfig,
        compression_stats: Optional[CompressionStats] = None
    ) -> List[MediaContent]:
        """Create MediaContent objects from rendered images."""
        import base64
        
        media_contents = []
        
        for i, img_path in enumerate(images):
            if not img_path.exists():
                continue
            
            # Read and encode image
            with open(img_path, 'rb') as f:
                image_data = f.read()
            
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Create metadata
            metadata = {
                "compression_ratio": compression_stats.compression_ratio if compression_stats else None,
                "quality_score": compression_stats.quality_score if compression_stats else None,
                "rendering_config": render_config.to_dict(),
                "provider_optimized": provider,
                "glyph_version": "1.0",
                "processing_time": compression_stats.processing_time if compression_stats else None,
                "image_index": i,
                "total_images": len(images),
                "original_content_length": len(original_content),
                "dpi": render_config.dpi,
                "font_config": {
                    "font_path": render_config.font_path,
                    "font_size": render_config.font_size,
                    "line_height": render_config.line_height
                }
            }
            
            # Create MediaContent
            media_content = MediaContent(
                media_type=MediaType.IMAGE,
                content=base64_data,
                content_format=ContentFormat.BASE64,
                mime_type="image/png",
                file_path=str(img_path),
                metadata=metadata
            )
            
            media_contents.append(media_content)
        
        return media_contents
    
    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Internal processing method for file-based compression.
        
        Args:
            file_path: Path to text file
            media_type: Detected media type
            **kwargs: Additional parameters including provider and model
            
        Returns:
            MediaContent with compressed representation
        """
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise CompressionError(f"Failed to read file {file_path}: {e}")
        
        # Extract provider and model from kwargs
        provider = kwargs.get('provider')
        model = kwargs.get('model')
        
        # Process text content
        media_contents = self.process_text(content, provider, model)
        
        # Return first image (for single file processing)
        if media_contents:
            return media_contents[0]
        else:
            raise CompressionError("No compressed content generated")
    
    def supports_media_type(self, media_type: MediaType) -> bool:
        """Check if this processor supports the media type."""
        # Glyph processor handles text content for compression
        return media_type in [MediaType.TEXT, MediaType.DOCUMENT]
    
    def supports_format(self, media_type: MediaType, format_ext: str) -> bool:
        """Check if this processor supports the format."""
        if media_type in [MediaType.TEXT, MediaType.DOCUMENT]:
            # Support text-based formats
            supported_formats = {'txt', 'md', 'csv', 'tsv', 'json', 'yaml', 'yml'}
            return format_ext.lower() in supported_formats
        return False
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            'processor': 'GlyphProcessor',
            'version': '1.0',
            'cache_stats': cache_stats,
            'config': {
                'enabled': self.config.enabled,
                'quality_threshold': self.config.quality_threshold,
                'min_token_threshold': self.config.min_token_threshold,
                'target_compression_ratio': self.config.target_compression_ratio
            },
            'provider_profiles': list(self.provider_profiles.keys())
        }

