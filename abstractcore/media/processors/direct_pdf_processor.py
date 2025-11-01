"""
Direct PDF-to-image processor for Glyph compression.

This processor converts PDF pages directly to images without text extraction,
preserving all visual elements including mathematical formulas, tables, and images.
Supports multi-page layouts (e.g., 2 pages per image) for optimal compression.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import tempfile
import os
import hashlib
import math

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    pdf2image = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat


class DirectPDFProcessor(BaseMediaHandler):
    """
    Direct PDF-to-image processor that preserves all visual elements.
    
    Converts PDF pages directly to images without text extraction,
    maintaining perfect fidelity of mathematical formulas, tables, and images.
    """
    
    def __init__(self, **kwargs):
        """Initialize the direct PDF processor."""
        super().__init__(**kwargs)
        
        if not PDF2IMAGE_AVAILABLE:
            raise MediaProcessingError("pdf2image is required for DirectPDFProcessor")
        
        if not PIL_AVAILABLE:
            raise MediaProcessingError("PIL/Pillow is required for DirectPDFProcessor")
        
        # Configuration
        self.pages_per_image = kwargs.get('pages_per_image', 2)  # 2 pages per image by default
        self.dpi = kwargs.get('dpi', 150)  # Higher DPI for better quality
        self.layout = kwargs.get('layout', 'horizontal')  # 'horizontal' or 'vertical'
        self.gap = kwargs.get('gap', 20)  # Gap between pages in pixels
        
        self.logger.debug(f"DirectPDFProcessor initialized: {self.pages_per_image} pages per image")
    
    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """Process PDF directly to images."""
        if media_type != MediaType.DOCUMENT:
            raise MediaProcessingError(f"DirectPDFProcessor only handles documents, got {media_type}")
        
        try:
            # Convert PDF pages to images
            image_paths = self._convert_pdf_to_combined_images(file_path)
            
            # For now, return the first combined image
            # In a full implementation, this would return all images
            if image_paths:
                with open(image_paths[0], 'rb') as f:
                    image_data = f.read()
                
                # Encode as base64 for MediaContent
                import base64
                encoded_data = base64.b64encode(image_data).decode('utf-8')
                
                # Get session info for metadata
                from ...config import get_config_manager
                config_manager = get_config_manager()
                glyph_cache_base = Path(config_manager.config.cache.glyph_cache_dir).expanduser()
                pdf_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                session_id = f"pdf_{pdf_hash}_{len(image_paths)}pages"
                
                metadata = {
                    'processing_method': 'direct_pdf_conversion',
                    'pages_per_image': self.pages_per_image,
                    'total_images': len(image_paths),
                    'dpi': self.dpi,
                    'layout': self.layout,
                    'image_paths': [str(p) for p in image_paths],
                    'glyph_session_id': session_id,
                    'glyph_cache_dir': str(glyph_cache_base / session_id)
                }
                
                return self._create_media_content(
                    content=encoded_data,
                    file_path=file_path,
                    media_type=MediaType.IMAGE,  # Return as image
                    content_format=ContentFormat.BASE64,
                    mime_type="image/png",
                    **metadata
                )
            else:
                raise MediaProcessingError("No images generated from PDF")
                
        except Exception as e:
            raise MediaProcessingError(f"Failed to process PDF directly: {str(e)}") from e
    
    def _convert_pdf_to_combined_images(self, pdf_path: Path) -> List[Path]:
        """Convert PDF to combined images with multiple pages per image."""
        
        # Convert all PDF pages to individual images
        individual_images = pdf2image.convert_from_path(
            pdf_path,
            dpi=self.dpi,
            fmt='PNG'
        )
        
        self.logger.info(f"Converted PDF to {len(individual_images)} individual page images")
        
        # Use AbstractCore's centralized Glyph cache directory
        from ...config import get_config_manager
        config_manager = get_config_manager()
        glyph_cache_base = Path(config_manager.config.cache.glyph_cache_dir).expanduser()
        
        # Calculate number of combined images that will be created
        num_combined_images = math.ceil(len(individual_images) / self.pages_per_image)
        
        # Create a unique subdirectory for this PDF processing session
        pdf_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()[:8]
        session_id = f"pdf_{pdf_hash}_{num_combined_images}pages"
        glyph_dir = glyph_cache_base / session_id
        glyph_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL DEBUG LOG: Show exactly where images are being generated
        self.logger.debug(f"🎯 GENERATING GLYPH IMAGES IN CACHE DIRECTORY: {glyph_dir}")
        self.logger.info(f"DirectPDFProcessor: Creating {self.pages_per_image} pages per image in {glyph_dir}")
        
        # Combine pages into multi-page images
        combined_images = []
        
        try:
            for i in range(0, len(individual_images), self.pages_per_image):
                # Get pages for this combined image
                pages_batch = individual_images[i:i + self.pages_per_image]
                
                # Create combined image
                combined_image = self._combine_pages(pages_batch, i // self.pages_per_image)
                
                # Save combined image in Glyph cache
                output_path = glyph_dir / f"image_{i // self.pages_per_image + 1:03d}.png"
                combined_image.save(output_path, 'PNG', optimize=True)
                combined_images.append(output_path)
                
                # CRITICAL DEBUG LOG: Show each image as it's created
                self.logger.debug(f"📄 CREATED GLYPH IMAGE: {output_path} (pages {i+1}-{min(i+self.pages_per_image, len(individual_images))})")
        
        except Exception as e:
            # Clean up cache directory on error
            import shutil
            shutil.rmtree(glyph_dir, ignore_errors=True)
            raise e
        
        # CRITICAL DEBUG LOG: Final summary with exact paths
        self.logger.info(f"Created {len(combined_images)} combined images from {len(individual_images)} pages")
        self.logger.debug(f"🎯 ALL GLYPH IMAGES STORED IN CACHE: {glyph_dir}")
        for i, img_path in enumerate(combined_images, 1):
            self.logger.debug(f"   📄 Image {i}: {img_path}")
        
        return combined_images
    
    def _combine_pages(self, pages: List['Image.Image'], batch_index: int) -> 'Image.Image':
        """Combine multiple PDF pages into a single image."""
        if not pages:
            raise ValueError("No pages to combine")
        
        if len(pages) == 1:
            return pages[0]
        
        # Calculate dimensions for combined image
        if self.layout == 'horizontal':
            # Side-by-side layout (like an open book)
            total_width = sum(page.width for page in pages) + self.gap * (len(pages) - 1)
            total_height = max(page.height for page in pages)
        else:
            # Vertical layout (pages stacked)
            total_width = max(page.width for page in pages)
            total_height = sum(page.height for page in pages) + self.gap * (len(pages) - 1)
        
        # Create new image with white background
        combined = Image.new('RGB', (total_width, total_height), 'white')
        
        # Paste pages into combined image
        current_x, current_y = 0, 0
        
        for page in pages:
            combined.paste(page, (current_x, current_y))
            
            if self.layout == 'horizontal':
                current_x += page.width + self.gap
            else:
                current_y += page.height + self.gap
        
        return combined
    
    def get_combined_image_paths(self, pdf_path: Path) -> List[Path]:
        """Get paths to all combined images (for external use)."""
        return self._convert_pdf_to_combined_images(pdf_path)
