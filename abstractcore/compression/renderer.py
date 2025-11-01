"""
ReportLab-based text renderer for Glyph compression.

Based on the actual Glyph implementation using reportlab for PDF generation
and pdf2image for conversion to optimized images.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import logging

from .config import GlyphConfig, RenderingConfig
from .exceptions import RenderingError
from ..utils.structured_logging import get_logger


class ReportLabRenderer:
    """Text renderer using ReportLab for PDF generation."""
    
    def __init__(self, config: GlyphConfig):
        """
        Initialize ReportLab renderer.
        
        Args:
            config: Glyph configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Check dependencies
        self._check_dependencies()
        
        self.logger.debug("ReportLabRenderer initialized")
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import reportlab
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
        except ImportError as e:
            raise RenderingError(
                f"ReportLab not available: {e}. Install with: pip install reportlab"
            )
        
        try:
            import pdf2image
        except ImportError as e:
            raise RenderingError(
                f"pdf2image not available: {e}. Install with: pip install pdf2image"
            )
    
    def text_to_images(
        self, 
        text: str, 
        config: RenderingConfig,
        output_dir: Optional[str] = None,
        unique_id: Optional[str] = None
    ) -> List[Path]:
        """
        Convert text to optimized images using ReportLab pipeline.
        
        Args:
            text: Text content to render
            config: Rendering configuration
            output_dir: Output directory for images
            unique_id: Unique identifier for this rendering
            
        Returns:
            List of paths to rendered images
        """
        try:
            # Import dependencies
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.units import inch
            import pdf2image
            
            # Setup output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="glyph_")
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            output_dir = Path(output_dir)
            unique_id = unique_id or "render"
            
            # Create PDF file
            pdf_path = output_dir / f"{unique_id}.pdf"
            
            # Setup canvas
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            page_width, page_height = A4
            
            # Register font if available
            font_name = "Helvetica"  # Default font
            try:
                if config.font_path and Path(config.font_path).exists():
                    pdfmetrics.registerFont(TTFont('CustomFont', config.font_path))
                    font_name = 'CustomFont'
                elif config.font_path == "Verdana.ttf":
                    # Try to find system Verdana font
                    verdana_paths = [
                        "/System/Library/Fonts/Verdana.ttf",  # macOS
                        "/Windows/Fonts/verdana.ttf",         # Windows
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux fallback
                    ]
                    for font_path in verdana_paths:
                        if Path(font_path).exists():
                            pdfmetrics.registerFont(TTFont('Verdana', font_path))
                            font_name = 'Verdana'
                            break
            except Exception as e:
                self.logger.warning(f"Failed to load custom font, using default: {e}")
            
            # Setup text rendering parameters
            c.setFont(font_name, config.font_size)
            
            # Calculate text area with multi-column support
            margin_x = config.margin_x
            margin_y = config.margin_y
            total_text_width = page_width - 2 * margin_x
            text_height = page_height - 2 * margin_y
            
            # Multi-column layout calculation (key optimization from original Glyph)
            columns = max(1, config.columns)
            column_gap = config.column_gap if columns > 1 else 0
            column_width = (total_text_width - (columns - 1) * column_gap) / columns
            
            # Process text with newline markup if configured
            if config.newline_markup and config.newline_markup != "\\n":
                # Simple newline markup processing
                processed_text = text.replace('\n', ' \\n ')
            else:
                processed_text = text
            
            # Split text into lines that fit the column width
            lines = self._wrap_text(processed_text, column_width, c, font_name, config.font_size)
            
            # Calculate lines per page and per column
            line_height = config.line_height
            lines_per_column = int(text_height / line_height)
            lines_per_page = lines_per_column * columns
            
            # Render pages with multi-column support
            current_line = 0
            page_count = 0
            
            while current_line < len(lines):
                if page_count > 0:
                    c.showPage()  # Start new page
                    c.setFont(font_name, config.font_size)
                
                page_count += 1
                
                # Render columns for this page
                for col in range(columns):
                    if current_line >= len(lines):
                        break
                    
                    # Calculate column position
                    col_x = margin_x + col * (column_width + column_gap)
                    y_position = page_height - margin_y - line_height
                    
                    # Render lines for this column
                    for i in range(lines_per_column):
                        if current_line >= len(lines):
                            break
                        
                        line = lines[current_line]
                        c.drawString(col_x, y_position, line)
                    y_position -= line_height
                        current_line += 1
            
            # Save PDF
            c.save()
            
            self.logger.debug(f"Generated PDF with {page_count} pages: {pdf_path}")
            
            # Convert PDF to images
            images = self._pdf_to_images(pdf_path, config, output_dir, unique_id)
            
            # Clean up PDF if not needed
            try:
                pdf_path.unlink()
            except Exception:
                pass
            
            return images
            
        except Exception as e:
            self.logger.error(f"Text rendering failed: {e}")
            raise RenderingError(f"Failed to render text: {e}") from e
    
    def _wrap_text(self, text: str, max_width: float, canvas, font_name: str, font_size: int) -> List[str]:
        """Wrap text to fit within specified width."""
        lines = []
        
        # Split by existing line breaks first
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                lines.append("")  # Empty line
                continue
            
            words = paragraph.split()
            current_line = ""
            
            for word in words:
                test_line = f"{current_line} {word}".strip()
                
                # Check if line fits
                text_width = canvas.stringWidth(test_line, font_name, font_size)
                
                if text_width <= max_width:
                    current_line = test_line
                else:
                    # Line too long, start new line
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            # Add remaining text
            if current_line:
                lines.append(current_line)
        
        return lines
    
    def _pdf_to_images(
        self, 
        pdf_path: Path, 
        config: RenderingConfig,
        output_dir: Path,
        unique_id: str
    ) -> List[Path]:
        """Convert PDF to optimized images."""
        try:
            import pdf2image
            from PIL import Image
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=config.dpi,
                fmt='PNG',
                thread_count=1  # Conservative for stability
            )
            
            image_paths = []
            
            for i, img in enumerate(images):
                # Apply cropping if configured
                if config.auto_crop_width or (config.auto_crop_last_page and i == len(images) - 1):
                    img = self._auto_crop_image(img)
                
                # Save image
                image_path = output_dir / f"{unique_id}_page_{i+1}.png"
                img.save(image_path, 'PNG', optimize=True)
                image_paths.append(image_path)
            
            self.logger.debug(f"Generated {len(image_paths)} images from PDF")
            return image_paths
            
        except Exception as e:
            self.logger.error(f"PDF to image conversion failed: {e}")
            raise RenderingError(f"Failed to convert PDF to images: {e}") from e
    
    def _auto_crop_image(self, img) -> 'Image':
        """Auto-crop image to remove excessive whitespace."""
        try:
            from PIL import Image, ImageOps
            
            # Convert to grayscale for easier processing
            gray = img.convert('L')
            
            # Find bounding box of non-white content
            # Invert image so text becomes white and background becomes black
            inverted = ImageOps.invert(gray)
            
            # Get bounding box
            bbox = inverted.getbbox()
            
            if bbox:
                # Add small margin
                margin = 10
                left, top, right, bottom = bbox
                left = max(0, left - margin)
                top = max(0, top - margin)
                right = min(img.width, right + margin)
                bottom = min(img.height, bottom + margin)
                
                # Crop image
                return img.crop((left, top, right, bottom))
            
            return img
            
        except Exception as e:
            self.logger.warning(f"Auto-crop failed, using original image: {e}")
            return img

