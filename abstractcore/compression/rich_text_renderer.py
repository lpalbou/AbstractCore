"""
Enhanced ReportLab renderer with rich text formatting support.

This renderer can handle TextSegment objects with proper bold and italic
font rendering using ReportLab's rich text capabilities.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from .config import GlyphConfig, RenderingConfig
from .text_formatter import TextSegment
from .exceptions import RenderingError
from ..utils.structured_logging import get_logger


class RichTextRenderer:
    """Enhanced text renderer with rich formatting support using ReportLab."""
    
    def __init__(self, config: GlyphConfig):
        """
        Initialize rich text renderer.
        
        Args:
            config: Glyph configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Check dependencies
        self._check_dependencies()
        
        self.logger.debug("RichTextRenderer initialized")
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import reportlab
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_LEFT
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
    
    def segments_to_images(
        self,
        segments: List[TextSegment],
        config: RenderingConfig,
        output_dir: Optional[str] = None,
        unique_id: Optional[str] = None
    ) -> List[Path]:
        """
        Convert TextSegment objects to optimized images using ReportLab rich text.
        
        Args:
            segments: List of TextSegment objects with formatting
            config: Rendering configuration
            output_dir: Output directory for images
            unique_id: Unique identifier for this rendering
            
        Returns:
            List of paths to rendered images
        """
        try:
            # Import dependencies
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import pdf2image
            
            # Setup output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="glyph_rich_")
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            output_dir = Path(output_dir)
            unique_id = unique_id or "render"
            
            # Create PDF file
            pdf_path = output_dir / f"{unique_id}.pdf"
            
            # Setup document with custom page size and margins
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                leftMargin=config.margin_x,
                rightMargin=config.margin_x,
                topMargin=config.margin_y,
                bottomMargin=config.margin_y
            )
            
            # Register fonts
            font_name = self._register_fonts(config)
            
            # Create styles
            styles = self._create_styles(config, font_name)
            
            # Convert segments to ReportLab story
            story = self._segments_to_story(segments, styles, config)
            
            # Build PDF
            doc.build(story)
            
            self.logger.debug(f"Generated rich text PDF: {pdf_path}")
            
            # Convert PDF to images
            images = self._pdf_to_images(pdf_path, config, output_dir, unique_id)
            
            # Clean up PDF if not needed
            try:
                pdf_path.unlink()
            except Exception:
                pass
            
            return images
            
        except Exception as e:
            self.logger.error(f"Rich text rendering failed: {e}")
            raise RenderingError(f"Failed to render rich text: {e}") from e
    
    def _register_fonts(self, config: RenderingConfig) -> str:
        """Register fonts with proper family mapping for bold/italic support."""
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
        
        font_name = "Helvetica"  # Default font
        
        try:
            # Always use Helvetica family which has guaranteed bold/italic support
            # Register the built-in Helvetica family with proper mapping
            
            # Helvetica fonts are built-in, but we need to map them properly
            addMapping('Helvetica', 0, 0, 'Helvetica')           # Normal
            addMapping('Helvetica', 0, 1, 'Helvetica-Oblique')  # Italic  
            addMapping('Helvetica', 1, 0, 'Helvetica-Bold')     # Bold
            addMapping('Helvetica', 1, 1, 'Helvetica-BoldOblique')  # Bold Italic
            
            font_name = "Helvetica"
            self.logger.info("Registered Helvetica font family with bold/italic mapping")
            
            # TODO: Add custom font support later if needed
            # For now, focus on getting Helvetica bold/italic working
                        
        except Exception as e:
            self.logger.warning(f"Failed to register font family mapping: {e}")
            font_name = "Helvetica"
        
        return font_name
    
    def _create_styles(self, config: RenderingConfig, font_name: str) -> dict:
        """Create paragraph styles for different formatting."""
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        
        styles = {}
        
        # Base style
        styles['normal'] = ParagraphStyle(
            'Normal',
            fontName=font_name,
            fontSize=config.font_size,
            leading=config.line_height,
            alignment=TA_LEFT,
            spaceAfter=0,
            spaceBefore=0
        )
        
        # Bold style - always use Helvetica-Bold which is guaranteed to exist
        styles['bold'] = ParagraphStyle(
            'Bold',
            parent=styles['normal'],
            fontName="Helvetica-Bold"
        )
        
        # Italic style - always use Helvetica-Oblique which is guaranteed to exist
        styles['italic'] = ParagraphStyle(
            'Italic',
            parent=styles['normal'],
            fontName="Helvetica-Oblique"
        )
        
        # Header styles
        styles['header1'] = ParagraphStyle(
            'Header1',
            parent=styles['bold'],
            fontSize=config.font_size + 2,
            spaceAfter=config.line_height // 2
        )
        
        styles['header2'] = ParagraphStyle(
            'Header2',
            parent=styles['bold'],
            fontSize=config.font_size + 1,
            spaceAfter=config.line_height // 3
        )
        
        styles['header3'] = ParagraphStyle(
            'Header3',
            parent=styles['bold'],
            fontSize=config.font_size,
            spaceAfter=config.line_height // 4
        )
        
        return styles
    
    def _segments_to_story(self, segments: List[TextSegment], styles: dict, config: RenderingConfig) -> List:
        """Convert TextSegment objects to ReportLab story elements."""
        from reportlab.platypus import Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        
        story = []
        current_paragraph_html = ""
        
        for segment in segments:
            if segment.text == "\n":
                # End current paragraph and start new one
                if current_paragraph_html.strip():
                    para = Paragraph(current_paragraph_html, styles['normal'])
                    story.append(para)
                    current_paragraph_html = ""
                # Add small spacer for line break
                story.append(Spacer(1, config.line_height // 2))
                
            elif segment.is_header:
                # Add header as inline bold text using HTML tags (NEVER as separate paragraph - rule #9)
                current_paragraph_html += f'<b>{self._escape_html(segment.text)}</b>'
                
            else:
                # Add to current paragraph with appropriate formatting using HTML tags
                if segment.is_bold and segment.is_italic:
                    current_paragraph_html += f'<b><i>{self._escape_html(segment.text)}</i></b>'
                elif segment.is_bold:
                    current_paragraph_html += f'<b>{self._escape_html(segment.text)}</b>'
                elif segment.is_italic:
                    current_paragraph_html += f'<i>{self._escape_html(segment.text)}</i>'
                else:
                    current_paragraph_html += self._escape_html(segment.text)
        
        # Add final paragraph if any
        if current_paragraph_html.strip():
            para = Paragraph(current_paragraph_html, styles['normal'])
            story.append(para)
        
        return story
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
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
            
            self.logger.debug(f"Generated {len(image_paths)} images from rich text PDF")
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
