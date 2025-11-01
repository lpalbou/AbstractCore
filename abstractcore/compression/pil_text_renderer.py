"""
PIL/Pillow-based text renderer with proper bold and italic font support.

This renderer directly creates images using PIL/Pillow, giving us complete control
over font rendering and text layout.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import math

from .config import GlyphConfig, RenderingConfig
from .text_formatter import TextSegment
from .exceptions import RenderingError
from ..utils.structured_logging import get_logger


class PILTextRenderer:
    """Direct text renderer using PIL/Pillow with proper font support."""
    
    def __init__(self, config: GlyphConfig):
        """
        Initialize PIL text renderer.
        
        Args:
            config: Glyph configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
        # Check dependencies
        self._check_dependencies()
        
        self.logger.debug("PILTextRenderer initialized")
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as e:
            raise RenderingError(
                f"PIL/Pillow not available: {e}. Install with: pip install pillow"
            )
    
    def segments_to_images(
        self,
        segments: List[TextSegment],
        config: RenderingConfig,
        output_dir: Optional[str] = None,
        unique_id: Optional[str] = None
    ) -> List[Path]:
        """
        Convert TextSegment objects to images using PIL/Pillow.
        
        Args:
            segments: List of TextSegment objects with formatting
            config: Rendering configuration
            output_dir: Output directory for images
            unique_id: Unique identifier for this rendering
            
        Returns:
            List of paths to rendered images
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Setup output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="glyph_pil_")
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            output_dir = Path(output_dir)
            unique_id = unique_id or "render"
            
            # Load fonts
            fonts = self._load_fonts(config)
            
            # Calculate text layout
            lines = self._layout_text(segments, fonts, config)
            
            # Calculate image dimensions
            img_width, img_height = self._calculate_image_size(lines, fonts, config)
            
            # Create image
            # Use white background for better compression
            image = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(image)
            
            # Render text
            self._render_text_to_image(draw, lines, fonts, config)
            
            # Save image
            image_path = output_dir / f"{unique_id}_page_1.png"
            image.save(image_path, 'PNG', optimize=True)
            
            self.logger.debug(f"Generated PIL image: {image_path} ({img_width}x{img_height})")
            
            return [image_path]
            
        except Exception as e:
            self.logger.error(f"PIL text rendering failed: {e}")
            raise RenderingError(f"Failed to render text with PIL: {e}") from e
    
    def _load_fonts(self, config: RenderingConfig) -> dict:
        """Load fonts for different styles."""
        from PIL import ImageFont
        
        fonts = {}
        font_size = config.font_size
        
        try:
            # Try to load system fonts
            # macOS system fonts
            font_paths = {
                'regular': [
                    '/System/Library/Fonts/Helvetica.ttc',
                    '/System/Library/Fonts/Arial.ttf',
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
                    'arial.ttf'  # Windows
                ],
                'bold': [
                    '/System/Library/Fonts/Helvetica.ttc',  # Contains bold variant
                    '/System/Library/Fonts/Arial Bold.ttf',
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',  # Linux
                    'arialbd.ttf'  # Windows
                ],
                'italic': [
                    '/System/Library/Fonts/Helvetica.ttc',  # Contains italic variant
                    '/System/Library/Fonts/Arial Italic.ttf',
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf',  # Linux
                    'ariali.ttf'  # Windows
                ],
                'bold_italic': [
                    '/System/Library/Fonts/Helvetica.ttc',  # Contains bold italic variant
                    '/System/Library/Fonts/Arial Bold Italic.ttf',
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf',  # Linux
                    'arialbi.ttf'  # Windows
                ]
            }
            
            # Load regular font
            fonts['regular'] = self._load_font_from_paths(font_paths['regular'], font_size, 'regular')
            
            # Load bold font
            fonts['bold'] = self._load_font_from_paths(font_paths['bold'], font_size, 'bold')
            
            # Load italic font  
            fonts['italic'] = self._load_font_from_paths(font_paths['italic'], font_size, 'italic')
            
            # Load bold italic font
            fonts['bold_italic'] = self._load_font_from_paths(font_paths['bold_italic'], font_size, 'bold_italic')
            
            self.logger.info(f"Loaded PIL fonts: regular={fonts['regular'] is not None}, "
                           f"bold={fonts['bold'] is not None}, italic={fonts['italic'] is not None}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load system fonts, using default: {e}")
            # Fallback to default font
            fonts = {
                'regular': ImageFont.load_default(),
                'bold': ImageFont.load_default(),
                'italic': ImageFont.load_default(),
                'bold_italic': ImageFont.load_default()
            }
        
        return fonts
    
    def _load_font_from_paths(self, paths: List[str], size: int, style: str = 'regular'):
        """Try to load font from a list of possible paths."""
        from PIL import ImageFont
        
        for path in paths:
            try:
                if Path(path).exists():
                    # For .ttc files (TrueType Collections), try different indices for different styles
                    if path.endswith('.ttc'):
                        # Map styles to font indices in Helvetica.ttc
                        style_indices = {
                            'regular': 0,      # Helvetica Regular
                            'bold': 1,         # Helvetica Bold  
                            'italic': 2,       # Helvetica Oblique
                            'bold_italic': 3   # Helvetica Bold Oblique
                        }
                        index = style_indices.get(style, 0)
                        try:
                            font = ImageFont.truetype(path, size, index=index)
                            self.logger.debug(f"Loaded {style} font from {path} index {index}")
                            return font
                        except Exception as e:
                            self.logger.warning(f"Failed to load {style} from {path} index {index}: {e}")
                            continue
                    else:
                        font = ImageFont.truetype(path, size)
                        self.logger.debug(f"Loaded {style} font from {path}")
                        return font
            except Exception as e:
                self.logger.debug(f"Failed to load font from {path}: {e}")
                continue
        
        # Fallback to default
        self.logger.warning(f"Using default font for {style}")
        return ImageFont.load_default()
    
    def _layout_text(self, segments: List[TextSegment], fonts: dict, config: RenderingConfig) -> List[List[dict]]:
        """
        Layout text segments into lines with word wrapping.
        
        Returns:
            List of lines, where each line is a list of text chunks with formatting info
        """
        lines = []
        current_line = []
        current_line_width = 0
        
        # Calculate available width
        page_width = config.page_width
        margin_x = config.margin_x
        available_width = page_width - 2 * margin_x
        
        # Handle multi-column layout
        columns = max(1, config.columns)
        column_gap = config.column_gap if columns > 1 else 0
        column_width = (available_width - (columns - 1) * column_gap) / columns
        
        self.logger.debug(f"Layout: page_width={page_width}, available_width={available_width}, "
                         f"columns={columns}, column_width={column_width}")
        
        for segment in segments:
            if segment.text == "\n":
                # Force line break
                if current_line:
                    lines.append(current_line)
                    current_line = []
                    current_line_width = 0
                continue
            
            # Get appropriate font
            font = self._get_font_for_segment(segment, fonts)
            
            # Handle space-only segments (like "   " from single newlines)
            if segment.text.strip() == "":
                # This is a space-only segment - preserve it
                space_width = self._get_text_width(segment.text, font)
                if current_line_width + space_width <= column_width:
                    # Add spaces to current line
                    current_line.append({
                        'text': segment.text,
                        'font': font,
                        'segment': segment
                    })
                    current_line_width += space_width
                else:
                    # Start new line with spaces
                    if current_line:
                        lines.append(current_line)
                    current_line = [{
                        'text': segment.text,
                        'font': font,
                        'segment': segment
                    }]
                    current_line_width = space_width
                continue
            
            # Split segment text into words (for non-space segments)
            words = segment.text.split()
            
            for word_idx, word in enumerate(words):
                # Calculate word width (without extra space for last word in segment)
                is_last_word_in_segment = (word_idx == len(words) - 1)
                word_text = word if is_last_word_in_segment else word + " "
                word_width = self._get_text_width(word_text, font)
                
                # Check if word fits on current line
                fits_on_line = (current_line_width + word_width <= column_width)
                
                if fits_on_line or not current_line:  # Always add first word to empty line
                    # Add to current line
                    current_line.append({
                        'text': word_text,
                        'font': font,
                        'segment': segment
                    })
                    current_line_width += word_width
                else:
                    # Start new line
                    if current_line:
                        lines.append(current_line)
                    current_line = [{
                        'text': word_text,
                        'font': font,
                        'segment': segment
                    }]
                    current_line_width = word_width
        
        # Add final line
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _get_font_for_segment(self, segment: TextSegment, fonts: dict):
        """Get the appropriate font for a text segment."""
        if segment.is_bold and segment.is_italic:
            font = fonts['bold_italic']
            self.logger.debug(f"Using bold_italic font for: '{segment.text[:20]}...'")
            return font
        elif segment.is_header or segment.is_bold:
            # Headers should ONLY be bold, never italic
            font = fonts['bold']
            self.logger.debug(f"Using bold font for: '{segment.text[:20]}...'")
            return font
        elif segment.is_italic:
            font = fonts['italic']
            self.logger.debug(f"Using italic font for: '{segment.text[:20]}...'")
            return font
        else:
            font = fonts['regular']
            self.logger.debug(f"Using regular font for: '{segment.text[:20]}...'")
            return font
    
    def _get_text_width(self, text: str, font) -> int:
        """Get the width of text in pixels."""
        from PIL import Image, ImageDraw
        
        if not text:
            return 0
            
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        try:
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]
        except:
            # Fallback for older PIL versions
            return temp_draw.textsize(text, font=font)[0]
    
    def _get_text_height(self, font) -> int:
        """Get the height of text in pixels."""
        from PIL import Image, ImageDraw
        
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        try:
            bbox = temp_draw.textbbox((0, 0), "Ag", font=font)
            return bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            return temp_draw.textsize("Ag", font=font)[1]
    
    def _calculate_image_size(self, lines: List[List[dict]], fonts: dict, config: RenderingConfig) -> Tuple[int, int]:
        """Calculate the required image size."""
        if not lines:
            return (config.page_width, 100)  # Minimum size
        
        # Calculate width (use page width)
        width = config.page_width
        
        # Calculate height with better spacing
        line_height = config.line_height
        line_spacing = int(line_height * 1.3)  # 30% more space between lines
        total_height = len(lines) * line_spacing + 2 * config.margin_y + 20
        
        # Add some padding
        height = max(total_height, 100)
        
        self.logger.debug(f"Image size calculation: lines={len(lines)}, "
                         f"line_height={line_height}, line_spacing={line_spacing}, height={height}")
        
        return (int(width), int(height))
    
    def _render_text_to_image(self, draw, lines: List[List[dict]], fonts: dict, config: RenderingConfig):
        """Render text lines to the image."""
        y = config.margin_y
        line_height = config.line_height
        line_spacing = int(line_height * 1.3)  # Match the spacing calculation
        
        for line in lines:
            x = config.margin_x
            
            for chunk in line:
                text = chunk['text']
                font = chunk['font']
                
                # Draw text
                draw.text((x, y), text, font=font, fill='black')
                
                # Move x position
                x += self._get_text_width(text, font)
            
            # Move to next line with better spacing
            y += line_spacing
