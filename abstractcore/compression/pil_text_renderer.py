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
        
        # Track if we're using OCRB fonts for special bold handling
        self.using_ocrb_fonts = False
        
        self.logger.debug("PILTextRenderer initialized")
    
    def _get_effective_dimensions(self, config: RenderingConfig) -> tuple[int, int]:
        """Get effective image dimensions (target dimensions or VLM-optimized defaults)."""
        if config.target_width and config.target_height:
            return (config.target_width, config.target_height)
        else:
            # VLM-optimized defaults: 1024x768 works well with most vision models
            # - Fits within Claude 3.5 Sonnet (1568x1568 max)
            # - Efficient for GPT-4o tokenization (~1700 tokens)
            # - Good text aspect ratio (4:3)
            # - Supported by all major VLM families
            return (1024, 768)
    
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
            
            # Calculate text layout (returns columns)
            columns_data = self._layout_text(segments, fonts, config)
            
            # Get target dimensions
            img_width, img_height = self._get_effective_dimensions(config)
            
            # Create image with exact dimensions
            # Use white background for better compression
            image = Image.new('RGB', (img_width, img_height), 'white')
            
            # Set DPI information on the image
            dpi_tuple = (config.dpi, config.dpi)
            image.info['dpi'] = dpi_tuple
            
            draw = ImageDraw.Draw(image)
            
            # Render text
            self._render_text_to_image(draw, columns_data, fonts, config)
            
            # Save image with DPI information
            image_path = output_dir / f"{unique_id}_page_1.png"
            image.save(image_path, 'PNG', optimize=True, dpi=dpi_tuple)
            
            self.logger.debug(f"Generated PIL image: {image_path} ({img_width}x{img_height})")
            
            return [image_path]
            
        except Exception as e:
            self.logger.error(f"PIL text rendering failed: {e}")
            raise RenderingError(f"Failed to render text with PIL: {e}") from e
    
    def _load_fonts(self, config: RenderingConfig) -> dict:
        """Load fonts for different styles with custom font support."""
        from PIL import ImageFont
        import os
        
        fonts = {}
        font_size = config.font_size
        
        self.logger.debug(f"Loading fonts with size: {font_size}")
        
        try:
            # Check if custom font path is specified
            if config.font_path and os.path.exists(config.font_path):
                self.logger.info(f"Using custom font path: {config.font_path}")
                try:
                    # Try to load the custom font for all styles
                    base_font = ImageFont.truetype(config.font_path, font_size)
                    fonts['regular'] = base_font
                    fonts['bold'] = base_font  # Use same font for all styles
                    fonts['italic'] = base_font
                    fonts['bold_italic'] = base_font
                    
                    self.logger.info(f"Successfully loaded custom font: {config.font_path}")
                    return fonts
                except Exception as e:
                    self.logger.warning(f"Failed to load custom font {config.font_path}: {e}. Falling back to system fonts.")
            
            # Check if custom font name is specified
            if config.font_name:
                self.logger.info(f"Trying to load font by name: {config.font_name}")
                
                # Special handling for OCRB font family
                if config.font_name.upper() == "OCRB":
                    return self._load_ocrb_font_family(font_size)
                
                try:
                    # Try to load by name (works on some systems)
                    base_font = ImageFont.truetype(config.font_name, font_size)
                    fonts['regular'] = base_font
                    fonts['bold'] = base_font
                    fonts['italic'] = base_font
                    fonts['bold_italic'] = base_font
                    
                    self.logger.info(f"Successfully loaded font by name: {config.font_name}")
                    return fonts
                except Exception as e:
                    self.logger.warning(f"Failed to load font by name {config.font_name}: {e}. Falling back to system fonts.")
            
            # Fall back to system fonts with good readability
            self.logger.info("Using default system fonts")
            font_paths = {
                'regular': [
                    '/System/Library/Fonts/Helvetica.ttc',  # macOS
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
    
    def _load_ocrb_font_family(self, font_size: int) -> dict:
        """Load OCRB font family with proper regular and italic variants."""
        from PIL import ImageFont
        import os
        
        fonts = {}
        
        # Define paths to OCRB font files relative to the package
        try:
            # Get the path to the assets directory
            assets_dir = Path(__file__).parent.parent / "assets"
            ocrb_regular_path = assets_dir / "OCRB.ttf"
            ocrb_italic_path = assets_dir / "OCRBL.ttf"
            
            self.logger.info(f"Loading OCRB font family from assets directory")
            
            # Load regular font
            if ocrb_regular_path.exists():
                fonts['regular'] = ImageFont.truetype(str(ocrb_regular_path), font_size)
                fonts['bold'] = ImageFont.truetype(str(ocrb_regular_path), font_size)  # Use regular for bold
                self.logger.info(f"Loaded OCRB regular font: {ocrb_regular_path}")
            else:
                self.logger.warning(f"OCRB regular font not found at: {ocrb_regular_path}")
                fonts['regular'] = ImageFont.load_default()
                fonts['bold'] = ImageFont.load_default()
            
            # Load italic font (OCRBL)
            if ocrb_italic_path.exists():
                fonts['italic'] = ImageFont.truetype(str(ocrb_italic_path), font_size)
                fonts['bold_italic'] = ImageFont.truetype(str(ocrb_italic_path), font_size)  # Use italic for bold-italic
                self.logger.info(f"Loaded OCRB italic font: {ocrb_italic_path}")
            else:
                self.logger.warning(f"OCRB italic font not found at: {ocrb_italic_path}")
                fonts['italic'] = fonts['regular']  # Fall back to regular
                fonts['bold_italic'] = fonts['regular']
            
            self.logger.info("Successfully loaded OCRB font family with proper italic support")
            self.using_ocrb_fonts = True  # Set flag for special bold handling
            return fonts
            
        except Exception as e:
            self.logger.error(f"Failed to load OCRB font family: {e}")
            # Fall back to default fonts
            return {
                'regular': ImageFont.load_default(),
                'bold': ImageFont.load_default(),
                'italic': ImageFont.load_default(),
                'bold_italic': ImageFont.load_default()
            }
    
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
    
    def _layout_text(self, segments: List[TextSegment], fonts: dict, config: RenderingConfig) -> List[List[List[dict]]]:
        """
        Layout text segments into columns and lines with word wrapping.
        
        Returns:
            List of columns, where each column is a list of lines,
            and each line is a list of text chunks with formatting info
        """
        lines = []
        current_line = []
        current_line_width = 0
        
        # Calculate available width using effective dimensions
        img_width, img_height = self._get_effective_dimensions(config)
        margin_x = config.margin_x
        available_width = img_width - 2 * margin_x
        
        # Handle multi-column layout properly
        columns = max(1, config.columns)
        column_gap = config.column_gap if columns > 1 else 0
        column_width = (available_width - (columns - 1) * column_gap) / columns
        
        self.logger.debug(f"Layout: img_width={img_width}, available_width={available_width}, "
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
                space_width = self._get_text_width(segment.text, font, segment)
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
            
            # Split segment text into words but preserve spaces more carefully
            # Use a simple approach: split on spaces but keep track of them
            import re
            
            # Split while preserving spaces - use regex to capture spaces
            parts = re.split(r'(\s+)', segment.text)
            
            for part in parts:
                if not part:  # Skip empty parts
                    continue
                    
                part_width = self._get_text_width(part, font, segment)
                
                # Check if part fits on current line
                if current_line_width + part_width <= column_width or not current_line:
                    # Add to current line
                    current_line.append({
                        'text': part,
                        'font': font,
                        'segment': segment
                    })
                    current_line_width += part_width
                else:
                    # Start new line
                    if current_line:
                        lines.append(current_line)
                    current_line = [{
                        'text': part,
                        'font': font,
                        'segment': segment
                    }]
                    current_line_width = part_width
        
        # Add final line
        if current_line:
            lines.append(current_line)
        
        # Now distribute lines among columns
        if columns == 1:
            return [lines]  # Single column
        
        # Multi-column: distribute lines evenly
        column_data = [[] for _ in range(columns)]
        lines_per_column = len(lines) // columns
        extra_lines = len(lines) % columns
        
        line_idx = 0
        for col in range(columns):
            lines_in_this_column = lines_per_column + (1 if col < extra_lines else 0)
            column_data[col] = lines[line_idx:line_idx + lines_in_this_column]
            line_idx += lines_in_this_column
        
        return column_data
    
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
    
    def _get_text_width(self, text: str, font, segment=None) -> int:
        """Get the width of text in pixels, accounting for stroke effects."""
        from PIL import Image, ImageDraw
        
        if not text:
            return 0
            
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        try:
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            base_width = bbox[2] - bbox[0]
        except:
            # Fallback for older PIL versions
            base_width = temp_draw.textsize(text, font=font)[0]
        
        # Add extra width for OCRB bold overlay effect
        if (self.using_ocrb_fonts and segment and 
            (segment.is_bold or segment.is_header) and not segment.is_italic):
            # Add width for enhanced horizontal overlays (0.6 pixel max offset)
            return int(base_width + 0.6)
        
        return base_width
    
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
    
    def _calculate_image_size(self, columns_data: List[List[List[dict]]], fonts: dict, config: RenderingConfig) -> Tuple[int, int]:
        """Calculate the required image size for multi-column layout with DPI scaling."""
        if not columns_data or not any(columns_data):
            return (config.page_width, self._scale_dimension(100, config))  # Minimum size
        
        # Calculate width (use page width - no scaling needed as it's already in points)
        width = config.page_width
        
        # Calculate height based on the tallest column with DPI scaling
        scaled_line_height = self._scale_dimension(config.line_height, config)
        scaled_line_spacing = int(scaled_line_height * 1.3)  # 30% more space between lines
        scaled_margin_y = self._scale_dimension(config.margin_y, config)
        scaled_padding = self._scale_dimension(20, config)
        
        max_lines_in_column = max(len(column) for column in columns_data)
        total_height = max_lines_in_column * scaled_line_spacing + 2 * scaled_margin_y + scaled_padding
        
        # Add some padding
        min_height = self._scale_dimension(100, config)
        height = max(total_height, min_height)
        
        self.logger.debug(f"Image size calculation: columns={len(columns_data)}, "
                         f"max_lines_in_column={max_lines_in_column}, "
                         f"scaled_line_height={scaled_line_height}, scaled_line_spacing={scaled_line_spacing}, height={height}")
        
        return (int(width), int(height))
    
    def _render_text_to_image(self, draw, columns_data: List[List[List[dict]]], fonts: dict, config: RenderingConfig):
        """Render text columns to the image with proper multi-column support."""
        line_height = config.line_height
        line_spacing = int(line_height * 1.3)  # Match the spacing calculation
        
        # Calculate column layout using effective dimensions
        columns = len(columns_data)
        img_width, img_height = self._get_effective_dimensions(config)
        available_width = img_width - 2 * config.margin_x
        column_gap = config.column_gap if columns > 1 else 0
        column_width = (available_width - (columns - 1) * column_gap) / columns
        
        # Render each column
        for col_idx, column_lines in enumerate(columns_data):
            # Calculate column x position
            column_x = config.margin_x + col_idx * (column_width + column_gap)
            
            # Render this column
            y = config.margin_y
            for line in column_lines:
                x = column_x
                
                for chunk in line:
                    text = chunk['text']
                    font = chunk['font']
                    segment = chunk.get('segment')
                    
                    # Draw text with special handling for OCRB bold
                    self._draw_text_with_effects(draw, (x, y), text, font, segment)
                    
                    # Move x position
                    x += self._get_text_width(text, font, segment)
                
                # Move to next line
                y += line_spacing
    
    def _draw_text_with_effects(self, draw, position, text, font, segment):
        """Draw text with special effects for OCRB bold text."""
        x, y = position
        
        # Check if this should be bold and we're using OCRB fonts
        if (self.using_ocrb_fonts and segment and 
            (segment.is_bold or segment.is_header) and not segment.is_italic):
            
            # Use improved bold effect for OCRB text
            self.logger.debug(f"Drawing OCRB bold text with enhanced effect: '{text[:20]}...'")
            
            # Method: Enhanced multiple overlays for more visible bold effect
            try:
                # Draw the base text
                draw.text((x, y), text, font=font, fill='black')
                
                # Add horizontal overlays for width (increased for 10% more visibility)
                draw.text((x + 0.2, y), text, font=font, fill='black')
                draw.text((x + 0.4, y), text, font=font, fill='black')
                draw.text((x + 0.6, y), text, font=font, fill='black')  # Additional overlay
                
                # Add vertical overlays for height (enhanced for better visibility)
                draw.text((x, y - 0.1), text, font=font, fill='black')
                draw.text((x, y + 0.1), text, font=font, fill='black')  # Bottom overlay
                
                # Add diagonal overlays for smoother appearance
                draw.text((x + 0.1, y - 0.05), text, font=font, fill='black')
                
            except Exception as e:
                # Fallback: just draw regular text if anything fails
                self.logger.debug(f"OCRB bold effect failed, using regular text: {e}")
                draw.text((x, y), text, font=font, fill='black')
        else:
            # Regular text drawing
            draw.text((x, y), text, font=font, fill='black')
