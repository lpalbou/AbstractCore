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
        
        # Track if we're using OCRB or OCRA fonts for special bold handling
        self.using_ocrb_fonts = False
        self.using_ocra_fonts = False
        
        self.logger.debug("PILTextRenderer initialized")
    
    def _get_effective_dimensions(self, config: RenderingConfig) -> tuple[int, int]:
        """Get effective image dimensions (target dimensions or VLM-optimized defaults)."""
        if config.target_width and config.target_height:
            return (config.target_width, config.target_height)
        else:
            # VLM-optimized defaults: 1024x1024 works well with most vision models
            # - Fits within Claude 3.5 Sonnet (1568x1568 max)
            # - Efficient for GPT-4o tokenization (~1800 tokens)
            # - Square aspect ratio (1:1) for consistent layout
            # - Supported by all major VLM families
            return (1024, 1024)
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as e:
            raise RenderingError(
                f"PIL/Pillow not available: {e}. Install with: pip install pillow"
            )
    
    def _estimate_text_capacity(self, config: RenderingConfig, fonts: dict) -> int:
        """
        Estimate how many characters can fit in the target image dimensions.
        
        Args:
            config: Rendering configuration
            fonts: Loaded fonts dictionary
            
        Returns:
            Estimated character capacity
        """
        img_width, img_height = self._get_effective_dimensions(config)
        
        # Calculate available space
        available_width = img_width - 2 * config.margin_x
        available_height = img_height - 2 * config.margin_y
        
        # Account for multi-column layout
        columns = max(1, config.columns)
        column_gap = config.column_gap if columns > 1 else 0
        column_width = (available_width - (columns - 1) * column_gap) / columns
        
        # Estimate character dimensions using regular font
        regular_font = fonts.get('regular')
        if regular_font:
            # Use average character width (test with common characters)
            test_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
            try:
                # Create temporary image to measure text
                from PIL import Image, ImageDraw
                temp_img = Image.new('RGB', (1, 1))
                temp_draw = ImageDraw.Draw(temp_img)
                
                try:
                    bbox = temp_draw.textbbox((0, 0), test_chars, font=regular_font)
                    total_width = bbox[2] - bbox[0]
                except AttributeError:
                    # Fallback for older PIL versions
                    total_width = temp_draw.textsize(test_chars, font=regular_font)[0]
                
                avg_char_width = total_width / len(test_chars)
            except:
                # Fallback estimate
                avg_char_width = config.font_size * 0.6
        else:
            # Fallback estimate
            avg_char_width = config.font_size * 0.6
        
        # Estimate line capacity
        line_height = int(config.line_height * 1.3)  # Match spacing calculation
        chars_per_line = int(column_width / avg_char_width)
        lines_per_column = int((available_height * 1.1 + line_height) / line_height)
        total_lines = lines_per_column * columns
        
        # More realistic capacity estimation
        # The previous 30% efficiency was way too conservative
        if min(img_width, img_height) < 600:
            # For small images like 448x448, use more realistic estimate
            efficiency_factor = 0.75  # Much higher to use available space
        else:
            efficiency_factor = 0.85
        
        estimated_capacity = int(chars_per_line * total_lines * efficiency_factor)
        
        self.logger.debug(f"Text capacity estimation: {estimated_capacity} chars "
                         f"({chars_per_line} chars/line × {total_lines} lines × {efficiency_factor} efficiency)")
        
        return max(estimated_capacity, 500)  # Minimum 500 chars per page for small images

    def _split_segments_into_pages(self, segments: List[TextSegment], capacity_per_page: int) -> List[List[TextSegment]]:
        """
        Split text segments into pages based on estimated capacity.
        Handles large segments by splitting them if needed.
        
        Args:
            segments: List of TextSegment objects
            capacity_per_page: Estimated character capacity per page
            
        Returns:
            List of pages, each containing a list of segments
        """
        pages = []
        current_page = []
        current_page_chars = 0
        
        for segment in segments:
            segment_length = len(segment.text)
            
            # If this single segment is larger than page capacity, split it
            if segment_length > capacity_per_page:
                # Finish current page if it has content
                if current_page:
                    pages.append(current_page)
                    current_page = []
                    current_page_chars = 0
                
                # Split the large segment into chunks
                text = segment.text
                while text:
                    # Take a chunk that fits the capacity
                    chunk_size = min(capacity_per_page, len(text))
                    
                    # Try to break at word boundaries if possible
                    if chunk_size < len(text):
                        # Look for a space within the last 20% of the chunk
                        search_start = max(0, int(chunk_size * 0.8))
                        space_pos = text.rfind(' ', search_start, chunk_size)
                        if space_pos > search_start:
                            chunk_size = space_pos + 1  # Include the space
                    
                    chunk_text = text[:chunk_size]
                    text = text[chunk_size:]
                    
                    # Create new segment with same formatting
                    chunk_segment = TextSegment(
                        text=chunk_text,
                        is_bold=segment.is_bold,
                        is_italic=segment.is_italic,
                        is_header=segment.is_header,
                        header_level=segment.header_level
                    )
                    
                    # Add as a new page
                    pages.append([chunk_segment])
                
            else:
                # Normal segment handling
                # Check if adding this segment would exceed capacity
                if current_page_chars + segment_length > capacity_per_page and current_page:
                    # Start new page
                    pages.append(current_page)
                    current_page = [segment]
                    current_page_chars = segment_length
                else:
                    # Add to current page
                    current_page.append(segment)
                    current_page_chars += segment_length
        
        # Add the last page if it has content
        if current_page:
            pages.append(current_page)
        
        # Log pagination statistics
        if pages:
            page_sizes = [sum(len(s.text) for s in page) for page in pages]
            avg_size = sum(page_sizes) / len(pages)
            min_size = min(page_sizes)
            max_size = max(page_sizes)
            
            self.logger.debug(f"Split {len(segments)} segments into {len(pages)} pages")
            self.logger.debug(f"Page sizes - avg: {avg_size:.0f}, min: {min_size}, max: {max_size} chars")
            
            # Warn about very small pages (less than 20% of capacity)
            small_pages = [i+1 for i, size in enumerate(page_sizes) if size < capacity_per_page * 0.2]
            if small_pages:
                self.logger.warning(f"Small pages detected (< 20% capacity): {small_pages[:5]}{'...' if len(small_pages) > 5 else ''}")
        
        return pages

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
            
            # Estimate text capacity and split into pages if needed
            capacity_per_page = self._estimate_text_capacity(config, fonts)
            total_chars = sum(len(segment.text) for segment in segments)
            
            self.logger.debug(f"Text pagination: {total_chars} total chars, "
                             f"{capacity_per_page} chars/page capacity")
            
            if total_chars > capacity_per_page:
                # Split into multiple pages
                pages = self._split_segments_into_pages(segments, capacity_per_page)
                self.logger.info(f"Text split into {len(pages)} pages for rendering")
            else:
                # Single page
                pages = [segments]
                self.logger.debug("Text fits in single page")
            
            # Render each page
            image_paths = []
            
            for page_idx, page_segments in enumerate(pages):
                self.logger.debug(f"Rendering page {page_idx + 1}/{len(pages)} "
                               f"({len(page_segments)} segments, "
                               f"{sum(len(s.text) for s in page_segments)} chars)")
                
                # Calculate text layout for this page
                columns_data = self._layout_text(page_segments, fonts, config)
                
                # Get target dimensions
                img_width, img_height = self._get_effective_dimensions(config)
                
                # Create image with exact dimensions
                # Use white background for better compression
                image = Image.new('RGB', (img_width, img_height), 'white')
                
                # Set DPI information on the image
                dpi_tuple = (config.dpi, config.dpi)
                image.info['dpi'] = dpi_tuple
                
                draw = ImageDraw.Draw(image)
                
                # Render text for this page
                self._render_text_to_image(draw, columns_data, fonts, config)
                
                # Save image with page number
                image_path = output_dir / f"{unique_id}_page_{page_idx + 1}.png"
                image.save(image_path, 'PNG', optimize=True, dpi=dpi_tuple)
                image_paths.append(image_path)
                
                self.logger.debug(f"Generated page {page_idx + 1}: {image_path} ({img_width}x{img_height})")
            
            self.logger.info(f"Rendered {len(pages)} pages total")
            return image_paths
            
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
                
                # Special handling for OCRA font family
                if config.font_name.upper() == "OCRA":
                    return self._load_ocra_font_family(font_size)
                
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
    
    def _load_ocra_font_family(self, font_size: int) -> dict:
        """Load OCRA font family with special handling (no italic variant available)."""
        from PIL import ImageFont
        import os
        
        fonts = {}
        
        # Define path to OCRA font file relative to the package
        try:
            # Get the path to the assets directory
            assets_dir = Path(__file__).parent.parent / "assets"
            ocra_regular_path = assets_dir / "OCRA.ttf"
            
            self.logger.info(f"Loading OCRA font family from assets directory")
            
            # Load regular font (OCRA only has one variant)
            if ocra_regular_path.exists():
                fonts['regular'] = ImageFont.truetype(str(ocra_regular_path), font_size)
                fonts['bold'] = ImageFont.truetype(str(ocra_regular_path), font_size)  # Use regular for bold
                fonts['italic'] = ImageFont.truetype(str(ocra_regular_path), font_size)  # Use regular for italic
                fonts['bold_italic'] = ImageFont.truetype(str(ocra_regular_path), font_size)  # Use regular for bold-italic
                self.logger.info(f"Loaded OCRA regular font: {ocra_regular_path}")
            else:
                self.logger.warning(f"OCRA regular font not found at: {ocra_regular_path}")
                fonts['regular'] = ImageFont.load_default()
                fonts['bold'] = ImageFont.load_default()
                fonts['italic'] = ImageFont.load_default()
                fonts['bold_italic'] = ImageFont.load_default()
            
            self.logger.info("Successfully loaded OCRA font family (using regular font for all styles)")
            self.using_ocra_fonts = True  # Set flag for special bold handling
            return fonts
            
        except Exception as e:
            self.logger.error(f"Failed to load OCRA font family: {e}")
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
            return fonts['bold_italic']
        elif segment.is_header or segment.is_bold:
            # Headers should ONLY be bold, never italic
            return fonts['bold']
        elif segment.is_italic:
            return fonts['italic']
        else:
            return fonts['regular']
    
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
        except AttributeError:
            # Fallback for older PIL versions
            try:
                base_width = temp_draw.textsize(text, font=font)[0]
            except AttributeError:
                # Ultimate fallback - estimate based on font size
                base_width = len(text) * config.font_size * 0.6
        
        # Add extra width for OCRB and OCRA bold overlay effect
        if ((self.using_ocrb_fonts or self.using_ocra_fonts) and segment and 
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
        """Draw text with special effects for OCRB and OCRA bold text."""
        x, y = position
        
        # Check if this should be bold and we're using OCRB or OCRA fonts
        if ((self.using_ocrb_fonts or self.using_ocra_fonts) and segment and 
            (segment.is_bold or segment.is_header) and not segment.is_italic):
            
            # Use improved bold effect for OCRB and OCRA text
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
                font_type = "OCRB" if self.using_ocrb_fonts else "OCRA"
                self.logger.debug(f"{font_type} bold effect failed, using regular text: {e}")
                draw.text((x, y), text, font=font, fill='black')
        else:
            # Regular text drawing
            draw.text((x, y), text, font=font, fill='black')
