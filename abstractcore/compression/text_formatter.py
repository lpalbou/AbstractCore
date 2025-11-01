"""
Text formatter for Glyph compression with markdown-like formatting support.

This module provides text preprocessing to improve readability in compressed images
by handling newlines, markdown formatting, and headers appropriately.

The formatter converts markdown-like syntax to ReportLab-compatible rich text
with proper bold and italic font rendering.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ..utils.structured_logging import get_logger


@dataclass
class TextSegment:
    """Represents a segment of text with formatting information."""
    text: str
    is_bold: bool = False
    is_italic: bool = False
    is_header: bool = False
    header_level: int = 0  # 1, 2, 3 for H1, H2, H3


@dataclass
class FormattingConfig:
    """Configuration for text formatting options."""
    
    # Newline handling - UPDATED RULES
    single_newline_to_space: bool = True        # Single \n becomes 1 space
    double_newline_to_two_spaces: bool = True   # \n\n becomes 2 spaces
    triple_newline_to_break: bool = True        # \n\n\n+ becomes single line break
    
    # Markdown formatting
    bold_formatting: bool = True                # **text** → BOLD TEXT
    italic_formatting: bool = True              # *text* → italic text
    
    # Header formatting
    header_formatting: bool = True              # Convert # ## ### to A) a) 1)
    header_bold_caps: bool = True              # Headers in BOLD AND ALL CAPS
    
    # Header numbering styles - HIERARCHICAL
    h1_style: str = "A"  # A. B. C. ...
    h2_style: str = "A.1"  # A.1. A.2. A.3. ...
    h3_style: str = "A.1.a"  # A.1.a. A.1.b. A.1.c. ...
    h4_style: str = "A.1.a.i"  # A.1.a.i. A.1.a.ii. A.1.a.iii. ...
    h5_style: str = "A.1.a.i.1"  # A.1.a.i.1. A.1.a.i.2. ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            'single_newline_to_space': self.single_newline_to_space,
            'double_newline_to_two_spaces': self.double_newline_to_two_spaces,
            'triple_newline_to_break': self.triple_newline_to_break,
            'bold_formatting': self.bold_formatting,
            'italic_formatting': self.italic_formatting,
            'header_formatting': self.header_formatting,
            'header_bold_caps': self.header_bold_caps,
            'h1_style': self.h1_style,
            'h2_style': self.h2_style,
            'h3_style': self.h3_style,
            'h4_style': self.h4_style
        }


class TextFormatter:
    """
    Text formatter for improving readability in Glyph-compressed images.
    
    Handles markdown-like formatting, newline processing, and header conversion
    to make text more readable when rendered as images.
    """
    
    def __init__(self, config: Optional[FormattingConfig] = None):
        """
        Initialize text formatter.
        
        Args:
            config: Formatting configuration
        """
        self.config = config or FormattingConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Header counters for numbering
        self._header_counters = {
            'h1': 0,
            'h2': 0, 
            'h3': 0,
            'h4': 0,
            'h5': 0
        }
        
        self.logger.debug("TextFormatter initialized", config=self.config.to_dict())
    
    def format_text(self, text: str) -> List[TextSegment]:
        """
        Apply formatting transformations to text and return structured segments.
        
        Args:
            text: Raw text to format
            
        Returns:
            List of TextSegment objects with formatting information
        """
        import time
        start_time = time.time()
        
        if not text:
            return [TextSegment(text="")]
            
        # Better header detection - check for any line starting with #
        has_headers = any(line.strip().startswith('#') for line in text.split('\n')[:100])  # Check first 100 lines for performance
        has_bold_markers = '**' in text
        has_italic_markers = '*' in text and '**' not in text
        
        self.logger.debug("Starting text formatting", 
                         original_length=len(text),
                         has_newlines='\n' in text,
                         has_bold_markers=has_bold_markers,
                         has_italic_markers=has_italic_markers,
                         has_headers=has_headers)
        
        # Performance optimization: For large files with no formatting, skip complex processing
        if len(text) > 50000 and not has_headers and not has_bold_markers and not has_italic_markers:
            self.logger.debug("Large file with no formatting detected - using fast path")
            # Just process newlines and return as single segment
            processed_text = self._process_newlines(text)
            return [TextSegment(text=processed_text)]
        
        # Reset header counters for each new text
        self._reset_counters()
        
        # Step 1: Parse into segments with formatting (before newline processing)
        step1_start = time.time()
        self.logger.debug("Step 1: Starting _parse_formatted_text")
        segments = self._parse_formatted_text(text)
        step1_time = time.time() - step1_start
        self.logger.debug(f"Step 1: _parse_formatted_text completed in {step1_time:.3f}s, segments={len(segments)}")
        
        # Step 2: Apply newline processing to the final segments
        step2_start = time.time()
        self.logger.debug("Step 2: Starting _apply_newline_processing_to_segments")
        segments = self._apply_newline_processing_to_segments(segments)
        step2_time = time.time() - step2_start
        self.logger.debug(f"Step 2: _apply_newline_processing_to_segments completed in {step2_time:.3f}s")
        
        total_time = time.time() - start_time
        self.logger.debug("Text formatting completed",
                         original_length=len(text),
                         segments_count=len(segments),
                         total_formatted_length=sum(len(s.text) for s in segments),
                         total_time_seconds=f"{total_time:.3f}")
        
        return segments
    
    def format_text_to_string(self, text: str) -> str:
        """
        Apply formatting and return as plain text (for backward compatibility).
        
        Args:
            text: Raw text to format
            
        Returns:
            Formatted text as plain string
        """
        segments = self.format_text(text)
        return ''.join(segment.text for segment in segments)
    
    def _reset_counters(self):
        """Reset header counters for new text."""
        self._header_counters = {'h1': 0, 'h2': 0, 'h3': 0, 'h4': 0, 'h5': 0}
    
    def _parse_formatted_text(self, text: str) -> List[TextSegment]:
        """
        Parse text with markdown formatting into structured segments.
        
        Args:
            text: Text with markdown formatting
            
        Returns:
            List of TextSegment objects
        """
        import time
        start_time = time.time()
        
        segments = []
        
        # Split text by lines first to handle headers
        lines = text.split('\n')
        total_lines = len(lines)
        
        self.logger.debug(f"_parse_formatted_text: Processing {total_lines} lines")
        
        for line_idx, line in enumerate(lines):
            # Progress logging every 1000 lines for large files
            if line_idx > 0 and line_idx % 1000 == 0:
                elapsed = time.time() - start_time
                self.logger.debug(f"_parse_formatted_text: Progress {line_idx}/{total_lines} lines ({line_idx/total_lines*100:.1f}%) in {elapsed:.2f}s")
            if line.strip():
                # Process headers first
                if self.config.header_formatting and line.strip().startswith('#'):
                    header_segment = self._process_header_line_to_segment(line)
                    if header_segment:
                        segments.append(header_segment)
                        # NEVER add line break after header (rule #9)
                        continue
                
                # Process inline formatting (bold/italic) for non-header lines
                line_segments = self._parse_inline_formatting(line)
                segments.extend(line_segments)
            else:
                # Empty line
                segments.append(TextSegment(text=""))
            
            # Add line break after each line (except the last one)
            if line_idx < len(lines) - 1 and not (self.config.header_formatting and line.strip().startswith('#')):
                segments.append(TextSegment(text="\n"))
        
        return segments
    
    def _apply_newline_processing_to_segments(self, segments: List[TextSegment]) -> List[TextSegment]:
        """
        Apply newline processing rules to segments.
        
        Args:
            segments: List of TextSegment objects
            
        Returns:
            List of TextSegment objects with newline processing applied
        """
        processed_segments = []
        
        for segment in segments:
            if segment.text == "\n":
                # Single newline becomes 1 space
                if self.config.single_newline_to_space:
                    processed_segments.append(TextSegment(text=" "))
                else:
                    processed_segments.append(segment)
            else:
                # Apply newline processing to text content
                processed_text = self._process_newlines(segment.text)
                processed_segments.append(TextSegment(
                    text=processed_text,
                    is_bold=segment.is_bold,
                    is_italic=segment.is_italic,
                    is_header=segment.is_header,
                    header_level=segment.header_level
                ))
        
        return processed_segments
    
    def _process_newlines(self, text: str) -> str:
        """
        Process newlines within text content according to updated rules:
        1) Single \n → 1 space
        2) Double \n\n → 2 spaces
        3) Triple+ \n\n\n → 1 linebreak
        
        Also handles literal \n sequences (backslash-n) in addition to actual newlines.
        """
        import re
        
        # First, convert literal \n sequences to actual newlines
        text = text.replace('\\n', '\n')
        
        # Process in order: triple+, double, single
        # Use placeholder to avoid conflicts
        
        # 1. Triple or more newlines → single line break (use placeholder first)
        if self.config.triple_newline_to_break:
            text = re.sub(r'\n{3,}', '___LINEBREAK___', text)
        
        # 2. Double newlines → 2 spaces
        if self.config.double_newline_to_two_spaces:
            text = re.sub(r'\n\n', '  ', text)
        
        # 3. Single newlines → 1 space
        if self.config.single_newline_to_space:
            text = re.sub(r'\n', ' ', text)
        
        # 4. Replace placeholder with actual line break
        text = text.replace('___LINEBREAK___', '\n')
        
        return text
    
    def _parse_inline_formatting(self, text: str) -> List[TextSegment]:
        """
        Parse inline formatting (bold, italic) in a line of text.
        
        Args:
            text: Line of text with potential formatting
            
        Returns:
            List of TextSegment objects for this line
        """
        segments = []
        
        if not text:
            return segments
        
        # Performance optimization: Skip inline parsing if no formatting markers
        if '**' not in text and '*' not in text:
            return [TextSegment(text=text)]
            
        # Process text sequentially to handle formatting correctly
        i = 0
        while i < len(text):
            # Check for bold formatting **text**
            if i < len(text) - 3 and text[i:i+2] == '**':
                # Find the closing **
                end_pos = text.find('**', i + 2)
                if end_pos != -1 and end_pos > i + 2:  # Must have content between
                    # Found bold text
                    bold_content = text[i+2:end_pos]
                    if bold_content and self.config.bold_formatting:
                        segments.append(TextSegment(text=bold_content, is_bold=True))
                    i = end_pos + 2
                    continue
            
            # Check for italic formatting *text* (but not part of **)
            if (i < len(text) - 2 and text[i] == '*' and 
                (i == 0 or text[i-1:i+1] != '**') and  # Not part of **
                (i >= len(text) - 2 or text[i:i+2] != '**')):  # Not start of **
                
                # Find the closing *
                end_pos = i + 1
                while end_pos < len(text) and text[end_pos] != '*':
                    end_pos += 1
                
                if end_pos < len(text) and end_pos > i + 1:  # Must have content between
                    # Make sure this isn't part of **
                    if end_pos >= len(text) - 1 or text[end_pos:end_pos+2] != '**':
                        italic_content = text[i+1:end_pos]
                        if italic_content and self.config.italic_formatting:
                            segments.append(TextSegment(text=italic_content, is_italic=True))
                        i = end_pos + 1
                        continue
            
            # Regular character - collect until next formatting marker or advance by 1
            start_pos = i
            while i < len(text) and text[i] != '*':
                i += 1
            
            if i > start_pos:
                plain_text = text[start_pos:i]
                if plain_text:  # Only add non-empty segments
                    segments.append(TextSegment(text=plain_text))
            else:
                # If we didn't advance, we hit a * that didn't match formatting
                # Add the single character and advance to prevent infinite loop
                segments.append(TextSegment(text=text[i]))
                i += 1
        
        return segments
    
    def _process_header_line_to_segment(self, line: str) -> Optional[TextSegment]:
        """
        Process a header line and return a TextSegment.
        
        Args:
            line: Line starting with # ## or ###
            
        Returns:
            TextSegment with header formatting, or None if not a valid header
        """
        stripped = line.strip()
        
        if stripped.startswith('#####'):
            # H5 header
            content = stripped[5:].strip()
            if content:
                self._header_counters['h5'] += 1
                number = self._get_header_number('h5', self._header_counters['h5'])
                # Process inline formatting in header content
                clean_content = self._strip_markdown_formatting(content)
                formatted_content = f"{number} {clean_content.upper() if self.config.header_bold_caps else clean_content}"
                return TextSegment(text=formatted_content, is_bold=True, is_header=True, header_level=5)
                
        elif stripped.startswith('####'):
            # H4 header
            content = stripped[4:].strip()
            if content:
                self._header_counters['h4'] += 1
                # Reset h5 counter when we encounter h4
                self._header_counters['h5'] = 0
                number = self._get_header_number('h4', self._header_counters['h4'])
                # Process inline formatting in header content
                clean_content = self._strip_markdown_formatting(content)
                formatted_content = f"{number} {clean_content.upper() if self.config.header_bold_caps else clean_content}"
                return TextSegment(text=formatted_content, is_bold=True, is_header=True, header_level=4)
                
        elif stripped.startswith('###'):
            # H3 header
            content = stripped[3:].strip()
            if content:
                self._header_counters['h3'] += 1
                # Reset h4 and h5 counters when we encounter h3
                self._header_counters['h4'] = 0
                self._header_counters['h5'] = 0
                number = self._get_header_number('h3', self._header_counters['h3'])
                # Process inline formatting in header content
                clean_content = self._strip_markdown_formatting(content)
                formatted_content = f"{number} {clean_content.upper() if self.config.header_bold_caps else clean_content}"
                return TextSegment(text=formatted_content, is_bold=True, is_header=True, header_level=3)
                
        elif stripped.startswith('##'):
            # H2 header
            content = stripped[2:].strip()
            if content:
                self._header_counters['h2'] += 1
                # Reset h3, h4, and h5 counters when we encounter h2
                self._header_counters['h3'] = 0
                self._header_counters['h4'] = 0
                self._header_counters['h5'] = 0
                number = self._get_header_number('h2', self._header_counters['h2'])
                # Process inline formatting in header content
                clean_content = self._strip_markdown_formatting(content)
                formatted_content = f"{number} {clean_content.upper() if self.config.header_bold_caps else clean_content}"
                return TextSegment(text=formatted_content, is_bold=True, is_header=True, header_level=2)
                
        elif stripped.startswith('#'):
            # H1 header - NO NUMBERING according to new rules
            content = stripped[1:].strip()
            if content:
                self._header_counters['h1'] += 1
                # Reset h2, h3, h4, and h5 counters when we encounter h1
                self._header_counters['h2'] = 0
                self._header_counters['h3'] = 0
                self._header_counters['h4'] = 0
                self._header_counters['h5'] = 0
                # Process inline formatting in header content
                clean_content = self._strip_markdown_formatting(content)
                formatted_content = f"{clean_content.upper() if self.config.header_bold_caps else clean_content}"
                return TextSegment(text=formatted_content, is_bold=True, is_header=True, header_level=1)
        
        return None
    
    def _strip_markdown_formatting(self, text: str) -> str:
        """Strip markdown formatting markers from text."""
        # Remove **bold** markers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # Remove *italic* markers
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)
        return text
    
    
    def _process_bold(self, text: str) -> str:
        """
        Process bold markdown formatting (**text** → BOLD TEXT).
        
        Note: In text rendering, we'll use uppercase to simulate bold
        since ReportLab bold fonts may not be available.
        """
        def bold_replacer(match):
            content = match.group(1)
            # Convert to uppercase to simulate bold in plain text
            return content.upper()
        
        # Match **text** patterns (non-greedy)
        return re.sub(r'\*\*(.*?)\*\*', bold_replacer, text)
    
    def _process_italic(self, text: str) -> str:
        """
        Process italic markdown formatting (*text* → italic text).
        
        Note: We'll keep italic text as-is since true italic rendering
        would require font changes in ReportLab.
        """
        def italic_replacer(match):
            content = match.group(1)
            # For now, just remove the markers and keep text as-is
            # In future, could add special markers for ReportLab italic rendering
            return content
        
        # Match *text* patterns (but not **text**) - single asterisks only
        return re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', italic_replacer, text)
    
    def _process_headers(self, text: str) -> str:
        """
        Process markdown headers and convert to numbered format.
        
        # Header → A) HEADER
        ## Header → a) HEADER  
        ### Header → 1) HEADER
        """
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            processed_line = self._process_header_line(line)
            processed_lines.append(processed_line)
        
        return '\n'.join(processed_lines)
    
    def _process_header_line(self, line: str) -> str:
        """Process a single line for header formatting."""
        stripped = line.strip()
        
        # Check for headers
        if stripped.startswith('###'):
            # H3 header
            content = stripped[3:].strip()
            if content:
                self._header_counters['h3'] += 1
                number = self._get_header_number('h3', self._header_counters['h3'])
                formatted_content = content.upper() if self.config.header_bold_caps else content
                return f"{number}) {formatted_content}"
                
        elif stripped.startswith('##'):
            # H2 header
            content = stripped[2:].strip()
            if content:
                self._header_counters['h2'] += 1
                # Reset h3 counter when we encounter h2
                self._header_counters['h3'] = 0
                number = self._get_header_number('h2', self._header_counters['h2'])
                formatted_content = content.upper() if self.config.header_bold_caps else content
                return f"{number}) {formatted_content}"
                
        elif stripped.startswith('#'):
            # H1 header
            content = stripped[1:].strip()
            if content:
                self._header_counters['h1'] += 1
                # Reset h2 and h3 counters when we encounter h1
                self._header_counters['h2'] = 0
                self._header_counters['h3'] = 0
                number = self._get_header_number('h1', self._header_counters['h1'])
                formatted_content = content.upper() if self.config.header_bold_caps else content
                return f"{number}) {formatted_content}"
        
        return line
    
    def _get_header_number(self, level: str, count: int) -> str:
        """
        Get the appropriate header number/letter based on level and count.
        
        NEW HIERARCHICAL FORMAT:
        H1: No numbering
        H2: A. B. C. ...
        H3: A.1. A.2. A.3. ...
        H4: A.1.a. A.1.b. A.1.c. ...
        H5: A.1.a.i. A.1.a.ii. A.1.a.iii. ...
        
        Args:
            level: Header level ('h1', 'h2', 'h3', 'h4', 'h5')
            count: Current count for this level
            
        Returns:
            Formatted number/letter (e.g., 'A.', 'A.1.', 'A.1.a.')
        """
        
        if level == 'h1':
            # H1: No numbering
            return ""
                
        elif level == 'h2':
            # H2: A. B. C. ...
            if count <= 26:
                letter = chr(ord('A') + count - 1)
            else:
                # After Z, use AA, BB, CC, etc.
                letter = chr(ord('A') + ((count - 1) % 26))
                letter = letter * ((count - 1) // 26 + 1)
            return f"{letter}."
                
        elif level == 'h3':
            # H3: A.1. A.2. A.3. ...
            h2_count = self._header_counters['h2']
            if h2_count <= 26:
                h2_letter = chr(ord('A') + h2_count - 1)
            else:
                h2_letter = chr(ord('A') + ((h2_count - 1) % 26))
                h2_letter = h2_letter * ((h2_count - 1) // 26 + 1)
            return f"{h2_letter}.{count}."
                
        elif level == 'h4':
            # H4: A.1.a. A.1.b. A.1.c. ...
            h2_count = self._header_counters['h2']
            h3_count = self._header_counters['h3']
            
            if h2_count <= 26:
                h2_letter = chr(ord('A') + h2_count - 1)
            else:
                h2_letter = chr(ord('A') + ((h2_count - 1) % 26))
                h2_letter = h2_letter * ((h2_count - 1) // 26 + 1)
                
            if count <= 26:
                h4_letter = chr(ord('a') + count - 1)
            else:
                h4_letter = chr(ord('a') + ((count - 1) % 26))
                h4_letter = h4_letter * ((count - 1) // 26 + 1)
                
            return f"{h2_letter}.{h3_count}.{h4_letter}."
        
        elif level == 'h5':
            # H5: A.1.a.i. A.1.a.ii. A.1.a.iii. ...
            h2_count = self._header_counters['h2']
            h3_count = self._header_counters['h3']
            h4_count = self._header_counters['h4']
            
            if h2_count <= 26:
                h2_letter = chr(ord('A') + h2_count - 1)
            else:
                h2_letter = chr(ord('A') + ((h2_count - 1) % 26))
                h2_letter = h2_letter * ((h2_count - 1) // 26 + 1)
                
            if h4_count <= 26:
                h4_letter = chr(ord('a') + h4_count - 1)
            else:
                h4_letter = chr(ord('a') + ((h4_count - 1) % 26))
                h4_letter = h4_letter * ((h4_count - 1) // 26 + 1)
                
            h5_roman = self._int_to_roman_lower(count)
            return f"{h2_letter}.{h3_count}.{h4_letter}.{h5_roman}."
        
        return str(count)
    
    def _int_to_roman_lower(self, num: int) -> str:
        """Convert integer to lowercase Roman numeral."""
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        literals = ['m', 'cm', 'd', 'cd', 'c', 'xc', 'l', 'xl', 'x', 'ix', 'v', 'iv', 'i']
        
        result = ""
        for i in range(len(values)):
            count = num // values[i]
            if count:
                result += literals[i] * count
                num -= values[i] * count
        return result
    
    def get_formatting_summary(self) -> Dict[str, Any]:
        """Get summary of formatting configuration and usage."""
        return {
            'config': self.config.to_dict(),
            'header_counters': self._header_counters.copy(),
            'formatter_version': '1.0'
        }


def create_default_formatter() -> TextFormatter:
    """Create a TextFormatter with default configuration."""
    return TextFormatter(FormattingConfig())


def create_minimal_formatter() -> TextFormatter:
    """Create a TextFormatter with minimal formatting (only newlines)."""
    config = FormattingConfig()
    config.bold_formatting = False
    config.italic_formatting = False
    config.header_formatting = False
    return TextFormatter(config)


def create_headers_only_formatter() -> TextFormatter:
    """Create a TextFormatter that only processes headers."""
    config = FormattingConfig()
    config.bold_formatting = False
    config.italic_formatting = False
    config.consecutive_newlines_to_break = False
    config.single_newline_to_spaces = False
    return TextFormatter(config)
