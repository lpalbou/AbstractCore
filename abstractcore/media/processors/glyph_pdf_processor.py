"""
Glyph-optimized PDF processor that preserves mathematical notation and table formatting.

This processor extracts PDF content while maintaining compact mathematical expressions
and tabular data formatting for optimal visual compression.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import re

try:
    import pymupdf as fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat


class GlyphPDFProcessor(BaseMediaHandler):
    """
    Glyph-optimized PDF processor that preserves mathematical notation and tables.
    
    Designed specifically for visual compression where maintaining compact
    mathematical expressions and table layouts is crucial for compression ratio.
    """
    
    def __init__(self, **kwargs):
        """Initialize the Glyph PDF processor."""
        super().__init__(**kwargs)
        
        if not PYMUPDF_AVAILABLE:
            raise MediaProcessingError("PyMuPDF is required for GlyphPDFProcessor")
        
        # Glyph-specific settings
        self.preserve_math_notation = kwargs.get('preserve_math_notation', True)
        self.preserve_table_layout = kwargs.get('preserve_table_layout', True)
        self.compact_whitespace = kwargs.get('compact_whitespace', True)
        
        self.logger.debug("GlyphPDFProcessor initialized for visual compression")
    
    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """Process PDF with Glyph optimization."""
        if media_type != MediaType.DOCUMENT:
            raise MediaProcessingError(f"GlyphPDFProcessor only handles documents, got {media_type}")
        
        try:
            # Extract content with Glyph optimizations
            content, metadata = self._extract_glyph_optimized_content(file_path)
            
            return self._create_media_content(
                content=content,
                file_path=file_path,
                media_type=media_type,
                content_format=ContentFormat.TEXT,
                mime_type="application/pdf",
                **metadata
            )
            
        except Exception as e:
            raise MediaProcessingError(f"Failed to process PDF for Glyph: {str(e)}") from e
    
    def _extract_glyph_optimized_content(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract PDF content optimized for Glyph visual compression."""
        doc = fitz.open(str(file_path))
        
        content_parts = []
        total_chars = 0
        page_count = 0
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_count += 1
                
                # Get text blocks with position information
                blocks = page.get_text("dict")
                
                # Process blocks to preserve mathematical notation and tables
                page_content = self._process_page_blocks(blocks, page_num)
                
                if page_content.strip():
                    content_parts.append(page_content)
                    total_chars += len(page_content)
            
            # Combine all pages
            full_content = "\n\n".join(content_parts)
            
            # Apply Glyph-specific optimizations
            optimized_content = self._apply_glyph_optimizations(full_content)
            
            metadata = {
                'page_count': page_count,
                'character_count': len(optimized_content),
                'processing_method': 'glyph_optimized',
                'math_notation_preserved': self.preserve_math_notation,
                'table_layout_preserved': self.preserve_table_layout
            }
            
            return optimized_content, metadata
            
        finally:
            doc.close()
    
    def _process_page_blocks(self, blocks: Dict, page_num: int) -> str:
        """Process page blocks while preserving mathematical and tabular content."""
        page_lines = []
        
        for block in blocks.get("blocks", []):
            if "lines" not in block:
                continue
            
            # Check if this block looks like a table
            if self._is_table_block(block):
                table_content = self._extract_table_content(block)
                if table_content:
                    page_lines.append(table_content)
            else:
                # Process as regular text, preserving math notation
                block_text = self._extract_block_text(block)
                if block_text.strip():
                    page_lines.append(block_text)
        
        return "\n".join(page_lines)
    
    def _is_table_block(self, block: Dict) -> bool:
        """Detect if a block represents tabular data."""
        lines = block.get("lines", [])
        if len(lines) < 2:
            return False
        
        # Look for patterns indicating tabular structure
        # - Multiple columns of aligned text
        # - Consistent spacing patterns
        # - Numeric data in columns
        
        x_positions = []
        for line in lines:
            for span in line.get("spans", []):
                bbox = span.get("bbox", [])
                if len(bbox) >= 4:
                    x_positions.append(bbox[0])  # Left x coordinate
        
        if len(x_positions) < 4:
            return False
        
        # Check for multiple distinct column positions
        unique_x = sorted(set(round(x, 1) for x in x_positions))
        return len(unique_x) >= 3  # At least 3 columns suggests a table
    
    def _extract_table_content(self, block: Dict) -> str:
        """Extract table content in a compact format."""
        lines = block.get("lines", [])
        table_rows = []
        
        for line in lines:
            row_parts = []
            spans = sorted(line.get("spans", []), key=lambda s: s.get("bbox", [0])[0])
            
            for span in spans:
                text = span.get("text", "").strip()
                if text:
                    row_parts.append(text)
            
            if row_parts:
                # Use compact table format (pipe-separated)
                table_rows.append(" | ".join(row_parts))
        
        if table_rows:
            return "\n".join(table_rows)
        return ""
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract text from a block, preserving mathematical notation."""
        lines = block.get("lines", [])
        block_lines = []
        
        for line in lines:
            line_text = ""
            for span in line.get("spans", []):
                text = span.get("text", "")
                
                # Preserve mathematical symbols and notation
                if self.preserve_math_notation:
                    text = self._preserve_math_symbols(text)
                
                line_text += text
            
            if line_text.strip():
                block_lines.append(line_text.strip())
        
        return "\n".join(block_lines)
    
    def _preserve_math_symbols(self, text: str) -> str:
        """Preserve mathematical symbols and compact notation."""
        # Don't expand mathematical symbols - keep them as-is
        # This prevents "α" from becoming "alpha", "∑" from becoming "sum", etc.
        
        # Remove excessive whitespace around mathematical operators
        text = re.sub(r'\s*([+\-×÷=<>≤≥≠∈∉∪∩∀∃∑∏∫])\s*', r'\1', text)
        
        # Preserve subscripts and superscripts in compact form
        # Keep Unicode mathematical symbols intact
        
        return text
    
    def _apply_glyph_optimizations(self, content: str) -> str:
        """Apply final optimizations for Glyph visual compression."""
        if not self.compact_whitespace:
            return content
        
        # Remove excessive blank lines (keep max 1)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in content.split('\n')]
        
        # Remove excessive spaces (keep max 2 consecutive spaces)
        optimized_lines = []
        for line in lines:
            line = re.sub(r'  +', '  ', line)  # Max 2 spaces
            optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
