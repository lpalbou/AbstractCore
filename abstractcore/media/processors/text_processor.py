"""
Text processor for various text-based file formats.

This module provides processing capabilities for text files, CSV/TSV data,
Markdown documents, and other text-based formats.
"""

import csv
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from ..base import BaseMediaHandler, MediaProcessingError
from ..types import MediaContent, MediaType, ContentFormat


class TextProcessor(BaseMediaHandler):
    """
    Text processor for various text-based file formats.

    Handles plain text, CSV/TSV files, Markdown, JSON, and other text formats
    with intelligent parsing and structure extraction.
    """

    def __init__(self, **kwargs):
        """
        Initialize the text processor.

        Args:
            **kwargs: Configuration parameters including:
                - encoding: Default text encoding ('utf-8')
                - csv_delimiter: Default CSV delimiter (',')
                - max_rows: Maximum rows to process for tabular data
                - preserve_structure: Whether to preserve document structure
        """
        super().__init__(**kwargs)

        # Text processing configuration
        self.default_encoding = kwargs.get('encoding', 'utf-8')
        self.csv_delimiter = kwargs.get('csv_delimiter', ',')
        self.max_rows = kwargs.get('max_rows', 10000)
        self.preserve_structure = kwargs.get('preserve_structure', True)

        # Set capabilities for text processing
        from ..types import MediaCapabilities
        # TextProcessor can handle any text file through its plain text fallback
        # We list common formats but the processor is not limited to these
        self.capabilities = MediaCapabilities(
            vision_support=False,
            audio_support=False,
            video_support=False,
            document_support=True,
            supported_document_formats=[
                # Core text formats
                'txt', 'md', 'markdown', 'csv', 'tsv',
                'json', 'jsonl', 'xml', 'html', 'htm',
                'yaml', 'yml', 'toml', 'ini', 'cfg', 'conf',
                # Programming languages (common examples)
                'py', 'js', 'java', 'c', 'cpp', 'go', 'rs', 'rb', 'php',
                'r', 'R', 'rmd', 'Rmd', 'sql', 'sh',
                # Notebooks and documentation
                'ipynb', 'qmd', 'rst', 'tex', 'bib',
                # Any other text-based format through fallback processing
            ],
            max_file_size=self.max_file_size
        )

        self.logger.debug(
            f"Initialized TextProcessor with encoding={self.default_encoding}, "
            f"max_rows={self.max_rows}, preserve_structure={self.preserve_structure}"
        )

    def _process_internal(self, file_path: Path, media_type: MediaType, **kwargs) -> MediaContent:
        """
        Process a text-based file and return structured content.

        Args:
            file_path: Path to the text file
            media_type: Detected media type (should be TEXT or DOCUMENT)
            **kwargs: Additional processing parameters:
                - encoding: Text encoding to use
                - format_output: Whether to format output ('raw', 'structured', 'summary')
                - extract_metadata: Whether to extract document metadata

        Returns:
            MediaContent with processed text content

        Raises:
            MediaProcessingError: If text processing fails
        """
        if media_type not in [MediaType.TEXT, MediaType.DOCUMENT]:
            raise MediaProcessingError(f"TextProcessor only handles text/document types, got {media_type}")

        try:
            # Override defaults with kwargs
            encoding = kwargs.get('encoding', self.default_encoding)
            format_output = kwargs.get('format_output', 'structured')
            extract_metadata = kwargs.get('extract_metadata', True)

            # Determine processing method based on file extension
            extension = file_path.suffix.lower().lstrip('.')

            if extension in ['csv', 'tsv']:
                content, metadata = self._process_tabular_file(file_path, extension, encoding, **kwargs)
            elif extension == 'json':
                content, metadata = self._process_json_file(file_path, encoding, **kwargs)
            elif extension in ['xml', 'html', 'htm']:
                content, metadata = self._process_markup_file(file_path, extension, encoding, **kwargs)
            elif extension == 'md':
                content, metadata = self._process_markdown_file(file_path, encoding, **kwargs)
            else:
                # Plain text processing
                content, metadata = self._process_plain_text(file_path, encoding, **kwargs)

            # Apply output formatting
            if format_output == 'structured':
                content = self._apply_structured_formatting(content, extension, metadata)
            elif format_output == 'summary':
                content = self._generate_content_summary(content, extension, metadata)
            # 'raw' format uses content as-is

            # Determine appropriate MIME type
            mime_type = self._get_mime_type_for_extension(extension)

            return self._create_media_content(
                content=content,
                file_path=file_path,
                media_type=media_type,
                content_format=ContentFormat.TEXT,
                mime_type=mime_type,
                format=extension,
                **metadata
            )

        except Exception as e:
            raise MediaProcessingError(f"Failed to process text file {file_path}: {str(e)}") from e

    def _process_tabular_file(self, file_path: Path, extension: str, encoding: str, **kwargs) -> tuple[str, Dict[str, Any]]:
        """
        Process CSV/TSV files with intelligent structure detection.

        Args:
            file_path: Path to the tabular file
            extension: File extension ('csv' or 'tsv')
            encoding: Text encoding
            **kwargs: Additional parameters

        Returns:
            Tuple of (processed_content, metadata)
        """
        delimiter = '\t' if extension == 'tsv' else ','
        delimiter = kwargs.get('delimiter', delimiter)

        try:
            if PANDAS_AVAILABLE:
                # Use pandas for robust CSV processing
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    nrows=self.max_rows,
                    on_bad_lines='skip'
                )

                # Generate structured content
                content_parts = []
                content_parts.append(f"# {file_path.name}")
                content_parts.append(f"Tabular data with {len(df)} rows and {len(df.columns)} columns\n")

                # Column information
                content_parts.append("## Columns:")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    null_count = df[col].isnull().sum()
                    content_parts.append(f"- {col} ({dtype}, {null_count} null values)")

                content_parts.append("\n## Sample Data:")
                content_parts.append(df.head(10).to_string(index=False))

                if len(df) > 10:
                    content_parts.append(f"\n... and {len(df) - 10} more rows")

                content = "\n".join(content_parts)

                metadata = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': df.columns.tolist(),
                    'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'delimiter': delimiter,
                    'has_header': True,
                    'null_values': df.isnull().sum().to_dict()
                }

            else:
                # Fallback to basic CSV processing
                with open(file_path, 'r', encoding=encoding) as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    rows = list(reader)

                if not rows:
                    content = f"Empty {extension.upper()} file"
                    metadata = {'row_count': 0, 'column_count': 0}
                else:
                    # Assume first row is header
                    header = rows[0]
                    data_rows = rows[1:self.max_rows + 1]

                    content_parts = []
                    content_parts.append(f"# {file_path.name}")
                    content_parts.append(f"Tabular data with {len(data_rows)} rows and {len(header)} columns\n")

                    content_parts.append("## Columns:")
                    for col in header:
                        content_parts.append(f"- {col}")

                    content_parts.append("\n## Sample Data:")
                    for i, row in enumerate(data_rows[:10]):
                        content_parts.append(f"Row {i+1}: {', '.join(row)}")

                    if len(data_rows) > 10:
                        content_parts.append(f"... and {len(data_rows) - 10} more rows")

                    content = "\n".join(content_parts)

                    metadata = {
                        'row_count': len(data_rows),
                        'column_count': len(header),
                        'columns': header,
                        'delimiter': delimiter,
                        'has_header': True
                    }

            return content, metadata

        except Exception as e:
            # Fallback to plain text if CSV parsing fails
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            metadata = {
                'processing_error': str(e),
                'fallback_to_plain_text': True,
                'delimiter': delimiter
            }

            return content, metadata

    def _process_json_file(self, file_path: Path, encoding: str, **kwargs) -> tuple[str, Dict[str, Any]]:
        """
        Process JSON files with structure analysis.

        Args:
            file_path: Path to the JSON file
            encoding: Text encoding
            **kwargs: Additional parameters

        Returns:
            Tuple of (processed_content, metadata)
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)

            # Generate structured content
            content_parts = []
            content_parts.append(f"# {file_path.name}")

            if isinstance(data, dict):
                content_parts.append(f"JSON object with {len(data)} keys\n")
                content_parts.append("## Structure:")
                content_parts.append(json.dumps(data, indent=2, ensure_ascii=False))
            elif isinstance(data, list):
                content_parts.append(f"JSON array with {len(data)} items\n")
                content_parts.append("## Sample items:")
                for i, item in enumerate(data[:5]):
                    content_parts.append(f"Item {i+1}: {json.dumps(item, ensure_ascii=False)}")
                if len(data) > 5:
                    content_parts.append(f"... and {len(data) - 5} more items")
            else:
                content_parts.append("JSON primitive value:")
                content_parts.append(json.dumps(data, indent=2, ensure_ascii=False))

            content = "\n".join(content_parts)

            metadata = {
                'json_type': type(data).__name__,
                'size': len(data) if isinstance(data, (list, dict)) else 1,
                'keys': list(data.keys()) if isinstance(data, dict) else None
            }

            return content, metadata

        except json.JSONDecodeError as e:
            # If JSON is invalid, treat as plain text
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            metadata = {
                'json_error': str(e),
                'fallback_to_plain_text': True
            }

            return content, metadata

    def _process_markup_file(self, file_path: Path, extension: str, encoding: str, **kwargs) -> tuple[str, Dict[str, Any]]:
        """
        Process markup files (XML, HTML) with basic structure extraction.

        Args:
            file_path: Path to the markup file
            extension: File extension
            encoding: Text encoding
            **kwargs: Additional parameters

        Returns:
            Tuple of (processed_content, metadata)
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Basic structure analysis
            lines = content.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]

            # Count basic markup elements
            tag_count = content.count('<')

            metadata = {
                'markup_type': extension,
                'line_count': len(lines),
                'non_empty_lines': len(non_empty_lines),
                'tag_count': tag_count,
                'character_count': len(content)
            }

            # For HTML, try to extract title
            if extension in ['html', 'htm']:
                import re
                title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
                if title_match:
                    metadata['title'] = title_match.group(1).strip()

            return content, metadata

        except Exception as e:
            metadata = {
                'processing_error': str(e),
                'markup_type': extension
            }
            return "", metadata

    def _process_markdown_file(self, file_path: Path, encoding: str, **kwargs) -> tuple[str, Dict[str, Any]]:
        """
        Process Markdown files with structure analysis.

        Args:
            file_path: Path to the Markdown file
            encoding: Text encoding
            **kwargs: Additional parameters

        Returns:
            Tuple of (processed_content, metadata)
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Basic Markdown structure analysis
            lines = content.split('\n')

            # Count different elements
            headers = [line for line in lines if line.strip().startswith('#')]
            code_blocks = content.count('```')
            links = content.count('[')
            images = content.count('![')

            metadata = {
                'line_count': len(lines),
                'header_count': len(headers),
                'code_block_count': code_blocks // 2,  # Pairs of ```
                'link_count': links,
                'image_count': images,
                'character_count': len(content),
                'word_count': len(content.split())
            }

            # Extract headers for structure
            if headers:
                metadata['headers'] = headers[:10]  # First 10 headers

            return content, metadata

        except Exception as e:
            metadata = {
                'processing_error': str(e)
            }
            return "", metadata

    def _process_plain_text(self, file_path: Path, encoding: str, **kwargs) -> tuple[str, Dict[str, Any]]:
        """
        Process plain text files with basic analysis.

        Args:
            file_path: Path to the text file
            encoding: Text encoding
            **kwargs: Additional parameters

        Returns:
            Tuple of (processed_content, metadata)
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

            # Basic text analysis
            lines = content.split('\n')
            words = content.split()

            metadata = {
                'line_count': len(lines),
                'word_count': len(words),
                'character_count': len(content),
                'non_empty_lines': len([line for line in lines if line.strip()]),
                'encoding_used': encoding
            }

            return content, metadata

        except UnicodeDecodeError:
            # Try different encodings
            for alt_encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=alt_encoding) as f:
                        content = f.read()

                    metadata = {
                        'encoding_used': alt_encoding,
                        'original_encoding_failed': encoding,
                        'character_count': len(content)
                    }

                    return content, metadata
                except:
                    continue

            # If all encodings fail, read as binary and decode with errors='replace'
            with open(file_path, 'rb') as f:
                raw_content = f.read()

            content = raw_content.decode('utf-8', errors='replace')
            metadata = {
                'encoding_used': 'utf-8-with-errors',
                'binary_fallback': True,
                'character_count': len(content)
            }

            return content, metadata

    def _apply_structured_formatting(self, content: str, extension: str, metadata: Dict[str, Any]) -> str:
        """Apply structured formatting to content based on file type."""
        if extension in ['csv', 'tsv']:
            # Content is already structured for tabular data
            return content
        elif extension == 'json':
            # Content is already structured for JSON
            return content
        elif extension == 'md':
            # Markdown is already structured
            return content
        else:
            # Add basic structure to plain text
            structured_parts = [f"# {metadata.get('file_name', 'Text Content')}"]

            if 'word_count' in metadata:
                structured_parts.append(f"Document Statistics: {metadata['word_count']} words, {metadata['line_count']} lines\n")

            structured_parts.append("## Content:")
            structured_parts.append(content)

            return "\n".join(structured_parts)

    def _generate_content_summary(self, content: str, extension: str, metadata: Dict[str, Any]) -> str:
        """Generate a summary of the content."""
        summary_parts = [f"# Summary of {metadata.get('file_name', 'file')}"]

        if extension in ['csv', 'tsv']:
            summary_parts.append(f"Tabular data with {metadata.get('row_count', 0)} rows and {metadata.get('column_count', 0)} columns")
            if 'columns' in metadata:
                summary_parts.append(f"Columns: {', '.join(metadata['columns'][:5])}")
        elif extension == 'json':
            summary_parts.append(f"JSON {metadata.get('json_type', 'data')} with {metadata.get('size', 0)} items")
        elif extension == 'md':
            summary_parts.append(f"Markdown document with {metadata.get('header_count', 0)} headers and {metadata.get('word_count', 0)} words")
        else:
            summary_parts.append(f"Text document with {metadata.get('word_count', 0)} words and {metadata.get('line_count', 0)} lines")

            # Add content preview
            preview = content[:500] + "..." if len(content) > 500 else content
            summary_parts.append(f"\nContent preview:\n{preview}")

        return "\n".join(summary_parts)

    def _get_mime_type_for_extension(self, extension: str) -> str:
        """Get MIME type for file extension."""
        mime_map = {
            'txt': 'text/plain',
            'md': 'text/markdown',
            'csv': 'text/csv',
            'tsv': 'text/tab-separated-values',
            'json': 'application/json',
            'xml': 'application/xml',
            'html': 'text/html',
            'htm': 'text/html'
        }
        return mime_map.get(extension, 'text/plain')

    def get_text_preview(self, file_path: Union[str, Path], max_chars: int = 1000) -> str:
        """
        Get a preview of text content without full processing.

        Args:
            file_path: Path to the text file
            max_chars: Maximum characters to preview

        Returns:
            Text preview
        """
        file_path = Path(file_path)

        try:
            with open(file_path, 'r', encoding=self.default_encoding) as f:
                content = f.read(max_chars)
                if len(content) == max_chars:
                    content += "..."
                return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get information about the text processor capabilities.

        Returns:
            Dictionary with processor information
        """
        return {
            'processor_type': 'TextProcessor',
            'supported_formats': self.capabilities.supported_document_formats,
            'supports_any_text_file': True,  # Through plain text fallback
            'capabilities': {
                'default_encoding': self.default_encoding,
                'csv_delimiter': self.csv_delimiter,
                'max_rows': self.max_rows,
                'preserve_structure': self.preserve_structure,
                'pandas_integration': PANDAS_AVAILABLE,
                'structured_formatting': True,
                'metadata_extraction': True,
                'plain_text_fallback': True  # Can handle any text file
            },
            'dependencies': {
                'pandas': PANDAS_AVAILABLE
            }
        }