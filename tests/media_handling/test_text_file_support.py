"""
Test comprehensive text file support for media handling.

This test verifies that the media module can handle various text-based file types
including R scripts, Jupyter notebooks, and other programming/markup languages.
"""

import pytest
from pathlib import Path
import tempfile
import json

from abstractcore.media.auto_handler import AutoMediaHandler
from abstractcore.media.types import detect_media_type, MediaType, is_text_file


class TestTextFileSupport:
    """Test that all text-based file types are supported."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def handler(self):
        """Create AutoMediaHandler instance."""
        return AutoMediaHandler()

    def create_test_file(self, temp_dir: Path, filename: str, content: str) -> Path:
        """Helper to create a test file."""
        filepath = temp_dir / filename
        filepath.write_text(content, encoding='utf-8')
        return filepath

    def test_r_script_detection(self, temp_dir):
        """Test that R scripts are detected as TEXT."""
        r_file = self.create_test_file(
            temp_dir,
            "analysis.R",
            "# R script\ndata <- read.csv('data.csv')\nplot(data$x, data$y)\n"
        )

        media_type = detect_media_type(r_file)
        assert media_type == MediaType.TEXT
        assert is_text_file(r_file)

    def test_r_markdown_detection(self, temp_dir):
        """Test that R Markdown files are detected as TEXT."""
        rmd_file = self.create_test_file(
            temp_dir,
            "report.Rmd",
            "---\ntitle: 'Analysis'\n---\n\n```{r}\nplot(1:10)\n```\n"
        )

        media_type = detect_media_type(rmd_file)
        assert media_type == MediaType.TEXT
        assert is_text_file(rmd_file)

    def test_jupyter_notebook_detection(self, temp_dir):
        """Test that Jupyter notebooks are detected as TEXT."""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["import numpy as np\nprint('Hello')"],
                    "outputs": []
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }

        ipynb_file = temp_dir / "notebook.ipynb"
        ipynb_file.write_text(json.dumps(notebook_content), encoding='utf-8')

        media_type = detect_media_type(ipynb_file)
        assert media_type == MediaType.TEXT
        assert is_text_file(ipynb_file)

    def test_various_programming_languages(self, temp_dir):
        """Test various programming language files."""
        test_files = {
            "script.py": "# Python\nprint('Hello')",
            "app.js": "// JavaScript\nconsole.log('Hello');",
            "Main.java": "// Java\npublic class Main { }",
            "main.go": "// Go\npackage main\n",
            "lib.rs": "// Rust\nfn main() { }",
            "script.rb": "# Ruby\nputs 'Hello'",
            "query.sql": "-- SQL\nSELECT * FROM users;",
            "script.sh": "#!/bin/bash\necho 'Hello'",
            "analysis.jl": "# Julia\nprintln(\"Hello\")",
        }

        for filename, content in test_files.items():
            filepath = self.create_test_file(temp_dir, filename, content)
            media_type = detect_media_type(filepath)
            assert media_type == MediaType.TEXT, f"Failed for {filename}"
            assert is_text_file(filepath), f"Failed is_text_file for {filename}"

    def test_config_and_markup_files(self, temp_dir):
        """Test configuration and markup files."""
        test_files = {
            "config.yaml": "name: test\nvalue: 123",
            "config.toml": "[section]\nkey = 'value'",
            "config.ini": "[section]\nkey=value",
            "README.md": "# Title\n\nContent here",
            "doc.rst": "Title\n=====\n\nContent",
            "style.css": "body { color: red; }",
        }

        for filename, content in test_files.items():
            filepath = self.create_test_file(temp_dir, filename, content)
            media_type = detect_media_type(filepath)
            assert media_type == MediaType.TEXT, f"Failed for {filename}"

    def test_r_script_processing(self, temp_dir, handler):
        """Test that R scripts can be processed."""
        r_file = self.create_test_file(
            temp_dir,
            "analysis.R",
            "# R script for data analysis\ndata <- read.csv('data.csv')\nplot(data$x, data$y)\n"
        )

        result = handler.process_file(r_file)

        assert result.success, f"Processing failed: {result.error_message}"
        assert result.media_content is not None
        assert "data analysis" in result.media_content.content.lower()

    def test_jupyter_notebook_processing(self, temp_dir, handler):
        """Test that Jupyter notebooks can be processed."""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Analysis Notebook"]
                },
                {
                    "cell_type": "code",
                    "source": ["import numpy as np\nprint('Hello, World!')"],
                    "outputs": []
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5
        }

        ipynb_file = temp_dir / "notebook.ipynb"
        ipynb_file.write_text(json.dumps(notebook_content, indent=2), encoding='utf-8')

        result = handler.process_file(ipynb_file)

        assert result.success, f"Processing failed: {result.error_message}"
        assert result.media_content is not None
        # The content should contain the notebook structure
        content = result.media_content.content
        assert "numpy" in content or "Analysis" in content

    def test_unknown_text_extension(self, temp_dir, handler):
        """Test that files with unknown but text extensions are processed."""
        # Create a file with a completely unknown extension but text content
        custom_file = self.create_test_file(
            temp_dir,
            "data.custom",
            "This is a custom text file format\nwith multiple lines\nof text content"
        )

        # Should be detected as text through content analysis
        media_type = detect_media_type(custom_file)
        assert media_type == MediaType.TEXT

        # Should be processable
        result = handler.process_file(custom_file)
        assert result.success, f"Processing failed: {result.error_message}"
        assert "custom text file" in result.media_content.content.lower()

    def test_binary_file_rejected(self, temp_dir):
        """Test that binary files are not detected as text."""
        # Create a binary file
        binary_file = temp_dir / "data.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05\xFF\xFE\xFD')

        # Should not be detected as text
        assert not is_text_file(binary_file)
        # Should be detected as DOCUMENT (fallback for unknown binary)
        media_type = detect_media_type(binary_file)
        assert media_type == MediaType.DOCUMENT

    def test_empty_file(self, temp_dir, handler):
        """Test that empty files are handled."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("", encoding='utf-8')

        assert is_text_file(empty_file)
        media_type = detect_media_type(empty_file)
        assert media_type == MediaType.TEXT

        result = handler.process_file(empty_file)
        assert result.success

    def test_supports_format_for_all_text_files(self, handler):
        """Test that handler supports format check returns True for any text type."""
        # Known extensions
        assert handler.supports_format(MediaType.TEXT, 'txt')
        assert handler.supports_format(MediaType.TEXT, 'py')
        assert handler.supports_format(MediaType.TEXT, 'r')
        assert handler.supports_format(MediaType.TEXT, 'R')
        assert handler.supports_format(MediaType.TEXT, 'ipynb')
        assert handler.supports_format(MediaType.TEXT, 'rmd')

        # Unknown extensions - should still return True for TEXT type
        assert handler.supports_format(MediaType.TEXT, 'unknown')
        assert handler.supports_format(MediaType.TEXT, 'custom')
        assert handler.supports_format(MediaType.TEXT, 'xyz')

    def test_cli_file_attachment_workflow(self, temp_dir, handler):
        """Test the complete workflow as used by the CLI @filepath feature."""
        # Create various test files
        files_to_test = {
            "analysis.R": "# R analysis\nlibrary(ggplot2)\n",
            "notebook.ipynb": json.dumps({
                "cells": [{"cell_type": "code", "source": ["import pandas"]}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5
            }),
            "query.sql": "SELECT * FROM users WHERE active = true;",
            "config.yaml": "database:\n  host: localhost\n  port: 5432",
        }

        for filename, content in files_to_test.items():
            filepath = self.create_test_file(temp_dir, filename, content)

            # This simulates what the CLI does:
            # 1. Detect media type
            media_type = detect_media_type(filepath)

            # 2. Check if supported
            format_ext = filepath.suffix.lower().lstrip('.')
            assert handler.supports_format(media_type, format_ext), \
                f"Format {format_ext} not supported for {filename}"

            # 3. Process the file
            result = handler.process_file(filepath)

            # 4. Verify success
            assert result.success, \
                f"Failed to process {filename}: {result.error_message}"
            assert result.media_content is not None, \
                f"No media content for {filename}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
