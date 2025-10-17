"""
Comprehensive tests for media processors.

Tests all media processors with real files to ensure proper functionality
across different file types and formats.
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image as PILImage
import io

from abstractcore.media.processors import ImageProcessor, TextProcessor, PDFProcessor
from abstractcore.media.types import MediaType, ContentFormat


class TestImageProcessor:
    """Test image processing functionality."""

    def setup_method(self):
        """Create test images for processing."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a test PNG image
        self.test_png = Path(self.temp_dir) / "test.png"
        img = PILImage.new('RGB', (100, 100), color='red')
        img.save(self.test_png)

        # Create a test JPEG image
        self.test_jpg = Path(self.temp_dir) / "test.jpg"
        img = PILImage.new('RGB', (200, 150), color='blue')
        img.save(self.test_jpg)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_png_processing(self):
        """Test PNG image processing."""
        processor = ImageProcessor()
        result = processor.process_file(self.test_png)

        assert result.success
        assert result.media_content.media_type == MediaType.IMAGE
        assert result.media_content.content_format == ContentFormat.BASE64
        assert result.media_content.mime_type == "image/png"
        assert len(result.media_content.content) > 0
        assert result.media_content.metadata["file_name"] == "test.png"

    def test_jpg_processing(self):
        """Test JPEG image processing."""
        processor = ImageProcessor()
        result = processor.process_file(self.test_jpg)

        assert result.success
        assert result.media_content.media_type == MediaType.IMAGE
        assert result.media_content.content_format == ContentFormat.BASE64
        assert result.media_content.mime_type == "image/jpeg"
        assert len(result.media_content.content) > 0

    def test_image_optimization(self):
        """Test image optimization features."""
        processor = ImageProcessor(optimize_for_vision=True, max_dimension=50)
        result = processor.process_file(self.test_jpg)

        assert result.success
        # Image should be resized
        metadata = result.media_content.metadata
        assert metadata["final_size"][0] <= 50
        assert metadata["final_size"][1] <= 50

    def test_unsupported_format(self):
        """Test handling of unsupported image format."""
        # Create a fake file with unsupported extension
        fake_file = Path(self.temp_dir) / "test.xyz"
        fake_file.write_text("not an image")

        processor = ImageProcessor()
        result = processor.process_file(fake_file)

        assert not result.success
        assert "Unsupported image format" in result.error_message


class TestTextProcessor:
    """Test text processing functionality."""

    def setup_method(self):
        """Create test text files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a test CSV file
        self.test_csv = Path(self.temp_dir) / "test.csv"
        self.test_csv.write_text("name,age,city\nJohn,25,NYC\nJane,30,LA\n")

        # Create a test markdown file
        self.test_md = Path(self.temp_dir) / "test.md"
        self.test_md.write_text("# Test Document\n\nThis is a **test** document.\n")

        # Create a test JSON file
        self.test_json = Path(self.temp_dir) / "test.json"
        self.test_json.write_text('{"name": "test", "value": 42}')

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_csv_processing(self):
        """Test CSV file processing with pandas integration."""
        processor = TextProcessor()
        result = processor.process_file(self.test_csv)

        assert result.success
        assert result.media_content.media_type == MediaType.TEXT
        assert result.media_content.content_format == ContentFormat.TEXT
        # Should contain formatted table data
        assert "name" in result.media_content.content
        assert "John" in result.media_content.content
        assert "25" in result.media_content.content

    def test_markdown_processing(self):
        """Test markdown file processing."""
        processor = TextProcessor()
        result = processor.process_file(self.test_md)

        assert result.success
        assert result.media_content.media_type == MediaType.TEXT
        assert "# Test Document" in result.media_content.content
        assert "**test**" in result.media_content.content

    def test_json_processing(self):
        """Test JSON file processing."""
        processor = TextProcessor()
        result = processor.process_file(self.test_json)

        assert result.success
        assert result.media_content.media_type == MediaType.TEXT
        # Should be formatted JSON
        assert "name" in result.media_content.content
        assert "test" in result.media_content.content


class TestPDFProcessor:
    """Test PDF processing functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pdf_processing_mock(self):
        """Test PDF processing with mocked PyMuPDF4LLM."""
        # Create a fake PDF file
        fake_pdf = Path(self.temp_dir) / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\nfake pdf content")

        processor = PDFProcessor()

        # Mock the dependency to avoid requiring PyMuPDF4LLM in tests
        try:
            result = processor.process_file(fake_pdf)
            # If PyMuPDF4LLM is available, test normally
            assert result.success or "PyMuPDF4LLM not installed" in result.error_message
        except Exception as e:
            # If dependency not available, ensure proper error handling
            assert "PyMuPDF4LLM not installed" in str(e) or "dependency" in str(e).lower()

    def test_non_pdf_file(self):
        """Test handling of non-PDF file."""
        fake_file = Path(self.temp_dir) / "test.txt"
        fake_file.write_text("not a pdf")

        processor = PDFProcessor()
        result = processor.process_file(fake_file)

        assert not result.success
        assert "Invalid PDF" in result.error_message or "not a PDF" in result.error_message


@pytest.mark.skipif(
    not os.getenv("TEST_WITH_OFFICE_DOCS"),
    reason="Office document tests require unstructured library and TEST_WITH_OFFICE_DOCS=1"
)
class TestOfficeProcessor:
    """Test Office document processing functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_office_processor_import(self):
        """Test that OfficeProcessor can be imported."""
        try:
            from abstractcore.media.processors import OfficeProcessor
            processor = OfficeProcessor()
            assert processor is not None
        except ImportError as e:
            pytest.skip(f"OfficeProcessor not available: {e}")

    def test_docx_processing_mock(self):
        """Test DOCX processing with mocked dependencies."""
        fake_docx = Path(self.temp_dir) / "test.docx"
        fake_docx.write_bytes(b"PK\x03\x04fake docx")  # Fake ZIP signature

        try:
            from abstractcore.media.processors import OfficeProcessor
            processor = OfficeProcessor()
            result = processor.process_file(fake_docx)
            # Test that it either works or gives proper error
            assert result.success or "unstructured not installed" in result.error_message
        except ImportError:
            pytest.skip("OfficeProcessor not available")


class TestAutoMediaHandler:
    """Test automatic media handler that selects appropriate processor."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auto_handler_import(self):
        """Test that AutoMediaHandler can be imported."""
        try:
            from abstractcore.media import AutoMediaHandler
            handler = AutoMediaHandler()
            assert handler is not None
        except ImportError as e:
            pytest.skip(f"AutoMediaHandler not available: {e}")

    def test_auto_processor_selection(self):
        """Test that AutoMediaHandler selects correct processor."""
        try:
            from abstractcore.media import AutoMediaHandler

            # Create test files
            test_txt = Path(self.temp_dir) / "test.txt"
            test_txt.write_text("Hello world")

            test_png = Path(self.temp_dir) / "test.png"
            img = PILImage.new('RGB', (10, 10), color='red')
            img.save(test_png)

            handler = AutoMediaHandler()

            # Test text file
            result = handler.process_file(test_txt)
            assert result.success
            assert result.media_content.media_type == MediaType.TEXT

            # Test image file
            result = handler.process_file(test_png)
            assert result.success
            assert result.media_content.media_type == MediaType.IMAGE

        except ImportError:
            pytest.skip("AutoMediaHandler not available")


class TestMediaTypeDetection:
    """Test media type detection functionality."""

    def test_detect_media_type(self):
        """Test media type detection from file extensions."""
        from abstractcore.media.types import detect_media_type

        assert detect_media_type(Path("test.jpg")) == MediaType.IMAGE
        assert detect_media_type(Path("test.png")) == MediaType.IMAGE
        assert detect_media_type(Path("test.pdf")) == MediaType.DOCUMENT
        assert detect_media_type(Path("test.txt")) == MediaType.TEXT
        assert detect_media_type(Path("test.csv")) == MediaType.TEXT
        assert detect_media_type(Path("test.md")) == MediaType.TEXT
        assert detect_media_type(Path("test.docx")) == MediaType.DOCUMENT


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/media_handling/test_media_processors.py -v
    pytest.main([__file__, "-v"])