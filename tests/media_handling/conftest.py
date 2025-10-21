"""
Pytest configuration for media handling tests.

Provides shared fixtures and configuration for all media handling tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image as PILImage


@pytest.fixture(scope="session")
def sample_media_files():
    """Create sample media files for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create sample image files
    sample_png = temp_dir / "sample.png"
    img = PILImage.new('RGB', (100, 100), color='red')
    img.save(sample_png)

    sample_jpg = temp_dir / "sample.jpg"
    img = PILImage.new('RGB', (150, 100), color='blue')
    img.save(sample_jpg)

    # Create sample text files
    sample_txt = temp_dir / "sample.txt"
    sample_txt.write_text("This is a sample text document for testing purposes.")

    sample_csv = temp_dir / "sample.csv"
    sample_csv.write_text("name,age,city\nAlice,25,NYC\nBob,30,LA\nCharlie,35,Chicago\n")

    sample_md = temp_dir / "sample.md"
    sample_md.write_text("# Sample Document\n\nThis is a **sample** markdown document.\n\n## Features\n\n- List item 1\n- List item 2")

    sample_json = temp_dir / "sample.json"
    sample_json.write_text('{"name": "sample", "type": "test", "data": [1, 2, 3]}')

    # Create fake PDF (for testing error handling)
    fake_pdf = temp_dir / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\nfake pdf content for testing")

    files = {
        'png': sample_png,
        'jpg': sample_jpg,
        'txt': sample_txt,
        'csv': sample_csv,
        'md': sample_md,
        'json': sample_json,
        'pdf': fake_pdf,
        'dir': temp_dir
    }

    yield files

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_provider_response():
    """Create test provider response for testing."""
    from unittest.mock import Mock

    response = Mock()
    response.content = "Mocked response content"
    response.model = "test-model"
    response.finish_reason = "stop"
    response.usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    }

    return response


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "requires_deps: marks tests that require optional dependencies"
    )
    config.addinivalue_line(
        "markers", "real_models: marks tests that require real model endpoints"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark tests that require dependencies
        if any(dep in item.nodeid for dep in ["office", "pdf", "unstructured"]):
            item.add_marker(pytest.mark.requires_deps)


# Skip certain tests if dependencies are not available
def pytest_runtest_setup(item):
    """Setup test runs with dependency checking."""
    if "requires_deps" in [mark.name for mark in item.iter_markers()]:
        # Check if optional dependencies are available
        try:
            import PIL
            import pandas
        except ImportError:
            pytest.skip("Optional dependencies not available")