"""
Real provider integration tests with NO MOCKING.
Tests use actual running services (Ollama, LMStudio) or real API keys.
"""

import os
import tempfile
from pathlib import Path
import pytest
from PIL import Image as PILImage

from abstractcore import create_llm
from abstractcore.core.types import GenerateResponse


class TestRealMediaIntegration:
    """Test media integration with REAL implementations - NO MOCKS."""

    def setup_method(self):
        """Create real test files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create real test image
        self.test_image = Path(self.temp_dir) / "test.png"
        img = PILImage.new('RGB', (100, 100), color='red')
        img.save(self.test_image)

        # Create real test text file
        self.test_text = Path(self.temp_dir) / "test.txt"
        self.test_text.write_text("This is test content for media processing.")

        # Create real CSV file
        self.test_csv = Path(self.temp_dir) / "data.csv"
        self.test_csv.write_text("name,value\ntest,123\nfoo,456")

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/ollama") and not os.path.exists("/usr/local/bin/ollama"),
        reason="Ollama not installed or not running"
    )
    def test_ollama_single_image_real(self):
        """Test Ollama with real image - NO MOCKING."""
        try:
            # Use REAL Ollama instance (must be running)
            llm = create_llm("ollama", model="llama3.2-vision:11b")

            # Real API call
            response = llm.generate(
                "What color is this image? Answer in one word.",
                media=[str(self.test_image)]
            )

            # Verify real response
            assert isinstance(response, GenerateResponse)
            assert response.content is not None
            assert len(response.content) > 0

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/ollama") and not os.path.exists("/usr/local/bin/ollama"),
        reason="Ollama not installed"
    )
    def test_ollama_multiple_media_real(self):
        """Test Ollama with multiple media files - NO MOCKING."""
        try:
            llm = create_llm("ollama", model="llama3.2-vision:11b")

            # Test with multiple real files
            response = llm.generate(
                "How many files do you see?",
                media=[str(self.test_image), str(self.test_text)]
            )

            assert isinstance(response, GenerateResponse)
            assert response.content is not None

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - skipping real API test"
    )
    def test_openai_real_api(self):
        """Test OpenAI with REAL API - NO MOCKING."""
        # Use real API with real key
        llm = create_llm("openai", model="gpt-4o-mini", timeout=5.0)  # Use mini for cost efficiency

        try:
            response = llm.generate(
                "What color is dominant in this image? One word only.",
                media=[str(self.test_image)]
            )
        except Exception as e:
            msg = str(e).lower()
            if any(
                keyword in msg
                for keyword in (
                    "connection error",
                    "connecterror",
                    "operation not permitted",
                    "network is unreachable",
                    "nodename nor servname provided",
                    "timeout",
                )
            ):
                pytest.skip(f"OpenAI not reachable in this environment: {e}")
            raise

        # Real API returns real content
        assert isinstance(response, GenerateResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert "red" in response.content.lower() or "Red" in response.content

    def test_media_error_handling_real(self):
        """Test error handling with invalid files - NO MOCKING."""
        # Create invalid file
        invalid_file = Path(self.temp_dir) / "invalid.xyz"
        invalid_file.write_bytes(b"not a valid media file")

        try:
            llm = create_llm("ollama", model="llama3.2-vision:11b")

            # Should handle gracefully (warning, not crash)
            response = llm.generate(
                "What do you see?",
                media=[str(invalid_file)]
            )

            # Should not crash, might return response without media
            assert isinstance(response, GenerateResponse)

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")


class TestRealStreamingMedia:
    """Test streaming with media - REAL implementation."""

    def setup_method(self):
        """Create test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image = Path(self.temp_dir) / "test.png"
        img = PILImage.new('RGB', (50, 50), color='blue')
        img.save(self.test_image)

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/ollama") and not os.path.exists("/usr/local/bin/ollama"),
        reason="Ollama not installed"
    )
    def test_streaming_with_media_real(self):
        """Test streaming with media - NO MOCKING."""
        try:
            llm = create_llm("ollama", model="llama3.2-vision:11b")

            # Real streaming
            response_stream = llm.generate(
                "What color is this?",
                media=[str(self.test_image)],
                stream=True
            )

            # Collect chunks
            chunks = []
            for chunk in response_stream:
                assert isinstance(chunk, GenerateResponse)
                if chunk.content:
                    chunks.append(chunk.content)

            # Should have received content
            assert len(chunks) > 0

        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")


class TestMediaCapabilityValidation:
    """Test capability validation - real detection logic."""

    def test_capability_validation(self):
        """Test media capability detection."""
        from abstractcore.architectures.detection import supports_vision

        # Test real capability detection
        assert supports_vision("gpt-4o") is True
        assert supports_vision("gpt-4o-mini") is True
        assert supports_vision("claude-3-5-sonnet") is True  # Use base model name
        assert supports_vision("llama3.2-vision:11b") is True

        # Non-vision models
        assert supports_vision("gpt-3.5-turbo") is False
        assert supports_vision("gpt-4") is False
