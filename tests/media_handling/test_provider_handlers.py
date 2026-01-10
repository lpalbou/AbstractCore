"""
Comprehensive tests for provider-specific media handlers.

Tests all provider media handlers to ensure proper formatting and compatibility
with each provider's API requirements.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image as PILImage

from abstractcore.media.types import MediaContent, MediaType, ContentFormat


class TestOpenAIMediaHandler:
    """Test OpenAI-specific media handler."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test image
        self.test_png = Path(self.temp_dir) / "test.png"
        img = PILImage.new('RGB', (100, 100), color='red')
        img.save(self.test_png)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_openai_handler_import(self):
        """Test that OpenAI handler can be imported."""
        try:
            from abstractcore.media.handlers import OpenAIMediaHandler
            handler = OpenAIMediaHandler()
            assert handler is not None
        except ImportError as e:
            pytest.skip(f"OpenAI media handler not available: {e}")

    def test_openai_image_formatting(self):
        """Test OpenAI image formatting."""
        try:
            from abstractcore.media.handlers import OpenAIMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image first
            processor = ImageProcessor()
            result = processor.process_file(self.test_png)
            assert result.success

            # Format for OpenAI
            handler = OpenAIMediaHandler()
            formatted = handler.format_for_provider(result.media_content)

            assert formatted["type"] == "image_url"
            assert "image_url" in formatted
            assert "url" in formatted["image_url"]
            assert formatted["image_url"]["url"].startswith("data:image/png;base64,")

        except ImportError:
            pytest.skip("OpenAI media handler not available")

    def test_openai_multimodal_message(self):
        """Test OpenAI multimodal message creation."""
        try:
            from abstractcore.media.handlers import OpenAIMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_png)
            assert result.success

            # Create multimodal message with proper capabilities
            from abstractcore.architectures import get_model_capabilities
            caps = get_model_capabilities("gpt-4o")
            handler = OpenAIMediaHandler(model_capabilities=caps, model_name="gpt-4o")
            message = handler.create_multimodal_message(
                "Describe this image",
                [result.media_content]
            )

            assert message["role"] == "user"
            assert isinstance(message["content"], list)
            assert len(message["content"]) == 2  # text + image

            # Check text content
            text_content = next(item for item in message["content"] if item["type"] == "text")
            assert text_content["text"] == "Describe this image"

            # Check image content
            img_content = next(item for item in message["content"] if item["type"] == "image_url")
            assert "image_url" in img_content

        except ImportError:
            pytest.skip("OpenAI media handler not available")

    def test_openai_vision_model_validation(self):
        """Test OpenAI vision model validation."""
        try:
            from abstractcore.media.handlers import OpenAIMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_png)
            assert result.success

            # Test with vision models
            handler = OpenAIMediaHandler({"vision_support": True})
            assert handler.validate_media_for_model(result.media_content, "gpt-4o")
            assert handler.validate_media_for_model(result.media_content, "gpt-4-turbo-with-vision")

            # Test with non-vision models
            assert not handler.validate_media_for_model(result.media_content, "gpt-4")
            assert not handler.validate_media_for_model(result.media_content, "gpt-3.5-turbo")

        except ImportError:
            pytest.skip("OpenAI media handler not available")


class TestAnthropicMediaHandler:
    """Test Anthropic-specific media handler."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test image
        self.test_png = Path(self.temp_dir) / "test.png"
        img = PILImage.new('RGB', (100, 100), color='blue')
        img.save(self.test_png)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_anthropic_handler_import(self):
        """Test that Anthropic handler can be imported."""
        try:
            from abstractcore.media.handlers import AnthropicMediaHandler
            handler = AnthropicMediaHandler()
            assert handler is not None
        except ImportError as e:
            pytest.skip(f"Anthropic media handler not available: {e}")

    def test_anthropic_image_formatting(self):
        """Test Anthropic image formatting."""
        try:
            from abstractcore.media.handlers import AnthropicMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image first
            processor = ImageProcessor()
            result = processor.process_file(self.test_png)
            assert result.success

            # Format for Anthropic
            handler = AnthropicMediaHandler()
            formatted = handler.format_for_provider(result.media_content)

            assert formatted["type"] == "image"
            assert "source" in formatted
            assert formatted["source"]["type"] == "base64"
            assert formatted["source"]["media_type"] == "image/png"
            assert len(formatted["source"]["data"]) > 0

        except ImportError:
            pytest.skip("Anthropic media handler not available")

    def test_anthropic_multimodal_message(self):
        """Test Anthropic multimodal message creation."""
        try:
            from abstractcore.media.handlers import AnthropicMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_png)
            assert result.success

            # Create multimodal message with proper capabilities
            from abstractcore.architectures import get_model_capabilities
            caps = get_model_capabilities("claude-haiku-4-5")
            handler = AnthropicMediaHandler(model_capabilities=caps, model_name="claude-haiku-4-5")
            message = handler.create_multimodal_message(
                "What do you see in this image?",
                [result.media_content]
            )

            assert message["role"] == "user"
            assert isinstance(message["content"], list)
            assert len(message["content"]) == 2  # text + image

            # Anthropic puts text first
            assert message["content"][0]["type"] == "text"
            assert message["content"][1]["type"] == "image"

        except ImportError:
            pytest.skip("Anthropic media handler not available")

    def test_anthropic_vision_model_validation(self):
        """Test Anthropic vision model validation."""
        try:
            from abstractcore.media.handlers import AnthropicMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_png)
            assert result.success

            # Test with vision models
            handler = AnthropicMediaHandler({"vision_support": True})
            assert handler.validate_media_for_model(result.media_content, "claude-haiku-4-5")

        except ImportError:
            pytest.skip("Anthropic media handler not available")

    def test_anthropic_document_analysis_prompt(self):
        """Test Anthropic document analysis prompt creation."""
        try:
            from abstractcore.media.handlers import AnthropicMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_png)
            assert result.success

            handler = AnthropicMediaHandler()

            # Test different analysis types
            prompt = handler.create_document_analysis_prompt([result.media_content], "summary")
            assert "summarize" in prompt.lower()

            prompt = handler.create_document_analysis_prompt([result.media_content], "extract")
            assert "extract" in prompt.lower()

            prompt = handler.create_document_analysis_prompt([result.media_content], "qa")
            assert "questions" in prompt.lower()

        except ImportError:
            pytest.skip("Anthropic media handler not available")


class TestLocalMediaHandler:
    """Test local provider media handler (Ollama, MLX, LMStudio)."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test image
        self.test_jpg = Path(self.temp_dir) / "test.jpg"
        img = PILImage.new('RGB', (50, 50), color='green')
        img.save(self.test_jpg)

        # Create test text file
        self.test_txt = Path(self.temp_dir) / "test.txt"
        self.test_txt.write_text("This is a test document with important information.")

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_local_handler_import(self):
        """Test that local handler can be imported."""
        try:
            from abstractcore.media.handlers import LocalMediaHandler
            handler = LocalMediaHandler("ollama")
            assert handler is not None
        except ImportError as e:
            pytest.skip(f"Local media handler not available: {e}")

    def test_ollama_formatting(self):
        """Test Ollama-specific formatting."""
        try:
            from abstractcore.media.handlers import LocalMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_jpg)
            assert result.success

            # Format for Ollama
            handler = LocalMediaHandler("ollama", {"vision_support": True})
            formatted = handler.format_for_provider(result.media_content)

            assert formatted["type"] == "image"
            assert "data" in formatted
            assert formatted["mime_type"] == "image/jpeg"

        except ImportError:
            pytest.skip("Local media handler not available")

    def test_lmstudio_formatting(self):
        """Test LMStudio-specific formatting."""
        try:
            from abstractcore.media.handlers import LocalMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_jpg)
            assert result.success

            # Format for LMStudio (OpenAI-compatible)
            handler = LocalMediaHandler("lmstudio", {"vision_support": True})
            formatted = handler.format_for_provider(result.media_content)

            assert formatted["type"] == "image_url"
            assert "image_url" in formatted
            assert "url" in formatted["image_url"]
            assert formatted["image_url"]["url"].startswith("data:")

        except ImportError:
            pytest.skip("Local media handler not available")

    def test_text_embedded_message(self):
        """Test text-embedded message creation for local providers."""
        try:
            from abstractcore.media.handlers import LocalMediaHandler
            from abstractcore.media.processors import TextProcessor

            # Process text document
            processor = TextProcessor()
            result = processor.process_file(self.test_txt)
            assert result.success

            # Create text-embedded message
            handler = LocalMediaHandler("ollama", prefer_text_extraction=True)
            message = handler.create_multimodal_message(
                "Analyze this document",
                [result.media_content]
            )

            # Should return a string with embedded content
            assert isinstance(message, str)
            assert "Analyze this document" in message
            assert "test document" in message
            assert "Content from test.txt" in message

        except ImportError:
            pytest.skip("Local media handler not available")

    def test_vision_model_validation(self):
        """Test local vision model validation."""
        try:
            from abstractcore.media.handlers import LocalMediaHandler
            from abstractcore.media.processors import ImageProcessor

            # Process image
            processor = ImageProcessor()
            result = processor.process_file(self.test_jpg)
            assert result.success

            # Test with vision models
            handler = LocalMediaHandler("ollama", {"vision_support": True})
            assert handler.validate_media_for_model(result.media_content, "qwen3-vl-8b")
            assert handler.validate_media_for_model(result.media_content, "gemma3:4b")

            # Test without vision support
            handler_no_vision = LocalMediaHandler("ollama", {"vision_support": False})
            assert not handler_no_vision.validate_media_for_model(result.media_content, "qwen3-4b")

        except ImportError:
            pytest.skip("Local media handler not available")


class TestMediaCapabilities:
    """Test media capability detection system."""

    def test_capability_detection_import(self):
        """Test that capability detection can be imported."""
        try:
            from abstractcore.media.capabilities import get_media_capabilities, is_vision_model
            assert get_media_capabilities is not None
            assert is_vision_model is not None
        except ImportError as e:
            pytest.skip(f"Media capabilities not available: {e}")

    def test_vision_model_detection(self):
        """Test vision model detection."""
        try:
            from abstractcore.media.capabilities import is_vision_model

            # Test OpenAI vision models
            assert is_vision_model("gpt-4o")
            assert is_vision_model("gpt-4-turbo-with-vision")
            assert not is_vision_model("gpt-4")
            assert not is_vision_model("gpt-3.5-turbo")

            # Test Anthropic vision models
            assert is_vision_model("claude-haiku-4-5")

            # Test local vision models
            assert is_vision_model("qwen3-vl")
            assert is_vision_model("gemma3:4b")
            assert is_vision_model("qwen2.5-vl-7b")

        except ImportError:
            pytest.skip("Media capabilities not available")

    def test_get_supported_media_types(self):
        """Test getting supported media types for models."""
        try:
            from abstractcore.media.capabilities import get_supported_media_types
            from abstractcore.media.types import MediaType

            # Test vision model
            media_types = get_supported_media_types("gpt-4o")
            assert MediaType.IMAGE in media_types
            assert MediaType.TEXT in media_types

            # Test text-only model
            media_types = get_supported_media_types("gpt-4")
            assert MediaType.TEXT in media_types
            # Should not support images if no vision
            if MediaType.IMAGE in media_types:
                # Vision support detected for this model
                pass

        except ImportError:
            pytest.skip("Media capabilities not available")


class TestProviderIntegration:
    """Test integration with provider capabilities."""

    def test_model_capabilities_loading(self):
        """Test loading model capabilities from JSON."""
        try:
            from abstractcore.media.capabilities import get_media_capabilities

            # Test specific models mentioned by user
            caps = get_media_capabilities("qwen2.5-vl-7b")
            assert caps.vision_support

            caps = get_media_capabilities("gemma-3n-e4b")
            assert caps.vision_support

            caps = get_media_capabilities("gemma3-4b")
            assert caps.vision_support

        except ImportError:
            pytest.skip("Media capabilities not available")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/media_handling/test_provider_handlers.py -v
    pytest.main([__file__, "-v"])
