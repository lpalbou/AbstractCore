"""
Integration tests for media handling across all providers.

Tests end-to-end media processing through the generate() method to ensure
complete functionality from file input to provider API calls.
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image as PILImage
from unittest.mock import Mock, patch, MagicMock

from abstractcore import create_llm
from abstractcore.core.types import GenerateResponse


class TestMediaIntegration:
    """Test media integration across all providers."""

    def setup_method(self):
        """Set up test environment with sample files."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test image
        self.test_image = Path(self.temp_dir) / "test.png"
        img = PILImage.new('RGB', (100, 100), color='red')
        img.save(self.test_image)

        # Create test text file
        self.test_text = Path(self.temp_dir) / "test.txt"
        self.test_text.write_text("This is a test document with sample content.")

        # Create test CSV file
        self.test_csv = Path(self.temp_dir) / "data.csv"
        self.test_csv.write_text("name,value\ntest,42\nsample,100\n")

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skipif(
        not pytest.importorskip("abstractcore.media", reason="Media handling not available"),
        reason="Media handling dependencies not available"
    )
    def test_openai_media_integration(self):
        """Test OpenAI provider with media files."""
        try:
            # Mock OpenAI responses
            with patch('abstractcore.providers.openai_provider.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client

                # Mock chat completion response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "I can see a red image."
                mock_response.choices[0].finish_reason = "stop"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 100
                mock_response.usage.completion_tokens = 20
                mock_response.usage.total_tokens = 120
                mock_client.chat.completions.create.return_value = mock_response

                # Create LLM instance
                llm = create_llm("openai", model="gpt-4o", api_key="test-key")

                # Test with image
                response = llm.generate(
                    "What do you see in this image?",
                    media=[str(self.test_image)]
                )

                assert isinstance(response, GenerateResponse)
                assert response.content == "I can see a red image."
                assert response.model == "gpt-4o"

                # Verify the API was called with proper multimodal format
                call_args = mock_client.chat.completions.create.call_args
                messages = call_args[1]['messages']

                # Should have a multimodal message
                user_message = next(msg for msg in messages if msg['role'] == 'user')
                assert isinstance(user_message['content'], list)

                # Should contain text and image
                content_types = [item['type'] for item in user_message['content']]
                assert 'text' in content_types
                assert 'image_url' in content_types

        except ImportError:
            pytest.skip("OpenAI provider or media handling not available")

    @pytest.mark.skipif(
        not pytest.importorskip("abstractcore.media", reason="Media handling not available"),
        reason="Media handling dependencies not available"
    )
    def test_anthropic_media_integration(self):
        """Test Anthropic provider with media files."""
        try:
            # Mock Anthropic client
            with patch('anthropic.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_anthropic.return_value = mock_client

                # Mock message response
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "This appears to be a red colored image."
                mock_response.stop_reason = "end_turn"
                mock_response.usage = Mock()
                mock_response.usage.input_tokens = 150
                mock_response.usage.output_tokens = 25
                mock_client.messages.create.return_value = mock_response

                # Create LLM instance
                llm = create_llm("anthropic", model="claude-3.5-sonnet", api_key="test-key")

                # Test with image
                response = llm.generate(
                    "Describe this image",
                    media=[str(self.test_image)]
                )

                assert isinstance(response, GenerateResponse)
                assert response.content == "This appears to be a red colored image."
                assert response.model == "claude-3.5-sonnet"

                # Verify the API was called with proper Anthropic format
                call_args = mock_client.messages.create.call_args
                messages = call_args[1]['messages']

                # Should have a multimodal message
                user_message = next(msg for msg in messages if msg['role'] == 'user')
                assert isinstance(user_message['content'], list)

                # Should contain text and image with Anthropic format
                content_types = [item['type'] for item in user_message['content']]
                assert 'text' in content_types
                assert 'image' in content_types

        except ImportError:
            pytest.skip("Anthropic provider or media handling not available")

    @pytest.mark.skipif(
        not pytest.importorskip("abstractcore.media", reason="Media handling not available"),
        reason="Media handling dependencies not available"
    )
    def test_ollama_media_integration(self):
        """Test Ollama provider with media files."""
        try:
            # Mock httpx for Ollama
            with patch('httpx.Client') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock Ollama response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "message": {
                        "role": "assistant",
                        "content": "I can see this is a red colored square image."
                    },
                    "done": True
                }
                mock_client.post.return_value = mock_response

                # Create LLM instance
                llm = create_llm("ollama", model="qwen3-vl:8b", base_url="http://localhost:11434")

                # Test with image
                response = llm.generate(
                    "What color is this image?",
                    media=[str(self.test_image)]
                )

                assert isinstance(response, GenerateResponse)
                assert response.content == "I can see this is a red colored square image."
                assert response.model == "qwen3-vl:8b"

                # Verify the API was called
                assert mock_client.post.called

        except ImportError:
            pytest.skip("Ollama provider or media handling not available")

    @pytest.mark.skipif(
        not pytest.importorskip("abstractcore.media", reason="Media handling not available"),
        reason="Media handling dependencies not available"
    )
    def test_lmstudio_media_integration(self):
        """Test LMStudio provider with media files."""
        try:
            # Mock httpx for LMStudio
            with patch('httpx.Client') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                # Mock LMStudio response
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "choices": [{
                        "message": {
                            "content": "The image shows a red square."
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 80,
                        "completion_tokens": 15,
                        "total_tokens": 95
                    }
                }
                mock_client.post.return_value = mock_response

                # Create LLM instance
                llm = create_llm("lmstudio", model="qwen/qwen2.5-vl-7b")

                # Test with image
                response = llm.generate(
                    "Describe this image",
                    media=[str(self.test_image)]
                )

                assert isinstance(response, GenerateResponse)
                assert response.content == "The image shows a red square."
                assert response.model == "qwen/qwen2.5-vl-7b"

        except ImportError:
            pytest.skip("LMStudio provider or media handling not available")

    @pytest.mark.skipif(
        not pytest.importorskip("abstractcore.media", reason="Media handling not available"),
        reason="Media handling dependencies not available"
    )
    def test_multiple_media_files(self):
        """Test providers with multiple media files."""
        try:
            # Mock OpenAI for this test
            with patch('abstractcore.providers.openai_provider.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client

                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "I see an image and text document."
                mock_response.choices[0].finish_reason = "stop"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 200
                mock_response.usage.completion_tokens = 30
                mock_response.usage.total_tokens = 230
                mock_client.chat.completions.create.return_value = mock_response

                llm = create_llm("openai", model="gpt-4o", api_key="test-key")

                # Test with multiple files
                response = llm.generate(
                    "Analyze these files",
                    media=[str(self.test_image), str(self.test_text), str(self.test_csv)]
                )

                assert isinstance(response, GenerateResponse)
                assert response.content == "I see an image and text document."

                # Verify multiple media items were processed
                call_args = mock_client.chat.completions.create.call_args
                messages = call_args[1]['messages']
                user_message = next(msg for msg in messages if msg['role'] == 'user')

                # Should have multiple content items (text + multiple media)
                assert len(user_message['content']) >= 3  # At least text + 2 media items

        except ImportError:
            pytest.skip("Provider or media handling not available")

    def test_media_error_handling(self):
        """Test error handling for invalid media files."""
        try:
            # Create invalid file
            invalid_file = Path(self.temp_dir) / "invalid.xyz"
            invalid_file.write_text("not a valid media file")

            # Use mocked provider to focus on media handling errors
            with patch('abstractcore.providers.openai_provider.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client

                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Response without media."
                mock_response.choices[0].finish_reason = "stop"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 50
                mock_response.usage.completion_tokens = 10
                mock_response.usage.total_tokens = 60
                mock_client.chat.completions.create.return_value = mock_response

                llm = create_llm("openai", model="gpt-4", api_key="test-key")

                # Should handle invalid media gracefully
                response = llm.generate(
                    "Test prompt",
                    media=[str(invalid_file)]
                )

                # Should still get a response (media processing failed but prompt succeeded)
                assert isinstance(response, GenerateResponse)

        except ImportError:
            pytest.skip("Provider not available")

    def test_media_without_vision_model(self):
        """Test media files with non-vision models."""
        try:
            # Mock OpenAI for non-vision model
            with patch('abstractcore.providers.openai_provider.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client

                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "I cannot process images."
                mock_response.choices[0].finish_reason = "stop"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 50
                mock_response.usage.completion_tokens = 10
                mock_response.usage.total_tokens = 60
                mock_client.chat.completions.create.return_value = mock_response

                # Use non-vision model
                llm = create_llm("openai", model="gpt-4", api_key="test-key")

                # Should handle gracefully
                response = llm.generate(
                    "Test with image",
                    media=[str(self.test_image)]
                )

                assert isinstance(response, GenerateResponse)

        except ImportError:
            pytest.skip("Provider not available")


class TestStreamingWithMedia:
    """Test streaming responses with media files."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test image
        self.test_image = Path(self.temp_dir) / "test.png"
        img = PILImage.new('RGB', (50, 50), color='blue')
        img.save(self.test_image)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.skipif(
        not pytest.importorskip("abstractcore.media", reason="Media handling not available"),
        reason="Media handling dependencies not available"
    )
    def test_streaming_with_media(self):
        """Test streaming responses with media content."""
        try:
            # Mock streaming response
            with patch('abstractcore.providers.openai_provider.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client

                # Mock streaming response
                def mock_stream():
                    chunks = [
                        {"choices": [{"delta": {"content": "I"}}]},
                        {"choices": [{"delta": {"content": " see"}}]},
                        {"choices": [{"delta": {"content": " a blue image."}}]},
                        {"choices": [{"finish_reason": "stop"}]}
                    ]
                    for chunk in chunks:
                        yield chunk

                mock_client.chat.completions.create.return_value = mock_stream()

                llm = create_llm("openai", model="gpt-4o", api_key="test-key")

                # Test streaming with media
                response_stream = llm.generate(
                    "What color is this image?",
                    media=[str(self.test_image)],
                    stream=True
                )

                # Collect streaming response
                full_content = ""
                for chunk in response_stream:
                    assert isinstance(chunk, GenerateResponse)
                    full_content += chunk.content

                assert "blue" in full_content.lower()

        except ImportError:
            pytest.skip("Provider or media handling not available")


class TestMediaCapabilityValidation:
    """Test media capability validation in providers."""

    def test_capability_validation(self):
        """Test that providers validate media capabilities correctly."""
        try:
            from abstractcore.media.capabilities import is_vision_model, supports_images

            # Test vision model detection
            assert is_vision_model("gpt-4o")
            assert is_vision_model("claude-3.5-sonnet")
            assert is_vision_model("qwen3-vl")
            assert is_vision_model("gemma3:4b")

            # Test non-vision models
            assert not is_vision_model("gpt-4")
            assert not is_vision_model("qwen3-4b")

            # Test image support
            assert supports_images("gpt-4o")
            assert supports_images("claude-3.5-sonnet")

        except ImportError:
            pytest.skip("Media capabilities not available")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/media_handling/test_provider_integration.py -v
    pytest.main([__file__, "-v"])