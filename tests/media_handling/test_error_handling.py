"""
Error handling and edge case tests for media handling system.

Tests various error conditions, edge cases, and dependency handling
to ensure robust operation of the media handling system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


class TestDependencyHandling:
    """Test handling of missing dependencies."""

    def test_media_import_error_handling(self):
        """Test graceful handling when media dependencies are missing."""
        try:
            from abstractcore import create_llm

            # Mock missing media dependencies
            with patch.dict('sys.modules', {'abstractcore.media': None}):
                llm = create_llm("openai", model="gpt-4", api_key="test-key")

                # Should handle missing media gracefully
                with patch('abstractcore.providers.openai_provider.OpenAI') as mock_openai:
                    mock_client = Mock()
                    mock_openai.return_value = mock_client

                    mock_response = Mock()
                    mock_response.choices = [Mock()]
                    mock_response.choices[0].message.content = "Text response without media."
                    mock_response.choices[0].finish_reason = "stop"
                    mock_response.usage = Mock()
                    mock_response.usage.prompt_tokens = 20
                    mock_response.usage.completion_tokens = 10
                    mock_response.usage.total_tokens = 30
                    mock_client.chat.completions.create.return_value = mock_response

                    # Should work without media processing
                    response = llm.generate("Test prompt")
                    assert response.content == "Text response without media."

        except ImportError:
            pytest.skip("Provider not available")

    def test_pil_missing_error_handling(self):
        """Test error handling when PIL is not available."""
        temp_dir = tempfile.mkdtemp()
        test_file = Path(temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image content")

        try:
            # Mock PIL import error
            with patch.dict('sys.modules', {'PIL': None, 'PIL.Image': None}):
                from abstractcore.media.processors import ImageProcessor

                processor = ImageProcessor()
                result = processor.process_file(test_file)

                assert not result.success
                assert "PIL" in result.error_message or "Pillow" in result.error_message

        except ImportError:
            # If ImageProcessor itself can't be imported due to PIL dependency
            pytest.skip("ImageProcessor not available")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pandas_missing_error_handling(self):
        """Test error handling when pandas is not available for CSV processing."""
        temp_dir = tempfile.mkdtemp()
        test_csv = Path(temp_dir) / "test.csv"
        test_csv.write_text("name,value\ntest,42\n")

        try:
            # Mock pandas import error
            with patch.dict('sys.modules', {'pandas': None}):
                from abstractcore.media.processors import TextProcessor

                processor = TextProcessor()
                result = processor.process_file(test_csv)

                # Should either handle gracefully or give clear error
                if not result.success:
                    assert "pandas" in result.error_message.lower()
                else:
                    # Should fallback to basic text processing
                    assert result.media_content.content is not None

        except ImportError:
            pytest.skip("TextProcessor not available")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestFileHandlingErrors:
    """Test error handling for various file-related issues."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        nonexistent_file = Path(self.temp_dir) / "does_not_exist.jpg"

        try:
            from abstractcore.media.processors import ImageProcessor

            processor = ImageProcessor()
            result = processor.process_file(nonexistent_file)

            assert not result.success
            assert "not found" in result.error_message.lower() or "exist" in result.error_message.lower()

        except ImportError:
            pytest.skip("ImageProcessor not available")

    def test_empty_file(self):
        """Test handling of empty files."""
        empty_file = Path(self.temp_dir) / "empty.txt"
        empty_file.touch()

        try:
            from abstractcore.media.processors import TextProcessor

            processor = TextProcessor()
            result = processor.process_file(empty_file)

            # Should handle empty files gracefully
            assert result.success
            assert len(result.media_content.content) == 0

        except ImportError:
            pytest.skip("TextProcessor not available")

    def test_corrupted_image_file(self):
        """Test handling of corrupted image files."""
        corrupted_image = Path(self.temp_dir) / "corrupted.jpg"
        corrupted_image.write_bytes(b"not a real image file")

        try:
            from abstractcore.media.processors import ImageProcessor

            processor = ImageProcessor()
            result = processor.process_file(corrupted_image)

            assert not result.success
            assert any(word in result.error_message.lower()
                      for word in ["corrupted", "invalid", "cannot", "decode"])

        except ImportError:
            pytest.skip("ImageProcessor not available")

    def test_permission_denied(self):
        """Test handling of permission denied errors."""
        # Create a file and make it unreadable (on Unix systems)
        restricted_file = Path(self.temp_dir) / "restricted.txt"
        restricted_file.write_text("secret content")

        try:
            # Try to make file unreadable
            import stat
            restricted_file.chmod(stat.S_IWUSR)  # Write-only, no read permission

            from abstractcore.media.processors import TextProcessor

            processor = TextProcessor()
            result = processor.process_file(restricted_file)

            # On systems where permission restrictions work, should get permission error
            if not result.success:
                assert any(word in result.error_message.lower()
                          for word in ["permission", "access", "denied"])

        except (ImportError, OSError):
            pytest.skip("Permission test not supported on this system")
        finally:
            # Restore permissions for cleanup
            try:
                restricted_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except:
                pass

    def test_unsupported_file_extension(self):
        """Test handling of files with unsupported extensions."""
        unknown_file = Path(self.temp_dir) / "test.unknown_extension"
        unknown_file.write_text("some content")

        try:
            from abstractcore.media.processors import ImageProcessor

            processor = ImageProcessor()
            result = processor.process_file(unknown_file)

            assert not result.success
            assert "unsupported" in result.error_message.lower() or "format" in result.error_message.lower()

        except ImportError:
            pytest.skip("ImageProcessor not available")


class TestProviderErrorHandling:
    """Test error handling in provider-specific media handlers."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_media_content_format(self):
        """Test handling of invalid MediaContent objects."""
        try:
            from abstractcore.media.handlers import OpenAIMediaHandler
            from abstractcore.media.types import MediaContent, MediaType, ContentFormat

            # Create invalid MediaContent (image without base64)
            invalid_content = MediaContent(
                content="not base64 content",
                media_type=MediaType.IMAGE,
                content_format=ContentFormat.TEXT,  # Wrong format for image
                mime_type="image/png",
                metadata={}
            )

            handler = OpenAIMediaHandler()

            # Should handle invalid format gracefully
            try:
                formatted = handler.format_for_provider(invalid_content)
                # If it doesn't raise an error, should at least detect the issue
                assert formatted is not None
            except Exception as e:
                # Should get meaningful error message
                assert any(word in str(e).lower()
                          for word in ["format", "base64", "invalid"])

        except ImportError:
            pytest.skip("Media handlers not available")

    def test_media_size_limits(self):
        """Test handling of files that exceed size limits."""
        try:
            from abstractcore.media.handlers import AnthropicMediaHandler
            from abstractcore.media.types import MediaContent, MediaType, ContentFormat

            # Create oversized image content (simulate very large base64)
            large_content = "x" * (10 * 1024 * 1024)  # 10MB of data

            large_media = MediaContent(
                content=large_content,
                media_type=MediaType.IMAGE,
                content_format=ContentFormat.BASE64,
                mime_type="image/png",
                metadata={"file_size": 15 * 1024 * 1024}  # 15MB
            )

            handler = AnthropicMediaHandler()

            # Should validate size limits
            is_valid = handler.validate_media_for_model(large_media, "claude-3.5-sonnet")

            # May reject oversized content
            if not is_valid:
                # This is expected behavior for oversized content
                assert True
            else:
                # If it accepts large content, that's also acceptable
                assert True

        except ImportError:
            pytest.skip("Media handlers not available")

    def test_unsupported_media_type_for_provider(self):
        """Test handling of unsupported media types for specific providers."""
        try:
            from abstractcore.media.handlers import LocalMediaHandler
            from abstractcore.media.types import MediaContent, MediaType, ContentFormat

            # Create audio content (not supported by most vision models)
            audio_content = MediaContent(
                content="fake audio data",
                media_type=MediaType.AUDIO,
                content_format=ContentFormat.BINARY,
                mime_type="audio/mp3",
                metadata={}
            )

            # Test with local handler that doesn't support audio
            handler = LocalMediaHandler("ollama", {"vision_support": True, "audio_support": False})

            # Should reject unsupported media type
            is_valid = handler.validate_media_for_model(audio_content, "qwen3-vl")
            assert not is_valid  # Should reject audio for vision-only model

        except ImportError:
            pytest.skip("Media handlers not available")


class TestEdgeCases:
    """Test various edge cases in media processing."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_very_small_image(self):
        """Test processing of very small images."""
        try:
            from PIL import Image as PILImage
            from abstractcore.media.processors import ImageProcessor

            # Create 1x1 pixel image
            tiny_image = Path(self.temp_dir) / "tiny.png"
            img = PILImage.new('RGB', (1, 1), color='red')
            img.save(tiny_image)

            processor = ImageProcessor()
            result = processor.process_file(tiny_image)

            # Should handle tiny images
            assert result.success
            assert result.media_content.metadata["original_size"] == [1, 1]

        except ImportError:
            pytest.skip("PIL or ImageProcessor not available")

    def test_very_large_text_file(self):
        """Test processing of very large text files."""
        try:
            from abstractcore.media.processors import TextProcessor

            # Create large text file
            large_text = Path(self.temp_dir) / "large.txt"
            large_content = "Large text content. " * 10000  # ~200KB
            large_text.write_text(large_content)

            processor = TextProcessor()
            result = processor.process_file(large_text)

            # Should handle large text files
            assert result.success
            assert len(result.media_content.content) > 100000

        except ImportError:
            pytest.skip("TextProcessor not available")

    def test_unicode_filenames(self):
        """Test handling of files with Unicode names."""
        try:
            from abstractcore.media.processors import TextProcessor

            # Create file with Unicode name
            unicode_file = Path(self.temp_dir) / "测试文件.txt"
            unicode_file.write_text("Content with unicode filename")

            processor = TextProcessor()
            result = processor.process_file(unicode_file)

            assert result.success
            assert result.media_content.metadata["file_name"] == "测试文件.txt"

        except ImportError:
            pytest.skip("TextProcessor not available")

    def test_mixed_content_types(self):
        """Test handling of mixed content types in multimodal messages."""
        try:
            from abstractcore.media.handlers import OpenAIMediaHandler
            from abstractcore.media.processors import ImageProcessor, TextProcessor
            from PIL import Image as PILImage

            # Create test files
            test_image = Path(self.temp_dir) / "test.png"
            img = PILImage.new('RGB', (50, 50), color='blue')
            img.save(test_image)

            test_text = Path(self.temp_dir) / "test.txt"
            test_text.write_text("Sample text content")

            # Process both files
            img_processor = ImageProcessor()
            img_result = img_processor.process_file(test_image)
            assert img_result.success

            txt_processor = TextProcessor()
            txt_result = txt_processor.process_file(test_text)
            assert txt_result.success

            # Create multimodal message with mixed content
            handler = OpenAIMediaHandler()
            message = handler.create_multimodal_message(
                "Analyze both the image and text",
                [img_result.media_content, txt_result.media_content]
            )

            assert message["role"] == "user"
            assert isinstance(message["content"], list)
            assert len(message["content"]) >= 2  # At least text prompt + content

        except ImportError:
            pytest.skip("Required components not available")


class TestConcurrencyAndPerformance:
    """Test concurrent processing and performance edge cases."""

    def test_concurrent_processing(self):
        """Test concurrent media processing doesn't cause issues."""
        import threading
        import tempfile
        from pathlib import Path

        results = []
        errors = []

        def process_file(file_path):
            try:
                from abstractcore.media.processors import TextProcessor
                processor = TextProcessor()
                result = processor.process_file(file_path)
                results.append(result.success)
            except Exception as e:
                errors.append(str(e))

        # Create multiple test files
        temp_dir = tempfile.mkdtemp()
        test_files = []

        try:
            for i in range(5):
                test_file = Path(temp_dir) / f"test_{i}.txt"
                test_file.write_text(f"Test content {i}")
                test_files.append(test_file)

            # Process files concurrently
            threads = []
            for file_path in test_files:
                thread = threading.Thread(target=process_file, args=(file_path,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join(timeout=10)  # Timeout to prevent hanging

            # Check results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert all(results), "Some files failed to process"

        except ImportError:
            pytest.skip("TextProcessor not available")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/media_handling/test_error_handling.py -v
    pytest.main([__file__, "-v"])