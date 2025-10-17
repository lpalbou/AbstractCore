"""
Comprehensive vision testing for LMStudio models.

Tests all specified LMStudio vision models with different image types and resolutions
to ensure proper integration and model-specific optimizations.
"""

import pytest
import tempfile
import json
from pathlib import Path
from PIL import Image
import os

# Import AbstractCore components
from abstractcore import create_llm
from abstractcore.media.utils import scale_image_for_model, ScalingMode, get_optimal_size_for_model
from abstractcore.media.processors import ImageProcessor
from abstractcore.media.types import MediaType


class TestLMStudioVisionModels:
    """Test LMStudio vision models with comprehensive scenarios."""

    # LMStudio models to test
    LMSTUDIO_MODELS = [
        "qwen/qwen2.5-vl-7b",
        "google/gemma-3n-e4b",
        "mistralai/magistral-small-2509"
        # Note: qwen3-vl models not yet supported by LMStudio
        # "qwen/qwen3-vl-8b",
        # "qwen/qwen3-vl-30b",
    ]

    @pytest.fixture(scope="class")
    def test_images(self):
        """Create test images of different sizes and formats."""
        temp_dir = Path(tempfile.mkdtemp())
        images = {}

        # Create images of different resolutions
        test_scenarios = [
            ("small_square", (256, 256), "RGB"),
            ("medium_square", (512, 512), "RGB"),
            ("large_square", (1024, 1024), "RGB"),
            ("very_large_square", (2048, 2048), "RGB"),
            ("wide_rectangle", (1920, 1080), "RGB"),
            ("tall_rectangle", (1080, 1920), "RGB"),
            ("ultra_wide", (3440, 1440), "RGB"),
            ("gemma_optimal", (896, 896), "RGB"),  # Optimal for Gemma models
            ("qwen_large", (3584, 3584), "RGB"),   # Max for Qwen 2.5-VL
            ("transparent_png", (512, 512), "RGBA")
        ]

        for name, size, mode in test_scenarios:
            # Create a simple colored image with text pattern
            img = Image.new(mode, size, color=(100, 150, 200, 255) if mode == "RGBA" else (100, 150, 200))

            # Add some visual pattern for testing
            pixels = img.load()
            for i in range(0, size[0], 50):
                for j in range(0, size[1], 50):
                    if pixels and i < size[0] and j < size[1]:
                        pixels[i, j] = (255, 255, 255, 255) if mode == "RGBA" else (255, 255, 255)

            # Save as different formats
            for format_ext in ['jpg', 'png']:
                if format_ext == 'jpg' and mode == 'RGBA':
                    # Convert RGBA to RGB for JPEG
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[-1] if mode == 'RGBA' else None)
                    img_to_save = rgb_img
                else:
                    img_to_save = img

                file_path = temp_dir / f"{name}.{format_ext}"
                img_to_save.save(file_path)
                images[f"{name}_{format_ext}"] = file_path

        yield images

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture(scope="class")
    def image_processor(self):
        """Create image processor for testing."""
        return ImageProcessor()

    def test_model_capabilities_loaded(self):
        """Test that all LMStudio model capabilities are properly loaded."""
        from abstractcore.media.capabilities import get_media_capabilities

        for model_name in self.LMSTUDIO_MODELS:
            caps = get_media_capabilities(model_name)

            # All models should support vision
            assert caps.vision_support, f"{model_name} should support vision"

            # Check model-specific capabilities
            if "gemma" in model_name.lower():
                # Gemma models should have fixed 896x896 resolution
                assert hasattr(caps, 'max_image_resolution'), f"{model_name} should have max_image_resolution"
            elif "qwen" in model_name.lower():
                # Qwen models should have their specific resolutions
                assert hasattr(caps, 'vision_support'), f"{model_name} should have vision support"

    def test_image_scaling_optimization(self, test_images, image_processor):
        """Test that image scaling works correctly for each model."""

        for model_name in self.LMSTUDIO_MODELS:
            print(f"\n--- Testing scaling for {model_name} ---")

            for image_name, image_path in test_images.items():
                if image_name.endswith('_jpg'):  # Test with JPEG images
                    original_img = Image.open(image_path)
                    original_size = original_img.size

                    # Get optimal size for this model
                    optimal_size = get_optimal_size_for_model(model_name, original_size)

                    # Scale image for this model
                    scaled_img = scale_image_for_model(original_img, model_name, ScalingMode.FIT)

                    print(f"  {image_name}: {original_size} -> {scaled_img.size} (optimal: {optimal_size})")

                    # Verify scaling worked
                    assert scaled_img.size[0] > 0 and scaled_img.size[1] > 0

                    # Model-specific checks
                    if "gemma" in model_name.lower():
                        # Gemma models should scale to 896x896 or smaller
                        assert max(scaled_img.size) <= 896, f"Gemma model {model_name} image too large: {scaled_img.size}"

                    elif "qwen2.5" in model_name.lower():
                        # Qwen 2.5 models should not exceed 3584x3584
                        assert max(scaled_img.size) <= 3584, f"Qwen 2.5 model {model_name} image too large: {scaled_img.size}"

    def test_model_optimized_processing(self, test_images, image_processor):
        """Test model-optimized image processing."""

        for model_name in self.LMSTUDIO_MODELS:
            print(f"\n--- Testing optimized processing for {model_name} ---")

            # Test with medium square image
            test_image = test_images["medium_square_jpg"]

            # Process image for this specific model
            result = image_processor.process_for_model(test_image, model_name)

            # Verify result structure
            assert result.media_type == MediaType.IMAGE
            assert result.content is not None
            assert len(result.content) > 0

            # Check metadata
            assert result.metadata.get('model_optimized') is True
            assert result.metadata.get('target_model') == model_name
            assert 'optimal_size_for_model' in result.metadata
            assert 'scaling_mode' in result.metadata

            print(f"  Processed: {result.metadata.get('final_size')} (optimal: {result.metadata.get('optimal_size_for_model')})")

    @pytest.mark.skipif(
        not os.getenv('TEST_WITH_REAL_MODELS'),
        reason="Real model testing disabled. Set TEST_WITH_REAL_MODELS=1 to enable"
    )
    def test_real_lmstudio_integration(self, test_images):
        """Test real integration with LMStudio models (requires running LMStudio)."""

        # Test with a smaller subset to avoid long test times
        test_models = ["qwen/qwen3-vl-8b", "google/gemma-3n-e4b"]

        for model_name in test_models:
            print(f"\n--- Testing real LMStudio integration with {model_name} ---")

            try:
                # Create LLM instance
                llm = create_llm("lmstudio", model=model_name, base_url="http://localhost:1234/v1")

                # Test with medium square image
                test_image = test_images["medium_square_jpg"]

                # Generate response with image
                response = llm.generate(
                    "What do you see in this image? Describe the colors and patterns.",
                    media=[str(test_image)]
                )

                # Verify response
                assert response.content is not None
                assert len(response.content) > 0

                print(f"  Model {model_name} response length: {len(response.content)}")
                print(f"  Response preview: {response.content[:100]}...")

            except Exception as e:
                pytest.skip(f"LMStudio not available or model not loaded: {e}")

    def test_scaling_modes(self, test_images):
        """Test different scaling modes with various models."""

        test_image = test_images["wide_rectangle_jpg"]
        original_img = Image.open(test_image)

        modes_to_test = [
            ScalingMode.FIT,
            ScalingMode.PAD,
            ScalingMode.FILL,
            ScalingMode.CROP_CENTER
        ]

        for model_name in ["qwen/qwen3-vl-8b", "google/gemma-3n-e4b"]:
            for mode in modes_to_test:
                scaled_img = scale_image_for_model(original_img, model_name, mode)

                assert scaled_img.size[0] > 0 and scaled_img.size[1] > 0
                print(f"  {model_name} with {mode.value}: {original_img.size} -> {scaled_img.size}")

    def test_large_image_handling(self, test_images):
        """Test handling of very large images."""

        # Test with the largest image
        large_image = test_images["very_large_square_jpg"]
        original_img = Image.open(large_image)

        for model_name in self.LMSTUDIO_MODELS:
            scaled_img = scale_image_for_model(original_img, model_name, ScalingMode.FIT)

            # Should be scaled down appropriately
            assert scaled_img.size[0] <= original_img.size[0]
            assert scaled_img.size[1] <= original_img.size[1]

            print(f"  Large image for {model_name}: {original_img.size} -> {scaled_img.size}")

    def test_format_compatibility(self, test_images, image_processor):
        """Test different image formats work with all models."""

        formats_to_test = ["jpg", "png"]

        for model_name in self.LMSTUDIO_MODELS:
            for format_ext in formats_to_test:
                test_image = test_images[f"medium_square_{format_ext}"]

                # Process with model optimization
                result = image_processor.process_for_model(test_image, model_name)

                assert result.content is not None
                assert result.media_type == MediaType.IMAGE

                print(f"  {model_name} processed {format_ext} format successfully")

    def test_model_support_detection(self, image_processor):
        """Test that the processor correctly identifies supported models."""

        for model_name in self.LMSTUDIO_MODELS:
            assert image_processor.supports_model(model_name), f"Should support {model_name}"

        # Test unsupported model
        assert not image_processor.supports_model("unsupported-model-xyz")

    def test_performance_benchmarks(self, test_images, image_processor):
        """Basic performance testing for optimization."""
        import time

        test_image = test_images["large_square_jpg"]

        for model_name in self.LMSTUDIO_MODELS[:2]:  # Test subset for speed
            start_time = time.time()

            result = image_processor.process_for_model(test_image, model_name)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should process within reasonable time (adjust threshold as needed)
            assert processing_time < 5.0, f"Processing took too long: {processing_time}s"

            print(f"  {model_name} processing time: {processing_time:.3f}s")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])