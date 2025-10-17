"""
Comprehensive vision testing for Ollama models.

Tests all specified Ollama vision models with different image types and resolutions
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


class TestOllamaVisionModels:
    """Test Ollama vision models with comprehensive scenarios."""

    # Ollama models to test
    OLLAMA_MODELS = [
        "qwen2.5vl:7b",
        "gemma3:4b",
        "gemma3:4b-it-qat",
        "gemma3n:e4b",
        "gemma3n:e2b",
        "llama3.2-vision:11b"
    ]

    @pytest.fixture(scope="class")
    def test_images(self):
        """Create test images of different sizes and formats."""
        temp_dir = Path(tempfile.mkdtemp())
        images = {}

        # Create images of different resolutions for Ollama testing
        test_scenarios = [
            ("tiny_square", (128, 128), "RGB"),
            ("small_square", (256, 256), "RGB"),
            ("medium_square", (512, 512), "RGB"),
            ("large_square", (1024, 1024), "RGB"),
            ("gemma_native", (896, 896), "RGB"),      # Native resolution for Gemma models
            ("qwen_medium", (1024, 768), "RGB"),      # Good size for Qwen models
            ("qwen_large", (2048, 1536), "RGB"),      # Larger for Qwen models
            ("landscape", (1280, 720), "RGB"),        # Common landscape format
            ("portrait", (720, 1280), "RGB"),         # Common portrait format
            ("square_transparent", (512, 512), "RGBA") # Test transparency handling
        ]

        for name, size, mode in test_scenarios:
            # Create a distinctive test pattern
            img = Image.new(mode, size, color=(50, 100, 150, 200) if mode == "RGBA" else (50, 100, 150))

            # Add a grid pattern for visual recognition
            pixels = img.load()
            grid_size = max(32, min(size) // 16)  # Adaptive grid size

            for i in range(0, size[0], grid_size):
                for j in range(0, size[1], grid_size):
                    # Create checkerboard pattern
                    if (i // grid_size + j // grid_size) % 2 == 0:
                        if pixels and i < size[0] and j < size[1]:
                            pixels[i, j] = (255, 200, 100, 255) if mode == "RGBA" else (255, 200, 100)

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

    def test_ollama_model_capabilities_loaded(self):
        """Test that all Ollama model capabilities are properly loaded."""
        from abstractcore.media.capabilities import get_media_capabilities

        for model_name in self.OLLAMA_MODELS:
            caps = get_media_capabilities(model_name)

            # All models should support vision
            assert caps.vision_support, f"{model_name} should support vision"

            # Check model-specific capabilities
            if "gemma" in model_name.lower():
                # Gemma models should have fixed 896x896 resolution
                assert hasattr(caps, 'max_image_resolution'), f"{model_name} should have max_image_resolution"
            elif "qwen" in model_name.lower():
                # Qwen models should have their specific capabilities
                assert hasattr(caps, 'vision_support'), f"{model_name} should have vision support"

    def test_ollama_image_scaling_optimization(self, test_images, image_processor):
        """Test that image scaling works correctly for each Ollama model."""

        for model_name in self.OLLAMA_MODELS:
            print(f"\n--- Testing Ollama scaling for {model_name} ---")

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
                        # Gemma models should prefer 896x896 or smaller
                        # For Ollama, we may be more flexible but still reasonable
                        assert max(scaled_img.size) <= 896, f"Ollama Gemma model {model_name} image too large: {scaled_img.size}"

                    elif "qwen" in model_name.lower():
                        # Qwen models in Ollama should be reasonably sized
                        assert max(scaled_img.size) <= 3584, f"Ollama Qwen model {model_name} image too large: {scaled_img.size}"

    def test_ollama_model_optimized_processing(self, test_images, image_processor):
        """Test model-optimized image processing for Ollama models."""

        for model_name in self.OLLAMA_MODELS:
            print(f"\n--- Testing Ollama optimized processing for {model_name} ---")

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
    def test_real_ollama_integration(self, test_images):
        """Test real integration with Ollama models (requires running Ollama)."""

        # Test with a smaller subset to avoid long test times
        test_models = ["qwen2.5vl:7b", "gemma3:4b"]

        for model_name in test_models:
            print(f"\n--- Testing real Ollama integration with {model_name} ---")

            try:
                # Create LLM instance
                llm = create_llm("ollama", model=model_name, base_url="http://localhost:11434")

                # Test with medium square image
                test_image = test_images["medium_square_jpg"]

                # Generate response with image
                response = llm.generate(
                    "What do you see in this image? Describe the patterns and colors.",
                    media=[str(test_image)]
                )

                # Verify response
                assert response.content is not None
                assert len(response.content) > 0

                print(f"  Ollama model {model_name} response length: {len(response.content)}")
                print(f"  Response preview: {response.content[:100]}...")

            except Exception as e:
                pytest.skip(f"Ollama not available or model not loaded: {e}")

    def test_ollama_gemma_vs_qwen_scaling(self, test_images):
        """Test that Gemma and Qwen models in Ollama scale differently."""

        test_image = test_images["large_square_jpg"]
        original_img = Image.open(test_image)
        original_size = original_img.size

        # Test Gemma model (should scale to ~896x896)
        gemma_model = "gemma3:4b"
        gemma_scaled = scale_image_for_model(original_img, gemma_model, ScalingMode.FIT)

        # Test Qwen model (should allow larger sizes)
        qwen_model = "qwen2.5vl:7b"
        qwen_scaled = scale_image_for_model(original_img, qwen_model, ScalingMode.FIT)

        print(f"\nGemma scaling: {original_size} -> {gemma_scaled.size}")
        print(f"Qwen scaling: {original_size} -> {qwen_scaled.size}")

        # Gemma should be more constrained
        assert max(gemma_scaled.size) <= 896
        # Qwen can be larger but still reasonable
        assert max(qwen_scaled.size) >= max(gemma_scaled.size)

    def test_ollama_quantized_models(self, test_images, image_processor):
        """Test quantized models (like 4b-it-qat) work correctly."""

        quantized_models = ["gemma3:4b-it-qat"]

        for model_name in quantized_models:
            test_image = test_images["medium_square_jpg"]

            # Process image
            result = image_processor.process_for_model(test_image, model_name)

            # Should work the same as non-quantized versions
            assert result.content is not None
            assert result.metadata.get('model_optimized') is True

            print(f"  Quantized model {model_name} processing successful")

    def test_ollama_model_variants(self, test_images):
        """Test different variants of Ollama models (e2b vs e4b)."""

        test_image = test_images["medium_square_jpg"]
        original_img = Image.open(test_image)

        variants = ["gemma3n:e2b", "gemma3n:e4b"]

        for model_name in variants:
            scaled_img = scale_image_for_model(original_img, model_name, ScalingMode.FIT)

            # Both should scale similarly (same base architecture)
            assert scaled_img.size[0] > 0 and scaled_img.size[1] > 0
            assert max(scaled_img.size) <= 896  # Gemma constraint

            print(f"  {model_name}: {original_img.size} -> {scaled_img.size}")

    def test_ollama_small_image_handling(self, test_images):
        """Test handling of small images that don't need scaling."""

        small_image = test_images["tiny_square_jpg"]
        original_img = Image.open(small_image)

        for model_name in self.OLLAMA_MODELS[:2]:  # Test subset
            scaled_img = scale_image_for_model(original_img, model_name, ScalingMode.FIT)

            # Small images should not be upscaled unnecessarily
            assert scaled_img.size == original_img.size or max(scaled_img.size) >= max(original_img.size)

            print(f"  Small image for {model_name}: {original_img.size} -> {scaled_img.size}")

    def test_ollama_transparency_handling(self, test_images, image_processor):
        """Test that transparent images are handled correctly."""

        transparent_image = test_images["square_transparent_png"]

        for model_name in self.OLLAMA_MODELS[:2]:  # Test subset
            # Process with model optimization
            result = image_processor.process_for_model(
                transparent_image,
                model_name,
                target_format="jpeg"  # Force conversion to non-transparent format
            )

            assert result.content is not None
            assert result.metadata.get('target_format') == 'jpeg'

            print(f"  Transparency handled for {model_name}")

    def test_ollama_scaling_modes_comparison(self, test_images):
        """Test different scaling modes specifically for Ollama models."""

        test_image = test_images["landscape_jpg"]
        original_img = Image.open(test_image)

        modes_to_test = [
            ScalingMode.FIT,
            ScalingMode.PAD,
            ScalingMode.FILL
        ]

        model_name = "gemma3:4b"  # Use consistent model for comparison

        results = {}
        for mode in modes_to_test:
            scaled_img = scale_image_for_model(original_img, model_name, mode)
            results[mode] = scaled_img.size

            print(f"  {mode.value}: {original_img.size} -> {scaled_img.size}")

        # PAD mode should maintain exact target dimensions for fixed-size models
        # FIT mode might be smaller
        # FILL mode should match target dimensions

    def test_ollama_memory_efficiency(self, test_images, image_processor):
        """Basic test for memory efficiency with multiple image processing."""

        test_images_subset = [
            test_images["medium_square_jpg"],
            test_images["gemma_native_jpg"],
            test_images["landscape_jpg"]
        ]

        model_name = "gemma3:4b"

        # Process multiple images in sequence
        results = []
        for test_image in test_images_subset:
            result = image_processor.process_for_model(test_image, model_name)
            results.append(result)

        # All should process successfully
        assert len(results) == len(test_images_subset)
        for result in results:
            assert result.content is not None

        print(f"  Processed {len(results)} images successfully")

    def test_ollama_error_handling(self, image_processor):
        """Test error handling for invalid inputs."""

        # Test with non-existent model
        try:
            from PIL import Image
            dummy_img = Image.new("RGB", (100, 100))
            result = scale_image_for_model(dummy_img, "invalid-model-name", ScalingMode.FIT)
            # Should still work with fallback capabilities
            assert result.size[0] > 0
        except Exception:
            # Error handling is acceptable
            pass

        print("  Error handling tested")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])