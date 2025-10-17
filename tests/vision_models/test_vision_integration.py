"""
Comprehensive vision integration tests for AbstractCore.

Tests the complete vision handling pipeline across all providers and models,
including real-world scenarios and edge cases.
"""

import pytest
import tempfile
import json
import os
import time
from pathlib import Path
from PIL import Image

# Import AbstractCore components
from abstractcore import create_llm
from abstractcore.media import AutoMediaHandler, process_file
from abstractcore.media.utils import scale_image_for_model, ScalingMode, get_optimal_size_for_model
from abstractcore.media.processors import ImageProcessor
from abstractcore.media.types import MediaType


class TestVisionIntegration:
    """Integration tests across all vision models and providers."""

    # All vision models to test
    ALL_VISION_MODELS = {
        "lmstudio": [
            "qwen/qwen3-vl-8b",
            "qwen/qwen3-vl-30b",
            "qwen/qwen2.5-vl-7b",
            "google/gemma-3n-e4b"
        ],
        "ollama": [
            "qwen2.5vl:7b",
            "gemma3:4b",
            "gemma3:4b-it-qat",
            "gemma3n:e4b",
            "gemma3n:e2b"
        ]
    }

    @pytest.fixture(scope="class")
    def comprehensive_test_images(self):
        """Create a comprehensive set of test images for integration testing."""
        temp_dir = Path(tempfile.mkdtemp())
        images = {}

        # Real-world test scenarios
        test_scenarios = [
            # Standard sizes
            ("icon", (64, 64), "RGB", "Simple icon size"),
            ("thumbnail", (150, 150), "RGB", "Thumbnail size"),
            ("social_media", (400, 400), "RGB", "Social media square"),
            ("photo_small", (800, 600), "RGB", "Small photo"),
            ("photo_medium", (1920, 1080), "RGB", "HD photo"),
            ("photo_large", (3840, 2160), "RGB", "4K photo"),

            # Model-specific optimal sizes
            ("gemma_optimal", (896, 896), "RGB", "Optimal for Gemma models"),
            ("qwen_small", (1024, 768), "RGB", "Good for Qwen models"),
            ("qwen_large", (3584, 3584), "RGB", "Max Qwen 2.5-VL"),

            # Aspect ratios
            ("ultrawide", (2560, 1080), "RGB", "Ultrawide monitor"),
            ("mobile_portrait", (1080, 1920), "RGB", "Mobile portrait"),
            ("square_large", (2048, 2048), "RGB", "Large square"),

            # Special cases
            ("transparent", (512, 512), "RGBA", "Transparent PNG"),
            ("very_small", (32, 32), "RGB", "Very small image"),
            ("very_tall", (400, 1600), "RGB", "Very tall image"),
            ("very_wide", (1600, 400), "RGB", "Very wide image")
        ]

        for name, size, mode, description in test_scenarios:
            # Create distinctive test pattern based on size
            img = Image.new(mode, size, color=(70, 130, 180, 180) if mode == "RGBA" else (70, 130, 180))

            # Add recognizable pattern
            pixels = img.load()

            # Create a more complex pattern for better visual testing
            center_x, center_y = size[0] // 2, size[1] // 2
            for x in range(size[0]):
                for y in range(size[1]):
                    # Distance from center
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

                    # Create concentric pattern
                    if int(dist) % 20 < 10:
                        if pixels and x < size[0] and y < size[1]:
                            pixels[x, y] = (255, 215, 0, 200) if mode == "RGBA" else (255, 215, 0)

                    # Add corner markers
                    if (x < 20 or x >= size[0] - 20) and (y < 20 or y >= size[1] - 20):
                        if pixels and x < size[0] and y < size[1]:
                            pixels[x, y] = (255, 0, 0, 255) if mode == "RGBA" else (255, 0, 0)

            # Save as multiple formats
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
                images[f"{name}_{format_ext}"] = {
                    'path': file_path,
                    'description': description,
                    'original_size': size,
                    'format': format_ext,
                    'mode': mode
                }

        yield images

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture(scope="class")
    def auto_media_handler(self):
        """Create AutoMediaHandler for testing."""
        return AutoMediaHandler()

    def test_all_models_capability_detection(self):
        """Test that all specified models are detected with proper capabilities."""
        from abstractcore.media.capabilities import get_media_capabilities

        all_models = []
        for provider_models in self.ALL_VISION_MODELS.values():
            all_models.extend(provider_models)

        for model_name in all_models:
            caps = get_media_capabilities(model_name)

            # All should support vision
            assert caps.vision_support, f"Model {model_name} should support vision"

            # Should have reasonable resolution information
            assert hasattr(caps, 'vision_support'), f"Model {model_name} missing vision capabilities"

            print(f"✓ {model_name}: vision_support={caps.vision_support}")

    def test_cross_provider_scaling_consistency(self, comprehensive_test_images):
        """Test that similar models across providers scale consistently."""

        # Compare Qwen models across providers
        qwen_models = {
            "lmstudio": "qwen/qwen2.5-vl-7b",
            "ollama": "qwen2.5vl:7b"
        }

        # Compare Gemma models (conceptually similar)
        gemma_models = {
            "lmstudio": "google/gemma-3n-e4b",
            "ollama": "gemma3:4b"
        }

        test_image_info = comprehensive_test_images["photo_medium_jpg"]
        original_img = Image.open(test_image_info['path'])

        print("\n--- Cross-provider scaling comparison ---")

        # Test Qwen model consistency
        qwen_sizes = {}
        for provider, model in qwen_models.items():
            scaled = scale_image_for_model(original_img, model, ScalingMode.FIT)
            qwen_sizes[provider] = scaled.size
            print(f"Qwen {provider}: {original_img.size} -> {scaled.size}")

        # Test Gemma model consistency
        gemma_sizes = {}
        for provider, model in gemma_models.items():
            scaled = scale_image_for_model(original_img, model, ScalingMode.FIT)
            gemma_sizes[provider] = scaled.size
            print(f"Gemma {provider}: {original_img.size} -> {scaled.size}")

        # Gemma models should be more constrained than Qwen
        for provider in ["lmstudio", "ollama"]:
            if provider in gemma_sizes and provider in qwen_sizes:
                gemma_max = max(gemma_sizes[provider])
                qwen_max = max(qwen_sizes[provider])
                print(f"  {provider}: Gemma max={gemma_max}, Qwen max={qwen_max}")

    def test_auto_media_handler_integration(self, comprehensive_test_images, auto_media_handler):
        """Test AutoMediaHandler with various image types."""

        test_cases = [
            "icon_jpg",
            "photo_medium_jpg",
            "gemma_optimal_png",
            "transparent_png"
        ]

        for test_case in test_cases:
            if test_case in comprehensive_test_images:
                image_info = comprehensive_test_images[test_case]
                image_path = image_info['path']

                # Process with AutoMediaHandler
                result = auto_media_handler.process_file(image_path)

                assert result.success, f"AutoMediaHandler failed for {test_case}: {result.error_message}"
                assert result.media_content is not None
                assert result.media_content.media_type == MediaType.IMAGE

                print(f"✓ AutoMediaHandler processed {test_case}: {image_info['original_size']}")

    def test_batch_processing_performance(self, comprehensive_test_images):
        """Test performance with batch processing of multiple images."""

        # Select a subset of images for performance testing
        test_images = [
            "icon_jpg",
            "thumbnail_jpg",
            "social_media_jpg",
            "photo_small_jpg"
        ]

        model_name = "qwen/qwen3-vl-8b"  # Use consistent model

        start_time = time.time()

        results = []
        for test_case in test_images:
            if test_case in comprehensive_test_images:
                image_info = comprehensive_test_images[test_case]
                original_img = Image.open(image_info['path'])

                scaled = scale_image_for_model(original_img, model_name, ScalingMode.FIT)
                results.append(scaled.size)

        end_time = time.time()
        processing_time = end_time - start_time

        assert len(results) == len(test_images)
        assert processing_time < 10.0, f"Batch processing too slow: {processing_time}s"

        print(f"✓ Batch processed {len(results)} images in {processing_time:.3f}s")

    def test_edge_case_handling(self, comprehensive_test_images):
        """Test handling of edge cases and unusual image sizes."""

        edge_cases = [
            "very_small_jpg",
            "very_tall_jpg",
            "very_wide_jpg",
            "ultrawide_jpg"
        ]

        models_to_test = [
            "qwen/qwen3-vl-8b",
            "gemma3:4b"
        ]

        for test_case in edge_cases:
            if test_case in comprehensive_test_images:
                image_info = comprehensive_test_images[test_case]
                original_img = Image.open(image_info['path'])

                for model_name in models_to_test:
                    try:
                        scaled = scale_image_for_model(original_img, model_name, ScalingMode.FIT)

                        # Should produce valid output
                        assert scaled.size[0] > 0 and scaled.size[1] > 0

                        print(f"✓ {test_case} with {model_name}: {original_img.size} -> {scaled.size}")

                    except Exception as e:
                        pytest.fail(f"Edge case {test_case} failed with {model_name}: {e}")

    def test_format_conversion_chain(self, comprehensive_test_images):
        """Test that the complete format conversion chain works."""

        processor = ImageProcessor()

        # Test PNG -> JPEG conversion with transparency
        if "transparent_png" in comprehensive_test_images:
            transparent_image = comprehensive_test_images["transparent_png"]['path']

            # Process with JPEG target format
            result = processor.process_for_model(
                transparent_image,
                "qwen/qwen3-vl-8b",
                target_format="jpeg"
            )

            assert result.content is not None
            assert result.metadata.get('target_format') == 'jpeg'

            print("✓ Transparency -> JPEG conversion successful")

    @pytest.mark.skipif(
        not os.getenv('TEST_WITH_REAL_MODELS'),
        reason="Real model testing disabled. Set TEST_WITH_REAL_MODELS=1 to enable"
    )
    def test_end_to_end_real_model_integration(self, comprehensive_test_images):
        """Test complete end-to-end integration with real models."""

        # Test one model from each provider if available
        test_scenarios = [
            ("lmstudio", "qwen/qwen3-vl-8b", "http://localhost:1234/v1"),
            ("ollama", "qwen2.5vl:7b", "http://localhost:11434")
        ]

        test_image = comprehensive_test_images["photo_small_jpg"]['path']

        for provider, model, base_url in test_scenarios:
            print(f"\n--- Testing {provider} with {model} ---")

            try:
                # Create LLM instance
                llm = create_llm(provider, model=model, base_url=base_url)

                # Test with image
                response = llm.generate(
                    "Describe what you see in this image. What patterns and shapes are visible?",
                    media=[str(test_image)]
                )

                # Verify response
                assert response.content is not None
                assert len(response.content) > 10  # Should be substantive

                print(f"✓ {provider} {model} responded with {len(response.content)} characters")
                print(f"  Preview: {response.content[:150]}...")

            except Exception as e:
                pytest.skip(f"{provider} not available: {e}")

    def test_scaling_mode_comprehensive(self, comprehensive_test_images):
        """Comprehensive test of all scaling modes across different models."""

        test_image_info = comprehensive_test_images["photo_medium_jpg"]
        original_img = Image.open(test_image_info['path'])

        modes = [ScalingMode.FIT, ScalingMode.PAD, ScalingMode.FILL, ScalingMode.CROP_CENTER]
        models = ["qwen/qwen3-vl-8b", "google/gemma-3n-e4b", "gemma3:4b"]

        results = {}

        for model in models:
            results[model] = {}
            for mode in modes:
                scaled = scale_image_for_model(original_img, model, mode)
                results[model][mode.value] = scaled.size

                print(f"  {model} {mode.value}: {original_img.size} -> {scaled.size}")

        # Verify all combinations produced valid results
        for model, mode_results in results.items():
            for mode, size in mode_results.items():
                assert size[0] > 0 and size[1] > 0, f"Invalid size for {model} {mode}: {size}"

    def test_model_optimization_metadata(self, comprehensive_test_images):
        """Test that model optimization metadata is properly recorded."""

        processor = ImageProcessor()
        test_image = comprehensive_test_images["photo_small_jpg"]['path']

        models_to_test = [
            "qwen/qwen3-vl-8b",
            "google/gemma-3n-e4b",
            "qwen2.5vl:7b",
            "gemma3:4b"
        ]

        for model_name in models_to_test:
            result = processor.process_for_model(test_image, model_name)

            # Check required metadata
            required_metadata = [
                'model_optimized',
                'target_model',
                'optimal_size_for_model',
                'scaling_mode',
                'final_size'
            ]

            for key in required_metadata:
                assert key in result.metadata, f"Missing metadata {key} for {model_name}"

            assert result.metadata['model_optimized'] is True
            assert result.metadata['target_model'] == model_name

            print(f"✓ {model_name} metadata complete: {result.metadata['final_size']}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])