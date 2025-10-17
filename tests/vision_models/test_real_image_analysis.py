"""
Real Image Vision Analysis Tests for AbstractCore.

Tests all specified vision models with actual images, measuring their ability
to accurately describe visual content through keyword overlap analysis.
"""

import pytest
import os
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import tempfile
import io
from PIL import Image

# Import AbstractCore components
from abstractcore import create_llm
from abstractcore.media.processors import ImageProcessor
from abstractcore.media.utils import scale_image_for_model, ScalingMode


class RealImageVisionTester:
    """Comprehensive real image vision testing with keyword overlap measurement."""

    # Reference keywords for each mystery image (hand-analyzed)
    REFERENCE_KEYWORDS = {
        "mystery1_mp.jpg": {
            "primary": ["mountain", "path", "trail", "fence", "railing", "hiking", "landscape", "outdoor", "nature"],
            "secondary": ["sun", "sky", "clouds", "dirt", "gravel", "vegetation", "plants", "sunny", "blue", "white", "green", "brown"],
            "description": "Mountain hiking trail with wooden fence and scenic landscape"
        },
        "mystery2_sc.jpg": {
            "primary": ["cat", "helmet", "space", "astronaut", "transparent", "dome", "controls", "spacecraft"],
            "secondary": ["glass", "buttons", "pet", "animal", "eyes", "futuristic", "science fiction", "sitting", "calm"],
            "description": "Cat inside space helmet with control panels"
        },
        "mystery3_us.jpg": {
            "primary": ["sunset", "dusk", "street", "urban", "city", "lamps", "lighting", "trees", "buildings"],
            "secondary": ["architecture", "sky", "pink", "orange", "purple", "colorful", "atmospheric", "pathway", "sidewalk", "bare trees", "evening"],
            "description": "Urban street at sunset with atmospheric lighting"
        },
        "mystery4_wh.jpg": {
            "primary": ["whale", "humpback", "breach", "breaching", "ocean", "sea", "water", "splash", "marine", "mammal"],
            "secondary": ["large", "gray", "grooves", "wildlife", "nature", "dramatic", "action", "airborne"],
            "description": "Humpback whale breaching from ocean water"
        }
    }

    # Models to test by provider
    LMSTUDIO_MODELS = [
        "qwen/qwen2.5-vl-7b",
        "google/gemma-3n-e4b",
        "mistralai/magistral-small-2509"
        # Note: qwen3-vl models not yet supported by LMStudio
        # "qwen/qwen3-vl-8b",
        # "qwen/qwen3-vl-30b",
    ]

    OLLAMA_MODELS = [
        "qwen2.5vl:7b",
        "gemma3:4b",
        "gemma3:4b-it-qat",
        "gemma3n:e4b",
        "gemma3n:e2b",
        "llama3.2-vision:11b"
    ]

    HUGGINGFACE_MODELS = [
        # Note: Qwen3-VL models require newer transformers architecture (as of 2025-10-17)
        # "Qwen/Qwen3-VL-8B-Instruct-FP8"  # Architecture too new for current transformers
    ]

    def __init__(self):
        self.processor = ImageProcessor()
        self.test_results = {}
        self.image_paths = {}

    def setup_images(self) -> Dict[str, Path]:
        """Setup paths to test images."""
        base_path = Path("tests/vision_examples")

        if not base_path.exists():
            raise FileNotFoundError(f"Test images directory not found: {base_path}")

        images = {}
        for filename in self.REFERENCE_KEYWORDS.keys():
            image_path = base_path / filename
            if image_path.exists():
                images[filename] = image_path
            else:
                print(f"⚠️  Warning: Test image not found: {image_path}")

        return images

    def extract_keywords_from_response(self, response_text: str) -> Set[str]:
        """
        Extract meaningful keywords from model response.

        Args:
            response_text: Raw response from vision model

        Returns:
            Set of lowercase keywords found in response
        """
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', response_text.lower())

        # Split into words
        words = text.split()

        # Filter out common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'am', 'very', 'quite', 'rather', 'too', 'so', 'just', 'now', 'then', 'here', 'there', 'where', 'when', 'how', 'what', 'which', 'who', 'why', 'as', 'like', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'some', 'any', 'no', 'not', 'only', 'own', 'same', 'such', 'than', 'more', 'most', 'other', 'another', 'much', 'many', 'little', 'few', 'several', 'all', 'both', 'each', 'every', 'either', 'neither', 'none', 'nothing', 'something', 'anything', 'everything', 'someone', 'anyone', 'everyone', 'somewhere', 'anywhere', 'everywhere', 'sometime', 'anytime', 'everytime'
        }

        # Filter meaningful words (length >= 3, not stop words)
        keywords = {word for word in words if len(word) >= 3 and word not in stop_words}

        return keywords

    def calculate_overlap(self, model_keywords: Set[str], reference_keywords: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate keyword overlap percentages.

        Args:
            model_keywords: Keywords extracted from model response
            reference_keywords: Reference keywords (primary and secondary)

        Returns:
            Dictionary with overlap percentages
        """
        primary_ref = set(kw.lower() for kw in reference_keywords['primary'])
        secondary_ref = set(kw.lower() for kw in reference_keywords['secondary'])
        all_ref = primary_ref | secondary_ref

        # Calculate overlaps
        primary_overlap = len(model_keywords & primary_ref) / len(primary_ref) if primary_ref else 0
        secondary_overlap = len(model_keywords & secondary_ref) / len(secondary_ref) if secondary_ref else 0
        total_overlap = len(model_keywords & all_ref) / len(all_ref) if all_ref else 0

        # Calculate precision (how many model keywords are relevant)
        precision = len(model_keywords & all_ref) / len(model_keywords) if model_keywords else 0

        return {
            'primary_recall': primary_overlap,
            'secondary_recall': secondary_overlap,
            'total_recall': total_overlap,
            'precision': precision,
            'f1_score': 2 * (precision * total_overlap) / (precision + total_overlap) if (precision + total_overlap) > 0 else 0,
            'matched_primary': list(model_keywords & primary_ref),
            'matched_secondary': list(model_keywords & secondary_ref),
            'total_model_keywords': len(model_keywords),
            'total_reference_keywords': len(all_ref)
        }

    def resize_image_for_model(self, image_path: Path, model_name: str) -> Image.Image:
        """
        Resize image in memory for specific model (no disk writes).

        Args:
            image_path: Path to original image
            model_name: Target model name

        Returns:
            PIL Image optimized for the model
        """
        # Load original image
        original_img = Image.open(image_path)

        # Use our scaling utility to optimize for the model
        scaled_img = scale_image_for_model(original_img, model_name, ScalingMode.FIT)

        return scaled_img

    def test_model_with_image(self, provider: str, model_name: str, image_filename: str, image_path: Path) -> Dict:
        """
        Test a specific model with a specific image.

        Args:
            provider: Provider name (lmstudio, ollama, huggingface)
            model_name: Model identifier
            image_filename: Name of test image file
            image_path: Path to image file

        Returns:
            Test result dictionary
        """
        print(f"🔍 Testing {provider}/{model_name} with {image_filename}")

        try:
            # Resize image for this model (in memory)
            optimized_img = self.resize_image_for_model(image_path, model_name)
            original_size = Image.open(image_path).size
            optimized_size = optimized_img.size

            print(f"   Image: {original_size} -> {optimized_size}")

            # Create LLM instance
            base_urls = {
                "lmstudio": "http://localhost:1234/v1",
                "ollama": "http://localhost:11434",
                "huggingface": None  # Uses default HuggingFace API or local model
            }

            base_url = base_urls.get(provider)
            if base_url:
                llm = create_llm(provider, model=model_name, base_url=base_url)
            else:
                llm = create_llm(provider, model=model_name)

            # Create a temporary file for the optimized image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                optimized_img.convert('RGB').save(temp_file.name, 'JPEG', quality=90)
                temp_path = temp_file.name

            try:
                # Test with standardized prompt
                prompt = "Describe this image in detail. List all the objects, animals, scenery, colors, and activities you can see. Be specific and comprehensive."

                start_time = time.time()
                response = llm.generate(prompt, media=[temp_path])
                end_time = time.time()

                # Extract keywords from response
                model_keywords = self.extract_keywords_from_response(response.content)

                # Calculate overlap with reference keywords
                reference_keywords = self.REFERENCE_KEYWORDS[image_filename]
                overlap_metrics = self.calculate_overlap(model_keywords, reference_keywords)

                result = {
                    'success': True,
                    'model': model_name,
                    'provider': provider,
                    'image': image_filename,
                    'response_length': len(response.content),
                    'response_time': end_time - start_time,
                    'original_size': original_size,
                    'optimized_size': optimized_size,
                    'model_keywords': list(model_keywords),
                    'overlap_metrics': overlap_metrics,
                    'response_preview': response.content[:200] + "..." if len(response.content) > 200 else response.content,
                    'full_response': response.content
                }

                print(f"   ✅ Success: {overlap_metrics['total_recall']:.1%} recall, {overlap_metrics['precision']:.1%} precision")

                # Output the actual model response for analysis
                print(f"   📝 Model response: {response.content[:200]}..." if len(response.content) > 200 else f"   📝 Model response: {response.content}")

                return result

            finally:
                # Clean up temporary file
                os.unlink(temp_path)

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            print(f"   🔧 Error details: Check model availability and compatibility")
            return {
                'success': False,
                'model': model_name,
                'provider': provider,
                'image': image_filename,
                'error': str(e),
                'original_size': Image.open(image_path).size if image_path.exists() else None
            }

    def run_comprehensive_tests(self, test_real_models: bool = False) -> Dict:
        """
        Run comprehensive tests across all models and images.

        Args:
            test_real_models: Whether to test with real model endpoints

        Returns:
            Complete test results
        """
        print("🎯 STARTING COMPREHENSIVE REAL IMAGE VISION TESTS")
        print("=" * 70)

        # Setup images
        self.image_paths = self.setup_images()
        print(f"📸 Found {len(self.image_paths)} test images")

        if not test_real_models:
            print("⚠️  Real model testing disabled. Set TEST_WITH_REAL_MODELS=1 to enable")
            return {"skipped": True, "reason": "Real model testing disabled"}

        all_results = {
            'summary': {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'average_scores': {}
            },
            'by_provider': {},
            'by_image': {},
            'detailed_results': []
        }

        # Test each provider (skip empty model lists)
        provider_model_pairs = [
            ("lmstudio", self.LMSTUDIO_MODELS),
            ("ollama", self.OLLAMA_MODELS),
            ("huggingface", self.HUGGINGFACE_MODELS)
        ]

        for provider, models in provider_model_pairs:
            if not models:  # Skip providers with no models
                print(f"\n🔧 SKIPPING {provider.upper()} MODELS (no models configured)")
                continue
            print(f"\n🔧 TESTING {provider.upper()} MODELS")
            print("-" * 50)

            provider_results = {
                'models_tested': 0,
                'successful_models': 0,
                'average_recall': 0,
                'average_precision': 0,
                'model_results': {}
            }

            for model_name in models:
                model_results = {
                    'images_tested': 0,
                    'successful_images': 0,
                    'average_recall': 0,
                    'average_precision': 0,
                    'image_results': {}
                }

                for image_filename, image_path in self.image_paths.items():
                    result = self.test_model_with_image(provider, model_name, image_filename, image_path)

                    all_results['detailed_results'].append(result)
                    all_results['summary']['total_tests'] += 1

                    if result['success']:
                        all_results['summary']['successful_tests'] += 1
                        model_results['successful_images'] += 1

                        # Accumulate metrics
                        metrics = result['overlap_metrics']
                        model_results['average_recall'] += metrics['total_recall']
                        model_results['average_precision'] += metrics['precision']
                    else:
                        all_results['summary']['failed_tests'] += 1

                    model_results['images_tested'] += 1
                    model_results['image_results'][image_filename] = result

                # Calculate model averages
                if model_results['successful_images'] > 0:
                    model_results['average_recall'] /= model_results['successful_images']
                    model_results['average_precision'] /= model_results['successful_images']
                    provider_results['successful_models'] += 1

                provider_results['models_tested'] += 1
                provider_results['model_results'][model_name] = model_results

                print(f"   {model_name}: {model_results['successful_images']}/{model_results['images_tested']} images, "
                      f"avg recall: {model_results['average_recall']:.1%}")

            # Calculate provider averages
            if provider_results['successful_models'] > 0:
                total_recall = sum(m['average_recall'] for m in provider_results['model_results'].values() if m['successful_images'] > 0)
                total_precision = sum(m['average_precision'] for m in provider_results['model_results'].values() if m['successful_images'] > 0)
                provider_results['average_recall'] = total_recall / provider_results['successful_models']
                provider_results['average_precision'] = total_precision / provider_results['successful_models']

            all_results['by_provider'][provider] = provider_results

        # Organize results by image
        for image_filename in self.image_paths.keys():
            image_results = [r for r in all_results['detailed_results'] if r['image'] == image_filename and r['success']]

            if image_results:
                avg_recall = sum(r['overlap_metrics']['total_recall'] for r in image_results) / len(image_results)
                avg_precision = sum(r['overlap_metrics']['precision'] for r in image_results) / len(image_results)

                all_results['by_image'][image_filename] = {
                    'description': self.REFERENCE_KEYWORDS[image_filename]['description'],
                    'successful_tests': len(image_results),
                    'average_recall': avg_recall,
                    'average_precision': avg_precision,
                    'best_model': max(image_results, key=lambda x: x['overlap_metrics']['f1_score']),
                    'results': image_results
                }

        return all_results

    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive report from test results."""
        if results.get('skipped'):
            return "❌ Tests were skipped. Enable real model testing with TEST_WITH_REAL_MODELS=1"

        report = []
        report.append("🎯 REAL IMAGE VISION ANALYSIS REPORT")
        report.append("=" * 70)

        # Summary
        summary = results['summary']
        report.append(f"\n📊 OVERALL SUMMARY")
        report.append(f"Total tests: {summary['total_tests']}")
        report.append(f"Successful: {summary['successful_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append(f"Success rate: {summary['successful_tests']/summary['total_tests']:.1%}")

        # Provider comparison
        report.append(f"\n🔧 PROVIDER COMPARISON")
        for provider, data in results['by_provider'].items():
            report.append(f"\n{provider.upper()}:")
            report.append(f"  Models tested: {data['models_tested']}")
            report.append(f"  Successful models: {data['successful_models']}")
            if data['successful_models'] > 0:
                report.append(f"  Average recall: {data['average_recall']:.1%}")
                report.append(f"  Average precision: {data['average_precision']:.1%}")

        # Image analysis
        report.append(f"\n📸 IMAGE ANALYSIS")
        for image, data in results['by_image'].items():
            report.append(f"\n{image}: {data['description']}")
            report.append(f"  Successful tests: {data['successful_tests']}")
            report.append(f"  Average recall: {data['average_recall']:.1%}")
            report.append(f"  Average precision: {data['average_precision']:.1%}")
            best = data['best_model']
            report.append(f"  Best model: {best['provider']}/{best['model']} (F1: {best['overlap_metrics']['f1_score']:.1%})")

        # Top performers
        successful_results = [r for r in results['detailed_results'] if r['success']]
        if successful_results:
            report.append(f"\n🏆 TOP PERFORMERS")

            # Best by F1 score
            best_f1 = max(successful_results, key=lambda x: x['overlap_metrics']['f1_score'])
            report.append(f"Best F1 Score: {best_f1['provider']}/{best_f1['model']} - {best_f1['overlap_metrics']['f1_score']:.1%}")

            # Best by recall
            best_recall = max(successful_results, key=lambda x: x['overlap_metrics']['total_recall'])
            report.append(f"Best Recall: {best_recall['provider']}/{best_recall['model']} - {best_recall['overlap_metrics']['total_recall']:.1%}")

            # Best by precision
            best_precision = max(successful_results, key=lambda x: x['overlap_metrics']['precision'])
            report.append(f"Best Precision: {best_precision['provider']}/{best_precision['model']} - {best_precision['overlap_metrics']['precision']:.1%}")

        # Add detailed model responses for each image
        report.append(f"\n📝 DETAILED MODEL RESPONSES BY IMAGE")
        report.append("=" * 60)

        for image_filename in sorted(results.get('by_image', {}).keys()):
            image_results = [r for r in results['detailed_results'] if r['image'] == image_filename]

            if image_results:
                image_desc = self.REFERENCE_KEYWORDS[image_filename]['description']
                report.append(f"\n🖼️  {image_filename}: {image_desc}")
                report.append("-" * 50)

                for i, result in enumerate(image_results, 1):
                    provider = result['provider']
                    model = result['model']

                    if result['success']:
                        response = result.get('full_response', result.get('response_preview', 'No response available'))
                        recall = result['overlap_metrics']['total_recall']
                        precision = result['overlap_metrics']['precision']
                        report.append(f"{i}. {provider}/{model} (R:{recall:.1%} P:{precision:.1%}):")
                        report.append(f"   \"{response}\"")
                    else:
                        error = result.get('error', 'Unknown error')
                        report.append(f"{i}. {provider}/{model} (FAILED):")
                        report.append(f"   Error: {error}")

                    report.append("")  # Add blank line between responses

        return "\n".join(report)


# Test class for pytest integration
class TestRealImageVisionAnalysis:
    """Pytest wrapper for real image vision analysis."""

    @pytest.fixture(scope="class")
    def tester(self):
        """Create vision tester instance."""
        return RealImageVisionTester()

    @pytest.mark.skipif(
        not os.getenv('TEST_WITH_REAL_MODELS'),
        reason="Real model testing disabled. Set TEST_WITH_REAL_MODELS=1 to enable"
    )
    def test_comprehensive_vision_analysis(self, tester):
        """Run comprehensive vision analysis tests."""
        results = tester.run_comprehensive_tests(test_real_models=True)

        # Generate and print report
        report = tester.generate_report(results)
        print("\n" + report)

        # Basic assertions
        assert not results.get('skipped'), "Tests should not be skipped when real models enabled"
        assert results['summary']['total_tests'] > 0, "Should have run some tests"
        assert results['summary']['successful_tests'] > 0, "Should have some successful tests"

    def test_keyword_extraction(self, tester):
        """Test keyword extraction functionality."""
        test_response = "This image shows a beautiful mountain landscape with a hiking trail. There are wooden fences along the path, blue sky with white clouds, and green vegetation."

        keywords = tester.extract_keywords_from_response(test_response)

        # Should extract relevant keywords
        expected_keywords = {"mountain", "landscape", "hiking", "trail", "wooden", "fences", "path", "blue", "sky", "white", "clouds", "green", "vegetation"}

        # Check that we extract meaningful keywords
        assert len(keywords) > 5, "Should extract multiple keywords"
        assert len(keywords & expected_keywords) > 5, "Should extract relevant keywords"

    def test_overlap_calculation(self, tester):
        """Test overlap calculation functionality."""
        model_keywords = {"mountain", "trail", "hiking", "fence", "sky", "clouds", "irrelevant", "keyword"}
        reference_keywords = {
            "primary": ["mountain", "trail", "hiking", "fence"],
            "secondary": ["sky", "clouds", "landscape", "nature"]
        }

        overlap = tester.calculate_overlap(model_keywords, reference_keywords)

        # Should calculate reasonable overlaps
        assert overlap['primary_recall'] == 1.0, "Should have 100% primary recall"
        assert overlap['secondary_recall'] == 0.5, "Should have 50% secondary recall"
        assert overlap['precision'] == 0.75, "Should have 75% precision (6/8 relevant)"

    def test_image_setup(self, tester):
        """Test that test images are properly accessible."""
        images = tester.setup_images()

        # Should find test images
        assert len(images) > 0, "Should find test images"

        for filename, path in images.items():
            assert path.exists(), f"Image file should exist: {path}"
            assert filename in tester.REFERENCE_KEYWORDS, f"Should have reference keywords for {filename}"


if __name__ == "__main__":
    # Allow running tests directly
    tester = RealImageVisionTester()

    # Run comprehensive tests if real models enabled
    if os.getenv('TEST_WITH_REAL_MODELS'):
        results = tester.run_comprehensive_tests(test_real_models=True)
        report = tester.generate_report(results)
        print(report)

        # Save detailed results
        with open('vision_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Detailed results saved to vision_test_results.json")
    else:
        print("❌ Real model testing disabled. Set TEST_WITH_REAL_MODELS=1 to enable")

        # Just test the functionality
        pytest.main([__file__, "-v", "-k", "not test_comprehensive_vision_analysis"])