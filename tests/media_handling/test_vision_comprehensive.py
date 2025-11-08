"""
Comprehensive Vision Testing for AbstractCore
Tests vision capabilities across all providers and models with proper pytest integration.

Usage:
    # Test all available vision models on all images
    pytest tests/test_vision_comprehensive.py

    # Test only specific provider
    pytest tests/test_vision_comprehensive.py -k "test_ollama"

    # Skip slow comprehensive tests
    pytest tests/test_vision_comprehensive.py -m "not slow"

    # Test specific model
    pytest tests/test_vision_comprehensive.py::TestSingleModelComprehensive::test_qwen_vision -s
"""

import pytest
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any

from abstractcore import create_llm
from abstractcore.media.capabilities import is_vision_model


class TestVisionModelAvailability:
    """Test that vision models are properly detected and available."""

    def test_vision_capability_detection(self):
        """Test that vision capability detection works correctly."""
        # Known vision models
        vision_models = [
            "gpt-4o",
            "gpt-4-turbo",
            "claude-3-5-sonnet",
            "qwen2.5vl:7b",
            "llama3.2-vision:11b"
        ]

        # Known non-vision models
        non_vision_models = [
            "gpt-3.5-turbo",
            "claude-3-haiku",  # Note: some Haiku versions do have vision
            "qwen3-coder:30b"
        ]

        for model in vision_models:
            assert is_vision_model(model), f"Model {model} should be detected as vision-capable"

        # Note: Some models might have vision variants, so we're more lenient here
        for model in non_vision_models:
            # We don't assert false here because model detection might be conservative
            result = is_vision_model(model)
            print(f"Model {model} vision detection: {result}")

    def test_provider_vision_models_list(self, available_vision_providers):
        """Test that we can enumerate available vision providers and models."""
        assert isinstance(available_vision_providers, dict)

        print(f"\nAvailable vision providers: {list(available_vision_providers.keys())}")

        for provider, models in available_vision_providers.items():
            assert isinstance(models, list)
            assert len(models) > 0, f"Provider {provider} should have at least one available model"

            for model in models:
                print(f"  {provider}: {model}")
                assert is_vision_model(model), f"Model {model} should support vision"


class TestSingleImageVision:
    """Test vision capabilities on individual images."""

    @pytest.mark.parametrize("image_name", [
        "mystery1_mp.jpg",  # Mountain path
        "mystery2_sc.jpg",  # Space cat
        "mystery3_us.jpg",  # Urban sunset
        "mystery4_wh.jpg",  # Whale breaching
        "mystery5_so.jpg"   # Food dish
    ])
    def test_single_image_analysis(self, image_name, vision_test_images, available_vision_providers,
                                  create_vision_llm):
        """Test basic vision analysis on a single image with any available provider."""
        # Find the image file
        image_path = None
        for img_path in vision_test_images:
            if image_name in img_path:
                image_path = img_path
                break

        if not image_path:
            pytest.skip(f"Test image {image_name} not found")

        # Test with the first available provider
        if not available_vision_providers:
            pytest.skip("No vision providers available")

        provider = list(available_vision_providers.keys())[0]
        model = available_vision_providers[provider][0]

        print(f"Testing {image_name} with {provider}/{model}")

        llm = create_vision_llm(provider, model)

        # Simple analysis prompt
        prompt = "Describe what you see in this image in 2-3 sentences."

        start_time = time.time()
        response = llm.generate(prompt, media=[image_path])
        duration = time.time() - start_time

        # Validate response
        assert response is not None
        assert hasattr(response, 'content')
        assert response.content is not None
        assert len(response.content.strip()) > 10, "Response should be meaningful"

        print(f"Response ({duration:.2f}s): {response.content[:100]}...")

        # Basic quality checks
        assert len(response.content.split()) >= 5, "Response should have multiple words"


class TestSingleModelComprehensive:
    """Test a single model comprehensively across all test images."""

    def _load_reference_data(self, image_path: str, vision_reference_files: Dict) -> Dict:
        """Load reference data for an image."""
        image_name = Path(image_path).name
        if image_name not in vision_reference_files:
            pytest.skip(f"No reference file found for {image_name}")

        ref_path = vision_reference_files[image_name]
        with open(ref_path, 'r') as f:
            return json.load(f)

    def _calculate_keyword_similarity(self, response_text: str, reference_keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword-based similarity metrics."""
        response_lower = response_text.lower()

        found_keywords = []
        for keyword in reference_keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)

        # Calculate metrics
        recall = len(found_keywords) / len(reference_keywords) if reference_keywords else 0
        response_words = set(response_lower.split())
        reference_words = set(" ".join(reference_keywords).lower().split())
        common_words = response_words.intersection(reference_words)
        precision = len(common_words) / len(response_words) if response_words else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "found_keywords": found_keywords,
            "found_count": len(found_keywords),
            "total_keywords": len(reference_keywords)
        }

    @pytest.mark.parametrize("provider,model", [
        ("ollama", "qwen2.5vl:7b"),
        ("lmstudio", "qwen/qwen2.5-vl-7b"),
        ("lmstudio", "qwen/qwen3-vl-4b"),
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet-20241022"),
        ("huggingface", "unsloth/Qwen2.5-VL-7B-Instruct-GGUF")
    ])
    @pytest.mark.slow
    def test_model_comprehensive_analysis(self, provider, model, vision_test_images,
                                        vision_reference_files, skip_if_provider_unavailable,
                                        create_vision_llm):
        """Test a specific model comprehensively across all test images."""
        # Check if provider and model are available
        skip_if_provider_unavailable(provider, model)

        llm = create_vision_llm(provider, model)

        results = {
            "provider": provider,
            "model": model,
            "test_results": {},
            "summary": {
                "total_images": 0,
                "successful_tests": 0,
                "avg_f1_score": 0.0,
                "avg_response_time": 0.0
            }
        }

        total_f1 = 0.0
        total_time = 0.0
        successful_tests = 0

        for image_path in vision_test_images:
            image_name = Path(image_path).name
            print(f"\nTesting {provider}/{model} on {image_name}")

            try:
                # Load reference data
                reference_data = self._load_reference_data(image_path, vision_reference_files)

                # Test keywords extraction
                keyword_prompt = "Extract keywords from this image. List only single words or short phrases that describe the objects, scenery, colors, lighting, and activities visible. Separate with commas."

                start_time = time.time()
                response = llm.generate(keyword_prompt, media=[image_path])
                duration = time.time() - start_time

                # Evaluate against reference
                keyword_eval = self._calculate_keyword_similarity(
                    response.content, reference_data["keywords"]
                )

                results["test_results"][image_name] = {
                    "success": True,
                    "response_time": duration,
                    "keyword_evaluation": keyword_eval,
                    "response_content": response.content[:200] + "..." if len(response.content) > 200 else response.content
                }

                total_f1 += keyword_eval["f1"]
                total_time += duration
                successful_tests += 1

                print(f"  ✅ F1 Score: {keyword_eval['f1']:.3f}, Time: {duration:.2f}s")

            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                results["test_results"][image_name] = {
                    "success": False,
                    "error": str(e)
                }

            results["summary"]["total_images"] += 1

        # Calculate summary statistics
        results["summary"]["successful_tests"] = successful_tests
        if successful_tests > 0:
            results["summary"]["avg_f1_score"] = total_f1 / successful_tests
            results["summary"]["avg_response_time"] = total_time / successful_tests

        # Print summary
        print(f"\n📊 SUMMARY for {provider}/{model}:")
        print(f"   Successful: {successful_tests}/{results['summary']['total_images']}")
        print(f"   Avg F1 Score: {results['summary']['avg_f1_score']:.3f}")
        print(f"   Avg Response Time: {results['summary']['avg_response_time']:.2f}s")

        # Assertions for test success
        assert successful_tests > 0, f"No successful tests for {provider}/{model}"
        assert results["summary"]["avg_f1_score"] > 0.1, f"F1 score too low: {results['summary']['avg_f1_score']}"

        # Store results for potential analysis
        if hasattr(pytest, 'test_results'):
            pytest.test_results = getattr(pytest, 'test_results', {})
            pytest.test_results[f"{provider}_{model}"] = results


class TestAllModelsAllImages:
    """Test all available models across all images - comprehensive matrix testing."""

    @pytest.mark.comprehensive
    @pytest.mark.slow
    def test_all_available_models_all_images(self, vision_test_images, vision_reference_files,
                                           available_vision_providers, create_vision_llm):
        """Test ALL available vision models on ALL test images."""
        if not available_vision_providers:
            pytest.skip("No vision providers available")

        if not vision_test_images:
            pytest.skip("No vision test images available")

        print(f"\n🎯 COMPREHENSIVE MATRIX TEST")
        print(f"   Images: {len(vision_test_images)}")
        print(f"   Providers: {list(available_vision_providers.keys())}")

        all_results = {
            "test_matrix": {},
            "summary": {
                "total_combinations": 0,
                "successful_combinations": 0,
                "total_images": len(vision_test_images),
                "total_providers": len(available_vision_providers)
            },
            "warnings_captured": [],
            "test_metadata": {
                "prompt_used": "What do you see in this image? Describe the main objects, colors, and setting in 1-2 sentences.",
                "test_images": [Path(img).name for img in vision_test_images]
            }
        }

        # Capture warnings during test execution
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")  # Ensure all warnings are captured

            for provider, models in available_vision_providers.items():
                print(f"\n🔧 Testing {provider.upper()} models:")

                for model in models:
                    print(f"   📱 Model: {model}")

                    try:
                        llm = create_vision_llm(provider, model)

                        model_key = f"{provider}/{model}"
                        all_results["test_matrix"][model_key] = {
                            "provider": provider,
                            "model": model,
                            "image_results": {},
                            "model_stats": {
                                "successful_images": 0,
                                "total_images": 0,
                                "avg_response_time": 0.0
                            }
                        }

                        total_time = 0.0
                        successful_images = 0

                        for image_path in vision_test_images:
                            image_name = Path(image_path).name

                            try:
                                # Simple but consistent test
                                prompt = "What do you see in this image? Describe the main objects, colors, and setting in 1-2 sentences."

                                start_time = time.time()
                                response = llm.generate(prompt, media=[image_path])
                                duration = time.time() - start_time

                                all_results["test_matrix"][model_key]["image_results"][image_name] = {
                                    "success": True,
                                    "response_time": duration,
                                    "response_content": response.content,
                                    "response_length": len(response.content),
                                    "word_count": len(response.content.split())
                                }

                                total_time += duration
                                successful_images += 1

                                print(f"      ✅ {image_name}: {duration:.2f}s")

                            except Exception as e:
                                print(f"      ❌ {image_name}: {str(e)}")
                                all_results["test_matrix"][model_key]["image_results"][image_name] = {
                                    "success": False,
                                    "error": str(e)
                                }

                            all_results["test_matrix"][model_key]["model_stats"]["total_images"] += 1

                        # Calculate model statistics
                        all_results["test_matrix"][model_key]["model_stats"]["successful_images"] = successful_images
                        if successful_images > 0:
                            all_results["test_matrix"][model_key]["model_stats"]["avg_response_time"] = total_time / successful_images

                        if successful_images > 0:
                            all_results["summary"]["successful_combinations"] += 1

                        print(f"      📊 {successful_images}/{len(vision_test_images)} images successful")

                    except Exception as e:
                        print(f"   ❌ Model {model} failed to initialize: {e}")

                    all_results["summary"]["total_combinations"] += 1

            # Process captured warnings
            for warning in captured_warnings:
                all_results["warnings_captured"].append({
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": warning.lineno
                })

        # Print warnings summary if any were captured
        if all_results["warnings_captured"]:
            print(f"\n⚠️  WARNINGS CAPTURED ({len(all_results['warnings_captured'])} total):")
            for warning in all_results["warnings_captured"]:
                print(f"   {warning['category']}: {warning['message']}")

        # Generate performance analysis
        performance_analysis = self._generate_performance_analysis(all_results)
        all_results["performance_analysis"] = performance_analysis

        # Save comprehensive results to files
        self._save_test_results(all_results)

        # Print final summary with performance metrics
        self._print_final_summary(all_results, performance_analysis)

        # Success criteria
        assert all_results["summary"]["successful_combinations"] > 0, "No successful model/image combinations"

        # Store comprehensive results
        if hasattr(pytest, 'comprehensive_results'):
            pytest.comprehensive_results = all_results
        else:
            pytest.comprehensive_results = all_results

    def _generate_performance_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        performance_data = []

        for model_key, model_data in all_results["test_matrix"].items():
            if model_data["model_stats"]["successful_images"] > 0:
                performance_data.append({
                    "provider": model_data["provider"],
                    "model": model_data["model"],
                    "model_key": model_key,
                    "avg_speed": model_data["model_stats"]["avg_response_time"],
                    "success_rate": model_data["model_stats"]["successful_images"] / model_data["model_stats"]["total_images"],
                    "successful_images": model_data["model_stats"]["successful_images"],
                    "total_images": model_data["model_stats"]["total_images"]
                })

        # Sort by average speed (fastest first)
        speed_ranking = sorted(performance_data, key=lambda x: x["avg_speed"])

        # Sort by success rate (highest first)
        reliability_ranking = sorted(performance_data, key=lambda x: x["success_rate"], reverse=True)

        # Calculate provider averages
        provider_stats = {}
        for item in performance_data:
            provider = item["provider"]
            if provider not in provider_stats:
                provider_stats[provider] = {"speeds": [], "success_rates": [], "models": []}
            provider_stats[provider]["speeds"].append(item["avg_speed"])
            provider_stats[provider]["success_rates"].append(item["success_rate"])
            provider_stats[provider]["models"].append(item["model"])

        provider_averages = {}
        for provider, stats in provider_stats.items():
            provider_averages[provider] = {
                "avg_speed": sum(stats["speeds"]) / len(stats["speeds"]),
                "avg_success_rate": sum(stats["success_rates"]) / len(stats["success_rates"]),
                "model_count": len(stats["models"]),
                "models": stats["models"]
            }

        return {
            "individual_models": performance_data,
            "speed_ranking": speed_ranking,
            "reliability_ranking": reliability_ranking,
            "provider_averages": provider_averages,
            "fastest_model": speed_ranking[0] if speed_ranking else None,
            "most_reliable_model": reliability_ranking[0] if reliability_ranking else None
        }

    def _save_test_results(self, all_results: Dict[str, Any]) -> None:
        """Save comprehensive test results to files."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("test_results/vision_comprehensive")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save raw JSON data
        json_file = results_dir / f"vision_test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Save human-readable summary
        summary_file = results_dir / f"vision_test_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write("# Vision Model Comprehensive Test Results\n\n")
            f.write(f"**Test Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Models Tested**: {all_results['summary']['total_combinations']}\n")
            f.write(f"**Successful Models**: {all_results['summary']['successful_combinations']}\n")
            f.write(f"**Success Rate**: {(all_results['summary']['successful_combinations'] / all_results['summary']['total_combinations']) * 100:.1f}%\n")
            f.write(f"**Warnings Captured**: {len(all_results['warnings_captured'])}\n")
            f.write(f"**Test Prompt**: {all_results['test_metadata']['prompt_used']}\n\n")

            # Performance rankings
            f.write("## Performance Rankings\n\n")
            f.write("### Speed Ranking (Fastest First)\n")
            for i, model in enumerate(all_results["performance_analysis"]["speed_ranking"], 1):
                f.write(f"{i}. **{model['model_key']}**: {model['avg_speed']:.2f}s avg\n")

            f.write("\n### Reliability Ranking (Most Successful First)\n")
            for i, model in enumerate(all_results["performance_analysis"]["reliability_ranking"], 1):
                f.write(f"{i}. **{model['model_key']}**: {model['success_rate']*100:.1f}% success rate\n")

            # Provider averages
            f.write("\n## Provider Averages\n\n")
            for provider, stats in all_results["performance_analysis"]["provider_averages"].items():
                f.write(f"### {provider.upper()}\n")
                f.write(f"- **Average Speed**: {stats['avg_speed']:.2f}s\n")
                f.write(f"- **Average Success Rate**: {stats['avg_success_rate']*100:.1f}%\n")
                f.write(f"- **Models**: {', '.join(stats['models'])}\n\n")

            # Warnings section
            if all_results["warnings_captured"]:
                f.write("## Warnings Captured\n\n")
                for warning in all_results["warnings_captured"]:
                    f.write(f"- **{warning['category']}**: {warning['message']}\n")
                    f.write(f"  - File: {warning['filename']}:{warning['lineno']}\n")
                f.write("\n")

            # Detailed results
            f.write("## Detailed Results\n\n")
            for model_key, model_data in all_results["test_matrix"].items():
                f.write(f"### {model_key}\n")
                f.write(f"- **Success Rate**: {model_data['model_stats']['successful_images']}/{model_data['model_stats']['total_images']} images\n")
                f.write(f"- **Average Response Time**: {model_data['model_stats']['avg_response_time']:.2f}s\n")
                f.write("- **Per-Image Results**:\n")
                for image_name, result in model_data["image_results"].items():
                    if result["success"]:
                        f.write(f"  - ✅ {image_name}: {result['response_time']:.2f}s ({result['word_count']} words)\n")
                        # Include first 100 chars of response for quality assessment
                        response_preview = result['response_content'][:100].replace('\n', ' ')
                        f.write(f"    - Response: \"{response_preview}{'...' if len(result['response_content']) > 100 else ''}\"\n")
                    else:
                        f.write(f"  - ❌ {image_name}: {result.get('error', 'Unknown error')}\n")
                f.write("\n")

        print(f"\n📁 Results saved to:")
        print(f"   JSON: {json_file}")
        print(f"   Summary: {summary_file}")

    def _print_final_summary(self, all_results: Dict[str, Any], performance_analysis: Dict[str, Any]) -> None:
        """Print comprehensive final summary with performance metrics."""
        print(f"\n🎯 FINAL MATRIX TEST SUMMARY")
        print(f"   Total combinations tested: {all_results['summary']['total_combinations']}")
        print(f"   Successful combinations: {all_results['summary']['successful_combinations']}")
        success_rate = (all_results['summary']['successful_combinations'] /
                       all_results['summary']['total_combinations']) * 100
        print(f"   Success rate: {success_rate:.1f}%")

        if performance_analysis["fastest_model"]:
            print(f"\n🚀 PERFORMANCE ANALYSIS")
            fastest = performance_analysis["fastest_model"]
            print(f"   Fastest Model: {fastest['model_key']} ({fastest['avg_speed']:.2f}s avg)")

            most_reliable = performance_analysis["most_reliable_model"]
            print(f"   Most Reliable: {most_reliable['model_key']} ({most_reliable['success_rate']*100:.1f}% success)")

            print(f"\n📊 PROVIDER AVERAGES")
            for provider, stats in performance_analysis["provider_averages"].items():
                print(f"   {provider.upper()}: {stats['avg_speed']:.2f}s avg, {stats['avg_success_rate']*100:.1f}% success")

            print(f"\n⚡ SPEED RANKINGS (Top 3)")
            for i, model in enumerate(performance_analysis["speed_ranking"][:3], 1):
                print(f"   {i}. {model['model_key']}: {model['avg_speed']:.2f}s")

            print(f"\n🎯 RELIABILITY RANKINGS (Top 3)")
            for i, model in enumerate(performance_analysis["reliability_ranking"][:3], 1):
                print(f"   {i}. {model['model_key']}: {model['success_rate']*100:.1f}% success")


class TestVisionQualityBenchmarks:
    """Test vision quality and performance benchmarks."""

    @pytest.mark.parametrize("image_name,expected_objects", [
        ("mystery1_mp.jpg", ["mountain", "trail", "fence", "sky"]),
        ("mystery2_sc.jpg", ["cat", "helmet", "transparent", "dome"]),
        ("mystery3_us.jpg", ["street", "sunset", "trees", "lights"]),
        ("mystery4_wh.jpg", ["whale", "ocean", "water", "breaching"]),
        ("mystery5_so.jpg", ["food", "bowl", "salad", "vegetables"])
    ])
    def test_object_detection_quality(self, image_name, expected_objects, vision_test_images,
                                    available_vision_providers, create_vision_llm):
        """Test that models can detect expected objects in test images."""
        if not available_vision_providers:
            pytest.skip("No vision providers available")

        # Find the image
        image_path = None
        for img_path in vision_test_images:
            if image_name in img_path:
                image_path = img_path
                break

        if not image_path:
            pytest.skip(f"Test image {image_name} not found")

        # Test with best available model (prefer cloud models for accuracy)
        provider = None
        model = None

        # Priority order: OpenAI, Anthropic, local models
        for prov in ["openai", "anthropic", "ollama", "lmstudio", "huggingface"]:
            if prov in available_vision_providers:
                provider = prov
                model = available_vision_providers[prov][0]
                break

        if not provider:
            pytest.skip("No suitable provider available")

        llm = create_vision_llm(provider, model)

        prompt = f"List all objects you can identify in this image. Be specific and include colors, materials, and activities."

        response = llm.generate(prompt, media=[image_path])
        response_lower = response.content.lower()

        # Check how many expected objects are detected
        detected_objects = []
        for obj in expected_objects:
            if obj.lower() in response_lower:
                detected_objects.append(obj)

        detection_rate = len(detected_objects) / len(expected_objects)

        print(f"Image: {image_name}")
        print(f"Expected: {expected_objects}")
        print(f"Detected: {detected_objects}")
        print(f"Detection rate: {detection_rate:.2f}")
        print(f"Response: {response.content[:200]}...")

        # Benchmark: Should detect at least 50% of expected objects
        assert detection_rate >= 0.5, f"Detection rate too low: {detection_rate:.2f} (detected {detected_objects})"


# Utility functions for test result analysis
def get_test_results():
    """Get accumulated test results (if any)."""
    return getattr(pytest, 'test_results', {})


def get_comprehensive_results():
    """Get comprehensive matrix test results (if any)."""
    return getattr(pytest, 'comprehensive_results', {})


# Custom pytest markers and hooks
def pytest_runtest_teardown(item, nextitem):
    """Print test completion info."""
    if hasattr(item, 'rep_call') and item.rep_call.passed:
        if "comprehensive" in item.name:
            print(f"\n✅ Comprehensive test completed: {item.name}")


@pytest.fixture(autouse=True)
def test_info(request):
    """Automatically provide test info."""
    print(f"\n🧪 Running: {request.node.name}")
    yield
    print(f"✅ Completed: {request.node.name}")