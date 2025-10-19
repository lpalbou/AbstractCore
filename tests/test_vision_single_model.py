"""
Single Model Vision Testing
Focused tests for testing one model comprehensively across all 5 test images.

Usage:
    # Test specific model on all images
    pytest tests/test_vision_single_model.py::TestSpecificModel::test_qwen_vision_all_images -s

    # Test with environment variable
    VISION_MODEL="ollama/qwen2.5vl:7b" pytest tests/test_vision_single_model.py::TestSpecificModel::test_env_model_all_images -s

    # Test all available models one by one
    pytest tests/test_vision_single_model.py::TestSpecificModel -s
"""

import pytest
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any

from abstractcore import create_llm
from abstractcore.media.capabilities import is_vision_model


class TestSpecificModel:
    """Test specific vision models comprehensively across all test images."""

    def _load_reference_data(self, image_path: str, vision_reference_files: Dict) -> Dict:
        """Load reference data for an image."""
        image_name = Path(image_path).name
        if image_name not in vision_reference_files:
            pytest.skip(f"No reference file found for {image_name}")

        ref_path = vision_reference_files[image_name]
        with open(ref_path, 'r') as f:
            return json.load(f)

    def _evaluate_response(self, response_text: str, reference_data: Dict, query_type: str) -> Dict:
        """Evaluate response against reference data."""
        if query_type == "keywords":
            return self._calculate_keyword_similarity(response_text, reference_data["keywords"])
        elif query_type == "summary":
            return self._evaluate_summary_quality(response_text, reference_data)
        elif query_type == "structured":
            return self._evaluate_structured_response(response_text, reference_data)
        else:
            return {"error": "Unknown query type"}

    def _calculate_keyword_similarity(self, response_text: str, reference_keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword-based similarity metrics."""
        response_lower = response_text.lower()

        found_keywords = []
        for keyword in reference_keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)

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

    def _evaluate_summary_quality(self, response_text: str, reference_data: Dict) -> Dict[str, Any]:
        """Evaluate summary quality based on reference."""
        reference_keywords = reference_data["keywords"]
        response_lower = response_text.lower()
        covered_keywords = [kw for kw in reference_keywords if kw.lower() in response_lower]

        coverage_score = len(covered_keywords) / len(reference_keywords) if reference_keywords else 0
        word_count = len(response_text.split())

        return {
            "coverage_score": coverage_score,
            "covered_keywords": covered_keywords,
            "total_keywords": len(reference_keywords),
            "word_count": word_count,
            "completeness": "high" if coverage_score > 0.7 else "medium" if coverage_score > 0.4 else "low"
        }

    def _evaluate_structured_response(self, response_text: str, reference_data: Dict) -> Dict[str, Any]:
        """Evaluate structured response completeness."""
        reference_structured = reference_data["structured"]
        reference_fields = list(reference_structured.keys())
        response_lower = response_text.lower()

        covered_fields = []
        for field in reference_fields:
            field_variations = [
                field,
                field.replace("_", " "),
                field.replace("_", "-"),
                field.split("_")[0]
            ]

            if any(var.lower() in response_lower for var in field_variations):
                covered_fields.append(field)

        structure_score = len(covered_fields) / len(reference_fields) if reference_fields else 0

        return {
            "structure_score": structure_score,
            "covered_fields": covered_fields,
            "expected_fields": reference_fields,
            "field_coverage": f"{len(covered_fields)}/{len(reference_fields)}",
            "organization": "structured" if ":" in response_text or "-" in response_text else "narrative"
        }

    def _run_comprehensive_test(self, provider: str, model: str, vision_test_images: List[str],
                              vision_reference_files: Dict, create_vision_llm) -> Dict:
        """Run comprehensive test for a specific model."""
        print(f"\n🎯 COMPREHENSIVE TEST: {provider}/{model}")
        print("=" * 60)

        llm = create_vision_llm(provider, model)

        query_types = ["keywords", "summary", "structured"]
        prompts = {
            "keywords": "Extract keywords from this image. List only single words or short phrases that describe the objects, scenery, colors, lighting, and activities visible. Separate with commas.",
            "summary": "Provide a detailed descriptive summary of this image. Describe what you see including objects, scenery, lighting, composition, and overall atmosphere in 3-4 sentences.",
            "structured": """Analyze this image and provide a structured response in the following format:

Theme: [Main theme/subject]
Mood: [Emotional tone/atmosphere]
Color_tone: [Overall color palette description]
Setting: [Location/environment type]
Weather: [Weather conditions visible]
Time_of_day: [Apparent time based on lighting]
Composition: [Photographic composition elements]
Main_objects: [Key objects visible]
Lighting: [Lighting conditions and quality]
Suggested_activity: [What activity this scene suggests]
Dominant_colors: [Primary colors present]
Visual_elements: [Notable visual features]
Landscape_type: [Type of terrain/landscape]
Human_presence: [Any signs of human activity]

Provide concise but descriptive answers for each field."""
        }

        results = {
            "provider": provider,
            "model": model,
            "test_results": {},
            "performance": {
                "total_images": 0,
                "successful_images": 0,
                "total_queries": 0,
                "successful_queries": 0,
                "avg_response_time": 0.0,
                "avg_scores": {
                    "keywords_f1": 0.0,
                    "summary_coverage": 0.0,
                    "structured_coverage": 0.0
                }
            }
        }

        total_time = 0.0
        total_scores = {"keywords": [], "summary": [], "structured": []}

        for image_path in vision_test_images:
            image_name = Path(image_path).name
            print(f"\n📸 Testing {image_name}")

            try:
                reference_data = self._load_reference_data(image_path, vision_reference_files)

                results["test_results"][image_name] = {
                    "queries": {},
                    "image_success": True,
                    "reference_theme": reference_data["structured"]["theme"]
                }

                image_successful_queries = 0

                for query_type in query_types:
                    print(f"   🔍 Query: {query_type}")

                    try:
                        start_time = time.time()
                        response = llm.generate(prompts[query_type], media=[image_path])
                        duration = time.time() - start_time

                        evaluation = self._evaluate_response(response.content, reference_data, query_type)

                        results["test_results"][image_name]["queries"][query_type] = {
                            "success": True,
                            "response_time": duration,
                            "evaluation": evaluation,
                            "response_content": response.content[:150] + "..." if len(response.content) > 150 else response.content
                        }

                        # Track scores for averages
                        if query_type == "keywords" and "f1" in evaluation:
                            total_scores["keywords"].append(evaluation["f1"])
                            print(f"      F1: {evaluation['f1']:.3f}")
                        elif query_type == "summary" and "coverage_score" in evaluation:
                            total_scores["summary"].append(evaluation["coverage_score"])
                            print(f"      Coverage: {evaluation['coverage_score']:.3f}")
                        elif query_type == "structured" and "structure_score" in evaluation:
                            total_scores["structured"].append(evaluation["structure_score"])
                            print(f"      Structure: {evaluation['structure_score']:.3f}")

                        total_time += duration
                        image_successful_queries += 1
                        results["performance"]["successful_queries"] += 1

                        print(f"      ✅ {duration:.2f}s")

                    except Exception as e:
                        print(f"      ❌ Query failed: {str(e)}")
                        results["test_results"][image_name]["queries"][query_type] = {
                            "success": False,
                            "error": str(e)
                        }

                    results["performance"]["total_queries"] += 1

                if image_successful_queries > 0:
                    results["performance"]["successful_images"] += 1
                else:
                    results["test_results"][image_name]["image_success"] = False

            except Exception as e:
                print(f"   ❌ Image failed: {str(e)}")
                results["test_results"][image_name] = {
                    "image_success": False,
                    "error": str(e)
                }

            results["performance"]["total_images"] += 1

        # Calculate final statistics
        if results["performance"]["successful_queries"] > 0:
            results["performance"]["avg_response_time"] = total_time / results["performance"]["successful_queries"]

        for score_type, scores in total_scores.items():
            if scores:
                if score_type == "keywords":
                    results["performance"]["avg_scores"]["keywords_f1"] = sum(scores) / len(scores)
                elif score_type == "summary":
                    results["performance"]["avg_scores"]["summary_coverage"] = sum(scores) / len(scores)
                elif score_type == "structured":
                    results["performance"]["avg_scores"]["structured_coverage"] = sum(scores) / len(scores)

        # Print final summary
        perf = results["performance"]
        print(f"\n📊 FINAL RESULTS for {provider}/{model}")
        print("=" * 60)
        print(f"Images: {perf['successful_images']}/{perf['total_images']} successful")
        print(f"Queries: {perf['successful_queries']}/{perf['total_queries']} successful")
        print(f"Avg Response Time: {perf['avg_response_time']:.2f}s")
        print(f"Avg Keyword F1: {perf['avg_scores']['keywords_f1']:.3f}")
        print(f"Avg Summary Coverage: {perf['avg_scores']['summary_coverage']:.3f}")
        print(f"Avg Structured Coverage: {perf['avg_scores']['structured_coverage']:.3f}")

        return results

    @pytest.mark.parametrize("provider,model", [
        ("ollama", "qwen2.5vl:7b"),
        ("ollama", "llama3.2-vision:11b"),
        ("lmstudio", "qwen/qwen2.5-vl-7b"),
        ("lmstudio", "qwen/qwen3-vl-4b"),
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-sonnet-20241022")
    ])
    def test_specific_model_all_images(self, provider, model, vision_test_images,
                                     vision_reference_files, skip_if_provider_unavailable,
                                     create_vision_llm):
        """Test a specific model comprehensively on all 5 test images."""
        skip_if_provider_unavailable(provider, model)

        results = self._run_comprehensive_test(
            provider, model, vision_test_images, vision_reference_files, create_vision_llm
        )

        # Assertions for test success
        assert results["performance"]["successful_images"] > 0, f"No successful images for {provider}/{model}"
        assert results["performance"]["successful_queries"] > 0, f"No successful queries for {provider}/{model}"

        # Quality thresholds
        avg_f1 = results["performance"]["avg_scores"]["keywords_f1"]
        assert avg_f1 > 0.1, f"Keyword F1 score too low: {avg_f1:.3f}"

        # Store results for analysis
        setattr(pytest, f'single_model_results_{provider}_{model}', results)

    def test_env_model_all_images(self, vision_test_images, vision_reference_files,
                                skip_if_provider_unavailable, create_vision_llm):
        """Test model specified by VISION_MODEL environment variable."""
        model_spec = os.getenv("VISION_MODEL")
        if not model_spec:
            pytest.skip("VISION_MODEL environment variable not set. Use format: provider/model")

        if "/" not in model_spec:
            pytest.skip("VISION_MODEL must be in format 'provider/model'")

        provider, model = model_spec.split("/", 1)

        print(f"Testing model from environment: {provider}/{model}")

        skip_if_provider_unavailable(provider, model)

        results = self._run_comprehensive_test(
            provider, model, vision_test_images, vision_reference_files, create_vision_llm
        )

        # Assertions
        assert results["performance"]["successful_images"] > 0, f"No successful images for {provider}/{model}"
        assert results["performance"]["successful_queries"] > 0, f"No successful queries for {provider}/{model}"

        # Store results
        setattr(pytest, f'env_model_results', results)

    @pytest.mark.parametrize("model_name", [
        "qwen2.5vl:7b",  # Ollama
        "qwen/qwen3-vl-4b",  # LMStudio
        "llama3.2-vision:11b",  # Ollama
        "gpt-4o",  # OpenAI
        "claude-3-5-sonnet-20241022"  # Anthropic
    ])
    def test_any_available_provider_for_model(self, model_name, vision_test_images,
                                            vision_reference_files, available_vision_providers,
                                            create_vision_llm):
        """Test a model with any available provider that supports it."""
        # Find provider that has this model
        provider = None
        for prov, models in available_vision_providers.items():
            if model_name in models:
                provider = prov
                break

        if not provider:
            pytest.skip(f"Model {model_name} not available in any provider")

        print(f"Testing {model_name} via {provider}")

        results = self._run_comprehensive_test(
            provider, model_name, vision_test_images, vision_reference_files, create_vision_llm
        )

        # Basic success assertions
        assert results["performance"]["successful_images"] > 0, f"No successful images for {model_name}"
        avg_f1 = results["performance"]["avg_scores"]["keywords_f1"]
        assert avg_f1 > 0.05, f"F1 score too low for {model_name}: {avg_f1:.3f}"


class TestQuickVisionSmoke:
    """Quick smoke tests for vision functionality."""

    def test_any_vision_model_basic_functionality(self, vision_test_images, available_vision_providers, create_vision_llm):
        """Quick test that at least one vision model works on at least one image."""
        if not available_vision_providers:
            pytest.skip("No vision providers available")

        if not vision_test_images:
            pytest.skip("No test images available")

        # Use first available provider and model
        provider = list(available_vision_providers.keys())[0]
        model = available_vision_providers[provider][0]
        image_path = vision_test_images[0]

        print(f"Quick test: {provider}/{model} on {Path(image_path).name}")

        llm = create_vision_llm(provider, model)

        response = llm.generate("What do you see in this image?", media=[image_path])

        assert response is not None
        assert hasattr(response, 'content')
        assert len(response.content.strip()) > 10
        assert len(response.content.split()) >= 3

        print(f"✅ Quick test passed: {response.content[:100]}...")

    @pytest.mark.parametrize("image_name", [
        "mystery1_mp.jpg", "mystery2_sc.jpg", "mystery3_us.jpg", "mystery4_wh.jpg", "mystery5_so.jpg"
    ])
    def test_each_image_with_best_available_model(self, image_name, vision_test_images,
                                                available_vision_providers, create_vision_llm):
        """Test each image with the best available model."""
        if not available_vision_providers:
            pytest.skip("No vision providers available")

        # Find the image
        image_path = None
        for img_path in vision_test_images:
            if image_name in img_path:
                image_path = img_path
                break

        if not image_path:
            pytest.skip(f"Image {image_name} not found")

        # Choose best available model (priority: OpenAI, Anthropic, others)
        provider, model = None, None
        for prov in ["openai", "anthropic", "ollama", "lmstudio", "huggingface"]:
            if prov in available_vision_providers:
                provider = prov
                model = available_vision_providers[prov][0]
                break

        if not provider:
            pytest.skip("No suitable provider available")

        print(f"Testing {image_name} with best available: {provider}/{model}")

        llm = create_vision_llm(provider, model)

        # Simple descriptive test
        response = llm.generate(
            "Describe this image in detail. What are the main objects, colors, and setting?",
            media=[image_path]
        )

        # Basic validation
        assert response.content is not None
        assert len(response.content.strip()) > 20
        assert len(response.content.split()) >= 10

        print(f"✅ {image_name}: {len(response.content)} chars, {len(response.content.split())} words")