#!/usr/bin/env python3
"""
Updated Comprehensive Vision Testing Framework
Tests each (provider, model) combination with 3 different query types using
dynamic reference loading for each specific image.

Key improvements:
1. Uses image-specific reference JSON files instead of hardcoded analysis
2. Supports testing multiple images in a single run
3. Better error handling and validation
4. Enhanced reporting with per-image results
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import sys

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from abstractcore import create_llm
from dynamic_reference_loader import (
    DynamicReferenceLoader,
    get_test_prompt,
    get_all_prompts
)

class UpdatedComprehensiveVisionTester:
    """Updated comprehensive vision testing with dynamic references."""

    def __init__(self, image_paths: List[str], providers: List[str] = None):
        self.image_paths = [Path(p) for p in image_paths]
        self.providers = providers or ["lmstudio", "ollama", "huggingface", "anthropic", "openai"]
        self.results = {}
        self.reference_loader = DynamicReferenceLoader()

        # Test configuration
        self.query_types = ["keywords", "summary", "structured"]

        # Model configurations for each provider
        self.provider_models = {
            "lmstudio": [
                "qwen/qwen2.5-vl-7b",
                "google/gemma-3n-e4b",
                "mistralai/magistral-small-2509"
            ],
            "ollama": [
                "qwen2.5vl:7b",
                "gemma3:4b",
                "llama3.2-vision:11b",
                "granite3.2-vision:2b"
            ],
            "huggingface": [
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
            ],
            "anthropic": [
                "claude-3-5-haiku-20241022"
            ],
            "openai": [
                "gpt-5-mini"
            ]
        }

        # Validate that reference files exist for all images
        self.validate_reference_files()

    def validate_reference_files(self):
        """Validate that reference files exist for all test images."""
        missing_references = []
        for image_path in self.image_paths:
            try:
                self.reference_loader.load_reference_for_image(str(image_path))
            except FileNotFoundError:
                missing_references.append(image_path.name)

        if missing_references:
            print(f"⚠️  Missing reference files for: {', '.join(missing_references)}")
            print("   Please ensure each image has a corresponding .json reference file")
            return False

        print(f"✅ All {len(self.image_paths)} reference files validated")
        return True

    def calculate_keyword_similarity(self, response_text: str, reference_keywords: List[str]) -> Dict[str, float]:
        """Calculate keyword-based similarity metrics."""
        response_lower = response_text.lower()

        # Find matches
        found_keywords = []
        for keyword in reference_keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)

        # Calculate metrics
        recall = len(found_keywords) / len(reference_keywords) if reference_keywords else 0

        # Simple precision based on content relevance
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

    def evaluate_summary_quality(self, response_text: str, image_path: str) -> Dict[str, Any]:
        """Evaluate summary quality based on image-specific reference."""
        reference_summary = self.reference_loader.get_reference_summary(image_path)
        reference_keywords = self.reference_loader.get_reference_keywords(image_path)

        # Extract key elements from reference keywords for this specific image
        response_lower = response_text.lower()
        covered_keywords = [kw for kw in reference_keywords if kw.lower() in response_lower]

        coverage_score = len(covered_keywords) / len(reference_keywords) if reference_keywords else 0
        word_count = len(response_text.split())

        return {
            "coverage_score": coverage_score,
            "covered_keywords": covered_keywords,
            "total_keywords": len(reference_keywords),
            "word_count": word_count,
            "completeness": "high" if coverage_score > 0.7 else "medium" if coverage_score > 0.4 else "low",
            "reference_summary_length": len(reference_summary)
        }

    def evaluate_structured_response(self, response_text: str, image_path: str) -> Dict[str, Any]:
        """Evaluate structured response completeness against image-specific reference."""
        reference_structured = self.reference_loader.get_reference_structured(image_path)
        reference_fields = list(reference_structured.keys())

        response_lower = response_text.lower()

        # Check which fields are addressed
        covered_fields = []
        for field in reference_fields:
            field_variations = [
                field,
                field.replace("_", " "),
                field.replace("_", "-"),
                field.split("_")[0]  # First part of compound fields
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

    async def test_model_with_query(self, provider: str, model: str, query_type: str, image_path: str) -> Dict[str, Any]:
        """Test a specific model with a specific query type on a specific image."""
        image_name = Path(image_path).name
        print(f"🔍 Testing {provider}/{model} - {query_type} - {image_name}")

        try:
            # Create LLM instance
            llm = create_llm(provider, model=model)

            # Get the prompt for this query type
            prompt = get_test_prompt(query_type)

            # Generate response with timing
            start_time = time.time()
            response = llm.generate(prompt, media=[str(image_path)])
            end_time = time.time()

            # Evaluate based on query type using image-specific reference
            if query_type == "keywords":
                reference_keywords = self.reference_loader.get_reference_keywords(image_path)
                evaluation = self.calculate_keyword_similarity(response.content, reference_keywords)
            elif query_type == "summary":
                evaluation = self.evaluate_summary_quality(response.content, image_path)
            elif query_type == "structured":
                evaluation = self.evaluate_structured_response(response.content, image_path)
            else:
                evaluation = {"error": "Unknown query type"}

            result = {
                "provider": provider,
                "model": model,
                "query_type": query_type,
                "image_name": image_name,
                "image_path": str(image_path),
                "success": True,
                "response_time": end_time - start_time,
                "response_content": response.content,
                "response_length": len(response.content),
                "word_count": len(response.content.split()),
                "evaluation": evaluation,
                "timestamp": datetime.now().isoformat()
            }

            print(f"   ✅ Success: {end_time - start_time:.2f}s")
            return result

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            return {
                "provider": provider,
                "model": model,
                "query_type": query_type,
                "image_name": image_name,
                "image_path": str(image_path),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests for all (provider, model, query_type, image) combinations."""
        print("🎯 UPDATED COMPREHENSIVE VISION TESTING")
        print("=" * 70)
        print(f"📸 Images: {len(self.image_paths)} files")
        for img_path in self.image_paths:
            print(f"   - {img_path.name}")
        print(f"🔧 Providers: {', '.join(self.providers)}")
        print(f"📋 Query types: {', '.join(self.query_types)}")
        print()

        # Results structure: results[image_name][model_key] = model_results
        results_by_image = {}

        for image_path in self.image_paths:
            image_name = image_path.name
            print(f"\n📸 TESTING IMAGE: {image_name}")
            print("=" * 50)

            # Load reference for this image
            try:
                reference_data = self.reference_loader.load_reference_for_image(str(image_path))
                print(f"📋 Reference loaded: {reference_data['description'][:80]}...")
            except Exception as e:
                print(f"❌ Failed to load reference: {e}")
                continue

            results_by_image[image_name] = {
                "reference_data": reference_data,
                "model_results": {},
                "image_stats": {
                    "total_tests": 0,
                    "successful_tests": 0,
                    "models_tested": 0,
                    "avg_response_time": 0
                }
            }

            # Test all models for this image
            for provider in self.providers:
                if provider not in self.provider_models:
                    print(f"⚠️  Unknown provider: {provider}")
                    continue

                print(f"\n🔧 TESTING {provider.upper()} MODELS")
                print("-" * 40)

                for model in self.provider_models[provider]:
                    print(f"\n📱 Model: {model}")
                    model_key = f"{provider}/{model}"

                    # Initialize result structure for this model
                    model_results = {
                        "keywords": None,
                        "summary": None,
                        "structured": None,
                        "evaluation": {},
                        "performance": {
                            "total_tests": 0,
                            "successful_tests": 0,
                            "response_times": {},
                            "success_rate": 0.0
                        },
                        "metadata": {
                            "provider": provider,
                            "model": model,
                            "image_name": image_name,
                            "test_timestamp": None
                        }
                    }

                    results_by_image[image_name]["model_results"][model_key] = model_results

                    # Test each query type for this model and image
                    successful_count = 0
                    total_time = 0

                    for query_type in self.query_types:
                        result = await self.test_model_with_query(provider, model, query_type, str(image_path))

                        model_results["performance"]["total_tests"] += 1
                        results_by_image[image_name]["image_stats"]["total_tests"] += 1

                        if result["success"]:
                            successful_count += 1
                            model_results["performance"]["successful_tests"] += 1
                            results_by_image[image_name]["image_stats"]["successful_tests"] += 1

                            # Store the response content
                            model_results[query_type] = result["response_content"]

                            # Store evaluation and performance data
                            model_results["evaluation"][query_type] = result["evaluation"]
                            model_results["performance"]["response_times"][query_type] = result["response_time"]
                            model_results["metadata"]["test_timestamp"] = result["timestamp"]

                            total_time += result["response_time"]

                    # Calculate success rate for this model
                    model_results["performance"]["success_rate"] = successful_count / len(self.query_types)

                    # Update image stats
                    results_by_image[image_name]["image_stats"]["models_tested"] += 1
                    if total_time > 0:
                        avg_time = total_time / successful_count if successful_count > 0 else 0
                        current_avg = results_by_image[image_name]["image_stats"]["avg_response_time"]
                        results_by_image[image_name]["image_stats"]["avg_response_time"] = (current_avg + avg_time) / 2

                    # Summary for this model
                    print(f"   📊 {successful_count}/{len(self.query_types)} tests successful")

        return {
            "test_config": {
                "image_paths": [str(p) for p in self.image_paths],
                "providers": self.providers,
                "query_types": self.query_types,
                "total_images": len(self.image_paths),
                "test_timestamp": datetime.now().isoformat()
            },
            "results_by_image": results_by_image,
            "summary": self.generate_comprehensive_summary(results_by_image)
        }

    def generate_comprehensive_summary(self, results_by_image: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive test summary from all image results."""
        # Overall statistics
        total_tests = sum(img_data["image_stats"]["total_tests"] for img_data in results_by_image.values())
        total_successful = sum(img_data["image_stats"]["successful_tests"] for img_data in results_by_image.values())

        # Per-image summary
        image_summaries = {}
        for image_name, img_data in results_by_image.items():
            stats = img_data["image_stats"]
            image_summaries[image_name] = {
                "tests_run": stats["total_tests"],
                "successful_tests": stats["successful_tests"],
                "success_rate": stats["successful_tests"] / stats["total_tests"] if stats["total_tests"] > 0 else 0,
                "models_tested": stats["models_tested"],
                "avg_response_time": stats["avg_response_time"]
            }

        # Provider performance across all images
        provider_stats = {}
        for provider in self.providers:
            provider_tests = 0
            provider_successful = 0
            provider_response_times = []

            for img_data in results_by_image.values():
                for model_key, model_results in img_data["model_results"].items():
                    if model_results["metadata"]["provider"] == provider:
                        provider_tests += model_results["performance"]["total_tests"]
                        provider_successful += model_results["performance"]["successful_tests"]
                        for time_val in model_results["performance"]["response_times"].values():
                            provider_response_times.append(time_val)

            if provider_tests > 0:
                provider_stats[provider] = {
                    "total_tests": provider_tests,
                    "successful_tests": provider_successful,
                    "success_rate": provider_successful / provider_tests,
                    "avg_response_time": sum(provider_response_times) / len(provider_response_times) if provider_response_times else 0
                }

        return {
            "overall": {
                "total_images": len(results_by_image),
                "total_tests": total_tests,
                "successful_tests": total_successful,
                "overall_success_rate": total_successful / total_tests if total_tests > 0 else 0,
                "test_timestamp": datetime.now().isoformat()
            },
            "by_image": image_summaries,
            "by_provider": provider_stats
        }

    def save_results(self, results: Dict, output_file: str = None) -> str:
        """Save results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_names = "_".join([p.stem for p in self.image_paths[:3]])  # First 3 image names
            if len(self.image_paths) > 3:
                image_names += f"_plus{len(self.image_paths)-3}"
            output_file = f"updated_vision_test_{image_names}_{timestamp}.json"

        output_path = Path("tests/vision_comprehensive") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_path}")
        return str(output_path)

async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Updated Comprehensive Vision Testing")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to test images")
    parser.add_argument("--providers", nargs="+",
                       default=["lmstudio", "ollama", "huggingface", "anthropic", "openai"],
                       help="Providers to test")
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")

    args = parser.parse_args()

    # Validate image paths
    valid_images = []
    for image_path in args.images:
        path = Path(image_path)
        if path.exists():
            valid_images.append(str(path))
        else:
            print(f"❌ Image not found: {image_path}")

    if not valid_images:
        print("❌ No valid images provided")
        return 1

    # Run tests
    tester = UpdatedComprehensiveVisionTester(valid_images, args.providers)
    results = await tester.run_comprehensive_tests()

    # Save results if requested
    if args.save_results:
        tester.save_results(results, args.output)

    # Print final summary
    print("\n🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    summary = results["summary"]
    print(f"Total images: {summary['overall']['total_images']}")
    print(f"Total tests: {summary['overall']['total_tests']}")
    print(f"Successful tests: {summary['overall']['successful_tests']}")
    print(f"Overall success rate: {summary['overall']['overall_success_rate']:.1%}")

    print("\nPer-image results:")
    for image_name, img_summary in summary["by_image"].items():
        print(f"  {image_name}: {img_summary['successful_tests']}/{img_summary['tests_run']} "
              f"({img_summary['success_rate']:.1%}) - {img_summary['avg_response_time']:.2f}s avg")

    return 0

if __name__ == "__main__":
    asyncio.run(main())