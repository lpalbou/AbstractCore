#!/usr/bin/env python3
"""
Comprehensive Vision Testing Framework
Tests each (provider, model) combination with 3 different query types:
1. Keywords extraction
2. Descriptive summary
3. Structured analysis

Each test compares model responses against reference analysis.
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
from reference_analysis import (
    REFERENCE_ANALYSIS,
    get_test_prompt,
    get_reference_keywords,
    get_reference_summary,
    get_reference_structured
)

class ComprehensiveVisionTester:
    """Comprehensive vision testing with structured evaluation."""

    def __init__(self, image_path: str, providers: List[str] = None):
        self.image_path = Path(image_path)
        self.providers = providers or ["lmstudio", "ollama", "huggingface", "anthropic", "openai"]
        self.results = []

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
                "gemma3:4b-it-qat",
                "gemma3n:e4b",
                "gemma3n:e2b",
                "llama3.2-vision:11b",
                "granite3.3:2b"
            ],
            "huggingface": [
                "unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
            ],
            "anthropic": [
                "claude-3.5-haiku"
            ],
            "openai": [
                "gpt-5-mini"
            ]
        }

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

        # Simple precision based on content relevance (rough estimate)
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

    def evaluate_summary_quality(self, response_text: str) -> Dict[str, Any]:
        """Evaluate summary quality based on content coverage."""
        reference = get_reference_summary()

        # Key elements that should be covered
        key_elements = [
            "mountain", "trail", "fence", "wooden", "path", "dirt", "gravel",
            "sky", "sun", "light", "shadow", "perspective", "valley", "nature"
        ]

        response_lower = response_text.lower()
        covered_elements = [elem for elem in key_elements if elem in response_lower]

        coverage_score = len(covered_elements) / len(key_elements)
        word_count = len(response_text.split())

        return {
            "coverage_score": coverage_score,
            "covered_elements": covered_elements,
            "key_elements_count": len(key_elements),
            "word_count": word_count,
            "completeness": "high" if coverage_score > 0.7 else "medium" if coverage_score > 0.4 else "low"
        }

    def evaluate_structured_response(self, response_text: str) -> Dict[str, Any]:
        """Evaluate structured response completeness."""
        reference_fields = get_reference_structured().keys()

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

        structure_score = len(covered_fields) / len(reference_fields)

        return {
            "structure_score": structure_score,
            "covered_fields": covered_fields,
            "expected_fields": list(reference_fields),
            "field_coverage": f"{len(covered_fields)}/{len(reference_fields)}",
            "organization": "structured" if ":" in response_text or "-" in response_text else "narrative"
        }

    async def test_model_with_query(self, provider: str, model: str, query_type: str) -> Dict[str, Any]:
        """Test a specific model with a specific query type."""
        print(f"🔍 Testing {provider}/{model} - Query: {query_type}")

        try:
            # Create LLM instance
            llm = create_llm(provider, model=model)

            # Get the prompt for this query type
            prompt = get_test_prompt(query_type)

            # Generate response with timing
            start_time = time.time()
            response = llm.generate(prompt, media=[str(self.image_path)])
            end_time = time.time()

            # Evaluate based on query type
            if query_type == "keywords":
                evaluation = self.calculate_keyword_similarity(response.content, get_reference_keywords())
            elif query_type == "summary":
                evaluation = self.evaluate_summary_quality(response.content)
            elif query_type == "structured":
                evaluation = self.evaluate_structured_response(response.content)
            else:
                evaluation = {"error": "Unknown query type"}

            result = {
                "provider": provider,
                "model": model,
                "query_type": query_type,
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
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests for all (provider, model, query_type) combinations."""
        print("🎯 COMPREHENSIVE VISION TESTING")
        print("=" * 60)
        print(f"📸 Image: {self.image_path.name}")
        print(f"🔧 Providers: {', '.join(self.providers)}")
        print(f"📋 Query types: {', '.join(self.query_types)}")
        print()

        model_results = {}

        for provider in self.providers:
            if provider not in self.provider_models:
                print(f"⚠️  Unknown provider: {provider}")
                continue

            print(f"🔧 TESTING {provider.upper()} MODELS")
            print("-" * 50)

            for model in self.provider_models[provider]:
                print(f"\n📱 Model: {model}")
                model_key = f"{provider}/{model}"

                # Initialize result structure for this model
                model_results[model_key] = {
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
                        "test_timestamp": None
                    }
                }

                # Test each query type for this model
                successful_count = 0
                for query_type in self.query_types:
                    result = await self.test_model_with_query(provider, model, query_type)

                    model_results[model_key]["performance"]["total_tests"] += 1

                    if result["success"]:
                        successful_count += 1
                        model_results[model_key]["performance"]["successful_tests"] += 1

                        # Store the response content in the structured format
                        model_results[model_key][query_type] = result["response_content"]

                        # Store evaluation and performance data
                        model_results[model_key]["evaluation"][query_type] = result["evaluation"]
                        model_results[model_key]["performance"]["response_times"][query_type] = result["response_time"]
                        model_results[model_key]["metadata"]["test_timestamp"] = result["timestamp"]

                # Calculate success rate
                model_results[model_key]["performance"]["success_rate"] = successful_count / len(self.query_types)

                # Summary for this model
                print(f"   📊 {successful_count}/{len(self.query_types)} tests successful")

        return {
            "test_config": {
                "image_path": str(self.image_path),
                "providers": self.providers,
                "query_types": self.query_types,
                "total_models": len(model_results)
            },
            "reference_analysis": REFERENCE_ANALYSIS,
            "results": model_results,
            "summary": self.generate_summary_from_structured(model_results)
        }

    def generate_summary_from_structured(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive test summary from structured results."""
        total_models = len(model_results)
        successful_models = len([r for r in model_results.values() if r["performance"]["success_rate"] > 0])

        # Performance by provider
        provider_stats = {}
        for provider in self.providers:
            provider_models = {k: v for k, v in model_results.items() if v["metadata"]["provider"] == provider}
            if provider_models:
                total_tests = sum(r["performance"]["total_tests"] for r in provider_models.values())
                successful_tests = sum(r["performance"]["successful_tests"] for r in provider_models.values())
                avg_times = [sum(r["performance"]["response_times"].values()) / len(r["performance"]["response_times"])
                            for r in provider_models.values() if r["performance"]["response_times"]]

                provider_stats[provider] = {
                    "models_tested": len(provider_models),
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                    "avg_response_time": sum(avg_times) / len(avg_times) if avg_times else 0
                }

        # Performance by query type
        query_stats = {}
        for query_type in self.query_types:
            models_with_query = [r for r in model_results.values() if query_type in r["evaluation"]]
            if models_with_query:
                if query_type == "keywords":
                    scores = [r["evaluation"][query_type]["f1"] for r in models_with_query]
                elif query_type == "summary":
                    scores = [r["evaluation"][query_type]["coverage_score"] for r in models_with_query]
                elif query_type == "structured":
                    scores = [r["evaluation"][query_type]["structure_score"] for r in models_with_query]
                else:
                    scores = [0]

                times = [r["performance"]["response_times"][query_type] for r in models_with_query if query_type in r["performance"]["response_times"]]

                query_stats[query_type] = {
                    "successful_tests": len(models_with_query),
                    "avg_quality_score": sum(scores) / len(scores) if scores else 0,
                    "avg_response_time": sum(times) / len(times) if times else 0
                }

        return {
            "overall": {
                "total_models": total_models,
                "successful_models": successful_models,
                "overall_success_rate": successful_models / total_models if total_models > 0 else 0,
                "test_timestamp": datetime.now().isoformat()
            },
            "by_provider": provider_stats,
            "by_query_type": query_stats
        }

    def save_results(self, results: Dict, output_file: str = None) -> str:
        """Save results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"comprehensive_vision_test_{timestamp}.json"

        output_path = Path("tests/vision_comprehensive") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_path}")
        return str(output_path)

async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Comprehensive Vision Testing")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--providers", nargs="+", default=["lmstudio", "ollama", "huggingface", "anthropic", "openai"],
                       help="Providers to test")
    parser.add_argument("--output", help="Output file name")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")

    args = parser.parse_args()

    # Find image file
    image_path = None
    possible_paths = [
        Path(args.image),
        Path(f"tests/vision_examples/{args.image}"),
        Path(f"tests/vision_examples/{args.image}.jpg"),
        Path(f"tests/vision_examples/{args.image}_mp.jpg")
    ]

    for path in possible_paths:
        if path.exists():
            image_path = path
            break

    if not image_path:
        print(f"❌ Image not found: {args.image}")
        return 1

    # Run tests
    tester = ComprehensiveVisionTester(image_path, args.providers)
    results = await tester.run_comprehensive_tests()

    # Save results if requested
    if args.save_results:
        tester.save_results(results, args.output)

    # Print final summary
    print("\n🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    summary = results["summary"]
    print(f"Total models: {summary['overall']['total_models']}")
    print(f"Successful models: {summary['overall']['successful_models']}")
    print(f"Success rate: {summary['overall']['overall_success_rate']:.1%}")

    return 0

if __name__ == "__main__":
    asyncio.run(main())