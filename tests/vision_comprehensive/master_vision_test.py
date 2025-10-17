#!/usr/bin/env python3
"""
Master Vision Testing Framework
Runs comprehensive vision tests across all available test images and aggregates statistics.
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

from comprehensive_vision_tester import ComprehensiveVisionTester

class MasterVisionTester:
    """Master tester that runs comprehensive tests across multiple images."""

    def __init__(self, providers: List[str] = None):
        self.providers = providers or ["lmstudio", "ollama", "huggingface"]
        self.test_images = [
            "mystery1_mp.jpg",
            "mystery2_sc.jpg",
            "mystery3_us.jpg",
            "mystery4_wh.jpg"
        ]
        self.results = {}
        self.aggregated_stats = {}

    async def run_master_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests on all images and aggregate results."""
        print("🎯 MASTER VISION TESTING FRAMEWORK")
        print("=" * 70)
        print(f"📸 Testing {len(self.test_images)} images: {', '.join([img.replace('_mp.jpg', '').replace('_sc.jpg', '').replace('_us.jpg', '').replace('_wh.jpg', '') for img in self.test_images])}")
        print(f"🔧 Providers: {', '.join(self.providers)}")
        print()

        master_start_time = time.time()

        # Run tests for each image
        for i, image_file in enumerate(self.test_images, 1):
            image_name = image_file.replace('_mp.jpg', '').replace('_sc.jpg', '').replace('_us.jpg', '').replace('_wh.jpg', '')
            image_path = Path("tests/vision_examples") / image_file

            if not image_path.exists():
                print(f"⚠️  Image not found: {image_path}")
                continue

            print(f"\n🖼️  [{i}/{len(self.test_images)}] TESTING IMAGE: {image_name}")
            print("-" * 50)

            try:
                # Create tester for this image
                tester = ComprehensiveVisionTester(image_path, self.providers)

                # Run the comprehensive test
                image_results = await tester.run_comprehensive_tests()

                # Store results
                self.results[image_name] = image_results

                # Print quick summary for this image
                summary = image_results["summary"]["overall"]
                print(f"   📊 {summary['successful_models']}/{summary['total_models']} models successful ({summary['overall_success_rate']:.1%})")

            except Exception as e:
                print(f"   ❌ Failed to test {image_name}: {str(e)}")
                self.results[image_name] = {"error": str(e)}

        master_end_time = time.time()

        # Aggregate statistics across all images
        self.aggregated_stats = self._aggregate_statistics()

        # Create master report
        master_report = {
            "test_config": {
                "images_tested": list(self.results.keys()),
                "providers": self.providers,
                "total_images": len(self.test_images),
                "successful_images": len([r for r in self.results.values() if "error" not in r]),
                "master_test_duration": master_end_time - master_start_time,
                "timestamp": datetime.now().isoformat()
            },
            "individual_results": self.results,
            "aggregated_statistics": self.aggregated_stats,
            "master_summary": self._generate_master_summary()
        }

        return master_report

    def _aggregate_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics across all successful image tests."""
        aggregated = {
            "by_provider": {},
            "by_model": {},
            "by_query_type": {},
            "performance_metrics": {},
            "success_rates": {}
        }

        successful_results = [r for r in self.results.values() if "error" not in r]

        if not successful_results:
            return aggregated

        # Aggregate by provider
        provider_stats = {}
        for provider in self.providers:
            total_tests = 0
            successful_tests = 0
            response_times = []

            for result in successful_results:
                if provider in result["summary"]["by_provider"]:
                    provider_data = result["summary"]["by_provider"][provider]
                    total_tests += provider_data["total_tests"]
                    successful_tests += provider_data["successful_tests"]
                    if provider_data["avg_response_time"] > 0:
                        response_times.append(provider_data["avg_response_time"])

            provider_stats[provider] = {
                "total_tests_across_images": total_tests,
                "successful_tests_across_images": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "images_tested": len(successful_results)
            }

        aggregated["by_provider"] = provider_stats

        # Aggregate by query type
        query_stats = {}
        query_types = ["keywords", "summary", "structured"]

        for query_type in query_types:
            successful_query_tests = 0
            quality_scores = []
            response_times = []

            for result in successful_results:
                if query_type in result["summary"]["by_query_type"]:
                    query_data = result["summary"]["by_query_type"][query_type]
                    successful_query_tests += query_data["successful_tests"]
                    quality_scores.append(query_data["avg_quality_score"])
                    response_times.append(query_data["avg_response_time"])

            query_stats[query_type] = {
                "successful_tests_across_images": successful_query_tests,
                "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "consistency": self._calculate_consistency(quality_scores)
            }

        aggregated["by_query_type"] = query_stats

        # Calculate model consistency across images
        model_consistency = {}
        all_models = set()
        for result in successful_results:
            all_models.update(result["results"].keys())

        for model in all_models:
            success_rates = []
            for result in successful_results:
                if model in result["results"]:
                    success_rates.append(result["results"][model]["performance"]["success_rate"])

            if success_rates:
                model_consistency[model] = {
                    "avg_success_rate": sum(success_rates) / len(success_rates),
                    "consistency": self._calculate_consistency(success_rates),
                    "images_tested": len(success_rates)
                }

        aggregated["by_model"] = model_consistency

        return aggregated

    def _calculate_consistency(self, values: List[float]) -> str:
        """Calculate consistency rating based on standard deviation."""
        if len(values) < 2:
            return "insufficient_data"

        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5

        # Normalize by average to get coefficient of variation
        cv = std_dev / avg if avg > 0 else float('inf')

        if cv < 0.1:
            return "very_consistent"
        elif cv < 0.2:
            return "consistent"
        elif cv < 0.3:
            return "moderate"
        else:
            return "inconsistent"

    def _generate_master_summary(self) -> Dict[str, Any]:
        """Generate a high-level master summary."""
        successful_images = len([r for r in self.results.values() if "error" not in r])
        total_images = len(self.test_images)

        # Calculate overall statistics
        total_tests = 0
        successful_tests = 0

        for result in self.results.values():
            if "error" not in result:
                summary = result["summary"]["overall"]
                total_tests += summary["total_models"] * 3  # 3 query types per model
                successful_tests += sum(
                    result["summary"]["by_query_type"][qt]["successful_tests"]
                    for qt in ["keywords", "summary", "structured"]
                    if qt in result["summary"]["by_query_type"]
                )

        return {
            "images_success_rate": successful_images / total_images if total_images > 0 else 0,
            "overall_test_success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_images_tested": total_images,
            "successful_images": successful_images,
            "total_model_tests": total_tests,
            "successful_model_tests": successful_tests,
            "most_consistent_provider": self._find_most_consistent_provider(),
            "most_consistent_query_type": self._find_most_consistent_query_type(),
            "top_performing_models": self._find_top_performing_models()
        }

    def _find_most_consistent_provider(self) -> Optional[str]:
        """Find the provider with the most consistent performance across images."""
        if not self.aggregated_stats.get("by_provider"):
            return None

        best_provider = None
        best_consistency = -1

        for provider, stats in self.aggregated_stats["by_provider"].items():
            # Consistency metric: success rate with low variance preferred
            if stats["success_rate"] > best_consistency:
                best_consistency = stats["success_rate"]
                best_provider = provider

        return best_provider

    def _find_most_consistent_query_type(self) -> Optional[str]:
        """Find the query type with the most consistent performance."""
        if not self.aggregated_stats.get("by_query_type"):
            return None

        best_query = None
        best_score = -1

        for query_type, stats in self.aggregated_stats["by_query_type"].items():
            if stats["avg_quality_score"] > best_score:
                best_score = stats["avg_quality_score"]
                best_query = query_type

        return best_query

    def _find_top_performing_models(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Find the top N performing models across all images."""
        if not self.aggregated_stats.get("by_model"):
            return []

        model_scores = []
        for model, stats in self.aggregated_stats["by_model"].items():
            model_scores.append({
                "model": model,
                "avg_success_rate": stats["avg_success_rate"],
                "consistency": stats["consistency"],
                "images_tested": stats["images_tested"]
            })

        # Sort by success rate descending
        model_scores.sort(key=lambda x: x["avg_success_rate"], reverse=True)
        return model_scores[:top_n]

    def save_master_report(self, report: Dict, output_file: str = None) -> str:
        """Save master report to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"master_vision_test_{timestamp}.json"

        output_path = Path("tests/vision_comprehensive") / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Master report saved to: {output_path}")
        return str(output_path)

    def print_master_summary(self, report: Dict):
        """Print a comprehensive master summary."""
        print("\n" + "=" * 70)
        print("🎯 MASTER VISION TESTING SUMMARY")
        print("=" * 70)

        config = report["test_config"]
        summary = report["master_summary"]

        print(f"📸 Images tested: {config['successful_images']}/{config['total_images']} ({summary['images_success_rate']:.1%})")
        print(f"⚡ Total duration: {config['master_test_duration']:.1f}s")
        print(f"🎯 Overall success rate: {summary['overall_test_success_rate']:.1%}")
        print(f"📊 Total tests: {summary['successful_model_tests']}/{summary['total_model_tests']}")

        if summary.get("most_consistent_provider"):
            print(f"🥇 Most consistent provider: {summary['most_consistent_provider']}")

        if summary.get("most_consistent_query_type"):
            print(f"🎨 Best query type: {summary['most_consistent_query_type']}")

        if summary.get("top_performing_models"):
            print(f"\n🏆 TOP PERFORMING MODELS:")
            for i, model in enumerate(summary["top_performing_models"], 1):
                print(f"   {i}. {model['model']}: {model['avg_success_rate']:.1%} success ({model['consistency']} consistency)")

async def main():
    """Main master test runner."""
    parser = argparse.ArgumentParser(description="Master Vision Testing - Tests all images")
    parser.add_argument("--providers", nargs="+", default=["lmstudio", "ollama", "huggingface"],
                       help="Providers to test")
    parser.add_argument("--output", help="Output file name for master report")
    parser.add_argument("--save-results", action="store_true", help="Save master report to file")

    args = parser.parse_args()

    # Run master tests
    master_tester = MasterVisionTester(args.providers)
    master_report = await master_tester.run_master_tests()

    # Save results if requested
    if args.save_results:
        master_tester.save_master_report(master_report, args.output)

    # Print comprehensive summary
    master_tester.print_master_summary(master_report)

    return 0

if __name__ == "__main__":
    asyncio.run(main())