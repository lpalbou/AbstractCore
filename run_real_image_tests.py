#!/usr/bin/env python3
"""
Real Image Vision Analysis Test Runner

Comprehensive test runner for analyzing real images with vision models,
measuring keyword overlap and generating detailed performance reports.

Usage:
    python run_real_image_tests.py                    # Test functionality without real models
    python run_real_image_tests.py --real-models      # Run with real model endpoints
    python run_real_image_tests.py --lmstudio-only    # Test only LMStudio models
    python run_real_image_tests.py --ollama-only      # Test only Ollama models
    python run_real_image_tests.py --image mystery1   # Test with specific image only
    python run_real_image_tests.py --save-results     # Save detailed JSON results
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path.cwd()))

from tests.vision_models.test_real_image_analysis import RealImageVisionTester


def check_test_images():
    """Check that test images are available."""
    base_path = Path("tests/vision_examples")

    if not base_path.exists():
        print(f"❌ Test images directory not found: {base_path}")
        print("   Please ensure the mystery images are placed in this directory")
        return False

    expected_images = [
        "mystery1_mp.jpg",
        "mystery2_sc.jpg",
        "mystery3_us.jpg",
        "mystery4_wh.jpg"
    ]

    found_images = []
    for img in expected_images:
        img_path = base_path / img
        if img_path.exists():
            found_images.append(img)
            print(f"✅ Found: {img}")
        else:
            print(f"❌ Missing: {img}")

    print(f"\n📸 Found {len(found_images)}/{len(expected_images)} test images")
    return len(found_images) > 0


def test_functionality_only():
    """Test the functionality without real models."""
    print("🔧 TESTING FUNCTIONALITY (NO REAL MODELS)")
    print("=" * 50)

    tester = RealImageVisionTester()

    # Test keyword extraction
    print("Testing keyword extraction...")
    test_response = "This image shows a beautiful mountain landscape with a hiking trail and wooden fences."
    keywords = tester.extract_keywords_from_response(test_response)
    print(f"✅ Extracted {len(keywords)} keywords: {list(keywords)[:10]}")

    # Test overlap calculation
    print("\nTesting overlap calculation...")
    model_keywords = {"mountain", "trail", "hiking", "fence", "sky"}
    reference_keywords = {
        "primary": ["mountain", "trail", "hiking"],
        "secondary": ["fence", "sky", "landscape"]
    }
    overlap = tester.calculate_overlap(model_keywords, reference_keywords)
    print(f"✅ Overlap metrics calculated: {overlap['total_recall']:.1%} recall, {overlap['precision']:.1%} precision")

    # Test image setup
    print("\nTesting image setup...")
    try:
        images = tester.setup_images()
        print(f"✅ Found {len(images)} test images")

        # Test image resizing (in memory)
        if images:
            first_image = next(iter(images.values()))
            resized = tester.resize_image_for_model(first_image, "qwen/qwen2.5-vl-7b")
            print(f"✅ Image resizing works: {resized.size}")

    except Exception as e:
        print(f"❌ Image setup failed: {e}")
        return False

    print("\n🎉 All functionality tests passed!")
    return True


def run_with_real_models(providers=None, specific_image=None, save_results=False):
    """Run tests with real models."""
    print("🎯 RUNNING REAL IMAGE VISION ANALYSIS")
    print("=" * 60)

    # Set environment variable for real model testing
    os.environ['TEST_WITH_REAL_MODELS'] = '1'

    tester = RealImageVisionTester()

    # Filter models if specific provider requested
    if providers:
        if "lmstudio" not in providers:
            tester.LMSTUDIO_MODELS = []
        if "ollama" not in providers:
            tester.OLLAMA_MODELS = []

    # Filter images if specific image requested
    if specific_image:
        original_setup = tester.setup_images
        def filtered_setup():
            all_images = original_setup()
            return {k: v for k, v in all_images.items() if specific_image in k}
        tester.setup_images = filtered_setup

    # Run comprehensive tests
    try:
        results = tester.run_comprehensive_tests(test_real_models=True)

        if results.get('skipped'):
            print("⚠️  Tests were skipped - check model availability")
            return False

        # Generate and display report
        report = tester.generate_report(results)
        print("\n" + report)

        # Save results if requested
        if save_results:
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"vision_test_results_{timestamp}.json"

            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n💾 Detailed results saved to: {results_file}")

        # Summary
        success_rate = results['summary']['successful_tests'] / results['summary']['total_tests'] if results['summary']['total_tests'] > 0 else 0
        print(f"\n📊 FINAL SUMMARY")
        print(f"Success rate: {success_rate:.1%} ({results['summary']['successful_tests']}/{results['summary']['total_tests']})")

        if success_rate > 0.8:
            print("🎉 Excellent performance across models!")
        elif success_rate > 0.6:
            print("✅ Good performance, some models may need optimization")
        else:
            print("⚠️  Performance concerns detected, check model availability")

        return success_rate > 0.5

    except Exception as e:
        print(f"❌ Real model testing failed: {e}")
        print("   Check that LMStudio/Ollama are running with vision models loaded")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run real image vision analysis tests')
    parser.add_argument('--real-models', action='store_true',
                       help='Test with real model endpoints (requires LMStudio/Ollama)')
    parser.add_argument('--lmstudio-only', action='store_true',
                       help='Test only LMStudio models')
    parser.add_argument('--ollama-only', action='store_true',
                       help='Test only Ollama models')
    parser.add_argument('--image', type=str,
                       help='Test with specific image only (e.g., "mystery1")')
    parser.add_argument('--save-results', action='store_true',
                       help='Save detailed JSON results to file')
    parser.add_argument('--check-images', action='store_true',
                       help='Only check if test images are available')

    args = parser.parse_args()

    print("🎯 REAL IMAGE VISION ANALYSIS TEST RUNNER")
    print("=" * 60)

    # Check test images first
    if not check_test_images():
        print("\n❌ Cannot proceed without test images")
        return 1

    if args.check_images:
        print("\n✅ Image check completed")
        return 0

    # Determine providers to test
    providers = []
    if not args.ollama_only:
        providers.append("lmstudio")
    if not args.lmstudio_only:
        providers.append("ollama")

    success = False

    if args.real_models:
        print(f"\n🚀 Testing with real models: {', '.join(providers)}")
        print("   Ensure LMStudio (port 1234) and/or Ollama (port 11434) are running")
        print("   with vision models loaded")

        success = run_with_real_models(
            providers=providers,
            specific_image=args.image,
            save_results=args.save_results
        )
    else:
        print("\n🔧 Testing functionality only (no real models)")
        success = test_functionality_only()

    if success:
        print("\n🎉 Tests completed successfully!")

        if not args.real_models:
            print("\n💡 To test with real models:")
            print("   1. Start LMStudio: http://localhost:1234")
            print("   2. Load vision models: qwen2.5-vl-7b, gemma-3n-e4b, magistral-small-2509")
            print("   3. Start Ollama: http://localhost:11434")
            print("   4. Pull vision models: qwen2.5vl:7b, gemma3:4b, gemma3n:e4b, gemma3n:e2b, llama3.2-vision:11b")
            print("   5. Optionally test HuggingFace: Qwen/Qwen3-VL-8B-Instruct-FP8")
            print("   6. Run: python run_real_image_tests.py --real-models")

        return 0
    else:
        print("\n❌ Tests failed or had significant issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())