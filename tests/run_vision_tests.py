#!/usr/bin/env python3
"""
Vision Test Runner for AbstractCore
Provides convenient ways to run vision tests with different configurations.

Usage:
    python tests/run_vision_tests.py --help
    python tests/run_vision_tests.py --smoke                    # Quick smoke tests
    python tests/run_vision_tests.py --single ollama qwen2.5vl:7b  # Test specific model
    python tests/run_vision_tests.py --comprehensive            # All models, all images
    python tests/run_vision_tests.py --available               # Show available models
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add abstractcore to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractcore import create_llm
from abstractcore.media.capabilities import is_vision_model


class VisionTestRunner:
    """Convenient test runner for vision capabilities."""

    def __init__(self):
        self.base_cmd = ["python", "-m", "pytest"]
        self.test_dir = Path(__file__).parent

    def run_smoke_tests(self):
        """Run quick smoke tests to verify basic vision functionality."""
        print("🔥 Running Vision Smoke Tests")
        print("=" * 50)

        cmd = self.base_cmd + [
            str(self.test_dir / "test_vision_single_model.py::TestQuickVisionSmoke"),
            "-v", "-s"
        ]

        return subprocess.run(cmd).returncode

    def run_single_model_test(self, provider: str, model: str):
        """Run comprehensive test for a single model."""
        print(f"🎯 Testing Single Model: {provider}/{model}")
        print("=" * 50)

        # Set environment variable for the test
        env = os.environ.copy()
        env["VISION_MODEL"] = f"{provider}/{model}"

        cmd = self.base_cmd + [
            str(self.test_dir / "test_vision_single_model.py::TestSpecificModel::test_env_model_all_images"),
            "-v", "-s"
        ]

        return subprocess.run(cmd, env=env).returncode

    def run_comprehensive_tests(self):
        """Run comprehensive tests across all available models and images."""
        print("🎯 Running Comprehensive Vision Tests")
        print("=" * 50)

        cmd = self.base_cmd + [
            str(self.test_dir / "test_vision_comprehensive.py::TestAllModelsAllImages::test_all_available_models_all_images"),
            "-v", "-s"
        ]

        return subprocess.run(cmd).returncode

    def run_provider_tests(self, provider: str):
        """Run tests for a specific provider."""
        print(f"🔧 Testing Provider: {provider}")
        print("=" * 50)

        cmd = self.base_cmd + [
            str(self.test_dir / "test_vision_comprehensive.py"),
            str(self.test_dir / "test_vision_single_model.py"),
            "-k", provider,
            "-v", "-s"
        ]

        return subprocess.run(cmd).returncode

    def run_image_tests(self, image_name: str):
        """Run tests for a specific image."""
        print(f"📸 Testing Image: {image_name}")
        print("=" * 50)

        cmd = self.base_cmd + [
            str(self.test_dir / "test_vision_comprehensive.py::TestSingleImageVision"),
            "-k", image_name,
            "-v", "-s"
        ]

        return subprocess.run(cmd).returncode

    def show_available_models(self):
        """Show available vision models."""
        print("📋 Available Vision Models")
        print("=" * 50)

        provider_models = {
            "ollama": ["qwen2.5vl:7b", "llama3.2-vision:11b", "gemma3:4b"],
            "lmstudio": ["qwen/qwen2.5-vl-7b", "qwen/qwen3-vl-4b", "google/gemma-3n-e4b"],
            "openai": ["gpt-4o", "gpt-4-turbo"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
            "huggingface": ["unsloth/Qwen2.5-VL-7B-Instruct-GGUF"]
        }

        available_count = 0

        for provider, models in provider_models.items():
            print(f"\n🔧 {provider.upper()}:")

            for model in models:
                try:
                    # Quick availability check
                    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
                        status = "❌ (No API Key)"
                    elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
                        status = "❌ (No API Key)"
                    else:
                        # Try to create LLM instance
                        llm = create_llm(provider, model=model)
                        if is_vision_model(model):
                            status = "✅ Available"
                            available_count += 1
                        else:
                            status = "⚠️ (No Vision)"

                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ["connection", "refused", "timeout", "not found"]):
                        status = "❌ (Not Running)"
                    else:
                        status = f"❌ ({str(e)[:30]}...)"

                print(f"  {model:<40} {status}")

        print(f"\n📊 Total Available Vision Models: {available_count}")

        # Show usage examples
        print(f"\n💡 Usage Examples:")
        print(f"   Smoke test: python {__file__} --smoke")
        print(f"   Single model: python {__file__} --single ollama qwen2.5vl:7b")
        print(f"   Provider: python {__file__} --provider ollama")
        print(f"   Comprehensive: python {__file__} --comprehensive")

    def run_all_tests(self):
        """Run all vision tests."""
        print("🎯 Running ALL Vision Tests")
        print("=" * 50)

        cmd = self.base_cmd + [
            str(self.test_dir / "test_vision_comprehensive.py"),
            str(self.test_dir / "test_vision_single_model.py"),
            "-v"
        ]

        return subprocess.run(cmd).returncode

    def run_quality_benchmarks(self):
        """Run quality benchmark tests."""
        print("📊 Running Quality Benchmarks")
        print("=" * 50)

        cmd = self.base_cmd + [
            str(self.test_dir / "test_vision_comprehensive.py::TestVisionQualityBenchmarks"),
            "-v", "-s"
        ]

        return subprocess.run(cmd).returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Vision Test Runner for AbstractCore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --smoke                           # Quick smoke tests
  %(prog)s --single ollama qwen2.5vl:7b     # Test specific model
  %(prog)s --provider ollama                 # Test all Ollama models
  %(prog)s --image mystery1_mp.jpg           # Test specific image
  %(prog)s --comprehensive                   # Full matrix test
  %(prog)s --quality                         # Quality benchmarks
  %(prog)s --available                       # Show available models
  %(prog)s --all                            # Run everything

Environment Variables:
  OPENAI_API_KEY      OpenAI API key for GPT-4o tests
  ANTHROPIC_API_KEY   Anthropic API key for Claude tests
  VISION_MODEL        Specific model for single model tests (provider/model)
        """
    )

    parser.add_argument("--smoke", action="store_true",
                       help="Run quick smoke tests")
    parser.add_argument("--single", nargs=2, metavar=("PROVIDER", "MODEL"),
                       help="Test specific model (e.g., --single ollama qwen2.5vl:7b)")
    parser.add_argument("--provider", metavar="PROVIDER",
                       help="Test specific provider (e.g., --provider ollama)")
    parser.add_argument("--image", metavar="IMAGE_NAME",
                       help="Test specific image (e.g., --image mystery1_mp.jpg)")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive tests (all models, all images)")
    parser.add_argument("--quality", action="store_true",
                       help="Run quality benchmark tests")
    parser.add_argument("--available", action="store_true",
                       help="Show available vision models")
    parser.add_argument("--all", action="store_true",
                       help="Run all vision tests")

    args = parser.parse_args()

    runner = VisionTestRunner()

    if args.available:
        runner.show_available_models()
        return 0

    elif args.smoke:
        return runner.run_smoke_tests()

    elif args.single:
        provider, model = args.single
        return runner.run_single_model_test(provider, model)

    elif args.provider:
        return runner.run_provider_tests(args.provider)

    elif args.image:
        return runner.run_image_tests(args.image)

    elif args.comprehensive:
        return runner.run_comprehensive_tests()

    elif args.quality:
        return runner.run_quality_benchmarks()

    elif args.all:
        return runner.run_all_tests()

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())