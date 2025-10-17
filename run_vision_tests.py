#!/usr/bin/env python3
"""
AbstractCore Vision Models Test Runner

Comprehensive test runner for all vision model functionality including:
- LMStudio model testing
- Ollama model testing
- Cross-provider integration testing
- Real model integration (optional)

Usage:
    python run_vision_tests.py                    # Run all tests (without real models)
    python run_vision_tests.py --real-models      # Run all tests including real model integration
    python run_vision_tests.py --lmstudio-only    # Test only LMStudio models
    python run_vision_tests.py --ollama-only      # Test only Ollama models
    python run_vision_tests.py --validation       # Run validation script first
    python run_vision_tests.py --verbose          # Verbose output
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("STDERR:", result.stderr)

    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        if not input("Continue anyway? (y/n): ").lower().startswith('y'):
            sys.exit(result.returncode)
    elif result.returncode == 0:
        print(f"✅ {description} completed successfully")

    return result.returncode == 0


def check_dependencies():
    """Check that required dependencies are installed."""
    print("🔍 Checking dependencies...")

    try:
        import PIL
        print(f"✅ PIL/Pillow: {PIL.__version__}")
    except ImportError:
        print("❌ PIL/Pillow not found. Install with: pip install Pillow")
        return False

    try:
        import pytest
        print(f"✅ pytest: {pytest.__version__}")
    except ImportError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False

    try:
        from abstractcore.media.utils import get_scaler
        print("✅ AbstractCore media utilities available")
    except ImportError as e:
        print(f"❌ AbstractCore media utilities not available: {e}")
        return False

    return True


def setup_environment(real_models=False):
    """Set up environment variables for testing."""
    print("🔧 Setting up test environment...")

    # Set up virtual environment
    venv_path = Path(".venv")
    if venv_path.exists():
        if os.name == 'nt':  # Windows
            activation_cmd = ".\\.venv\\Scripts\\activate.bat"
        else:  # Unix/Linux/macOS
            activation_cmd = "source .venv/bin/activate"
        print(f"✅ Using virtual environment: {venv_path}")
    else:
        activation_cmd = ""
        print("⚠️  No virtual environment found (.venv), using system Python")

    # Set environment variables for testing
    os.environ['PYTHONPATH'] = str(Path.cwd())

    if real_models:
        os.environ['TEST_WITH_REAL_MODELS'] = '1'
        print("✅ Real model testing enabled")
    else:
        os.environ.pop('TEST_WITH_REAL_MODELS', None)
        print("ℹ️  Real model testing disabled (use --real-models to enable)")

    return activation_cmd


def main():
    parser = argparse.ArgumentParser(description='Run AbstractCore vision model tests')
    parser.add_argument('--real-models', action='store_true',
                       help='Enable testing with real model endpoints (requires running LMStudio/Ollama)')
    parser.add_argument('--lmstudio-only', action='store_true',
                       help='Run only LMStudio model tests')
    parser.add_argument('--ollama-only', action='store_true',
                       help='Run only Ollama model tests')
    parser.add_argument('--validation', action='store_true',
                       help='Run media system validation first')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')

    args = parser.parse_args()

    print("🚀 AbstractCore Vision Models Test Runner")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)

    # Setup environment
    activation_cmd = setup_environment(args.real_models)

    success = True

    # Run validation if requested
    if args.validation:
        cmd = f"{activation_cmd} && python validate_media_system.py" if activation_cmd else "python validate_media_system.py"
        if not run_command(cmd, "Running media system validation", check=False):
            print("⚠️  Validation had issues, but continuing with tests...")

    # Build pytest command
    pytest_args = []

    if args.verbose:
        pytest_args.append("-v")

    if args.coverage:
        pytest_args.extend(["--cov=abstractcore.media", "--cov-report=html", "--cov-report=term"])

    # Determine which tests to run
    if args.lmstudio_only:
        test_files = ["tests/vision_models/test_lmstudio_vision.py"]
        description = "LMStudio vision model tests"
    elif args.ollama_only:
        test_files = ["tests/vision_models/test_ollama_vision.py"]
        description = "Ollama vision model tests"
    else:
        test_files = [
            "tests/vision_models/test_lmstudio_vision.py",
            "tests/vision_models/test_ollama_vision.py",
            "tests/vision_models/test_vision_integration.py"
        ]
        description = "All vision model tests"

    # Add benchmark tests if requested
    if args.benchmark:
        pytest_args.append("-k")
        pytest_args.append("benchmark or performance")
        description += " (with benchmarks)"

    # Run the tests
    for test_file in test_files:
        if Path(test_file).exists():
            cmd_parts = []
            if activation_cmd:
                cmd_parts.append(activation_cmd)

            cmd_parts.append(f"python -m pytest {test_file}")
            cmd_parts.extend(pytest_args)

            cmd = " && ".join(cmd_parts) if len(cmd_parts) > 1 else cmd_parts[0]

            if not run_command(cmd, f"Running {test_file}", check=False):
                success = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
            success = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    if success:
        print("🎉 All vision model tests completed successfully!")

        if args.real_models:
            print("✅ Real model integration tests passed")
        else:
            print("ℹ️  Real model tests skipped (use --real-models to enable)")

        print("\n📋 What was tested:")
        print("• Image scaling utility for different model resolutions")
        print("• Model-specific optimization (Qwen vs Gemma)")
        print("• Cross-provider consistency")
        print("• Edge case handling (small/large images, aspect ratios)")
        print("• Format conversion (PNG, JPEG, transparency)")
        print("• Performance and memory efficiency")

        if args.real_models:
            print("• Real model endpoints (LMStudio/Ollama)")
            print("• End-to-end vision processing")

        print("\n🚀 Vision model system is ready for use!")

        print("\n📖 To test with real models:")
        print("1. Start LMStudio: http://localhost:1234")
        print("2. Start Ollama: http://localhost:11434")
        print("3. Load vision models (qwen3-vl, gemma3-4b, etc.)")
        print("4. Run: python run_vision_tests.py --real-models")

    else:
        print("❌ Some tests failed or had issues")
        print("🔍 Check the output above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()