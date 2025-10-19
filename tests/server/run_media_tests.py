#!/usr/bin/env python3
"""
Comprehensive Server Media Tests Runner

This script runs all server media tests across different data modalities,
validates the implementation, and provides a detailed report.

Usage:
    python tests/server/run_media_tests.py [--quick] [--provider ollama|lmstudio] [--verbose]
"""

import argparse
import sys
import os
import subprocess
import time
import requests
import json
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.server.media_vision import VisionTestHelper
from tests.server.media_documents import DocumentTestHelper
from tests.server.media_data import DataTestHelper
from tests.server.media_mixed import MixedMediaTestHelper

class ServerMediaTestRunner:
    """Comprehensive test runner for server media capabilities."""

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.test_modules = [
            "tests.server.media_vision",
            "tests.server.media_documents",
            "tests.server.media_data",
            "tests.server.media_mixed"
        ]

    def check_server_health(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Server is healthy and responding")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False

    def check_available_models(self) -> Dict[str, List[str]]:
        """Check available models for each provider."""
        providers = ["ollama", "lmstudio"]
        available_models = {}

        for provider in providers:
            try:
                response = requests.get(f"{self.server_url}/providers/{provider}/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    available_models[provider] = models
                    print(f"📝 {provider.title()}: {len(models)} models available")

                    # Show specific model types
                    vision_models = [m for m in models if any(term in m.lower() for term in ["vl", "vision"])]
                    text_models = [m for m in models if not any(term in m.lower() for term in ["vl", "vision"])]

                    if vision_models:
                        print(f"   👁️  Vision models: {vision_models[:3]}{'...' if len(vision_models) > 3 else ''}")
                    if text_models:
                        print(f"   📄 Text models: {text_models[:3]}{'...' if len(text_models) > 3 else ''}")
                else:
                    print(f"⚠️  {provider.title()}: No models available")
                    available_models[provider] = []
            except Exception as e:
                print(f"❌ Failed to get {provider} models: {e}")
                available_models[provider] = []

        return available_models

    def validate_dependencies(self) -> bool:
        """Validate required dependencies for tests."""
        print("\n🔍 Checking test dependencies...")

        required_packages = [
            "PIL",           # For image creation
            "reportlab",     # For PDF creation
            "openpyxl",      # For Excel files
            "python-docx",   # For Word documents
            "python-pptx",   # For PowerPoint
            "requests",      # For HTTP requests
            "pytest"         # For test execution
        ]

        missing_packages = []

        for package in required_packages:
            try:
                if package == "PIL":
                    import PIL
                elif package == "python-docx":
                    import docx
                elif package == "python-pptx":
                    import pptx
                else:
                    __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - MISSING")
                missing_packages.append(package)

        if missing_packages:
            print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
            print("📦 Install with: pip install " + " ".join(missing_packages))
            return False

        print("✅ All dependencies available")
        return True

    def run_quick_validation(self) -> bool:
        """Run quick validation tests to ensure basic functionality."""
        print("\n🚀 Running quick validation tests...")

        helpers = [
            ("Vision", VisionTestHelper()),
            ("Documents", DocumentTestHelper()),
            ("Data", DataTestHelper()),
            ("Mixed Media", MixedMediaTestHelper())
        ]

        for name, helper in helpers:
            try:
                if hasattr(helper, 'create_test_image'):
                    # Test image creation
                    image_data = helper.create_test_image("Validation Test")
                    assert len(image_data) > 0
                    print(f"   ✅ {name}: Image creation")

                elif hasattr(helper, 'create_test_pdf'):
                    # Test document creation
                    pdf_path = helper.create_test_pdf()
                    assert os.path.exists(pdf_path)
                    os.unlink(pdf_path)  # Cleanup
                    print(f"   ✅ {name}: Document creation")

                elif hasattr(helper, 'create_test_csv'):
                    # Test data file creation
                    csv_path = helper.create_test_csv()
                    assert os.path.exists(csv_path)
                    os.unlink(csv_path)  # Cleanup
                    print(f"   ✅ {name}: Data file creation")

                elif hasattr(helper, 'create_comprehensive_test_suite'):
                    # Test mixed media creation
                    files = helper.create_comprehensive_test_suite()
                    assert len(files) > 0
                    helper.cleanup_files(files)
                    print(f"   ✅ {name}: Mixed media creation")

            except Exception as e:
                print(f"   ❌ {name}: {e}")
                return False

        print("✅ Quick validation passed")
        return True

    def run_pytest_suite(self, provider_filter: Optional[str] = None, verbose: bool = False) -> Dict[str, any]:
        """Run the full pytest suite."""
        print(f"\n🧪 Running comprehensive test suite...")

        # Build pytest command
        cmd = ["python", "-m", "pytest"]

        # Add test modules
        for module in self.test_modules:
            cmd.append(module.replace(".", "/") + ".py")

        # Add filters
        if provider_filter:
            cmd.extend(["-k", f"provider and {provider_filter}"])

        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        # Add other options
        cmd.extend([
            "--tb=short",           # Short traceback format
            "--maxfail=10",         # Stop after 10 failures
            "--disable-warnings",   # Reduce noise
            "-x"                    # Stop on first failure for debugging
        ])

        print(f"   Running: {' '.join(cmd)}")

        # Run tests
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            duration = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": "Tests timed out after 10 minutes",
                "returncode": -1
            }

    def generate_report(self, test_results: Dict[str, any], available_models: Dict[str, List[str]]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("📊 AbstractCore Server Media Tests - Comprehensive Report")
        report.append("=" * 80)
        report.append("")

        # Server status
        report.append("🖥️  Server Status:")
        report.append(f"   URL: {self.server_url}")
        report.append(f"   Health: {'✅ Healthy' if self.check_server_health() else '❌ Unhealthy'}")
        report.append("")

        # Available models
        report.append("🤖 Available Models:")
        for provider, models in available_models.items():
            if models:
                vision_models = [m for m in models if any(term in m.lower() for term in ["vl", "vision"])]
                text_models = [m for m in models if not any(term in m.lower() for term in ["vl", "vision"])]

                report.append(f"   {provider.title()}: {len(models)} total")
                if vision_models:
                    report.append(f"      👁️  Vision: {', '.join(vision_models)}")
                if text_models:
                    report.append(f"      📄 Text: {', '.join(text_models[:3])}{'...' if len(text_models) > 3 else ''}")
            else:
                report.append(f"   {provider.title()}: ❌ No models available")
        report.append("")

        # Test results
        report.append("🧪 Test Results:")
        if test_results["success"]:
            report.append(f"   Status: ✅ PASSED")
        else:
            report.append(f"   Status: ❌ FAILED")

        report.append(f"   Duration: {test_results['duration']:.1f} seconds")
        report.append(f"   Return Code: {test_results['returncode']}")
        report.append("")

        # Parse pytest output for summary
        stdout = test_results.get("stdout", "")
        if "failed" in stdout.lower() or "error" in stdout.lower():
            report.append("❌ Failures/Errors:")
            for line in stdout.split('\n'):
                if any(term in line.lower() for term in ["failed", "error", "assert"]):
                    report.append(f"   {line.strip()}")
            report.append("")

        # Test coverage
        report.append("📋 Test Coverage:")
        test_areas = [
            ("Vision Processing", "Images with vision models (OpenAI format, streaming, error handling)"),
            ("Document Processing", "PDF, DOCX, XLSX, PPTX files (@filename syntax)"),
            ("Data Processing", "CSV, TSV, JSON, XML, TXT, MD files"),
            ("Mixed Media", "Multiple file types in single requests"),
            ("Streaming", "Real-time responses with media attachments"),
            ("Error Handling", "Invalid files, size limits, format validation"),
            ("Format Compatibility", "OpenAI vs AbstractCore syntax equivalence")
        ]

        for area, description in test_areas:
            report.append(f"   ✅ {area}: {description}")
        report.append("")

        # Supported formats summary
        report.append("📁 Supported File Formats:")
        formats = [
            ("Images", "PNG, JPEG, GIF, WEBP, BMP, TIFF"),
            ("Documents", "PDF, DOCX, XLSX, PPTX"),
            ("Data", "CSV, TSV, JSON, XML"),
            ("Text", "TXT, MD")
        ]

        for category, types in formats:
            report.append(f"   {category}: {types}")
        report.append("")

        # Usage examples
        report.append("💡 Usage Examples:")
        report.append("")
        report.append("   OpenAI Vision API Format:")
        report.append('   {')
        report.append('     "model": "ollama/qwen2.5vl:7b",')
        report.append('     "messages": [{')
        report.append('       "role": "user",')
        report.append('       "content": [')
        report.append('         {"type": "text", "text": "Analyze this image"},')
        report.append('         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}')
        report.append('       ]')
        report.append('     }]')
        report.append('   }')
        report.append("")
        report.append("   AbstractCore @filename Format:")
        report.append('   {')
        report.append('     "model": "lmstudio/qwen3-next-80b",')
        report.append('     "messages": [{')
        report.append('       "role": "user",')
        report.append('       "content": "Summarize @report.pdf and @data.csv"')
        report.append('     }]')
        report.append('   }')
        report.append("")

        # Next steps
        if test_results["success"]:
            report.append("🎉 All Tests Passed!")
            report.append("")
            report.append("✅ The server media integration is working correctly across all modalities.")
            report.append("✅ OpenAI client compatibility is confirmed.")
            report.append("✅ AbstractCore's universal media system is accessible via standard endpoints.")
            report.append("")
            report.append("📋 Next Steps:")
            report.append("   1. Deploy to production environment")
            report.append("   2. Update client documentation with media examples")
            report.append("   3. Monitor performance metrics in production")
            report.append("   4. Consider rate limiting for large file uploads")
        else:
            report.append("⚠️  Some Tests Failed")
            report.append("")
            report.append("🔍 Recommended Actions:")
            report.append("   1. Check server logs for detailed error information")
            report.append("   2. Verify all required models are available")
            report.append("   3. Ensure sufficient disk space for temporary files")
            report.append("   4. Review specific test failures above")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Run AbstractCore server media tests")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation only")
    parser.add_argument("--provider", choices=["ollama", "lmstudio"],
                       help="Test specific provider only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--server", default="http://localhost:8000",
                       help="Server URL (default: http://localhost:8000)")

    args = parser.parse_args()

    runner = ServerMediaTestRunner(args.server)

    print("🚀 AbstractCore Server Media Tests")
    print("=" * 50)

    # Check server health
    if not runner.check_server_health():
        print("\n❌ Server is not running or unhealthy.")
        print("💡 Start the server with: uvicorn abstractcore.server.app:app --port 8000")
        return 1

    # Check dependencies
    if not runner.validate_dependencies():
        return 1

    # Check available models
    available_models = runner.check_available_models()

    # Check if any models are available
    total_models = sum(len(models) for models in available_models.values())
    if total_models == 0:
        print("\n⚠️  No models available for testing.")
        print("💡 Make sure Ollama and/or LMStudio are running with models loaded.")
        print("💡 Ollama: ollama pull qwen2.5vl:7b")
        print("💡 LMStudio: Load a vision-capable model")
        if not args.quick:
            return 1

    # Run validation
    if not runner.run_quick_validation():
        return 1

    # Run full tests unless --quick specified
    if args.quick:
        print("\n✅ Quick validation completed successfully!")
        print("💡 Run full tests with: python tests/server/run_media_tests.py")
        return 0

    # Run comprehensive tests
    test_results = runner.run_pytest_suite(args.provider, args.verbose)

    # Generate and display report
    report = runner.generate_report(test_results, available_models)
    print(report)

    # Save report to file
    report_file = "media_tests_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"📄 Full report saved to: {report_file}")

    return 0 if test_results["success"] else 1

if __name__ == "__main__":
    sys.exit(main())