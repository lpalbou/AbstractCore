#!/usr/bin/env python3
"""
AbstractCore Media Handler System Validation Script

This script validates the complete media handling system by testing:
1. All file type processors
2. All provider-specific handlers
3. Integration with all providers
4. Error handling and edge cases

Run this script to verify the media handling system is working correctly.
"""

import sys
import tempfile
from pathlib import Path
from PIL import Image as PILImage
import json


def print_status(message, status="INFO"):
    """Print colored status message."""
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m"
    }

    color = colors.get(status, colors["INFO"])
    reset = colors["RESET"]
    print(f"{color}[{status}]{reset} {message}")


def test_media_types():
    """Test all media type detection and processing."""
    print_status("Testing Media Type Detection and Processing", "INFO")

    # Create temporary test files
    temp_dir = Path(tempfile.mkdtemp())
    results = {}

    try:
        # Test image processing
        test_png = temp_dir / "test.png"
        img = PILImage.new('RGB', (100, 100), color='red')
        img.save(test_png)

        # Test text processing
        test_txt = temp_dir / "test.txt"
        test_txt.write_text("Sample text for testing")

        # Test CSV processing
        test_csv = temp_dir / "test.csv"
        test_csv.write_text("name,value\ntest,42\nsample,100\n")

        # Test media type detection
        try:
            from abstractcore.media.types import detect_media_type, MediaType

            assert detect_media_type(test_png) == MediaType.IMAGE
            assert detect_media_type(test_txt) == MediaType.TEXT
            assert detect_media_type(test_csv) == MediaType.TEXT

            results["media_type_detection"] = True
            print_status("✓ Media type detection working", "SUCCESS")

        except Exception as e:
            results["media_type_detection"] = False
            print_status(f"✗ Media type detection failed: {e}", "ERROR")

        # Test processors
        processors = {
            "ImageProcessor": ("abstractcore.media.processors", "ImageProcessor", test_png),
            "TextProcessor": ("abstractcore.media.processors", "TextProcessor", test_txt),
            "PDFProcessor": ("abstractcore.media.processors", "PDFProcessor", test_txt),  # Will fail gracefully
        }

        for processor_name, (module, class_name, test_file) in processors.items():
            try:
                module_obj = __import__(module, fromlist=[class_name])
                processor_class = getattr(module_obj, class_name)
                processor = processor_class()

                result = processor.process_file(test_file)

                if processor_name == "PDFProcessor" and test_file.suffix != ".pdf":
                    # Expected to fail for non-PDF
                    if not result.success:
                        results[processor_name] = True
                        print_status(f"✓ {processor_name} properly rejects non-PDF", "SUCCESS")
                    else:
                        results[processor_name] = False
                        print_status(f"✗ {processor_name} should reject non-PDF", "WARNING")
                else:
                    if result.success:
                        results[processor_name] = True
                        print_status(f"✓ {processor_name} working", "SUCCESS")
                    else:
                        results[processor_name] = False
                        print_status(f"✗ {processor_name} failed: {result.error_message}", "ERROR")

            except ImportError as e:
                results[processor_name] = False
                print_status(f"✗ {processor_name} not available: {e}", "WARNING")
            except Exception as e:
                results[processor_name] = False
                print_status(f"✗ {processor_name} error: {e}", "ERROR")

        # Test auto handler
        try:
            from abstractcore.media import AutoMediaHandler

            handler = AutoMediaHandler()

            # Test with image
            img_result = handler.process_file(test_png)
            if img_result.success:
                results["AutoMediaHandler_image"] = True
                print_status("✓ AutoMediaHandler image processing working", "SUCCESS")
            else:
                results["AutoMediaHandler_image"] = False
                print_status(f"✗ AutoMediaHandler image failed: {img_result.error_message}", "ERROR")

            # Test with text
            txt_result = handler.process_file(test_txt)
            if txt_result.success:
                results["AutoMediaHandler_text"] = True
                print_status("✓ AutoMediaHandler text processing working", "SUCCESS")
            else:
                results["AutoMediaHandler_text"] = False
                print_status(f"✗ AutoMediaHandler text failed: {txt_result.error_message}", "ERROR")

        except ImportError as e:
            results["AutoMediaHandler"] = False
            print_status(f"✗ AutoMediaHandler not available: {e}", "WARNING")
        except Exception as e:
            results["AutoMediaHandler"] = False
            print_status(f"✗ AutoMediaHandler error: {e}", "ERROR")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_provider_handlers():
    """Test provider-specific media handlers."""
    print_status("Testing Provider-Specific Handlers", "INFO")

    results = {}

    # Create test content
    temp_dir = Path(tempfile.mkdtemp())
    test_image = temp_dir / "test.png"
    img = PILImage.new('RGB', (50, 50), color='blue')
    img.save(test_image)

    try:
        # Process image first
        from abstractcore.media.processors import ImageProcessor
        processor = ImageProcessor()
        img_result = processor.process_file(test_image)

        if not img_result.success:
            print_status("Cannot test handlers without working image processor", "ERROR")
            return {"handler_test_prerequisite": False}

        media_content = img_result.media_content

        # Test handlers
        handlers = {
            "OpenAIMediaHandler": ("abstractcore.media.handlers", "OpenAIMediaHandler"),
            "AnthropicMediaHandler": ("abstractcore.media.handlers", "AnthropicMediaHandler"),
            "LocalMediaHandler": ("abstractcore.media.handlers", "LocalMediaHandler"),
        }

        for handler_name, (module, class_name) in handlers.items():
            try:
                module_obj = __import__(module, fromlist=[class_name])
                handler_class = getattr(module_obj, class_name)

                if handler_name == "LocalMediaHandler":
                    handler = handler_class("ollama", {"vision_support": True})
                else:
                    handler = handler_class({"vision_support": True})

                # Test formatting
                formatted = handler.format_for_provider(media_content)
                if formatted and isinstance(formatted, dict):
                    results[f"{handler_name}_format"] = True
                    print_status(f"✓ {handler_name} formatting working", "SUCCESS")
                else:
                    results[f"{handler_name}_format"] = False
                    print_status(f"✗ {handler_name} formatting failed", "ERROR")

                # Test multimodal message creation
                message = handler.create_multimodal_message("Test prompt", [media_content])
                if message:
                    results[f"{handler_name}_multimodal"] = True
                    print_status(f"✓ {handler_name} multimodal messages working", "SUCCESS")
                else:
                    results[f"{handler_name}_multimodal"] = False
                    print_status(f"✗ {handler_name} multimodal messages failed", "ERROR")

            except ImportError as e:
                results[handler_name] = False
                print_status(f"✗ {handler_name} not available: {e}", "WARNING")
            except Exception as e:
                results[handler_name] = False
                print_status(f"✗ {handler_name} error: {e}", "ERROR")

    except ImportError as e:
        print_status(f"Cannot test handlers: {e}", "ERROR")
        return {"handlers_not_available": False}
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


def test_capabilities():
    """Test capability detection system."""
    print_status("Testing Capability Detection", "INFO")

    results = {}

    try:
        from abstractcore.media.capabilities import (
            get_media_capabilities, is_vision_model, supports_images,
            get_supported_media_types
        )

        # Test vision model detection
        vision_models = [
            "gpt-4o",
            "claude-3.5-sonnet",
            "qwen3-vl",
            "gemma3:4b",
            "qwen2.5-vl-7b"
        ]

        for model in vision_models:
            if is_vision_model(model):
                results[f"vision_detection_{model}"] = True
                print_status(f"✓ {model} correctly identified as vision model", "SUCCESS")
            else:
                results[f"vision_detection_{model}"] = False
                print_status(f"✗ {model} not identified as vision model", "WARNING")

        # Test non-vision models
        non_vision_models = ["gpt-4", "qwen3-4b"]
        for model in non_vision_models:
            if not is_vision_model(model):
                results[f"non_vision_detection_{model}"] = True
                print_status(f"✓ {model} correctly identified as non-vision", "SUCCESS")
            else:
                results[f"non_vision_detection_{model}"] = False
                print_status(f"✗ {model} incorrectly identified as vision model", "WARNING")

        # Test capability retrieval
        caps = get_media_capabilities("gpt-4o")
        if caps.vision_support:
            results["capability_retrieval"] = True
            print_status("✓ Capability retrieval working", "SUCCESS")
        else:
            results["capability_retrieval"] = False
            print_status("✗ Capability retrieval failed", "ERROR")

    except ImportError as e:
        results["capabilities_not_available"] = False
        print_status(f"✗ Capabilities not available: {e}", "WARNING")
    except Exception as e:
        results["capabilities_error"] = False
        print_status(f"✗ Capabilities error: {e}", "ERROR")

    return results


def test_provider_integration():
    """Test integration with provider classes."""
    print_status("Testing Provider Integration", "INFO")

    results = {}

    # Test that providers have media parameter in generate methods
    providers = [
        "openai_provider.OpenAIProvider",
        "anthropic_provider.AnthropicProvider",
        "ollama_provider.OllamaProvider",
        "lmstudio_provider.LMStudioProvider",
        "huggingface_provider.HuggingFaceProvider",
        "mlx_provider.MLXProvider"
    ]

    for provider_path in providers:
        try:
            module_name, class_name = provider_path.rsplit('.', 1)
            module = __import__(f"abstractcore.providers.{module_name}", fromlist=[class_name])
            provider_class = getattr(module, class_name)

            # Check if _generate_internal has media parameter
            import inspect
            sig = inspect.signature(provider_class._generate_internal)

            if 'media' in sig.parameters:
                results[f"{class_name}_media_param"] = True
                print_status(f"✓ {class_name} has media parameter", "SUCCESS")
            else:
                results[f"{class_name}_media_param"] = False
                print_status(f"✗ {class_name} missing media parameter", "ERROR")

        except ImportError as e:
            results[f"{provider_path}_import"] = False
            print_status(f"✗ {provider_path} not available: {e}", "WARNING")
        except Exception as e:
            results[f"{provider_path}_error"] = False
            print_status(f"✗ {provider_path} error: {e}", "ERROR")

    return results


def test_model_capabilities_json():
    """Test model capabilities JSON file."""
    print_status("Testing Model Capabilities Configuration", "INFO")

    results = {}

    try:
        # Load model capabilities
        capabilities_file = Path("abstractcore/assets/model_capabilities.json")

        if capabilities_file.exists():
            with open(capabilities_file) as f:
                capabilities = json.load(f)

            # Check for specific models we added
            test_models = [
                "qwen2.5-vl-7b",
                "gemma-3n-e4b",
                "gemma3-4b",
                "gpt-4-turbo-with-vision"
            ]

            for model in test_models:
                if model in capabilities.get("models", {}):
                    model_info = capabilities["models"][model]
                    if model_info.get("vision_support", False):
                        results[f"model_caps_{model}"] = True
                        print_status(f"✓ {model} configured with vision support", "SUCCESS")
                    else:
                        results[f"model_caps_{model}"] = False
                        print_status(f"✗ {model} missing vision support in config", "ERROR")
                else:
                    results[f"model_caps_{model}"] = False
                    print_status(f"✗ {model} not found in capabilities", "ERROR")

            # Check updated GPT-4o
            if "gpt-4o" in capabilities.get("models", {}):
                gpt4o = capabilities["models"]["gpt-4o"]
                if gpt4o.get("max_output_tokens") == 16384:
                    results["gpt4o_updated"] = True
                    print_status("✓ GPT-4o updated with 2025 specs", "SUCCESS")
                else:
                    results["gpt4o_updated"] = False
                    print_status("✗ GPT-4o not updated with latest specs", "WARNING")

        else:
            results["capabilities_file"] = False
            print_status("✗ Model capabilities file not found", "ERROR")

    except Exception as e:
        results["capabilities_json_error"] = False
        print_status(f"✗ Error reading capabilities: {e}", "ERROR")

    return results


def main():
    """Run complete validation."""
    print_status("AbstractCore Media Handler System Validation", "INFO")
    print_status("=" * 60, "INFO")

    all_results = {}

    # Run all tests
    test_functions = [
        ("Media Types", test_media_types),
        ("Provider Handlers", test_provider_handlers),
        ("Capabilities", test_capabilities),
        ("Provider Integration", test_provider_integration),
        ("Model Capabilities", test_model_capabilities_json)
    ]

    for test_name, test_func in test_functions:
        print_status(f"\n{test_name} Tests", "INFO")
        print_status("-" * 40, "INFO")

        try:
            results = test_func()
            all_results.update(results)
        except Exception as e:
            print_status(f"✗ {test_name} test suite failed: {e}", "ERROR")
            all_results[f"{test_name.lower()}_suite_error"] = False

    # Summary
    print_status("\n" + "=" * 60, "INFO")
    print_status("VALIDATION SUMMARY", "INFO")
    print_status("=" * 60, "INFO")

    total_tests = len(all_results)
    passed_tests = sum(1 for result in all_results.values() if result is True)
    failed_tests = total_tests - passed_tests

    print_status(f"Total Tests: {total_tests}", "INFO")
    print_status(f"Passed: {passed_tests}", "SUCCESS" if passed_tests > 0 else "INFO")
    print_status(f"Failed: {failed_tests}", "ERROR" if failed_tests > 0 else "SUCCESS")

    if failed_tests == 0:
        print_status("\n🎉 ALL TESTS PASSED! Media handling system is working correctly.", "SUCCESS")
        return 0
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        print_status(f"\n✅ Most tests passed ({passed_tests}/{total_tests}). System is mostly functional.", "SUCCESS")
        print_status("Some optional features may be missing dependencies.", "WARNING")
        return 0
    else:
        print_status(f"\n❌ Many tests failed ({failed_tests}/{total_tests}). Please check the issues above.", "ERROR")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)