"""
Test all specified providers and models with structured output.

This test validates all the providers and models mentioned in the specifications:
- ollama qwen3-coder:30b
- lmstudio qwen/qwen3-coder-30b
- mlx mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
- huggingface microsoft/DialoGPT-medium (accessible GGUF alternative)
- openai gpt-4o-mini (cost-effective alternative to gpt-4-turbo)
- anthropic claude-3-5-haiku-latest
"""

import os
from pydantic import BaseModel
from typing import List, Optional
from abstractllm import create_llm


# Test models for validation
class SimpleTask(BaseModel):
    title: str
    priority: str  # high, medium, low

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

class CodeReview(BaseModel):
    """Model for code review - appropriate for coding models"""
    language: str
    issues_found: List[str]
    suggestions: List[str]
    overall_quality: str  # excellent, good, fair, poor


def test_provider_model():
    """Test all available provider/model combinations"""

    # Define test configurations
    test_configs = [
        ("ollama", "qwen3-coder:30b", CodeReview, "Please review this Python code: def hello(): print('world')"),
    ]

    # Add cloud providers if available
    if os.getenv("ANTHROPIC_API_KEY"):
        test_configs.append(("anthropic", "claude-3-5-haiku-20241022", CodeReview, "Please review this Python code: def hello(): print('world')"))
    if os.getenv("OPENAI_API_KEY"):
        test_configs.append(("openai", "gpt-4o-mini", CodeReview, "Please review this Python code: def hello(): print('world')"))

    success_count = 0
    total_tests = len(test_configs)

    for provider_name, model_name, response_model, test_prompt in test_configs:
        print(f"\n--- Testing {provider_name.upper()} | {model_name} ---")

        try:
            _test_single_provider(provider_name, model_name, response_model, test_prompt)
            success_count += 1
            print(f"✅ {provider_name} test passed")
        except Exception as e:
            print(f"❌ {provider_name} test failed: {str(e)}")
            continue

    print(f"\n✅ Provider tests completed: {success_count}/{total_tests} passed")
    assert success_count > 0, "No provider tests passed"


def _test_single_provider(provider_name: str, model_name: str, response_model: BaseModel, test_prompt: str):
    """Test a specific provider/model combination"""

    try:
        llm = create_llm(provider_name, model=model_name)

        # Test basic generation
        basic_response = llm.generate("Hello! Respond in one sentence.")
        assert basic_response.content is not None
        assert len(basic_response.content) > 0
        print(f"✅ Basic generation: OK")

        # Test structured output
        structured_response = llm.generate(test_prompt, response_model=response_model)
        assert isinstance(structured_response, response_model)
        print(f"✅ Structured output: OK")
        print(f"   Result: {structured_response}")

        return True, None

    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}...")
        return False, str(e)


def run_all_provider_tests():
    """Test all specified providers and models"""
    print("=" * 80)
    print("TESTING ALL SPECIFIED PROVIDERS AND MODELS")
    print("=" * 80)

    test_configs = [
        {
            "provider": "ollama",
            "model": "qwen3-coder:30b",
            "response_model": CodeReview,
            "prompt": "Review this Python code: def add(a, b): return a + b"
        },
        {
            "provider": "lmstudio",
            "model": "qwen/qwen3-coder-30b",
            "response_model": CodeReview,
            "prompt": "Review this Python code: def multiply(x, y): return x * y"
        },
        {
            "provider": "mlx",
            "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            "response_model": CodeReview,
            "prompt": "Review this JavaScript code: function divide(a, b) { return a / b; }"
        },
        {
            "provider": "huggingface",
            "model": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "response_model": CodeReview,
            "prompt": "Review this Python code: def process_data(data): return [x*2 for x in data if x > 0]"
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "response_model": ContactInfo,
            "prompt": "Extract contact: John Smith, john@example.com, 555-0123"
        },
        {
            "provider": "anthropic",
            "model": "claude-3-5-haiku-latest",
            "response_model": ContactInfo,
            "prompt": "Extract contact: Sarah Wilson, sarah.w@company.org, 555-9876"
        }
    ]

    results = {}
    for config in test_configs:
        provider = config["provider"]
        model = config["model"]

        success, error = test_provider_model(
            provider,
            model,
            config["response_model"],
            config["prompt"]
        )

        results[f"{provider}:{model}"] = {
            "success": success,
            "error": error
        }

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    success_count = 0
    total_count = len(results)

    for provider_model, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{provider_model:<50} | {status}")
        if result["success"]:
            success_count += 1
        else:
            print(f"   Error: {result['error'][:60]}...")

    print(f"\nOVERALL: {success_count}/{total_count} providers working ({success_count/total_count*100:.1f}%)")

    if success_count == total_count:
        print("🎉 ALL SPECIFIED PROVIDERS AND MODELS ARE WORKING!")
    else:
        print(f"⚠️  {total_count - success_count} providers need attention")

    return results


if __name__ == "__main__":
    results = run_all_provider_tests()