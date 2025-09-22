"""
Progressive complexity testing for structured output across all providers.

Tests three levels of complexity:
1. Flat JSON (1 level) - simple fields only
2. Nested JSON (2 levels) - objects with nested properties
3. Deep Nested JSON (4 levels) - complex hierarchical structures

Providers tested:
- Ollama: qwen3-coder:30b
- LMStudio: qwen/qwen3-coder-30b
- Anthropic: claude-3-5-haiku-latest
- OpenAI: gpt-4o-mini
- MLX: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict
from enum import Enum
import json
from abstractllm import create_llm


# Level 1: Flat JSON (1 level)
class FlatUser(BaseModel):
    """Simple flat structure with basic types."""
    name: str
    age: int
    email: str
    active: bool
    score: float


# Level 2: Nested JSON (2 levels)
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class NestedUser(BaseModel):
    """Structure with one level of nesting."""
    name: str
    age: int
    email: str
    address: Address
    theme: str  # Simplified from preferences dict


# Level 3: Deep Nested JSON (4 levels)
class ContactInfo(BaseModel):
    email: str
    phone: str

class Location(BaseModel):
    latitude: float
    longitude: float
    timezone: str

class DetailedAddress(BaseModel):
    street: str
    city: str
    state: str
    country: str
    postal_code: str
    location: Location

class Profile(BaseModel):
    bio: str
    skills: List[str]
    contact: ContactInfo

class Department(BaseModel):
    name: str
    code: str
    budget: float

class Company(BaseModel):
    name: str
    department: Department
    address: DetailedAddress

class DeepNestedUser(BaseModel):
    """Complex structure with 4 levels of nesting."""
    name: str
    age: int
    profile: Profile
    company: Company
    clearance_level: str  # Simplified from complex metadata


# Test data for each complexity level
TEST_DATA = {
    "flat": {
        "input": "User: John Doe, 28 years old, email john@example.com, active user, score 85.5",
        "expected_fields": ["name", "age", "email", "active", "score"],
        "model": FlatUser
    },
    "nested": {
        "input": """
        User: Sarah Johnson, age 34, email sarah@company.com
        Address: 123 Main St, Boston, MA, USA, 02101
        Theme: dark
        """,
        "expected_fields": ["name", "age", "email", "address", "theme"],
        "model": NestedUser
    },
    "deep": {
        "input": """
        Employee: Dr. Alice Chen, 42 years old
        Profile: Senior Research Scientist with expertise in AI, ML, NLP. Contact: alice.chen@techcorp.com, +1-555-0123
        Company: TechCorp Research Division (R&D), budget $2.5M
        Address: 456 Innovation Drive, San Francisco, CA 94105, USA
        Location: 37.7749 latitude, -122.4194 longitude, Pacific Time
        Clearance: high
        """,
        "expected_fields": ["name", "age", "profile", "company", "clearance_level"],
        "model": DeepNestedUser
    }
}

# Provider configurations
PROVIDERS = {
    "ollama": {
        "model": "qwen3-coder:30b",
        "strategy": "native_json"
    },
    "lmstudio": {
        "model": "qwen/qwen3-coder-30b",
        "strategy": "prompted"
    },
    "anthropic": {
        "model": "claude-3-5-haiku-latest",
        "strategy": "tool_trick"
    },
    "openai": {
        "model": "gpt-4o-mini",
        "strategy": "native_strict"
    },
    "mlx": {
        "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "strategy": "prompted"
    }
}


def test_provider_complexity(provider_name: str, provider_config: dict, complexity: str, test_data: dict):
    """Test a specific provider with a specific complexity level."""

    print(f"\n--- {provider_name.upper()} | {complexity.upper()} ---")

    try:
        # Create LLM instance
        llm = create_llm(provider_name, model=provider_config["model"])

        # Generate structured output
        result = llm.generate(
            f"Extract structured data from: {test_data['input']}",
            response_model=test_data["model"]
        )

        # Validate result
        assert isinstance(result, test_data["model"]), f"Expected {test_data['model'].__name__}, got {type(result)}"

        # Check all expected fields are present
        for field in test_data["expected_fields"]:
            assert hasattr(result, field), f"Missing field: {field}"
            value = getattr(result, field)
            assert value is not None, f"Field {field} is None"

        print(f"✅ SUCCESS")
        print(f"   Strategy: {provider_config['strategy']}")
        print(f"   Model: {provider_config['model']}")
        print(f"   Type: {type(result).__name__}")

        # Print key extracted data
        if complexity == "flat":
            print(f"   Data: {result.name}, age {result.age}, score {result.score}")
        elif complexity == "nested":
            print(f"   Data: {result.name} from {result.address.city}, {result.address.country}, theme: {result.theme}")
        elif complexity == "deep":
            print(f"   Data: {result.name}, {result.company.name} ({result.company.department.name})")
            print(f"   Location: {result.company.address.location.latitude}, {result.company.address.location.longitude}")
            print(f"   Clearance: {result.clearance_level}")

        return True, None

    except Exception as e:
        print(f"❌ FAILED: {str(e)[:100]}...")
        return False, str(e)


def run_progressive_tests():
    """Run all tests across providers and complexity levels."""

    print("="*80)
    print("PROGRESSIVE STRUCTURED OUTPUT TESTING")
    print("="*80)

    results = {}

    # Test each complexity level
    for complexity in ["flat", "nested", "deep"]:
        print(f"\n{'='*20} COMPLEXITY: {complexity.upper()} {'='*20}")

        test_data = TEST_DATA[complexity]
        results[complexity] = {}

        # Test each provider
        for provider_name, provider_config in PROVIDERS.items():
            success, error = test_provider_complexity(
                provider_name, provider_config, complexity, test_data
            )
            results[complexity][provider_name] = {
                "success": success,
                "error": error,
                "strategy": provider_config["strategy"]
            }

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)

    for complexity in ["flat", "nested", "deep"]:
        print(f"\n{complexity.upper()} JSON:")
        for provider in PROVIDERS.keys():
            result = results[complexity][provider]
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            strategy = result["strategy"]
            print(f"  {provider:<12} | {strategy:<15} | {status}")

    # Calculate success rates
    print(f"\nOVERALL SUCCESS RATES:")
    for provider in PROVIDERS.keys():
        successes = sum(1 for complexity in results.values() if complexity[provider]["success"])
        total = len(results)
        rate = (successes / total) * 100
        print(f"  {provider:<12} | {successes}/{total} | {rate:.1f}%")

    return results


def test_specific_examples():
    """Test specific challenging examples."""

    print("\n" + "="*80)
    print("SPECIFIC CHALLENGING EXAMPLES")
    print("="*80)

    # Test with enum validation
    class TaskPriority(str, Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    class ValidatedTask(BaseModel):
        title: str
        priority: TaskPriority
        hours: float

        @field_validator('hours')
        @classmethod
        def validate_hours(cls, v):
            if v <= 0:
                raise ValueError('Hours must be positive')
            return v

    # Test validation retry behavior
    print("\n--- VALIDATION RETRY TEST ---")

    for provider_name in ["ollama", "openai"]:
        try:
            print(f"\nTesting {provider_name.upper()}...")
            llm = create_llm(provider_name, model=PROVIDERS[provider_name]["model"])

            result = llm.generate(
                "Task: Finish the quarterly report - this is very urgent and will take 8 hours",
                response_model=ValidatedTask
            )

            print(f"✅ Success: {result.title}, priority={result.priority}, hours={result.hours}")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run main progressive tests
    results = run_progressive_tests()

    # Run specific challenging examples
    test_specific_examples()

    print("\n" + "="*80)
    print("TESTING COMPLETED!")
    print("="*80)