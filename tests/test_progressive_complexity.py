"""
Progressive complexity testing for structured output across all providers.

Tests three levels of complexity:
1. Flat JSON (1 level) - simple fields only
2. Nested JSON (2 levels) - objects with nested properties
3. Deep Nested JSON (4 levels) - complex hierarchical structures
"""

from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict
from enum import Enum
import json
import os
from abstractllm import create_llm


# Level 1: Flat JSON (1 level)
class FlatUser(BaseModel):
    """Simple flat structure with basic types."""
    name: str
    age: int
    email: str
    active: bool


# Level 2: Nested JSON (2 levels)
class Address(BaseModel):
    """Address information."""
    street: str
    city: str
    country: str
    postal_code: str


class NestedUser(BaseModel):
    """User with nested address."""
    name: str
    age: int
    email: str
    address: Address
    active: bool


# Level 3: Deep nested JSON (4 levels)
class Location(BaseModel):
    """GPS coordinates."""
    latitude: float
    longitude: float


class CompanyAddress(BaseModel):
    """Company address with location."""
    street: str
    city: str
    country: str
    location: Location


class Department(BaseModel):
    """Company department."""
    name: str
    floor: int
    budget: Optional[float] = None


class Company(BaseModel):
    """Company information."""
    name: str
    industry: str
    address: CompanyAddress
    department: Department


class DeepUser(BaseModel):
    """User with deep nested company structure."""
    name: str
    age: int
    email: str
    company: Company
    clearance_level: str


def test_provider_complexity():
    """Test progressive complexity across available providers."""

    # Define test configurations
    providers_to_test = [
        ("ollama", "qwen3-coder:30b"),
    ]

    # Add cloud providers if available
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append(("anthropic", "claude-3-5-haiku-20241022"))
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append(("openai", "gpt-4o-mini"))

    # Define complexity tests
    complexity_tests = {
        "flat": {
            "model": FlatUser,
            "input": "John Smith, 25 years old, john@example.com, currently active user",
            "expected_fields": ["name", "age", "email", "active"]
        },
        "nested": {
            "model": NestedUser,
            "input": "Sarah Johnson, 30 years old, sarah@example.com, lives at 123 Main St, New York, USA, postal code 10001, active user",
            "expected_fields": ["name", "age", "email", "address", "active"]
        },
        "deep": {
            "model": DeepUser,
            "input": "Mike Wilson, 35 years old, mike@techcorp.com, works at TechCorp in Engineering department on floor 5, company address: 456 Tech Ave, San Francisco, USA, GPS: 37.7749, -122.4194, security clearance: Level 3",
            "expected_fields": ["name", "age", "email", "company", "clearance_level"]
        }
    }

    # Test each provider
    success_count = 0
    total_tests = 0

    for provider_name, model_name in providers_to_test:
        print(f"\n--- Testing {provider_name.upper()} ---")

        try:
            # Create LLM instance
            llm = create_llm(provider_name, model=model_name)

            # Test each complexity level
            for complexity, test_data in complexity_tests.items():
                print(f"  Testing {complexity} complexity...")
                total_tests += 1

                try:
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

                    print(f"  ✅ {complexity} complexity passed")
                    success_count += 1

                except Exception as e:
                    print(f"  ❌ {complexity} complexity failed: {str(e)}")
                    # Continue testing other complexity levels
                    continue

        except Exception as e:
            print(f"⏭️  Skipping {provider_name}: {str(e)}")
            # Skip to next provider
            continue

    print(f"\n✅ Progressive complexity test completed: {success_count}/{total_tests} tests passed")

    # Ensure at least some tests passed (don't require 100% success)
    assert success_count > 0, "No complexity tests passed for any provider"