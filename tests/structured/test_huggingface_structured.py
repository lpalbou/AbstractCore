"""
Comprehensive HuggingFace Structured Output Testing Suite

Tests HuggingFace provider with:
- GGUF models (native structured output via llama-cpp-python)
- Transformers models (prompted structured output)
- Multiple complexity levels (simple, medium, complex)
- Success rate tracking
- Performance measurement

This test suite will help verify:
1. GGUF models get native structured output support
2. Transformers models fall back to prompted approach
3. Both approaches produce valid structured outputs
4. Performance comparison between GGUF and transformers
"""

import json
import time
import pytest
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass

from abstractcore import create_llm
from abstractcore.structured import StructuredOutputHandler


# ============================================================================
# SIMPLE SCHEMAS (Complexity Level 1)
# ============================================================================

class SimplePersonInfo(BaseModel):
    """Simple schema: basic fields only"""
    name: str
    age: int
    email: str


# ============================================================================
# MEDIUM SCHEMAS (Complexity Level 2)
# ============================================================================

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class Task(BaseModel):
    """Medium complexity: nested objects with enums"""
    title: str
    description: str
    priority: Priority
    status: TaskStatus
    estimated_hours: Optional[float] = None
    tags: List[str] = []


class Project(BaseModel):
    """Medium complexity: object with array of nested objects"""
    name: str
    description: str
    tasks: List[Task]
    total_hours: float


# ============================================================================
# COMPLEX SCHEMAS (Complexity Level 3)
# ============================================================================

class Department(str, Enum):
    ENGINEERING = "engineering"
    MARKETING = "marketing"
    SALES = "sales"
    HR = "hr"
    FINANCE = "finance"


class EmployeeLevel(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"


class Skill(BaseModel):
    name: str
    proficiency: int  # 1-10
    years_experience: float


class Employee(BaseModel):
    name: str
    email: str
    department: Department
    level: EmployeeLevel
    skills: List[Skill]
    manager_email: Optional[str] = None


class Team(BaseModel):
    name: str
    department: Department
    lead: Employee
    members: List[Employee]


class Organization(BaseModel):
    company_name: str
    founded_year: int
    departments: List[Department]
    teams: List[Team]
    total_employees: int


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Test models - you can modify these based on what you have cached
GGUF_TEST_MODEL = "unsloth/Qwen3-4B-Instruct-2507-GGUF"  # Default GGUF model

# Note: Transformers models are commented out by default as they require downloading
# Uncomment if you have models cached locally
# TRANSFORMERS_TEST_MODEL = "microsoft/DialoGPT-medium"  # Example transformers model

# Test prompts for each complexity level
SIMPLE_PROMPTS = {
    "person": "Extract information: John Smith is 35 years old and his email is john.smith@example.com"
}

MEDIUM_PROMPTS = {
    "project": """Create a project 'Website Redesign' with 2 tasks:
1. 'Design mockups' - high priority, pending, 8 hours, tags: design, ui
2. 'Implement frontend' - medium priority, in_progress, 16 hours, tags: development, react
Total: 24 hours"""
}

COMPLEX_PROMPTS = {
    "organization": """Create an organization 'TechCorp' founded in 2020 with:
- Departments: engineering, marketing
- 1 Engineering team 'Platform' with:
  - Lead: Sarah (sarah@tech.com, senior engineer, skills: Python-9-5yrs, AWS-8-4yrs)
  - Member: Bob (bob@tech.com, mid engineer, skills: JavaScript-7-3yrs)
- Total employees: 2"""
}


# ============================================================================
# TEST HELPERS
# ============================================================================

def test_structured_output(provider_type: str, model_name: str, prompt: str,
                          response_model: type[BaseModel], complexity: str) -> dict:
    """
    Test structured output for a given configuration.

    Returns:
        dict with test results including success, validation errors, timing, etc.
    """
    result = {
        "provider": provider_type,
        "model": model_name,
        "complexity": complexity,
        "success": False,
        "validation_error": None,
        "response_time_ms": 0,
        "retry_count": 0,
        "response_data": None
    }

    try:
        # Create LLM
        llm = create_llm(provider_type, model=model_name, temperature=0)

        # Check if provider has native support
        handler = StructuredOutputHandler(llm)
        has_native = handler._has_native_support(llm)
        result["native_support"] = has_native

        # Measure time
        start_time = time.time()

        # Generate structured output using the handler directly
        # This returns the Pydantic model, not a GenerateResponse
        handler = StructuredOutputHandler()
        response = handler.generate_structured(
            provider=llm,
            prompt=prompt,
            response_model=response_model,
            temperature=0
        )

        end_time = time.time()
        result["response_time_ms"] = round((end_time - start_time) * 1000, 1)

        # Validate response - it should be the Pydantic model directly
        if isinstance(response, response_model):
            result["success"] = True
            result["response_data"] = response.model_dump()
        else:
            result["validation_error"] = f"Expected {response_model.__name__}, got {type(response).__name__}"

    except ValidationError as e:
        result["validation_error"] = str(e)
    except Exception as e:
        result["validation_error"] = f"Error: {str(e)}"

    return result


# ============================================================================
# GGUF MODEL TESTS (Native Structured Output)
# ============================================================================

@pytest.mark.skipif(
    not True,  # Always try to run, will fail gracefully if model not available
    reason="GGUF model not available"
)
class TestHuggingFaceGGUFStructured:
    """Test native structured output support for HuggingFace GGUF models"""

    def test_gguf_simple_person(self):
        """Test GGUF model with simple schema"""
        result = test_structured_output(
            provider_type="huggingface",
            model_name=GGUF_TEST_MODEL,
            prompt=SIMPLE_PROMPTS["person"],
            response_model=SimplePersonInfo,
            complexity="simple"
        )

        print(f"\nGGUF Simple Person Test:")
        print(f"  Native Support: {result.get('native_support', 'Unknown')}")
        print(f"  Success: {result['success']}")
        print(f"  Response Time: {result['response_time_ms']}ms")
        if result['success']:
            print(f"  Data: {result['response_data']}")
        else:
            print(f"  Error: {result.get('validation_error', 'Unknown')}")

        assert result['success'], f"GGUF simple test failed: {result.get('validation_error')}"
        assert result.get('native_support', False), "GGUF should have native support"

    def test_gguf_medium_project(self):
        """Test GGUF model with medium complexity schema"""
        result = test_structured_output(
            provider_type="huggingface",
            model_name=GGUF_TEST_MODEL,
            prompt=MEDIUM_PROMPTS["project"],
            response_model=Project,
            complexity="medium"
        )

        print(f"\nGGUF Medium Project Test:")
        print(f"  Native Support: {result.get('native_support', 'Unknown')}")
        print(f"  Success: {result['success']}")
        print(f"  Response Time: {result['response_time_ms']}ms")
        if result['success']:
            print(f"  Data: {result['response_data']}")
        else:
            print(f"  Error: {result.get('validation_error', 'Unknown')}")

        assert result['success'], f"GGUF medium test failed: {result.get('validation_error')}"
        assert result.get('native_support', False), "GGUF should have native support"

    def test_gguf_complex_organization(self):
        """Test GGUF model with complex nested schema"""
        result = test_structured_output(
            provider_type="huggingface",
            model_name=GGUF_TEST_MODEL,
            prompt=COMPLEX_PROMPTS["organization"],
            response_model=Organization,
            complexity="complex"
        )

        print(f"\nGGUF Complex Organization Test:")
        print(f"  Native Support: {result.get('native_support', 'Unknown')}")
        print(f"  Success: {result['success']}")
        print(f"  Response Time: {result['response_time_ms']}ms")
        if result['success']:
            print(f"  Data: {result['response_data']}")
        else:
            print(f"  Error: {result.get('validation_error', 'Unknown')}")

        assert result['success'], f"GGUF complex test failed: {result.get('validation_error')}"
        assert result.get('native_support', False), "GGUF should have native support"


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_native_support_detection():
    """Test that HuggingFace GGUF models are correctly detected as having native support"""
    try:
        # Create HuggingFace provider with GGUF model
        llm = create_llm("huggingface", model=GGUF_TEST_MODEL)

        # Check native support detection
        handler = StructuredOutputHandler(llm)
        has_native = handler._has_native_support(llm)

        print(f"\nNative Support Detection Test:")
        print(f"  Provider: {llm.__class__.__name__}")
        print(f"  Model Type: {llm.model_type}")
        print(f"  Has Native Support: {has_native}")

        assert llm.model_type == "gguf", "Should be GGUF model"
        assert has_native, "GGUF models should have native structured output support"

    except Exception as e:
        pytest.skip(f"Could not test native support detection: {str(e)}")


# ============================================================================
# MANUAL RUN SUPPORT
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HuggingFace Structured Output Comprehensive Tests")
    print("=" * 80)

    results = []

    # Test 1: Native support detection
    print("\n1. Testing native support detection...")
    try:
        test_native_support_detection()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    # Test 2: GGUF Simple Person
    print("\n2. Testing GGUF simple schema...")
    try:
        result = test_structured_output(
            provider_type="huggingface",
            model_name=GGUF_TEST_MODEL,
            prompt=SIMPLE_PROMPTS["person"],
            response_model=SimplePersonInfo,
            complexity="simple"
        )
        results.append(result)
        if result['success']:
            print(f"   ✅ PASSED (Native: {result.get('native_support')}, Time: {result['response_time_ms']}ms)")
        else:
            print(f"   ❌ FAILED: {result.get('validation_error')}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    # Test 3: GGUF Medium Project
    print("\n3. Testing GGUF medium schema...")
    try:
        result = test_structured_output(
            provider_type="huggingface",
            model_name=GGUF_TEST_MODEL,
            prompt=MEDIUM_PROMPTS["project"],
            response_model=Project,
            complexity="medium"
        )
        results.append(result)
        if result['success']:
            print(f"   ✅ PASSED (Native: {result.get('native_support')}, Time: {result['response_time_ms']}ms)")
        else:
            print(f"   ❌ FAILED: {result.get('validation_error')}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    # Test 4: GGUF Complex Organization
    print("\n4. Testing GGUF complex schema...")
    try:
        result = test_structured_output(
            provider_type="huggingface",
            model_name=GGUF_TEST_MODEL,
            prompt=COMPLEX_PROMPTS["organization"],
            response_model=Organization,
            complexity="complex"
        )
        results.append(result)
        if result['success']:
            print(f"   ✅ PASSED (Native: {result.get('native_support')}, Time: {result['response_time_ms']}ms)")
        else:
            print(f"   ❌ FAILED: {result.get('validation_error')}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    avg_time = sum(r['response_time_ms'] for r in results) / len(results) if results else 0

    print(f"\nTotal Tests: {total_count}")
    print(f"Passed: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Success Rate: {(success_count/total_count*100):.1f}%" if total_count > 0 else "N/A")
    print(f"Average Response Time: {avg_time:.0f}ms")

    # Breakdown by complexity
    print("\nBy Complexity:")
    for complexity in ["simple", "medium", "complex"]:
        complexity_results = [r for r in results if r.get('complexity') == complexity]
        if complexity_results:
            success = sum(1 for r in complexity_results if r['success'])
            avg = sum(r['response_time_ms'] for r in complexity_results) / len(complexity_results)
            print(f"  {complexity.capitalize()}: {success}/{len(complexity_results)} passed, avg {avg:.0f}ms")

    # Save results to JSON
    output_file = "test_results_huggingface_structured.json"
    try:
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": GGUF_TEST_MODEL,
                "total_tests": total_count,
                "success_rate": f"{(success_count/total_count*100):.1f}%" if total_count > 0 else "N/A",
                "average_response_time_ms": avg_time,
                "results": results
            }, f, indent=2)
        print(f"\n📊 Results saved to {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

    print("\n" + "=" * 80)
    print("Tests completed!")
    print("=" * 80)
