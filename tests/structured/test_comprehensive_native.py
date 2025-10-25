"""
Comprehensive Native Structured Output Testing Suite

Tests both Ollama and LMStudio providers with:
- Multiple models (small, medium, large)
- Three complexity levels (simple, medium, complex)
- Retry strategy analysis
- Success rate tracking
- Performance measurement

This test suite will help determine:
1. Are native structured outputs truly "guaranteed"?
2. When do we need retry strategies?
3. Which models work best for structured outputs?
4. What are the practical limitations?
"""

import json
import time
import pytest
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass
from datetime import datetime

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
    CANCELLED = "cancelled"


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
    active_projects: List[Project]


class Organization(BaseModel):
    """Complex schema: deeply nested with multiple enums and arrays"""
    company_name: str
    founded_year: int
    departments: List[Department]
    teams: List[Team]
    total_employees: int


# ============================================================================
# TEST RESULT TRACKING
# ============================================================================

@dataclass
class TestResult:
    """Track results of a single test"""
    provider: str
    model: str
    schema_complexity: str  # simple, medium, complex
    attempt_number: int
    success: bool
    response_time_ms: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    validation_error: bool = False
    required_retry: bool = False


class TestResultTracker:
    """Track all test results for analysis"""

    def __init__(self):
        self.results: List[TestResult] = []

    def add_result(self, result: TestResult):
        self.results.append(result)

    def get_success_rate(self, provider: str = None, model: str = None, complexity: str = None) -> float:
        """Calculate success rate for given filters"""
        filtered = self.results
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if model:
            filtered = [r for r in filtered if r.model == model]
        if complexity:
            filtered = [r for r in filtered if r.schema_complexity == complexity]

        if not filtered:
            return 0.0

        successful = len([r for r in filtered if r.success])
        return (successful / len(filtered)) * 100

    def get_avg_response_time(self, provider: str = None, model: str = None) -> float:
        """Calculate average response time"""
        filtered = self.results
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if model:
            filtered = [r for r in filtered if r.model == model]

        if not filtered:
            return 0.0

        successful = [r for r in filtered if r.success]
        if not successful:
            return 0.0

        return sum(r.response_time_ms for r in successful) / len(successful)

    def get_retry_rate(self, provider: str = None, model: str = None, complexity: str = None) -> float:
        """Calculate retry necessity rate"""
        filtered = self.results
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if model:
            filtered = [r for r in filtered if r.model == model]
        if complexity:
            filtered = [r for r in filtered if r.schema_complexity == complexity]

        if not filtered:
            return 0.0

        required_retry = len([r for r in filtered if r.required_retry])
        return (required_retry / len(filtered)) * 100

    def save_to_json(self, filepath: str):
        """Save results to JSON file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "results": [
                {
                    "provider": r.provider,
                    "model": r.model,
                    "schema_complexity": r.schema_complexity,
                    "attempt_number": r.attempt_number,
                    "success": r.success,
                    "response_time_ms": r.response_time_ms,
                    "error_type": r.error_type,
                    "error_message": r.error_message,
                    "validation_error": r.validation_error,
                    "required_retry": r.required_retry
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# Models to test for each provider
OLLAMA_MODELS = [
    "qwen3:4b-instruct-2507-q4_K_M",  # Small model (~4B params)
    "gpt-oss:20b",                     # Medium model (~20B params)
]

LMSTUDIO_MODELS = [
    "qwen/qwen3-4b-2507",             # Small model
    "openai/gpt-oss-20b",             # Medium model
]

# Test prompts for each complexity level
SIMPLE_PROMPTS = [
    "Generate a person: John Doe, 35 years old, john@example.com",
    "Create a person: Alice Smith, 28, alice.smith@company.com",
]

MEDIUM_PROMPTS = [
    """Create a project 'Website Redesign' with 2 tasks:
    1. 'Design mockups' - high priority, pending, 8 hours, tags: design, ui
    2. 'Implement frontend' - medium priority, in progress, 16 hours, tags: development, react
    Total: 24 hours""",

    """Create a project 'Mobile App' with 2 tasks:
    1. 'API Integration' - critical priority, in progress, 12 hours, tags: backend, api
    2. 'Testing' - high priority, pending, 6 hours, tags: qa, testing
    Total: 18 hours""",
]

COMPLEX_PROMPTS = [
    """Create an organization 'TechCorp' founded in 2020 with:
    - Departments: engineering, marketing
    - 1 Engineering team 'Platform' with:
      - Lead: Sarah (sarah@tech.com, senior engineer, skills: Python-9-5yrs, AWS-8-4yrs)
      - Member: Bob (bob@tech.com, mid engineer, skills: JavaScript-7-3yrs)
    - 1 Marketing team 'Growth' with:
      - Lead: Alice (alice@tech.com, staff marketing, skills: SEO-9-6yrs)
      - Member: Charlie (charlie@tech.com, junior marketing, skills: Content-6-2yrs)
    - Total employees: 4""",
]


# ============================================================================
# TEST CLASS
# ============================================================================

class TestComprehensiveNativeStructured:
    """Comprehensive testing of native structured outputs"""

    tracker = TestResultTracker()

    def _run_single_test(
        self,
        provider_name: str,
        model: str,
        schema_class: type[BaseModel],
        prompt: str,
        complexity: str,
        max_retries: int = 3
    ) -> None:
        """Run a single test with retry tracking"""

        try:
            # Create LLM instance
            llm = create_llm(provider_name, model=model)
            handler = StructuredOutputHandler()

            for attempt in range(1, max_retries + 1):
                start_time = time.time()
                error_occurred = None
                success = False

                try:
                    result = handler.generate_structured(
                        provider=llm,
                        prompt=prompt,
                        response_model=schema_class,
                        temperature=0  # Deterministic
                    )

                    response_time = (time.time() - start_time) * 1000
                    success = True

                    # Verify it's the correct type
                    assert isinstance(result, schema_class)

                    # Record successful result
                    self.tracker.add_result(TestResult(
                        provider=provider_name,
                        model=model,
                        schema_complexity=complexity,
                        attempt_number=attempt,
                        success=True,
                        response_time_ms=response_time,
                        required_retry=(attempt > 1)
                    ))

                    print(f"✅ {provider_name}/{model}/{complexity} - Success on attempt {attempt} ({response_time:.0f}ms)")
                    return  # Success, exit retry loop

                except ValidationError as e:
                    response_time = (time.time() - start_time) * 1000
                    error_occurred = e

                    # Record validation error
                    self.tracker.add_result(TestResult(
                        provider=provider_name,
                        model=model,
                        schema_complexity=complexity,
                        attempt_number=attempt,
                        success=False,
                        response_time_ms=response_time,
                        error_type="ValidationError",
                        error_message=str(e),
                        validation_error=True,
                        required_retry=True
                    ))

                    if attempt < max_retries:
                        print(f"⚠️  {provider_name}/{model}/{complexity} - ValidationError on attempt {attempt}, retrying...")
                    else:
                        print(f"❌ {provider_name}/{model}/{complexity} - Failed after {max_retries} attempts")

                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    error_occurred = e

                    # Record other errors
                    self.tracker.add_result(TestResult(
                        provider=provider_name,
                        model=model,
                        schema_complexity=complexity,
                        attempt_number=attempt,
                        success=False,
                        response_time_ms=response_time,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        required_retry=True
                    ))

                    print(f"❌ {provider_name}/{model}/{complexity} - {type(e).__name__}: {str(e)[:100]}")
                    return  # Non-recoverable error, don't retry

        except Exception as e:
            # Provider initialization failed
            pytest.skip(f"Provider {provider_name} not available: {e}")

    # ========================================================================
    # OLLAMA TESTS
    # ========================================================================

    @pytest.mark.parametrize("model", OLLAMA_MODELS)
    def test_ollama_simple_schema(self, model):
        """Test Ollama with simple schema"""
        for prompt in SIMPLE_PROMPTS:
            self._run_single_test(
                provider_name="ollama",
                model=model,
                schema_class=SimplePersonInfo,
                prompt=prompt,
                complexity="simple"
            )

    @pytest.mark.parametrize("model", OLLAMA_MODELS)
    def test_ollama_medium_schema(self, model):
        """Test Ollama with medium complexity schema"""
        for prompt in MEDIUM_PROMPTS:
            self._run_single_test(
                provider_name="ollama",
                model=model,
                schema_class=Project,
                prompt=prompt,
                complexity="medium"
            )

    @pytest.mark.parametrize("model", OLLAMA_MODELS)
    def test_ollama_complex_schema(self, model):
        """Test Ollama with complex nested schema"""
        for prompt in COMPLEX_PROMPTS:
            self._run_single_test(
                provider_name="ollama",
                model=model,
                schema_class=Organization,
                prompt=prompt,
                complexity="complex"
            )

    # ========================================================================
    # LMSTUDIO TESTS
    # ========================================================================

    @pytest.mark.parametrize("model", LMSTUDIO_MODELS)
    def test_lmstudio_simple_schema(self, model):
        """Test LMStudio with simple schema"""
        for prompt in SIMPLE_PROMPTS:
            self._run_single_test(
                provider_name="lmstudio",
                model=model,
                schema_class=SimplePersonInfo,
                prompt=prompt,
                complexity="simple"
            )

    @pytest.mark.parametrize("model", LMSTUDIO_MODELS)
    def test_lmstudio_medium_schema(self, model):
        """Test LMStudio with medium complexity schema"""
        for prompt in MEDIUM_PROMPTS:
            self._run_single_test(
                provider_name="lmstudio",
                model=model,
                schema_class=Project,
                prompt=prompt,
                complexity="medium"
            )

    @pytest.mark.parametrize("model", LMSTUDIO_MODELS)
    def test_lmstudio_complex_schema(self, model):
        """Test LMStudio with complex nested schema"""
        for prompt in COMPLEX_PROMPTS:
            self._run_single_test(
                provider_name="lmstudio",
                model=model,
                schema_class=Organization,
                prompt=prompt,
                complexity="complex"
            )


# ============================================================================
# TEST RUNNER AND RESULTS
# ============================================================================

def run_all_tests_and_save_results():
    """Run all tests and save results to JSON"""
    print("\n" + "="*80)
    print("COMPREHENSIVE NATIVE STRUCTURED OUTPUT TESTING")
    print("="*80)

    test_instance = TestComprehensiveNativeStructured()

    # Run all tests
    print("\n📊 Testing Ollama Simple Schemas...")
    for model in OLLAMA_MODELS:
        for prompt in SIMPLE_PROMPTS:
            test_instance._run_single_test("ollama", model, SimplePersonInfo, prompt, "simple")

    print("\n📊 Testing Ollama Medium Schemas...")
    for model in OLLAMA_MODELS:
        for prompt in MEDIUM_PROMPTS:
            test_instance._run_single_test("ollama", model, Project, prompt, "medium")

    print("\n📊 Testing Ollama Complex Schemas...")
    for model in OLLAMA_MODELS:
        for prompt in COMPLEX_PROMPTS:
            test_instance._run_single_test("ollama", model, Organization, prompt, "complex")

    print("\n📊 Testing LMStudio Simple Schemas...")
    for model in LMSTUDIO_MODELS:
        for prompt in SIMPLE_PROMPTS:
            test_instance._run_single_test("lmstudio", model, SimplePersonInfo, prompt, "simple")

    print("\n📊 Testing LMStudio Medium Schemas...")
    for model in LMSTUDIO_MODELS:
        for prompt in MEDIUM_PROMPTS:
            test_instance._run_single_test("lmstudio", model, Project, prompt, "medium")

    print("\n📊 Testing LMStudio Complex Schemas...")
    for model in LMSTUDIO_MODELS:
        for prompt in COMPLEX_PROMPTS:
            test_instance._run_single_test("lmstudio", model, Organization, prompt, "complex")

    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)

    tracker = test_instance.tracker

    # Overall stats
    total_tests = len(tracker.results)
    overall_success_rate = tracker.get_success_rate()
    print(f"\nTotal Tests: {total_tests}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")

    # By provider
    print("\n--- By Provider ---")
    for provider in ["ollama", "lmstudio"]:
        success_rate = tracker.get_success_rate(provider=provider)
        avg_time = tracker.get_avg_response_time(provider=provider)
        retry_rate = tracker.get_retry_rate(provider=provider)
        print(f"{provider.capitalize()}: {success_rate:.1f}% success, {avg_time:.0f}ms avg, {retry_rate:.1f}% needed retries")

    # By complexity
    print("\n--- By Complexity ---")
    for complexity in ["simple", "medium", "complex"]:
        success_rate = tracker.get_success_rate(complexity=complexity)
        retry_rate = tracker.get_retry_rate(complexity=complexity)
        print(f"{complexity.capitalize()}: {success_rate:.1f}% success, {retry_rate:.1f}% needed retries")

    # By model
    print("\n--- By Model ---")
    all_models = OLLAMA_MODELS + LMSTUDIO_MODELS
    for model in all_models:
        success_rate = tracker.get_success_rate(model=model)
        avg_time = tracker.get_avg_response_time(model=model)
        print(f"{model}: {success_rate:.1f}% success, {avg_time:.0f}ms avg")

    # Save to JSON
    output_file = "test_results_native_structured.json"
    tracker.save_to_json(output_file)
    print(f"\n✅ Results saved to: {output_file}")

    print("\n" + "="*80)

    return tracker


if __name__ == "__main__":
    run_all_tests_and_save_results()
