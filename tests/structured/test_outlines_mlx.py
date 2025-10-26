"""
Comprehensive test suite for MLX with Outlines native structured output.
Tests simple, medium, and complex schema complexity with real MLX models on Apple Silicon.
"""

import json
import time
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

from abstractcore import create_llm


# ============================================================================
# Schema Definitions (Simple, Medium, Complex) - Same as HuggingFace tests
# ============================================================================

class SimplePersonInfo(BaseModel):
    """Simple schema: 3 basic fields"""
    name: str
    age: int
    email: str


class Priority(str, Enum):
    """Enum for task priority"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """Medium complexity schema component"""
    title: str
    priority: Priority
    estimated_hours: float


class MediumProjectSchema(BaseModel):
    """Medium schema: nested objects with enums and arrays"""
    project_name: str
    description: str
    start_date: str
    tasks: List[Task]
    total_hours: float


class Department(str, Enum):
    """Enum for departments"""
    ENGINEERING = "engineering"
    SALES = "sales"
    MARKETING = "marketing"


class EmployeeLevel(str, Enum):
    """Enum for employee levels"""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"


class Skill(BaseModel):
    """Skill component"""
    name: str
    proficiency: int  # 1-10
    years_experience: float


class Employee(BaseModel):
    """Employee component"""
    name: str
    email: str
    department: Department
    level: EmployeeLevel
    skills: List[Skill]
    manager_email: Optional[str] = None


class Team(BaseModel):
    """Team component (level 2)"""
    name: str
    department: Department
    lead: Employee
    members: List[Employee]


class ComplexOrganizationSchema(BaseModel):
    """Complex schema: 3+ levels deep with multiple enums and nested arrays"""
    company_name: str
    founded_year: int
    teams: List[Team]
    total_employees: int


# ============================================================================
# Test Result Tracking
# ============================================================================

class TestResult(BaseModel):
    """Track results of each test"""
    provider: str
    model: str
    schema_name: str
    schema_complexity: str
    success: bool
    response_time_ms: float
    validation_error: Optional[str] = None
    used_outlines: bool = False


class TestResultTracker:
    """Track all test results"""
    def __init__(self):
        self.results: List[TestResult] = []

    def add_result(self, result: TestResult):
        self.results.append(result)

    def get_success_rate(self, provider: str = None, complexity: str = None) -> float:
        filtered = self.results
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if complexity:
            filtered = [r for r in filtered if r.schema_complexity == complexity]

        if not filtered:
            return 0.0

        successful = [r for r in filtered if r.success]
        return len(successful) / len(filtered) * 100

    def get_avg_response_time(self, provider: str = None, complexity: str = None) -> float:
        filtered = self.results
        if provider:
            filtered = [r for r in filtered if r.provider == provider]
        if complexity:
            filtered = [r for r in filtered if r.schema_complexity == complexity]

        successful = [r for r in filtered if r.success]
        if not successful:
            return 0.0

        return sum(r.response_time_ms for r in successful) / len(successful)

    def save_results(self, filename: str):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(
                [r.model_dump() for r in self.results],
                f,
                indent=2
            )


# ============================================================================
# Test Functions
# ============================================================================

def test_simple_schema(llm, tracker: TestResultTracker):
    """Test simple schema (3 fields)"""
    prompt = "Extract: Alice Smith, 28 years old, alice.smith@example.com"

    start_time = time.time()
    try:
        result = llm.generate(prompt, response_model=SimplePersonInfo, temperature=0)
        response_time_ms = (time.time() - start_time) * 1000

        # Verify it's the right type
        assert isinstance(result, SimplePersonInfo), f"Expected SimplePersonInfo, got {type(result)}"
        assert result.name, "Name should not be empty"
        assert result.age > 0, "Age should be positive"
        assert "@" in result.email, "Email should contain @"

        tracker.add_result(TestResult(
            provider="mlx",
            model=llm.model,
            schema_name="SimplePersonInfo",
            schema_complexity="simple",
            success=True,
            response_time_ms=response_time_ms,
            used_outlines=hasattr(llm, '_outlines_model') and llm._outlines_model is not None
        ))

        print(f"  ✅ Simple schema: {response_time_ms:.0f}ms - {result.name}, {result.age}, {result.email}")
        return True

    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        tracker.add_result(TestResult(
            provider="mlx",
            model=llm.model,
            schema_name="SimplePersonInfo",
            schema_complexity="simple",
            success=False,
            response_time_ms=response_time_ms,
            validation_error=str(e)
        ))
        print(f"  ❌ Simple schema failed: {e}")
        return False


def test_medium_schema(llm, tracker: TestResultTracker):
    """Test medium schema (nested with enums)"""
    prompt = """Create project: Mobile App Development
Description: Build iOS and Android app for customer portal
Start date: 2025-02-01
Tasks:
1. UI/UX Design (high priority, 15 hours)
2. iOS Development (critical priority, 50 hours)
3. Android Development (critical priority, 50 hours)
4. Backend Integration (medium priority, 25 hours)
Total: 140 hours"""

    start_time = time.time()
    try:
        result = llm.generate(prompt, response_model=MediumProjectSchema, temperature=0)
        response_time_ms = (time.time() - start_time) * 1000

        # Verify structure
        assert isinstance(result, MediumProjectSchema)
        assert result.project_name, "Project name should not be empty"
        assert len(result.tasks) > 0, "Should have at least one task"
        assert all(isinstance(t.priority, Priority) for t in result.tasks), "All tasks should have valid priority"

        tracker.add_result(TestResult(
            provider="mlx",
            model=llm.model,
            schema_name="MediumProjectSchema",
            schema_complexity="medium",
            success=True,
            response_time_ms=response_time_ms,
            used_outlines=hasattr(llm, '_outlines_model') and llm._outlines_model is not None
        ))

        print(f"  ✅ Medium schema: {response_time_ms:.0f}ms - {result.project_name} with {len(result.tasks)} tasks")
        return True

    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        tracker.add_result(TestResult(
            provider="mlx",
            model=llm.model,
            schema_name="MediumProjectSchema",
            schema_complexity="medium",
            success=False,
            response_time_ms=response_time_ms,
            validation_error=str(e)
        ))
        print(f"  ❌ Medium schema failed: {e}")
        return False


def test_complex_schema(llm, tracker: TestResultTracker):
    """Test complex schema (3+ levels deep)"""
    prompt = """Create organization: DataCorp, founded 2018
Team: Data Science
Lead: Maria Garcia (maria@data.com, senior data scientist, Python-9/10-6yrs, SQL-8/10-5yrs, ML-9/10-4yrs)
Members:
- Tom Zhang (tom@data.com, mid data scientist, Python-7/10-3yrs, SQL-7/10-3yrs, manager: maria@data.com)
- Emma Johnson (emma@data.com, junior data scientist, Python-6/10-1.5yrs, manager: maria@data.com)
Total employees: 3"""

    start_time = time.time()
    try:
        result = llm.generate(prompt, response_model=ComplexOrganizationSchema, temperature=0)
        response_time_ms = (time.time() - start_time) * 1000

        # Verify deep nesting
        assert isinstance(result, ComplexOrganizationSchema)
        assert result.company_name, "Company name should not be empty"
        assert len(result.teams) > 0, "Should have at least one team"
        assert isinstance(result.teams[0].lead, Employee), "Team lead should be Employee"
        assert len(result.teams[0].lead.skills) > 0, "Lead should have skills"

        tracker.add_result(TestResult(
            provider="mlx",
            model=llm.model,
            schema_name="ComplexOrganizationSchema",
            schema_complexity="complex",
            success=True,
            response_time_ms=response_time_ms,
            used_outlines=hasattr(llm, '_outlines_model') and llm._outlines_model is not None
        ))

        print(f"  ✅ Complex schema: {response_time_ms:.0f}ms - {result.company_name} with {result.total_employees} employees")
        return True

    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        tracker.add_result(TestResult(
            provider="mlx",
            model=llm.model,
            schema_name="ComplexOrganizationSchema",
            schema_complexity="complex",
            success=False,
            response_time_ms=response_time_ms,
            validation_error=str(e)
        ))
        print(f"  ❌ Complex schema failed: {e}")
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests with available MLX models"""
    print("=" * 80)
    print("MLX WITH OUTLINES NATIVE STRUCTURED OUTPUT TESTING")
    print("=" * 80)
    print()

    tracker = TestResultTracker()

    # Test with available MLX model
    print("🍎 Testing MLX Model on Apple Silicon")
    print("-" * 80)
    try:
        model_name = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
        print(f"Model: {model_name}")
        llm = create_llm("mlx", model=model_name)

        test_simple_schema(llm, tracker)
        test_medium_schema(llm, tracker)
        test_complex_schema(llm, tracker)

        print()
    except Exception as e:
        print(f"❌ Failed to test MLX model: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Print summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"Total tests: {len(tracker.results)}")
    print(f"Successful: {len([r for r in tracker.results if r.success])}")
    print(f"Failed: {len([r for r in tracker.results if not r.success])}")
    print()

    if tracker.results:
        print("By Complexity:")
        for complexity in ["simple", "medium", "complex"]:
            results_for_complexity = [r for r in tracker.results if r.schema_complexity == complexity]
            if results_for_complexity:
                success_rate = tracker.get_success_rate(complexity=complexity)
                avg_time = tracker.get_avg_response_time(complexity=complexity)
                print(f"  {complexity.capitalize()}: {success_rate:.1f}% success, {avg_time:.0f}ms avg")
        print()

        # Check Outlines usage
        outlines_used = any(r.used_outlines for r in tracker.results)
        print(f"Outlines used: {'Yes' if outlines_used else 'No'}")
        print()

    # Save results
    output_file = "test_results_mlx_outlines.json"
    tracker.save_results(output_file)
    print(f"Results saved to: {output_file}")
    print()

    return tracker


if __name__ == "__main__":
    tracker = run_all_tests()
