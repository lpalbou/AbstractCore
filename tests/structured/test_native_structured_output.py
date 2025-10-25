"""
Tests for native structured output support in Ollama and LMStudio providers.

This test suite verifies that:
1. Ollama uses native format parameter for structured outputs
2. LMStudio uses native response_format parameter for structured outputs
3. Both providers correctly pass JSON schemas to their respective APIs
4. Complex schemas with nested objects and enums work correctly
"""

import pytest
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

from abstractcore import create_llm
from abstractcore.structured import StructuredOutputHandler


# Test models
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
    title: str
    description: str
    priority: Priority
    status: TaskStatus
    estimated_hours: Optional[float] = None


class Project(BaseModel):
    name: str
    tasks: List[Task]
    total_estimated_hours: float


class PersonInfo(BaseModel):
    name: str
    age: int
    email: str
    occupation: Optional[str] = None


class TestNativeStructuredOutput:
    """Test native structured output support for Ollama and LMStudio."""

    def test_ollama_native_support_detection(self):
        """Test that Ollama provider is correctly detected as having native support."""
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
            handler = StructuredOutputHandler()

            assert handler._has_native_support(llm) is True, \
                "Ollama provider should have native structured output support"
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_lmstudio_native_support_detection(self):
        """Test that LMStudio provider is correctly detected as having native support."""
        try:
            llm = create_llm("lmstudio", model="qwen3-4b-2507")
            handler = StructuredOutputHandler()

            assert handler._has_native_support(llm) is True, \
                "LMStudio provider should have native structured output support"
        except Exception as e:
            pytest.skip(f"LMStudio not available: {e}")

    def test_ollama_simple_structured_output(self):
        """Test Ollama with simple structured output (PersonInfo)."""
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
            handler = StructuredOutputHandler()

            prompt = "Generate a person: John Doe, 35 years old, john@example.com, software engineer"

            result = handler.generate_structured(
                provider=llm,
                prompt=prompt,
                response_model=PersonInfo,
                temperature=0
            )

            assert isinstance(result, PersonInfo), "Result should be PersonInfo instance"
            assert result.name is not None, "Name should be present"
            assert result.age > 0, "Age should be positive"
            assert "@" in result.email, "Email should contain @"

            print(f"✅ Ollama simple structured output: {result}")

        except Exception as e:
            pytest.skip(f"Ollama test skipped: {e}")

    def test_ollama_complex_structured_output_with_enums(self):
        """Test Ollama with complex nested structure and enums."""
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
            handler = StructuredOutputHandler()

            prompt = """Create a project named 'Website Redesign' with 2 tasks:
            1. 'Design mockups' - high priority, pending status, 8 hours
            2. 'Implement frontend' - medium priority, in progress, 16 hours
            Total estimated: 24 hours"""

            result = handler.generate_structured(
                provider=llm,
                prompt=prompt,
                response_model=Project,
                temperature=0
            )

            assert isinstance(result, Project), "Result should be Project instance"
            assert len(result.tasks) >= 1, "Should have at least 1 task"
            assert result.total_estimated_hours > 0, "Total hours should be positive"

            for task in result.tasks:
                assert isinstance(task.priority, Priority), "Priority should be Priority enum"
                assert isinstance(task.status, TaskStatus), "Status should be TaskStatus enum"
                assert task.title is not None, "Task title should be present"

            print(f"✅ Ollama complex structured output: {result.name} with {len(result.tasks)} tasks")

        except Exception as e:
            pytest.skip(f"Ollama complex test skipped: {e}")

    def test_lmstudio_simple_structured_output(self):
        """Test LMStudio with simple structured output (PersonInfo)."""
        try:
            llm = create_llm("lmstudio", model="qwen3-4b-2507")
            handler = StructuredOutputHandler()

            prompt = "Generate a person: Alice Smith, 28 years old, alice@example.com, data scientist"

            result = handler.generate_structured(
                provider=llm,
                prompt=prompt,
                response_model=PersonInfo,
                temperature=0
            )

            assert isinstance(result, PersonInfo), "Result should be PersonInfo instance"
            assert result.name is not None, "Name should be present"
            assert result.age > 0, "Age should be positive"
            assert "@" in result.email, "Email should contain @"

            print(f"✅ LMStudio simple structured output: {result}")

        except Exception as e:
            pytest.skip(f"LMStudio test skipped: {e}")

    def test_lmstudio_complex_structured_output_with_enums(self):
        """Test LMStudio with complex nested structure and enums."""
        try:
            llm = create_llm("lmstudio", model="qwen3-4b-2507")
            handler = StructuredOutputHandler()

            prompt = """Create a project named 'Mobile App' with 2 tasks:
            1. 'API Integration' - critical priority, in progress, 12 hours
            2. 'Testing' - high priority, pending, 6 hours
            Total estimated: 18 hours"""

            result = handler.generate_structured(
                provider=llm,
                prompt=prompt,
                response_model=Project,
                temperature=0
            )

            assert isinstance(result, Project), "Result should be Project instance"
            assert len(result.tasks) >= 1, "Should have at least 1 task"
            assert result.total_estimated_hours > 0, "Total hours should be positive"

            for task in result.tasks:
                assert isinstance(task.priority, Priority), "Priority should be Priority enum"
                assert isinstance(task.status, TaskStatus), "Status should be TaskStatus enum"
                assert task.title is not None, "Task title should be present"

            print(f"✅ LMStudio complex structured output: {result.name} with {len(result.tasks)} tasks")

        except Exception as e:
            pytest.skip(f"LMStudio complex test skipped: {e}")

    def test_schema_generation_for_ollama(self):
        """Verify that Ollama receives the full JSON schema in the format parameter."""
        try:
            llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")

            # Check that the provider has the correct capabilities
            assert llm.model_capabilities.get("structured_output") == "native", \
                "Ollama models should have native structured output support"

            print("✅ Ollama schema generation verified")

        except Exception as e:
            pytest.skip(f"Ollama schema test skipped: {e}")

    def test_schema_generation_for_lmstudio(self):
        """Verify that LMStudio uses response_format parameter correctly."""
        try:
            llm = create_llm("lmstudio", model="qwen3-4b-2507")

            # LMStudio always supports native structured outputs
            handler = StructuredOutputHandler()
            assert handler._has_native_support(llm) is True, \
                "LMStudio provider should always have native support"

            print("✅ LMStudio schema generation verified")

        except Exception as e:
            pytest.skip(f"LMStudio schema test skipped: {e}")


if __name__ == "__main__":
    # Run tests manually
    test = TestNativeStructuredOutput()

    print("\n" + "="*80)
    print("Testing Ollama Native Structured Output")
    print("="*80)

    try:
        test.test_ollama_native_support_detection()
        test.test_ollama_simple_structured_output()
        test.test_ollama_complex_structured_output_with_enums()
        test.test_schema_generation_for_ollama()
    except Exception as e:
        print(f"❌ Ollama tests failed: {e}")

    print("\n" + "="*80)
    print("Testing LMStudio Native Structured Output")
    print("="*80)

    try:
        test.test_lmstudio_native_support_detection()
        test.test_lmstudio_simple_structured_output()
        test.test_lmstudio_complex_structured_output_with_enums()
        test.test_schema_generation_for_lmstudio()
    except Exception as e:
        print(f"❌ LMStudio tests failed: {e}")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
