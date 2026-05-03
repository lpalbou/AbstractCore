"""Compact native structured-output coverage for local providers.

The original version of this file ran a large Ollama x LM Studio x model x
schema x prompt matrix on every pytest invocation. That made ordinary test runs
slow and noisy whenever local model servers happened to be available.

Default coverage here is intentionally cheap:
- one fake-native unit test verifies that the structured handler uses the native
  path and validates a nested enum schema;
- live local-provider checks are gated behind environment flags.

Live modes:
- ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1
  runs one compact nested-schema smoke case per configured local provider.
- ABSTRACTCORE_RUN_COMPREHENSIVE_NATIVE_STRUCTURED_TESTS=1
  expands the live check to simple, nested, and deep schemas. Add comma-separated
  model lists with ABSTRACTCORE_OLLAMA_STRUCTURED_TEST_MODELS or
  ABSTRACTCORE_LMSTUDIO_STRUCTURED_TEST_MODELS when benchmarking several models.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pytest
from pydantic import BaseModel

from abstractcore import create_llm
from abstractcore.core.types import GenerateResponse
from abstractcore.structured import StructuredOutputHandler
from abstractcore.structured.retry import FeedbackRetry


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: Iterable[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or list(default)


def _short_error(error: Exception) -> str:
    """Keep skip output small when providers include long model lists."""
    text = str(error).strip()
    for marker in ("✅ Available models", "Available models", "\n  •"):
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx].strip()
            break
    text = " ".join(text.split())
    if len(text) > 240:
        text = text[:237].rstrip() + "..."
    return text or type(error).__name__


def _looks_like_local_availability_error(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in text
        for marker in (
            "connection",
            "connecterror",
            "connection refused",
            "model not found",
            "not found for",
            "not running",
            "operation not permitted",
            "timeout",
            "timed out",
        )
    )


class SimplePersonInfo(BaseModel):
    name: str
    age: int
    email: str


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
    description: str
    tasks: list[Task]
    total_hours: float


class Department(str, Enum):
    ENGINEERING = "engineering"
    MARKETING = "marketing"


class Employee(BaseModel):
    name: str
    email: str
    department: Department
    skills: list[str]


class Team(BaseModel):
    name: str
    department: Department
    lead: Employee
    members: list[Employee]
    active_projects: list[Project]


class Organization(BaseModel):
    company_name: str
    founded_year: int
    departments: list[Department]
    teams: list[Team]
    total_employees: int


class OllamaProvider:
    """Tiny fake provider whose class name exercises the native handler branch."""

    model = "fake-native"
    model_capabilities = {"structured_output": "native"}

    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict[str, object]] = []

    def _generate_internal(self, prompt: str, response_model=None, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "response_model": response_model,
                "kwargs": dict(kwargs),
            }
        )
        return GenerateResponse(content=self.content, model=self.model, finish_reason="stop")


def test_native_handler_validates_nested_enum_schema_without_backend():
    """Fast regression coverage for the native structured-output path."""
    provider = OllamaProvider(
        json.dumps(
            {
                "name": "Website Redesign",
                "description": "Refresh the public site.",
                "tasks": [
                    {
                        "title": "Design mockups",
                        "description": "Create the main screens.",
                        "priority": "high",
                        "status": "pending",
                        "estimated_hours": 8,
                    }
                ],
                "total_hours": 8,
            }
        )
    )

    handler = StructuredOutputHandler(retry_strategy=FeedbackRetry(max_attempts=1))
    result = handler.generate_structured(
        provider=provider,
        prompt="Create a project with one design task.",
        response_model=Project,
        temperature=0,
        max_output_tokens=256,
    )

    assert isinstance(result, Project)
    assert result.tasks[0].priority is Priority.HIGH
    assert provider.calls[0]["response_model"] is Project
    assert provider.calls[0]["kwargs"] == {"temperature": 0, "max_output_tokens": 256}


@dataclass(frozen=True)
class SchemaCase:
    name: str
    response_model: type[BaseModel]
    prompt: str
    max_output_tokens: int


SCHEMA_CASES = {
    "simple": SchemaCase(
        name="simple",
        response_model=SimplePersonInfo,
        prompt="Extract a person: John Doe, 35 years old, john@example.com.",
        max_output_tokens=160,
    ),
    "nested": SchemaCase(
        name="nested",
        response_model=Project,
        prompt=(
            "Create project 'Website Redesign' with one task: 'Design mockups', "
            "high priority, pending status, 8 hours. Total: 8 hours."
        ),
        max_output_tokens=256,
    ),
    "deep": SchemaCase(
        name="deep",
        response_model=Organization,
        prompt=(
            "Create organization 'TechCorp', founded in 2020, departments engineering "
            "and marketing. Include one engineering team 'Platform' led by Sarah "
            "(sarah@tech.com, skills Python and AWS), with Bob (bob@tech.com, skill "
            "JavaScript) as member. One active project named 'Runtime'. Total employees: 2."
        ),
        max_output_tokens=640,
    ),
}


@dataclass(frozen=True)
class LiveCase:
    provider: str
    model: str
    schema: SchemaCase

    @property
    def id(self) -> str:
        return f"{self.provider}-{self.model}-{self.schema.name}".replace("/", "_")


def _provider_models(provider: str) -> list[str]:
    if provider == "ollama":
        return _csv_env(
            "ABSTRACTCORE_OLLAMA_STRUCTURED_TEST_MODELS",
            _csv_env(
                "ABSTRACTCORE_OLLAMA_STRUCTURED_TEST_MODEL",
                [os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")],
            ),
        )
    if provider == "lmstudio":
        return _csv_env(
            "ABSTRACTCORE_LMSTUDIO_STRUCTURED_TEST_MODELS",
            _csv_env(
                "ABSTRACTCORE_LMSTUDIO_STRUCTURED_TEST_MODEL",
                [os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-4b-2507")],
            ),
        )
    raise ValueError(f"Unsupported native structured provider: {provider}")


def _live_cases() -> list[LiveCase]:
    providers = _csv_env("ABSTRACTCORE_STRUCTURED_NATIVE_PROVIDERS", ["ollama", "lmstudio"])
    schema_names = ["nested"]
    if _env_flag("ABSTRACTCORE_RUN_COMPREHENSIVE_NATIVE_STRUCTURED_TESTS"):
        schema_names = ["simple", "nested", "deep"]

    cases: list[LiveCase] = []
    for provider in providers:
        if provider not in {"ollama", "lmstudio"}:
            continue
        for model in _provider_models(provider):
            for schema_name in schema_names:
                cases.append(LiveCase(provider=provider, model=model, schema=SCHEMA_CASES[schema_name]))
    return cases


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (
        _env_flag("ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS")
        or _env_flag("ABSTRACTCORE_RUN_COMPREHENSIVE_NATIVE_STRUCTURED_TESTS")
    ),
    reason=(
        "Local native structured tests disabled; set ABSTRACTCORE_RUN_LOCAL_PROVIDER_TESTS=1 "
        "for smoke coverage or ABSTRACTCORE_RUN_COMPREHENSIVE_NATIVE_STRUCTURED_TESTS=1 "
        "for the three-level matrix"
    ),
)
@pytest.mark.parametrize("case", _live_cases(), ids=lambda case: case.id)
def test_local_native_structured_output_live(case: LiveCase):
    """Live smoke/matrix coverage for Ollama and LM Studio native schema paths."""
    timeout = float(os.getenv("ABSTRACTCORE_STRUCTURED_NATIVE_TEST_TIMEOUT", "45"))
    try:
        llm = create_llm(case.provider, model=case.model, timeout=timeout)
    except Exception as error:
        pytest.skip(f"{case.provider}/{case.model} unavailable: {_short_error(error)}")

    handler = StructuredOutputHandler(retry_strategy=FeedbackRetry(max_attempts=1))
    try:
        result = handler.generate_structured(
            provider=llm,
            prompt=case.schema.prompt,
            response_model=case.schema.response_model,
            temperature=0,
            max_output_tokens=case.schema.max_output_tokens,
        )
    except Exception as error:
        if _looks_like_local_availability_error(error):
            pytest.skip(f"{case.provider}/{case.model} unavailable during generation: {_short_error(error)}")
        raise

    assert isinstance(result, case.schema.response_model)
