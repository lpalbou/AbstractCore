# Structured Output

AbstractCore implements structured output generation using Pydantic models with automatic schema validation and provider-specific optimizations. The system employs a dual-strategy architecture that adapts to provider capabilities, delivering reliable schema compliance across all supported LLM providers.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Provider Implementation](#provider-implementation)
4. [Usage Guide](#usage-guide)
5. [Schema Design](#schema-design)
6. [Performance Characteristics](#performance-characteristics)
7. [Error Handling](#error-handling)
8. [Production Deployment](#production-deployment)
9. [API Reference](#api-reference)

---

## Overview

### What is Structured Output?

Structured output constrains LLM responses to conform to predefined schemas, enabling direct deserialization into typed objects. AbstractCore uses Pydantic BaseModel classes to define schemas and validate responses.

### Basic Example

```python
from abstractcore import create_llm
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    email: str

llm = create_llm("openai", model="gpt-4o-mini")
person = llm.generate(
    "Extract: John Doe, 35 years old, john@example.com",
    response_model=Person
)

# person is a validated Person instance
assert isinstance(person, Person)
assert person.name == "John Doe"
assert person.age == 35
```

### Key Benefits

- **Type Safety**: Responses are validated Pydantic instances with full IDE support
- **Schema Compliance**: Automatic validation ensures data conforms to defined structure
- **Provider Agnostic**: Identical API across OpenAI, Anthropic, Ollama, LMStudio, HuggingFace, MLX
- **Automatic Strategy Selection**: Framework selects optimal implementation based on provider capabilities
- **Test Coverage**: Supported strategies are exercised by the repository test suite (see `tests/structured/`)

---

## Architecture

### Dual-Strategy Design

AbstractCore implements two distinct strategies for structured output generation:

#### Strategy 1: Native Structured Output (Server-Side Enforcement)

**Mechanism**: Provider API accepts JSON schema and enforces compliance before returning response.

**Providers**:
- OpenAI (via `response_format` parameter)
- Anthropic (via tool-calling mechanism)
- Ollama (via `format` parameter)
- LMStudio (via `response_format` parameter)
- HuggingFace GGUF models (via `response_format` parameter with llama-cpp-python)

**Characteristics**:
- Server-side schema validation
- Zero client-side validation retries required
- Deterministic schema compliance
- Optimal performance for production workloads

**Validation**:
- Structured output behavior is covered by automated tests in this repo (see `tests/structured/`).
- Exact success rates and latency depend on provider/model/schema complexity.

#### Strategy 2: Prompted with Validation (Client-Side Enforcement)

**Mechanism**: Schema embedded in system prompt; response extracted, validated, and retried if necessary.

**Providers**:
- HuggingFace (Transformers models)
- MLX
- Any provider without native support

**Characteristics**:
- Schema injected into enhanced prompt
- Client-side Pydantic validation
- Automatic retry with error feedback (up to 3 attempts)
- Fallback for providers without native support

### Automatic Strategy Selection

The `StructuredOutputHandler` selects the appropriate strategy automatically:

```python
def _has_native_support(self, provider) -> bool:
    """Detect native structured output capability"""
    provider_name = provider.__class__.__name__

    # Ollama and LMStudio always have native support
    if provider_name in ['OllamaProvider', 'LMStudioProvider']:
        return True

    # HuggingFace GGUF models (via llama-cpp-python)
    if provider_name == 'HuggingFaceProvider':
        if hasattr(provider, 'model_type') and provider.model_type == 'gguf':
            return True

    # Check model capabilities for other providers
    capabilities = getattr(provider, 'model_capabilities', {})
    return capabilities.get("structured_output") == "native"
```

No configuration required—the framework handles strategy selection transparently.

---

## Provider Implementation

### OpenAI

**Implementation**: Native support via `response_format` parameter

```python
# AbstractCore implementation (simplified)
payload["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": response_model.__name__,
        "schema": response_model.model_json_schema()
    }
}
```

**Models with Native Support**:
- gpt-4o, gpt-4o-mini
- gpt-4-turbo
- gpt-3.5-turbo

**Reference**: [OpenAI Structured Outputs Documentation](https://platform.openai.com/docs/guides/structured-outputs)

---

### Anthropic

**Implementation**: Native support via tool-calling mechanism

The provider forces execution of a tool whose input schema matches the desired output structure.

**Models with Native Support**:
- claude-haiku-4-5
- claude-sonnet-4-5
- claude-opus-4-5

**Reference**: [Anthropic API Documentation](https://docs.anthropic.com/)

---

### Ollama

**Implementation**: Native support via `format` parameter

```python
# AbstractCore implementation (abstractcore/providers/ollama_provider.py:147-152)
if response_model and PYDANTIC_AVAILABLE:
    json_schema = response_model.model_json_schema()
    payload["format"] = json_schema  # Full schema, server-side validation
```

**Mechanism**:
1. Full JSON schema passed to Ollama API
2. Server-side constrained sampling enforces schema compliance
3. Response is expected to follow the schema (provider/model dependent)

**Notes**:
- Native structured output depends on the Ollama server/build and the selected model.
- For example coverage, see `tests/structured/`.

**Supported Models**: Many models, including:
- Llama 3.1, 3.2, 3.3 family
- Qwen 2.5, 3, 3-coder family
- Gemma 2b, 7b, gemma2, gemma3
- Mistral, Phi-3, Phi-4, GLM-4, DeepSeek-R1

**Reference**: [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

---

### LMStudio

**Implementation**: Native support via OpenAI-compatible `response_format` parameter

```python
# AbstractCore implementation (abstractcore/providers/lmstudio_provider.py:211-222)
if response_model and PYDANTIC_AVAILABLE:
    json_schema = response_model.model_json_schema()
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": json_schema
        }
    }
```

**Mechanism**:
1. OpenAI-compatible format passed to LMStudio server
2. Server-side schema enforcement via underlying inference engine
3. Response is expected to follow the schema (server/model dependent)

**Notes**:
- Behavior depends on the LMStudio server version and underlying model/runtime.
- For example coverage, see `tests/structured/`.

**Reference**: [LMStudio Documentation](https://lmstudio.ai/docs)

---

### HuggingFace

**Implementation**: Dual strategy based on model type

#### GGUF Models (Native Support)

**Backend**: llama-cpp-python with native structured output

```python
# AbstractCore implementation (abstractcore/providers/huggingface_provider.py:669-680)
if response_model and PYDANTIC_AVAILABLE:
    json_schema = response_model.model_json_schema()
    generation_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": json_schema
        }
    }
```

**Notes**:
- GGUF structured output support depends on the llama-cpp-python backend and model.
- For example coverage, see `tests/structured/`.

#### Transformers Models (Native via Outlines)

**Backend**: Hugging Face Transformers library with Outlines

**Implementation**: Native support via Outlines constrained generation

```python
# AbstractCore implementation (abstractcore/providers/huggingface_provider.py:514-548)
if response_model and PYDANTIC_AVAILABLE and OUTLINES_AVAILABLE:
    # Cache Outlines model wrapper
    if not hasattr(self, '_outlines_model'):
        self._outlines_model = outlines.from_transformers(
            self.model_instance,
            self.tokenizer
        )

    # Generate with constrained decoding
    generator = self._outlines_model(
        input_text,
        outlines.json_schema(response_model),
        max_tokens=max_tokens
    )

    # Return validated instance
    validated_obj = response_model.model_validate(generator)
```

**Mechanism**:
1. Outlines wraps transformers model and tokenizer
2. JSON schema passed to constrained generator
3. Server-side logit filtering ensures only valid tokens are sampled
4. Schema compliance is enforced via constrained decoding (provider/model dependent)
5. Automatic fallback to prompted approach if Outlines unavailable

**Installation**:
```bash
pip install "abstractcore[huggingface]"  # Includes Outlines automatically
```

**Characteristics**:
- Schema compliance via constrained decoding (still validated client-side)
- Zero or minimal validation retries when supported
- Works with many transformers-compatible models
- Automatic detection and activation when Outlines is installed
- Graceful fallback to prompted approach if Outlines is missing

**Fallback behavior**:
- If Outlines isn't available (or a backend doesn't support constrained decoding), AbstractCore falls back to prompted structured output with validation and retries.
- Exact success rates and latency depend on provider/model/hardware/schema complexity.

---

### MLX (Apple Silicon)

**Implementation**: Native via Outlines

**Backend**: MLX with Outlines constrained generation

```python
# AbstractCore implementation (abstractcore/providers/mlx_provider.py:165-197)
if response_model and PYDANTIC_AVAILABLE and OUTLINES_AVAILABLE:
    # Cache Outlines MLX model wrapper
    if not hasattr(self, '_outlines_model'):
        self._outlines_model = outlines_models.mlxlm(self.model)

    # Generate with constrained decoding
    generator = self._outlines_model(
        full_prompt,
        outlines.json_schema(response_model),
        max_tokens=max_tokens
    )

    # Return validated instance
    validated_obj = response_model.model_validate(generator)
```

**Mechanism**:
1. Outlines MLX backend wraps mlx-lm model
2. JSON schema converted to token constraints
3. Constrained sampling on Apple Silicon hardware
4. Server-side schema enforcement
5. Automatic fallback to prompted approach if Outlines unavailable

**Installation**:
```bash
pip install "abstractcore[mlx]"  # Includes Outlines automatically
```

**Models**:
- mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
- mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
- All MLX-compatible models

**Characteristics**:
- Schema compliance via constrained decoding (still validated client-side)
- Zero or minimal validation retries when supported
- Optimized for Apple M-series processors
- Automatic detection and activation when Outlines installed
- Graceful fallback to prompted approach if Outlines missing

**Performance notes**:
- Prompted structured output (validation + retry) is the default fallback and is often the simplest to run.
- Constrained decoding can be slower or faster depending on backend/model/schema; benchmark on your hardware if it matters.

---

## Usage Guide

### Basic Usage

```python
from abstractcore import create_llm
from pydantic import BaseModel

class ExtractedData(BaseModel):
    name: str
    age: int
    email: str

llm = create_llm("ollama", model="qwen3:4b")
result = llm.generate(
    "Extract: Alice Johnson, 28, alice@example.com",
    response_model=ExtractedData,
    temperature=0  # Recommended for deterministic output
)

print(f"{result.name} ({result.age}): {result.email}")
```

### Using Enums

Enums provide type-safe categorical values:

```python
from enum import Enum
from pydantic import BaseModel

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Task(BaseModel):
    title: str
    priority: Priority
    estimated_hours: float

llm = create_llm("lmstudio", model="qwen/qwen3-4b-2507")
task = llm.generate(
    "Create task: Fix authentication bug, critical priority, 8 hours estimated",
    response_model=Task
)

assert isinstance(task.priority, Priority)
print(f"Priority: {task.priority.value}")  # "critical"
```

**Notes**: Enums are supported and exercised by tests; exact behavior depends on provider/model.

### Nested Objects

```python
from typing import List
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    postal_code: str

class Person(BaseModel):
    name: str
    email: str
    address: Address

llm = create_llm("openai", model="gpt-4o-mini")
person = llm.generate(
    """Extract: John Smith, john@example.com
    Address: 123 Main St, Boston, MA 02101""",
    response_model=Person
)

assert isinstance(person.address, Address)
```

### Complex Hierarchies

Complex schemas with multiple nesting levels are supported:

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

class Department(str, Enum):
    ENGINEERING = "engineering"
    SALES = "sales"
    MARKETING = "marketing"

class EmployeeLevel(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"

class Skill(BaseModel):
    name: str
    proficiency: int  # 1-10 scale
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
    teams: List[Team]
    total_employees: int

llm = create_llm("anthropic", model="claude-haiku-4-5")
org = llm.generate(
    """Create organization: TechCorp, founded 2020
    Team: Platform (engineering)
    Lead: Sarah Chen (sarah@tech.com, senior, Python-9/10-5yrs, AWS-8/10-4yrs)
    Member: Bob Lee (bob@tech.com, mid, JavaScript-7/10-3yrs, manager: sarah@tech.com)
    Total employees: 2""",
    response_model=Organization
)
```

**Notes**: Deeply nested schemas are supported; validate against your target provider/model and see `tests/structured/` for examples.

### Direct Handler Usage

For advanced use cases requiring custom retry configuration:

```python
from abstractcore.structured import StructuredOutputHandler, FeedbackRetry

# Configure custom retry strategy
handler = StructuredOutputHandler(
    retry_strategy=FeedbackRetry(max_attempts=5)
)

result = handler.generate_structured(
    provider=llm,
    prompt="Extract complex data from document...",
    response_model=ComplexSchema,
    temperature=0
)
```

---

## Schema Design

### Design Principles

Well-designed schemas improve validation success rates and reduce response times.

#### 1. Clear Field Naming

Use descriptive, unambiguous field names:

```python
# Recommended
class Employee(BaseModel):
    employee_id: str
    hire_date: str
    department: str
    annual_salary: float

# Avoid
class Employee(BaseModel):
    id: str  # Ambiguous
    date: str  # What date?
    dept: str  # Abbreviation unclear
    salary: float  # Currency? Period?
```

#### 2. Leverage Enums for Categorical Data

Enums provide validation and type safety:

```python
# Recommended
class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class User(BaseModel):
    status: Status  # Only valid enum values accepted

# Avoid
class User(BaseModel):
    status: str  # Any string accepted, no validation
```

#### 3. Use Optional Fields Appropriately

Distinguish required from optional fields:

```python
from typing import Optional, List

class Task(BaseModel):
    # Required fields
    title: str
    created_at: str

    # Optional with defaults
    description: str = ""
    tags: List[str] = []

    # Truly optional (may be None)
    assigned_to: Optional[str] = None
    due_date: Optional[str] = None
```

#### 4. Logical Hierarchy

Group related fields into nested objects:

```python
# Recommended
class ContactInfo(BaseModel):
    email: str
    phone: str
    address: str

class Person(BaseModel):
    name: str
    contact: ContactInfo  # Logical grouping

# Avoid flat structure
class Person(BaseModel):
    name: str
    email: str
    phone: str
    address: str
```

### Complexity Guidelines

Schema complexity affects latency and cost; keep schemas as small as practical.

#### Simple Schemas (< 10 fields, 1 level)

**Example**:
```python
class PersonInfo(BaseModel):
    name: str
    age: int
    email: str
    occupation: str
```

**Recommended for**: User profiles, data extraction, form processing

#### Medium Schemas (10-30 fields, 1-2 levels)

**Example**:
```python
class Project(BaseModel):
    name: str
    description: str
    start_date: str
    tasks: List[Task]  # Nested objects
    total_hours: float
```

**Recommended for**: Project management, task tracking, structured data extraction

#### Complex Schemas (30+ fields, 3+ levels)

**Example**:
```python
class Organization(BaseModel):
    company_name: str
    teams: List[Team]  # Level 2
    # Team contains:
    #   lead: Employee  # Level 3
    #   members: List[Employee]  # Level 3
    #     # Employee contains:
    #     #   skills: List[Skill]  # Level 4
```

**Recommended for**: Organizational hierarchies, knowledge graphs, complex data models

### Anti-Patterns

Avoid these patterns that can degrade performance or reliability:

#### 1. Excessive Nesting Depth (>4 levels)

```python
# Avoid
class Level1(BaseModel):
    level2: Level2
    # Level2 -> Level3 -> Level4 -> Level5 (too deep)
```

**Impact**: Increased token usage, longer response times

#### 2. Ambiguous Enum Values

```python
# Avoid
class Status(str, Enum):
    ONE = "1"
    TWO = "2"
    THREE = "3"

# Recommended
class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
```

#### 3. Overly Long Field Names

```python
# Avoid
class Data(BaseModel):
    very_long_and_descriptive_field_name_that_uses_many_tokens: str

# Recommended
class Data(BaseModel):
    user_email: str  # Clear but concise
```

**Impact**: Increases token count, affecting cost and context window

---

## Performance Characteristics

Structured output performance is highly dependent on:
- Provider/backend strategy (native constrained decoding vs prompted validation/retry)
- Schema complexity (field count + nesting depth)
- Model choice, server configuration, and hardware
- Sampling settings (use `temperature=0` when you care about schema fidelity)

If performance matters, benchmark on your target provider/model/hardware.
Historical benchmark notes (non-authoritative) may exist under `docs/reports/`.

### Temperature Settings

**Recommendation**: Use `temperature=0` for structured outputs

**Rationale**:
- Deterministic responses
- Consistent schema compliance
- Reduced sampling overhead

**When to increase temperature**:
- Creative content generation within schema constraints
- Diverse response generation for the same prompt
- Exploratory data generation

---

## Error Handling

### Error Categories

#### 1. Infrastructure Errors (Retriable)

Network failures, timeouts, server unavailability—retry with exponential backoff:

```python
import time
from requests.exceptions import ConnectionError, Timeout

def generate_with_retry(llm, prompt, response_model, max_retries=3):
    """Retry infrastructure errors with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return llm.generate(
                prompt,
                response_model=response_model,
                temperature=0
            )
        except (ConnectionError, Timeout) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
                continue
            raise

result = generate_with_retry(llm, "Extract data...", DataModel)
```

**Retriable errors**:
- `ConnectionError`: Network connectivity issues
- `TimeoutError`: Request timeout
- HTTP 5xx: Server errors
- Token limit exceeded (retry with simplified schema or chunking)

#### 2. Validation Errors (Non-Retriable)

Schema validation failures indicate schema or prompt issues—do not retry:

```python
from pydantic import ValidationError

try:
    result = llm.generate(
        "Extract user data...",
        response_model=UserModel
    )
except ValidationError as e:
    # Log validation errors
    print("Schema validation failed:")
    for error in e.errors():
        field = " -> ".join(str(loc) for loc in error['loc'])
        print(f"  {field}: {error['msg']}")

    # Fix schema or prompt—do not retry
    raise
```

**Common validation errors**:
- Missing required fields: Schema too strict or prompt unclear
- Type mismatches: Field type incompatible with LLM output
- Enum validation failures: LLM returned invalid enum value

**Resolution**: Revise schema or improve prompt clarity

#### 3. Token Limit Errors

Context window exceeded—simplify schema or split request:

```python
try:
    result = llm.generate(prompt, response_model=ComplexModel)
except Exception as e:
    if "token" in str(e).lower() or "context" in str(e).lower():
        print("Token limit exceeded. Options:")
        print("1. Simplify schema (reduce fields or nesting)")
        print("2. Split into multiple requests")
        print("3. Use model with larger context window")
        raise
```

### Retry Strategy Details

The default `FeedbackRetry` strategy:

1. **Maximum attempts**: 3 (configurable)
2. **Retry condition**: Only `ValidationError` exceptions
3. **Feedback mechanism**: Provides detailed error descriptions to LLM

**Example error feedback**:
```
Your previous response had validation errors:
• Missing required field: 'department'
• Field 'employee_level': Expected one of: junior, mid, senior
• Field 'age': Expected integer, received string
```

The LLM uses this feedback to self-correct on subsequent attempts.

**Configuration**:
```python
from abstractcore.structured import StructuredOutputHandler, FeedbackRetry

handler = StructuredOutputHandler(
    retry_strategy=FeedbackRetry(max_attempts=5)
)
```

---

## Production Deployment

### Pre-Deployment Checklist

Before deploying structured outputs to production:

- [ ] Schema validated locally with Pydantic: `Model.model_validate(test_data)`
- [ ] Success rate measured with target model (target: >95%)
- [ ] Response time benchmarked under expected load
- [ ] Error handling implemented for infrastructure failures
- [ ] Logging configured for validation errors and retries
- [ ] Monitoring configured for success rates and latencies
- [ ] Fallback strategy defined for structured output failures
- [ ] Token limits verified: `len(prompt) + len(schema) + len(response) < context_window`

### Monitoring Metrics

Track these metrics in production:

**Success Metrics**:
- Validation success rate (target: >95%)
- First-attempt success rate
- Average retry count

**Performance Metrics**:
- p50, p95, p99 response times
- Response time by schema complexity
- Token usage statistics

**Error Metrics**:
- Validation error rate by field
- Infrastructure error rate
- Token limit exceeded rate

### Configuration Best Practices

**Temperature**: Set to 0 for deterministic structured outputs
```python
llm.generate(prompt, response_model=Model, temperature=0)
```

**Timeout**: Configure appropriate timeouts based on schema complexity
```python
# Simple schemas: 30s
# Medium schemas: 60s
# Complex schemas: 120s
```

**Provider Selection**:
- Development: Use local providers (Ollama, LMStudio) for cost efficiency
- Production: Select based on performance requirements and budget

### Schema Versioning

Maintain schema version compatibility:

```python
from pydantic import BaseModel, Field

class UserV1(BaseModel):
    name: str
    email: str

class UserV2(BaseModel):
    name: str
    email: str
    department: str = Field(default="unassigned")  # Backward compatible
```

Use optional fields with defaults for backward-compatible schema evolution.

---

## API Reference

### Core Function

```python
llm.generate(
    prompt: str,
    response_model: Type[BaseModel],
    temperature: float = 0.0,
    **kwargs
) -> BaseModel
```

**Parameters**:
- `prompt` (str): Input prompt describing extraction/generation task
- `response_model` (Type[BaseModel]): Pydantic model class defining output schema
- `temperature` (float): Sampling temperature (0.0 = deterministic, 1.0 = creative)
- `**kwargs`: Additional provider-specific parameters

**Returns**:
- Instance of `response_model`, validated and type-safe

**Raises**:
- `ValidationError`: Schema validation failed after all retry attempts
- `ConnectionError`: Network/infrastructure error
- `TimeoutError`: Request timeout

**Example**:
```python
person = llm.generate(
    "Extract: John Doe, age 35",
    response_model=Person,
    temperature=0
)
```

### StructuredOutputHandler

Advanced handler for custom retry strategies:

```python
from abstractcore.structured import StructuredOutputHandler

handler = StructuredOutputHandler(retry_strategy=None)
```

**Methods**:

```python
handler.generate_structured(
    provider: LLMProvider,
    prompt: str,
    response_model: Type[BaseModel],
    **kwargs
) -> BaseModel
```

Generates structured output with automatic strategy selection (native or prompted).

### Retry Strategies

```python
from abstractcore.structured import FeedbackRetry

retry = FeedbackRetry(max_attempts=3)
```

**Parameters**:
- `max_attempts` (int): Maximum retry attempts including initial attempt

**Methods**:
- `should_retry(attempt, error)`: Returns True if retry should occur
- `prepare_retry_prompt(prompt, error, attempt)`: Creates retry prompt with validation feedback

---

## Related Documentation

- [Getting Started](getting-started.md#structured-output) - Quick introduction
- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples.md#structured-output-examples) - Real-world usage patterns
- [Response Model Parameter Analysis](archive/structured-response-keyword.md) - Why `response_model`
- [Native Implementation Test Results](archive/improved-structured-response.md) - Detailed test data

---

## References

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

---

## Testing and Validation

Structured output behavior is exercised by automated tests under `tests/structured/`.

### Running tests

From this repository:

```bash
pip install -e ".[test]"
pytest tests/structured -q
```

Some provider-specific tests require additional extras:

- HuggingFace / Outlines: `pip install -e ".[huggingface]"`
- MLX: `pip install -e ".[mlx]"` (macOS + Apple Silicon only)

If you're installing from PyPI and just want the test dependencies:

```bash
pip install "abstractcore[test]"
pytest -q
```

### Notes

- Performance and success rates vary widely by provider/model/schema complexity and are not guaranteed.
- If performance matters, benchmark on your target hardware/provider setup.
