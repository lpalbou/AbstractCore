# Structured Output Module

## Purpose

The **structured** module provides robust structured output capabilities for AbstractCore, enabling LLM responses to be automatically validated against Pydantic models. The module implements two distinct strategies for structured output generation:

1. **Native support** - Leverages provider-specific server-side schema enforcement for guaranteed compliance
2. **Prompted approach** - Uses schema-enhanced prompts with validation, self-healing, and intelligent retry mechanisms

This dual-strategy approach ensures maximum reliability across all providers, automatically selecting the optimal method based on provider capabilities.

## Architecture Position

**Layer**: Processing Layer (between Core and Providers)

**Dependencies**:
- `pydantic` - Schema definition and validation
- `abstractcore.providers` - LLM provider implementations
- `abstractcore.utils.structured_logging` - Structured logging
- `abstractcore.utils.self_fixes` - JSON repair utilities
- `abstractcore.events` - Event emission system

**Used By**:
- `abstractcore.core.base_llm` - BaseLLM base class for structured generation
- `abstractcore.core.factory` - LLM factory for creating instances
- User applications - Direct usage for structured outputs

## Component Structure

```
abstractcore/structured/
├── __init__.py                # Public API exports
├── retry.py                   # Retry strategy implementations
├── handler.py                 # Core structured output handler
└── README.md                  # This documentation
```

### File Overview

| File | Purpose | Key Classes |
|------|---------|-------------|
| `retry.py` | Retry strategies for validation failures | `Retry`, `FeedbackRetry` |
| `handler.py` | Main structured output orchestration | `StructuredOutputHandler` |

## Detailed Components

### retry.py - Retry Strategies

Implements intelligent retry mechanisms for handling validation failures during structured output generation.

#### Core Classes

**1. `Retry` (Abstract Base Class)**

Base class defining the retry strategy interface.

```python
class Retry(ABC):
    @abstractmethod
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if retry should be attempted"""
        pass

    @abstractmethod
    def prepare_retry_prompt(self, original_prompt: str, error: Exception, attempt: int) -> str:
        """Prepare prompt with error feedback for retry"""
        pass
```

**2. `FeedbackRetry` (Production Implementation)**

The primary retry strategy that feeds Pydantic validation errors back to the LLM for self-correction.

```python
class FeedbackRetry(Retry):
    def __init__(self, max_attempts: int = 3):
        """
        Initialize retry strategy.

        Args:
            max_attempts: Maximum attempts including initial (default: 3)
        """
```

**Key Features**:
- Retries only for `ValidationError` exceptions
- Formats validation errors into clear, actionable feedback
- Enhances prompts with specific error details
- Respects maximum attempt limits
- User-friendly error message formatting

**Error Formatting**:
- `missing` errors → "Missing required field: 'field_name'"
- `int_parsing` errors → "Expected an integer, got text that can't be converted"
- `string_type` errors → "Expected a string"
- `value_error` → Field-specific custom validation messages
- `json_invalid` → JSON parsing error details

**Usage Example**:

```python
from abstractcore.structured import FeedbackRetry

# Conservative retry strategy
retry = FeedbackRetry(max_attempts=2)

# Standard retry strategy (default)
retry = FeedbackRetry(max_attempts=3)

# Aggressive retry strategy
retry = FeedbackRetry(max_attempts=5)
```

### handler.py - Structured Output Handler

The core orchestrator that manages structured output generation using native or prompted strategies.

#### Core Class: `StructuredOutputHandler`

**Responsibilities**:
1. Detect provider native support capabilities
2. Route to native or prompted generation
3. Validate responses against Pydantic schemas
4. Handle JSON extraction and repair
5. Manage retry logic with validation feedback
6. Emit structured logging and events

**Initialization**:

```python
class StructuredOutputHandler:
    def __init__(self, retry_strategy: Optional[FeedbackRetry] = None):
        """
        Initialize handler with optional custom retry strategy.

        Args:
            retry_strategy: Retry strategy (defaults to FeedbackRetry())
        """
```

**Main Method**:

```python
def generate_structured(
    self,
    provider,
    prompt: str,
    response_model: Type[BaseModel],
    **kwargs
) -> BaseModel:
    """
    Generate structured output using best available strategy.

    Args:
        provider: LLM provider instance
        prompt: Input prompt
        response_model: Pydantic model for validation
        **kwargs: Additional provider parameters

    Returns:
        Validated instance of response_model

    Raises:
        ValidationError: If validation fails after all retries
    """
```

**Strategy Detection**:

```python
def _has_native_support(self, provider) -> bool:
    """
    Detect native structured output support.

    Detection Logic:
    - Ollama: Always native (uses 'format' parameter)
    - LMStudio: Always native (uses 'response_format' parameter)
    - HuggingFace GGUF: Native (llama-cpp-python)
    - HuggingFace Transformers: Native if Outlines installed
    - MLX: Native if Outlines installed
    - Others: Check model_capabilities.json

    Returns:
        True if provider supports native structured outputs
    """
```

**Native Generation**:

```python
def _generate_native(
    self,
    provider,
    prompt: str,
    response_model: Type[BaseModel],
    **kwargs
) -> BaseModel:
    """
    Generate using provider's native structured output support.

    Process:
    1. Pass response_model to provider's _generate_internal()
    2. Provider enforces schema server-side
    3. Parse and validate response
    4. Fallback to prompted if native fails

    Returns:
        Validated instance of response_model
    """
```

**Prompted Generation**:

```python
def _generate_prompted(
    self,
    provider,
    prompt: str,
    response_model: Type[BaseModel],
    **kwargs
) -> BaseModel:
    """
    Generate using prompted approach with validation and retry.

    Process:
    1. Create schema-enhanced prompt
    2. Generate response
    3. Extract and repair JSON
    4. Validate against schema
    5. Retry with feedback on validation errors

    Returns:
        Validated instance of response_model

    Raises:
        ValidationError: If all retry attempts fail
    """
```

**Schema Enhancement**:

```python
def _create_schema_prompt(self, prompt: str, response_model: Type[BaseModel]) -> str:
    """
    Create prompt with embedded JSON schema.

    Features:
    - Includes full JSON schema definition
    - Generates example structure
    - Simplifies enum schemas for prompted providers
    - Provides clear formatting instructions

    Returns:
        Enhanced prompt with schema information
    """
```

**JSON Extraction**:

```python
def _extract_json(self, content: str) -> str:
    """
    Extract JSON from responses with additional text.

    Extraction Strategies:
    1. Direct parsing (if starts with '{' and ends with '}')
    2. Code block extraction (```json ... ```)
    3. Pattern matching for JSON objects
    4. Fallback to original content

    Returns:
        Extracted JSON string
    """
```

**Enum Simplification** (Prompted Providers):

For prompted providers, the handler simplifies enum schemas to avoid LLM confusion:

```python
def _simplify_enum_schemas(self, schema: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """
    Simplify enum schemas while preserving mappings.

    Process:
    1. Identify enum definitions in $defs
    2. Replace enum references with inline definitions
    3. Add clear enum value descriptions
    4. Store mappings for response preprocessing

    Returns:
        Tuple of (simplified_schema, enum_mappings)
    """
```

**Enum Response Preprocessing**:

```python
def _preprocess_enum_response(self, data: Dict[str, Any], enum_mappings: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Convert Python enum notation back to valid enum values.

    Handles:
    - "EnumClass.VALUE_NAME" notation
    - "<EnumClass.VALUE_NAME: 'value'>" repr format

    Returns:
        Preprocessed data with corrected enum values
    """
```

## Native vs Prompted Approaches

### Native Support (Server-Side Schema Enforcement)

**When Used**:
- **Ollama**: All models (uses `format` parameter with JSON schema)
- **LMStudio**: All models (uses `response_format` with OpenAI-compatible schema)
- **HuggingFace GGUF**: Models loaded via llama-cpp-python
- **HuggingFace Transformers**: When Outlines library is installed
- **MLX**: When Outlines library is installed

**Advantages**:
- ✅ **Guaranteed schema compliance** - Server enforces structure
- ✅ **No validation retries needed** - Structure always correct
- ✅ **Handles complex schemas** - Deep nesting, enums, constraints
- ✅ **Faster for complex schemas** - No retry overhead

**Limitations**:
- ⚠️ May be slower for simple schemas due to constrained generation overhead
- ⚠️ Requires provider support

**Performance** (from comprehensive testing):
- Success rate: **100%** (20/20 tests)
- Retry rate: **0%** (no validation retries)
- Schema compliance: **Perfect** (zero violations)

### Prompted Approach (Schema-Enhanced Prompts)

**When Used**:
- **OpenAI**: All models (native support not yet implemented)
- **Anthropic**: All models (native support not yet implemented)
- **Google**: All models (native support not yet implemented)
- **HuggingFace Transformers**: When Outlines not installed
- **MLX**: When Outlines not installed
- **Fallback**: When native generation fails

**Advantages**:
- ✅ **Universal compatibility** - Works with all providers
- ✅ **Faster for simple schemas** - No constrained generation overhead
- ✅ **Self-healing** - Automatic JSON repair attempts
- ✅ **Intelligent retry** - LLM learns from validation errors

**Features**:
- Schema-enhanced prompts with examples
- JSON extraction from code blocks
- Automatic JSON repair (via `fix_json`)
- Validation error feedback to LLM
- Configurable retry strategies
- Enum simplification for LLM clarity

**Performance** (from comprehensive testing):
- Success rate: **100%** for quality models
- Retry rate: **Low** (typically 0-1 retries)
- Speed: **2-5x faster** than native for simple schemas

## Usage Patterns

### Basic Usage (Automatic Strategy Selection)

The handler automatically selects the optimal strategy based on provider capabilities:

```python
from abstractcore import create_llm
from pydantic import BaseModel

# Define your schema
class Person(BaseModel):
    name: str
    age: int
    email: str

# Create LLM (any provider)
llm = create_llm("ollama", model="qwen3:4b")  # Native support
# llm = create_llm("openai", model="gpt-4")   # Prompted approach

# Generate structured output
response = llm.generate(
    prompt="Extract: John Doe, 35, john@example.com",
    response_model=Person,
    temperature=0  # Recommended for deterministic outputs
)

# Access validated data
print(response.name)   # "John Doe"
print(response.age)    # 35
print(response.email)  # "john@example.com"
```

### Native Support Detection

Check if a provider supports native structured outputs:

```python
from abstractcore import create_llm
from abstractcore.structured import StructuredOutputHandler

llm = create_llm("ollama", model="qwen3:4b")
handler = StructuredOutputHandler()

# Check native support
if handler._has_native_support(llm):
    print("Native support available - guaranteed schema compliance")
else:
    print("Using prompted approach - validation with retry")
```

### Complex Schemas with Nested Objects

```python
from typing import List
from pydantic import BaseModel
from enum import Enum

class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class Task(BaseModel):
    title: str
    description: str
    status: TaskStatus
    priority: int

class Project(BaseModel):
    name: str
    tasks: List[Task]
    owner: str

llm = create_llm("lmstudio", model="qwen3-4b")

response = llm.generate(
    prompt="""
    Extract project information:
    Project: Website Redesign
    Owner: Alice
    Tasks:
    1. Design mockups (in progress, priority 1)
    2. Implement frontend (todo, priority 2)
    3. Deploy to staging (todo, priority 3)
    """,
    response_model=Project,
    temperature=0
)

print(f"Project: {response.name}")
print(f"Tasks: {len(response.tasks)}")
for task in response.tasks:
    print(f"  - {task.title}: {task.status.value} (P{task.priority})")
```

### Custom Retry Configuration

Configure retry behavior for specific use cases:

```python
from abstractcore.structured import FeedbackRetry, StructuredOutputHandler
from abstractcore import create_llm
from pydantic import BaseModel

class Data(BaseModel):
    value: int
    label: str

# Conservative: 2 attempts (1 initial + 1 retry)
retry = FeedbackRetry(max_attempts=2)
handler = StructuredOutputHandler(retry_strategy=retry)

# Aggressive: 5 attempts (1 initial + 4 retries)
retry = FeedbackRetry(max_attempts=5)
handler = StructuredOutputHandler(retry_strategy=retry)

# Use with LLM
llm = create_llm("openai", model="gpt-4o-mini")
result = handler.generate_structured(
    provider=llm,
    prompt="Extract: value is 42, label is 'test'",
    response_model=Data
)
```

### Handling Validation Errors

```python
from pydantic import BaseModel, field_validator, ValidationError
from abstractcore import create_llm

class PositiveNumber(BaseModel):
    value: int

    @field_validator('value')
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('value must be positive')
        return v

llm = create_llm("ollama", model="qwen3:4b")

try:
    response = llm.generate(
        prompt="Extract: the value is -5",
        response_model=PositiveNumber,
        temperature=0
    )
except ValidationError as e:
    print(f"Validation failed after all retries:")
    for error in e.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        print(f"  {field}: {error['msg']}")
```

## Integration Points

### With Providers

Providers integrate with structured outputs by accepting `response_model` parameter:

```python
# In provider implementations
def _generate_internal(
    self,
    prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    **kwargs
) -> GenerationResponse:
    """
    Generate with optional structured output.

    Args:
        response_model: Optional Pydantic model for structured output
    """
    if response_model:
        # Native providers pass schema to model
        if self._supports_native_structured():
            json_schema = response_model.model_json_schema()
            # Pass schema to model API
            # (implementation varies by provider)

    # Generate response
    # ...
```

**Native Support Implementations**:

**Ollama** (`ollama_provider.py`):
```python
generation_kwargs["format"] = response_model.model_json_schema()
```

**LMStudio** (`lmstudio_provider.py`):
```python
generation_kwargs["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": response_model.__name__,
        "schema": response_model.model_json_schema()
    }
}
```

**HuggingFace GGUF** (`huggingface_provider.py`):
```python
generation_kwargs["response_format"] = {
    "type": "json_schema",
    "json_schema": {
        "name": response_model.__name__,
        "schema": json_schema
    }
}
```

**HuggingFace Transformers with Outlines** (`huggingface_provider.py`):
```python
import outlines
outlines_model = outlines.from_transformers(self.llm, self.tokenizer)
generator = outlines.json_schema(outlines_model, json_schema)
response = generator(prompt)
```

**MLX with Outlines** (`mlx_provider.py`):
```python
import outlines
outlines_model = outlines.from_mlxlm(self.llm, self.tokenizer)
generator = outlines.json_schema(outlines_model, json_schema)
response = generator(prompt)
```

### With BaseLLM

The `BaseLLM` class exposes structured outputs through its public API:

```python
class BaseLLM:
    def generate(
        self,
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ):
        """
        Generate with optional structured output.

        If response_model is provided, uses StructuredOutputHandler
        to ensure schema compliance.
        """
        if response_model:
            handler = StructuredOutputHandler()
            return handler.generate_structured(
                provider=self,
                prompt=prompt,
                response_model=response_model,
                **kwargs
            )
        else:
            # Regular generation
            return self._generate_internal(prompt, **kwargs)
```

### With Events System

The handler emits events for monitoring and debugging:

**Event Types**:
- `VALIDATION_FAILED` - Validation error during prompted generation
- `ERROR` - Fatal error during generation

**Event Data**:
```python
{
    "response_model": "ClassName",
    "validation_attempt": 1,
    "validation_error": "Missing required field: 'name'",
    "error_type": "ValidationError",
    "response_length": 150,
    "success": False
}
```

**Subscribe to Events**:
```python
from abstractcore.events import subscribe, EventType

def handle_validation_failure(event):
    print(f"Validation failed: {event['data']['validation_error']}")
    print(f"Attempt: {event['data']['validation_attempt']}")

subscribe(EventType.VALIDATION_FAILED, handle_validation_failure)
```

## Best Practices

### Schema Design

**DO**:
- ✅ Use clear, descriptive field names
- ✅ Add field descriptions for complex schemas
- ✅ Use `Optional` for fields that may be missing
- ✅ Define enums for categorical data
- ✅ Use nested models for logical grouping
- ✅ Add custom validators for business logic
- ✅ Use type hints for all fields

**DON'T**:
- ❌ Use overly complex nested structures (>3 levels)
- ❌ Define ambiguous field names
- ❌ Omit type hints
- ❌ Use generic names like "data", "info", "value"
- ❌ Create circular references
- ❌ Mix unrelated data in single model

**Example**:

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

# GOOD: Clear, well-structured schema
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str = Field(description="Task title or summary")
    description: str = Field(description="Detailed task description")
    priority: Priority = Field(description="Task priority level")
    assignee: Optional[str] = Field(None, description="Person assigned to task")
    estimated_hours: Optional[int] = Field(None, description="Estimated hours to complete")

# BAD: Ambiguous, poorly structured
class Data(BaseModel):
    info: str
    values: list
    status: str
```

### Temperature Configuration

**Recommended Settings**:

```python
# Deterministic structured outputs (recommended)
llm.generate(
    prompt="...",
    response_model=MyModel,
    temperature=0  # No randomness
)

# Creative but structured
llm.generate(
    prompt="...",
    response_model=MyModel,
    temperature=0.3  # Slight creativity
)

# Avoid high temperature for structured outputs
# temperature > 0.7 may produce invalid JSON
```

### Retry Strategy Selection

**Conservative** (2 attempts):
- Use for: Fast responses, simple schemas
- Benefit: Minimal latency

```python
retry = FeedbackRetry(max_attempts=2)
```

**Standard** (3 attempts - default):
- Use for: Most use cases
- Benefit: Good balance of reliability and speed

```python
retry = FeedbackRetry(max_attempts=3)
```

**Aggressive** (5 attempts):
- Use for: Critical data extraction, complex schemas
- Benefit: Maximum reliability

```python
retry = FeedbackRetry(max_attempts=5)
```

### Error Handling

**Always catch validation errors**:

```python
from pydantic import ValidationError

try:
    response = llm.generate(
        prompt="...",
        response_model=MyModel
    )
except ValidationError as e:
    # Log detailed error information
    for error in e.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        print(f"Field '{field}': {error['msg']}")

    # Implement fallback behavior
    response = use_default_values()
```

### Prompt Engineering for Structured Outputs

**Effective prompts**:

```python
# GOOD: Clear extraction instructions
prompt = """
Extract the following information about the person:
Name: John Doe
Age: 35
Email: john@example.com
"""

# BETTER: Structured input format
prompt = """
Parse this JSON-like data:
{
  "person": {
    "name": "John Doe",
    "age": 35,
    "email": "john@example.com"
  }
}
"""

# BEST: Natural language with clear context
prompt = """
Analyze this user profile and extract key information:

User Profile:
- Full Name: John Doe
- Age: 35 years old
- Contact: john@example.com
- Location: San Francisco, CA

Extract the name, age, and email address.
"""
```

### Testing Structured Outputs

```python
import pytest
from pydantic import BaseModel, ValidationError

class TestStructuredOutput:
    """Test structured output generation."""

    def test_simple_extraction(self, llm):
        """Test basic field extraction."""
        class Person(BaseModel):
            name: str
            age: int

        response = llm.generate(
            prompt="Extract: John Doe, 35 years old",
            response_model=Person,
            temperature=0
        )

        assert response.name == "John Doe"
        assert response.age == 35

    def test_validation_enforcement(self, llm):
        """Test that validation is enforced."""
        class PositiveInt(BaseModel):
            value: int

            @field_validator('value')
            @classmethod
            def must_be_positive(cls, v):
                if v <= 0:
                    raise ValueError('must be positive')
                return v

        # Should raise ValidationError after retries
        with pytest.raises(ValidationError):
            llm.generate(
                prompt="Extract: value is -5",
                response_model=PositiveInt,
                temperature=0
            )
```

## Common Pitfalls

### 1. Validation Errors After All Retries

**Problem**: LLM consistently fails to produce valid output.

**Causes**:
- Schema too complex for model capability
- Ambiguous prompt or field names
- Model hallucinating invalid values
- Enum values unclear to LLM

**Solutions**:

```python
# Simplify schema
class SimpleModel(BaseModel):
    # Instead of complex nested structure
    name: str
    category: str  # Instead of enum if failing

# Add clear descriptions
class BetterModel(BaseModel):
    name: str = Field(description="Full name of the person")
    status: Status = Field(description="Must be one of: active, inactive, pending")

# Improve prompt
prompt = """
Extract person information. The status must be exactly one of these values:
- active
- inactive
- pending

Person: John Doe, status is active
"""
```

### 2. JSON Parsing Failures

**Problem**: Response contains valid JSON but extraction fails.

**Causes**:
- LLM wraps JSON in markdown code blocks
- Additional text before/after JSON
- Malformed JSON with syntax errors

**Solutions**:

The handler already implements multi-strategy extraction:
1. Direct parsing
2. Code block extraction
3. Pattern matching
4. Automatic JSON repair via `fix_json()`

If issues persist:

```python
# Check raw response
response = llm.generate(prompt="...", temperature=0)
print(response.content)  # Inspect raw output

# Adjust prompt to be more explicit
prompt = """
{your instructions}

IMPORTANT: Return ONLY the JSON object with no additional text,
explanations, or markdown formatting.
"""
```

### 3. Enum Validation Failures

**Problem**: LLM returns Python enum notation instead of values.

**Example**:
```json
{"status": "Status.ACTIVE"}  // Wrong
{"status": "active"}          // Correct
```

**Solutions**:

The handler automatically:
- Simplifies enum schemas for prompted providers
- Adds clear enum value descriptions
- Preprocesses responses to convert enum notation

Manual fix if needed:

```python
from enum import Enum

class Status(str, Enum):
    ACTIVE = "active"      # Use clear string values
    INACTIVE = "inactive"
    PENDING = "pending"

# In prompt, be explicit
prompt = """
Extract status. Use exact string values:
- "active" (not Status.ACTIVE)
- "inactive" (not Status.INACTIVE)
- "pending" (not Status.PENDING)
"""
```

### 4. Slow Performance with Native Support

**Problem**: Native structured outputs slower than expected.

**Cause**: Constrained generation has per-token overhead.

**Performance Expectations**:
- **Simple schemas**: May be 2-5x slower than prompted
- **Medium schemas**: Comparable to prompted
- **Complex schemas**: Faster than prompted (fewer retries)

**Solutions**:

```python
# Use prompted for simple schemas
from abstractcore.structured import StructuredOutputHandler

handler = StructuredOutputHandler()

# Force prompted approach for simple schemas
if is_simple_schema(MyModel):
    # Temporarily disable native support detection
    # (not officially supported, but can monkeypatch for testing)
    result = handler._generate_prompted(
        provider=llm,
        prompt=prompt,
        response_model=MyModel
    )
```

### 5. Missing Required Fields

**Problem**: LLM omits required fields.

**Causes**:
- Field not present in input data
- Field name ambiguous
- Model capability insufficient

**Solutions**:

```python
# Make fields optional if data may be missing
from typing import Optional

class Person(BaseModel):
    name: str  # Always required
    age: Optional[int] = None  # May be missing
    email: Optional[str] = None  # May be missing

# Add default values
class Config(BaseModel):
    enabled: bool = True  # Defaults to True if not specified
    timeout: int = 30      # Defaults to 30 if not specified

# Improve prompt clarity
prompt = """
Extract all fields (provide null if not available):
- name (required)
- age (optional, set to null if not mentioned)
- email (optional, set to null if not mentioned)

Data: John Doe is 35 years old
"""
```

### 6. Type Coercion Issues

**Problem**: Values have wrong types (e.g., string instead of int).

**Cause**: Pydantic's strict validation vs. coercion modes.

**Solutions**:

```python
from pydantic import BaseModel, Field, field_validator

# Option 1: Enable coercion (Pydantic default)
class CoerciveModel(BaseModel):
    age: int  # "35" will be coerced to 35

# Option 2: Add custom validator
class ValidatedModel(BaseModel):
    age: int

    @field_validator('age', mode='before')
    @classmethod
    def coerce_age(cls, v):
        if isinstance(v, str):
            return int(v)
        return v

# Option 3: Use strict mode (raises error on type mismatch)
class StrictModel(BaseModel):
    model_config = {"strict": True}
    age: int  # "35" will fail validation
```

## Testing Strategy

### Unit Tests for Retry Logic

```python
import pytest
from pydantic import BaseModel, ValidationError
from abstractcore.structured import FeedbackRetry

def test_retry_on_validation_error():
    """Test retry is triggered for ValidationError."""
    retry = FeedbackRetry(max_attempts=3)

    class Model(BaseModel):
        value: int

    try:
        Model(value="not_an_int")
    except ValidationError as e:
        assert retry.should_retry(1, e) is True
        assert retry.should_retry(2, e) is True
        assert retry.should_retry(3, e) is False

def test_no_retry_on_other_errors():
    """Test retry not triggered for non-ValidationError."""
    retry = FeedbackRetry(max_attempts=3)

    error = ValueError("Some error")
    assert retry.should_retry(1, error) is False
```

### Integration Tests with Real Models

```python
import pytest
from pydantic import BaseModel
from abstractcore import create_llm

@pytest.fixture
def llm():
    """Fixture providing test LLM instance."""
    return create_llm("ollama", model="qwen3:4b")

def test_simple_structured_output(llm):
    """Test basic structured output."""
    class Person(BaseModel):
        name: str
        age: int

    response = llm.generate(
        prompt="Extract: Alice, 28 years old",
        response_model=Person,
        temperature=0
    )

    assert response.name == "Alice"
    assert response.age == 28

def test_nested_structured_output(llm):
    """Test nested model structures."""
    from typing import List

    class Task(BaseModel):
        title: str
        priority: int

    class Project(BaseModel):
        name: str
        tasks: List[Task]

    response = llm.generate(
        prompt="""
        Project: Website
        Tasks:
        1. Design (priority 1)
        2. Code (priority 2)
        """,
        response_model=Project,
        temperature=0
    )

    assert response.name == "Website"
    assert len(response.tasks) == 2
    assert response.tasks[0].title == "Design"
    assert response.tasks[0].priority == 1
```

### Testing Native vs Prompted Strategies

```python
def test_native_support_detection(llm):
    """Test native support is correctly detected."""
    from abstractcore.structured import StructuredOutputHandler

    handler = StructuredOutputHandler()

    # Ollama should have native support
    if llm.__class__.__name__ == "OllamaProvider":
        assert handler._has_native_support(llm) is True

    # OpenAI should use prompted approach
    if llm.__class__.__name__ == "OpenAIProvider":
        assert handler._has_native_support(llm) is False
```

### Performance Testing

```python
import time
from pydantic import BaseModel

def test_structured_output_performance(llm):
    """Test structured output performance."""
    class SimpleModel(BaseModel):
        value: int
        label: str

    start = time.time()

    response = llm.generate(
        prompt="Extract: value 42, label 'test'",
        response_model=SimpleModel,
        temperature=0
    )

    duration = time.time() - start

    # Performance assertions
    assert duration < 10.0  # Should complete in < 10 seconds
    assert response.value == 42
    assert response.label == "test"
```

### Testing Error Handling

```python
def test_validation_error_on_invalid_data(llm):
    """Test ValidationError raised after all retries."""
    from pydantic import field_validator

    class PositiveInt(BaseModel):
        value: int

        @field_validator('value')
        @classmethod
        def must_be_positive(cls, v):
            if v <= 0:
                raise ValueError('must be positive')
            return v

    with pytest.raises(ValidationError):
        llm.generate(
            prompt="Extract: value is -10",
            response_model=PositiveInt,
            temperature=0
        )
```

## Public API

### Recommended Imports

```python
# Core structured output functionality
from abstractcore.structured import (
    StructuredOutputHandler,  # Main handler
    FeedbackRetry,            # Retry strategy
    Retry                     # Base retry class
)

# Use via BaseLLM (recommended)
from abstractcore import create_llm
from pydantic import BaseModel

class MyModel(BaseModel):
    field: str

llm = create_llm("ollama", model="qwen3:4b")
response = llm.generate(
    prompt="...",
    response_model=MyModel
)
```

### Direct Handler Usage (Advanced)

```python
from abstractcore.structured import StructuredOutputHandler, FeedbackRetry
from abstractcore import create_llm
from pydantic import BaseModel

# Custom configuration
retry = FeedbackRetry(max_attempts=5)
handler = StructuredOutputHandler(retry_strategy=retry)

# Generate with custom handler
llm = create_llm("openai", model="gpt-4")
response = handler.generate_structured(
    provider=llm,
    prompt="Extract person: John Doe, 35",
    response_model=Person
)
```

### Custom Retry Strategy (Advanced)

```python
from abstractcore.structured import Retry
from pydantic import ValidationError

class CustomRetry(Retry):
    """Custom retry strategy implementation."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Custom retry logic."""
        # Only retry for specific error types
        if isinstance(error, ValidationError):
            return attempt < self.max_attempts
        return False

    def prepare_retry_prompt(self, original_prompt: str, error: Exception, attempt: int) -> str:
        """Custom prompt preparation."""
        return f"{original_prompt}\n\nPrevious attempt failed. Try again."

# Use custom retry
from abstractcore.structured import StructuredOutputHandler

retry = CustomRetry(max_attempts=4)
handler = StructuredOutputHandler(retry_strategy=retry)
```

## Performance Characteristics

### Native Support (from comprehensive testing)

**Test Configuration**: 20 tests across 2 providers, 4 models, 3 complexity levels

**Results**:
- **Success Rate**: 100% (20/20 tests passed)
- **Retry Rate**: 0% (no validation retries needed)
- **Schema Compliance**: Perfect (zero violations)

**Performance by Provider**:
- **Ollama**: Avg 22,828ms, best with gpt-oss:20b (10,170ms)
- **LMStudio**: Avg 31,442ms, best with qwen3-4b (3,623ms)

**Performance by Complexity**:
- **Simple schemas**: 439ms - 8,473ms
- **Medium schemas**: 2,123ms - 146,408ms
- **Complex schemas**: 9,194ms - 163,556ms

**Key Insight**: Native support guarantees schema compliance with zero retries, making it ideal for production use.

### Prompted Approach (from comprehensive testing)

**Test Configuration**: MLX with qwen3-30b, 3 complexity levels

**Results**:
- **Success Rate**: 100% (same as native)
- **Retry Rate**: 0% (zero retries needed)
- **Speed**: 2-5x faster than native for simple schemas

**Performance**:
- **Simple schema**: 745ms (vs 2,031ms native)
- **Medium schema**: 1,945ms (vs 9,904ms native)
- **Complex schema**: 4,193ms (vs 9,840ms native)

**Key Insight**: Prompted approach achieves identical success rate at significantly better performance for quality models.

### Recommendations

**For Simple Schemas** (1-5 fields, no nesting):
- **Preferred**: Prompted approach (2-5x faster)
- **Alternative**: Native if guaranteed compliance required

**For Medium Schemas** (5-15 fields, 1-2 levels nesting):
- **Preferred**: Prompted approach (comparable speed, works everywhere)
- **Alternative**: Native for mission-critical applications

**For Complex Schemas** (15+ fields, 3+ levels nesting):
- **Preferred**: Native support (fewer retries, guaranteed compliance)
- **Alternative**: Prompted with aggressive retry (5 attempts)

**For Production**:
- Use **native support** when available (Ollama, LMStudio, HuggingFace GGUF)
- Use **prompted approach** for universal compatibility
- Both achieve 100% success rate with quality models

---

## Related Documentation

- **Provider Registry**: `/Users/albou/projects/abstractcore/abstractcore/providers/README.md`
- **Core Factory**: `/Users/albou/projects/abstractcore/abstractcore/core/README.md`
- **Events System**: `/Users/albou/projects/abstractcore/abstractcore/events/README.md`
- **Utils**: `/Users/albou/projects/abstractcore/abstractcore/utils/README.md`

## Version Information

This documentation reflects the structured output implementation as of **AbstractCore v2.5.2+**, including:
- Native support for Ollama, LMStudio, HuggingFace GGUF
- Optional Outlines integration for HuggingFace Transformers and MLX
- Comprehensive testing with 100% success rates
- JSON self-healing via `fix_json()`
- Enum simplification for prompted providers
- Event system integration

## Related Modules

**Direct dependencies**:
- [`core/`](../core/README.md) - Response model types and base abstractions
- [`providers/`](../providers/README.md) - Native structured output detection
- [`architectures/`](../architectures/README.md) - Model capability queries
- [`exceptions/`](../exceptions/README.md) - Validation error handling
- [`events/`](../events/README.md) - Retry and validation events
- [`utils/`](../utils/README.md) - Validation utilities, logging

**Used by**:
- [`processing/`](../processing/README.md) - High-level processors use response models
- [`apps/`](../apps/README.md) - Application-level structured outputs
- [`server/`](../server/README.md) - API response schemas

**Related systems**:
- [`tools/`](../tools/README.md) - Tool call structured formats
