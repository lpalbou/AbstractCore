"""
Test retry logging functionality and observability.

This script demonstrates and tests the comprehensive logging system
for structured output with retry behavior.
"""

import logging
import json
import time
from pathlib import Path
from pydantic import BaseModel, field_validator
from abstractllm import create_llm
from abstractllm.utils import configure_logging, get_logger, capture_session
from abstractllm.structured import StructuredOutputHandler, FeedbackRetry

# Clean up any existing logs for clean test
import shutil
if Path("test_logs").exists():
    shutil.rmtree("test_logs")

# Configure logging for testing
configure_logging(
    console_level=logging.INFO,   # Show retry attempts in console
    file_level=logging.DEBUG,     # Detailed logs to file
    log_dir="test_logs",          # Test log directory
    verbatim_enabled=True,        # Capture all interactions
    console_json=False,           # Human-readable console
    file_json=True               # Machine-readable files
)

print("🔍 Testing Retry Logging and Observability")
print("=" * 50)

# Create test models with validation that will likely fail initially
class StrictNumber(BaseModel):
    value: int

    @field_validator('value')
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive integer')
        return v

class ComplexData(BaseModel):
    items: list[str]
    count: int
    average: float

    @field_validator('count')
    @classmethod
    def validate_count(cls, v, info):
        if 'items' in info.data:
            items = info.data['items']
            if v != len(items):
                raise ValueError(f'Count {v} does not match items length {len(items)}')
        return v

    @field_validator('average')
    @classmethod
    def validate_average(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Average must be between 0 and 100')
        return v

# Test 1: Basic retry with validation errors
print("\n📝 Test 1: Basic Retry with Validation Errors")
print("-" * 45)

try:
    llm = create_llm("ollama", model="qwen3-coder:30b")

    # This should trigger retries due to requesting a negative number
    result = llm.generate(
        "Give me a negative number like -5 or -10",
        response_model=StrictNumber
    )
    print(f"✅ Unexpected success: {result}")

except Exception as e:
    print(f"❌ Expected failure after retries: {e}")

# Test 2: Complex validation with detailed logging
print("\n📊 Test 2: Complex Validation with Session Monitoring")
print("-" * 52)

with capture_session("complex_validation_test") as session_logger:
    session_logger.info("Starting complex validation test")

    try:
        # This should trigger retries due to mismatched count
        result = llm.generate(
            "Create data with 5 items but set count to 3",
            response_model=ComplexData
        )
        session_logger.info("Unexpected success", result=str(result))

    except Exception as e:
        session_logger.error("Expected validation failure", error=str(e))

# Test 3: Custom retry strategy with logging
print("\n⚙️  Test 3: Custom Retry Strategy")
print("-" * 32)

class TestRetry(FeedbackRetry):
    def __init__(self):
        super().__init__(max_attempts=2)  # Only 2 attempts for faster testing
        self.logger = get_logger(__name__)

    def prepare_retry_prompt(self, original_prompt: str, error, attempt: int) -> str:
        self.logger.info("Preparing custom retry prompt",
                        attempt=attempt,
                        error_type=type(error).__name__)
        return super().prepare_retry_prompt(original_prompt, error, attempt)

# Use custom retry strategy
handler = StructuredOutputHandler(retry_strategy=TestRetry())

try:
    # Manually use handler to test custom retry
    result = handler.generate_structured(
        provider=llm,
        prompt="Give me a negative number: -100",
        response_model=StrictNumber
    )
    print(f"✅ Unexpected success: {result}")

except Exception as e:
    print(f"❌ Expected failure with custom retry: {e}")

# Test 4: Successful retry to show the complete flow
print("\n🎯 Test 4: Successful Retry Flow")
print("-" * 30)

try:
    # This should succeed after potential retries
    result = llm.generate(
        "Give me a positive integer between 1 and 100",
        response_model=StrictNumber
    )
    print(f"✅ Success: {result}")

except Exception as e:
    print(f"❌ Unexpected failure: {e}")

# Give logs time to flush
time.sleep(1)

# Analyze the logs
print("\n📋 Log Analysis")
print("-" * 15)

log_dir = Path("test_logs")
if log_dir.exists():
    # Find verbatim logs
    verbatim_files = list(log_dir.glob("verbatim_*.jsonl"))

    if verbatim_files:
        latest_verbatim = max(verbatim_files, key=lambda f: f.stat().st_mtime)
        print(f"📄 Reading verbatim log: {latest_verbatim.name}")

        interactions = []
        with open(latest_verbatim) as f:
            for line in f:
                try:
                    interaction = json.loads(line)
                    interactions.append(interaction)
                except json.JSONDecodeError:
                    continue

        print(f"📊 Found {len(interactions)} total interactions")

        # Show retry attempts
        for i, interaction in enumerate(interactions):
            metadata = interaction.get('metadata', {})
            success = metadata.get('success', True)
            prompt_preview = interaction.get('prompt', '')[:50] + "..."
            response_preview = interaction.get('response', '')[:50] + "..."

            print(f"   Interaction {i+1}: {'✅' if success else '❌'}")
            print(f"     Prompt: {prompt_preview}")
            print(f"     Response: {response_preview}")
            if not success:
                print(f"     Error: {metadata.get('error', 'Unknown')}")
            print()

    else:
        print("❌ No verbatim logs found")

    # Check for structured log files
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        print(f"📊 Found {len(log_files)} structured log files")
        # Could parse these for structured log analysis
    else:
        print("❌ No structured log files found")

else:
    print("❌ No log directory found")

print("\n✨ Observability Test Complete!")
print("=" * 35)
print("📁 Check 'test_logs/' directory for:")
print("   • verbatim_*.jsonl - Complete prompt/response pairs")
print("   • *.log - Structured log entries with retry details")
print("   • All retry attempts, validation errors, and timing data")