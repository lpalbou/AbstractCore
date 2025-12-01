# TACTICAL-002: Structured Logging Standardization

**Status**: ✅ COMPLETED (Phase 1)
**Priority**: P2 - Medium
**Effort Estimated**: 6-12 hours (original proposal)
**Effort Actual**: 2 hours (Phase 1 simplified approach)
**Type**: Code Quality / Observability
**Completed Version**: 2.5.4 (Patch Release)
**Completion Date**: 2025-12-01

---

## ✅ COMPLETION SUMMARY

**Phase 1 Completed**: Basic structured logging migration
- **14 files successfully migrated** to `get_logger()`
- **Zero breaking changes**
- **All tests passing**
- **Production ready**

**Scope Decision**: User chose **Phase 1 only** (simple migration)
- ✅ Phase 1: Standardize on structured logging (COMPLETED)
- ⏭️ Phase 2: Trace ID propagation (SKIPPED - not needed)
- ⏭️ Phase 3: Enhanced context binding (SKIPPED - not needed)

**Files Migrated**:
1. abstractcore/tools/common_tools.py
2. abstractcore/tools/handler.py
3. abstractcore/tools/parser.py
4. abstractcore/tools/registry.py
5. abstractcore/tools/syntax_rewriter.py
6. abstractcore/tools/tag_rewriter.py
7. abstractcore/architectures/detection.py
8. abstractcore/core/retry.py
9. abstractcore/embeddings/manager.py
10. abstractcore/media/processors/office_processor.py
11. abstractcore/media/utils/image_scaler.py
12. abstractcore/media/vision_fallback.py
13. abstractcore/providers/streaming.py
14. abstractcore/utils/self_fixes.py

**Correctly Skipped**: abstractcore/utils/cli.py (local scope usage only)

**Results**:
- ✅ 100% success rate (14/14 files migrated)
- ✅ Zero `logging.getLogger(__name__)` in migrated files
- ✅ SOTA best practices followed (PEP 282, cloud-native, Django patterns)
- ✅ No over-engineering (simple import replacement only)
- ✅ All modules import successfully
- ✅ All tests passing

**Why Simplified**:
- Original proposal was over-engineered for actual needs
- Phase 1 migration sufficient for improved observability
- trace_id propagation not needed (interaction tracing already provides this)
- Context binding already works in existing structured_logging.py
- Simple approach = 2 hours vs 6-12 hours (5-6x more efficient)

---

## ORIGINAL PROPOSAL BELOW

_(Note: The implementation was simplified based on user requirements. Only Phase 1 was completed.)_

---

## Executive Summary

AbstractCore has a sophisticated structured logging system (`utils/structured_logging.py`) but adoption is inconsistent across the codebase. Some modules use the SOTA `get_logger()` from structured_logging, while others use standard `logging.getLogger()`. This proposal standardizes logging across all modules and adds correlation IDs for complete request tracing.

**Expected Benefits**:
- Consistent structured logs across all components
- Better production debugging with correlated traces
- Machine-readable logs for ELK/Datadog/Splunk
- Enhanced observability with context propagation
- Integration with interaction tracing

---

## Problem Statement

### Current State: Mixed Logging Approaches

**Pattern 1: Structured Logging** (✅ Correct, but inconsistent adoption):
```python
# abstractcore/providers/base.py:46
from ..utils.structured_logging import get_logger

class BaseProvider:
    def __init__(self, ...):
        self.logger = get_logger(self.__class__.__name__)  # ✅ SOTA approach
```

**Pattern 2: Standard Logging** (❌ Inconsistent, 6+ modules):
```python
# abstractcore/tools/common_tools.py:6
import logging
logger = logging.getLogger(__name__)  # ❌ Standard logging

# abstractcore/tools/syntax_rewriter.py
# abstractcore/tools/handler.py
# abstractcore/tools/registry.py
# abstractcore/tools/parser.py
# abstractcore/tools/tag_rewriter.py
# abstractcore/embeddings/manager.py
# ... and more
```

**Evidence** (from grep analysis):
```bash
$ grep -r "import logging\|from logging\|getLogger" abstractcore/ | head -20
abstractcore/tools/common_tools.py:import logging
abstractcore/tools/common_tools.py:logger = logging.getLogger(__name__)
abstractcore/tools/syntax_rewriter.py:import logging
abstractcore/tools/syntax_rewriter.py:logger = logging.getLogger(__name__)
# ... 10+ more files with standard logging
```

### Issues with Current Approach

1. **Inconsistent Log Format**: Some logs structured, others plain text
   ```
   # From structured_logging modules:
   {"timestamp": "2025-11-25T10:30:00", "level": "INFO", "message": "Generation completed", "tokens": 150, "latency_ms": 1234.5}

   # From standard logging modules:
   2025-11-25 10:30:00 [INFO] Generation completed
   ```

2. **No Context Propagation**: Standard logging doesn't carry context
   - Can't correlate logs across components
   - No request ID tracking
   - No session ID in logs
   - Can't trace full request lifecycle

3. **Limited Machine Readability**: Standard logs hard to parse
   - ELK/Splunk ingestion requires grok patterns
   - No structured fields for filtering
   - Manual log parsing required

4. **No Trace Correlation**: Logs don't correlate with interaction traces
   - Interaction tracing has `trace_id`
   - Logs don't include trace_id
   - Can't correlate traces with logs

### Impact

**Development**: Harder to debug complex flows
**Production**: More difficult log analysis and monitoring
**Observability**: Incomplete tracing picture
**Cost**: Higher log processing costs (unstructured data)

---

## Proposed Solution

### Phase 1: Standardize on Structured Logging

**Migration Pattern**:

```python
# BEFORE (standard logging)
import logging
logger = logging.getLogger(__name__)

logger.info(f"Processing request with {len(items)} items")

# AFTER (structured logging)
from abstractcore.utils.structured_logging import get_logger

logger = get_logger(__name__)

logger.info("Processing request", item_count=len(items), component="processor")
```

**Benefits**:
- Consistent log format
- Structured fields automatically
- Context propagation support
- Better aggregation in log systems

### Phase 2: Add Trace ID Propagation

**Enhancement to StructuredLogger**:

```python
# abstractcore/utils/structured_logging.py

import contextvars

# Context variable for trace ID propagation
trace_id_var = contextvars.ContextVar('trace_id', default=None)

class StructuredLogger:
    """Enhanced logger with trace correlation."""

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with trace correlation."""
        # Merge context
        log_data = {**self._context, **kwargs}

        # Auto-inject trace_id if available
        trace_id = trace_id_var.get()
        if trace_id:
            log_data['trace_id'] = trace_id

        # ... existing logging code ...

# Helper functions for trace management
def set_trace_id(trace_id: str):
    """Set trace ID for current context."""
    trace_id_var.set(trace_id)

def get_trace_id() -> Optional[str]:
    """Get current trace ID."""
    return trace_id_var.get()

@contextmanager
def trace_context(trace_id: Optional[str] = None):
    """Context manager for trace ID."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())

    token = trace_id_var.set(trace_id)
    try:
        yield trace_id
    finally:
        trace_id_var.reset(token)
```

**Integration with BaseProvider**:

```python
# abstractcore/providers/base.py

class BaseProvider:
    def _capture_trace(self, ...):
        """Capture trace and set logging context."""
        trace_id = str(uuid.uuid4())

        # Set trace ID for logging context
        from ..utils.structured_logging import set_trace_id
        set_trace_id(trace_id)

        # ... existing trace capture code ...

        return trace_id

    async def agenerate(self, ...):
        """Async generation with trace context."""
        from ..utils.structured_logging import trace_context

        with trace_context() as trace_id:
            # All logs in this context include trace_id
            response = await self._agenerate_internal(prompt, **kwargs)
            return response
```

### Phase 3: Enhanced Context Binding

```python
# Usage example with rich context

logger = get_logger(__name__)

# Bind persistent context
request_logger = logger.bind(
    user_id="user_123",
    session_id="session_456",
    provider="openai",
    model="gpt-4o-mini"
)

# All subsequent logs include bound context
request_logger.info("Starting generation", prompt_length=150)
# Output: {"level": "INFO", "message": "Starting generation",
#          "user_id": "user_123", "session_id": "session_456",
#          "provider": "openai", "model": "gpt-4o-mini",
#          "prompt_length": 150, "trace_id": "abc-123-def"}
```

---

## Implementation Plan

### Step 1: Identify All Modules Using Standard Logging (1 hour)

```bash
# Create audit report
grep -r "import logging" abstractcore/ --include="*.py" | \
    grep -v "structured_logging" | \
    cut -d: -f1 | \
    sort -u > logging_audit.txt

# Expected ~15-20 files to migrate
```

### Step 2: Create Migration Script (2 hours)

```python
# scripts/migrate_to_structured_logging.py

import re
from pathlib import Path

def migrate_file(file_path: Path):
    """Migrate a file to structured logging."""
    content = file_path.read_text()

    # Replace import
    content = re.sub(
        r'import logging\nlogger = logging\.getLogger\(__name__\)',
        r'from abstractcore.utils.structured_logging import get_logger\nlogger = get_logger(__name__)',
        content
    )

    # Replace f-string logging with structured logging
    # Before: logger.info(f"Processing {count} items")
    # After: logger.info("Processing items", item_count=count)

    # This requires manual review for accuracy
    # Flag candidates for manual review:
    candidates = re.findall(r'logger\.\w+\(f["\'].*?\)', content)
    if candidates:
        print(f"{file_path}: {len(candidates)} f-string logs need review")

    file_path.write_text(content)

# Run on all files
for file_path in Path("abstractcore").rglob("*.py"):
    if "logging.getLogger" in file_path.read_text():
        migrate_file(file_path)
```

### Step 3: Migrate Modules (4-6 hours)

**Priority order** (by usage frequency):

1. **High Priority** (core infrastructure):
   - `tools/common_tools.py`
   - `tools/handler.py`
   - `embeddings/manager.py`

2. **Medium Priority** (frequently used):
   - `tools/syntax_rewriter.py`
   - `tools/registry.py`
   - `tools/parser.py`

3. **Low Priority** (less critical):
   - `tools/tag_rewriter.py`
   - Other utility modules

**Migration process per file**:
1. Run migration script
2. Manual review of logging statements
3. Convert f-strings to structured parameters
4. Test module functionality
5. Verify logs output correctly

### Step 4: Add Trace ID Propagation (2-3 hours)

1. Update `structured_logging.py` with contextvars
2. Add `set_trace_id()`, `get_trace_id()`, `trace_context()`
3. Update `BaseProvider._capture_trace()` to set context
4. Update `BaseProvider.generate()` to use trace context
5. Test trace ID appears in logs

### Step 5: Testing & Verification (2-3 hours)

```python
# tests/logging/test_structured_logging_standard.py

def test_all_modules_use_structured_logging():
    """Verify all modules use structured logging."""
    import subprocess

    # Find files still using standard logging
    result = subprocess.run([
        "grep", "-r", "import logging",
        "abstractcore/", "--include=*.py"
    ], capture_output=True, text=True)

    violations = [
        line for line in result.stdout.split('\n')
        if line and 'structured_logging' not in line
    ]

    assert len(violations) == 0, f"Files still using standard logging: {violations}"

def test_trace_id_propagation():
    """Verify trace_id appears in logs."""
    from abstractcore.utils.structured_logging import trace_context, get_logger

    logger = get_logger("test")

    with trace_context("test-trace-123") as trace_id:
        # Capture log output
        with capture_logs() as logs:
            logger.info("Test message", test_field="value")

        # Verify trace_id in logs
        assert trace_id in str(logs)
        assert "test-trace-123" in str(logs)

def test_context_binding():
    """Verify context binding works correctly."""
    from abstractcore.utils.structured_logging import get_logger

    logger = get_logger("test")
    bound_logger = logger.bind(user_id="user_123", session_id="session_456")

    with capture_logs() as logs:
        bound_logger.info("Bound test")

    # Verify bound context in logs
    assert "user_123" in str(logs)
    assert "session_456" in str(logs)
```

**Total Estimated Time**: 11-15 hours

---

## Testing & Verification

### Migration Verification

```bash
# 1. Verify no standard logging remains
grep -r "import logging" abstractcore/ --include="*.py" | \
    grep -v "structured_logging" | \
    wc -l
# Expected: 0

# 2. Verify all modules import structured logging
grep -r "from.*structured_logging import get_logger" abstractcore/ | \
    wc -l
# Expected: ~15-20 modules

# 3. Test log output format
python -c "
from abstractcore import create_llm
from abstractcore.utils.structured_logging import configure_logging, trace_context
import logging

# Configure JSON logging
configure_logging(console_level=logging.INFO, console_json=True)

llm = create_llm('ollama', model='qwen3:4b')

with trace_context('test-trace-123'):
    response = llm.generate('Test')

# Verify JSON output with trace_id
"
# Expected: JSON logs with trace_id field
```

### Functional Tests

```python
# tests/logging/test_correlation.py

def test_trace_correlation_with_interaction_tracing():
    """Verify logs correlate with interaction traces."""
    from abstractcore import create_llm
    from abstractcore.utils.structured_logging import get_trace_id

    llm = create_llm('ollama', model='qwen3:4b', enable_tracing=True)

    # Capture logs and trace
    with capture_logs() as logs:
        response = llm.generate('Test')
        trace_id = response.metadata['trace_id']

    # Verify same trace_id in logs
    assert trace_id in str(logs)

    # Get interaction trace
    trace = llm.get_traces(trace_id=trace_id)
    assert trace['trace_id'] == trace_id
```

---

## Success Criteria

1. **Zero Standard Logging**: All modules use structured logging
2. **Trace Correlation**: 100% of logs include trace_id when available
3. **Context Propagation**: Bound context appears in all child logs
4. **Backwards Compatibility**: No breaking changes
5. **Performance**: No measurable performance impact (<1% overhead)
6. **Test Coverage**: 100% of new logging code covered

---

## Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Log format breaking downstream tools | Low | Medium | Gradual rollout, compatibility mode |
| Performance overhead | Very Low | Low | Benchmark tests, optimize if needed |
| Incomplete migration | Medium | Medium | Automated verification tests |
| Context leaks | Low | Medium | Proper context manager usage |

---

## Dependencies

**Code Dependencies**:
- structlog (optional, already in use)
- contextvars (built-in Python 3.7+)

**Feature Dependencies**:
- Interaction tracing (for trace_id integration)

---

## Backwards Compatibility

**Breaking Changes**: None

**Log Format**: Configurable (JSON or text)

**Migration**: Transparent to users

---

## Rollout Plan

1. **Phase 1**: Core infrastructure modules (3-4 hours)
2. **Phase 2**: Remaining modules (3-4 hours)
3. **Phase 3**: Add trace propagation (2-3 hours)
4. **Phase 4**: Testing and verification (2-3 hours)
5. **Phase 5**: Documentation updates (1 hour)

---

## Monitoring & Metrics

### Key Metrics

```python
# Log statistics
{
  "logs_per_second": 150,
  "structured_percentage": 100,  # Goal: 100%
  "trace_correlation_rate": 95,  # % of logs with trace_id
  "context_binding_usage": 45    # % of logs with bound context
}
```

### Alerts

- Alert if structured logging percentage drops below 95%
- Alert if trace correlation rate drops below 80%
- Monitor log volume for unexpected spikes

---

## Follow-up Actions

After implementation:

1. **ELK Integration**: Document structured log ingestion
2. **Datadog Dashboard**: Create observability dashboard
3. **Performance Profiling**: Verify minimal overhead
4. **Advanced Features**: Consider adding:
   - Request ID propagation
   - Span IDs for distributed tracing
   - Log sampling for high-volume scenarios
5. **Documentation**: Update logging guide with best practices

---

## References

- Current implementation: `abstractcore/utils/structured_logging.py`
- Interaction tracing: `docs/interaction-tracing.md`
- Python contextvars: https://docs.python.org/3/library/contextvars.html
- structlog docs: https://www.structlog.org/

---

**Document Version**: 1.0
**Created**: 2025-11-25
**Author**: Expert Code Review
**Status**: Ready for Implementation
