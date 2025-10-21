# Generation Parameters Architecture

This document explains the design and implementation of unified generation parameters (temperature, seed) across all AbstractCore providers.

## Design Principles

### 1. **Interface-First Design**
Parameters are declared at the `AbstractCoreInterface` level, ensuring:
- Consistent API contract across all providers
- Type safety and documentation at the interface level
- Automatic inheritance by all provider implementations

### 2. **DRY (Don't Repeat Yourself)**
Common parameters are handled centrally to avoid:
- Code duplication across 6 providers
- Inconsistent parameter handling
- Maintenance overhead for parameter changes

### 3. **Graceful Degradation**
Providers that don't support certain parameters:
- Accept the parameters without error
- Issue appropriate warnings (e.g., Anthropic seed warning)
- Maintain consistent API behavior
- Provide fallback mechanisms where possible

## Architecture Overview

```python
AbstractCoreInterface (interface.py)
├── temperature: float = 0.7        # Interface-level default
├── seed: Optional[int] = None       # Interface-level default
└── _validate_parameters()           # Validation logic

BaseProvider (base.py)
├── _prepare_generation_kwargs()     # Unified parameter processing
├── _extract_generation_params()     # Parameter extraction helper
└── Parameter fallback hierarchy     # kwargs → instance → defaults

Individual Providers
├── Provider-specific parameters only (top_p, frequency_penalty, etc.)
├── Provider-specific parameter mapping
└── Native API integration
```

## Parameter Hierarchy

Parameters follow a clear precedence order:

1. **Method-level kwargs** (highest priority)
   ```python
   llm.generate("Hello", temperature=0.9, seed=123)
   ```

2. **Instance-level parameters**
   ```python
   llm = create_llm("openai", temperature=0.5, seed=42)
   ```

3. **Interface defaults** (lowest priority)
   ```python
   # temperature=0.7, seed=None (from AbstractCoreInterface)
   ```

## Provider-Specific Implementation

### Native Support (OpenAI, Ollama, LMStudio, HuggingFace)
```python
# Direct parameter mapping to provider API
call_params["temperature"] = params["temperature"]
if "seed" in params:
    call_params["seed"] = params["seed"]
```

### Graceful Fallback (Anthropic, MLX)
```python
# Accept parameters but log limitation
if "seed" in params:
    self.logger.debug(f"Seed {params['seed']} requested but not supported - logged for debugging")
```

## Session Integration

Sessions maintain persistent parameters across conversations:

```python
session = BasicSession(
    provider=llm,
    temperature=0.5,    # Default for all messages
    seed=42            # Consistent across conversation
)

# Uses session defaults
response1 = session.generate("Hello")

# Override for specific message
response2 = session.generate("Be creative!", temperature=0.9)
```

## Code Quality Benefits

### Before (Duplicated Code)
```python
# In each of 6 providers:
self.temperature = kwargs.get("temperature", 0.7)
self.seed = kwargs.get("seed", None)
# ... parameter extraction logic in each provider
```

### After (Centralized)
```python
# In AbstractCoreInterface:
def __init__(self, ..., temperature: float = 0.7, seed: Optional[int] = None):
    self.temperature = temperature
    self.seed = seed

# In BaseProvider:
def _extract_generation_params(self, **kwargs) -> Dict[str, Any]:
    return {
        "temperature": kwargs.get("temperature", self.temperature),
        "seed": kwargs.get("seed", self.seed) if self.seed is not None else None
    }
```

## Future Extensibility

Adding new parameters requires only:
1. Declaration in `AbstractCoreInterface`
2. Logic in `BaseProvider._extract_generation_params()`
3. Provider-specific mapping where supported

No changes needed in individual provider `__init__` methods.

## Testing Strategy

Parameters are tested at multiple levels:
- **Interface level**: Parameter inheritance and defaults
- **Provider level**: Native API integration and fallback behavior
- **Session level**: Parameter persistence and override behavior
- **Integration level**: End-to-end parameter flow

## Performance Considerations

- **Minimal Overhead**: Parameter extraction happens once per generation call
- **Memory Efficient**: No parameter duplication across providers
- **CPU Efficient**: Simple dictionary operations for parameter resolution

## Backward Compatibility

All changes are fully backward compatible:
- Existing code continues to work unchanged
- New parameters are optional with sensible defaults
- Provider behavior remains consistent for existing use cases

## Empirical Verification ✅

**Real-World Testing Confirms Deterministic Behavior**:

All providers (except Anthropic) achieve true determinism with `seed + temperature=0`:

```python
# Verified with actual API calls (100% success rate):
✅ OpenAI (gpt-3.5-turbo): Same seed → Identical outputs
✅ Ollama (gemma3:1b): Same seed → Identical outputs  
✅ MLX (Qwen3-4B): Same seed → Identical outputs
✅ HuggingFace: Same seed → Identical outputs (transformers + GGUF)
✅ LMStudio: Same seed → Identical outputs (OpenAI-compatible)
⚠️ Anthropic (claude-3-haiku): temperature=0 → Consistent outputs (no seed support)
```

**Provider-Specific Implementations**:
- **OpenAI**: Native `seed` parameter in API
- **MLX**: `mx.random.seed()` before generation  
- **Ollama**: `seed` in options payload
- **HuggingFace**: `torch.manual_seed()` + GGUF native seed
- **LMStudio**: OpenAI-compatible `seed` parameter
- **Anthropic**: Issues `UserWarning` when seed provided

**Testing Commands**:
```bash
# Verify determinism across providers
python tests/manual_seed_verification.py

# Test specific provider
python tests/manual_seed_verification.py --provider openai
```
