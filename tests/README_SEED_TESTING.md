# SEED and Temperature Determinism Testing

This directory contains comprehensive tests for SEED and temperature parameter functionality across all AbstractCore providers.

## Test Files

### 1. `test_seed_temperature_basic.py`
**Purpose**: Basic parameter handling tests that run in CI/CD environments.

**What it tests**:
- Parameter inheritance from interface to providers
- Default parameter values
- Parameter override behavior
- Session-level parameter persistence
- Parameter fallback hierarchy (kwargs → instance → defaults)

**Usage**:
```bash
# Run all basic tests
pytest tests/test_seed_temperature_basic.py -v

# Run specific test class
pytest tests/test_seed_temperature_basic.py::TestSeedTemperatureParameters -v
```

### 2. `test_seed_determinism.py`
**Purpose**: Comprehensive determinism testing across all providers.

**What it tests**:
- Same seed produces identical outputs (determinism)
- Different seeds produce different outputs (variation)
- Session-level seed persistence
- Provider-specific behavior (native support vs. fallback)

**Usage**:
```bash
# Run determinism tests (requires provider setup)
pytest tests/test_seed_determinism.py -v

# Run specific provider test
pytest tests/test_seed_determinism.py::test_openai_seed_determinism -v
```

### 3. `manual_seed_verification.py`
**Purpose**: Interactive script for manual testing and verification.

**What it does**:
- Tests actual determinism with real providers
- Provides visual feedback and detailed analysis
- Can test specific providers or all available ones
- Generates comprehensive reports

**Usage**:
```bash
# Test all available providers
python tests/manual_seed_verification.py

# Test specific provider
python tests/manual_seed_verification.py --provider openai

# Use custom prompt
python tests/manual_seed_verification.py --prompt "Count to 10"
```

## Provider Support Matrix

| Provider | Native SEED Support | Temperature Support | Notes |
|----------|-------------------|-------------------|-------|
| OpenAI | ✅ Yes | ✅ Yes | Except o1 reasoning models |
| Anthropic | ❌ No | ✅ Yes | **Warns when seed provided** |
| HuggingFace | ✅ Yes | ✅ Yes | Both transformers & GGUF |
| Ollama | ✅ Yes | ✅ Yes | Native support |
| LMStudio | ✅ Yes | ✅ Yes | OpenAI-compatible |
| MLX | ✅ Yes | ✅ Yes | **Via mx.random.seed()** |

## Expected Behavior

### ✅ Deterministic Providers (OpenAI, HuggingFace, Ollama, LMStudio, MLX)
- Same seed + temperature=0 → **Identical outputs**
- Different seeds → **Different outputs**
- Session persistence → **Consistent across conversation**

### ⚠️ Limited Support Providers (Anthropic)
- Same seed + temperature=0 → **May vary** (no native seed support)
- Different seeds → **May not vary** (seed ignored with warning)
- Session persistence → **Works for temperature only**
- Seed parameter → **Issues UserWarning when provided**

## Test Environment Setup

### Prerequisites
```bash
# Install test dependencies
pip install pytest

# Optional: Set up provider API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Optional: Start local providers
ollama serve  # For Ollama tests
# Start LMStudio server for LMStudio tests
```

### Running Tests

#### CI/CD Environment (Basic Tests Only)
```bash
# These tests don't require external services
pytest tests/test_seed_temperature_basic.py
```

#### Development Environment (Full Tests)
```bash
# Run all seed-related tests
pytest tests/test_seed_temperature_basic.py tests/test_seed_determinism.py -v

# Manual verification
python tests/manual_seed_verification.py
```

## Understanding Test Results

### ✅ Perfect Determinism
```
Call 1: 'Python programming language'
Call 2: 'Python programming language'
Call 3: 'Python programming language'
Deterministic: ✅ YES
```

### ⚠️ Partial Determinism
```
Call 1: 'Python programming language'
Call 2: 'Python programming language'
Call 3: 'Python coding language'
Deterministic: ❌ NO (2/3 identical)
```

### ❌ No Determinism
```
Call 1: 'Python programming language'
Call 2: 'JavaScript web development'
Call 3: 'Machine learning algorithms'
Deterministic: ❌ NO (all different)
```

## Troubleshooting

### Common Issues

1. **"Provider not available"**
   - Check API keys are set
   - Verify provider services are running (Ollama, LMStudio)
   - Check network connectivity

2. **"No determinism detected"**
   - Verify provider supports seed parameter
   - Check temperature is set to 0.0
   - Some models may have inherent randomness

3. **"Tests skipped"**
   - Missing API keys (expected for CI/CD)
   - Provider services not running (expected)
   - This is normal behavior

### Debug Mode
```bash
# Run with verbose output
python tests/manual_seed_verification.py --verbose

# Run specific provider with custom prompt
python tests/manual_seed_verification.py --provider openai --prompt "Be deterministic"
```

## Integration with CI/CD

The basic tests (`test_seed_temperature_basic.py`) are designed to run in CI/CD environments without external dependencies. They test:

- Parameter inheritance and defaults
- Interface contracts
- Session behavior
- Provider functionality testing

The determinism tests require actual provider setup and are intended for:
- Development environments
- Manual verification
- Integration testing with real services

## Contributing

When adding new providers or modifying parameter handling:

1. Update `test_seed_temperature_basic.py` with new parameter tests
2. Add provider-specific tests to `test_seed_determinism.py`
3. Update the provider support matrix in this README
4. Test with `manual_seed_verification.py` to verify real-world behavior

## Architecture Notes

The test suite validates the architectural improvements made in version 2.4.6:

- **Interface-level parameters**: Tests verify parameters are inherited from `AbstractCoreInterface`
- **Centralized logic**: Tests validate `_extract_generation_params()` method
- **Provider consistency**: Tests ensure uniform behavior across all providers
- **Graceful degradation**: Tests verify fallback behavior for unsupported features
