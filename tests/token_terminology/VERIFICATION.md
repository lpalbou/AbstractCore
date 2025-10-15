# Token Terminology Migration - Verification Guide

## Quick Verification Commands

### 1. Run the Comprehensive Test Suite
```bash
source .venv/bin/activate
python -m pytest tests/token_terminology/test_max_tokens_migration.py -v
```

**Expected Result**: 25/25 tests PASSED ✅

---

### 2. Verify Specific Models

#### Test GPT-4
```bash
source .venv/bin/activate
python -c "
from abstractcore.architectures import get_model_capabilities, get_context_limits
from abstractcore.providers.openai_provider import OpenAIProvider

caps = get_model_capabilities('gpt-4')
limits = get_context_limits('gpt-4')
provider = OpenAIProvider('gpt-4')

print('GPT-4 Verification:')
print(f'  max_tokens from capabilities: {caps.get(\"max_tokens\")}')
print(f'  max_tokens from get_context_limits(): {limits.get(\"max_tokens\")}')
print(f'  max_tokens on provider: {provider.max_tokens}')
print(f'  ✅ All values should be 128000')
"
```

#### Test Qwen3 with Alias
```bash
source .venv/bin/activate
python -c "
from abstractcore.architectures import get_model_capabilities, get_context_limits
from abstractcore.providers.ollama_provider import OllamaProvider

# Test alias resolution
alias = 'qwen/qwen3-next-80b'
caps = get_model_capabilities(alias)
limits = get_context_limits(alias)

print('Qwen3-Next Alias Test:')
print(f'  Using alias: {alias}')
print(f'  max_tokens: {caps.get(\"max_tokens\")}')
print(f'  ✅ Should be 262144 (alias resolved correctly)')
"
```

#### Test Ollama Provider
```bash
source .venv/bin/activate
python -c "
from abstractcore.providers.ollama_provider import OllamaProvider

provider = OllamaProvider('qwen3-coder:30b')

print('Ollama Provider Verification:')
print(f'  Model: {provider.model}')
print(f'  Architecture: {provider.architecture}')
print(f'  max_tokens: {provider.max_tokens}')
print(f'  max_output_tokens: {provider.max_output_tokens}')
print(f'  ✅ max_tokens should be 32768')
"
```

---

### 3. Verify JSON Has No Legacy References

#### Check for context_length (should be 0)
```bash
grep -r "context_length" abstractcore/assets/model_capabilities.json
# Expected: No output (0 matches)
```

#### Count max_tokens occurrences (should be 85+)
```bash
grep -o "max_tokens" abstractcore/assets/model_capabilities.json | wc -l
# Expected: 86 (85 models + 1 in default_capabilities)
```

---

### 4. Verify Code Has No Legacy References

#### Check detection.py
```bash
grep -i "context_length" abstractcore/architectures/detection.py
# Expected: No output (0 matches)
```

#### Check base.py
```bash
grep -i "context_length" abstractcore/providers/base.py | grep -v "^#"
# Expected: No output (0 active code references)
```

---

### 5. Verify All 85 Models

#### Count models with max_tokens
```bash
source .venv/bin/activate
python -c "
import json
from pathlib import Path

with open('abstractcore/assets/model_capabilities.json', 'r') as f:
    data = json.load(f)

models = data.get('models', {})
with_max_tokens = [m for m, d in models.items() if 'max_tokens' in d]
with_context_length = [m for m, d in models.items() if 'context_length' in d]

print(f'Total models: {len(models)}')
print(f'Models with max_tokens: {len(with_max_tokens)}')
print(f'Models with context_length: {len(with_context_length)}')
print()
print('✅ Expected: 85 total, 85 with max_tokens, 0 with context_length')
"
```

---

### 6. Run Existing Integration Tests

```bash
source .venv/bin/activate
python -m pytest tests/integration/test_system_integration.py::TestJSONCapabilitiesIntegration -v
```

**Expected Result**: All capability tests pass ✅

---

### 7. Verify Backward Compatibility

```bash
source .venv/bin/activate
python -c "
from abstractcore.architectures import detect_architecture, get_model_capabilities, get_context_limits
from abstractcore.providers.openai_provider import OpenAIProvider

# Test that all existing APIs still work
model = 'gpt-4'

# 1. Architecture detection
arch = detect_architecture(model)
assert arch == 'gpt', f'Architecture detection failed: {arch}'

# 2. Capabilities
caps = get_model_capabilities(model)
assert 'architecture' in caps, 'Capabilities missing architecture'

# 3. Context limits (NEW API)
limits = get_context_limits(model)
assert 'max_tokens' in limits, 'Context limits missing max_tokens'
assert 'max_output_tokens' in limits, 'Context limits missing max_output_tokens'

# 4. Provider creation
provider = OpenAIProvider(model)
assert provider.model == model, 'Provider creation failed'

print('✅ All backward compatibility checks passed')
print(f'   - Architecture detection: {arch}')
print(f'   - max_tokens: {limits[\"max_tokens\"]}')
print(f'   - max_output_tokens: {limits[\"max_output_tokens\"]}')
"
```

---

## Expected Outputs Summary

| Test | Command | Expected Output |
|------|---------|----------------|
| Full Test Suite | `pytest tests/token_terminology/...` | `25 passed in ~9s` |
| GPT-4 Verification | Python script | `max_tokens: 128000` |
| Qwen3 Alias | Python script | `max_tokens: 262144` |
| Ollama Provider | Python script | `max_tokens: 32768` |
| JSON Legacy Check | `grep context_length` | No output |
| Code Legacy Check | `grep context_length` | No output |
| Model Count | Python script | `85 total, 85 with max_tokens` |
| Integration Tests | `pytest ...Integration` | All pass |
| Backward Compat | Python script | All checks passed |

---

## Visual Verification Checklist

Use this checklist to manually verify the migration:

### ✅ JSON Verification
- [ ] Open `abstractcore/assets/model_capabilities.json`
- [ ] Search for "context_length" → Should find 0 matches
- [ ] Search for "max_tokens" → Should find 86 matches (85 models + 1 default)
- [ ] Verify sample models:
  - [ ] gpt-4: `"max_tokens": 128000`
  - [ ] claude-3.5-sonnet: `"max_tokens": 200000`
  - [ ] llama-3.1-8b: `"max_tokens": 128000`

### ✅ Code Verification
- [ ] Open `abstractcore/architectures/detection.py`
  - [ ] Line 255-256: Function returns `"max_tokens"` key
  - [ ] No references to `context_length`
- [ ] Open `abstractcore/providers/base.py`
  - [ ] Provider initialization uses `max_tokens`
  - [ ] No active references to `context_length`

### ✅ Test Verification
- [ ] Run test suite: All 25 tests pass
- [ ] Check test output: No errors or warnings
- [ ] Review test report: 100% coverage confirmed

### ✅ Integration Verification
- [ ] Existing tests still pass
- [ ] No breaking changes detected
- [ ] Provider creation works correctly

---

## Troubleshooting

### Issue: Tests fail with import errors
**Solution**: Make sure virtual environment is activated
```bash
source .venv/bin/activate
```

### Issue: "context_length" still appears in grep
**Solution**: Check if it's in a comment or documentation (safe)
```bash
grep -n "context_length" <file>  # Shows line numbers
```

### Issue: Provider has wrong max_tokens
**Solution**: Verify model name matches JSON exactly
```python
from abstractcore.architectures import get_model_capabilities
caps = get_model_capabilities('your-model-name')
print(caps.get('max_tokens'))  # Should show the correct value
```

### Issue: Unknown model gets wrong default
**Solution**: Check default_capabilities in JSON
```bash
grep -A 5 "default_capabilities" abstractcore/assets/model_capabilities.json
# Should show: "max_tokens": 16384
```

---

## Files to Review

### Primary Files Modified
1. `/Users/albou/projects/abstractcore_core/abstractcore/assets/model_capabilities.json`
2. `/Users/albou/projects/abstractcore_core/abstractcore/architectures/detection.py`
3. `/Users/albou/projects/abstractcore_core/abstractcore/providers/base.py`
4. `/Users/albou/projects/abstractcore_core/abstractcore/utils/cli.py`
5. `/Users/albou/projects/abstractcore_core/tests/integration/test_system_integration.py`

### Test Files Created
1. `/Users/albou/projects/abstractcore_core/tests/token_terminology/test_max_tokens_migration.py`
2. `/Users/albou/projects/abstractcore_core/tests/token_terminology/TEST_REPORT.md`
3. `/Users/albou/projects/abstractcore_core/tests/token_terminology/SUMMARY.md`
4. `/Users/albou/projects/abstractcore_core/tests/token_terminology/VERIFICATION.md` (this file)

---

## Quick Pass/Fail Test

Run this single command to verify the migration:

```bash
source .venv/bin/activate && python -c "
import json
from pathlib import Path
from abstractcore.architectures import get_context_limits
from abstractcore.providers.openai_provider import OpenAIProvider

# Load JSON
with open('abstractcore/assets/model_capabilities.json') as f:
    data = json.load(f)

# Check all models have max_tokens
models = data.get('models', {})
has_max_tokens = all('max_tokens' in m for m in models.values())
has_context_length = any('context_length' in m for m in models.values())

# Check get_context_limits returns max_tokens
limits = get_context_limits('gpt-4')
limits_correct = 'max_tokens' in limits and 'context_length' not in limits

# Check provider uses max_tokens
provider = OpenAIProvider('gpt-4')
provider_correct = provider.max_tokens == 128000

# Final result
success = has_max_tokens and not has_context_length and limits_correct and provider_correct

if success:
    print('✅ PASS: Migration verified successfully')
    print(f'   - All {len(models)} models have max_tokens')
    print(f'   - No models have context_length')
    print(f'   - get_context_limits() uses max_tokens')
    print(f'   - Providers use max_tokens from JSON')
else:
    print('❌ FAIL: Migration verification failed')
    print(f'   - Models with max_tokens: {sum(1 for m in models.values() if \"max_tokens\" in m)}/{len(models)}')
    print(f'   - Models with context_length: {sum(1 for m in models.values() if \"context_length\" in m)}')
    print(f'   - get_context_limits correct: {limits_correct}')
    print(f'   - Provider correct: {provider_correct}')
"
```

**Expected Output**: `✅ PASS: Migration verified successfully`

---

**Last Updated**: 2025-10-11
**Status**: ✅ Production Ready
