# Feature Request: Accept `system` as alias for `system_prompt` in generate()

## Context
The `generate()` method accepts `system_prompt=` but silently ignores `system=` via `**kwargs`. This caused a silent failure in an experiment where 864 tests ran overnight without any system prompt being applied — producing invalid results and wasting ~8 hours of computation.

## Problem
```python
# This works:
llm.generate("Hello", system_prompt="You are a pirate.")

# This is silently ignored — no error, no warning:
llm.generate("Hello", system="You are a pirate.")
```

The `system=` parameter is consumed by `**kwargs` in the `generate()` → `generate_with_telemetry()` → `_generate_internal()` chain but never reaches `system_prompt`. No warning is emitted.

## Proposed Fix
Either:
1. **Accept `system` as an alias** for `system_prompt` in `_generate_internal()`:
   ```python
   system_prompt = system_prompt or kwargs.pop('system', None)
   ```
2. **Or raise a warning** when `system` is passed as a kwarg:
   ```python
   if 'system' in kwargs:
       warnings.warn("Use 'system_prompt=' instead of 'system='. The 'system' parameter was ignored.", UserWarning)
   ```

Option 2 is safer (explicit > implicit) and follows the project's ADR on fallbacks requiring warnings.

## Impact
- Any user coming from OpenAI's API (which uses `messages` with `role: system`) or from other frameworks that use `system=` will hit this silently.
- The `**kwargs` pattern makes this particularly dangerous — the parameter is consumed but never used.

## Requested by
Laurent-Philippe Albou, 2026-02-27
Discovered during unit prompt experiment v7 (864 tests invalidated).
