# Known bug: structured output retries can hit HTTP timeouts (local servers)

When using structured output (`response_model=...`) against some local OpenAI-compatible servers (for example, LMStudio), you may see repeated validation failures followed by an HTTP timeout.

## Symptoms

- Repeated warnings like `Validation attempt failed` from `abstractcore.structured.handler`
- The request eventually fails with an `httpx.ReadTimeout` (often wrapped in `ProviderAPIError`)

## Why it happens

Structured output is enforced by validating the model response against your Pydantic schema. If the backend returns:

- invalid JSON
- a JSON shape that doesn’t match the schema
- values outside constraints (e.g., a `float` outside `0..1`)

…AbstractCore retries with feedback (up to the configured retry limit). On slow models/backends, those retries can take long enough to hit the HTTP timeout.

## Workarounds

- Increase timeout for the run:
  - CLI apps: `intent ... --timeout 600` (or `--timeout 0` for unlimited)
  - Python: pass `timeout=...` to `create_llm(...)` (or `timeout=None` for unlimited)
- Use a backend/model with more reliable structured output support.
- Reduce schema complexity (fewer nested objects, tighter prompts).
- Reduce workload size (smaller input, smaller `--chunk-size`, or lower analysis depth where applicable).

## Notes

- Retry behavior is controlled by `FeedbackRetry` and the structured output handler in `abstractcore/structured/`.
