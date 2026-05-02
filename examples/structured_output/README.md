# Structured Output Examples

## What This Folder Teaches

How to ask an LLM for a *structured* result (JSON-like / schema-bound), then validate it.

This is the pattern you use when “natural language” is not enough:
- you need typed fields,
- you want to reject/repair invalid output,
- or you want to feed the output into downstream automation safely.

## Key AbstractCore Concepts

- Structured output is a contract between your app and the model.
- Validation is required: you must assume the model can output invalid structure.

## Scripts

- `example_structured_output.py`
  - Demonstrates: generating structured output and validating it (schema-first workflow).
  - How it works: pass `response_model=YourPydanticModel` to `llm.generate(...)`; AbstractCore will
    request JSON that matches the schema and validate (with optional retry/repair).
  - Run (local examples):
    - `python examples/structured_output/example_structured_output.py --provider ollama --model llama3.2:3b`
    - `python examples/structured_output/example_structured_output.py --provider lmstudio --model qwen3.5-4b@q4_k_m --base-url http://localhost:1234/v1`
  - Takeaway: treat models as probabilistic parsers—always validate before acting.

## Key Takeaways

- Keep schemas small and explicit; large schemas increase failure rate.
- Prefer “validate + retry/repair” over “hope the model follows the format”.
