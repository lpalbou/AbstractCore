# Reasoning / “Thinking” Examples

## What This Folder Teaches

How to use AbstractCore’s unified `thinking=` control and how to validate what the backend actually returned.

Different providers/models expose “reasoning traces” differently (or not at all). AbstractCore tries to normalize:
- requesting reasoning (`thinking=...`), and
- extracting reasoning (when the backend supports it).

## Prereqs

- LM Studio: start the server (`http://localhost:1234/v1`) and load a model
- Ollama: `ollama serve`

## Key AbstractCore Concepts

- `thinking=` is best-effort: if a provider doesn’t support it, AbstractCore should warn and continue.
- “Reasoning” is not the same as “final answer”: some backends return it as a separate field; others embed it in text.
- For product work, you should assume reasoning is optional metadata, not required output.

## Scripts

- `qwen_thinking_repl.py`
  - Demonstrates: interactive probing across providers/models; prints raw payloads where possible.
  - Takeaway: don’t assume a model obeyed `thinking=`—verify.

- `reasoning_control_demo.py`
  - Demonstrates: minimal “one call per provider” usage of `thinking=`.
  - Takeaway: this is the smallest example to copy into your own tests.

- `local_qwen3_5_thinking_probe.py` (dev-oriented)
  - Demonstrates: a more instrumented probe (model listing, heuristics, payload preview).
  - Takeaway: useful when you’re debugging a backend integration.

## Key Takeaways

- Treat `thinking=` as a capability, not a contract.
- When you need determinism, set temperature=0 and keep max tokens tight; “reasoning” tends to increase output variance.
