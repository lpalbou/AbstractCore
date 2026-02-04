# Example 01: Basic usage (Ollama)

This example shows the smallest useful AbstractCore program: create a provider and call `generate()`.

## Prerequisites

- Python 3.9+
- An Ollama server running locally (or reachable over the network)

Install Ollama: see https://ollama.com/

## Install

```bash
pip install abstractcore
```

Ollama uses the core install (no provider extra is required).

## Prepare a model

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

## Code (non-streaming)

```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
resp = llm.generate("Tell me a short joke.")
print(resp.content)
```

## Code (streaming)

```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
for chunk in llm.generate("Write a short poem about distributed systems.", stream=True):
    print(chunk.content or "", end="", flush=True)
```

## Troubleshooting

- **Connection errors**: ensure Ollama is running (`ollama serve`) and that your base URL is correct.
  - programmatic: `create_llm("ollama", ..., base_url="http://localhost:11434")`
  - env vars: `OLLAMA_BASE_URL` (or legacy `OLLAMA_HOST`)
- **Model not found**: confirm the model exists: `ollama list`

## Reference docs

- [Getting Started](../../docs/getting-started.md)
- [Prerequisites](../../docs/prerequisites.md)
- [Troubleshooting](../../docs/troubleshooting.md)

## Next steps

After completing this example, move to `example_02_providers` to explore more advanced provider configurations.

## Contribute & feedback

- Found an issue? [Open an issue](https://github.com/lpalbou/AbstractCore/issues)
- Want to improve the example? Submit a pull request!

---
**Complexity**: Beginner  
**Last Updated**: 2026-02-04
