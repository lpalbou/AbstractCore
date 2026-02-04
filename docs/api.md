# API (Python)

This page is a user-facing map of the **public Python API**. For a complete listing of functions/classes (including events), see **[API Reference](api-reference.md)**.

## Core entrypoints

### `create_llm(...)`

Create a provider instance:

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")  # requires: pip install "abstractcore[openai]"
resp = llm.generate("Hello!")
print(resp.content)
```

### `BasicSession`

Keep conversation state:

```python
from abstractcore import BasicSession, create_llm

session = BasicSession(create_llm("anthropic", model="claude-haiku-4-5"))  # requires: abstractcore[anthropic]
print(session.generate("Give me 3 name ideas.").content)
print(session.generate("Pick the best one.").content)
```

## Responses (`GenerateResponse`)

Most calls return a `GenerateResponse` object (or an iterator of them for streaming). Common fields:

- `content`: cleaned assistant text
- `tool_calls`: structured tool calls (pass-through by default)
- `usage`: token usage (provider-dependent)
- `metadata`: provider/model specific fields (for example extracted reasoning text when configured)

## Tool calling

Tools are passed explicitly to `generate()` / `agenerate()`:

```python
from abstractcore import create_llm, tool

@tool
def get_weather(city: str) -> str:
    return f"{city}: 22°C and sunny"

llm = create_llm("openai", model="gpt-4o-mini")
resp = llm.generate("Use the tool.", tools=[get_weather])
print(resp.tool_calls)
```

See **[Tool Calling](tool-calling.md)** and **[Tool Syntax Rewriting](tool-syntax-rewriting.md)**.

## Structured output

Pass a Pydantic model via `response_model=...` to receive a typed result:

```python
from pydantic import BaseModel
from abstractcore import create_llm

class Answer(BaseModel):
    title: str
    bullets: list[str]

llm = create_llm("openai", model="gpt-4o-mini")
result = llm.generate("Summarize HTTP/3 in 3 bullets.", response_model=Answer)
print(result.bullets)
```

See **[Structured Output](structured-output.md)**.

## Media input

Media handling is opt-in:

```bash
pip install "abstractcore[media]"
```

Then pass `media=[...]` to `generate()` / `agenerate()` (or use the media pipeline). See **[Media Handling](media-handling-system.md)**.

## HTTP API (optional)

If you want an OpenAI-compatible `/v1` gateway, install and run the server:

```bash
pip install "abstractcore[server]"
python -m abstractcore.server.app
```

See **[Server](server.md)**.

