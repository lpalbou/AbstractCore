# BasicIntentAnalyzer — Intent analysis (with deception indicators)

`BasicIntentAnalyzer` is a structured intent-analysis component built on AbstractCore. It can analyze:

- a single text input (`analyze_intent`)
- a multi-participant conversation (`analyze_conversation_intents`)

Deception indicators are always included in the structured output schema.

## Install

```bash
# Core (lightweight)
pip install abstractcore

# Provider extras (install only what you use)
pip install "abstractcore[openai]"
pip install "abstractcore[anthropic]"
pip install "abstractcore[huggingface]"  # heavy (torch/transformers)

# Optional: read PDFs / Office docs in the CLI
pip install "abstractcore[media]"
```

## Quick start (Python)

```python
from abstractcore import create_llm
from abstractcore.processing import BasicIntentAnalyzer, IntentContext, IntentDepth

llm = create_llm("openai", model="gpt-4o-mini")  # requires `abstractcore[openai]`
analyzer = BasicIntentAnalyzer(llm)

result = analyzer.analyze_intent(
    "I'm struggling to understand this concept and need help.",
    context_type=IntentContext.STANDALONE,
    depth=IntentDepth.UNDERLYING,
)

print(result.primary_intent.intent_type.value)
print(result.primary_intent.underlying_goal)
print(result.primary_intent.deception_analysis.deception_likelihood)
```

## CLI usage (`intent`)

The `intent` CLI analyzes either direct text or a file path:

```bash
# Direct text
intent "I need help with this problem" --depth underlying

# File input
intent ./email.txt --context document --depth comprehensive --format plain

# Conversation log input (expects lines like `USER: ...` / `ASSISTANT: ...`)
intent ./chat.txt --conversation-mode --focus-participant user --format plain

# Session archive JSON created via `BasicSession.save(...)` (auto-enables conversation mode)
intent ./saved_session.json --focus-participant user --depth comprehensive --format plain
```

### Key flags

- `--context`: `standalone`, `conversational`, `document`, `interactive`
- `--depth`: `surface`, `underlying`, `comprehensive`
- `--conversation-mode`: parse multi-message inputs; use `--focus-participant` to target a role
- `--provider` + `--model`: override the default app model (must be provided together)
- `--format`: `json`, `yaml`, `plain`
- `--timeout`: HTTP timeout in seconds (`0` = unlimited)

Tip: to see every option, run `intent --help`.

## Interactive chat integration (`/intent`)

The interactive chat CLI (`abstractcore-chat`) supports `/intent` to analyze the current conversation state:

```bash
abstractcore-chat --provider ollama --model gemma3:1b-it-qat

# In the REPL:
/intent
/intent user
/intent assistant
```

## Events (optional)

You can monitor generations via the global event bus:

```python
from abstractcore.events import EventType, on_global

def monitor(event):
    if event.type == EventType.GENERATION_COMPLETED:
        duration_ms = event.data.get("duration_ms")
        print("generation_completed", duration_ms)

on_global(EventType.GENERATION_COMPLETED, monitor)
```

## Notes

- Long inputs are chunked and combined (map-reduce). Use `--chunk-size` in the CLI (or `max_chunk_size` in Python) to control chunk size.
- Output schemas live in `abstractcore/processing/basic_intent.py` (`IntentAnalysisOutput`, `IntentType`, `DeceptionIndicators`).
