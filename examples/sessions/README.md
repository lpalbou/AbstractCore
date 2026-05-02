# Sessions Examples

## What This Folder Teaches

AbstractCore sessions are the “application layer” above a provider:
- they hold conversation history,
- they keep your system prompt and tool catalog together,
- and they provide convenience methods for multi-turn chat and persistence.

If you’re building anything beyond a single `llm.generate(...)` call, you typically want a session.

## Key AbstractCore Concepts

- `BasicSession`: minimal conversation tracking (system prompt + message history + generate/stream).
- Session persistence: save/load a chat transcript as JSON so you can resume.

## Scripts

- `basic_session_repl.py`
  - Demonstrates: a minimal multi-turn REPL built on `BasicSession`, plus `:save`/`:load`.
  - Takeaway: sessions keep your app logic simple as soon as you have multiple turns.

## Key Takeaways

- Providers are interchangeable backends; sessions are where your app’s conversational state lives.
- Keep your session boundary explicit: “what’s in history” is what the model can see.

