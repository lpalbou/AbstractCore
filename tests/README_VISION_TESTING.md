# Vision testing

AbstractCore’s vision and media tests live under `tests/media_handling/` and are designed to be safe to run in many environments (tests that need providers/models typically **skip** when unavailable).

## Quick start

```bash
# Run all vision-related media tests
pytest -q tests/media_handling/test_vision_*.py

# Run the comprehensive suite (may be slow; many cases are opt-in/skipped)
pytest -q tests/media_handling/test_vision_comprehensive.py
```

## Notes

- Vision capabilities depend on your configured providers/models.
  - Cloud providers require API keys (`abstractcore --configure` or env vars).
  - Local providers require a running server (Ollama/LMStudio/vLLM) and a vision-capable model.
- Media handling (images/PDFs/Office docs) requires optional dependencies; see `docs/media-handling-system.md`.

