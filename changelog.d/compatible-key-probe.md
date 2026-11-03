### Fixed

- **A key check on an OpenAI-compatible endpoint with a public model list checks the KEY**
  (`openai_compatible_provider.py`). OVH AI Endpoints answers `GET /v1/models` with 200 for no key and for a wrong
  one (measured 2026-10-29), so the gateway's Connect said "Connected · N models" to a wrong key. With
  `raise_on_error=True` (a key check) and a key set, when `/models` also answers WITHOUT the key, the listing now
  asks `POST /chat/completions` for a model that does not exist (`abstractcore-key-check`, nothing is generated):
  401/403 raises `AuthenticationError` with the service's reason (OVH: 403 "authentication failed"); any other
  answer (404 "model not found") keeps the key. An endpoint whose `/models` needs the key (Mistral, Groq, …) is not
  probed; plain listing is unchanged; OpenRouter keeps its `GET /key` check (`PUBLIC_MODELS_KEY_PROBE = False`).
  Tests: `tests/providers/test_compatible_key_probe_unit.py`.
