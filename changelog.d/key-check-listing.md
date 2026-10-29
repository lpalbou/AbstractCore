### Fixed

- **A key check says whether the KEY works** (`openrouter_provider.py`, `anthropic_provider.py`; the gateway's setup
  guide checks a pasted key by listing models with `raise_on_error=True`). OpenRouter's `GET /models` is public: it
  answered 200 with the whole catalogue for a wrong key (measured 2026-10-29), so the check said "Key works · N models"
  to a typo. `OpenRouterProvider.list_available_models` now asks the authenticated `GET /key` first: a 401/403 lists
  nothing, and with `raise_on_error=True` raises `AuthenticationError` with OpenRouter's reason ("User not found.").
  Any other answer (a proxy `base_url` without `/key`) falls through to the listing as before. Anthropic's listing
  returned `[]` with no reason for a wrong key even with `raise_on_error=True`; it now raises `AuthenticationError`
  ("Anthropic API error (401): invalid x-api-key"), or `ProviderAPIError` for another status. Without
  `raise_on_error` both keep their old answer (`[]`). Tests: `tests/providers/test_key_check_listing_unit.py`.
