### Fixed

- **Endpoint profiles accept every provider the registry reaches over HTTP** (`config/provider_profiles.py`,
  `providers/registry.py`). The profile validator kept its own list of eight families, so the new decision
  providers were refused on save ("Unsupported provider family 'typesafe'") although `create_llm("typesafe")`
  worked. The accepted families are now `ProviderRegistry.endpoint_provider_families()`: every registered
  provider the light install covers that is not an in-process engine (MLX, local HuggingFace stay refused). The
  registry's logger is created on first use: the first config load runs inside `utils.structured_logging`'s
  import, and building the registry there to validate a stored profile would otherwise be an import cycle,
  which the loader reads as a corrupt config (defaults for the session). Tests:
  `tests/config/test_provider_profiles_config.py`.
