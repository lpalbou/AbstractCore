### Fixed

- **A provider profile whose family this AbstractCore does not know no longer makes the whole config read as
  corrupt** (`config/provider_profiles.py`, `config/manager.py`). 2026-11-03: the abstractcore 2.31 installs in the
  gateway and assistant environments (whose family list predates `typesafe`/`systemone`, added in 2.32.0) read a
  config holding a TypeSafe connection. The row's "Unsupported provider family 'typesafe'" error quarantined the file
  (a `.corrupt-<stamp>.bak` copy on every import) and ran the session on DEFAULTS, without any of the other profiles
  or settings. Such a row is now skipped with one WARNING naming it ("provider profile 'typesafe-key' is not loaded by
  this AbstractCore (...); it stays in the config file unchanged"). The rest of the config loads, nothing is written,
  and a later save keeps the row exactly as stored (the same carry that keeps a newer Core's route keys and columns).
  The same applies to any profile row that does not validate. `typesafe` and `systemone` are profile families since
  2.32.0, derived from the provider registry.
