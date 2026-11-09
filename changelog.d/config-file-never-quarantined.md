### Fixed

- **A settings file that cannot be parsed stops the program instead of being renamed and replaced by defaults**
  (`abstractcore/config/manager.py`, `abstractcore/config/provider_profiles.py`, `abstractcore/server/app.py`). The
  loader raises `ConfigParseError` with the file path and the parse error (`OSError` when the file cannot be read);
  the file is never copied to `abstractcore.json.corrupt-<ts>.bak`, never shadowed by default settings, and no later
  save can overwrite it. A provider profile's family is accepted as written while the provider registry is still
  importing (it is validated when the profile is used), so a load that happens during `import abstractcore` no
  longer fails. The `abstractcore config` CLI prints the error and exits with status 1, and the server refuses to
  start on such a file instead of coming up with no settings. `ConfigParseError` is exported from
  `abstractcore.config`.
