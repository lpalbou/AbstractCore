### Added

- **An optional Context7 refresh after each release** (`.github/workflows/release.yml`, `context7.json`). When the
  repository variable `CONTEXT7_REFRESH` is `true` and the `CONTEXT7_API_KEY` secret is set, the release workflow
  asks Context7 to re-index this repository's user docs once the release is on PyPI, so coding agents reading
  AbstractCore through the Context7 MCP see the published version. `context7.json` limits the index to the user
  docs (`docs/`, the README and the llms files) and keeps the backlog, ADRs, reports and sources out. Off by default.
