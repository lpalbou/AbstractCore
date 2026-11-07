### Added

- **Context7 re-indexes the user docs after each release** (`.github/workflows/release.yml`, `context7.json`). Once the
  release is on PyPI the workflow asks Context7 to re-index this repository, so coding agents reading AbstractCore
  through the Context7 MCP see the published version. The committed `context7.json` is the switch and limits the
  index to the user docs (`docs/`, the README and the llms files), keeping the backlog, ADRs, reports and sources
  out; the `CONTEXT7_API_KEY` repository secret is the only credential, and without it the job reports a warning
  and does nothing.
