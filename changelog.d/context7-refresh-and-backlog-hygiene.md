### Added

- **Context7 re-indexes the user docs after each release** (`.github/workflows/release.yml`, `context7.json`). Once the
  release is on PyPI the workflow asks Context7 to re-index this repository, so coding agents reading AbstractCore
  through the Context7 MCP see the published version. `context7.json` limits the index to the user docs
  (`docs/`, the README and the llms files), keeping the backlog, ADRs, reports and sources out. The only credential
  is `CONTEXT7_API_KEY` in the GitHub environment `context7`; without it the job reports a warning and does nothing,
  and required reviewers on that environment turn the refresh into an approved step. The same step runs on demand
  from the Actions tab (`context7-refresh.yml`) after a docs-only change.
