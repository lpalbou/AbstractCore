# 0884 — Release 2.35.0 (bumped on main 2026-11-07, never tagged): unreachable endpoints typed, MCP handshake for strict servers, tool parameter descriptions

> Type: release
> Created: 2026-11-08
> Priority: high
> Labels: release, providers, mcp, tools
> Links: `changelog.d/endpoint-unreachable-mcp-handshake-speculation-default.md`, `changelog.d/tool-parameter-descriptions-from-docstrings.md`, root backlog 1162 / 1164 / 1167 / 1168 (framework repository), [0885](../proposed/0885_tool_parameters_that_name_environment_variables.md)

## Summary

Main carries, unreleased, the 2026-11-08 patch series (`44897db3` … `4b339d64`): `EndpointUnreachableError`
(typed connect failures with `base_url`, `cause`, `detail`, `errno`, `elapsed_s`, `retryable`; no breaker, no
retry after a connect timeout; 503 `endpoint_unreachable` envelope on the HTTP server; pickles across the
model-host child), MCP `initialize` with `capabilities: {}` and re-initialize on a 404 session, the embeddings
residency race, speculation defaults degrading to MTP off (`metadata.speculation.applied=false`), and tool
parameter descriptions taken from docstring `Args:` sections (182 parameters on 31 built-ins). AbstractRuntime
main `da03304` already parks a step on the 503 envelope and the pickled exception; the runtime's and the
gateway's `abstractcore>=` floors rise to the release that carries `b8bcda1d` or later. Until that release
exists the resilience half cannot ship.

## Current code reality

- Code, tests and docs are on main; `scripts/ci_local.sh abstractcore` was green on `6c1e7a0e` and on the test
  job of `b8bcda1d`/`4b339d64` apart from the shell-session race fixed in `4b339d64`.
- Two changelog fragments wait in `changelog.d/`; `abstractcore/utils/version.py` already says 2.35.0 (e4e1a6f3); the CHANGELOG `[2.35.0]` section exists and the 2026-11-08 fragments are folded into it (re-date it at the cut).
- `README.md` "What's new" points at 2.35.0.

## Scope

1. Re-date the `[2.35.0]` section to the cut day (fragments already folded, version already bumped), README what's-new pointer, docs index
   and `llms*.txt` regenerated (`scripts/update_llms.py --check`), the `coredoc` release checklist.
2. Tag and publish through `release.yml`; the Context7 re-index runs by itself after PyPI.
3. Tell the framework session (root 1168) the published version so the runtime and gateway floors move.

## Also in this release (2026-11-09 audit)

- The config-loader fix: a valid config quarantined as corrupt by a circular import during `import abstractcore` (187 `.corrupt-*.bak` copies; processes ran on defaults). Tests: `tests/config/test_config_load_internal_error_unit.py`.
- 0869: the console theme copy regenerated against the ui-kit the apps are built on.

## Non-goals

No new code; 0885 and root 1160 stay separate.

## Validation

- `pip install abstractcore==2.35.0` in a clean venv; `from abstractcore.exceptions import EndpointUnreachableError`.
- Context7 shows the 2.35.0 README text.
