# 0854-abstractcore: [FEATURE] Models & engines backend follow-ups (catalog verification, geometry, HF quant deletes, LM Studio fallbacks)

> Created: 2026-09-23
> Status: Planned
> Type: feature
> Priority: P2
> Labels: models, engines, catalog, consoles

## Summary

The models & engines backend (host profile, fit estimator, engine inventory/install, host
jobs, installed inventory, delete, curated catalog, CLI verbs and `/acore` routes; see
`docs/models.md` and `docs/engines.md`) shipped with known gaps. This item collects them so
the gateway and console work that builds on these payloads has a single follow-up list.

## Current code reality

Audited 2026-09-23 on branch `feat/models-engines` (integration notes below):

- `abstractcore/assets/model_downloads_catalog.json` has 76 rows / 46 families. Artifacts
  observed on a real engine store carry `verified: true` and exact sizes; the rest follow each
  engine's naming convention and are unverified (sizes `null`, shown as `estimate` from
  parameters x bits). LM Studio ids without `@quant` rely on LM Studio's default build.
- `utils/model_fit.py` uses the rough KV rule (0.5 MiB/token per 8B params) whenever geometry
  is not on local disk; it over-estimates KV for GQA models (e.g. 8B@Q4 reads `tight` on a
  16 GB CPU box). `catalog(hub=True)` fetches file sizes but not `config.json` geometry.
- HF deletes are repo-level (`delete_revisions`); a `repo:QUANT` delete removes every quant.
  GGUF downloads do not fetch `mmproj*` projector files for vision GGUF repos.
- LM Studio listing requires the `lms` CLI; without it `list_installed("lmstudio")` reports an
  error (no directory-scan fallback). LM Studio downloads report text lines only (no byte %);
  the REST `POST /api/v1/models/download` lane is not used. There is no `lms rm`; deletes
  remove files under the models root plus the hub manifest directory.
- Ollama `/api/tags` field names were taken from Ollama's documentation and the on-disk
  manifest format; the server on the audit machine was not running, so the live shape was not
  re-observed. `ollama --version` parsing covers `ollama version is X` and
  `client version is X`.
- Engine installs run with stdin closed: the Linux Ollama script and the macOS script fallback
  fail at a sudo prompt instead of prompting. No elevation flow exists (by design).
- Windows paths (winget/PowerShell plans, `%LOCALAPPDATA%` detection) are unit-tested as data
  only; nothing has run on a Windows host.
- The core server records its bind host in `ABSTRACTCORE_SERVER_BIND_HOST` from
  `run_server`/`run_server_with_args`; an app mounted another way (e.g. `uvicorn
  abstractcore.server.app:app`) has an unknown bind and therefore installs off by default.

### Integration notes (2.14.0 integration branch, 2026-09-23)

The backend, the web console and the terminal console were merged and verified together (see
the 2.14.0 changelog). What that pass left open:

- **Ollama shape, partly re-observed.** On the integration run Ollama 0.20.2 was running;
  `models list` returned its rows (e.g. `gemma3:1b`, 815,319,791 bytes) through `/api/tags`, and
  a delete dry run planned `DELETE /api/delete`. The offline fixtures are still the documented
  shape; refreshing them from a live capture remains open (acceptance criterion below).
- **Metal ceiling fallback.** On a Mac with no `iogpu.wired_limit_mb` set and without `mlx` in
  the interpreter, `host profile` reports `ceiling_source: ram_75pct`. That is correct per the
  order, but the console could say why ("install MLX for the Metal working-set figure").
- **First run on loopback.** `uvicorn abstractcore.server.app:app` (and the Docker image) skip the
  first run by design and need `ABSTRACTCORE_AUTH_TOKEN`. On Windows the 0600 mode of
  `server-token` is not enforced (NTFS ACLs of the user profile apply); claim records are pruned
  only when the next code is minted, and a crash mid-redeem can leave a `.redeeming` file behind
  (harmless, never redeemable). A `serve --claim-url` against a server started with an explicit
  token exits 2 by design.
- **`abstractcore serve` now binds 127.0.0.1 by default** (was 0.0.0.0): a behaviour change for
  anyone relying on the old default for LAN access; they now pass `--host 0.0.0.0` with a token.
- **Terminal console, routes screen (3) download.** It reads only the final NDJSON line (fixed);
  it shows elapsed time, not progress. Screen 9 (Models) is the lane with a progress bar.
- **Test isolation.** Config tests that isolate through `HOME` (e.g.
  `tests/config/test_capability_defaults_config.py`) fail when `ABSTRACTCORE_CONFIG_DIR` is set in
  the caller's environment; they should clear it themselves.
- **Pre-existing failure, unrelated:** `tests/test_prompt_cache_bloc_composition.py` errors at
  collection on hosts without at least three whitelisted local tokenizers (by design of that
  guard); the WS1 subset runs with `--ignore` for it.
- **First crates.io publish of `abstractcore-console` 0.2.0 is manual** (trusted publishing needs
  an existing crate); the crate's dev-only `scripts/pty_smoke.py` still falls back to a workspace
  venv path (not packaged in the crate).

## Scope and non-goals

In scope: verify or correct catalog ids against the live registries (Ollama library,
LM Studio hub, Hugging Face) with a repeatable script that writes sizes and flips `verified`;
fetch `config.json` geometry under `hub=True` (cached with the 24 h hub cache); quant-level HF
deletes via `HFCacheInfo.delete_files` where available; `mmproj` selection for vision GGUF;
LM Studio directory-scan fallback and REST download lane with byte progress; a live Ollama
`/api/tags` fixture refresh; a Windows smoke run of detection and install plans.

Non-goals: automatic elevation (sudo/UAC), unattended engine installs from remote gateways,
auto-downloading anything during a probe or catalog read, a second catalog store.

## Acceptance criteria

- [ ] A catalog verification script (network, maintainer-run) updates sizes and `verified` flags; the seed schema test still passes.
- [ ] Fit verdicts use real geometry for catalog rows when `hub=True`, with `confidence` raised accordingly.
- [ ] Deleting `org/Repo-GGUF:Q4_K_M` removes only that quant's files; other quants remain and are still listed.
- [ ] `list_installed("lmstudio")` returns rows from the models directory when `lms` is absent, flagged as a fallback.
- [ ] LM Studio download jobs report `total_bytes`/`percent`.
- [ ] Ollama `/api/tags` test fixtures are refreshed from a live server capture.
- [ ] Engine detection and plans verified on a Windows host.
- [ ] Config tests that set `HOME` also clear `ABSTRACTCORE_CONFIG_DIR` / `ABSTRACTCORE_CONFIG_FILE`.
- [ ] First run verified on Windows (token file permissions, `serve --print-token`).

## Testing

`python -m pytest tests/config/test_model_catalog.py tests/config/test_models_installed_and_delete.py tests/config/test_host_jobs_and_engines.py tests/config/test_models_engines_cli.py tests/server/test_server_host_routes.py -q`
plus new offline fakes for each added lane.

## Dependencies and ADR status

Builds on the models & engines contracts (host_profile_v1, engines_status_v1,
model_catalog_v1, models_installed_v1, host_job_v1) consumed by the AbstractGateway and the
core/gateway consoles; keep field names stable. ADR impact: none expected; the install-safety
policy (fixed argv allowlist, loopback default, principal required) is documented in
`docs/engines.md` and should become an ADR if the gateway adopts it unchanged.
