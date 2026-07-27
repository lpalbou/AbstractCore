# abstractcore-console

A keyboard-first terminal console for configuring
[AbstractCore](https://github.com/lpalbou/abstractcore) — the
`abstractcore --config` wizard's job with browse-anywhere freedom,
honest live state, and validation. Rendered by
[AbstractTUI](https://crates.io/crates/abstracttui).

## Run

```bash
cargo run                 # needs an interactive terminal (headless: prints a skip line, exit 0)
cargo run -- --help
cargo run -- --theme catppuccin-mocha   # or ABSTRACTTUI_THEME=...
```

The console reads `~/.abstractcore/config/abstractcore.json` directly
(honoring `ABSTRACTCORE_CONFIG_FILE` / `ABSTRACTCORE_CONFIG_DIR`) and
shells out to the `abstractcore` CLI for the derived views (routes
coverage, redacted profiles). Binary resolution: `$ABSTRACTCORE_BIN` →
`abstractcore` on PATH → the framework venv fallback. The header always
names the file being shown and the CLI being used.

## What it does

- **Browse** (digits 1-8): every section's honest state — set /
  default / broken, secrets fingerprinted, unknown keys flagged.
- **Edit** (Enter/e, x clears): typed editors per field; coupled
  fields go through the `abstractcore` CLI setters; CLI-less fields
  through unknown-key-preserving direct writes. Every write is
  verified against a fresh re-read before it is reported done.
- **Wizard** (w; the default on a machine with no config): a guided
  walk over the CLI wizard's 8 phases. Browse stays free.
- **Test** (t on Providers/Routes; g anywhere): live model discovery
  per provider (`config test-provider`), route-model membership
  checks, and one cheap generation over YOUR configured default route
  (`abstractcore-chat --prompt`). Verdicts are honest three-state:
  proven / NOT PROVEN / failed — the CLI's `ok:true, count:0` answer
  for a dead server is never presented as success (a TCP reachability
  check on known local endpoints names the actual cause). Evidence
  lands on Review (8). Note: `g` tests the GLOBAL default
  (`default_models.global_*` — the library's fallback route);
  `abstractcore-chat`'s own CLI default is `app_defaults.cli`, a
  different slot.

## Safety posture

- Never writes on open; a corrupt config file is a hard stop that
  points at the timestamped backups, never a silent reset. A file that
  parses here but that PYTHON's loader refuses (e.g. a profile row
  with an unknown field) is flagged loudly — both from the mirror's
  own fold and from the CLI's `#FALLBACK` stderr.
- Writes refuse structurally: corrupt/unreadable files refuse all
  writes; Python-refused files refuse CLI setters (which would reset
  the file) while direct preserving writes stay allowed; a drift guard
  (mtime+ino+size) refuses when another writer landed in between.
- Secrets render as `set · fp <sha256[:8]>` — the same fingerprint
  convention the Python side uses. Values are never logged or echoed.
- Tests only reach endpoints the config names (profile base_url) or a
  provider's documented LOCAL default (ollama/lmstudio); never https,
  never cloud endpoints — those get CLI-only verdicts.

## Layout

```
src/
  lib.rs       arg parsing, headless guard, mount, worker spawn, boot load
  schema.rs    the DISPLAY schema (sections/fields/defaults/validation) — never a write schema
  config.rs    config path resolution, parse, fold to a redacted display model
  cli.rs       the abstractcore(-chat) CLI subprocess client + error taxonomy
  store.rs     signals per domain; Loadable<T> honest states
  worker.rs    ONE background thread owning all file/subprocess/socket I/O
  writes.rs    the write vocabulary: specs, verbs, verified expectations
  probes.rs    the test vocabulary: probe specs + pure verdict folds
  ui/          one module per screen over a PageHost shell
tests/
  headless_ui.rs   CaptureTerm+Driver harness; fixtures; the chrome matrix
```

## Test

```bash
cargo test            # headless: no network, no real config file touched
cargo clippy --all-targets   # zero warnings is the bar
python3 scripts/pty_smoke.py          # live: real CLI, scratch configs
python3 scripts/definition_of_done.py # the chartered end-to-end walk
```

See `LAUNCH-PROMPT.md` for the full charter and
`docs/config-surface-inventory.md` for the config surface this console
mirrors.
