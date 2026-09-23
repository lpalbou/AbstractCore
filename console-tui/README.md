# abstractcore-console

A keyboard-first terminal console for configuring
[AbstractCore](https://github.com/lpalbou/abstractcore) — the
`abstractcore --config` wizard's job with browse-anywhere freedom,
honest live state, and validation. Rendered by
[AbstractTUI](https://crates.io/crates/abstracttui).

It also browses, downloads and deletes local models and installs local
engines (Ollama, LM Studio, MLX, llama.cpp), and it is a Rust library:
the same **Models** and **Engines** screens run inside the
[AbstractGateway console](https://crates.io/crates/abstractgateway-console).

## Install

```bash
cargo install abstractcore-console     # Rust 1.87+
abstractcore-console                   # needs the abstractcore CLI: pip install abstractcore
```

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

- **Browse** (digits 1-9, then 0 for the tenth screen): every section's honest state — set /
  default / broken, secrets fingerprinted, unknown keys flagged.
- **Providers** (3): ONE list, the AbstractGateway console's columns —
  `provider | family | base URL | API key | models | enabled | origin`
  — with every stored endpoint profile inline as its `endpoint:<id>`
  row. `a` adds a connection (as many as you need; it lands in this
  file's `provider_profiles`), `e` configures the selected row, `d`
  deletes a connection, `m` lists its models, `t` probes it, `k` edits
  a builtin provider's key. `origin` says where a row comes from:
  `config` · `env` · `auto` (a local server that answered) ·
  `registry` (known, nothing configured yet).
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
- **Models** (9): the model catalog fitted to this machine (host
  profile on top; fit `fits / tight / too large / partial offload /
  unknown`; weights `installed / not downloaded / unknown / remote`).
  `w` downloads the selected artifact with live progress, `d` deletes
  after a confirm that names what blocks it (a loaded model, a cache
  another engine shares — forcing is a separate answer), `/` filters,
  `f` shows only what fits, `e` picks the engine, `v` flips between the
  catalog and what is installed, `c` cancels the running job.
- **Engines** (0): which engines are installed and running. `i` installs
  the selected one after a confirm that shows the exact command, the
  host it runs on and whether it needs sudo/UAC (or answers with a dry
  run); `o` opens the vendor's download page (LM Studio is a desktop
  app); `r` probes the local servers.

Every Models/Engines action is an `abstractcore` CLI call you can run
yourself — the job strip shows its CLI equivalent
(`abstractcore models download ollama qwen3:8b`,
`abstractcore engines install ollama`).

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

## Use as a library

The Models and Engines screens are backend-agnostic: they talk to a
`ConsoleTransport`, which returns the JSON documents of the shared
contracts (`host_profile_v1`, `engines_status_v1`, `model_catalog_v1`,
`models_installed_v1`, `host_job_v1`). This binary uses `CliTransport`
(`abstractcore … --json` subprocesses); the gateway console implements
the trait over its HTTP client and mounts the same screens:

```toml
[dependencies]
abstractcore-console = "0.2"
abstracttui = "0.3.6"   # the same engine version: one reactive runtime
```

```rust
use std::sync::Arc;
use abstractcore_console::screens::{self, ScreensCtx, ScreensOptions};
use abstractcore_console::transport::{ConsoleTransport, TransportError};
use serde_json::Value;

struct HttpTransport { /* your client */ }

impl ConsoleTransport for HttpTransport {
    fn host_profile(&self) -> Result<Value, TransportError> { /* GET …/host/profile */ todo!() }
    // engines_status, models_catalog, models_installed, start_download,
    // delete_model, engine_install, job, cancel_job — one route each;
    // map 403/409 to TransportError::refused(msg, Some(body)).
    fn host_label(&self) -> String { "gateway.example.lan".into() }
}

// In your mount closure (UI thread), once:
let sctx = ScreensCtx::new(cx, Arc::new(HttpTransport { /* … */ }), overlays.clone(),
    ScreensOptions { notice: Some(store.notice), ..ScreensOptions::default() });
// …then as two more PageHost pages:
//   .page("catalog", "Models",  move |pcx| screens::catalog(pcx, &sctx_a))
//   .page("engines", "Engines", move |pcx| screens::engines(pcx, &sctx_b))
```

Each screen loads its data on first entry and binds its own keys
(`w d / f e v r c` on Models, `i o r c` on Engines);
`screens::catalog::HINTS` and `screens::engines::HINTS` are the footer
pairs to show. Full API: <https://docs.rs/abstractcore-console>.

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
  transport.rs ConsoleTransport + TransportError (the Models/Engines seam)
  transport/cli.rs  CliTransport: abstractcore … --json, child jobs
  screens/     the shared Models/Engines screens (library): store, worker, confirms
  ui/          one module per screen over a PageHost shell
tests/
  headless_ui.rs    CaptureTerm+Driver harness; fixtures; the chrome matrix;
                    Models/Engines over a MockTransport
  cli_transport.rs  CliTransport against a fake `abstractcore` on PATH
  fixtures/         contract A–E JSON documents
```

## Test

```bash
cargo test --locked   # headless: no network, no real config file touched
cargo clippy --all-targets -- -D warnings   # zero warnings is the bar
cargo fmt --check
python3 scripts/pty_smoke.py          # live: real CLI, scratch configs
python3 scripts/definition_of_done.py # the chartered end-to-end walk
```

See `LAUNCH-PROMPT.md` for the full charter and
`docs/config-surface-inventory.md` for the config surface this console
mirrors.
