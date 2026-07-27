# abstractcore-console — the builder brief (checked in: this IS the charter)

You are the builder of `abstractcore-console`: a keyboard-first terminal
console that makes configuring AbstractCore pleasant — the
`abstractcore --config` wizard's job with browse-anywhere freedom, honest
live state, and validation, rendered by AbstractTUI 0.2.22. You own this
directory. You never edit AbstractCore's Python or the engine. Work with
one adversarial fable5 subagent per wave — findings are half the mission
(see the feedback protocol).

## The two briefing documents — read FULLY before any code

Both live in `docs/` (probe reports from the engine seat, every claim
cited to source `file:line`):

1. `docs/config-surface-inventory.md` — WHAT you are building over:
   where config lives, the 8-phase/~26-prompt wizard, the 17-section /
   85-field surface, the 24 capability routes, providers and secrets,
   the write-path rules, the risk map.
2. `docs/setup-pattern-recommendations.md` — HOW to build it: the proven
   gateway-console architecture (this app's sibling), the day-one engine
   kit, the born-knowing lessons, the test harness pattern.

## Mission

A user (or the operator) runs `abstractcore-console` and can, without
reading docs: see the WHOLE config state honestly (what is set, what is
default, what is broken), walk a wizard mode covering at least the CLI
wizard's 8 phases (default model → vision → API keys → server → audio →
video → embeddings → logging), browse any section directly (PageHost
screens), edit values with the right control per type (Select/Combobox
for enums+models, masked TextInput for keys, ChoicePrompt for coupled
decisions), test the result (provider connectivity, model listing, a
cheap generation), and leave with a valid file — never a corrupted one.

## The integration contract (settled by the probe — do not re-litigate)

- Reads: parse `~/.abstractcore/config/abstractcore.json` directly
  (`serde_json::Value` — NEVER a typed schema snapshot) + shell out to
  `abstractcore config defaults --json` / `providers --json` for the
  DERIVED views (coverage badges, redacted profiles).
- Writes: through `abstractcore` CLI setters wherever one exists — they
  enforce coupled-field invariants (global default also writes
  `input.text`; embeddings mirror into `embedding.text`;
  `--set-audio-strategy` sets `audio_strategy_explicit`). Direct
  read-modify-write (tmp+rename, 0600, unknown keys PRESERVED) only for
  the ~26 config-only fields with no CLI. Verify every write by
  re-reading.
- The abstractcore HTTP server is optional enrichment only (it is
  usually not running): capability-defaults CRUD + read-only discovery
  when up; the console must be fully functional without it.

## The five facts you must not get wrong (inventory §risk-map)

1. Python's own save drops unknown keys — YOUR direct writes must
   preserve them, and you may never stash console metadata in the file.
2. Coupled writes go through the CLI (the setter list is in the
   inventory) — independent field writes desync status from runtime.
3. Env precedence is OPPOSITE by domain: API keys in config OVERRIDE
   env; server settings YIELD to env. Follow the code, not the old doc.
4. Never auto-save on open. A corrupt file loads as defaults after a
   timestamped backup — persisting what you loaded completes a known
   data-loss incident. Parse failure = stop, point at `.corrupt-*.bak`.
5. Secrets are plaintext in the file: render only Set/Not set +
   sha256[:8] fingerprints; never log values; never auto-probe
   openai-compatible/vllm without a configured base_url.

## The five born-knowing lessons (pattern doc §lessons)

1. Chrome pins + heavy fixtures from day one: `shrink(0.0)` on every
   fixed row; the chrome-survival matrix test ships WITH the first
   screen, not after the first operator screenshot.
2. `.focusable().autofocus()` on every modal's content root — or all
   keys die and it reads as a freeze (engine finding 1000, open).
3. Honest states are the product: `Loadable<T>` four states;
   unreachable ≠ 401 ≠ 403 ≠ port-squatter; body over transport; a
   picker must never fabricate a selection; refusals say why.
4. ONE worker thread owns all subprocess/file I/O (posted closures
   back); writes are write → verify-by-re-read → journal; Secret
   newtypes redact structurally.
5. Render `use_startup_notices` in the footer — the engine names layout
   crushes into that lane; unread means invisible.

## Day-one engine kit (0.2.22 — docs/api.md sections exist for each)

PageHost (screens) · Disclosure (config groups — progressive
disclosure) · Select/Combobox (pickers; modal popup anchoring fixed) ·
masked TextInput (keys) · ChoicePrompt (decision gates) · Drawer (detail
inspector) · Table/List + on_activate (inventories, double-click) ·
ThemeSwitcher (one line in the footer) · Block::on_close (dismissible
panels) · Screenshot (SVG artifacts for reports + tests) ·
CaptureTerm+Driver headless harness. Explicitly NOT needed:
Meter/TimeSeries, Feed, the graph crates, reactive::connection (this
client is probe-shaped).

## Milestones (each ends: tests green + a fable5 adversarial review + a report)

- M1 — the honest mirror: shell (PageHost + chrome + footer with
  startup notices + ThemeSwitcher), config load + parse + the Overview
  screen (every section's set/default/broken state, secrets
  fingerprinted), read-only. The chrome matrix test ships here.
- M2 — edit + wizard: section screens with typed editors, the write
  paths (CLI setters + RMW with unknown-key preservation,
  verify-by-re-read), wizard mode (gated linear walk over the 8 phases;
  browse stays free), corrupt-file refusal path.
- M3 — test & prove: provider connectivity checks (model listing per
  provider; a cheap generation for the default route), the Review
  screen, SVG screenshot artifacts, pty smoke, the final adversarial
  acceptance (one keyboard-driven end-to-end: fresh config → wizard →
  verified working default model → file valid for the Python side).

## Definition of done

A screen recording (or SVG series) of: launch on a machine with no
config → wizard completes with a local provider (lmstudio/ollama) →
test passes → `abstractcore config defaults --json` (the Python side)
agrees with what the console shows → browse mode edits one capability
route → re-verify. Plus: headless suite green, pty smoke green, zero
clippy warnings, the chrome matrix green at 80x24/100x24/60x16.

## Feedback protocol (half the mission)

Engine findings (bugs, gaps, footguns in abstracttui) are filed in THE
ENGINE repo band
`~/tmp/abstractframework/abstracttui/docs/backlog/proposed/field-core/`
(range: 1100-1190; README there) — one file per finding, `file:line`
evidence, the workaround shipped meanwhile. AbstractCore findings
(config bugs, wizard gaps — e.g. the inventory already found the
wizard's embeddings validation omits vllm) go to the CORE seat via a
note in the report; never edit the Python. The engine ships fast: when
a release note lands mid-build, read it and adopt what applies; version
bumps are semver-checked.

## Dev commands

- Engine docs: `~/tmp/abstractframework/abstracttui/docs/api.md`
  (+ `examples/` — `shell.rs` is the app-shell reference).
- AbstractCore venv python: the framework venv at
  `~/tmp/abstractframework/.venv/bin/python` (`abstractcore` CLI on its
  PATH); verify with `abstractcore config defaults --json`.
- Run: `cargo run` in this directory; headless guard pattern per the
  sibling apps (exit 0 when stdout is not a tty).
- NEVER run git. NEVER edit outside this directory (findings go to the
  bands above).
