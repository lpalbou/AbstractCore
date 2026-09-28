# Changelog — abstractcore-console

All notable changes, one entry per build wave, each with its gate line
(build/test/clippy state at the wave's close). Charter:
`LAUNCH-PROMPT.md`.

## [Unreleased]

## [0.4.0] — 2026-09-28

Released with AbstractCore 2.18.0. A minor bump for `cargo semver-checks`, but
`Opener`'s error type changed, which breaks hosts that supply their own opener.

### Added

- `OpenError` (`NoDisplay { why }`, `Refused`, `Failed`), `display_available()`
  and `display_from(os, env)`: an SSH session (`SSH_CONNECTION`, `SSH_CLIENT`,
  `SSH_TTY`) or Linux/BSD with neither `DISPLAY` nor `WAYLAND_DISPLAY` has no
  display, and `system_open` launches nothing there.
- Routes: `engine_missing` (AbstractCore 2.18.0) reads `engine not installed`; the
  detail line gives Core's reason with the install command, and the Engines
  row that installs it. The apply-recommended report says which engine a
  written route still needs.
- Models: fit `needs GPU limit` with the exact `sudo sysctl
  iogpu.wired_limit_mb=…` command on the detail line (`ArtifactRow::gpu_limit`,
  `GpuLimit`).

### Changed

- **Breaking:** `Opener` is `Rc<dyn Fn(&str) -> Result<(), OpenError>>`, and
  `ScreensCtx::open_url` returns that result. On `NoDisplay` the notice reads
  "open <url> on your computer — no display on this machine (<why>)".
- Engines: a `remote_only` engine (vLLM without CUDA) reads `remote only`, the
  kind column is in words, and the detail line prints the engine's own reason.

### Migrating from 0.3

- A custom opener returns `Err(OpenError::Failed(msg))` where it returned
  `Err(msg)`; handle or explicitly ignore the `Result` from `open_url`.

Gate: `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `cargo test`
(stable and MSRV 1.87) green.

## [0.3.0] — 2026-09-27

Terminal-console parity with the gateway's web console for the shared
**Models** (9) and **Engines** (0) screens. **Breaking** for library
hosts (the gateway console): see *Migrating from 0.2* below.

### Models and Engines

- **Optional transport verbs** (`TransportCaps`, all off by default,
  each answering `Unsupported` until a transport opts in): Hugging Face
  search (`h`), the default text model (`u`), the downloads feed (the
  Downloads view, `v`), app-engine install location (just me / all
  users), continuing a paused install (`a`, `y` copies its command) and
  engine servers (`s` start/stop). The footer says "not here" for a verb
  the backend lacks.
- **Downloads run in parallel**; a paused install (`needs_admin` /
  `needs_tools`) blocks nothing. `c` asks before stopping a download
  (the web's two-step cancel) and cancels it on the download route
  (`ConsoleTransport::cancel_download`).
- **Every download is polled on the download route**
  (`ConsoleTransport::download_job`, default `job`), a "download all"
  group and its children adopted from the feed included.
- **Admin-only verbs** (`Access`): `w d u c` on Models and `i s a c` on
  Engines follow a host-owned `Signal<Access>`. Read-only: each is
  refused first, before any capability or state answer, with the web
  console's sentence plus the host's reason
  ("only an admin can download models — signed in as ana, not an
  admin"), the footer says "admin only", browsing stays open. The core
  console is the local operator: `Access::Admin`.
- **`t` filters by model type** — text, thinking, tools, vision, audio,
  embedding, voice, image, video: the web console's capability chips
  (`capabilities.video_generation` included), filtered locally.
- **`w` sends the catalog's size** (`expected_bytes`) when the catalog
  vouches for it (`size_source` `catalog`/`hf_api`), so the gateway
  pre-checks the disk. The CLI transport has no such flag to pass.
- **`u` sets the route to exactly the chosen model**: the previous
  model's `base_url`, `reasoning` and `options` are cleared.
- **`q` refuses only while a job is WORKING**
  (`ScreensStore::job_running`); a paused install never holds it.
- **An empty or failed list keeps the screen's keys**
  (`screens::focus_holder`): `v` leaves an empty installed list, `r`
  retries a failed engines read.

### Capability routes

- **`a` (apply recommended) journals the CLI's report**: the write runs
  `config apply-recommended [--force] --json` and every route's outcome
  lands in the journal and the notice — applied, kept (with what was
  recommended), already, and routes this computer cannot run with the
  reason ("output.video: nothing recommended runs on this computer —
  …; left unset"), routes `--force` `cleared` (a broken route with
  nothing runnable to replace it), and the report's totals (`cleared`
  included). The prompt names no route list: Core owns which routes it
  recommends.
- **Routes this computer cannot run are flagged** (optional Core fields):
  a configured row with `route_unavailable` reads `cannot run here` and
  its detail line says why (Enter edits it, x clears it); an unset row
  with `recommendation_unavailable` says which recommendation cannot run
  and why.

### Migrating from 0.2

- `ScreensCtx::new(cx, transport, overlays, access, options)`: pass a
  `Signal<Access>` kept current from the signed-in principal.
- `ConsoleTransport::start_download(provider, artifact, expected_bytes)`:
  forward `expected_bytes` (gateway body field `expected_bytes`).
- Implement `download_job` for a backend whose job route does not know
  download groups (gateway `GET /models/download/{id}`).
- `set_text_default` clears the route's `base_url`, `reasoning` and
  `options` (gateway body `{provider, model, base_url: "", reasoning:
  "", options: {}}`).
- `ScreensStore::job_active` → `job_running` (paused installs excluded).
- `catalog::hints(caps, &access)` / `engines::hints(caps, &access)`;
  `HINTS` gained `t`.
- `#[non_exhaustive]`: `TransportErrorKind`, `CatalogView`, `ScreenCmd`,
  `InstallLocation`, `WriteVerb`, `Access` (match with a wildcard arm);
  every `ScreenCmd` struct variant (drive the lane through the
  `ScreensCtx` methods); `store::RouteRow` and `store::RouteUnavailable`;
  `TransportCaps` (start from `ALL`/`default()` and set fields),
  `JobPoll` (`JobPoll::new`), `ScreensStore` and every parsed contract
  view in `screens::data` (read fields; the parsers build them).
  `ScreensOptions` stays constructible with `..Default::default()`.

Gate: `cargo build`, `cargo test --locked` (lib 90, headless 88 + 1 ignored,
cli_transport 1, doc 5), `cargo clippy --all-targets -D warnings` and
`cargo fmt --check` clean; `cargo semver-checks` vs 0.2.0: major
(6 major lints, expected for 0.3.0; 0 minor).

## [0.2.0] — 2026-09-23

First crates.io release (`cargo install abstractcore-console`). The crate
is now a LIBRARY too: the shared **Models** and **Engines** screens that
the gateway console mounts over its own transport.

### Models and Engines, one implementation for both consoles

- **Library API (contract H).** `transport::ConsoleTransport` (host
  profile, engines status, catalog, installed, download, delete, engine
  install, job, cancel) returning the contract A–E JSON documents, with a
  classified `TransportError` (`Refused` carries the backend's blockers).
  `screens::catalog()` and `screens::engines()` are plain page builders
  over `ScreensCtx` (store signals + the screens' own worker lane); the
  job-poll helper `schedule_job_poll` re-sends `PollJob` from a timer
  thread, the gateway console's `schedule_poll` pattern.
- **`CliTransport`** drives `abstractcore … --json` on this machine. Exit
  code 2 maps to *refused*. Downloads and real engine installs run as
  child processes the transport owns (the job registry lives inside one
  Python process, so a second CLI call could not poll it): NDJSON
  `host_job_v1` lines update progress, stderr feeds the log tail, `c`
  terminates the child. Ids starting with `-` are refused, never passed.
- **Screen 9 — Models** (`catalog`): the catalog with host profile, fit
  badges (`fits / tight / too large / partial offload / unknown`) and
  weight labels (`installed / not downloaded / unknown / remote`). `w`
  downloads (one key; a confirm only when the model is too large or the
  disk is short), `d` deletes after a confirm that spells out blockers
  (`loaded`, shared cache; force is a separate danger answer), `/`
  filters, `f` fits-only, `e` cycles the engine, `v` flips to what is
  installed, `c` cancels the running job.
- **Screen 0 — Engines** (`engines`): Ollama, LM Studio, MLX, llama.cpp,
  vLLM, Hugging Face — installed, version, server state, models. `i`
  installs after a confirm showing the exact argv, the host it runs on
  and the sudo/UAC notes, with a dry-run answer; `o` opens the vendor's
  download page; `r` probes the local servers.
- One job at a time, shown in a progress strip on both screens; its
  outcome lands as a toast, and a finished download/delete/install
  re-reads what it changed. `q` refuses while a job runs.

- **Routes-screen downloads read the streamed job.** `abstractcore models
  download … --json` (abstractcore 2.14.0) prints one `host_job_v1` line
  per progress step; the routes screen's `w` download now keeps the final
  line (`CoreCli::run_json_last`) and reports a failed job's own error.

### Platform

- abstracttui 0.3.0 → **0.3.6**, the version abstractgateway-console
  builds on (the two crates must share one engine). No API breakage.
- `cargo fmt` applied across the crate; CI now gates fmt, clippy
  `-D warnings`, `cargo test --locked` and the 1.87 MSRV build.
- Screens 1–9 keep their digits; the tenth screen is `0`. The footer and
  `--help` say `1-9, 0`.

Gate: `cargo test --locked` 80 lib + 73 headless + 1 CLI-transport + 5
doc tests green, `cargo clippy --all-targets -D warnings` clean,
`cargo fmt --check` clean, `cargo publish --dry-run --locked` passes.

### Earlier waves (0.1.x, never published)

### The weights banner stops warning a configured machine (2026-09-06, operator ruling)

> "I do not like that it shows 'Recommended defaults — 2 of 3 models
> present. Missing: lmstudio qwen/qwen3.5-9b@4bit' when we already have
> models installed. The recommendations are for a clean fresh system with
> no detected models, not for an already configured system. Otherwise it
> appears like an error / something to do — and I will never install it
> since I have better locally."

The Routes weights line read `recommended models: 2 of 3 present ·
missing: lmstudio qwen/qwen3.5-9b@4bit` in warn amber on a machine whose
every route was answered. A warning whose only cure is installing the
model you chose against is noise, and noise on the healthy path teaches
an operator to skip the line that matters.

`abstractcore models status --json` now carries `recommended.gaps` — the
recommended models that are absent AND whose route has nothing else
serving it (`mark_recommended_route_gaps`). The banner reads those and
nothing else: `1 route with no model yet · input.text · recommended:
lmstudio qwen/qwen3.5-9b@4bit · w downloads the selected route's
weights`, and it paints nothing at all when there are no gaps.
`would_download` remains the fallback for an older `abstractcore`.

Gate: `cargo test` 71 lib + 60 headless green, `cargo check --tests`
clean.

### One providers screen, two doors (2026-08-01, operator ruling)

> "I do not understand why the providers are displayed in a different
> fashion between gateway and core; they should have the exact same.
> Gateway is the one we want. Profiles are just indicated as profile of
> the openai-compatible endpoint, and we should have a way to configure
> as many as necessary, like in the gateway console."

The Providers screen was TWO tables in two vocabularies: a provider
inventory (`provider | kind | key/endpoint | answering`, "local
server", "nothing to configure") and, under it, a second table listing
the endpoint profiles again — the same objects, twice. It is one table
now, cell for cell the gateway console-TUI's:

- **`provider | family | base URL | API key | models | enabled |
  origin`**, `family` and `models` appearing at ≥104 cells like the
  gateway's; widths still solved by `ui::widths` (untouched).
- **Endpoint profiles are rows**, inline as `endpoint:<id>`, carrying
  the family they are a profile OF (`openai-compatible`). The second
  table is gone; its editing moved onto those rows.
  `ProfilesData::connections()` composes the list from core's own
  surfaces — the `config providers --probe --json` inventory joined
  with the `provider_profiles` rows — so the screen renders rows
  instead of performing a join, and the composition is unit-tested.
- **`origin` says where a row lives**: `config` (this file — an
  endpoint profile, or a key in `api_keys`) · `env` (resolved from the
  environment) · `auto` (a local server that ANSWERED at its default
  address) · `registry` (known, nothing configured). Key precedence is
  now readable per row instead of being a footnote under the table.
- **Verbs, the gateway's set, per row**: `a` adds a connection (a real
  `provider_profiles` entry through `config set-provider` — as many as
  wanted), `e`/Enter/double-click configures the selected row (profile
  editor, or the masked `api_keys` editor for a keyed builtin), `d`
  deletes a stored connection, `m` lists its models, `t` probes it.
  `k` stays the explicit key verb. Every refusal names the row and the
  reason. `t` could not be selected-row before — the old top table WAS
  the `api_keys` section, so keyless lmstudio/ollama had no row to
  select and the verb had to open a picker over a list the screen did
  not show; the picker is gone with the reason for it.
- **Footer, gateway-shaped**: the selected row's summary, the
  configured `core default: <provider> / <model>`, and the
  `not configured yet (k sets a key · a adds a connection): …` line.
- **Adapted honestly, not faked**: `enabled` prints `—` on a registry
  row (core has no enable switch for a builtin — `yes` beside a column
  whose other value is `NO` would advertise a toggle with nothing to
  write to), and `origin: registry` is a row the gateway does not have
  at all (core keeps it: ruling 2026-08-01, "how come we don't have
  ollama, lmstudio, huggingface and mlx?").

Two defects the live pty run caught, both fixed here:

- **The post-write refresh dropped `--probe`.** It replaces the whole
  providers view, so a probe-less re-read silently downgraded every
  probed row after any write: live model counts vanished and `origin`
  flipped `auto` → `registry` — a wrong word about a server that was
  still answering.
- **"Python REFUSES this file", one keystroke after a successful add.**
  `config set-provider` stamps `scope` + `capabilities` onto every
  profile row it saves (ONE STORE FOR PROVIDER CONFIG,
  `provider_profiles.py:140-165`); the console's `PROFILE_FIELDS` did
  not know them, so its refusal detector flagged a file Python had just
  written. Both fields are now known (Core reads neither; nothing here
  edits them).

Gates: `cargo build` · `cargo test` 130 passed (71 lib + 59 headless) ·
`cargo clippy --all-targets -D warnings` clean · live pty round-trip
against the real store (`a` → `endpoint:paritytest` appears → `d` →
disappears), store restored byte-identical (sha256 `28fe8d4d…`).

### One config language, two doors (2026-08-01, operator ruling)

Harmonized with `abstractgateway-console` for the configuration the two
entry points SHARE. The gateway console is the reference; where a better
shared pattern emerged it was applied to both.

- **One state vocabulary** on the routes grid, `RouteRow::state_label()`
  in both crates, same four strings: `configured` · `covered by <key>` ·
  `derived ← <key>` · `not configured`. The old spellings (`= input.text`,
  `default`) are gone; `default` read as "a default is set" on a screen
  literally about defaults, so BOTH consoles now say `not configured`.
- **Derived-ness comes from the payload**, not from a hardcoded key.
  `RouteRow` folds `derived_from` and `editable()` reads it (the gateway
  console's exact body), so a second derived row needs no console edit.
  Refusals speak the same sentence in both: `output.text derives from
  input.text — edit that route instead`.
- **Source attribution per row**: a `source` column at ≥112 cells, the
  payload's own string — gateway parity.
- **The editor leads with the stored truth**: `Applies now: <provider> /
  <model> · reasoning <effort> (source: …)`, derived only from what the
  store answers, never from local picks. `Clear override` now lives in
  the editor too, behind the same danger confirm as the table's `x`.
- **Reasoning is a select, not free text** — `not set / minimal / low /
  medium / high`, the web console's list verbatim, and only on the
  text-generation route (`input.text` / `output.text`), where the store
  keeps it. Deliberately NOT the engine's `ReasoningSelect`: that control
  is capability-driven and renders locked without `ReasoningFacts`, which
  a route editor does not have.
- **The save sends only what the operator edited.** `set-default` is
  field-preserving (`update_capability_default` keeps every field a
  command does not name), so the editor diffs each control against the
  value it opened with: an untouched field carries no flag, an emptied
  one carries an empty flag (`--base-url ""`) that clears it, and options
  send `--option ""` to drop them all. Echoing the rendered row back as a
  write would let a stale grid overwrite a value set from the gateway
  console between render and save. An untouched Save refuses instead
  ("nothing changed"). `writes::RouteEdit` carries the diff;
  `set_route_sends_only_the_edited_fields` pins the argv.
- **The write proves itself with the resulting row**: `input.text =
  endpoint:airelay / gpt-5.4 · reasoning medium (source: …)` — the same
  sentence the gateway console prints after its PUT. `Expect::RouteEq`
  gained `reasoning` and now verifies exactly the fields the write named.
- **Providers screen speaks the gateway's connection vocabulary**: the
  API-key cell is `stored (fp)` / `none` / `none ($VAR)`, and the first
  column is the PROVIDER NAME (`endpoint:<id>`) — the spelling a route's
  provider field takes, so the Providers screen and the route editor's
  dropdown stop naming the same thing two ways.
- **Fixed: the embeddings model filter never fired.**
  `class_for_modality` read the row's modality, but `embedding.text`
  carries kind `embedding` and modality `text` — so the one picker whose
  class filter earns its keep offered chat models. Now
  `class_for_route(kind, modality)`, pinned against the live row shapes.

Gate: `cargo build`, `cargo test` (57 lib + 46 headless), `cargo clippy
--all-targets` all clean. Live: driven over a pty against
`runtime/config/abstractcore.json` — set a reasoning default from this
console and from the gateway console, journal and store verified after
each, config restored byte-identical to its pre-session state.

### Class-filtered model pickers (2026-07-26, operator request)

- **Selecting a provider now populates a real model dropdown** in the
  pair and route editors: the model row becomes a Combobox over live
  discovery, FILTERED to the class the field is for — embedding models
  for the embeddings pair and `embedding.*` routes, generative models
  (embedding-shaped names excluded) everywhere else. Discovery gives
  names only, so the class is a name heuristic (`src/models.rs`:
  *embed*, minilm, bge, gte, e5, sentence-transformers families) —
  hidden models are COUNTED on the status line, a whiffed filter falls
  back to the full list (labeled), and finer classes (vision vs text)
  are honestly not pretended.
- **Prefilled providers kick discovery at OPEN** (both editors): the
  picker is populated by the time the operator reaches it, not only
  after re-committing a provider they already had. Editing an existing
  pair also prefills provider + model now (the editor used to open
  blank over a set pair).
- **Free typing survives**: while discovery is loading/failed/absent
  the row stays a TextInput with the state named; the picker's last
  option (`✎ type a custom id…`) and `c` flip to custom typing;
  Ctrl+P returns to the picker — an undiscovered or heuristic-missed
  id can always be entered. A prefilled value discovery doesn't list
  keeps the row in custom mode instead of misrepresenting it.
- Engine-behavior fix caught live (pyte-rendered pty bisect): the row's
  dyn_view TRACKED the model value, so committing from the popup
  regenerated the row and destroyed the focused Combobox — Tab then
  landed on the provider Select instead of Save. The value is now read
  untracked (the Combobox owns its display); commits keep focus.
- `scripts/definition_of_done.py` drives both editors through the real
  picker now (populate → type-to-filter → commit → save).

Gate: build green · 56 unit + 37 headless (+1 ignored minter) · clippy
zero · full pty smoke green · definition-of-done walk green (picker
flow live against LM Studio, both editors).

### M3 — test & prove (2026-07-25)

- **The probe lane** (`probes.rs` + the worker): three test verbs with
  honest three-state verdicts (proven / NOT PROVEN / failed) —
  - `t` (Providers): a provider test picker over ALL 10 canonical
    providers + every endpoint profile (the api_keys table alone could
    never reach keyless lmstudio/ollama) → live model discovery via
    `config test-provider --json`.
  - `t` (Routes): the selected route's model must be AMONG what the
    provider actually serves — capability-agnostic membership (voice/
    image routes can't be chat-tested; model existence always can).
  - `g` (anywhere): one cheap generation over the CONFIGURED default
    route via `abstractcore-chat --prompt`, with a local pre-check —
    probed: the chat CLI on an empty config silently invents a
    huggingface default, so testing without a configured route would
    lie about YOUR route.
- **The CLI's third liar class, folded honestly**: `test-provider`
  answers `ok:true, count:0, errors:[]` against a DEAD server
  (live-probed), and `abstractcore-chat` exits 0 printing `❌ Error:`
  on failures. Zero-count success folds to NOT PROVEN — upgraded to a
  named cause by a TCP reachability check on KNOWN endpoints only
  (profile base_url or the ollama/lmstudio local defaults; never
  https, never cloud). The TCP evidence LEADS the message (notices
  truncate from the right).
- **Review is the evidence surface**: latest result per target
  (re-tests replace), verdict-colored, with a teaching empty state;
  probe results also land in the session journal. Probes are
  single-flight (a queued duplicate would silently double the cost).
- Wizard review step teaches `g`; footer hints carry the test verbs.
- `scripts/definition_of_done.py`: the chartered end-to-end walk,
  live-green — fresh machine → wizard boots → default set to
  lmstudio via the pair editor (coupled CLI write) → `g` generation
  PROVEN → wizard finish → `config defaults --json` agrees → browse
  edits route input.text to another live model → `t` membership
  PROVEN → Python re-read agrees.

Gate: build green · 50 unit + 35 headless (+1 ignored minter) ·
clippy zero · full pty smoke green (7 phases incl. the M3 test-verb
phase against the real LM Studio) · negative lanes live-verified
(dead ollama → NOT PROVEN with "looks DOWN"; empty config `g` →
honest refusal) · definition-of-done walk green end-to-end ·
captures: `docs/captures/m3-review-evidence.svg`.

Adversarial review (fable5, `docs/reviews/m3-adversarial-review.md`:
2 P1 / 4 P2 / 13 P3, both P1s live-proven) — fix wave applied same
day:

- P1-1 (`g` on an `endpoint:<id>` default route reported ✗ FAILED for
  a WORKING route — the chat CLI's argparse knows no endpoint
  providers): keyless profiles now expand to `--provider <family>
  --base-url <url>` (expansion disclosed in the evidence as
  "via …"); keyed/disabled profiles refuse honestly as NOT PROVEN
  (argv never carries secrets; a guessed key lane would mint 401
  lies); a default naming a missing profile is Failed. Pinned with an
  argv-proving fake chat + live re-proven (endpoint default over the
  real LM Studio → ✓ PROVEN).
- P1-2 (TCP disambiguation probed only the FIRST resolved address —
  `localhost` resolves `::1` first, local servers often bind IPv4
  only, so an UP server read "looks DOWN"): reachability now judges
  ALL resolved addresses (Connected if any accepts; Refused prefers
  the IPv4 error text), extracted pure and pinned with real
  listeners.
- P2 fixes: `is_log_line` no longer panics on multibyte model output
  (`get(8..)`, pinned); the single-flight guard latches
  `probe_busy` SYNCHRONOUSLY at send (queued probes behind a busy
  worker were invisible to it); route-test evidence labels are
  pair-free so re-tests after edits supersede stale rows (pinned);
  the Routes lane resolves an endpoint route's PROFILE base_url for
  reach parity with the Providers picker (pinned).
- P3 wave: CliError carries the failing PROGRAM (chat failures no
  longer wear "abstractcore"); PATH-resolved chat binaries are
  disclosed in evidence; userinfo URLs refused (secret-shaped hosts
  never render); Proven derives from the MODELS LIST, never the count
  field; journal renders NOT PROVEN under `?`, not `✗`; keyed-cloud
  zero-listing names the likely no-key cause; "N models available"
  (hf/mlx list caches, nothing is "served"); evidence overflow says
  "… and N older"; route detail puts the pair AFTER the cause;
  RouteEq failure wording aligned; DoD warns on ambient
  `ABSTRACTCORE_BIN`; README teaches the `g`-vs-`app_defaults.cli`
  distinction; fold_generation's stdout-scraping cost documented.
- Audit adoption: the smoke's M3 phase gained the automated NEGATIVE
  lane (ollama → NOT PROVEN with the TCP cause; environment-tolerant
  if ollama is up); tcp reachability + endpoint-generation lanes have
  unit pins; the review-evidence capture now shows all three verdicts
  plus a route-membership failure.

Gate after fixes: build green · 53 unit + 36 headless (+1 ignored
minter) · clippy zero · full pty smoke green incl. the negative lane ·
definition-of-done walk green · P1-1 endpoint repro live-verified ✓.

### M2 — edit + wizard (2026-07-25)

- **The write lane** (`writes.rs` + the worker's three-phase
  execution): every editable surface writes through a `WriteSpec` —
  CLI setters for every coupled field (global default ↔ route
  input.text, embeddings ↔ embedding.text, audio strategy ↔ the
  explicit flag, vision pair), direct read-modify-write (fresh read →
  mutate → unique tmp + rename + 0600, unknown keys preserved by
  construction) only for the CLI-less fields. Every spec carries
  value-level expectations verified against a FRESH re-read (+ fresh
  derived views for route/profile writes) — load-bearing, since both
  of the CLI's success signals lie (probed: `--set-*` exits 0 on
  refusals; `--set-app-default` prints ✅ for dropped writes; the
  worker tests pin both liar classes against fake CLIs).
- **Write refusals are structural**: corrupt/unreadable files refuse
  all writes; a Python-refused file (P1-1 class) refuses CLI verbs
  specifically (a setter against it would RESET the file to defaults —
  the historical incident, executed by us) while unknown-key-preserving
  RMW stays allowed; a drift guard refuses when the file changed since
  the operator loaded it (no lock exists; last-writer-wins).
- **Typed editors** (`ui/editors.rs` + per-screen verbs): scalar/enum/
  toggle (UNSAFE flags get danger confirms) / masked secret (blank
  keeps, explicit clear, fingerprint verify) / provider+model pair
  editors with live model discovery (`config models P --json`) /
  vision strategy + fallback-chain editors / route editor (options as
  k=v, coverage-aware refusals) / profile editor. Section pages became
  editable field tables (Enter/e edits, x clears) with a pinned
  selected-row truth line.
- **Wizard mode** (`ui/wizard.rs`): a guided walk covering the CLI
  wizard's 8 phases in its order (default model → vision → API keys →
  server → audio → video → embeddings → logging) plus orientation and
  review; section pages filter to the step's focus; digits/free-nav
  disarmed with reasons; f finishes; w re-enters; adaptive default
  (wizard on a machine with no config file, browse otherwise;
  --wizard/--browse override).
- Ctrl+L moved to the GLOBAL action registry so the repaint stays live
  inside focus-trapped modals; forms follow the sibling console's
  plumbing (single modal slot, dirty-Esc guard, write_done routing,
  message slot).
- Engine finding filed: field-core 1100 (TextInput cannot open with
  the cursor at the end — prefilled editors insert at position 0).

Gate: build green · 33 unit + 31 headless tests (+1 ignored capture
minter) · clippy zero · pty smoke green end-to-end INCLUDING the M2
write phase (fresh scratch config → wizard boot → editor → real
`abstractcore` setter → file created with the value → Python-side
`defaults --json` reads it cleanly).

Adversarial review (fable5, `docs/reviews/m2-adversarial-review.md`:
1 P1 / 6 P2 / 12 P3) — fix wave applied same day:

- P1-1 (array-index expectations never evaluate): the expectation
  walker could not index arrays, so every fallback-chain write reported
  failure AFTER landing (retries appended silent duplicates) and remove
  verification was vacuous. Fixed with numeric-segment array walking;
  pinned at unit level, through `execute_write` with the real chain
  builders, in a headless chain-editor test, and re-proven live with
  the reviewer's repro (journal now `✓`, file ground truth agrees).
- P2 fixes: "disabled" vision strategy names its blast radius in a
  danger confirm (edit and clear paths); `clear_global_default` runs
  CLI-first with a CLI-presence pre-check; the UI door now honors the
  refused-file CLI/RMW split the worker already had; audio.strategy
  clear resets value + explicit flag (both spellings); route editor
  round-trips `reasoning`; chain length folds from the model
  (`list_len`), not the display string.
- P3 wave: per-form base stamps + reactive "applies now" lines;
  `FileStamp` (mtime, ino, size) drift identity; quit refuses
  mid-write; leading-dash secret refusal; options round-trip with
  quoted values; empty-string prefill; clear-on-missing-file refusal;
  dead plumbing removed + `reset_domains` wired into reload; dirty
  guards track non-text fields; pair truth line; chain-add in-flight
  guard; vision-clear refuse-before-confirm ordering.
- Post-review: successful writes no longer refresh `providers --json`
  unless the spec touches profiles — the unconditional refresh added
  5-15s of CLI tail to every write; pty smoke `wait_fresh` reads until
  the frame settles (fixed windows truncated repaints under load).

Gate after fixes: build green · 42 unit + 32 headless (+1 ignored
minter) · clippy zero · pty smoke green incl. write phase · P1
regression live-verified · M2 captures minted
(`docs/captures/m2-{wizard-model,editor}.svg`).

### M1 — the honest mirror (2026-07-25)

- Shell: PageHost over 8 screens (Overview, Model, Providers, Routes,
  Media, Embeddings, Server, Review), pinned chrome (header/footer
  `shrink(0.0)` from day one), footer with busy strip, app notices,
  engine startup notices (humanized), key hints, and a ThemeSwitcher.
- Config mirror: direct parse of `abstractcore.json` (path resolution
  honoring `ABSTRACTCORE_CONFIG_FILE`/`_DIR`), folded to a redacted
  display model — per-field set/default/broken states against the
  documented dataclass defaults, secrets as sha256[:8] fingerprints,
  unknown sections/keys surfaced with the Python drop-on-save warning,
  corrupt files refused with backups listed (never rewritten).
- Derived views: `abstractcore config defaults --json` (routes +
  coverage) and `config providers --json` (redacted profiles) via one
  worker thread; config-file identity cross-checked between the CLI
  echo and the console's own path.
- Read-only: no write paths in this milestone.

Adversarial review (fable5, `docs/reviews/m1-adversarial-review.md`:
1 P1 / 4 P2 / 14 P3, zero engine defects) — fix wave applied same day:

- P1-1 (the mirror vouched for files Python refuses): the fold now
  models Python's ONLY loader raise-surface (profile-row construction:
  unknown fields, non-dict rows, invalid id/family/base_url/env-var,
  including the `profiles`-key-absent quirk) and flags the file; the
  CLI client surfaces `#FALLBACK` stderr from exit-0 runs as a loud
  banner; the header, agreement line and Overview all refuse to vouch.
  Live-verified end-to-end against a scratch config Python refuses.
- P2 fixes: api_keys broken states visible on the Overview; env config
  paths expanduser'd with exact Python truthiness; empty-string
  secrets classify as not-set. P2-2 is PARTIAL by design: the
  profiles half (the raising one) is covered; a malformed-shape
  advisory for the two tolerant special sections is deferred to M2.
- All 14 P3s fixed (EMPTY fingerprint canonicalization, nullable
  app_defaults/embeddings fields, float-typed ints, flag truthiness,
  route counting, dead ureq removed, README tense, Scroll autofocus,
  directory hint, cell-exact padding, option-value masking, $VAR
  resolution honesty, panic busy-leak, component-wise path compare) +
  header re-ordered state-first (a truncated CORRUPT flag is a lying
  header).
- Test additions per the review's suite audit: refusal banners (both
  lanes), unreadable, DIFFERENT-FILES + same-path guard, api_keys
  broken, negative command-queue assert, corrupt refusal at 60x16,
  exit-0-stderr surfacing, non-object JSON, directory-at-path.

Gate: build green · 21 unit + 22 headless tests + 1 ignored capture
minter · clippy zero · pty smoke green on the real machine · P1
regression live-verified.
