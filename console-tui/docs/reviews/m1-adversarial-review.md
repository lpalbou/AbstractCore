# M1 adversarial review — abstractcore-console (the honest mirror)

- Date: 2026-07-25
- Reviewer: adversarial spec-conformance + edge-case audit (M1 charter scope)
- Baseline: working tree at review time; `abstracttui` resolves to exactly 0.2.22
  (Cargo.lock); Python side `abstractcore 2.13.38` (both the pyenv PATH install and
  the framework venv — no version skew on this machine).
- Gates re-run by the reviewer: `cargo build --all-targets` green, `cargo test`
  15 passed / 1 ignored, `cargo clippy --all-targets` zero warnings,
  `python3 scripts/pty_smoke.py` all gates passed against the real machine config.
- Live probes were run in a scratch dir under `target/` (deleted after) with
  `ABSTRACTCORE_CONFIG_FILE` pointed at scratch files; the real
  `~/.abstractcore/config/abstractcore.json` was read only through the app/CLI and
  never modified. No secret values appear in this report.

## Verdict

M1 is a solid, honestly-built read-only mirror: the worker/signal architecture is
implemented as chartered, secrets are structurally redacted at parse time (verified
at the model level, the screen level, and by code read), the chrome matrix and pty
smoke both pass, and the 85-field/17-section schema matches the Python dataclasses
byte-for-byte on names and defaults. But the headline claim — "the honest mirror" —
has one genuine hole, proven live: the console's definition of "corrupt" is
narrower than Python's. A valid-JSON config whose `provider_profiles` row carries
one unknown key (the exact forward-compat scenario the charter's fact #1 exists
for) renders as `loaded ●` / `✓ same file` / fields "set", while the Python side
treats that same file as whole-file corrupt — silently loading defaults, minting a
new `.corrupt-*.bak` on every CLI invocation (two per console reload), and
emitting the one honest signal (`#FALLBACK` on stderr, exit 0) into a pipe the
console captures and discards. In that state the mirror shows values Python does
not run with, and the operator it reassures is one Python-side `--set-*` away from
the historical data-loss incident. That, plus a cluster of smaller
state-classification deviations (api_keys broken-state invisible on the Overview,
empty-string secrets counted as Set, no `expanduser` on env paths), is the gap
between "renders honestly" and "is honest about what Python will do". Everything
else I attacked held.

## P1 findings

### P1-1 — Python-corrupt files render as loaded: the mirror can vouch for a config Python refuses

**Claim violated**: charter M1 "honest mirror … what is broken"; charter fact #4
(corrupt = stop, point at backups); README "a corrupt config file is a hard stop".

The console treats exactly two states as corrupt: JSON parse failure and
non-object root (`src/config.rs:186-199`). Python's corrupt path is wider:
`_load_config` wraps **the whole dict-to-config conversion** in its backup-and-
default fallback (`abstractcore/config/manager.py:510-528`), and
`provider_profiles_from_dict` RAISES on real-world shapes:

- a profile row with an unknown key — `ProviderProfile(**payload)` has no kwargs
  filtering, unlike every other section (`provider_profiles.py:224-229` vs
  `manager.py:368-372`);
- a non-dict profile row (`provider_profiles.py:225-226`);
- an invalid id / family / base_url / env-var name
  (`provider_profiles.py:38-49, 63-68, 71-77, 87-93`).

**Live proof** (scratch file, this machine, 2026-07-25): a config containing one
profile row with `"future_field": true` plus `"video": {"max_frames": 7}`:

```
abstractcore config providers --json   → exit 0, ok: true, profiles: []
stderr: "#FALLBACK abstractcore config at …/a.json could not be parsed
 (ProviderProfile.__init__() got an unexpected keyword argument 'future_field');
 falling back to DEFAULTS for this session. The unreadable file was backed up to
 …/a.json.corrupt-20260725-184424.bak …"
```

A new `.corrupt-<stamp>.bak` is minted **on every load** — the console runs two
CLI subprocesses per reload (`src/worker.rs:88-115`), so every press of `r`
silently creates two more backup files beside the operator's config.

What the console shows for that file: header `● loaded`; Overview
`provider_profiles · ● 1 set`; `video · ● 1 set` with `max_frames=7` — while
Python runs `max_frames=3` and zero profiles (the corrupt fallback discards the
WHOLE file, not just the bad row). The agreement line shows `✓ the abstractcore
CLI reads the same file` (`src/ui/overview.rs:107-115`) because it compares only
path strings, and the derived views show `ok: true` (the CLI hardcodes it,
`main.py:1442-1452`) — body-over-transport cannot catch this because the body
lies too. The only honest signal is the `#FALLBACK` stderr line of an exit-0
subprocess, which `run_json` reads and then drops on the success path
(`src/cli.rs:182-195` — stderr is only consulted inside `error_line` on nonzero
exits).

**Why this is the charter's own scenario**: fact #1 exists because "a NEWER
abstractcore than the TUI's model will have fields the TUI doesn't know". The
same skew in the other direction — a newer abstractcore wrote a profile field,
the operator downgraded, or a hand-edit added one — is this repro. The mirror's
job in that state is to say "Python cannot load this file"; instead it vouches
for it, and every value it shows for every section is wrong relative to the
running Python.

**Remediation options** (any one restores honesty; (b) is one line of leverage):
(a) mirror Python's profile-row strictness in the fold — rows must be objects
with only the 11 known profile fields and valid id/family/base_url/env-var, else
mark the section (and the file) "Python will refuse this file"; (b) surface
stderr `#FALLBACK`/`WARNING` lines from exit-0 CLI runs as a loud notice — that
catches this entire divergence class forever, including ones Rust-side
validation will never predict; (c) cross-check `profiles_in_file > 0` against a
Ready-and-empty derived view and render the contradiction as a warning instead
of two quietly disagreeing rows.

## P2 findings

### P2-1 — Overview api_keys row drops the Broken state entirely

`src/ui/overview.rs:147-166`: the api_keys branch counts only `FieldState::Set`
fields; `broken_count` is never consulted. `"api_keys": {"openai": 12345}` (the
non-string edge from the schema's own validation) renders the row as `· none /
no keys stored` — claim 1 says the Overview shows every section's
set/default/broken state, and this section is the one holding secrets. The
Providers screen does show the broken row, so the workaround exists. The `of 7`
literal (`overview.rs:157`) will also silently drift if the key list ever grows.

### P2-2 — The two special sections can never show Broken anywhere

`SectionKind::Routes`/`Profiles` carry no field specs (`src/schema.rs:532-539`)
and the fold reads their bodies only through well-shaped accessors
(`src/config.rs:262-273`), so `"capability_defaults": "garbage"` or
`"provider_profiles": {"profiles": 3}` renders `· default` / `· none` with no
broken signal on any screen — while a string-valued **scalar** section marks
every field broken (`src/config.rs:306-318`). The two sections with the least
mirror validation are exactly the ones whose malformation Python either
silently empties (capability_defaults, `capability_defaults.py:290-310`) or
hard-refuses the whole file over (profiles, P1-1). Structural sibling of P1-1.

### P2-3 — Env-var config paths: no `expanduser`, so the mirror can read a different file than Python

Python expands `~` in both env vars (`manager.py:339-348` — `.expanduser()` on
every branch); the console does `PathBuf::from(file.trim())`
(`src/config.rs:51-62`). Live-proven: `ABSTRACTCORE_CONFIG_FILE='~/x.json'` →
CLI echoes `/Users/albou/x.json`; the console reads the literal relative path
`~/x.json` → renders Missing/built-in defaults for a file Python reads fine.
Literal tildes in env vars are how launchd plists, `.env` files, and quoted
shell assignments deliver paths. The agreement line catches the string mismatch
*after* the CLI loads, but the header state ("no config file yet") is wrong the
whole time. Claim 2's precedence chain (FILE > DIR > default) is correct; the
value handling is not Python's.

Related, same function: the doc comment "Blank env values count as unset
(Python truthiness)" is factually wrong about Python — only `""` is falsy;
whitespace is truthy. Live-proven: `ABSTRACTCORE_CONFIG_FILE='   '` → Python
uses a file literally named `'   '` (CLI echoed `config_file: '   '`); the
console falls through to the default path. Absurd input, but the comment claims
parity the code doesn't have.

### P2-4 — Empty-string secrets classify as Set while displaying "not set"

`fold_field` validates `""` as a legal Secret, then compares against
`Dv::Null` → `FieldState::Set` (`src/config.rs:344-357`); the renderer
trims and shows `not set` (`src/config.rs:378-384`). Result: one row that says
**set** (bold green state badge, counted into `● N of 7`) and **not set**
(value cell) simultaneously. Python treats empty/whitespace keys as not set
everywhere (truthiness at `manager.py:437` injection, `manager.py:933-942`
status). `--set-api-key openai ""` and hand-edits both produce this shape.
Whitespace-only secrets hit the same path.

## P3 findings

### P3-1 — Fingerprint convention is not "exactly Python's": the EMPTY fold is missing

Python fingerprints `normalize_api_key(value)`, which strips AND canonicalizes
any case variant of `EMPTY` to `"EMPTY"` (`provider_profiles.py:80-84,
120-124`); the console hashes the trimmed value only (`src/config.rs:167-172`).
Live: key `"empty"` → Python `cc1d2f83`, console `2e1cfa82`. `EMPTY` is the
documented placeholder for keyless endpoints, so case variants are plausible in
`api_keys.*` too (the shadow-warning fingerprints at `manager.py:445-455` are
the Python surface an operator would compare against). The inventory's own
wording ("sha256 of trimmed value") is imprecise about this. One `if
s.eq_ignore_ascii_case("EMPTY")` closes it.

### P3-2 — Null marks 12 legally-nullable fields as broken

`app_defaults` (all 10, schema `Str`, `src/schema.rs:295-306`) and
`embeddings.provider`/`model` (`Enum`/`Str`, `src/schema.rs:282-291`) are
`Optional[str]` in Python (`manager.py:129-149`) — `null` loads fine there but
renders `broken — expected a string, file has null` here. Python never writes
null to them, so it takes a hand-edit; still a false "broken" against claim 5's
"what Python actually enforces".

### P3-3 — Float-typed integers: `validate` and `Dv::matches` contradict each other

`Dv::I(3).matches(json!(3.0))` is deliberately true ("JSON does not distinguish
3 from 3.0", `src/schema.rs:71`), but validation runs first and
`int_in` uses `as_i64`, so `"max_frames": 3.0` renders broken
(`src/schema.rs:689-703`) — for a value Python loads without complaint (no type
enforcement at load; live-verified tolerance for the sibling case). Either
stance is defensible; holding both in one file is the defect.

### P3-4 — `audio_strategy_explicit` reading diverges on degenerate shapes

Python: key-presence check + `bool()` truthiness, and a present top-level key
suppresses the nested one entirely (`manager.py:514-521` — `"false"` → True;
top-level `null` → False, nested ignored). Console: `as_bool` with fall-through
(`src/config.rs:289-298` — top-level `null` + nested `true` → True; string
`"true"` → nested/false). Only reachable by hand-edits.

### P3-5 — `routes_in_file` overcounts against what Python loads

The fold counts raw keys in `capability_defaults.routes`
(`src/config.rs:262-267`); Python normalizes aliases, silently skips
unparseable keys, and drops unconfigured `{}` route values
(`capability_defaults.py:296-304`). A file with `"input.stt": {}` shows
`● 1 set` on the Overview for a section Python loads as empty. The derived view
corrects it when the CLI is up; the file-only fallback lane misleads.

### P3-6 — Dead `ureq` dependency in M1

`Cargo.toml:36` declares ureq (with TLS) for "optional server enrichment";
nothing under `src/` references it (grep-verified). Compile weight and
audit surface for a milestone whose charter is file+CLI only. Add it in the
milestone that uses it.

### P3-7 — README describes M2 write behavior in the present tense

README "Safety posture": "Direct file writes … preserve every unknown key and
go through tmp+rename with 0600" and "shells out to the abstractcore CLI for
derived views **and writes**" — no write path exists (claim 3, verified). The
CHANGELOG says read-only correctly; the README should too, or a reviewer
auditing M1 safety chases writes that aren't there.

### P3-8 — Section pages: arrow keys dead until Tab

Overview/Routes autofocus their tables (`src/ui/overview.rs:262`,
`src/ui/routes.rs:127`); Model/Media/Embeddings/Server pages have only a
non-autofocused `Scroll` (engine Scroll is focusable with arrows/PgUp/PgDn,
`abstracttui/src/widgets/scroll.rs:407-414,484`). First keystroke behavior
differs per screen; on a 24-row terminal the Server page (8 sections, ~70 rows)
scrolls only after Tab, and nothing on screen says so beyond the generic
footer hint.

### P3-9 — `Unreadable` hint is wrong for directories

`ABSTRACTCORE_CONFIG_FILE` pointing at a directory → `FileState::Unreadable`
with hint "fix the file permissions, then press r" (`src/ui/sections.rs:34-42`).
Python treats that as corrupt-fallback (defaults + warning, backup impossible).
The console's distinct state is fine; the advice is misleading for the
IsADirectory/NotADirectory error kinds.

### P3-10 — Char-count formatting misaligns wide glyphs and can push state off-screen

`field_row` pads with `{:<22}`/`{:<28}` (char counts, `src/ui/sections.rs:220-223`)
and `util::ellipsize` truncates by chars (`src/ui/util.rs:173-179`) while the
render path measures cells. CJK/emoji model ids (the unicode-in-model-names
edge) misalign the value/state columns; at ≤80 cols a long value clips the
state/broken-reason off the row (honest `…`, but the state vocabulary is the
product). `fit_width` itself is width-aware and correct.

### P3-11 — Route options render verbatim; key-shaped option values would echo on screen

`src/ui/routes.rs:65-67` prints the selected route's `options` dict raw.
Options are the one route field with operator-free content
(`capability_defaults.py:246-287` folds unknown scalars into it); someone
storing `api_key=…` as a route option (plausible for ad-hoc endpoint routes)
gets it echoed. Python's surfaces carry it as data but don't display it
unprompted. Consider masking key-suffixed option names in M2's editors.

### P3-12 — Profile `$VAR` display doesn't say whether it resolves

`src/ui/providers.rs:86-98`: an env-var reference renders `$TEAM_KEY`
regardless of `api_key_set` (which is `bool(resolved_key)` from the payload,
`provider_profiles.py:168-181`). An unset var with no stored key reads
identically to a resolving one — "configured-looking but keyless" is the
fabricated-selection family the design laws care about.

### P3-13 — Worker panic leaks the busy op forever

`src/worker.rs:55-66` catches per-command panics and posts a notice, but never
ends the ops begun by that command — a ghost `⟳ …` entry keeps the 500ms ticker
alive for the rest of the session (`src/ui/mod.rs:257-278`), violating
zero-idle-cost after a one-off internal error. Handlers are panic-light today;
the fix is cheap (drain the op ids in the catch arm).

### P3-14 — Path-string agreement can false-alarm on unnormalized paths

Python `str(Path(...))` normalizes doubled slashes; `PathBuf` display does not.
`ABSTRACTCORE_CONFIG_DIR=/etc/ac//` → console `/etc/ac//abstractcore.json` vs
CLI echo `/etc/ac/abstractcore.json` → `✗ DIFFERENT FILES` on a same-file
config (`src/ui/overview.rs:98-131` compares raw strings). Not live-tested;
pathlib normalization is standard behavior.

## Engine-suspect findings

None reproduced at 0.2.22 in this app's compositions. Specifically checked:

- **Autofocus inside a regenerating region** (first-app finding 0220's class):
  `sections_table(...).autofocus()` and the routes table both live inside
  `dyn_view_scoped` regenerations and neither panics nor misbehaves across
  fixture loads, reloads, and screen switches (headless suite + live smoke).
  Watch-item, not a finding: the autofocus refires on every snapshot reload;
  if focus semantics ever tighten engine-side, these two call sites are the
  canary.
- **Chrome pins under content pressure**: heavy fixtures at 60x16 across all 8
  screens kept header/tab/footer rows intact — the 0.2.15+ collapse behavior
  plus app-side `shrink(0.0)` did their job.
- **Startup-notice lane**: empty during the live smoke; the app's
  "caps"-prefix filter (`src/ui/mod.rs:382-386`) is an app-side choice that
  would also hide any future engine diagnostic that happens to start with
  "caps" — engine keeps the ambient-summary prefix stable, or this filter
  needs a tighter match. Filed here as a coupling note, not an engine defect.

## Test-suite audit

Fixtures are live-shape-faithful: I diffed the fixture payloads against the
real `config defaults --json` / `providers --json` on this machine — top-level
keys, route-row keys (including provider/model **absent**, not null, on
unconfigured routes), and profile-row keys all match; `RouteRow`/`ProfileRow`
tolerant parsing covers the differences that remain (`task`, `created_at`).
The chrome matrix is real (all 8 screens × 3 sizes × heavy fixtures, asserting
rows 0/1/last). The strongest secret guarantee is correctly placed at the model
level (`secrets_never_reach_the_display_model` asserts over the full Debug dump)
with the screen sweep as belt.

Gaps and weak pins:

1. **The two P2 state defects are exactly the untested states**: no test folds
   an api_keys section with a broken-typed or empty-string key, so
   `overview_states_set_default_broken` pins "● 1 of 7" without ever noticing
   the branch drops broken counts. The fixture's two broken fields live in
   sections whose Overview branch handles them.
2. **`✗ 1 broken` cannot distinguish its two sources**: video and logging each
   render that exact string; the assert passes if either row exists. Same test:
   `s.contains("openai")` is a weak needle (any provider string containing it
   would satisfy).
3. **Untested honest states**: `Unreadable` (never rendered in any test), the
   `✗ DIFFERENT FILES` agreement branch (only ✓ is pinned, via fixtures whose
   paths agree), `RoutesData.ok == false` with rows present (renders as a
   normal table; unreachable via today's CLI which hardcodes `ok: true`, but
   the field exists and is folded).
4. **Non-object JSON is handled but unpinned**: `load()`'s `!raw.is_object()`
   branch (`src/config.rs:195-199`) has no unit test — the corrupt test uses
   invalid bytes only. This is the one corrupt shape where console and Python
   agree exactly (live-verified `[1,2,3]` → Python backup + defaults), and
   nothing pins it.
5. **Negative asserts**: `drain_cmds` exists and correctly drains the whole
   queue, but it is used once, positively (`r_reloads_everything`). No test
   proves navigation/typing sends NO commands — the sibling-console law the
   harness comment cites is available but unused.
6. **Chrome matrix omits the refusal states**: corrupt/missing renders are
   tested at 110x34 only; the corrupt panel's backup list at 60x16 (5 backup
   lines + guidance under pinned chrome) is unpinned. 60x16 as the floor
   matches the charter's definition of done; nothing below it is claimed or
   tested — fine, but worth saying that at 60 wide the section-page field rows
   (22+28 cell budget before the state column) clip the state vocabulary, and
   no test would catch a regression there because none asserts state text at
   60 wide (the matrix asserts chrome rows only).
7. **pty smoke**: gates and needles are sound (the `wait_fresh` full-redraw
   loop is the right lesson applied); two nits: `wait_for("loaded")` would also
   match "not loaded" if the NotAsked state ever rendered at that moment
   (currently unreachable — boot sets Loading before `run()`), and the smoke
   inherits the ambient environment — my shell carried a leaked
   `ABSTRACTCORE_CONFIG_FILE` pointing at a deleted probe dir, which would fail
   gate [1] as "no config file yet" with no hint that the env var is the
   cause. A one-line preflight echo of `ABSTRACTCORE_CONFIG_FILE`/`_DIR` (or a
   guard) would save the next person the confusion.
8. **`cli.rs` subprocess tests** hit real processes (echo/sleep/nonexistent)
   including the timeout kill — good; nothing covers the exit-0-with-stderr
   case (which is P1-1's signal lane).

## What holds (verified, not assumed)

- **Read-only claim (3)**: no write call exists under `src/` outside `#[cfg(test)]`
  (grep-verified); the only test writer targets `std::env::temp_dir()` and the
  ignored `mint_captures` writes SVGs into `docs/captures/` as designed.
- **Worker/threading claim (2)**: one worker thread owns the file read and both
  subprocesses; every result crosses back exclusively as `wake.post` closures;
  signals are touched only in those closures and the UI thread
  (`src/worker.rs` whole-file read). Boot triple-load and `r` reload-all are
  test-pinned.
- **Path precedence chain**: FILE > DIR > default with blank-as-unset is
  unit-tested and matches Python for well-formed absolute values (live smoke
  agreement line ✓ on the real machine). The deviations are the value-handling
  edges in P2-3.
- **Secrets (charter fact 5)**: redaction is parse-time and structural; numeric
  or otherwise broken secret values render "not set" + a type reason without
  echoing the value; profile keys come only from the pre-redacted CLI surface;
  fingerprints match Python for ordinary keys (live: `sk-test` → `f3abf2a6`
  both sides, trim behavior identical).
- **Corrupt/missing/unreadable are distinct honest states**; JSON-parse corrupt
  is a hard stop listing both backup shapes newest-first
  (`src/config.rs:223-240`), and the non-object shape matches Python's own
  corrupt semantics (live-verified).
- **Missing file** renders the fold of `{}` — the exact in-memory defaults
  Python runs with (`manager.py:529-530` parity), labeled as such.
- **Schema (claim 5)**: 17 sections / 85 scalar fields — recounted against
  `manager.py:73-322` field-by-field; every name and default matches, including
  the audio alias sets and both effective-value projections
  (`manager.py:395-400` ⇄ `src/config.rs:420-437`). No dataclass field is
  missing from the schema, so the "unknown keys" warning cannot fire falsely
  against current Python (the legacy nested meta flag is correctly exempted).
- **Chrome pins + footer notices + ThemeSwitcher (claim 4)**: present, pinned
  (`shrink(0.0)` on header and both footer rows), matrix-tested at the three
  definition-of-done sizes with heavy fixtures, and live-smoked end-to-end
  (boot → routes → providers → reload ack → review agreement → clean exit 0).
- **Build gates**: build/test/clippy all green as claimed in the CHANGELOG gate
  line; engine pin resolves to exactly 0.2.22; headless guard prints its skip
  line and exits 0; `--help`/`--version` exit 0, unknown args exit 2.
