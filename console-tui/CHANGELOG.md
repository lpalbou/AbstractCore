# Changelog — abstractcore-console

All notable changes, one entry per build wave, each with its gate line
(build/test/clippy state at the wave's close). Charter:
`LAUNCH-PROMPT.md`.

## [Unreleased]

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
