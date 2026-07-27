# M2 report — edit + wizard (2026-07-25)

Builder report for milestone M2 of `abstractcore-console` (charter:
`LAUNCH-PROMPT.md`). Gates were re-run after the adversarial fix wave;
the P1 was re-proven live with the reviewer's own repro.

## What shipped

The write lane, typed editors, and wizard mode on top of the M1 mirror:

- **Three-phase writes** (`writes.rs` + `worker.rs`): every editable
  surface produces a `WriteSpec` — refuse-unless-writable pre-checks
  (corrupt/unreadable/Python-refused/drift/CLI-presence), then verbs
  (CLI setters for every coupled field; unknown-key-preserving
  read-modify-write only for the CLI-less fields), then verification of
  value-level expectations against a FRESH re-read (+ fresh derived
  views for route/profile writes). Verify-by-re-read is load-bearing:
  both of the CLI's success signals lie (probed in M1; both liar
  classes pinned against fake CLIs).
- **Structural refusals**: corrupt/unreadable refuses everything; a
  Python-refused file refuses CLI verbs specifically (a setter would
  RESET it — the historical incident) while RMW stays allowed with a
  warning; drift (the file changed since the operator loaded it)
  refuses with a reload hint. File identity is `(mtime, ino, size)`,
  not mtime alone.
- **Typed editors** (`ui/editors.rs` + per-screen verbs): scalar/enum/
  toggle (UNSAFE flags and destructive strategy flips get danger
  confirms), masked secrets (blank keeps, explicit clear, fingerprint
  verify, leading-dash refusal), provider+model pair editors with live
  model discovery, vision strategy/chain editors, route editor
  (options as k=v with quoted-value round-trip, reasoning preserved),
  profile editor. Section pages are editable field tables with a
  pinned selected-row truth line ("applies now", reactive).
- **Wizard mode** (`ui/wizard.rs`): a guided walk covering the CLI
  wizard's 8 phases in its order, plus orientation and review; section
  pages filter to the step's focus; free-nav disarmed with reasons;
  adaptive default (wizard on a machine with no config, browse
  otherwise; `--wizard`/`--browse` override).

## Evidence

- `cargo test`: 42 unit + 32 headless + 1 ignored capture minter.
  `cargo clippy --all-targets`: zero.
- Chrome matrix: 8 screens × 80x24/100x24/60x16 × heavy fixtures +
  corrupt refusal at 60x16 (M1, still green).
- `scripts/pty_smoke.py`: green end-to-end INCLUDING the M2 write
  phase — fresh scratch config → wizard boot → editor → real
  `abstractcore` setter → file created with the value → Python-side
  `defaults --json` reads the same scratch cleanly.
- P1 fix live re-verified with the reviewer's repro: chain
  `[ollama/llava2]` → add `lmstudio/qwen-vl` → journal now carries
  `✓ add vision fallback lmstudio/qwen-vl` (was `✗ …[1].provider is
  absent`) and the file ground truth agrees (2 entries).
- SVG artifacts: `docs/captures/m2-{wizard-model,editor}.svg`.

## Adversarial review round-trip

`docs/reviews/m2-adversarial-review.md` (fable5): verdict "the
three-phase lane is the right shape; fix the P1 and the two
coupled-write P2s before calling it trustworthy" — 1 P1, 6 P2, 12 P3.
All fixed same day:

- **P1-1 (array-index expectations never evaluate)**: the expectation
  walker used `Value::get(&String)`, which never indexes arrays — every
  chain write verified as "absent" even after the CLI appended
  correctly, and the natural retry appended silent duplicates. Fixed
  with a `walk` that resolves numeric segments as array indices; pinned
  at the unit level (`expect_paths_walk_arrays`) AND through
  `execute_write` with the real builders
  (`fallback_chain_writes_verify_with_array_paths`), plus headless
  chain-editor coverage and the live repro above.
- **P2-1**: vision strategy "disabled" now names its blast radius
  (caption pair + N-entry chain) in a danger confirm before executing;
  the same disclosure guards `x` on vision.strategy.
- **P2-2**: `clear_global_default` runs its fallible CLI half FIRST
  (RMW second); specs with CLI verbs refuse upfront when no CLI is
  resolvable. Pinned (`clear_global_default_is_cli_first_and_refuses_
  without_cli`).
- **P2-3**: the UI door now matches the worker's split — CLI-routed
  editors refuse on a Python-refused file, RMW-routed editors proceed
  with a warning. The test that enshrined the defect was rewritten
  (`refused_file_door_splits_cli_from_rmw`).
- **P2-4**: clearing audio.strategy routes through a dedicated
  `ResetAudioStrategy` RMW that clears the value AND both spellings of
  the explicit flag — the true default state, not "explicitly default".
- **P2-5**: `reasoning` round-trips through the route editor (row →
  field → CLI arg).
- **P2-6**: chain length comes from a fold-time `list_len` on the
  display model, never from counting `/` in the display string —
  `org/model` ids no longer break removal.
- **P3 wave**: in-flight guard on chain-add; per-form base stamp
  captured at open + reactive "applies now" lines; `FileStamp`
  identity; quit refuses mid-write; leading-dash secrets refused with a
  teaching message; options round-trip (quoted values, type-preserving)
  pinned; empty-string prefills; `x` on a missing file refuses instead
  of minting `{}`; dead plumbing removed (`prompt_open`, `open_form`)
  and `reset_domains` actually wired into reload; dirty-Esc guards
  track selects/checkboxes; pair editor truth line; vision-strategy
  clear ordering (refuse before confirm).

**Post-review regression caught by the smoke re-run**: `handle_write`
refreshed BOTH derived views after every successful write —
`providers --json` added 5-15s of CLI tail to writes that cannot touch
profiles. Profiles now refresh only for specs that declare profile
expectations (key/profile CRUD); the routes/defaults refresh stays (it
carries the agreement line and the `#FALLBACK` lane). Write end-to-end
time roughly halved.

**Test-audit adoption**: gaps 1-3 closed (array paths through the real
builders; `eval_derived_expect` extracted pure + tested; multi-verb
order pinned); gap 7 partially (chain editor headless; the P1
surface). Deferred to M3 with the test verbs they belong to:
SecretFp-through-execute_write, timeout/file-disappeared refusals,
`parse_args` unit tests, remaining editor submit coverage,
double-submit interleavings (the in-flight guards shipped; dedicated
tests pending).

## Findings for the CORE seat (Python-side; console never edits it)

Nothing NEW beyond the M1 report's seven (all reconfirmed relevant
during this milestone's write probes). One operator-experience note
worth relaying: every `abstractcore` invocation is a cold Python
import — 4-15s per call under load on this machine. The console
absorbs it (busy strip with elapsed, journal, single-flight worker),
but any interactive caller pays it per write; a long-lived
`--serve-json` mode (or a batch flag form) would change the class.

## Engine findings (abstracttui 0.2.22)

- **field-core 1100** (TextInput opens with cursor at byte 0; no API to
  open at end or select-all) — filed during the build, independently
  confirmed by the reviewer. Workaround shipped: tests send End first;
  operators must End/→ before appending to a prefilled value. This is
  the one real papercut in the editors.
- The reviewer's weak signal (overlay layers possibly repainting one
  frame behind the base tree after a global redraw) did not reproduce
  under stabilized frame reads — the probe's fixed 0.5s settle window
  was truncating repaints under system load. `wait_fresh` now reads
  until the frame is quiet (0.3s) before judging; not filed.
- Verified NOT an issue: prompt-over-modal stacking ("topmost modal
  wins" holds in 0.2.22's overlay store).

## M3 inputs settled during this milestone

- The write lane is the substrate for M3's test verbs: `WriteSpec` +
  journal generalize to probe specs (connectivity, model listing, cheap
  generation) with the same busy/notice/journal surfaces.
- `config test-provider` semantics (M1 finding 5): `ok:true, count:0`
  against a dead server means the console's Test verb must treat
  zero-count success as NOT proven and pair it with its own health
  probe.
- CLI latency shapes M3's UX: test verbs must be per-provider
  single-flight with visible elapsed, never a fan-out of blocking
  calls.
