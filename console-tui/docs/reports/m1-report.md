# M1 report — the honest mirror (2026-07-25)

Builder report for milestone M1 of `abstractcore-console` (charter:
`LAUNCH-PROMPT.md`). Everything below is verified, not remembered;
gates were re-run after the adversarial fix wave.

## What shipped

A read-only configuration mirror on abstracttui 0.2.22:

- **Shell**: PageHost over 8 screens (Overview, Model, Providers,
  Routes, Media, Embeddings, Server, Review), chrome pinned
  (`shrink(0.0)` on header + footer from day one), footer with busy
  strip / app notices / humanized engine startup notices / key hints /
  ThemeSwitcher. Header leads with the file STATE (state-first ordering
  — a truncated CORRUPT flag is a lying header), then the path.
- **The mirror**: direct parse of `abstractcore.json` (path resolution
  with Python-exact env semantics incl. `expanduser`), folded at parse
  time into a REDACTED display model — per-field set/default/broken
  against the documented dataclass defaults (schema.rs: 17 sections /
  85 fields, recounted by the reviewer against `manager.py:73-322`),
  secrets as sha256[:8] fingerprints (Python's own convention incl.
  the EMPTY canonicalization), unknown sections/keys surfaced with the
  drop-on-Python-save warning, corrupt files refused with backups
  listed, missing files rendered as the built-in defaults Python runs
  with.
- **Python-refusal honesty (the review's P1)**: two independent lanes
  flag a file Python's loader raises on — the fold models the ONE
  raising surface (profile-row construction) and every exit-0 CLI run's
  stderr is scanned for `#FALLBACK`. Either lane flips the header,
  poisons the agreement line, and banners the consequence ("every
  abstractcore run backs it up and uses DEFAULTS").
- **Derived views**: `config defaults --json` (24 routes, coverage
  badges, output.text alias) and `config providers --json` (redacted
  profiles, $VAR resolution honesty) through ONE worker thread; the
  CLI's `config_file` echo is cross-checked component-wise against the
  console's own path.

## Evidence

- `cargo test`: 21 unit + 22 headless (CaptureTerm/Driver over the real
  UI, fixtures shape-diffed against live CLI payloads by the reviewer)
  + 1 ignored capture minter. `cargo clippy --all-targets`: zero.
- Chrome matrix: 8 screens × 80x24/100x24/60x16 × heavy fixtures, plus
  the corrupt refusal at 60x16.
- `scripts/pty_smoke.py`: green against the real machine (boot →
  agreement → routes → providers → reload ack → review → clean exit).
- P1 regression live-verified: a scratch config with one unknown
  profile-row field rendered both refusal banners, the named field,
  and the flipped header — while the CLI call it triggered minted a
  fresh `.corrupt-*.bak` in the scratch dir, confirming the incident
  mechanics.
- SVG artifacts: `docs/captures/m1-{overview,routes,server,corrupt}.svg`.

## Adversarial review round-trip

`docs/reviews/m1-adversarial-review.md` (fable5): verdict "solid,
honestly-built" with one genuine hole — 1 P1, 4 P2, 14 P3, zero engine
defects. All fixed same day except one deliberate partial:

- **P2-2 partial**: the raising half (profiles) is fully covered by the
  P1 fix; a malformed-shape advisory for the two TOLERANT special
  sections (`capability_defaults: "garbage"` renders `· default`,
  which matches what Python actually runs — empty routes — but does
  not say "your section is nonsense") is deferred to M2's editors,
  where section shape gets touched anyway.
- Not adopted: alias-normalized route counting in the file lane (the
  derived view is the exact truth when the CLI is up; the file lane now
  at least drops `{}` routes); a `RoutesData.ok == false` render test
  (unreachable via today's CLI, which hardcodes `ok: true`).

## Findings for the CORE seat (Python-side; console never edits it)

Live-verified 2026-07-25 on abstractcore 2.13.38, scratch configs only:

1. **Profile rows lack kwargs filtering** — one unknown key in a
   `provider_profiles` row makes `_dict_to_config` raise, sending the
   WHOLE file down the corrupt-fallback path (backup + defaults),
   unlike every other section (`_filter_dataclass_kwargs`,
   manager.py:368-372 vs provider_profiles.py:224-229). Forward-compat
   consequence: a config written by a NEWER abstractcore (or one
   hand-added field) makes every older invocation run on defaults.
2. **A fresh `.corrupt-<stamp>.bak` is minted on EVERY load** of such a
   file (manager.py:495-566) — a monitoring loop or repeated CLI calls
   litter the config dir with backups (this console throttles nothing
   yet; it runs two CLI calls per reload).
3. **The flags CLI exits 0 on refused writes** — `--set-server-port
   99999` prints `❌ Error: Invalid server port: 99999`, exits 0, file
   unchanged. Machine callers cannot script `--set-*` safely; the
   `config` subcommands (set-default etc.) do exit honestly.
4. **`--set-app-default` success-lies for unknown apps** — prints
   `✅ Set bogusapp default to: ollama/m`, persists nothing
   (`set_app_default` returns False; main.py:1816-1819 ignores it).
5. **`config test-provider ollama --json` reports `ok:true, count:0`,
   exit 0 while Ollama is DOWN** (connection refused) — the
   raise_on_error lane swallows the unreachable case; a "test" that
   passes against a dead server. (Consequence adopted for M3: the
   console's Test verb treats `ok:true + count:0` as not-proven and
   pairs it with a native health probe.)
6. **No CLI clear** exists for the global default or chat/code models
   (`--set-global-default ''` is a silent exit-0 no-op);
   `--set-api-key P ''` stores `""` rather than null (semantically
   not-set — falsy — so harmless, but asymmetric with the null
   default).
7. Pre-known from the inventory, re-confirmed relevant: the wizard's
   embeddings validation omits `vllm`; `config
   set-default/clear-default/set-provider` accept no `--json`.

## Engine findings (abstracttui 0.2.22)

**None filed this milestone — and that is a claim, not an omission.**
The adversarial reviewer specifically attacked the engine-suspect
surfaces and reproduced nothing: autofocus inside `dyn_view_scoped`
regenerations behaved across reload/switch cycles (watch-item: these
call sites are the canary if focus semantics tighten), chrome pins held
at 60x16 under heavy fixtures, Modal/popup surfaces are not yet in use
(M1 has no editors — M2 is where Select/Combobox/ChoicePrompt/masked
TextInput pressure starts, which is where the sibling apps found most
of their engine findings). The `field-core` band (1100-1190) is
registered and empty; expectation is that M2's form work changes that.

## M2 inputs settled during this milestone

`docs/backlog/proposed/0001_write_lane_design.md` — the write lane
grounded in live-verified CLI behavior: verify-by-re-read is
load-bearing (two of the CLI's own success signals lie, findings 3+4),
`❌ Error:` scanning on exit-0, unique tmp names for direct writes,
drift detection before save, and the corrupt/refused states as
structural write-refusals.
