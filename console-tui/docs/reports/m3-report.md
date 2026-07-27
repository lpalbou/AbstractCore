# M3 report — test & prove (2026-07-25)

Builder report for milestone M3 of `abstractcore-console` (charter:
`LAUNCH-PROMPT.md`). Gates were re-run after the adversarial fix wave;
both P1s were re-proven fixed live.

## What shipped

The test lane on top of the M1 mirror and M2 write lane:

- **The probe vocabulary** (`probes.rs`, pure and unit-tested): probe
  specs + verdict folds with an honest three-state outcome — Proven
  (positive evidence only: models listed, reply produced), NOT PROVEN
  (the CLI's ambiguous successes land here, never in Proven), Failed
  (a named error). The worker (`handle_probe`) owns the subprocess and
  socket mechanics.
- **Three test verbs**:
  - `t` (Providers): a provider test picker over all 10 canonical
    providers + every endpoint profile → live model discovery via
    `config test-provider --json`. The api_keys table alone could
    never reach keyless lmstudio/ollama — the wizard's own targets.
  - `t` (Routes): the selected route's model must be AMONG what the
    provider actually serves — capability-agnostic membership (voice/
    image routes can't be chat-tested; model existence always can).
  - `g` (anywhere): one cheap generation over the CONFIGURED default
    route via `abstractcore-chat --prompt`. A local pre-check refuses
    when the file names no route — probed: the chat CLI on an empty
    config silently generates through a built-in huggingface default.
    `endpoint:<id>` defaults expand to `--provider <family>
    --base-url <url>` for keyless profiles (disclosed as "via …" in
    the evidence); keyed/disabled profiles refuse honestly.
- **The TCP disambiguation**: `config test-provider` answers `ok:true,
  count:0, errors:[]` against a DEAD server (live-probed, reconfirmed
  by the reviewer). Zero-count success folds to NOT PROVEN, upgraded
  to a named cause by a reachability check over ALL resolved addresses
  of KNOWN endpoints only — profile base_url or the ollama/lmstudio
  local defaults; never https, never cloud, userinfo URLs refused.
  The cause LEADS the message (surfaces truncate from the right).
- **Evidence surfaces**: Review carries the latest result per target
  (re-tests replace; overflow says "… and N older"); every probe also
  journals (NOT PROVEN under `?`, not `✗`); probes are single-flight
  (latched synchronously at send).

## Evidence

- `cargo test`: 53 unit + 36 headless + 1 ignored capture minter.
  `cargo clippy --all-targets`: zero.
- `scripts/pty_smoke.py`: green — 7 phases, now including the M3
  test-verb phase against the real LM Studio AND an automated negative
  lane (ollama down → `? test ollama` with the TCP cause leading;
  environment-tolerant if ollama is up).
- `scripts/definition_of_done.py`: the chartered walk, green
  end-to-end in ~25s — fresh machine → wizard boots → default set to
  lmstudio through the pair editor (coupled CLI write) → `g`
  generation PROVEN → wizard finish → `config defaults --json` agrees
  → browse edits route input.text to another live model → `t`
  membership PROVEN → Python re-read agrees.
- P1 fixes live re-proven: an `endpoint:scratch-lm` default over the
  real LM Studio → `✓ generation test (default route):
  endpoint:scratch-lm/gemma-3-1b-it — replied in 2s: "PONG" (via
  lmstudio @ …)` (was ✗ FAILED on an argparse error).
- SVG artifacts: `docs/captures/m3-review-evidence.svg` (all three
  verdicts + a route-membership failure).

## Adversarial review round-trip

`docs/reviews/m3-adversarial-review.md` (fable5): verdict "the core
idea is right and mostly real … but M3 ships with two P1s, both
live-proven" — 2 P1, 4 P2, 13 P3. All fixed same day except the
documented deferrals below:

- **P1-1**: `g` on an `endpoint:<id>` default route reported ✗ FAILED
  for a WORKING route — `abstractcore-chat`'s argparse `choices` list
  knows only the 10 static providers, and the console's own pair
  editor authors endpoint defaults (the framework's flagship shape).
  Fixed by expansion (keyless) + honest NOT PROVEN refusals (keyed —
  argv must never carry secrets, and a guessed key lane would mint
  401 lies on working routes) + Failed for missing profiles. Pinned
  with an argv-proving fake; live re-proven.
- **P1-2**: the TCP check probed only the FIRST resolved address;
  `localhost` resolves `::1` first on this machine while LM Studio
  serves IPv4 only — so the exact branch the feature exists for
  (server UP, zero models: the fresh-install wizard audience) said
  "the server looks DOWN". Fixed: reachability judges ALL addresses
  (Connected if any accepts; Refused prefers the IPv4 error text);
  extracted pure (`probe_addr_list`) and pinned with real listeners.
- **P2 wave**: multibyte panic in `is_log_line` (a `[8..]` slice on
  arbitrary model output — now `get(8..)`, pinned); the single-flight
  guard was advisory (set only by the worker's begin-post — now
  latched synchronously at send, pinned); route-evidence labels
  embedded the pair so re-tests after edits accumulated stale rows
  (pair-free labels, pinned — the reviewer read this file mid-fix;
  the pin landed with the wave); Routes-lane endpoint routes now
  resolve the PROFILE's base_url for reach parity (pinned).
- **P3 wave**: program-named CLI errors, chat-binary provenance
  disclosure, userinfo-URL refusal, models-list-derived Proven,
  journal `?` rendering, keyed-cloud no-key hint, "available" not
  "served", evidence overflow line, pair-after-cause detail order,
  RouteEq wording, DoD ambient-var warning, README `g` semantics
  line, documented stdout-scraping cost.

**Deferred with reasons**:
- `handle_probe`'s store-effect wiring (probe_busy ordering, journal
  lanes, notice format) is tested pure at `probe_generation` and by
  the headless verb tests, but the record/journal/notice fold itself
  has no Store+wake harness — same extraction class the M2 review
  named for `handle_write`; scheduled, not silently skipped.
- `fold_generation`'s `❌`/`Error:` reply-shape false positive stands
  as a documented cost (the chat CLI has no machine verdict channel);
  the probe prompt ("Reply with exactly: PONG") makes it adversarial
  rather than expected.

## Findings for the CORE seat (Python-side; console never edits it)

New this milestone (all live-verified on abstractcore 2.13.38):

1. **`abstractcore-chat` cannot name `endpoint:<id>` providers** —
   argparse `choices` hardcodes the 10 static names
   (`utils/cli.py:2558`), while the config system happily persists
   endpoint defaults and `create_llm` accepts them. Any CLI caller
   testing an endpoint route must expand family+base_url itself (this
   console now does); a `choices` widening (or dropping `choices` for
   the library's own validation) would close the gap for everyone.
2. **`abstractcore-chat` exits 0 on generation failures**, printing
   `❌ Error: …` to stdout — the write-lane liar class, third
   appearance (machine callers cannot script it safely).
3. **Bare `abstractcore-chat` never reads the global default**: its
   default resolution is `app_defaults.cli` (dataclass default
   huggingface/Qwen3-4B), so an empty or default-less config silently
   generates through a local HF model — surprising for operators who
   just set `default_models.global_*`.
4. Reconfirmed: `config test-provider` answers `ok:true, count:0,
   errors:[]` with exit 0 against a dead server (M1 finding 5); the
   console pairs it with its own TCP evidence.
5. `openai`'s `list_available_models` returns `[]` when no key
   resolves — silently, even under `raise_on_error`
   (`openai_provider.py:1125`), indistinguishable from an empty
   account; an errors[] entry would make key problems diagnosable.

## Engine findings (abstracttui 0.2.22)

None new this milestone. ChoicePrompt (the test picker), Select popups
(pair/route editors), and the global-action repaint contract all
behaved as documented under both headless and pty drives. field-core
1100 (TextInput cursor-at-0 on prefill) remains the one open papercut,
worked around with End-first key sequences in scripts and noted for
operators.

## Definition of done — status

Chartered: "launch on a machine with no config → wizard completes with
a local provider → test passes → `abstractcore config defaults --json`
agrees with what the console shows → browse mode edits one capability
route → re-verify. Plus: headless suite green, pty smoke green, zero
clippy warnings, the chrome matrix green at 80x24/100x24/60x16."

All delivered: `scripts/definition_of_done.py` is the scripted,
repeatable form of the walk (green); the chrome matrix (M1) still
passes at all three sizes with heavy fixtures; the SVG series lives in
`docs/captures/`; the smoke and suite gates are green as of this
report.
