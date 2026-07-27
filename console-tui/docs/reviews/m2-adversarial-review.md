# M2 adversarial review — abstractcore-console (edit + wizard)

- Date: 2026-07-25 · Reviewer: adversarial subagent (spec-conformance + edge-case audit)
- Scope: the M2 write lane (`writes.rs`, `worker.rs`, `cli.rs`), typed editors
  (`ui/editors.rs`, `ui/forms.rs`, `ui/sections.rs`, `ui/providers.rs`,
  `ui/routes.rs`), wizard mode (`ui/wizard.rs`, `ui/mod.rs`, `lib.rs`), plus
  regression checks on the M1 fixes.
- Method: full source read; `cargo build --all-targets` green, `cargo test`
  green (33 unit + 31 headless + 1 ignored), `cargo clippy --all-targets`
  zero warnings; Python source cross-reads under
  `abstractcore/config/{manager,main,provider_profiles}.py`; live pty drives
  of the real binary against scratch configs (`ABSTRACTCORE_CONFIG_FILE`
  env-pointed; the real config untouched; scratch dirs deleted); one live
  probe of the real `abstractcore` flags CLI against a scratch config.
- Line numbers are from the working tree at review time.

## Verdict

The three-phase write lane is the right shape and its two probed liar-class
defenses are real (both pinned against fake CLIs, and the ✅-liar catch is
what my live P1 repro rode in on — verification fired; it was the *evaluator*
that was wrong). Refusal layering (corrupt / unreadable / Python-refused /
drift) is present at both the UI door and the worker, secrets redact
structurally on every surface I could reach, and the wizard matches the CLI's
8 phases with honest refusals. The M1 fixes did not regress.

But M2 ships with **one live-proven P1**: the vision fallback chain — the one
write class whose expectations use array indices — never verifies, because
the expectation walker cannot index arrays. A successful add is reported as a
failure on screen, in the notice, and in the journal, and the natural retry
appends silent duplicates. Around it sit six P2s: a partial-application trap
in the one multi-verb spec, a claim-vs-behavior mismatch on the
Python-refused-file rule, two destructive/coupled writes that fire without
disclosure ("disabled" wipes the chain; "reset audio" flips the explicit
flag), a silent `reasoning` drop in the route editor, and a display-derived
chain length that breaks removal for `org/model` ids. The verification lane
for routes/profiles (`handle_write`'s derived-view half) has zero test
coverage. Fix the P1 and the two coupled-write P2s before calling the write
lane trustworthy; the rest is schedulable.

---

## P1 findings

### P1-1 — Array-index expectations never evaluate: vision-fallback add reports failure after the write LANDED; retries append duplicates; remove verification is vacuous (live-proven)

`eval_file_expect`'s path walker resolves each segment with
`Value::get(&String)`:

- `src/writes.rs:198-204` — `cur = cur.get(seg)?` where `seg: &String`.

serde_json's `Index` impl for `str`/`String` only indexes **objects**; on an
array it returns `None` unconditionally (only `usize` indexes arrays). Every
expectation whose path carries a numeric segment therefore resolves to
"absent":

- `src/writes.rs:656-664` — `add_vision_fallback` expects
  `Eq(["vision","fallback_chain","<new_len-1>","provider"], provider)` →
  `get` returns `None` → `Err("… is absent (expected …)")` **always**, even
  when the CLI appended correctly.
- `src/writes.rs:683-701` — `remove_vision_fallback` expects
  `Cleared(["vision","fallback_chain","<n>","provider"])` → `None` →
  `Ok("cleared")` **always** — the removal verification pins nothing (it
  would report success even if the entry were still there).

**Live repro** (real binary, real `abstractcore` 2.13.38, scratch config):
chain `[ollama/llava2]` → chain editor → Add → `lmstudio/qwen-vl` → Save.
Final frame, all simultaneously true:

```
vision  fallback_chain  [ollama/llava2, lmstudio/qwen-vl]  ● set     ← the write LANDED
│✗ vision.fallback_chain.1.provider is absent (expected "lmstudio")│  ← form says FAILED
✗ add vision fallback lmstudio/qwen-vl — vision.fallback_chain.1.provider is absent …  ← journal/notice say FAILED
```

File after: `[{"provider":"ollama","model":"llava2"},{"provider":"lmstudio","model":"qwen-vl"}]`.

Compounding facts:

1. **Retry appends duplicates.** Python's `add_vision_fallback` is a plain
   append with no dedupe (`abstractcore/config/manager.py:1569-1584`), and
   after the failed verify `handle_write` refreshes the mirror
   (`src/worker.rs:210-217`), so the form's next Save reads a fresh
   `base_mtime` (`src/ui/mod.rs:154-161` reads the store at submit time) and
   the drift guard passes. Each retry of the "failed" write appends one more
   copy while the UI keeps saying failure.
2. **The journal is now a false record**: a `✗` row for a write that changed
   the file (the Review screen's session audit misleads exactly where the
   operator goes to verify).
3. This is a direct violation of charter/CHANGELOG claim 1 ("EVERY write
   verified against a fresh re-read") — for this write class the verifier
   lies in both directions (false-fail on add, vacuous-pass on remove).

Fix direction: teach the walker numeric segments (`seg.parse::<usize>()` when
`cur` is an array), or carry typed path segments (`enum Seg { Key(String),
Idx(usize) }`). Then add the missing test: run `add_vision_fallback`'s own
builder through `execute_write` against a real post-write file (see the test
audit — this exact hole is untested; the generic `Eq` tests all use 2-segment
object paths).

---

## P2 findings

### P2-1 — Vision strategy "disabled" silently DESTROYS the caption pair and the whole fallback chain, with no confirm and no disclosure

`open_vision_strategy` fires the write immediately on selecting "disabled"
(`src/ui/editors.rs:843-855`); the option copy says "disabled — no vision
fallback". Python's `--disable-vision` does much more than set the enum:
`disable_vision` nulls `caption_provider`/`caption_model` **and empties
`fallback_chain`** (`abstractcore/config/manager.py:1586-1596`). The spec's
expects check only `vision.strategy` (`src/writes.rs:628-637`), so the
journal reports a clean "verified" while operator-built config was erased
from its only copy.

Contrast: a mere bool flip gets a ChoicePrompt confirm
(`src/ui/editors.rs:398-429` — "a single keystroke silently toggling server
config would be too cheap an accident"), yet this genuinely destructive verb
runs confirm-free. By the letter of this review's severity rubric (data
loss) this borders P1; I rate it P2 because the wipe is the documented
contract of the CLI verb the console chose — the console defect is offering
it as a plain enum choice with zero disclosure. Fix: name the consequence in
the option detail and route it through `confirm_danger` when the pair or
chain is non-empty; optionally verify (or at least journal) the coupled
clearing.

### P2-2 — The one multi-verb spec half-applies on the LIKELY failure, and nothing tells the operator

`clear_global_default` is the only two-verb spec: RMW-null the
`default_models` fields, then CLI `config clear-default input.text`
(`src/writes.rs:484-509`). The worker runs verbs in order and aborts on the
first error (`src/worker.rs:414-436`) — but the most likely verb-2 failures
(CLI missing: `worker.rs:417-419`; Python env broken; setter refusal) all
strike **after** verb 1 already rewrote the file. Result: `global_provider`/
`global_model` nulled while route `input.text` still routes — exactly the
"status desyncs from runtime" coupled-invariant break the charter's fact #2
names — and the error the operator sees ("abstractcore CLI not found — this
write needs it") implies nothing happened.

Notes:

- `writable_now` (`src/ui/mod.rs:167-197`) does not pre-check CLI presence,
  so the no-CLI case is reachable from the UI door.
- Fix direction: order the fallible CLI verb first (RMW after a
  just-successful fresh read is the near-infallible half), or pre-check
  `cli.is_some()` for specs containing CLI verbs, and make multi-verb
  failures name what already applied ("the fields were cleared; the route
  clear did NOT run — route input.text still routes").

### P2-3 — Claim vs behavior: "Python-refused file → RMW stays allowed" is true only in a layer no user can reach

CHANGELOG (lines 23-28) and review claim 2 state the Python-refused file
"refuses CLI verbs specifically … while unknown-key-preserving RMW stays
allowed". The worker implements exactly that (`src/worker.rs:380-394`, pinned
by `python_refused_file_blocks_cli_verbs_but_not_rmw`). But every editor
opens through `Ctx::writable_now`, which refuses **everything** when
`python_refusals` is non-empty or the CLI `#FALLBACK` signal is set
(`src/ui/mod.rs:170-189`), and the headless test
`editors_refuse_while_python_refuses_the_file` enshrines the door refusal.
So no RMW write can be issued from the UI in that state: an operator with a
Python-refused file cannot even edit the harmless `offline.*`/`email.*`
fields the console is "the first UI for". Either the door should
distinguish (allow RMW-routed editors with a warning, refuse CLI-routed
ones — matching the worker and the claim), or the claim/CHANGELOG must stop
saying RMW stays allowed. As shipped, the claim is dead code plus a test
that pins the opposite behavior.

### P2-4 — "Reset audio.strategy to its default" does not produce the default state (explicit flag silently set)

The generic `x`/reset path writes the default value through the field's CLI
route (`src/ui/sections.rs:361-408` → `writes::set_scalar` →
`--set-audio-strategy auto`). Python's setter also sets
`audio_strategy_explicit = true` (`abstractcore/config/manager.py:661-676`).
Post-"reset", the file is NOT in the default state: the smart default stops
applying, and on a machine without abstractvoice the effective strategy
changes from `native_only` to `auto` (the console's own effective-note logic,
`src/config.rs:638-655`, documents this divergence). The confirm label
("Reset audio.strategy to its default (auto)?") and the spec label
(`set audio.strategy = "auto"`) never mention the flag, and the expects don't
check it — the enum-editor path special-cases `set_audio_strategy` with the
"marks it explicit" label and the flag expectation
(`src/ui/editors.rs:314-320`, `src/writes.rs:914-939`), but the reset path
bypasses that special case. An honest reset for this field is RMW: remove
`audio.strategy` AND clear the explicit flag (both spellings), or at minimum
the same "(marks it explicit)" labeling.

### P2-5 — The route editor silently deletes `reasoning` on every edit, and verification can't see it

`config set-default` is REPLACE semantics: `set_capability_default` builds a
fresh route from only the passed args (`abstractcore/config/manager.py:
1266-1300`). The console's route editor round-trips provider/model/base_url/
options (`src/ui/routes.rs:188-292`) but has **no reasoning field** and never
passes `--reasoning` (`src/writes.rs:791-832`). Editing any route that
carries `reasoning` (settable via the CLI today) silently drops it, and
`Expect::RouteEq` checks provider/model only (`src/worker.rs:248-277`), so
the write verifies "clean". `RouteRow` doesn't even parse the field
(`src/store.rs:58-96`), so the editor cannot warn. Fix: parse + prefill +
resend `reasoning` (one more optional text field), or refuse to edit routes
that carry fields the editor doesn't round-trip.

### P2-6 — Chain length derived by counting '/' in the DISPLAY string: `org/model` ids break removal

`open_chain_editor` computes the entry count as
`f.display.matches('/').count()` (`src/ui/editors.rs:871-883`) over the
rendered `[p/m, p/m]` string (`src/config.rs:611-627`). Any entry whose model
itself contains `/` — every HuggingFace-style id, including abstractcore's
own built-in default `unsloth/Qwen3-4B-Instruct-2507-GGUF` — counts twice.
Consequences: the menu title miscounts ("(2 entries)" for one), **Remove
targets `chain_len - 1` = a nonexistent index** and is refused by the RMW op
("entry N no longer exists") — removal is impossible while any such entry
exists — and Add computes `new_len` one too high (its expect would point past
the end; currently masked by P1-1, which makes all array expects fail
anyway). Fix: count entries from the snapshot's raw array length (fold it
into `FieldView` or re-derive from `routes_in_file`-style raw access), never
from display text.

---

## P3 findings

### P3-1 — `open_chain_add` submit lacks the `in_flight` guard every other editor has

`src/ui/editors.rs:935-950`: no `if in_flight.get_untracked() { return; }`.
A double-Enter queues two writes sharing one `form_id`; the worker is serial
so the second dies on the drift guard with the misleading "the file changed
since you loaded" — and its outcome arrives after the form closed (stale
`write_done` slot). The drift guard happens to prevent the double-append, but
by accident of ordering, not design.

### P3-2 — The drift guard is one-shot and self-rearming; the form's "review" surface is stale

After any refusal/failure, `handle_write` refreshes the mirror
(`src/worker.rs:210-217, 329-331`), and `write_base()` reads the store at
SUBMIT time (`src/ui/mod.rs:151-161`) — so the guard's "press r to reload,
review, then retry" is satisfied by a blind immediate re-Save: the refusal
fires exactly once and the operator never has to actually look. Meanwhile the
form's "applies now" line is a non-reactive snapshot from form OPEN
(`src/ui/editors.rs:238-258` uses `with_untracked` once), so reviewing
*inside* the form shows pre-conflict state. Base-at-submit also means a
mirror refresh that lands while a form is open (another write completing)
silently re-arms the guard under an operator who saw older values. Smallest
honest fix: capture `base_mtime` at editor OPEN, and make `applies_now_line`
reactive.

### P3-3 — Drift identity is mtime-only

`Snapshot.mtime` (`src/config.rs:136`, compared at `src/worker.rs:395-401`)
is `SystemTime` equality. On coarse-mtime filesystems a same-second external
rewrite passes the guard. Python's save mints a new inode every time
(tmp + `replace`), so `(mtime, ino, size)` would be strictly stronger at zero
cost — the workspace's recorded JsonFileRunStore lesson. Low likelihood on
APFS; cheap to close.

### P3-4 — Quit abandons in-flight writes; the lib.rs comment is stale M1 text

`src/lib.rs:190-196`: "M1 commands are reads, safe to abandon" — no longer
true. `q`/Ctrl+C during an in-flight write exits immediately; a spawned
`abstractcore` setter is a separate process and lands AFTER exit, unverified
and unjournaled (the operator watched "applying…" and quit believing nothing
happened). The Cancel button likewise closes the form while the write
proceeds (`install_write_done`'s outcome then hits a dead form). At minimum:
refuse-or-warn on quit while `store.busy` holds a write, rename the comment,
and label Cancel-during-flight ("the write continues; check the journal").

### P3-5 — Leading-dash secrets cannot be stored, and the surfaced error is cryptic (no leak, verified)

Live-probed: `--set-server-auth-token -fake-lead-dash` → argparse exits 2
with "expected one argument" and does NOT echo the value (no leak — claim 3
holds). But `secrets.token_urlsafe` can mint leading-dash tokens, so a pasted
real token occasionally hits this, and the operator sees the CLI's raw
"expected one argument" with no hint. Worth a pre-submit check in the secret
editors ("values starting with '-' can't pass the flags CLI").

### P3-6 — Route options round-trip hazards: space-containing values can't re-save; string scalars silently change type

The options field re-parses `k=v` tokens split on whitespace
(`src/ui/routes.rs:263-275`): an option value containing a space (legal via
`--option "prompt=hello world"`) prefills as `prompt=hello world` and then
refuses to re-submit ("\"world\" is not k=v") — that route becomes uneditable
without deleting the option. And `_parse_capability_options` JSON-parses
values (`abstractcore/config/main.py:1395-1414`), so a STRING-typed stored
value that looks like a scalar (`"true"`, `"42"`) flips type on any console
re-save. Same class as the CLI's own hazard, but the editor invites it by
round-tripping.

### P3-7 — Empty-string values prefill as the literal two-character text `""`

`render_value` displays an empty string as `""` (`src/config.rs:607`) and
`prefill` feeds the display back as the edit buffer
(`src/ui/editors.rs:84-89`) — saving writes a two-quote string. Reachable for
any Str field set to `""` (e.g. `email.smtp_host` explicitly emptied…
by a non-default route) and for OptStr set to `""`.

### P3-8 — `x` (clear) on a missing file creates an empty `{}` config file

`rmw_write` treats Missing as an empty object base (`src/worker.rs:458-460`),
so clearing an already-default field on a machine with no config file mints a
`{}` file (semantically harmless to Python, but the console just created a
file the operator never asked for, and the header flips from "no config file
yet" to "loaded").

### P3-9 — Dead plumbing that implies unenforced invariants

`ui.prompt_open` is incremented/decremented (`src/ui/forms.rs:85-89`) and
consumed by nothing — the comment's promise ("anything that could stack over
a live prompt must wait on this") is enforced nowhere. `Store::reset_domains`
(`src/store.rs:292-310`) and `forms::open_form` (`src/ui/forms.rs:31-33`)
have zero call sites. Either wire them or delete them; dead guards read as
protection they don't provide.

### P3-10 — Dirty-Esc guards track only text fields

Profile editor: family select, `enabled`, and `clear the stored key` are not
in the dirty set (`src/ui/providers.rs:299-311`); route editor: the provider
select isn't (`src/ui/routes.rs:219-229`). A form dirty only in those
controls discards on the FIRST Esc with no warning — the exact accident the
guard exists to prevent.

### P3-11 — The pair editor has no "applies now" truth line

Scalar and enum editors carry it (`src/ui/editors.rs:177-179, 346`); the pair
editor (global default / embeddings / vision pair / app defaults) shows only
the discovery line — the current pair is visible only in the table behind the
modal. CHANGELOG claim 4 groups all editors under the truth-line rule.

### P3-12 — `x` on vision.strategy: confirm first, refuse after

`clear_selected`'s generic branch confirms ("Reset vision.strategy to its
default (disabled)?") and only then hits `set_scalar`'s refusal for the
VisionStrategy route (`src/ui/sections.rs:376-407` → `src/writes.rs:407`),
surfacing "routes to VisionStrategy — use its editor" as a notice after the
operator already confirmed. Route the refusal before the confirm.

---

## Engine-suspect findings (abstracttui 0.2.22)

1. **field-core 1100 (already filed by the builder) — confirmed
   independently.** TextInput opens prefilled with the cursor at byte 0; the
   M2 headless test itself must send End before editing
   (`tests/headless_ui.rs:790-794`), and the live pty phase types into empty
   editors only. No new filing needed; the workaround (End-first) is fragile
   for operators, which is why P3-7's prefill artifacts sting twice.
2. **Weak signal, NOT filed as a defect**: during pty probing under load,
   fresh frames captured ~0.5-0.7s after a Ctrl+L (global `request_full_redraw`)
   sometimes lacked the open modal while later frames always included it. My
   two early "form closed" misreads came from this. Most likely probe-side
   settle timing; but if the engine repaints overlay layers a frame behind the
   base tree after a full redraw, that's worth a verify-in-passing on the
   engine side. Evidence is not strong enough to file.
3. **Verified NOT an issue — prompt-over-modal stacking**: `popups.rs:56`
   routes ALL input to the topmost modal layer while visible ("topmost modal
   wins"), so the pair editor's `m` ChoicePrompt over the open form modal
   receives keys; the abstractcode-tui-era "equal-z prefers oldest" hazard
   does not apply to 0.2.22's overlay store. Also verified: root shortcuts
   (Esc→wizard-back, digits, q) are not on a modal's focus path, so no
   double-dispatch when a form is open (consistent with the Ctrl+L
   global-action rationale in `src/lib.rs:124-133`).

---

## Test-suite audit

**Faithful pins (verified by reading the tests against the probed Python
behaviors):**

- Both CLI liar classes ride fake CLIs shaped exactly like the probe: exit-0
  `❌ Error:` (the `--set-server-port` class) and exit-0 `✅` with no write
  (the `--set-app-default` class) — `worker.rs:749-794`. The honest-fake
  round trip pins the Eq path.
- RMW: unknown-section/key preservation, 0600 on the result, drift refusal
  leaving the file untouched, corrupt refusal leaving bytes untouched,
  refused-file CLI-vs-RMW split (worker-layer), missing-file creation.
- Secrets: `Debug` redaction on specs and commands, screen-level absence on
  all 8 screens, blank-refusal teaching, fingerprint convention incl. the
  `EMPTY` canonicalization (M1).
- Wizard: full phase walk with per-step screen+title asserts, digit/q
  refusals with reasons, Esc-back, f-finish re-arming digits.
- Chrome matrix at 80x24/100x24/60x16 with heavy fixtures; corrupt refusal
  at 60x16.

**Fidelity gaps in the fake-CLI worker tests:**

- The honest fake mutates ONE field in place. The real flags CLI rewrites the
  WHOLE file (live-verified: 18 top-level keys after one setter) and drops
  unknown keys — no test covers a spec whose expects pass while the CLI
  quietly restructured everything else, nor the `#FALLBACK`-mid-write branch
  (`worker.rs:424-432`, the corrupt-between-check-and-verb reset case) — that
  branch has zero coverage despite being the last line of defense for the
  historical data-loss incident.
- No fake covers `run_setter`'s nonzero-exit lane end-to-end through
  `execute_write` (exit-2 argparse refusals — the leading-dash case).

**Write-lane states with NO coverage anywhere:**

1. `eval_file_expect` with array-index paths — the exact P1-1 hole. A single
   test running `add_vision_fallback(…)` / `remove_vision_fallback(…)`
   builders through `execute_write` against a real post-write file would have
   caught it. The existing Eq/Cleared tests all use 2-segment object paths.
2. `handle_write`'s derived-view verification — `RouteEq`, `RouteCleared`,
   `ProfileExists`, `ProfileAbsent` evaluation (`worker.rs:246-319`) is
   entirely untested (it needs a Store+wake harness or extraction into a pure
   function; extract it — it's a pure fold over `routes_new`/`profiles_new`).
3. Multi-verb specs: no test for `clear_global_default`'s verb order or the
   verb-1-succeeded/verb-2-failed partial state (P2-2).
4. `Expect::SecretFp` through `execute_write` (only unit-tested at the
   `eval_file_expect` level).
5. Timeout and `file DISAPPEARED` (base `Some` + state Missing) refusals.
6. `parse_args` / `--wizard`/`--browse` / the adaptive boot default
   (`lib.rs:115-117`) — no unit test; only the pty smoke pins
   missing-file→wizard, and nothing pins existing-file→browse.
7. Headless: no editor SUBMIT coverage for the pair editor, profile editor,
   route editor, chain editor, toggle confirms, UNSAFE danger confirm, or the
   `x` clear verbs (only open/refusal/danger-default paths are pinned). The
   chain editor — the P1 surface — has zero headless coverage.
8. Same-field/in-flight interleavings (double-submit, edit-while-writing).

**A test that enshrines a defect:** `editors_refuse_while_python_refuses_the_file`
pins the UI door refusing ALL editors on a Python-refused file — the opposite
of the CHANGELOG's "RMW stays allowed" (P2-3). Whichever way P2-3 resolves,
this test or the claim must change.

**pty smoke (write phase):** real and valuable — wizard boot on a fresh
config, a real CLI setter, file-state proof, Python-side re-read. But it
covers exactly one CLI-routed field; no RMW-routed field, no secret, no pair
write ever crosses the real CLI in any automated lane.

---

## What holds (verified only)

- **Both probed liar classes are caught in the shipped binary.** Worker tests
  pin them, and the live P1 repro doubles as proof the verify lane runs after
  real CLI writes (it fired — with a broken evaluator, but it fired;
  exit-0 `❌` scanning is in `cli.rs:163-168`).
- **Refusals are structural at two layers.** Corrupt/unreadable refuse all
  writes; Python-refused refuses CLI verbs at the worker (`worker.rs:380-394`)
  with the reset-to-defaults rationale named; drift refusal on a fresh
  pre-write read; RMW re-reads fresh inside the verb, so a file that becomes
  corrupt mid-sequence refuses honestly (`worker.rs:455-461`); a CLI verb
  racing a corrupt file is caught by the `#FALLBACK` stderr lane
  (`worker.rs:424-432` — code path present, though untested).
- **Unknown keys survive direct writes by construction** (mutate the fresh
  raw `Value`; test-pinned incl. unknown sections), tmp name is unique
  (`.tmp-console-<pid>`, never Python's fixed `.tmp`), rename + 0600 both
  pinned.
- **Secrets:** structural redaction end-to-end (Arg::Secret Debug, journal
  labels authored redacted, `redact_scalar` in verify errors, fingerprint-only
  display). Live-verified that the flags CLI does not echo secret argv values
  in its error lines (`set_api_key` errors name the provider only; argparse
  refusals don't echo leading-dash values). Argv exposure remains the
  documented, accepted local-tool tradeoff.
- **Fabricated-selection law:** placeholders at index 0 in the enum, pair,
  profile-family, and route-provider selects; an on-disk alias value ("stt")
  correctly pre-selects NOTHING; model text stays the source of truth with
  discovery as an aid whose Loading/Failed/empty states are honest and whose
  failure explicitly says "type the id by hand".
- **Coupled writes route through the CLI and verify their couplings**:
  global default (fields + route input.text via fresh `config defaults
  --json`), embeddings (pair + route embedding.text), audio strategy (+
  explicit flag), vision pair (strategy + both fields), app defaults (the
  ✅-liar class covered by value expects). The field-route table matches the
  inventory's CLI column across all 17 sections (checked field by field).
- **`endpoint:` providers work through the pair editors**: Python's
  `_split_provider_model` only commits the `:` branch for known prefixes
  (`manager.py:52-71`), so `endpoint:x/model` and ollama's colon-bearing
  model ids both parse correctly from the console's `provider/model` join.
- **Unicode model names survive the round trip**: argv passes bytes through;
  Python writes `ensure_ascii` escapes; serde decodes back to identical
  strings; `Expect::Eq` compares decoded values.
- **Wizard:** 8 phases in the CLI's order plus orientation and review, every
  step skippable, digits and q refused WITH reasons (root-level so the
  refusal renders even though PageHost jumps are disarmed via empty chords +
  `number_jump(false)`), Esc back, f finish, w re-enter, browse fully free
  after finish; wizard on a corrupt file degrades to the corrupt panel with
  editors refused (code-verified; browse is the adaptive default for corrupt
  since only Missing selects wizard).
- **M1 fixes intact** (all still test-pinned green): both refusal-banner
  lanes (fold-side `python_refusals`, CLI `#FALLBACK` stderr), header
  state-first ordering (`ui/mod.rs:566-577`), agreement lines with
  component-wise path compare and the DIFFERENT-FILES alarm, fingerprint
  canonicalization, api_keys broken-state visibility, empty-secret
  classification.
- **Gates:** build green, 33 unit + 31 headless green (+1 ignored minter),
  clippy zero, engine pinned at 0.2.22, `ureq` still absent per the M1
  review's P3-6.
