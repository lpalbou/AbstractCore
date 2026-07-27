# Core console TUI — setup pattern & recommendations (PATTERN-PROBE)

- Date: 2026-07-25
- Prepared for: the launch of `abstractcore/console-tui` (the AbstractCore
  configuration console — validator app #3 on AbstractTUI)
- Probe scope (all read-only): `abstractgateway/console-tui` at 0.3.6
  (Cargo/src/tests/docs/CHANGELOG/LAUNCH-PROMPT), `agora-tui`'s
  LAUNCH-PROMPT.md, the engine repo at **0.2.22** (docs/api.md, CHANGELOG
  0.2.8→0.2.22, backlog overview + field bands, the engine seat's
  recommendation letter `reviews/gateway-console-recommendations-2026-07-25.md`),
  and the field findings 0900–1050 across both repos.
- Peer: CORE-PROBE owns the abstractcore config-surface inventory. This
  report is seam-agnostic where their inventory decides (marked ⇠CORE-PROBE).

---

## 1. The pattern verdict

**Copy the gateway console-tui setup nearly wholesale.** It is the proven
member of the pair for this app class: same shape (a config console:
wizard + browse over a small set of screens, forms, pickers, tables,
secrets), it survived six adversarial cycles plus live operator use, and
its architecture was explicitly endorsed by the engine seat's survey
("the worker/WakeHandle threading contract, the write→verify→journal law,
the fabricated-selection law … all read as the engine's documented idioms
applied correctly; nothing stale found").

What made it succeed, in order of load-bearingness:

1. **A launch prompt with verified-live grounding.** §2 of the gateway
   brief was studied from source with exact dev commands, port hazards,
   boot patience, weak-token fail-fast, and the `-P` footgun. The builder
   never guessed an endpoint or a startup incantation. The agora brief
   had the same virtue (live-probed hub, schema-accurate example rows,
   verified traps like receipt-recording reads).
2. **Design laws stated as paid-for lessons, not style advice** (the
   fabricated-selection incident, body-over-transport, secrets
   discipline, honest states, verify-after-write). Every one of these
   shows up in the shipped app as a named, test-pinned law.
3. **The feedback protocol as half the mission**, with the band, the
   file grammar, and an inline skeleton in the brief itself. The console
   filed 16 findings (0900–1050); three of its four P1s were fixed in
   the engine within days (fusion class → 0.2.15, `s`-consumption →
   0.2.17, popup anchors → verified fixed by 0.2.20) and round-tripped
   back as one-line version bumps.
4. **One worker thread + Loadable stores + headless CaptureTerm suite**
   — the engine's live-data law applied without exception, which is what
   made 47+ headless tests possible over the real UI with zero network.
5. **The engine round-trip channel in both directions**: upgrade briefs
   (`reviews/gateway-console-v2-upgrade-prompt.md`) and recommendation
   letters flowed engine→app; findings flowed app→engine. The 0.2.12
   PageHost/Drawer adoption deleted ~83 lines of hand-rolled tab bar
   including a mouse hit-test that had already drifted once.

What to improve (hindsight from the console's own incident history):

- **Ship the chrome-survival matrix with the first screen, not after the
  first operator screenshot.** The vanishing-title-bar incident (0.3.3;
  findings 1020/1030) existed because light fixtures never over-demand
  height — "an app that looked correct through every empty-fixture
  review loses its header the first time a real gateway serves seven
  entities." The new brief must mandate heavy fixtures + tight sizes
  from day one.
- **Born-knowing the focus-init recipe.** Two modals shipped with the
  dead-keys trap (finding 1000) and cost a night hour of pty forensics.
  The recipe is one line; put it in the brief's design laws, not in the
  findings the builder discovers later.
- **Expect same-day engine releases.** The console launched against
  0.2.8 and lived through 0.2.9/0.2.12/0.2.16/0.2.17/0.2.20 during its
  own build. The brief should name the adopt-on-release protocol (check
  the engine CHANGELOG before working around anything; expect upgrade
  briefs; gate every bump on your own suite + smoke).
- **One canonical findings location.** The band contract says findings
  live in the ENGINE repo, but the console's later findings (1020–1050)
  physically live in the app repo (`console-tui/docs/backlog/proposed/field-gateway/`)
  while 0900–1010 live engine-side. The engine seat read both, so
  nothing was lost — but the new brief should re-pin one location
  (engine repo) so ledgers and round-trip status live in one place.
- **Register a band that won't collide.** field-gateway was registered
  0900–0990 and overflowed to 1050; field-agora was registered 0800–0890
  and overflowed to 0910 (numerically colliding with field-gateway's
  0900/0905/0910 — harmless only because bands are per-directory).
  Propose field-core with headroom and a stated overflow rule (§6).

From the agora brief specifically, adopt: the **staged milestone plan**
(M1 one-evening vertical slice → M2 widen → M3 harden + soak + report),
the **budget note** ("treat significant overrun as a finding about the
engine track — file it, don't push through silently"), and the **ground
rules about peer lanes** (don't touch other apps' paths or bands).

---

## 2. The setup recipe (project skeleton)

Create `abstractcore/console-tui/` inside the abstractcore package repo
(the gateway precedent: the TUI lives inside the Python package's repo so
one package serves both surfaces; `publish = false`). Target directory
does not exist yet — this is the full initial contents:

```
console-tui/
  Cargo.toml
  LAUNCH-PROMPT.md          # the builder brief (§3) — checked in, it IS the charter
  README.md                 # run/test/layout; updated as screens land
  CHANGELOG.md              # per-wave entries with gate lines (the console's discipline)
  src/
    main.rs                 # 4 lines: argv → lib::run_cli → exit code
    lib.rs                  # arg parsing, headless guard, mount, worker spawn, auto-probe
    api.rs                  # the side-effect client + error taxonomy   ⇠CORE-PROBE seam
    store.rs                # Store of signals; Loadable<T>; typed rows; reset_domains
    worker.rs               # ONE thread owning all side effects; Cmd enum; three-phase writes
    health.rs               # (phase 2) centralized connection/validity authority
    ui/
      mod.rs                # root shell: wizard/browse over PageHost; UiState; Ctx; shared form plumbing
      util.rs               # line/span/field helpers (with the R4 rect.is_empty guard FROM DAY ONE)
      <screen>.rs           # one module per screen, plain fn(cx, &Ctx) -> View
  tests/
    headless_ui.rs          # CaptureTerm+Driver harness; fixtures; chrome matrix
    live_e2e.rs             # #[ignore] live test: writes + verify + RAII restore
  scripts/
    pty_smoke.py            # keyboard-driven live proof of the definition of done
  docs/backlog/proposed/    # app-side items only (worker design notes etc.), NOT engine findings
```

Cargo.toml shape (verbatim pattern from the precedent, updated pin):

```toml
[package]
name = "abstractcore-console"
version = "0.1.0"
edition = "2021"
rust-version = "1.87"        # MSRV floor comes from the engine
license = "MIT"
description = "…"
publish = false

[lib]
name = "abstractcore_console"
path = "src/lib.rs"

[[bin]]
name = "abstractcore-console"
path = "src/main.rs"

[dependencies]
abstracttui = "0.2.22"       # comment the version rationale, keep it current
ureq = { version = "2.12", default-features = false, features = ["tls", "gzip"] }  # if the seam is HTTP ⇠CORE-PROBE
serde_json = "1"
```

Dependency policy: the engine's five-dep posture is the engine's, not the
app's — but the precedent pair (ureq blocking + serde_json) fits the
worker model exactly. **No async runtime.** If CORE-PROBE's inventory says
the seam is local config files rather than HTTP, drop ureq and keep
`std::fs` in the worker — the architecture is identical either way (see
"seam note" below).

The architectural invariants the skeleton encodes (each one is a law the
precedent paid for):

- **lib.rs**: `run_cli(argv) -> i32`; `--help`/`--version`; headless
  guard (`!term::have_tty()` → print a skip line, exit 0 — CI-safe);
  `--theme` + `ABSTRACTTUI_THEME`; secrets via env preferred over argv
  ("argv is visible in `ps`"); auto-probe/auto-load at boot
  (zero-keystroke first paint of real state); at quit `drop(tx)` and
  deliberately **no join** of a worker possibly mid-call.
- **store.rs**: every remote/disk domain is a
  `Loadable<T> { NotAsked, Loading, Ready(T), Failed(err) }` — the four
  honest states render distinctly, "never render a guess." Typed rows
  parsed tolerantly (unknown fields ignored, absent = `None` → `—` with
  a reason). `Store::reset_domains()` uses an **exhaustive destructure
  (no `..`)** so adding a field fails compilation until the
  reset-or-exempt decision is made — the stale-data class becomes
  structurally impossible.
- **worker.rs**: one background thread owns the client and every side
  effect. Commands in via `mpsc::Sender<Cmd>`; results back **only** as
  closures posted through `WakeHandle` (signals are UI-thread-only — the
  engine law). Writes are **three-phase by construction**: write →
  verify via a fresh read → journal entry; putting it in the worker
  makes it impossible for a screen to forget. `form_id` correlates a
  write with the form modal that issued it (close on success, stay open
  with data intact on failure). `Secret`/`Body` newtypes redact in their
  own `Debug` so a new Cmd variant cannot leak a key. The worker
  survives per-command panics loudly and releases the issuing form.
  Serial total-order is a correctness feature — record a second-lane
  design if it ever hurts, don't build it speculatively (the precedent's
  backlog 0001).
- **ui/mod.rs**: screens are **plain component functions** over a
  cloneable `Ctx { tx, overlays, quitter, store, ui, modal, … }`.
  Durable UI state (form fields, selections) lives in a Copy `UiState`
  of signals created at the ROOT — PageHost disposes page scopes on
  every switch, so anything screen-local dies on tab change by design.
  ONE modal slot (`Rc<RefCell<Option<Modal>>>`) — stacked modals are an
  engine hazard; sequence them. A `prompt_open` count gates anything
  that could stack over a live ChoicePrompt (finding 0945: equal-z keys
  belong to the OLDEST layer — an invisible prompt under a visible modal
  owns the keyboard).
- **Dual-mode shell** (the wizard/browse pattern, engine-hosted since
  0.2.12): one `PageHost` carries the tab bar + page region for all
  screens; `ui.screen: usize` stays the source of truth with a two-way
  equality-guarded bridge to PageHost's string `active`. Browse mode
  arms free navigation (`.number_jump(true)` + Ctrl+PgUp/PgDn chords);
  wizard mode fully DISARMS the free surface (empty chord sets, digits
  off) and the app's gate logic (`wizard_next`/`wizard_back`, digit
  refusals **with reasons**) writes the screen signal. Each wizard step
  carries one muted goal line answering "what is this step for / can I
  skip it" (the 0.3.6 lesson: the wizard must GUIDE, not just gate).
- **Injected side effects for testability**: the precedent's `ProberSlot`
  pattern — production installs a thread-spawning closure at mount;
  the headless harness installs a recorder. Any out-of-band side effect
  (health probes, file watchers) gets an injection slot, never a direct
  spawn from UI code.

Seam note (⇠CORE-PROBE): the gateway console's seam is an admin HTTP API,
so `api.rs` is a ureq client with an error taxonomy
(unreachable / 401 / 403 / detail-carrying HTTP / protocol). AbstractCore's
surface is likely a mix: local config files (`abstractcore/config/`:
manager, capability_defaults, provider_profiles, vision_config) plus
live provider probes (model discovery, test generations). The pattern
holds unchanged — the worker owns file I/O exactly as it owns HTTP, the
error taxonomy becomes {file unreadable / parse error / validation
refused / provider unreachable / provider auth}, and verify-after-write
becomes **write config → re-read the file → journal**. If a running
AbstractCore server is an optional target, that is a second client in
the same worker, not a second architecture.

---

## 3. The launch-prompt skeleton

Model: the gateway brief (the stronger of the two for this app class),
with the agora brief's staged plan and budget note folded in, plus the
hindsight sections. Write it as `LAUNCH-PROMPT.md` in the app directory,
checked in — the precedents prove the brief doubles as the app's charter
and gets cited by CHANGELOG entries for the life of the build.

Section by section, with what goes in each:

1. **Header — who you are, what this is.** One paragraph: the app, where
   it lives, the engine pin (`abstracttui = "0.2.22"`, crates.io), the
   target surface, "zero <host>-side changes." Name it validator app #3
   and state plainly: **filing engine findings is as much the mission as
   shipping the app.** Point at the epic/backlog item that rules scope.
   State that a compiling scaffold exists (build one first — both
   precedents launched from a green scaffold, so cycle one starts at
   "replace main.rs", not "fight the toolchain").
2. **What to build.** The screen list in wizard-first order, each with
   its verbs. The reference UX if one exists (the gateway brief pointed
   at the served web console "open it beside your TUI"; for core the
   reference is `abstractcore --config`'s phases and the installer
   wizard ⇠CORE-PROBE). Explicitly: wizard and browse share every screen
   component; start with a narrow end-to-end path, then widen.
3. **Target grounding (studied from source, verified live, dated).** The
   most load-bearing section of the precedent. For every surface: exact
   paths/endpoints/schemas, what each field means, secrets semantics
   (write-only keys, once-shown tokens), error shapes, and — critically —
   **the exact dev commands the builder can run before writing any
   Rust**, with the operational footguns spelled out (interpreter paths,
   ports already taken by the operator's own services and NEVER killing
   them, boot patience, config-file locations and their locking story
   ⇠CORE-PROBE). Every claim dated and verified, not remembered.
4. **Wizard shape (suggested, not mandated).** Step order, per-step
   validation gates, Esc semantics, keyboard-first, what persists
   client-side (nothing secret; say on screen what is persisted where).
5. **Design laws (paid-for lessons — do not relearn).** Verbatim from
   the precedent, they all apply to a config console:
   - the fabricated-selection law (placeholder at index 0, explicit
     default-vs-override mode, "Applies now: … (source)" resolution
     lines, model pickers reset on provider switch — never a fabricated
     pair);
   - body over transport (a 200 carrying `ok:false`/error renders as
     failure; for the file seam: a successful write syscall is not a
     successful config change — re-read and compare);
   - secrets discipline (masked input, never echo a stored secret into
     an edit form, fingerprint display + blank-keeps + explicit clear;
     structural redaction in Debug);
   - honest states everywhere (unreachable ≠ unauthorized ≠ empty ≠
     loading; `—` with a reason beats a guess);
   - verify after write (re-render from the verify read, never from
     optimistic local state);
   - **refusals speak** (every advertised key that refuses SAYS why —
     toast + footer; dead gestures are the class the precedent hunted
     all build long);
   - **the focus-init law** (new — from finding 1000, see §5.2);
   - **chrome pins from day one** (new — from findings 1020/1030, see
     §5.1).
6. **Engine guide (0.2.22).** Docs pointers (docs.rs, repo `llms-full.txt`,
   `docs/api.md`, `docs/design/01-damage-contract.md`). The examples to
   read first (`components.rs` — THE component pattern; `decide.rs` —
   decision gates; `shell.rs` — PageHost + Drawer + ThemeSwitcher
   footer; `attachments.rs` if file fields exist; `hello.rs` — the
   skeleton). The load-bearing API list **verified present against the
   pinned version** (§4 below is that list). The "known engine state
   you will feel" paragraph: app-kits 0510 (form kit) / 0520 (wizard
   flow) / 0530 (rich table cells) are still proposed, not shipped —
   this build is more promotion evidence; hand-roll field rows,
   validation gating, and badges-as-styled-cells, and file what you
   wish existed.
7. **Dependency policy.** Engine five-dep posture is the engine's; apps
   take deps. The ureq+serde_json precedent pair, or std-only if the
   seam is purely local files. No async runtime — nothing here needs one.
8. **THE FEEDBACK PROTOCOL.** Band name + range (field-core, §6), the
   one-file-per-finding grammar with the inline markdown skeleton
   (Metadata created/status/severity/class; Context with the exact
   composition; Current code reality citing engine `file:line` against
   the PINNED version; Repro; Workaround in the field "delete when
   fixed"), the README table row, the severity ladder (P1 blocks / P2
   costs real time, workaround holds / P3 paper cut), the class
   vocabulary (bug / footgun / API gap / capability gap / UX defect /
   rendering defect / feature). Plus the two protocol facts the
   precedents proved: expect same-day round-trips (check the engine
   CHANGELOG and the completed tables before working around something
   twice), and findings live in the ENGINE repo only — the app repo
   carries app-side design notes, never the engine band.
9. **Quality bar.** cargo build/test green after every meaningful
   change; clippy zero (the engine holds zero warnings — match it);
   headless CaptureTerm coverage for every form's logic with fixtures,
   no network in tests; keyboard-first with visible hints; theme tokens
   only (`ABSTRACTTUI_THEME=` must restyle everything); zero idle cost;
   headless guard prints a skip line and exits 0; **no git operations —
   the maintainer owns commits** (both precedents carry this rule).
10. **Non-goals.** What this console deliberately does not touch (for
    core: ⇠CORE-PROBE, but the precedent's shape holds — no host-side
    code changes; configuration only, not monitoring; name the adjacent
    surfaces left alone and why).
11. **Staged plan + budget note** (from the agora brief): M1 = one
    vertical slice end-to-end (connection/probe + one config domain
    written + verified), M2 = widen to all screens, M3 = harden
    (adversarial pass, chrome matrix, live smoke) + the experience
    report. Budget stated; significant overrun is itself a finding.
12. **Definition of done.** One end-to-end scenario driven only by the
    keyboard against a REAL local target, with writes verified by
    re-read, headless coverage for every step's form logic, findings
    filed for every friction point ("an empty findings directory after
    this build would itself be a surprising claim — say so explicitly
    in your report if it happens"), and a final report: what shipped,
    the live evidence, test counts, findings list.

---

## 4. The day-one engine kit (0.2.22)

The honest kit for a config console — selective, with one-line whys.
Everything below is verified present in the 0.2.22 tree (api.md +
CHANGELOG).

**Take on day one:**

- **PageHost** (0.2.12) — screens. The engine owns the tab bar,
  windowing, clicks, chords, digit jumps; one draw plan feeds paint AND
  hit-test so they can never drift (the class that bit the hand-rolled
  bar). Wizard mode = disarm free nav, keep writing the `active` signal.
- **ChoicePrompt / ChoiceSequence** — every decision gate: apply
  confirms, danger-tinted deletes, "emptied section: reset vs deny-all"
  consequences. Wrap it once (`confirm_danger()`) so keep-defaults are
  structural. Respect the 0945 stacking law (one prompt-open count).
- **Select / Combobox / MultiSelect + SelectHandle** — enum fields,
  provider/model pickers (Combobox for long model lists: typing
  filters). The modal-popup displacement P1 (finding 1050:
  Select/Combobox popups inside a Modal anchored layer-local and opened
  at the screen's top-left) is FIXED as of 0.2.20 — the precedent
  verified it with a regression pin at its 0.2.12→0.2.20 bump
  (`select_popup_inside_modal_opens_adjacent_to_its_field`); no anchor
  workaround needed, but copy that pin habit (the engine CHANGELOG
  carries no explicit 1050 entry — the pin is the proof).
- **TextInput `.masked(true)`** — API keys. Masked on screen AND in the
  a11y export. Pair with write-only semantics: "key stored (fp: …)" /
  "no key", blank keeps, explicit clear.
- **Table + `on_activate`** (0.2.16/0.2.17) — inventories (providers,
  routes, profiles). Enter/Space/double-click native; consumed only when
  bound; the `s`-swallowing footgun (0980) is fixed in 0.2.17. Bind
  activation to the SAME handler the edit key calls so refusal reasons
  ride along. Width-aware column drops remain app-side (finding 0900
  still open — drop secondary columns, never the payload column).
- **List** — thin pickers/rosters where Table is overkill; know the
  vocabulary split (`on_select` fires on movement; `on_activate` is the
  commit).
- **Disclosure** (0.2.11) — config groups and advanced sections; the
  progressive-disclosure widget (the precedent's 0.3.6 simplification
  wave moved this direction by hand). Folded by default costs one row;
  `max_body_rows` caps server-fed lists so reference data can never
  starve working data (the P2.1 lesson: unbounded rows inside pinned
  chrome is the 1020 class reborn).
- **Drawer** (0.2.12) — the detail inspector beside a roster (Passive
  focus mode: the list keeps the keyboard). Decisions stay Modals; the
  drawer is a reader.
- **ThemeSwitcher** (0.2.21) — the one-cell theme control for the
  chrome; menu face or `::toggle()`. Deletes any hand-rolled theme
  cycling; `ABSTRACTTUI_THEME` still honored at boot.
- **Block::on_close** (0.2.22) — dismissible panels (mouse-only ✕,
  never focusable; keyboard close stays an app key). Use for optional
  info/journal panes; survivors re-flex into the space.
- **FilePicker + `on_paste`/`input::paste::classify`** (0.2.20) — only
  if the config surface has file-path fields (model paths, CA bundles,
  config import/export ⇠CORE-PROBE). The classifier turns a terminal
  file-drop into a picked path with the refuse-when-ambiguous policy;
  don't hand-roll drop parsing.
- **Screenshot family** (0.2.14) — `self.term.screen().screenshot()` in
  the headless harness (one line per scene → deterministic `.svg`
  evidence), and one root shortcut on `app::request_screenshot` for
  live operator reports. The engine seat's letter: "your headless suite
  is one line away from minting SVG evidence."
- **testing::CaptureTerm + app::Driver + `push_resize`** — the whole
  test story (§5.4). Fixed `Capabilities` in RunConfig so the host TERM
  never steers assertions.
- **`use_startup_notices` rendered in the footer's idle lane** (recipe
  R3) — the engine names every zero-collapse into it; a notice nobody
  renders is a debugging session someone else pays for. Filter the
  ambient caps summary (the precedent found it reads as a permanent
  warning); humanize the raw `layout:` diagnostic for operators and
  keep the verbatim line behind a debug env var.
- **Toast + Badge + the notice lane** — acknowledgment discipline:
  every probe/refresh/refusal lands a visible, timestamped trace.
- **Theme tokens only** (`use_theme(cx)`) — no hardcoded colors, ever.

**Explicitly NOT needed (say so in the brief so nobody decorates):**

- **Meter / AudioScope / TimeSeries / Sparkline / charts** — a config
  console fetches no live levels and no time series; the one latency
  number is a scalar. The engine seat's letter endorsed exactly this
  non-adoption for the gateway console ("nothing to chart honestly").
  If a polled stats surface ever appears, the engine owns the ring and
  ballistics — do not hand-roll then.
- **`reactive::connection` + Backoff** — the console class is
  PROBE-shaped, not stream-shaped (finding 0950, engine-endorsed):
  the operator fixes a URL/key and probes on demand; auto-retry against
  a wrong token would hammer 401s. Hand-roll the small `ConnPhase`
  enum instead (or its file-seam analogue), plus generation-guarded
  settles.
- **Feed / follow_tail** — no streaming transcript in a config console
  (revisit only if a log-tail screen enters scope).
- **Graph/mermaid extension crates** — config relations are one hop
  ("covered by", "derives from"); a table with a state column plus a
  refusal reason in prose beats a graph that decorates.
- **Canvas / Viewport3D / gfx images / PushToTalk** — no fit.

**The R1–R4 small-terminal recipes** (api.md "Small terminals & content
pressure" + `reviews/wave10/size-ratio-sweep.md` §3) go in the brief
verbatim — they are app-side jobs by design:

- R1: chrome you want incompressible gets `shrink(0.0)` (bare
  `line(n)`/`Cells(n)` is a starting size, not a promise).
- R2: oversized middles go in a `Scroll` (default `grow(1) basis(0)`
  absorbs volume with no sibling pressure).
- R3: render `use_startup_notices` somewhere visible.
- R4: draw closures guard `rect.is_empty()` and clip on BOTH axes —
  the engine makes zero-area unreachable (0.2.15), but a partially
  crushed rect still arrives smaller than asked. Put the guard in the
  shared `ui/util.rs` line helpers on day one and every hand-rolled row
  inherits it.

---

## 5. Born-knowing-these lessons

The condensed field history the new builder must start with — each cost
the precedent real time and is now either engine-fixed (adopt the fix)
or an app-side recipe (apply from day one).

1. **Chrome pins + heavy fixtures from the first screen** (findings
   1020/1030, the vanishing-title-bar incident). Fixed rows silently
   flex-shrink to zero at the ROOT when loaded content over-demands
   height; the trigger is DATA VOLUME with a rounding threshold, so
   empty-fixture reviews never catch it. Pin `shrink(0.0)` on the
   header, separator, footer rows, message/safety slots, and result
   slots; give every screen a heavy fixture; ship the chrome-survival
   matrix test (all screens × both modes × tight sizes, asserting
   header row 0 / separator row 1 / tab bar row 2) in the SAME wave as
   the shell. The fusion half (zero-area rects painting over siblings)
   is engine-fixed since 0.2.15 — collapse is clean absence now — but
   pins remain the difference between "content yields" and "chrome
   vanishes".
2. **The async-modal focus-init recipe** (finding 1000, still open at
   engine HEAD as a structural ask). A modal whose only focusables
   mount after an async load has NO focus owner at mount — every key
   except Tab is dead, including Esc, indistinguishable from a frozen
   app. Recipe: `.focusable().autofocus()` on the modal's OUTER content
   root (built once in the modal scope, never inside a regenerating
   region). Audit EVERY modal for an at-mount focusable; the engine
   letter verified the precedent applies it consistently — start
   consistent.
3. **Honest states are the product.** `Loadable<T>` four-state rendering
   everywhere; a connection-phase taxonomy where unreachable ≠ 401 ≠
   403 ≠ "something answered, but not like the target" (port squatters
   are a documented recurring event); body over transport; the
   fabricated-selection law (an unconfigured picker showing catalog
   row 0 as if configured was one accidental Save from routing through
   a keyless provider — placeholder index 0, default-vs-override mode,
   "Applies now:" lines derived only from verified state); refusals
   always say why; probes/refreshes always acknowledge visibly (the
   "probe does nothing" incident was pure acknowledgment failure — the
   probe worked in 2 ms and repainted identical pixels).
4. **One worker, three-phase writes, injected side effects.** All I/O on
   one background thread; results as posted closures; write → verify by
   re-read → journal in the worker so screens can't skip it; `form_id`
   correlation; `Secret` newtypes redacting structurally; reset-on-
   reconnect via ONE exhaustive-destructure reset function (the
   stale-data P1: both hand-maintained reset lists missed the seven
   newest domains — a same-named entity on gateway B rendered gateway
   A's data). Out-of-band probes get injection slots so the headless
   harness can install recorders. Quit never joins a worker mid-call.
5. **The feedback loop is half the mission — and it's fast.** File one
   finding per friction point with engine `file:line` against the
   pinned version and the workaround-to-delete; add the README table
   row; check the engine CHANGELOG before working around anything
   (three of the console's four P1 classes were engine-fixed within
   days — fusion 0.2.15, `s`-swallow 0.2.17, popup anchors verified
   fixed by 0.2.20 — and the new app starts AFTER those fixes). Expect upgrade briefs and
   recommendation letters back from the engine seat; gate every version
   bump on your own suite + smoke, then delete workarounds the release
   obsoletes.

Also carry (smaller, but each was a real diagnosis): the headless settle
rule (`turns(n)` = the effect-depth of the change, not magic); negative
asserts drain the WHOLE command queue (absence-of-match hides
mis-sends); test with fixed terminal Capabilities; know that PageHost
disposes page scopes on switch (durable state lives at the root); the
wizard's locked-future-steps look is still an open engine ask (1010 —
PageHost has one active/idle style; interim: reactive `.badge(id, …)`
or accept refusal-with-reason as the teacher).

---

## 6. The field-band proposal

Current band map (docs/backlog/overview.md "Bands:" + observed reality):

| Range | Owner | Status |
| --- | --- | --- |
| 0010–0090 | live-data | registered |
| 0100–0190 | app-widgets | registered |
| 0200–0298 | ports + first-app findings | registered |
| 0300–0390 | control-plane | registered |
| 0400–0490 | extensions | registered |
| 0500–0590 | app-kits | registered |
| 0600–0690 | media-av | registered |
| 0700–0790 | games | registered |
| 0800–0890 | field-agora | registered — **overflowed to 0910** in its dir |
| 0900–0990 | field-gateway | registered — **overflowed to 1050** (1020–1050 currently in the app repo) |
| 0990 | wave11 (one item) | numeric reuse, different dir |

Proposal: **field-core owns 1100–1190**, registered in
`docs/backlog/proposed/field-core/` with a README carrying the standard
house-grammar header and the findings table. Rationale: 1000–1090 is
already field-gateway's de-facto overflow zone (1000–1050 exist), and
both prior bands overran their 90-slot allocation — so field-core gets
the next clean century with the overflow rule stated up front: *if the
band fills, extend into 1190+ within the same directory and note it in
overview.md — never spill into a neighbor's century.* The overview.md
"Bands:" section gains one line at registration time (the engine seat's
edit, not the app builder's).

Contract restated for the brief (what both precedents converged on):

- Findings live in the ENGINE repo:
  `abstracttui/docs/backlog/proposed/field-core/NNNN_snake_case_title.md`
  plus a row in that directory's README table. The only engine-repo
  files the builder touches.
- One file per finding; house grammar: Metadata (created / status /
  severity P1|P2|P3 / class), Context (the exact composition), Current
  code reality (engine `file:line` at the pinned version), Repro
  (steps or a ~10-line snippet), Workaround in the field ("delete when
  fixed" — the engine fix's acceptance test is deleting it).
- Class vocabulary: bug / footgun / API gap / capability gap / UX
  defect / rendering defect / feature.
- Workaround app-side first, always — never stall the build on the
  engine, never edit the engine.
- Expected finding sources for THIS app (state them so the builder
  watches): form-field focus order and validation gating (0510/0520
  evidence — still unshipped), Select/Combobox in dense forms, Table
  badge/cell gaps (0530-shaped), masked-input UX, Disclosure-heavy
  layouts, FilePicker in modals, and wizard-state persistence
  (control-plane 0340 still unshipped — note where it would have saved
  you).

---

## 7. Handoff notes

- **To CORE-PROBE / the brief author**: §3.3 (target grounding) is
  yours to fill — config file locations + locking/concurrent-writer
  story, the `abstractcore --config` phase list as the wizard's
  reference UX, provider-key storage semantics (write-only? at-rest
  plaintext? fingerprintable?), capability-route vocabulary overlap
  with the gateway console (same
  `abstractcore/config/capability_defaults.py` kinds/modalities/tasks —
  the routes screen can share its design laws verbatim), model
  discovery / test-generation lanes for the "Test" verb, and whether a
  running AbstractCore server is an optional second target. Each claim
  verified live and dated, per the precedent.
- **Scaffold first**: both precedents launched from a compiling
  scaffold ("cargo build green, q quits, headless exit 0") so the
  builder's first cycle is app work. Whoever cuts the directory should
  include the scaffold and say so in the brief's header.
- **The engine seat's round-trip channel**: plan for recommendation
  letters and upgrade briefs (`reviews/` in the engine repo is where
  they land). The new app should expect at least one same-week engine
  release during its build and treat bumps as gated, test-verified
  adoptions — the precedent's bump gate (suite green at the pin bump
  BEFORE any migration edit, then adopt, then smoke) is the discipline
  to copy.
