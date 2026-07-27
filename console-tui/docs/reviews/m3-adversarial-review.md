# M3 adversarial review — abstractcore-console (test & prove)

- Date: 2026-07-25 · Reviewer: adversarial subagent (spec-conformance + edge-case audit)
- Scope: the M3 probe lane (`probes.rs`, the worker's `Cmd::Probe`/`handle_probe`/
  `probe_generation`/`tcp_probe`, `cli.rs`'s `resolve_chat_bin`/`run_chat`/
  `run_raw_at`, `store.rs`'s `tests`/`record_test`/`probe_busy`), its UI surfaces
  (`ui/providers.rs::open_test_picker`, `ui/routes.rs::test_selected`,
  `ui/mod.rs::send_probe` + the `g` shortcut + footer hints, `ui/review.rs`'s
  evidence block, `ui/wizard.rs`'s review goal), the M3 halves of
  `scripts/definition_of_done.py` and `scripts/pty_smoke.py`, plus regression
  checks on the M1/M2 surfaces this wave touched (post-write refresh gating,
  `wait_fresh`, the capture minter).
- Baseline: working tree at review time; `abstracttui` 0.2.22 (Cargo.lock);
  Python side `abstractcore 2.13.38` (framework venv).
- Gates re-run by the reviewer: `cargo build --all-targets` green, `cargo test`
  50 unit + 35 headless green (+1 ignored minter), `cargo clippy --all-targets`
  zero warnings.
- Cross-reads: `abstractcore/config/main.py` (`test-provider` → `_handle_config_models`),
  `abstractcore/providers/registry.py::get_available_models`, the per-provider
  `list_available_models` impls (openai/ollama/openai-compatible/anthropic/
  huggingface/mlx), `abstractcore/utils/cli.py` (the `abstractcore-chat` entry),
  `abstractcore/config/manager.py` (`set_global_default_model`, `AppDefaults`,
  `get_app_default`).
- Live probes: scratch configs only (`ABSTRACTCORE_CONFIG_FILE` env-pointed at
  `/tmp` dirs, deleted after; `~/.abstractcore` never touched). Verified live:
  `test-provider lmstudio` → ok:true/count:48 (LM Studio serving this machine);
  `test-provider ollama` → ok:true/count:0/errors:[]/exit 0 (ollama down — the
  builder's dead-server claim reproduces exactly); `test-provider
  endpoint:<scratch-profile>` → 48 models (the picker's target string is
  accepted); `test-provider` on a missing profile → `❌ Error: Unknown provider`
  exit 1; `abstractcore-chat --provider endpoint:x` → argparse exit 2 (no
  network); two pty drives of the real binary (dead-ollama NotProven lane;
  the endpoint-default `g` lane). One standalone `rustc` repro of the
  `is_log_line` panic, and one of Rust's `to_socket_addrs("localhost")`
  ordering. Zero generations were spent: every verdict lane needed was
  reachable without one (the DoD's own green run already covers the Proven
  generation lane).
- Line numbers are from the working tree at review time.

## Verdict

The M3 core idea is right and mostly real: the three-state verdict vocabulary is
enforced where it matters (zero-count success can never fold to Proven — pinned
and live-verified against the actual dead-ollama CLI shape), the folds read the
live payload shapes I re-probed, the generation pre-check genuinely kills the
chat CLI's silent huggingface-default lie (the `AppDefaults` dataclass defaults
ARE `huggingface/Qwen3-4B…` — confirmed in `manager.py`), the picker fabricates
nothing, probe argv is injection-safe by construction, and `probe_busy` is
cleared on every arm including the worker's panic handler. The DoD walk and the
pty test phase assert glyph-prefixed, value-specific needles that cannot pass
vacuously on their happy paths.

But M3 ships with **two P1s, both live-proven**. First: the definition-of-done
verb itself lies on the framework's flagship config shape — a default route on
an `endpoint:<id>` profile (which the console's own pair editor offers and the
flags CLI happily persists) makes `g` run `abstractcore-chat --provider
endpoint:…`, which dies on a hardcoded argparse `choices` list, and the console
folds that to **✗ FAILED** on a fully working route (proven end-to-end in a pty:
"✗ generation test (default route): abstractcore exited with 2 … invalid
choice"). Second: the TCP disambiguation — M3's one honest addition over the
CLI's ambiguous zero — probes only the FIRST resolved address; on macOS
`localhost` resolves `::1` first and LM Studio serves IPv4 only (both verified
live, including Rust's own resolution order), so the exact branch the feature
exists for (server UP, zero models — the fresh-install wizard audience) reports
"the server looks DOWN" as evidence. Around them: a reproducible panic in the
generation fold on legitimate multibyte model output, a single-flight guard
that is advisory (the CHANGELOG's "single-flight" claim doesn't hold while the
worker is busy), route-test evidence that never replaces across route edits
(contradicting both the worker's own comment and the CHANGELOG's
"latest per target"), and an endpoint-route reach gap that makes the same
target produce weaker evidence from the Routes screen than from Providers. Fix
the two P1s and the panic before calling the test lane trustworthy; the rest is
schedulable. The negative verdict lanes also need an automated e2e pin — today
a regression in the NotProven fold would ride every green gate.

---

## P1 findings

### P1-1 — `g` on an `endpoint:<id>` default route reports ✗ FAILED for a working route: the definition-of-done verb lies on the framework's flagship config shape (live-proven end-to-end)

The chat CLI validates `--provider` against a hardcoded argparse choices list
that knows only the 10 static providers:

- `abstractcore/utils/cli.py:2558-2560` — `choices=['openai', 'anthropic',
  'openrouter', 'portkey', 'openai-compatible', 'vllm', 'ollama',
  'huggingface', 'mlx', 'lmstudio']`. `endpoint:<id>` is not in it.

The console's generation probe pre-resolves the configured default from the
file and passes it explicitly:

- `src/worker.rs:355-368` — `run_chat(&["--provider", &provider, "--model",
  &model, …])` with `provider` read from `default_models.global_provider`.

And the console's own editors author exactly that config: the Model-screen pair
editor offers every enabled endpoint profile for the global default
(`src/ui/editors.rs:625-643` — `PairKind::GlobalDefault` falls into the
`_ => STATIC_PROVIDERS` arm and then appends `prof.virtual_provider()`), the
write goes through `--set-global-default endpoint:<id>/<model>`
(`src/writes.rs:471-501`), and the Python side persists it without any
provider-name validation (`manager.py:1323-1338` — `_split_provider_model`,
no registry check). This is not an exotic shape: it is the operator's primary
substrate pattern in this framework (`endpoint:ovh-provider`).

**Live repro, both layers.** CLI layer (scratch config):

```
$ abstractcore-chat --provider endpoint:ovh-provider --model gpt-oss-120b --prompt "Reply with exactly: PONG" --max-output-tokens 24
abstractcore-chat: error: argument --provider: invalid choice: 'endpoint:ovh-provider' (choose from openai, …, lmstudio)
EXIT=2
```

Full console (pty drive, scratch config with `global_provider:
"endpoint:scratch-lm"` pointing at the live LM Studio through a real profile —
a route that WORKS through the library): press `g` →

```
✗ generation test (default route): abstractcore exited with 2: abstr…
```

The verdict is Failed; Review and the journal record it as Failed. The wizard's
review step says "PROVE it: g runs a cheap generation over your default route"
(`src/ui/wizard.rs:83`) — the operator who just configured their (working)
endpoint route through this very console gets a red ✗ whose detail is an
argparse usage error. That is a wrong verdict, not a wrong message: the honesty
bar says Failed must mean the tested thing failed.

Compounding wrinkle: bare `abstractcore-chat` (no `--provider`) would not hit
argparse — but it would not test the global route either: `cli.py:2600-2624`
resolves `get_app_default('cli')`, i.e. `app_defaults.cli_provider`, whose
dataclass default is `huggingface` (`manager.py:138-146`) — the exact
silent-fallback lie the pre-check exists to prevent. So "just drop the flags"
is not the fix.

**Fix shape.** Two honest options, in order of effort: (a) `probe_generation`
checks the resolved provider against the chat CLI's accepted set (the same 10
names as `schema::STATIC_PROVIDERS`) and REFUSES with a teaching detail when it
is an `endpoint:<id>` ("the chat CLI cannot name endpoint profiles — test the
profile's models with t on Providers; generation testing for endpoint routes
needs the family+base-url expansion") — a refusal is honest where a FAILED is a
lie; (b) the real fix: expand `endpoint:<id>` to `--provider <family>
--base-url <profile base_url>` (both flags exist in the chat CLI,
`cli.py:2570-2571`), passing the profile's key via the environment
(`OPENAI_API_KEY` through `Command::env`), never argv — `run_chat`'s
"argv never carries secrets" contract (`src/cli.rs:176-179`) survives. Either
way, pin it: no test anywhere constructs a Generate probe against an
`endpoint:` default (see the audit).

### P1-2 — `tcp_probe` checks only the first resolved address: on macOS the count==0 disambiguation says "the server looks DOWN" while the server is UP — the exact branch the feature exists for, inverted (mechanism fully live-verified)

- `src/worker.rs:384-390` — `to_socket_addrs()` → `addrs.first()` → one
  `connect_timeout`. Every other address is ignored.

The four-link chain, each link verified live on this machine:

1. The ambiguous branch triggers only on `count == 0` (`worker.rs:226-230`),
   where the fold's message is decided by the TCP evidence
   (`probes.rs:200-210`).
2. Rust resolves `("localhost", 1234)` to `[[::1]:1234, 127.0.0.1:1234]` —
   `::1` FIRST (standalone `rustc` repro of `to_socket_addrs`, this machine's
   getaddrinfo; Python confirms the same order).
3. LM Studio serves `127.0.0.1:1234` and REFUSES `[::1]:1234` (live socket
   check: IPv4 CONNECTED, IPv6 `[Errno 61] Connection refused`).
4. `Reach::Refused` renders "TCP localhost:1234 → Connection refused — the
   server looks DOWN" (`probes.rs:158-160`).

So: LM Studio up, IPv4-serving, with an empty model list — which is precisely
the fresh-install machine the wizard's definition-of-done scenario targets
(LM Studio installed, no models downloaded yet; `/v1/models` answers
`{"data": []}` → the CLI's count:0 shape) — and the console's own TCP
evidence asserts the server is down. The comment at `probes.rs:200-202` says
the cause "is the part the operator must see"; here the console fabricates the
cause. The same inversion applies to any `http://localhost:*` profile base_url
whose server binds IPv4-only, and to ollama-up-with-zero-models
(ollama binds 127.0.0.1 by default). The verdict stays NotProven either way —
the failure is in the evidence text, which is exactly what M3 added over M2.

Note the fold's OTHER branch is fine: dead-on-both-families servers (today's
actual ollama) correctly read Refused → "looks DOWN" (verified in the pty
drive: `? test ollama` with the TCP cause leading).

**Fix shape.** Iterate ALL resolved addresses; `Connected` if ANY accepts;
`Refused` only when every one refuses (report the last error, or prefer the
IPv4 error text — "Connection refused" on `::1` alone is what produced the
lie). Extract the iteration into a pure helper over `Vec<SocketAddr>` so a
unit test can pin it without sockets, and add one socket test with a real
`std::net::TcpListener` on `127.0.0.1:0` (Connected) + a closed ephemeral port
(Refused) — `tcp_probe` currently has zero coverage.

---

## P2 findings

### P2-1 — `is_log_line` panics on legitimate multibyte model output: the generation probe dies as "internal error" and the evidence is lost (repro'd)

- `src/probes.rs:276-283` — after two byte probes, `t[8..]` byte-slices the
  line. `&str[8..]` panics when byte 8 is not a char boundary.

Any reply line ≥ 11 bytes with `:` at bytes 2 and 5 and a multibyte char
spanning byte 8 satisfies the guards and panics. Standalone repro (exact
function copied):

```
is_log_line("ab:cd:xé[test] whatever")
→ panicked: byte index 8 is not a char boundary; it is inside 'é' (bytes 7..9)
```

`fold_generation` feeds every stdout line of the model's reply through it
(`probes.rs:255-260`) — model output is arbitrary text (French/accented text,
CJK, emoji; the timestamp-like prefix `ab:cd:` shapes occur in tabular or
time-like replies). The worker's `catch_unwind` contains it (`worker.rs:88-108`:
busy cleared, `probe_busy` cleared, loud "internal error while handling
Probe(…)" notice — verified by code read; the panic handler is real), so
nothing wedges and nothing corrupts — but the probe records NO verdict, no
journal row, and the operator sees an internal error for a generation that
succeeded. A model whose replies routinely hit the shape can never be
generation-tested.

**Fix shape.** `t.get(8..)` instead of `t[8..]`
(`.get(8..).is_some_and(|r| r.trim_start().starts_with('['))`) — `get` returns
None on a non-boundary. Add the adversarial line to the
`generation_folds_reply_error_and_silence` test. (Also note: the containment
depends on the default `panic = "unwind"` profile — Cargo.toml declares no
override today; if a future wave sets `panic = "abort"`, this class becomes
app-fatal.)

### P2-2 — The single-flight guard is advisory: `probe_busy` is set only by the worker's posted closure, so probes queued behind a busy worker double the generation cost the guard exists to prevent

- `src/ui/mod.rs:150-160` — `send_probe` checks `probe_busy.get_untracked()`
  and sends.
- `src/worker.rs:213-216` — `probe_busy.set(true)` is posted only when the
  worker BEGINS handling the probe.

The worker is serial: while it is inside any earlier command — a models
discovery for a picker (up to 60s, `MODELS_TIMEOUT`), a write (up to 60s), a
prior probe still in the channel — a sent `Cmd::Probe` sits queued and
`probe_busy` stays false. Every additional `g`/`t` in that window passes the
guard and queues another real generation/discovery. The operator-shaped repro
is mundane: press `g`, see the "⟳ generation test …" notice, nothing visibly
happens (the worker is grinding a models listing the route editor kicked off),
press `g` again. Both generations run and both bill. The store comment
("a queued duplicate would silently double the cost", `store.rs:277-279`) and
the CHANGELOG ("Probes are single-flight") describe intent, not behavior.
`open_test_picker`'s door check (`providers.rs:207-214`) has the same window.

**Fix shape.** Set `probe_busy` synchronously in `send_probe` (the UI thread
owns the signal); the worker's completion post still clears it, and the
worker-side `set(true)` becomes a harmless reassert. One headless test: two
`g` presses across turns with no worker → exactly one `Cmd::Probe` drained.

### P2-3 — Route-test evidence never replaces across route edits: the label embeds the pair, so "latest per target" silently accumulates stale rows — contradicting the worker's own comment and the CHANGELOG

- `src/probes.rs:309` — `label: format!("test route {capability}
  ({provider}/{model})")`.
- `src/worker.rs:264-266` — the comment claims the opposite: "The label names
  only the route (evidence replaces per target) — the tested pair rides the
  detail" (and the detail DOES carry the pair, `worker.rs:266`).
- `src/store.rs:327-332` — `record_test` replaces by exact label equality.

Test route `input.text (lmstudio/m1)` → edit the route to `ollama/m2` → test
again: two rows now coexist ("test route input.text (lmstudio/m1)" and
"… (ollama/m2)"), the old one describing a configuration that no longer
exists. The Review block renders both, newest first, titled "latest per
target" (`review.rs:95`), and with `take(6)` (`review.rs:80`) each stale pair
evicts a live result from view. Re-tests of the SAME pair do replace — the
design only breaks exactly when the operator followed the Failed verdict's own
advice ("edit the route (e) and pick from the live list",
`probes.rs:237`) and re-verified, i.e. the mainline fix loop. The pair is also
rendered twice in the meantime (label + detail prefix).

**Fix shape.** `label: format!("test route {capability}")` — one line; the
pair already rides the detail. The DoD's needle matches on the detail
("model … is among") and the headless tests assert kinds/fields, so nothing
else moves. Add a pin: two `route_check` specs for one capability with
different pairs → one row in `tests` after both record.

### P2-4 — Routes-screen tests of `endpoint:<id>` routes get no reach evidence and claim "no known endpoint to reach-check" while the console knows the endpoint: same target, weaker evidence than the Providers picker

- `src/ui/routes.rs:125` — `route_check(&row.key, &provider, &model,
  row.base_url.as_deref())` passes only the ROUTE row's base_url (the
  route-level override, absent for typical endpoint-profile routes).
- `src/probes.rs:299-316` — `endpoint_for(provider, profile_base_url)`; an
  `endpoint:<id>` provider matches no local default, so `reach = None`.
- `src/ui/providers.rs:247-251` — the picker, for the SAME `endpoint:<id>`
  target, resolves the profile's base_url from the profiles store and passes
  it.

Consequence: with the profile's server dead (the CLI's ok:true/count:0 shape),
`t` on the Providers screen names the cause ("TCP host:port → … looks DOWN")
while `t` on the route row folds to "… ZERO models …; **no known endpoint to
reach-check**" (`probes.rs:206-209`) — a false claim: the endpoint is in
`store.profiles` under that exact id, one lookup away. The verdict is right;
the console under-uses its own knowledge and says so in the wrong words.

**Fix shape.** In `test_selected`, when the provider starts with `endpoint:`
and `row.base_url` is None, resolve the profile's base_url from
`store.profiles` (same untracked read the picker does) and pass it. Pin with a
headless test asserting `reach.is_some()` for an http-profile route (the
https→None case is already pinned in `test_verbs_send_probe_commands`).

---

## P3 findings

### P3-1 — Chat-binary failures wear the wrong name: "abstractcore exited with 2"

`CliErrorKind::Exit`'s headline is hardwired to "abstractcore exited with
{code}" (`src/cli.rs:45`), and `run_chat` reuses it (`cli.rs:187-192`).
Observed live in the P1-1 pty drive: the review row reads "abstractcore exited
with 2: abstractcore-chat: error: …" — the failing binary is
`abstractcore-chat`. The stderr tail usually self-corrects the reader, but the
headline misattributes. Thread a binary name into the error or headline.

### P3-2 — `resolve_chat_bin`'s PATH fallback contradicts its own same-install rationale, silently

The doc comment (`src/cli.rs:128-131`) argues the sibling lookup "keeps both
binaries from the SAME install (a PATH hit from a different venv would test a
different abstractcore)" — then falls back to exactly that PATH scan
(`cli.rs:136-143`) with no marker. A `$ABSTRACTCORE_BIN` venv without the chat
script plus a different venv's `abstractcore-chat` on PATH = discovery tests
install A, generation tests install B, and nothing on any screen says so
(CliInfo carries only the core bin). Either surface the chat bin's provenance
(Review/Overview line) or warn in the generation detail when the chat bin is
not the sibling.

### P3-3 — `parse_http_host_port` mis-parses userinfo and IPv6-literal URLs

`src/probes.rs:110-127`: `http://user:pass@host:1234` → host
`"user:pass@host"` — the password lands in `HostPort.host` and renders through
`Reach::describe` into details/journal/notices ("cannot resolve
user:pass@host:1234 — …"). Mitigating: `base_url` is already an unredacted
surface (profiles table renders it, `providers.rs:285`), so this widens an
existing exposure rather than minting a new class — but the probe detail and
journal are new carriers. `http://[::1]:1234` → host `"[::1]"` (brackets
kept) → always Unresolvable. Strip userinfo before the port split (or refuse
URLs containing `@`), and either strip brackets or accept the honest
Unresolvable for IPv6 literals with a test naming it.

### P3-4 — `fold_list_models` proves on `count` alone; a count/models mismatch renders "N models served · e.g. ?"

`src/probes.rs:188-198`: Proven requires only `count > 0`; the models array is
consulted just for the sample (fallback `"?"`). The real CLI computes
`count = len(models)` (`config/main.py:1651-1656`) so the divergence is
unreachable through today's abstractcore — but the fold is the honesty
chokepoint and "Proven with no nameable model" is a fold-level fabrication if
any future/other CLI drifts. Cheap hardening: derive the count from the parsed
models list (or require a non-empty models array for Proven). Non-string
`errors` entries fall back to raw JSON via `e.to_string()` (`probes.rs:178-181`)
— fine, but untested.

### P3-5 — The journal renders NotProven with the failure glyph

`src/worker.rs:290-294` folds NotProven into the journal's `Err` lane as
`"NOT PROVEN — {detail}"`, and the journal renders every `Err` as `✗ …`
(`review.rs:114-117`). The evidence block says `? NOT PROVEN` while the journal
row for the same probe says `✗ NOT PROVEN — …`. The words disambiguate, the
glyph misleads; a three-state journal outcome (or `? ` prefix inside the Err
string) would keep the vocabulary consistent.

### P3-6 — Keyless cloud providers read as the generic ZERO-models ambiguity with no key hint

`openai`'s `list_available_models` returns `[]` when no key resolves
(`providers/openai_provider.py:1125-1127` — silently, even under
`raise_on_error`, which is not threaded into the classmethod path), so
`test openai` with no key folds to NotProven "… ZERO models …; no known
endpoint to reach-check". True but unteaching: for keyed cloud providers the
dominant cause is "no API key configured" and the console knows key state
(api_keys table + profile rows). A one-clause hint in the no-reach branch for
cloud providers ("for cloud providers this is also the no-API-key answer")
would name the likely cause. (Live note: on this reviewer's shell the ambient
`OPENAI_API_KEY` made `test openai` return real models — the console inherits
the operator's env for subprocesses, which is correct behavior worth a line in
the docs: the test proves the key the PROVIDER resolves, config or env.)

### P3-7 — `fold_generation`'s failure scan can misjudge legitimate replies

`src/probes.rs:248-252`: any reply line containing `❌` anywhere or starting
with `Error:` folds the whole probe to Failed. A model that echoes those
shapes (asked about error handling; or an ❌ emoji in a reply) reads as a
failed generation. This is the documented cost of scraping the chat CLI's
mixed stdout (the CLI gives no machine verdict channel) — but it deserves a
comment/doc line, and the `Error:`-prefix check could be tightened to the
CLI's actual `❌ Error:` shape now that the ❌ scan exists.

### P3-8 — "N models served" overclaims for cache-listing providers

`test huggingface` / `test mlx` list LOCAL CACHES
(`huggingface_provider.py:6228-6240`, `mlx_provider.py:2448-2457`) — no server
exists. Proven is defensible (positive evidence the models are present and
loadable) but "served" is the wrong verb, and the picker's subtitle ("live
model discovery") reinforces it. "N models available" (or a per-family verb)
would keep the sentence true for all ten targets.

### P3-9 — The evidence block shows at most 6 rows with no overflow indicator

`src/ui/review.rs:80` — `results.iter().take(6)`, title "latest per target".
Seven distinct tests in a session and the oldest silently vanishes from the
one surface the wizard sends operators to. Combined with P2-3's stale-label
accumulation the window shrinks further. Render "… and N older" (or scroll).

### P3-10 — "default route" tests `default_models.global_*`, which bare `abstractcore-chat` never reads

`probe_generation` resolves `default_models.global_provider/model`
(`worker.rs:332-352`); the chat app's own default resolution is
`get_app_default('cli')` → `app_defaults.cli_*` (`utils/cli.py:2602`,
`manager.py:1720-1725`), which Python's save persists at its dataclass
defaults (huggingface) even when the global pair is set. So the console's `g`
tests the pair the LIBRARY's global fallback uses — the right thing to test,
and passing it explicitly is the only way to test it — but an operator reading
"✓ generation test (default route)" may conclude bare `abstractcore-chat`
now uses their route; it uses `app_defaults.cli` instead. One teaching line in
the detail or docs ("tests the global default; the chat app's own default is
app_defaults.cli") closes the gap.

### P3-11 — Long route pairs push the TCP cause out of the truncated detail

The route-check detail prefixes the pair before the cause
(`worker.rs:266` — `"{provider}/{model} — {d}"`), while `fold_list_models`
deliberately puts TCP evidence FIRST because "notices/rows truncate from the
right" (`probes.rs:200-202`). With `endpoint:` providers plus HF-style model
ids (60+ chars), the notice's `ellipsize(…, 90)` (`worker.rs:300`) leaves
little or none of the cause visible; the Review row's 110 is similar. The
label already carries the route; once P2-3 moves the pair out of the label the
detail prefix is load-bearing — consider trimming it (pair after cause, or
capability-only prefix).

### P3-12 — `eval_derived_expect`'s RouteEq misattributes a failed refresh

`src/worker.rs:528-530`: when the post-write `defaults --json` reload itself
failed, `RouteEq` reports "route {key} not found in the fresh view" — implying
the route vanished. `RouteCleared`'s sibling message names the real
possibility ("CLI reload failed?", `worker.rs:552-554`). Align RouteEq's
wording. (M2 lane, but this wave's refresh-gating rework touched the call
sites.)

### P3-13 — DoD/pty env hygiene: full `os.environ` inheritance, noted in one script but not the other

`definition_of_done.py:30-34` builds the child env as `dict(os.environ)` +
scratch `ABSTRACTCORE_CONFIG_FILE`. An exported `ABSTRACTCORE_BIN` silently
repoints the CONSOLE's CLI while `python_side()` uses `shutil.which` from
PATH — the two halves of the gate can test different installs.
`pty_smoke.py:213-216` at least prints the ambient-var note; the DoD prints
nothing. Strip or print the same note (the workspace's parameter-explicit
isolation lesson).

---

## Test-suite audit

**Faithful pins (verified against the live CLI shapes I re-probed):**

- The zero-count honesty rule is pinned three ways (`zero_count_success_is_not_proven`:
  bare, Refused, Connected) and the fixture matches the REAL dead-ollama JSON
  byte-shape (re-probed live: `ok:true, count:0, errors:[], models:[]`, exit 0).
- The generation liar class (`exit 0 + ❌`) is pinned at the fold
  (`generation_folds_reply_error_and_silence`) AND through `probe_generation`
  with a fake chat script (`generation_probe_folds_reply_and_exit0_error`),
  including the argv-carries-the-file's-default assert (`PONG from $2/$4` —
  a genuinely clever positive proof).
- The huggingface-invention pre-check is pinned for both the empty-config and
  missing-file cases with the teaching text asserted
  (`generation_probe_refuses_without_a_default_route`).
- https profiles are pinned to reach=None end-to-end through the picker
  (`test_verbs_send_probe_commands`), and the door-level busy refusal is
  pinned with a drained-to-empty negative assert.
- Route membership Proven/Failed/ambiguous are pinned at the fold; the
  routes-screen refusal for unset routes is pinned with zero-commands drained.
- The pty test phase and DoD assert glyph-prefixed, value-specific needles
  ("✓ test lmstudio", '✓ generation test', "verified: … = \"lmstudio\"",
  "model granite-4.1-3b is among") — none can pass on a NotProven or Failed.
  The one weak assert (`assert "proven" in text`, `pty_smoke.py:196`) is
  lowercase-exact so it cannot match "NOT PROVEN", and it sits behind two
  ✓-needle waits — weak but not vacuous.

**States with NO coverage anywhere (ranked):**

1. **The endpoint-default generation lane (P1-1).** No unit, headless, pty, or
   DoD lane ever builds a Generate probe over an `endpoint:` default — the
   fake chat scripts accept any argv, and both live scripts use `lmstudio`.
   The flagship config shape is untested end-to-end.
2. **`tcp_probe` — zero tests** (P1-2's home). Nothing pins Connected/Refused,
   let alone multi-address resolution. A `TcpListener` on `127.0.0.1:0` makes
   the Connected pin trivial.
3. **No automated NEGATIVE e2e lane.** The CHANGELOG's "negative lanes
   live-verified (dead ollama → NOT PROVEN …)" was a manual, one-time check.
   `test_phase` exercises only two green lanes; the DoD only green. A
   regression in the NotProven fold (or the TCP-gate wiring in
   `handle_probe`) would ride every green gate. The dead-ollama pick is
   scriptable exactly like my pty drive (one Up from the openai initial).
4. **`is_log_line`/`fold_generation` on multibyte input** — the P2-1 panic
   line belongs in the fold's unit test.
5. **Stale-label accumulation across route edits** (P2-3) — the review test
   re-records under one label only; two pairs for one capability would have
   caught the contradiction with the worker comment.
6. **`handle_probe`'s store effects** (probe_busy set/clear ordering, journal
   Err-lane for NotProven, notice format) — `probe_generation` is tested pure,
   but the wiring that records/journals/notices is untested (needs the same
   Store+wake extraction the M2 review recommended for `handle_write`; the
   panic-path clearing in `spawn` is likewise untested).
7. **The send_probe race** (P2-2) — untestable until the guard is synchronous;
   then one headless test pins it.
8. **Routes-lane reach resolution** (P2-4) — no test asserts what `reach` the
   routes screen passes for endpoint-profile rows.
9. `fold_list_models` payload-drift edges: count>0 with empty/non-string
   models; non-string errors; huge counts (P3-4).
10. Review overflow behavior beyond 6 rows (P3-9).

**Script-level notes:** `wait_fresh`'s stabilized re-read (clear → Ctrl+L →
read-until-quiet, `pty_smoke.py:96-122`) is sound and matches the engine's
global-action repaint contract; `nav`'s Ctrl+L trick is unchanged. The picker
navigation in `test_phase` (5×Up, 2×Down from the assumed openai initial,
`pty_smoke.py:181-187`) hardcodes index arithmetic that silently depends on
STATIC_PROVIDERS order, keys_sel=0, and a profile-less scratch config — it
fails LOUDLY if any of those drift (timeout on a ✓-needle), which is
acceptable, but a comment naming the three assumptions would save the next
editor a puzzle. The capture minter's M3 TestResults are hand-authored demo
values — verified faithful to the live shapes (the NotProven detail string
matches the real dead-ollama fold output word-for-word).

---

## What holds (verified only)

- **The three-state honesty core is real.** Zero-count success cannot fold to
  Proven (pinned + live-shape-verified); errors fold to Failed with the CLI's
  own line; Proven requires count>0 (live: lmstudio → "✓ … 48 models served ·
  e.g. …"). The dead-server CLI shape the whole design rests on reproduces
  exactly as the builder documented (`ok:true, count:0, errors:[]`, exit 0).
- **The generation pre-check kills a real lie.** `AppDefaults`' dataclass
  defaults are `huggingface/unsloth/Qwen3-4B-Instruct-2507-GGUF`
  (`manager.py:138-146`) — an empty config WOULD generate through an invented
  default; the fresh-file-read pre-check refuses first (pinned both for empty
  and missing configs, teaching text asserted).
- **`endpoint:<id>` targets are correct for discovery.** `config test-provider
  endpoint:<id>` resolves the profile and live-discovers through its family
  (live: 48 models through a scratch lmstudio-family profile); missing
  profiles fail loudly (`❌ Error: Unknown provider`, exit 1 → Failed). The
  picker's target strings and per-profile base_url threading are right.
- **The picker fabricates nothing.** Options are the 10 registry providers +
  real profiles only; the api_keys-row initial maps `openai_compatible` →
  `openai-compatible` correctly and FILTERS out `google` (reserved key, no
  registry provider — matches `registry.py`); no mapping → no initial, never a
  guess. Busy → refused at the door with the teaching notice (pinned).
- **Probe argv is injection-safe by construction.** `Command::args` (no
  shell); config-sourced values land as single argv elements; leading-dash
  values are refused by argparse loudly ("expected one argument" /
  "invalid choice"), never silently honored — and `run_chat`'s argv genuinely
  carries no secrets (provider/model/prompt only; keys resolve
  provider-side from config/env).
- **`probe_busy` cannot stick.** Every `handle_probe` arm (including
  no-CLI) runs the same completion post; the worker's `catch_unwind` handler
  clears busy ops AND `probe_busy` with a loud notice (`worker.rs:88-108`),
  and the crate builds with the default unwind profile. The serial-worker
  argument for clearing ALL busy entries holds (at most the panicked command's
  own ops are in flight).
- **Timeouts are honest.** `run_raw_at` enforces the deadline with kill +
  reader joins (pinned via `/bin/sleep`); a >120s generation folds to Failed
  naming the timeout and the invocation. `REACH_TIMEOUT` (1.5s) bounds the TCP
  check; `GENERATE_TIMEOUT`'s 120s is visible the whole way in the busy strip.
- **TCP scope is as chartered.** Only http endpoints the config names or the
  two documented local defaults; https and garbage parse to None (pinned;
  cloud providers get no reach). The `ureq` abstinence stands — the one socket
  is `std::net` with a connect timeout.
- **Secrets stay out of the probe surfaces** (with the P3-3 userinfo caveat,
  which requires the operator to have embedded credentials in a base_url —
  already an unredacted surface). Labels are target/route names; details are
  CLI stdout lines and TCP text; the chat argv is secret-free; `Cmd::Probe`'s
  Debug carries no key material.
- **Evidence surfaces render what happened.** Live pty: dead ollama → `? test
  ollama` with the TCP cause leading the truncated notice (the TCP-first
  design working as intended); the endpoint `g` → `✗ … exited with 2` on
  Review; journal rows and notices carry verdict glyph + label + detail;
  re-tests of the SAME label replace (pinned).
- **`reset_domains` exemptions are deliberate and compile-enforced.** The
  exhaustive destructure forces the tests/probe_busy decision per field with
  rationale in place (`store.rs:304-324`); dated evidence surviving a reload
  is a defensible reading of "evidence is about NOW" (rows carry timestamps).
- **Generation label collisions are the GOOD kind.** "generation test (default
  route)" is deliberately constant: after editing the default and re-testing,
  the latest evidence replaces the stale pair's (the detail names the tested
  pair). `Generate{Some,Some}` is never constructed today (grep-verified).
- **M1/M2 did not regress under this wave.** The profiles-refresh gating is
  sound: every profile-mutating spec (`save_profile`, `delete_profile`)
  declares a profile expect (grep-verified exhaustively), key writes touch
  only `api_keys` (mirror always re-reads), and routes still refresh on every
  successful write. All M2-review pins still pass (35 headless green
  including the refused-file split, chain-editor, wizard walks); the M1
  refusal banners and agreement lines are untouched by M3 code.
- **Gates:** build green, 50 unit + 35 headless (+1 ignored minter), clippy
  zero, engine pinned at 0.2.22.
