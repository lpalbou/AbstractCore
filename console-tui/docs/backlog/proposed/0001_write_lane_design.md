# 0001 — M2 write lane: design grounded in live-verified CLI behavior

- Created: 2026-07-25 · Status: proposed (M2 input) · Owner: builder

Every claim below was verified live on 2026-07-25 against abstractcore
2.13.38 via a scratch config (`ABSTRACTCORE_CONFIG_FILE=/tmp/...`,
deleted after probing — the real config was never touched).

## Verified facts that shape the lane

1. **The flags CLI (`--set-*`) exits 0 on REFUSED writes.**
   `--set-server-port 99999` prints `❌ Error: Invalid server port:
   99999` and exits 0; the file is unchanged. Exit codes are honest
   only for `config <subcommand>` verbs (`set-default bogus.route` →
   exit 1). → The write lane must (a) scan stdout for `❌ Error:` even
   on exit 0, and (b) never trust either signal alone: **verify by
   re-read is the only truth.**
2. **`--set-app-default` prints ✅ success for an UNKNOWN app and
   persists nothing** (`manager.set_app_default` returns False —
   ValueError swallowed — and `main.py:1816-1819` ignores the return).
   → Same defense; also a core-seat finding (report).
3. **Python saves drop unknown keys, live-confirmed**: an injected
   top-level section and an injected key inside `video` both vanished
   after one `--set-video-strategy` run. → Direct RMW writes must
   re-read fresh, mutate minimally, preserve everything else.
4. **Coupled writes confirmed**: `--set-global-default lmstudio/m` also
   writes route `input.text`; `--set-embeddings-model ollama/m` mirrors
   into route `embedding.text`; `--set-audio-strategy` sets
   `audio_strategy_explicit: true`. Blank values clear optional fields
   (`--set-stt-language ''` → null).
5. **`config set-default/clear-default/set-provider` accept NO
   `--json`** (argparse rejects; only the listing verbs have it). Their
   stdout is `✅/❌` human lines; exit codes are honest here.
6. **`ABSTRACTCORE_CONFIG_FILE` steers both CLI shapes** (flags and
   `config` subcommands) — the env-inheritance lane works for scratch
   isolation and for honoring the operator's own env.
7. **Every setter rewrites the whole file** (all 17 sections + meta
   flag appear after the first write) at mode 600.

## The lane (worker-owned, three-phase by construction)

```
Write = { label, verb, verify }
verb   = Cli { args: Vec<Arg> }          // Arg::Plain | Arg::Secret (redacting Debug)
       | Rmw { mutate: fn(&mut Value) -> Result<(), String> }
verify = fn(&ReVerifyView) -> Result<String, String>
         // runs over a FRESH load + (when routes are involved) a fresh
         // `config defaults --json`; returns the human proof line
```

Worker sequence per write: refuse unless the file state is `Ready` or
`Missing` (fact #4 of the charter: never write over Corrupt/Unreadable)
→ run verb → scan stdout for `❌ Error:` (Cli verb) → reload config
(+ derived views when touched) → run `verify` → journal
`{when, action, outcome, verified}` → post store updates + form_id
completion. RMW verb: fresh read → mutate → serde write to
`<file>.tmp-console-<pid>` → set 0600 → rename → re-read.
(Unique tmp name: the Python side uses a fixed `<file>.tmp`; colliding
with it would interleave writers — the sibling's SessionStore lesson.)

Drift guard: the mirror keeps the load-time mtime; a write whose
pre-write fresh read differs from the mirror's snapshot marks the form
with "the file changed since you loaded — review the mirror (r) before
saving" instead of silently last-writer-winning.

## Core-seat findings to carry in the M2 report

- `--set-*` refusals exit 0 (❌ line only) — machine callers cannot
  script the flags CLI safely.
- `--set-app-default` success-lies for unknown apps (returns-False
  ignored at `main.py:1816-1819`).
- (Pre-known from the inventory) the wizard's embeddings validation
  omits `vllm`.
- `config set-default`/`set-provider`/`clear-default` lack `--json`.
- **`config test-provider ollama --json` reports `ok:true, count:0`,
  exit 0 while Ollama is DOWN** (connection refused, live-verified
  2026-07-25): the raise_on_error lane swallows the unreachable case
  for ollama. Consequence for M3: the console's Test verb must render
  `ok:true + count:0` as "answered nothing — not proven", and pair it
  with a native base-URL health probe (inventory §7.3) before any
  green.
- Clearing: no CLI clear exists for the global default or
  chat/code models (`--set-global-default ''` is a silent no-op,
  exit 0); `--set-api-key P ''` stores an EMPTY STRING (not null) —
  semantically not-set to Python (falsy), so the console classifies
  `""` secrets as not-set.
