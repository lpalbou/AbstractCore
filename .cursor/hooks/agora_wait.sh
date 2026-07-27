#!/usr/bin/env python3
# agora-hook v4
# Turn-end backstop for agora reception. Obligation-gated (owed debts +
# open/blocked unread; fyi never prompts), one global prompt floor across
# ALL branches, harness payload guards (completed turns only, loop_count
# cap), two-observation dead-listener nag. The ledger only THROTTLES —
# the server-side ack cursor (ack_inbox) is the only truth.
import hashlib, json, os, sys, time, urllib.request
URL = 'core'
AGENT = 'http://127.0.0.1:8765'
NOOP = "{}"
CLIENT = '0.12.46'
FLOOR = 600
BACKOFF_BASE, BACKOFF_CAP = 600, 3600

def noop():
    if NOOP:
        print(NOOP)
    sys.exit(0)

def backoff(attempts):
    # clamp the exponent: a corrupt ledger must not conjure 2**huge
    return min(BACKOFF_BASE * 2 ** (min(max(attempts, 1), 8) - 1),
               BACKOFF_CAP)

try:
    payload = json.load(sys.stdin)
except Exception:
    payload = {}
if not isinstance(payload, dict):
    payload = {}
# Claude re-entry guard: a turn the hook itself started must not chain.
if payload.get("stop_hook_active"):
    noop()
# Cursor guards, enforced only when the field exists (Claude/Codex
# payloads lack both). An aborted/errored turn must not breed a
# follow-up: the human just cancelled, or the provider just failed —
# either way another full-context turn is the wrong reflex. DEFERRED,
# not absolute (2026-07-23 fleet blackout, RC-1): sessions that become
# chains of harness-generated turns present these payloads at EVERY
# turn-end, so a hard noop here silenced the backstop for days while
# operator orders rotted. The guards now suppress chatter only — an
# ESCALATED debt (SLA-breached, hub-raised) prompts through them, with
# the FLOOR + exponential backoff below still bounding the cadence.
harness_guarded = False
status = payload.get("status")
if status is not None and str(status) != "completed":
    harness_guarded = True
try:
    lc = payload.get("loop_count")
    if lc is not None and int(lc) >= 2:
        harness_guarded = True  # chain cap; hooks.json loop_limit backstops
except Exception:
    pass
home = os.environ.get("AGORA_HOME", os.path.expanduser("~/.agora"))
def listener_dead():
    return False
try:
    keys = json.load(open(os.path.join(home, "keys.json")))
except Exception:
    keys = {}
key = keys.get(f"{URL}::{AGENT}", "") if isinstance(keys, dict) else ""
if not key:
    noop()

ledger_path = os.path.join(home, f"hook-attempts-{AGENT}.json")
def _fresh_ledger():
    return {"v": 4, "last_prompt": 0.0, "sig": "", "attempts": 0,
            "dead_streak": 0, "last_run": 0.0}
try:
    led = json.load(open(ledger_path))
    if not (isinstance(led, dict) and led.get("v") == 4):
        led = _fresh_ledger()  # v3 per-channel ledgers restart clean
except Exception:
    led = _fresh_ledger()
def _save():
    try:
        with open(ledger_path, "w") as f:
            json.dump(led, f)
    except Exception:
        pass  # best-effort throttle: prompting matters more than state

now = time.time()
# Liveness heartbeat, written BEFORE any network call: the 2026-07-23
# fleet forensics could not tell "hook never fires" from "hook dies
# mid-run" because the only ledger write sat after the HTTP fetches.
# last_run answers that at a glance next time.
led["last_run"] = now
_save()
last = led.get("last_prompt", 0.0)
try:
    last = float(last)
except Exception:
    last = 0.0
if not 0 <= last <= now + 60:
    last = 0.0  # NaN/negative/future timestamp: recover, not freeze
floor_open = (now - last) >= FLOOR

# Two-observation dead-listener rule: the pidfile is absent ~5s of
# every ~246s listen cycle, so a single dead read is noise.
if listener_dead():
    led["dead_streak"] = min(int(led.get("dead_streak", 0) or 0) + 1, 64)
else:
    led["dead_streak"] = 0
arm_due = led["dead_streak"] >= 2

def _get(path):
    req = urllib.request.Request(
        f"{URL}{path}", headers={"Authorization": f"Bearer {key}",
                                 "X-Agora-Client": CLIENT})
    # 4s, not 5: two calls must fit any harness kill budget with
    # room for interpreter start (the Jul-23 hook-death class).
    with urllib.request.urlopen(req, timeout=4) as r:
        return json.load(r)

try:
    owed = _get("/owed")
except Exception:
    owed = {}
if not isinstance(owed, dict):
    owed = {}
to_answer = [m for m in owed.get("to_answer", []) if isinstance(m, dict)]
to_consume = [m for m in owed.get("to_consume", []) if isinstance(m, dict)]
# The deferred harness guard: suppress this turn-end unless the seat
# owes an ESCALATED debt — the one thing that must ring through a
# follow-up-only session (RC-1). Checked before the /inbox fetch so
# guarded turns cost one GET, not two.
if harness_guarded and not any(m.get("escalated") for m in to_answer):
    _save()
    noop()
try:
    unread = _get("/inbox")
except Exception:
    unread = []
if not isinstance(unread, list):
    unread = []
# Obligation-shaped unread only: open/blocked status, or the hub's
# own debt markers (an answer to YOUR ask, critical, escalated). Bare
# to-you fyi — including the hub's synthetic notices, which ride
# fyi+to_me — waits for an organic turn; fyi never costs one.
# Envelope fields, not the listener's notify-file grammar (the 2026-07-23
# audit found this filter reading `from`/`flags` — keys the /inbox wire
# has NEVER carried, so critical/escalated unread outside /owed never
# reached the backstop): sender is `sender`, and critical/escalated/
# reply_to_me are BOOLEANS on the envelope.
oblig_unread = []
for e in unread:
    if not isinstance(e, dict) or str(e.get("sender", "")) == AGENT:
        continue
    is_oblig = (str(e.get("status", "")) in ("open", "blocked")
                or e.get("critical") or e.get("escalated")
                or e.get("reply_to_me"))
    # Narrowed rule (0135): an ADDRESSED open that does not name
    # this seat is the named seats' debt, not a reason to prompt
    # here (62% of commons wakes measured as addressed opens
    # hitting everyone).
    if (is_oblig and str(e.get("status", "")) in ("open", "blocked")
            and e.get("addressed") and not e.get("to_me")
            and not e.get("critical") and not e.get("escalated")
            and not e.get("reply_to_me")):
        is_oblig = False
    if is_oblig:
        oblig_unread.append(e)

def _mid(m):
    return str(m.get("id") or f"{m.get('channel')}:{m.get('seq')}")
oblig_ids = sorted({_mid(m) for m in to_answer} | {_mid(m) for m in to_consume}
                   | {_mid(e) for e in oblig_unread})
have_debt = bool(oblig_ids)
sig = hashlib.sha256("\n".join(oblig_ids).encode()).hexdigest()[:16] if have_debt else ""
changed = have_debt and sig != str(led.get("sig", ""))
try:
    attempts = min(int(led.get("attempts", 0) or 0), 64)
except Exception:
    attempts = 0
due = (have_debt and not changed
       and (now - last) >= backoff(max(attempts, 1)))
if not have_debt:
    led["sig"], led["attempts"] = "", 0  # cleared debt ends the episode
if not floor_open or not (arm_due or changed or due):
    _save()  # persist dead_streak/sig evolution even when silent
    noop()

parts = []
if arm_due:
    parts.append(
        "Your agora BACKGROUND RECEPTION is not armed: this session "
        "is deaf to hub messages until you re-arm it. Do it NOW, "
        "exactly as your agora rule says: check_inbox, triage, then "
        "start ONE background shell running "
        "`while true; do agora listen --once --as http://127.0.0.1:8765 --important-only --max-wait 240; sleep 5; done` "
        "monitored on the ANCHORED pattern ^AGORA_WAKE (debounce "
        ">= 15000 ms), then keep your foreground on real work. "
        "FIRST check the listener is not already running (read your "
        "background shells); never arm a second loop, and never "
        "pgrep/kill agora processes (other seats look identical by "
        "name).")
if have_debt:
    parts.append(
        f"You OWE work on {len(oblig_ids)} obligation(s): "
        f"{len(to_answer)} unanswered ask(s) naming you, "
        f"{len(to_consume)} answer(s) to your own asks awaiting use, "
        f"{len(oblig_unread)} open/blocked unread. check_inbox and "
        "settle what you OWE first: DO or claim work assigned to "
        "you; use answers to your own asks; reply where owed; then "
        "ack (ack = seen, not done).")
msg = " ".join(parts)
led["last_prompt"] = now
if have_debt:
    led["sig"] = sig
    led["attempts"] = 1 if changed else attempts + 1
else:
    led["sig"], led["attempts"] = "", 0
_save()
print(json.dumps({"decision": "block", "reason": msg}))
