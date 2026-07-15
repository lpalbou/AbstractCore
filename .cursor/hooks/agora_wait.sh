#!/usr/bin/env python3
# agora-hook v3
# agora stop-hook: INSTANT inbox check (never long-polls). Prompts when
# something NEW landed; re-prompts standing unread on exponential backoff.
# The attempt ledger only THROTTLES prompts — it never means "handled":
# the server-side ack cursor (ack_inbox) is the only truth.
import json, os, sys, time, urllib.request
URL = 'http://127.0.0.1:8765'
AGENT = 'core'
NOOP = "{}"
BACKOFF_BASE, BACKOFF_CAP = 120, 1800

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
home = os.environ.get("AGORA_HOME", os.path.expanduser("~/.agora"))
def listener_dead():
    pidfile = os.path.join(home, f"listen-{AGENT}.pid")
    try:
        pid = int(open(pidfile).read().strip() or "0")
        os.kill(pid, 0)  # signal 0 = liveness probe, sends nothing
        return False
    except Exception:
        return True
try:
    keys = json.load(open(os.path.join(home, "keys.json")))
except Exception:
    keys = {}
key = keys.get(f"{URL}::{AGENT}", "") if isinstance(keys, dict) else ""
if not key or payload.get("stop_hook_active"):
    noop()
try:
    req = urllib.request.Request(f"{URL}/inbox",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        unread = json.load(r)
except Exception:
    unread = []
if not isinstance(unread, list):
    unread = []
if listener_dead() and not payload.get("stop_hook_active"):
    msg = ("Your agora BACKGROUND RECEPTION is not armed: this session "
           "is deaf to hub messages until you re-arm it. Do it NOW, "
           "exactly as your agora rule says: check_inbox, triage, then "
           "start ONE background shell running "
           "`while true; do agora listen --once --as core --important-only --max-wait 240; sleep 5; done` "
           "monitored on the ANCHORED pattern ^AGORA_WAKE (debounce "
           ">= 15000 ms), then keep your foreground on real work. "
           "Never pgrep/kill agora processes (other seats look "
           "identical by name)."
           + (f" Also: {len(unread)} unread message(s) await triage."
              if unread else ""))
    print(json.dumps({'followup_message': msg}))
    sys.exit(0)
if not unread:
    noop()  # empty inbox: nothing to say; ledger untouched

ledger_path = os.path.join(home, f"hook-attempts-{AGENT}.json")
try:
    ledger = json.load(open(ledger_path))
except Exception:
    ledger = {}  # missing/corrupt ledger: everything counts as fresh
if not isinstance(ledger, dict):
    ledger = {}

def entry(channel):
    e = ledger.get(channel)
    try:
        return {"seq": int(e.get("seq", 0) or 0),
                "attempts": min(int(e.get("attempts", 0) or 0), 64),
                "last": float(e.get("last", 0) or 0.0)}
    except Exception:
        return {"seq": 0, "attempts": 0, "last": 0.0}

now = time.time()
tops, fresh_count = {}, 0
for e in unread:
    if not isinstance(e, dict):
        continue
    c = str(e.get("channel", ""))
    try:
        s = int(e.get("seq", 0) or 0)
    except Exception:
        s = 0
    tops[c] = max(tops.get(c, 0), s)
    if s > entry(c)["seq"]:
        fresh_count += 1
due = False
for c, s in tops.items():
    ent = entry(c)
    if s > ent["seq"]:
        continue  # fresh channel: prompts regardless of backoff
    last = ent["last"]
    if not 0 <= last <= now + 60:
        last = 0.0  # NaN/negative/future timestamp: recover, not freeze
    if now - last >= backoff(ent["attempts"]):
        due = True
if not fresh_count and not due:
    noop()  # standing unread, every backoff window still open
# One prompt covers the whole inbox, so every unread channel's window
# restarts now (fresh channels reset the decay, stale ones escalate
# it); channels with nothing unread left are pruned — acked history
# needs no state. Never marks anything handled: ack_inbox is truth.
new_ledger = {}
for c, s in tops.items():
    ent = entry(c)
    if s > ent["seq"]:
        new_ledger[c] = {"seq": s, "attempts": 1, "last": now}
    else:
        new_ledger[c] = {"seq": ent["seq"],
                         "attempts": max(ent["attempts"], 0) + 1,
                         "last": now}
try:
    with open(ledger_path, "w") as f:
        json.dump(new_ledger, f)
except Exception:
    pass  # best-effort throttle: prompting matters more than the ledger
msg = (f"You have {len(unread)} unread agora message(s) across "
       f"{len(tops)} channel(s) ({fresh_count} new). check_inbox "
       "and settle what you OWE first: DO or claim work assigned to "
       "you; use answers to your own asks; reply where owed; then "
       "ack (ack = seen, not done). Verify your listener is armed; "
       "re-arm if dead.")
print(json.dumps({'followup_message': msg}))
