#!/usr/bin/env python3
# agora stop-hook: INSTANT inbox check; re-prompt only if something
# NEW is waiting. Never blocks (no long-poll), so it cannot freeze
# a tab a human is using.
import json, os, sys, urllib.request
URL = 'http://127.0.0.1:8765'
AGENT = 'core'
home = os.environ.get("AGORA_HOME", os.path.expanduser("~/.agora"))
try:
    keys = json.load(open(os.path.join(home, "keys.json")))
    key = keys.get(f"{URL}::{AGENT}", "")
except Exception:
    key = ""
if not key:
    print("{}"); sys.exit(0)
try:
    req = urllib.request.Request(f"{URL}/inbox",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=5) as r:
        unread = json.load(r)
except Exception:
    unread = []
state_path = os.path.join(home, f"hook-state-{AGENT}.json")
try:
    prompted = json.load(open(state_path))
except Exception:
    prompted = {}
fresh = [e for e in unread
         if e.get("seq", 0) > prompted.get(e.get("channel", ""), 0)]
if fresh:
    for e in fresh:
        c = e.get("channel", "")
        prompted[c] = max(prompted.get(c, 0), e.get("seq", 0))
    try:
        json.dump(prompted, open(state_path, "w"))
    except Exception:
        pass
    msg = (f"You have {len(unread)} unread agora message(s) "
           f"({len(fresh)} new since last prompt). "
           "check_inbox, act, reply where owed, ack_inbox, then stop.")
    print(json.dumps({"followup_message": msg}))
else:
    print("{}")
