#!/usr/bin/env python3
"""The chartered definition-of-done walk, end to end, keyboard-only:

  fresh machine (no config) → wizard boots → default model set to a
  LOCAL provider (lmstudio) through the pair editor → generation test
  passes → wizard finishes → the PYTHON side (config defaults --json)
  agrees with what the console showed → browse mode edits one
  capability route → re-verify (route membership test + Python re-read).

Everything runs against a SCRATCH config (env-pointed); the operator's
real file is never touched. Requires LM Studio serving on :1234 (the
machine's live local provider) and the framework venv's abstractcore.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pty_smoke import REPO, Tui  # noqa: E402

MODEL = "gemma-3-1b-it"
ROUTE_MODEL = "granite-4.1-3b"


def cli_env(cfg):
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["ABSTRACTCORE_CONFIG_FILE"] = cfg
    # Ambient-var honesty (M3 review P3-13): an exported
    # ABSTRACTCORE_BIN repoints the CONSOLE's CLI while python_side()
    # resolves from PATH — the two halves of this gate would test
    # different installs. Name it loudly instead of guessing.
    if os.environ.get("ABSTRACTCORE_BIN"):
        print(f"[pre] WARNING: ABSTRACTCORE_BIN={os.environ['ABSTRACTCORE_BIN']} is exported — "
              "the console will use it while the Python-side checks use PATH")
    return env


def python_side(core, env, *args):
    out = subprocess.run(
        [core, *args], capture_output=True, text=True, timeout=90, env=env, check=True
    )
    return json.loads(out.stdout)


def route_row(defaults, key):
    """`config defaults --json` carries routes as a LIST of rows."""
    return next(r for r in defaults["routes"] if r.get("key") == key)


def main():
    core = shutil.which("abstractcore")
    if not core:
        print("abstractcore not on PATH — activate the framework venv first")
        sys.exit(2)
    binary = os.path.join(REPO, "target", "debug", "abstractcore-console")
    scratch = tempfile.mkdtemp(prefix="acc-dod-")
    cfg = os.path.join(scratch, "abstractcore.json")
    env = cli_env(cfg)
    print(f"[pre] scratch config: {cfg} (missing — fresh machine)")

    try:
        # ---- A: fresh machine boots into the wizard -------------------
        print("[A] launch with no config → wizard")
        tui = Tui([binary], env)
        tui.wait_for("AbstractCore Console", 10, "app painted")
        tui.wait_fresh("Step 1/10", 15, "adaptive default = wizard on a fresh machine")

        # ---- B: wizard step 2 — set the default model (pair editor) ---
        print("[B] wizard step 2: default model → lmstudio (pair editor)")
        tui.send(b"\x0e", settle=0.6)  # Ctrl+N → the default-model phase
        tui.wait_fresh("Step 2/10", 10, "step 2 (Model screen, filtered)")
        tui.send(b"e", settle=0.8)  # row 0 = global_provider → pair editor
        tui.wait_for("Default model (global + route input.text)", 15, "pair editor open")
        tui.send(b"\r", settle=0.4)  # open the provider popup
        for _ in range(3):  # placeholder → anthropic → huggingface → lmstudio
            tui.send(b"\x1b[B", settle=0.15)
        tui.send(b"\r", settle=0.4)  # commit lmstudio → discovery kicks
        # The model row becomes a class-filtered Combobox once live
        # discovery lands (the status line names the count).
        tui.wait_for("generative models", 60, "model picker populated from discovery")
        tui.send(b"\t", settle=0.3)  # → the model Combobox
        tui.send(b"\r", settle=0.5)  # open its popup
        tui.send(MODEL.encode(), settle=0.5)  # type-to-filter
        tui.send(b"\r", settle=0.4)  # commit the exact match
        tui.send(b"\t", settle=0.3)  # → Save
        tui.send(b"\r", settle=1.0)
        # The CLI setter also writes route input.text (coupled write).
        # (String proofs render JSON-quoted: `= "lmstudio"`.)
        tui.wait_fresh('verified: default_models.global_provider = "lmstudio"', 90,
                       "pair written + verified")

        # ---- C: the generation test over the JUST-SET default ---------
        print("[C] g → cheap generation over the configured default route")
        tui.send(b"g", settle=1.0)
        tui.wait_fresh("✓ generation test (default route)", 150, "generation PROVEN")

        # ---- D: finish; Review agreement; quit; Python-side check -----
        print("[D] finish wizard → Review agreement → Python agrees")
        tui.send(b"f", settle=0.8)  # finish → browse (stays on the screen)
        tui.nav(b"8")
        tui.wait_fresh("Test evidence", 15, "review evidence block")
        tui.wait_fresh("✓ same file", 30, "identity agreement lines")
        tui.send(b"q", settle=0.5)
        code = tui.stop()
        print(f"  ✓ app exited with {code}")
        if code != 0:
            sys.exit(1)

        defaults = python_side(core, env, "config", "defaults", "--json")
        assert defaults["config_file"] == cfg, defaults["config_file"]
        route = route_row(defaults, "input.text")
        assert route["provider"] == "lmstudio" and route["model"] == MODEL, route
        print(f"  ✓ PYTHON SIDE: input.text = lmstudio/{MODEL} (coupled write held)")
        raw = json.load(open(cfg))
        gp = raw["default_models"]["global_provider"]
        gm = raw["default_models"]["global_model"]
        assert (gp, gm) == ("lmstudio", MODEL), (gp, gm)
        print("  ✓ FILE: default_models carries the pair")

        # ---- E: browse mode edits ONE capability route -----------------
        print("[E] browse: edit route input.text → a different live model")
        tui = Tui([binary, "--browse"], env)
        tui.wait_for("AbstractCore Console", 10, "app painted (browse)")
        tui.nav(b"4")
        tui.wait_fresh("input.text", 20, "routes screen")
        # input.text is the first row (route table order); its detail
        # line names it — assert before editing.
        tui.send(b"e", settle=0.8)
        tui.wait_for("Route — Text Input (input.text)", 15, "route editor open")
        # The prefilled provider kicks discovery at OPEN; the model row
        # becomes the filtered Combobox once it lands.
        tui.wait_for("generative models", 60, "route model picker populated")
        tui.send(b"\t", settle=0.3)  # provider → the model Combobox
        tui.send(b"\r", settle=0.5)  # open its popup
        tui.send(b"granite", settle=0.5)  # filter to the one match
        tui.send(b"\r", settle=0.4)  # commit granite-4.1-3b
        # base URL → reasoning → options → Save.
        for _ in range(4):
            tui.send(b"\t", settle=0.2)
        tui.send(b"\r", settle=1.0)
        tui.wait_fresh(f"✓ set route input.text = lmstudio/{ROUTE_MODEL}", 90,
                       "route write verified")

        # ---- F: re-verify — membership test + Python re-read ----------
        print("[F] re-verify: t (route membership) + Python re-read")
        tui.send(b"t", settle=1.0)
        tui.wait_fresh(f"model {ROUTE_MODEL} is among", 60,
                       "route model PROVEN against the live list")
        tui.send(b"q", settle=0.5)
        code = tui.stop()
        print(f"  ✓ app exited with {code}")
        if code != 0:
            sys.exit(1)
        defaults = python_side(core, env, "config", "defaults", "--json")
        route = route_row(defaults, "input.text")
        assert route["provider"] == "lmstudio" and route["model"] == ROUTE_MODEL, route
        print(f"  ✓ PYTHON SIDE: input.text = lmstudio/{ROUTE_MODEL}")
        print("DEFINITION OF DONE: ALL GATES PASSED")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
