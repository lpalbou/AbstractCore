#!/usr/bin/env python3
"""PTY smoke (M1): drive the REAL binary with real keyboard bytes
against the REAL machine state — the one lane the headless suite never
exercises (the worker thread + abstractcore subprocesses + the real
config file). Read-only: this milestone writes nothing, so the smoke is
safe on an operator's live config.

Needles come from the LIVE machine (the config file + the CLI's own
JSON), never from one operator's remembered data.

Usage: python3 scripts/pty_smoke.py
Exit 0 = every gate passed.
"""

import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time

ANSI = re.compile(
    rb"\x1b\[[0-9;:?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[=>]|\x1b\([0-9A-B]"
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_abstractcore():
    explicit = os.environ.get("ABSTRACTCORE_BIN", "").strip()
    if explicit:
        return explicit
    for d in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(d, "abstractcore")
        if os.path.isfile(cand):
            return cand
    venv = os.path.expanduser("~/tmp/abstractframework/.venv/bin/abstractcore")
    if os.path.isfile(venv):
        return venv
    raise SystemExit("no abstractcore CLI found — the smoke needs the real machine")


def cli_json(bin_path, args):
    out = subprocess.run(
        [bin_path, *args], capture_output=True, text=True, timeout=60, check=True
    )
    return json.loads(out.stdout)


class Tui:
    def __init__(self, argv, env, cols=150, rows=42):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave, env=env,
            start_new_session=True,
        )
        os.close(slave)
        self.raw = bytearray()

    def pump(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            r, _, _ = select.select([self.master], [], [], 0.1)
            if self.master in r:
                try:
                    chunk = os.read(self.master, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                self.raw.extend(chunk)

    def text(self):
        return ANSI.sub(b"", bytes(self.raw)).decode("utf-8", "replace")

    def wait_for(self, needle, timeout, label):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if needle in self.text():
                print(f"  ✓ {label}")
                return
            self.pump(0.2)
        print(f"  ✗ TIMEOUT waiting for {needle!r} ({label})")
        print("---- last 2000 chars of stripped output ----")
        print(self.text()[-2000:])
        self.stop()
        sys.exit(1)

    def wait_fresh(self, needle, timeout, label):
        """For needles that arrive from ASYNC loads: the diff-based
        presenter emits only changed cells, so a needle can land
        fragmented across partial repaints. Loop: clear the tap,
        request a full redraw (Ctrl+L), read until the frame settles
        (a fixed 0.5s window truncates repaints under system load)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.raw.clear()
            os.write(self.master, b"\x0c")
            # Read until quiet for 0.3s (frame complete), max 2.5s.
            frame_deadline = time.time() + 2.5
            last_len = -1
            while time.time() < frame_deadline:
                self.pump(0.3)
                if len(self.raw) == last_len and last_len > 0:
                    break
                last_len = len(self.raw)
            if needle in self.text():
                print(f"  ✓ {label}")
                return
            time.sleep(0.4)
        print(f"  ✗ TIMEOUT waiting for {needle!r} ({label})")
        print("---- last full frame ----")
        print(self.text()[-2000:])
        self.stop()
        sys.exit(1)

    def send(self, data, settle=0.25):
        os.write(self.master, data)
        self.pump(settle)

    def nav(self, data, settle=0.5):
        """Send keys, then Ctrl+L: the diff-based presenter re-emits
        only CHANGED cells on a screen switch, fragmenting needles
        across the byte stream — a full-frame emission makes every
        needle contiguous (the abstractcode-tui pty lesson)."""
        self.send(data, settle)
        self.send(b"\x0c", settle=0.3)

    def stop(self):
        if self.proc.poll() is None:
            self.send(b"\x03", settle=0.3)  # Ctrl+C — the app's own quit
        if self.proc.poll() is None:
            time.sleep(0.5)
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
        return self.proc.wait(timeout=5)


def test_phase(core):
    """M3: the test verbs against REAL providers, on a scratch config
    the smoke owns. LM Studio serves this machine (live-verified in the
    M3 probes); the default route points at it, so:
    - t → provider picker → lmstudio → live discovery must PROVE.
    - g → one cheap generation over the configured default → PROVEN
      with a real reply (the definition-of-done verb).
    Evidence lands on Review (8) with honest verdicts."""
    import shutil
    import tempfile

    print("[7] M3 test verbs (real providers; scratch config)")
    scratch = tempfile.mkdtemp(prefix="acc-smoke-m3-")
    cfg = os.path.join(scratch, "abstractcore.json")
    with open(cfg, "w") as f:
        json.dump(
            {
                "default_models": {
                    "global_provider": "lmstudio",
                    "global_model": "gemma-3-1b-it",
                }
            },
            f,
        )
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["ABSTRACTCORE_CONFIG_FILE"] = cfg
    binary = os.path.join(REPO, "target", "debug", "abstractcore-console")
    tui = Tui([binary, "--browse"], env)
    try:
        tui.wait_for("AbstractCore Console", 10, "app painted (scratch)")
        tui.nav(b"3")  # Providers
        tui.wait_fresh("api_keys", 20, "providers screen")
        tui.send(b"t", settle=0.8)
        tui.wait_for("Test which provider?", 10, "test picker open")
        # anthropic → … → lmstudio is index 2; initial follows the key
        # row (openai, index 5) — Home-like: go up to the top then down.
        for _ in range(5):
            tui.send(b"\x1b[A", settle=0.1)
        for _ in range(2):
            tui.send(b"\x1b[B", settle=0.1)
        tui.send(b"\r", settle=1.0)
        # Live discovery against the real LM Studio: ~1-15s.
        tui.wait_fresh("✓ test lmstudio", 60, "provider test PROVEN (live discovery)")
        # g: the cheap generation over the configured default route.
        tui.send(b"g", settle=1.0)
        tui.wait_fresh("✓ generation test", 150, "generation PROVEN (real reply)")
        # NEGATIVE lane (M3 review audit #3: a NotProven-fold regression
        # must not ride green gates). ollama: environment-tolerant —
        # NOT PROVEN with the TCP cause when it is down (usual on this
        # machine), honest ✓ when someone started it.
        tui.send(b"t", settle=0.8)
        tui.wait_for("Test which provider?", 10, "test picker open (negative lane)")
        for _ in range(5):
            tui.send(b"\x1b[A", settle=0.1)
        for _ in range(4):  # anthropic → … → ollama (index 4)
            tui.send(b"\x1b[B", settle=0.1)
        tui.send(b"\r", settle=1.0)
        deadline = time.time() + 60
        verdict = None
        while time.time() < deadline:
            tui.raw.clear()
            tui.send(b"\x0c", settle=0.8)
            text = tui.text()
            if "? test ollama" in text:
                verdict = "notproven"
                assert "looks DOWN" in text or "TCP" in text, \
                    f"the TCP cause must lead the ambiguity:\n{text[-600:]}"
                break
            if "✓ test ollama" in text:
                verdict = "proven"  # someone runs ollama — honest pass
                break
            time.sleep(0.4)
        assert verdict, "ollama test never resolved"
        print(f"  ✓ NEGATIVE LANE: ollama folded honestly ({verdict})")
        tui.nav(b"8")  # Review: the evidence surface
        tui.wait_fresh("Test evidence", 10, "review evidence block")
        text = tui.text()
        assert "proven" in text, f"verdict words render:\n{text[-800:]}"
        print("  ✓ REVIEW: evidence lists the proven tests")
        code = tui.stop()
        print(f"  ✓ app exited with {code}")
        if code != 0:
            sys.exit(1)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    core = resolve_abstractcore()
    print(f"[pre] abstractcore: {core}")
    # The smoke inherits the ambient env — a leaked ABSTRACTCORE_CONFIG_*
    # var silently retargets both the app and the CLI (review nit: a
    # stale probe var once made gate [1] fail as "no config file yet"
    # with nothing naming the cause).
    for var in ("ABSTRACTCORE_CONFIG_FILE", "ABSTRACTCORE_CONFIG_DIR", "ABSTRACTCORE_BIN"):
        val = os.environ.get(var)
        if val:
            print(f"[pre] NOTE: {var}={val!r} is set — the smoke runs against it")
    defaults = cli_json(core, ["config", "defaults", "--json"])
    config_file = defaults["config_file"]
    print(f"[pre] config file per the CLI: {config_file}")
    configured = [r["key"] for r in defaults.get("routes", []) if r.get("configured")]
    route_needle = configured[0] if configured else "input.text"
    profiles = cli_json(core, ["config", "providers", "--json"])
    profile_needle = next(
        (p["id"] for p in profiles.get("profiles", []) if p.get("id")), None
    )

    binary = os.path.join(REPO, "target", "debug", "abstractcore-console")
    if not os.path.exists(binary):
        raise SystemExit(f"build first: cargo build ({binary} missing)")

    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    tui = Tui([binary], env)

    print("[1] boot: file mirror + derived views load with zero keystrokes")
    tui.wait_for("AbstractCore Console", 10, "app painted")
    tui.wait_for(os.path.basename(config_file), 10, "real config path in the header")
    # The state label lands via an async diff — fresh-frame wait, and
    # the needle includes the dot so "not loaded"/"reloading" can never
    # satisfy it.
    tui.wait_fresh("● loaded", 10, "file state: loaded")
    tui.wait_fresh(
        "reads the same file", 30, "CLI agreement (defaults --json ran + echo matched)"
    )
    tui.wait_fresh("default_models", 5, "sections table rendered")

    print("[2] digit 4 → Routes (live derived view)")
    tui.send(b"4", settle=0.5)
    tui.wait_fresh(route_needle, 30, f"live route row rendered ({route_needle})")
    tui.wait_fresh("configured", 5, "banner counts")

    print("[3] digit 3 → Providers")
    tui.send(b"3", settle=0.5)
    if profile_needle:
        tui.wait_fresh(profile_needle, 30, f"live profile rendered ({profile_needle})")
    tui.wait_fresh("API keys", 15, "api_keys block present")

    print("[4] r → reload acknowledgment + fresh state")
    tui.raw.clear()
    tui.send(b"r", settle=0.6)
    tui.wait_for("reloading config", 5, "reload acknowledged visibly")
    tui.wait_fresh("API keys", 20, "screen repainted after reload")

    print("[5] digit 8 → Review (identity proof)")
    tui.send(b"8", settle=0.5)
    tui.wait_fresh("same file", 20, "review agreement lines")

    code = tui.stop()
    print(f"[exit] app exited with {code}")
    if code != 0:
        sys.exit(1)

    write_phase(core)
    test_phase(core)
    print("PTY SMOKE: ALL GATES PASSED")


def write_phase(core):
    """M2: a REAL verified write, against a SCRATCH config the smoke
    owns (env-pointed; the operator's real file is never touched).
    Keyboard-only: wizard boot on the fresh file → browse → Media →
    select video.max_frames → editor → type → Save → the file changes
    on disk and the journal carries the verify line."""
    import shutil
    import tempfile

    print("[6] M2 write phase (scratch config; real abstractcore setter)")
    scratch = tempfile.mkdtemp(prefix="acc-smoke-")
    cfg = os.path.join(scratch, "abstractcore.json")
    env = dict(os.environ)
    env["TERM"] = "xterm-256color"
    env["ABSTRACTCORE_CONFIG_FILE"] = cfg
    binary = os.path.join(REPO, "target", "debug", "abstractcore-console")
    tui = Tui([binary], env)
    try:
        tui.wait_for("AbstractCore Console", 10, "app painted (scratch)")
        # Fresh machine → adaptive default = wizard mode.
        tui.wait_fresh("Step 1/10", 15, "wizard mode on a missing config")
        tui.send(b"f", settle=0.5)  # finish → browse for the targeted edit
        tui.send(b"5", settle=0.5)  # Media
        tui.wait_fresh("max_frames", 20, "media field table")
        # Rows: vision 5 + audio 6 = 11 → video.strategy; 12 = max_frames.
        for _ in range(12):
            tui.send(b"\x1b[B", settle=0.08)
        # The modal's own open PAINTS it — accumulate, don't clear (the
        # fresh-frame trick would wipe the just-painted needle).
        tui.send(b"e", settle=0.8)
        tui.wait_for("Edit video.max_frames", 20, "editor open")
        tui.send(b"7", settle=0.3)
        # Save → CLI setter + a defaults --json refresh; each abstractcore
        # invocation is a cold Python import (4-15s under load).
        tui.send(b"\r", settle=1.0)
        tui.wait_fresh("verified: video.max_frames = 7", 90, "write + verify journaled")
        deadline = time.time() + 10
        while time.time() < deadline:
            if os.path.exists(cfg):
                with open(cfg) as f:
                    v = json.load(f)
                if v.get("video", {}).get("max_frames") == 7:
                    print("  ✓ FILE STATE: video.max_frames = 7 (created + written by the setter)")
                    break
            time.sleep(0.3)
        else:
            print("  ✗ the scratch file never showed the write")
            tui.stop()
            sys.exit(1)
        # Cross-check with the Python side reading the same scratch.
        out = subprocess.run(
            [core, "config", "defaults", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=True,
        )
        echo = json.loads(out.stdout)["config_file"]
        assert echo == cfg, f"CLI reads {echo}, expected {cfg}"
        print("  ✓ PYTHON SIDE: defaults --json reads the scratch file cleanly")
        code = tui.stop()
        print(f"  ✓ app exited with {code}")
        if code != 0:
            sys.exit(1)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
