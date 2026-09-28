"""One display rule for every Python surface that could launch a browser
(`abstractcore.utils.display`, the console crate's `display_from` twin): an SSH
session or a Linux/BSD session without DISPLAY/WAYLAND_DISPLAY never launches
one; the URL is printed instead."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from abstractcore.utils import display


@pytest.mark.parametrize(
    "platform, env, blocked",
    [
        ("darwin", {}, None),
        ("win32", {}, None),
        ("linux", {"DISPLAY": ":0"}, None),
        ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, None),
        ("freebsd14", {"DISPLAY": ":0"}, None),
        ("linux", {}, "DISPLAY and WAYLAND_DISPLAY are unset"),
        ("linux", {"DISPLAY": ""}, "DISPLAY and WAYLAND_DISPLAY are unset"),
        ("freebsd14", {}, "DISPLAY and WAYLAND_DISPLAY are unset"),
        ("darwin", {"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}, "SSH session"),
        ("win32", {"SSH_CLIENT": "1.2.3.4 5 22"}, "SSH session"),
        ("linux", {"DISPLAY": ":0", "SSH_TTY": "/dev/pts/1"}, "SSH session"),
    ],
)
def test_the_display_rule(platform, env, blocked):
    reason = display.no_display_reason(platform=platform, environ=env)
    if blocked is None:
        assert reason is None
    else:
        assert blocked in reason


def test_open_url_never_launches_without_a_display(monkeypatch):
    import webbrowser

    launched: list = []
    monkeypatch.setattr(webbrowser, "open", lambda url: launched.append(url) or True)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")
    assert display.open_url("https://example.invalid/") is False
    monkeypatch.delenv("SSH_CONNECTION")
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    assert display.open_url("https://example.invalid/") is True
    assert launched == ["https://example.invalid/"]


def test_engines_open_over_ssh_prints_the_link_and_launches_nothing():
    env = {**os.environ, "SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}
    proc = subprocess.run(
        [sys.executable, "-m", "abstractcore.config.main", "engines", "open", "lmstudio", "--json"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["opened"] is False and body["url"] == "https://lmstudio.ai/download"
    assert "SSH session" in body["not_opened"]
    proc = subprocess.run(
        [sys.executable, "-m", "abstractcore.config.main", "engines", "open", "lmstudio"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert proc.stdout.strip() == "https://lmstudio.ai/download"
    assert "no display here" in proc.stderr
