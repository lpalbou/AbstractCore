"""A light install sends no telemetry on a user's behalf.

`import unstructured` runs Scarf telemetry (an HTTP GET to
packages.unstructured.io, plus an `nvidia-smi` probe). AutoMediaHandler used
to import it just to learn whether Office documents are supported, so every
handler -- even one built for a .txt file -- phoned home. Each case runs in a
fresh interpreter (another test may already have imported unstructured) with
every socket connection and name lookup refused and recorded, and every
subprocess recorded.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_GUARD = textwrap.dedent(
    """
    import json, socket, subprocess, sys
    attempts = []

    def _refuse_connect(self, address, *a, **k):
        attempts.append(["connect", repr(address)])
        raise OSError("network refused by test guard")

    def _refuse_create_connection(address, *a, **k):
        attempts.append(["create_connection", repr(address)])
        raise OSError("network refused by test guard")

    def _refuse_getaddrinfo(host, *a, **k):
        attempts.append(["getaddrinfo", repr(host)])
        raise OSError("name resolution refused by test guard")

    _real_popen_init = subprocess.Popen.__init__

    def _record_popen(self, args, *a, **k):
        # Recorded, not refused: the host probe's own `sysctl` read is fine.
        attempts.append(["subprocess", repr(args)])
        return _real_popen_init(self, args, *a, **k)

    socket.socket.connect = _refuse_connect
    socket.socket.connect_ex = _refuse_connect
    socket.create_connection = _refuse_create_connection
    socket.getaddrinfo = _refuse_getaddrinfo
    subprocess.Popen.__init__ = _record_popen
    """
)


def _run(body: str, tmp_path: Path) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.endswith("_KEY") or k.endswith("_TOKEN") or k in ("SCARF_NO_ANALYTICS", "DO_NOT_TRACK"))
    }
    env["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(exist_ok=True)
    env["ABSTRACTCORE_TEST_HERMETIC_MODEL_DISCOVERY"] = "1"
    script = _GUARD + textwrap.dedent(body) + "\nprint('RESULT=' + json.dumps(result))\n"
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT=")][-1]
    return json.loads(line[len("RESULT="):])


def test_auto_media_handler_on_a_text_file_makes_no_network_call_and_never_imports_unstructured(tmp_path: Path) -> None:
    sample = tmp_path / "note.txt"
    sample.write_text("hello from a plain text file\n", encoding="utf-8")
    result = _run(
        f"""
        from abstractcore.media.auto_handler import AutoMediaHandler
        handler = AutoMediaHandler()
        out = handler.process_file({str(sample)!r})
        result = {{
            "attempts": attempts,
            "unstructured_imported": "unstructured" in sys.modules,
            "success": bool(getattr(out, "success", False)),
            "office": handler._available_processors.get("office"),
        }}
        """,
        tmp_path,
    )
    assert result["success"] is True
    network = [a for a in result["attempts"] if a[0] != "subprocess"]
    assert network == [], network
    assert not [a for a in result["attempts"] if "nvidia-smi" in a[1]], result["attempts"]
    assert result["unstructured_imported"] is False
    # Availability is still reported, from the spec alone.
    assert result["office"] is (importlib.util.find_spec("unstructured") is not None)


@pytest.mark.skipif(importlib.util.find_spec("unstructured") is None, reason="unstructured is not installed")
def test_office_processor_imports_unstructured_with_telemetry_opted_out(tmp_path: Path) -> None:
    result = _run(
        """
        import os
        from abstractcore.media.processors.office_processor import OfficeProcessor
        proc = OfficeProcessor()
        result = {
            "attempts": attempts,
            "available": proc._unstructured_available,
            "scarf": os.environ.get("SCARF_NO_ANALYTICS"),
            "dnt": os.environ.get("DO_NOT_TRACK"),
        }
        """,
        tmp_path,
    )
    assert result["available"] is True
    assert result["scarf"] == "true" and result["dnt"] == "true"
    # No telemetry call. (unstructured's docx partitioner also fetches NLTK
    # data from raw.githubusercontent.com on first use when it is missing --
    # a data download, not telemetry -- and its `nvidia-smi` probe runs before
    # the opt-out check; neither reaches packages.unstructured.io.)
    telemetry = [a for a in result["attempts"] if "unstructured.io" in a[1]]
    assert telemetry == [], telemetry
