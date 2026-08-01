"""The model materializer: one probe/download abstraction, honest about what it knows.

These tests pin the four rules `model_materializer`'s docstring states, because
each of them was a way an earlier "just check if the model is there" helper
lied to an operator:

  1. probing never downloads,
  2. `unknown` is a real answer and is never dressed up as `installed`,
  3. a served model id is not a download reference (`@4bit`),
  4. a failed download reports the provider tool's own words.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from abstractcore.config import model_materializer as mm
from abstractcore.config.capability_defaults import (
    RECOMMENDED_CAPABILITY_DEFAULT_ROUTES,
    RECOMMENDED_MODEL_DOWNLOADS,
)


# ---------------------------------------------------------------------------
# Rule 3: served ids vs download refs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("qwen/qwen3.5-9b@4bit", ("qwen/qwen3.5-9b", "4bit")),
        ("qwen/qwen3.5-9b@8bit", ("qwen/qwen3.5-9b", "8bit")),
        ("llama-3.1-8b@q4_k_m", ("llama-3.1-8b", "q4_k_m")),
        ("model@bf16", ("model", "bf16")),
        # No quant suffix: the whole string is the reference. `-8bit` inside the
        # repo NAME is not an `@quant` and must survive untouched, or the image
        # default would be fetched from a repo that does not exist.
        ("AbstractFramework/flux.2-klein-4b-8bit", ("AbstractFramework/flux.2-klein-4b-8bit", None)),
        ("supertonic-3", ("supertonic-3", None)),
        # An `@` that is not a quantization stays part of the id.
        ("some/model@weird-suffix", ("some/model@weird-suffix", None)),
    ],
)
def test_split_artifact_knows_the_quant_convention(raw, expected):
    assert mm.split_artifact(raw) == expected


def test_recommended_text_download_is_the_four_bit_build():
    """The ROUTE stores the served id; the DOWNLOAD names the 4-bit weights.

    Operator ruling 2026-08-01: "the recommended text default MEANS the 4-bit
    quantized qwen3.5-9b". LM Studio serves a single-quant install under the
    bare id, so the two strings differ on purpose -- and the download surface
    must resolve the artifact, never the route's model.
    """

    route = RECOMMENDED_CAPABILITY_DEFAULT_ROUTES["input.text"]
    download = RECOMMENDED_MODEL_DOWNLOADS["input.text"]
    assert route.model == "qwen/qwen3.5-9b"
    assert download["artifact"] == "qwen/qwen3.5-9b@4bit"
    base, quant = mm.split_artifact(download["artifact"])
    assert base == route.model
    assert quant == "4bit"


def test_bare_installed_id_satisfies_a_quantized_artifact():
    """Tolerant in ONE direction (see `_matches_installed_id`)."""

    assert mm._matches_installed_id("qwen/qwen3.5-9b", "qwen/qwen3.5-9b@4bit")
    assert mm._matches_installed_id("qwen/qwen3.5-9b@4bit", "qwen/qwen3.5-9b")
    assert mm._matches_installed_id("QWEN/Qwen3.5-9B", "qwen/qwen3.5-9b@4bit")
    # A DIFFERENT model must never match, quant suffix or not.
    assert not mm._matches_installed_id("qwen/qwen3.5-4b", "qwen/qwen3.5-9b@4bit")
    assert not mm._matches_installed_id("", "qwen/qwen3.5-9b")


# ---------------------------------------------------------------------------
# Rule 2: `unknown` is legal, and never faked into `installed`
# ---------------------------------------------------------------------------


def test_lmstudio_unknown_when_no_cli_and_no_server(monkeypatch):
    monkeypatch.setattr(mm, "_lms_cli", lambda: None)
    monkeypatch.setattr(mm, "_http_json", lambda url, **kw: (None, "connection refused"))
    presence = mm.probe("lmstudio", "qwen/qwen3.5-9b@4bit")
    assert presence.status == mm.PRESENCE_UNKNOWN
    assert "lms" in (presence.instruction or "").lower() or "lm studio" in (presence.instruction or "").lower()
    assert presence.downloadable is False


def test_lmstudio_http_miss_is_unknown_not_absent(monkeypatch):
    """The HTTP endpoint lists SERVED models, a subset of what is downloaded.

    Reporting `absent` from that miss would invite a re-download of weights
    already on disk, which is the expensive way to be wrong.
    """

    monkeypatch.setattr(mm, "_lms_cli", lambda: None)
    monkeypatch.setattr(
        mm,
        "_http_json",
        lambda url, **kw: ({"data": [{"id": "some-other-model"}]}, ""),
    )
    presence = mm.probe("lmstudio", "qwen/qwen3.5-9b@4bit")
    assert presence.status == mm.PRESENCE_UNKNOWN


def test_lmstudio_cli_miss_is_absent(monkeypatch):
    """With `lms ls` available the downloaded set IS known, so a miss is absent."""

    monkeypatch.setattr(mm, "_lms_downloaded_ids", lambda: (["other/model"], ""))
    presence = mm.probe("lmstudio", "qwen/qwen3.5-9b@4bit")
    assert presence.status == mm.PRESENCE_ABSENT
    assert presence.instruction == "lms get qwen/qwen3.5-9b@4bit"
    assert presence.downloadable is True


def test_lmstudio_bare_downloaded_id_reads_as_installed(monkeypatch):
    monkeypatch.setattr(mm, "_lms_downloaded_ids", lambda: (["qwen/qwen3.5-9b"], ""))
    presence = mm.probe("lmstudio", "qwen/qwen3.5-9b@4bit")
    assert presence.status == mm.PRESENCE_INSTALLED
    assert "qwen/qwen3.5-9b" in presence.detail


def test_ollama_installed_and_absent(monkeypatch):
    monkeypatch.setattr(
        mm,
        "_http_json",
        lambda url, **kw: ({"models": [{"name": "gemma3:1b"}, {"name": "all-minilm:22m"}]}, ""),
    )
    assert mm.probe("ollama", "gemma3:1b").status == mm.PRESENCE_INSTALLED
    # Ollama's own `:latest` is implicit, so both spellings are one tag.
    assert mm.probe("ollama", "gemma3:1b:latest").status == mm.PRESENCE_INSTALLED
    absent = mm.probe("ollama", "not-pulled:7b")
    assert absent.status == mm.PRESENCE_ABSENT
    assert absent.instruction == "ollama pull not-pulled:7b"


def test_ollama_unknown_when_daemon_down_and_cli_missing(monkeypatch):
    monkeypatch.setattr(mm, "_http_json", lambda url, **kw: (None, "connection refused"))
    monkeypatch.setattr(mm, "_ollama_cli_ids", lambda: (None, "the `ollama` CLI is not installed"))
    presence = mm.probe("ollama", "gemma3:1b")
    assert presence.status == mm.PRESENCE_UNKNOWN
    assert "ollama serve" in (presence.instruction or "")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter", "endpoint:ovh-provider"])
def test_cloud_and_relay_providers_are_not_applicable(provider):
    presence = mm.probe(provider, "some-model")
    assert presence.status == mm.PRESENCE_NOT_APPLICABLE
    assert presence.downloadable is False
    # And a download attempt says the same thing rather than failing obscurely.
    outcome = mm.download(provider, "some-model")
    assert outcome.status == "not_applicable"
    assert outcome.ok is False


def test_unsupported_provider_is_unknown_with_the_supported_list():
    presence = mm.probe("stable-audio", "stabilityai/stable-audio-open-small")
    assert presence.status == mm.PRESENCE_UNKNOWN
    assert "no local-weights probe" in presence.detail
    assert "ollama" in (presence.instruction or "")


def test_probe_never_raises_and_never_returns_a_bogus_state(monkeypatch):
    def boom(*_a, **_kw):
        raise RuntimeError("hub exploded")

    monkeypatch.setattr(mm, "_probe_ollama", boom)
    presence = mm.probe("ollama", "gemma3:1b")
    assert presence.status == mm.PRESENCE_UNKNOWN
    assert "hub exploded" in presence.detail
    assert presence.status in mm.PRESENCE_STATES


# ---------------------------------------------------------------------------
# Rule 1: probing never downloads
# ---------------------------------------------------------------------------


def test_probing_the_recommended_set_runs_no_download_tool(monkeypatch):
    """A grid render must be free. No subprocess that fetches, no hub call."""

    spawned = []
    real_popen = subprocess.Popen
    fetch_verbs = {"get", "pull", "download", "fetch"}

    def spy_popen(args, *rest, **kwargs):
        argv = list(args) if isinstance(args, (list, tuple)) else [str(args)]
        spawned.append(argv)
        assert not (fetch_verbs & {str(a).lower() for a in argv}), f"probe spawned a FETCH: {argv}"
        return real_popen(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy_popen)
    # No downloader may be reached at all -- `lms ls` / `ollama list` are reads
    # and stay allowed, but the download lane must never open during a probe.
    for name in ("_download_lmstudio", "_download_ollama", "_download_supertonic", "_download_huggingface"):
        monkeypatch.setattr(mm, name, lambda *a, **k: pytest.fail("probe called a downloader"))
    monkeypatch.setattr(mm, "_run_streaming", lambda *a, **k: pytest.fail("probe streamed a download"))

    plan = mm.recommended_plan()
    assert plan["total"] == len(RECOMMENDED_MODEL_DOWNLOADS)
    for row in plan["recommended"]:
        assert row["status"] in mm.PRESENCE_STATES


def test_recommended_plan_would_download_only_absent_artifacts(monkeypatch):
    states = {
        "qwen/qwen3.5-9b@4bit": mm.PRESENCE_ABSENT,
        "supertonic-3": mm.PRESENCE_INSTALLED,
        "AbstractFramework/flux.2-klein-4b-8bit": mm.PRESENCE_UNKNOWN,
    }
    monkeypatch.setattr(
        mm,
        "probe",
        lambda provider, artifact, **kw: mm.ModelPresence(provider, artifact, states[artifact]),
    )
    plan = mm.recommended_plan()
    assert (plan["total"], plan["installed"], plan["absent"], plan["unknown"]) == (3, 1, 1, 1)
    assert [item["artifact"] for item in plan["would_download"]] == ["qwen/qwen3.5-9b@4bit"]
    # An `unknown` row is NOT queued for download: we do not spend gigabytes on
    # a guess, we tell the operator we could not tell.
    assert all(item["artifact"] != "AbstractFramework/flux.2-klein-4b-8bit" for item in plan["would_download"])


# ---------------------------------------------------------------------------
# download(): explicit, idempotent, dry-runnable
# ---------------------------------------------------------------------------


def test_dry_run_resolves_the_command_without_spending_a_byte(monkeypatch):
    monkeypatch.setattr(
        mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT)
    )
    monkeypatch.setattr(mm, "_download_ollama", lambda *a, **k: pytest.fail("dry run downloaded"))
    outcome = mm.download("ollama", "gemma3:1b", dry_run=True)
    assert outcome.status == "planned"
    assert outcome.ok is True
    assert outcome.command == ["ollama", "pull", "gemma3:1b"]


def test_download_short_circuits_when_already_installed(monkeypatch):
    monkeypatch.setattr(
        mm,
        "probe",
        lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_INSTALLED, location="/cache/x"),
    )
    monkeypatch.setattr(mm, "_download_ollama", lambda *a, **k: pytest.fail("re-downloaded an installed model"))
    outcome = mm.download("ollama", "gemma3:1b")
    assert (outcome.ok, outcome.status) == (True, "already_installed")
    assert outcome.location == "/cache/x"


def test_download_still_runs_when_presence_is_unknown(monkeypatch):
    """`unknown` must not block the operator: they asked, so we try."""

    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_UNKNOWN))
    ran = {}

    def fake(artifact, emit, base_url):
        ran["artifact"] = artifact
        return mm.DownloadOutcome("ollama", artifact, True, "completed")

    monkeypatch.setattr(mm, "_DOWNLOADERS", {**mm._DOWNLOADERS, "ollama": fake})
    outcome = mm.download("ollama", "gemma3:1b")
    assert ran["artifact"] == "gemma3:1b"
    assert outcome.status == "completed"


def test_lmstudio_download_without_the_cli_returns_the_install_instruction(monkeypatch):
    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    monkeypatch.setattr(mm, "_lms_cli", lambda: None)
    outcome = mm.download("lmstudio", "qwen/qwen3.5-9b@4bit")
    assert outcome.ok is False
    assert "lms get qwen/qwen3.5-9b@4bit" in (outcome.instruction or "")
    assert outcome.command == ["lms", "get", "qwen/qwen3.5-9b@4bit", "--yes"]


def test_ollama_download_failure_carries_the_tools_own_words(monkeypatch):
    """Rule 4: the provider said it; we relay it, then add ONE line."""

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            yield b'{"status":"pulling manifest"}\n'
            yield b'{"error":"pull model manifest: file does not exist"}\n'

    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    monkeypatch.setattr(mm.urllib.request, "urlopen", lambda *a, **k: _Resp())
    seen = []
    outcome = mm.download("ollama", "nope:9b", progress_cb=seen.append)
    assert outcome.ok is False
    assert outcome.message == "pull model manifest: file does not exist"
    assert "pull model manifest: file does not exist" in outcome.output
    assert "ollama pull nope:9b" in (outcome.instruction or "")
    assert seen and seen[-1].status is mm.DownloadStatus.ERROR


def test_ollama_download_streams_real_byte_progress(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            yield b'{"status":"pulling 797b","total":1000,"completed":500}\n'
            yield b'{"status":"success"}\n'

    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    monkeypatch.setattr(mm.urllib.request, "urlopen", lambda *a, **k: _Resp())
    seen = []
    outcome = mm.download("ollama", "tiny:1b", progress_cb=seen.append)
    assert outcome.ok is True
    halfway = [p for p in seen if p.percent == 50.0]
    assert halfway and halfway[0].downloaded_bytes == 500 and halfway[0].total_bytes == 1000
    assert seen[-1].status is mm.DownloadStatus.COMPLETE


# ---------------------------------------------------------------------------
# Grid annotation: what every console renders
# ---------------------------------------------------------------------------


def test_annotate_maps_the_recommended_route_to_its_quantized_artifact(monkeypatch):
    probed = []

    def fake_probe(provider, artifact, **kw):
        probed.append((provider, artifact))
        return mm.ModelPresence(provider, artifact, mm.PRESENCE_ABSENT)

    monkeypatch.setattr(mm, "probe", fake_probe)
    rows = mm.annotate_route_availability(
        [
            {"key": "input.text", "provider": "lmstudio", "model": "qwen/qwen3.5-9b"},
            {"key": "output.voice", "provider": "supertonic", "model": "supertonic-3"},
            {"key": "input.video", "provider": "", "model": ""},
        ]
    )
    text = next(r for r in rows if r["key"] == "input.text")
    # THE POINT: the row's model is the served id; the thing we would FETCH is
    # the 4-bit artifact, and it is the artifact that was probed.
    assert text["download_artifact"] == "qwen/qwen3.5-9b@4bit"
    assert ("lmstudio", "qwen/qwen3.5-9b@4bit") in probed
    assert text["availability"]["status"] == mm.PRESENCE_ABSENT

    unconfigured = next(r for r in rows if r["key"] == "input.video")
    assert unconfigured["availability"]["status"] == mm.PRESENCE_UNKNOWN
    assert unconfigured["availability"]["evidence"] == "route not configured"
    assert "download_artifact" not in unconfigured


def test_a_covered_row_fetches_what_its_covering_row_fetches(monkeypatch):
    """ONE set of files, ONE instruction.

    `input.image` served by the text model IS the text model's weights. Left to
    resolve on its own it produced `lms get qwen/qwen3.5-9b` right next to
    `lms get qwen/qwen3.5-9b@4bit` for the same download -- two instructions,
    one of them naming no quantization at all.
    """

    probed = []
    monkeypatch.setattr(
        mm,
        "probe",
        lambda p, a, **kw: (probed.append(a), mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))[1],
    )
    rows = mm.annotate_route_availability(
        [
            {"key": "input.text", "provider": "lmstudio", "model": "qwen/qwen3.5-9b"},
            {"key": "input.image", "provider": "lmstudio", "model": "qwen/qwen3.5-9b", "covered_by": "input.text"},
            {"key": "output.text", "provider": "lmstudio", "model": "qwen/qwen3.5-9b", "derived_from": "input.text"},
        ]
    )
    assert {row["download_artifact"] for row in rows} == {"qwen/qwen3.5-9b@4bit"}
    assert set(probed) == {"qwen/qwen3.5-9b@4bit"}, f"one artifact, not two: {probed}"


def test_annotate_keeps_an_operator_override_as_written(monkeypatch):
    """An operator who pinned a DIFFERENT model gets that model probed.

    The recommendation only supplies the artifact when the row still names the
    recommended model; otherwise the console would offer to download weights
    the operator deliberately replaced.
    """

    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_INSTALLED))
    rows = mm.annotate_route_availability(
        [{"key": "input.text", "provider": "ollama", "model": "gemma3:1b"}]
    )
    assert "download_artifact" not in rows[0]
    assert rows[0]["availability"]["artifact"] == "gemma3:1b"


def test_annotate_probes_each_distinct_pair_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mm,
        "probe",
        lambda p, a, **kw: (calls.append((p, a)), mm.ModelPresence(p, a, mm.PRESENCE_INSTALLED))[1],
    )
    mm.annotate_route_availability(
        [
            {"key": "input.text", "provider": "ollama", "model": "gemma3:1b"},
            {"key": "input.image", "provider": "ollama", "model": "gemma3:1b"},
            {"key": "output.voice", "provider": "supertonic", "model": "supertonic-3"},
        ]
    )
    assert len(calls) == 2, f"one probe per distinct provider/model pair, got {calls}"


# ---------------------------------------------------------------------------
# The CLI surface
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, capsys, argv):
    from abstractcore.config import main as config_main

    code = config_main.main(argv)
    return code, capsys.readouterr().out


def test_models_status_json_is_machine_readable(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    config_file = tmp_path / "abstractcore.json"
    code, out = _run_cli(
        monkeypatch, capsys, ["models", "--config-file", str(config_file), "status", "--json"]
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
    assert isinstance(payload["routes"], list) and payload["routes"]
    assert payload["recommended"]["total"] == len(RECOMMENDED_MODEL_DOWNLOADS)
    assert set(payload["providers"]) >= {"lmstudio", "ollama", "supertonic", "mlx-gen"}
    for row in payload["routes"]:
        assert row["availability"]["status"] in mm.PRESENCE_STATES


def test_models_download_recommended_dry_run_names_the_four_bit_artifact(monkeypatch, capsys):
    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    monkeypatch.setattr(mm, "_lms_cli", lambda: "/usr/local/bin/lms")
    for name in ("_download_lmstudio", "_download_ollama", "_download_supertonic", "_download_huggingface"):
        monkeypatch.setattr(mm, name, lambda *a, **k: pytest.fail("--dry-run downloaded"))
    code, out = _run_cli(monkeypatch, capsys, ["models", "download", "--recommended", "--dry-run", "--json"])
    assert code == 0
    payload = json.loads(out[out.index("{") :])
    artifacts = {r["artifact"]: r for r in payload["results"]}
    assert artifacts["qwen/qwen3.5-9b@4bit"]["status"] == "planned"
    assert artifacts["qwen/qwen3.5-9b@4bit"]["command"][:2] == ["/usr/local/bin/lms", "get"]
    assert payload["dry_run"] is True


def test_models_download_requires_an_explicit_target(monkeypatch, capsys):
    code, out = _run_cli(monkeypatch, capsys, ["models", "download"])
    assert code == 1
    assert "--recommended" in out


def test_models_download_refuses_recommended_plus_a_named_artifact(monkeypatch, capsys):
    code, out = _run_cli(monkeypatch, capsys, ["models", "download", "--recommended", "ollama", "gemma3:1b"])
    assert code == 1
    assert "do not also name" in out


def test_models_status_accepts_a_provider_model_pair(monkeypatch, capsys):
    monkeypatch.setattr(
        mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_INSTALLED, detail="pulled")
    )
    code, out = _run_cli(monkeypatch, capsys, ["models", "status", "ollama/gemma3:1b"])
    assert code == 0
    assert "ollama/gemma3:1b" in out and "installed" in out


def test_cli_progress_printer_collapses_repeats(capsys):
    from abstractcore.config.main import _models_progress_printer

    emit, finish = _models_progress_printer()
    emit(SimpleNamespace(message="pulling manifest", percent=None))
    emit(SimpleNamespace(message="pulling manifest", percent=None))
    emit(SimpleNamespace(message="verifying", percent=None))
    finish()
    out = capsys.readouterr().out
    assert out.count("pulling manifest") == 1
    assert "verifying" in out


# ---------------------------------------------------------------------------
# Rule 2, the hardest cases: losing the evidence must never read as a fact
# ---------------------------------------------------------------------------


def _hf_repo(root, repo_id, *, files=("model.safetensors",), incomplete=()):
    """Build a Hugging Face hub cache the way `huggingface_hub` writes one."""

    repo = root / ("models--" + repo_id.replace("/", "--"))
    blobs = repo / "blobs"
    snap = repo / "snapshots" / "rev1"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    for index, name in enumerate(files):
        blob = blobs / f"sha{index}"
        blob.write_bytes(b"\x00" * 16)
        # The hub symlinks a snapshot entry ONLY once the blob is whole.
        (snap / name).symlink_to(blob)
    for index, _ in enumerate(incomplete):
        (blobs / f"pending{index}.incomplete").write_bytes(b"\x00" * 32)
    return repo


def test_an_interrupted_hf_download_is_absent_not_installed(monkeypatch, tmp_path):
    """THE WORST LIE: a half-downloaded repo whose snapshot dir looks perfect.

    `huggingface_hub` writes an in-flight file to `blobs/<sha>.incomplete` and
    only symlinks it into `snapshots/<rev>/` once it is COMPLETE. So an eight
    shard model interrupted after shard one leaves a snapshot directory in
    which every file is whole -- and a `.incomplete` scan of that directory
    finds nothing, because the evidence is one level up in `blobs/`.

    Reported as `installed`, that costs the operator a run that dies at load
    time with a shape error. It is `absent`: the repair is known, and the same
    download resumes exactly where it stopped.
    """

    cache = tmp_path / "hub"
    cache.mkdir()
    _hf_repo(cache, "acme/half-model", files=("shard-1.safetensors",), incomplete=("a", "b", "c"))
    monkeypatch.setattr(mm, "_hf_cache_dirs", lambda: [cache])

    presence = mm.probe("mlx-gen", "acme/half-model")
    assert presence.status == mm.PRESENCE_ABSENT, "a partial snapshot must never read as installed"
    assert "interrupted" in presence.evidence
    assert "3 interrupted file(s)" in presence.detail
    assert presence.downloadable is True
    assert "resume" in (presence.instruction or "")


def test_a_complete_hf_snapshot_is_still_installed(monkeypatch, tmp_path):
    """The interruption check must not turn every cached model into a re-download."""

    cache = tmp_path / "hub"
    cache.mkdir()
    _hf_repo(cache, "acme/whole-model", files=("a.safetensors", "b.safetensors"))
    monkeypatch.setattr(mm, "_hf_cache_dirs", lambda: [cache])

    presence = mm.probe("mlx-gen", "acme/whole-model")
    assert presence.status == mm.PRESENCE_INSTALLED
    assert presence.location and "acme--whole-model" in presence.location


def test_an_lms_output_shape_we_do_not_recognise_is_unknown(monkeypatch):
    """`lms` is a third-party CLI. Losing its shape loses the evidence.

    Reading an unrecognised payload as "0 models downloaded" turned every
    LM Studio row `absent` at once -- an offer to re-fetch a library already on
    disk, from no evidence at all.
    """

    monkeypatch.setattr(mm, "_lms_cli", lambda: "/fake/lms")
    monkeypatch.setattr(mm, "_lmstudio_served_ids", lambda _b: (None, "server down"))

    def answering(payload):
        return lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    # An object instead of a list, and rows whose id keys were renamed.
    for payload in ({"models": []}, [{"newKeyName": "qwen/qwen3.5-9b"}]):
        monkeypatch.setattr(mm.subprocess, "run", answering(payload))
        presence = mm.probe("lmstudio", "qwen/qwen3.5-9b")
        assert presence.status == mm.PRESENCE_UNKNOWN, f"{payload!r} must not read as an empty library"

    # A genuinely EMPTY library is different: that is real evidence of absence.
    monkeypatch.setattr(mm.subprocess, "run", answering([]))
    assert mm.probe("lmstudio", "qwen/qwen3.5-9b").status == mm.PRESENCE_ABSENT


def test_a_remote_ollama_is_never_answered_by_the_local_cli(monkeypatch):
    """The `ollama` CLI answers about ONE daemon: the local one.

    Falling back to it for a route pinned to another Ollama reported the wrong
    machine's library and printed `ollama pull ...`, an instruction that fetches
    the weights onto the wrong host. Host AND port count -- a second daemon on
    another port is as much "not this one" as a box across the network.
    """

    monkeypatch.setattr(mm, "_http_json", lambda url, **kw: (None, "connection refused"))
    monkeypatch.setattr(mm, "_ollama_cli_ids", lambda: (["local-only:1b"], ""))

    for url in ("http://10.0.0.9:11434", "http://127.0.0.1:59999"):
        presence = mm.probe("ollama", "local-only:1b", base_url=url)
        assert presence.status == mm.PRESENCE_UNKNOWN, f"{url} answered from the local CLI"
        assert "ollama pull" not in (presence.instruction or "")
        assert url in (presence.instruction or "")

    # The loopback SPELLINGS of the default daemon really are the local one.
    for url in ("http://localhost:11434", "http://127.0.0.1:11434"):
        assert mm.probe("ollama", "local-only:1b", base_url=url).status == mm.PRESENCE_INSTALLED


def test_a_sweep_reads_each_provider_listing_once(monkeypatch):
    """A wedged `lms ls` must cost its timeout ONCE per payload, not once per row.

    One availability payload probes the grid and the recommended set. Before
    the sweep that was three `lms ls` runs, so a hung CLI turned one console
    refresh into three full timeouts.
    """

    runs = []

    def counting(cmd, *a, **kw):
        runs.append(cmd)
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"modelKey": "already/here"}]), stderr="")

    monkeypatch.setattr(mm, "_lms_cli", lambda: "/fake/lms")
    monkeypatch.setattr(mm.subprocess, "run", counting)
    rows = [
        {"key": "input.text", "provider": "lmstudio", "model": "qwen/qwen3.5-9b"},
        {"key": "embedding.text", "provider": "lmstudio", "model": "some/embedder"},
        {"key": "output.text", "provider": "lmstudio", "model": "another/model"},
    ]

    with mm.presence_sweep():
        mm.annotate_route_availability(rows)
        mm.recommended_plan()
    assert len(runs) == 1, f"one listing per sweep, ran {len(runs)}"

    # OUTSIDE a sweep nothing is cached: no hidden state between callers.
    runs.clear()
    mm.probe("lmstudio", "qwen/qwen3.5-9b")
    mm.probe("lmstudio", "qwen/qwen3.5-9b")
    assert len(runs) == 2, "a lone probe() must stay uncached"


def test_lms_get_is_verified_because_it_searches_rather_than_fetches(monkeypatch):
    """`lms get` resolves a SEARCH, and `--yes` accepts the first match.

    Its own docs: "if there are multiple models matching the search term, the
    first one will be used". A `@quant` that does not exist fails loudly (good),
    but a stale or mistyped reference can still fetch a different repo, approved
    without a prompt. So the outcome is checked against the request.
    """

    monkeypatch.setattr(mm, "_lms_cli", lambda: "/fake/lms")
    monkeypatch.setattr(
        mm,
        "_run_streaming",
        lambda cmd, provider, artifact, emit: mm.DownloadOutcome(
            provider, artifact, True, "completed", message="downloaded", output="Downloaded some-other-model"
        ),
    )

    # The probe positively says the requested artifact is NOT there: the tool
    # succeeded at fetching SOMETHING, but not what was asked for.
    monkeypatch.setattr(
        mm,
        "probe",
        lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT, evidence="lms ls --json"),
    )
    out = mm._download_lmstudio("qwen/qwen3.5-9b@4bit", lambda _p: None, None)
    assert out.ok is False
    assert "is not among the downloaded models" in out.message
    assert "lms ls" in (out.instruction or "")

    # `unknown` is NOT evidence of a mismatch and must not fail a good download.
    monkeypatch.setattr(
        mm,
        "probe",
        lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_UNKNOWN, evidence="no lms CLI"),
    )
    assert mm._download_lmstudio("qwen/qwen3.5-9b@4bit", lambda _p: None, None).ok is True


def test_a_download_never_answers_from_a_sweep_taken_before_it(monkeypatch):
    """A sweep caches a library listing; a download changes that library.

    Reusing the pre-download read to answer "did it land?" would report the
    weights absent immediately after fetching them.
    """

    reads = []

    def listing():
        reads.append(1)
        return (["already/here"] if len(reads) > 1 else [], "")

    monkeypatch.setattr(mm, "_lms_cli", lambda: "/fake/lms")
    monkeypatch.setattr(mm, "_read_lms_downloaded_ids", listing)
    monkeypatch.setattr(
        mm,
        "_run_streaming",
        lambda cmd, provider, artifact, emit: mm.DownloadOutcome(
            provider, artifact, True, "completed", message="downloaded"
        ),
    )

    with mm.presence_sweep():
        mm.probe("lmstudio", "already/here")  # caches the EMPTY library
        outcome = mm.download("lmstudio", "already/here")
    assert outcome.ok is True, "the post-download check must re-read, not reuse the sweep"
    assert len(reads) > 1
