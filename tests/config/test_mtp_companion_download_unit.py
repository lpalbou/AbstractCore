"""An MLX model's MTP companion is downloaded, listed and deleted WITH the model.

The registry (`model_capabilities.json` `speculation.runtimes.mlx.drafter`)
names a separate drafter repo for several MLX models; loading never downloads
it (offline_first), so before this a fresh install ran without MTP forever and
nobody said so (mission W2). Here: one job fetches both (the companion as child
file rows), sizes add up, cancel covers both, `completed`/`installed` only when
both are whole, `models list` shows the companion under its model, and a delete
offers to remove it. The hub is faked: no network, no real cache.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from abstractcore.config import host_jobs
from abstractcore.config import model_materializer as mm
from abstractcore.download import DownloadProgress, DownloadStatus

MODEL = "mlx-works/Qwen3.5-9B-oQ4e-mtp"
COMPANION = "mlx-community/Qwen3.5-9B-MTP-4bit"
SIZES = {MODEL: [("model-00001-of-00002.safetensors", 4000), ("model-00002-of-00002.safetensors", 2000)],
         COMPANION: [("model.safetensors", 160)]}


def test_the_registry_names_the_companion():
    assert mm.companion_artifacts("mlx", MODEL) == [COMPANION]
    assert mm.companion_artifacts("mlx", "Jundot/Qwen3.8-27B-oQ4e-mtp") == ["mlx-community/Qwen3.8-27B-MTP-4bit"]


def test_built_in_heads_and_drafters_have_no_companion():
    assert mm.companion_artifacts("mlx", "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp") == []
    assert mm.companion_artifacts("mlx", COMPANION) == []
    assert mm.companion_artifacts("ollama", MODEL) == []


class _Hub:
    """A fake Hugging Face: per-repo presence, file plans, and a downloader."""

    def __init__(self, monkeypatch, *, installed=(), fail=None, cancel_during=None):
        self.installed = set(installed)
        self.fetched: List[str] = []
        self.fail = fail
        self.cancel_during = cancel_during
        hub = self

        def probe_hf(provider, artifact):
            repo, _q, _p = mm.hf_artifact_parts(artifact)
            if repo in hub.installed:
                return mm.ModelPresence(provider, artifact, mm.PRESENCE_INSTALLED, location=f"/cache/{repo}")
            return mm.ModelPresence(provider, artifact, mm.PRESENCE_ABSENT)

        def plan(repo, patterns, token):
            return [{"name": n, "size": s, "etag": n} for n, s in SIZES[repo]], "sha", ""

        def download(artifact, emit, base_url):
            hub.fetched.append(artifact)
            files = [{"name": n, "bytes_done": 0, "bytes_total": s, "state": "pending"} for n, s in SIZES[artifact]]
            total = sum(s for _n, s in SIZES[artifact])
            emit(DownloadProgress(status=DownloadStatus.STARTING, message=f"reading the file list of {artifact}", phase="resolving"))
            done = 0
            for row in files:
                if hub.cancel_during == artifact:
                    control = host_jobs.current_job_control()
                    control.cancel()
                    return mm.DownloadOutcome("huggingface", artifact, False, "cancelled", message="cancelled")
                row.update(bytes_done=row["bytes_total"], state="done")
                done += row["bytes_total"]
                emit(DownloadProgress(status=DownloadStatus.DOWNLOADING, message=f"{artifact}: {done}", downloaded_bytes=done,
                                      total_bytes=total, files=[dict(f) for f in files], current_file=row["name"], phase="downloading"))
            if hub.fail == artifact:
                return mm.DownloadOutcome("huggingface", artifact, False, "failed", message="HTTP 500 from the hub")
            hub.installed.add(artifact)
            emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"cached at /cache/{artifact}", percent=100.0))
            return mm.DownloadOutcome("huggingface", artifact, True, "completed", location=f"/cache/{artifact}")

        monkeypatch.setattr(mm, "_probe_huggingface", probe_hf)
        monkeypatch.setattr(mm, "_hf_file_plan", plan)
        monkeypatch.setattr(mm, "_download_huggingface", download)
        monkeypatch.setattr(mm, "_disk_shortfall", lambda *a, **k: None)


def _job(artifact=MODEL, **kw) -> Dict[str, Any]:
    reg = host_jobs.HostJobRegistry(tick_s=0.05)
    started = host_jobs.start_download_job("mlx", artifact, registry=reg, run_inline=True, **kw)
    return reg.get(started["job_id"]) or started


def test_one_job_fetches_the_model_and_its_companion_as_child_rows(monkeypatch):
    hub = _Hub(monkeypatch)
    job = _job()
    assert hub.fetched == [MODEL, COMPANION]
    assert job["status"] == "completed" and job["state"] == "done"
    assert job["bytes_total"] == 6000 + 160, "the job's size must include the companion"
    assert job["size_unknown"] is False
    names = [f["name"] for f in job["files"]]
    assert f"{COMPANION}/model.safetensors" in names
    companion_rows = [f for f in job["files"] if f.get("role") == "mtp_companion"]
    assert companion_rows and all(f["artifact"] == COMPANION and f["state"] == "done" for f in companion_rows)
    result = job["result"]
    assert result["status"] == "completed"
    assert result["companions"][0]["artifact"] == COMPANION
    assert result["companions"][0]["status"] == "completed"
    assert result["companions"][0]["size_bytes"] == 160
    assert mm.probe("mlx", MODEL).status == mm.PRESENCE_INSTALLED


def test_the_total_is_known_before_the_companion_starts(monkeypatch):
    _Hub(monkeypatch)
    seen: List[Any] = []
    mm.download("mlx", MODEL, progress_cb=seen.append)
    first_with_total = next(p for p in seen if p.total_bytes)
    assert first_with_total.total_bytes == 6160
    assert any(f.get("role") == "mtp_companion" and f["state"] == "pending" for f in (first_with_total.files or []))


def test_a_failed_companion_fails_the_job_and_the_model_is_not_installed(monkeypatch):
    hub = _Hub(monkeypatch, fail=COMPANION)
    job = _job()
    assert job["status"] == "failed" and job["state"] == "failed"
    assert COMPANION in job["error"] and "not reported installed" in job["error"]
    assert job["result"]["companions"][0]["status"] == "failed"
    presence = mm.probe("mlx", MODEL)
    assert presence.status == mm.PRESENCE_ABSENT
    assert COMPANION in presence.detail and "MTP" in presence.detail
    assert hub.installed == {MODEL}


def test_model_present_companion_missing_fetches_only_the_companion(monkeypatch):
    hub = _Hub(monkeypatch, installed={MODEL})
    assert mm.probe("mlx", MODEL).status == mm.PRESENCE_ABSENT
    job = _job()
    assert hub.fetched == [COMPANION]
    assert job["status"] == "completed"
    assert job["bytes_total"] == 160


def test_both_present_is_already_installed(monkeypatch):
    hub = _Hub(monkeypatch, installed={MODEL, COMPANION})
    job = _job()
    assert hub.fetched == []
    assert job["result"]["status"] == "already_installed"
    assert job["result"]["companions"][0]["status"] == "already_installed"


def test_cancel_covers_the_companion(monkeypatch):
    hub = _Hub(monkeypatch, cancel_during=MODEL)
    job = _job()
    assert job["status"] == "cancelled"
    assert hub.fetched == [MODEL], "the companion must not start after a cancel"
    assert job["result"]["companions"][0]["status"] == "cancelled"
    assert mm.probe("mlx", MODEL).status == mm.PRESENCE_ABSENT


def test_dry_run_names_the_companion_and_its_size(monkeypatch):
    hub = _Hub(monkeypatch)
    out = mm.download("mlx", MODEL, dry_run=True)
    assert hub.fetched == []
    assert out.status == "planned"
    assert COMPANION in out.message
    assert out.to_dict()["companions"][0]["status"] == "planned"
    assert out.to_dict()["companions"][0]["size_bytes"] == 160


def test_flash_next_downloads_alone(monkeypatch):
    SIZES["Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"] = [("model.safetensors", 10)]
    try:
        hub = _Hub(monkeypatch)
        job = _job("Jundot/Qwen3.8-Flash-Next-oQ4e-mtp")
        assert hub.fetched == ["Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"]
        assert "companions" not in (job["result"] or {})
    finally:
        SIZES.pop("Jundot/Qwen3.8-Flash-Next-oQ4e-mtp")


# --- listing & delete ----------------------------------------------------------


def _fake_cache(monkeypatch, repos):
    monkeypatch.setattr(
        mm,
        "_hf_repos",
        lambda: (
            [
                {"repo_id": r, "repo_path": Path(f"/cache/models--{r.replace('/', '--')}"), "size_bytes": size,
                 "revisions": ["abc"], "nb_files": 1, "cache_dir": Path("/cache")}
                for r, size in repos.items()
            ],
            None,
        ),
    )
    monkeypatch.setattr(mm, "_classify_hf_repo", lambda repo_id, path: "mlx")
    monkeypatch.setattr(mm, "_hf_interrupted_downloads", lambda repo_id: (0, 0, None))
    monkeypatch.setattr(mm, "_hf_local_kind_tasks", lambda path: {})


def test_the_listing_shows_the_companion_under_its_model(monkeypatch):
    _fake_cache(monkeypatch, {MODEL: 6000, COMPANION: 160})
    payload = mm.list_installed("mlx")
    artifacts = [r["artifact"] for r in payload["rows"]]
    assert artifacts == [MODEL], f"the companion must not be a stray row: {artifacts}"
    comp = payload["rows"][0]["companions"][0]
    assert comp["artifact"] == COMPANION and comp["installed"] is True and comp["size_bytes"] == 160
    assert comp["companion_of"] == [MODEL]
    assert payload["totals"]["size_bytes"] == 6160


def test_a_missing_companion_is_listed_as_not_installed(monkeypatch):
    _fake_cache(monkeypatch, {MODEL: 6000})
    row = mm.list_installed("mlx")["rows"][0]
    assert row["companions"] == [dict(row["companions"][0], artifact=COMPANION, installed=False)]


def test_an_orphan_companion_stays_visible_and_marked(monkeypatch):
    _fake_cache(monkeypatch, {COMPANION: 160})
    rows = mm.list_installed("mlx")["rows"]
    assert [r["artifact"] for r in rows] == [COMPANION]
    assert rows[0]["role"] == "mtp_companion" and rows[0]["companion_of"] == []


def _deletes(monkeypatch) -> List[str]:
    removed: List[str] = []

    def delete_hf(row, out, dry_run):
        removed.append(row["artifact"])
        out.update(ok=True, status="planned" if dry_run else "deleted", message=f"deleted {row['artifact']}")
        return out

    monkeypatch.setattr(mm, "_delete_hf", delete_hf)
    # delete_blockers lists EVERY engine's installed models; unstubbed, the
    # Ollama probe reached the operator's live :11434 (network guard finding,
    # 2026-09-24). This test's world holds only the faked HF cache.
    monkeypatch.setattr(mm, "_installed_ollama", lambda *a, **k: ([], None))
    monkeypatch.setattr(mm, "_installed_lmstudio", lambda *a, **k: ([], None))
    return removed


def test_delete_offers_the_companion_without_removing_it(monkeypatch):
    _fake_cache(monkeypatch, {MODEL: 6000, COMPANION: 160})
    removed = _deletes(monkeypatch)
    out = mm.delete_artifact("mlx", MODEL)
    assert removed == [MODEL]
    offer = out["companion_offer"][0]
    assert offer["artifact"] == COMPANION and offer["shared_with"] == [] and offer["size_bytes"] == 160
    assert offer["command"] == f"abstractcore models delete mlx {COMPANION} --yes"


def test_delete_with_companions_removes_it(monkeypatch):
    _fake_cache(monkeypatch, {MODEL: 6000, COMPANION: 160})
    removed = _deletes(monkeypatch)
    out = mm.delete_artifact("mlx", MODEL, with_companions=True)
    assert removed == [MODEL, COMPANION]
    assert out["companions_deleted"][0]["ok"] is True


def test_a_shared_companion_is_kept(monkeypatch):
    other = "mlx-community/Qwen3.5-9B-MLX-4bit"
    _fake_cache(monkeypatch, {MODEL: 6000, other: 5900, COMPANION: 160})
    removed = _deletes(monkeypatch)
    out = mm.delete_artifact("mlx", MODEL, with_companions=True)
    assert removed == [MODEL]
    assert out["companions_deleted"][0]["status"] == "kept"
    assert other in out["companion_offer"][0]["shared_with"]


def test_the_companion_can_still_be_deleted_by_name(monkeypatch):
    _fake_cache(monkeypatch, {MODEL: 6000, COMPANION: 160})
    removed = _deletes(monkeypatch)
    out = mm.delete_artifact("mlx", COMPANION)
    assert out["ok"] is True and removed == [COMPANION]


# --- CLI help ----------------------------------------------------------------------


def test_download_help_lists_mlx(capsys):
    from abstractcore.config.main import _handle_models_subcommand

    with pytest.raises(SystemExit):
        _handle_models_subcommand(["download", "--help"])
    text = capsys.readouterr().out
    provider_help = text.split("provider", 2)[-1]
    assert " mlx," in provider_help or " mlx\n" in provider_help, text


def test_verify_is_a_models_subcommand(capsys):
    from abstractcore.config.main import _handle_models_subcommand

    with pytest.raises(SystemExit):
        _handle_models_subcommand(["verify", "--help"])
    assert "speculation.used" in capsys.readouterr().out


def test_size_is_not_reported_unknown_while_downloading_with_a_known_total(monkeypatch):
    _Hub(monkeypatch)
    reg = host_jobs.HostJobRegistry(tick_s=0.05)
    seen: List[Dict[str, Any]] = []
    reg.add_listener(seen.append)
    host_jobs.start_download_job("mlx", MODEL, registry=reg, run_inline=True)
    mid = [s for s in seen if s["state"] == "downloading" and s["bytes_total"]]
    assert mid, "no in-flight snapshot was recorded"
    assert all(s["size_unknown"] is False for s in mid), [s["size_unknown"] for s in mid]


def test_the_catalog_companions_agree_with_the_registry(tmp_path, monkeypatch):
    """SEAM with mission W1: the catalog payload's `companions` (what the console and
    the downloader read) come FROM the registry drafter (what the download fetches
    and the provider loads) -- one source. The seed carries no hand-typed
    companions, only their verified sizes (`companion_sizes`); a registry companion
    with no recorded size fails here, loudly."""
    from abstractcore.config import model_catalog as mc
    from tests.models_engines_fakes import isolate_host, synthetic_host

    isolate_host(tmp_path, monkeypatch)
    seed = mc.load_seed()
    assert not any("companions" in a for r in seed["rows"] for a in r["artifacts"]), "hand-typed companions are back in the seed"
    payload = mc.catalog(host=synthetic_host("metal64"))
    by_artifact = {a["artifact"]: a for r in payload["rows"] for a in r["artifacts"]}
    assert by_artifact[MODEL]["companions"] == [COMPANION], "the 9B MTP build lost its companion"
    assert by_artifact["Jundot/Qwen3.8-27B-oQ4e-mtp"]["companions"] == ["mlx-community/Qwen3.8-27B-MTP-4bit"]
    assert by_artifact["mlx-community/Qwen3.8-27B-4bit"]["companions"] == ["mlx-community/Qwen3.8-27B-MTP-4bit"]
    assert by_artifact["mlx-community/Qwen3.5-9B-MLX-4bit"]["companions"] == [COMPANION]
    assert by_artifact["Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"]["companions"] == []  # built-in head
    for art in by_artifact.values():
        assert mm.companion_artifacts(art["provider"], art["artifact"]) == art["companions"], art["artifact"]
        if art["companions"]:
            assert isinstance(art["companion_bytes"], int) and art["companion_bytes"] > 0, art["artifact"]
