"""Contract D (installed models, with sizes) and the delete verb.

Every engine is a fake: an HTTP Ollama on a free port, an `lms` executable on
PATH that answers with the field names the real CLI prints, and a Hugging Face
cache laid out in tmp_path exactly as huggingface_hub writes it.
"""

from __future__ import annotations

import json

import pytest

from abstractcore.config import model_materializer as mm
from tests.models_engines_fakes import (
    REAL_SHAPED_LMS_ROWS,
    FakeOllama,
    install_fake_lms,
    isolate_host,
    lms_calls,
    make_hf_repo,
    ollama_tag,
)


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


def _lmstudio_files(models_root):
    """On-disk layout matching REAL_SHAPED_LMS_ROWS (+ the hub manifest)."""

    (models_root / "lmstudio-community" / "Qwen3.8-27B-GGUF").mkdir(parents=True)
    (models_root / "lmstudio-community" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-Q4_K_M.gguf").write_bytes(b"g" * 64)
    mlx_dir = models_root / "mlx-community" / "Llama-3.2-1B-Instruct-4bit"
    mlx_dir.mkdir(parents=True)
    (mlx_dir / "model.safetensors").write_bytes(b"s" * 32)
    emb = models_root / "Qwen" / "Qwen3-Embedding-0.6B-GGUF"
    emb.mkdir(parents=True)
    (emb / "Qwen3-Embedding-0.6B-Q8_0.gguf").write_bytes(b"e" * 16)
    (emb / "Qwen3-Embedding-0.6B-f16.gguf").write_bytes(b"f" * 16)
    hub = models_root.parent / "hub" / "models" / "qwen" / "qwen3.8-27b"
    hub.mkdir(parents=True)
    (hub / "manifest.json").write_text(
        json.dumps(
            {
                "type": "model",
                "owner": "qwen",
                "name": "qwen3.8-27b",
                "dependencies": [
                    {
                        "type": "model",
                        "purpose": "baseModel",
                        "modelKeys": ["lmstudio-community/qwen3.8-27b-gguf"],
                        "sources": [{"type": "huggingface", "user": "lmstudio-community", "repo": "Qwen3.8-27B-GGUF"}],
                    }
                ],
                "revision": 2,
            }
        )
    )
    return hub


# ---------------------------------------------------------------------------
# list_installed
# ---------------------------------------------------------------------------


def test_ollama_rows_keep_size_params_and_quant_from_api_tags(host, monkeypatch):
    models = [ollama_tag("qwen3:8b", 5_225_388_164, "8.2B", "Q4_K_M"), ollama_tag("gemma3:1b", 815_319_791, "999.89M", "Q4_K_M", "gemma3")]
    with FakeOllama(models, loaded=["qwen3:8b"]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)  # the daemon THIS machine's CLI talks to
        payload = mm.list_installed("ollama")
    assert payload["schema"] == "models_installed_v1"
    assert payload["engines_probed"] == ["ollama"]
    assert payload["errors"] == {}
    rows = {r["artifact"]: r for r in payload["rows"]}
    assert rows["qwen3:8b"]["size_bytes"] == 5_225_388_164
    assert rows["qwen3:8b"]["params_total"] == 8_200_000_000
    assert rows["qwen3:8b"]["quant"] == "q4_k_m"
    assert rows["qwen3:8b"]["loaded"] is True
    assert rows["qwen3:8b"]["delete_blockers"] == ["loaded"]
    assert rows["gemma3:1b"]["loaded"] is False
    assert rows["gemma3:1b"]["delete_blockers"] == []
    assert rows["qwen3:8b"]["catalog_id"] == "qwen3-8b"


def test_an_ollama_daemon_on_another_host_is_a_remote_engine(host):
    with FakeOllama([ollama_tag("qwen3:8b", 10, "8B", "Q4_K_M")]) as fake:
        # OLLAMA_BASE_URL still names the (dead) local daemon: fake.url is "elsewhere".
        payload = mm.list_installed("ollama", base_urls={"ollama": fake.url})
        refused = mm.delete_artifact("ollama", "qwen3:8b", base_url=fake.url)
    assert payload["rows"][0]["delete_blockers"] == ["remote_engine"]
    assert payload["rows"][0]["location"] == fake.url
    assert refused["status"] == "refused" and refused["delete_blockers"] == ["remote_engine"]


def test_unreachable_local_ollama_lists_on_disk_manifests_and_says_so(host):
    manifest = host["ollama_models"] / "manifests" / "registry.ollama.ai" / "library" / "gemma3"
    manifest.mkdir(parents=True)
    (manifest / "1b").write_text(json.dumps({"config": {"size": 492}, "layers": [{"size": 815310432}, {"size": 358}]}))
    payload = mm.list_installed("ollama")
    assert "unreachable" in payload["errors"]["ollama"]
    row = payload["rows"][0]
    assert row["artifact"] == "gemma3:1b"
    assert row["size_bytes"] == 815310432 + 358 + 492
    assert row["deletable"] is False
    assert row["delete_blockers"] == ["engine_not_running"]


def test_lmstudio_rows_use_lms_ls_json_fields_and_resolve_locations(host, monkeypatch):
    install_fake_lms(host["fakebin"], monkeypatch, REAL_SHAPED_LMS_ROWS, ps=[{"identifier": "llama-3.2-1b-instruct", "modelKey": "llama-3.2-1b-instruct"}])
    _lmstudio_files(host["lms_models"])
    payload = mm.list_installed("lmstudio")
    assert payload["errors"] == {}
    rows = {r["artifact"]: r for r in payload["rows"]}
    hub_row = rows["qwen/qwen3.8-27b@q4_k_m"]  # selectedVariant wins over modelKey
    assert hub_row["size_bytes"] == 17742039110
    assert hub_row["params_total"] == 27_000_000_000
    assert hub_row["quant"] == "q4_k_m"
    assert hub_row["location"].endswith("lmstudio-community/Qwen3.8-27B-GGUF")
    assert hub_row["max_context"] == 262144
    loaded = rows["llama-3.2-1b-instruct"]
    assert loaded["loaded"] is True and loaded["delete_blockers"] == ["loaded"]
    emb = rows["text-embedding-qwen3-embedding-0.6b"]
    assert emb["location"].endswith("Qwen3-Embedding-0.6B-Q8_0.gguf")
    assert emb["type"] == "embedding"
    # Bundled with the app: no file under the models root -> not deletable.
    nomic = rows["text-embedding-nomic-embed-text-v1.5"]
    assert nomic["deletable"] is False
    assert nomic["delete_blockers"] == ["unknown_location"]


def test_lms_missing_is_an_error_not_an_empty_library(host):
    payload = mm.list_installed("lmstudio")
    assert payload["rows"] == []
    assert "lms" in payload["errors"]["lmstudio"]


def test_hf_cache_splits_into_mlx_and_huggingface_rows_with_sizes(host):
    make_hf_repo(host["hf"], "mlx-community/Qwen3-8B-4bit", {"model.safetensors": b"m" * 100, "config.json": b"{}"})
    make_hf_repo(host["hf"], "unsloth/Qwen3-8B-GGUF", {"Qwen3-8B-Q4_K_M.gguf": b"q" * 300})
    make_hf_repo(host["hf"], "Qwen/Qwen3-4B", {"model.safetensors": b"w" * 50}, incomplete=1)
    payload = mm.list_installed()
    rows = {(r["provider"], r["artifact"]): r for r in payload["rows"]}
    assert ("mlx", "mlx-community/Qwen3-8B-4bit") in rows
    assert ("huggingface", "unsloth/Qwen3-8B-GGUF") in rows
    assert rows[("huggingface", "unsloth/Qwen3-8B-GGUF")]["size_bytes"] == 300
    assert rows[("mlx", "mlx-community/Qwen3-8B-4bit")]["quant"] == "4bit"
    assert rows[("mlx", "mlx-community/Qwen3-8B-4bit")]["params_total"] == 8_000_000_000
    assert rows[("huggingface", "Qwen/Qwen3-4B")]["incomplete_files"] == 1
    assert set(payload["engines_probed"]) == {"ollama", "lmstudio", "mlx", "huggingface"}
    assert payload["totals"]["count"] == len(payload["rows"])


def test_remote_provider_is_reported_not_listed(host):
    payload = mm.list_installed("openai")
    assert payload["rows"] == []
    assert "remote" in payload["errors"]["openai"]


# ---------------------------------------------------------------------------
# delete_artifact
# ---------------------------------------------------------------------------


def test_ollama_delete_uses_api_delete_and_refuses_loaded_without_force(host, monkeypatch):
    models = [ollama_tag("qwen3:8b", 1000, "8.2B", "Q4_K_M"), ollama_tag("gemma3:1b", 10, "1B", "Q4_K_M")]
    with FakeOllama(models, loaded=["qwen3:8b"]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        refused = mm.delete_artifact("ollama", "qwen3:8b", base_url=fake.url)
        assert refused["status"] == "refused" and refused["delete_blockers"] == ["loaded"]
        assert not [r for r in fake.requests if r["method"] == "DELETE"]

        planned = mm.delete_artifact("ollama", "gemma3:1b", base_url=fake.url, dry_run=True)
        assert planned["status"] == "planned" and planned["ok"]
        assert not [r for r in fake.requests if r["method"] == "DELETE"]

        done = mm.delete_artifact("ollama", "gemma3:1b", base_url=fake.url)
        assert done["status"] == "deleted" and done["freed_bytes"] == 10
        forced = mm.delete_artifact("ollama", "qwen3:8b", base_url=fake.url, force=True)
        assert forced["status"] == "deleted"
        unload = [r for r in fake.requests if r["path"] == "/api/generate"]
        assert unload and unload[0]["body"] == {"model": "qwen3:8b", "keep_alive": 0}
        deletes = [r["body"]["model"] for r in fake.requests if r["method"] == "DELETE"]
        assert deletes == ["gemma3:1b", "qwen3:8b"]
        assert fake.models == []


def test_delete_of_a_model_that_is_not_installed_is_not_found(host, monkeypatch):
    with FakeOllama([]) as fake:
        monkeypatch.setenv("OLLAMA_BASE_URL", fake.url)
        out = mm.delete_artifact("ollama", "nope:1b", base_url=fake.url)
    assert out["status"] == "not_found" and not out["ok"]


def test_lmstudio_delete_unloads_then_removes_files_and_hub_entry(host, monkeypatch):
    state = install_fake_lms(
        host["fakebin"], monkeypatch, REAL_SHAPED_LMS_ROWS,
        ps=[{"identifier": "llama-3.2-1b-instruct", "modelKey": "llama-3.2-1b-instruct"}],
    )
    hub = _lmstudio_files(host["lms_models"])
    refused = mm.delete_artifact("lmstudio", "llama-3.2-1b-instruct")
    assert refused["status"] == "refused" and "loaded" in refused["delete_blockers"]

    forced = mm.delete_artifact("lmstudio", "llama-3.2-1b-instruct", force=True)
    assert forced["status"] == "deleted", forced
    assert ["unload", "llama-3.2-1b-instruct"] in lms_calls(state)
    assert not (host["lms_models"] / "mlx-community" / "Llama-3.2-1B-Instruct-4bit").exists()

    hub_delete = mm.delete_artifact("lmstudio", "qwen/qwen3.8-27b@q4_k_m")
    assert hub_delete["status"] == "deleted"
    assert not (host["lms_models"] / "lmstudio-community" / "Qwen3.8-27B-GGUF").exists()
    assert not hub.exists()

    # One quant file inside a multi-quant repo: only that file goes.
    emb = mm.delete_artifact("lmstudio", "text-embedding-qwen3-embedding-0.6b")
    assert emb["status"] == "deleted"
    emb_dir = host["lms_models"] / "Qwen" / "Qwen3-Embedding-0.6B-GGUF"
    assert not (emb_dir / "Qwen3-Embedding-0.6B-Q8_0.gguf").exists()
    assert (emb_dir / "Qwen3-Embedding-0.6B-f16.gguf").exists()


def test_lmstudio_bundled_model_cannot_be_deleted(host, monkeypatch):
    install_fake_lms(host["fakebin"], monkeypatch, REAL_SHAPED_LMS_ROWS)
    out = mm.delete_artifact("lmstudio", "text-embedding-nomic-embed-text-v1.5", force=True)
    assert out["status"] == "refused"
    assert "unknown_location" in out["delete_blockers"]


def test_hf_delete_removes_the_repo_via_the_cache_api(host):
    folder = make_hf_repo(host["hf"], "unsloth/Qwen3-8B-GGUF", {"Qwen3-8B-Q4_K_M.gguf": b"q" * 300})
    planned = mm.delete_artifact("huggingface", "unsloth/Qwen3-8B-GGUF", dry_run=True)
    assert planned["status"] == "planned" and planned["freed_bytes"] == 300
    assert folder.exists()
    done = mm.delete_artifact("huggingface", "unsloth/Qwen3-8B-GGUF")
    assert done["status"] == "deleted", done
    assert not folder.exists()


def test_hf_delete_through_the_other_engine_is_a_shared_cache_refusal(host):
    folder = make_hf_repo(host["hf"], "mlx-community/Qwen3-8B-4bit", {"model.safetensors": b"m" * 10})
    out = mm.delete_artifact("huggingface", "mlx-community/Qwen3-8B-4bit")
    assert out["status"] == "refused"
    assert out["delete_blockers"] == ["shared_cache:mlx,huggingface"]
    assert folder.exists()
    forced = mm.delete_artifact("huggingface", "mlx-community/Qwen3-8B-4bit", force=True)
    assert forced["status"] == "deleted"
    assert not folder.exists()


def test_safe_remove_refuses_paths_outside_a_known_store(host, tmp_path):
    outside = tmp_path / "precious"
    outside.mkdir()
    ok, why = mm._safe_remove(outside, [host["lms_models"]])
    assert not ok and "not inside" in why
    assert outside.exists()
    # The root itself is never removed either.
    ok, _ = mm._safe_remove(host["lms_models"], [host["lms_models"]])
    assert not ok and host["lms_models"].exists()


def test_relay_providers_have_nothing_to_delete(host):
    out = mm.delete_artifact("openai", "gpt-4o")
    assert out["status"] == "not_applicable"


# ---------------------------------------------------------------------------
# HF quant selection (`repo:QUANT`) and the disk pre-check
# ---------------------------------------------------------------------------


def test_repo_colon_quant_names_one_gguf_file_set():
    repo, quant, patterns = mm.hf_artifact_parts("unsloth/Qwen3-8B-GGUF:Q4_K_M")
    assert repo == "unsloth/Qwen3-8B-GGUF" and quant == "Q4_K_M"
    assert "*Q4_K_M*.gguf" in patterns and "*q4_k_m*.gguf" in patterns
    # MLX repos are one quant already: no narrowing.
    assert mm.hf_artifact_parts("mlx-community/Qwen3-8B-4bit") == ("mlx-community/Qwen3-8B-4bit", None, None)
    assert mm._planned_command("huggingface", "unsloth/Qwen3-8B-GGUF:Q4_K_M")[-2:] == ["--include", "*q4_k_m*.gguf"]


def test_probe_needs_the_named_quant_file_not_just_the_repo(host):
    make_hf_repo(host["hf"], "unsloth/Qwen3-8B-GGUF", {"Qwen3-8B-Q8_0.gguf": b"q" * 30})
    miss = mm.probe("huggingface", "unsloth/Qwen3-8B-GGUF:Q4_K_M")
    assert miss.status == mm.PRESENCE_ABSENT and "Q4_K_M" in miss.detail
    hit = mm.probe("huggingface", "unsloth/Qwen3-8B-GGUF:Q8_0")
    assert hit.status == mm.PRESENCE_INSTALLED


def test_hf_download_passes_allow_patterns_and_refuses_when_disk_is_short(host, monkeypatch):
    import types

    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(host["hf"] / "snap")

    fake_hub = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hub)
    monkeypatch.setattr(mm, "_hf_remote_total", lambda repo, patterns, token: (4_000_000, ""))
    out = mm.download("huggingface", "unsloth/Qwen3-8B-GGUF:Q4_K_M")
    assert out.status == "completed", out
    assert calls["allow_patterns"][0] == "*Q4_K_M*.gguf"
    assert calls["repo_id"] == "unsloth/Qwen3-8B-GGUF"

    monkeypatch.setattr(mm, "_hf_remote_total", lambda repo, patterns, token: (10**18, ""))
    calls.clear()
    short = mm.download("huggingface", "unsloth/Qwen3-8B-GGUF:Q5_K_M")
    assert short.status == "failed" and "not enough disk" in short.message
    assert calls == {}  # refused before the first byte


def test_expected_bytes_arms_the_disk_check_for_other_engines(host, monkeypatch):
    monkeypatch.setattr(mm, "probe", lambda p, a, **kw: mm.ModelPresence(p, a, mm.PRESENCE_ABSENT))
    monkeypatch.setitem(mm._DOWNLOADERS, "lmstudio", lambda *a: pytest.fail("downloaded despite no disk"))
    out = mm.download("lmstudio", "qwen/qwen3.5-9b@4bit", expected_bytes=10**18)
    assert out.status == "failed" and "not enough disk" in out.message
    planned = mm.download("lmstudio", "qwen/qwen3.5-9b@4bit", expected_bytes=10**18, dry_run=True)
    assert planned.status == "planned" and "WARNING" in planned.message
