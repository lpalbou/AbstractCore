"""Contract C: the curated download catalog, its schema, presence, fit and hub enrichment.

Offline by construction: engines are unreachable fakes, the HF cache is a
tmp dir, and `hub=True` runs against `FakeHfApi` with an on-disk cache in
tmp_path.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from abstractcore.config import model_catalog as mc
from abstractcore.config.capability_defaults import RECOMMENDED_MODEL_DOWNLOADS
from tests.models_engines_fakes import FakeHfApi, HubRateLimited, isolate_host, make_hf_repo, synthetic_host

ASSETS = Path(mc.__file__).resolve().parent.parent / "assets"


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# The seed and its schema
# ---------------------------------------------------------------------------


def test_seed_is_valid():
    seed = mc.load_seed()
    assert mc.validate_catalog(seed) == []


def test_seed_validation_goes_red_when_a_required_field_is_removed():
    seed = mc.load_seed()
    broken = copy.deepcopy(seed)
    del broken["rows"][0]["artifacts"][0]["provider"]
    assert any("missing 'provider'" in e for e in mc.validate_catalog(broken))
    broken = copy.deepcopy(seed)
    broken["rows"][1]["id"] = broken["rows"][0]["id"]
    assert any("duplicated" in e for e in mc.validate_catalog(broken))
    broken = copy.deepcopy(seed)
    broken["rows"][0]["artifacts"][0]["download_bytes"] = None
    broken["rows"][0]["artifacts"][0]["size_source"] = "catalog"
    assert mc.validate_catalog(broken)


def test_seed_matches_the_json_schema_when_jsonschema_is_available():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ASSETS / "model_downloads_catalog.schema.json").read_text())
    jsonschema.validate(mc.load_seed(), schema)


def test_schema_file_and_validator_agree_on_required_fields():
    """The hand validator mirrors the schema: required lists must match."""

    schema = json.loads((ASSETS / "model_downloads_catalog.schema.json").read_text())
    row_required = set(schema["$defs"]["row"]["required"])
    art_required = set(schema["$defs"]["artifact"]["required"])
    seed = mc.load_seed()
    for field in row_required:
        broken = copy.deepcopy(seed)
        del broken["rows"][0][field]
        assert mc.validate_catalog(broken), f"validator ignores missing row field {field}"
    for field in art_required:
        broken = copy.deepcopy(seed)
        del broken["rows"][0]["artifacts"][0][field]
        assert mc.validate_catalog(broken), f"validator ignores missing artifact field {field}"


def test_seed_covers_at_least_forty_families_across_the_required_kinds():
    rows = mc.load_seed()["rows"]
    families = {r["family"] for r in rows}
    assert len(families) >= 40
    tags = {t for r in rows for t in r["tags"]}
    assert {"chat", "coding", "vision", "embedding"} <= tags
    vendors = {r["vendor"] for r in rows}
    assert {"Qwen", "Meta", "Google", "Mistral AI", "DeepSeek", "OpenAI", "Microsoft", "Hugging Face", "Nomic AI", "BAAI"} <= vendors


def test_every_recommended_download_is_a_starter_row():
    arts = {(a["provider"], a["artifact"]): r for r in mc.load_seed()["rows"] for a in r["artifacts"]}
    for spec in RECOMMENDED_MODEL_DOWNLOADS.values():
        row = arts.get((spec["provider"], spec["artifact"]))
        assert row is not None, spec
        assert row["starter"] is True
    qwen = arts[("lmstudio", "qwen/qwen3.5-9b@4bit")]
    assert qwen["id"] == "qwen3.5-9b"


def test_catalog_id_lookup_is_tolerant_like_presence():
    assert mc.catalog_id_for("lmstudio", "qwen/qwen3.5-9b") == "qwen3.5-9b"
    assert mc.catalog_id_for("ollama", "gemma3:1b:latest") == "gemma-3-1b"
    assert mc.catalog_id_for("huggingface", "unsloth/Qwen3-8B-GGUF") == "qwen3-8b"
    assert mc.catalog_id_for("ollama", "no-such:model") is None


# ---------------------------------------------------------------------------
# catalog(): shape, presence, fit, pre-selection, filters
# ---------------------------------------------------------------------------

_ROW_KEYS = {"id", "family", "display_name", "vendor", "params_total", "params_active", "license", "capabilities", "source", "tags", "artifacts"}
_ART_KEYS = {"provider", "artifact", "quant", "bits", "quant_class", "quant_class_source", "companions", "companion_bytes", "note", "options", "download_bytes", "size_source", "presence", "fit", "downloadable", "recommended"}
_FIT_KEYS = {"verdict", "need_bytes", "ceiling_bytes", "free_now_bytes", "fits_now", "disk_ok", "max_context", "confidence", "notes"}


def test_catalog_payload_follows_contract_c(host):
    payload = mc.catalog(host=synthetic_host("metal128"))
    assert payload["schema"] == "model_catalog_v1"
    assert payload["host_profile"]["accelerator"] == "metal"
    assert payload["hub"] is None
    assert payload["counts"]["rows"] == len(payload["rows"]) >= 70
    for row in payload["rows"]:
        assert _ROW_KEYS <= set(row), row["id"]
        assert row["source"] == "curated"
        assert {"text", "vision", "audio", "tools", "thinking", "max_tokens", "embedding"} <= set(row["capabilities"])
        assert sum(1 for a in row["artifacts"] if a["recommended"]) <= 1
        for art in row["artifacts"]:
            assert _ART_KEYS <= set(art)
            assert _FIT_KEYS <= set(art["fit"])
            assert art["presence"]["status"] in {"installed", "absent", "unknown", "not_applicable"}
            assert art["fit"]["verdict"] in {"fits", "tight", "too_large", "partial_offload", "unknown"}
            assert art["size_source"] in {"hf_api", "catalog", "engine", "estimate", "unknown"}
            assert art["quant_class"] in mc.QUANT_CLASSES


def test_capabilities_are_joined_from_the_registry(host):
    rows = {r["id"]: r for r in mc.catalog(host=synthetic_host("metal128"))["rows"]}
    assert rows["qwen3-8b"]["capabilities"]["tools"] == "native"
    assert rows["qwen3-8b"]["capabilities"]["thinking"] is True
    assert rows["qwen3-8b"]["capabilities"]["max_tokens"] == 131072
    assert rows["qwen3-vl-8b"]["capabilities"]["vision"] is True
    assert rows["nomic-embed-text-v1.5"]["capabilities"]["embedding"] is True
    assert rows["nomic-embed-text-v1.5"]["capabilities"]["text"] is False


def test_the_starter_text_model_is_preselected_on_apple_silicon(host):
    """On a Mac the MLX lane is pre-selected and the starter is the memory tier."""

    rows = {r["id"]: r for r in mc.catalog(host=synthetic_host("metal128"))["rows"]}
    nine = rows["qwen3.5-9b"]
    assert [a["artifact"] for a in nine["artifacts"] if a["recommended"]] == [mc._tier_artifact(mc.APPLE_TEXT_TIERS[0])]
    assert nine["starter"] is False  # 128 GiB is the Flash-Next tier
    assert rows["qwen3.8-flash-next"]["starter"] is True
    assert [r["id"] for r in rows.values() if r["starter"] and "chat" in r["tags"]] == ["qwen3.8-flash-next"]
    # LM Studio / Ollama builds stay listed and downloadable, just not pre-selected.
    lms = next(a for a in nine["artifacts"] if a["provider"] == "lmstudio")
    assert lms["recommended"] is False and lms["supported_on_host"] is True


def test_the_portable_starter_is_preselected_off_apple_silicon(host):
    row = next(r for r in mc.catalog(host=synthetic_host("cuda24"))["rows"] if r["id"] == "qwen3.5-9b")
    assert [a["artifact"] for a in row["artifacts"] if a["recommended"]] == ["qwen/qwen3.5-9b@4bit"]
    assert row["starter"] is True


def _no_local_engines(monkeypatch):
    """engine_inventory() reads THIS interpreter (llama-cpp-python, mlx and
    transformers importable in the dev venv): a synthetic host must not
    inherit it."""

    from abstractcore.config import engines

    real = engines.engine_inventory

    def inventory(*args, **kwargs):
        out = real(*args, **kwargs)
        for e in out["engines"]:
            e["installed"] = False
        return out

    monkeypatch.setattr(engines, "engine_inventory", inventory)


def test_mlx_artifacts_are_not_downloadable_off_apple_silicon(host, monkeypatch):
    _no_local_engines(monkeypatch)
    rows = mc.catalog(host=synthetic_host("cuda24"))["rows"]
    mlx = [a for r in rows for a in r["artifacts"] if a["provider"] == "mlx"]
    assert mlx and all(not a["downloadable"] and not a["supported_on_host"] for a in mlx)
    qwen8 = next(r for r in rows if r["id"] == "qwen3-8b")
    assert next(a for a in qwen8["artifacts"] if a["recommended"])["provider"] == "ollama"


def test_an_installed_engine_outranks_the_provider_order(host, monkeypatch):
    """The documented rule that made the test above environment-dependent:
    with llama-cpp-python importable, the GGUF (llamacpp engine) wins."""

    from abstractcore.config import engines

    real = engines.engine_inventory

    def inventory(*args, **kwargs):
        out = real(*args, **kwargs)
        for e in out["engines"]:
            e["installed"] = e["id"] == "llamacpp"
        return out

    monkeypatch.setattr(engines, "engine_inventory", inventory)
    qwen8 = next(r for r in mc.catalog(host=synthetic_host("cuda24"))["rows"] if r["id"] == "qwen3-8b")
    assert next(a for a in qwen8["artifacts"] if a["recommended"])["engine"] == "llamacpp"


def test_presence_reflects_the_hf_cache(host):
    make_hf_repo(host["hf"], "unsloth/Qwen3-4B-GGUF", {"Qwen3-4B-Q4_K_M.gguf": b"x" * 64})
    row = next(r for r in mc.catalog(host=synthetic_host("metal128"))["rows"] if r["id"] == "qwen3-4b")
    art = next(a for a in row["artifacts"] if a["provider"] == "huggingface")
    assert art["presence"]["status"] == "installed"
    assert art["size_source"] == "catalog"  # the seed's observed size, not the tiny fake file
    assert "already installed" in art["fit"]["notes"]


def test_fit_verdicts_differ_by_machine(host):
    def verdict(kind, row_id, provider):
        rows = mc.catalog(host=synthetic_host(kind))["rows"]
        row = next(r for r in rows if r["id"] == row_id)
        return next(a for a in row["artifacts"] if a["provider"] == provider)["fit"]["verdict"]

    assert verdict("metal128", "llama-3.3-70b", "ollama") in ("fits", "tight")
    assert verdict("cuda24", "llama-3.3-70b", "ollama") == "partial_offload"
    assert verdict("cpu16", "llama-3.3-70b", "ollama") == "too_large"


def test_query_engine_fits_and_tag_filters(host):
    q = mc.catalog("qwen3 8b", host=synthetic_host("metal128"))
    ids = {r["id"] for r in q["rows"]}
    assert "qwen3-8b" in ids and "qwen3.5-0.8b" not in ids

    only_ollama = mc.catalog(engine="ollama", host=synthetic_host("metal128"))
    assert all(a["provider"] == "ollama" for r in only_ollama["rows"] for a in r["artifacts"])

    llamacpp = mc.catalog(engine="llamacpp", host=synthetic_host("metal128"))
    assert llamacpp["rows"] and all(a["engine"] == "llamacpp" for r in llamacpp["rows"] for a in r["artifacts"])

    small = mc.catalog(fits=True, host=synthetic_host("cpu16"))
    assert small["rows"]
    assert all(a["fit"]["verdict"] in ("fits", "tight") for r in small["rows"] for a in r["artifacts"])
    assert "llama-3.3-70b" not in {r["id"] for r in small["rows"]}

    emb = mc.catalog(tags=["embedding"], host=synthetic_host("metal128"))
    assert emb["rows"] and all("embedding" in r["tags"] for r in emb["rows"])


# ---------------------------------------------------------------------------
# hub=True: enrichment, search, cache, 429, offline
# ---------------------------------------------------------------------------


def test_hub_enrichment_uses_exact_file_sizes_for_the_named_quant(host, tmp_path):
    api = FakeHfApi(
        repos={
            "unsloth/Qwen3-8B-GGUF": {
                "files": {"Qwen3-8B-Q4_K_M.gguf": 5_027_783_488, "Qwen3-8B-Q8_0.gguf": 8_709_519_168, "README.md": 10},
                "params": 8_190_735_360,
            }
        }
    )
    cache = tmp_path / "hub.json"
    payload = mc.catalog("qwen3 8b", hub=True, hf_api=api, hub_cache=cache, host=synthetic_host("metal128"))
    row = next(r for r in payload["rows"] if r["id"] == "qwen3-8b")
    gguf = next(a for a in row["artifacts"] if a["provider"] == "huggingface")
    assert gguf["download_bytes"] == 5_027_783_488
    assert gguf["size_source"] == "hf_api"
    assert payload["hub"]["ok"] is True
    assert cache.exists()

    # Second call inside the TTL: served from the on-disk cache, no API call.
    before = [c for c in api.calls if c[0] == "model_info"]
    mc.catalog("qwen3 8b", hub=True, hf_api=api, hub_cache=cache, host=synthetic_host("metal128"))
    after = [c for c in api.calls if c[0] == "model_info"]
    assert len(after) == len(before)


def test_hub_search_adds_hf_rows_with_a_picked_quant(host, tmp_path):
    api = FakeHfApi(
        repos={"someone/Cool-7B-GGUF": {"files": {"Cool-7B-Q4_K_M.gguf": 4_000_000_000, "Cool-7B-Q8_0.gguf": 7_000_000_000}}},
        search={"gguf:cool": ["someone/Cool-7B-GGUF"], "mlx:cool": []},
    )
    payload = mc.catalog("cool", hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("metal128"))
    hf_rows = [r for r in payload["rows"] if r["source"] == "hf_search"]
    assert len(hf_rows) == 1
    art = hf_rows[0]["artifacts"][0]
    assert art["artifact"] == "someone/Cool-7B-GGUF:Q4_K_M"
    assert art["download_bytes"] == 4_000_000_000 and art["size_source"] == "hf_api"
    assert hf_rows[0]["id"] == "hf:someone/cool-7b-gguf"


def test_rate_limit_degrades_to_the_offline_answer(host, tmp_path):
    api = FakeHfApi()
    api.fail_with = HubRateLimited()
    payload = mc.catalog("qwen3 8b", hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("metal128"))
    assert payload["hub"]["ok"] is False
    assert any("429" in e for e in payload["hub"]["errors"])
    assert any(r["id"] == "qwen3-8b" for r in payload["rows"])  # curated rows still there
    # One 429 stops the lane: no hammering.
    assert len(api.calls) == 1


def test_offline_hub_is_reported_not_raised(host, tmp_path):
    api = FakeHfApi()
    api.fail_with = OSError("network is unreachable")
    payload = mc.catalog("qwen3 8b", hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("metal128"))
    assert payload["hub"]["ok"] is False
    assert payload["rows"]
    # Without a query (enrichment only) every lookup fails -> still not ok.
    browse = mc.catalog(hub=True, hf_api=api, hub_cache=tmp_path / "hub2.json", host=synthetic_host("metal128"))
    assert browse["hub"]["ok"] is False and browse["hub"]["errors"]
