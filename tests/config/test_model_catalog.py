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
from abstractcore.utils.model_fit import FIT_VERDICTS
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
    qwen = arts[("lmstudio", "qwen/qwen3.5-9b@q4_k_m")]
    assert qwen["id"] == "qwen3.5-9b"


def test_every_voice_artifact_carries_its_download_size():
    """A download prompt must never say "size not published" for a voice model:
    every artifact of a voice row (TTS or STT) carries a real byte count with
    `size_source: catalog` and a note saying where it was read."""

    seed = mc.load_seed()
    voice = [
        (row["id"], art)
        for row in seed["rows"]
        for art in row["artifacts"]
        if (row.get("capabilities_override") or {}).get("voice") or "voice" in (row.get("tags") or [])
        or art["provider"] in {"supertonic", "faster-whisper", "piper"}
    ]
    assert voice, "the catalog has voice rows (supertonic-3 at least)"
    for row_id, art in voice:
        assert isinstance(art.get("download_bytes"), int) and art["download_bytes"] > 0, row_id
        assert art.get("size_source") == "catalog" and art.get("note"), row_id
    supertonic = next(art for row_id, art in voice if art["artifact"] == "supertonic-3")
    assert supertonic["download_bytes"] == 401276744


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
            assert art["fit"]["verdict"] in set(FIT_VERDICTS)
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


def test_the_portable_starter_is_preselected_off_apple_silicon(host, monkeypatch):
    _no_local_engines(monkeypatch)  # with no engine installed the curated LM Studio build leads
    row = next(r for r in mc.catalog(host=synthetic_host("cuda24"))["rows"] if r["id"] == "qwen3.5-9b")
    assert [a["artifact"] for a in row["artifacts"] if a["recommended"]] == ["qwen/qwen3.5-9b@q4_k_m"]
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


def _only_engines(monkeypatch, *installed):
    from abstractcore.config import engines

    real = engines.engine_inventory

    def inventory(*args, **kwargs):
        out = real(*args, **kwargs)
        for e in out["engines"]:
            e["installed"] = e["id"] in installed
        return out

    monkeypatch.setattr(engines, "engine_inventory", inventory)


def test_a_host_with_only_llamacpp_is_offered_the_gguf_first(host, monkeypatch):
    """Backlog 0863 (the OVH server): llama.cpp Ready, LM Studio and Ollama absent. The row's
    pre-selection and first artifact is the HF GGUF llama.cpp serves, not the curated LM Studio
    build; the absent engines' builds stay listed after it."""
    _only_engines(monkeypatch, "llamacpp")
    rows = {r["id"]: r for r in mc.catalog(host=synthetic_host("cuda24"))["rows"]}
    arts = rows["qwen3.5-9b"]["artifacts"]
    first = arts[0]
    assert (first["provider"], first["artifact"], first["engine"]) == (
        "huggingface", "unsloth/Qwen3.5-9B-GGUF:Q4_K_M", "llamacpp"
    )
    assert first["recommended"] is True and first["engine_installed"] is True
    assert [a["provider"] for a in arts if a["recommended"]] == ["huggingface"]
    assert {"lmstudio", "ollama"} <= {a["provider"] for a in arts[1:]}
    assert not any(a["engine_installed"] for a in arts if a["provider"] in ("lmstudio", "ollama"))
    # Installed-engine artifacts are listed before any artifact of an absent engine, on every row.
    for row in rows.values():
        flags = [a["engine_installed"] for a in row["artifacts"] if not a["recommended"]]
        assert flags == sorted(flags, reverse=True), row["id"]


def test_every_curated_text_row_has_a_build_llamacpp_serves():
    """Backlog 0863: no curated chat row offers only engines a llama.cpp-only host lacks.
    (Qwen3.8 Flash-Next: the hub has no plain GGUF quant, only its MTP drafter files.)"""
    missing = []
    for r in mc.load_seed()["rows"]:
        if "chat" not in (r.get("tags") or []) and "coding" not in (r.get("tags") or []):
            continue
        provs = {a["provider"] for a in r["artifacts"]}
        if not provs & {"lmstudio", "ollama", "mlx"}:
            continue
        if not any(mc._engine_for(a["provider"], a.get("quant")) == "llamacpp" for a in r["artifacts"]):
            missing.append(r["id"])
    assert missing == ["qwen3.8-flash-next"]


def test_mlx_is_never_offered_off_apple_silicon(host, monkeypatch):
    """Operator 2026-10-29: "on Linux never offer MLX" — no mlx / mlx-gen / mlx-vlm artifact off Apple
    silicon (they were listed, not downloadable), and a row with only MLX builds is not listed."""
    _no_local_engines(monkeypatch)
    apple = mc.catalog(host=synthetic_host("metal128"))["rows"]
    mlx_only = {r["id"] for r in apple if r["artifacts"] and all(mc._engine_for(a["provider"], a.get("quant")) == "mlx" for a in r["artifacts"])}
    assert any(mc._engine_for(a["provider"], a.get("quant")) == "mlx" for r in apple for a in r["artifacts"])
    rows = mc.catalog(host=synthetic_host("cuda24"))["rows"]
    assert not [a for r in rows for a in r["artifacts"] if mc._engine_for(a["provider"], a.get("quant")) == "mlx"]
    assert mlx_only and not mlx_only & {r["id"] for r in rows}
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


def test_a_hub_gguf_this_llamacpp_cannot_load_is_flagged_before_download(host, tmp_path, monkeypatch):
    """Backlog 0868 (OVH: XHToken/Spark-X2.6-4B-GGUF downloaded, then failed at Test: `spark2_5`). The
    Hub's `gguf.architecture` is checked against this computer's llama.cpp before anything downloads."""
    from abstractcore.utils import model_cache

    monkeypatch.setattr(model_cache, "llamacpp_knows_architecture", lambda arch: {"spark2_5": False, "qwen3": True}.get(arch))
    api = FakeHfApi(
        repos={
            "XHToken/Spark-X2.6-4B-GGUF": {"files": {"Spark-X2.6-4B-Q4_K_M.gguf": 2_500_000_000}, "architecture": "spark2_5"},
            "someone/Cool-4B-GGUF": {"files": {"Cool-4B-Q4_K_M.gguf": 2_400_000_000}, "architecture": "qwen3"},
            "other/Plain-4B-GGUF": {"files": {"Plain-4B-Q4_K_M.gguf": 2_400_000_000}},
        },
        search={"gguf:4b": ["XHToken/Spark-X2.6-4B-GGUF", "someone/Cool-4B-GGUF", "other/Plain-4B-GGUF"], "mlx:4b": []},
    )
    payload = mc.catalog("4b", hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("cuda24"))
    arts = {r["artifacts"][0]["artifact"].split(":")[0]: r["artifacts"][0] for r in payload["rows"] if r["source"] == "hf_search"}
    spark = arts["XHToken/Spark-X2.6-4B-GGUF"]
    assert spark["gguf_architecture"] == "spark2_5" and spark["loadable"] is False
    assert spark["unloadable_reason"].startswith("This computer's llama.cpp (llama-cpp-python")
    assert "cannot load the spark2_5 architecture" in spark["unloadable_reason"]
    assert spark["downloadable"] is False and spark["recommended"] is False
    assert spark["unloadable_reason"] in spark["fit"]["notes"]
    cool = arts["someone/Cool-4B-GGUF"]
    assert cool["loadable"] is True and cool["unloadable_reason"] is None and cool["downloadable"] is True
    # The Hub did not say (or no llama-cpp-python): nothing is claimed.
    plain = arts["other/Plain-4B-GGUF"]
    assert plain["gguf_architecture"] is None and plain["loadable"] is None and plain["downloadable"] is True
    # Unknown to the check (None): offered, never refused.
    monkeypatch.setattr(model_cache, "llamacpp_knows_architecture", lambda arch: None)
    again = mc.catalog("4b", hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("cuda24"))
    spark2 = next(r["artifacts"][0] for r in again["rows"] if r["id"] == "hf:xhtoken/spark-x2.6-4b-gguf")
    assert spark2["loadable"] is None and spark2["downloadable"] is True


def test_hub_search_pages_and_says_how_many_and_whether_more(host, tmp_path):
    """Backlog 0864: Hugging Face results come a page at a time, and `hub.search` says how many
    rows the page added and whether another page exists."""
    repos = {f"someone/Model{i:02d}-GGUF": {"files": {f"Model{i:02d}-Q4_K_M.gguf": 1_000_000_000}} for i in range(45)}
    api = FakeHfApi(repos=repos, search={"gguf:model": sorted(repos)})
    kw = dict(hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("cuda24"))
    pages = []
    for page in (1, 2, 3):
        payload = mc.catalog("model", hub_page=page, **kw)
        ids = [r["id"] for r in payload["rows"] if r["source"] == "hf_search"]
        pages.append(ids)
        search = payload["hub"]["search"]
        assert search["page"] == page and search["page_size"] == 20 and search["rows"] == len(ids)
        assert search["has_more"] is (page < 3)
    assert [len(p) for p in pages] == [20, 20, 5]
    assert len(set(pages[0]) | set(pages[1]) | set(pages[2])) == 45  # no repeats across pages
    assert pages[1][0] == "hf:someone/model20-gguf"


def test_an_exact_handle_finds_the_repo_and_its_builds(host, tmp_path):
    """Backlog 0864: `someone/Cool-7B` lists that repo first (a Transformers checkpoint, fetched
    whole) and the GGUF builds the hub has for its name, `Cool-7B`."""
    api = FakeHfApi(
        repos={
            "someone/Cool-7B": {"files": {"model-00001-of-00002.safetensors": 8_000_000_000, "config.json": 1_000}},
            "other/Cool-7B-GGUF": {"files": {"Cool-7B-Q4_K_M.gguf": 4_000_000_000}},
        },
        search={"gguf:Cool-7B": ["other/Cool-7B-GGUF"]},
    )
    payload = mc.catalog("someone/Cool-7B", hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("cuda24"))
    hf_rows = [r for r in payload["rows"] if r["source"] == "hf_search"]
    assert [r["id"] for r in hf_rows] == ["hf:someone/cool-7b", "hf:other/cool-7b-gguf"]
    exact = hf_rows[0]
    assert exact["exact"] is True and hf_rows[1]["exact"] is False
    assert (exact["artifacts"][0]["provider"], exact["artifacts"][0]["artifact"]) == ("huggingface", "someone/Cool-7B")
    assert exact["artifacts"][0]["engine"] == "huggingface"
    assert payload["hub"]["search"]["handle"] == "someone/Cool-7B" and payload["hub"]["search"]["handle_found"] is True
    assert ("list_models", "Cool-7B", "gguf") in api.calls  # the name, not the handle


def test_a_handle_that_does_not_exist_still_searches_its_name(host, tmp_path):
    api = FakeHfApi(repos={"other/Cool-7B-GGUF": {"files": {"Cool-7B-Q4_K_M.gguf": 4_000_000_000}}},
                    search={"gguf:Cool-7B": ["other/Cool-7B-GGUF"]})
    payload = mc.catalog("nobody/Cool-7B", hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("cuda24"))
    assert [r["id"] for r in payload["rows"] if r["source"] == "hf_search"] == ["hf:other/cool-7b-gguf"]
    assert payload["hub"]["search"]["handle_found"] is False


def test_a_handle_or_hyphenated_name_finds_the_curated_row(host):
    """Backlog 0864: the query splits like the row, so "Qwen/Qwen3.5-9B" and "qwen3.5-9B" find
    qwen3.5-9b (the console's Hugging Face mode asks the catalog with the typed text)."""
    for q in ("Qwen/Qwen3.5-9B", "qwen3.5-9B", "qwen3.5 9b"):
        assert "qwen3.5-9b" in [r["id"] for r in mc.catalog(q, host=synthetic_host("cuda24"))["rows"]], q


def test_extended_lists_every_quant_of_the_rows_own_gguf_repos(host, tmp_path):
    """Batch item (b), "All" quants: with `extended`, a row gains every quant its own GGUF repo
    ships (exact size from the file list, its fit), marked `extended`, never the pre-selection;
    a label whose download pattern would fetch other files too (Q6_K also matches UD-Q6_K_XL;
    any Q4_K_M rides an MTP drafter's) is left out; mmproj is not a quant."""
    repo = "unsloth/Qwen3-8B-GGUF"
    files = {
        "Qwen3-8B-Q4_K_M.gguf": 5_000_000_000, "Qwen3-8B-Q8_0.gguf": 8_700_000_000,
        "Qwen3-8B-UD-IQ1_S.gguf": 2_275_000_000, "Qwen3-8B-Q3_K_M.gguf": 4_100_000_000,
        "Qwen3-8B-Q6_K.gguf": 6_700_000_000, "Qwen3-8B-UD-Q6_K_XL.gguf": 7_400_000_000,
        "BF16/Qwen3-8B-BF16-00001-of-00002.gguf": 8_000_000_000, "BF16/Qwen3-8B-BF16-00002-of-00002.gguf": 8_300_000_000,
        "mmproj-F16.gguf": 600_000_000,
    }
    api = FakeHfApi(repos={repo: {"files": files}})
    kw = dict(hub=True, hf_api=api, hub_cache=tmp_path / "hub.json", host=synthetic_host("cuda24"))
    row = next(r for r in mc.catalog("qwen3 8b", extended=True, **kw)["rows"] if r["id"] == "qwen3-8b")
    ext = {a["artifact"].split(":", 1)[1]: a for a in row["artifacts"] if a.get("extended")}
    assert sorted(ext) == ["BF16", "Q3_K_M", "UD-IQ1_S", "UD-Q6_K_XL"]
    assert ext["BF16"]["download_bytes"] == 16_300_000_000 and ext["Q3_K_M"]["download_bytes"] == 4_100_000_000
    assert all(a["provider"] == "huggingface" and a["engine"] == "llamacpp" and a["fit"]["verdict"] for a in ext.values())
    assert not any(a["recommended"] for a in ext.values())
    plain = next(r for r in mc.catalog("qwen3 8b", **kw)["rows"] if r["id"] == "qwen3-8b")
    assert not any(a.get("extended") for a in plain["artifacts"])


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


def test_the_lfm_families_are_in_the_catalog_with_gguf_and_mlx(host):
    """Operator QA 2026-10-28: searching "lfm" showed one row (LFM2.5 2.6B, MLX only). Both Liquid AI families,
    the 24B-A2B models included, are listed with a GGUF (llama.cpp) and an MLX artifact each, every artifact
    with its verified size; repo ids are aliases of their capabilities entries."""
    rows = {r["id"]: r for r in mc.catalog("lfm", host=synthetic_host("metal128"))["rows"]}
    for family, ids in {
        "lfm2": ["lfm2-350m", "lfm2-700m", "lfm2-1.2b", "lfm2-2.6b", "lfm2-8b-a1b", "lfm2-24b-a2b"],
        "lfm2.5": ["lfm2.5-350m", "lfm2.5-1.2b-instruct", "lfm2.5-1.2b-thinking", "lfm2.5-2.6b", "lfm2.5-8b-a1b",
                   "lfm2.5-24b-a2b"],
    }.items():
        for rid in ids:
            assert rid in rows, f"search 'lfm' misses {rid}"
            row = rows[rid]
            assert row["family"] == family
            seed = next(r for r in mc.load_seed()["rows"] if r["id"] == rid)
            providers = {a["provider"] for a in seed["artifacts"]}
            assert {"huggingface", "mlx"} <= providers, (rid, providers)
            assert any(a["artifact"].endswith((":Q4_K_M", ":Q8_0")) for a in seed["artifacts"] if a["provider"] == "huggingface")
            # Every artifact carries its size: upstream-verified (hf_api) or seen on a real engine (verified).
            assert all(a["download_bytes"] and (a.get("verified") or a.get("upstream", {}).get("method") == "hf_api")
                       for a in seed["artifacts"]), rid
    caps = json.loads((Path(mc.__file__).parent.parent / "assets" / "model_capabilities.json").read_text())["models"]
    assert caps["lfm2.5-24b-a2b"]["max_tokens"] == 131072
    assert "LiquidAI/LFM2.5-24B-A2B-GGUF" in caps["lfm2.5-24b-a2b"]["aliases"]
    assert "mlx-community/LFM2-24B-A2B-4bit" in caps["lfm2-24b-a2b"]["aliases"]


def test_no_liquidai_q4_k_m_gguf_names_a_repo_with_a_hip_optimized_twin():
    """LiquidAI's LFM2 GGUF repos ship `-Q4_K_M.gguf` AND `-Q4_K_M-hip-optimized.gguf`: the `*Q4_K_M*.gguf`
    download pattern would fetch both and the loader could pick either (backlog 0860). Their Q4_K_M comes
    from unsloth; LiquidAI's LFM2.5 repos have no such twin."""
    twins = {"LiquidAI/LFM2-350M-GGUF", "LiquidAI/LFM2-700M-GGUF", "LiquidAI/LFM2-1.2B-GGUF", "LiquidAI/LFM2-2.6B-GGUF"}
    bad = [a["artifact"] for r in mc.load_seed()["rows"] for a in r["artifacts"]
           if a["provider"] == "huggingface" and a["artifact"].split(":")[0] in twins and "Q4_K_M" in a["artifact"].upper()]
    assert bad == []


def test_an_extended_quant_is_never_the_preselection():
    def art(name, verdict, **kw):
        return {"provider": "huggingface", "artifact": name, "engine": "llamacpp", "supported_on_host": True,
                "downloadable": True, "fit": {"verdict": verdict}, "presence": {"status": "absent"}, **kw}
    row = {"id": "r", "artifacts": [art("r:Q4_K_M", "too_large"), art("r:UD-IQ1_S", "fits", extended=True)]}
    mc._pick_recommended(row, "cuda", {"llamacpp": True})
    assert [a["artifact"] for a in row["artifacts"] if a["recommended"]] == ["r:Q4_K_M"]
