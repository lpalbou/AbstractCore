"""The recommended model for every capability x every machine class.

`recommendations.recommended_models(host)` is the per-host answer and
`recommendation_matrix()` the export (`abstractcore models recommendations`)
the docs page and the website render. These tests pin:

  - completeness: every class x capability is decided (recommended, covered,
    or unavailable WITH a reason) and carries the facts a table needs;
  - one source: the text cells are `recommended_text_model()`'s picks, the
    starter cells are exactly what the writers download
    (`recommended_model_downloads`), and the starter set itself is unchanged;
  - honesty: the platform and memory facts (MLX-only engines, CTranslate2 and
    PyTorch builds, the video/music memory gate, the 128 GB sysctl);
  - the generated docs block is current;
  - the CLI.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.config import recommendations as rec
from tests.models_engines_fakes import synthetic_host

ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_IDS = ["text", "vision", "speech_output", "speech_input", "image", "video", "music"]
STATUSES = {"recommended", "covered", "unavailable"}


@pytest.fixture(scope="module")
def matrix():
    return rec.recommendation_matrix()


def _cls(matrix, class_id):
    return next(c for c in matrix["classes"] if c["id"] == class_id)


def _apple_class_for(matrix, gib):
    return next(c for c in matrix["classes"] if c["family"] == "apple_silicon" and gib in c["memory_gib"])


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_the_matrix_covers_every_capability_and_machine_class(matrix):
    assert matrix["schema"] == "model_recommendations_v1"
    assert [c["id"] for c in matrix["capabilities"]] == CAPABILITY_IDS
    families = [c["family"] for c in matrix["classes"]]
    assert families[-3:] == ["nvidia", "cpu", "intel_mac"]
    # Every Apple memory size lands in exactly one band, in order.
    apple = [c for c in matrix["classes"] if c["family"] == "apple_silicon"]
    assert [g for c in apple for g in c["memory_gib"]] == list(rec.APPLE_MEMORY_SIZES_GIB)
    for a, b in zip(apple, apple[1:]):
        assert a["memory_gib_below"] == b["memory_gib_min"]
    assert apple[-1]["memory_gib_below"] is None
    for cls in matrix["classes"]:
        assert list(cls["entries"]) == CAPABILITY_IDS, cls["id"]


def test_every_cell_is_decided_with_the_facts_a_table_needs(matrix):
    for cls, cid, e in rec.iter_entries(matrix):
        where = f"{cls['id']}/{cid}"
        assert e["status"] in STATUSES, where
        if e["status"] == "unavailable":
            assert isinstance(e["reason"], str) and len(e["reason"]) > 20, where
            continue
        assert e["reason"] is None, where
        for key in ("provider", "engine", "device", "model", "artifact", "download_provider", "catalog_id", "fit"):
            assert e[key], f"{where}: {key}"
        assert isinstance(e["memory_need_bytes"], int) and e["memory_need_bytes"] > 0, where
        assert e["memory_need_source"] in ("measured", "estimated"), where
        # A size is exact (catalog) or honestly absent; never invented.
        assert e["download_bytes"] is None or e["download_bytes"] > 0, where
        if e["fit"] == "needs_gpu_limit":
            assert e["gpu_limit_command"].startswith("sudo sysctl iogpu.wired_limit_mb="), where
        else:
            assert e["gpu_limit_command"] is None, where


def test_image_input_is_covered_by_the_text_model_everywhere(matrix):
    for cls in matrix["classes"]:
        vision, text = cls["entries"]["vision"], cls["entries"]["text"]
        assert vision["status"] == "covered" and vision["covered_by"] == "text", cls["id"]
        assert (vision["provider"], vision["model"]) == (text["provider"], text["model"])


# ---------------------------------------------------------------------------
# One source of truth
# ---------------------------------------------------------------------------

# The Apple silicon text tiers every surface showed before this matrix existed.
TEXT_BY_APPLE_GIB = {
    **{g: "mlx-community/Qwen3.5-9B-MLX-4bit" for g in (8, 16, 18)},
    **{g: "mlx-community/Qwen3.8-27B-4bit" for g in (24, 32, 36, 48, 64, 96)},
    **{g: "mlx-community/Qwen3.8-Flash-Next-4bit" for g in (128, 192, 256, 512)},
}
TEXT_BY_CLASS = {
    "nvidia": ("lmstudio", "qwen/qwen3.5-9b", "qwen/qwen3.5-9b@4bit"),
    "cpu": ("lmstudio", "qwen/qwen3.5-9b", "qwen/qwen3.5-9b@4bit"),
    "intel_mac": ("ollama", "qwen3.5:9b", "qwen3.5:9b"),
}


def test_text_cells_are_the_recommended_text_model(matrix):
    for gib, artifact in TEXT_BY_APPLE_GIB.items():
        e = _apple_class_for(matrix, gib)["entries"]["text"]
        pick = mc.recommended_text_model(rec._apple_host(gib))
        assert (e["provider"], e["artifact"]) == ("mlx", artifact) == (pick["provider"], pick["artifact"]), gib
        assert e["fit"] == pick["fit"]["verdict"], gib
        assert e["warning"] == pick["warning"], gib
    for class_id, (provider, model, artifact) in TEXT_BY_CLASS.items():
        e = _cls(matrix, class_id)["entries"]["text"]
        assert (e["provider"], e["model"], e["artifact"]) == (provider, model, artifact), class_id


def test_the_starter_set_the_writers_use_is_unchanged():
    # What the fresh-install seed, apply-recommended and download --recommended act on.
    assert {k: (r.provider, r.model) for k, r in cd.RECOMMENDED_CAPABILITY_DEFAULT_ROUTES.items()} == {
        "input.text": ("lmstudio", "qwen/qwen3.5-9b"),
        "output.voice": ("supertonic", "supertonic-3"),
        "output.image": ("mlx-gen", "AbstractFramework/flux.2-klein-4b-8bit"),
        "output.video": ("mlx-gen", "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"),
    }
    assert cd.RECOMMENDED_MODEL_DOWNLOADS == {
        "input.text": {"provider": "lmstudio", "artifact": "qwen/qwen3.5-9b@4bit"},
        "output.voice": {"provider": "supertonic", "artifact": "supertonic-3"},
        "output.image": {"provider": "mlx-gen", "artifact": "AbstractFramework/flux.2-klein-4b-8bit"},
        "output.video": {"provider": "mlx-gen", "artifact": "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"},
    }
    # ...and they are views of the one table, never a second copy.
    assert {k for k, r in cd.RECOMMENDED_MODELS.items() if r.starter} == set(cd.RECOMMENDED_MODEL_DOWNLOADS)
    assert {k for k, r in cd.RECOMMENDED_MODELS.items() if not r.starter} == {"input.voice", "output.music"}


def _hosts(matrix):
    for cls in matrix["classes"]:
        if cls["family"] == "apple_silicon":
            for gib in cls["memory_gib"]:
                yield cls, rec._apple_host(gib)
        else:
            spec = next(s for s in rec.MACHINE_CLASSES if s["id"] == cls["id"])
            yield cls, dict(spec["host"])


def test_starter_cells_are_exactly_what_the_writers_download(matrix):
    """The matrix and `recommended_model_downloads` (seed, apply, download
    --recommended, console tiles) agree on every host, cell by cell."""

    route_of = {cid: route for cid, route, _l, _t in rec.RECOMMENDATION_CAPABILITIES}
    for cls, host in _hosts(matrix):
        entries = rec.recommended_models(host)
        from_matrix = {
            route_of[cid]: {"provider": e["download_provider"], "artifact": e["artifact"]}
            for cid, e in entries.items()
            if e["status"] == "recommended" and e["starter"]
        }
        assert from_matrix == cd.recommended_model_downloads(host), cls["id"]
        unavailable = {route_of[cid] for cid, e in entries.items() if e["status"] == "unavailable" and e["starter"]}
        assert unavailable == set(cd.recommended_unavailable_routes(host)), cls["id"]


def test_class_labels_never_over_claim_their_variants():
    for spec in rec.MACHINE_CLASSES:
        reference = rec.recommended_models(dict(spec["host"]))
        for variant in spec["variants"]:
            got = rec.recommended_models(dict(spec["host"], **variant))
            assert rec._signature(got) == rec._signature(reference), (spec["id"], variant)


# ---------------------------------------------------------------------------
# Platform and memory facts
# ---------------------------------------------------------------------------


def test_apple_only_engines_are_unavailable_elsewhere_with_the_reason(matrix):
    for class_id in ("nvidia", "cpu", "intel_mac"):
        entries = _cls(matrix, class_id)["entries"]
        for cid in ("image", "video"):
            assert entries[cid]["status"] == "unavailable", (class_id, cid)
            assert "MLX runs only on Apple Silicon" in entries[cid]["reason"]


def test_video_is_memory_gated_on_apple_silicon(matrix):
    for gib in rec.APPLE_MEMORY_SIZES_GIB:
        video = _apple_class_for(matrix, gib)["entries"]["video"]
        if gib < 96:
            assert video["status"] == "unavailable", gib
            assert "(measured)" in video["reason"] and "more unified memory" in video["reason"]
        else:
            assert video["status"] == "recommended" and video["fit"] in ("fits", "tight"), gib
            assert video["memory_need_source"] == "measured"


def test_the_128_gb_text_tier_carries_the_sysctl(matrix):
    text = _apple_class_for(matrix, 128)["entries"]["text"]
    assert text["fit"] == "needs_gpu_limit"
    assert text["gpu_limit_command"] == "sudo sysctl iogpu.wired_limit_mb=117760"
    assert "sudo sysctl iogpu.wired_limit_mb=117760" in text["warning"]


def test_speech_input_follows_the_ctranslate2_builds():
    cpu = synthetic_host("cpu16")
    win_arm = rec.recommended_models(dict(cpu, os="windows", arch="arm64"))["speech_input"]
    assert win_arm["status"] == "unavailable"
    assert "CTranslate2 has no windows build for arm64" in win_arm["reason"]
    assert "transformers-asr" in win_arm["reason"]
    for host in (cpu, synthetic_host("cuda24"), synthetic_host("metal16"), dict(cpu, os="darwin", arch="x86_64")):
        e = rec.recommended_models(host)["speech_input"]
        assert (e["status"], e["provider"], e["model"], e["starter"]) == ("recommended", "faster-whisper", "base", False)
    assert rec.recommended_models(synthetic_host("cuda24"))["speech_input"]["device"] == "NVIDIA GPU (CUDA)"
    assert rec.recommended_models(synthetic_host("metal16"))["speech_input"]["device"] == "processor"


def test_music_follows_the_pytorch_builds_and_the_memory_gate(matrix):
    intel = _cls(matrix, "intel_mac")["entries"]["music"]
    assert intel["status"] == "unavailable" and "PyTorch" in intel["reason"] and "2.2.2" in intel["reason"]
    nvidia = _cls(matrix, "nvidia")["entries"]["music"]
    assert (nvidia["status"], nvidia["provider"], nvidia["device"]) == ("recommended", "acestep", "NVIDIA GPU (CUDA)")
    small = _apple_class_for(matrix, 16)["entries"]["music"]
    assert small["status"] == "unavailable" and "(estimated)" in small["reason"]
    assert _apple_class_for(matrix, 24)["entries"]["music"]["status"] == "recommended"
    cpu = _cls(matrix, "cpu")["entries"]["music"]
    assert cpu["notes"] == ["On the processor AbstractMusic runs it in float32: about twice the memory shown."]


def test_every_recommended_artifact_is_a_catalog_row_with_a_verified_size():
    for key, r in cd.RECOMMENDED_MODELS.items():
        if key == "input.text":
            continue  # the text tiers are pinned by test_recommended_text_tiers
        row_id = mc.catalog_id_for(r.download["provider"], r.download["artifact"])
        assert row_id is not None, key
        _row, art = mc._seed_row_and_artifact(row_id, r.download["provider"], r.download["artifact"])
        assert isinstance(art["download_bytes"], int) and art["size_source"] == "catalog", key


# ---------------------------------------------------------------------------
# Generated docs block + CLI
# ---------------------------------------------------------------------------


def _doc_script():
    spec = importlib.util.spec_from_file_location("update_recommended_models_doc", ROOT / "scripts" / "update_recommended_models_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_generated_docs_block_is_current():
    script = _doc_script()
    page = script.DOC.read_text(encoding="utf-8")
    assert script.updated_page(page, script.rendered_block()) == page, (
        "docs/recommended-models.md is stale: run python scripts/update_recommended_models_doc.py"
    )


def test_the_docs_script_refuses_a_page_without_its_markers():
    script = _doc_script()
    with pytest.raises(ValueError):
        script.updated_page("# Recommended models\n\nno markers here\n", "| table |\n")


def test_cli_exports_the_matrix_as_json(capsys, tmp_path):
    from abstractcore.config.main import main

    assert main(["models", "recommendations", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(json.dumps(rec.recommendation_matrix()))
    out = tmp_path / "recs.json"
    assert main(["models", "recommendations", "--json", "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_cli_host_answers_for_this_machine(capsys, monkeypatch):
    from abstractcore.config.main import main
    from abstractcore.utils import host_profile as hp

    monkeypatch.setattr(hp, "host_profile", lambda **_kw: synthetic_host("metal128"))
    assert main(["models", "recommendations", "--host", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["host"]["os"] == "darwin" and payload["host"]["memory_gib"] == 128.0
    assert payload["entries"]["text"]["artifact"] == "mlx-community/Qwen3.8-Flash-Next-4bit"
    assert payload["entries"]["video"]["status"] == "recommended"
    assert main(["models", "recommendations", "--host"]) == 0
    assert "Music generation: acestep" in capsys.readouterr().out
