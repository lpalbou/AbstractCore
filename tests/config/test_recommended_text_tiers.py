"""The recommended text model: Apple-silicon memory tiers, one function.

Operator ruling 2026-09-24: on accelerator `metal` the text recommendation is
chosen by unified memory (GiB as the host probe reports it): < 24 -> Qwen3.5
9B, 24 <= m < 128 -> Qwen3.8 27B, >= 128 -> Qwen3.8 Flash-Next; the MLX lane
is the recommended one. `model_catalog.MTP_RECOMMENDED` picks the MTP build of
each tier instead of the plain 4-bit one; every test here runs under BOTH
values so flipping the switch is a one-line change.

Offline: synthetic hosts, no engine, no hub.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from abstractcore.config import capability_defaults as cd
from abstractcore.config import model_catalog as mc
from abstractcore.utils.model_fit import bits_for_quant
from tests.models_engines_fakes import isolate_host, synthetic_host

PLAIN = {
    "9b": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "27b": "mlx-community/Qwen3.8-27B-4bit",
    "flash": "mlx-community/Qwen3.8-Flash-Next-4bit",
}
MTP = {
    "9b": "mlx-works/Qwen3.5-9B-oQ4e-mtp",
    "27b": "Jundot/Qwen3.8-27B-oQ4e-mtp",
    "flash": "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
}
ROW = {"9b": "qwen3.5-9b", "27b": "qwen3.8-27b", "flash": "qwen3.8-flash-next"}
OPERATOR_MTP_OPTIONS = {"speculation": {"mode": "native_mtp", "num_draft_tokens": 2, "require_acceleration": False}}

TIER_BY_HOST = [
    ("metal16", "9b"),
    ("metal23.9", "9b"),
    ("metal24", "27b"),
    ("metal32", "27b"),
    ("metal64", "27b"),
    ("metal96", "27b"),
    ("metal127", "27b"),
    ("metal128", "flash"),
    ("metal192", "flash"),
]


@pytest.fixture(params=[False, True], ids=["plain", "mtp"])
def mtp_switch(request, monkeypatch):
    monkeypatch.setattr(mc, "MTP_RECOMMENDED", request.param)
    return request.param


@pytest.fixture
def host(tmp_path, monkeypatch):
    return isolate_host(tmp_path, monkeypatch)


def _expected(tier: str, mtp: bool) -> str:
    return (MTP if mtp else PLAIN)[tier]


# ---------------------------------------------------------------------------
# The one function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind,tier", TIER_BY_HOST)
def test_apple_silicon_tier_by_unified_memory(kind, tier, mtp_switch):
    pick = mc.recommended_text_model(synthetic_host(kind))
    assert pick["provider"] == "mlx"
    assert pick["artifact"] == _expected(tier, mtp_switch)
    assert pick["model"] == pick["artifact"]
    assert pick["catalog_id"] == ROW[tier]
    assert pick["basis"] == "apple_silicon_tiers"
    assert pick["mtp"] is mtp_switch
    # The MTP policy is host-wide (the portable route's), whichever build is picked.
    assert pick["options"] == OPERATOR_MTP_OPTIONS == cd.RECOMMENDED_CAPABILITY_DEFAULT_ROUTES["input.text"].options


@pytest.mark.parametrize("kind", ["cuda24", "cpu16", "rocm32"])
def test_other_hosts_keep_the_portable_default(kind, mtp_switch):
    pick = mc.recommended_text_model(synthetic_host(kind))
    route = cd.RECOMMENDED_CAPABILITY_DEFAULT_ROUTES["input.text"]
    assert (pick["provider"], pick["artifact"], pick["model"]) == ("lmstudio", "qwen/qwen3.5-9b@4bit", "qwen/qwen3.5-9b")
    assert pick["options"] == route.options
    assert pick["basis"] == "portable_default" and pick["mtp"] is False
    # The host-aware tables equal the portable ones exactly off Apple silicon.
    routes = cd.recommended_capability_default_routes(synthetic_host(kind))
    assert {k: v.to_dict() for k, v in routes.items()} == {
        k: v.to_dict() for k, v in cd.RECOMMENDED_CAPABILITY_DEFAULT_ROUTES.items()
    }
    assert cd.recommended_model_downloads(synthetic_host(kind)) == cd.RECOMMENDED_MODEL_DOWNLOADS


def test_the_module_switch_is_what_decides(monkeypatch):
    host = synthetic_host("metal64")
    monkeypatch.setattr(mc, "MTP_RECOMMENDED", True)
    assert mc.recommended_text_model(host)["artifact"] == MTP["27b"]
    assert cd.recommended_model_downloads(host)["input.text"]["artifact"] == MTP["27b"]
    monkeypatch.setattr(mc, "MTP_RECOMMENDED", False)
    assert mc.recommended_text_model(host)["artifact"] == PLAIN["27b"]
    assert cd.recommended_capability_default_routes(host)["input.text"].model == PLAIN["27b"]


def test_unknown_memory_on_a_mac_says_so():
    host = dict(synthetic_host("metal64"), ram_bytes=None)
    pick = mc.recommended_text_model(host, mtp=False)
    assert pick["artifact"] == PLAIN["9b"]
    assert "unknown" in pick["tier"] and pick["memory_gib"] is None


# ---------------------------------------------------------------------------
# A tier that does not fit stays the tier, and says so
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind,tier", TIER_BY_HOST)
def test_fit_is_reported_never_a_silent_tier_change(host, kind, tier, mtp_switch):
    h = synthetic_host(kind)
    pick = mc.recommended_text_model(h)
    assert pick["artifact"] == _expected(tier, mtp_switch)  # never another tier
    verdict = pick["fit"]["verdict"]
    # The pick's verdict IS the catalog's verdict for that artifact.
    row = next(r for r in mc.catalog(host=h)["rows"] if r["id"] == ROW[tier])
    art = next(a for a in row["artifacts"] if a["artifact"] == pick["artifact"])
    assert art["fit"]["verdict"] == verdict
    if verdict in ("too_large", "partial_offload"):
        assert pick["fits"] is False
        assert pick["warning"] and "may not fit" in pick["warning"]
    elif verdict == "tight":
        assert pick["fits"] is True and "tightly" in pick["warning"]
    else:
        assert pick["fits"] is True and pick["warning"] is None


def test_a_tier_that_does_not_fit_is_still_the_tier():
    pick = mc.recommended_text_model(synthetic_host("metal24"), mtp=False)
    assert pick["fit"]["verdict"] == "too_large"
    assert pick["artifact"] == PLAIN["27b"] and pick["fits"] is False and pick["warning"]


# ---------------------------------------------------------------------------
# The fit estimate uses the real upstream size and the real KV geometry
# ---------------------------------------------------------------------------

GIB = 1024**3


def _expected_need(row_id: str, artifact: str, context: int = 8192) -> int:
    """estimate_fit's documented formula, recomputed from the seed:
    need = W (upstream weight bytes) + KV (2 x layers x kv_heads x head_dim x
    2 bytes x context, full-attention layers only) + O (max(0.5 GiB, 5% W))."""

    row = next(r for r in mc.load_seed()["rows"] if r["id"] == row_id)
    art = next(a for a in row["artifacts"] if a["artifact"] == artifact)
    geo = row["kv_geometry"]
    w = art["download_bytes"]
    kv = 2 * geo["n_layers"] * geo["n_kv_heads"] * geo["head_dim"] * 2 * context
    return w + kv + int(max(0.5 * GIB, 0.05 * w))


@pytest.mark.parametrize("key", ["plain", "mtp"])
def test_flash_next_fit_uses_the_real_size_on_metal128(key):
    artifact = PLAIN["flash"] if key == "plain" else MTP["flash"]
    pick = mc.recommended_text_model(synthetic_host("metal128"), mtp=(key == "mtp"))
    fit = pick["fit"]
    assert fit["weight_bytes"] == {"plain": 111519423247, "mtp": 106294664646}[key]  # upstream listing
    assert fit["confidence"] == "exact"
    assert fit["kv_bytes"] == 2 * 12 * 2 * 256 * 2 * 8192  # 12 full-attention layers of 48
    assert fit["need_bytes"] == _expected_need("qwen3.8-flash-next", artifact)
    assert fit["need_bytes"] < 110 * GIB  # was ~199 GiB with the params-based KV guess
    assert pick["catalog_id"] == "qwen3.8-flash-next"  # never moved to the 27B


def test_the_operators_128_gib_mac_reads_the_real_numbers():
    """The host probe on the operator's M5 Max (2026-09-24): 128 GiB, Metal's
    recommended working set 107.52 GiB. The plain 4-bit needs ~109 GiB: the
    warning says so with those numbers; the tier stays Flash-Next."""

    host = dict(synthetic_host("metal128"), ceiling_bytes=int(107.52 * GIB), ceiling_source="metal_recommended")
    pick = mc.recommended_text_model(host, mtp=False)
    assert pick["artifact"] == PLAIN["flash"]
    assert pick["fit"]["verdict"] == "too_large"
    # The sentence states what the verdict compared: the TOTAL need against
    # the USABLE memory (107.52 GiB ceiling minus the 5% system reserve).
    assert "needs about 109.2 GiB in total" in pick["warning"]
    assert "can give a model about 102.1 GiB" in pick["warning"]
    assert "at most" not in pick["warning"] and "107.5 GiB" in pick["warning"]


def test_a_tight_fit_is_said_as_tight():
    need = _expected_need("qwen3.8-flash-next", PLAIN["flash"])
    # c_eff = ceiling - max(2 GiB, 5% ceiling); tight = 0.8 c_eff < need <= c_eff
    ceiling = int(need / 0.95) + GIB
    pick = mc.recommended_text_model(dict(synthetic_host("metal128"), ceiling_bytes=ceiling), mtp=False)
    assert pick["fit"]["verdict"] == "tight"
    assert pick["fits"] is True and "fits, but tightly" in pick["warning"]
    assert pick["artifact"] == PLAIN["flash"]


def test_metal192_fits_the_flash_tier_without_a_warning():
    pick = mc.recommended_text_model(synthetic_host("metal192"), mtp=False)
    assert pick["fit"]["verdict"] == "fits" and pick["warning"] is None


# ---------------------------------------------------------------------------
# Every consumer derives from the one function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind,tier", TIER_BY_HOST)
def test_catalog_flags_follow_the_pick(host, kind, tier, mtp_switch):
    rows = {r["id"]: r for r in mc.catalog(host=synthetic_host(kind))["rows"]}
    text_starters = [r["id"] for r in rows.values() if r["starter"] and "chat" in r["tags"]]
    assert text_starters == [ROW[tier]]
    for t, row_id in ROW.items():
        picked = [a["artifact"] for a in rows[row_id]["artifacts"] if a["recommended"]]
        assert picked == [_expected(t, mtp_switch)], row_id


@pytest.mark.parametrize("kind,tier", TIER_BY_HOST)
def test_seed_routes_and_downloads_follow_the_pick(kind, tier, mtp_switch):
    h = synthetic_host(kind)
    expected = _expected(tier, mtp_switch)
    assert cd.recommended_model_downloads(h)["input.text"] == {"provider": "mlx", "artifact": expected}
    route = cd.recommended_capability_default_routes(h)["input.text"]
    assert (route.provider, route.model) == ("mlx", expected)
    assert route.options == OPERATOR_MTP_OPTIONS
    seeded = cd.seed_recommended_capability_defaults(cd.CapabilityDefaultsConfig(), host=h)
    assert seeded.routes["input.text"].model == expected
    plan = {p["key"]: p for p in cd.plan_recommended_capability_defaults({}, host=h)}
    assert plan["input.text"]["after"]["model"] == expected
    assert plan["input.text"]["download"] == {"provider": "mlx", "artifact": expected}
    # voice and image are untouched by the tiers
    assert cd.recommended_model_downloads(h)["output.voice"] == cd.RECOMMENDED_MODEL_DOWNLOADS["output.voice"]


def test_materializer_downloads_and_plan_follow_the_pick(host, monkeypatch, mtp_switch):
    from abstractcore.config import model_materializer as mm
    from abstractcore.utils import host_profile as hp

    monkeypatch.setattr(hp, "host_profile", lambda **_k: synthetic_host("metal24"))
    expected = _expected("27b", mtp_switch)
    items = {i["route"]: i for i in mm.recommended_downloads()}
    assert items["input.text"] == {"route": "input.text", "provider": "mlx", "artifact": expected}
    plan = {r["route"]: r for r in mm.recommended_plan()["recommended"]}
    text = plan["input.text"]
    assert text["artifact"] == expected and text["catalog_id"] == "qwen3.8-27b"
    assert text["fit_verdict"] == "too_large" and text["fits"] is False and text["warning"]


def test_mlx_is_the_recommended_lane_on_a_mac(host, monkeypatch):
    """Rows outside the tiers: with an MLX and an LM Studio build of equal fit,
    the MLX build is pre-selected on Apple silicon (and LM Studio elsewhere)."""

    from abstractcore.config import engines

    real = engines.engine_inventory

    def inventory(*args, **kwargs):
        out = real(*args, **kwargs)
        for e in out["engines"]:
            e["installed"] = False
        return out

    monkeypatch.setattr(engines, "engine_inventory", inventory)
    rows = {r["id"]: r for r in mc.catalog(host=synthetic_host("metal128"))["rows"]}
    for row_id in ("gemma-3-1b", "qwen3.6-35b-a3b"):
        pick = next(a for a in rows[row_id]["artifacts"] if a["recommended"])
        assert pick["provider"] == "mlx", (row_id, pick["artifact"])
    assert mc._HOST_PREFERENCE["metal"][0] == "mlx"


# ---------------------------------------------------------------------------
# The seed carries every tier artifact, verified upstream
# ---------------------------------------------------------------------------


def test_every_tier_artifact_is_an_upstream_verified_seed_artifact():
    rows = {r["id"]: r for r in mc.load_seed()["rows"]}
    assert [t["row"] for t in mc.APPLE_TEXT_TIERS] == [ROW["9b"], ROW["27b"], ROW["flash"]]
    for key, tier in zip(("9b", "27b", "flash"), mc.APPLE_TEXT_TIERS):
        assert (tier["plain"], tier["mtp"]) == (PLAIN[key], MTP[key])
        for artifact in (tier["plain"], tier["mtp"]):
            art = next(a for a in rows[tier["row"]]["artifacts"] if a["artifact"] == artifact)
            assert art["provider"] == "mlx"
            assert art["upstream"]["method"] == "hf_api"
            assert isinstance(art["download_bytes"], int) and art["download_bytes"] > 0
        mtp_art = next(a for a in rows[tier["row"]]["artifacts"] if a["artifact"] == tier["mtp"])
        plain_art = next(a for a in rows[tier["row"]]["artifacts"] if a["artifact"] == tier["plain"])
        assert mtp_art["options"] == OPERATOR_MTP_OPTIONS
        assert "options" not in plain_art


def test_boundaries_are_exact():
    gib = 1024**3
    for mem, row in ((23.999, ROW["9b"]), (24.0, ROW["27b"]), (127.999, ROW["27b"]), (128.0, ROW["flash"])):
        host = dict(synthetic_host("metal64"), ram_bytes=int(mem * gib))
        assert mc.recommended_text_model(host, mtp=False)["catalog_id"] == row, mem


# ---------------------------------------------------------------------------
# quant_class
# ---------------------------------------------------------------------------

QUANT_TABLE = [
    ("q2_k", "2bit"),
    ("iq2_xs", "2bit"),
    ("q3_k_m", "3bit"),
    ("q3_k_l", "3bit"),
    ("3bit", "3bit"),
    ("q4_k_m", "4bit"),
    ("Q4_K_M", "4bit"),
    ("q4_0", "4bit"),
    ("iq4_xs", "4bit"),
    ("UD-Q4_K_XL", "4bit"),
    ("4bit", "4bit"),
    ("4-bit", "4bit"),
    ("mxfp4", "4bit"),
    ("oq4e", "4bit"),
    ("q5_k_m", "5bit"),
    ("5bit", "5bit"),
    ("q6_k", "6bit"),
    ("6bit", "6bit"),
    ("Q8_0", "8bit"),
    ("8bit", "8bit"),
    ("fp8", "8bit"),
    ("int8", "8bit"),
    ("bf16", "16bit"),
    ("f16", "16bit"),
    ("fp16", "16bit"),
    ("f32", "full"),
    ("fp32", "full"),
    (None, "unknown"),
    ("", "unknown"),
    ("iq1_s", "unknown"),
    ("mystery", "unknown"),
]


@pytest.mark.parametrize("quant,expected", QUANT_TABLE)
def test_quant_class_table(quant, expected):
    assert mc.quant_class(quant, bits_for_quant(quant)) == expected


@pytest.mark.parametrize("bits,expected", [(4.5, "4bit"), (4.85, "4bit"), (8.5, "8bit"), (6.6, "6bit"), (16.0, "16bit"), (32.0, "full"), (1.6, "unknown"), (None, "unknown")])
def test_quant_class_from_bits_alone(bits, expected):
    assert mc.quant_class(None, bits) == expected


@pytest.mark.parametrize(
    "provider,quant,expected",
    [
        ("ollama", "q4_k_m", ("4bit", "stated")),
        ("ollama", "q8_0", ("8bit", "stated")),
        ("lmstudio", "4bit", ("4bit", "stated")),
        ("mlx", "oq4e", ("4bit", "stated")),
        ("ollama", None, ("4bit", "assumed")),  # a bare Ollama tag: Q4_K_M by default
        ("lmstudio", None, ("4bit", "assumed")),  # a bare LM Studio id: a 4-bit build
        ("huggingface", None, ("unknown", None)),  # no quant information at all
        ("mlx", None, ("unknown", None)),
        ("ollama", "mystery", ("unknown", None)),
    ],
)
def test_quant_class_source(provider, quant, expected):
    assert mc.quant_class_for(provider, quant) == expected


def test_catalog_labels_an_assumed_quant_as_assumed(host):
    row = next(r for r in mc.catalog(host=synthetic_host("metal64"))["rows"] if r["id"] == "qwen3.5-9b")
    tag = next(a for a in row["artifacts"] if a["artifact"] == "qwen3.5:9b")
    assert tag["quant"] is None  # the reference names no quant...
    assert (tag["quant_class"], tag["quant_class_source"]) == ("4bit", "assumed")  # ...the default is assumed
    lms = next(a for a in row["artifacts"] if a["provider"] == "lmstudio")
    assert (lms["quant_class"], lms["quant_class_source"]) == ("4bit", "stated")  # `@4bit`
    q8 = next(a for a in row["artifacts"] if a["artifact"] == "qwen3.5:9b-q8_0")
    assert (q8["quant_class"], q8["quant_class_source"]) == ("8bit", "stated")
    for r in mc.catalog(host=synthetic_host("metal64"))["rows"]:
        for a in r["artifacts"]:
            assert (a["quant_class"] == "unknown") == (a["quant_class_source"] is None), a


# ---------------------------------------------------------------------------
# MTP companions
# ---------------------------------------------------------------------------


def test_companions_come_from_the_drafter_registry_with_verified_sizes(host):
    """One source: `companions` is the MLX drafter registry's answer
    (`model_materializer.companion_artifacts`), never a hand-typed seed list;
    the seed records only each companion's verified size, which the payload
    folds into `download_bytes` (and reports as `companion_bytes`)."""

    from abstractcore.config import model_materializer as mm

    seed = mc.load_seed()
    assert not any("companions" in a for r in seed["rows"] for a in r["artifacts"])
    rows = {r["id"]: r for r in seed["rows"]}
    payload = {r["id"]: r for r in mc.catalog(host=synthetic_host("metal64"))["rows"]}
    drafter = {"9b": "mlx-community/Qwen3.5-9B-MTP-4bit", "27b": "mlx-community/Qwen3.8-27B-MTP-4bit", "flash": None}
    for key in ("9b", "27b", "flash"):
        for artifact in (PLAIN[key], MTP[key]):
            shown = next(a for a in payload[ROW[key]]["artifacts"] if a["artifact"] == artifact)
            seeded = next(a for a in rows[ROW[key]]["artifacts"] if a["artifact"] == artifact)
            assert shown["companions"] == mm.companion_artifacts("mlx", artifact)
            if drafter[key] is None:
                assert shown["companions"] == [] and shown["companion_bytes"] is None
                assert shown["download_bytes"] == seeded["download_bytes"]
            else:
                assert shown["companions"] == [drafter[key]]
                size = seed["companion_sizes"][drafter[key]]["download_bytes"]
                assert shown["companion_bytes"] == size
                assert shown["download_bytes"] == seeded["download_bytes"] + size
    pick = mc.recommended_text_model(synthetic_host("metal64"), mtp=True)
    assert pick["companions"] == [drafter["27b"]]
    assert pick["fit"]["weight_bytes"] == 16971681558 + 238934137  # model + drafter, both upstream
    plain = mc.recommended_text_model(synthetic_host("metal64"), mtp=False)
    assert plain["companions"] == [drafter["27b"]]  # the MTP policy is on: the plain build uses it too
    assert mc.recommended_text_model(synthetic_host("metal128"), mtp=True)["companions"] == []


def test_every_registry_companion_of_a_seed_artifact_has_a_verified_size():
    from abstractcore.config import model_materializer as mm

    seed = mc.load_seed()
    sizes = seed["companion_sizes"]
    for r in seed["rows"]:
        for a in r["artifacts"]:
            for repo in mm.companion_artifacts(a["provider"], a["artifact"]):
                assert repo in sizes, (a["artifact"], repo)
                assert sizes[repo]["upstream"]["method"] == "hf_api"


def test_mtp_builds_carry_a_note():
    rows = {r["id"]: r for r in mc.load_seed()["rows"]}
    for key in ("9b", "27b", "flash"):
        assert next(a for a in rows[ROW[key]]["artifacts"] if a["artifact"] == MTP[key])["note"]


# ---------------------------------------------------------------------------
# The seed validator and the new fields
# ---------------------------------------------------------------------------


def _upstream_art(seed):
    for i, r in enumerate(seed["rows"]):
        for j, a in enumerate(r["artifacts"]):
            if "upstream" in a:
                return i, j
    raise AssertionError("no upstream-verified artifact in the seed")


def test_validator_accepts_the_new_rows():
    seed = mc.load_seed()
    assert mc.validate_catalog(seed) == []
    ids = {r["id"] for r in seed["rows"]}
    assert "qwen3.8-flash-next" in ids
    assert sum(1 for r in seed["rows"] for a in r["artifacts"] if "upstream" in a) >= 90


def test_validator_rejects_an_upstream_row_without_a_verifiable_size():
    seed = mc.load_seed()
    i, j = _upstream_art(seed)
    broken = copy.deepcopy(seed)
    broken["rows"][i]["artifacts"][j]["download_bytes"] = None
    broken["rows"][i]["artifacts"][j]["size_source"] = "unknown"
    assert any("verified download_bytes" in e for e in mc.validate_catalog(broken))
    broken = copy.deepcopy(seed)
    broken["rows"][i]["artifacts"][j]["upstream"]["method"] = "guessed"
    assert mc.validate_catalog(broken)
    broken = copy.deepcopy(seed)
    del broken["rows"][i]["artifacts"][j]["upstream"]["checked"]
    assert mc.validate_catalog(broken)


def test_validator_rejects_bad_kv_geometry_and_companions():
    seed = mc.load_seed()
    i = next(i for i, r in enumerate(seed["rows"]) if r["id"] == "qwen3.8-flash-next")
    broken = copy.deepcopy(seed)
    broken["rows"][i]["kv_geometry"]["n_layers"] = 0
    assert any("kv_geometry.n_layers" in e for e in mc.validate_catalog(broken))
    broken = copy.deepcopy(seed)
    del broken["rows"][i]["kv_geometry"]["source"]
    assert mc.validate_catalog(broken)
    j = next(j for j, r in enumerate(seed["rows"]) if r["id"] == "qwen3.8-27b")
    broken = copy.deepcopy(seed)
    broken["rows"][j]["artifacts"][0]["companions"] = ["mlx-community/Qwen3.8-27B-MTP-4bit"]
    assert any("unknown field 'companions'" in e for e in mc.validate_catalog(broken))  # the registry owns them
    broken = copy.deepcopy(seed)
    del broken["companion_sizes"]["mlx-community/Qwen3.8-27B-MTP-4bit"]["download_bytes"]
    assert any("companion_sizes" in e for e in mc.validate_catalog(broken))


def test_validator_rejects_bad_speculation_options():
    seed = mc.load_seed()
    row = next(i for i, r in enumerate(seed["rows"]) if r["id"] == "qwen3.8-27b")
    j = next(j for j, a in enumerate(seed["rows"][row]["artifacts"]) if "options" in a)
    broken = copy.deepcopy(seed)
    broken["rows"][row]["artifacts"][j]["options"]["speculation"]["num_drafts"] = 2
    assert any("speculation" in e for e in mc.validate_catalog(broken))
    broken = copy.deepcopy(seed)
    broken["rows"][row]["artifacts"][j]["options"]["temperature"] = 0
    assert mc.validate_catalog(broken)


def test_every_upstream_artifact_is_eight_bit_or_a_tier_or_a_reverified_id():
    """No row was invented: an upstream-verified artifact is an 8-bit twin, a
    tier build, or an existing id whose size was re-read upstream."""

    tier_ids = set(PLAIN.values()) | set(MTP.values())
    reverified = {"mlx-community/Qwen3.5-9B-4bit", "mlx-community/Qwen3.8-27B-4bit"}
    for r in mc.load_seed()["rows"]:
        for a in r["artifacts"]:
            if "upstream" not in a:
                continue
            assert a["artifact"] in tier_ids or a["artifact"] in reverified or mc.quant_class(a["quant"]) == "8bit", a


# ---------------------------------------------------------------------------
# Importing AbstractCore never loads an engine
# ---------------------------------------------------------------------------


def test_import_on_a_fresh_home_seeds_without_loading_an_engine(tmp_path):
    """`import abstractcore` builds the config manager, which seeds a fresh
    store with the host-aware recommendation. The tier needs only the LIGHT
    host reading (os/arch/RAM): mlx, torch and transformers stay unloaded."""

    import json
    import os
    import subprocess
    import sys

    home = tmp_path / "fresh-home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ABSTRACTCORE_", "HF_", "ABSTRACTGATEWAY_"))}
    env["HOME"] = str(home)
    code = (
        "import sys, json\n"
        "import abstractcore\n"
        "import abstractcore.media.openai_parts\n"
        "from abstractcore.config.manager import get_config_manager\n"
        "route = get_config_manager().config.capability_defaults.routes['input.text'].to_dict()\n"
        "print(json.dumps({'loaded': [m for m in ('mlx', 'mlx.core', 'torch', 'transformers') if m in sys.modules],"
        " 'route': route}))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120,
                          cwd=str(Path(mc.__file__).resolve().parents[2]))
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["loaded"] == [], out
    assert out["route"]["provider"] and out["route"]["model"]  # the seed still ran


def test_the_route_tables_use_the_light_host_reading(monkeypatch):
    from abstractcore.utils import host_profile as hp

    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return synthetic_host("metal64")

    monkeypatch.setattr(hp, "host_profile", probe)
    cd.recommended_capability_default_routes()
    cd.recommended_model_downloads()
    assert calls and all(c.get("light") is True for c in calls), calls
