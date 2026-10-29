"""A GGUF the installed llama.cpp cannot load: say whether updating llama.cpp fixes it
(the newest llama-cpp-python release knows the architecture) and expose the update for
the console's "Update llama.cpp" (operator 2026-10-29: "all of them should be known and
supported; you are fixing the symptoms, not the cause")."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractcore.utils import model_cache as mc

ASSET = Path(mc.__file__).resolve().parents[1] / "assets" / "llamacpp_known_architectures.json"


def test_the_newest_release_table_is_shipped_and_knows_the_server_models():
    data = json.loads(ASSET.read_text(encoding="utf-8"))
    archs = set(data["architectures"])
    assert data["llama_cpp_python"] == "0.3.36"
    # spark2_5 (XHToken/Spark-X2.6-4B on the OVH server) and every catalog GGUF architecture
    # (checked against the Hub's gguf.architecture on 2026-10-29)
    for a in ("spark2_5", "qwen35", "qwen35moe", "gemma4", "lfm2", "lfm2moe", "qwen3vl", "granitehybrid", "hy_v3",
              "minimax-m2", "gpt-oss", "deepseek2", "nomic-bert", "gemma3n"):
        assert a in archs, a


@pytest.fixture
def old_install(monkeypatch):
    monkeypatch.setattr(mc, "llamacpp_knows_architecture", lambda a: False)
    monkeypatch.setattr(mc, "_installed_llamacpp_version", lambda: "0.3.35")


def test_an_architecture_the_newest_release_knows_offers_the_update(old_install):
    assert mc.llamacpp_update_for("spark2_5") == {"architecture": "spark2_5", "installed": "0.3.35", "update_to": "0.3.36"}
    reason = mc.gguf_unloadable_reason("spark2_5")
    assert "llama-cpp-python 0.3.36 can: update llama.cpp" in reason, "MUTANT: no update hint -> only greyed -> RED"


def test_an_architecture_no_release_knows_stays_greyed_with_the_sentence(old_install):
    assert mc.llamacpp_update_for("not_a_real_arch") is None
    assert mc.gguf_unloadable_reason("not_a_real_arch").startswith("No llama.cpp release supports the not_a_real_arch architecture yet")


def test_a_loadable_architecture_says_nothing(monkeypatch):
    monkeypatch.setattr(mc, "llamacpp_knows_architecture", lambda a: True)
    assert mc.gguf_unloadable_reason("spark2_5") is None and mc.llamacpp_update_for("spark2_5") is None
