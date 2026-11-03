"""The mlx provider lists the decision models it can run from the local caches."""

from __future__ import annotations

from abstractcore.decision.loader import DecisionModelInfo
from abstractcore.providers import mlx_provider


def _info(**kw):
    base = {"model_id": "x", "layout": "adapter", "path": "/p"}
    base.update(kw)
    return DecisionModelInfo(**base)


def test_adapters_with_their_base_on_disk_and_merged_mlx_conversions_are_listed(monkeypatch):
    table = {
        "jaredpalmer/kev-0.8b": _info(kind="kev", base="Qwen/Qwen3.5-0.8B-Base", base_path="/hub/base"),
        "interfaze-ai/lev": _info(kind="lev", base="Qwen/Qwen3.5-4B-Base", base_path="/hub/base4"),
        "jaredpalmer/kev-9b": _info(kind="kev", base="Qwen/Qwen3.5-9B-Base", base_path=None),   # base not downloaded
        "AbstractFramework/kev-4b-mlx-8bit": _info(layout="merged", kind="kev", runtime="mlx"),
        "jaredpalmer/kev-4b-merged-hf": _info(layout="merged", kind="kev", runtime="transformers"),
        "convaiinnovations/laya": _info(layout="merged", kind="laya", runtime="transformers"),
        "ggml-org/Kev-4B-GGUF": _info(layout="gguf", kind="kev"),
    }
    monkeypatch.setattr("abstractcore.decision.loader.describe_decision_model", lambda model, provider=None: table.get(model))
    listed = [name for name in table if mlx_provider._decision_model_runs_on_mlx(name)]
    assert listed == ["jaredpalmer/kev-0.8b", "interfaze-ai/lev", "AbstractFramework/kev-4b-mlx-8bit"]
    assert mlx_provider._decision_model_runs_on_mlx("mlx-community/Qwen3.6-27B-4bit") is False


def test_a_failing_description_never_breaks_the_listing(monkeypatch):
    def boom(model, provider=None):
        raise RuntimeError("corrupt folder")

    monkeypatch.setattr("abstractcore.decision.loader.describe_decision_model", boom)
    assert mlx_provider._decision_model_runs_on_mlx("jaredpalmer/kev-4b") is False
