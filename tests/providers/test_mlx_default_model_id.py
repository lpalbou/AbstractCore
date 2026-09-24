"""The default MLX model ids must name repos that exist on the Hub.

`mlx-community/Qwen3-4B` (no quant suffix) was the default and does not
exist: every mlx-community repo carries its quantisation in its name. These
checks are offline: they pin the id and cross-check it against the curated
downloads catalog, whose MLX artifacts name published repos.
"""

from abstractcore.config.model_catalog import load_seed, quant_class
from abstractcore.providers.registry import ProviderRegistry

EXPECTED_MLX_DEFAULT = "mlx-community/Qwen3-4B-4bit"


def _catalog_mlx_artifacts():
    arts = {}
    for row in load_seed()["rows"]:
        for art in row.get("artifacts") or []:
            if art.get("provider") == "mlx":
                arts[art["artifact"]] = art
    return arts


def _mlx_defaults():
    reg = ProviderRegistry()
    infos = [reg.get_provider_info(name) for name in reg.list_provider_names()]
    return {info.name: info.default_model for info in infos if info and info.name == "mlx"}


def test_registry_mlx_default_is_the_published_4bit_repo():
    assert _mlx_defaults() == {"mlx": EXPECTED_MLX_DEFAULT}


def test_every_registry_mlx_default_ends_with_a_catalog_quant_suffix():
    arts = _catalog_mlx_artifacts()
    catalog_quants = {str(a.get("quant") or "").lower() for a in arts.values()} - {""}
    defaults = _mlx_defaults()
    assert defaults, "the registry registers no MLX provider"
    for name, model in defaults.items():
        suffix = model.rsplit("-", 1)[-1].lower()
        assert suffix in catalog_quants, f"{name} default {model!r} has no quant suffix the catalog knows"
        assert quant_class(suffix) != "unknown", f"{name} default {model!r}: unknown quant {suffix!r}"
        assert model in arts, f"{name} default {model!r} is not a curated MLX artifact"


def test_endpoint_default_model_is_the_published_4bit_repo(monkeypatch):
    from abstractcore.endpoint.app import _parse_args

    monkeypatch.delenv("ABSTRACTENDPOINT_MODEL", raising=False)
    monkeypatch.delenv("ABSTRACTENDPOINT_PROVIDER", raising=False)
    cfg = _parse_args([])
    assert (cfg.provider, cfg.model) == ("mlx", EXPECTED_MLX_DEFAULT)
