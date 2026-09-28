"""Mission KK (2026-09-24): the fit warning states the numbers its verdict compared.

A fresh 24 GiB Mac read "Qwen3.8 27B ... may not fit. It needs about 16 GiB;
this computer can give a model about 18 GiB." The verdict compared the TOTAL
need (weights + KV cache + overhead, 16.4 GiB) with the USABLE memory (the
18 GiB Metal ceiling minus a 2 GiB system reserve = 16.0 GiB), while the
sentence printed the need rounded to 16 and the raw ceiling. These tests pin
need / usable / verdict / sentence to one another for the 27B tier row.

Since the 2026-09-28 rulings (operator measurement, 24 GB Mac mini) the Metal
sentence names the GPU memory limit itself (18 GiB, ~75% of RAM; measured
~17.8) and what is left after MLX's 2 GiB of working buffers, never the
remainder alone as "what this computer can give"; the 27B on a 24 GiB Mac
fits with a small context (`tight`, `small_context`). `_27b_on` asks the
catalog's fit and warning for the 27B tier's artifact directly.

Offline: synthetic hosts, no engine, no hub.
"""

from __future__ import annotations

import re

import pytest

from abstractcore.config import model_catalog as mc
from abstractcore.utils.model_fit import estimate_fit
from tests.models_engines_fakes import synthetic_host

GIB = 1024**3
MAC24 = dict(synthetic_host("metal24"), ceiling_bytes=18 * GIB, ceiling_source="metal_recommended", free_now_bytes=12 * GIB)


def _27b_on(host: dict) -> dict:
    """The 27B tier artifact's fit and warning on `host`, shaped like a pick."""
    artifact = "mlx-community/Qwen3.8-27B-4bit"
    row, art = mc._seed_row_and_artifact("qwen3.8-27b", "mlx", artifact)
    fit = mc._fit_for_seed_artifact(row, art, host)
    return {"artifact": artifact, "fit": fit, "fits": fit["verdict"] in ("fits", "tight"), "warning": mc._fit_warning(row, fit)}


def _numbers(sentence: str) -> dict:
    need = re.search(r"needs about ([0-9.]+) GiB in total \(([0-9.]+) GiB of weights plus ([0-9.]+) GiB", sentence)
    usable = re.search(
        r"GPU memory limit (?:on this Mac|raised on this Mac) is ([0-9.]+) GiB, and after ([0-9.]+) GiB of working "
        r"buffers about ([0-9.]+) GiB is left for a model", sentence)
    assert need and usable, sentence
    return {
        "need": float(need.group(1)), "weights": float(need.group(2)), "rest": float(need.group(3)),
        "ceiling": float(usable.group(1)), "reserve": float(usable.group(2)), "usable": float(usable.group(3)),
    }


def test_the_24_gib_mac_27b_warning_states_what_the_verdict_compared():
    pick = _27b_on(MAC24)
    fit = pick["fit"]
    assert pick["artifact"] == "mlx-community/Qwen3.8-27B-4bit"
    # At 8k tokens of context it would not fit; the weights and working
    # memory do: it runs with a small context (the measurement).
    assert fit["verdict"] == "tight" and fit["small_context"] is True and pick["fits"] is True
    # The verdict's own quantities are in the fit block.
    assert fit["usable_bytes"] == 16 * GIB  # 18 GiB GPU limit - 2 GiB working buffers
    assert fit["reserve_bytes"] == 2 * GIB
    assert fit["need_bytes"] == fit["weight_bytes"] + fit["kv_bytes"] + fit["overhead_bytes"]
    assert fit["need_bytes"] > fit["usable_bytes"] > fit["weight_bytes"] + fit["overhead_bytes"]
    # The sentence compares what decides "it runs": the weights against the
    # limit itself -- and claims no token count.
    assert "Its weights (15.2 GiB) fit macOS's GPU memory limit on this Mac (18.0 GiB)" in pick["warning"]
    assert "Tight: it runs with a small context by default" in pick["warning"] and "may not fit" not in pick["warning"]
    assert "can give a model about" not in pick["warning"], "the limit itself is said, not the remainder alone"
    assert "tokens of context" not in pick["warning"]


def test_a_tight_27b_on_a_bigger_ceiling_says_fits_but_tightly_with_consistent_numbers():
    base = _27b_on(MAC24)["fit"]
    need = base["need_bytes"]
    # tight: 0.8 * usable < need <= usable, with usable = ceiling - 2 GiB
    ceiling = need + 2 * GIB + GIB // 2
    pick = _27b_on(dict(MAC24, ceiling_bytes=ceiling))
    fit = pick["fit"]
    assert fit["verdict"] == "tight" and pick["fits"] is True
    assert 0.8 * fit["usable_bytes"] < fit["need_bytes"] <= fit["usable_bytes"]
    assert "fits, but tightly" in pick["warning"] and "may not fit" not in pick["warning"]
    n = _numbers(pick["warning"])
    assert n["need"] <= n["usable"]


def test_rounding_never_hides_the_comparison():
    # need and usable within 0.05 GiB: one decimal would print them equal.
    fit = estimate_fit(host={"ceiling_bytes": 20 * GIB}, weight_bytes=int(17.162 * GIB), context=1)
    assert round(fit["need_bytes"] / GIB, 1) == round(fit["usable_bytes"] / GIB, 1)  # one decimal would tie
    assert fit["verdict"] == "too_large"
    sentence = mc._fit_warning({"display_name": "X"}, fit)
    n = re.search(r"needs about ([0-9.]+) GiB in total.*can give a model about ([0-9.]+) GiB", sentence)
    assert float(n.group(1)) > float(n.group(2)), sentence


@pytest.mark.parametrize("verdict", ["fits", "unknown"])
def test_no_warning_when_nothing_is_in_doubt(verdict):
    assert mc._fit_warning({"display_name": "X"}, {"verdict": verdict, "need_bytes": 1, "usable_bytes": 2}) is None
