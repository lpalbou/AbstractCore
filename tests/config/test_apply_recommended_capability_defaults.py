"""`apply-recommended`: make an EXISTING machine match the recommendation.

THE HOLE THIS FILLS. `seed_recommended_capability_defaults` runs only when the
store file has never existed -- deliberately, because an operator who cleared a
route meant it. That safety left the product with no answer at all to "make my
machine match what you recommend", and an operator fell straight into it:

    "I asked for qwen3.5-9b everywhere, I see qwen3-0.6b"   (2026-08-01)

Their store carried `input.text: lmstudio/qwen3-0.6b` from months earlier; the
recommendation said `lmstudio/qwen/qwen3.5-9b`; nothing in the CLI, the Gateway
console or either console-TUI could close that gap.

THE RULES the action keeps, and the reasons they are rules:

  1. An EMPTY route is filled. That is the whole point.
  2. A route the operator configured DIFFERENTLY is kept and REPORTED. A
     "recommended" button that silently replaced a deliberate choice would be
     the same class of defect as the one it exists to fix.
  3. `--force` is the explicit overrule, and even then it only touches
     provider/model.
  4. FIELD-PRESERVING throughout: a pinned `base_url`, a reasoning effort and
     plugin options (`{voice: M2}`) describe THIS machine, not the
     recommendation, and survive every outcome.
"""

from __future__ import annotations

import json

import pytest

from abstractcore.config.capability_defaults import (
    RECOMMENDED_CAPABILITY_DEFAULT_ROUTES,
    RECOMMENDED_SELECTORS,
)
from abstractcore.config.manager import ConfigurationManager


def _manager(tmp_path) -> ConfigurationManager:
    return ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)


def _routes_on_disk(tmp_path) -> dict:
    return json.loads((tmp_path / "abstractcore.json").read_text(encoding="utf-8"))[
        "capability_defaults"
    ]["routes"]


def _row(report: dict, key: str) -> dict:
    return next(row for row in report["routes"] if row["key"] == key)


def test_every_recommended_route_has_a_selector_word() -> None:
    """The `--only` vocabulary covers the recommendation exactly.

    One table behind the CLI flag, the Gateway request body and both TUIs: a
    fourth recommended route with no word would be unreachable from every
    surface, and a word with no route would be a dead flag.
    """
    assert set(RECOMMENDED_SELECTORS.values()) == set(RECOMMENDED_CAPABILITY_DEFAULT_ROUTES)


def test_empty_routes_are_filled(tmp_path) -> None:
    manager = _manager(tmp_path)
    for key in list(manager.config.capability_defaults.routes):
        manager.clear_capability_default(key)

    report = manager.apply_recommended_capability_defaults()

    assert report["changed"] == len(RECOMMENDED_CAPABILITY_DEFAULT_ROUTES)
    routes = _routes_on_disk(tmp_path)
    for key, recommended in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES.items():
        assert routes[key]["provider"] == recommended.provider
        assert routes[key]["model"] == recommended.model
        assert _row(report, key)["action"] == "apply"


def test_a_route_configured_differently_is_kept_and_reported(tmp_path) -> None:
    manager = _manager(tmp_path)
    manager.set_capability_default(
        "input", "text", provider="lmstudio", model="qwen3-0.6b", base_url="http://localhost:1234/v1"
    )

    report = manager.apply_recommended_capability_defaults()

    row = _row(report, "input.text")
    assert row["action"] == "kept"
    assert row["changed"] is False
    assert row["before"]["model"] == "qwen3-0.6b"
    assert row["recommended"]["model"] == "qwen/qwen3.5-9b"
    assert _routes_on_disk(tmp_path)["input.text"]["model"] == "qwen3-0.6b", (
        "the operator's own choice is not touched without --force"
    )


def test_force_overwrites_but_preserves_the_rest_of_the_row(tmp_path) -> None:
    """The pinned base URL is the operator's machine, not the recommendation."""
    manager = _manager(tmp_path)
    manager.set_capability_default(
        "input",
        "text",
        provider="lmstudio",
        model="qwen3-0.6b",
        base_url="http://localhost:1234/v1",
        reasoning="high",
    )

    report = manager.apply_recommended_capability_defaults(only=["text"], force=True)

    row = _row(report, "input.text")
    assert row["action"] == "overwrite"
    stored = _routes_on_disk(tmp_path)["input.text"]
    assert stored["model"] == "qwen/qwen3.5-9b"
    assert stored["base_url"] == "http://localhost:1234/v1", "a pinned base URL survives"
    assert stored["reasoning"] == "high", "a reasoning effort survives"


def test_options_survive_an_already_matching_route(tmp_path) -> None:
    manager = _manager(tmp_path)
    manager.set_capability_default(
        "output", "voice", provider="supertonic", model="supertonic-3", options={"voice": "M2"}
    )

    report = manager.apply_recommended_capability_defaults(force=True)

    assert _row(report, "output.voice")["action"] == "already"
    assert _routes_on_disk(tmp_path)["output.voice"]["options"] == {"voice": "M2"}


def test_only_limits_the_blast_radius(tmp_path) -> None:
    manager = _manager(tmp_path)
    for key in list(manager.config.capability_defaults.routes):
        manager.clear_capability_default(key)

    report = manager.apply_recommended_capability_defaults(only=["image"])

    assert [row["key"] for row in report["routes"]] == ["output.image"]
    routes = _routes_on_disk(tmp_path)
    assert "output.image" in routes
    assert "output.voice" not in routes


def test_dry_run_writes_nothing_and_still_reports(tmp_path) -> None:
    manager = _manager(tmp_path)
    for key in list(manager.config.capability_defaults.routes):
        manager.clear_capability_default(key)
    before = (tmp_path / "abstractcore.json").read_text(encoding="utf-8")

    report = manager.apply_recommended_capability_defaults(dry_run=True)

    assert report["dry_run"] is True
    assert report["changed"] == len(RECOMMENDED_CAPABILITY_DEFAULT_ROUTES)
    assert (tmp_path / "abstractcore.json").read_text(encoding="utf-8") == before


def test_an_unknown_selector_refuses_with_the_vocabulary(tmp_path) -> None:
    with pytest.raises(ValueError) as exc:
        _manager(tmp_path).apply_recommended_capability_defaults(only=["audio"])
    assert "audio" in str(exc.value)
    for word in RECOMMENDED_SELECTORS:
        assert word in str(exc.value)


def test_the_cli_prints_before_and_after_and_names_what_it_kept(tmp_path, capsys) -> None:
    from abstractcore.config import main as config_main

    cfg = tmp_path / "abstractcore.json"
    seed = ConfigurationManager(config_file=cfg, apply_env=False)
    seed.set_capability_default("input", "text", provider="lmstudio", model="qwen3-0.6b")

    assert config_main.main(["config", "--config-file", str(cfg), "apply-recommended", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Would apply" in out
    assert "kept yours lmstudio/qwen3-0.6b" in out
    assert "recommended lmstudio/qwen/qwen3.5-9b" in out
    assert "--force" in out, "the output names the way to overrule it"
    assert "Nothing was written" in out
