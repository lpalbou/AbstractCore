"""THE AUTHORITY CONTRACT for configuration AbstractCore owns.

AbstractCore and AbstractGateway are the two entry points to the framework. Where
AbstractCore holds a configuration value, that value is the single source of
truth: both entry points read and write the same store, and neither keeps a copy.

This file pins the AbstractCore side of that contract, parametrized over every
core-owned domain rather than restated per domain:

  - a write persists to the store and reads back identically
  - a route is reported as configured whenever it carries any field, including a
    reasoning effort with no provider or model beside it
  - `set_capability_default` writes a WHOLE ROW, which is why every caller that
    edits one field goes through `update_capability_default` instead: it keeps
    the fields the caller did not name, clears the ones it is given empty, and
    is the rule the CLI, the server's config routes and the Gateway all share
  - clearing removes the row
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractcore.config.capability_defaults import (
    TEXT_ROUTE_KEY,
    TEXT_ROUTE_KEYS,
    TEXT_ROUTE_STORAGE_KEY,
    capability_default_reasoning,
)
from abstractcore.config.manager import ConfigurationManager


# (route as written, storage key, fields) -- one row per core-owned domain.
_DOMAINS = [
    ("output.text", "input.text", {"provider": "lmstudio", "model": "qwen3-30b"}),
    ("output.text", "input.text", {"provider": "lmstudio", "model": "qwen3-30b", "reasoning": "high"}),
    ("input.text", "input.text", {"provider": "ollama", "model": "granite"}),
    ("output.image.text_to_image", "output.image.text_to_image", {"provider": "mlx-gen", "model": "flux"}),
    ("output.image.image_to_image", "output.image.image_to_image", {"provider": "mlx-gen", "model": "kontext"}),
    ("output.video.text_to_video", "output.video.text_to_video", {"provider": "hf", "model": "ltx"}),
    ("output.voice", "output.voice", {"provider": "abstractvoice", "model": "supertonic"}),
    ("input.voice", "input.voice", {"provider": "abstractvoice", "model": "whisper-small"}),
    ("output.music", "output.music", {"provider": "abstractmusic", "model": "musicgen"}),
    ("output.sound", "output.sound", {"provider": "abstractsound", "model": "audiogen"}),
    ("output.scene3d.text_to_scene3d", "output.scene3d.text_to_scene3d", {"provider": "abstract3d", "model": "trellis"}),
    ("embedding.text", "embedding.text", {"provider": "huggingface", "model": "all-minilm-l6-v2"}),
]

_IDS = [f"{route}:{'+'.join(sorted(fields))}" for route, _key, fields in _DOMAINS]


def _manager(tmp_path: Path) -> ConfigurationManager:
    return ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)


@pytest.mark.parametrize(("route", "storage_key", "fields"), _DOMAINS, ids=_IDS)
def test_a_write_persists_and_reads_back(tmp_path: Path, route: str, storage_key: str, fields: dict) -> None:
    manager = _manager(tmp_path)
    assert manager.set_capability_default(route, **fields)

    stored = json.loads((tmp_path / "abstractcore.json").read_text(encoding="utf-8"))
    row = stored["capability_defaults"]["routes"][storage_key]
    for name, value in fields.items():
        assert row[name] == value, f"{route}.{name} did not reach the store"

    read_back = manager.get_capability_default(route)
    for name, value in fields.items():
        assert read_back[name] == value, f"{route}.{name} did not read back"
    assert read_back["source"] == "abstractcore.capability_defaults"


@pytest.mark.parametrize(("route", "storage_key", "fields"), _DOMAINS, ids=_IDS)
def test_a_written_route_is_reported_configured(tmp_path: Path, route: str, storage_key: str, fields: dict) -> None:
    manager = _manager(tmp_path)
    assert manager.set_capability_default(route, **fields)

    rows = {row["key"]: row for row in manager.list_capability_defaults()}
    assert rows[storage_key]["configured"] is True
    for name, value in fields.items():
        assert rows[storage_key][name] == value


@pytest.mark.parametrize(("route", "storage_key", "fields"), _DOMAINS, ids=_IDS)
def test_clearing_removes_the_row(tmp_path: Path, route: str, storage_key: str, fields: dict) -> None:
    manager = _manager(tmp_path)
    assert manager.set_capability_default(route, **fields)
    assert manager.clear_capability_default(route)

    stored = json.loads((tmp_path / "abstractcore.json").read_text(encoding="utf-8"))
    assert storage_key not in stored["capability_defaults"]["routes"]
    assert manager.get_capability_default(route)["source"] == "not_configured"


def test_a_reasoning_only_route_is_configured(tmp_path: Path) -> None:
    """A reasoning effort with no provider beside it is a configured route.

    The route dataclass counts reasoning towards `configured()`, so the grid rows
    must too: a row that reported otherwise would hide the operator's setting
    from every reader of the grid, including the Gateway console.
    """
    manager = _manager(tmp_path)
    assert manager.set_capability_default("output.text", reasoning="medium")

    rows = {row["key"]: row for row in manager.list_capability_defaults()}
    assert rows[TEXT_ROUTE_STORAGE_KEY]["reasoning"] == "medium"
    assert rows[TEXT_ROUTE_STORAGE_KEY]["configured"] is True
    assert rows[TEXT_ROUTE_KEY]["configured"] is True


def test_a_partial_write_replaces_the_whole_row(tmp_path: Path) -> None:
    """The store keeps a route as one row, so a write is a replacement.

    This is the reason a caller that edits one field of a route reads the stored
    row and merges over it first. Pinning the behavior here keeps that
    requirement visible to anyone adding a new writer.
    """
    manager = _manager(tmp_path)
    assert manager.set_capability_default("output.text", provider="lmstudio", model="qwen3", reasoning="high")
    assert manager.set_capability_default("output.text", provider="lmstudio", model="qwen3")

    assert manager.get_capability_default("output.text").get("reasoning") is None


def test_a_partial_update_keeps_the_fields_it_does_not_name(tmp_path: Path) -> None:
    """`update_capability_default` is the rule every writer of this store shares.

    The store has more than one writer -- the `abstractcore config` CLI, the
    AbstractCore server's config routes, and AbstractGateway through them. If
    they did not agree that an unnamed field survives, setting a model from one
    would silently discard a reasoning effort set from another.
    """
    manager = _manager(tmp_path)
    assert manager.set_capability_default(
        "output.text",
        provider="lmstudio",
        model="qwen3",
        base_url="http://127.0.0.1:1234/v1",
        reasoning="high",
        options={"profile": "local"},
    )

    assert manager.update_capability_default("output.text", model="qwen3-next")

    row = manager.get_capability_default("output.text")
    assert row["model"] == "qwen3-next"
    assert row["provider"] == "lmstudio"
    assert row["base_url"] == "http://127.0.0.1:1234/v1"
    assert row["reasoning"] == "high"
    assert row["options"] == {"profile": "local"}


def test_a_partial_update_clears_the_field_it_is_given_empty(tmp_path: Path) -> None:
    """`""` is how a writer says "clear this one field" without dropping the row."""
    manager = _manager(tmp_path)
    assert manager.set_capability_default("output.text", provider="lmstudio", model="qwen3", reasoning="high")

    assert manager.update_capability_default("output.text", reasoning="")

    row = manager.get_capability_default("output.text")
    assert row.get("reasoning") is None
    assert row["model"] == "qwen3"


def test_a_partial_update_never_persists_a_derived_row(tmp_path: Path) -> None:
    """A route answered by another route has nothing stored of its own.

    `input.image` is served from the text route while it is unset. Merging over
    what a reader was shown would persist that derivation, freezing coverage
    into a copy of whatever the text route said at save time.
    """
    manager = _manager(tmp_path)
    assert manager.set_capability_default("input.text", provider="lmstudio", model="qwen3-vl")

    assert manager.update_capability_default("input.image", provider="mlx-vlm")

    stored = manager.stored_capability_default("input.image")
    assert stored == {"provider": "mlx-vlm"}


def test_the_legacy_global_default_model_writes_the_text_route_without_clearing_it(tmp_path: Path) -> None:
    """`--set-default-model` is the legacy spelling of "set the text route".

    It writes the same row every other entry point writes, so it owes that row
    the same care: naming a provider and a model must not discard the reasoning
    effort, base URL or options already on it.
    """
    manager = _manager(tmp_path)
    assert manager.set_capability_default(
        "output.text",
        provider="lmstudio",
        model="qwen3",
        reasoning="high",
        base_url="http://127.0.0.1:1234/v1",
        options={"profile": "local"},
    )

    assert manager.set_global_default_model("ollama/granite")

    row = manager.get_capability_default("output.text")
    assert (row["provider"], row["model"]) == ("ollama", "granite")
    assert row["reasoning"] == "high"
    assert row["base_url"] == "http://127.0.0.1:1234/v1"
    assert row["options"] == {"profile": "local"}


def test_the_legacy_embeddings_setters_keep_the_rest_of_the_embedding_route(tmp_path: Path) -> None:
    """`--set-embeddings-model` names three fields; the route can carry more.

    The `embeddings` section mirrors onto `embedding.text`, and a mirror that
    replaced the row would drop plugin options an operator set on that route
    through the capability-defaults surface.
    """
    manager = _manager(tmp_path)
    assert manager.set_capability_default(
        "embedding.text",
        provider="huggingface",
        model="all-minilm-l6-v2",
        options={"dimensions": 384},
    )

    assert manager.set_embeddings_model("openai/text-embedding-3-small")

    row = manager.get_capability_default("embedding.text")
    assert (row["provider"], row["model"]) == ("openai", "text-embedding-3-small")
    assert row["options"] == {"dimensions": 384}


def test_status_reports_the_global_default_from_the_route_not_the_legacy_field(tmp_path: Path) -> None:
    """One store, one answer.

    `default_models.global_*` is written only by the legacy flag, so a route set
    through `config set-default` or from the Gateway leaves it behind. Status
    reads the route, and keeps the legacy pair beside it under its own name.
    """
    manager = _manager(tmp_path)
    assert manager.set_capability_default("output.text", provider="lmstudio", model="qwen3", reasoning="high")

    status = manager.get_status()["global_defaults"]
    assert (status["provider"], status["model"]) == ("lmstudio", "qwen3")
    assert status["reasoning"] == "high"
    assert status["source"] == "abstractcore.capability_defaults"
    assert status["legacy"] == {"provider": None, "model": None}


def test_the_config_cli_keeps_the_fields_the_operator_did_not_type(tmp_path: Path) -> None:
    """`abstractcore config set-default` is a partial update, like every writer.

    An operator who sets the model from the CLI has not asked to drop the
    reasoning effort or the voice options on that route, and a flag they did not
    type must not act as if they had.
    """
    from argparse import Namespace

    from abstractcore.config.main import _handle_config_set_default

    config_file = tmp_path / "abstractcore.json"
    manager = ConfigurationManager(config_file=config_file, apply_env=False)
    assert manager.set_capability_default(
        "output.text", provider="lmstudio", model="qwen3", reasoning="high", options={"profile": "local"}
    )

    args = Namespace(
        config_file=str(config_file),
        config_dir=None,
        route="output.text",
        provider=None,
        model="qwen3-next",
        base_url=None,
        reasoning=None,
        option=[],
    )
    assert _handle_config_set_default(args) == 0

    row = ConfigurationManager(config_file=config_file, apply_env=False).get_capability_default("output.text")
    assert row["model"] == "qwen3-next"
    assert row["provider"] == "lmstudio"
    assert row["reasoning"] == "high"
    assert row["options"] == {"profile": "local"}


def test_reasoning_resolves_from_either_text_route_key() -> None:
    """`output.text` answers first; `input.text` is the storage key beneath it."""
    assert capability_default_reasoning({TEXT_ROUTE_KEY: {"reasoning": "High"}}) == "high"
    assert capability_default_reasoning({TEXT_ROUTE_STORAGE_KEY: {"reasoning": "low"}}) == "low"
    assert (
        capability_default_reasoning(
            {
                TEXT_ROUTE_KEY: {"source": "not_configured"},
                TEXT_ROUTE_STORAGE_KEY: {"reasoning": "medium"},
            }
        )
        == "medium"
    )
    assert capability_default_reasoning({TEXT_ROUTE_KEY: {"provider": "lmstudio"}}) is None
    assert capability_default_reasoning({}) is None
    assert capability_default_reasoning(None) is None
    assert TEXT_ROUTE_KEYS == (TEXT_ROUTE_KEY, TEXT_ROUTE_STORAGE_KEY)


def test_the_embeddings_section_mirrors_the_embedding_route_both_ways(tmp_path: Path) -> None:
    """One question, one answer, whichever entry point asked it.

    The `embedding.text` capability route is what the framework routes on, and
    the `embeddings` section is the shape the embeddings commands and
    `--show-config` read. They mirror in both directions so a route write and an
    `abstractcore config --set-embeddings-model` never disagree.
    """
    manager = _manager(tmp_path)

    manager.set_embeddings_model("huggingface/all-minilm-l6-v2")
    assert manager.get_capability_default("embedding.text")["provider"] == "huggingface"

    assert manager.set_capability_default(
        "embedding.text", provider="lmstudio", model="text-embedding-qwen3", base_url="http://127.0.0.1:1234/v1"
    )
    reread = _manager(tmp_path)
    assert reread.config.embeddings.provider == "lmstudio"
    assert reread.config.embeddings.model == "text-embedding-qwen3"
    assert reread.config.embeddings.base_url == "http://127.0.0.1:1234/v1"

    assert reread.clear_capability_default("embedding.text")
    cleared = _manager(tmp_path)
    assert cleared.config.embeddings.provider is None
    assert cleared.config.embeddings.model is None


def test_the_text_route_canonicalizes_to_one_cell(tmp_path: Path) -> None:
    """`output.text` and `input.text` name the same stored cell, never two."""
    manager = _manager(tmp_path)
    assert manager.set_capability_default("output.text", provider="lmstudio", model="qwen3", reasoning="high")

    stored = json.loads((tmp_path / "abstractcore.json").read_text(encoding="utf-8"))
    routes = stored["capability_defaults"]["routes"]
    assert TEXT_ROUTE_STORAGE_KEY in routes
    assert TEXT_ROUTE_KEY not in routes

    derived = manager.get_capability_default(TEXT_ROUTE_KEY)
    assert derived["derived_from"] == TEXT_ROUTE_STORAGE_KEY
    assert derived["reasoning"] == "high"
