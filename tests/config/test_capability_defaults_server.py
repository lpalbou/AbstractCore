from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.basic


def test_server_capability_defaults_routes_persist_to_core_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    with client:
        created = client.put(
            "/v1/config/capability-defaults/output/text",
            json={
                "provider": "lmstudio",
                "model": "qwen/qwen3.6-35b-a3b",
                "base_url": "http://127.0.0.1:1234/v1",
                "options": {"profile": "local"},
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        route = next(item for item in body["routes"] if item["key"] == "output.text")
        assert route["kind"] == "output"
        assert route["provider"] == "lmstudio"
        assert route["model"] == "qwen/qwen3.6-35b-a3b"
        assert route["base_url"] == "http://127.0.0.1:1234/v1"
        assert route["options"] == {"profile": "local"}
        assert route["source"] == "abstractcore.capability_defaults"

        listed = client.get("/v1/config/capability-defaults")
        assert listed.status_code == 200, listed.text
        assert any(item["key"] == "output.text" and item["configured"] for item in listed.json()["routes"])

        cleared = client.delete("/v1/config/capability-defaults/output/text")
        assert cleared.status_code == 200, cleared.text
        cleared_route = next(item for item in cleared.json()["routes"] if item["key"] == "output.text")
        assert cleared_route["configured"] is False


def test_server_task_capability_defaults_routes_persist_to_core_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    # An EXISTING empty store (fresh-absent installs seed recommended defaults).
    store = tmp_path / ".abstractcore" / "config" / "abstractcore.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{}")

    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    with client:
        created = client.put(
            "/v1/config/capability-defaults/output/image/image_to_image",
            json={
                "provider": "mlx-gen",
                "model": "AbstractFramework/qwen-image-edit-2511-4bit",
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        route = next(item for item in body["routes"] if item["key"] == "output.image.image_to_image")
        assert route["kind"] == "output"
        assert route["modality"] == "image"
        assert route["task"] == "image_to_image"
        assert route["provider"] == "mlx-gen"
        assert route["model"] == "AbstractFramework/qwen-image-edit-2511-4bit"
        assert route["source"] == "abstractcore.capability_defaults"

        listed = client.get("/v1/config/capability-defaults")
        assert listed.status_code == 200, listed.text
        routes = {item["key"]: item for item in listed.json()["routes"]}
        assert routes["output.image.image_to_image"]["configured"] is True
        assert routes["output.image"]["configured"] is False

        cleared = client.delete("/v1/config/capability-defaults/output/image/image_to_image")
        assert cleared.status_code == 200, cleared.text
        cleared_route = next(item for item in cleared.json()["routes"] if item["key"] == "output.image.image_to_image")
        assert cleared_route["configured"] is False


def _route(client, key: str = "output.text") -> dict:
    listed = client.get("/v1/config/capability-defaults")
    assert listed.status_code == 200, listed.text
    return next(item for item in listed.json()["routes"] if item["key"] == key)


def test_the_server_config_route_updates_a_field_without_replacing_the_row(monkeypatch, tmp_path) -> None:
    """A field the request does not name keeps its stored value.

    The config routes are how a split AbstractGateway writes this store, and the
    Gateway console saves a provider and a model. If naming those two cleared
    the rest of the row, an operator's `abstractcore config set-default
    output.text --reasoning high` would vanish the next time anyone touched the
    model, and the two entry points would stop agreeing about one store.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    from abstractcore.config.manager import ConfigurationManager

    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    with client:
        assert ConfigurationManager().set_capability_default(
            "output",
            "text",
            provider="lmstudio",
            model="qwen3",
            base_url="http://127.0.0.1:1234/v1",
            reasoning="high",
            options={"profile": "local"},
        )

        updated = client.put(
            "/v1/config/capability-defaults/output/text",
            json={"provider": "lmstudio", "model": "qwen3-next"},
        )
        assert updated.status_code == 200, updated.text

        route = _route(client)
        assert route["model"] == "qwen3-next"
        assert route["reasoning"] == "high"
        assert route["base_url"] == "http://127.0.0.1:1234/v1"
        assert route["options"] == {"profile": "local"}


def test_the_server_config_route_writes_and_clears_the_reasoning_effort(monkeypatch, tmp_path) -> None:
    """The reasoning effort is settable through the server, and clearable by name."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    with client:
        assert client.put(
            "/v1/config/capability-defaults/output/text",
            json={"provider": "lmstudio", "model": "qwen3", "reasoning": "medium"},
        ).status_code == 200
        assert _route(client)["reasoning"] == "medium"

        assert client.put(
            "/v1/config/capability-defaults/output/text", json={"reasoning": ""}
        ).status_code == 200
        route = _route(client)
        assert "reasoning" not in route
        assert route["model"] == "qwen3", "clearing one field must not drop the row"


def test_the_server_config_route_never_persists_a_derived_row(monkeypatch, tmp_path) -> None:
    """`input.image` is answered by the text route until it is set for real.

    A partial update to a derived route must merge over what is STORED for that
    route -- nothing -- not over the row the reader was being shown, or coverage
    that follows the text route freezes into a copy of it.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    from abstractcore.config.manager import ConfigurationManager

    server_app = importlib.import_module("abstractcore.server.app")
    client = TestClient(server_app.app)

    with client:
        assert ConfigurationManager().set_capability_default(
            "output", "text", provider="lmstudio", model="qwen3-vl"
        )
        assert client.put(
            "/v1/config/capability-defaults/input/image", json={"provider": "mlx-vlm"}
        ).status_code == 200

        stored = ConfigurationManager().stored_capability_default("input", "image")
        assert stored.get("provider") == "mlx-vlm"
        assert "model" not in stored, "the text route's model must not be copied into input.image"
