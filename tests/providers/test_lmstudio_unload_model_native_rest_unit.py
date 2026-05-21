import httpx

from abstractcore.providers.lmstudio_provider import LMStudioProvider


def test_lmstudio_unload_model_posts_instance_id(monkeypatch) -> None:
    # Avoid any dependency on a running LMStudio server during provider init.
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)

    provider = LMStudioProvider(model="qwen/qwen3-4b-2507", base_url="http://localhost:1234/v1")

    called = {}

    def _fake_post(url: str, *, json=None, headers=None, timeout=None):  # noqa: ANN001
        called["url"] = url
        called["json"] = json
        called["headers"] = headers
        called["timeout"] = timeout
        req = httpx.Request("POST", url)
        return httpx.Response(200, json={"instance_id": json.get("instance_id")}, request=req)

    monkeypatch.setattr(httpx, "post", _fake_post)

    provider.unload_model("my-instance")

    assert called["url"] == "http://localhost:1234/api/v1/models/unload"
    assert called["json"] == {"instance_id": "my-instance"}


def test_lmstudio_load_model_posts_model_key(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen/qwen3-4b-2507", base_url="http://localhost:1234/v1")

    called = {}

    def _fake_post(url: str, *, json=None, headers=None, timeout=None):  # noqa: ANN001
        called["url"] = url
        called["json"] = json
        called["headers"] = headers
        called["timeout"] = timeout
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"type": "llm", "instance_id": json.get("model"), "status": "loaded"},
            request=req,
        )

    monkeypatch.setattr(httpx, "post", _fake_post)

    out = provider.load_model("qwen/qwen3-4b-2507")

    assert called["url"] == "http://localhost:1234/api/v1/models/load"
    assert called["json"] == {"model": "qwen/qwen3-4b-2507"}
    assert out["supported"] is True
    assert out["operation"] == "load"
    assert out["raw"]["status"] == "loaded"


def test_lmstudio_unload_model_resolves_loaded_instance_id_when_model_key_is_not_instance_id(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)

    provider = LMStudioProvider(model="qwen3.5-4b@q4_k_m", base_url="http://localhost:1234/v1")
    calls = []

    def _fake_post(url: str, *, json=None, headers=None, timeout=None):  # noqa: ANN001
        calls.append(("post", url, json, headers, timeout))
        req = httpx.Request("POST", url)
        if json == {"instance_id": "qwen3.5-4b@q4_k_m"}:
            return httpx.Response(400, json={"error": "instance not found"}, request=req)
        return httpx.Response(200, json={"instance_id": json.get("instance_id")}, request=req)

    def _fake_get(url: str, *, headers=None, timeout=None):  # noqa: ANN001
        calls.append(("get", url, None, headers, timeout))
        req = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "key": "qwen3.5-4b",
                        "selected_variant": "qwen3.5-4b@q4_k_m",
                        "variants": ["qwen3.5-4b@q4_k_m"],
                        "loaded_instances": [{"id": "loaded-instance-1"}],
                    }
                ]
            },
            request=req,
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    monkeypatch.setattr(httpx, "get", _fake_get)

    provider.unload_model("qwen3.5-4b@q4_k_m")

    post_payloads = [call[2] for call in calls if call[0] == "post"]
    assert post_payloads == [
        {"instance_id": "qwen3.5-4b@q4_k_m"},
        {"instance_id": "loaded-instance-1"},
    ]
    assert any(call[0] == "get" and call[1] == "http://localhost:1234/api/v1/models" for call in calls)


def test_lmstudio_unload_model_falls_back_when_success_status_contains_error(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)

    provider = LMStudioProvider(model="qwen3.5-4b", base_url="http://localhost:1234/v1")
    post_payloads = []

    def _fake_post(url: str, *, json=None, headers=None, timeout=None):  # noqa: ANN001
        post_payloads.append(json)
        req = httpx.Request("POST", url)
        if json == {"instance_id": "qwen3.5-4b"}:
            return httpx.Response(200, json={"error": "unsupported instance id"}, request=req)
        return httpx.Response(200, json={"instance_id": json.get("instance_id")}, request=req)

    def _fake_get(url: str, *, headers=None, timeout=None):  # noqa: ANN001
        req = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "key": "qwen3.5-4b",
                        "loaded_instances": [{"id": "resolved-instance"}],
                    }
                ]
            },
            request=req,
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    monkeypatch.setattr(httpx, "get", _fake_get)

    provider.unload_model("qwen3.5-4b")

    assert post_payloads == [
        {"instance_id": "qwen3.5-4b"},
        {"instance_id": "resolved-instance"},
    ]


def test_lmstudio_get_model_residency_reports_loaded_instances(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen3.5-4b", base_url="http://localhost:1234/v1")

    def _fake_get(url: str, *, headers=None, timeout=None):  # noqa: ANN001
        req = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={"models": [{"key": "qwen3.5-4b", "loaded_instances": [{"id": "loaded-instance"}]}]},
            request=req,
        )

    monkeypatch.setattr(httpx, "get", _fake_get)

    residency = provider.get_model_residency(model="qwen3.5-4b")

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is True
    assert residency["loaded"] is True
    assert residency["provider_instance_ids"] == ["loaded-instance"]
    assert residency["source"] == "abstractcore.provider.lmstudio.native_rest"


def test_lmstudio_get_model_residency_reports_not_loaded(monkeypatch) -> None:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    provider = LMStudioProvider(model="qwen3.5-4b", base_url="http://localhost:1234/v1")

    def _fake_get(url: str, *, headers=None, timeout=None):  # noqa: ANN001
        req = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={"models": [{"key": "qwen3.5-4b", "loaded_instances": []}]},
            request=req,
        )

    monkeypatch.setattr(httpx, "get", _fake_get)

    residency = provider.get_model_residency(model="qwen3.5-4b")

    assert residency["provider_residency_verified"] is True
    assert residency["provider_resident"] is False
    assert residency["loaded"] is False
    assert residency["state"] == "not_loaded"
