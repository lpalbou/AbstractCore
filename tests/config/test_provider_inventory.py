"""THE PROVIDER LIST IS THE REGISTRY, NOT THE KEY STORE.

Every surface that answered "which providers do I have" by enumerating the
`api_keys` config section hid every provider that needs no key. The operator
asked the question the defect makes unanswerable:

    "how come we don't have ollama, lmstudio, huggingface and mlx?"  (2026-08-01)

`api_keys` is where secrets live; it has a row for `google` (which is not a
provider at all) and no row for ollama, lmstudio, mlx, huggingface or any media
engine (which are). `provider_inventory` is the list itself, drawn from the
provider registry -- the same source as Core's own "Unknown provider: x.
Available providers: ..." refusal -- plus the media/engine backends the
weights surface already knows about.

WHAT MATTERS DIFFERS PER PROVIDER, which is why each row carries a `kind` and
its own state: a key for cloud APIs, a base URL (and whether anything answers
there) for local servers, and nothing at all for local engines. A single
"status" column that treated those as one question is what made the old screen
useless even for the providers it did list.
"""

from __future__ import annotations

import json

from abstractcore.config import model_materializer
from abstractcore.config.manager import ConfigurationManager


def _inventory(tmp_path, **kwargs):
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    return manager, {
        row["provider"]: row for row in model_materializer.provider_inventory(manager, **kwargs)
    }


def test_keyless_providers_are_listed(tmp_path) -> None:
    _, rows = _inventory(tmp_path)
    for provider in ("ollama", "lmstudio", "mlx", "huggingface"):
        assert provider in rows, f"{provider} needs no API key and must still be listed"


def test_media_engines_are_listed(tmp_path) -> None:
    _, rows = _inventory(tmp_path)
    for provider in ("mlx-gen", "supertonic", "diffusers"):
        assert provider in rows
        assert rows[provider]["kind"] == "local_engine"
        assert rows[provider]["auth"] == "none"


def test_the_llm_list_matches_the_registry_core_would_refuse_against(tmp_path) -> None:
    """No second list. A surface must not offer what Core will reject."""
    from abstractcore.providers.registry import get_provider_registry

    _, rows = _inventory(tmp_path)
    for name in get_provider_registry().list_provider_names():
        assert name in rows


def test_no_row_for_a_provider_that_does_not_exist(tmp_path) -> None:
    """`api_keys.google` is a reserved FIELD; there is no google provider.

    The old screen listed it because it listed `api_keys`. A registry-driven
    list cannot invent it -- and the field stays editable where every other raw
    field is.
    """
    manager, rows = _inventory(tmp_path)
    assert "google" not in rows
    assert hasattr(manager.config.api_keys, "google"), "the reserved key field still exists"


def test_local_servers_carry_their_address_and_its_source(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    _, rows = _inventory(tmp_path)
    assert rows["lmstudio"]["kind"] == "local_server"
    assert rows["lmstudio"]["base_url"] == "http://localhost:1234/v1"
    assert rows["lmstudio"]["base_url_source"] == "default"
    assert rows["ollama"]["base_url"] == "http://localhost:11434"

    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://box.local:9999/v1")
    _, rows = _inventory(tmp_path)
    assert rows["lmstudio"]["base_url"] == "http://box.local:9999/v1"
    assert rows["lmstudio"]["base_url_source"] == "env:LMSTUDIO_BASE_URL", (
        "the row says WHERE the address came from, not just what it is"
    )


def test_vllm_has_no_assumed_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    _, rows = _inventory(tmp_path)
    assert rows["vllm"]["base_url"] == "", "guessing a vLLM address would be a lie"


def test_a_key_is_reported_by_presence_and_fingerprint_never_by_value(tmp_path) -> None:
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    manager.set_api_key("openai", "sk-live-topsecret-body")
    rows = {row["provider"]: row for row in model_materializer.provider_inventory(manager)}

    row = rows["openai"]
    assert row["api_key_set"] is True
    assert row["api_key_source"] == "config"
    assert row["api_key_field"] == "openai"
    assert row["api_key_env_var"] == "OPENAI_API_KEY"
    assert row["api_key_fingerprint"] and row["api_key_fingerprint"] != "sk-live-topsecret-body"
    assert "sk-live-topsecret-body" not in json.dumps(rows), "no key material anywhere in the payload"


def test_a_bare_env_var_still_counts_as_configured(tmp_path, monkeypatch) -> None:
    """The provider will read it, so the row must not say "no key"."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-the-environment")
    _, rows = _inventory(tmp_path)
    assert rows["anthropic"]["api_key_set"] is True
    assert rows["anthropic"]["api_key_source"] == "env:ANTHROPIC_API_KEY"


def test_huggingface_names_hf_token_for_gated_repos(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    _, rows = _inventory(tmp_path)
    assert rows["huggingface"]["auth"] == "optional"
    assert rows["huggingface"]["api_key_env_var"] == "HF_TOKEN"
    assert rows["huggingface"]["api_key_set"] is False
    assert "gated" in rows["huggingface"]["note"]

    monkeypatch.setenv("HF_TOKEN", "hf_sometoken")
    _, rows = _inventory(tmp_path)
    assert rows["huggingface"]["api_key_set"] is True


def test_endpoint_profiles_ride_the_same_list_with_their_own_key(tmp_path) -> None:
    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    manager.set_provider_profile("ovh", base_url="https://ovh.invalid/v1", api_key="sk-profile-key")
    rows = {row["provider"]: row for row in model_materializer.provider_inventory(manager)}

    row = rows["endpoint:ovh"]
    assert row["kind"] == "endpoint_profile"
    assert row["base_url"] == "https://ovh.invalid/v1"
    assert row["api_key_set"] is True
    assert "sk-profile-key" not in json.dumps(rows)


def test_probe_is_opt_in_and_never_guesses(tmp_path) -> None:
    """`None` (not probed) and `False` (probed, silent) are different answers."""
    _, unprobed = _inventory(tmp_path)
    assert unprobed["lmstudio"]["reachable"] is None
    assert unprobed["lmstudio"]["reachability"] == ""

    manager = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    manager.set_provider_profile("dead", base_url="http://127.0.0.1:9/v1")
    rows = {
        row["provider"]: row
        for row in model_materializer.provider_inventory(manager, probe=True)
    }
    assert rows["endpoint:dead"]["reachable"] is False
    assert rows["endpoint:dead"]["reachability"], "an unreachable server says WHY"


def test_the_cli_lists_providers_and_says_how_to_probe(tmp_path, capsys) -> None:
    from abstractcore.config import main as config_main

    cfg = tmp_path / "abstractcore.json"
    assert config_main.main(["config", "--config-file", str(cfg), "providers"]) == 0
    out = capsys.readouterr().out
    for provider in ("ollama", "lmstudio", "mlx", "huggingface"):
        assert provider in out
    assert "local server" in out
    assert "local engine" in out
    assert "--probe" in out, "the listing names the way to check reachability"
