"""Recipient policy, send limits and the credential vault (backlog 0992 WP1) — pure units."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from abstractcore.comms.email import (
    EmailPolicyRefused,
    EmailRateLimited,
    EmailSecret,
    EmailSecretUnavailable,
    RecipientPolicy,
    SecretVault,
    SendLimits,
    SendRateLimiter,
    evaluate,
    normalize_address,
    parse_recipients,
)
from abstractcore.comms.email.errors import EmailInvalidSettings

pytestmark = pytest.mark.basic


# ------------------------------------------------------------------ policy


def test_normalisation_is_case_insensitive_and_idna() -> None:
    assert normalize_address("Me@Example.TEST") == "me@example.test"
    assert normalize_address("x@bücher.example") == "x@xn--bcher-kva.example"
    assert normalize_address("x@example.test.") == "x@example.test"
    policy = RecipientPolicy.build("allowlist", ["bücher.example", "Boss@Corp.Test"])
    assert policy.entries == ("xn--bcher-kva.example", "boss@corp.test")
    assert evaluate(policy, to=["BOSS@corp.test", "anyone@XN--BCHER-KVA.example"]).allowed


def test_domain_entry_matches_that_domain_only_not_subdomains() -> None:
    policy = RecipientPolicy.build("allowlist", ["corp.test"])
    decision = evaluate(policy, to=["a@corp.test", "b@eu.corp.test"])
    assert not decision.allowed
    assert [v.address for v in decision.refused] == ["b@eu.corp.test"]
    policy2 = RecipientPolicy.build("allowlist", ["corp.test", "eu.corp.test"])
    assert evaluate(policy2, to=["b@eu.corp.test"]).allowed


def test_allowlist_refuses_the_whole_message_naming_addresses_fields_and_rule() -> None:
    policy = RecipientPolicy.build("allowlist", ["me@example.test"])
    decision = evaluate(policy, to=["me@example.test"], cc=["x@other.test"], bcc=["hidden@other.test"])
    assert not decision.allowed
    with pytest.raises(EmailPolicyRefused) as info:
        decision.raise_if_refused()
    err = info.value
    assert "x@other.test (cc: not in the allowlist)" in err.cause
    assert "hidden@other.test (bcc: not in the allowlist)" in err.cause
    assert "nothing was sent" in err.cause
    refused = err.details["refused"]
    assert {r["field"] for r in refused} == {"cc", "bcc"}
    assert all(r["rule"] == "allowlist" for r in refused)


def test_denylist_refuses_listed_addresses_and_domains_with_the_matching_rule() -> None:
    policy = RecipientPolicy.build("denylist", ["@competitor.test", "ex@example.test"])
    d = evaluate(policy, to=["friend@example.test", "Ex@Example.test"], bcc=["spy@competitor.test"])
    refused = {v.address: v.rule for v in d.refused}
    assert refused == {"Ex@Example.test": "ex@example.test", "spy@competitor.test": "competitor.test"}
    assert evaluate(policy, to=["friend@example.test"]).allowed


def test_no_pattern_language_and_no_partial_matches() -> None:
    for bad in ("*@example.test", "*.example.test", "example.*", "^a@b.test$", "a|b.test", "exa?ple.test"):
        with pytest.raises(EmailInvalidSettings):
            RecipientPolicy.build("allowlist", [bad])
    policy = RecipientPolicy.build("allowlist", ["example.test"])
    # A domain that merely ends with the entry's text is a different domain.
    assert not evaluate(policy, to=["a@notexample.test"]).allowed


def test_empty_message_and_invalid_addresses_are_refused() -> None:
    policy = RecipientPolicy.build("denylist", [])
    with pytest.raises(EmailPolicyRefused):
        evaluate(policy).raise_if_refused()
    d = evaluate(policy, to=["not an address"])
    assert not d.allowed and d.refused[0].reason == "not a valid email address"


def test_default_policy_is_an_allowlist_of_the_registered_address() -> None:
    p = RecipientPolicy.default_for("Me@Example.test")
    assert p.mode == "allowlist" and p.entries == ("me@example.test",)
    assert evaluate(p, to=["me@example.test"]).allowed
    assert not evaluate(p, to=["other@example.test"]).allowed


def test_display_names_are_ignored_by_recipient_parsing() -> None:
    assert parse_recipients('"Doe, John" <john@example.test>, jane@example.test; x@y.test') == [
        "john@example.test",
        "jane@example.test",
        "x@y.test",
    ]
    with pytest.raises(ValueError):
        parse_recipients("not-an-address")


def test_policy_edits_add_remove_mode() -> None:
    p = RecipientPolicy.build("allowlist", ["a@example.test"])
    p2 = p.with_changes(add=["example.org"], mode="denylist")
    assert p2.mode == "denylist" and p2.entries == ("a@example.test", "example.org")
    assert p2.with_changes(remove=["A@EXAMPLE.test"]).entries == ("example.org",)
    with pytest.raises(EmailInvalidSettings):
        p2.with_changes(remove=["missing@example.test"])


# ------------------------------------------------------------------ limits


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


def test_hour_and_day_windows_are_enforced_and_durable(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "sends.json"
    limiter = SendRateLimiter(path, SendLimits.build(2, 3), clock=clock)
    limiter.reserve()
    limiter.reserve()
    with pytest.raises(EmailRateLimited) as info:
        limiter.reserve()
    assert info.value.details["window"] == "hour" and info.value.details["limit"] == 2
    assert info.value.details["retry_after_s"] == pytest.approx(3600.0)
    # A new limiter over the same file (another process, a restart) sees the same count.
    clock.t += 3601
    again = SendRateLimiter(path, SendLimits.build(2, 3), clock=clock)
    again.reserve()
    with pytest.raises(EmailRateLimited) as info2:
        again.reserve()
    assert info2.value.details["window"] == "day"
    clock.t += 86401
    again.reserve()  # the day window has passed
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_refund_returns_a_slot_and_zero_means_no_sending(tmp_path: Path) -> None:
    clock = Clock()
    limiter = SendRateLimiter(tmp_path / "s.json", SendLimits.build(1, 10), clock=clock)
    token = limiter.reserve()
    limiter.refund(token)
    limiter.reserve()
    none = SendRateLimiter(tmp_path / "z.json", SendLimits.build(0, 10), clock=clock)
    with pytest.raises(EmailRateLimited):
        none.reserve()
    assert limiter.usage()["used_last_hour"] == 1


def test_limits_validation() -> None:
    assert SendLimits().to_dict() == {"per_hour": 20, "per_day": 100}
    with pytest.raises(EmailInvalidSettings):
        SendLimits.build(-1, 10)
    with pytest.raises(EmailInvalidSettings):
        SendLimits.build("many", 10)


# ------------------------------------------------------------------ vault


def test_vault_file_backend_seals_and_never_stores_plaintext(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path / "email", key_backend="file")
    secret = EmailSecret("sentinel-Pa55word-7f3a9c")
    assert vault.store(secret.sealed_payload()) == "file"
    for path in (vault.sealed_path, vault.key_path):
        assert b"sentinel-Pa55word-7f3a9c" not in path.read_bytes()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "email").stat().st_mode) == 0o700
    assert EmailSecret.from_sealed_payload(vault.load()).password == "sentinel-Pa55word-7f3a9c"
    assert vault.permissions_ok()


def test_vault_keyring_backend_keeps_the_key_out_of_the_folder(tmp_path: Path, memory_keyring) -> None:
    vault = SecretVault(tmp_path / "email", key_backend="auto")
    assert vault.store({"password": "pw-1"}) == "keyring"
    assert not vault.key_path.exists()
    assert len(memory_keyring.items) == 1
    assert vault.load()["password"] == "pw-1"
    vault.delete()
    assert not vault.sealed_path.exists() and memory_keyring.items == {}


def test_vault_falls_back_to_a_key_file_when_no_keychain(tmp_path: Path, monkeypatch) -> None:
    import keyring
    from keyring.backends import fail

    keyring.set_keyring(fail.Keyring())
    vault = SecretVault(tmp_path / "email")
    assert vault.store({"password": "x"}) == "file"
    assert "0600 file" in vault.key_warning
    with pytest.raises(EmailSecretUnavailable):
        SecretVault(tmp_path / "other", key_backend="keyring").store({"password": "x"})


def test_vault_tampering_and_missing_key_are_typed_errors(tmp_path: Path, memory_keyring) -> None:
    vault = SecretVault(tmp_path / "email")
    vault.store({"password": "x"})
    doc = json.loads(vault.sealed_path.read_text())
    doc["ct"] = doc["ct"][:-4] + ("AAAA" if not doc["ct"].endswith("AAAA") else "BBBB")
    vault.sealed_path.write_text(json.dumps(doc))
    with pytest.raises(EmailSecretUnavailable):
        vault.load()
    vault.store({"password": "y"})
    memory_keyring.items.clear()  # keychain entry gone (moved folder, other machine)
    with pytest.raises(EmailSecretUnavailable) as info:
        vault.load()
    assert "keychain" in info.value.cause


def test_secret_repr_and_str_redact() -> None:
    s = EmailSecret("hunter2-secret", refresh_token="rt-secret", client_secret="cs-secret")
    for text in (repr(s), str(s), f"{s}"):
        assert "hunter2" not in text and "rt-secret" not in text and "cs-secret" not in text
    import pickle

    with pytest.raises(TypeError):
        pickle.dumps(s)
