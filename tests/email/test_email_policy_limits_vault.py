"""Recipient policy, send limits and the credential vault (backlog 0992 WP1) — pure units."""

from __future__ import annotations

import base64
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


def test_domain_entry_covers_its_subdomains_by_dot_suffix_only() -> None:
    """Round 3 recipient rules: "A domain also covers its subdomains" (dot-suffix, never a
    bare string suffix)."""
    policy = RecipientPolicy.build("allowlist", ["corp.test"])
    decision = evaluate(policy, to=["a@corp.test", "b@eu.corp.test", "c@badcorp.test"])
    assert not decision.allowed
    assert [v.address for v in decision.refused] == ["c@badcorp.test"]


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
    assert p.with_changes(add=["example.org"]).entries == ("a@example.test", "example.org")
    # add/remove/clear act on the list the (new) mode uses: denylist -> Always denied.
    p2 = p.with_changes(add=["example.org", "a@example.test"], mode="denylist")
    assert p2.mode == "denylist" and p2.entries == ("example.org", "a@example.test")
    assert p2.always_allow == ("a@example.test",)  # the Allowed list is kept
    assert p2.with_changes(remove=["A@EXAMPLE.test"]).entries == ("example.org",)
    assert p2.with_changes(clear=True).always_deny == () and p2.with_changes(clear=True).always_allow == ("a@example.test",)
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
    assert SendLimits().to_dict() == {"per_hour": 100, "per_day": 1000}
    assert SendLimits.from_dict({}).to_dict() == {"per_hour": 100, "per_day": 1000}
    with pytest.raises(EmailInvalidSettings):
        SendLimits.build(-1, 10)
    with pytest.raises(EmailInvalidSettings):
        SendLimits.build("many", 10)


# ------------------------------------------------------------------ vault


def test_vault_seals_with_the_key_file_and_never_stores_plaintext(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path / "email")
    assert vault.key_file == tmp_path / "secrets" / "sealing.key"
    assert not vault.key_file.exists()
    secret = EmailSecret("sentinel-Pa55word-7f3a9c")
    assert vault.store(secret.sealed_payload()) == "sealing-key"
    for path in (vault.sealed_path, vault.key_file):
        assert b"sentinel-Pa55word-7f3a9c" not in path.read_bytes()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "email").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "secrets").stat().st_mode) == 0o700
    assert len(base64.b64decode(vault.key_file.read_bytes().strip())) == 32
    assert not vault.key_path.exists() and vault.key_warning == ""
    assert EmailSecret.from_sealed_payload(vault.load()).password == "sentinel-Pa55word-7f3a9c"
    assert vault.permissions_ok() and vault.location() == "sealing-key"


def test_one_key_file_serves_every_store_and_a_moved_folder(tmp_path: Path) -> None:
    import shutil

    key = tmp_path / "data" / "secrets" / "sealing.key"
    a = SecretVault(tmp_path / "data" / "a", key_file=key)
    b = SecretVault(tmp_path / "data" / "x" / "b", key_file=key)
    a.store({"password": "pa"})
    b.store({"password": "pb"})
    assert json.loads(a.sealed_path.read_text())["kid"] == json.loads(b.sealed_path.read_text())["kid"]
    # A sealed file copied into another store does not open (its place is the associated data).
    shutil.copy(a.sealed_path, b.sealed_path)
    with pytest.raises(EmailSecretUnavailable):
        b.load()
    # The folder moved with its secrets/: still opens. Without secrets/: a typed sentence.
    shutil.copytree(tmp_path / "data", tmp_path / "moved")
    moved = SecretVault(tmp_path / "moved" / "a", key_file=tmp_path / "moved" / "secrets" / "sealing.key")
    assert moved.load() == {"password": "pa"}
    shutil.rmtree(tmp_path / "moved" / "secrets")
    with pytest.raises(EmailSecretUnavailable) as info:
        moved.load()
    assert "sealing key" in info.value.cause


def test_the_keychain_is_gone(tmp_path: Path, memory_keyring) -> None:
    with pytest.raises(EmailInvalidSettings):
        SecretVault(tmp_path / "email", key_backend="keyring")
    assert SecretVault(tmp_path / "email", key_backend="file").store({"password": "x"}) == "sealing-key"
    # A store sealed by an older version with the key in a keychain is NEVER opened.
    old = SecretVault(tmp_path / "old")
    old.directory.mkdir()
    old.sealed_path.write_text(json.dumps({"v": 1, "alg": "AES-256-GCM", "key": "keyring", "key_id": "0" * 24,
                                           "nonce": "AAAAAAAAAAAAAAAA", "ct": "AAAA"}))
    assert old.legacy_keychain() and old.location() == "keyring"
    with pytest.raises(EmailSecretUnavailable) as info:
        old.load()
    assert info.value.code == "email_secret_sealed_with_keychain" and "connect the account again" in info.value.cause
    assert old.retire_legacy_keychain() and not old.exists() and (old.directory / "secret.keychain-old.enc").exists()
    assert memory_keyring.attempts == []
    import abstractcore.comms.email.vault as vault_module

    assert "import keyring" not in Path(vault_module.__file__).read_text()


def test_an_older_key_file_store_is_resealed_under_the_sealing_key(tmp_path: Path) -> None:
    import secrets as _secrets

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    # Exactly what AbstractCore < 2.26 wrote without a keychain: secret.key beside secret.enc.
    d = tmp_path / "email"
    d.mkdir()
    key = AESGCM.generate_key(bit_length=256)
    (d / "secret.key").write_bytes(base64.b64encode(key))
    nonce = _secrets.token_bytes(12)
    key_id = "f" * 24
    ct = AESGCM(key).encrypt(nonce, json.dumps({"password": "old-pw"}).encode(), b"abstractcore-email-secret-v1|" + key_id.encode())
    (d / "secret.enc").write_text(json.dumps({"v": 1, "alg": "AES-256-GCM", "key": "file", "key_id": key_id,
                                              "nonce": base64.b64encode(nonce).decode(), "ct": base64.b64encode(ct).decode()}))
    readonly = SecretVault(d, reseal_legacy=False)
    assert readonly.load() == {"password": "old-pw"} and readonly.location() == "file"
    vault = SecretVault(d)
    assert vault.load() == {"password": "old-pw"}
    assert vault.location() == "sealing-key" and not (d / "secret.key").exists()
    assert vault.load() == {"password": "old-pw"}


def test_vault_tampering_and_missing_key_are_typed_errors(tmp_path: Path) -> None:
    vault = SecretVault(tmp_path / "email")
    vault.store({"password": "x"})
    doc = json.loads(vault.sealed_path.read_text())
    doc["ct"] = doc["ct"][:-4] + ("AAAA" if not doc["ct"].endswith("AAAA") else "BBBB")
    vault.sealed_path.write_text(json.dumps(doc))
    with pytest.raises(EmailSecretUnavailable):
        vault.load()
    vault.store({"password": "y"})
    vault.key_file.unlink()  # the folder copied without secrets/
    with pytest.raises(EmailSecretUnavailable) as info:
        vault.load()
    assert "sealing key" in info.value.cause


def test_secret_repr_and_str_redact() -> None:
    s = EmailSecret("hunter2-secret", refresh_token="rt-secret", client_secret="cs-secret")
    for text in (repr(s), str(s), f"{s}"):
        assert "hunter2" not in text and "rt-secret" not in text and "cs-secret" not in text
    import pickle

    with pytest.raises(TypeError):
        pickle.dumps(s)


def test_self_follows_the_registered_address_in_allowlist_mode() -> None:
    """`self_addresses` are always allowed, whatever the mode."""
    from abstractcore.comms.email.policy import RecipientPolicy, evaluate

    policy = RecipientPolicy.default_for("old@example.test")
    stale = evaluate(policy, to=["new@example.test"])
    assert stale.allowed is False
    fresh = evaluate(policy, to=["new@example.test"], self_addresses=("new@example.test", "mailbox@example.test"))
    assert fresh.allowed is True
    own = evaluate(policy, to=["mailbox@example.test"], self_addresses=("new@example.test", "mailbox@example.test"))
    assert own.allowed is True
    # Anyone else is still refused.
    assert evaluate(policy, to=["boss@example.test"], self_addresses=("new@example.test",)).allowed is False
    # Self is its own precedence step (round 3, recipient rules): the own address is allowed in
    # denylist mode too, even when the denylist names it; before, self was merged into the
    # mode's matched set and a denylist therefore DENIED it.
    deny = RecipientPolicy.build("denylist", ["new@example.test"])
    assert evaluate(deny, to=["new@example.test"], self_addresses=("new@example.test",)).allowed is True
    assert evaluate(deny, to=["new@example.test"]).allowed is False
