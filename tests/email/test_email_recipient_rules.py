"""Recipient rules: Always allowed / Always denied lists on top of the mode (round 3, §4/§13.3).

Precedence (one function, `evaluate`): own address -> allowed; Always denied -> refused;
Always allowed -> allowed; else the mode (allowlist -> refused, denylist -> allowed). A domain
covers its subdomains. To, Cc and Bcc are all checked; one refusal refuses the message.
"""

from __future__ import annotations

import json

import pytest

from abstractcore.comms.email import EmailAccountStore, EmailSecret
from abstractcore.comms.email.errors import EmailInvalidSettings, EmailPolicyRefused
from abstractcore.comms.email.policy import RecipientPolicy, evaluate

from email_fixtures import ME, PASSWORD, account_for

pytestmark = pytest.mark.basic


def _both(mode: str = "allowlist", allow=("abstractframework.ai",), deny=("denied.gov",)) -> RecipientPolicy:
    return RecipientPolicy.build(mode, (), always_allow=list(allow), always_deny=list(deny))


def test_deny_beats_allow_for_the_same_recipient() -> None:
    policy = _both(allow=("example.test",), deny=("bad.example.test", "boss@example.test"))
    d = evaluate(policy, to=["x@bad.example.test", "boss@example.test", "ok@example.test"])
    assert [(v.address, v.allowed, v.source) for v in d.verdicts] == [
        ("x@bad.example.test", False, "always_deny"),
        ("boss@example.test", False, "always_deny"),
        ("ok@example.test", True, "always_allow"),
    ]
    # Denied wins in both modes, and even when the same entry is on both lists.
    same = RecipientPolicy.build("denylist", (), always_allow=["denied.gov"], always_deny=["denied.gov"])
    assert evaluate(same, to=["x@denied.gov"]).allowed is False


def test_domain_covers_subdomains_by_dot_suffix_only() -> None:
    policy = _both(mode="allowlist")
    assert evaluate(policy, to=["anyone@abstractframework.ai"]).allowed
    assert evaluate(policy, to=["a@mail.abstractframework.ai"]).allowed
    assert not evaluate(policy, to=["a@notabstractframework.ai"]).allowed
    assert not evaluate(policy, to=["x@sub.denied.gov"]).allowed
    assert evaluate(_both(mode="denylist"), to=["x@undenied.gov"]).allowed


def test_cc_and_bcc_are_checked_and_one_refusal_refuses_the_message() -> None:
    policy = _both(mode="denylist")
    for field in ("to", "cc", "bcc"):
        kwargs = {"to": ["friend@example.test"], field: ["x@denied.gov"]} if field != "to" else {"to": ["x@denied.gov"]}
        d = evaluate(policy, **kwargs)
        assert d.allowed is False, field
        assert [(v.field, v.rule) for v in d.refused] == [(field, "denied.gov")]


def test_refusal_names_the_recipient_and_the_always_denied_rule() -> None:
    d = evaluate(_both(), to=["x@denied.gov"])
    with pytest.raises(EmailPolicyRefused) as info:
        d.raise_if_refused()
    assert info.value.cause == "Not sent: x@denied.gov is on your Always denied list (denied.gov)."
    exact = evaluate(_both(deny=("x@denied.gov",)), to=["X@Denied.gov"])
    with pytest.raises(EmailPolicyRefused) as info2:
        exact.raise_if_refused()
    assert info2.value.cause == "Not sent: X@Denied.gov is on your Always denied list."
    many = evaluate(_both(), to=["anyone@abstractframework.ai"], cc=["x@denied.gov"])
    with pytest.raises(EmailPolicyRefused) as info3:
        many.raise_if_refused()
    assert info3.value.cause.startswith("Not sent: x@denied.gov is on your Always denied list (denied.gov).")
    assert "nothing was sent to anyone" in info3.value.cause


def test_mode_decides_only_recipients_on_neither_list() -> None:
    assert not evaluate(_both(mode="allowlist"), to=["stranger@else.test"]).allowed
    assert evaluate(_both(mode="denylist"), to=["stranger@else.test"]).allowed


def test_own_address_is_always_allowed_even_when_denied_in_denylist_mode() -> None:
    """The latent bug: self used to be merged into the mode's matched set, so a denylist DENIED it."""
    policy = RecipientPolicy.build("denylist", (), always_deny=["me@example.test", "example.test"])
    assert evaluate(policy, to=["me@example.test"], self_addresses=["Me@Example.test"]).allowed is True
    assert evaluate(policy, to=["me@example.test"]).allowed is False
    assert evaluate(policy, to=["other@example.test"], self_addresses=["me@example.test"]).allowed is False
    v = evaluate(policy, to=["me@example.test"], self_addresses=["me@example.test"]).verdicts[0]
    assert (v.source, v.reason) == ("self", "your own address")


def test_entries_are_structurally_validated_without_patterns() -> None:
    p = _both(allow=("Name@Example.TEST", "@corp.test", "bücher.example"), deny=())
    assert p.always_allow == ("name@example.test", "corp.test", "xn--bcher-kva.example")
    for bad in ("*.gov", "gov", "a@@b.test", "exa mple.test", "-x.test", "a@b"):
        with pytest.raises(EmailInvalidSettings):
            _both(allow=(bad,), deny=())
        with pytest.raises(EmailInvalidSettings):
            _both(allow=(), deny=(bad,))


def test_migration_old_allowlist_and_denylist_load_into_the_lists() -> None:
    old_allow = RecipientPolicy.from_dict({"mode": "allowlist", "entries": ["me@example.test", "corp.test"]})
    assert old_allow.mode == "allowlist"
    assert old_allow.always_allow == ("me@example.test", "corp.test") and old_allow.always_deny == ()
    assert evaluate(old_allow, to=["me@example.test", "a@corp.test"]).allowed
    assert not evaluate(old_allow, to=["x@other.test"]).allowed
    old_deny = RecipientPolicy.from_dict({"mode": "denylist", "entries": ["spam.test"]})
    assert old_deny.always_deny == ("spam.test",) and old_deny.always_allow == ()
    assert not evaluate(old_deny, to=["x@spam.test"]).allowed and evaluate(old_deny, to=["x@ok.test"]).allowed
    # The stored shape keeps `entries` = the mode's list (older readers keep their meaning).
    doc = _both(mode="denylist").to_dict()
    assert doc == {"mode": "denylist", "entries": ["denied.gov"], "always_allow": ["abstractframework.ai"], "always_deny": ["denied.gov"]}
    assert RecipientPolicy.from_dict(doc) == _both(mode="denylist")


def test_store_keeps_an_old_allowlist_and_replaces_lists_when_given(config_file) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"email": {"policy": {"mode": "allowlist", "entries": ["me@example.test", "corp.test"]}}}))
    store = EmailAccountStore(config_file)
    st = store.settings()
    assert st.policy.always_allow == ("me@example.test", "corp.test") and not st.policy_is_default
    doc = store.set_policy(always_deny=["denied.gov"])
    assert doc["policy"]["always_allow"] == ["me@example.test", "corp.test"]
    assert doc["policy"]["always_deny"] == ["denied.gov"]
    doc = store.set_policy(mode="denylist", always_allow=["abstractframework.ai"])
    assert doc["policy"]["mode"] == "denylist" and doc["policy"]["entries"] == ["denied.gov"]
    assert doc["policy"]["always_allow"] == ["abstractframework.ai"]
    stored = json.loads(config_file.read_text())["email"]["policy"]
    assert stored["always_deny"] == ["denied.gov"] and stored["always_allow"] == ["abstractframework.ai"]


def test_agent_send_goes_through_the_rules_and_self_is_allowed(imap, smtp, ca, config_file) -> None:
    from abstractcore.tools import comms_tools as t

    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    store.set_agent_tools(True)
    domain = ME.split("@", 1)[1]
    store.set_policy(mode="denylist", clear=True, always_allow=["abstractframework.ai"], always_deny=[domain, "denied.gov"])
    refused = t.send_email(to="anyone@abstractframework.ai", cc="x@denied.gov", subject="s", body_text="b")
    assert refused["success"] is False and refused["error_code"] == "email_policy_refused"
    assert "Not sent: x@denied.gov is on your Always denied list (denied.gov)." in json.dumps(refused)
    assert smtp.messages == []
    # The own address is allowed although its domain is on Always denied.
    ok = t.send_email(to=ME, bcc="anyone@abstractframework.ai", subject="s", body_text="b")
    assert ok["success"], ok
    assert len(smtp.messages) == 1
