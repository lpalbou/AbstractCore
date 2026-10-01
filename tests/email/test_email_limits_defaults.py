"""Send-limit defaults (100 per hour, 1000 per day) and how stored limits survive the change.

Rule (`store.limits_source`): nothing stored follows the defaults; limits a user set carry
`set_by: user` and are never changed; values stored by 2.21 or earlier without the marker are
kept as stored, because a stored 20 / 100 cannot be told apart from a user who chose it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abstractcore.comms.email import EmailAccountStore, EmailSecret
from abstractcore.comms.email.models import DEFAULT_PER_DAY, DEFAULT_PER_HOUR

from email_fixtures import ME, PASSWORD, account_for

pytestmark = pytest.mark.basic


def _write_section(config_file: Path, limits) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(config_file.read_text()) if config_file.is_file() else {}
    doc.setdefault("email", {})["limits"] = limits
    config_file.write_text(json.dumps(doc))


def _stored(config_file: Path):
    return json.loads(config_file.read_text())["email"]["limits"]


def _lim(store: EmailAccountStore):
    st = store.settings()
    return (st.limits.per_hour, st.limits.per_day, st.limits_source)


def test_the_defaults_are_100_per_hour_and_1000_per_day() -> None:
    assert (DEFAULT_PER_HOUR, DEFAULT_PER_DAY) == (100, 1000)


def test_nothing_stored_follows_the_defaults(config_file) -> None:
    store = EmailAccountStore(config_file)
    assert _lim(store) == (100, 1000, "default")
    assert store.public()["limits"] is not None and store.public()["limits"]["source"] == "default"


def test_connect_does_not_store_the_defaults(config_file, imap, smtp, ca) -> None:
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    assert _stored(config_file) == {}
    assert _lim(store) == (100, 1000, "default")


def test_a_user_set_old_default_is_kept(config_file) -> None:
    store = EmailAccountStore(config_file)
    store.set_limits(per_hour=20, per_day=100)
    assert _stored(config_file) == {"per_hour": 20, "per_day": 100, "set_by": "user"}
    assert _lim(EmailAccountStore(config_file)) == (20, 100, "user")


def test_an_unmarked_old_default_pair_follows_the_new_defaults(config_file) -> None:
    # 2.21 stored {20, 100} at connect (operator ruling 2026-10-01): it is the old default -> 100 / 1000.
    _write_section(config_file, {"per_hour": 20, "per_day": 100})
    store = EmailAccountStore(config_file)
    assert _lim(store) == (100, 1000, "default")
    assert _stored(config_file) == {"per_hour": 20, "per_day": 100}  # reading writes nothing


def test_an_unmarked_half_old_default_is_kept_as_legacy(config_file) -> None:
    _write_section(config_file, {"per_hour": 20, "per_day": 300})
    assert _lim(EmailAccountStore(config_file)) == (20, 300, "legacy")


def test_a_legacy_custom_value_is_kept(config_file) -> None:
    _write_section(config_file, {"per_hour": 5, "per_day": 50})
    assert _lim(EmailAccountStore(config_file)) == (5, 50, "legacy")


def test_setting_one_window_stores_only_that_window(config_file) -> None:
    store = EmailAccountStore(config_file)
    store.set_limits(per_hour=7)
    assert _stored(config_file) == {"per_hour": 7, "set_by": "user"}
    assert _lim(store) == (7, 1000, "user")
    store.set_limits(per_day=70)
    assert _stored(config_file) == {"per_hour": 7, "per_day": 70, "set_by": "user"}


def test_setting_a_window_on_a_legacy_record_marks_it_user_and_keeps_the_other(config_file) -> None:
    _write_section(config_file, {"per_hour": 20, "per_day": 100})
    store = EmailAccountStore(config_file)
    store.set_limits(per_hour=30)
    assert _stored(config_file) == {"per_hour": 30, "per_day": 100, "set_by": "user"}


def test_reset_follows_the_defaults_again(config_file, capsys) -> None:
    from abstractcore.config.email_cli import handle_email

    _write_section(config_file, {"per_hour": 30, "per_day": 300})
    assert handle_email(["limits", "show"]) == 0
    assert "stored by an earlier version" in capsys.readouterr().out
    assert handle_email(["limits", "reset", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["per_hour"], out["per_day"], out["source"]) == (100, 1000, "default")
    assert _stored(config_file) == {}


def test_connect_keeps_a_stored_user_value(config_file, imap, smtp, ca) -> None:
    store = EmailAccountStore(config_file)
    store.set_limits(per_hour=3, per_day=9)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    assert _lim(store) == (3, 9, "user")
