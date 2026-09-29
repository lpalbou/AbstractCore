"""OAuth2 sign-in (backlog 0992 WP1, operator addition): XOAUTH2 on IMAP and SMTP, token
refresh and rotation, re-authorisation errors, device-code and loopback flows through the CLI.

The token endpoint is a local https fake signed by the throwaway CA; Google and Microsoft are
never contacted (their presets are only checked as data).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from abstractcore.comms.email import (
    EmailAccount,
    EmailClient,
    EmailOAuthFailed,
    EmailOAuthReauthorize,
    EmailSecret,
    ImapSettings,
    OAuthSettings,
    OAuthTokenClient,
    OAuthTokenProvider,
    SmtpSettings,
    provider_preset,
    xoauth2_string,
)
from abstractcore.comms.email.errors import EmailInvalidSettings

from email_fixtures import ME, PASSWORD

pytestmark = pytest.mark.basic


def oauth_settings(server, **kw) -> OAuthSettings:
    return OAuthSettings.build(
        "custom",
        server.client_id,
        token_endpoint=f"{server.base_url}/token",
        device_authorization_endpoint=f"{server.base_url}/device",
        authorization_endpoint=f"{server.base_url}/authorize",
        scopes=["mail"],
        **kw,
    )


def oauth_account(imap, smtp, ca, server) -> EmailAccount:
    return EmailAccount.build(
        address=ME,
        imap=ImapSettings.build("localhost", port=imap.port, security="ssl", ca_file=str(ca.ca_pem)) if imap else None,
        smtp=SmtpSettings.build("localhost", port=smtp.port, security=smtp.security, ca_file=str(ca.ca_pem)) if smtp else None,
        auth_kind="oauth2",
        oauth=oauth_settings(server),
    )


def test_presets_are_data_for_google_and_microsoft() -> None:
    g = provider_preset("google")
    assert g["imap"]["host"] == "imap.gmail.com" and g["smtp"]["host"] == "smtp.gmail.com"
    assert g["scopes"] == ["https://mail.google.com/"] and g["default_flow"] == "loopback"
    m = provider_preset("microsoft", tenant="contoso.example")
    assert m["token_endpoint"] == "https://login.microsoftonline.com/contoso.example/oauth2/v2.0/token"
    assert "https://outlook.office.com/SMTP.Send" in m["scopes"] and "offline_access" in m["scopes"]
    assert m["smtp"] == {"host": "smtp.office365.com", "port": 587, "security": "starttls"}


def test_endpoints_must_be_https_and_contexts_must_verify(ca) -> None:
    with pytest.raises(EmailInvalidSettings):
        OAuthSettings.build("custom", "cid", token_endpoint="http://127.0.0.1:1/token", scopes=["s"])
    import ssl

    oauth = OAuthSettings.build("custom", "cid", token_endpoint="https://localhost:1/token", scopes=["s"])
    with pytest.raises(EmailInvalidSettings):
        OAuthTokenClient(oauth, verify=ssl._create_unverified_context())


def test_xoauth2_sign_in_on_imap_and_smtp_with_refresh(imap, smtp, ca, oauth_server, tokens) -> None:
    refresh = oauth_server.issue_refresh_token(ME)
    secret = EmailSecret(refresh_token=refresh, client_secret=oauth_server.client_secret)
    updates = []
    provider = OAuthTokenProvider(oauth_account(imap, smtp, ca, oauth_server).oauth, secret, on_update=updates.append, verify=ca.client_context())
    c = EmailClient(oauth_account(imap, smtp, ca, oauth_server), secret, token_provider=provider)
    result = c.test()
    assert result["imap"]["ok"] is True and result["smtp"]["ok"] is True
    assert imap.logins[0][0] == ME and imap.logins[0][1].startswith("at-")
    assert smtp.logins[0][0] == ME
    assert len(updates) == 1 and updates[0].access_token  # refreshed once, then reused
    assert [r["grant_type"] for r in oauth_server.requests] == ["refresh_token"]
    assert "rt-" not in repr(updates[0])


def test_rotated_refresh_token_is_handed_back_for_sealing(ca, oauth_server) -> None:
    oauth_server.rotate_refresh = True
    first = oauth_server.issue_refresh_token(ME)
    oauth = oauth_settings(oauth_server)
    got = []
    p = OAuthTokenProvider(oauth, EmailSecret(refresh_token=first, client_secret="test-secret"), on_update=got.append, verify=ca.client_context())
    p.access_token()
    assert got[0].refresh_token != first and got[0].refresh_token in oauth_server.refresh_tokens
    p.access_token(force_refresh=True)  # uses the rotated token
    assert len(got) == 2


def test_revoked_grant_asks_to_sign_in_again_and_bad_client_is_named(ca, oauth_server) -> None:
    oauth = oauth_settings(oauth_server)
    refresh = oauth_server.issue_refresh_token(ME)
    oauth_server.revoke_refresh_tokens()
    with pytest.raises(EmailOAuthReauthorize) as info:
        OAuthTokenClient(oauth, client_secret="test-secret", verify=ca.client_context()).refresh(refresh)
    assert info.value.details["oauth_error"] == "invalid_grant"
    assert "Sign in again" in info.value.fix
    with pytest.raises(EmailOAuthFailed) as info2:
        OAuthTokenClient(oauth, client_secret="wrong", verify=ca.client_context()).refresh("rt-x")
    assert info2.value.details["oauth_error"] == "invalid_client"


def test_rejected_access_token_is_an_auth_error_naming_re_authorisation(imap, ca, oauth_server, tokens) -> None:
    secret = EmailSecret(access_token="at-forged", expires_at=time.time() + 3600, refresh_token="rt-x")
    c = EmailClient(oauth_account(imap, None, ca, oauth_server), secret)
    result = c.test()
    assert result["imap"]["code"] == "email_auth_failed"
    assert "OAuth2" in result["imap"]["cause"]
    assert imap.logins == []


def test_untrusted_token_endpoint_is_refused(oauth_server) -> None:
    oauth = oauth_settings(oauth_server)
    from abstractcore.comms.email import EmailTlsFailed

    with pytest.raises(EmailTlsFailed):
        OAuthTokenClient(oauth, client_secret="test-secret").refresh("rt-x")  # system trust store only


def test_cli_device_flow_connects_and_seals_tokens(imap, smtp, ca, oauth_server, config_file, capsys) -> None:
    from abstractcore.config.email_cli import handle_email

    def approve_soon() -> None:
        for _ in range(100):
            if oauth_server.devices:
                oauth_server.approve_all_devices()
                return
            time.sleep(0.05)

    threading.Thread(target=approve_soon, daemon=True).start()
    code = handle_email([
        "connect", "--address", ME, "--oauth", "custom", "--client-id", oauth_server.client_id,
        "--client-secret", oauth_server.client_secret, "--oauth-flow", "device",
        "--token-endpoint", f"{oauth_server.base_url}/token", "--device-endpoint", f"{oauth_server.base_url}/device",
        "--scope", "mail", "--imap-host", "localhost", "--imap-port", str(imap.port), "--imap-security", "ssl",
        "--smtp-host", "localhost", "--smtp-port", str(smtp.port), "--smtp-security", "starttls",
        "--ca-file", str(ca.ca_pem), "--json",
    ])
    out = capsys.readouterr()
    assert code == 0, out
    assert "WDJB-MJHT" in out.err  # the code the person types, shown on stderr in --json mode
    doc = json.loads(out.out)
    assert doc["auth_kind"] == "oauth2" and doc["oauth"]["client_id"] == oauth_server.client_id
    assert doc["status"]["legs"]["imap"]["ok"] is True
    blob = config_file.read_text() + "".join(p.read_text(errors="replace") for p in (config_file.parent / "email").iterdir())
    assert oauth_server.client_secret not in blob and "rt-" not in blob and "at-" not in blob


def test_cli_loopback_flow_uses_pkce_and_the_one_shot_listener(imap, ca, oauth_server, config_file, capsys, monkeypatch) -> None:
    from abstractcore.config import email_cli

    def browser(url: str) -> bool:
        redirect = oauth_server.authorize(url, ME)  # the provider's consent page
        assert redirect.startswith("http://127.0.0.1:")
        threading.Thread(target=lambda: urllib.request.urlopen(redirect, timeout=5).read(), daemon=True).start()
        return True

    monkeypatch.setattr(email_cli.webbrowser, "open", browser)
    code = email_cli.handle_email([
        "connect", "--address", ME, "--oauth", "custom", "--client-id", oauth_server.client_id,
        "--client-secret", oauth_server.client_secret, "--oauth-flow", "loopback",
        "--token-endpoint", f"{oauth_server.base_url}/token", "--authorization-endpoint", f"{oauth_server.base_url}/authorize",
        "--scope", "mail", "--imap-host", "localhost", "--imap-port", str(imap.port), "--imap-security", "ssl",
        "--ca-file", str(ca.ca_pem), "--oauth-timeout", "20",
    ])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "code_challenge_method=S256" in out
    assert [r["grant_type"] for r in oauth_server.requests] == ["authorization_code"]


def test_stored_oauth_account_refreshes_and_persists_rotation_through_the_store(imap, ca, oauth_server, config_file) -> None:
    from abstractcore.comms.email import EmailAccountStore

    oauth_server.rotate_refresh = True
    oauth_server.access_ttl_s = 1  # expires at once: the next use refreshes
    first = oauth_server.issue_refresh_token(ME)
    store = EmailAccountStore(config_file)
    store.connect(oauth_account(imap, None, ca, oauth_server), EmailSecret(refresh_token=first, client_secret="test-secret"), test=False)
    ctx = store.context()
    assert ctx.client().test_imap()["ok"] is True
    sealed = EmailSecret.from_sealed_payload(store.vault.load())
    assert sealed.refresh_token != first and sealed.refresh_token in oauth_server.refresh_tokens
