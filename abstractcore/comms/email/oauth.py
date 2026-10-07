"""OAuth2 sign-in for mail accounts: provider presets, token exchange, refresh, XOAUTH2.

What lives here (host-agnostic, no UI):

- `provider_preset(name)`: Google and Microsoft endpoints, scopes and IMAP/SMTP hosts.
- `OAuthTokenClient`: the token endpoint (RFC 6749) — authorization-code exchange with PKCE
  (RFC 7636), device authorization (RFC 8628) start and poll, refresh. Errors are classified
  from the RFC 6749 `error` code of the response, never from text.
- `LoopbackAuthorization`: the loopback redirect flow for a person at this machine: a one-shot
  listener on 127.0.0.1, the authorization URL to open, the code that comes back.
- `OAuthTokenProvider`: a valid access token for each connection, refreshed ahead of expiry;
  a rotated refresh token is handed to the caller's `on_update` so it is sealed again.
- `xoauth2_string(user, token)`: the SASL XOAUTH2 initial response used for IMAP and SMTP.

Which flow for which provider: Microsoft supports the device-code flow for its IMAP/SMTP
scopes. Google documents that its device flow does not allow Gmail scopes, so Google accounts
use the loopback redirect flow (a browser on the machine running the command) or a host
(the gateway) running the redirect flow.

Every endpoint must be https; certificates are verified (an explicit `verify` SSL context is
accepted for private CAs and tests, never an unverified one).
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import secrets
import socketserver
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from .errors import (
    EmailError,
    EmailInvalidSettings,
    EmailOAuthFailed,
    EmailOAuthPending,
    EmailOAuthReauthorize,
    EmailServerError,
    EmailTlsFailed,
    EmailUnreachable,
)

_PRESETS: Dict[str, Dict[str, Any]] = {
    "google": {
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "device_authorization_endpoint": "",
        "scopes": ["https://mail.google.com/"],
        "imap": {"host": "imap.gmail.com", "port": 993, "security": "ssl"},
        "smtp": {"host": "smtp.gmail.com", "port": 465, "security": "ssl"},
        "default_flow": "loopback",
        "tenant": "",
    },
    "microsoft": {
        "authorization_endpoint": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
        "token_endpoint": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        "device_authorization_endpoint": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode",
        "scopes": [
            "https://outlook.office.com/IMAP.AccessAsUser.All",
            "https://outlook.office.com/SMTP.Send",
            "offline_access",
        ],
        "imap": {"host": "outlook.office365.com", "port": 993, "security": "ssl"},
        "smtp": {"host": "smtp.office365.com", "port": 587, "security": "starttls"},
        "default_flow": "device",
        "tenant": "common",
    },
}


# The AbstractFramework OAuth clients registered with each provider (operator decision
# 2026-09-29: a built-in client AND bring-your-own). A built-in client is an "installed
# application" client (RFC 8252): its id is public and its secret, where the provider issues
# one, is not a confidential credential. The table stays empty until the registrations exist
# (Google: an OAuth client for the restricted Gmail scope; Microsoft: a multi-tenant Entra app
# with publisher verification); until then every OAuth sign-in brings its own client
# (`--client-id <value> --client-secret <value>`, or the gateway's admin setting).
BUILTIN_CLIENTS: Dict[str, Dict[str, str]] = {}


def builtin_client(provider: str) -> Optional[Dict[str, str]]:
    """The built-in client `{client_id, client_secret}` of a provider, or None."""

    entry = BUILTIN_CLIENTS.get(str(provider or "").strip().lower())
    if not entry or not str(entry.get("client_id") or "").strip():
        return None
    return {"client_id": str(entry["client_id"]).strip(), "client_secret": str(entry.get("client_secret") or "")}


_PROVIDER_LABELS = {"google": "Google", "microsoft": "Microsoft"}


def oauth_providers_public(configured: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """`[{id, available, reason}]` for the "Sign in with Google / Microsoft" buttons.

    `available` when this version has a built-in client for the provider, or when `configured`
    (an embedding host's own clients, e.g. a gateway admin setting: `{provider: truthy}`) names
    one. `reason` says why not (None when available).
    """

    configured = configured or {}
    out: List[Dict[str, Any]] = []
    for prov in ("google", "microsoft"):
        label = _PROVIDER_LABELS[prov]
        available = builtin_client(prov) is not None or bool(configured.get(prov))
        out.append(
            {
                "id": prov,
                "available": available,
                "reason": None if available else f"No built-in {label} sign-in client in this version: add your own client id under Sign-in app.",
            }
        )
    return out


def resolve_oauth_client(provider: str, client_id: str = "", client_secret: str = "") -> Dict[str, str]:
    """Which OAuth client signs in: `{client_id, client_secret, source}`.

    A given client id is used as is (`source: "own"`, bring your own client). Without one,
    the built-in AbstractFramework client of a known provider (`source: "builtin"`). Without
    either, a typed error that names both ways forward.
    """

    prov = str(provider or "").strip().lower()
    cid = str(client_id or "").strip()
    if cid:
        return {"client_id": cid, "client_secret": str(client_secret or ""), "source": "own"}
    if str(client_secret or "").strip():
        raise EmailInvalidSettings(
            "An OAuth client secret was given without its client id.",
            "Give both --client-id <value> and --client-secret <value> of your OAuth client.",
        )
    builtin = builtin_client(prov) if prov in _PRESETS else None
    if builtin is not None:
        return {**builtin, "source": "builtin"}
    label = {"google": "Google", "microsoft": "Microsoft"}.get(prov, prov or "this provider")
    if prov in _PRESETS:
        cause = f"No built-in AbstractFramework OAuth client is registered for {label} in this version."
    else:
        cause = f"A custom OAuth provider needs the client id of an OAuth client registered with {label}."
    raise EmailInvalidSettings(
        cause,
        f"Bring your own OAuth client registered with {label} for the mail scopes: --client-id <value> "
        "--client-secret <value> (or sign in with an app password instead).",
    )


def provider_preset(name: str, *, tenant: str = "") -> Dict[str, Any]:
    """Endpoints, scopes and mail hosts of a known provider (a copy; `{}` for unknown)."""

    preset = _PRESETS.get(str(name or "").strip().lower())
    if not preset:
        return {}
    out = {k: (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)) for k, v in preset.items()}
    t = str(tenant or out.get("tenant") or "").strip()
    out["tenant"] = t
    for key in ("authorization_endpoint", "token_endpoint", "device_authorization_endpoint"):
        if out.get(key):
            out[key] = str(out[key]).replace("{tenant}", t or "common")
    return out


def xoauth2_string(user: str, access_token: str) -> str:
    """The SASL XOAUTH2 initial client response (before base64)."""

    return f"user={user}\x01auth=Bearer {access_token}\x01\x01"


@dataclass(frozen=True)
class TokenSet:
    access_token: str
    expires_at: float
    refresh_token: str = ""
    scope: str = ""
    token_type: str = "Bearer"

    def __repr__(self) -> str:
        return f"TokenSet(access_token=«set», refresh_token={'«set»' if self.refresh_token else '«empty»'}, expires_at={self.expires_at})"


@dataclass(frozen=True)
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    expires_at: float
    interval: float
    verification_uri_complete: str = ""

    def __repr__(self) -> str:
        return f"DeviceAuthorization(user_code={self.user_code!r}, verification_uri={self.verification_uri!r})"

    def public(self) -> Dict[str, Any]:
        """What a person needs to see (the device code itself stays private)."""

        return {
            "user_code": self.user_code,
            "verification_uri": self.verification_uri,
            "verification_uri_complete": self.verification_uri_complete,
            "expires_at": self.expires_at,
        }


def make_pkce() -> Dict[str, str]:
    """A PKCE verifier and its S256 challenge (RFC 7636)."""

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(48)).decode("ascii").rstrip("=")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")
    return {"verifier": verifier, "challenge": challenge, "method": "S256"}


def _find_ssl_error(exc: BaseException) -> Optional[BaseException]:
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


def _classify_oauth_error(error: str, *, stage: str, status: int) -> EmailError:
    details = {"protocol": "oauth", "oauth_error": error, "stage": stage, "http_status": status}
    if error == "invalid_grant":
        return EmailOAuthReauthorize(
            "The mail provider no longer accepts this sign-in (the grant expired, was revoked, or the "
            "password changed).",
            "Sign in again: connect the account with OAuth2 once more.",
            details=details,
        )
    if error in {"invalid_client", "unauthorized_client"}:
        return EmailOAuthFailed(
            "The mail provider refused the OAuth client (client id or client secret).",
            "Check the client id and client secret of the OAuth client registered with the provider.",
            details=details,
        )
    if error in {"invalid_scope"}:
        return EmailOAuthFailed(
            "The mail provider refused the requested mail scopes for this OAuth client.",
            "Enable the IMAP/SMTP (or Gmail) scopes on the OAuth client registration, then sign in again.",
            details=details,
        )
    if error in {"access_denied"}:
        return EmailOAuthFailed(
            "The sign-in was declined.",
            "Start the sign-in again and approve the requested access.",
            details=details,
        )
    if error in {"expired_token"}:
        return EmailOAuthFailed(
            "The sign-in code expired before it was approved.",
            "Start the sign-in again and approve it within the time shown.",
            details=details,
        )
    if error in {"authorization_pending", "slow_down"}:
        return EmailOAuthPending("Waiting for approval.", "Approve the sign-in in the browser.", details=details)
    if error in {"temporarily_unavailable", "server_error"} or status >= 500:
        return EmailServerError(
            "The mail provider's sign-in service is temporarily unavailable.",
            "Retry later.",
            retryable=True,
            details=details,
        )
    return EmailOAuthFailed(
        "The mail provider refused the sign-in request.",
        "Check the OAuth client settings (client id, secret, endpoints) and sign in again.",
        details=details,
    )


Verify = Union[bool, ssl.SSLContext, None]


class OAuthTokenClient:
    """Talks to one provider's OAuth2 endpoints (https only, certificates verified)."""

    def __init__(self, oauth: Any, *, client_secret: str = "", verify: Verify = None, timeout: float = 30.0) -> None:
        self.oauth = oauth
        self._client_secret = str(client_secret or "")
        if isinstance(verify, ssl.SSLContext):
            if verify.verify_mode != ssl.CERT_REQUIRED or not verify.check_hostname:
                raise EmailInvalidSettings(
                    "An SSL context that does not verify certificates was given for OAuth.",
                    "Pass a context from ssl.create_default_context() (optionally with a private CA loaded).",
                )
            self._verify: Any = verify
        else:
            self._verify = True
        self._timeout = float(timeout)

    def __repr__(self) -> str:
        return f"OAuthTokenClient(provider={getattr(self.oauth, 'provider', '?')!r})"

    def _post(self, url: str, data: Dict[str, str], *, stage: str) -> Dict[str, Any]:
        import httpx

        if not str(url or "").lower().startswith("https://"):
            raise EmailInvalidSettings(
                f"The OAuth {stage} endpoint is not an https URL.",
                "Use the provider's https endpoint; tokens are never sent over plain HTTP.",
            )
        try:
            with httpx.Client(verify=self._verify, timeout=self._timeout, follow_redirects=False) as client:
                resp = client.post(url, data=data, headers={"Accept": "application/json"})
        except Exception as exc:  # network / TLS
            host = urllib.parse.urlsplit(url).hostname or ""
            details = {"protocol": "oauth", "host": host, "stage": stage}
            ssl_err = _find_ssl_error(exc)
            if ssl_err is not None:
                raise EmailTlsFailed(
                    "The mail provider's sign-in endpoint presented a certificate that could not be verified.",
                    "Check the endpoint URL; for a private CA, provide that CA.",
                    details=details,
                ) from None
            raise EmailUnreachable(
                "The mail provider's sign-in endpoint could not be reached.",
                "Check the network connection and the endpoint URL; then retry.",
                details=details,
            ) from None
        try:
            payload = resp.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if resp.status_code >= 400 or "error" in payload:
            raise _classify_oauth_error(str(payload.get("error") or ""), stage=stage, status=int(resp.status_code))
        return payload

    def _base(self) -> Dict[str, str]:
        data = {"client_id": self.oauth.client_id}
        if self._client_secret:
            data["client_secret"] = self._client_secret
        return data

    def _tokens(self, payload: Dict[str, Any], *, previous_refresh: str = "") -> TokenSet:
        token = str(payload.get("access_token") or "")
        if not token:
            raise EmailOAuthFailed(
                "The token endpoint answered without an access token.",
                "Check the OAuth client settings and sign in again.",
                details={"protocol": "oauth"},
            )
        try:
            expires_in = float(payload.get("expires_in") or 3600)
        except (TypeError, ValueError):
            expires_in = 3600.0
        return TokenSet(
            access_token=token,
            expires_at=time.time() + max(0.0, expires_in),
            refresh_token=str(payload.get("refresh_token") or previous_refresh or ""),
            scope=str(payload.get("scope") or ""),
            token_type=str(payload.get("token_type") or "Bearer"),
        )

    # -- flows ---------------------------------------------------------------------------

    def authorization_url(self, *, redirect_uri: str, state: str, code_challenge: str, login_hint: str = "") -> str:
        if not self.oauth.authorization_endpoint:
            raise EmailInvalidSettings(
                "This OAuth client has no authorization endpoint.",
                "Give --authorization-endpoint for a custom provider, or use the device-code flow.",
            )
        params = {
            "response_type": "code",
            "client_id": self.oauth.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(self.oauth.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if self.oauth.provider == "google":
            params["access_type"] = "offline"
            params["prompt"] = "consent"
        if login_hint:
            params["login_hint"] = login_hint
        sep = "&" if "?" in self.oauth.authorization_endpoint else "?"
        return self.oauth.authorization_endpoint + sep + urllib.parse.urlencode(params)

    def exchange_code(self, code: str, *, redirect_uri: str, code_verifier: str) -> TokenSet:
        data = self._base()
        data.update(
            {
                "grant_type": "authorization_code",
                "code": str(code),
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            }
        )
        return self._tokens(self._post(self.oauth.token_endpoint, data, stage="code_exchange"))

    def start_device_authorization(self) -> DeviceAuthorization:
        if not self.oauth.device_authorization_endpoint:
            raise EmailInvalidSettings(
                f"The {self.oauth.provider} OAuth client has no device authorization endpoint.",
                "Use the loopback flow (--oauth-flow loopback) on a machine with a browser.",
            )
        data = self._base()
        data["scope"] = " ".join(self.oauth.scopes)
        payload = self._post(self.oauth.device_authorization_endpoint, data, stage="device_authorization")
        try:
            expires_in = float(payload.get("expires_in") or 900)
            interval = float(payload.get("interval") or 5)
        except (TypeError, ValueError):
            expires_in, interval = 900.0, 5.0
        device_code = str(payload.get("device_code") or "")
        user_code = str(payload.get("user_code") or "")
        uri = str(payload.get("verification_uri") or payload.get("verification_url") or "")
        if not (device_code and user_code and uri):
            raise EmailOAuthFailed(
                "The device authorization answer is incomplete.",
                "Check the device authorization endpoint of the OAuth client.",
                details={"protocol": "oauth"},
            )
        return DeviceAuthorization(
            device_code=device_code,
            user_code=user_code,
            verification_uri=uri,
            verification_uri_complete=str(payload.get("verification_uri_complete") or ""),
            expires_at=time.time() + expires_in,
            interval=max(1.0, interval),
        )

    def poll_device_authorization(
        self,
        device: DeviceAuthorization,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> TokenSet:
        interval = device.interval
        while True:
            if clock() >= device.expires_at:
                raise _classify_oauth_error("expired_token", stage="device_poll", status=400)
            try:
                return self.poll_device_once(device)
            except EmailOAuthPending as pending:
                if pending.details.get("oauth_error") == "slow_down":
                    interval += 5.0
                sleep(interval)

    def poll_device_once(self, device: DeviceAuthorization) -> TokenSet:
        """One poll: the tokens, or `EmailOAuthPending` while the person has not approved yet."""

        data = self._base()
        data.update({"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": device.device_code})
        return self._tokens(self._post(self.oauth.token_endpoint, data, stage="device_poll"))

    def refresh(self, refresh_token: str) -> TokenSet:
        if not refresh_token:
            raise EmailOAuthReauthorize(
                "No OAuth refresh token is stored for this account.",
                "Sign in again: connect the account with OAuth2 once more.",
            )
        data = self._base()
        data.update({"grant_type": "refresh_token", "refresh_token": refresh_token})
        if self.oauth.provider != "google":
            data["scope"] = " ".join(self.oauth.scopes)
        return self._tokens(self._post(self.oauth.token_endpoint, data, stage="refresh"), previous_refresh=refresh_token)


class OAuthTokenProvider:
    """A valid access token per connection, refreshed ahead of expiry.

    `on_update(secret)` receives the new `EmailSecret` after each refresh (the refresh token
    may rotate) so the caller seals it again. Thread-safe.
    """

    def __init__(
        self,
        oauth: Any,
        secret: Any,
        *,
        on_update: Optional[Callable[[Any], None]] = None,
        verify: Verify = None,
        skew_s: float = 60.0,
    ) -> None:
        self._oauth = oauth
        self._secret = secret
        self._on_update = on_update
        self._verify = verify
        self._skew = float(skew_s)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return "OAuthTokenProvider(«redacted»)"

    def access_token(self, *, force_refresh: bool = False) -> str:
        from .models import EmailSecret

        with self._lock:
            if not force_refresh and self._secret.access_token_valid(skew_s=self._skew):
                return self._secret.access_token
            client = OAuthTokenClient(self._oauth, client_secret=self._secret.client_secret, verify=self._verify)
            tokens = client.refresh(self._secret.refresh_token)
            self._secret = EmailSecret(
                refresh_token=tokens.refresh_token or self._secret.refresh_token,
                access_token=tokens.access_token,
                expires_at=tokens.expires_at,
                client_secret=self._secret.client_secret,
            )
            if self._on_update is not None:
                self._on_update(self._secret)
            return self._secret.access_token


# ---------------------------------------------------------------------------------------
# Loopback redirect flow
# ---------------------------------------------------------------------------------------


class _OneShotServer(socketserver.TCPServer):
    allow_reuse_address = True


class LoopbackAuthorization:
    """Authorization-code flow with PKCE and a one-shot listener on 127.0.0.1.

        flow = LoopbackAuthorization(token_client)
        url = flow.start()            # show / open this URL
        tokens = flow.finish(timeout_s=300)

    The listener accepts exactly one redirect carrying the expected `state`, answers a short
    page, and stops. Nothing listens on another interface.
    """

    def __init__(self, token_client: OAuthTokenClient, *, login_hint: str = "", path: str = "/oauth/callback") -> None:
        self._client = token_client
        self._login_hint = login_hint
        self._path = path
        self._state = secrets.token_urlsafe(24)
        self._pkce = make_pkce()
        self._result: Dict[str, str] = {}
        self._done = threading.Event()
        self._server: Optional[_OneShotServer] = None
        self._thread: Optional[threading.Thread] = None
        self.redirect_uri = ""

    def start(self) -> str:
        flow = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # never log the query (it carries the code)
                return

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                parts = urllib.parse.urlsplit(self.path)
                if parts.path != flow._path:
                    self.send_response(404)
                    self.end_headers()
                    return
                query = dict(urllib.parse.parse_qsl(parts.query))
                if query.get("state") != flow._state:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"State mismatch; this sign-in was not started here.")
                    return
                flow._result = {"code": query.get("code", ""), "error": query.get("error", "")}
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Sign-in received. You can close this window and return to the terminal.")
                flow._done.set()

        self._server = _OneShotServer(("127.0.0.1", 0), Handler)
        port = int(self._server.server_address[1])
        self.redirect_uri = f"http://127.0.0.1:{port}{self._path}"
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()
        return self._client.authorization_url(
            redirect_uri=self.redirect_uri,
            state=self._state,
            code_challenge=self._pkce["challenge"],
            login_hint=self._login_hint,
        )

    def wait(self, timeout_s: float) -> bool:
        """True once the redirect came back (then call `finish`)."""

        return self._done.wait(timeout_s)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def finish(self, *, timeout_s: float = 300.0) -> TokenSet:
        try:
            if not self._done.wait(timeout_s):
                raise EmailOAuthFailed(
                    "No sign-in came back before the time ran out.",
                    "Start the sign-in again and complete it in the browser.",
                    details={"protocol": "oauth", "stage": "loopback"},
                )
        finally:
            self.close()
        if self._result.get("error"):
            raise _classify_oauth_error(self._result["error"], stage="authorization", status=400)
        code = self._result.get("code") or ""
        if not code:
            raise EmailOAuthFailed(
                "The sign-in redirect carried no authorization code.",
                "Start the sign-in again.",
                details={"protocol": "oauth", "stage": "loopback"},
            )
        return self._client.exchange_code(code, redirect_uri=self.redirect_uri, code_verifier=self._pkce["verifier"])


def scopes_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [s for s in value.split() if s]
    if isinstance(value, (list, tuple)):
        return [str(s).strip() for s in value if str(s).strip()]
    return []
