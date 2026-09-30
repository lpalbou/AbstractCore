//! Email: the email address and the mailbox of this AbstractCore
//! install — the web console's Email tab, card for card and word for
//! word (the framework's account-page design, single user):
//!
//! 1. **Email address** — one field with its own inline Save (where
//!    notifications go, and the first address agents may write to).
//! 2. **Mailbox** — not connected: tabs Google / Microsoft / Other. The
//!    Other tab asks only for the address and the password; the mail
//!    servers are found from the address (`abstractcore email discover`)
//!    and shown as one summary line, with "Server settings" folded (it
//!    opens by itself, with the reason, when nothing is found). ONE
//!    primary action, Connect (store + connection test). Connected: one
//!    status line + Test + Disconnect (inline confirmation).
//! 3. **Agent email tools** — one switch, unavailable with its reason
//!    until a mailbox is connected and in use.
//! 4. **Advanced** (folded) — recipient rules (add/remove apply at
//!    once), send limits (saved on Enter or when the field loses
//!    focus), the folder, and the "Use this mailbox" switch.
//!
//! No Save per section: switches apply immediately, the only Save is the
//! Email address field's own.
//!
//! Everything goes through `abstractcore email … --json` (the worker's
//! `Cmd::LoadEmail` / `Cmd::Email` / `Cmd::EmailDiscover`): the CLI
//! validates, tests the connection, seals the password (encrypted,
//! never printed) and owns the recipient policy and send limits. This
//! screen renders the `email_settings_v1` document and builds argv.
//! Every value rides `--flag=value` (or a checked positional), so an
//! entry starting with `-` cannot be read as a flag. A password or
//! OAuth client secret never goes on a command line (every local user
//! can read argv with `ps`): the command gets `--password-stdin` /
//! `--client-secret-stdin` and the secret is written to its stdin.
//!
//! The mailbox shown is the one of this AbstractCore install (the core
//! settings file); a gateway keeps one per user, in the gateway
//! console's My account page.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use abstracttui::prelude::*;
use abstracttui::widgets::{ButtonStyle, Disclosure};
use serde_json::Value;

use crate::store::Loadable;
use crate::worker::{cancel_email_oauth, next_form_id, Cmd, EmailAction};
use crate::writes::{Arg, StdinSecret};

use super::switch::SwitchRow;
use super::util::{error_panel, field_w, line, span, span_bold, Switch};
use super::Ctx;

/// The footer's keys for this screen.
pub const HINTS: &[(&str, &str)] = &[
    ("Tab", "next control"),
    ("space", "switch"),
    ("Enter", "press / save"),
    ("←/→", "mailbox tabs"),
];

// ---------------------------------------------------------------------
// Words (the framework vocabulary: "Email address" vs "Mailbox").
// ---------------------------------------------------------------------

pub const ADDRESS_HELP: &str =
    "Where notifications go, and the first address your agents may write to.";
pub const PASSWORD_HELP: &str = "Use an app password if your provider needs one.";
pub const AGENT_TOOLS_DESC: &str = "Your agents and workflows may list, search, read, send and reply to your mail. Every send still follows your recipient rules, your limits and the approval gate.";
pub const USE_MAILBOX_DESC: &str =
    "Off keeps the settings but stops watching, sending and notifications.";
pub const DISCONNECT_CONFIRM: &str = "Disconnect this mailbox? Your agents lose email until you connect again. Policy and limits are kept.";
pub const NO_MAILBOX: &str = "Connect a mailbox first.";
pub const MAILBOX_NOT_IN_USE: &str = "\u{201c}Use this mailbox\u{201d} is off (Advanced).";
pub const TABS: [&str; 3] = ["Google", "Microsoft", "Other"];

/// Label column of the screen's fields.
const LABEL_W: i32 = 15;

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

fn b(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

// ---------------------------------------------------------------------
// Durable screen state (root scope: survives tab switches and remounts,
// and a timer firing after the page closed never writes a dead signal).
// ---------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct EmailUi {
    /// Mailbox tab: 0 Google, 1 Microsoft, 2 Other.
    pub tab: Signal<usize>,
    /// The person picked a tab: the lookup no longer chooses one.
    tab_chosen: Signal<bool>,
    /// The Email address card's field, and its inline outcome.
    pub address: Signal<String>,
    pub address_note: Signal<Option<Result<String, String>>>,
    /// The stored email address the field was last seeded from.
    seen_address: Signal<Option<String>>,
    /// The mailbox address (every tab; prefilled from the email address).
    pub mb_address: Signal<String>,
    pub password: Signal<String>,
    pub servers_folded: Signal<bool>,
    /// Why Server settings opened by itself (discovery found nothing).
    pub servers_reason: Signal<Option<String>>,
    pub username: Signal<String>,
    pub display_name: Signal<String>,
    pub imap_host: Signal<String>,
    pub imap_port: Signal<String>,
    pub imap_sec: Signal<usize>,
    pub folder: Signal<String>,
    pub smtp_host: Signal<String>,
    pub smtp_port: Signal<String>,
    pub smtp_sec: Signal<usize>,
    pub ca_file: Signal<String>,
    pub oauth_folded: Signal<bool>,
    pub client_id: Signal<String>,
    pub client_secret: Signal<String>,
    pub tenant: Signal<String>,
    pub flow: Signal<usize>,
    /// The inline error of the last Connect / sign-in.
    pub connect_error: Signal<Option<String>>,
    pub connecting: Signal<bool>,
    pub confirm_disconnect: Signal<bool>,
    pub advanced_folded: Signal<bool>,
    pub policy_add: Signal<String>,
    pub per_hour: Signal<String>,
    pub per_day: Signal<String>,
    seen_limits: Signal<Option<(i64, i64)>>,
    /// The last values sent (Enter then leaving the field sends once).
    limits_sent: Signal<Option<(String, String)>>,
    folder_sent: Signal<Option<String>>,
    /// Advanced → Folder (saved on Enter or when it loses focus).
    pub folder_edit: Signal<String>,
    seen_folder: Signal<Option<String>>,
    pub folder_note: Signal<Option<Result<String, String>>>,
    pub limits_note: Signal<Option<Result<String, String>>>,
    /// The address a server lookup is running for.
    pub pending_discovery: Signal<Option<String>>,
    seeded: Signal<bool>,
    flash_gen: Signal<u64>,
    pub fid_address: u64,
    pub fid_connect: u64,
    pub fid_limits: u64,
    pub fid_folder: u64,
}

impl EmailUi {
    pub fn create(cx: Scope) -> EmailUi {
        EmailUi {
            tab: cx.signal(0),
            tab_chosen: cx.signal(false),
            address: cx.signal(String::new()),
            address_note: cx.signal(None),
            seen_address: cx.signal(None),
            mb_address: cx.signal(String::new()),
            password: cx.signal(String::new()),
            servers_folded: cx.signal(true),
            servers_reason: cx.signal(None),
            username: cx.signal(String::new()),
            display_name: cx.signal(String::new()),
            imap_host: cx.signal(String::new()),
            imap_port: cx.signal(String::new()),
            imap_sec: cx.signal(0),
            folder: cx.signal(String::new()),
            smtp_host: cx.signal(String::new()),
            smtp_port: cx.signal(String::new()),
            smtp_sec: cx.signal(0),
            ca_file: cx.signal(String::new()),
            oauth_folded: cx.signal(true),
            client_id: cx.signal(String::new()),
            client_secret: cx.signal(String::new()),
            tenant: cx.signal(String::new()),
            flow: cx.signal(0),
            connect_error: cx.signal(None),
            connecting: cx.signal(false),
            confirm_disconnect: cx.signal(false),
            advanced_folded: cx.signal(true),
            policy_add: cx.signal(String::new()),
            per_hour: cx.signal(String::new()),
            per_day: cx.signal(String::new()),
            seen_limits: cx.signal(None),
            limits_sent: cx.signal(None),
            folder_sent: cx.signal(None),
            folder_edit: cx.signal(String::new()),
            seen_folder: cx.signal(None),
            folder_note: cx.signal(None),
            limits_note: cx.signal(None),
            pending_discovery: cx.signal(None),
            seeded: cx.signal(false),
            flash_gen: cx.signal(0),
            fid_address: next_form_id(),
            fid_connect: next_form_id(),
            fid_limits: next_form_id(),
            fid_folder: next_form_id(),
        }
    }
}

// ---------------------------------------------------------------------
// Pure helpers (unit-tested below).
// ---------------------------------------------------------------------

/// A syntactic check before a server lookup (one `@`, a dotted domain,
/// no spaces, not starting with `-`) — never a guess about meaning; the
/// CLI's own validation stays the authority.
pub fn plausible_address(addr: &str) -> bool {
    let addr = addr.trim();
    if addr.starts_with('-') || addr.chars().any(char::is_whitespace) {
        return false;
    }
    match addr.split_once('@') {
        Some((local, domain)) => {
            !local.is_empty()
                && !domain.contains('@')
                && domain.contains('.')
                && !domain.starts_with('.')
                && !domain.ends_with('.')
        }
        None => false,
    }
}

fn domain_of(addr: &str) -> &str {
    addr.trim().rsplit_once('@').map(|(_, d)| d).unwrap_or("")
}

fn security_word(sec: &str) -> &'static str {
    if sec == "starttls" {
        "STARTTLS"
    } else {
        "SSL"
    }
}

/// "imap.fastmail.com · 993 · SSL  ·  smtp.fastmail.com · 465 · SSL".
pub fn servers_summary(found: &Value) -> String {
    ["imap", "smtp"]
        .iter()
        .filter_map(|leg| found.get(*leg).filter(|v| v.is_object()))
        .map(|srv| {
            format!(
                "{} · {} · {}",
                s(srv, "host"),
                srv.get("port").and_then(Value::as_i64).unwrap_or(0),
                security_word(s(srv, "security"))
            )
        })
        .collect::<Vec<_>>()
        .join("  ·  ")
}

/// The discovery answer about `addr`, if the last lookup was for it.
fn answer_for(d: &Option<Value>, addr: &str) -> Option<Value> {
    d.as_ref()
        .filter(|v| s(v, "address").eq_ignore_ascii_case(addr.trim()))
        .cloned()
}

fn provider_label(p: &str) -> &'static str {
    match p {
        "google" => "Google",
        "microsoft" => "Microsoft",
        _ => "OAuth2",
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Seconds since the epoch of an RFC 3339 timestamp
/// (`2026-09-29T20:00:00+00:00`, `…Z`, fractions allowed).
pub fn parse_rfc3339(ts: &str) -> Option<i64> {
    let ts = ts.trim();
    if ts.len() < 19 {
        return None;
    }
    let num = |a: usize, b: usize| ts.get(a..b)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (h, mi, se) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    let mut rest = &ts[19..];
    if let Some(r) = rest.strip_prefix('.') {
        let cut = r.find(|c: char| !c.is_ascii_digit()).unwrap_or(r.len());
        rest = &r[cut..];
    }
    let offset = match rest {
        "" | "Z" | "z" => 0,
        o if o.len() == 6 => {
            let sign = if o.starts_with('-') { -1 } else { 1 };
            let oh = o.get(1..3)?.parse::<i64>().ok()?;
            let om = o.get(4..6)?.parse::<i64>().ok()?;
            sign * (oh * 3600 + om * 60)
        }
        _ => return None,
    };
    Some(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se - offset)
}

/// "just now" / "2 min ago" / "3 h ago" / "4 days ago".
pub fn ago(ts: &str, now: i64) -> Option<String> {
    let then = parse_rfc3339(ts)?;
    let d = (now - then).max(0);
    Some(match d {
        0..=59 => "just now".into(),
        60..=3599 => format!("{} min ago", d / 60),
        3600..=86_399 => format!("{} h ago", d / 3600),
        _ => {
            let days = d / 86_400;
            format!("{days} day{} ago", if days == 1 { "" } else { "s" })
        }
    })
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// "Connected as me@x.com · Password · checked 2 min ago".
pub fn connected_line(doc: &Value, now: i64) -> String {
    let method = if s(doc, "auth_kind") == "oauth2" {
        provider_label(
            doc.get("oauth")
                .map(|o| s(o, "provider"))
                .unwrap_or_default(),
        )
        .to_string()
    } else {
        "Password".to_string()
    };
    let st = doc.get("status").cloned().unwrap_or(Value::Null);
    let checked = match ago(s(&st, "last_test"), now) {
        Some(a) => format!("checked {a}"),
        None => "not checked yet".into(),
    };
    format!("Connected as {} · {method} · {checked}", s(doc, "address"))
}

/// One line for the Overview's Email row.
pub fn summary_line(doc: &Value) -> String {
    if !b(doc, "configured") {
        return "mailbox not connected".into();
    }
    let in_use = doc.get("enabled").and_then(Value::as_bool).unwrap_or(true);
    let tools = doc
        .get("agent_tools")
        .map(|a| b(a, "active"))
        .unwrap_or(false);
    let failed = doc
        .get("status")
        .and_then(|st| st.get("last_error"))
        .is_some_and(|e| !e.is_null());
    format!(
        "mailbox {}{} · agent email tools {} · {}",
        s(doc, "address"),
        if in_use { "" } else { " (not in use)" },
        if tools { "on" } else { "off" },
        if failed { "last check failed" } else { "ok" }
    )
}

/// The "Agent email tools" switch state.
pub fn agent_tools_switch(doc: &Value) -> Switch {
    if !b(doc, "configured") {
        return Switch::Unavailable(NO_MAILBOX.into());
    }
    if !doc.get("enabled").and_then(Value::as_bool).unwrap_or(true) {
        return Switch::Unavailable(MAILBOX_NOT_IN_USE.into());
    }
    Switch::from_bool(
        doc.get("agent_tools")
            .map(|a| b(a, "enabled"))
            .unwrap_or(false),
    )
}

/// The "Use this mailbox" switch state.
pub fn use_mailbox_switch(doc: &Value) -> Switch {
    if !b(doc, "configured") {
        return Switch::Unavailable(NO_MAILBOX.into());
    }
    Switch::from_bool(doc.get("enabled").and_then(Value::as_bool).unwrap_or(true))
}

/// Whether "Sign in with <provider>" can start: this version has a
/// built-in client (`oauth_providers` of the status document), or the
/// person typed their own client id. The reason otherwise.
pub fn oauth_availability(doc: &Value, provider: &str, own_client: bool) -> Result<(), String> {
    if own_client {
        return Ok(());
    }
    let entry = doc
        .get("oauth_providers")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().find(|p| s(p, "id") == provider));
    match entry {
        Some(p) if !b(p, "available") => Err(match s(p, "reason") {
            "" => format!(
                "No built-in {} sign-in client in this version: add your own client id under Advanced.",
                provider_label(provider)
            ),
            r => r.to_string(),
        }),
        _ => Ok(()),
    }
}

/// Which mailbox this screen configures (the file first, so a narrow
/// terminal keeps it).
pub fn scope_text(doc: &Value) -> String {
    let file = match s(doc, "config_file") {
        "" => "(default file)",
        f => f,
    };
    format!(
        "Core settings: {file} — the mailbox of this AbstractCore install (a gateway user's mailbox is set in the gateway console, My account)."
    )
}

/// The words of an `oauth_prompt` (what the person must do to sign in).
pub fn oauth_prompt_text(p: &Value) -> String {
    if s(p, "flow") == "device" {
        format!(
            "Open {} and enter the code {}",
            s(p, "verification_uri"),
            s(p, "user_code")
        )
    } else {
        format!(
            "Open this address in a browser on this machine (opened automatically): {}",
            s(p, "authorization_url")
        )
    }
}

// ---------------------------------------------------------------------
// Actions (argv builders; unit-tested below).
// ---------------------------------------------------------------------

const SECURITY: [&str; 2] = ["ssl", "starttls"];

fn security_index(v: &str) -> usize {
    SECURITY.iter().position(|x| *x == v).unwrap_or(0)
}

/// The Other tab's values, as typed (trimmed where the form trims).
#[derive(Clone, Debug, Default)]
pub(crate) struct ConnectFields {
    pub address: String,
    pub display_name: String,
    pub username: String,
    /// Never trimmed: spaces can belong to a password.
    pub password: String,
    pub imap_host: String,
    pub imap_port: String,
    pub imap_security: usize,
    pub folder: String,
    pub smtp_host: String,
    pub smtp_port: String,
    pub smtp_security: usize,
    pub ca_file: String,
}

/// Connect: `email connect --address=… --password-stdin …`, the password
/// on stdin. Without hosts the CLI finds the servers from the address
/// (and refuses with `email_discovery_failed` when it can't). Errors are
/// the form's words.
pub(crate) fn connect_action(
    f: &ConnectFields,
    form_id: Option<u64>,
) -> Result<EmailAction, String> {
    if f.address.is_empty() {
        return Err("Enter the mailbox's email address.".into());
    }
    if f.password.is_empty() {
        return Err("Enter the password (it is stored encrypted and never shown again).".into());
    }
    let secret = StdinSecret::new(f.password.clone())
        .map_err(|_| "The password cannot contain a line break.".to_string())?;
    for (label, p) in [("IMAP port", &f.imap_port), ("SMTP port", &f.smtp_port)] {
        if !p.is_empty() && p.parse::<u16>().map(|n| n == 0).unwrap_or(true) {
            return Err(format!("{label} must be a number (1-65535)."));
        }
    }
    let mut args = vec![
        Arg::p("connect"),
        Arg::p(format!("--address={}", f.address)),
        Arg::p("--password-stdin"),
    ];
    let mut opt = |flag: &str, value: &str| {
        if !value.is_empty() {
            args.push(Arg::p(format!("--{flag}={value}")));
        }
    };
    opt("display-name", &f.display_name);
    opt("username", &f.username);
    if !f.imap_host.is_empty() {
        opt("imap-host", &f.imap_host);
        opt("imap-port", &f.imap_port);
        opt("imap-security", SECURITY[f.imap_security.min(1)]);
    }
    if !f.folder.is_empty() {
        opt("imap-folder", &f.folder);
    }
    if !f.smtp_host.is_empty() {
        opt("smtp-host", &f.smtp_host);
        opt("smtp-port", &f.smtp_port);
        opt("smtp-security", SECURITY[f.smtp_security.min(1)]);
    }
    opt("ca-file", &f.ca_file);
    Ok(EmailAction {
        label: "Connect".to_string(),
        args,
        form_id,
        stdin_secret: Some(secret),
        oauth: false,
        ok_notice: Some(format!("Mailbox connected as {}.", f.address)),
    })
}

/// Tab index → provider id.
const OAUTH_PROVIDERS: [&str; 2] = ["google", "microsoft"];
const OAUTH_FLOWS: [&str; 3] = ["", "device", "loopback"];

/// The Google / Microsoft tab's values (trimmed, except the secret).
#[derive(Clone, Debug, Default)]
pub(crate) struct OAuthFields {
    /// 0 Google, 1 Microsoft (the tab).
    pub provider: usize,
    pub address: String,
    pub client_id: String,
    pub client_secret: String,
    pub tenant: String,
    pub flow: usize,
}

/// Sign in: `email connect --address=… --oauth=… [--client-secret-stdin]`,
/// the client secret (when given) on stdin.
pub(crate) fn oauth_action(f: &OAuthFields, form_id: Option<u64>) -> Result<EmailAction, String> {
    if f.address.is_empty() {
        return Err("Enter the mailbox's email address.".into());
    }
    let prov = OAUTH_PROVIDERS[f.provider.min(1)];
    let mut args = vec![
        Arg::p("connect"),
        Arg::p(format!("--address={}", f.address)),
        Arg::p(format!("--oauth={prov}")),
    ];
    if !f.client_id.is_empty() {
        args.push(Arg::p(format!("--client-id={}", f.client_id)));
    }
    let mut stdin_secret = None;
    if !f.client_secret.is_empty() {
        stdin_secret = Some(
            StdinSecret::new(f.client_secret.clone())
                .map_err(|_| "The client secret cannot contain a line break.".to_string())?,
        );
        args.push(Arg::p("--client-secret-stdin"));
    }
    if prov == "microsoft" && !f.tenant.is_empty() {
        args.push(Arg::p(format!("--tenant={}", f.tenant)));
    }
    let flow = OAUTH_FLOWS[f.flow.min(2)];
    if !flow.is_empty() {
        args.push(Arg::p(format!("--oauth-flow={flow}")));
    }
    Ok(EmailAction {
        label: format!("Sign in with {}", provider_label(prov)),
        args,
        form_id,
        stdin_secret,
        oauth: true,
        ok_notice: Some(format!("Mailbox connected as {}.", f.address)),
    })
}

/// Save the email address: `email registered-address <address>` ("" clears).
pub(crate) fn address_action(addr: &str, form_id: Option<u64>) -> Result<EmailAction, String> {
    let addr = addr.trim();
    if addr.starts_with('-') || (!addr.is_empty() && !plausible_address(addr)) {
        return Err(format!(
            "{addr:?} is not an email address (name@example.com)."
        ));
    }
    Ok(EmailAction {
        label: "Email address".into(),
        args: vec![Arg::p("registered-address"), Arg::p(addr)],
        form_id,
        stdin_secret: None,
        oauth: false,
        ok_notice: Some(if addr.is_empty() {
            "Email address cleared.".into()
        } else {
            "Email address saved.".into()
        }),
    })
}

/// A switch write: the notice names the NEW state.
pub(crate) fn switch_action(feature: &str, on: bool) -> EmailAction {
    let (args, notice) = match feature {
        "agent-tools" => (
            vec![Arg::p("agent-tools"), Arg::p(if on { "on" } else { "off" })],
            format!("Agent email tools are {}.", if on { "on" } else { "off" }),
        ),
        _ => (
            vec![Arg::p(if on { "enable" } else { "disable" })],
            if on {
                "This mailbox is in use.".to_string()
            } else {
                "This mailbox is not in use (settings kept).".to_string()
            },
        ),
    };
    EmailAction {
        label: if feature == "agent-tools" {
            "Agent email tools".into()
        } else {
            "Use this mailbox".into()
        },
        args,
        form_id: None,
        stdin_secret: None,
        oauth: false,
        ok_notice: Some(notice),
    }
}

/// Send limits: both numbers, or the form's words.
pub(crate) fn limits_action(
    per_hour: &str,
    per_day: &str,
    form_id: Option<u64>,
) -> Result<EmailAction, String> {
    let (h, d) = (per_hour.trim(), per_day.trim());
    for (label, v) in [("Per hour", h), ("Per day", d)] {
        if v.parse::<u32>().is_err() {
            return Err(format!("{label} must be a whole number."));
        }
    }
    Ok(EmailAction {
        label: "Send limits".into(),
        args: vec![
            Arg::p("limits"),
            Arg::p("set"),
            Arg::p(format!("--per-hour={h}")),
            Arg::p(format!("--per-day={d}")),
        ],
        form_id,
        stdin_secret: None,
        oauth: false,
        ok_notice: Some(format!("Send limits: {h} per hour, {d} per day.")),
    })
}

/// The folder the mailbox is read from: `email folder <name>` ("" = INBOX).
pub(crate) fn folder_action(name: &str, form_id: Option<u64>) -> Result<EmailAction, String> {
    let name = name.trim();
    if name.starts_with('-') {
        return Err(format!(
            "{name:?}: a folder name here cannot start with \"-\"."
        ));
    }
    Ok(EmailAction {
        label: "Folder".into(),
        args: vec![Arg::p("folder"), Arg::p(name)],
        form_id,
        stdin_secret: None,
        oauth: false,
        ok_notice: Some(format!(
            "The mailbox is read from the folder {}.",
            if name.is_empty() { "INBOX" } else { name }
        )),
    })
}

fn plain_action(label: &str, args: Vec<Arg>, ok_notice: String) -> EmailAction {
    EmailAction {
        label: label.to_string(),
        args,
        form_id: None,
        stdin_secret: None,
        oauth: false,
        ok_notice: Some(ok_notice),
    }
}

fn send(ctx: &Ctx, action: EmailAction) {
    ctx.send(Cmd::Email(Box::new(action)));
}

/// Ask for the mail servers of the mailbox address, once per address.
fn request_discovery(ctx: &Ctx) {
    let ui = ctx.ui.email;
    let addr = ui.mb_address.get_untracked().trim().to_string();
    if !plausible_address(&addr) {
        return;
    }
    let answered = ctx
        .store
        .email_discovery
        .with_untracked(|d| answer_for(d, &addr).is_some());
    let pending = ui
        .pending_discovery
        .with_untracked(|p| p.as_deref() == Some(addr.as_str()));
    if answered || pending {
        return;
    }
    ui.pending_discovery.set(Some(addr.clone()));
    ctx.send(Cmd::EmailDiscover { address: addr });
}

/// Fill the empty Server settings fields from the lookup's answer.
fn prefill_servers(ctx: &Ctx) {
    let ui = ctx.ui.email;
    let addr = ui.mb_address.get_untracked();
    let Some(answer) = ctx
        .store
        .email_discovery
        .with_untracked(|d| answer_for(d, &addr))
    else {
        return;
    };
    let Some(found) = answer.get("result").filter(|r| b(r, "found")) else {
        return;
    };
    let port = |v: &Value| {
        v.get("port")
            .and_then(Value::as_i64)
            .map(|p| p.to_string())
            .unwrap_or_default()
    };
    if let Some(imap) = found.get("imap").filter(|v| v.is_object()) {
        if ui.imap_host.with_untracked(String::is_empty) {
            ui.imap_host.set(s(imap, "host").to_string());
            ui.imap_port.set(port(imap));
            ui.imap_sec.set(security_index(s(imap, "security")));
        }
    }
    if let Some(smtp) = found.get("smtp").filter(|v| v.is_object()) {
        if ui.smtp_host.with_untracked(String::is_empty) {
            ui.smtp_host.set(s(smtp, "host").to_string());
            ui.smtp_port.set(port(smtp));
            ui.smtp_sec.set(security_index(s(smtp, "security")));
        }
    }
    let user = s(found, "username");
    if ui.username.with_untracked(String::is_empty) && !user.is_empty() && user != addr.trim() {
        ui.username.set(user.to_string());
    }
}

fn flash(ui: EmailUi, note: Signal<Option<Result<String, String>>>, text: &str) {
    let g = ui.flash_gen.get_untracked() + 1;
    ui.flash_gen.set(g);
    note.set(Some(Ok(text.to_string())));
    abstracttui::reactive::after(Duration::from_secs(2), move || {
        if ui.flash_gen.get_untracked() == g {
            note.set(None);
        }
    });
}

// ---------------------------------------------------------------------
// The screen.
// ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Shape {
    Loading,
    Failed,
    Ready { configured: bool },
}

fn ensure_loaded(ctx: &Ctx) {
    let store = ctx.store;
    if matches!(store.email.get_untracked(), Loadable::NotAsked) {
        store.email.set(Loadable::Loading);
        ctx.send(Cmd::LoadEmail);
    }
}

fn doc_untracked(ctx: &Ctx) -> Value {
    ctx.store
        .email
        .with_untracked(|e| e.ready().cloned())
        .unwrap_or(Value::Null)
}

pub fn view(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    {
        let ctx_load = ctx.clone();
        cx.effect(move || {
            let _ = store.email.with(|e| matches!(e, Loadable::NotAsked));
            ensure_loaded(&ctx_load);
        });
    }
    // The page remounts only when its SHAPE changes (loading → ready,
    // connected ↔ not connected), never on a status re-read: focus and
    // typed drafts survive every write.
    let shape = cx.signal(Shape::Loading);
    cx.effect(move || {
        let now = store.email.with(|e| match e {
            Loadable::NotAsked | Loadable::Loading => Shape::Loading,
            Loadable::Failed(_) => Shape::Failed,
            Loadable::Ready(d) => Shape::Ready {
                configured: b(d, "configured"),
            },
        });
        if shape.with_untracked(|s| *s != now) {
            shape.set(now);
        }
    });
    install_effects(cx, ctx);

    let ctx_b = ctx.clone();
    // Full width: a max width on this region is not applied when the
    // cards measure their wrapped text, so a cap would let a card size
    // itself for a wider line than it gets (rows overlap).
    let body = dyn_view_scoped(col(), move |gcx| {
        let t = theme.get().tokens;
        match shape.get() {
            Shape::Loading => line(vec![span(
                "⟳ loading email (abstractcore email status --json)…",
                t.info,
            )]),
            Shape::Failed => store.email.with_untracked(|e| match e {
                Loadable::Failed(err) => error_panel(&t, err),
                _ => line(vec![span(String::new(), t.text)]),
            }),
            Shape::Ready { configured } => page(gcx, &ctx_b, theme, configured),
        }
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .focusable()
        .autofocus()
        .child(
            Scroll::new(body)
                .scrollbar_auto_hide(true)
                .layout(LayoutStyle::default().grow(1.0))
                .view(cx),
        )
        .build()
}

/// Seeding, completions and the discovery follow-ups (page scope).
fn install_effects(cx: Scope, ctx: &Ctx) {
    let store = ctx.store;
    let ui = ctx.ui.email;
    let uis = ctx.ui;

    // Seed the drafts from the document: once for the mailbox address,
    // and again whenever the STORED email address or limits change (a
    // save lands, another console wrote).
    cx.effect(move || {
        let Some(doc) = store.email.with(|e| e.ready().cloned()) else {
            return;
        };
        let stored = doc
            .get("registered_address_stored")
            .and_then(Value::as_str)
            .unwrap_or_else(|| s(&doc, "registered_address"))
            .to_string();
        if ui
            .seen_address
            .with_untracked(|a| a.as_deref() != Some(&stored))
        {
            ui.seen_address.set(Some(stored.clone()));
            ui.address.set(stored.clone());
        }
        if !ui.seeded.get_untracked() {
            ui.seeded.set(true);
            let mb = if stored.is_empty() {
                s(&doc, "address").to_string()
            } else {
                stored
            };
            ui.mb_address.set(mb);
            if s(&doc, "auth_kind") == "password" {
                ui.tab.set(2);
                ui.tab_chosen.set(true);
            } else if let Some(p) = doc.get("oauth").map(|o| s(o, "provider")) {
                ui.tab.set(usize::from(p == "microsoft"));
                ui.tab_chosen.set(true);
            }
        }
        let folder = doc
            .get("imap")
            .map(|i| s(i, "folder").to_string())
            .filter(|f| !f.is_empty())
            .unwrap_or_else(|| "INBOX".into());
        if ui
            .seen_folder
            .with_untracked(|f| f.as_deref() != Some(&folder))
        {
            ui.seen_folder.set(Some(folder.clone()));
            ui.folder_edit.set(folder);
        }
        if let Some(lim) = doc.get("limits").filter(|l| l.is_object()) {
            let n = |k: &str| lim.get(k).and_then(Value::as_i64).unwrap_or(0);
            let now = (n("per_hour"), n("per_day"));
            if ui.seen_limits.get_untracked() != Some(now) {
                ui.seen_limits.set(Some(now));
                ui.per_hour.set(now.0.to_string());
                ui.per_day.set(now.1.to_string());
            }
        }
    });

    // Completions of this screen's writes (by form id).
    cx.effect(move || {
        let Some((fid, outcome)) = uis.write_done.get() else {
            return;
        };
        if fid == ui.fid_address {
            uis.write_done.set(None);
            match outcome {
                Ok(_) => flash(ui, ui.address_note, "Saved"),
                Err(e) => ui.address_note.set(Some(Err(e))),
            }
        } else if fid == ui.fid_connect {
            uis.write_done.set(None);
            ui.connecting.set(false);
            match outcome {
                Ok(_) => {
                    ui.password.set(String::new());
                    ui.client_secret.set(String::new());
                    ui.connect_error.set(None);
                    ui.confirm_disconnect.set(false);
                }
                Err(e) => ui.connect_error.set(Some(e)),
            }
        } else if fid == ui.fid_folder {
            uis.write_done.set(None);
            match outcome {
                Ok(_) => flash(ui, ui.folder_note, "Saved"),
                Err(e) => ui.folder_note.set(Some(Err(e))),
            }
        } else if fid == ui.fid_limits {
            uis.write_done.set(None);
            match outcome {
                Ok(_) => flash(ui, ui.limits_note, "Saved"),
                Err(e) => ui.limits_note.set(Some(Err(e))),
            }
        }
    });

    // A lookup answered: it is no longer pending, and when it found
    // nothing for the address in the field, Server settings open with
    // the reason.
    cx.effect(move || {
        let answer = store.email_discovery.get();
        let addr = ui.mb_address.get();
        if let Some(a) = answer.as_ref() {
            let who = s(a, "address").to_string();
            if ui
                .pending_discovery
                .with_untracked(|p| p.as_deref() == Some(who.as_str()))
            {
                ui.pending_discovery.set(None);
            }
        }
        // Until the person picks a tab, the lookup picks the one that
        // matches the address: Google, Microsoft, or Other (any other
        // provider, or none found: the servers are typed by hand).
        if let Some(found) = answer_for(&answer, &addr).and_then(|a| a.get("result").cloned()) {
            if !ui.tab_chosen.get_untracked() {
                let tab = match s(&found, "provider") {
                    "google" => 0,
                    "microsoft" => 1,
                    _ => 2,
                };
                if ui.tab.get_untracked() != tab {
                    ui.tab.set(tab);
                }
            }
        }
        match answer_for(&answer, &addr) {
            Some(a) if a.get("result").is_some_and(|r| !b(r, "found")) => {
                ui.servers_reason.set(Some(format!(
                    "Couldn't find the mail servers for {}. Enter them here.",
                    domain_of(&addr)
                )));
                ui.servers_folded.set(false);
            }
            Some(a)
                if a.get("result").is_some_and(|r| b(r, "found"))
                    && ui.servers_reason.with_untracked(Option::is_some) =>
            {
                ui.servers_reason.set(None);
            }
            _ => {}
        }
    });

    // Connect refused because the CLI found no servers either.
    cx.effect(move || {
        if store.email_error_code.get().as_deref() == Some("email_discovery_failed") {
            let addr = ui.mb_address.get_untracked();
            ui.servers_reason.set(Some(format!(
                "Couldn't find the mail servers for {}. Enter them here.",
                domain_of(&addr)
            )));
            ui.servers_folded.set(false);
        }
    });
}

fn heading(t: &TokenSet, text: &str) -> View {
    line(vec![span_bold(format!(" {text}"), t.text)])
}

/// Muted helper text that WRAPS at the card width (a 60-column
/// terminal keeps every word) and measures its own height.
fn helper(t: &TokenSet, text: &str) -> View {
    wrapped(text.to_string(), t.text_muted)
}

fn wrapped(text: String, ink: Rgba) -> View {
    wrapped_styled(text, ink, false)
}

/// `wrapped` in bold (status lines).
fn wrapped_bold(text: String, ink: Rgba) -> View {
    wrapped_styled(text, ink, true)
}

fn wrapped_styled(text: String, ink: Rgba, bold: bool) -> View {
    let measured = text.clone();
    Element::new()
        .style(LayoutStyle::default().width(Dimension::Percent(1.0)))
        .measure(move |avail| abstracttui::text::measure(&measured, avail))
        .draw(move |canvas, rect| {
            if rect.is_empty() {
                return;
            }
            let mut style = abstracttui::render::Style::new().fg(ink);
            if bold {
                style = style.bold();
            }
            for (i, row) in abstracttui::text::wrap(&text, rect.w)
                .iter()
                .take(rect.h.max(0) as usize)
                .enumerate()
            {
                canvas.print_styled(Point::new(rect.x, rect.y + i as i32), row, &style);
            }
        })
        .build()
}

fn input(
    cx: Scope,
    t: &TokenSet,
    label: &str,
    value: Signal<String>,
    masked: bool,
    max_w: i32,
) -> View {
    field_w(
        t,
        label,
        LABEL_W,
        TextInput::new()
            .layout(LayoutStyle::default().grow(1.0).max_w(max_w).h(1))
            .value(value)
            .masked(masked)
            .view(cx),
    )
}

fn danger_style() -> ButtonStyle {
    ButtonStyle {
        fg: TokenId::Error,
        ..ButtonStyle::default()
    }
}

/// A full-width column (reactive regions whose content wraps must know
/// their width when measured).
fn col() -> LayoutStyle {
    LayoutStyle::column().width(Dimension::Percent(1.0))
}

/// A card: a titled block whose body keeps one cell off the border.
fn card(t: &TokenSet, title: &str, body: View) -> View {
    Block::new()
        .title(title)
        .child(
            Element::new()
                .style(LayoutStyle::column().padding(Edges::hv(1, 0)))
                .child(body)
                .build(),
        )
        .element(t)
        .build()
}

fn empty() -> View {
    Element::new().style(LayoutStyle::default().h(0)).build()
}

fn page(
    cx: Scope,
    ctx: &Ctx,
    theme: Signal<&'static abstracttui::theme::Theme>,
    configured: bool,
) -> View {
    let t = theme.get_untracked().tokens;
    let store = ctx.store;
    let doc = doc_untracked(ctx);
    let mut col = Element::new().style(LayoutStyle::column());
    if let Some(notices) = doc.get("notices").and_then(Value::as_array) {
        for n in notices.iter().filter_map(Value::as_str) {
            col = col.child(wrapped(format!(" ⚠ {n}"), t.warn));
        }
    }
    col = col
        .child(address_card(cx, ctx, theme))
        .child(mailbox_card(cx, ctx, theme, configured))
        .child(agent_tools_card(cx, ctx, theme))
        .child(advanced(cx, ctx, theme))
        .child(dyn_view(LayoutStyle::line(1), move || {
            let t = theme.get().tokens;
            let text = store
                .email
                .with(|e| e.ready().map(scope_text).unwrap_or_default());
            line(vec![span(format!(" {text}"), t.text_faint)])
        }));
    col.build()
}

fn address_card(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let t = theme.get_untracked().tokens;
    let ui = ctx.ui.email;
    let save = {
        let ctx = ctx.clone();
        move || match address_action(&ui.address.get_untracked(), Some(ui.fid_address)) {
            Ok(a) => {
                ui.address_note.set(None);
                send(&ctx, a);
            }
            Err(e) => ui.address_note.set(Some(Err(e))),
        }
    };
    let save2 = save.clone();
    let row = Element::new()
        .style(LayoutStyle::row().gap(1))
        .child(field_w(
            &t,
            "Email address",
            LABEL_W,
            TextInput::new()
                .layout(LayoutStyle::default().grow(1.0).max_w(44).h(1))
                .value(ui.address)
                .on_submit(move |_| save2())
                .view(cx),
        ))
        .child(Button::new("Save").on_click(save).view(cx))
        .child(dyn_view(
            LayoutStyle::row().h(1).w(9).shrink(0.0),
            move || {
                let t = theme.get().tokens;
                match ui.address_note.get() {
                    Some(Ok(w)) => line(vec![span_bold(format!("✓ {w}"), t.ok)]),
                    _ => empty(),
                }
            },
        ))
        .build();
    card(
        &t,
        "Email address",
        Element::new()
            .style(LayoutStyle::column())
            .child(row)
            .child(dyn_view(col(), move || {
                let t = theme.get().tokens;
                match ui.address_note.get() {
                    Some(Err(e)) => wrapped_bold(format!("✗ {e}"), t.error),
                    _ => empty(),
                }
            }))
            .child(helper(&t, ADDRESS_HELP))
            .build(),
    )
}

fn mailbox_card(
    cx: Scope,
    ctx: &Ctx,
    theme: Signal<&'static abstracttui::theme::Theme>,
    configured: bool,
) -> View {
    let t = theme.get_untracked().tokens;
    let body = if configured {
        connected_view(ctx, theme)
    } else {
        not_connected_view(cx, ctx, theme)
    };
    card(&t, "Mailbox", body)
}

fn connected_view(ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui.email;
    let status = dyn_view(col(), move || {
        let t = theme.get().tokens;
        let Some(doc) = store.email.with(|e| e.ready().cloned()) else {
            return empty();
        };
        let mut col = Element::new()
            .style(LayoutStyle::column())
            .child(wrapped_bold(connected_line(&doc, now_secs()), t.ok));
        if !doc.get("enabled").and_then(Value::as_bool).unwrap_or(true) {
            col = col.child(wrapped(MAILBOX_NOT_IN_USE.to_string(), t.warn));
        }
        if let Some(err) = doc
            .get("status")
            .and_then(|st| st.get("last_error"))
            .filter(|e| e.is_object())
        {
            col = col.child(wrapped(
                format!("Last check failed: {}", s(err, "cause")),
                t.error,
            ));
            if !s(err, "fix").is_empty() {
                col = col.child(wrapped(format!("Fix: {}", s(err, "fix")), t.text));
            }
        }
        if !s(&doc, "secret_warning").is_empty() {
            col = col.child(wrapped(format!("⚠ {}", s(&doc, "secret_warning")), t.warn));
        }
        col.build()
    });
    let actions = {
        let ctx = ctx.clone();
        dyn_view_scoped(col(), move |acx| {
            let t = theme.get().tokens;
            if ui.confirm_disconnect.get() {
                let ctx_yes = ctx.clone();
                Element::new()
                    .style(LayoutStyle::column())
                    .child(helper(&t, DISCONNECT_CONFIRM))
                    .child(
                        Element::new()
                            .style(LayoutStyle::row().gap(2).h(1))
                            .child(
                                Button::new("Disconnect")
                                    .style(danger_style())
                                    .on_click(move || {
                                        ui.confirm_disconnect.set(false);
                                        send(
                                            &ctx_yes,
                                            plain_action(
                                                "Disconnect",
                                                vec![Arg::p("disconnect"), Arg::p("--yes")],
                                                "Mailbox disconnected. Policy and limits are kept."
                                                    .into(),
                                            ),
                                        );
                                    })
                                    .view(acx),
                            )
                            .child(
                                Button::new("Cancel")
                                    .on_click(move || ui.confirm_disconnect.set(false))
                                    .element(acx, &t)
                                    .autofocus()
                                    .build(),
                            )
                            .build(),
                    )
                    .build()
            } else {
                let ctx_test = ctx.clone();
                Element::new()
                    .style(LayoutStyle::row().gap(2).h(1))
                    .child(
                        Button::new("Test")
                            .on_click(move || {
                                send(
                                    &ctx_test,
                                    plain_action(
                                        "Test",
                                        vec![Arg::p("test")],
                                        "Connection test passed (IMAP and SMTP).".into(),
                                    ),
                                )
                            })
                            .view(acx),
                    )
                    .child(
                        Button::new("Disconnect")
                            .style(danger_style())
                            .on_click(move || ui.confirm_disconnect.set(true))
                            .view(acx),
                    )
                    .build()
            }
        })
    };
    Element::new()
        .style(LayoutStyle::column())
        .child(status)
        .child(actions)
        .build()
}

fn not_connected_view(
    cx: Scope,
    ctx: &Ctx,
    theme: Signal<&'static abstracttui::theme::Theme>,
) -> View {
    let t = theme.get_untracked().tokens;
    let ui = ctx.ui.email;
    let store = ctx.store;
    // A prefilled address is looked up at once (it picks the tab).
    request_discovery(ctx);
    // The bar only (panels need a scope of their own: built below).
    let mut tabs = Tabs::new();
    for title in TABS {
        tabs = tabs.tab(title, || abstracttui::ui::text(""));
    }
    let bar = tabs
        .active(ui.tab)
        .on_change(move |_| ui.tab_chosen.set(true))
        .layout(LayoutStyle::column().h(2).shrink(0.0))
        .element(cx, &t)
        .build();
    let ctx_p = ctx.clone();
    let panel = dyn_view_scoped(col(), move |pcx| {
        let tab = ui.tab.get();
        if tab == 2 {
            other_panel(pcx, &ctx_p, theme)
        } else {
            oauth_panel(pcx, &ctx_p, theme, tab)
        }
    });
    let ctx_c = ctx.clone();
    let waiting = dyn_view_scoped(col(), move |wcx| {
        let t = theme.get().tokens;
        if !ui.connecting.get() {
            return empty();
        }
        let text = match store.email_oauth_prompt.get() {
            Some(p) => oauth_prompt_text(&p),
            None if ui.tab.get_untracked() < 2 => "starting the sign-in…".into(),
            None => "connecting — signing in to the IMAP and SMTP servers…".into(),
        };
        let _ = &ctx_c;
        let mut col = Element::new()
            .style(LayoutStyle::column())
            .child(wrapped_bold(format!("⟳ {text}"), t.info));
        if ui.tab.get_untracked() < 2 {
            col = col.child(
                Element::new()
                    .style(LayoutStyle::row().h(1))
                    .child(
                        Button::new("Cancel sign-in")
                            .on_click(cancel_email_oauth)
                            .view(wcx),
                    )
                    .build(),
            );
        }
        col.build()
    });
    let error = dyn_view(col(), move || {
        let t = theme.get().tokens;
        match ui.connect_error.get() {
            Some(e) => wrapped_bold(format!("✗ {e}"), t.error),
            None => empty(),
        }
    });
    Element::new()
        .style(LayoutStyle::column())
        .child(bar)
        .child(panel)
        .child(waiting)
        .child(error)
        .build()
}

fn oauth_panel(
    cx: Scope,
    ctx: &Ctx,
    theme: Signal<&'static abstracttui::theme::Theme>,
    tab: usize,
) -> View {
    let t = theme.get_untracked().tokens;
    let ui = ctx.ui.email;
    let store = ctx.store;
    let provider = OAUTH_PROVIDERS[tab.min(1)];
    let ctx_btn = ctx.clone();
    let button = dyn_view_scoped(col(), move |bcx| {
        let t = theme.get().tokens;
        let own = ui.client_id.with(|c| !c.trim().is_empty());
        let avail = store.email.with(|e| {
            e.ready()
                .map(|d| oauth_availability(d, provider, own))
                .unwrap_or(Ok(()))
        });
        let ctx2 = ctx_btn.clone();
        let label = format!("Sign in with {}", provider_label(provider));
        let start = move || {
            if ui.connecting.get_untracked() {
                return;
            }
            let v = |sig: Signal<String>| sig.get_untracked().trim().to_string();
            let f = OAuthFields {
                provider: tab,
                address: v(ui.mb_address),
                client_id: v(ui.client_id),
                client_secret: ui.client_secret.get_untracked(),
                tenant: v(ui.tenant),
                flow: ui.flow.get_untracked(),
            };
            match oauth_action(&f, Some(ui.fid_connect)) {
                Ok(a) => {
                    ui.connect_error.set(None);
                    ui.connecting.set(true);
                    send(&ctx2, a);
                }
                Err(e) => ui.connect_error.set(Some(e)),
            }
        };
        let mut col = Element::new().style(LayoutStyle::column()).child(
            Element::new()
                .style(LayoutStyle::row().h(1))
                .child(
                    Button::new(label)
                        .disabled(avail.is_err())
                        .on_click(start)
                        .view(bcx),
                )
                .build(),
        );
        if let Err(reason) = avail {
            col = col.child(helper(&t, &reason));
        }
        col.build()
    });
    let advanced = Disclosure::new("Advanced: your own sign-in client")
        .folded(ui.oauth_folded)
        .max_body_rows(0)
        .body(move |dcx| {
            let t = theme.get_untracked().tokens;
            let mut col = Element::new()
                .style(LayoutStyle::column())
                .child(input(dcx, &t, "Client id", ui.client_id, false, 60))
                .child(input(dcx, &t, "Client secret", ui.client_secret, true, 60));
            if tab == 1 {
                col = col.child(input(dcx, &t, "Tenant", ui.tenant, false, 40));
            }
            col.child(field_w(
                &t,
                "Sign-in flow",
                LABEL_W,
                Select::new(vec![
                    SelectOption::new("provider default"),
                    SelectOption::new("device code"),
                    SelectOption::new("browser on this machine"),
                ])
                .value(ui.flow)
                .layout(LayoutStyle::default().grow(1.0).max_w(30).h(1))
                .view(dcx),
            ))
            .child(helper(
                &t,
                "No client id = the built-in AbstractFramework client, when this version has one. Tokens are stored encrypted.",
            ))
            .build()
        })
        .view(cx);
    Element::new()
        .style(LayoutStyle::column())
        .child(mailbox_address_input(cx, ctx, &t))
        .child(button)
        .child(advanced)
        .build()
}

/// The mailbox address field (every tab): the servers are looked up
/// when it loses focus or is submitted.
fn mailbox_address_input(cx: Scope, ctx: &Ctx, t: &TokenSet) -> View {
    let ui = ctx.ui.email;
    let focused = cx.signal(false);
    let was_focused = cx.signal(false);
    {
        let ctx = ctx.clone();
        cx.effect(move || {
            let f = focused.get();
            if was_focused.get_untracked() && !f {
                request_discovery(&ctx);
            }
            was_focused.set(f);
        });
    }
    let ctx_submit = ctx.clone();
    field_w(
        t,
        "Email address",
        LABEL_W,
        TextInput::new()
            .layout(LayoutStyle::default().grow(1.0).max_w(44).h(1))
            .value(ui.mb_address)
            .on_submit(move |_| request_discovery(&ctx_submit))
            .element(cx, t)
            .focus_signal(focused)
            .build(),
    )
}

fn other_panel(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let t = theme.get_untracked().tokens;
    let ui = ctx.ui.email;
    let store = ctx.store;
    let address = mailbox_address_input(cx, ctx, &t);
    let summary = dyn_view(col(), move || {
        let t = theme.get().tokens;
        let addr = ui.mb_address.get().trim().to_string();
        let answer = store.email_discovery.with(|d| answer_for(d, &addr));
        let pending = ui
            .pending_discovery
            .with(|p| p.as_deref() == Some(addr.as_str()));
        if !plausible_address(&addr) {
            return helper(&t, "The mail servers are found from your email address.");
        }
        match answer {
            None if pending => line(vec![span(
                format!("⟳ finding the mail servers for {}…", domain_of(&addr)),
                t.info,
            )]),
            None => helper(
                &t,
                &format!(
                    "Press Enter in the address field to find the mail servers for {}.",
                    domain_of(&addr)
                ),
            ),
            Some(a) => {
                match a.get("result") {
                    Some(found) if b(found, "found") => {
                        let mut col = Element::new()
                            .style(LayoutStyle::column())
                            .child(wrapped(servers_summary(found), t.text));
                        let prov = s(found, "provider");
                        if !prov.is_empty() {
                            let p = provider_label(prov);
                            col = col.child(helper(
                            &t,
                            &format!("This is a {p} mailbox: the {p} tab signs in without a password."),
                        ));
                        }
                        col.build()
                    }
                    Some(_) => wrapped(
                        format!(
                            "Couldn't find the mail servers for {}. Enter them in Server settings.",
                            domain_of(&addr)
                        ),
                        t.warn,
                    ),
                    None => wrapped(
                        format!("Couldn't look up the mail servers: {}", s(&a, "error")),
                        t.warn,
                    ),
                }
            }
        }
    });
    let ctx_prefill = ctx.clone();
    let servers = Disclosure::new("Server settings")
        .folded(ui.servers_folded)
        .max_body_rows(0)
        .on_toggle(move |folded| {
            if !folded {
                prefill_servers(&ctx_prefill);
            }
        })
        .body(move |dcx| {
            let t = theme.get_untracked().tokens;
            let sec = || vec![SelectOption::new("SSL"), SelectOption::new("STARTTLS")];
            let reason = dyn_view(col(), move || {
                let t = theme.get().tokens;
                match ui.servers_reason.get() {
                    Some(r) => wrapped_bold(r, t.warn),
                    None => empty(),
                }
            });
            // Port and security side by side on a wide terminal, one
            // above the other on a narrow one (no wrapping row: a
            // wrapped row measures one line and paints two).
            let wide = use_viewport(dcx).get_untracked().w >= 100;
            let pair = |_dcx: Scope, a: View, b_: View| {
                let cell = |w: i32, v: View| {
                    Element::new()
                        .style(LayoutStyle::default().w(w).h(1).shrink(0.0))
                        .child(v)
                        .build()
                };
                if wide {
                    Element::new()
                        .style(LayoutStyle::row().gap(2).h(1))
                        .child(cell(LABEL_W + 1 + 8, a))
                        .child(cell(13 + 1 + 14, b_))
                        .build()
                } else {
                    Element::new()
                        .style(LayoutStyle::column())
                        .child(a)
                        .child(b_)
                        .build()
                }
            };
            Element::new()
                .style(LayoutStyle::column())
                .child(reason)
                .child(input(dcx, &t, "IMAP host", ui.imap_host, false, 44))
                .child(pair(
                    dcx,
                    input(dcx, &t, "IMAP port", ui.imap_port, false, 7),
                    field_w(
                        &t,
                        "IMAP security",
                        if wide { 13 } else { LABEL_W },
                        Select::new(sec())
                            .value(ui.imap_sec)
                            .layout(LayoutStyle::default().w(14).h(1))
                            .view(dcx),
                    ),
                ))
                .child(input(dcx, &t, "Folder", ui.folder, false, 30))
                .child(input(dcx, &t, "SMTP host", ui.smtp_host, false, 44))
                .child(pair(
                    dcx,
                    input(dcx, &t, "SMTP port", ui.smtp_port, false, 7),
                    field_w(
                        &t,
                        "SMTP security",
                        if wide { 13 } else { LABEL_W },
                        Select::new(sec())
                            .value(ui.smtp_sec)
                            .layout(LayoutStyle::default().w(14).h(1))
                            .view(dcx),
                    ),
                ))
                .child(input(dcx, &t, "User name", ui.username, false, 44))
                .child(input(dcx, &t, "Display name", ui.display_name, false, 44))
                .child(input(dcx, &t, "CA file", ui.ca_file, false, 60))
                .child(helper(
                    &t,
                    "Empty fields use what was found; the user name defaults to the address, the folder to INBOX.",
                ))
                .build()
        })
        .view(cx);
    let ctx_connect = ctx.clone();
    let connect = move || {
        if ui.connecting.get_untracked() {
            return;
        }
        let v = |sig: Signal<String>| sig.get_untracked().trim().to_string();
        let f = ConnectFields {
            address: v(ui.mb_address),
            display_name: v(ui.display_name),
            username: v(ui.username),
            password: ui.password.get_untracked(),
            imap_host: v(ui.imap_host),
            imap_port: v(ui.imap_port),
            imap_security: ui.imap_sec.get_untracked(),
            folder: v(ui.folder),
            smtp_host: v(ui.smtp_host),
            smtp_port: v(ui.smtp_port),
            smtp_security: ui.smtp_sec.get_untracked(),
            ca_file: v(ui.ca_file),
        };
        match connect_action(&f, Some(ui.fid_connect)) {
            Ok(a) => {
                ui.connect_error.set(None);
                ui.connecting.set(true);
                send(&ctx_connect, a);
            }
            Err(e) => ui.connect_error.set(Some(e)),
        }
    };
    Element::new()
        .style(LayoutStyle::column())
        .child(address)
        .child(input(cx, &t, "Password", ui.password, true, 44))
        .child(helper(&t, PASSWORD_HELP))
        .child(summary)
        .child(servers)
        .child(
            Element::new()
                .style(LayoutStyle::row().h(1))
                .child(Button::new("Connect").on_click(connect).view(cx))
                .build(),
        )
        .build()
}

fn agent_tools_card(
    cx: Scope,
    ctx: &Ctx,
    theme: Signal<&'static abstracttui::theme::Theme>,
) -> View {
    let t = theme.get_untracked().tokens;
    let store = ctx.store;
    let ctx_sw = ctx.clone();
    let notice = store.notice;
    card(
        &t,
        "Agent email tools",
        Element::new()
            .style(LayoutStyle::column())
            .child(
                SwitchRow::new(
                    "Agent email tools",
                    move || {
                        store
                            .email
                            .with(|e| e.ready().map(agent_tools_switch))
                            .unwrap_or(Switch::Unavailable(NO_MAILBOX.into()))
                    },
                    move |on| send(&ctx_sw, switch_action("agent-tools", on)),
                )
                .on_refused(move |r| notice.set(Some(format!("Agent email tools: {r}"))))
                .view(cx),
            )
            .child(helper(&t, AGENT_TOOLS_DESC))
            .build(),
    )
}

fn advanced(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let ui = ctx.ui.email;
    let ctx_b = ctx.clone();
    Disclosure::new("Advanced")
        .folded(ui.advanced_folded)
        .max_body_rows(0)
        .body(move |bcx| advanced_body(bcx, &ctx_b, theme))
        .view(cx)
}

fn advanced_body(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let t = theme.get_untracked().tokens;
    let store = ctx.store;
    let ui = ctx.ui.email;

    // Recipient rules: the mode applies on change, entries on add/remove.
    let mode = cx.signal(0usize);
    let entries: Signal<Vec<String>> = cx.signal(Vec::new());
    cx.effect(move || {
        let (m, e) = store.email.with(|e| {
            let pol = e
                .ready()
                .and_then(|d| d.get("policy").cloned())
                .unwrap_or(Value::Null);
            let m = usize::from(s(&pol, "mode") == "denylist");
            let list: Vec<String> = pol
                .get("entries")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            (m, list)
        });
        if mode.get_untracked() != m {
            mode.set(m);
        }
        if entries.with_untracked(|x| *x != e) {
            entries.set(e);
        }
    });
    let ctx_mode = ctx.clone();
    let mode_select = field_w(
        &t,
        "Mode",
        LABEL_W,
        Select::new(vec![
            SelectOption::new("Only these recipients (allowlist)"),
            SelectOption::new("Everyone except these (denylist)"),
        ])
        .value(mode)
        .layout(LayoutStyle::default().grow(1.0).max_w(38).h(1))
        .on_change(move |i| {
            let (m, words) = if i == 1 {
                ("denylist", "everyone except the listed recipients")
            } else {
                ("allowlist", "only the listed recipients")
            };
            send(
                &ctx_mode,
                plain_action(
                    "Recipient rules",
                    vec![
                        Arg::p("policy"),
                        Arg::p("set"),
                        Arg::p(format!("--mode={m}")),
                    ],
                    format!("Recipient rules: {words}."),
                ),
            );
        })
        .view(cx),
    );
    let ctx_rm = ctx.clone();
    let list = dyn_view_scoped(col(), move |lcx| {
        let t = theme.get().tokens;
        let items = entries.get();
        let mut col = Element::new().style(LayoutStyle::column());
        if items.is_empty() {
            col = col.child(helper(
                &t,
                if mode.get() == 0 {
                    "No entries: an empty allowlist refuses every recipient."
                } else {
                    "No entries: every recipient is allowed."
                },
            ));
        }
        for e in items {
            let ctx2 = ctx_rm.clone();
            let entry = e.clone();
            col = col.child(
                Element::new()
                    .style(LayoutStyle::row().gap(1).h(1))
                    .child(line_w(&t, &format!("  · {e}"), 40))
                    .child(
                        Button::new("Remove")
                            .on_click(move || {
                                send(
                                    &ctx2,
                                    plain_action(
                                        "Recipient rules",
                                        vec![
                                            Arg::p("policy"),
                                            Arg::p("set"),
                                            Arg::p(format!("--remove={entry}")),
                                        ],
                                        format!("Removed {entry} from the recipient rules."),
                                    ),
                                )
                            })
                            .view(lcx),
                    )
                    .build(),
            );
        }
        col.build()
    });
    let add = {
        let ctx = ctx.clone();
        move || {
            let e = ui.policy_add.get_untracked().trim().to_string();
            if e.is_empty() {
                return;
            }
            ui.policy_add.set(String::new());
            send(
                &ctx,
                plain_action(
                    "Recipient rules",
                    vec![
                        Arg::p("policy"),
                        Arg::p("set"),
                        Arg::p(format!("--add={e}")),
                    ],
                    format!("Added {e} to the recipient rules."),
                ),
            );
        }
    };
    let add2 = add.clone();
    let add_row = Element::new()
        .style(LayoutStyle::row().gap(1).h(1))
        .child(field_w(
            &t,
            "Add",
            LABEL_W,
            TextInput::new()
                .layout(LayoutStyle::default().grow(1.0).max_w(36).h(1))
                .value(ui.policy_add)
                .on_submit(move |_| add2())
                .view(cx),
        ))
        .child(Button::new("Add").on_click(add).view(cx))
        .build();

    // Send limits: saved on Enter, or when a field loses focus.
    let save_limits = {
        let ctx = ctx.clone();
        move || {
            let (h, d) = (ui.per_hour.get_untracked(), ui.per_day.get_untracked());
            let same = ui
                .seen_limits
                .get_untracked()
                .is_some_and(|(sh, sd)| h.trim() == sh.to_string() && d.trim() == sd.to_string());
            let pending = (h.trim().to_string(), d.trim().to_string());
            if same || ui.limits_sent.get_untracked().as_ref() == Some(&pending) {
                return;
            }
            match limits_action(&h, &d, Some(ui.fid_limits)) {
                Ok(a) => {
                    ui.limits_note.set(None);
                    ui.limits_sent.set(Some(pending));
                    send(&ctx, a);
                }
                Err(e) => ui.limits_note.set(Some(Err(e))),
            }
        }
    };
    let limit_input = |label: &str, value: Signal<String>, label_w: i32| {
        let focused = cx.signal(false);
        let was = cx.signal(false);
        let save = save_limits.clone();
        cx.effect(move || {
            let f = focused.get();
            if was.get_untracked() && !f {
                save();
            }
            was.set(f);
        });
        let save2 = save_limits.clone();
        field_w(
            &t,
            label,
            label_w,
            TextInput::new()
                .layout(LayoutStyle::default().w(7).h(1))
                .value(value)
                .on_submit(move |_| save2())
                .element(cx, &t)
                .focus_signal(focused)
                .build(),
        )
    };
    let limits_row = Element::new()
        .style(LayoutStyle::row().gap(2))
        .child(limit_input("Per hour", ui.per_hour, LABEL_W))
        .child(limit_input("Per day", ui.per_day, 8))
        .child(dyn_view(
            LayoutStyle::row().h(1).w(9).shrink(0.0),
            move || {
                let t = theme.get().tokens;
                match ui.limits_note.get() {
                    Some(Ok(w)) => line(vec![span_bold(format!("✓ {w}"), t.ok)]),
                    _ => empty(),
                }
            },
        ))
        .build();
    let limits_error = dyn_view(col(), move || {
        let t = theme.get().tokens;
        match ui.limits_note.get() {
            Some(Err(e)) => wrapped_bold(format!("✗ {e}"), t.error),
            _ => empty(),
        }
    });
    let usage = dyn_view(col(), move || {
        let t = theme.get().tokens;
        let lim = store
            .email
            .with(|e| e.ready().and_then(|d| d.get("limits").cloned()))
            .unwrap_or(Value::Null);
        let n = |k: &str| lim.get(k).and_then(Value::as_i64).unwrap_or(0);
        helper(
            &t,
            &format!(
                "{} sent in the last hour, {} in the last day.",
                n("used_last_hour"),
                n("used_last_day")
            ),
        )
    });
    let save_folder = {
        let ctx = ctx.clone();
        move || {
            let name = ui.folder_edit.get_untracked();
            let trimmed = name.trim().to_string();
            if ui.seen_folder.get_untracked().as_deref() == Some(trimmed.as_str())
                || ui.folder_sent.get_untracked().as_deref() == Some(trimmed.as_str())
            {
                return;
            }
            let connected = store
                .email
                .with_untracked(|e| e.ready().map(|d| b(d, "configured")).unwrap_or(false));
            if !connected {
                ui.folder_note.set(Some(Err(NO_MAILBOX.into())));
                return;
            }
            match folder_action(&name, Some(ui.fid_folder)) {
                Ok(a) => {
                    ui.folder_note.set(None);
                    ui.folder_sent.set(Some(trimmed));
                    send(&ctx, a);
                }
                Err(e) => ui.folder_note.set(Some(Err(e))),
            }
        }
    };
    let folder_focused = cx.signal(false);
    let folder_was = cx.signal(false);
    {
        let save = save_folder.clone();
        cx.effect(move || {
            let f = folder_focused.get();
            if folder_was.get_untracked() && !f {
                save();
            }
            folder_was.set(f);
        });
    }
    let folder = Element::new()
        .style(LayoutStyle::column())
        .child(
            Element::new()
                .style(LayoutStyle::row().gap(2).h(1))
                .child(field_w(
                    &t,
                    "Folder",
                    LABEL_W,
                    TextInput::new()
                        .layout(LayoutStyle::default().w(24).h(1))
                        .value(ui.folder_edit)
                        .on_submit(move |_| save_folder())
                        .element(cx, &t)
                        .focus_signal(folder_focused)
                        .build(),
                ))
                .child(dyn_view(
                    LayoutStyle::row().h(1).w(9).shrink(0.0),
                    move || {
                        let t = theme.get().tokens;
                        match ui.folder_note.get() {
                            Some(Ok(w)) => line(vec![span_bold(format!("✓ {w}"), t.ok)]),
                            _ => empty(),
                        }
                    },
                ))
                .build(),
        )
        .child(dyn_view(col(), move || {
            let t = theme.get().tokens;
            match ui.folder_note.get() {
                Some(Err(e)) => wrapped_bold(format!("✗ {e}"), t.error),
                _ => empty(),
            }
        }))
        .child(helper(&t, "The folder your agents read. Empty = INBOX."))
        .build();
    let ctx_use = ctx.clone();
    let notice = store.notice;
    let use_mailbox = SwitchRow::new(
        "Use this mailbox",
        move || {
            store
                .email
                .with(|e| e.ready().map(use_mailbox_switch))
                .unwrap_or(Switch::Unavailable(NO_MAILBOX.into()))
        },
        move |on| send(&ctx_use, switch_action("use-mailbox", on)),
    )
    .on_refused(move |r| notice.set(Some(format!("Use this mailbox: {r}"))))
    .view(cx);

    Element::new()
        .style(LayoutStyle::column())
        .child(heading(&t, "Recipient rules"))
        .child(mode_select)
        .child(list)
        .child(add_row)
        .child(helper(
            &t,
            "Exact addresses or domains (a subdomain only as its own entry). To, Cc and Bcc are checked; one refused recipient refuses the message.",
        ))
        .child(heading(&t, "Send limits"))
        .child(limits_row)
        .child(limits_error)
        .child(usage)
        .child(folder)
        .child(use_mailbox)
        .child(helper(&t, USE_MAILBOX_DESC))
        .build()
}

fn line_w(t: &TokenSet, text: &str, w: i32) -> View {
    Element::new()
        .style(LayoutStyle::default().w(w).h(1).shrink(1.0))
        .child(line(vec![span(text.to_string(), t.text)]))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::build_command;
    use crate::worker::email_argv;
    use serde_json::json;
    use std::path::Path;

    const PW: &str = " pw-SENTINEL-7f3a -x ";
    const CS: &str = "cs-SENTINEL-91be";

    fn fields() -> ConnectFields {
        ConnectFields {
            address: "me@example.test".into(),
            password: PW.into(),
            imap_host: "imap.example.test".into(),
            smtp_host: "smtp.example.test".into(),
            smtp_port: "587".into(),
            smtp_security: 1,
            ..ConnectFields::default()
        }
    }

    /// The args of the process the worker would spawn for this action.
    fn spawned_args(action: &EmailAction) -> Vec<String> {
        let argv = email_argv(action);
        let command = build_command(
            Path::new("/usr/local/bin/abstractcore"),
            &argv,
            action.stdin_secret.is_some(),
        );
        command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// Connect: the password is nowhere on the command line (nor in
    /// Debug); the command reads it from stdin, as one line, untrimmed.
    #[test]
    fn connect_puts_the_password_on_stdin_never_in_argv() {
        let action = connect_action(&fields(), Some(7)).unwrap();
        let args = spawned_args(&action);
        assert!(args.iter().all(|a| !a.contains("SENTINEL")), "{args:?}");
        assert_eq!(
            &args[..4],
            [
                "email",
                "connect",
                "--address=me@example.test",
                "--password-stdin"
            ]
        );
        assert!(
            args.contains(&"--smtp-security=starttls".to_string()),
            "{args:?}"
        );
        assert_eq!(args.last().map(String::as_str), Some("--json"));
        assert_eq!(
            action.stdin_secret.as_ref().unwrap().line(),
            format!("{PW}\n")
        );
        assert!(!format!("{action:?}").contains("SENTINEL"));
        assert!(!action.oauth);
        assert_eq!(
            action.ok_notice.as_deref(),
            Some("Mailbox connected as me@example.test.")
        );
    }

    /// Address + password only: no host flags, the CLI finds the servers.
    #[test]
    fn connect_without_servers_lets_the_cli_discover_them() {
        let f = ConnectFields {
            address: "me@fastmail.test".into(),
            password: "pw".into(),
            ..ConnectFields::default()
        };
        let args = spawned_args(&connect_action(&f, None).unwrap());
        assert_eq!(
            args,
            [
                "email",
                "connect",
                "--address=me@fastmail.test",
                "--password-stdin",
                "--json"
            ]
        );
    }

    #[test]
    fn connect_refuses_what_the_form_refuses() {
        let mut f = fields();
        f.password = "two\nlines".into();
        assert!(connect_action(&f, None).unwrap_err().contains("line break"));
        f.password.clear();
        assert!(connect_action(&f, None)
            .unwrap_err()
            .starts_with("Enter the password"));
        let mut f = fields();
        f.imap_port = "0".into();
        assert_eq!(
            connect_action(&f, None).unwrap_err(),
            "IMAP port must be a number (1-65535)."
        );
        let mut f = fields();
        f.address.clear();
        assert_eq!(
            connect_action(&f, None).unwrap_err(),
            "Enter the mailbox's email address."
        );
    }

    /// OAuth2 sign-in: an own client secret goes on stdin with
    /// `--client-secret-stdin`; without one, stdin carries nothing.
    #[test]
    fn oauth_puts_the_client_secret_on_stdin_never_in_argv() {
        let f = OAuthFields {
            provider: 1,
            address: "me@example.test".into(),
            client_id: "my-client".into(),
            client_secret: CS.into(),
            tenant: "contoso".into(),
            flow: 1,
        };
        let action = oauth_action(&f, Some(3)).unwrap();
        let args = spawned_args(&action);
        assert!(args.iter().all(|a| !a.contains("SENTINEL")), "{args:?}");
        for want in [
            "--oauth=microsoft",
            "--client-id=my-client",
            "--client-secret-stdin",
            "--tenant=contoso",
            "--oauth-flow=device",
        ] {
            assert!(args.contains(&want.to_string()), "{want} missing: {args:?}");
        }
        assert_eq!(
            action.stdin_secret.as_ref().unwrap().line(),
            format!("{CS}\n")
        );
        assert!(!format!("{action:?}").contains("SENTINEL"));
        assert!(action.oauth);
        assert_eq!(action.label, "Sign in with Microsoft");

        let google = oauth_action(
            &OAuthFields {
                provider: 0,
                client_secret: String::new(),
                ..f
            },
            None,
        )
        .unwrap();
        assert!(google.stdin_secret.is_none());
        let args = spawned_args(&google);
        assert!(args.contains(&"--oauth=google".to_string()), "{args:?}");
        assert!(!args.iter().any(|a| a.contains("client-secret")));
        assert!(
            !args.iter().any(|a| a.starts_with("--tenant")),
            "tenant is Microsoft's"
        );
    }

    #[test]
    fn switch_actions_name_the_new_state() {
        let on = switch_action("agent-tools", true);
        assert_eq!(spawned_args(&on), ["email", "agent-tools", "on", "--json"]);
        assert_eq!(on.ok_notice.as_deref(), Some("Agent email tools are on."));
        let off = switch_action("use-mailbox", false);
        assert_eq!(spawned_args(&off), ["email", "disable", "--json"]);
        assert_eq!(
            off.ok_notice.as_deref(),
            Some("This mailbox is not in use (settings kept).")
        );
        assert_eq!(
            spawned_args(&switch_action("use-mailbox", true)),
            ["email", "enable", "--json"]
        );
    }

    #[test]
    fn address_and_limits_actions() {
        let a = address_action(" me@example.test ", Some(1)).unwrap();
        assert_eq!(
            spawned_args(&a),
            ["email", "registered-address", "me@example.test", "--json"]
        );
        assert_eq!(a.ok_notice.as_deref(), Some("Email address saved."));
        let clear = address_action("", None).unwrap();
        assert_eq!(
            spawned_args(&clear),
            ["email", "registered-address", "", "--json"]
        );
        assert!(address_action("--yes", None).is_err(), "never a flag");
        assert!(address_action("not an address", None).is_err());
        let l = limits_action("5", " 50", None).unwrap();
        assert_eq!(
            spawned_args(&l),
            [
                "email",
                "limits",
                "set",
                "--per-hour=5",
                "--per-day=50",
                "--json"
            ]
        );
        assert_eq!(
            limits_action("x", "1", None).unwrap_err(),
            "Per hour must be a whole number."
        );
    }

    #[test]
    fn folder_action_sets_the_folder_or_inbox() {
        let a = folder_action(" Archive ", Some(4)).unwrap();
        assert_eq!(spawned_args(&a), ["email", "folder", "Archive", "--json"]);
        assert_eq!(
            a.ok_notice.as_deref(),
            Some("The mailbox is read from the folder Archive.")
        );
        let inbox = folder_action("", None).unwrap();
        assert_eq!(spawned_args(&inbox), ["email", "folder", "", "--json"]);
        assert!(folder_action("--yes", None).is_err(), "never a flag");
    }

    #[test]
    fn switch_states_follow_the_document() {
        let none = json!({"configured": false});
        assert_eq!(
            agent_tools_switch(&none),
            Switch::Unavailable(NO_MAILBOX.into())
        );
        assert_eq!(
            use_mailbox_switch(&none),
            Switch::Unavailable(NO_MAILBOX.into())
        );
        let on = json!({"configured": true, "enabled": true,
                        "agent_tools": {"enabled": true, "active": true}});
        assert_eq!(agent_tools_switch(&on), Switch::On);
        assert_eq!(use_mailbox_switch(&on), Switch::On);
        let paused = json!({"configured": true, "enabled": false,
                            "agent_tools": {"enabled": true}});
        assert_eq!(
            agent_tools_switch(&paused),
            Switch::Unavailable(MAILBOX_NOT_IN_USE.into())
        );
        assert_eq!(use_mailbox_switch(&paused), Switch::Off);
    }

    #[test]
    fn oauth_availability_reads_the_providers_list() {
        let doc = json!({"oauth_providers": [
            {"id": "google", "available": false, "reason": "No built-in Google sign-in client in this version: add your own client id under Advanced."},
            {"id": "microsoft", "available": true, "reason": null}]});
        assert!(oauth_availability(&doc, "google", false)
            .unwrap_err()
            .starts_with("No built-in Google"));
        assert!(oauth_availability(&doc, "google", true).is_ok());
        assert!(oauth_availability(&doc, "microsoft", false).is_ok());
    }

    #[test]
    fn times_and_summaries() {
        let t = parse_rfc3339("2026-09-29T20:00:00+00:00").unwrap();
        assert_eq!(parse_rfc3339("2026-09-29T22:00:00.123+02:00"), Some(t));
        assert_eq!(parse_rfc3339("2026-09-29T20:00:00Z"), Some(t));
        assert_eq!(
            ago("2026-09-29T20:00:00+00:00", t + 125).unwrap(),
            "2 min ago"
        );
        assert_eq!(ago("2026-09-29T20:00:00Z", t + 30).unwrap(), "just now");
        assert_eq!(
            ago("2026-09-29T20:00:00Z", t + 86_400).unwrap(),
            "1 day ago"
        );
        assert!(ago("", t).is_none());
        let found = json!({"found": true,
            "imap": {"host": "imap.fastmail.com", "port": 993, "security": "ssl"},
            "smtp": {"host": "smtp.fastmail.com", "port": 465, "security": "ssl"}});
        assert_eq!(
            servers_summary(&found),
            "imap.fastmail.com · 993 · SSL  ·  smtp.fastmail.com · 465 · SSL"
        );
        let doc = json!({"address": "me@x.test", "auth_kind": "oauth2",
                         "oauth": {"provider": "google"},
                         "status": {"last_test": "2026-09-29T20:00:00Z"}});
        assert_eq!(
            connected_line(&doc, t + 120),
            "Connected as me@x.test · Google · checked 2 min ago"
        );
        assert!(plausible_address("me@fastmail.com"));
        for bad in [
            "me",
            "me@",
            "@x.com",
            "me@x",
            "-me@x.com",
            "m e@x.com",
            "a@b@c.com",
        ] {
            assert!(!plausible_address(bad), "{bad}");
        }
    }
}
