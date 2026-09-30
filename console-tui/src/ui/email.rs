//! Email: the account of this AbstractCore install — the web console's
//! Email tab, field for field and word for word (abstractcore/console/
//! web.py `_EMAIL_HTML`).
//!
//! Everything goes through `abstractcore email … --json` (the worker's
//! `Cmd::LoadEmail` / `Cmd::Email`): the CLI validates, tests the
//! connection, seals the password (encrypted, never printed) and owns
//! the recipient policy and send limits. This screen only renders the
//! `email_settings_v1` document and builds argv.
//!
//! Verbs: `c` connect (Account form; Save and test), `g` Sign in with
//! OAuth2 (Microsoft / Google; the device code or sign-in address shows
//! while the command waits for the approval), `t` Test, `o` Turn off /
//! Turn on, `x` Disconnect (confirm), `p` Recipient policy, `l` Send
//! limits, `a` Agent email tools on/off (default off; `abstractcore
//! email agent-tools on|off`). Every value is passed as `--flag=value`,
//! so an entry starting with `-` cannot be read as a flag. A password or
//! OAuth client secret never goes on the command line (every local user
//! can read argv with `ps` while the command runs): the command gets
//! `--password-stdin` / `--client-secret-stdin` and the secret is
//! written to its stdin (`EmailAction::stdin_secret`).
//!
//! The account shown is the one of this AbstractCore install (the core
//! settings file); a gateway keeps its own account per user, configured
//! in the gateway console (My email).

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Button};
use serde_json::Value;

use crate::store::Loadable;
use crate::worker::{cancel_email_oauth, next_form_id, Cmd, EmailAction};
use crate::writes::{Arg, StdinSecret};

use super::forms::{
    confirm_danger, install_dirty_guard, install_write_done, message_slot, open_form_guarded,
};
use super::util::{error_panel, field, line, span, span_bold};
use super::Ctx;

/// The footer's verbs for this screen (same words as the web console).
pub const HINTS: &[(&str, &str)] = &[
    ("c", "connect"),
    ("g", "OAuth2 sign-in"),
    ("t", "test"),
    ("o", "turn on/off"),
    ("x", "disconnect"),
    ("p", "recipient policy"),
    ("l", "send limits"),
    ("a", "agent email tools"),
];

fn s<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

fn leg_text(leg: Option<&Value>) -> String {
    match leg {
        None => "-".into(),
        Some(l) => match l.get("ok").and_then(Value::as_bool) {
            Some(true) => "ok".into(),
            Some(false) => format!(
                "failed: {}",
                l.get("cause").and_then(Value::as_str).unwrap_or("?")
            ),
            None => "not configured".into(),
        },
    }
}

fn server_text(v: Option<&Value>, with_folder: bool) -> String {
    match v {
        Some(x) if x.is_object() => {
            let mut out = format!(
                "{}:{} {}",
                s(x, "host"),
                x.get("port").and_then(Value::as_i64).unwrap_or(0),
                s(x, "security")
            );
            if with_folder {
                out.push_str(&format!(", folder {}", s(x, "folder")));
            }
            out
        }
        _ => "not configured".into(),
    }
}

/// One line for the Overview's Email row.
pub fn summary_line(doc: &Value) -> String {
    if !doc
        .get("configured")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return "not connected".into();
    }
    let on = doc.get("enabled").and_then(Value::as_bool).unwrap_or(true);
    let pol = doc.get("policy").cloned().unwrap_or(Value::Null);
    format!(
        "{} ({}) · policy {} · {}",
        s(doc, "address"),
        if on { "on" } else { "off" },
        s(&pol, "mode"),
        doc.get("status")
            .and_then(|st| st.get("last_error"))
            .filter(|e| !e.is_null())
            .map(|_| "last test failed")
            .unwrap_or("ok")
    )
}

fn ensure_loaded(ctx: &Ctx) {
    let store = ctx.store;
    if matches!(store.email.get_untracked(), Loadable::NotAsked) {
        store.email.set(Loadable::Loading);
        ctx.send(Cmd::LoadEmail);
    }
}

fn send_action(ctx: &Ctx, label: &str, args: Vec<Arg>, form_id: Option<u64>) {
    ctx.send(Cmd::Email(Box::new(EmailAction {
        label: label.to_string(),
        args,
        form_id,
        stdin_secret: None,
        oauth: false,
    })));
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

/// The "Agent email tools" line (the gateway consoles' words).
pub fn agent_tools_text(doc: &Value) -> String {
    let at = doc.get("agent_tools").cloned().unwrap_or(Value::Null);
    if at.get("active").and_then(Value::as_bool).unwrap_or(false) {
        return "on: your agents and workflows have the email tools (policy, limits and approval still apply)".into();
    }
    match s(&at, "reason") {
        "" => "off".into(),
        reason => format!("off — {reason}"),
    }
}

/// Which account this screen configures: the core settings of this
/// install (the file first, so a narrow terminal keeps it).
pub fn scope_text(doc: &Value) -> String {
    let file = s(doc, "config_file");
    let file = if file.is_empty() {
        "(default file)"
    } else {
        file
    };
    format!(
        "Core settings: {file} — the email account of this AbstractCore install (a gateway user's account is set in the gateway console, My email)."
    )
}

fn kv(t: &TokenSet, label: &str, value: String) -> View {
    line(vec![
        span(format!("  {:<20}", label), t.text_muted),
        span(value, t.text),
    ])
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
    let (c_connect, c_test, c_toggle, c_disc, c_pol, c_lim, c_oauth, c_agent) = (
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
    );

    let body = dyn_view_scoped(LayoutStyle::column().grow(1.0), move |_gcx| {
        let t = theme.get().tokens;
        match store.email.get() {
            Loadable::NotAsked | Loadable::Loading => line(vec![span(
                "⟳ loading email (abstractcore email status --json)…",
                t.info,
            )]),
            Loadable::Failed(e) => error_panel(&t, &e),
            Loadable::Ready(doc) => {
                let prompt = store.email_oauth_prompt.get();
                let page = render(&t, &doc);
                match prompt {
                    Some(p) => Element::new()
                        .style(LayoutStyle::column())
                        .child(line(vec![span_bold(
                            format!(" Sign-in waiting: {}", oauth_prompt_text(&p)),
                            t.info,
                        )]))
                        .child(page)
                        .build(),
                    None => page,
                }
            }
        }
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .focusable()
        .autofocus()
        .shortcut(KeyChord::plain(Key::Char('c')), move |_| {
            open_connect_form(cx, &c_connect)
        })
        .shortcut(KeyChord::plain(Key::Char('g')), move |_| {
            open_oauth_form(cx, &c_oauth)
        })
        .shortcut(KeyChord::plain(Key::Char('t')), move |_| {
            if configured(&c_test) {
                send_action(&c_test, "email test", vec![Arg::p("test")], None);
            }
        })
        .shortcut(KeyChord::plain(Key::Char('o')), move |_| {
            let on = c_toggle.store.email.with_untracked(|e| {
                e.ready()
                    .and_then(|d| d.get("enabled").and_then(Value::as_bool))
                    .unwrap_or(true)
            });
            let (label, verb) = if on {
                ("Turn off", "disable")
            } else {
                ("Turn on", "enable")
            };
            send_action(
                &c_toggle,
                &format!("email {label}"),
                vec![Arg::p(verb)],
                None,
            );
        })
        .shortcut(KeyChord::plain(Key::Char('x')), move |_| {
            if !configured(&c_disc) {
                return;
            }
            let ctx2 = c_disc.clone();
            confirm_danger(
                cx,
                c_disc.ui,
                "Disconnect deletes the stored password or tokens (policy and limits are kept)."
                    .into(),
                "Disconnect now",
                "Cancel",
                move || {
                    send_action(
                        &ctx2,
                        "email Disconnect",
                        vec![Arg::p("disconnect"), Arg::p("--yes")],
                        None,
                    )
                },
            );
        })
        .shortcut(KeyChord::plain(Key::Char('p')), move |_| {
            open_policy_form(cx, &c_pol)
        })
        .shortcut(KeyChord::plain(Key::Char('l')), move |_| {
            open_limits_form(cx, &c_lim)
        })
        .shortcut(KeyChord::plain(Key::Char('a')), move |_| {
            let on = c_agent.store.email.with_untracked(|e| {
                e.ready()
                    .and_then(|d| d.get("agent_tools"))
                    .and_then(|a| a.get("enabled"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            });
            let state = if on { "off" } else { "on" };
            send_action(
                &c_agent,
                &format!("email Agent email tools {state}"),
                vec![Arg::p("agent-tools"), Arg::p(state)],
                None,
            );
        })
        .child(body)
        .build()
}

fn configured(ctx: &Ctx) -> bool {
    let ok = ctx.store.email.with_untracked(|e| {
        e.ready()
            .and_then(|d| d.get("configured").and_then(Value::as_bool))
            .unwrap_or(false)
    });
    if !ok {
        ctx.store.notice.set(Some(
            "no email account is connected — press c to connect one".into(),
        ));
    }
    ok
}

fn render(t: &TokenSet, doc: &Value) -> View {
    let configured = doc
        .get("configured")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let enabled = doc.get("enabled").and_then(Value::as_bool).unwrap_or(true);
    let state = if !configured {
        span_bold("○ not connected", t.text_muted)
    } else if !enabled {
        span_bold("● off", t.warn)
    } else {
        span_bold("● connected", t.ok)
    };
    let storage = match s(doc, "secret_storage") {
        "os-keychain" => "Credentials encrypted, key in the OS keychain",
        "key-file" => "Credentials encrypted, key in a 0600 file",
        _ => "",
    };
    let mut col = Element::new().style(LayoutStyle::column()).child(line(vec![
        span(" Email  ", t.text_faint),
        state,
        span(format!("  {storage}"), t.text_muted),
    ]));
    if let Some(notices) = doc.get("notices").and_then(Value::as_array) {
        for n in notices.iter().filter_map(Value::as_str) {
            col = col.child(line(vec![span(format!(" ⚠ {n}"), t.warn)]));
        }
    }
    if !s(doc, "secret_warning").is_empty() {
        col = col.child(line(vec![span(
            format!(" ⚠ {}", s(doc, "secret_warning")),
            t.warn,
        )]));
    }

    let username = s(doc, "username");
    let account = Element::new()
        .style(LayoutStyle::column())
        .child(kv(
            t,
            "Address",
            if configured {
                s(doc, "address").to_string()
            } else {
                "not connected".into()
            },
        ))
        .child(kv(t, "Display name", s(doc, "display_name").to_string()))
        .child(kv(t, "User name", username.to_string()))
        .child(kv(t, "Sign-in", s(doc, "auth_kind").to_string()))
        .child(kv(t, "IMAP (read)", server_text(doc.get("imap"), true)))
        .child(kv(t, "SMTP (send)", server_text(doc.get("smtp"), false)))
        .child(kv(
            t,
            "Registered address",
            s(doc, "registered_address").to_string(),
        ))
        .child(kv(t, "Agent email tools", agent_tools_text(doc)))
        .build();

    let st = doc.get("status").cloned().unwrap_or(Value::Null);
    let legs = st.get("legs").cloned().unwrap_or(Value::Null);
    let mut status = Element::new()
        .style(LayoutStyle::column())
        .child(kv(
            t,
            "Last test",
            if s(&st, "last_test").is_empty() {
                "never".into()
            } else {
                s(&st, "last_test").to_string()
            },
        ))
        .child(kv(t, "IMAP test", leg_text(legs.get("imap"))))
        .child(kv(t, "SMTP test", leg_text(legs.get("smtp"))));
    if let Some(err) = st.get("last_error").filter(|e| e.is_object()) {
        status = status
            .child(line(vec![
                span("  Last error          ", t.text_muted),
                span(s(err, "cause").to_string(), t.error),
            ]))
            .child(line(vec![
                span("                      Fix: ", t.text_muted),
                span(s(err, "fix").to_string(), t.text),
            ]));
    }

    let pol = doc.get("policy").cloned().unwrap_or(Value::Null);
    let entries: Vec<String> = pol
        .get("entries")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mode = s(&pol, "mode");
    let mut policy = Element::new().style(LayoutStyle::column()).child(kv(
        t,
        "Mode",
        match mode {
            "allowlist" => "allowlist: only these recipients".into(),
            "denylist" => "denylist: everyone except these".into(),
            other => other.to_string(),
        },
    ));
    if entries.is_empty() {
        policy = policy.child(line(vec![span(
            if mode == "allowlist" {
                "  No entries: an empty allowlist refuses every recipient."
            } else {
                "  No entries: every recipient is allowed."
            },
            t.text_muted,
        )]));
    }
    for e in &entries {
        policy = policy.child(line(vec![span(format!("  · {e}"), t.text)]));
    }

    let lim = doc.get("limits").cloned().unwrap_or(Value::Null);
    let n = |k: &str| lim.get(k).and_then(Value::as_i64).unwrap_or(0);
    let limits = Element::new()
        .style(LayoutStyle::column())
        .child(kv(
            t,
            "Per hour",
            format!(
                "{} ({} sent in the last hour)",
                n("per_hour"),
                n("used_last_hour")
            ),
        ))
        .child(kv(
            t,
            "Per day",
            format!("{} ({} in the last day)", n("per_day"), n("used_last_day")),
        ))
        .build();

    col.child(Block::new().title("Account").child(account).element(t).build())
        .child(Block::new().title("Status").child(status.build()).element(t).build())
        .child(Block::new().title("Recipient policy").child(policy.build()).element(t).build())
        .child(Block::new().title("Send limits").child(limits).element(t).build())
        .child(line(vec![span(format!(" {}", scope_text(doc)), t.text_faint)]))
        .child(line(vec![span(
            " a: Agent email tools on/off (default off; they work only with a connected, turned-on account). The mailbox is only read (never marked read, moved or deleted).",
            t.text_faint,
        )]))
        .build()
}

// ---------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------

fn text_row(
    mcx: Scope,
    t: &TokenSet,
    label: &str,
    value: Signal<String>,
    placeholder: &str,
    masked: bool,
) -> View {
    field(
        t,
        label,
        TextInput::new()
            .layout(LayoutStyle::default().grow(1.0).h(1))
            .value(value)
            .masked(masked)
            .placeholder(placeholder)
            .view(mcx),
    )
}

const SECURITY: [&str; 2] = ["ssl", "starttls"];

/// The Account form's values, as typed (trimmed where the form trims).
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
    pub registered: String,
}

/// Save and test: `email connect --address=… --password-stdin …`, the
/// password on stdin. Errors are the form's words.
pub(crate) fn connect_action(
    f: &ConnectFields,
    form_id: Option<u64>,
) -> Result<EmailAction, String> {
    if f.address.is_empty() {
        return Err("Address is required".into());
    }
    if f.password.is_empty() {
        return Err("Password is required (it is stored encrypted and never shown again)".into());
    }
    let secret = StdinSecret::new(f.password.clone())
        .map_err(|_| "The password cannot contain a line break".to_string())?;
    if f.imap_host.is_empty() && f.smtp_host.is_empty() {
        return Err("give an IMAP host (read) and/or an SMTP host (send)".into());
    }
    for (label, p) in [("IMAP port", &f.imap_port), ("SMTP port", &f.smtp_port)] {
        if !p.is_empty() && p.parse::<u16>().map(|n| n == 0).unwrap_or(true) {
            return Err(format!("{label} must be a number (1-65535)"));
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
        opt("imap-folder", &f.folder);
    }
    if !f.smtp_host.is_empty() {
        opt("smtp-host", &f.smtp_host);
        opt("smtp-port", &f.smtp_port);
        opt("smtp-security", SECURITY[f.smtp_security.min(1)]);
    }
    opt("ca-file", &f.ca_file);
    opt("registered-address", &f.registered);
    Ok(EmailAction {
        label: "email Save and test".to_string(),
        args,
        form_id,
        stdin_secret: Some(secret),
        oauth: false,
    })
}

fn security_index(v: &str) -> usize {
    SECURITY.iter().position(|x| *x == v).unwrap_or(0)
}

/// The Account form (`c`): the web console's fields in its order.
fn open_connect_form(cx: Scope, ctx: &Ctx) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let doc = ctx
        .store
        .email
        .with_untracked(|e| e.ready().cloned())
        .unwrap_or(Value::Null);
    open_form_guarded(ctx, cx, Size::new(86, 24), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let imap = doc.get("imap").cloned().unwrap_or(Value::Null);
        let smtp = doc.get("smtp").cloned().unwrap_or(Value::Null);
        let port = |v: &Value| {
            v.get("port")
                .and_then(Value::as_i64)
                .map(|p| p.to_string())
                .unwrap_or_default()
        };
        let address = mcx.signal(s(&doc, "address").to_string());
        let display_name = mcx.signal(s(&doc, "display_name").to_string());
        let username0 = if s(&doc, "username") == s(&doc, "address") {
            String::new()
        } else {
            s(&doc, "username").to_string()
        };
        let username = mcx.signal(username0);
        let password = mcx.signal(String::new());
        let imap_host = mcx.signal(s(&imap, "host").to_string());
        let imap_port = mcx.signal(port(&imap));
        let imap_sec = mcx.signal(security_index(s(&imap, "security")));
        let folder = mcx.signal(s(&imap, "folder").to_string());
        let smtp_host = mcx.signal(s(&smtp, "host").to_string());
        let smtp_port = mcx.signal(port(&smtp));
        let smtp_sec = mcx.signal(security_index(s(&smtp, "security")));
        let ca = if !s(&imap, "ca_file").is_empty() {
            s(&imap, "ca_file")
        } else {
            s(&smtp, "ca_file")
        };
        let ca_file = mcx.signal(ca.to_string());
        let reg0 = if s(&doc, "registered_address") == s(&doc, "address") {
            String::new()
        } else {
            s(&doc, "registered_address").to_string()
        };
        let registered = mcx.signal(reg0);

        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        install_dirty_guard(
            mcx,
            &guard,
            vec![
                (address, address.get_untracked()),
                (display_name, display_name.get_untracked()),
                (username, username.get_untracked()),
                (password, String::new()),
                (imap_host, imap_host.get_untracked()),
                (imap_port, imap_port.get_untracked()),
                (folder, folder.get_untracked()),
                (smtp_host, smtp_host.get_untracked()),
                (smtp_port, smtp_port.get_untracked()),
                (ca_file, ca_file.get_untracked()),
                (registered, registered.get_untracked()),
            ],
            esc_armed,
            form_error,
        );
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let v = |sig: Signal<String>| sig.get_untracked().trim().to_string();
            let fields = ConnectFields {
                address: v(address),
                display_name: v(display_name),
                username: v(username),
                password: password.get_untracked(),
                imap_host: v(imap_host),
                imap_port: v(imap_port),
                imap_security: imap_sec.get_untracked(),
                folder: v(folder),
                smtp_host: v(smtp_host),
                smtp_port: v(smtp_port),
                smtp_security: smtp_sec.get_untracked(),
                ca_file: v(ca_file),
                registered: v(registered),
            };
            match connect_action(&fields, Some(form_id)) {
                Ok(action) => {
                    in_flight.set(true);
                    ctx3.send(Cmd::Email(Box::new(action)));
                }
                Err(e) => form_error.set(Some(e)),
            }
        };
        let sec_opts = || vec![SelectOption::new("SSL"), SelectOption::new("STARTTLS")];
        Block::new()
            .title("Account — Save and test")
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(0))
                    .child(text_row(mcx, &t, "Address", address, "me@example.com", false))
                    .child(text_row(mcx, &t, "Display name", display_name, "", false))
                    .child(text_row(mcx, &t, "User name", username, "(the address)", false))
                    .child(text_row(mcx, &t, "Password", password, "app password", true))
                    .child(text_row(mcx, &t, "IMAP host", imap_host, "imap.example.com", false))
                    .child(text_row(mcx, &t, "IMAP port", imap_port, "993", false))
                    .child(field(&t, "IMAP security", Select::new(sec_opts()).value(imap_sec).view(mcx)))
                    .child(text_row(mcx, &t, "Folder", folder, "INBOX", false))
                    .child(text_row(mcx, &t, "SMTP host", smtp_host, "smtp.example.com", false))
                    .child(text_row(mcx, &t, "SMTP port", smtp_port, "465", false))
                    .child(field(&t, "SMTP security", Select::new(sec_opts()).value(smtp_sec).view(mcx)))
                    .child(text_row(mcx, &t, "CA file", ca_file, "(system trust store)", false))
                    .child(text_row(mcx, &t, "Registered address", registered, "(the address)", false))
                    .child(line(vec![span(
                        "  Passwords are stored encrypted and never shown again. Many providers need an app password.",
                        t.text_faint,
                    )]))
                    .child(message_slot(theme, form_error, in_flight))
                    .child(
                        Element::new()
                            .style(LayoutStyle::row().gap(2).shrink(0.0))
                            .child(Button::new("Save and test").on_click(submit).view(mcx))
                            .child(Button::new("Cancel").on_click({
                                let close = close.clone();
                                move || close()
                            }).view(mcx))
                            .build(),
                    )
                    .build(),
            )
            .element(&t)
            .build()
    });
}

const OAUTH_PROVIDERS: [&str; 2] = ["microsoft", "google"];
const OAUTH_FLOWS: [&str; 3] = ["", "device", "loopback"];

/// The OAuth2 form's values (trimmed, except the client secret).
#[derive(Clone, Debug, Default)]
pub(crate) struct OAuthFields {
    pub provider: usize,
    pub address: String,
    pub client_id: String,
    pub client_secret: String,
    pub tenant: String,
    pub flow: usize,
}

/// Start sign-in: `email connect --address=… --oauth=… [--client-secret-stdin]`,
/// the client secret (when given) on stdin.
pub(crate) fn oauth_action(f: &OAuthFields, form_id: Option<u64>) -> Result<EmailAction, String> {
    if f.address.is_empty() {
        return Err("Address is required".into());
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
                .map_err(|_| "The client secret cannot contain a line break".to_string())?,
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
        label: "email OAuth2 sign-in".to_string(),
        args,
        form_id,
        stdin_secret,
        oauth: true,
    })
}

/// The OAuth2 sign-in form (`g`): the web console's "Sign in with
/// OAuth2" card, field for field. Submitting runs `abstractcore email
/// connect --oauth … --json`, which waits for the approval; the form
/// stays open showing the device code or sign-in address (streamed from
/// the command's `oauth_prompt` line) until it connects or fails.
/// "Cancel sign-in" kills the command (nothing is stored).
fn open_oauth_form(cx: Scope, ctx: &Ctx) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let doc = ctx
        .store
        .email
        .with_untracked(|e| e.ready().cloned())
        .unwrap_or(Value::Null);
    open_form_guarded(ctx, cx, Size::new(90, 16), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let oauth = doc.get("oauth").cloned().unwrap_or(Value::Null);
        let provider = mcx.signal(if s(&oauth, "provider") == "google" {
            1usize
        } else {
            0
        });
        let address = mcx.signal(s(&doc, "address").to_string());
        let own_client = s(&oauth, "client_source") != "builtin";
        let client_id = mcx.signal(if own_client {
            s(&oauth, "client_id").to_string()
        } else {
            String::new()
        });
        let client_secret = mcx.signal(String::new());
        let tenant = mcx.signal(match s(&oauth, "tenant") {
            "common" => String::new(),
            other => other.to_string(),
        });
        let flow = mcx.signal(0usize);
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        install_dirty_guard(
            mcx,
            &guard,
            vec![
                (address, address.get_untracked()),
                (client_id, client_id.get_untracked()),
                (client_secret, String::new()),
                (tenant, tenant.get_untracked()),
            ],
            esc_armed,
            form_error,
        );
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let v = |sig: Signal<String>| sig.get_untracked().trim().to_string();
            let fields = OAuthFields {
                provider: provider.get_untracked(),
                address: v(address),
                client_id: v(client_id),
                client_secret: client_secret.get_untracked(),
                tenant: v(tenant),
                flow: flow.get_untracked(),
            };
            match oauth_action(&fields, Some(form_id)) {
                Ok(action) => {
                    in_flight.set(true);
                    ctx3.send(Cmd::Email(Box::new(action)));
                }
                Err(e) => form_error.set(Some(e)),
            }
        };
        let store = ctx2.store;
        let prompt_line = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
            let t = theme.get().tokens;
            match store.email_oauth_prompt.get() {
                Some(p) if in_flight.get() => line(vec![span_bold(
                    format!("  {}", oauth_prompt_text(&p)),
                    t.info,
                )]),
                _ if in_flight.get() => line(vec![span("  starting the sign-in…", t.info)]),
                _ => line(vec![span(String::new(), t.text)]),
            }
        });
        Block::new()
            .title("Sign in with OAuth2")
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(0))
                    .child(field(
                        &t,
                        "Provider",
                        Select::new(vec![
                            SelectOption::new("Microsoft (Outlook, Microsoft 365)"),
                            SelectOption::new("Google (Gmail)"),
                        ])
                        .value(provider)
                        .view(mcx),
                    ))
                    .child(text_row(mcx, &t, "Address", address, "me@outlook.com", false))
                    .child(text_row(mcx, &t, "Client id", client_id, "(built-in client)", false))
                    .child(text_row(mcx, &t, "Client secret", client_secret, "(none)", true))
                    .child(text_row(mcx, &t, "Tenant", tenant, "common (Microsoft only)", false))
                    .child(field(
                        &t,
                        "Sign-in flow",
                        Select::new(vec![
                            SelectOption::new("provider default"),
                            SelectOption::new("device code"),
                            SelectOption::new("browser on this machine"),
                        ])
                        .value(flow)
                        .view(mcx),
                    ))
                    .child(line(vec![span(
                        "  No client id = the built-in AbstractFramework client, when registered; else your own client. Tokens are stored encrypted.",
                        t.text_faint,
                    )]))
                    .child(prompt_line)
                    .child(message_slot(theme, form_error, in_flight))
                    .child(
                        Element::new()
                            .style(LayoutStyle::row().gap(2).shrink(0.0))
                            .child(Button::new("Start sign-in").on_click(submit).view(mcx))
                            .child(Button::new("Cancel sign-in").on_click({
                                let close = close.clone();
                                move || {
                                    if in_flight.get_untracked() {
                                        cancel_email_oauth();
                                    }
                                    close()
                                }
                            }).view(mcx))
                            .build(),
                    )
                    .build(),
            )
            .element(&t)
            .build()
    });
}

/// The Recipient policy form (`p`): mode + entries (comma-separated).
fn open_policy_form(cx: Scope, ctx: &Ctx) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let pol = ctx
        .store
        .email
        .with_untracked(|e| e.ready().and_then(|d| d.get("policy").cloned()))
        .unwrap_or(Value::Null);
    open_form_guarded(ctx, cx, Size::new(80, 11), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let mode = mcx.signal(if s(&pol, "mode") == "denylist" {
            1usize
        } else {
            0
        });
        let entries0: Vec<String> = pol
            .get("entries")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let entries = mcx.signal(entries0.join(", "));
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        install_dirty_guard(
            mcx,
            &guard,
            vec![(entries, entries.get_untracked())],
            esc_armed,
            form_error,
        );
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());
        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let m = if mode.get_untracked() == 1 {
                "denylist"
            } else {
                "allowlist"
            };
            let mut args = vec![
                Arg::p("policy"),
                Arg::p("set"),
                Arg::p("--clear"),
                Arg::p(format!("--mode={m}")),
            ];
            for e in entries
                .get_untracked()
                .split(',')
                .map(str::trim)
                .filter(|e| !e.is_empty())
            {
                args.push(Arg::p(format!("--add={e}")));
            }
            in_flight.set(true);
            send_action(&ctx3, "email Save policy", args, Some(form_id));
        };
        Block::new()
            .title("Recipient policy")
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(0))
                    .child(field(
                        &t,
                        "Mode",
                        Select::new(vec![
                            SelectOption::new("allowlist: only these recipients"),
                            SelectOption::new("denylist: everyone except these"),
                        ])
                        .value(mode)
                        .view(mcx),
                    ))
                    .child(text_row(mcx, &t, "Entries", entries, "me@example.com, example.org", false))
                    .child(line(vec![span(
                        "  Exact addresses or domains (a subdomain only as its own entry); To, Cc and Bcc; any refused recipient refuses the message.",
                        t.text_faint,
                    )]))
                    .child(message_slot(theme, form_error, in_flight))
                    .child(
                        Element::new()
                            .style(LayoutStyle::row().gap(2).shrink(0.0))
                            .child(Button::new("Save policy").on_click(submit).view(mcx))
                            .child(Button::new("Cancel").on_click({
                                let close = close.clone();
                                move || close()
                            }).view(mcx))
                            .build(),
                    )
                    .build(),
            )
            .element(&t)
            .build()
    });
}

/// The Send limits form (`l`).
fn open_limits_form(cx: Scope, ctx: &Ctx) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let lim = ctx
        .store
        .email
        .with_untracked(|e| e.ready().and_then(|d| d.get("limits").cloned()))
        .unwrap_or(Value::Null);
    open_form_guarded(ctx, cx, Size::new(64, 9), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let n = |k: &str| {
            lim.get(k)
                .and_then(Value::as_i64)
                .map(|v| v.to_string())
                .unwrap_or_default()
        };
        let per_hour = mcx.signal(n("per_hour"));
        let per_day = mcx.signal(n("per_day"));
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        install_dirty_guard(
            mcx,
            &guard,
            vec![
                (per_hour, per_hour.get_untracked()),
                (per_day, per_day.get_untracked()),
            ],
            esc_armed,
            form_error,
        );
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());
        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let h = per_hour.get_untracked().trim().to_string();
            let d = per_day.get_untracked().trim().to_string();
            for (label, v) in [("Per hour", &h), ("Per day", &d)] {
                if v.parse::<u32>().is_err() {
                    form_error.set(Some(format!("{label} must be a whole number")));
                    return;
                }
            }
            in_flight.set(true);
            send_action(
                &ctx3,
                "email Save limits",
                vec![
                    Arg::p("limits"),
                    Arg::p("set"),
                    Arg::p(format!("--per-hour={h}")),
                    Arg::p(format!("--per-day={d}")),
                ],
                Some(form_id),
            );
        };
        Block::new()
            .title("Send limits")
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(0))
                    .child(text_row(mcx, &t, "Per hour", per_hour, "20", false))
                    .child(text_row(mcx, &t, "Per day", per_day, "100", false))
                    .child(message_slot(theme, form_error, in_flight))
                    .child(
                        Element::new()
                            .style(LayoutStyle::row().gap(2).shrink(0.0))
                            .child(Button::new("Save limits").on_click(submit).view(mcx))
                            .child(
                                Button::new("Cancel")
                                    .on_click({
                                        let close = close.clone();
                                        move || close()
                                    })
                                    .view(mcx),
                            )
                            .build(),
                    )
                    .build(),
            )
            .element(&t)
            .build()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::build_command;
    use crate::worker::email_argv;
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

    /// Save and test: the password is nowhere on the command line (nor in
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
    }

    #[test]
    fn connect_refuses_what_the_form_refuses() {
        let mut f = fields();
        f.password = "two\nlines".into();
        assert!(connect_action(&f, None).unwrap_err().contains("line break"));
        f.password.clear();
        assert!(connect_action(&f, None)
            .unwrap_err()
            .starts_with("Password is required"));
        let mut f = fields();
        f.imap_port = "0".into();
        assert_eq!(
            connect_action(&f, None).unwrap_err(),
            "IMAP port must be a number (1-65535)"
        );
        let mut f = fields();
        f.address.clear();
        assert_eq!(connect_action(&f, None).unwrap_err(), "Address is required");
    }

    /// OAuth2 sign-in: an own client secret goes on stdin with
    /// `--client-secret-stdin`; without one, stdin carries nothing.
    #[test]
    fn oauth_puts_the_client_secret_on_stdin_never_in_argv() {
        let f = OAuthFields {
            provider: 0,
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

        let public = oauth_action(
            &OAuthFields {
                client_secret: String::new(),
                ..f
            },
            None,
        )
        .unwrap();
        assert!(public.stdin_secret.is_none());
        assert!(!spawned_args(&public)
            .iter()
            .any(|a| a.contains("client-secret")));
    }
}
