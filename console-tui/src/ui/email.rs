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
//! Verbs: `c` connect (Account form; Save and test), `t` Test, `o` Turn
//! off / Turn on, `x` Disconnect (confirm), `p` Recipient policy, `l`
//! Send limits. Every value is passed as `--flag=value`, so a password
//! or entry starting with `-` cannot be read as a flag.

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Button};
use serde_json::Value;

use crate::store::Loadable;
use crate::worker::{next_form_id, Cmd, EmailAction};
use crate::writes::Arg;

use super::forms::{
    confirm_danger, install_dirty_guard, install_write_done, message_slot, open_form_guarded,
};
use super::util::{error_panel, field, line, span, span_bold};
use super::Ctx;

/// The footer's verbs for this screen (same words as the web console).
pub const HINTS: &[(&str, &str)] = &[
    ("c", "connect"),
    ("t", "test"),
    ("o", "turn on/off"),
    ("x", "disconnect"),
    ("p", "recipient policy"),
    ("l", "send limits"),
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
    })));
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
    let (c_connect, c_test, c_toggle, c_disc, c_pol, c_lim) = (
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
            Loadable::Ready(doc) => render(&t, &doc),
        }
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .focusable()
        .autofocus()
        .shortcut(KeyChord::plain(Key::Char('c')), move |_| {
            open_connect_form(cx, &c_connect)
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
        .child(line(vec![span(
            " The mailbox is only read (never marked read, moved or deleted). CLI: abstractcore email status",
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
            let addr = v(address);
            if addr.is_empty() {
                form_error.set(Some("Address is required".into()));
                return;
            }
            let pw = password.get_untracked();
            if pw.is_empty() {
                form_error.set(Some(
                    "Password is required (it is stored encrypted and never shown again)".into(),
                ));
                return;
            }
            if v(imap_host).is_empty() && v(smtp_host).is_empty() {
                form_error.set(Some(
                    "give an IMAP host (read) and/or an SMTP host (send)".into(),
                ));
                return;
            }
            for (label, p) in [("IMAP port", v(imap_port)), ("SMTP port", v(smtp_port))] {
                if !p.is_empty() && p.parse::<u16>().map(|n| n == 0).unwrap_or(true) {
                    form_error.set(Some(format!("{label} must be a number (1-65535)")));
                    return;
                }
            }
            let mut args = vec![
                Arg::p("connect"),
                Arg::p(format!("--address={addr}")),
                Arg::Secret(format!("--password={pw}")),
            ];
            let mut opt = |flag: &str, value: String| {
                if !value.is_empty() {
                    args.push(Arg::p(format!("--{flag}={value}")));
                }
            };
            opt("display-name", v(display_name));
            opt("username", v(username));
            if !v(imap_host).is_empty() {
                opt("imap-host", v(imap_host));
                opt("imap-port", v(imap_port));
                opt(
                    "imap-security",
                    SECURITY[imap_sec.get_untracked().min(1)].to_string(),
                );
                opt("imap-folder", v(folder));
            }
            if !v(smtp_host).is_empty() {
                opt("smtp-host", v(smtp_host));
                opt("smtp-port", v(smtp_port));
                opt(
                    "smtp-security",
                    SECURITY[smtp_sec.get_untracked().min(1)].to_string(),
                );
            }
            opt("ca-file", v(ca_file));
            opt("registered-address", v(registered));
            in_flight.set(true);
            send_action(&ctx3, "email Save and test", args, Some(form_id));
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
