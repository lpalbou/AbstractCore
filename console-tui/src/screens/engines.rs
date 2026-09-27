//! The **Engines** screen (page id `engines`): which local engines exist
//! on the host (Ollama, LM Studio, MLX, llama.cpp, vLLM, Hugging Face),
//! whether they run, and a confirmed one-key install (`i`; app engines
//! choose where) or the vendor's download page (`o`). Servers start and
//! stop with `s`. An install that PAUSES for an administrator or for
//! developer tools says exactly what to run (`y` copies it) and `a`
//! continues it; a paused install never blocks any other work.

use abstracttui::app::{ChoiceOutcome, ChoicePrompt};
use abstracttui::prelude::*;

use super::data::{bytes_label, EngineRow, JobView};
use super::{
    confirm_install, confirm_install_location, focus_holder, job_strip, remote_line, Access,
    Remote, ScreensCtx, ScreensStore,
};
use crate::transport::{ServerAction, TransportCaps};
use crate::ui::util::{line, span, span_bold};
use crate::ui::widths::{self, ColRule};

/// The Engines screen. Mount as a page; it loads its data on first entry.
///
/// ```no_run
/// # use abstracttui::prelude::*;
/// # use abstractcore_console::screens::{self, ScreensCtx};
/// # fn page(cx: Scope, sctx: &ScreensCtx) -> View {
/// screens::engines(cx, sctx)
/// # }
/// ```
pub fn engines(cx: Scope, sctx: &ScreensCtx) -> View {
    let theme = use_theme(cx);
    let store = sctx.store;
    {
        let s = sctx.clone();
        cx.effect(move || s.ensure_engines());
    }

    let host_label = sctx.host_label.clone();
    let host_line = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let summary = match store.host.get() {
            Remote::Ready(h) => h.summary(),
            Remote::Loading => "⟳ reading the host profile…".to_string(),
            Remote::Failed(e) => format!("host profile {}", e.headline()),
            Remote::NotAsked => String::new(),
        };
        line(vec![
            span_bold(format!(" {host_label} "), t.accent),
            span(summary, t.text_muted),
        ])
    });

    let status_line = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        match store.engines.get() {
            Remote::Ready(d) => {
                let installed = d
                    .engines
                    .iter()
                    .filter(|e| e.installed == Some(true))
                    .count();
                let running = d.engines.iter().filter(|e| e.running == Some(true)).count();
                line(vec![
                    span_bold(" engines", t.accent),
                    span(
                        format!(
                            " · {} known · {installed} installed · {running} running",
                            d.engines.len()
                        ),
                        t.text,
                    ),
                    span(" · r probes the local servers", t.text_faint),
                ])
            }
            other => remote_line(&t, "engines", &other).expect("not ready"),
        }
    });

    let table = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |tcx| {
        let t = theme.get().tokens;
        let w = abstracttui::app::use_viewport(tcx).get().w;
        // Placeholders hold focus (see `focus_holder`): `r` must retry
        // a failed read.
        match store.engines.get() {
            Remote::Ready(d) if d.engines.is_empty() => focus_holder(line(vec![span(
                " ∅ the backend reported no engines",
                t.text_muted,
            )])),
            Remote::Ready(d) => engines_table(tcx, &t, &d.engines, store, w),
            Remote::Failed(_) => {
                // Never a dead end: the desktop engines' own pages.
                focus_holder(line(vec![span(
                    " the backend could not list its engines — Ollama: https://ollama.com/download · LM Studio: https://lmstudio.ai/download",
                    t.text_muted,
                )]))
            }
            _ => focus_holder(line(vec![span(String::new(), t.text)])),
        }
    });

    let detail = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        let _ = store.jobs.get();
        match selected_engine(&store, true) {
            Some(e) => engine_detail(&t, &e, store.active_install(&e.id).as_ref()),
            None => Element::new().style(LayoutStyle::default().h(0)).build(),
        }
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .shortcut(KeyChord::plain(Key::Char('i')), {
            let s = sctx.clone();
            move |_| install_selected(cx, &s)
        })
        .shortcut(KeyChord::plain(Key::Char('o')), {
            let s = sctx.clone();
            move |_| match selected_engine(&s.store, false) {
                Some(e) => match e.open_url() {
                    Some(url) => s.open_url(url),
                    None => s
                        .store
                        .notice
                        .set(Some(format!("{} names no download page", e.name))),
                },
                None => s.store.notice.set(Some("no engine selected".into())),
            }
        })
        .shortcut(KeyChord::plain(Key::Char('r')), {
            let s = sctx.clone();
            move |_| {
                s.store
                    .notice
                    .set(Some("⟳ probing the engines on this host…".into()));
                s.refresh_engines();
            }
        })
        .shortcut(KeyChord::plain(Key::Char('c')), {
            let s = sctx.clone();
            move |_| match selected_engine(&s.store, false)
                .and_then(|e| s.store.active_install(&e.id))
            {
                Some(j) => s.cancel_job(&j),
                None => s.cancel(),
            }
        })
        .shortcut(KeyChord::plain(Key::Char('s')), {
            let s = sctx.clone();
            move |_| start_stop_selected(&s)
        })
        .shortcut(KeyChord::plain(Key::Char('a')), {
            let s = sctx.clone();
            move |_| continue_selected(cx, &s)
        })
        .shortcut(KeyChord::plain(Key::Char('y')), {
            let s = sctx.clone();
            move |_| copy_selected_command(&s)
        })
        .child(host_line)
        .child(status_line)
        .child(table)
        .child(detail)
        .child(job_strip(sctx, theme))
        .build()
}

/// The footer hint pairs for this screen with EVERY optional verb (a
/// full backend such as the gateway) and admin access. [`hints`] tailors
/// them.
pub const HINTS: &[(&str, &str)] = &[
    ("i", "install"),
    ("o", "open download page"),
    ("r", "probe"),
    ("c", "cancel install"),
    ("s", "start/stop"),
    ("a", "continue paused install"),
    ("y", "copy its command"),
];

/// The footer pairs for a transport and the person at the console: a
/// verb the backend lacks says so ("not here"), a verb only an admin may
/// run says so ("admin only") — never a key that silently does nothing.
pub fn hints(caps: TransportCaps, access: &Access) -> Vec<(&'static str, &'static str)> {
    let admin = access.is_admin();
    HINTS
        .iter()
        .filter_map(|&(k, label)| match k {
            "s" if !caps.engine_server => Some((k, "start/stop: not here")),
            // Without continuation no install ever pauses here, and
            // there is no command to copy either.
            "a" if !caps.engine_continue => Some((k, "continue: not here")),
            "y" if !caps.engine_continue => None,
            "i" if !admin => Some((k, "install: admin only")),
            "c" if !admin => Some((k, "cancel: admin only")),
            "s" if !admin => Some((k, "start/stop: admin only")),
            "a" if !admin => Some((k, "continue: admin only")),
            _ => Some((k, label)),
        })
        .collect()
}

/// The highlighted engine (`tracked` = reactive read).
pub fn selected_engine(store: &ScreensStore, tracked: bool) -> Option<EngineRow> {
    let i = if tracked {
        store.engine_sel.get()
    } else {
        store.engine_sel.get_untracked()
    };
    let pick = |r: &Remote<super::EnginesData>| r.ready().and_then(|d| d.engines.get(i).cloned());
    if tracked {
        store.engines.with(pick)
    } else {
        store.engines.with_untracked(pick)
    }
}

fn engines_table(cx: Scope, t: &TokenSet, data: &[EngineRow], store: ScreensStore, w: i32) -> View {
    let mut rows: Vec<Vec<String>> = data
        .iter()
        .map(|e| {
            let status = match store.active_install(&e.id) {
                Some(j) if j.state.as_deref() == Some("needs_admin") => "needs admin".to_string(),
                Some(j) if j.state.as_deref() == Some("needs_tools") => "needs tools".to_string(),
                Some(j) => match j.percent {
                    Some(p) => format!("installing {p:.0}%"),
                    None => "installing".to_string(),
                },
                None => e.install_label().to_string(),
            };
            let mut row = vec![e.name.clone(), status];
            row.push(e.version.clone().unwrap_or_else(|| "—".into()));
            row.push(e.running_label().to_string());
            row.push(
                e.models_count
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "—".into()),
            );
            if w >= 90 {
                row.push(e.kind.clone().unwrap_or_else(|| "—".into()));
            }
            row.push(install_cell(e));
            row
        })
        .collect();
    let mut rules = vec![
        ColRule::tail("engine", 10),
        ColRule::head("status", 15),
        ColRule::tail("version", 7),
        ColRule::head("server", 11),
        ColRule::head("models", 6),
    ];
    if w >= 90 {
        rules.push(ColRule::head("kind", 12));
    }
    rules.push(ColRule::tail("install", 12));
    let cols = widths::columns(&rules, &mut rows, w);
    Table::new(cols)
        .rows(rows)
        .selection(store.engine_sel)
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .autofocus()
        .build()
}

/// What `i` would do for this row, in a word or two.
fn install_cell(e: &EngineRow) -> String {
    if e.supported_on_host == Some(false) {
        return "—".into();
    }
    if e.installed == Some(true) {
        return "done".into();
    }
    if !e.install.available {
        return if e.open_url().is_some() {
            "o: page".into()
        } else {
            "—".into()
        };
    }
    match e.install.method.as_deref() {
        Some("download_page") | None => "o: page".into(),
        Some(m) => format!("i: {m}"),
    }
}

fn engine_detail(t: &TokenSet, e: &EngineRow, job: Option<&JobView>) -> View {
    let mut first = vec![span_bold(format!(" {} ", e.name), t.text)];
    if let Some(l) = &e.install_location {
        first.push(span(format!("at {l} "), t.text_muted));
    }
    if let Some(u) = &e.base_url {
        let reach = match e.reachable {
            Some(true) => t.ok,
            Some(false) => t.warn,
            None => t.text_muted,
        };
        first.push(span(format!("· {u} "), reach));
    }
    if let Some(v) = &e.version {
        first.push(span(format!("· v{v} "), t.text_faint));
    }
    if let Some(j) = job.filter(|j| j.is_paused()) {
        return paused_detail(t, first, j);
    }
    let second = if e.supported_on_host == Some(false) {
        vec![span(
            format!(
                "   not supported on this host: {}",
                e.unsupported_reason.as_deref().unwrap_or("no reason given")
            ),
            t.warn,
        )]
    } else if e.installed == Some(true) {
        vec![span("   installed", t.ok)]
    } else if e.install.available && !e.install.argv.is_empty() {
        let mut v = vec![
            span("   i runs: ", t.text_muted),
            span(e.install.argv.join(" "), t.text),
        ];
        if let Some(b) = e.install.estimated_bytes {
            v.push(span(format!("  (~{})", bytes_label(Some(b))), t.text_faint));
        }
        v
    } else if let Some(url) = e.open_url() {
        vec![
            span("   o opens ", t.text_muted),
            span(url.to_string(), t.text),
        ]
    } else {
        vec![span("   no install plan for this host", t.text_muted)]
    };
    Element::new()
        .style(LayoutStyle::column().shrink(0.0))
        .child(line(first))
        .child(line(second))
        .build()
}

/// A paused install: what it waits for, the EXACT command (its own
/// line, nothing else on it, so a terminal selection copies it clean),
/// where it runs, and the verbs.
fn paused_detail(t: &TokenSet, first: Vec<crate::ui::util::SpanSpec>, j: &JobView) -> View {
    let mut col = Element::new()
        .style(LayoutStyle::column().shrink(0.0))
        .child(line(first))
        .child(line(vec![span(
            format!(
                "   ⏸ {}",
                j.message
                    .as_deref()
                    .or(j.paused_label())
                    .unwrap_or("waiting for a person")
            ),
            t.warn,
        )]));
    let (lead, cmd, where_) = match j.state.as_deref() {
        Some("needs_admin") => {
            let p = j.admin_prompt.clone().unwrap_or_default();
            let lead = if p.method.as_deref() == Some("manual") {
                "   run this yourself, then press a (re-check):"
            } else {
                "   it will run, as administrator:"
            };
            (lead, p.command, p.where_)
        }
        _ => {
            let p = j.tools_prompt.clone().unwrap_or_default();
            let lead = if p.started {
                "   the tools installer is open on the host; press a when it finishes:"
            } else {
                "   the missing tools install with:"
            };
            (lead, p.command, None)
        }
    };
    col = col.child(line(vec![span(lead, t.text_muted)]));
    if let Some(c) = cmd {
        col = col.child(line(vec![span(c, t.text)]));
    }
    let mut verbs = Vec::new();
    if let Some(w) = where_ {
        verbs.push(format!("where: {w}"));
    }
    let actions: Vec<String> = j
        .continue_actions
        .iter()
        .map(|a| continue_label(j, a))
        .collect();
    if !actions.is_empty() {
        verbs.push(format!("a: {}", actions.join(" / ")));
    }
    if j.copyable_command().is_some() {
        verbs.push("y copies the command".into());
    }
    verbs.push("c cancels".into());
    col.child(line(vec![span(
        format!("   {}", verbs.join(" · ")),
        t.text_faint,
    )]))
    .build()
}

/// The words for one `continue_actions` entry (the backend's button
/// label when it sent one).
fn continue_label(j: &JobView, action: &str) -> String {
    match action {
        "approve_admin" => j
            .admin_prompt
            .as_ref()
            .and_then(|p| p.button.clone())
            .unwrap_or_else(|| "Continue with administrator password".into()),
        "install_tools" => j
            .tools_prompt
            .as_ref()
            .and_then(|p| p.button.clone())
            .unwrap_or_else(|| "Install tools".into()),
        "recheck" => "Re-check".into(),
        other => other.to_string(),
    }
}

/// `i`: refuse with the reason, or confirm with the exact argv.
fn install_selected(cx: Scope, sctx: &ScreensCtx) {
    if !sctx.require_admin("install engines") {
        return;
    }
    let store = sctx.store;
    let Some(e) = selected_engine(&store, false) else {
        store.notice.set(Some("no engine selected".into()));
        return;
    };
    let refusal = if e.supported_on_host == Some(false) {
        Some(format!(
            "{} is not supported on this host: {}",
            e.name,
            e.unsupported_reason.as_deref().unwrap_or("no reason given")
        ))
    } else if e.installed == Some(true) {
        Some(format!(
            "{} is already installed{}",
            e.name,
            e.version
                .as_deref()
                .map(|v| format!(" (v{v})"))
                .unwrap_or_default()
        ))
    } else if let Some(j) = store.active_install(&e.id) {
        Some(if j.is_paused() {
            format!(
                "the {} install is waiting ({}) — a continues it, c cancels it",
                e.name,
                j.paused_label().unwrap_or("paused")
            )
        } else {
            format!("the {} install is still running — c cancels it", e.name)
        })
    } else if let Some(a) = e.action("install").filter(|a| !a.enabled) {
        Some(format!(
            "{}: {}",
            e.name,
            a.reason
                .as_deref()
                .unwrap_or("installing is not available right now")
        ))
    } else if !e.install.available
        || e.install.argv.is_empty()
        || e.install.method.as_deref() == Some("download_page")
    {
        Some(match e.open_url() {
            Some(url) => format!("{} installs from its download page — o opens {url}", e.name),
            None => format!("{} has no install plan for this host", e.name),
        })
    } else {
        None
    };
    if let Some(msg) = refusal {
        store.notice.set(Some(msg));
        return;
    }
    if e.is_app_install() && sctx.caps.install_location {
        confirm_install_location(cx, sctx, &e);
        return;
    }
    let os = store
        .host
        .with_untracked(|h| h.ready().and_then(|h| h.os.clone()));
    confirm_install(cx, sctx, &e, os.as_deref());
}

/// `s`: the row's own start/stop action (gateway v2 `actions`, which
/// also carry the admin guard), refused with the reason otherwise.
fn start_stop_selected(sctx: &ScreensCtx) {
    let store = sctx.store;
    let Some(e) = selected_engine(&store, false) else {
        store.notice.set(Some("no engine selected".into()));
        return;
    };
    if !sctx.caps.engine_server {
        store.notice.set(Some(format!(
            "starting or stopping engine servers is not available over {}",
            sctx.host_label
        )));
        return;
    }
    let pick = e
        .action("stop")
        .map(|a| (a, ServerAction::Stop))
        .or_else(|| e.action("start").map(|a| (a, ServerAction::Start)));
    match pick {
        Some((a, _)) if !a.enabled => store.notice.set(Some(format!(
            "{}: {}",
            e.name,
            a.reason.as_deref().unwrap_or("not available right now")
        ))),
        Some((_, act)) => sctx.server(&e.id, act),
        None => {
            let why = if e.supported_on_host == Some(false) {
                format!("{} does not run on this host", e.name)
            } else if e.installed != Some(true) {
                format!("{} is not installed", e.name)
            } else if e.running.is_none() && e.base_url.is_some() {
                format!(
                    "whether {} runs is not known yet — r probes it first",
                    e.name
                )
            } else {
                format!(
                    "{} is not a server: it runs inside the host when a model uses it",
                    e.name
                )
            };
            store.notice.set(Some(why));
        }
    }
}

/// `a`: continue the selected engine's paused install. One offered
/// action runs at once (the web's single button); several ask which.
fn continue_selected(cx: Scope, sctx: &ScreensCtx) {
    if !sctx.caps.engine_continue {
        sctx.store.notice.set(Some(format!(
            "continuing a paused install is not available over {}",
            sctx.host_label
        )));
        return;
    }
    if !sctx.require_admin("continue an install") {
        return;
    }
    let store = sctx.store;
    let Some(e) = selected_engine(&store, false) else {
        store.notice.set(Some("no engine selected".into()));
        return;
    };
    let Some(j) = store.active_install(&e.id).filter(|j| j.is_paused()) else {
        store.notice.set(Some(format!(
            "no {} install is waiting for anything",
            e.name
        )));
        return;
    };
    match j.continue_actions.as_slice() {
        [] => sctx.continue_job(&j, None),
        [one] => sctx.continue_job(&j, Some(one)),
        many => {
            let mut prompt = ChoicePrompt::new(format!("Continue the {} install?", e.name));
            for a in many {
                prompt = prompt.option(a.as_str(), continue_label(&j, a));
            }
            let s = sctx.clone();
            let job = j.clone();
            prompt
                .option("not_now", "Not now")
                .initial("not_now")
                .on_resolve(move |outcome| {
                    if let ChoiceOutcome::Answered(ans) = outcome {
                        match ans.selected.first().map(String::as_str) {
                            Some("not_now") | None => {}
                            Some(a) => s.continue_job(&job, Some(a)),
                        }
                    }
                })
                .open(cx);
        }
    }
}

/// `y`: copy the paused install's command (OSC 52).
fn copy_selected_command(sctx: &ScreensCtx) {
    let store = sctx.store;
    let cmd = selected_engine(&store, false)
        .and_then(|e| store.active_install(&e.id))
        .and_then(|j| j.copyable_command().map(str::to_string));
    match cmd {
        Some(c) => {
            abstracttui::prelude::copy_to_clipboard(c.clone());
            store.notice.set(Some(format!("copied: {c}")));
        }
        None => store
            .notice
            .set(Some("no paused install with a command to copy".into())),
    }
}
