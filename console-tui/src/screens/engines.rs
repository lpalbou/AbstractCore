//! The **Engines** screen (page id `engines`): which local engines exist
//! on the host (Ollama, LM Studio, MLX, llama.cpp, vLLM, Hugging Face),
//! whether they run, and a confirmed one-key install (`i`) or the
//! vendor's download page (`o`).

use abstracttui::prelude::*;

use super::data::{bytes_label, EngineRow};
use super::{confirm_install, job_strip, remote_line, Remote, ScreensCtx, ScreensStore};
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
        match store.engines.get() {
            Remote::Ready(d) if d.engines.is_empty() => line(vec![span(
                " ∅ the backend reported no engines",
                t.text_muted,
            )]),
            Remote::Ready(d) => engines_table(tcx, &t, &d.engines, store, w),
            _ => line(vec![span(String::new(), t.text)]),
        }
    });

    let detail = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        match selected_engine(&store, true) {
            Some(e) => engine_detail(&t, &e),
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
            move |_| s.cancel()
        })
        .child(host_line)
        .child(status_line)
        .child(table)
        .child(detail)
        .child(job_strip(sctx, theme))
        .build()
}

/// The footer hint pairs for this screen (hosts append them).
pub const HINTS: &[(&str, &str)] = &[
    ("i", "install"),
    ("o", "open download page"),
    ("r", "probe"),
    ("c", "cancel job"),
];

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
            let mut row = vec![e.name.clone(), e.install_label().to_string()];
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
        ColRule::head("status", 13),
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

fn engine_detail(t: &TokenSet, e: &EngineRow) -> View {
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

/// `i`: refuse with the reason, or confirm with the exact argv.
fn install_selected(cx: Scope, sctx: &ScreensCtx) {
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
    let os = store
        .host
        .with_untracked(|h| h.ready().and_then(|h| h.os.clone()));
    confirm_install(cx, sctx, &e, os.as_deref());
}
