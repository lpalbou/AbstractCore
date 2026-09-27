//! The **Models** screen (page id `catalog`): browse what can be
//! downloaded, see whether it fits this host, download (`w`, several at
//! once), delete (`d`), filter (`/`, `f`, `e`), search Hugging Face
//! (`h`), make an installed model the default text model (`u`), and
//! cycle catalog → installed → downloads (`v`) — the downloads view is
//! the live feed, where `c` cancels the selected download.

use std::rc::Rc;

use abstracttui::app::{ChoiceOption, ChoiceOutcome, ChoicePrompt, Modal};
use abstracttui::prelude::*;

use super::data::{
    bytes_label, fit_label, params_label, served_model_id, weights_label, ArtifactRow,
    InstalledRow, JobView,
};
use super::{
    confirm_cancel_download, confirm_delete, job_strip, open_filter, remote_line, CatalogView,
    Remote, ScreensCtx, ScreensStore,
};
use crate::transport::TransportCaps;
use crate::ui::util::{line, span, span_bold};
use crate::ui::widths::{self, ColRule};

/// The Models screen. Mount as a page; it loads its data on first entry.
///
/// ```no_run
/// # use abstracttui::prelude::*;
/// # use abstractcore_console::screens::{self, ScreensCtx};
/// # fn page(cx: Scope, sctx: &ScreensCtx) -> View {
/// screens::catalog(cx, sctx)
/// # }
/// ```
pub fn catalog(cx: Scope, sctx: &ScreensCtx) -> View {
    let theme = use_theme(cx);
    let store = sctx.store;
    {
        // Untracked reads only: runs once per mount — entering the
        // screen asks for what was never asked, nothing more.
        let s = sctx.clone();
        cx.effect(move || s.ensure_catalog());
    }

    let host_line = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        match store.host.get() {
            Remote::Ready(h) => line(vec![
                span_bold(" host ", t.accent),
                span(h.summary(), t.text),
            ]),
            other => remote_line(&t, "host profile", &other)
                .unwrap_or_else(|| line(vec![span(String::new(), t.text)])),
        }
    });

    let status_line = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let view = store.view.get();
        let q = store.query.get();
        let fits = store.fits_only.get();
        let engine = store.engine_filter.get();
        let count = match view {
            CatalogView::Catalog => store
                .catalog
                .with(|c| c.ready().map(|d| format!("{} artifacts", d.rows.len()))),
            CatalogView::Installed => store.installed.with(|i| {
                i.ready()
                    .map(|d| format!("{} installed", installed_rows(&store, &d.rows).len()))
            }),
            CatalogView::Downloads => {
                let jobs = store.download_jobs(true);
                let live = jobs.iter().filter(|j| j.is_active()).count();
                Some(format!("{} downloads · {live} running", jobs.len()))
            }
        };
        let hub = store.hub.get();
        let mut spans = vec![span_bold(
            match view {
                CatalogView::Catalog if hub.is_some() => " Hugging Face",
                CatalogView::Catalog => " catalog",
                CatalogView::Installed => " installed",
                CatalogView::Downloads => " downloads",
            },
            t.accent,
        )];
        if let Some(c) = count {
            spans.push(span(format!(" · {c}"), t.text));
        }
        if view == CatalogView::Downloads {
            if let Remote::Failed(e) = store.feed.get() {
                spans.push(span(
                    format!(" · feed {}: {}", e.headline(), e.message),
                    t.warn,
                ));
            }
            spans.push(span(" · c cancels the selected one · v back to the catalog", t.text_faint));
            return line(spans);
        }
        if let (CatalogView::Catalog, Some(h)) = (view, hub.as_ref()) {
            spans.push(span(format!(" · \"{h}\" · h changes it, empty leaves"), t.text_muted));
            if let Some(d) = store.catalog.with(|c| c.ready().cloned()) {
                if d.hub_ok == Some(false) {
                    let why = if d.hub_errors.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", d.hub_errors.join("; "))
                    };
                    spans.push(span(
                        format!(" · Hugging Face answered in part — results may be incomplete{why}"),
                        t.warn,
                    ));
                }
            }
            return line(spans);
        }
        if let Some((p, m)) = store.text_default.with(|d| d.ready().cloned().flatten()) {
            spans.push(span(format!(" · default text {p} · {m}"), t.text_faint));
        }
        spans.push(span(
            if q.is_empty() {
                " · no filter".to_string()
            } else {
                format!(" · filter \"{q}\"")
            },
            t.text_muted,
        ));
        spans.push(span(
            format!(" · engine {}", engine.as_deref().unwrap_or("all")),
            t.text_muted,
        ));
        if view == CatalogView::Catalog {
            spans.push(span(
                if fits { " · fits only" } else { " · any fit" },
                if fits { t.ok } else { t.text_muted },
            ));
        }
        if let Some(errors) = store.installed.with(|i| {
            i.ready().filter(|d| !d.errors.is_empty()).map(|d| {
                d.errors
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
        }) {
            spans.push(span(format!(" · not read: {errors}"), t.warn));
        }
        line(spans)
    });

    let table = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |tcx| {
        let t = theme.get().tokens;
        let w = abstracttui::app::use_viewport(tcx).get().w;
        // The installed list filters locally: track the filters here.
        let _ = (store.query.get(), store.engine_filter.get());
        match store.view.get() {
            CatalogView::Catalog => match store.catalog.get() {
                Remote::Ready(d) if d.rows.is_empty() => line(vec![span(
                    " ∅ nothing matches — / changes the filter, f toggles fits-only, e the engine",
                    t.text_muted,
                )]),
                Remote::Ready(d) => catalog_table(tcx, &t, &d.rows, store, w),
                other => remote_line(
                    &t,
                    if store.hub.get_untracked().is_some() { "Hugging Face" } else { "catalog" },
                    &other,
                )
                .expect("not ready"),
            },
            CatalogView::Downloads => {
                let jobs = store.download_jobs(true);
                if jobs.is_empty() {
                    match store.feed.get() {
                        Remote::Loading => line(vec![span(" ⟳ reading the downloads…", t.info)]),
                        _ => line(vec![span(
                            " ∅ no downloads yet — v shows the catalog, w downloads the selected model",
                            t.text_muted,
                        )]),
                    }
                } else {
                    downloads_table(tcx, &t, &jobs, store, w)
                }
            }
            CatalogView::Installed => match store.installed.get() {
                Remote::Ready(d) => {
                    let rows = installed_rows(&store, &d.rows);
                    if rows.is_empty() {
                        line(vec![span(
                            " ∅ no models on disk match — v shows the catalog",
                            t.text_muted,
                        )])
                    } else {
                        installed_table(tcx, &t, &rows, store, w)
                    }
                }
                other => remote_line(&t, "installed models", &other).expect("not ready"),
            },
        }
    });

    let detail = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        let _ = (store.query.get(), store.engine_filter.get());
        match store.view.get() {
            CatalogView::Catalog => match selected_artifact(&store, true) {
                Some(r) => artifact_detail(&t, &r, &store),
                None => line(vec![span(String::new(), t.text)]),
            },
            CatalogView::Installed => match selected_installed(&store, true) {
                Some(r) => installed_detail(&t, &r),
                None => line(vec![span(String::new(), t.text)]),
            },
            CatalogView::Downloads => match selected_download(&store, true) {
                Some(j) => download_detail(&t, &j),
                None => line(vec![span(String::new(), t.text)]),
            },
        }
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .shortcut(KeyChord::plain(Key::Char('w')), {
            let s = sctx.clone();
            move |_| download_selected(cx, &s)
        })
        .shortcut(KeyChord::plain(Key::Char('d')), {
            let s = sctx.clone();
            move |_| delete_selected(cx, &s)
        })
        .shortcut(KeyChord::plain(Key::Char('/')), {
            let s = sctx.clone();
            move |_| open_filter(cx, &s)
        })
        .shortcut(KeyChord::plain(Key::Char('f')), {
            let s = sctx.clone();
            move |_| {
                let on = !s.store.fits_only.get_untracked();
                s.store.fits_only.set(on);
                s.store.catalog_sel.set(0);
                s.reload_catalog();
                s.store.notice.set(Some(
                    if on {
                        "showing only what fits this host"
                    } else {
                        "showing every fit verdict"
                    }
                    .into(),
                ));
            }
        })
        .shortcut(KeyChord::plain(Key::Char('e')), {
            let s = sctx.clone();
            move |_| cycle_engine(&s)
        })
        .shortcut(KeyChord::plain(Key::Char('v')), {
            let s = sctx.clone();
            move |_| {
                let next = match s.store.view.get_untracked() {
                    CatalogView::Catalog => CatalogView::Installed,
                    CatalogView::Installed => CatalogView::Downloads,
                    CatalogView::Downloads => CatalogView::Catalog,
                };
                if next == CatalogView::Downloads {
                    // The feed may hold downloads started elsewhere.
                    s.load_downloads();
                }
                s.store.view.set(next);
            }
        })
        .shortcut(KeyChord::plain(Key::Char('r')), {
            let s = sctx.clone();
            move |_| {
                s.store
                    .notice
                    .set(Some("⟳ re-reading the catalog and what is on disk…".into()));
                s.refresh_catalog();
            }
        })
        .shortcut(KeyChord::plain(Key::Char('c')), {
            let s = sctx.clone();
            move |_| cancel_selected(cx, &s)
        })
        .shortcut(KeyChord::plain(Key::Char('h')), {
            let s = sctx.clone();
            move |_| open_hub_search(cx, &s)
        })
        .shortcut(KeyChord::plain(Key::Char('u')), {
            let s = sctx.clone();
            move |_| use_as_default(&s)
        })
        .child(host_line)
        .child(status_line)
        .child(table)
        .child(detail)
        .child(job_strip(sctx, theme))
        .build()
}

/// The footer hint pairs for this screen with EVERY optional verb (a
/// full backend such as the gateway). [`hints`] tailors them.
pub const HINTS: &[(&str, &str)] = &[
    ("w", "download"),
    ("d", "delete"),
    ("h", "Hugging Face"),
    ("u", "use as default"),
    ("/", "filter"),
    ("f", "fits only"),
    ("e", "engine"),
    ("v", "installed/downloads"),
    ("c", "cancel download"),
];

/// The footer pairs for a transport: a verb the backend lacks says so
/// ("not here") instead of silently doing nothing.
pub fn hints(caps: TransportCaps) -> Vec<(&'static str, &'static str)> {
    HINTS
        .iter()
        .map(|&(k, label)| match k {
            "h" if !caps.hub_search => (k, "Hugging Face: not here"),
            "u" if !caps.text_default => (k, "default: not here"),
            _ => (k, label),
        })
        .collect()
}

/// Installed rows after the screen's filters (query + engine; the
/// installed read has no server-side text filter).
pub fn installed_rows(store: &ScreensStore, rows: &[InstalledRow]) -> Vec<InstalledRow> {
    let q = store.query.get_untracked().to_lowercase();
    let engine = store.engine_filter.get_untracked();
    rows.iter()
        .filter(|r| engine.as_deref().is_none_or(|e| r.provider == e))
        .filter(|r| {
            q.is_empty()
                || r.artifact.to_lowercase().contains(&q)
                || r.provider.to_lowercase().contains(&q)
                || r.catalog_id
                    .as_deref()
                    .is_some_and(|c| c.to_lowercase().contains(&q))
        })
        .cloned()
        .collect()
}

/// The highlighted catalog artifact (`tracked` = reactive read).
pub fn selected_artifact(store: &ScreensStore, tracked: bool) -> Option<ArtifactRow> {
    let i = if tracked {
        store.catalog_sel.get()
    } else {
        store.catalog_sel.get_untracked()
    };
    let pick = |c: &Remote<super::CatalogData>| c.ready().and_then(|d| d.rows.get(i).cloned());
    if tracked {
        store.catalog.with(pick)
    } else {
        store.catalog.with_untracked(pick)
    }
}

/// The highlighted download (Downloads view).
pub fn selected_download(store: &ScreensStore, tracked: bool) -> Option<JobView> {
    let i = if tracked {
        store.downloads_sel.get()
    } else {
        store.downloads_sel.get_untracked()
    };
    store.download_jobs(tracked).get(i).cloned()
}

/// The highlighted installed row (after filters).
pub fn selected_installed(store: &ScreensStore, tracked: bool) -> Option<InstalledRow> {
    let i = if tracked {
        store.installed_sel.get()
    } else {
        store.installed_sel.get_untracked()
    };
    let pick = |r: &Remote<super::InstalledData>| {
        r.ready()
            .and_then(|d| installed_rows(store, &d.rows).get(i).cloned())
    };
    if tracked {
        store.installed.with(pick)
    } else {
        store.installed.with_untracked(pick)
    }
}

fn catalog_table(
    cx: Scope,
    t: &TokenSet,
    data: &[ArtifactRow],
    store: ScreensStore,
    w: i32,
) -> View {
    let mut rows: Vec<Vec<String>> = data
        .iter()
        .map(|r| {
            let star = if r.recommended { "★ " } else { "" };
            let mut row = vec![
                format!("{star}{}", r.model_name),
                r.provider.clone(),
                r.artifact.clone(),
            ];
            if w >= 100 {
                row.push(r.quant.clone().unwrap_or_else(|| "—".into()));
            }
            row.push(bytes_label(r.download_bytes));
            row.push(if r.supported_on_host == Some(false) {
                "not for this host".to_string()
            } else {
                fit_label(&r.fit).to_string()
            });
            row.push(match store.active_download(&r.provider, &r.artifact) {
                Some(j) => match j.percent {
                    Some(p) => format!("⟳ {p:.0}%"),
                    None => "⟳ downloading".to_string(),
                },
                None => weights_label(&r.presence).to_string(),
            });
            row
        })
        .collect();
    let mut rules = vec![
        ColRule::tail("model", 12),
        ColRule::head("engine", 8),
        ColRule::tail("artifact", 14),
    ];
    if w >= 100 {
        rules.push(ColRule::head("quant", 6));
    }
    // Closed vocabularies keep their widest word as the floor.
    rules.push(ColRule::head("size", 9));
    rules.push(ColRule::head("fit", 17));
    rules.push(ColRule::head("weights", 14));
    let cols = widths::columns(&rules, &mut rows, w);
    Table::new(cols)
        .rows(rows)
        .selection(store.catalog_sel)
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .autofocus()
        .build()
}

fn installed_table(
    cx: Scope,
    t: &TokenSet,
    data: &[InstalledRow],
    store: ScreensStore,
    w: i32,
) -> View {
    let mut rows: Vec<Vec<String>> = data
        .iter()
        .map(|r| {
            let mut row = vec![r.provider.clone(), r.artifact.clone()];
            if w >= 100 {
                row.push(r.quant.clone().unwrap_or_else(|| "—".into()));
            }
            row.push(bytes_label(r.size_bytes));
            row.push(match r.loaded {
                Some(true) => "loaded".into(),
                Some(false) => "idle".into(),
                None => "—".into(),
            });
            row.push(if r.delete_blockers.is_empty() {
                if r.deletable {
                    "deletable".into()
                } else {
                    "not deletable".into()
                }
            } else {
                r.delete_blockers.join(", ")
            });
            row
        })
        .collect();
    let mut rules = vec![ColRule::head("engine", 8), ColRule::tail("artifact", 18)];
    if w >= 100 {
        rules.push(ColRule::head("quant", 6));
    }
    rules.push(ColRule::head("size", 9));
    rules.push(ColRule::head("state", 6));
    rules.push(ColRule::tail("delete", 13));
    let cols = widths::columns(&rules, &mut rows, w);
    Table::new(cols)
        .rows(rows)
        .selection(store.installed_sel)
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .autofocus()
        .build()
}

fn artifact_detail(t: &TokenSet, r: &ArtifactRow, store: &ScreensStore) -> View {
    let fit_ink = match r.fit.as_str() {
        "fits" => t.ok,
        "tight" | "partial_offload" => t.warn,
        "too_large" => t.error,
        _ => t.text_muted,
    };
    let mut spans = vec![
        span_bold(format!(" {} ", r.model_name), t.text),
        span(
            format!("{} params · ", params_label(r.params_total)),
            t.text_muted,
        ),
    ];
    if r.supported_on_host == Some(false) {
        spans.push(span("its engine does not run on this host", t.warn));
    } else {
        spans.push(span(format!("fit {}", fit_label(&r.fit)), fit_ink));
    }
    // The web console's fit explanation: the need against what this host
    // can give a model (the number the verdict compares with).
    match (r.need_bytes, r.usable_bytes) {
        (Some(need), Some(usable)) => spans.push(span(
            format!(
                " (needs about {} of the {} this host can give a model)",
                bytes_label(Some(need)),
                bytes_label(Some(usable))
            ),
            t.text_muted,
        )),
        (Some(need), None) => spans.push(span(
            format!(" (needs {})", bytes_label(Some(need))),
            t.text_muted,
        )),
        _ => {}
    }
    if let Some(fr) = r.free_now_bytes {
        spans.push(span(format!(" · {} free now", bytes_label(Some(fr))), t.text_faint));
    }
    if r.disk_ok == Some(false) {
        spans.push(span(" · not enough disk", t.error));
    }
    if let Some(ctx) = r.max_context {
        spans.push(span(format!(" · ctx {ctx}"), t.text_muted));
    }
    let mut second = Vec::new();
    if r.starter {
        second.push(span(" starter set", t.accent));
    }
    if r.source.as_deref() == Some("hf_search") {
        second.push(span(" Hugging Face", t.accent));
    }
    if let Some(n) = r.fit_notes.first() {
        second.push(span(format!(" {n}"), t.text_faint));
    }
    let is_default = store.text_default.with(|d| {
        d.ready().cloned().flatten().is_some_and(|(p, m)| {
            p == r.provider && m == served_model_id(&r.provider, &r.artifact)
        })
    });
    match r.presence.as_str() {
        "installed" if is_default => second.push(span(" · default text model", t.ok)),
        "installed" if r.can_be_text_default() => {
            second.push(span(" · d deletes · u makes it the default text model", t.text_faint))
        }
        "installed" => second.push(span(" · d deletes", t.text_faint)),
        "absent" if r.downloadable => second.push(span(" · w downloads", t.text_faint)),
        _ => {}
    }
    Element::new()
        .style(LayoutStyle::column().shrink(0.0))
        .child(line(spans))
        .child(line(second))
        .build()
}

fn downloads_table(cx: Scope, t: &TokenSet, data: &[JobView], store: ScreensStore, w: i32) -> View {
    let mut rows: Vec<Vec<String>> = data
        .iter()
        .map(|j| {
            let glyph = match j.status.as_str() {
                "completed" => "✓",
                "failed" => "✗",
                "cancelled" => "⊘",
                _ => "⟳",
            };
            let state = j.state.clone().unwrap_or_else(|| j.status.clone());
            let mut row = vec![
                format!("{glyph} {state}"),
                j.provider.clone().unwrap_or_else(|| "—".into()),
                match (&j.artifact, j.kind.as_str()) {
                    (Some(a), _) => a.clone(),
                    (None, "download_group") => "download all".into(),
                    _ => j.job_id.clone(),
                },
                j.percent.map(|p| format!("{p:.0}%")).unwrap_or_else(|| "—".into()),
                match (j.downloaded_bytes, j.total_bytes) {
                    (Some(d), Some(tot)) => format!("{} / {}", bytes_label(Some(d)), bytes_label(Some(tot))),
                    (Some(d), None) => bytes_label(Some(d)),
                    _ => "—".into(),
                },
            ];
            if w >= 100 {
                row.push(match (j.bytes_per_second, j.eta_s) {
                    (Some(b), Some(e)) if b > 0.0 && j.is_active() => {
                        format!("{}/s · {e}s left", bytes_label(Some(b as u64)))
                    }
                    (Some(b), _) if b > 0.0 && j.is_active() => format!("{}/s", bytes_label(Some(b as u64))),
                    _ => "—".into(),
                });
            }
            row
        })
        .collect();
    let mut rules = vec![
        ColRule::head("state", 12),
        ColRule::head("engine", 8),
        ColRule::tail("artifact", 18),
        ColRule::head("done", 5),
        ColRule::head("bytes", 12),
    ];
    if w >= 100 {
        rules.push(ColRule::head("speed", 10));
    }
    let cols = widths::columns(&rules, &mut rows, w);
    Table::new(cols)
        .rows(rows)
        .selection(store.downloads_sel)
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .autofocus()
        .build()
}

fn download_detail(t: &TokenSet, j: &JobView) -> View {
    let (text, ink) = if let Some(e) = j.error.as_ref().filter(|_| j.status == "failed") {
        (format!(" ✗ {e}"), t.error)
    } else if let Some(r) = j.ended_reason.as_ref().filter(|_| !j.is_active()) {
        (format!(" {r}"), t.text_muted)
    } else if let Some(m) = &j.message {
        (format!(" {m}"), t.text_muted)
    } else {
        (String::new(), t.text)
    };
    let mut spans = vec![span(text, ink)];
    if let Some(p) = &j.parent_job {
        spans.push(span(format!(" · part of {p}"), t.text_faint));
    }
    if j.is_active() {
        spans.push(span(" · c cancels (asks first)", t.text_faint));
    }
    line(spans)
}

fn installed_detail(t: &TokenSet, r: &InstalledRow) -> View {
    let mut spans = vec![span_bold(
        format!(" {} {} ", r.provider, r.artifact),
        t.text,
    )];
    if let Some(l) = &r.location {
        spans.push(span(format!("at {l} "), t.text_muted));
    }
    if let Some(c) = &r.catalog_id {
        spans.push(span(format!("· catalog {c} "), t.text_faint));
    }
    line(spans)
}

/// `w`.
fn download_selected(cx: Scope, sctx: &ScreensCtx) {
    let store = sctx.store;
    match store.view.get_untracked() {
        CatalogView::Installed => {
            store.notice.set(Some(
                "this list is what is already on disk — v shows the catalog to download from".into(),
            ));
            return;
        }
        CatalogView::Downloads => {
            store.notice.set(Some(
                "this list is the downloads — v back to the catalog to start one".into(),
            ));
            return;
        }
        CatalogView::Catalog => {}
    }
    let Some(r) = selected_artifact(&store, false) else {
        store
            .notice
            .set(Some("no model selected — nothing to download".into()));
        return;
    };
    let why_not = match r.presence.as_str() {
        "installed" => Some(format!("{} is already on disk", r.artifact)),
        "not_applicable" => Some(format!(
            "{} runs remotely — there are no weights to download",
            r.artifact
        )),
        _ if !r.downloadable => Some(format!(
            "{} has no download verb for {} — install it with the engine's own tool",
            r.artifact, r.provider
        )),
        _ => None,
    };
    if let Some(msg) = why_not {
        store.notice.set(Some(msg));
        return;
    }
    let oversized = r.fit == "too_large" || r.disk_ok == Some(false);
    if !oversized {
        sctx.download(&r.provider, &r.artifact);
        return;
    }
    let reason = if r.disk_ok == Some(false) {
        "there is not enough free disk for it".to_string()
    } else {
        format!(
            "it is too large for this host (needs {})",
            bytes_label(r.need_bytes)
        )
    };
    let s = sctx.clone();
    let (p, a) = (r.provider.clone(), r.artifact.clone());
    ChoicePrompt::new(format!("Download {}? {reason}.", r.artifact))
        .option_with(ChoiceOption::new("go", "Download anyway").danger(true))
        .option("keep", "Don't download")
        .initial("keep")
        .on_resolve(move |outcome| {
            if let ChoiceOutcome::Answered(ans) = outcome {
                if ans.selected.iter().any(|x| x == "go") {
                    s.download(&p, &a);
                }
            }
        })
        .open(cx);
}

/// `d`.
fn delete_selected(cx: Scope, sctx: &ScreensCtx) {
    let store = sctx.store;
    let target = match store.view.get_untracked() {
        CatalogView::Downloads => {
            store.notice.set(Some(
                "this list is the downloads — v shows what is installed, to delete from".into(),
            ));
            return;
        }
        CatalogView::Installed => selected_installed(&store, false)
            .map(|r| (r.provider, r.artifact, r.size_bytes, r.delete_blockers)),
        CatalogView::Catalog => match selected_artifact(&store, false) {
            Some(r) if r.presence == "installed" => {
                // The installed read knows the blockers (loaded, shared
                // cache); the catalog row only knows presence.
                let known = store.installed.with_untracked(|i| {
                    i.ready()
                        .and_then(|d| d.find(&r.provider, &r.artifact).cloned())
                });
                Some(match known {
                    Some(k) => (k.provider, k.artifact, k.size_bytes, k.delete_blockers),
                    None => (r.provider, r.artifact, r.download_bytes, Vec::new()),
                })
            }
            Some(r) => {
                store.notice.set(Some(format!(
                    "{} is {} — nothing to delete",
                    r.artifact,
                    weights_label(&r.presence)
                )));
                return;
            }
            None => None,
        },
    };
    match target {
        Some((p, a, size, blockers)) => confirm_delete(cx, sctx, &p, &a, size, blockers),
        None => store
            .notice
            .set(Some("no model selected — nothing to delete".into())),
    }
}

/// `e`: all → each seen engine → all.
fn cycle_engine(sctx: &ScreensCtx) {
    let store = sctx.store;
    let seen = store.providers_seen.get_untracked();
    if seen.is_empty() {
        store
            .notice
            .set(Some("no engines seen yet — r reads the catalog".into()));
        return;
    }
    let next = match store.engine_filter.get_untracked() {
        None => Some(seen[0].clone()),
        Some(cur) => seen
            .iter()
            .position(|p| *p == cur)
            .and_then(|i| seen.get(i + 1).cloned()),
    };
    store.notice.set(Some(format!(
        "engine: {}",
        next.as_deref().unwrap_or("all")
    )));
    store.engine_filter.set(next);
    store.catalog_sel.set(0);
    store.installed_sel.set(0);
    sctx.reload_catalog();
}

/// `c`: the download the screen points at — the selected row of the
/// Downloads view, the selected catalog artifact's live download — else
/// the one live job. Downloads ask first (the web's two-step cancel).
fn cancel_selected(cx: Scope, sctx: &ScreensCtx) {
    let store = sctx.store;
    let target = match store.view.get_untracked() {
        CatalogView::Downloads => match selected_download(&store, false) {
            Some(j) if j.is_active() => Some(j),
            Some(j) => {
                store.notice.set(Some(format!(
                    "{} {} is {} — nothing to cancel",
                    j.verb(),
                    j.subject(),
                    j.status
                )));
                return;
            }
            None => None,
        },
        CatalogView::Catalog => selected_artifact(&store, false)
            .and_then(|r| store.active_download(&r.provider, &r.artifact)),
        CatalogView::Installed => None,
    };
    match target {
        Some(j) => confirm_cancel_download(cx, sctx, j),
        None => {
            let live: Vec<JobView> = store
                .jobs
                .with_untracked(|v| v.iter().filter(|j| j.is_active()).cloned().collect());
            match live.as_slice() {
                [one] if matches!(one.kind.as_str(), "download" | "download_group") => {
                    confirm_cancel_download(cx, sctx, one.clone())
                }
                _ => sctx.cancel(),
            }
        }
    }
}

/// `u`: the selected INSTALLED, text-capable artifact becomes the
/// default text model (`output.text`), verified from a fresh read.
fn use_as_default(sctx: &ScreensCtx) {
    let store = sctx.store;
    if !sctx.caps.text_default {
        store.notice.set(Some(format!(
            "changing the default text model is not available over {}",
            sctx.host_label
        )));
        return;
    }
    if store.view.get_untracked() != CatalogView::Catalog {
        store.notice.set(Some(
            "u works on the catalog — v back to it and select an installed model".into(),
        ));
        return;
    }
    let Some(r) = selected_artifact(&store, false) else {
        store.notice.set(Some("no model selected".into()));
        return;
    };
    let why_not = if r.presence != "installed" {
        Some(format!("{} is not downloaded yet — w first, then u", r.artifact))
    } else if r.embedding == Some(true) {
        Some(format!("{} is an embedding model — it cannot answer text", r.artifact))
    } else if r.text_capable != Some(true) {
        Some(format!("{} is not a text model", r.model_name))
    } else {
        None
    };
    if let Some(msg) = why_not {
        store.notice.set(Some(msg));
        return;
    }
    let model = served_model_id(&r.provider, &r.artifact);
    let already = store
        .text_default
        .with_untracked(|d| d.ready().cloned().flatten() == Some((r.provider.clone(), model.clone())));
    if already {
        store
            .notice
            .set(Some(format!("{} · {model} is already the default text model", r.provider)));
        return;
    }
    sctx.set_text_default(&r.provider, &r.artifact);
}

/// `h`: one field; Enter searches Hugging Face (ONE request per Enter —
/// the hub rate-limits), an empty Enter goes back to the catalog.
fn open_hub_search(cx: Scope, sctx: &ScreensCtx) {
    let store = sctx.store;
    if !sctx.caps.hub_search {
        store.notice.set(Some(format!(
            "Hugging Face search is not available over {}",
            sctx.host_label
        )));
        return;
    }
    let viewport = abstracttui::app::use_viewport(cx).get_untracked();
    let value = cx.signal(store.hub.get_untracked().unwrap_or_default());
    let theme = use_theme(cx);
    let slot: Rc<std::cell::RefCell<Option<Modal>>> = Rc::new(std::cell::RefCell::new(None));
    let size = Size::new(60.min(viewport.w - 4).max(20), 5);
    let apply_ctx = sctx.clone();
    let (apply_slot, esc_slot) = (slot.clone(), slot.clone());
    let modal = Modal::open(&sctx.overlays, cx, viewport, size, move |mcx| {
        let t = theme.get_untracked().tokens;
        let slot = apply_slot.clone();
        let ctx = apply_ctx.clone();
        let esc = esc_slot.clone();
        Element::new()
            .style(LayoutStyle::column().grow(1.0))
            .shortcut(KeyChord::plain(Key::Escape), move |_| {
                if let Some(m) = esc.borrow_mut().take() {
                    m.close();
                }
            })
            .child(line(vec![span_bold(" Search Hugging Face", t.accent)]))
            .child(
                TextInput::new()
                    .layout(LayoutStyle::default().grow(1.0).h(1))
                    .value(value)
                    .placeholder("model name — Enter searches, empty goes back to the catalog")
                    .on_submit(move |_| {
                        if let Some(m) = slot.borrow_mut().take() {
                            m.close();
                        }
                        ctx.hub_search(&value.get_untracked());
                    })
                    .view(mcx),
            )
            .child(line(vec![span(
                " Enter searches · Esc keeps what is shown",
                t.text_faint,
            )]))
            .build()
    });
    *slot.borrow_mut() = Some(modal);
}
