//! The **Models** screen (page id `catalog`): browse what can be
//! downloaded, see whether it fits this host, download (`w`), delete
//! (`d`), filter (`/`, `f`, `e`), and flip to what is installed (`v`).

use abstracttui::app::{ChoiceOption, ChoiceOutcome, ChoicePrompt};
use abstracttui::prelude::*;

use super::data::{bytes_label, fit_label, params_label, weights_label, ArtifactRow, InstalledRow};
use super::{
    confirm_delete, job_strip, open_filter, remote_line, CatalogView, Remote, ScreensCtx,
    ScreensStore,
};
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
        };
        let mut spans = vec![span_bold(
            match view {
                CatalogView::Catalog => " catalog",
                CatalogView::Installed => " installed",
            },
            t.accent,
        )];
        if let Some(c) = count {
            spans.push(span(format!(" · {c}"), t.text));
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
                other => remote_line(&t, "catalog", &other).expect("not ready"),
            },
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

    let detail = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let _ = (store.query.get(), store.engine_filter.get());
        match store.view.get() {
            CatalogView::Catalog => match selected_artifact(&store, true) {
                Some(r) => artifact_detail(&t, &r),
                None => line(vec![span(String::new(), t.text)]),
            },
            CatalogView::Installed => match selected_installed(&store, true) {
                Some(r) => installed_detail(&t, &r),
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
                    CatalogView::Installed => CatalogView::Catalog,
                };
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
    ("w", "download"),
    ("d", "delete"),
    ("/", "filter"),
    ("f", "fits only"),
    ("e", "engine"),
    ("v", "installed/catalog"),
    ("c", "cancel job"),
];

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
            row.push(fit_label(&r.fit).to_string());
            row.push(weights_label(&r.presence).to_string());
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
    rules.push(ColRule::head("fit", 15));
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

fn artifact_detail(t: &TokenSet, r: &ArtifactRow) -> View {
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
        span(format!("fit {}", fit_label(&r.fit)), fit_ink),
    ];
    if let Some(need) = r.need_bytes {
        spans.push(span(
            format!(" (needs {})", bytes_label(Some(need))),
            t.text_muted,
        ));
    }
    if r.disk_ok == Some(false) {
        spans.push(span(" · not enough disk", t.error));
    }
    if let Some(ctx) = r.max_context {
        spans.push(span(format!(" · ctx {ctx}"), t.text_muted));
    }
    if let Some(n) = r.fit_notes.first() {
        spans.push(span(format!(" · {n}"), t.text_faint));
    }
    match r.presence.as_str() {
        "installed" => spans.push(span(" · d deletes", t.text_faint)),
        "absent" if r.downloadable => spans.push(span(" · w downloads", t.text_faint)),
        _ => {}
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
    if store.view.get_untracked() == CatalogView::Installed {
        store.notice.set(Some(
            "this list is what is already on disk — v shows the catalog to download from".into(),
        ));
        return;
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
