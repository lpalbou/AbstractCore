//! The section screens: one field TABLE per page (section · field ·
//! value · state), Enter/e opens the right editor for the selected
//! row, x clears optional fields. In wizard mode the table filters to
//! the step's focused section so each phase reads as one purposeful
//! form. Corrupt/missing/unreadable render their honest states.

use abstracttui::prelude::*;
use abstracttui::widgets::{ColWidth, Column, Table};
use serde_json::json;

use crate::config::{FieldState, FileState, Snapshot};
use crate::schema::FieldKind;
use crate::store::Loadable;
use crate::worker::Cmd;
use crate::writes;

use super::editors::open_field_editor;
use super::util::{ellipsize, error_panel, line, span, span_bold};
use super::Ctx;

/// Render `f` over the current snapshot, with honest states for every
/// other file condition. A MISSING file renders the fold of an empty
/// object — exactly the in-memory defaults Python runs with
/// (manager.py:529-530) — so the section pages still teach what the
/// defaults ARE (and the wizard can configure a fresh machine).
/// Corrupt/unreadable refuse with the reason.
pub fn with_snapshot(
    t: &TokenSet,
    store: &crate::store::Store,
    f: impl FnOnce(&Snapshot, bool) -> View,
) -> View {
    match store.cfg.get() {
        Loadable::NotAsked => line(vec![span("— not loaded yet (r reloads)", t.text_muted)]),
        Loadable::Loading => line(vec![span("⟳ reading config…", t.info)]),
        Loadable::Failed(e) => error_panel(t, &e),
        Loadable::Ready(m) => match &m.state {
            FileState::Ready(snap) => f(snap, false),
            FileState::Missing => f(&crate::config::fold(&json!({}), 0, None, None), true),
            FileState::Corrupt { error, backups } => corrupt_panel(t, error, backups),
            FileState::Unreadable { error } => Element::new()
                .style(LayoutStyle::column())
                .child(line(vec![span_bold("✗ config file unreadable", t.error)]))
                .child(line(vec![span(format!("  {error}"), t.text)]))
                .child(line(vec![span(
                    "  fix the path or its permissions, then press r",
                    t.text_muted,
                )]))
                .build(),
        },
    }
}

/// The corrupt-file refusal: the error, the recovery artifacts, and
/// the one rule that prevents the known data-loss incident. The
/// headline rows are PINNED — under height pressure the artifact list
/// yields, never the refusal itself (the safety copy is the point).
pub fn corrupt_panel(t: &TokenSet, error: &str, backups: &[String]) -> View {
    let head = Element::new()
        .style(LayoutStyle::column().shrink(0.0))
        .child(line(vec![span_bold(
            "✗ config file corrupt — this console will NOT write until it is fixed",
            t.error,
        )]))
        .child(line(vec![span(format!("  parse error: {error}"), t.text)]))
        .child(line(vec![span(
            "  (a Python-side save from this state would reset it to defaults — fix first)",
            t.warn,
        )]))
        .build();
    let mut rest = Element::new().style(LayoutStyle::column());
    if backups.is_empty() {
        rest = rest.child(line(vec![span(
            "  no backups found beside it — repair the JSON by hand, then press r",
            t.text_muted,
        )]));
    } else {
        rest = rest.child(line(vec![span(
            "  recovery artifacts beside it (newest first):",
            t.text_muted,
        )]));
        for b in backups.iter().take(5) {
            rest = rest.child(line(vec![span(format!("    {b}"), t.text)]));
        }
        rest = rest.child(line(vec![span(
            "  compare/restore one by hand, then press r",
            t.text_muted,
        )]));
    }
    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .child(head)
        .child(rest.build())
        .build()
}

/// The rows a page shows: (section, key) pairs in schema order,
/// filtered to the wizard's focused section when one is set.
fn page_rows(
    snap: &Snapshot,
    section_names: &'static [&'static str],
    focus: Option<&'static str>,
) -> Vec<(&'static str, &'static str)> {
    let mut out = Vec::new();
    for sv in &snap.sections {
        if !section_names.contains(&sv.spec.name) {
            continue;
        }
        if let Some(f) = focus {
            if sv.spec.name != f {
                continue;
            }
        }
        for fv in &sv.fields {
            out.push((sv.spec.name, fv.key));
        }
    }
    out
}

/// A section screen: the field table + editing verbs.
pub fn page(
    cx: Scope,
    ctx: &Ctx,
    theme: Signal<&'static abstracttui::theme::Theme>,
    section_names: &'static [&'static str],
    sel: Signal<usize>,
) -> View {
    let store = ctx.store;
    let ui = ctx.ui;
    let ctx_edit = ctx.clone();
    let ctx_clear = ctx.clone();
    let ctx_table = ctx.clone();

    let body = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |gcx| {
        let t = theme.get().tokens;
        let focus = ui.focus_section.get().filter(|_| ui.wizard.get());
        with_snapshot(&t, &store, |snap, defaults_only| {
            let mut col = Element::new().style(LayoutStyle::column());
            if defaults_only {
                col = col.child(line(vec![span(
                    " no config file yet — these are the built-in defaults; saving a field creates it",
                    t.info,
                )]));
            }
            // Section-level warnings ride above the table (unknown
            // keys would be dropped by a Python save).
            for sv in &snap.sections {
                if section_names.contains(&sv.spec.name) && !sv.unknown_keys.is_empty() {
                    col = col.child(line(vec![
                        span_bold(format!(" ⚠ {} unknown keys: ", sv.spec.name), t.warn),
                        span(sv.unknown_keys.join(", "), t.text),
                        span("  — a Python-side save DROPS these", t.text_muted),
                    ]));
                }
            }
            col = col.child(fields_table(
                gcx,
                &ctx_table,
                &t,
                snap,
                section_names,
                focus,
                sel,
            ));
            col.build()
        })
    });

    // Selected-row truth line: the table's note/broken cells truncate
    // at narrow widths — the full state and note of the SELECTED field
    // always have one uncramped row.
    let detail = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let focus = ui.focus_section.get().filter(|_| ui.wizard.get());
        let pair = store.cfg.with(|c| {
            c.ready().and_then(|m| match &m.state {
                FileState::Ready(snap) => {
                    let rows = page_rows(snap, section_names, focus);
                    rows.get(sel.get()).map(|(s, k)| {
                        let sv = snap.sections.iter().find(|x| x.spec.name == *s)?;
                        let fv = sv.fields.iter().find(|f| f.key == *k)?;
                        Some(((*s).to_string(), (*k).to_string(), fv.clone()))
                    })?
                }
                _ => None,
            })
        });
        match pair {
            Some((section, key, fv)) => {
                let mut spans = vec![
                    span_bold(format!(" {section}.{key} "), t.accent),
                    span(fv.display.clone(), t.text),
                ];
                spans.push(span("  ", t.text));
                spans.extend(super::util::state_spans(&t, &fv.state));
                if let Some(n) = &fv.note {
                    spans.push(span(format!("  {n}"), t.text_faint));
                }
                line(spans)
            }
            None => line(vec![span(String::new(), t.text)]),
        }
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .shortcut(KeyChord::plain(Key::Char('e')), move |_| {
            activate_selected(cx, &ctx_edit, section_names, sel);
        })
        .shortcut(KeyChord::plain(Key::Char('x')), move |_| {
            clear_selected(cx, &ctx_clear, section_names, sel);
        })
        .child(body)
        .child(detail)
        .build()
}

fn fields_table(
    cx: Scope,
    ctx: &Ctx,
    t: &TokenSet,
    snap: &Snapshot,
    section_names: &'static [&'static str],
    focus: Option<&'static str>,
    sel: Signal<usize>,
) -> View {
    let pairs = page_rows(snap, section_names, focus);
    let w = abstracttui::app::use_viewport(cx).get().w;
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (section, key) in &pairs {
        let sv = snap
            .sections
            .iter()
            .find(|s| s.spec.name == *section)
            .expect("section exists");
        let fv = sv.fields.iter().find(|f| f.key == *key).expect("field exists");
        let state = match &fv.state {
            FieldState::Default => "· default".to_string(),
            FieldState::Set => "● set".to_string(),
            FieldState::Broken(r) => format!("✗ {}", ellipsize(r, 40)),
        };
        let mut row = vec![(*section).to_string(), (*key).to_string()];
        row.push(ellipsize(&fv.display, 38));
        row.push(state);
        if w >= 100 {
            row.push(fv.note.clone().unwrap_or_default());
        }
        rows.push(row);
    }
    let mut cols = vec![
        Column::new("section", ColWidth::Cells(15)),
        Column::new("field", ColWidth::Cells(22)),
        Column::new("value", ColWidth::Cells(40)),
        Column::new("state", ColWidth::Flex(1.0)),
    ];
    if w >= 100 {
        cols.push(Column::new("note", ColWidth::Flex(1.0)));
    }
    let ctx = ctx.clone();
    let pairs2 = pairs.clone();
    Table::new(cols)
        .rows(rows)
        .selection(sel)
        .on_activate(move |i| {
            if let Some((section, key)) = pairs2.get(i) {
                open_field_editor(cx, &ctx, section, key);
            }
        })
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .autofocus()
        .build()
}

/// The selected (section, key) resolved from live state — shared by
/// the e and x verbs.
fn selected_pair(
    ctx: &Ctx,
    section_names: &'static [&'static str],
    sel: Signal<usize>,
) -> Option<(&'static str, &'static str)> {
    let focus = ctx
        .ui
        .focus_section
        .get_untracked()
        .filter(|_| ctx.ui.wizard.get_untracked());
    ctx.store.cfg.with_untracked(|c| {
        c.ready().and_then(|m| match &m.state {
            FileState::Ready(snap) => {
                page_rows(snap, section_names, focus).get(sel.get_untracked()).copied()
            }
            FileState::Missing => {
                let snap = crate::config::fold(&json!({}), 0, None, None);
                page_rows(&snap, section_names, focus)
                    .get(sel.get_untracked())
                    .copied()
            }
            _ => None,
        })
    })
}

fn activate_selected(
    cx: Scope,
    ctx: &Ctx,
    section_names: &'static [&'static str],
    sel: Signal<usize>,
) {
    match selected_pair(ctx, section_names, sel) {
        Some((section, key)) => open_field_editor(cx, ctx, section, key),
        None => ctx
            .store
            .notice
            .set(Some("no field selected — nothing to edit".into())),
    }
}

fn clear_selected(
    cx: Scope,
    ctx: &Ctx,
    section_names: &'static [&'static str],
    sel: Signal<usize>,
) {
    let Some((section, key)) = selected_pair(ctx, section_names, sel) else {
        ctx.store
            .notice
            .set(Some("no field selected — nothing to clear".into()));
        return;
    };
    // A missing config file holds nothing to clear — resetting a
    // default would only mint an empty `{}` file the operator never
    // asked for (M2 review P3-8).
    let missing = ctx
        .store
        .cfg
        .with_untracked(|c| matches!(c.ready().map(|m| &m.state), Some(FileState::Missing)));
    if missing {
        ctx.store.notice.set(Some(
            "no config file yet — everything is already at its default".into(),
        ));
        return;
    }
    // Refusals BEFORE confirms — confirming a reset and then being
    // told "use its editor" is backwards (M2 review P3-12).
    if matches!(
        writes::field_route(section, key),
        writes::FieldRoute::VisionStrategy
    ) {
        ctx.store.notice.set(Some(
            "vision.strategy resets through its own editor (Enter) — the disabled default \
             also clears the caption pair and chain"
                .into(),
        ));
        return;
    }
    if !ctx.writable_now() {
        return;
    }
    // audio.strategy's TRUE default state is value-removed + explicit
    // flag cleared — the CLI setter would set the flag and freeze the
    // smart default (M2 review P2-4).
    if section == "audio" && key == "strategy" {
        let ctx2 = ctx.clone();
        super::forms::confirm_danger(
            cx,
            ctx.ui,
            "Reset audio.strategy to the smart default? (also clears the explicit flag)".into(),
            "Reset it",
            "Keep it",
            move || {
                let spec = writes::reset_audio_strategy(ctx2.write_base(), None);
                ctx2.send(Cmd::Write(Box::new(spec)));
            },
        );
        return;
    }
    // Clear = reset to default. Only meaningful for fields that can
    // differ; pair/secret fields clear through their own editors.
    let spec = crate::schema::section(section)
        .and_then(|s| s.fields.iter().find(|f| f.key == key));
    let Some(fs) = spec else { return };
    match writes::field_route(section, key) {
        writes::FieldRoute::Secret => {
            super::editors::open_field_editor(cx, ctx, section, key);
        }
        writes::FieldRoute::Pair if section == "default_models" => {
            let ctx2 = ctx.clone();
            super::forms::confirm_danger(
                cx,
                ctx.ui,
                "Clear the global default model? (also clears route input.text)".into(),
                "Clear it",
                "Keep it",
                move || {
                    let spec = writes::clear_global_default(ctx2.write_base(), None);
                    ctx2.send(Cmd::Write(Box::new(spec)));
                },
            );
        }
        writes::FieldRoute::Pair => {
            ctx.store.notice.set(Some(format!(
                "{section}.{key} is pair-coupled — set a new pair with Enter"
            )));
        }
        _ => {
            let nullable = matches!(
                fs.kind,
                FieldKind::OptStr | FieldKind::OptPath | FieldKind::OptInt { .. } | FieldKind::OptEnum(_)
            );
            let ctx2 = ctx.clone();
            let label = if nullable {
                format!("Clear {section}.{key}?")
            } else {
                format!(
                    "Reset {section}.{key} to its default ({})?",
                    fs.default.render()
                )
            };
            let default = fs.default;
            super::forms::confirm_danger(
                cx,
                ctx.ui,
                label,
                "Reset it",
                "Keep it",
                move || {
                    let spec = if nullable {
                        writes::clear_scalar(section, key, ctx2.write_base(), None)
                    } else {
                        // Non-nullable fields reset by WRITING the
                        // default (absent ≡ default anyway, but the
                        // CLI setter keeps coupled flags honest).
                        let v = match default {
                            crate::schema::Dv::S(s) => json!(s),
                            crate::schema::Dv::B(b) => json!(b),
                            crate::schema::Dv::I(i) => json!(i),
                            crate::schema::Dv::F(f) => json!(f),
                            _ => json!(null),
                        };
                        if v.is_null() {
                            writes::clear_scalar(section, key, ctx2.write_base(), None)
                        } else {
                            writes::set_scalar(section, key, v, ctx2.write_base(), None)
                        }
                    };
                    match spec {
                        Ok(spec) => ctx2.send(Cmd::Write(Box::new(spec))),
                        Err(e) => ctx2.store.notice.set(Some(e)),
                    }
                },
            );
        }
    }
}

/// The Embeddings screen: the field table + the mirror rule that makes
/// the legacy pair confusing in the wild (reads prefer route
/// embedding.text; setters write both — manager.py:830-848, 1421-1431).
pub fn embeddings_page(
    cx: Scope,
    ctx: &Ctx,
    theme: Signal<&'static abstracttui::theme::Theme>,
) -> View {
    let store = ctx.store;
    let table = page(cx, ctx, theme, &["embeddings"], ctx.ui.embeddings_sel);
    let mirror = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        let route_line: Vec<super::util::SpanSpec> = match store.routes.get() {
            Loadable::Ready(r) => match r.rows.iter().find(|row| row.key == "embedding.text") {
                Some(row) if row.configured => {
                    let pair = format!(
                        "{} / {}",
                        row.provider.clone().unwrap_or_else(|| "?".into()),
                        row.model.clone().unwrap_or_else(|| "?".into())
                    );
                    vec![
                        span(" route embedding.text: ", t.text_muted),
                        span_bold(pair, t.text),
                    ]
                }
                _ => vec![span(
                    " route embedding.text: not configured — the legacy pair above applies",
                    t.text_muted,
                )],
            },
            Loadable::Loading => vec![span(" route embedding.text: ⟳ loading…", t.info)],
            Loadable::Failed(_) => vec![span(
                " route embedding.text: unavailable (CLI view failed — see Review)",
                t.warn,
            )],
            Loadable::NotAsked => vec![span(" route embedding.text: not loaded", t.text_muted)],
        };
        Element::new()
            .style(LayoutStyle::column())
            .child(line(route_line))
            .child(line(vec![span(
                " reads prefer the route; setters keep both in sync",
                t.text_faint,
            )]))
            .build()
    });
    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .child(table)
        .child(mirror)
        .build()
}

/// Legacy read-only renderer kept for the Providers screen's api_keys
/// block (that screen renders keys as rows inside its own Block).
pub fn section_rows(t: &TokenSet, sv: &crate::config::SectionView) -> View {
    let mut col = Element::new().style(LayoutStyle::column());
    for fv in &sv.fields {
        col = col.child(field_row(t, fv));
    }
    if !sv.unknown_keys.is_empty() {
        col = col.child(line(vec![
            span_bold("  ⚠ unknown keys: ", t.warn),
            span(sv.unknown_keys.join(", "), t.text),
            span(
                "  — a Python-side save DROPS these; this console preserves them",
                t.text_muted,
            ),
        ]));
    }
    col.build()
}

/// One field row: key, value (ink by state), state, note. Cell-exact
/// padding — char-count padding misaligns CJK/emoji values.
fn field_row(t: &TokenSet, fv: &crate::config::FieldView) -> View {
    let value_ink = match fv.state {
        FieldState::Set => t.text,
        FieldState::Default => t.text_muted,
        FieldState::Broken(_) => t.error,
    };
    let mut spans = vec![
        span(format!(" {}", super::util::pad_cells(fv.key, 22)), t.text_muted),
        span(super::util::pad_cells(&fv.display, 28), value_ink),
        span(" ", t.text),
    ];
    spans.extend(super::util::state_spans(t, &fv.state));
    if let Some(n) = &fv.note {
        spans.push(span(format!("  {n}"), t.text_faint));
    }
    line(spans)
}
