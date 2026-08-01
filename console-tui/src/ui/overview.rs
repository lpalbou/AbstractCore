//! The Overview — the honest mirror. One glance answers: which file,
//! which CLI, does Python agree, and what is set / default / broken in
//! every section. Enter on a section row jumps to its owning screen.

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Table};

use crate::config::Snapshot;
use crate::store::Loadable;

use super::sections::with_snapshot;
use super::util::{line, span, span_bold};
use super::widths;
use super::{config_identity_line, screen_for_section, Ctx};

pub fn view(_cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui;

    // Identity block: fixed rows, pinned — this is the "which config am
    // I editing" answer and must survive any content pressure below.
    // Built inside the reactive region so theme switches retint the
    // chrome, not just the text.
    let identity = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        let rows = Element::new()
            .style(LayoutStyle::column())
            .child(line({
                let mut spans = vec![span(" file  ", t.text_faint)];
                spans.extend(config_identity_line(&t, &store));
                spans
            }))
            .child(line({
                let mut spans = vec![span(" cli   ", t.text_faint)];
                match store.cli.get() {
                    Some(info) => {
                        spans.push(span(info.bin.display().to_string(), t.text));
                        spans.push(span(format!("  ({})", info.source), t.text_faint));
                    }
                    None => {
                        spans.push(span_bold("not found", t.warn));
                        spans.push(span(
                            "  — file mirror works; derived views and writes need it ($ABSTRACTCORE_BIN)",
                            t.text_muted,
                        ));
                    }
                }
                spans
            }))
            .build();
        let mut col = Element::new().style(LayoutStyle::column()).child(rows);
        for l in agreement_lines(&t, &store) {
            col = col.child(line(l));
        }
        // The P1-1 lane: Python told us (exit-0 stderr `#FALLBACK`)
        // that it refuses this file and runs on defaults — the single
        // most important line the mirror can show, pinned here.
        if let Some(fb) = store.python_fallback.get() {
            col = col
                .child(line(vec![span_bold(
                    " ✗ PYTHON REFUSES THIS FILE — every abstractcore run backs it up and uses DEFAULTS",
                    t.error,
                )]))
                .child(line(vec![span(
                    // `line()` fits this to the row it is drawn on; a
                    // 200-char pre-cut only hid the end on a wide term.
                    format!("   {fb}"),
                    t.text_muted,
                )]));
        }
        Block::new()
            .title("Config identity")
            .child(col.build())
            .element(&t)
            .build()
    });

    let table = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |gcx| {
        let t = theme.get().tokens;
        with_snapshot(&t, &store, |snap, defaults_only| {
            let mut col = Element::new().style(LayoutStyle::column());
            if defaults_only {
                col = col.child(line(vec![span(
                    " no config file yet — Python runs on built-in defaults; the first write creates it",
                    t.info,
                )]));
            }
            // Shapes Python's loader raises on (the mirror's own
            // detection — belt to the #FALLBACK stderr braces): name
            // each one so the operator can fix the exact row.
            for refusal in snap.python_refusals.iter().take(4) {
                col = col.child(line(vec![
                    span_bold(" ✗ Python will refuse this file: ", t.error),
                    span(refusal.clone(), t.text),
                ]));
            }
            col = col.child(sections_table(gcx, Jump { ui, store }, &t, snap));
            if !snap.unknown_sections.is_empty() {
                col = col.child(line(vec![
                    span_bold(" ⚠ unknown sections: ", t.warn),
                    span(snap.unknown_sections.join(", "), t.text),
                    span(
                        "  — preserved by this console; a Python-side save drops them",
                        t.text_muted,
                    ),
                ]));
            }
            col.build()
        })
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .child(identity)
        .child(table)
        .build()
}

/// A Copy bundle so the table's on_activate can jump screens without
/// cloning the whole Ctx into the closure.
#[derive(Clone, Copy)]
struct Jump {
    ui: super::UiState,
    store: crate::store::Store,
}

/// Same-file check by path COMPONENTS, not strings — pathlib
/// normalizes doubled slashes, PathBuf display does not, and a false
/// "DIFFERENT FILES" alarm on the same file would be the worst kind
/// of wrong (review P3-14).
pub fn same_path(a: &str, b: &str) -> bool {
    use std::path::Path;
    Path::new(a).components().eq(Path::new(b).components())
}

fn agreement_lines(t: &TokenSet, store: &crate::store::Store) -> Vec<Vec<super::util::SpanSpec>> {
    let mine = match store.cfg.get() {
        Loadable::Ready(m) => Some(m.path.path.display().to_string()),
        _ => None,
    };
    let cli_echo = match store.routes.get() {
        Loadable::Ready(r) => r.config_file,
        _ => None,
    };
    let lead = || span(" agree ", t.text_faint);
    if store.python_fallback.with(|f| f.is_some()) {
        // Path agreement is meaningless while Python refuses the file
        // — saying "✓ same file" under the refusal banner would
        // re-vouch for it.
        return vec![vec![
            lead(),
            span_bold("✗ ", t.error),
            span(
                "the CLI reads the same path but REFUSES its content (see above)",
                t.text,
            ),
        ]];
    }
    match (mine, cli_echo) {
        (Some(a), Some(b)) if same_path(&a, &b) => vec![vec![
            lead(),
            span_bold("✓ ", t.ok),
            span(
                "the abstractcore CLI reads the same file this console shows",
                t.text_muted,
            ),
        ]],
        // Two lines: the second names the fix — one line truncated the
        // teaching away at ≤110 cols (the alarm without the cause).
        (Some(a), Some(b)) => vec![
            vec![
                lead(),
                span_bold("✗ DIFFERENT FILES ", t.error),
                span(format!("— the CLI reads {b}"), t.text),
            ],
            vec![
                span("       ", t.text_faint),
                span(
                    format!("console shows {a} — check ABSTRACTCORE_CONFIG_FILE/_DIR"),
                    t.text_muted,
                ),
            ],
        ],
        _ => vec![vec![
            lead(),
            span(
                "— unknown until the CLI view loads (r reloads)",
                t.text_muted,
            ),
        ]],
    }
}

fn sections_table(cx: Scope, jump: Jump, t: &TokenSet, snap: &Snapshot) -> View {
    use crate::schema::SectionKind;

    struct Row {
        name: &'static str,
        state: String,
        details: String,
    }
    let mut rows_data: Vec<Row> = Vec::new();
    for sv in &snap.sections {
        let (state, details) = match sv.spec.kind {
            SectionKind::Fields => {
                let set = sv.set_count();
                let broken = sv.broken_count();
                if sv.spec.name == "api_keys" {
                    // Broken keys must not vanish from the section
                    // holding the secrets (review P2-1); the total is
                    // derived, never a literal.
                    let set_keys: Vec<&str> = sv
                        .fields
                        .iter()
                        .filter(|f| matches!(f.state, crate::config::FieldState::Set))
                        .map(|f| f.key)
                        .collect();
                    if broken > 0 {
                        rows_data.push(Row {
                            name: sv.spec.name,
                            state: format!("✗ {broken} broken"),
                            details: first_broken_detail(sv),
                        });
                        continue;
                    }
                    let total = sv.fields.len();
                    let state = if set_keys.is_empty() {
                        "· none".to_string()
                    } else {
                        format!("● {} of {total}", set_keys.len())
                    };
                    (
                        state,
                        if set_keys.is_empty() {
                            "no keys stored".to_string()
                        } else {
                            set_keys.join(", ")
                        },
                    )
                } else if broken > 0 {
                    (format!("✗ {broken} broken"), first_broken_detail(sv))
                } else if set > 0 {
                    (format!("● {set} set"), set_preview(sv))
                } else {
                    ("· default".to_string(), String::new())
                }
            }
            SectionKind::Routes => {
                let in_file = snap.routes_in_file.len();
                let details = match jump.store.routes.get() {
                    Loadable::Ready(r) if r.ok => {
                        format!(
                            "{} of {} routes configured",
                            r.configured_count(),
                            r.rows.len()
                        )
                    }
                    Loadable::Loading => "⟳ loading the derived view…".to_string(),
                    Loadable::Failed(_) => {
                        format!("{in_file} route entries in file (CLI view failed)")
                    }
                    _ => format!("{in_file} route entries in file"),
                };
                let state = if in_file > 0 {
                    format!("● {in_file} set")
                } else {
                    "· default".to_string()
                };
                (state, details)
            }
            SectionKind::Profiles => {
                let n = snap.profiles_in_file;
                let details = match jump.store.profiles.get() {
                    Loadable::Ready(p) => {
                        let with_key = p.profiles.iter().filter(|x| x.api_key_set).count();
                        let ids: Vec<String> = p.profiles.iter().map(|x| x.id.clone()).collect();
                        format!("{} ({with_key} with key): {}", ids.len(), ids.join(", "))
                    }
                    Loadable::Loading => "⟳ loading the derived view…".to_string(),
                    Loadable::Failed(_) => format!("{n} in file (CLI view failed)"),
                    _ => format!("{n} in file"),
                };
                let state = if n > 0 {
                    format!("● {n} set")
                } else {
                    "· none".to_string()
                };
                (state, details)
            }
        };
        rows_data.push(Row {
            name: sv.spec.name,
            state,
            details,
        });
    }

    let w = abstracttui::app::use_viewport(cx).get().w;
    let mut rows: Vec<Vec<String>> = rows_data
        .iter()
        .map(|r| {
            let mut row = vec![r.name.to_string(), r.state.clone()];
            // Uncapped: `ui::widths` sizes this column to the summary and
            // cuts it only when the terminal cannot carry it.
            row.push(r.details.clone());
            if w >= 96 {
                row.push(format!("→ {}", super::SCREENS[screen_for_section(r.name)]));
            }
            row
        })
        .collect();

    // `details` is a SENTENCE, so it keeps its head; section names and
    // the destination screen are short labels that never earn a cut.
    let mut rules = vec![
        widths::ColRule::head("section", 14),
        widths::ColRule::head("state", 11),
        widths::ColRule::head("details", 24),
    ];
    if w >= 96 {
        rules.push(widths::ColRule::head("edit on", 12));
    }
    let cols = widths::columns(&rules, &mut rows, w);

    let names: Vec<&'static str> = rows_data.iter().map(|r| r.name).collect();
    Table::new(cols)
        .rows(rows)
        .selection(jump.ui.overview_sel)
        .on_activate(move |i| {
            if let Some(name) = names.get(i) {
                let target = screen_for_section(name);
                jump.ui.screen.set(target);
                jump.store
                    .notice
                    .set(Some(format!("→ {}", super::SCREENS[target])));
            }
        })
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .autofocus()
        .build()
}

fn first_broken_detail(sv: &crate::config::SectionView) -> String {
    sv.fields
        .iter()
        .find_map(|f| match &f.state {
            crate::config::FieldState::Broken(reason) => Some(format!("{}: {reason}", f.key)),
            _ => None,
        })
        .unwrap_or_default()
}

/// Up to three set fields as `k=v` — enough to recognize the section's
/// configuration at a glance without opening it.
fn set_preview(sv: &crate::config::SectionView) -> String {
    let parts: Vec<String> = sv
        .fields
        .iter()
        .filter(|f| matches!(f.state, crate::config::FieldState::Set))
        .take(3)
        .map(|f| format!("{}={}", f.key, f.display))
        .collect();
    parts.join("  ")
}
