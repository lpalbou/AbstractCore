//! Review: the session journal, the config-file identity proof (does
//! the Python side read the same file this console shows?), and — in
//! M3 — the provider connectivity and generation test verbs.

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Table};

use crate::store::Loadable;

use super::util::{line, span, span_bold};
use super::widths;
use super::Ctx;

pub fn view(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;

    let identity = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        let mut col = Element::new().style(LayoutStyle::column()).child(line({
            let mut spans = vec![span(" file  ", t.text_faint)];
            spans.extend(super::config_identity_line(&t, &store));
            spans
        }));
        // The one-file identity proof, spelled out where the operator
        // comes to verify: both derived views echo the config_file the
        // Python side actually read.
        let mine = match store.cfg.get() {
            Loadable::Ready(m) => Some(m.path.path.display().to_string()),
            _ => None,
        };
        for (label, echo) in [
            (
                "defaults",
                store.routes.with(|r| r.ready().and_then(|d| d.config_file.clone())),
            ),
            (
                "providers",
                store
                    .profiles
                    .with(|p| p.ready().and_then(|d| d.config_file.clone())),
            ),
        ] {
            let spans = match (&mine, echo) {
                (Some(a), Some(b)) if *a == b => vec![
                    span(format!(" {label:<9}"), t.text_faint),
                    span_bold("✓ same file ", t.ok),
                    span(b, t.text_muted),
                ],
                (Some(_), Some(b)) => vec![
                    span(format!(" {label:<9}"), t.text_faint),
                    span_bold("✗ DIFFERENT ", t.error),
                    span(format!("CLI read {b}"), t.text),
                ],
                _ => vec![
                    span(format!(" {label:<9}"), t.text_faint),
                    span("— no CLI echo yet (r reloads)", t.text_muted),
                ],
            };
            col = col.child(line(spans));
        }
        Block::new()
            .title("Python-side agreement")
            .child(col.build())
            .element(&t)
            .build()
    });

    // Test evidence: the latest result per target, verdicts colored,
    // never fabricated — an empty state teaches the verbs instead.
    let viewport = abstracttui::app::use_viewport(cx);
    let tests = dyn_view(LayoutStyle::column().shrink(0.0), move || {
        let t = theme.get().tokens;
        let avail = viewport.get().w - widths::BLOCK_CHROME;
        let results = store.tests.get();
        let mut col = Element::new().style(LayoutStyle::column());
        if results.is_empty() {
            col = col.child(line(vec![span(
                " no tests run yet — t tests a provider/profile/route on its screen · g runs a \
                 cheap generation over the default route",
                t.text_muted,
            )]));
        } else {
            for r in results.iter().take(6) {
                let color = match r.verdict {
                    crate::probes::Verdict::Proven => t.ok,
                    crate::probes::Verdict::NotProven => t.warn,
                    crate::probes::Verdict::Failed => t.error,
                };
                // The detail is the ELASTIC span of this line: budgeted
                // against the row it is drawn on, not a constant that cut
                // it at 110 with ninety cells still free.
                let when = format!(" {} ", r.when);
                let verdict =
                    format!("{} {:<11}", r.verdict.glyph(), r.verdict.word());
                let label = format!("{} — ", r.label);
                let detail = widths::head_fit(
                    &r.detail,
                    widths::elastic_budget(&[&when, &verdict, &label], avail),
                );
                col = col.child(line(vec![
                    span(when, t.text_faint),
                    span_bold(verdict, color),
                    span(label, t.text),
                    span(detail, t.text_muted),
                ]));
            }
            // Never let evidence vanish silently (M3 review P3-9).
            if results.len() > 6 {
                col = col.child(line(vec![span(
                    format!(" … and {} older (journal below has all)", results.len() - 6),
                    t.text_faint,
                )]));
            }
        }
        Block::new()
            .title("Test evidence (latest per target)")
            .child(col.build())
            .element(&t)
            .build()
    });

    let journal = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |gcx| {
        let t = theme.get().tokens;
        let entries = store.journal.get();
        let body = if entries.is_empty() {
            line(vec![span(
                " no actions recorded this session — loads journal only when they fail",
                t.text_muted,
            )])
        } else {
            let w = abstracttui::app::use_viewport(gcx).get().w;
            let mut rows: Vec<Vec<String>> = entries
                .iter()
                .rev()
                .map(|e| {
                    // "NOT PROVEN — " Errs are the probe lane's third
                    // state (written in worker::handle_probe) — the ?
                    // glyph, never the failure ✗ (M3 review P3-5).
                    let outcome = match &e.outcome {
                        Ok(s) => format!("✓ {s}"),
                        Err(s) if s.starts_with("NOT PROVEN") => format!("? {s}"),
                        Err(s) => format!("✗ {s}"),
                    };
                    // Uncapped: the journal is EVIDENCE, and a wide
                    // terminal has no reason to hide the end of it.
                    vec![e.when.clone(), e.action.clone(), outcome]
                })
                .collect();
            // The action names a route/provider/field — its TAIL says
            // which one; the outcome is prose and reads from the left.
            // Two cells for the block's own border, the same chrome the
            // routes screen budgets for.
            let rules = [
                widths::ColRule::head("when", 8),
                widths::ColRule::tail("action", 24),
                widths::ColRule::head("outcome", 30),
            ];
            let cols = widths::columns(&rules, &mut rows, w - widths::BLOCK_CHROME);
            Table::new(cols)
                .rows(rows)
                .layout(LayoutStyle::default().grow(1.0))
                .element(gcx, &t)
                .build()
        };
        Block::new()
            .title("Session journal")
            .child(body)
            .element(&t)
            .build()
    });

    // The identity block is FIXED content (~5 rows) — pinned, never
    // scrolled: a basis-0 Scroll beside the journal's grow starved it
    // to zero rows (the providers-screen class, caught live by the
    // smoke). The page wrapper is focusable+autofocused so the
    // PageHost digit/chord surface has a focus owner after a
    // mode-flip rebuild (keys target the tree root when nothing is
    // focused — the host's shortcuts are not on that path).
    let _ = cx;
    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .focusable()
        .autofocus()
        .child(
            Element::new()
                .style(LayoutStyle::column().shrink(0.0))
                .child(identity)
                .build(),
        )
        .child(tests)
        .child(journal)
        .build()
}
