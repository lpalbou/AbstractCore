//! Capability routes: the 24-route derived view from
//! `abstractcore config defaults --json` — coverage decorations,
//! the output.text alias, per-route provider/model/options — with the
//! route editor (e) and clear (x) writing through `config
//! set-default`/`clear-default`, verified against the fresh view.

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Button, ColWidth, Column, Table};

use crate::schema;
use crate::store::{Loadable, RouteRow, RoutesData};
use crate::worker::{next_form_id, Cmd};
use crate::writes;

use super::forms::{
    confirm_danger, install_write_done, message_slot, open_form_guarded,
};
use super::util::{ellipsize, field, line, loadable_view, or_dash, span, span_bold};
use super::Ctx;

pub fn view(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui;
    let ctx_edit = ctx.clone();
    let ctx_clear = ctx.clone();

    let banner = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        match store.routes.get() {
            Loadable::Ready(d) => {
                let mut spans = vec![
                    span(
                        format!(" {} of {} configured", d.configured_count(), d.rows.len()),
                        t.text,
                    ),
                    span("  ·  output.text mirrors input.text", t.text_faint),
                ];
                if !d.errors.is_empty() {
                    spans.push(span_bold(
                        format!("  ·  errors: {}", d.errors.join(" | ")),
                        t.error,
                    ));
                }
                line(spans)
            }
            _ => line(vec![span(String::new(), t.text)]),
        }
    });

    let table = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |gcx| {
        let t = theme.get().tokens;
        let data = store.routes.get();
        loadable_view(
            &t,
            &data,
            |d: &RoutesData| d.rows.is_empty(),
            "the CLI reported no routes",
            |d| routes_table(gcx, &t, d, ui.route_sel),
        )
    });

    // Selected-row extras: options / base_url / package hint — the
    // fields that don't earn a column but answer "why is this row odd".
    let detail = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let row: Option<RouteRow> = store
            .routes
            .with(|d| d.ready().and_then(|d| d.rows.get(ui.route_sel.get()).cloned()));
        match row {
            Some(r) => {
                let mut spans = vec![span_bold(format!(" {} ", r.key), t.accent)];
                if let Some(u) = &r.base_url {
                    spans.push(span(format!("base_url {u}  "), t.text_muted));
                }
                if let Some(o) = &r.options {
                    spans.push(span(format!("options {}  ", mask_options(o)), t.text_muted));
                }
                if !r.configured {
                    if let Some(h) = &r.package_hint {
                        spans.push(span(format!("needs: {h}"), t.text_faint));
                    }
                }
                line(spans)
            }
            None => line(vec![span(String::new(), t.text)]),
        }
    });

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .shortcut(KeyChord::plain(Key::Char('e')), move |_| {
            edit_selected(cx, &ctx_edit);
        })
        .shortcut(KeyChord::plain(Key::Char('x')), move |_| {
            clear_selected(cx, &ctx_clear);
        })
        .shortcut(KeyChord::plain(Key::Char('t')), {
            let ctx_test = ctx.clone();
            move |_| test_selected(&ctx_test)
        })
        .child(banner)
        .child(table)
        .child(detail)
        .build()
}

/// Test the selected route: the configured model must be among what
/// the provider ACTUALLY serves (live discovery) — capability-agnostic
/// (a voice route can't be chat-tested, but its model's existence can
/// always be checked). The effective pair covers alias/covered rows.
fn test_selected(ctx: &Ctx) {
    let Some(row) = selected_route(ctx) else {
        ctx.store
            .notice
            .set(Some("no route selected — nothing to test".into()));
        return;
    };
    let (Some(provider), Some(model)) = (row.provider.clone(), row.model.clone()) else {
        ctx.store.notice.set(Some(format!(
            "{} resolves to no provider/model — nothing to test (e edits a route)",
            row.key
        )));
        return;
    };
    // `endpoint:<id>` routes usually carry no route-level base_url —
    // the PROFILE holds it, and the console knows the profile. Without
    // this lookup the same target got weaker evidence from this screen
    // than from the Providers picker ("no known endpoint to
    // reach-check" while the endpoint sat in store.profiles — M3
    // review P2-4).
    let base_url = row.base_url.clone().or_else(|| {
        provider.strip_prefix("endpoint:").and_then(|id| {
            ctx.store.profiles.with_untracked(|d| {
                d.ready()
                    .and_then(|d| d.profiles.iter().find(|p| p.id == id))
                    .map(|p| p.base_url.clone())
            })
        })
    });
    let spec = crate::probes::route_check(&row.key, &provider, &model, base_url.as_deref());
    ctx.send_probe(spec);
}

fn selected_route(ctx: &Ctx) -> Option<RouteRow> {
    let idx = ctx.ui.route_sel.get_untracked();
    ctx.store
        .routes
        .with_untracked(|d| d.ready().and_then(|d| d.rows.get(idx).cloned()))
}

fn edit_selected(cx: Scope, ctx: &Ctx) {
    let Some(row) = selected_route(ctx) else {
        ctx.store
            .notice
            .set(Some("no route selected — nothing to edit".into()));
        return;
    };
    if !row.editable() {
        let reason = if row.key == "output.text" {
            "output.text mirrors input.text — edit that route instead".to_string()
        } else if row.covered_by.is_some() {
            format!("{} is covered and not overrideable", row.key)
        } else {
            format!("{} is read-only", row.key)
        };
        ctx.store.notice.set(Some(reason));
        return;
    }
    if !ctx.writable_now() {
        return;
    }
    open_route_editor(cx, ctx, row);
}

fn clear_selected(cx: Scope, ctx: &Ctx) {
    let Some(row) = selected_route(ctx) else {
        ctx.store
            .notice
            .set(Some("no route selected — nothing to clear".into()));
        return;
    };
    if row.key == "output.text" {
        ctx.store.notice.set(Some(
            "output.text mirrors input.text — clear that route instead".into(),
        ));
        return;
    }
    if !row.configured {
        ctx.store
            .notice
            .set(Some(format!("{} is not configured — nothing to clear", row.key)));
        return;
    }
    if !ctx.writable_now() {
        return;
    }
    let ctx2 = ctx.clone();
    let key = row.key.clone();
    confirm_danger(
        cx,
        ctx.ui,
        format!("Clear the route {key}? The engine default applies again."),
        "Clear it",
        "Keep it",
        move || {
            let spec = writes::clear_route(&key, ctx2.write_base(), None);
            ctx2.send(Cmd::Write(Box::new(spec)));
        },
    );
}

/// The route editor: provider (Select with placeholder), model (text +
/// m picker over live discovery), base URL, options as `k=v` pairs.
fn open_route_editor(cx: Scope, ctx: &Ctx, row: RouteRow) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    // Provider choices: the 10 static + enabled endpoint profiles.
    let mut providers: Vec<String> = schema::STATIC_PROVIDERS
        .iter()
        .map(|s| s.to_string())
        .collect();
    if let Loadable::Ready(p) = ctx.store.profiles.get_untracked() {
        for prof in &p.profiles {
            if prof.enabled {
                providers.push(prof.virtual_provider());
            }
        }
    }
    // The drift base is captured at editor OPEN — a base read at
    // submit time would be silently re-armed by any mirror refresh
    // landing while the form is up (M2 review P3-2).
    let base = ctx.write_base();
    let providers = std::rc::Rc::new(providers);
    // A prefilled provider kicks discovery at OPEN: the model picker
    // must be populated when the operator reaches it, not only after
    // re-committing the provider they already had.
    if let Some(p) = row.provider.as_deref() {
        if providers.iter().any(|q| q == p) {
            super::model_field::kick_discovery(ctx, p);
        }
    }
    open_form_guarded(ctx, cx, Size::new(84, 19), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let mut popts = vec![SelectOption::new("— engine decides —")];
        let mut pinitial = 0usize;
        for (i, p) in providers.iter().enumerate() {
            popts.push(SelectOption::new(p.clone()));
            if Some(p.as_str()) == row.provider.as_deref() {
                pinitial = i + 1;
            }
        }
        let provider_sel = mcx.signal(pinitial);
        let model = mcx.signal(row.model.clone().unwrap_or_default());
        let base_url = mcx.signal(row.base_url.clone().unwrap_or_default());
        let reasoning = mcx.signal(row.reasoning.clone().unwrap_or_default());
        let options = mcx.signal(
            row.options
                .as_ref()
                .and_then(|o| o.as_object())
                .map(render_options)
                .unwrap_or_default(),
        );
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        // The dirty set must track EVERY editable control — a form
        // dirty only in the select used to discard on the first Esc
        // (M2 review P3-10).
        let p0 = pinitial;
        let model_init = model.get_untracked();
        let url_init = base_url.get_untracked();
        let reasoning_init = reasoning.get_untracked();
        let options_init = options.get_untracked();
        super::forms::install_dirty_guard_with(
            mcx,
            &guard,
            move || {
                provider_sel.get_untracked() != p0
                    || model.with_untracked(|v| v != &model_init)
                    || base_url.with_untracked(|v| v != &url_init)
                    || reasoning.with_untracked(|v| v != &reasoning_init)
                    || options.with_untracked(|v| v != &options_init)
            },
            move || {
                let _ = provider_sel.get();
                let _ = model.get();
                let _ = base_url.get();
                let _ = reasoning.get();
                let _ = options.get();
            },
            esc_armed,
            form_error,
        );
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        // Provider commit kicks model discovery (same lane the pair
        // editor uses).
        let providers2 = providers.clone();
        let ctx_models = ctx2.clone();
        let on_provider = move |i: usize| {
            if i > 0 {
                super::model_field::kick_discovery(&ctx_models, &providers2[i - 1]);
            }
        };

        let key = row.key.clone();
        let providers3 = providers.clone();
        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let i = provider_sel.get_untracked();
            let provider = (i > 0).then(|| providers3[i - 1].clone());
            let m = model.get_untracked().trim().to_string();
            let u = base_url.get_untracked().trim().to_string();
            let r = reasoning.get_untracked().trim().to_string();
            let opts = match parse_options(&options.get_untracked()) {
                Ok(o) => o,
                Err(e) => {
                    form_error.set(Some(e));
                    return;
                }
            };
            if provider.is_none() && m.is_empty() && u.is_empty() && r.is_empty() && opts.is_empty()
            {
                form_error.set(Some(
                    "nothing to set — pick a provider/model, or x clears the route".into(),
                ));
                return;
            }
            let spec = writes::set_route(
                &key,
                provider.as_deref(),
                (!m.is_empty()).then_some(m.as_str()),
                (!u.is_empty()).then_some(u.as_str()),
                (!r.is_empty()).then_some(r.as_str()),
                &opts,
                base,
                Some(form_id),
            );
            in_flight.set(true);
            ctx3.send(Cmd::Write(Box::new(spec)));
        };
        let submit_btn = submit.clone();

        // The model row: discovery-backed picker filtered by the
        // route's MODALITY (embedding.* → embedding models; the rest
        // → generative), free-typing fallback preserved.
        let mf = super::model_field::model_field(
            mcx,
            theme,
            super::model_field::ModelFieldSpec {
                store: ctx2.store,
                providers: providers.clone(),
                provider_sel,
                model_text: model,
                class: crate::models::class_for_modality(&row.modality),
                submit: None,
            },
        );
        let mf_handle = mf.handle.clone();
        let mf_custom = mf.custom;

        let hint = row
            .option_examples
            .as_ref()
            .map(|e| format!("option examples: {e}"))
            .unwrap_or_default();
        Block::new()
            .title(format!("Route — {} ({})", row.label, row.key))
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(0))
                    .shortcut(KeyChord::plain(Key::Char('m')), {
                        let ctx_m = ctx2.clone();
                        move |_| {
                            if !mf_handle.open() {
                                ctx_m.store.notice.set(Some(
                                    "no picker to open — pick a provider first (or Ctrl+P \
                                     leaves custom typing)"
                                        .into(),
                                ));
                            }
                        }
                    })
                    .shortcut(KeyChord::plain(Key::Char('c')), move |_| {
                        mf_custom.set(Some(true));
                    })
                    .shortcut(KeyChord::new(Mods::CTRL, Key::Char('p')), move |_| {
                        mf_custom.set(Some(false));
                    })
                    .child(field(
                        &t,
                        "provider",
                        Select::new(popts)
                            .value(provider_sel)
                            .on_change(on_provider)
                            .view(mcx),
                    ))
                    .child(mf.view)
                    .child(field(
                        &t,
                        "base URL",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(base_url)
                            .placeholder("optional — http(s)://…")
                            .view(mcx),
                    ))
                    .child(field(
                        &t,
                        "reasoning",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(reasoning)
                            .placeholder("optional — effort hint (low/medium/high)")
                            .view(mcx),
                    ))
                    .child(field(
                        &t,
                        "options",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(options)
                            .placeholder("k=v pairs, space-separated (voice=M2, k=\"a b\")")
                            .view(mcx),
                    ))
                    .child(line(vec![span(format!(" {hint}"), t.text_faint)]))
                    .child(message_slot(theme, form_error, in_flight))
                    .child(
                        Element::new()
                            .style(LayoutStyle::row().gap(2).shrink(0.0))
                            .child(Button::new("Save").on_click(submit_btn).view(mcx))
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

/// Render stored options for the edit buffer. Values containing
/// whitespace are double-quoted (so they re-parse), and STRING values
/// that would JSON-parse as scalars ("true", "42") are quoted too —
/// the CLI JSON-parses option values, so an unquoted round trip would
/// silently flip their type (M2 review P3-6).
fn render_options(o: &serde_json::Map<String, serde_json::Value>) -> String {
    o.iter()
        .map(|(k, v)| {
            let vs = match v {
                serde_json::Value::String(s) => {
                    let ambiguous = serde_json::from_str::<serde_json::Value>(s)
                        .map(|p| !p.is_string())
                        .unwrap_or(false);
                    if s.contains(char::is_whitespace) || ambiguous {
                        format!("\"{s}\"")
                    } else {
                        s.clone()
                    }
                }
                other => other.to_string(),
            };
            format!("{k}={vs}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse the edit buffer back to k=v pairs. Double quotes group a
/// value containing spaces AND force string typing downstream (the
/// quotes travel in argv; Python's json.loads reads them as a string).
pub fn parse_options(raw: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in raw.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                cur.push(ch);
            }
            c if c.is_whitespace() && !in_quotes => {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if in_quotes {
        return Err("unterminated quote in options".into());
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    for tok in tokens {
        match tok.split_once('=') {
            Some((k, v)) if !k.is_empty() => out.push((k.to_string(), v.to_string())),
            _ => {
                return Err(format!(
                    "options are space-separated k=v pairs (quote spaced values: k=\"a b\") — \
                     \"{tok}\" is not"
                ))
            }
        }
    }
    Ok(out)
}

/// Route options are the one route field with free-form operator
/// content (unknown scalars fold into it, capability_defaults.py:246-287)
/// — a key-shaped option value must not echo on screen (review P3-11).
fn mask_options(options: &serde_json::Value) -> String {
    let Some(obj) = options.as_object() else {
        return options.to_string();
    };
    let parts: Vec<String> = obj
        .iter()
        .map(|(k, v)| {
            let kl = k.to_ascii_lowercase();
            if kl.contains("key") || kl.contains("token") || kl.contains("secret") {
                format!("{k}=«redacted»")
            } else {
                format!("{k}={v}")
            }
        })
        .collect();
    format!("{{{}}}", parts.join(", "))
}

fn routes_table(cx: Scope, t: &TokenSet, data: &RoutesData, sel: Signal<usize>) -> View {
    // Width-aware columns: the model column is the payload — it keeps
    // real width at every breakpoint; source drops first, provider
    // second (the sibling's 0900-class recipe).
    let w = abstracttui::app::use_viewport(cx).get().w;
    let rows: Vec<Vec<String>> = data
        .rows
        .iter()
        .map(|r| {
            let state = if r.key == "output.text" {
                "= input.text".to_string()
            } else if let Some(by) = &r.covered_by {
                format!("covered by {by}")
            } else if r.configured {
                "configured".to_string()
            } else {
                "default".to_string()
            };
            let lock = if !r.editable() { " 🔒" } else { "" };
            let mut row = vec![format!("{}{}", r.key, lock), state];
            if w >= 96 {
                row.push(or_dash(&r.provider));
            }
            row.push(ellipsize(&or_dash(&r.model), 40));
            row
        })
        .collect();
    let mut cols = vec![
        Column::new("route", ColWidth::Cells(30)),
        Column::new("state", ColWidth::Cells(22)),
    ];
    if w >= 96 {
        cols.push(Column::new("provider", ColWidth::Cells(14)));
    }
    cols.push(Column::new("model", ColWidth::Flex(1.0)));
    Table::new(cols)
        .rows(rows)
        .selection(sel)
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .autofocus()
        .build()
}

#[cfg(test)]
mod tests {
    use super::{parse_options, render_options};
    use serde_json::json;

    /// The options round trip (M2 review P3-6): spaced values quote
    /// and re-parse; ambiguous strings keep their type via quoting.
    #[test]
    fn options_round_trip_spaces_and_types() {
        let stored = json!({
            "voice": "M2",
            "prompt": "hello world",
            "flag": true,
            "stringy": "true",
            "n": 42
        });
        let rendered = render_options(stored.as_object().unwrap());
        assert!(rendered.contains("voice=M2"), "{rendered}");
        assert!(rendered.contains("prompt=\"hello world\""), "{rendered}");
        assert!(rendered.contains("flag=true"), "{rendered}");
        assert!(
            rendered.contains("stringy=\"true\""),
            "string-typed 'true' quotes so a re-save keeps it a string: {rendered}"
        );
        let parsed = parse_options(&rendered).unwrap();
        assert!(parsed
            .iter()
            .any(|(k, v)| k == "prompt" && v == "\"hello world\""));
        assert!(parsed.iter().any(|(k, v)| k == "stringy" && v == "\"true\""));

        assert!(parse_options("k=\"unterminated").is_err());
        let err = parse_options("novalue").unwrap_err();
        assert!(err.contains("k=v"), "{err}");
        assert!(parse_options("").unwrap().is_empty());
    }
}
