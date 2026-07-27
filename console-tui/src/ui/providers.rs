//! Providers & keys: the api_keys table (k sets/clears the selected
//! key through the masked editor) and the provider endpoint profiles
//! (a adds, e edits, d deletes — all through `config set-provider`/
//! `delete-provider`, verified by re-read).

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Button, Checkbox, ColWidth, Column, Table};

use crate::config::FieldState;
use crate::schema;
use crate::store::{ProfileRow, ProfilesData};
use crate::worker::{next_form_id, Cmd};
use crate::writes;

use super::forms::{
    confirm_danger, install_write_done, message_slot, open_form_guarded,
};
use super::util::{ellipsize, field, line, loadable_view, span};
use super::Ctx;

pub fn view(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui;
    let ctx_key = ctx.clone();
    let ctx_add = ctx.clone();
    let ctx_edit = ctx.clone();
    let ctx_del = ctx.clone();

    // Two halves, both grow(1.0) with zero basis: they split whatever
    // height exists instead of one half's content starving the other
    // to zero rows.
    let keys = dyn_view_scoped(
        LayoutStyle::default().grow(1.0).basis(Dimension::Cells(0)),
        move |gcx| {
            let t = theme.get().tokens;
            let ctx_tbl = ctx_key.clone();
            let body = super::sections::with_snapshot(&t, &store, |snap, _| {
                let Some(sv) = snap.sections.iter().find(|s| s.spec.name == "api_keys") else {
                    return line(vec![span("api_keys section missing?!", t.error)]);
                };
                let rows: Vec<Vec<String>> = sv
                    .fields
                    .iter()
                    .map(|f| {
                        let state = match &f.state {
                            FieldState::Default => "not set".to_string(),
                            FieldState::Set => f.display.clone(),
                            FieldState::Broken(r) => format!("✗ {}", ellipsize(r, 36)),
                        };
                        vec![
                            f.key.to_string(),
                            state,
                            f.note.clone().unwrap_or_default(),
                        ]
                    })
                    .collect();
                let names: Vec<&'static str> = sv.fields.iter().map(|f| f.key).collect();
                let ctx_act = ctx_tbl.clone();
                Element::new()
                    .style(LayoutStyle::column())
                    .child(
                        Table::new(vec![
                            Column::new("provider key", ColWidth::Cells(20)),
                            Column::new("state", ColWidth::Cells(26)),
                            Column::new("note", ColWidth::Flex(1.0)),
                        ])
                        .rows(rows)
                        .selection(ui.keys_sel)
                        .on_activate(move |i| {
                            if let Some(name) = names.get(i) {
                                if ctx_act.writable_now() {
                                    super::editors::open_field_editor(
                                        gcx, &ctx_act, "api_keys", name,
                                    );
                                }
                            }
                        })
                        .layout(LayoutStyle::default().grow(1.0))
                        .element(gcx, &t)
                        .autofocus()
                        .build(),
                    )
                    .child(line(vec![span(
                        " keys stored HERE override the environment (a differing env var is shadowed at load)",
                        t.text_faint,
                    )]))
                    .build()
            });
            Block::new()
                .title("API keys · api_keys — Enter/k edits the selected key")
                .layout(LayoutStyle::column().grow(1.0))
                .child(body)
                .element(&t)
                .build()
        },
    );

    let profiles = dyn_view_scoped(
        LayoutStyle::default().grow(1.0).basis(Dimension::Cells(0)),
        move |gcx| {
            let t = theme.get().tokens;
            let data = store.profiles.get();
            let body = loadable_view(
                &t,
                &data,
                |d: &ProfilesData| d.profiles.is_empty(),
                "no provider endpoint profiles — a adds one (endpoint:<id> providers)",
                |d| profiles_table(gcx, &t, d, ui.profile_sel),
            );
            Block::new()
                .title("Provider endpoint profiles · provider_profiles — a add · e edit · d delete")
                .layout(LayoutStyle::column().grow(1.0))
                .child(body)
                .element(&t)
                .build()
        },
    );

    let ctx_k = ctx.clone();
    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .shortcut(KeyChord::plain(Key::Char('k')), move |_| {
            let idx = ui.keys_sel.get_untracked();
            let name = schema::section("api_keys")
                .and_then(|s| s.fields.get(idx))
                .map(|f| f.key);
            match name {
                Some(name) => {
                    if ctx_k.writable_now() {
                        super::editors::open_field_editor(cx, &ctx_k, "api_keys", name);
                    }
                }
                None => ctx_k.store.notice.set(Some("no key selected".into())),
            }
        })
        .shortcut(KeyChord::plain(Key::Char('a')), move |_| {
            if ctx_add.writable_now() {
                open_profile_editor(cx, &ctx_add, None);
            }
        })
        .shortcut(KeyChord::plain(Key::Char('e')), move |_| {
            match selected_profile(&ctx_edit) {
                Some(p) => {
                    if ctx_edit.writable_now() {
                        open_profile_editor(cx, &ctx_edit, Some(p));
                    }
                }
                None => ctx_edit
                    .store
                    .notice
                    .set(Some("no profile selected — a adds one".into())),
            }
        })
        .shortcut(KeyChord::plain(Key::Char('t')), {
            let ctx_t = ctx.clone();
            move |_| open_test_picker(cx, &ctx_t)
        })
        .shortcut(KeyChord::plain(Key::Char('d')), move |_| {
            match selected_profile(&ctx_del) {
                Some(p) => {
                    if !ctx_del.writable_now() {
                        return;
                    }
                    let ctx2 = ctx_del.clone();
                    confirm_danger(
                        cx,
                        ctx_del.ui,
                        format!(
                            "Delete profile {} (endpoint:{})? Its stored key goes with it.",
                            p.id, p.id
                        ),
                        "Delete it",
                        "Keep it",
                        move || {
                            let spec =
                                writes::delete_profile(&p.id, ctx2.write_base(), None);
                            ctx2.send(Cmd::Write(Box::new(spec)));
                        },
                    );
                }
                None => ctx_del
                    .store
                    .notice
                    .set(Some("no profile selected — nothing to delete".into())),
            }
        })
        .child(keys)
        .child(profiles)
        .build()
}

fn selected_profile(ctx: &Ctx) -> Option<ProfileRow> {
    let idx = ctx.ui.profile_sel.get_untracked();
    ctx.store
        .profiles
        .with_untracked(|d| d.ready().and_then(|d| d.profiles.get(idx).cloned()))
}

/// The provider test picker (M3): one keyboard-driven prompt over ALL
/// canonical providers + every endpoint profile — the api_keys table
/// only lists KEYED providers, so a selected-row-only verb could never
/// test lmstudio/ollama, the wizard's own recommended targets. Initial
/// selection follows the selected key row when it maps to a provider
/// id (api_keys field names spell `openai_compatible` with an
/// underscore; provider ids use a hyphen).
pub fn open_test_picker(cx: Scope, ctx: &Ctx) {
    if ctx.store.probe_busy.get_untracked() {
        // Refuse at the door, not after the pick — send_probe's guard
        // stays as the backstop.
        ctx.store.notice.set(Some(
            "a test is already running — its result lands in the journal and on Review (8)".into(),
        ));
        return;
    }
    let mut prompt = abstracttui::app::ChoicePrompt::new(
        "Test which provider? (live model discovery via config test-provider)",
    );
    for p in schema::STATIC_PROVIDERS {
        prompt = prompt.option(*p, *p);
    }
    // Profiles ride the same picker with their base_url for the
    // reachability disambiguation.
    let profiles: Vec<ProfileRow> = ctx
        .store
        .profiles
        .with_untracked(|d| d.ready().map(|d| d.profiles.clone()).unwrap_or_default());
    for p in &profiles {
        let id = format!("endpoint:{}", p.id);
        prompt = prompt.option(id.clone(), format!("{id} ({})", ellipsize(&p.base_url, 32)));
    }
    let selected_key_provider = {
        let idx = ctx.ui.keys_sel.get_untracked();
        schema::section("api_keys")
            .and_then(|s| s.fields.get(idx))
            .map(|f| f.key.replace('_', "-"))
            .filter(|p| schema::STATIC_PROVIDERS.contains(&p.as_str()))
    };
    if let Some(p) = &selected_key_provider {
        prompt = prompt.initial(p.clone());
    }
    let ctx2 = ctx.clone();
    super::forms::open_prompt(cx, ctx.ui, prompt, move |outcome| {
        if let abstracttui::app::ChoiceOutcome::Answered(a) = outcome {
            let Some(choice) = a.selected.first() else {
                return;
            };
            let base_url = choice
                .strip_prefix("endpoint:")
                .and_then(|id| profiles.iter().find(|p| p.id == id))
                .map(|p| p.base_url.clone());
            ctx2.send_probe(crate::probes::list_models(choice, base_url.as_deref()));
        }
    });
}

fn profiles_table(cx: Scope, t: &TokenSet, d: &ProfilesData, sel: Signal<usize>) -> View {
    let w = abstracttui::app::use_viewport(cx).get().w;
    let rows: Vec<Vec<String>> = d
        .profiles
        .iter()
        .map(|p| {
            // api_key_set reflects the RESOLVED key (env var value if
            // set, else the stored key): a `$VAR` reference that
            // resolves to nothing must not read as configured.
            let key = if let Some(env) = &p.api_key_env_var {
                if p.api_key_set {
                    format!("${env} ✓")
                } else {
                    format!("${env} — EMPTY")
                }
            } else if p.api_key_set {
                match &p.api_key_fingerprint {
                    Some(f) => format!("fp {f}"),
                    None => "set".to_string(),
                }
            } else {
                "not set".to_string()
            };
            let models = if p.allowed_models.is_empty() {
                "live discovery".to_string()
            } else {
                format!("{} pinned", p.allowed_models.len())
            };
            let mut row = vec![p.id.clone(), p.family.clone()];
            row.push(ellipsize(&p.base_url, 40));
            row.push(key);
            if w >= 100 {
                row.push(models);
            }
            row.push(if p.enabled { "yes".into() } else { "NO".into() });
            row
        })
        .collect();
    let mut cols = vec![
        Column::new("id", ColWidth::Cells(16)),
        Column::new("family", ColWidth::Cells(18)),
        Column::new("base URL", ColWidth::Flex(1.0)),
        Column::new("API key", ColWidth::Cells(16)),
    ];
    if w >= 100 {
        cols.push(Column::new("models", ColWidth::Cells(14)));
    }
    cols.push(Column::new("enabled", ColWidth::Cells(7)));
    Table::new(cols)
        .rows(rows)
        .selection(sel)
        .layout(LayoutStyle::default().grow(1.0))
        .element(cx, t)
        .build()
}

/// The profile editor: create (id editable) or edit (id fixed).
/// Secrets: masked, blank keeps, explicit clear; the `$VAR` reference
/// form rides the same field (`--api-key '$VAR'` stores the NAME).
pub fn open_profile_editor(cx: Scope, ctx: &Ctx, existing: Option<ProfileRow>) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    // Drift base captured at OPEN (M2 review P3-2).
    let base = ctx.write_base();
    open_form_guarded(ctx, cx, Size::new(84, 19), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let is_new = existing.is_none();
        let p = existing.clone();
        let id = mcx.signal(p.as_ref().map(|p| p.id.clone()).unwrap_or_default());
        let family0 = p.as_ref().map(|p| p.family.clone()).unwrap_or_default();
        let base_url = mcx.signal(p.as_ref().map(|p| p.base_url.clone()).unwrap_or_default());
        let name = mcx.signal(p.as_ref().map(|p| p.display_name.clone()).unwrap_or_default());
        let desc = mcx.signal(p.as_ref().map(|p| p.description.clone()).unwrap_or_default());
        let key_input = mcx.signal(String::new());
        let clear_key = mcx.signal(false);
        let enabled = mcx.signal(p.as_ref().map(|p| p.enabled).unwrap_or(true));
        let key_now = p
            .as_ref()
            .map(|p| {
                if let Some(env) = &p.api_key_env_var {
                    format!("${env}")
                } else if p.api_key_set {
                    format!(
                        "stored (fp {})",
                        p.api_key_fingerprint.clone().unwrap_or_default()
                    )
                } else {
                    "not set".into()
                }
            })
            .unwrap_or_else(|| "not set".into());

        // Family select with placeholder — never fabricate.
        let mut fopts = vec![SelectOption::new("— choose a family —")];
        let mut finitial = 0usize;
        for (i, fam) in schema::PROFILE_FAMILIES.iter().enumerate() {
            fopts.push(SelectOption::new(*fam));
            if *fam == family0 {
                finitial = i + 1;
            }
        }
        let family_sel = mcx.signal(finitial);

        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        // The dirty set tracks EVERY editable control — a form dirty
        // only in the select/checkboxes used to discard on the first
        // Esc (M2 review P3-10).
        {
            let id0 = id.get_untracked();
            let url0 = base_url.get_untracked();
            let name0 = name.get_untracked();
            let desc0 = desc.get_untracked();
            let fam0 = finitial;
            let enabled0 = enabled.get_untracked();
            super::forms::install_dirty_guard_with(
                mcx,
                &guard,
                move || {
                    id.with_untracked(|v| v != &id0)
                        || base_url.with_untracked(|v| v != &url0)
                        || name.with_untracked(|v| v != &name0)
                        || desc.with_untracked(|v| v != &desc0)
                        || key_input.with_untracked(|v| !v.is_empty())
                        || family_sel.get_untracked() != fam0
                        || enabled.get_untracked() != enabled0
                        || clear_key.get_untracked()
                },
                move || {
                    let _ = id.get();
                    let _ = base_url.get();
                    let _ = name.get();
                    let _ = desc.get();
                    let _ = key_input.get();
                    let _ = family_sel.get();
                    let _ = enabled.get();
                    let _ = clear_key.get();
                },
                esc_armed,
                form_error,
            );
        }
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let idv = id.get_untracked().trim().to_string();
            if idv.is_empty() {
                form_error.set(Some("the profile id is required".into()));
                return;
            }
            let fi = family_sel.get_untracked();
            if fi == 0 {
                form_error.set(Some("choose a provider family".into()));
                return;
            }
            let family = schema::PROFILE_FAMILIES[fi - 1];
            let url = base_url.get_untracked().trim().to_string();
            let url_ok =
                url.is_empty() || url.starts_with("http://") || url.starts_with("https://");
            if !url_ok {
                form_error.set(Some("base URL must start with http:// or https://".into()));
                return;
            }
            let typed_key = key_input.get_untracked().trim().to_string();
            if typed_key.starts_with('-') {
                // argparse would read it as a flag and die with a
                // cryptic "expected one argument" (M2 review P3-5).
                form_error.set(Some(
                    "keys starting with '-' cannot pass the abstractcore flags CLI".into(),
                ));
                return;
            }
            let spec = writes::save_profile(
                &idv,
                family,
                &url,
                (!typed_key.is_empty()).then_some(typed_key.as_str()),
                clear_key.get_untracked(),
                name.get_untracked().trim(),
                desc.get_untracked().trim(),
                enabled.get_untracked(),
                base,
                Some(form_id),
            );
            in_flight.set(true);
            ctx3.send(Cmd::Write(Box::new(spec)));
        };
        let submit_btn = submit.clone();

        Block::new()
            .title(if is_new {
                "Add provider endpoint profile".to_string()
            } else {
                format!("Edit profile {}", id.get_untracked())
            })
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(0))
                    .child(field(
                        &t,
                        "id",
                        if is_new {
                            TextInput::new()
                                .layout(LayoutStyle::default().grow(1.0).h(1))
                                .value(id)
                                .view(mcx)
                        } else {
                            // Ids are identities — renaming would mint
                            // a second profile.
                            dyn_view(LayoutStyle::line(1), move || {
                                line(vec![span(id.get(), t.text_muted)])
                            })
                        },
                    ))
                    .child(field(
                        &t,
                        "family",
                        Select::new(fopts).value(family_sel).view(mcx),
                    ))
                    .child(field(
                        &t,
                        "base URL",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(base_url)
                            .placeholder("https://host/v1")
                            .view(mcx),
                    ))
                    .child(field(
                        &t,
                        "API key",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(key_input)
                            .masked(true)
                            .placeholder("blank keeps · $VAR stores an env reference")
                            .view(mcx),
                    ))
                    .child(line(vec![
                        span("                    stored now: ", t.text_faint),
                        span(key_now.clone(), t.text_muted),
                    ]))
                    .child(field(
                        &t,
                        "",
                        Checkbox::new("clear the stored key").checked(clear_key).view(mcx),
                    ))
                    .child(field(
                        &t,
                        "display name",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(name)
                            .view(mcx),
                    ))
                    .child(field(
                        &t,
                        "description",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(desc)
                            .view(mcx),
                    ))
                    .child(field(
                        &t,
                        "",
                        Checkbox::new("enabled").checked(enabled).view(mcx),
                    ))
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
