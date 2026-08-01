//! Providers: ONE unified provider list — the AbstractGateway console-
//! TUI's Providers screen, cell for cell (operator ruling 2026-08-01:
//! "I do not understand why the providers are displayed in a different
//! fashion between gateway and core; they should have the exact same.
//! Gateway is the one we want. Profiles are just indicated as profile
//! of the openai-compatible endpoint, and we should have a way to
//! configure as many as necessary, like in the gateway console.").
//!
//! THIS SCREEN USED TO BE TWO TABLES. The top one enumerated the
//! provider registry with its own vocabulary (`kind` / `key/endpoint` /
//! `answering`, "local server", "nothing to configure"); a second table
//! underneath listed the endpoint profiles again — the same objects,
//! twice, in two spellings. The gateway has ONE table, so this does
//! too: `provider | family | base URL | API key | models | enabled |
//! origin`, with every stored profile INLINE as its `endpoint:<id>` row
//! (`store::ProfilesData::connections` composes it), and the profile
//! editing that used to justify the second table moved onto those rows.
//!
//! Where core differs from the gateway it says so instead of faking a
//! cell:
//!   - `enabled` is `—` on a registry row: core has no enable switch
//!     for a builtin provider, and `yes` beside a column whose other
//!     value is `NO` would advertise a toggle with nothing to write to.
//!   - `origin` reads `config | env | auto | registry` — core has one
//!     store and no scopes, and `registry` (known, nothing configured)
//!     is a row the gateway does not have at all.
//!
//! Verbs, per row, the gateway's set: `a` adds a connection (a real
//! `provider_profiles` entry, as many as wanted), `e` configures the
//! selected row, `d` deletes a profile, `m` lists models, `t` probes it.
//! `k` stays the explicit key verb for the rows that have an `api_keys`
//! field. Every refusal SAYS why — a footer-advertised key that does
//! nothing reads as a dead app.

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Button, Checkbox, Scroll, Table};

use crate::config::FileState;
use crate::schema;
use crate::store::{ConnectionRow, Loadable, Origin, ProfileRow, ProfilesData};
use crate::worker::{next_form_id, Cmd};
use crate::writes;

use super::forms::{confirm_danger, install_write_done, message_slot, open_form_guarded};
use super::util::{error_panel, field, line, loadable_view, span, span_bold};
use super::widths;
use super::Ctx;

pub fn view(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui;

    // Deleting the last row must not strand the highlight past the end.
    clamp_selection(cx, ui.profile_sel, move || {
        store
            .profiles
            .with(|d| d.ready().map(|d| d.connections().len()).unwrap_or(0))
    });

    let ctx_add = ctx.clone();
    let ctx_edit = ctx.clone();
    let ctx_del = ctx.clone();
    let ctx_models = ctx.clone();
    let ctx_test = ctx.clone();
    let ctx_key = ctx.clone();
    let ctx_table = ctx.clone();

    Element::new()
        .style(LayoutStyle::column().grow(1.0))
        .shortcut(KeyChord::plain(Key::Char('a')), move |_| {
            if ctx_add.writable_now() {
                open_profile_editor(cx, &ctx_add, None);
            }
        })
        .shortcut(KeyChord::plain(Key::Char('e')), move |_| {
            configure_selected(cx, &ctx_edit);
        })
        .shortcut(KeyChord::plain(Key::Char('k')), move |_| {
            edit_selected_key(cx, &ctx_key);
        })
        .shortcut(KeyChord::plain(Key::Char('d')), move |_| {
            delete_selected(cx, &ctx_del);
        })
        .shortcut(KeyChord::plain(Key::Char('m')), move |_| {
            match selected(&ctx_models) {
                Some(row) => open_models_modal(cx, &ctx_models, row.provider),
                None => ctx_models
                    .store
                    .notice
                    .set(Some("no provider selected — no models to browse".into())),
            }
        })
        .shortcut(KeyChord::plain(Key::Char('t')), move |_| {
            match selected(&ctx_test) {
                Some(row) => {
                    // The unified list is why this verb can be
                    // selected-row now: the old screen's top table was
                    // the api_keys section, so a row-scoped test could
                    // never reach keyless lmstudio/ollama — the
                    // wizard's own recommended targets. Every provider
                    // has a row here.
                    let base_url = (!row.base_url.is_empty()).then_some(row.base_url.as_str());
                    ctx_test.send_probe(crate::probes::list_models(&row.provider, base_url));
                }
                None => ctx_test
                    .store
                    .notice
                    .set(Some("no provider selected — nothing to test".into())),
            }
        })
        .child(dyn_view_scoped(
            LayoutStyle::default().grow(1.0),
            move |gcx| {
                let t = theme.get().tokens;
                let ctx_tbl = ctx_table.clone();
                let data = store.profiles.get();
                let body = loadable_view(
                    &t,
                    &data,
                    |d: &ProfilesData| d.connections().is_empty(),
                    "the CLI reported no providers — is `abstractcore` on PATH?",
                    |d| unified_table(gcx, cx, &ctx_tbl, &t, d, ui.profile_sel),
                );
                Block::new()
                    .title("Available providers (a adds a connection)")
                    .layout(LayoutStyle::column().grow(1.0))
                    .child(body)
                    // Selected-row action honesty — the TUI stand-in
                    // for the web's per-row buttons.
                    .child(selection_hint(&t, &data, ui.profile_sel.get()))
                    // The facts under the one list: what answers when
                    // nothing names a provider, and what is known but
                    // unconfigured.
                    .child(defaults_footer(&t, ctx_tbl.store))
                    .element(&t)
                    .build()
            },
        ))
        .build()
}

/// Keep a selection inside the list it points at (the gateway console's
/// `util::clamp_selection`, same body): deleting the last row must not
/// strand the highlight past the end.
fn clamp_selection(cx: Scope, sel: Signal<usize>, len_of: impl Fn() -> usize + 'static) {
    cx.effect(move || {
        let len = len_of();
        let cur = sel.get();
        if len == 0 {
            if cur != 0 {
                sel.set(0);
            }
        } else if cur >= len {
            sel.set(len - 1);
        }
    });
}

/// THE ONE provider table. Every row is a connection row — a builtin
/// the registry knows, a stored `endpoint:<id>` profile, or a local
/// server the probe found answering. The first column carries the
/// provider NAME the row answers to, because that string — not the
/// internal profile id — is what routes and flows reference.
fn unified_table(
    gcx: Scope,
    page_cx: Scope,
    ctx: &Ctx,
    t: &TokenSet,
    d: &ProfilesData,
    sel: Signal<usize>,
) -> View {
    // Which columns APPEAR is a breakpoint decision (narrow terminals
    // get fewer, honest columns instead of a silently amputated payload
    // column); how wide the survivors are is MEASURED from the rows by
    // `ui::widths`, so a wide terminal prints every base URL whole.
    let w = abstracttui::app::use_viewport(gcx).get().w;
    let wide = w >= 104;
    let mut rows: Vec<Vec<String>> = d
        .connections()
        .iter()
        .map(|c| {
            let mut row = vec![c.provider.clone()];
            if wide {
                row.push(c.family.clone());
            }
            // The FULL base URL: two endpoints on the same host differ
            // in their path, so a fixed cap printed one string for both.
            row.push(c.base_url_text());
            row.push(c.api_key.clone());
            if wide {
                row.push(c.models.clone());
            }
            row.push(c.enabled_text());
            row.push(c.origin.label().to_string());
            row
        })
        .collect();
    // Provider names and base URLs discriminate on their TAIL (`…/v1`
    // vs `…/v1/openai`); the rest print bounded phrases whose floor is
    // their widest word.
    let mut rules = vec![widths::ColRule::tail("provider", 16)];
    if wide {
        rules.push(widths::ColRule::head("family", 12));
    }
    rules.push(widths::ColRule::tail("base URL", 20));
    rules.push(widths::ColRule::head("API key", 14));
    if wide {
        rules.push(widths::ColRule::head("models", 8));
    }
    rules.push(widths::ColRule::head("enabled", 7));
    rules.push(widths::ColRule::head("origin", 8));
    // The grid lives inside a bordered Block, so the budget is the
    // viewport minus that border.
    let cols = widths::columns(&rules, &mut rows, w - widths::BLOCK_CHROME);
    let ctx_act = ctx.clone();
    Table::new(cols)
        .rows(rows)
        .selection(sel)
        // Enter / Space / double-click = the `e` verb. The modal opens
        // on the PAGE scope, never `gcx`: this dyn re-renders on every
        // config change, and a form parented here died when a reload
        // landed mid-edit.
        .on_activate(move |_| configure_selected(page_cx, &ctx_act))
        .layout(LayoutStyle::default().grow(1.0))
        .element(gcx, t)
        .autofocus()
        .build()
}

fn selected(ctx: &Ctx) -> Option<ConnectionRow> {
    let idx = ctx.ui.profile_sel.get_untracked();
    ctx.store
        .profiles
        .with_untracked(|d| d.ready().and_then(|d| d.connections().into_iter().nth(idx)))
}

/// The per-row action line — what THIS row supports and why (the web
/// shows per-row buttons; a TUI says it under the table).
fn selection_hint(t: &TokenSet, data: &Loadable<ProfilesData>, sel: usize) -> View {
    let Some(row) = data
        .ready()
        .and_then(|d| d.connections().into_iter().nth(sel))
    else {
        return line(vec![span(String::new(), t.text_faint)]);
    };
    let verbs = if row.is_profile() {
        "e edit · d delete · m models · t test"
    } else if row.takes_key() {
        "k/e set the key · m models · t test"
    } else {
        "no key to set · a adds an endpoint connection · m models · t test"
    };
    line(vec![
        span_bold(format!(" {}", row.provider), t.accent),
        span(format!(" — {} · {verbs}", row.origin_detail()), t.text_muted),
    ])
}

/// The facts under the one list, mirroring the gateway's discovery
/// footer: the configured default pair (what answers when nothing names
/// a provider) and the providers with no configuration yet. Degrades
/// independently of the table — an unread config never blanks the rows.
fn defaults_footer(t: &TokenSet, store: crate::store::Store) -> View {
    let default_line = match store.cfg.get() {
        Loadable::NotAsked => None,
        Loadable::Loading => Some(span("⟳ reading the config file…", t.info)),
        Loadable::Failed(e) => Some(span(
            format!(" core default unavailable — {}", e.headline()),
            t.warn,
        )),
        Loadable::Ready(m) => Some(match &m.state {
            FileState::Ready(snap) => {
                let get = |key: &str| {
                    snap.sections
                        .iter()
                        .find(|s| s.spec.name == "default_models")
                        .and_then(|s| s.fields.iter().find(|f| f.key == key))
                        .map(|f| f.display.clone())
                        .unwrap_or_else(|| "—".into())
                };
                let (p, m) = (get("global_provider"), get("global_model"));
                if p == "—" && m == "—" {
                    span(
                        " core default: none set (screen 2 sets the global pair)".to_string(),
                        t.text_muted,
                    )
                } else {
                    span(format!(" core default: {p} / {m}"), t.text_muted)
                }
            }
            FileState::Missing => span(
                " core default: no config file yet — nothing configured".to_string(),
                t.text_muted,
            ),
            _ => span(
                " core default unavailable — the config file is unreadable".to_string(),
                t.warn,
            ),
        }),
    };
    let free: Vec<String> = store.profiles.with(|d| {
        d.ready()
            .map(|d| {
                d.connections()
                    .iter()
                    .filter(|c| c.origin == Origin::Registry)
                    .map(|c| c.provider.clone())
                    .collect()
            })
            .unwrap_or_default()
    });
    let mut col = Element::new().style(LayoutStyle::column().shrink(0.0));
    if let Some(l) = default_line {
        col = col.child(line(vec![l]));
    }
    if !free.is_empty() {
        // Affordance FIRST: the line right-truncates on narrow
        // terminals, and the teaching must survive over tail names.
        col = col.child(line(vec![span(
            format!(
                " not configured yet (k sets a key · a adds a connection): {}",
                free.join(", ")
            ),
            t.text_faint,
        )]));
    }
    col.build()
}

/// `e` and row activation — ONE body, so the refusals and the write
/// door can never differ between keyboard and mouse. A stored profile
/// opens the profile editor; a key-taking builtin opens the masked
/// `api_keys` field editor (that IS how a builtin is configured in
/// core); anything else refuses with the reason.
fn configure_selected(cx: Scope, ctx: &Ctx) {
    let Some(row) = selected(ctx) else {
        ctx.store
            .notice
            .set(Some("no provider selected — a adds a connection".into()));
        return;
    };
    if let Some(p) = row.profile {
        if ctx.writable_now() {
            open_profile_editor(cx, ctx, Some(p));
        }
        return;
    }
    open_key_editor(cx, ctx, &row);
}

/// `k` — the explicit key verb, on the rows that have an `api_keys`
/// field. A stored profile keeps its key in the profile, so `k` sends
/// the operator to the editor that owns it rather than to a field that
/// says nothing about this row.
fn edit_selected_key(cx: Scope, ctx: &Ctx) {
    let Some(row) = selected(ctx) else {
        ctx.store.notice.set(Some("no provider selected".into()));
        return;
    };
    if let Some(p) = row.profile {
        ctx.store.notice.set(Some(format!(
            "{} keeps its key in the profile — e edits it there",
            row.provider
        )));
        let _ = p;
        return;
    }
    open_key_editor(cx, ctx, &row);
}

/// The masked `api_keys` field editor for a builtin row — or the
/// refusal that names the provider and why. "takes no API key" is an
/// answer, and it is the answer the operator came to this screen for.
fn open_key_editor(cx: Scope, ctx: &Ctx, row: &ConnectionRow) {
    if !row.takes_key() {
        ctx.store.notice.set(Some(format!(
            "{} takes no API key — a adds an endpoint connection if you need a keyed one",
            row.provider
        )));
        return;
    }
    let field_name = schema::section("api_keys")
        .and_then(|s| s.fields.iter().find(|f| f.key == row.api_key_field))
        .map(|f| f.key);
    match field_name {
        Some(name) => {
            if ctx.writable_now() {
                super::editors::open_field_editor(cx, ctx, "api_keys", name);
            }
        }
        None => ctx.store.notice.set(Some(format!(
            "no api_keys field named {} — edit it on the Sections screen",
            row.api_key_field
        ))),
    }
}

/// `d` — only a stored profile can be deleted here. A registry row is
/// not ours to remove; the refusal says where it comes from instead of
/// swallowing the key.
fn delete_selected(cx: Scope, ctx: &Ctx) {
    let Some(row) = selected(ctx) else {
        ctx.store
            .notice
            .set(Some("no provider selected — nothing to delete".into()));
        return;
    };
    let Some(p) = row.profile.clone() else {
        // Affordance FIRST: a notice truncates at the right edge, and
        // the rule must survive over the row's name.
        let from = match row.origin {
            Origin::Registry => "the provider registry",
            Origin::Env => "the environment",
            Origin::Auto => "a local-server probe",
            Origin::Config => "this config's api_keys section",
        };
        ctx.store.notice.set(Some(format!(
            "only stored connections delete here — '{}' comes from {from} (a adds a connection)",
            row.provider
        )));
        return;
    };
    if !ctx.writable_now() {
        return;
    }
    let ctx2 = ctx.clone();
    confirm_danger(
        cx,
        ctx.ui,
        format!(
            "Delete provider connection '{}'? Routes and flows pointing at endpoint:{} \
             will stop resolving, and its stored key goes with it.",
            p.id, p.id
        ),
        "Delete it",
        "Keep it",
        move || {
            let spec = writes::delete_profile(&p.id, ctx2.write_base(), None);
            ctx2.send(Cmd::Write(Box::new(spec)));
        },
    );
}

/// `m` — the models drill-in for any row, by the provider NAME it
/// answers to (`config models <name> --json`, which knows both bare
/// providers and `endpoint:<id>` profiles).
pub fn open_models_modal(cx: Scope, ctx: &Ctx, provider: String) {
    let store = ctx.store;
    if store
        .models
        .with_untracked(|m| m.get(&provider).and_then(|l| l.ready()).is_none())
    {
        store.models.update(|m| {
            m.insert(provider.clone(), Loadable::Loading);
        });
        ctx.send(Cmd::LoadModels {
            provider: provider.clone(),
        });
    }
    let theme = use_theme(cx);
    open_form_guarded(ctx, cx, Size::new(70, 22), move |_mcx, close, _guard| {
        let p_title = provider.clone();
        let p_body = provider.clone();
        // This modal's only focusable (Close) mounts inside a dyn — the
        // engine's focus-init runs before it exists, so the content root
        // owns focus itself (engine finding 1000's recipe). Without it
        // Esc and Enter would land nowhere and the dialog would read as
        // frozen.
        Element::new()
            .style(LayoutStyle::column().gap(1))
            .focusable()
            .autofocus()
            .child(dyn_view(LayoutStyle::line(1), move || {
                let t = theme.get().tokens;
                line(vec![span_bold(format!("Models — {p_title}"), t.accent)])
            }))
            .child(dyn_view_scoped(
                LayoutStyle::default().grow(1.0),
                move |gcx| {
                    let t = theme.get().tokens;
                    let entry = store
                        .models
                        .with(|m| m.get(&p_body).cloned())
                        .unwrap_or(Loadable::NotAsked);
                    match entry {
                        Loadable::NotAsked | Loadable::Loading => {
                            line(vec![span("⟳ discovering models…", t.info)])
                        }
                        Loadable::Failed(e) => error_panel(&t, &e),
                        Loadable::Ready(models) if models.is_empty() => line(vec![span(
                            "∅ no models reported (endpoint offline, or nothing loaded)",
                            t.text_muted,
                        )]),
                        Loadable::Ready(models) => {
                            let count = models.len();
                            Element::new()
                                .style(LayoutStyle::column().grow(1.0))
                                .child(line(vec![span(format!("{count} models"), t.text_muted)]))
                                .child(
                                    Scroll::new(
                                        Element::new()
                                            .style(LayoutStyle::column())
                                            .children(
                                                models
                                                    .iter()
                                                    .map(|m| {
                                                        line(vec![span(format!("  {m}"), t.text)])
                                                    })
                                                    .collect::<Vec<_>>(),
                                            )
                                            .build(),
                                    )
                                    .view(gcx),
                                )
                                .build()
                        }
                    }
                },
            ))
            .child(dyn_view_scoped(
                LayoutStyle::default().h(1).shrink(0.0),
                move |gcx| {
                    let t = theme.get().tokens;
                    let close = close.clone();
                    Element::new()
                        .style(LayoutStyle::row().gap(2))
                        .child(
                            Button::new("Close (Esc)")
                                .on_click(move || close())
                                .element(gcx, &t)
                                .build(),
                        )
                        .build()
                },
            ))
            .build()
    });
}

/// The profile editor: create (id editable) or edit (id fixed) — the
/// `a` door and the `e` door onto an `endpoint:<id>` row. Writes go
/// through `config set-provider`, so a connection created here lands in
/// core's own `provider_profiles` section, as many as the operator
/// wants. Secrets: masked, blank keeps, explicit clear; the `$VAR`
/// reference form rides the same field (`--api-key '$VAR'` stores the
/// NAME).
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
        // Same vocabulary as the table above and as the gateway
        // console's key note — one way to say "there is a key here".
        let key_now = p
            .as_ref()
            .map(|p| match (&p.api_key_env_var, p.api_key_set) {
                (Some(env), true) => format!("stored (${env})"),
                (Some(env), false) => format!("none (${env})"),
                (None, true) => format!(
                    "stored ({})",
                    p.api_key_fingerprint.clone().unwrap_or_default()
                ),
                (None, false) => "none".into(),
            })
            .unwrap_or_else(|| "none".into());

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
                // The `a` verb's own words: this creates a CONNECTION —
                // an `endpoint:<id>` provider in provider_profiles.
                "Add a provider connection (endpoint:<id>)".to_string()
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
                                .placeholder("lowercase, digits, - _ (answers as endpoint:<id>)")
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
                        span("                    key now: ", t.text_faint),
                        span(key_now.clone(), t.text_muted),
                        span(
                            " — leave blank to keep it; type to replace; check clear to remove",
                            t.text_faint,
                        ),
                    ]))
                    .child(field(
                        &t,
                        "",
                        Checkbox::new("clear the stored key")
                            .checked(clear_key)
                            .view(mcx),
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
