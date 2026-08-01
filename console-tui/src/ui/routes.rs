//! Capability routes: the 24-route derived view from
//! `abstractcore config defaults --json` — coverage decorations,
//! the output.text alias, per-route provider/model/options — with the
//! route editor (e) and clear (x) writing through `config
//! set-default`/`clear-default`, verified against the fresh view.

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Button, Table};

use crate::schema;
use crate::store::{Loadable, RouteRow, RoutesData};
use crate::worker::{next_form_id, Cmd};
use crate::writes;

use super::forms::{
    confirm_danger, install_write_done, message_slot, open_form_guarded,
};
use super::util::{field, line, loadable_view, or_dash, span, span_bold};
use super::widths;
use super::Ctx;

pub fn view(cx: Scope, ctx: &Ctx, theme: Signal<&'static abstracttui::theme::Theme>) -> View {
    let store = ctx.store;
    let ui = ctx.ui;
    let ctx_edit = ctx.clone();
    let ctx_clear = ctx.clone();

    // The banner reads exactly like the gateway console's: the write
    // door first (writable / read-only), the grid's coverage second.
    // The alias teaching moved INTO the row that carries it ("derived
    // ← input.text") — a permanent banner sentence about one row was
    // noise on the other 23 and made every substring assertion lie.
    let banner = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        match store.routes.get() {
            Loadable::Ready(d) => {
                let mut spans = if d.writable {
                    vec![span(" writable", t.ok)]
                } else {
                    vec![span_bold(" read-only (config file refused?)", t.warn)]
                };
                spans.push(span(
                    format!("  ·  {} of {} configured", d.configured_count(), d.rows.len()),
                    t.text,
                ));
                // The other half of the weights banner below: that line
                // says whether the recommended MODELS are here, this verb
                // makes the ROUTES name them. It rides the always-present
                // banner because the weights line can fill with a missing
                // list and truncate anything appended to it.
                spans.push(span(
                    "  ·  a applies the recommended routes".to_string(),
                    t.text_faint,
                ));
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

    // WEIGHTS BANNER. A fresh install has the three recommended routes
    // configured and possibly none of their models on disk; the grid
    // alone reads as "all set". This line says how many can actually
    // run, names what is missing, and points at the one verb that fixes
    // it. Absent entirely while the probe has not answered — a blank is
    // honest, a zero would not be.
    //
    // THE MISSING LIST IS THE ELASTIC PART. It can name a dozen
    // artifacts; the verb after it is the only actionable text on the
    // row, and `line()` clips last-span-first — which is how the
    // operator saw `w downloads the selecte…`. So the verb (and the
    // count, and the `missing:` label) are reserved, the list gets what
    // is left, and it keeps its TAIL: the `@4bit`-style tag is what
    // distinguishes one absent artifact from another.
    let viewport = abstracttui::app::use_viewport(cx);
    let weights_banner = dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let avail = viewport.get().w;
        match store.availability.get() {
            Loadable::Ready(a) if a.total > 0 => {
                let head = format!(" recommended models: {} of {} present", a.installed, a.total);
                let unknown = if a.unknown > 0 {
                    format!("  ·  {} unknown", a.unknown)
                } else {
                    String::new()
                };
                let mut spans = vec![span_bold(
                    head.clone(),
                    if a.absent == 0 { t.ok } else { t.warn },
                )];
                if !unknown.is_empty() {
                    spans.push(span(unknown.clone(), t.text_muted));
                }
                if !a.missing.is_empty() {
                    const LABEL: &str = "  ·  missing: ";
                    const VERB: &str = "  ·  w downloads the selected route's weights";
                    let list = a
                        .missing
                        .iter()
                        .map(|(p, art)| format!("{p} {art}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let budget = widths::elastic_budget(&[&head, &unknown, LABEL, VERB], avail);
                    let count = format!("  ·  {} missing", a.missing.len());
                    // Three honest lines, widest first: name the artifacts,
                    // or count them, or say neither — but never at the
                    // verb's expense, and never by clipping a word.
                    if budget >= 8 {
                        spans.push(span(
                            format!("{LABEL}{}", widths::middle_fit(&list, budget)),
                            t.warn,
                        ));
                    } else if widths::fits(&[&head, &unknown, &count, VERB], avail) {
                        spans.push(span(count, t.warn));
                    }
                    spans.push(span(VERB.to_string(), t.text_faint));
                }
                line(spans)
            }
            Loadable::Failed(e) => line(vec![span(
                format!(" model availability unavailable: {e}"),
                t.text_muted,
            )]),
            _ => line(vec![span(String::new(), t.text)]),
        }
    });

    let ctx_table = ctx.clone();
    let table = dyn_view_scoped(LayoutStyle::default().grow(1.0), move |gcx| {
        let t = theme.get().tokens;
        let data = store.routes.get();
        loadable_view(
            &t,
            &data,
            |d: &RoutesData| d.rows.is_empty(),
            "the CLI reported no routes",
            |d| routes_table(gcx, cx, &ctx_table, &t, d, ui.route_sel),
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
                // The FULL key leads, always: the route column now shows a
                // task row as `└ text_to_image` under its parent, and this
                // is where `output.image.text_to_image` stays readable and
                // copyable for `config set-default`.
                let mut spans = vec![span_bold(format!(" {} ", r.key), t.accent)];
                // WHAT THE PARENT ROW IS FOR, in words. The grid alone made
                // an operator ask whether `output.image` was dead code next
                // to t2i/i2i/upscale; it is the opposite — the ONE value
                // that serves every image task with no row of its own, and
                // the simple setting for someone who wants one image model.
                if r.is_task_parent() {
                    spans.push(span(
                        format!("serves any {} task with no row of its own  ", r.modality),
                        t.text_muted,
                    ));
                    if r.covered_by_tasks {
                        spans.push(span(
                            format!("· all {} task rows below are set, so nothing reads it  ", r.task_keys.len()),
                            t.text_faint,
                        ));
                    }
                } else if let Some(parent) = &r.broad_key {
                    spans.push(span(
                        if r.inherits_broad {
                            format!("no value of its own — {parent} answers it  ")
                        } else {
                            format!("overrides {parent}  ")
                        },
                        t.text_muted,
                    ));
                }
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
                // The weights answer for THIS row, with the evidence or
                // the actionable line the probe produced — the detail a
                // 14-cell column cannot carry.
                if let Some(wr) = store
                    .availability
                    .with(|d| d.ready().and_then(|a| a.by_route.get(&r.key).cloned()))
                {
                    let tone = match wr.status.as_str() {
                        "installed" => t.ok,
                        "absent" => t.warn,
                        _ => t.text_muted,
                    };
                    spans.push(span(format!("  weights {} ", wr.label()), tone));
                    if wr.status == "absent" && !wr.artifact.is_empty() {
                        spans.push(span(format!("· w downloads {} ", wr.artifact), t.text_faint));
                    } else if !wr.detail.is_empty() {
                        spans.push(span(format!("· {} ", wr.detail), t.text_faint));
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
        // `w` for WEIGHTS, and deliberately not `d`. `d` deletes on the
        // Providers screen, and a key that means "delete" one screen over must
        // not mean "download" here: downloads are safe and frequent, so `d`
        // would train a fast confirm-and-move-on reflex that an operator then
        // carries onto a destructive prompt. `w` also names the column the
        // operator is looking at.
        .shortcut(KeyChord::plain(Key::Char('w')), {
            let ctx_dl = ctx.clone();
            move |_| download_selected(cx, &ctx_dl)
        })
        // `a` for APPLY, the other half of the weights banner. Safe by
        // default: without --force the CLI keeps every route the operator
        // configured differently, so this key can never quietly overrule a
        // deliberate choice — the prompt offers that as a separate answer.
        .shortcut(KeyChord::plain(Key::Char('a')), {
            let ctx_apply = ctx.clone();
            move |_| apply_recommended(cx, &ctx_apply)
        })
        .child(banner)
        .child(weights_banner)
        .child(table)
        .child(detail)
        .build()
}

/// `d` — download the selected route's model weights.
///
/// NEVER AUTOMATIC, ALWAYS CONFIRMED. This is the one verb in the
/// console that spends gigabytes, so it names the artifact and the
/// provider tool before it runs. It refuses (with the reason) rather
/// than guessing when: nothing is selected, the route names no model,
/// the weights are already here, the provider has no download verb, or
/// availability says `unknown` — a guess there costs the operator's
/// disk, and the honest answer is the instruction the probe already
/// produced.
fn download_selected(cx: Scope, ctx: &Ctx) {
    let Some(row) = selected_route(ctx) else {
        ctx.store
            .notice
            .set(Some("no route selected — nothing to download".into()));
        return;
    };
    let weights = ctx
        .store
        .availability
        .with_untracked(|d| d.ready().and_then(|a| a.by_route.get(&row.key).cloned()));
    let Some(weights) = weights else {
        ctx.store.notice.set(Some(format!(
            "{}: no weight information yet (r reloads) — nothing to download",
            row.key
        )));
        return;
    };
    match weights.status.as_str() {
        "installed" => {
            ctx.store.notice.set(Some(format!(
                "{} is already installed{}",
                weights.artifact,
                if weights.detail.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", weights.detail)
                }
            )));
            return;
        }
        "not_applicable" => {
            ctx.store.notice.set(Some(format!(
                "{} serves models remotely — there is nothing to download",
                weights.provider
            )));
            return;
        }
        "unknown" => {
            let hint = if weights.instruction.is_empty() {
                "the provider's tool could not be consulted".to_string()
            } else {
                weights.instruction.clone()
            };
            ctx.store.notice.set(Some(format!(
                "{}: availability is unknown — {hint}",
                row.key
            )));
            return;
        }
        _ => {}
    }
    if !weights.downloadable {
        ctx.store.notice.set(Some(if weights.instruction.is_empty() {
            format!("{} has no download tool on this machine", weights.provider)
        } else {
            weights.instruction.clone()
        }));
        return;
    }

    let ctx2 = ctx.clone();
    let provider = weights.provider.clone();
    let artifact = weights.artifact.clone();
    super::forms::confirm_danger(
        cx,
        ctx.ui,
        format!(
            "Download {artifact} with {provider}? This runs the provider's own tool and \
             may fetch several gigabytes."
        ),
        "Download",
        "Not now",
        move || {
            ctx2.send(Cmd::DownloadModel {
                provider: provider.clone(),
                artifact: artifact.clone(),
            });
            // The worker lane is SERIAL, so this re-probe runs after the
            // download returns — the weights column tells the truth the
            // moment the operator looks back at it, with no timer and no
            // second definition of "finished".
            ctx2.send(Cmd::LoadAvailability);
        },
    );
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

/// The ONE refusal vocabulary, word for word the gateway console's —
/// derived from the row's own payload fields, never a hardcoded key.
fn not_editable_reason(row: &RouteRow) -> String {
    if let Some(d) = &row.derived_from {
        format!("{} derives from {} — edit that route instead", row.key, d)
    } else if row.covered_by.is_some() {
        format!("{} is covered and not overrideable", row.key)
    } else {
        format!("{} is read-only", row.key)
    }
}

fn edit_selected(cx: Scope, ctx: &Ctx) {
    let Some(row) = selected_route(ctx) else {
        ctx.store
            .notice
            .set(Some("no route selected — nothing to edit".into()));
        return;
    };
    if !row.editable() {
        ctx.store.notice.set(Some(not_editable_reason(&row)));
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
    // Clearing only makes sense for an explicitly configured row — a
    // derived or covered one has no override of its own to drop (the
    // gateway console's exact gate).
    if !row.editable() {
        ctx.store.notice.set(Some(not_editable_reason(&row)));
        return;
    }
    if !row.configured || row.covered_by.is_some() {
        ctx.store.notice.set(Some(format!(
            "{} has no explicit override to clear",
            row.key
        )));
        return;
    }
    if !ctx.writable_now() {
        return;
    }
    confirm_clear(cx, ctx, row);
}

/// `a` — make this machine's routes match the framework recommendation.
///
/// Three answers, because the honest action has three: fill only the
/// empty routes (safe, the default), replace the operator's choices too
/// (the `--force` spelling, offered as its own option so it can never be
/// the accidental one), or nothing. The routes the CLI keeps are named in
/// its report, which rides the journal like every other write.
fn apply_recommended(cx: Scope, ctx: &Ctx) {
    if !ctx.writable_now() {
        return;
    }
    let ctx_keep = ctx.clone();
    let ctx_force = ctx.clone();
    let prompt = abstracttui::app::ChoicePrompt::new(
        "Apply the framework's recommended routes (text, voice, image)?"
            .to_string(),
    )
    .option("keep", "Apply — keep routes I configured")
    .option_with(
        abstracttui::app::ChoiceOption::new("force", "Apply — replace mine too").danger(true),
    )
    .option("cancel", "Cancel")
    .initial("keep");
    super::forms::open_prompt(cx, ctx.ui, prompt, move |outcome| {
        if let abstracttui::app::ChoiceOutcome::Answered(a) = outcome {
            let choice = a.selected.first().cloned().unwrap_or_default();
            let (ctx2, force) = match choice.as_str() {
                "keep" => (ctx_keep, false),
                "force" => (ctx_force, true),
                _ => return,
            };
            let spec = writes::apply_recommended(force, ctx2.write_base(), None);
            ctx2.send(Cmd::Write(Box::new(spec)));
        }
    });
}

/// ONE confirm policy for clearing a route — shared by the table's `x`
/// and the editor's Clear button (the editor closes itself first, then
/// prompts on the screen scope).
fn confirm_clear(cx: Scope, ctx: &Ctx, row: RouteRow) {
    let ctx2 = ctx.clone();
    let key = row.key.clone();
    confirm_danger(
        cx,
        ctx.ui,
        format!(
            "Clear the override on {} ({})? The engine default takes over.",
            row.key,
            row.pair_text()
        ),
        "Clear the route",
        "Keep the override",
        move || {
            let spec = writes::clear_route(&key, ctx2.write_base(), None);
            ctx2.send(Cmd::Write(Box::new(spec)));
        },
    );
}

/// "Applies now" — derived ONLY from the stored row, never from local
/// picks. The gateway console's line, verbatim, plus the reasoning the
/// text route carries.
fn applies_now(t: &TokenSet, row: &RouteRow) -> View {
    let mut spans = vec![span_bold("Applies now: ".to_string(), t.text)];
    if row.configured {
        spans.push(span(row.pair_text(), t.ok));
        if let Some(r) = &row.reasoning {
            spans.push(span(format!("  · reasoning {r}"), t.ok));
        }
        if let Some(by) = &row.covered_by {
            spans.push(span(format!("  (covered by {by})"), t.info));
        } else {
            spans.push(span(format!("  (source: {})", row.source), t.text_muted));
        }
    } else {
        spans.push(span(
            "nothing configured — engine decides".to_string(),
            t.text_muted,
        ));
        if let Some(hint) = &row.package_hint {
            spans.push(span(format!("  (needs {hint})"), t.text_faint));
        }
    }
    line(spans)
}

/// The route editor: "Applies now" (stored truth), provider (Select
/// with a placeholder at index 0), model (text + `m` picker over live
/// discovery), base URL, reasoning (text routes only), options as
/// `k=v` pairs — the gateway console's field set and wording, over
/// core's CLI write path.
///
/// THE SAVE SENDS WHAT THE OPERATOR EDITED, AND NOTHING ELSE. Every
/// control is diffed against the value it opened with; an untouched
/// field is not named at all, so the store keeps it. Echoing back what
/// the grid last rendered would let a stale row overwrite a setting
/// made from the gateway console between render and save.
fn open_route_editor(cx: Scope, ctx: &Ctx, row: RouteRow) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    // The SCREEN scope: the editor's Clear button prompts on it after
    // closing the editor (the modal scope dies with the editor).
    let screen_cx = cx;
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
    open_form_guarded(ctx, cx, Size::new(84, 22), move |mcx, close, guard| {
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
        // Reasoning is a fixed vocabulary, not free text: the same
        // five choices the web console offers, index 0 = "not set".
        let is_text = row.is_text_generation();
        let reasoning_init_ix = crate::store::reasoning_index(row.reasoning.as_deref());
        let reasoning_sel = mcx.signal(reasoning_init_ix);
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
        // These are also the DIFF BASE the save reads: what the form
        // opened with, so an untouched field is never sent.
        let p0 = pinitial;
        let model_init = model.get_untracked();
        let url_init = base_url.get_untracked();
        let options_init = options.get_untracked();
        {
            let model_init = model_init.clone();
            let url_init = url_init.clone();
            let options_init = options_init.clone();
            super::forms::install_dirty_guard_with(
                mcx,
                &guard,
                move || {
                    provider_sel.get_untracked() != p0
                        || model.with_untracked(|v| v != &model_init)
                        || base_url.with_untracked(|v| v != &url_init)
                        || reasoning_sel.get_untracked() != reasoning_init_ix
                        || options.with_untracked(|v| v != &options_init)
                },
                move || {
                    let _ = provider_sel.get();
                    let _ = model.get();
                    let _ = base_url.get();
                    let _ = reasoning_sel.get();
                    let _ = options.get();
                },
                esc_armed,
                form_error,
            );
        }
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
            let m = model.get_untracked().trim().to_string();
            let u = base_url.get_untracked().trim().to_string();
            let r_ix = reasoning_sel.get_untracked();
            let opts = match parse_options(&options.get_untracked()) {
                Ok(o) => o,
                Err(e) => {
                    form_error.set(Some(e));
                    return;
                }
            };
            // THE DIFF. Each field is named only when the operator
            // moved it; an emptied field is named as "" so the store
            // CLEARS it rather than silently keeping the old value.
            let edit = writes::RouteEdit {
                provider: (i != p0).then(|| {
                    if i > 0 {
                        providers3[i - 1].clone()
                    } else {
                        String::new()
                    }
                }),
                model: (m != model_init.trim()).then(|| m.clone()),
                base_url: (u != url_init.trim()).then(|| u.clone()),
                reasoning: (is_text && r_ix != reasoning_init_ix).then(|| {
                    crate::store::REASONING_LEVELS
                        .get(r_ix.wrapping_sub(1))
                        .map(|s| (*s).to_string())
                        .unwrap_or_default()
                }),
                options: (options.get_untracked() != options_init).then_some(opts),
            };
            if edit.is_empty() {
                form_error.set(Some(
                    "nothing changed — edit a field, or Clear override drops the route".into(),
                ));
                return;
            }
            let spec = writes::set_route(&key, &edit, base, Some(form_id));
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
                class: crate::models::class_for_route(&row.kind, &row.modality),
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
        let reasoning_opts: Vec<SelectOption> = std::iter::once(SelectOption::new("not set"))
            .chain(
                crate::store::REASONING_LEVELS
                    .iter()
                    .map(|l| SelectOption::new(*l)),
            )
            .collect();
        let row_for_clear = row.clone();
        let ctx_clear = ctx2.clone();
        let close_after_clear = close.clone();
        let clear_enabled = row.configured && row.covered_by.is_none();
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
                    .child(applies_now(&t, &row))
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
                    // Reasoning belongs to TEXT GENERATION only — the
                    // row is absent (not disabled) on every other
                    // route, exactly as the web console hides it.
                    .child(if is_text {
                        field(
                            &t,
                            "reasoning",
                            Select::new(reasoning_opts).value(reasoning_sel).view(mcx),
                        )
                    } else {
                        Element::new().style(LayoutStyle::default().h(0)).build()
                    })
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
                                // ONE policy for a destructive clear:
                                // the same danger confirm as the
                                // table's x. Close this editor first
                                // (prompt-over-modal would stack two
                                // modals — the engine hazard), then
                                // prompt on the SCREEN scope, which
                                // outlives the editor.
                                Button::new("Clear override")
                                    .disabled(!clear_enabled)
                                    .on_click(move || {
                                        close_after_clear();
                                        confirm_clear(screen_cx, &ctx_clear, row_for_clear.clone());
                                    })
                                    .view(mcx),
                            )
                            .child(
                                Button::new("Cancel (Esc)")
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

/// `gcx` builds the table; `page_cx` is the scope activation opens its
/// modal on — the routes dyn is disposed on every reload, and a form
/// parented to it would vanish mid-edit.
fn routes_table(
    gcx: Scope,
    page_cx: Scope,
    ctx: &Ctx,
    t: &TokenSet,
    data: &RoutesData,
    sel: Signal<usize>,
) -> View {
    // Width-aware columns: which columns APPEAR is a breakpoint decision
    // (source drops first, provider second — the sibling's 0900-class
    // recipe); how wide the survivors are is measured from the rows
    // themselves by `ui::widths`, never from a constant. The old policy
    // spent constants here (`Cells(30)`, `Cells(20)`, `ellipsize(model,
    // 40)`) and handed the leftover to a Flex model column, which is how
    // a 200-cell terminal ended up printing
    // `AbstractFramework/wan2.2-t2v-a14b-diffu…` beside seventy blank
    // cells.
    let w = abstracttui::app::use_viewport(gcx).get().w;
    // Read reactively: when the weights probe lands, the grid repaints.
    // An empty map (probe not answered) prints nothing, never "absent".
    let weights = ctx
        .store
        .availability
        .with(|d| d.ready().map(|a| a.by_route.clone()).unwrap_or_default());
    let mut rows: Vec<Vec<String>> = data
        .rows
        .iter()
        .map(|r| {
            // THE shared state vocabulary, straight off the row model —
            // the same four strings the gateway console prints.
            let state = r.state_label();
            // `⊘` U+2298, NOT `🔒` U+1F512: the padlock is Emoji=Yes and
            // measures 2 cells, and terminals/fonts routinely draw it at
            // a different advance than the engine measured — the row's
            // later columns then slide and overlap (visible on the alias
            // row, and only there, since it is the row that carries the
            // marker). The engine's own glyph research rejects emoji
            // outright and lands on U+2298: width 1 under BOTH
            // unicode-width conventions, in Mathematical Operators, a
            // block emoji-data never touches. The `state` column already
            // spells the reason in words, so the glyph is garnish.
            let lock = if !r.editable() { " ⊘" } else { "" };
            // THE ROUTE COLUMN CARRIES THE HIERARCHY. `output.image` is
            // the PARENT of `output.image.*` — one value for every image
            // task, overridden per task by the rows beneath it. Printed
            // as four flat siblings with the parent on top reading "not
            // configured", it looked like a leftover key ("why do we have
            // output.image AND t2i/i2i/upscale?"). `display_key()` indents
            // the children under a tree marker and drops the repeated
            // parent prefix; the parent names what it is for.
            let mut row = vec![format!("{}{}", r.display_key(), lock), state];
            if w >= 96 {
                row.push(or_dash(&r.provider));
            }
            // The FULL model name — the column solver sizes to it and
            // cuts it only if the terminal cannot carry it. A constant
            // cap here truncated the payload while the space to print it
            // whole sat unused one column over.
            row.push(or_dash(&r.model));
            // WEIGHTS: is this route's model actually on the machine?
            // Blank while unprobed and for rows that name no model — the
            // absence of an answer must not read as an answer.
            row.push(
                weights
                    .get(&r.key)
                    .map(|w| w.label().to_string())
                    .unwrap_or_default(),
            );
            if w >= 112 {
                row.push(r.source.clone());
            }
            row
        })
        .collect();
    // The rules carry a FLOOR, not a width: nothing is capped while the
    // terminal has room. Route keys, provider ids, model artifacts and
    // source modules all discriminate on their TAIL
    // (`…image_to_scene3d`, `…-t2v-a14b-diffusers-8bit`), so those cut in
    // the middle; the state and weights vocabularies read from the left.
    // `state` and `weights` print CLOSED VOCABULARIES, so their floor is
    // the widest word each can say — a squeezed vocabulary column is not
    // a shorter answer, it is a different (wrong) one. The open columns
    // (route/provider/model/source) carry operator content and take the
    // squeeze on its behalf.
    let mut rules = vec![
        widths::ColRule::tail("route", 18),
        widths::ColRule::head("state", 21),
    ];
    if w >= 96 {
        rules.push(widths::ColRule::tail("provider", 10));
    }
    rules.push(widths::ColRule::tail("model", 22));
    // The weights column earns its floor at every width: "configured
    // but not downloaded" is the single most common reason a route that
    // LOOKS right does not run, and hiding it on a narrow terminal
    // hides exactly the machine most likely to be a fresh install. 14
    // cells is the widest label it prints ("not downloaded"), so this
    // floor is the whole vocabulary, never a stub.
    rules.push(widths::ColRule::head("weights", 14));
    if w >= 112 {
        // Source attribution per row — the payload's own `source`
        // string, the same column the gateway console shows.
        rules.push(widths::ColRule::tail("source", 12));
    }
    // This screen mounts straight into PageHost's page region, which adds
    // no horizontal chrome of its own — the table's rect IS the viewport.
    // (The gateway console's routes screen wraps the same grid in a
    // bordered block and subtracts 2; the policy is shared, the chrome is
    // per-screen.)
    let cols = widths::columns(&rules, &mut rows, w);
    let ctx_act = ctx.clone();
    Table::new(cols)
        .rows(rows)
        .selection(sel)
        // Enter, Space and double-click = the `e` verb, to the row the
        // Table just selected (select() runs before activate, so
        // `ui.route_sel` is already the activated row). Routing through
        // edit_selected keeps ONE set of refusals — the output.text
        // alias, covered/read-only rows, and the write door all answer
        // the same way whether the user pressed a key or double-clicked.
        .on_activate(move |_| edit_selected(page_cx, &ctx_act))
        .layout(LayoutStyle::default().grow(1.0))
        .element(gcx, t)
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
