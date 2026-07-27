//! The typed field editors: the right control per FieldKind, each
//! submitting one verified WriteSpec. Every editor:
//!   - shows "Applies now" truth, never fabricates a selection
//!     (placeholders at index 0),
//!   - validates BEFORE submitting (the schema's validate is the same
//!     rule the mirror classifies by),
//!   - closes on verified success, stays open with the verbatim error
//!     on failure (write_done routing).

use abstracttui::prelude::*;
use abstracttui::widgets::{Block, Button, Checkbox};
use serde_json::Value;

use crate::config::{FieldState, FieldView};
use crate::schema::{self, FieldKind};
use crate::store::Loadable;
use crate::worker::{next_form_id, Cmd};
use crate::writes::{self, FieldRoute};

use super::forms::{
    install_dirty_guard, install_write_done, message_slot, open_form_guarded, open_prompt,
};
use super::util::{field, line, span};
use super::Ctx;

/// Open the right editor for one (section, key). The entry point every
/// screen and the wizard use — refusals name their reason.
pub fn open_field_editor(cx: Scope, ctx: &Ctx, section: &'static str, key: &'static str) {
    let Some(spec) = schema::section(section) else {
        ctx.store.notice.set(Some(format!("unknown section {section}")));
        return;
    };
    let Some(fs) = spec.fields.iter().find(|f| f.key == key) else {
        ctx.store.notice.set(Some(format!("unknown field {section}.{key}")));
        return;
    };
    // The door matches the field's WRITE ROUTE: direct-write fields
    // stay editable on a Python-refused file (they preserve every
    // byte); CLI-routed fields refuse there (a setter would reset the
    // file to defaults) — the same split the worker enforces.
    let rmw_routed = matches!(writes::field_route(section, key), FieldRoute::Rmw);
    let door_open = if rmw_routed {
        ctx.writable_now_rmw()
    } else {
        ctx.writable_now()
    };
    if !door_open {
        return; // the door posted the reason
    }
    match writes::field_route(section, key) {
        FieldRoute::Pair => open_pair_editor(cx, ctx, section, key),
        FieldRoute::Secret => open_secret_editor(cx, ctx, section, key),
        FieldRoute::VisionStrategy => open_vision_strategy(cx, ctx),
        FieldRoute::Toggle { .. } => open_toggle(cx, ctx, section, key, fs.note),
        _ => match fs.kind {
            FieldKind::Enum(choices) | FieldKind::OptEnum(choices) => {
                open_enum_editor(cx, ctx, section, key, choices)
            }
            FieldKind::EnumLoose { canon, .. } => {
                // The CLI's own choice list excludes the reserved
                // `caption` (main.py:279-286) — offer what the setter
                // accepts.
                let offered: Vec<&'static str> = canon
                    .iter()
                    .copied()
                    .filter(|c| !(section == "audio" && *c == "caption"))
                    .collect();
                open_enum_editor_vec(cx, ctx, section, key, offered)
            }
            FieldKind::Bool => open_toggle(cx, ctx, section, key, fs.note),
            FieldKind::FallbackChain => open_chain_editor(cx, ctx, section),
            _ => open_scalar_editor(cx, ctx, section, key, fs.kind),
        },
    }
}

/// Current field view from the mirror (value display + state).
fn field_view(ctx: &Ctx, section: &str, key: &str) -> Option<FieldView> {
    ctx.store.cfg.with_untracked(|c| {
        c.ready().and_then(|m| match &m.state {
            crate::config::FileState::Ready(snap) => snap
                .sections
                .iter()
                .find(|s| s.spec.name == section)
                .and_then(|s| s.fields.iter().find(|f| f.key == key))
                .cloned(),
            _ => None,
        })
    })
}

/// The raw current string value for prefill (never secrets — secret
/// editors don't prefill). The display's honest-empty rendering `""`
/// maps back to an actually-empty buffer (M2 review P3-7: feeding the
/// two-quote TEXT back would save a literal `""` string).
fn prefill(ctx: &Ctx, section: &str, key: &str) -> String {
    field_view(ctx, section, key)
        .filter(|f| !matches!(f.state, FieldState::Default))
        .map(|f| match f.display.as_str() {
            "—" | "\"\"" => String::new(),
            _ => f.display,
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------
// Scalar (text) editor: Str/OptStr/Path/Int/Float/EnvVarName.
// ---------------------------------------------------------------------

fn open_scalar_editor(
    cx: Scope,
    ctx: &Ctx,
    section: &'static str,
    key: &'static str,
    kind: FieldKind,
) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let base = ctx.write_base();
    let initial = prefill(ctx, section, key);
    let nullable = matches!(
        kind,
        FieldKind::OptStr | FieldKind::OptPath | FieldKind::OptInt { .. } | FieldKind::OptEnum(_)
    );
    open_form_guarded(ctx, cx, Size::new(74, 12), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let value = mcx.signal(initial.clone());
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        install_dirty_guard(
            mcx,
            &guard,
            vec![(value, initial.clone())],
            esc_armed,
            form_error,
        );
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        let submit = {
            let ctx = ctx2.clone();
            move || {
                if in_flight.get_untracked() {
                    return;
                }
                form_error.set(None);
                let raw = value.get_untracked();
                let trimmed = raw.trim().to_string();
                let spec = if trimmed.is_empty() && nullable {
                    writes::clear_scalar(section, key, base, Some(form_id))
                } else {
                    match typed_value(&kind, &trimmed) {
                        Ok(v) => match schema::validate(&kind, &v) {
                            Ok(()) => {
                                writes::set_scalar(section, key, v, base, Some(form_id))
                            }
                            Err(e) => Err(e),
                        },
                        Err(e) => Err(e),
                    }
                };
                match spec {
                    Ok(spec) => {
                        in_flight.set(true);
                        ctx.send(Cmd::Write(Box::new(spec)));
                    }
                    Err(e) => form_error.set(Some(e)),
                }
            }
        };
        let submit_btn = submit.clone();

        let hint = kind_hint(&kind, nullable);
        Block::new()
            .title(format!("Edit {section}.{key}"))
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(1))
                    .child(field(
                        &t,
                        "value",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(value)
                            .on_submit({
                                let s = submit.clone();
                                move |_| s()
                            })
                            .view(mcx),
                    ))
                    .child(line(vec![span(format!(" {hint}"), t.text_faint)]))
                    .child(applies_now_line(theme, &ctx2, section, key))
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

/// Parse the typed value for the kind (argv spelling + Expect value).
fn typed_value(kind: &FieldKind, s: &str) -> Result<Value, String> {
    match kind {
        FieldKind::Int { .. } | FieldKind::OptInt { .. } => s
            .parse::<i64>()
            .map(Value::from)
            .map_err(|_| format!("\"{s}\" is not an integer")),
        FieldKind::Float { .. } | FieldKind::FloatFree => s
            .parse::<f64>()
            .map(Value::from)
            .map_err(|_| format!("\"{s}\" is not a number")),
        _ => Ok(Value::String(s.to_string())),
    }
}

fn kind_hint(kind: &FieldKind, nullable: bool) -> String {
    let base = match kind {
        FieldKind::Int { min, max } | FieldKind::OptInt { min, max } => {
            if *max == i64::MAX {
                format!("integer ≥ {min}")
            } else {
                format!("integer {min}..={max}")
            }
        }
        FieldKind::Float { min } => format!("number ≥ {min} (0 = unlimited)"),
        FieldKind::FloatFree => "number".into(),
        FieldKind::EnvVarName => "an environment variable NAME (not the secret)".into(),
        FieldKind::Path | FieldKind::OptPath => "filesystem path (~ ok)".into(),
        _ => "text".into(),
    };
    if nullable {
        format!("{base} — leave blank to clear")
    } else {
        base
    }
}

/// The one truth line: what the mirror holds NOW (value + state), so
/// the form never pretends an unsaved value applies. REACTIVE — a
/// write completing while the form is open must update it, not show
/// the open-time snapshot (M2 review P3-2).
fn applies_now_line(
    theme: Signal<&'static abstracttui::theme::Theme>,
    ctx: &Ctx,
    section: &'static str,
    key: &'static str,
) -> View {
    let store = ctx.store;
    dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        let now = store.cfg.with(|c| {
            c.ready().and_then(|m| match &m.state {
                crate::config::FileState::Ready(snap) => snap
                    .sections
                    .iter()
                    .find(|s| s.spec.name == section)
                    .and_then(|s| s.fields.iter().find(|f| f.key == key))
                    .cloned(),
                _ => None,
            })
        });
        let spans = match now {
            Some(f) => {
                let state = match f.state {
                    FieldState::Default => "default",
                    FieldState::Set => "set",
                    FieldState::Broken(_) => "BROKEN",
                };
                vec![
                    span(" applies now: ", t.text_faint),
                    span(f.display, t.text_muted),
                    span(format!("  ({state})"), t.text_faint),
                ]
            }
            None => vec![span(" applies now: — (file not loaded)", t.text_faint)],
        };
        line(spans)
    })
}

// ---------------------------------------------------------------------
// Enum editor: a Select over the canonical choices.
// ---------------------------------------------------------------------

fn open_enum_editor(
    cx: Scope,
    ctx: &Ctx,
    section: &'static str,
    key: &'static str,
    choices: &'static [&'static str],
) {
    open_enum_editor_vec(cx, ctx, section, key, choices.to_vec());
}

fn open_enum_editor_vec(
    cx: Scope,
    ctx: &Ctx,
    section: &'static str,
    key: &'static str,
    choices: Vec<&'static str>,
) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let base = ctx.write_base();
    let current = prefill(ctx, section, key);
    open_form_guarded(ctx, cx, Size::new(64, 12), move |mcx, close, _guard| {
        let t = theme.get().tokens;
        // Placeholder at index 0 — a picker must never fabricate a
        // selection (the sibling's combo-fabrication law).
        let mut options = vec![SelectOption::new("— choose —")];
        let mut initial = 0usize;
        for (i, c) in choices.iter().enumerate() {
            options.push(SelectOption::new(*c));
            if *c == current {
                initial = i + 1;
            }
        }
        let picked = mcx.signal(initial);
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        let choices2 = choices.clone();
        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            let i = picked.get_untracked();
            if i == 0 {
                form_error.set(Some("choose a value first".into()));
                return;
            }
            let value = choices2[i - 1];
            let spec = if section == "audio" && key == "strategy" {
                Ok(writes::set_audio_strategy(
                    value,
                    base,
                    Some(form_id),
                ))
            } else {
                writes::set_scalar(
                    section,
                    key,
                    Value::String(value.into()),
                    base,
                    Some(form_id),
                )
            };
            match spec {
                Ok(spec) => {
                    in_flight.set(true);
                    ctx3.send(Cmd::Write(Box::new(spec)));
                }
                Err(e) => form_error.set(Some(e)),
            }
        };
        let submit_btn = submit.clone();

        Block::new()
            .title(format!("Edit {section}.{key}"))
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(1))
                    .child(field(&t, "value", Select::new(options).value(picked).view(mcx)))
                    .child(applies_now_line(theme, &ctx2, section, key))
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

// ---------------------------------------------------------------------
// Bool toggle — with the danger confirm on the UNSAFE server flags.
// ---------------------------------------------------------------------

fn open_toggle(
    cx: Scope,
    ctx: &Ctx,
    section: &'static str,
    key: &'static str,
    note: Option<&'static str>,
) {
    let current_true = field_view(ctx, section, key)
        .map(|f| f.display == "true")
        .unwrap_or(false);
    let target = !current_true;
    let unsafe_flag = note.map(|n| n.starts_with("UNSAFE")).unwrap_or(false);
    let ctx2 = ctx.clone();
    let base = ctx.write_base();
    let submit = move || {
        match writes::set_scalar(section, key, Value::Bool(target), base, None) {
            Ok(spec) => ctx2.send(Cmd::Write(Box::new(spec))),
            Err(e) => ctx2.store.notice.set(Some(e)),
        }
    };
    if unsafe_flag && target {
        super::forms::confirm_danger(
            cx,
            ctx.ui,
            format!(
                "{section}.{key} = true is flagged UNSAFE in abstractcore's own status. Enable it?"
            ),
            "Enable (unsafe)",
            "Keep it off",
            submit,
        );
    } else {
        // A plain flip still confirms — a single keystroke silently
        // toggling server config would be too cheap an accident.
        open_prompt(
            cx,
            ctx.ui,
            abstracttui::app::ChoicePrompt::new(format!(
                "Set {section}.{key} = {target}? (now {current_true})"
            ))
            .option("go", format!("Set {target}"))
            .option("keep", "Keep as is")
            .initial("go"),
            move |outcome| {
                if let abstracttui::app::ChoiceOutcome::Answered(a) = outcome {
                    if a.selected.iter().any(|s| s == "go") {
                        submit();
                    }
                }
            },
        );
    }
}

// ---------------------------------------------------------------------
// Secret editor: masked input, blank keeps, explicit clear.
// ---------------------------------------------------------------------

fn open_secret_editor(cx: Scope, ctx: &Ctx, section: &'static str, key: &'static str) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let base = ctx.write_base();
    let current = field_view(ctx, section, key)
        .map(|f| f.display)
        .unwrap_or_else(|| "not set".into());
    open_form_guarded(ctx, cx, Size::new(74, 13), move |mcx, close, guard| {
        let t = theme.get().tokens;
        let value = mcx.signal(String::new());
        let clear = mcx.signal(false);
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let esc_armed = mcx.signal(false);
        install_dirty_guard(mcx, &guard, vec![(value, String::new())], esc_armed, form_error);
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let typed = value.get_untracked();
            let typed = typed.trim();
            let wants_clear = clear.get_untracked();
            let spec = if wants_clear {
                if section == "server" {
                    writes::clear_server_auth_token(base, Some(form_id))
                } else {
                    writes::clear_api_key(key, base, Some(form_id))
                }
            } else if typed.is_empty() {
                form_error.set(Some(
                    "blank keeps the stored secret — type a new one or tick clear".into(),
                ));
                return;
            } else if typed.starts_with('-') {
                // The flags CLI parses argv with argparse — a value
                // starting with '-' reads as a flag and dies with a
                // cryptic "expected one argument" (M2 review P3-5).
                form_error.set(Some(
                    "values starting with '-' cannot pass the abstractcore flags CLI — \
                     regenerate the key/token"
                        .into(),
                ));
                return;
            } else if section == "server" {
                writes::set_server_auth_token(typed, base, Some(form_id))
            } else {
                writes::set_api_key(key, typed, base, Some(form_id))
            };
            in_flight.set(true);
            ctx3.send(Cmd::Write(Box::new(spec)));
        };
        let submit_btn = submit.clone();

        Block::new()
            .title(format!("Secret — {section}.{key}"))
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(1))
                    .child(line(vec![
                        span(" stored: ", t.text_faint),
                        span(current.clone(), t.text_muted),
                        span("  (never echoed — fingerprint only)", t.text_faint),
                    ]))
                    .child(field(
                        &t,
                        "new value",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(value)
                            .masked(true)
                            .on_submit({
                                let s = submit.clone();
                                move |_| s()
                            })
                            .view(mcx),
                    ))
                    .child(field(
                        &t,
                        "",
                        Checkbox::new("clear the stored secret").checked(clear).view(mcx),
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

// ---------------------------------------------------------------------
// The provider/model pair editor — global default, embeddings, vision
// caption, app defaults. One form, four couplings.
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub enum PairKind {
    GlobalDefault,
    Embeddings,
    VisionCaption,
    App(&'static str),
}

impl PairKind {
    fn title(&self) -> String {
        match self {
            PairKind::GlobalDefault => "Default model (global + route input.text)".into(),
            PairKind::Embeddings => "Embeddings (legacy pair + route embedding.text)".into(),
            PairKind::VisionCaption => "Vision caption pair (sets strategy two_stage)".into(),
            PairKind::App(app) => format!("App default — {app}"),
        }
    }

    /// The (section, provider_key, model_key) this pair reads from —
    /// the "applies now" truth line's source.
    fn fields(&self) -> (&'static str, String, String) {
        match self {
            PairKind::GlobalDefault => (
                "default_models",
                "global_provider".into(),
                "global_model".into(),
            ),
            PairKind::Embeddings => ("embeddings", "provider".into(), "model".into()),
            PairKind::VisionCaption => (
                "vision",
                "caption_provider".into(),
                "caption_model".into(),
            ),
            PairKind::App(app) => (
                "app_defaults",
                format!("{app}_provider"),
                format!("{app}_model"),
            ),
        }
    }

    /// What the model picker filters FOR (crate::models): embedding
    /// models for the embeddings pair; generative for the rest
    /// (vision-capable vs text cannot be told apart by name — no
    /// finer pretense).
    fn model_class(&self) -> crate::models::ModelClass {
        match self {
            PairKind::Embeddings => crate::models::ModelClass::Embedding,
            _ => crate::models::ModelClass::Generative,
        }
    }

    /// The provider choices honest for this pair.
    fn providers(&self, ctx: &Ctx) -> Vec<String> {
        let mut out: Vec<String> = match self {
            PairKind::Embeddings => schema::EMBEDDING_PROVIDERS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            _ => schema::STATIC_PROVIDERS.iter().map(|s| s.to_string()).collect(),
        };
        // endpoint:<id> profiles extend the registry at runtime.
        if !matches!(self, PairKind::Embeddings) {
            if let Loadable::Ready(p) = ctx.store.profiles.get_untracked() {
                for prof in &p.profiles {
                    if prof.enabled {
                        out.push(prof.virtual_provider());
                    }
                }
            }
        }
        out
    }
}

pub fn open_pair_editor(cx: Scope, ctx: &Ctx, section: &'static str, key: &'static str) {
    let kind = match (section, key) {
        ("default_models", _) => PairKind::GlobalDefault,
        ("embeddings", _) => PairKind::Embeddings,
        ("vision", _) => PairKind::VisionCaption,
        ("app_defaults", k) => {
            let app = schema::APPS
                .iter()
                .find(|a| k.starts_with(&format!("{}_", a)))
                .copied();
            match app {
                Some(a) => PairKind::App(a),
                None => {
                    ctx.store.notice.set(Some(format!("no app for {key}")));
                    return;
                }
            }
        }
        _ => {
            ctx.store.notice.set(Some(format!("{section}.{key} is not a pair field")));
            return;
        }
    };
    open_pair_editor_kind(cx, ctx, kind);
}

pub fn open_pair_editor_kind(cx: Scope, ctx: &Ctx, kind: PairKind) {
    if !ctx.writable_now() {
        return;
    }
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let base = ctx.write_base();
    let providers = std::rc::Rc::new(kind.providers(ctx));
    // Edit semantics: the CURRENT pair prefills both controls, and a
    // prefilled provider kicks discovery at OPEN — the picker must be
    // populated by the time the operator reaches the model row, not
    // only after they re-commit a provider they already had.
    let (section, pkey, mkey) = kind.fields();
    let provider_now = prefill(ctx, section, &pkey);
    let model_now = prefill(ctx, section, &mkey);
    let initial_provider = providers
        .iter()
        .position(|p| *p == provider_now)
        .map(|i| i + 1)
        .unwrap_or(0);
    if initial_provider > 0 {
        super::model_field::kick_discovery(ctx, &provider_now);
    }
    open_form_guarded(ctx, cx, Size::new(78, 15), move |mcx, close, _guard| {
        let t = theme.get().tokens;
        let mut popts = vec![SelectOption::new("— choose a provider —")];
        for p in providers.iter() {
            popts.push(SelectOption::new(p.clone()));
        }
        let provider_sel = mcx.signal(initial_provider);
        let model_text = mcx.signal(model_now.clone());
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());

        // Provider commit → kick model discovery for the picker.
        let providers2 = providers.clone();
        let ctx_models = ctx2.clone();
        let on_provider = move |i: usize| {
            if i > 0 {
                super::model_field::kick_discovery(&ctx_models, &providers2[i - 1]);
            }
        };

        let providers3 = providers.clone();
        let ctx3 = ctx2.clone();
        let submit = move || {
            if in_flight.get_untracked() {
                return;
            }
            form_error.set(None);
            let i = provider_sel.get_untracked();
            if i == 0 {
                form_error.set(Some("choose a provider first".into()));
                return;
            }
            let provider = providers3[i - 1].clone();
            let model = model_text.get_untracked().trim().to_string();
            if model.is_empty() {
                form_error.set(Some("type or pick a model".into()));
                return;
            }
            // `base` captured at editor OPEN (outer scope) — never
            // re-read at submit (M2 review P3-2).
            let spec = match kind {
                PairKind::GlobalDefault => {
                    writes::set_global_default(&provider, &model, base, Some(form_id))
                }
                PairKind::Embeddings => {
                    writes::set_embeddings(&provider, &model, base, Some(form_id))
                }
                PairKind::VisionCaption => {
                    writes::set_vision_pair(&provider, &model, base, Some(form_id))
                }
                PairKind::App(app) => {
                    writes::set_app_default(app, &provider, &model, base, Some(form_id))
                }
            };
            in_flight.set(true);
            ctx3.send(Cmd::Write(Box::new(spec)));
        };
        let submit_btn = submit.clone();

        // The model row: a discovery-backed Combobox filtered to this
        // pair's class (embedding models for the embeddings pair,
        // generative otherwise), with the free-typing fallback.
        let mf = super::model_field::model_field(
            mcx,
            theme,
            super::model_field::ModelFieldSpec {
                store: ctx2.store,
                providers: providers.clone(),
                provider_sel,
                model_text,
                class: kind.model_class(),
                submit: Some(std::rc::Rc::new(submit.clone())),
            },
        );
        let mf_handle = mf.handle.clone();
        let mf_custom = mf.custom;

        // The pair's own truth line (M2 review P3-11): what the mirror
        // holds NOW, reactive so an in-flight completion updates it.
        let pair_now = {
            let store = ctx2.store;
            let (section, pkey, mkey) = kind.fields();
            dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
                let t = theme.get().tokens;
                let read = |key: &str| {
                    store.cfg.with(|c| {
                        c.ready().and_then(|m| match &m.state {
                            crate::config::FileState::Ready(snap) => snap
                                .sections
                                .iter()
                                .find(|s| s.spec.name == section)
                                .and_then(|s| s.fields.iter().find(|f| f.key == key))
                                .map(|f| f.display.clone()),
                            _ => None,
                        })
                    })
                };
                let p = read(&pkey).unwrap_or_else(|| "—".into());
                let m = read(&mkey).unwrap_or_else(|| "—".into());
                line(vec![
                    span(" applies now: ", t.text_faint),
                    span(format!("{p} / {m}"), t.text_muted),
                ])
            })
        };

        Block::new()
            .title(kind.title())
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(1))
                    .shortcut(KeyChord::plain(Key::Char('m')), {
                        let ctx_m = ctx2.clone();
                        move |_| {
                            if !mf_handle.open() {
                                ctx_m.store.notice.set(Some(
                                    "no picker to open — choose a provider first (or Ctrl+P \
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
                    .child(pair_now)
                    .child(field(
                        &t,
                        "provider",
                        Select::new(popts)
                            .value(provider_sel)
                            .on_change(on_provider)
                            .view(mcx),
                    ))
                    .child(mf.view)
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

// ---------------------------------------------------------------------
// vision.strategy: three values, three verbs.
// ---------------------------------------------------------------------

fn open_vision_strategy(cx: Scope, ctx: &Ctx) {
    let ctx2 = ctx.clone();
    // What "disabled" would destroy — the confirm below must name it
    // (Python's --disable-vision nulls the caption pair AND empties
    // the fallback chain; M2 review P2-1).
    let pair_set = field_view(ctx, "vision", "caption_provider")
        .map(|f| matches!(f.state, FieldState::Set))
        .unwrap_or(false);
    let chain_len = field_view(ctx, "vision", "fallback_chain")
        .and_then(|f| f.list_len)
        .unwrap_or(0);
    open_prompt(
        cx,
        ctx.ui,
        abstracttui::app::ChoicePrompt::new("Vision strategy")
            .option_detail(
                "two_stage",
                "two_stage — caption images with a vision model",
                "opens the caption provider/model pair editor",
            )
            .option_detail(
                "disabled",
                "disabled — no vision fallback",
                "ALSO clears the caption pair and the whole fallback chain",
            )
            .option_detail(
                "basic_metadata",
                "basic_metadata",
                "metadata only (no CLI setter — written directly)",
            ),
        move |outcome| {
            if let abstracttui::app::ChoiceOutcome::Answered(a) = outcome {
                match a.selected.first().map(String::as_str) {
                    Some("two_stage") => open_pair_editor_kind(cx, &ctx2, PairKind::VisionCaption),
                    Some("disabled") if pair_set || chain_len > 0 => {
                        // Destructive beyond the enum: confirm, naming
                        // exactly what is about to be erased.
                        let ctx3 = ctx2.clone();
                        super::forms::confirm_danger(
                            cx,
                            ctx2.ui,
                            format!(
                                "Disabling vision ALSO erases the caption pair and the \
                                 fallback chain ({chain_len} entries). Proceed?"
                            ),
                            "Disable and erase",
                            "Keep vision config",
                            move || {
                                match writes::set_vision_strategy(
                                    "disabled",
                                    ctx3.write_base(),
                                    None,
                                ) {
                                    Ok(spec) => ctx3.send(Cmd::Write(Box::new(spec))),
                                    Err(e) => ctx3.store.notice.set(Some(e)),
                                }
                            },
                        );
                    }
                    Some(s @ ("disabled" | "basic_metadata")) => {
                        match writes::set_vision_strategy(s, ctx2.write_base(), None) {
                            Ok(spec) => ctx2.send(Cmd::Write(Box::new(spec))),
                            Err(e) => ctx2.store.notice.set(Some(e)),
                        }
                    }
                    _ => {}
                }
            }
        },
    );
}

// ---------------------------------------------------------------------
// vision.fallback_chain: add (CLI append) / remove (direct write).
// ---------------------------------------------------------------------

fn open_chain_editor(cx: Scope, ctx: &Ctx, section: &'static str) {
    if section != "vision" {
        ctx.store
            .notice
            .set(Some("audio.fallback_chain is reserved — nothing to edit yet".into()));
        return;
    }
    // The REAL entry count from the folded array — counting '/' in the
    // display string double-counted `org/model` ids and made removal
    // impossible while one existed (M2 review P2-6).
    let chain_len = field_view(ctx, "vision", "fallback_chain")
        .and_then(|f| f.list_len)
        .unwrap_or(0);
    let ctx2 = ctx.clone();
    let ctx3 = ctx.clone();
    open_prompt(
        cx,
        ctx.ui,
        abstracttui::app::ChoicePrompt::new(format!(
            "vision.fallback_chain ({chain_len} entries)"
        ))
        .option("add", "Add an entry (provider/model)")
        .option_detail(
            "remove",
            "Remove the LAST entry",
            "the CLI can only append; removal is a direct write",
        ),
        move |outcome| {
            if let abstracttui::app::ChoiceOutcome::Answered(a) = outcome {
                match a.selected.first().map(String::as_str) {
                    Some("add") => open_chain_add(cx, &ctx2, chain_len),
                    Some("remove") => {
                        if chain_len == 0 {
                            ctx3.store
                                .notice
                                .set(Some("the chain is empty — nothing to remove".into()));
                            return;
                        }
                        let spec = writes::remove_vision_fallback(
                            chain_len - 1,
                            chain_len,
                            ctx3.write_base(),
                            None,
                        );
                        ctx3.send(Cmd::Write(Box::new(spec)));
                    }
                    _ => {}
                }
            }
        },
    );
}

fn open_chain_add(cx: Scope, ctx: &Ctx, old_len: usize) {
    let theme = use_theme(cx);
    let ctx2 = ctx.clone();
    let base = ctx.write_base();
    open_form_guarded(ctx, cx, Size::new(70, 11), move |mcx, close, _guard| {
        let t = theme.get().tokens;
        let pair = mcx.signal(String::new());
        let form_error: Signal<Option<String>> = mcx.signal(None);
        let in_flight = mcx.signal(false);
        let form_id = next_form_id();
        install_write_done(mcx, &ctx2, form_id, in_flight, form_error, close.clone());
        let ctx3 = ctx2.clone();
        let submit = move || {
            // Same double-submit guard every other editor has (M2
            // review P3-1) — a second Enter must not queue a second
            // append.
            if in_flight.get_untracked() {
                return;
            }
            let v = pair.get_untracked();
            let Some((p, m)) = v.trim().split_once('/') else {
                form_error.set(Some("format: provider/model".into()));
                return;
            };
            let spec = writes::add_vision_fallback(
                p.trim(),
                m.trim(),
                old_len + 1,
                base,
                Some(form_id),
            );
            in_flight.set(true);
            ctx3.send(Cmd::Write(Box::new(spec)));
        };
        let submit_btn = submit.clone();
        Block::new()
            .title("Add vision fallback")
            .layout(LayoutStyle::column().grow(1.0))
            .child(
                Element::new()
                    .style(LayoutStyle::column().gap(1))
                    .child(field(
                        &t,
                        "provider/model",
                        TextInput::new()
                            .layout(LayoutStyle::default().grow(1.0).h(1))
                            .value(pair)
                            .placeholder("ollama/qwen2.5vl")
                            .on_submit({
                                let s = submit.clone();
                                move |_| s()
                            })
                            .view(mcx),
                    ))
                    .child(message_slot(theme, form_error, in_flight))
                    .child(
                        Element::new()
                            .style(LayoutStyle::row().gap(2).shrink(0.0))
                            .child(Button::new("Add").on_click(submit_btn).view(mcx))
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
