//! The shared model field for pair/route editors: a discovery-backed
//! Combobox filtered to the class the field is FOR (embedding models
//! for embeddings, generative for text pairs — `crate::models`), with
//! a free-typing lane that stays the source of truth.
//!
//! Honesty rules:
//! - The picker only ever offers DISCOVERED ids; while discovery is
//!   loading/failed/absent the row is a plain TextInput with the state
//!   named beside it — never an empty dropdown pretending to know.
//! - The class filter is a name heuristic: hidden models are COUNTED
//!   on the status line, a whiffed filter falls back to the full list
//!   (labeled), and custom typing can always enter any id.
//! - `model_text` is the one value: Combobox commits write it, custom
//!   mode edits it directly, and a prefill that discovery does not
//!   list keeps the field in custom mode instead of blanking it.

use std::rc::Rc;

use abstracttui::prelude::*;

use crate::models::{filter_for, ModelClass};
use crate::store::{Loadable, Store};

use super::util::{field, line, span};

/// Kick model discovery for a provider unless the domain already
/// holds it — shared by every editor that shows the picker, including
/// at form OPEN for prefilled providers.
pub fn kick_discovery(ctx: &super::Ctx, provider: &str) {
    let known = ctx
        .store
        .models
        .with_untracked(|m| m.contains_key(provider));
    if !known {
        let p = provider.to_string();
        ctx.store.models.update(|m| {
            m.insert(p.clone(), Loadable::Loading);
        });
        ctx.send(crate::worker::Cmd::LoadModels { provider: p });
    }
}

pub struct ModelField {
    pub view: View,
    /// `m` (or any caller shortcut) opens the picker popup when the
    /// row is in picker mode; refuses (false) in custom mode.
    pub handle: SelectHandle,
    /// None = auto (picker when discovery lists the current value or
    /// the field is empty); Some(true) = free typing; Some(false) =
    /// picker. Callers may flip it from shortcuts.
    pub custom: Signal<Option<bool>>,
}

/// Everything the model row needs from its host form. `providers` and
/// `provider_sel` mirror the provider Select above it (index 0 = the
/// placeholder); `submit` runs on Enter in custom mode (form-submit
/// parity with the old TextInput).
pub struct ModelFieldSpec {
    pub store: Store,
    pub providers: Rc<Vec<String>>,
    pub provider_sel: Signal<usize>,
    pub model_text: Signal<String>,
    pub class: ModelClass,
    pub submit: Option<Rc<dyn Fn()>>,
}

/// Build the model row + its status line.
pub fn model_field(
    mcx: Scope,
    theme: Signal<&'static abstracttui::theme::Theme>,
    spec: ModelFieldSpec,
) -> ModelField {
    let ModelFieldSpec {
        store,
        providers,
        provider_sel,
        model_text,
        class,
        submit,
    } = spec;
    let handle = SelectHandle::new();
    let custom: Signal<Option<bool>> = mcx.signal(None);
    let handle2 = handle.clone();

    let view = dyn_view_scoped(LayoutStyle::column().shrink(0.0), move |gcx| {
        let t = theme.get().tokens;
        let i = provider_sel.get();
        let provider = (i > 0).then(|| providers[i - 1].clone());
        let discovery = provider
            .as_ref()
            .and_then(|p| store.models.with(|m| m.get(p).cloned()));
        // UNTRACKED on purpose: the Combobox owns its own display, and
        // a TRACKED read would regenerate this row on every commit —
        // destroying the focused control mid-Tab-flow (live-caught:
        // Tab after a commit landed on the provider Select because the
        // Combobox that had focus was rebuilt). The value only shapes
        // mode/initial-index decisions, which the REAL regen triggers
        // (provider change, discovery landing, mode flip) re-derive.
        let current = model_text.get_untracked();

        // The filtered option set, when discovery is usable.
        let filtered = match &discovery {
            Some(Loadable::Ready(models)) if !models.is_empty() => {
                Some(filter_for(class, models))
            }
            _ => None,
        };

        // Mode: explicit choice wins; auto = picker unless the current
        // value is one discovery does not list (a picker that cannot
        // display the real value would lie about it).
        let auto_custom = match (&filtered, current.trim()) {
            (Some((options, _)), text) if !text.is_empty() => {
                !options.iter().any(|m| m == text)
            }
            _ => false,
        };
        let custom_now = custom.get().unwrap_or(auto_custom);
        let picker_mode = filtered.is_some() && !custom_now;

        let class_word = match class {
            ModelClass::Embedding => "embedding",
            ModelClass::Generative => "generative",
        };

        let (row, status) = if picker_mode {
            let (options, hidden) = filtered.clone().expect("picker_mode implies filtered");
            let whiffed = hidden == 0
                && options
                    .iter()
                    .any(|m| !crate::models::matches_class(m, class));
            let initial = options
                .iter()
                .position(|m| *m == current)
                .unwrap_or(usize::MAX);
            let idx = gcx.signal(initial);
            let mut opts: Vec<SelectOption> = options
                .iter()
                .map(|m| SelectOption::new(m.clone()))
                .collect();
            opts.push(SelectOption::new("✎ type a custom id…"));
            let n = options.len();
            let options2 = options.clone();
            let row = field(
                &t,
                "model",
                Combobox::new(opts)
                    .value(idx)
                    .placeholder(format!("pick from {n} {class_word} models"))
                    .handle(&handle2)
                    .on_change(move |k| {
                        if k < options2.len() {
                            // No mode write here: commit implies picker
                            // mode already, and a signal write would
                            // regenerate the row under the focused
                            // trigger (the Tab-flow breaker above).
                            model_text.set(options2[k].clone());
                        } else {
                            custom.set(Some(true));
                        }
                    })
                    .layout(LayoutStyle::default().grow(1.0))
                    .view(gcx),
            );
            let status = if whiffed {
                line(vec![span(
                    format!(
                        " no {class_word}-shaped names among {n} discovered — showing all \
                         (name heuristic; c types a custom id)"
                    ),
                    t.warn,
                )])
            } else if hidden > 0 {
                line(vec![span(
                    format!(
                        " {n} {class_word} models · {hidden} other hidden · m opens · c types \
                         a custom id"
                    ),
                    t.text_faint,
                )])
            } else {
                line(vec![span(
                    format!(" {n} {class_word} models · m opens · c types a custom id"),
                    t.text_faint,
                )])
            };
            (row, status)
        } else {
            let submit2 = submit.clone();
            let mut input = TextInput::new()
                .layout(LayoutStyle::default().grow(1.0).h(1))
                .value(model_text)
                .placeholder("type a model id");
            if let Some(s) = submit2 {
                input = input.on_submit(move |_| s());
            }
            let row = field(&t, "model", input.view(gcx));
            let status = match (&provider, &discovery) {
                (None, _) => line(vec![span(
                    " models: choose a provider to discover",
                    t.text_faint,
                )]),
                (_, Some(Loadable::Loading)) => {
                    line(vec![span(" models: ⟳ discovering…", t.info)])
                }
                (_, Some(Loadable::Failed(e))) => line(vec![span(
                    format!(
                        " models: discovery failed — {} (typing stays open)",
                        e.headline()
                    ),
                    t.warn,
                )]),
                (_, Some(Loadable::Ready(models))) if models.is_empty() => line(vec![span(
                    " models: none reported — type the id by hand",
                    t.text_muted,
                )]),
                (_, Some(Loadable::Ready(models))) => {
                    // Custom mode over usable discovery.
                    let (options, _) = filter_for(class, models);
                    line(vec![span(
                        format!(
                            " typing freely — Ctrl+P returns to the {}-model picker",
                            options.len()
                        ),
                        t.text_faint,
                    )])
                }
                _ => line(vec![span(" models: not loaded", t.text_faint)]),
            };
            (row, status)
        };

        Element::new()
            .style(LayoutStyle::column())
            .child(row)
            .child(status)
            .build()
    });

    ModelField {
        view,
        handle,
        custom,
    }
}
