//! Form plumbing: the single-modal slot, the dirty-Esc guard, the
//! write-completion routing, and the danger confirm — the sibling
//! console's proven invariants, carried over whole:
//!
//! - ONE modal at a time (stacked same-z modals give keys to the
//!   OLDEST layer — an invisible key owner).
//! - The first Esc on a dirty form WARNS and arms; the second
//!   discards; any edit disarms. One implementation, every form.
//! - A write's outcome routes back by form_id: close on success, stay
//!   open with data intact (and the verbatim error) on failure.

use std::cell::RefCell;
use std::rc::Rc;

use abstracttui::app::{ChoiceOutcome, ChoicePrompt, Modal};
use abstracttui::prelude::*;

use super::util::{line, span, span_bold};
use super::Ctx;

/// Modal close handle passed into form builders (the Modal object only
/// exists after `Modal::open` returns, while the builder runs inside).
pub type CloserFn = Rc<dyn Fn()>;

/// The Esc guard a form installs: return true to BLOCK this Esc (the
/// form warned and armed itself); false lets the modal close.
pub type GuardSlot = Rc<RefCell<Option<Box<dyn Fn() -> bool>>>>;

/// Open a form modal, closing any previous one. Esc closes through the
/// guard; the builder gets a `CloserFn` for its own Save/Cancel.
pub fn open_form_guarded(
    ctx: &Ctx,
    cx: Scope,
    size: Size,
    build: impl FnOnce(Scope, CloserFn, GuardSlot) -> View,
) {
    ctx.close_modal();
    let viewport = abstracttui::app::use_viewport(cx).get_untracked();
    let slot = ctx.modal.clone();
    let epoch = ctx.ui.modal_epoch;
    let closer: CloserFn = Rc::new(move || {
        if let Some(m) = slot.borrow_mut().take() {
            m.close();
        }
        epoch.update(|e| *e += 1);
    });
    let guard: GuardSlot = Rc::new(RefCell::new(None));
    let guard_esc = guard.clone();
    let c_esc = closer.clone();
    // Focus-init note (engine finding 1000): the dead-keys recipe
    // (.focusable().autofocus() on the content root) applies to modals
    // whose focusables mount AFTER an async load. Every form here
    // mounts its inputs synchronously, so Modal's own focus-init lands
    // on the first input — a wrapper autofocus would STEAL the
    // keyboard from it (typing until Tab would go nowhere).
    let modal = Modal::open(&ctx.overlays, cx, viewport, size, move |mcx| {
        Element::new()
            .style(LayoutStyle::fill())
            .shortcut(KeyChord::plain(Key::Escape), move |_| {
                if let Some(g) = guard_esc.borrow().as_ref() {
                    if g() {
                        return; // the form warned and armed itself
                    }
                }
                c_esc()
            })
            .child(build(mcx, closer.clone(), guard.clone()))
            .build()
    });
    *ctx.modal.borrow_mut() = Some(modal);
}

/// Open a ChoicePrompt — the one wrapper every prompt goes through
/// (0.2.22 routes input to the TOPMOST modal, so no stacking counter
/// is needed; keeping the single entry point preserves the seam if
/// that ever changes).
pub fn open_prompt(
    cx: Scope,
    _ui: super::UiState,
    prompt: ChoicePrompt,
    resolve: impl FnOnce(ChoiceOutcome) + 'static,
) {
    prompt.on_resolve(resolve).open(cx);
}

/// ONE danger confirm: message → one danger-tinted option → one keep
/// option, DEFAULTING to keep (structural, impossible to forget).
pub fn confirm_danger(
    cx: Scope,
    ui: super::UiState,
    message: String,
    danger_label: &str,
    keep_label: &str,
    on_confirm: impl FnOnce() + 'static,
) {
    open_prompt(
        cx,
        ui,
        ChoicePrompt::new(message)
            .option_with(abstracttui::app::ChoiceOption::new("go", danger_label).danger(true))
            .option("keep", keep_label)
            .initial("keep"),
        move |outcome| {
            if let ChoiceOutcome::Answered(a) = outcome {
                if a.selected.iter().any(|s| s == "go") {
                    on_confirm();
                }
            }
        },
    );
}

/// The dirty-form Esc warning — one string, compared verbatim by the
/// disarm effect so a REAL error is never cleared by accident.
pub const ESC_WARNING: &str = "unsaved changes — press Esc again to discard";

/// THE dirty-Esc contract: `dirty` runs UNTRACKED comparisons against
/// the form's initial values; `track` performs TRACKED reads of every
/// editable signal so the disarm effect re-runs on edits.
pub fn install_dirty_guard_with(
    mcx: Scope,
    guard: &GuardSlot,
    dirty: impl Fn() -> bool + 'static,
    track: impl Fn() + 'static,
    esc_armed: Signal<bool>,
    form_error: Signal<Option<String>>,
) {
    *guard.borrow_mut() = Some(Box::new(move || {
        if !dirty() || esc_armed.get_untracked() {
            return false;
        }
        esc_armed.set(true);
        form_error.set(Some(ESC_WARNING.into()));
        true
    }));
    mcx.effect(move || {
        track();
        if esc_armed.get_untracked() {
            esc_armed.set(false);
            if form_error.with_untracked(|e| e.as_deref() == Some(ESC_WARNING)) {
                form_error.set(None);
            }
        }
    });
}

/// String-pairs convenience for forms whose whole state is text fields.
pub fn install_dirty_guard(
    mcx: Scope,
    guard: &GuardSlot,
    fields: Vec<(Signal<String>, String)>,
    esc_armed: Signal<bool>,
    form_error: Signal<Option<String>>,
) {
    let fields2 = fields.clone();
    install_dirty_guard_with(
        mcx,
        guard,
        move || fields.iter().any(|(s, init)| s.get_untracked() != *init),
        move || {
            for (s, _) in &fields2 {
                let _ = s.get();
            }
        },
        esc_armed,
        form_error,
    );
}

/// write_done routing, one implementation: close on success, verbatim
/// error on failure, in-flight released either way.
pub fn install_write_done(
    mcx: Scope,
    ctx: &Ctx,
    form_id: u64,
    in_flight: Signal<bool>,
    form_error: Signal<Option<String>>,
    close: CloserFn,
) {
    let ui = ctx.ui;
    mcx.effect(move || {
        if let Some((fid, outcome)) = ui.write_done.get() {
            if fid == form_id {
                ui.write_done.set(None);
                in_flight.set(false);
                match outcome {
                    Ok(_) => close(),
                    Err(e) => form_error.set(Some(e)),
                }
            }
        }
    });
}

/// The message line every write form shows: error/warning WINS over
/// the busy line, then busy, then blank. Pinned — the dirty-Esc
/// warning disappearing under pressure would make the second Esc
/// silently destructive.
pub fn message_slot(
    theme: Signal<&'static abstracttui::theme::Theme>,
    form_error: Signal<Option<String>>,
    in_flight: Signal<bool>,
) -> View {
    dyn_view(LayoutStyle::line(1).shrink(0.0), move || {
        let t = theme.get().tokens;
        if let Some(e) = form_error.get() {
            return line(vec![span_bold(format!("✗ {e}"), t.error)]);
        }
        if in_flight.get() {
            return line(vec![span(
                "⟳ applying… (write + verify by re-read)",
                t.info,
            )]);
        }
        line(vec![span(String::new(), t.text)])
    })
}
