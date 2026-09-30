//! The terminal switch: one persistent on/off setting, labelled by the
//! FEATURE and showing its state (the framework's state-toggle rule).
//!
//! `[x] Agent email tools` ON (accent + bold) · `[ ] Agent email tools`
//! OFF (plain) · `[-] Agent email tools — Connect a mailbox first.`
//! UNAVAILABLE (faint, the reason after an em dash). Space or Enter
//! switches while focused; a click switches. An unavailable switch stays
//! focusable (so a keyboard user can reach it and read why) and answers
//! a switch attempt with its reason instead of a change.
//!
//! The switch never stores anything itself: `state` reads the live
//! document (a tracked read — the row repaints when the document
//! changes) and `on_switch` receives the NEW state to apply. The
//! caller's write re-reads the document, so a refused write shows the
//! old state again on its own.

use std::cell::RefCell;
use std::rc::Rc;

use abstracttui::base::Point;
use abstracttui::prelude::*;
use abstracttui::render::Style;
use abstracttui::ui::{MouseButton, MouseKind, Phase, UiEvent};

use super::util::{fit_width, switch_spans, Switch};

type StateFn = Rc<dyn Fn() -> Switch>;
type SwitchFn = Rc<RefCell<Box<dyn FnMut(bool)>>>;
type RefusedFn = Rc<RefCell<Box<dyn FnMut(String)>>>;

pub struct SwitchRow {
    label: String,
    state: StateFn,
    on_switch: SwitchFn,
    on_refused: RefusedFn,
}

impl SwitchRow {
    /// `state` is read inside the row's own reactive view (tracked);
    /// `on_switch(new_state)` runs on Space / Enter / click when the
    /// switch is available.
    pub fn new(
        label: impl Into<String>,
        state: impl Fn() -> Switch + 'static,
        on_switch: impl FnMut(bool) + 'static,
    ) -> SwitchRow {
        SwitchRow {
            label: label.into(),
            state: Rc::new(state),
            on_switch: Rc::new(RefCell::new(Box::new(on_switch))),
            on_refused: Rc::new(RefCell::new(Box::new(|_| {}))),
        }
    }

    /// What a switch attempt on an UNAVAILABLE row does with its reason
    /// (the screens post it to the status line).
    pub fn on_refused(mut self, f: impl FnMut(String) + 'static) -> SwitchRow {
        self.on_refused = Rc::new(RefCell::new(Box::new(f)));
        self
    }

    pub fn view(self, cx: Scope) -> View {
        let theme = use_theme(cx);
        let focused = cx.signal(false);
        let label = self.label;
        let state = self.state;
        let attempt = {
            let state = state.clone();
            let on_switch = self.on_switch.clone();
            let on_refused = self.on_refused.clone();
            move || match untrack(|| state()) {
                Switch::On => (on_switch.borrow_mut())(false),
                Switch::Off => (on_switch.borrow_mut())(true),
                Switch::Unavailable(reason) => (on_refused.borrow_mut())(reason),
            }
        };
        let access_state = state.clone();
        let paint_state = state.clone();
        let paint_label = label.clone();
        Element::new()
            .style(LayoutStyle::line(1).shrink(0.0))
            .role(abstracttui::ui::Role::Checkbox)
            .access_label(label.clone())
            .access_value(move || match untrack(|| access_state()) {
                Switch::On => "on".into(),
                Switch::Off => "off".into(),
                Switch::Unavailable(r) => format!("unavailable: {r}"),
            })
            .focusable()
            .focus_signal(focused)
            .on(Phase::Bubble, move |ctx, ev| match ev {
                UiEvent::Key(k) if k.key == Key::Enter || k.key == Key::Char(' ') => {
                    if focused.get_untracked() {
                        attempt();
                        ctx.stop_propagation();
                    }
                }
                UiEvent::Mouse(m) if matches!(m.kind, MouseKind::Down(MouseButton::Left)) => {
                    attempt();
                    ctx.stop_propagation();
                }
                _ => {}
            })
            .child(dyn_view(LayoutStyle::line(1), move || {
                let t = theme.get().tokens;
                let st = paint_state();
                let focus = focused.get();
                let spans = switch_spans(&t, &paint_label, &st);
                let (sel_fg, sel_bg) = (t.selection_fg, t.selection_bg);
                Element::new()
                    .style(LayoutStyle::line(1))
                    .draw(move |canvas, rect| {
                        if rect.is_empty() {
                            return;
                        }
                        let mut x = rect.x;
                        let right = rect.x + rect.w;
                        for (text, ink, bold) in &spans {
                            if x >= right {
                                break;
                            }
                            let mut style = if focus {
                                Style::new().fg(sel_fg).bg(sel_bg)
                            } else {
                                Style::new().fg(*ink)
                            };
                            if *bold {
                                style = style.bold();
                            }
                            let fitted = fit_width(text, (right - x).max(0) as usize);
                            canvas.print_styled(Point::new(x, rect.y), &fitted, &style);
                            x += abstracttui::text::width(&fitted);
                        }
                    })
                    .build()
            }))
            .build()
    }
}
