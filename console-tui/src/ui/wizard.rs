//! Wizard mode: a guided linear walk covering the CLI wizard's 8
//! phases (main.py:730-975 — default model → vision → API keys →
//! server → audio → video → embeddings → logging), in the CLI's own
//! order, plus an orientation step and the review. Every phase is
//! OPTIONAL (exactly like the CLI's [y/N] prompts) — Ctrl+N always
//! advances; the goal line says what the step is for and when to skip.
//!
//! Steps target (screen, focused section): the section pages filter to
//! the focused section while the wizard walks, so each phase reads as
//! one purposeful form rather than a management table in a wizard
//! costume (the sibling's cycle-1 lesson).

use super::Ctx;

#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub screen: usize,
    /// The section this step edits (section pages filter to it).
    pub focus: Option<&'static str>,
    pub title: &'static str,
    pub goal: &'static str,
}

pub const STEPS: &[Step] = &[
    Step {
        screen: 0,
        focus: None,
        title: "Overview",
        goal: "this is your config file and its honest state — Ctrl+N starts the walk; every step is optional",
    },
    Step {
        screen: 1,
        focus: Some("default_models"),
        title: "Default model (1/8)",
        goal: "pick the global provider/model — Enter on global_provider opens the pair editor (also writes route input.text)",
    },
    Step {
        screen: 4,
        focus: Some("vision"),
        title: "Vision (2/8)",
        goal: "optional: a caption model lets text-only models see images — Enter on strategy; skip with Ctrl+N",
    },
    Step {
        screen: 2,
        focus: Some("api_keys"),
        title: "API keys (3/8)",
        goal: "optional: store cloud keys (k on a row) — keys stored here OVERRIDE the environment",
    },
    Step {
        screen: 6,
        focus: Some("server"),
        title: "Server (4/8)",
        goal: "optional: only needed if you run `abstractcore serve` — token, allowlists, host/port",
    },
    Step {
        screen: 4,
        focus: Some("audio"),
        title: "Audio (5/8)",
        goal: "how audio inputs are handled — setting a strategy marks it explicit; skip to keep the smart default",
    },
    Step {
        screen: 4,
        focus: Some("video"),
        title: "Video (6/8)",
        goal: "video strategy + frame knobs — the defaults are sane; skip unless you know you need this",
    },
    Step {
        screen: 5,
        focus: Some("embeddings"),
        title: "Embeddings (7/8)",
        goal: "the embedding pair (Enter on provider opens the pair editor) — mirrors into route embedding.text",
    },
    Step {
        screen: 6,
        focus: Some("logging"),
        title: "Logging (8/8)",
        goal: "console verbosity (Enter on console_level) — file logging and the json knobs live here too",
    },
    Step {
        screen: 7,
        focus: None,
        title: "Review",
        goal: "PROVE it: g runs a cheap generation over your default route; the journal + Python-agreement lines are below — Finish (f) switches to browse",
    },
];

/// Apply step `i`: set the screen and the section focus.
pub fn apply_step(ctx: &Ctx, i: usize) {
    let i = i.min(STEPS.len() - 1);
    let step = &STEPS[i];
    ctx.ui.step.set(i);
    ctx.ui.screen.set(step.screen);
    ctx.ui.focus_section.set(step.focus);
}

pub fn next(ctx: &Ctx) {
    let i = ctx.ui.step.get_untracked();
    if i + 1 >= STEPS.len() {
        ctx.store.notice.set(Some(
            "last step — f finishes the wizard (browse mode); Ctrl+C quits".into(),
        ));
        return;
    }
    apply_step(ctx, i + 1);
}

pub fn back(ctx: &Ctx) {
    let i = ctx.ui.step.get_untracked();
    if i == 0 {
        ctx.store.notice.set(Some("already on the first step".into()));
        return;
    }
    apply_step(ctx, i - 1);
}

/// Leave the wizard for browse mode (the free surface re-arms).
pub fn finish(ctx: &Ctx) {
    ctx.ui.wizard.set(false);
    ctx.ui.focus_section.set(None);
    ctx.store.notice.set(Some(
        "browse mode — 1-8 jump screens, Enter edits, q quits".into(),
    ));
}
