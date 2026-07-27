//! abstractcore-console: the AbstractCore configuration console.
//!
//! Architecture (the proven gateway-console shape): the UI thread owns
//! all signals; one worker thread owns every file read and every
//! `abstractcore` subprocess. Commands cross via mpsc, results come
//! back as closures posted through `WakeHandle` (the engine's
//! live-data law).

pub mod cli;
pub mod config;
pub mod models;
pub mod probes;
pub mod schema;
pub mod store;
pub mod ui;
pub mod worker;
pub mod writes;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

use abstracttui::prelude::*;

use store::Store;
use ui::{Ctx, UiState};

pub const ENGINE_VERSION: &str = "0.2.22";

const HELP: &str = "\
abstractcore-console — configure AbstractCore from the terminal

USAGE:
  abstractcore-console [--wizard|--browse] [--theme ID]

The console mirrors ~/.abstractcore/config/abstractcore.json (honoring
ABSTRACTCORE_CONFIG_FILE / ABSTRACTCORE_CONFIG_DIR) and drives the
abstractcore CLI ($ABSTRACTCORE_BIN, PATH, or the framework venv).
Writes go through the CLI setters (coupled fields) or a direct
unknown-key-preserving rewrite (CLI-less fields), and every write is
verified by re-reading.

OPTIONS:
  --wizard       start in the guided wizard (default when no config
                 file exists yet)
  --browse       start in browse mode (default with an existing config)
  --theme ID     abstracttui theme id (also $ABSTRACTTUI_THEME)
  -h, --help     this help
  --version      print the version

KEYS: 1-8 screens (browse) · Ctrl+N/P next/prev · Tab focus ·
      Enter/e edit · x clear · k set key · w wizard · f finish wizard ·
      r reload · Ctrl+L repaint · q (browse) / Ctrl+C quit
";

struct Args {
    theme: Option<String>,
    /// None = adaptive (wizard on a fresh machine, browse otherwise).
    wizard: Option<bool>,
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut theme = None;
    let mut wizard = None;
    let mut it = argv.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(None);
            }
            "--version" => {
                println!("abstractcore-console {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "--wizard" => wizard = Some(true),
            "--browse" => wizard = Some(false),
            "--theme" => theme = Some(it.next().cloned().ok_or("--theme needs a value")?),
            other => return Err(format!("unknown argument: {other} (see --help)")),
        }
    }
    Ok(Some(Args { theme, wizard }))
}

/// CLI entry — returns the process exit code.
pub fn run_cli(argv: &[String]) -> i32 {
    let args = match parse_args(argv) {
        Ok(Some(a)) => a,
        Ok(None) => return 0,
        Err(e) => {
            eprintln!("abstractcore-console: {e}");
            return 2;
        }
    };

    // Headless guard (CI / piped runs): skip cleanly, exit 0.
    if !abstracttui::term::have_tty() {
        println!("abstractcore-console: needs an interactive terminal — skipping cleanly");
        return 0;
    }

    if let Some(id) = args
        .theme
        .or_else(|| std::env::var("ABSTRACTTUI_THEME").ok())
    {
        set_theme_by_id(&id);
    }

    // Resolve the two identities BEFORE mount (cheap: env + PATH scan +
    // existence checks) so the first paint can already name them.
    let config_path = config::resolve_config_path_from_env();
    let cli_info = cli::resolve_bin_from_env();

    // Adaptive mode default: a machine with no config file gets the
    // wizard (the definition-of-done scenario); an existing config
    // opens in browse. Flags override.
    let start_wizard = args.wizard.unwrap_or_else(|| {
        matches!(config::load(&config_path.path), config::FileState::Missing)
    });

    let mut app = App::new(Size::new(110, 32));
    let overlays = app.overlays();
    let quitter = app.quitter();
    let (tx, rx) = mpsc::channel::<worker::Cmd>();

    // Ctrl+L as a GLOBAL ACTION, not a root shortcut: global actions
    // resolve last in the key path, so the repaint works even inside a
    // focus-trapped modal (a root shortcut is not on a modal's focus
    // path — the repaint key an app owes its users must never go dead
    // behind a form).
    app.actions().register(
        "repaint (full redraw)",
        Some(KeyChord::new(Mods::CTRL, Key::Char('l'))),
        abstracttui::app::request_full_redraw,
    );

    let store_slot: Rc<RefCell<Option<Store>>> = Rc::new(RefCell::new(None));
    let store_out = store_slot.clone();
    let ui_slot: Rc<RefCell<Option<UiState>>> = Rc::new(RefCell::new(None));
    let ui_out = ui_slot.clone();

    let tx_mount = tx.clone();
    let cli_for_store = cli_info.clone();
    if let Err(e) = app.mount(move |cx| {
        let store = Store::create(cx);
        store.cli.set(cli_for_store.clone());
        *store_out.borrow_mut() = Some(store);
        let ui_state = UiState::create(cx);
        ui_state.wizard.set(start_wizard);
        *ui_out.borrow_mut() = Some(ui_state);
        let ctx = Ctx {
            tx: tx_mount.clone(),
            overlays: overlays.clone(),
            quitter: quitter.clone(),
            store,
            ui: ui_state,
            modal: Rc::new(RefCell::new(None)),
        };
        if start_wizard {
            ui::wizard::apply_step(&ctx, 0);
        }
        ui::root(cx, ctx)
    }) {
        eprintln!("abstractcore-console: mount failed: {e}");
        return 1;
    }

    // Worker thread: owns file reads/writes + subprocesses; posts
    // results back. The done sink marshals write completions to the
    // one open form through the UI signal.
    let wake = abstracttui::reactive::wake_handle();
    let store = store_slot.borrow().expect("store created");
    let ui_state = ui_slot.borrow().expect("ui state created");
    let done_sink: worker::DoneSink = {
        let wake = wake.clone();
        Box::new(move |form_id, outcome| {
            let ui = ui_state;
            wake.post(move || ui.write_done.set(Some((form_id, outcome.clone()))));
        })
    };
    let core_cli = cli_info.map(|i| cli::CoreCli::new(i.bin));
    let worker_handle = worker::spawn(store, wake, rx, config_path, core_cli, done_sink);

    // Boot load: zero-keystroke first paint of real state.
    store.cfg.set(store::Loadable::Loading);
    store.routes.set(store::Loadable::Loading);
    store.profiles.set(store::Loadable::Loading);
    let _ = tx.send(worker::Cmd::LoadConfig);
    let _ = tx.send(worker::Cmd::LoadRoutes);
    let _ = tx.send(worker::Cmd::LoadProfiles);

    let result = app.run();
    // Drop the sender so an idle worker unblocks and ends. Deliberately
    // NO join: a worker mid-subprocess would hang quit with the
    // terminal already restored — process exit reaps the thread. The
    // q key refuses while a write is in flight (the honest guard);
    // Ctrl+C remains the force-quit, abandoning at most one write
    // whose setter completes Python-side unverified.
    drop(tx);
    drop(worker_handle);

    match result {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("abstractcore-console: {e}");
            1
        }
    }
}
