//! Headless UI tests: the REAL interface driven through AbstractTUI's
//! capture harness — same pipeline as production, no pty, no network,
//! and no real config file (fixtures are applied to the store between
//! frames exactly as the worker's posted closures would).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

use abstracttui::app::Driver;
use abstracttui::prelude::*;
use abstracttui::testing::CaptureTerm;
use serde_json::{json, Value};

use abstractcore_console::config::{self, ConfigPath, FileState, PathSource};
use abstractcore_console::store::{ConfigMirror, Loadable, ProfilesData, RoutesData, Store};
use abstractcore_console::ui::{self, Ctx, UiState};
use abstractcore_console::worker::Cmd;

struct Harness {
    app: App,
    term: CaptureTerm,
    driver: Driver,
    store: Store,
    ui: UiState,
    rx: mpsc::Receiver<Cmd>,
}

fn harness() -> Harness {
    harness_sized(Size::new(110, 34))
}

fn harness_sized(size: Size) -> Harness {
    abstracttui::app::set_theme_by_id("abstract-dark");
    let mut app = App::new(size);
    let overlays = app.overlays();
    let quitter = app.quitter();
    let (tx, rx) = mpsc::channel::<Cmd>();
    let store_slot: Rc<RefCell<Option<Store>>> = Rc::new(RefCell::new(None));
    let store_out = store_slot.clone();
    let ui_slot: Rc<RefCell<Option<UiState>>> = Rc::new(RefCell::new(None));
    let ui_out = ui_slot.clone();
    app.mount(move |cx| {
        let store = Store::create(cx);
        *store_out.borrow_mut() = Some(store);
        let ui_state = UiState::create(cx);
        *ui_out.borrow_mut() = Some(ui_state);
        let ctx = Ctx {
            tx: tx.clone(),
            overlays: overlays.clone(),
            quitter: quitter.clone(),
            store,
            ui: ui_state,
            modal: Rc::new(RefCell::new(None)),
        };
        ui::root(cx, ctx)
    })
    .expect("mount");
    let mut term = CaptureTerm::new(size);
    let cfg = RunConfig {
        probe: false,
        // Fixed capabilities: the host's TERM must not steer assertions.
        caps: Some(abstracttui::term::Capabilities::with(|c| {
            c.truecolor = true;
            c.colors_256 = true;
            c.unicode_ok = true;
        })),
        ..RunConfig::default()
    };
    let driver = Driver::new(&mut app, &mut term, cfg).expect("driver");
    let store = store_slot.borrow().expect("store created");
    let ui = ui_slot.borrow().expect("ui state created");
    Harness {
        app,
        term,
        driver,
        store,
        ui,
        rx,
    }
}

impl Harness {
    fn turn(&mut self) -> String {
        self.driver
            .turn(&mut self.app, &mut self.term)
            .expect("turn");
        self.term.screen().to_text()
    }

    /// One turn = one render pass; an effect triggered by a signal
    /// write runs on the NEXT pass — `turns(n)` is the effect-depth of
    /// the change under test, not magic.
    fn turns(&mut self, n: usize) -> String {
        let mut last = String::new();
        for _ in 0..n {
            last = self.turn();
        }
        last
    }

    fn key(&mut self, bytes: &[u8]) {
        self.term.push_input(bytes);
    }

    fn type_text(&mut self, text: &str) {
        self.term.push_input(text.as_bytes());
    }

    fn press_escape(&mut self) {
        // Bare-ESC disambiguation: byte arrives, ~30ms deadline, then
        // the parser resolves it as a lone Escape.
        self.term.push_input(&[0x1b]);
        self.turn();
        std::thread::sleep(std::time::Duration::from_millis(45));
        self.turn();
    }

    /// Drain EVERY queued worker command — negative asserts must
    /// assert emptiness, not absence-of-match.
    fn drain_cmds(&mut self) -> Vec<Cmd> {
        let mut out = Vec::new();
        while let Ok(c) = self.rx.try_recv() {
            out.push(c);
        }
        out
    }

    fn load_fixtures(&mut self) {
        self.store
            .cfg
            .set(Loadable::Ready(mirror_of(config_fixture_value())));
        self.store
            .routes
            .set(Loadable::Ready(RoutesData::from_value(&routes_fixture())));
        self.store
            .profiles
            .set(Loadable::Ready(ProfilesData::from_value(
                &profiles_fixture(),
            )));
        self.turns(2);
    }

    fn goto_screen(&mut self, n: usize) -> String {
        self.ui.screen.set(n);
        self.turns(3)
    }
}

fn test_path() -> ConfigPath {
    ConfigPath {
        path: "/tmp/console-test/abstractcore.json".into(),
        source: PathSource::Default,
    }
}

fn mirror_of(raw: Value) -> ConfigMirror {
    ConfigMirror {
        path: test_path(),
        state: FileState::Ready(config::fold(&raw, 4321, None, Some(0o600))),
        loaded_at: "12:00:00Z".into(),
    }
}

/// Heavy, shape-faithful config: every section present (Python's save
/// writes all of them), several set values, two broken fields, one
/// secret, one unknown section, one unknown key.
fn config_fixture_value() -> Value {
    json!({
        "audio_strategy_explicit": false,
        "vision": {"strategy": "disabled", "caption_provider": null,
                   "caption_model": null, "fallback_chain": [], "local_models_path": null},
        "audio": {"strategy": "auto", "stt_backend_id": null, "stt_language": null,
                  "caption_provider": null, "caption_model": null, "fallback_chain": []},
        "video": {"strategy": "auto", "max_frames": "three", "max_frames_native": 8,
                  "frame_format": "jpg", "sampling_strategy": "uniform",
                  "max_frame_side": 1024, "max_video_size_bytes": null,
                  "brand_new_knob": true},
        "embeddings": {"provider": "lmstudio",
                       "model": "text-embedding-qwen3-embedding-0.6b", "base_url": null},
        "app_defaults": {
            "cli_provider": "huggingface", "cli_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "summarizer_provider": "huggingface", "summarizer_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "extractor_provider": "huggingface", "extractor_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "judge_provider": "huggingface", "judge_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
            "intent_provider": "huggingface", "intent_model": "unsloth/Qwen3-4B-Instruct-2507-GGUF"
        },
        "default_models": {"global_provider": "lmstudio",
                           "global_model": "qwen/qwen3.6-35b-a3b",
                           "chat_model": null, "code_model": null},
        "capability_defaults": {"version": 1, "routes": {
            "input.text": {"provider": "lmstudio", "model": "qwen3-0.6b",
                            "base_url": "http://localhost:1234/v1"},
            "output.voice": {"provider": "supertonic", "model": "supertonic-3",
                              "options": {"voice": "M2"}},
            "embedding.text": {"provider": "lmstudio",
                                "model": "text-embedding-qwen3-embedding-0.6b"}
        }},
        "provider_profiles": {"profiles": {
            "ovh-provider": {"id": "ovh-provider", "display_name": "OVH Provider",
                "description": "hosted", "provider_family": "openai-compatible",
                "base_url": "https://oai.example.net/v1",
                "api_key": "sk-live-topsecret-body", "api_key_env_var": "",
                "allowed_models": [], "enabled": true,
                "created_at": "2026-06-02T19:32:36Z", "updated_at": "2026-06-02T19:32:36Z"}
        }},
        "api_keys": {"openai": "sk-test-secret-abc", "anthropic": null,
                     "openrouter": null, "portkey": null, "openai_compatible": null,
                     "vllm": null, "google": null},
        "server": {"auth_token": "tok-super-secret", "allow_unauthenticated": true,
                   "base_url_allowlist": null, "url_fetch_allowlist": null,
                   "media_root": null, "allow_local_files": false,
                   "host": null, "port": 8000},
        "cache": {"default_cache_dir": "~/.cache/abstractcore",
                  "huggingface_cache_dir": "~/.cache/huggingface",
                  "local_models_cache_dir": "~/.abstractcore/models",
                  "glyph_cache_dir": "~/.abstractcore/glyph_cache"},
        "logging": {"console_level": "chatty", "file_level": "DEBUG",
                    "file_logging_enabled": false, "log_base_dir": null,
                    "verbatim_enabled": true, "console_json": false, "file_json": true},
        "streaming": {"cli_stream_default": false},
        "timeouts": {"default_timeout": 600.0, "tool_timeout": 600.0},
        "offline": {"offline_first": true, "allow_network": false,
                    "force_local_files_only": true},
        "maintenance": {"triage_llm_enabled": false,
                        "triage_llm_base_url": "http://localhost:1234",
                        "triage_llm_model": "qwen/qwen3-next-80b",
                        "triage_llm_temperature": 0.2, "triage_llm_max_tokens": 800,
                        "triage_llm_timeout_s": 30.0},
        "email": {"smtp_host": "", "smtp_port": 587, "smtp_username": "",
                  "smtp_password_env_var": "EMAIL_PASSWORD", "smtp_use_starttls": true,
                  "from_email": null, "reply_to": null, "imap_host": "",
                  "imap_port": 993, "imap_username": "",
                  "imap_password_env_var": "EMAIL_PASSWORD", "imap_folder": "INBOX"},
        "future_section": {"anything": 1}
    })
}

/// Shape-faithful to `abstractcore config defaults --json` (live probe
/// 2026-07-25), covering all 24 route keys with a realistic mix.
fn routes_fixture() -> Value {
    let mut routes = vec![
        json!({"key": "input.text", "kind": "input", "modality": "text",
               "label": "Text Input", "provider": "lmstudio", "model": "qwen3-0.6b",
               "base_url": "http://localhost:1234/v1", "configured": true,
               "source": "abstractcore.capability_defaults"}),
        json!({"key": "input.image", "kind": "input", "modality": "image",
               "label": "Image Input", "provider": "lmstudio", "model": "qwen3-0.6b",
               "configured": true, "covered_by": "input.text", "read_only": true,
               "source": "abstractcore.capability_defaults"}),
        json!({"key": "input.video", "kind": "input", "modality": "video",
               "label": "Video Input", "provider": "lmstudio", "model": "qwen3-0.6b",
               "configured": true, "covered_by": "input.text", "overrideable": true,
               "source": "abstractcore.capability_defaults"}),
        json!({"key": "output.text", "kind": "output", "modality": "text",
               "label": "Text Output", "provider": "lmstudio", "model": "qwen3-0.6b",
               "configured": true, "source": "abstractcore.capability_defaults"}),
        json!({"key": "output.voice", "kind": "output", "modality": "voice",
               "label": "Voice Output", "provider": "supertonic", "model": "supertonic-3",
               "options": {"voice": "M2"}, "configured": true,
               "source": "abstractcore.capability_defaults"}),
        json!({"key": "embedding.text", "kind": "embedding", "modality": "text",
               "label": "Text Embeddings", "provider": "lmstudio",
               "model": "text-embedding-qwen3-embedding-0.6b", "configured": true,
               "source": "abstractcore.capability_defaults"}),
    ];
    for key in [
        "input.voice",
        "input.sound",
        "input.music",
        "input.scene3d",
        "output.image",
        "output.image.text_to_image",
        "output.image.image_to_image",
        "output.image.image_upscale",
        "output.video",
        "output.video.text_to_video",
        "output.video.image_to_video",
        "output.sound",
        "output.music",
        "output.scene3d",
        "output.scene3d.text_to_scene3d",
        "output.scene3d.image_to_scene3d",
        "embedding.image",
        "rerank.text",
    ] {
        let segs: Vec<&str> = key.split('.').collect();
        routes.push(json!({
            "key": key, "kind": segs[0], "modality": segs[1],
            "label": key, "configured": false, "source": "not_configured",
            "package_hint": "abstractvision or a vision-capable LLM"
        }));
    }
    json!({
        "ok": true, "authority": "abstractcore.local",
        "config_file": "/tmp/console-test/abstractcore.json",
        "writable": true, "errors": [], "routes": routes
    })
}

fn profiles_fixture() -> Value {
    json!({
        "ok": true, "writable": true,
        "config_file": "/tmp/console-test/abstractcore.json",
        "profiles": [
            {"id": "ovh-provider", "display_name": "OVH Provider",
             "description": "hosted endpoint", "provider_family": "openai-compatible",
             "base_url": "https://oai.example.net/v1",
             "api_key_set": true, "api_key_fingerprint": "35982521",
             "api_key_env_var": "", "allowed_models": [], "enabled": true,
             "virtual_provider": "endpoint:ovh-provider"},
            {"id": "team-proxy", "display_name": "Team proxy",
             "description": "", "provider_family": "openai",
             "base_url": "https://proxy.example/v1",
             "api_key_set": false, "api_key_fingerprint": null,
             "api_key_env_var": "TEAM_KEY", "allowed_models": ["gpt-x"], "enabled": false}
        ]
    })
}

// =======================================================================
// The honest mirror
// =======================================================================

#[test]
fn boot_shows_loading_then_the_mirror() {
    let mut h = harness();
    h.store.cfg.set(Loadable::Loading);
    let s = h.turns(2);
    assert!(s.contains("reading"), "loading state renders:\n{s}");

    h.load_fixtures();
    let s = h.turns(2);
    assert!(
        s.contains("AbstractCore Console"),
        "header present:\n{s}"
    );
    assert!(s.contains("loaded"), "file state in header:\n{s}");
    assert!(
        s.contains("abstractcore.json"),
        "config path named:\n{s}"
    );
    assert!(
        s.contains("default_models"),
        "sections table renders:\n{s}"
    );
}

#[test]
fn overview_states_set_default_broken() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.turns(2);
    // default_models is set (2 fields differ from defaults).
    assert!(s.contains("● 2 set"), "set counts render:\n{s}");
    // video has one broken field, logging another.
    assert!(s.contains("✗ 1 broken"), "broken sections marked:\n{s}");
    // api_keys shows which keys, never values.
    assert!(s.contains("● 1 of 7"), "api key count renders:\n{s}");
    assert!(s.contains("openai"), "the set key is named:\n{s}");
    // Unknown section warning.
    assert!(
        s.contains("future_section"),
        "unknown sections surfaced:\n{s}"
    );
    // Python-agreement line (fixture echoes the same path).
    assert!(
        s.contains("reads the same file"),
        "agreement line:\n{s}"
    );
}

#[test]
fn secrets_never_render_on_any_screen() {
    let mut h = harness();
    h.load_fixtures();
    for screen in 0..8 {
        let s = h.goto_screen(screen);
        assert!(
            !s.contains("sk-test-secret-abc"),
            "api key leaked on screen {screen}:\n{s}"
        );
        assert!(
            !s.contains("tok-super-secret"),
            "auth token leaked on screen {screen}:\n{s}"
        );
        assert!(
            !s.contains("sk-live-topsecret-body"),
            "profile key leaked on screen {screen}:\n{s}"
        );
    }
    // The Providers screen shows presence + fingerprint instead.
    let s = h.goto_screen(2);
    assert!(s.contains("set · fp"), "fingerprint presence renders:\n{s}");
}

#[test]
fn corrupt_file_is_a_hard_stop_with_backups() {
    let mut h = harness();
    h.store.cfg.set(Loadable::Ready(ConfigMirror {
        path: test_path(),
        state: FileState::Corrupt {
            error: "expected `,` at line 3 column 7".into(),
            backups: vec![
                "abstractcore.json.corrupt-20260725-080000.bak".into(),
                "abstractcore.json.bak-repair-143715".into(),
            ],
        },
        loaded_at: "12:00:00Z".into(),
    }));
    let s = h.turns(2);
    assert!(s.contains("CORRUPT"), "header names the state:\n{s}");
    assert!(
        s.contains("will NOT write"),
        "the refusal is explicit:\n{s}"
    );
    assert!(
        s.contains("corrupt-20260725-080000"),
        "backups listed:\n{s}"
    );
    assert!(
        s.contains("expected `,`"),
        "the parse error is shown:\n{s}"
    );
}

#[test]
fn missing_file_shows_the_defaults_honestly() {
    let mut h = harness();
    h.store.cfg.set(Loadable::Ready(ConfigMirror {
        path: test_path(),
        state: FileState::Missing,
        loaded_at: "12:00:00Z".into(),
    }));
    let s = h.turns(2);
    assert!(
        s.contains("no config file yet"),
        "missing state named:\n{s}"
    );
    assert!(
        s.contains("built-in defaults"),
        "explains what Python runs with:\n{s}"
    );
    assert!(
        s.contains("default_models"),
        "the section inventory still teaches:\n{s}"
    );
}

/// The P1-1 mirror lane: a valid-JSON file whose profile row carries
/// one unknown key is a file PYTHON refuses (whole-file defaults +
/// backup) — the mirror must say so on the Overview AND in the header,
/// never vouch for it.
#[test]
fn python_refused_file_is_flagged_not_vouched_for() {
    let mut h = harness();
    let mut raw = config_fixture_value();
    raw["provider_profiles"]["profiles"]["ovh-provider"]["future_field"] =
        serde_json::json!(true);
    h.store.cfg.set(Loadable::Ready(mirror_of(raw)));
    let s = h.turns(2);
    assert!(
        s.contains("Python will refuse this file"),
        "fold-side refusal banner:\n{s}"
    );
    assert!(
        s.contains("unknown field \"future_field\""),
        "the exact row/field is named:\n{s}"
    );
    assert!(
        s.contains("Python REFUSES it"),
        "the header must not wear the green label:\n{s}"
    );
}

/// The P1-1 CLI lane: an exit-0 abstractcore run that printed a
/// #FALLBACK stderr line (Python refused the file, ran on defaults)
/// surfaces as a loud banner and poisons the agreement line.
#[test]
fn python_fallback_stderr_becomes_a_loud_banner() {
    let mut h = harness();
    h.load_fixtures();
    h.store.python_fallback.set(Some(
        "#FALLBACK abstractcore config at /tmp/console-test/abstractcore.json could not \
         be parsed (…); falling back to DEFAULTS for this session."
            .into(),
    ));
    let s = h.turns(2);
    assert!(
        s.contains("PYTHON REFUSES THIS FILE"),
        "the banner renders:\n{s}"
    );
    assert!(
        s.contains("backs it up and uses DEFAULTS"),
        "the consequence is spelled out:\n{s}"
    );
    assert!(
        s.contains("REFUSES its content"),
        "the agreement line must not say ✓ same file:\n{s}"
    );
    assert!(
        !s.contains("reads the same file this console shows"),
        "the vouching line is gone:\n{s}"
    );
}

#[test]
fn unreadable_file_renders_distinctly() {
    let mut h = harness();
    h.store.cfg.set(Loadable::Ready(ConfigMirror {
        path: test_path(),
        state: FileState::Unreadable {
            error: "/x is a DIRECTORY — ABSTRACTCORE_CONFIG_FILE must name a file \
                    (ABSTRACTCORE_CONFIG_DIR names the directory)"
                .into(),
        },
        loaded_at: "12:00:00Z".into(),
    }));
    let s = h.turns(2);
    assert!(s.contains("unreadable"), "header names the state:\n{s}");
    assert!(s.contains("DIRECTORY"), "the dir mistake is named:\n{s}");
    assert!(
        s.contains("fix the path or its permissions"),
        "the hint covers both causes:\n{s}"
    );
}

/// The ✗ DIFFERENT FILES branch — and its inverse guard: an
/// unnormalized-but-identical path must NOT alarm (component compare).
#[test]
fn agreement_alarm_fires_only_for_truly_different_files() {
    let mut h = harness();
    h.load_fixtures();
    // Doubled slash, same file: no alarm.
    let mut same = routes_fixture();
    same["config_file"] = serde_json::json!("/tmp/console-test//abstractcore.json");
    h.store
        .routes
        .set(Loadable::Ready(RoutesData::from_value(&same)));
    let s = h.turns(2);
    assert!(
        s.contains("reads the same file"),
        "unnormalized same path stays ✓:\n{s}"
    );

    let mut other = routes_fixture();
    other["config_file"] = serde_json::json!("/etc/elsewhere/abstractcore.json");
    h.store
        .routes
        .set(Loadable::Ready(RoutesData::from_value(&other)));
    let s = h.turns(2);
    assert!(s.contains("DIFFERENT FILES"), "the alarm fires:\n{s}");
    assert!(
        s.contains("ABSTRACTCORE_CONFIG_FILE"),
        "and teaches the cause:\n{s}"
    );
}

/// review P2-1: the api_keys Overview row must not hide broken keys —
/// the section holding the secrets is the last place to drop states.
#[test]
fn api_keys_broken_state_visible_on_overview() {
    let mut h = harness();
    let mut raw = config_fixture_value();
    raw["api_keys"]["openai"] = serde_json::json!(12345);
    h.store.cfg.set(Loadable::Ready(mirror_of(raw)));
    let s = h.turns(2);
    assert!(
        s.contains("✗ 1 broken") && s.contains("openai:"),
        "api_keys row shows the broken key:\n{s}"
    );
    assert!(!s.contains("12345"), "the value never renders:\n{s}");
}

/// Navigation and browsing send NOTHING to the worker — reads happen
/// at boot and on r only (negative assert drains the whole queue).
#[test]
fn navigation_sends_no_commands() {
    let mut h = harness();
    h.load_fixtures();
    h.drain_cmds();
    for key in [b"2" as &[u8], b"4", b"7", &[0x0e], &[0x10], b"\t", b"\x1b[B"] {
        h.key(key);
        h.turns(2);
    }
    let cmds = h.drain_cmds();
    assert!(
        cmds.is_empty(),
        "browsing must not trigger loads: {cmds:?}"
    );
}

#[test]
fn cli_missing_degrades_honestly() {
    let mut h = harness();
    h.store
        .cfg
        .set(Loadable::Ready(mirror_of(config_fixture_value())));
    h.store.routes.set(Loadable::Failed(
        abstractcore_console::cli::CliError::core(
            abstractcore_console::cli::CliErrorKind::NotFound,
            "no $ABSTRACTCORE_BIN, nothing on PATH, no venv fallback".into(),
        ),
    ));
    let s = h.turns(2);
    assert!(
        s.contains("not found"),
        "cli line says not found:\n{s}"
    );
    let s = h.goto_screen(3);
    assert!(
        s.contains("abstractcore CLI not found"),
        "routes screen explains:\n{s}"
    );
    assert!(
        s.contains("ABSTRACTCORE_BIN"),
        "and teaches the fix:\n{s}"
    );
}

// =======================================================================
// Navigation
// =======================================================================

#[test]
fn digits_and_chords_switch_screens() {
    let mut h = harness();
    h.load_fixtures();
    h.key(b"4");
    let s = h.turns(3);
    assert!(
        s.contains("route") && s.contains("input.text"),
        "digit 4 lands on Routes:\n{s}"
    );
    assert_eq!(h.ui.screen.get_untracked(), 3);

    // Ctrl+N advances to Media.
    h.key(&[0x0e]);
    let s = h.turns(3);
    assert_eq!(h.ui.screen.get_untracked(), 4, "Ctrl+N advances:\n{s}");
    // Ctrl+P goes back.
    h.key(&[0x10]);
    h.turns(3);
    assert_eq!(h.ui.screen.get_untracked(), 3);
}

#[test]
fn overview_enter_jumps_to_the_owning_screen() {
    let mut h = harness();
    h.load_fixtures();
    // Row 0 = default_models → Model screen (index 1).
    h.key(b"\r");
    let s = h.turns(3);
    assert_eq!(h.ui.screen.get_untracked(), 1, "Enter jumps:\n{s}");
    assert!(
        s.contains("global_provider"),
        "the Model screen shows the section:\n{s}"
    );
}

#[test]
fn r_reloads_everything() {
    let mut h = harness();
    h.load_fixtures();
    h.drain_cmds();
    h.key(b"r");
    h.turns(2);
    let cmds = h.drain_cmds();
    let dbg = format!("{cmds:?}");
    assert!(dbg.contains("LoadConfig"), "reload sends LoadConfig: {dbg}");
    assert!(dbg.contains("LoadRoutes"), "reload sends LoadRoutes: {dbg}");
    assert!(
        dbg.contains("LoadProfiles"),
        "reload sends LoadProfiles: {dbg}"
    );
}

// =======================================================================
// Screens
// =======================================================================

#[test]
fn routes_screen_shows_coverage_and_alias() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(3);
    assert!(
        s.contains("6 of 24 configured"),
        "the banner counts:\n{s}"
    );
    assert!(
        s.contains("covered by input.text"),
        "coverage decorations:\n{s}"
    );
    assert!(s.contains("= input.text"), "output.text alias:\n{s}");
}

#[test]
fn providers_screen_shows_profiles_and_env_refs() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(2);
    assert!(s.contains("ovh-provider"), "profile row:\n{s}");
    assert!(s.contains("35982521"), "profile fingerprint:\n{s}");
    assert!(s.contains("$TEAM_KEY"), "env-var reference renders:\n{s}");
    assert!(
        s.contains("keys stored HERE override the environment"),
        "the precedence rule is taught:\n{s}"
    );
}

#[test]
fn embeddings_screen_shows_the_route_mirror() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(5);
    assert!(
        s.contains("route embedding.text"),
        "route mirror line:\n{s}"
    );
    assert!(
        s.contains("setters keep both in sync"),
        "the mirror rule is taught:\n{s}"
    );
}

#[test]
fn review_screen_proves_the_one_file_identity() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(7);
    assert!(
        s.contains("✓ same file"),
        "agreement per derived view:\n{s}"
    );
    assert!(
        s.contains("no actions recorded"),
        "empty journal is honest:\n{s}"
    );
}

#[test]
fn audio_effective_note_renders_on_media() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(4);
    // Select the audio.strategy row (vision's 5 fields precede it) —
    // the detail line under the table carries the FULL effective-rule
    // note that the table cell truncates.
    h.ui.media_sel.set(5);
    let s = h.turns(2);
    assert!(
        s.contains("audio.strategy"),
        "detail line names the field:\n{s}"
    );
    assert!(
        s.contains("not explicit — effective"),
        "the smart-default rule shows in full:\n{s}"
    );
}

// =======================================================================
// M2: editors + the write lane's UI half + wizard mode
// =======================================================================

/// The scalar editor: open on a section-page row, validate a bad
/// value, save a good one, and prove the whole write-done round trip
/// (form closes only on the verified outcome).
#[test]
fn scalar_editor_validates_submits_and_closes_on_success() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(4);
    // vision(5) + audio(6) precede video; max_frames is video's 2nd.
    h.ui.media_sel.set(12);
    h.drain_cmds();
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("Edit video.max_frames"),
        "editor modal opens:\n{s}"
    );
    assert!(s.contains("integer ≥ 1"), "the kind hint teaches:\n{s}");
    assert!(s.contains("applies now"), "the truth line renders:\n{s}");

    // The fixture value is broken ("three") — the editor prefills it
    // with the cursor at 0 (engine gap: no cursor-at-end open; End
    // first). Clear it, type garbage, submit: validation refuses
    // BEFORE any command is sent.
    h.key(b"\x1b[F"); // End
    for _ in 0..8 {
        h.key(&[0x7f]); // backspace
    }
    h.turn();
    h.type_text("abc");
    h.turn();
    h.key(b"\r");
    let s = h.turns(2);
    assert!(
        s.contains("not an integer"),
        "validation refuses locally:\n{s}"
    );
    assert!(h.drain_cmds().is_empty(), "no command left the form");

    for _ in 0..3 {
        h.key(&[0x7f]);
    }
    h.turn();
    h.type_text("9");
    h.turn();
    h.key(b"\r");
    let dbg_screen = h.turns(2);
    let cmds = h.drain_cmds();
    let write = cmds
        .iter()
        .find_map(|c| match c {
            Cmd::Write(spec) => Some(spec.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a write was sent — cmds: {cmds:?}\nscreen:\n{dbg_screen}"));
    let dbg = format!("{write:?}");
    assert!(
        dbg.contains("--set-video-max-frames"),
        "routes through the CLI setter: {dbg}"
    );
    assert!(dbg.contains("\"9\""), "argv carries the value: {dbg}");

    // Simulate the worker's verified completion → the form closes.
    let form_id = write.form_id.expect("form correlated");
    h.ui.write_done.set(Some((form_id, Ok("verified".into()))));
    let s = h.turns(3);
    assert!(
        !s.contains("Edit video.max_frames"),
        "form closed on success:\n{s}"
    );

    // And a FAILED outcome keeps the next form open with the error.
    h.key(b"e");
    h.turns(2);
    h.key(b"\x1b[F");
    for _ in 0..8 {
        h.key(&[0x7f]);
    }
    h.type_text("7");
    h.turn();
    h.key(b"\r");
    h.turns(2);
    let write2 = h
        .drain_cmds()
        .into_iter()
        .find_map(|c| match c {
            Cmd::Write(spec) => Some(spec),
            _ => None,
        })
        .expect("second write");
    h.ui.write_done.set(Some((
        write2.form_id.unwrap(),
        Err("route input.text verifies as —/— (configured: false)".into()),
    )));
    let s = h.turns(3);
    assert!(
        s.contains("Edit video.max_frames"),
        "form stays open on failure:\n{s}"
    );
    assert!(
        s.contains("verifies as"),
        "the verbatim verify error shows:\n{s}"
    );
}

/// The dirty-Esc guard: the first Esc warns, the second discards.
#[test]
fn dirty_form_esc_warns_then_discards() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(4);
    h.ui.media_sel.set(12);
    h.key(b"e");
    h.turns(2);
    h.type_text("5");
    h.turn();
    h.press_escape();
    let s = h.turns(2);
    assert!(
        s.contains("Esc again to discard"),
        "first Esc warns:\n{s}"
    );
    assert!(s.contains("Edit video.max_frames"), "form still open:\n{s}");
    h.press_escape();
    let s = h.turns(2);
    assert!(
        !s.contains("Edit video.max_frames"),
        "second Esc discards:\n{s}"
    );
}

/// The secret editor: typed keys render as bullets, never plaintext;
/// the drained command redacts structurally; the label carries no
/// secret.
#[test]
fn secret_editor_masks_and_redacts() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(2);
    h.ui.keys_sel.set(0); // openai
    h.drain_cmds();
    h.key(b"k");
    let s = h.turns(2);
    assert!(
        s.contains("Secret — api_keys.openai"),
        "masked editor opens:\n{s}"
    );
    h.type_text("sk-typed-secret-xyz");
    let s = h.turns(2);
    assert!(
        !s.contains("sk-typed-secret-xyz"),
        "typed secret never renders:\n{s}"
    );
    assert!(s.contains("••"), "bullets render instead:\n{s}");
    h.key(b"\r");
    h.turns(2);
    let cmds = h.drain_cmds();
    let dbg = format!("{cmds:?}");
    assert!(dbg.contains("set API key openai"), "{dbg}");
    assert!(
        !dbg.contains("sk-typed-secret-xyz"),
        "the command debug redacts: {dbg}"
    );
    assert!(dbg.contains("«redacted»"), "{dbg}");
}

/// Blank submit in the secret editor is refused with the teaching
/// (blank keeps; clear is explicit).
#[test]
fn secret_editor_blank_refuses_with_reason() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(2);
    h.drain_cmds();
    h.key(b"k");
    h.turns(2);
    h.key(b"\r");
    let s = h.turns(2);
    assert!(
        s.contains("blank keeps the stored secret"),
        "refusal teaches:\n{s}"
    );
    assert!(h.drain_cmds().is_empty(), "nothing sent");
}

/// The route editor opens for editable rows and refuses the alias/
/// locked rows with reasons; x clears through the confirm.
#[test]
fn route_editor_and_clear_respect_editability() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(3);
    h.drain_cmds();

    // output.text (row 3) is the alias — e refuses with the teaching.
    h.ui.route_sel.set(3);
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("mirrors input.text"),
        "alias refusal teaches:\n{s}"
    );

    // input.text (row 0) is editable — the editor opens prefilled.
    h.ui.route_sel.set(0);
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("Route — Text Input (input.text)"),
        "editor opens:\n{s}"
    );
    h.press_escape();
    h.turns(2);

    // x on a configured route asks first; the danger option clears.
    h.ui.route_sel.set(4); // output.voice, configured
    h.key(b"x");
    let s = h.turns(2);
    assert!(
        s.contains("Clear the route output.voice"),
        "confirm prompt:\n{s}"
    );
    // Danger confirms default to KEEP — move up to the clear option.
    h.key(b"\x1b[A");
    h.turn();
    h.key(b"\r");
    h.turns(2);
    let cmds = h.drain_cmds();
    let dbg = format!("{cmds:?}");
    assert!(
        dbg.contains("clear-default") && dbg.contains("output.voice"),
        "the clear went through the honest CLI verb: {dbg}"
    );
}

/// Keep (the default) on the danger confirm sends NOTHING — drained
/// to emptiness, not absence-of-match.
#[test]
fn danger_confirm_defaults_to_keep() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(3);
    h.drain_cmds();
    h.ui.route_sel.set(4);
    h.key(b"x");
    h.turns(2);
    h.key(b"\r"); // commit the DEFAULT (keep)
    h.turns(2);
    assert!(h.drain_cmds().is_empty(), "keep sends nothing");
}

/// The write door on a Python-refused file matches the worker's split
/// (M2 review P2-3): CLI-routed editors refuse (a setter would RESET
/// the file to defaults); direct-write editors stay open with a
/// warning (RMW preserves every byte — including the refused rows).
#[test]
fn refused_file_door_splits_cli_from_rmw() {
    let mut h = harness();
    let mut raw = config_fixture_value();
    raw["provider_profiles"]["profiles"]["ovh-provider"]["future_field"] =
        serde_json::json!(true);
    h.store.cfg.set(Loadable::Ready(mirror_of(raw)));
    h.store
        .routes
        .set(Loadable::Ready(RoutesData::from_value(&routes_fixture())));
    h.turns(2);
    h.goto_screen(4);
    h.drain_cmds();

    // Row 0 = vision.strategy → CLI-routed → refused with the reason.
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("CLI writes would RESET it"),
        "the CLI door refuses with the reason:\n{s}"
    );
    assert!(
        !s.contains("Edit vision.strategy") && !s.contains("Vision strategy"),
        "no editor opened:\n{s}"
    );
    assert!(h.drain_cmds().is_empty(), "nothing sent");

    // Row 8 = audio.caption_provider → RMW-routed → the editor OPENS
    // with the careful-warning (the file stays preservable).
    h.ui.media_sel.set(8);
    h.turns(2);
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("Edit audio.caption_provider"),
        "the RMW editor opens:\n{s}"
    );
    assert!(
        s.contains("careful: Python refuses this file"),
        "with the warning notice:\n{s}"
    );
}

/// The chain editor counts REAL entries (an `org/model` id used to
/// double-count via display-slash counting — review P2-6) and the add
/// form sends the array-path-verified write (review P1-1's surface).
#[test]
fn chain_editor_counts_real_entries_and_adds() {
    let mut h = harness();
    let mut raw = config_fixture_value();
    raw["vision"]["fallback_chain"] = serde_json::json!([
        {"provider": "huggingface", "model": "unsloth/Qwen3-4B-Instruct-2507-GGUF"}
    ]);
    h.store.cfg.set(Loadable::Ready(mirror_of(raw)));
    h.turns(2);
    h.goto_screen(4);
    h.ui.media_sel.set(3); // vision.fallback_chain
    h.drain_cmds();
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("(1 entries)"),
        "org/model id counts ONCE:\n{s}"
    );
    // First option (Add) is highlighted; Enter commits it.
    h.key(b"\r");
    let s = h.turns(2);
    assert!(s.contains("Add vision fallback"), "add form opens:\n{s}");
    h.type_text("ollama/qwen2.5vl");
    h.turn();
    h.key(b"\r");
    h.turns(2);
    let cmds = h.drain_cmds();
    let dbg = format!("{cmds:?}");
    assert!(
        dbg.contains("--add-vision-fallback"),
        "the CLI append verb: {dbg}"
    );
    assert!(
        dbg.contains("fallback_chain") && dbg.contains("\"1\""),
        "the expectation walks the array to the NEW slot: {dbg}"
    );
}

// =======================================================================
// Wizard mode
// =======================================================================

#[test]
fn wizard_walks_the_phases_and_filters_sections() {
    let mut h = harness();
    h.load_fixtures();
    // w enters the wizard at step 1 (Overview orientation).
    h.key(b"w");
    let s = h.turns(3);
    assert!(s.contains("Step 1/10"), "step line renders:\n{s}");
    assert!(s.contains("wizard"), "header mode flips:\n{s}");

    // Ctrl+N → the default-model phase on the Model screen, filtered
    // to default_models (app_defaults rows are hidden).
    h.key(&[0x0e]);
    let s = h.turns(3);
    assert!(s.contains("Step 2/10"), "step advances:\n{s}");
    assert!(s.contains("Default model (1/8)"), "phase title:\n{s}");
    assert!(s.contains("global_provider"), "focused section shows:\n{s}");
    assert!(
        !s.contains("summarizer_provider"),
        "other sections filtered while the wizard focuses:\n{s}"
    );
    assert_eq!(h.ui.screen.get_untracked(), 1);

    // Digits are refused with a reason in wizard mode.
    h.key(b"5");
    let s = h.turns(2);
    assert_eq!(h.ui.screen.get_untracked(), 1, "digit did not jump");
    assert!(
        s.contains("digit jumps work in browse mode"),
        "refusal teaches:\n{s}"
    );

    // Walk the remaining phases: vision → keys → server → audio →
    // video → embeddings → logging → review.
    let expects = [
        (4usize, "Vision (2/8)"),
        (2, "API keys (3/8)"),
        (6, "Server (4/8)"),
        (4, "Audio (5/8)"),
        (4, "Video (6/8)"),
        (5, "Embeddings (7/8)"),
        (6, "Logging (8/8)"),
        (7, "Review"),
    ];
    for (screen, title) in expects {
        h.key(&[0x0e]);
        let s = h.turns(3);
        assert_eq!(h.ui.screen.get_untracked(), screen, "screen for {title}");
        assert!(s.contains(title), "step title {title}:\n{s}");
    }

    // Past the last step: refused with the finish teaching.
    h.key(&[0x0e]);
    let s = h.turns(2);
    assert!(s.contains("f finishes"), "last-step refusal:\n{s}");

    // f finishes into browse; the free surface re-arms.
    h.key(b"f");
    let s = h.turns(3);
    assert!(s.contains("browse"), "browse mode:\n{s}");
    h.key(b"4");
    h.turns(3);
    assert_eq!(h.ui.screen.get_untracked(), 3, "digits live again");
}

/// Esc walks back through wizard steps; q is refused with a reason.
#[test]
fn wizard_esc_back_and_q_refusal() {
    let mut h = harness();
    h.load_fixtures();
    h.key(b"w");
    h.turns(2);
    h.key(&[0x0e]);
    h.turns(2);
    assert_eq!(h.ui.step.get_untracked(), 1);
    h.press_escape();
    assert_eq!(h.ui.step.get_untracked(), 0, "Esc steps back");
    h.key(b"q");
    let s = h.turns(2);
    assert!(
        s.contains("q quits in browse mode"),
        "q refused with reason:\n{s}"
    );
}

// =======================================================================
// M3 test verbs: the probe commands the screens send, the single-flight
// guard, and the Review evidence render.
// =======================================================================

/// `t` on Providers opens the provider test picker (ALL canonical
/// providers + endpoint profiles — the api_keys table alone could
/// never reach keyless lmstudio/ollama); `g` (anywhere) sends the
/// default-route generation. Each pick posts exactly one Probe.
#[test]
fn test_verbs_send_probe_commands() {
    use abstractcore_console::probes::ProbeKind;
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(2); // Providers
    h.drain_cmds();

    h.key(b"t");
    let s = h.turns(2);
    assert!(s.contains("Test which provider?"), "picker opens:\n{s}");
    assert!(s.contains("lmstudio"), "keyless providers offered:\n{s}");
    // (Profiles ride below the list window's fold — their presence is
    // proven by PICKING one further down.)
    // Initial selection follows the selected key row (row 0 = openai).
    h.key(b"\r");
    h.turns(2);
    let cmds = h.drain_cmds();
    let [Cmd::Probe(spec)] = cmds.as_slice() else {
        panic!("expected exactly one Probe, got {cmds:?}");
    };
    let ProbeKind::ListModels { target, .. } = &spec.kind else {
        panic!("expected ListModels, got {spec:?}");
    };
    assert_eq!(target, "openai", "initial pick follows the key row");

    // The single-flight guard: while a probe runs, the picker refuses
    // at the door with the teaching notice.
    h.store.probe_busy.set(true);
    h.key(b"t");
    let s = h.turns(2);
    assert!(h.drain_cmds().is_empty(), "no probe queued while busy");
    assert!(s.contains("a test is already running"), "{s}");
    assert!(!s.contains("Test which provider?"), "picker refused while busy");
    h.store.probe_busy.set(false);

    // Pick the endpoint profile: initial openai is index 5 of the 10
    // static providers; the profile rides at index 10.
    h.key(b"t");
    h.turns(2);
    for _ in 0..5 {
        h.key(b"\x1b[B");
        h.turn();
    }
    h.key(b"\r");
    h.turns(2);
    let cmds = h.drain_cmds();
    let [Cmd::Probe(spec)] = cmds.as_slice() else {
        panic!("expected exactly one Probe, got {cmds:?}");
    };
    let ProbeKind::ListModels { target, reach } = &spec.kind else {
        panic!("expected ListModels, got {spec:?}");
    };
    assert_eq!(target, "endpoint:ovh-provider");
    // The fixture profile is httpS — the console never speaks TLS, so
    // the reach check honestly stays out (CLI-only verdict). The http
    // lane is pinned in probes::tests::endpoints_config_first_then_….
    assert!(reach.is_none(), "https base_url must NOT be TCP-probed");

    // send_probe latches probe_busy SYNCHRONOUSLY (M3 review P2-2 —
    // waiting for the worker's begin-post left queued probes invisible
    // to the guard); with no worker in this harness, completion is
    // simulated by clearing it.
    assert!(h.store.probe_busy.get_untracked(), "send latches the guard");
    h.store.probe_busy.set(false);

    // g from any screen: the default-route generation.
    h.key(b"g");
    h.turns(2);
    let cmds = h.drain_cmds();
    let [Cmd::Probe(spec)] = cmds.as_slice() else {
        panic!("expected exactly one Probe, got {cmds:?}");
    };
    assert!(
        matches!(
            &spec.kind,
            ProbeKind::Generate {
                provider: None,
                model: None
            }
        ),
        "{spec:?}"
    );
}

/// `t` on Routes checks the SELECTED route's model against the live
/// list; unset routes refuse with a reason instead of probing nothing.
#[test]
fn route_test_verb_checks_membership_or_refuses() {
    use abstractcore_console::probes::ProbeKind;
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(3); // Routes
    h.drain_cmds();

    h.key(b"t");
    h.turns(2);
    let cmds = h.drain_cmds();
    let [Cmd::Probe(spec)] = cmds.as_slice() else {
        panic!("expected exactly one Probe, got {cmds:?}");
    };
    let ProbeKind::RouteCheck {
        capability,
        provider,
        model,
        ..
    } = &spec.kind
    else {
        panic!("expected RouteCheck, got {spec:?}");
    };
    assert!(!capability.is_empty() && !provider.is_empty() && !model.is_empty());

    // An unset route: selection moved to a row with no provider/model
    // resolves to a refusal notice, zero commands.
    let unset_idx = h.store.routes.with_untracked(|r| {
        r.ready()
            .and_then(|d| d.rows.iter().position(|row| row.provider.is_none()))
    });
    if let Some(idx) = unset_idx {
        h.ui.route_sel.set(idx);
        h.turns(1);
        h.key(b"t");
        let s = h.turns(2);
        assert!(h.drain_cmds().is_empty(), "unset route sends nothing");
        assert!(s.contains("nothing to test"), "{s}");
    }
}

/// The pair editor's model picker: a prefilled provider kicks
/// discovery at OPEN, and once the domain holds models the row is a
/// Combobox filtered to the pair's class — embedding models for the
/// embeddings pair, with the hidden generative count named.
#[test]
fn pair_editor_populates_a_class_filtered_model_picker() {
    let mut h = harness();
    h.load_fixtures();

    // Discovery already known: mixed lmstudio list (2 embedding-shaped,
    // 2 generative).
    h.store.models.update(|m| {
        m.insert(
            "lmstudio".into(),
            Loadable::Ready(vec![
                "text-embedding-qwen3-embedding-0.6b".into(),
                "gemma-3-1b-it".into(),
                "bge-small-en-v1.5".into(),
                "granite-4.1-3b".into(),
            ]),
        );
    });
    h.goto_screen(5); // Embeddings; row 0 = provider (the pair field)
    h.drain_cmds();
    h.key(b"e");
    let s = h.turns(3);
    assert!(
        s.contains("Embeddings (legacy pair"),
        "pair editor opens:\n{s}"
    );
    assert!(
        s.contains("2 embedding models · 2 other hidden"),
        "class filter + hidden count:\n{s}"
    );
    // The CURRENT model (fixture pair) displays as the picker value.
    assert!(
        s.contains("text-embedding-qwen3-embedding-0.6b"),
        "prefilled value visible:\n{s}"
    );
    assert!(
        h.drain_cmds().is_empty(),
        "known discovery is not re-kicked"
    );

    // Second editor with NO discovery: the open KICKS it and the row
    // stays a typeable input with the loading state named.
    let mut h2 = harness();
    h2.load_fixtures();
    h2.goto_screen(5);
    h2.drain_cmds();
    h2.key(b"e");
    let s = h2.turns(3);
    assert!(s.contains("discovering"), "kick at open shows:\n{s}");
    let cmds = h2.drain_cmds();
    assert!(
        cmds.iter()
            .any(|c| matches!(c, Cmd::LoadModels { provider } if provider == "lmstudio")),
        "prefilled provider kicks discovery at open: {cmds:?}"
    );

    // Discovery landing WHILE the form is open (the live sequence):
    // the row swaps input → picker in place; the MODAL SURVIVES.
    h2.store.models.update(|m| {
        m.insert(
            "lmstudio".into(),
            Loadable::Ready(vec![
                "text-embedding-qwen3-embedding-0.6b".into(),
                "gemma-3-1b-it".into(),
            ]),
        );
    });
    let s = h2.turns(3);
    assert!(
        s.contains("Embeddings (legacy pair"),
        "the modal survives the domain update:\n{s}"
    );
    assert!(
        s.contains("1 embedding models · 1 other hidden"),
        "row swapped to the filtered picker:\n{s}"
    );
}

/// Routes-lane reach parity (M3 review P2-4): an `endpoint:<id>` route
/// with no route-level base_url resolves the PROFILE's http endpoint
/// for the TCP disambiguation — the same target must not give weaker
/// evidence from Routes than from the Providers picker.
#[test]
fn route_test_resolves_endpoint_profile_reach() {
    use abstractcore_console::probes::ProbeKind;
    let mut h = harness();
    h.load_fixtures();
    h.store.routes.set(Loadable::Ready(RoutesData::from_value(&json!({
        "ok": true, "config_file": "/tmp/console-test/abstractcore.json",
        "routes": [
            {"key": "input.text", "kind": "input", "modality": "text",
             "label": "Text Input", "provider": "endpoint:local-lab",
             "model": "m1", "configured": true, "source": "configured"},
        ]
    }))));
    h.store.profiles.set(Loadable::Ready(ProfilesData::from_value(&json!({
        "ok": true, "config_file": "/tmp/console-test/abstractcore.json",
        "profiles": [
            {"id": "local-lab", "display_name": "Lab", "description": "",
             "provider_family": "lmstudio", "base_url": "http://localhost:9999/v1",
             "api_key_set": false, "api_key_fingerprint": null,
             "api_key_env_var": "", "allowed_models": [], "enabled": true}
        ]
    }))));
    h.goto_screen(3);
    h.drain_cmds();
    h.key(b"t");
    h.turns(2);
    let cmds = h.drain_cmds();
    let [Cmd::Probe(spec)] = cmds.as_slice() else {
        panic!("expected exactly one Probe, got {cmds:?}");
    };
    let ProbeKind::RouteCheck {
        provider, reach, ..
    } = &spec.kind
    else {
        panic!("expected RouteCheck, got {spec:?}");
    };
    assert_eq!(provider, "endpoint:local-lab");
    let hp = reach.as_ref().expect("profile base_url feeds the reach check");
    assert_eq!((hp.host.as_str(), hp.port), ("localhost", 9999));
}

/// Review renders the latest evidence per target with honest verdicts —
/// and an empty state that teaches the verbs.
#[test]
fn review_renders_test_evidence() {
    use abstractcore_console::probes::{TestResult, Verdict};
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(7);
    assert!(s.contains("no tests run yet"), "teaching empty state:\n{s}");

    h.store.record_test(TestResult {
        when: "12:00:01Z".into(),
        label: "test lmstudio".into(),
        verdict: Verdict::Proven,
        detail: "48 models served · e.g. gemma-3-1b-it".into(),
    });
    h.store.record_test(TestResult {
        when: "12:00:02Z".into(),
        label: "generation test (default route)".into(),
        verdict: Verdict::NotProven,
        detail: "the CLI reports success with ZERO models".into(),
    });
    let s = h.turns(3);
    assert!(s.contains("✓ proven") && s.contains("48 models"), "{s}");
    assert!(s.contains("? NOT PROVEN"), "{s}");

    // A re-test REPLACES its prior entry (latest evidence per target).
    h.store.record_test(TestResult {
        when: "12:00:03Z".into(),
        label: "test lmstudio".into(),
        verdict: Verdict::Failed,
        detail: "boom".into(),
    });
    let s = h.turns(3);
    assert!(!s.contains("48 models"), "stale evidence replaced:\n{s}");
    assert!(s.contains("✗ FAILED"), "{s}");
    let count = h.store.tests.with_untracked(|t| t.len());
    assert_eq!(count, 2, "one entry per label");
}

// =======================================================================
// Capture minting (evidence artifacts, not assertions): run on demand
// with `cargo test --test headless_ui mint_captures -- --ignored`.
// Writes deterministic SVGs of the fixture-loaded screens into
// docs/captures/ for reports.
// =======================================================================

#[test]
#[ignore = "mints report artifacts; run explicitly"]
fn mint_captures() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/captures");
    std::fs::create_dir_all(&dir).expect("captures dir");
    let mut h = harness_sized(Size::new(110, 30));
    h.load_fixtures();
    for (screen, name) in [(0usize, "m1-overview"), (3, "m1-routes"), (6, "m1-server")] {
        h.goto_screen(screen);
        let shot = h.term.screen().screenshot();
        shot.write_svg(dir.join(format!("{name}.svg"))).expect("svg");
    }
    // The corrupt refusal — the state worth showing in every report.
    h.store.cfg.set(Loadable::Ready(ConfigMirror {
        path: test_path(),
        state: FileState::Corrupt {
            error: "expected `,` at line 3 column 7".into(),
            backups: vec!["abstractcore.json.corrupt-20260725-080000.bak".into()],
        },
        loaded_at: "12:00:00Z".into(),
    }));
    h.goto_screen(0);
    let shot = h.term.screen().screenshot();
    shot.write_svg(dir.join("m1-corrupt.svg")).expect("svg");

    // M2: the wizard walk and an open typed editor.
    let mut h = harness_sized(Size::new(110, 30));
    h.load_fixtures();
    h.key(b"w");
    h.key(&[0x0e]); // Ctrl+N → step 2: the default-model phase
    h.turns(3);
    let shot = h.term.screen().screenshot();
    shot.write_svg(dir.join("m2-wizard-model.svg")).expect("svg");

    let mut h = harness_sized(Size::new(110, 30));
    h.load_fixtures();
    h.goto_screen(4); // Media
    h.turns(2);
    for _ in 0..12 {
        h.key(b"\x1b[B"); // down to video.max_frames — a scalar editor
    }
    h.key(b"e");
    h.turns(3);
    let shot = h.term.screen().screenshot();
    shot.write_svg(dir.join("m2-editor.svg")).expect("svg");

    // M3: the Review evidence surface with all three verdicts.
    let mut h = harness_sized(Size::new(110, 30));
    h.load_fixtures();
    for r in [
        abstractcore_console::probes::TestResult {
            when: "12:00:01Z".into(),
            label: "test lmstudio".into(),
            verdict: abstractcore_console::probes::Verdict::Proven,
            detail: "48 models available · e.g. gemma-3-1b-it".into(),
        },
        abstractcore_console::probes::TestResult {
            when: "12:00:07Z".into(),
            label: "generation test (default route)".into(),
            verdict: abstractcore_console::probes::Verdict::Proven,
            detail: "lmstudio/gemma-3-1b-it — replied in 3s: “PONG”".into(),
        },
        abstractcore_console::probes::TestResult {
            when: "12:00:12Z".into(),
            label: "test ollama".into(),
            verdict: abstractcore_console::probes::Verdict::NotProven,
            detail: "TCP localhost:11434 → Connection refused (os error 61) — the server \
                     looks DOWN; the CLI reports success with ZERO models (also its answer \
                     for a dead server)"
                .into(),
        },
        abstractcore_console::probes::TestResult {
            when: "12:00:15Z".into(),
            label: "test route input.text".into(),
            verdict: abstractcore_console::probes::Verdict::Failed,
            detail: "model ghost-9b is NOT among the 48 the provider serves — edit the \
                     route (e) and pick from the live list — lmstudio/ghost-9b"
                .into(),
        },
    ] {
        h.store.record_test(r);
    }
    h.goto_screen(7);
    h.turns(2);
    let shot = h.term.screen().screenshot();
    shot.write_svg(dir.join("m3-review-evidence.svg")).expect("svg");
}

// =======================================================================
// The chrome-survival matrix (born-knowing lesson #1): all screens ×
// the definition-of-done sizes, WITH heavy fixtures — the header, tab
// bar and footer hints must never be flex-crushed by content.
// =======================================================================

/// The refusal states carry safety copy — it must survive the tightest
/// chartered size too (review audit #6).
#[test]
fn corrupt_refusal_survives_tight_size() {
    let mut h = harness_sized(Size::new(60, 16));
    h.store.cfg.set(Loadable::Ready(ConfigMirror {
        path: test_path(),
        state: FileState::Corrupt {
            error: "expected `,` at line 3 column 7".into(),
            backups: vec![
                "abstractcore.json.corrupt-20260725-080000.bak".into(),
                "abstractcore.json.corrupt-20260724-120000.bak".into(),
                "abstractcore.json.bak-repair-143715".into(),
            ],
        },
        loaded_at: "12:00:00Z".into(),
    }));
    let s = h.turns(3);
    let lines: Vec<&str> = s.lines().collect();
    assert!(
        lines[0].contains("AbstractCore Console"),
        "chrome survives:\n{s}"
    );
    assert!(
        s.contains("will NOT write"),
        "the refusal line survives 60x16:\n{s}"
    );
}

#[test]
fn chrome_survives_every_screen_at_every_size() {
    for (w, hgt) in [(80u16, 24u16), (100, 24), (60, 16)] {
        let mut h = harness_sized(Size::new(w as i32, hgt as i32));
        h.load_fixtures();
        for screen in 0..8 {
            let s = h.goto_screen(screen);
            let lines: Vec<&str> = s.lines().collect();
            assert!(
                lines
                    .first()
                    .map(|l| l.contains("AbstractCore Console"))
                    .unwrap_or(false),
                "title bar at row 0 ({w}x{hgt} screen={screen}):\n{s}"
            );
            // The strip windows around the active tab at narrow widths
            // — the ACTIVE label is the one always guaranteed visible.
            let active_label = format!("{} {}", screen + 1, ui::SCREENS[screen]);
            assert!(
                lines
                    .get(1)
                    .map(|l| l.contains(&active_label))
                    .unwrap_or(false),
                "tab bar at row 1 shows {active_label} ({w}x{hgt} screen={screen}):\n{s}"
            );
            let hint_row = lines.last().unwrap_or(&"");
            assert!(
                hint_row.contains("1-8"),
                "footer hints at the last row ({w}x{hgt} screen={screen}):\n{s}"
            );
        }
    }
}
