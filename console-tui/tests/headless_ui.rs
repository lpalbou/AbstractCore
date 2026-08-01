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
use abstractcore_console::store::{
    AvailabilityData, ConfigMirror, Loadable, ProfilesData, RoutesData, Store,
};
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
        // Production posture (lib.rs) — the harness drives the SAME
        // pipeline, so the hover-ink mouse mode must not differ.
        hover_ink: true,
        // The one deliberate divergence: the host-clipboard fallback
        // spawns pbcopy/wl-copy/xclip synchronously. A test suite must
        // never overwrite the clipboard of whoever runs it — the exact
        // bug abstracttui 0.3.0 fixed in its OWN suite, and the reason
        // the flag exists.
        platform_clipboard: false,
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
        self.store
            .availability
            .set(Loadable::Ready(AvailabilityData::from_value(
                &availability_fixture(),
            )));
        self.turns(2);
    }

    fn goto_screen(&mut self, n: usize) -> String {
        self.ui.screen.set(n);
        self.turns(3)
    }

    /// Select a route row BY KEY, so a test never hardcodes a grid index
    /// that reordering or a new route would silently shift.
    fn select_route(&mut self, key: &str) -> String {
        let idx = self
            .store
            .routes
            .with_untracked(|d| {
                d.ready()
                    .and_then(|d| d.rows.iter().position(|r| r.key == key))
            })
            .unwrap_or_else(|| panic!("route {key} is not in the fixture grid"));
        self.ui.route_sel.set(idx);
        self.turns(2)
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
        // The derived text-output row, in the LIVE shape: the CLI marks
        // it `derived_from` + `read_only` (manager.py:1138-1151), and
        // the console reads derived-ness from those fields — never from
        // a hardcoded key.
        json!({"key": "output.text", "kind": "output", "modality": "text",
               "label": "Text Output", "provider": "lmstudio", "model": "qwen3-0.6b",
               "configured": true, "read_only": true, "derived_from": "input.text",
               "source": "abstractcore.capability_defaults"}),
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
        let mut row = json!({
            "key": key, "kind": segs[0], "modality": segs[1],
            "label": key, "configured": false, "source": "not_configured",
            "package_hint": "abstractvision or a vision-capable LLM"
        });
        // THE ROUTE HIERARCHY the payload actually carries
        // (`manager._decorate_route_hierarchy`): a task row names its
        // parent, a parent names its task rows. Without these the
        // fixture rendered four flat `output.image*` siblings — the very
        // shape that made an operator ask whether the parent was a
        // remnant. `covered_by_tasks` is absent here on purpose: this
        // fixture's task rows are unconfigured, so the parent IS the
        // missing setting and must still read "not configured".
        if segs.len() >= 3 {
            row["broad_key"] = json!(format!("{}.{}", segs[0], segs[1]));
        } else if ["output.image", "output.video", "output.scene3d"].contains(&key) {
            let tasks: Vec<String> = [
                "output.image.text_to_image",
                "output.image.image_to_image",
                "output.image.image_upscale",
                "output.video.text_to_video",
                "output.video.image_to_video",
                "output.scene3d.text_to_scene3d",
                "output.scene3d.image_to_scene3d",
            ]
            .iter()
            .filter(|t| t.starts_with(&format!("{key}.")))
            .map(|t| t.to_string())
            .collect();
            row["task_keys"] = json!(tasks);
        }
        routes.push(row);
    }
    json!({
        "ok": true, "authority": "abstractcore.local",
        "config_file": "/tmp/console-test/abstractcore.json",
        "writable": true, "errors": [], "routes": routes
    })
}

/// `abstractcore models status --json` for the fixture machine: the
/// text route's weights are MISSING (the recommended 4-bit build), the
/// voice route's are here, the embedding route's provider cannot be
/// consulted, and an unconfigured route reports `route not configured`
/// — which is not a missing download and must never be offered as one.
fn availability_fixture() -> Value {
    json!({
        "ok": true,
        "routes": [
            {"key": "input.text", "provider": "lmstudio", "model": "qwen3-0.6b",
             "download_artifact": "qwen/qwen3.5-9b@4bit",
             "availability": {"provider": "lmstudio", "artifact": "qwen/qwen3.5-9b@4bit",
                              "status": "absent", "downloadable": true,
                              "evidence": "lms ls --json",
                              "instruction": "lms get qwen/qwen3.5-9b@4bit"}},
            {"key": "output.voice", "provider": "supertonic", "model": "supertonic-3",
             "availability": {"provider": "supertonic", "artifact": "supertonic-3",
                              "status": "installed", "downloadable": true,
                              "location": "/cache/supertonic-3"}},
            {"key": "embedding.text", "provider": "lmstudio",
             "model": "text-embedding-qwen3-embedding-0.6b",
             "availability": {"provider": "lmstudio",
                              "artifact": "text-embedding-qwen3-embedding-0.6b",
                              "status": "unknown", "downloadable": false,
                              "evidence": "no lms CLI, server unreachable",
                              "instruction": "Install LM Studio and enable its CLI"}},
            {"key": "output.image", "provider": "", "model": "",
             "availability": {"status": "unknown", "evidence": "route not configured"}}
        ],
        "recommended": {
            "total": 3, "installed": 2, "absent": 1, "unknown": 0,
            "would_download": [
                {"provider": "lmstudio", "artifact": "qwen/qwen3.5-9b@4bit", "route": "input.text"}
            ]
        }
    })
}

fn profiles_fixture() -> Value {
    json!({
        "ok": true, "writable": true, "probed": true,
        "config_file": "/tmp/console-test/abstractcore.json",
        // THE PROVIDER INVENTORY, not the api_keys section: the screen
        // used to enumerate api_keys and therefore hid ollama, lmstudio,
        // mlx and huggingface entirely. Row 0 stays `openai` so the key
        // editor's fixed selection still lands on a key-taking row.
        //
        // The stored profiles ride this SAME array as `endpoint:<id>`
        // rows (live shape: model_materializer.provider_inventory folds
        // the profile store into the inventory), which is what lets the
        // screen render ONE list.
        "providers": [
            {"provider": "openai", "kind": "cloud_api", "auth": "required",
             "api_key_field": "openai", "api_key_env_var": "OPENAI_API_KEY",
             "api_key_set": true, "api_key_source": "config",
             "api_key_fingerprint": "9f1c33aa", "base_url": "", "base_url_source": "",
             "reachable": null, "reachability": "", "note": ""},
            {"provider": "anthropic", "kind": "cloud_api", "auth": "required",
             "api_key_field": "anthropic", "api_key_env_var": "ANTHROPIC_API_KEY",
             "api_key_set": false, "api_key_source": "", "api_key_fingerprint": "",
             "base_url": "", "base_url_source": "",
             "reachable": null, "reachability": "", "note": ""},
            // A key that comes from the ENVIRONMENT — the row core can
            // show and the gateway's `env` origin means.
            {"provider": "openrouter", "kind": "cloud_api", "auth": "required",
             "api_key_field": "openrouter", "api_key_env_var": "OPENROUTER_API_KEY",
             "api_key_set": true, "api_key_source": "env:OPENROUTER_API_KEY",
             "api_key_fingerprint": "c0ffee11", "base_url": "", "base_url_source": "",
             "reachable": null, "reachability": "", "note": ""},
            {"provider": "lmstudio", "kind": "local_server", "auth": "none",
             "api_key_field": "", "api_key_env_var": "", "api_key_set": false,
             "api_key_source": "", "api_key_fingerprint": "",
             "base_url": "http://localhost:1234/v1", "base_url_source": "default",
             "reachable": true, "reachability": "reachable (43 models)", "note": ""},
            {"provider": "ollama", "kind": "local_server", "auth": "none",
             "api_key_field": "", "api_key_env_var": "", "api_key_set": false,
             "api_key_source": "", "api_key_fingerprint": "",
             "base_url": "http://localhost:11434", "base_url_source": "default",
             "reachable": false, "reachability": "GET .../api/tags unreachable", "note": ""},
            {"provider": "mlx", "kind": "local_engine", "auth": "none",
             "api_key_field": "", "api_key_env_var": "", "api_key_set": false,
             "api_key_source": "", "api_key_fingerprint": "",
             "base_url": "", "base_url_source": "",
             "reachable": null, "reachability": "",
             "note": "Apple Silicon text/vision inference"},
            {"provider": "huggingface", "kind": "local_engine", "auth": "optional",
             "api_key_field": "", "api_key_env_var": "HF_TOKEN", "api_key_set": false,
             "api_key_source": "", "api_key_fingerprint": "",
             "base_url": "", "base_url_source": "",
             "reachable": null, "reachability": "",
             "note": "HF_TOKEN only for gated/private repos"},
            {"provider": "endpoint:ovh-provider", "kind": "endpoint_profile",
             "auth": "optional", "api_key_field": "", "api_key_env_var": "",
             "api_key_set": true, "api_key_source": "profile",
             "api_key_fingerprint": "35982521",
             "base_url": "https://oai.example.net/v1", "base_url_source": "endpoint profile",
             "reachable": true, "reachability": "reachable (22 models)",
             "note": "endpoint profile (openai-compatible)"},
            {"provider": "endpoint:team-proxy", "kind": "endpoint_profile",
             "auth": "optional", "api_key_field": "", "api_key_env_var": "TEAM_KEY",
             "api_key_set": false, "api_key_source": "", "api_key_fingerprint": "",
             "base_url": "https://proxy.example/v1", "base_url_source": "endpoint profile",
             "reachable": null, "reachability": "",
             "note": "endpoint profile (openai)"}
        ],
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
    assert!(
        s.contains("stored (9f1c33aa)"),
        "fingerprint presence renders:\n{s}"
    );
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

/// THE SHARED STATE VOCABULARY. These four strings are the words the
/// gateway console's routes table prints for the same four states —
/// one grid, whichever entry point the operator opened.
#[test]
fn routes_screen_speaks_the_shared_state_vocabulary() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(3);
    assert!(s.contains("writable"), "the write door leads:\n{s}");
    assert!(
        s.contains("6 of 24 configured"),
        "the banner counts:\n{s}"
    );
    assert!(s.contains("configured"), "configured state:\n{s}");
    assert!(
        s.contains("covered by input.text"),
        "coverage decorations:\n{s}"
    );
    assert!(s.contains("derived ← input.text"), "derived state:\n{s}");
    assert!(s.contains("not configured"), "unconfigured state:\n{s}");
}

/// THE ROUTE GRID READS AS A HIERARCHY, NOT AS DUPLICATE KEYS. The
/// operator's question (2026-08-01) was "why do we have output.image and
/// output.video? are those remnants?" — asked because the grid printed
/// the parent as a flat sibling ABOVE its own three children with a red
/// "not configured". It is the opposite of a remnant: it is the one
/// value that serves every image task with no row of its own, which is
/// exactly what the fresh-install seed writes. So the parent keeps its
/// full key and the task rows indent beneath it.
#[test]
fn routes_screen_groups_task_rows_under_their_modality_row() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(3);
    assert!(
        s.contains("output.image") && s.contains("output.video"),
        "the parent rows are present and keep their full key:\n{s}"
    );
    for task in ["text_to_image", "image_to_image", "image_upscale", "image_to_video"] {
        assert!(
            s.contains(&format!("└ {task}")),
            "{task} indents under its modality row:\n{s}"
        );
    }
    assert!(
        !s.contains("output.image.text_to_image"),
        "the child does not repeat its parent's prefix — that width is the model column's:\n{s}"
    );
    // A modality with no task rows is the PRIMARY key, not a fallback:
    // it must stay a plain, un-indented row.
    assert!(
        s.contains("output.voice") && !s.contains("└ output.voice"),
        "voice/sound/music have no task rows and take no indent:\n{s}"
    );
}

/// The detail line answers "what IS this row" for both hierarchy levels:
/// the parent says what it serves, a task row names what it overrides,
/// and the FULL key stays readable there now that the grid abbreviates
/// it (it is what `config set-default` needs typed).
#[test]
fn routes_detail_line_explains_the_hierarchy_level() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(3);
    let s = h.select_route("output.image");
    assert!(
        s.contains("output.image") && s.contains("serves any image task with no row of its own"),
        "the parent row says what it is for:\n{s}"
    );
    let s = h.select_route("output.image.text_to_image");
    assert!(
        s.contains("output.image.text_to_image"),
        "the full key stays copyable on the detail line:\n{s}"
    );
    assert!(
        s.contains("overrides output.image"),
        "a task row names the parent it overrides:\n{s}"
    );
}

/// The Providers screen at a width where the payload fits whole. The
/// width policy (`ui::widths`) is measured, not capped, so a narrow
/// terminal legitimately middle-truncates a 21-character
/// `endpoint:<id>`; these tests are about the CELLS, not the cut.
fn providers_harness() -> Harness {
    harness_sized(Size::new(140, 40))
}

/// Select a provider row BY NAME — the unified list's order is the
/// CLI's, and a hardcoded index would silently shift when core adds a
/// backend or the operator adds a connection.
fn select_provider(h: &mut Harness, name: &str) -> String {
    let idx = h
        .store
        .profiles
        .with_untracked(|d| {
            d.ready()
                .and_then(|d| d.connections().iter().position(|c| c.provider == name))
        })
        .unwrap_or_else(|| panic!("{name} is not in the unified provider list"));
    h.ui.profile_sel.set(idx);
    h.turns(2)
}

/// THE PARITY RULING (2026-08-01, with screenshots of both consoles):
/// "I do not understand why the providers are displayed in a different
/// fashion between gateway and core; they should have the exact same.
/// Gateway is the one we want. Profiles are just indicated as profile
/// of the openai-compatible endpoint, and we should have a way to
/// configure as many as necessary, like in the gateway console."
///
/// ONE table, the gateway console-TUI's seven columns, every stored
/// profile INLINE as its `endpoint:<id>` row — and no second table.
#[test]
fn providers_screen_is_one_table_in_the_gateway_columns() {
    let mut h = providers_harness();
    h.load_fixtures();
    let s = h.goto_screen(2);
    for column in [
        "provider", "family", "base URL", "API key", "models", "enabled", "origin",
    ] {
        assert!(s.contains(column), "gateway column `{column}` is here:\n{s}");
    }
    assert!(
        s.contains("Available providers (a adds a connection)"),
        "the gateway's own block title:\n{s}"
    );
    // The second table is GONE — its editing moved onto the rows.
    assert!(
        !s.contains("Provider endpoint profiles"),
        "one list, not two:\n{s}"
    );
    // The old screen's vocabulary went with it.
    for gone in ["local server", "local engine", "cloud API", "answering", "key / endpoint"] {
        assert!(!s.contains(gone), "the old vocabulary `{gone}` is gone:\n{s}");
    }

    // A profile is a row like any other, saying which endpoint family
    // it is a profile OF.
    let (_, row) = find_row(&s, "endpoint:ovh-provider").expect("the profile row is IN the list");
    let cells = s.lines().nth(row as usize).unwrap_or_default();
    assert!(
        cells.contains("openai-compatible"),
        "the profile row names its family:\n{cells}"
    );
    assert!(
        cells.contains("https://oai.example.net/v1") && cells.contains("stored (35982521)"),
        "base URL + the shared key vocabulary on one row:\n{cells}"
    );
    assert!(
        cells.contains("22 live") && cells.contains("yes") && cells.contains("config"),
        "models / enabled / origin:\n{cells}"
    );
    // As many connections as necessary: the second one is a row too.
    assert!(
        find_row(&s, "endpoint:team-proxy").is_some(),
        "every stored profile gets a row:\n{s}"
    );
}

/// The `origin` column answers "where does this row come from?" — the
/// gateway's question, in core's four words. Key precedence (a stored
/// key beats the environment) is READABLE from the column instead of
/// being a footnote under the table.
#[test]
fn origin_column_says_where_each_row_comes_from() {
    let mut h = providers_harness();
    h.load_fixtures();
    let s = h.goto_screen(2);
    let origin_of = |screen: &str, provider: &str| {
        let (_, row) = find_row(screen, provider).unwrap_or_else(|| panic!("{provider} row"));
        screen
            .lines()
            .nth(row as usize)
            .unwrap_or_default()
            .trim_matches(|c: char| c == ' ' || c == '│')
            .split_whitespace()
            .next_back()
            .unwrap_or_default()
            .to_string()
    };
    assert_eq!(origin_of(&s, "openai"), "config", "a key stored in this file");
    assert_eq!(
        origin_of(&s, "openrouter"),
        "env",
        "a key resolved from the environment"
    );
    assert_eq!(
        origin_of(&s, "lmstudio"),
        "auto",
        "a local server that ANSWERED at its default address"
    );
    assert_eq!(
        origin_of(&s, "ollama"),
        "registry",
        "a default address with nothing behind it is not configured"
    );
    assert_eq!(origin_of(&s, "endpoint:ovh-provider"), "config");

    // The footer's two facts, gateway-shaped.
    assert!(
        s.contains("core default: lmstudio / qwen/qwen3.6-35b-a3b"),
        "what answers when nothing names a provider:\n{s}"
    );
    assert!(
        s.contains("not configured yet (k sets a key · a adds a connection)"),
        "the affordance leads the unconfigured line:\n{s}"
    );
    assert!(
        s.contains("anthropic"),
        "an unconfigured backend is named there:\n{s}"
    );
}

/// THE EARLIER DEFECT, still fixed: the screen once listed the
/// `api_keys` SECTION, so every provider that takes no key had no row
/// at all ("how come we don't have ollama, lmstudio, huggingface and
/// mlx?", 2026-08-01). Unifying the list must not lose them again.
#[test]
fn providers_screen_lists_every_provider_not_just_keyed_ones() {
    let mut h = providers_harness();
    h.load_fixtures();
    let s = h.goto_screen(2);
    for keyless in ["ollama", "lmstudio", "mlx", "huggingface"] {
        assert!(
            find_row(&s, keyless).is_some(),
            "keyless provider {keyless} has a row:\n{s}"
        );
    }
    // A probe that counted models says how many are live, beside the
    // address it counted them at.
    assert!(s.contains("43 live"), "the probe's live count:\n{s}");
    assert!(
        s.contains("http://localhost:1234/v1"),
        "the address it was probed at:\n{s}"
    );
    // Key presence stays a FINGERPRINT, never key material; a missing
    // key names the var it would have come from.
    assert!(s.contains("stored (9f1c33aa)"), "key presence proof:\n{s}");
    assert!(
        s.contains("none ($ANTHROPIC_API_KEY)"),
        "a missing key names the var it would come from:\n{s}"
    );
    // A registry provider has no enable switch in core — the cell says
    // so instead of advertising a toggle with nothing behind it.
    let (_, row) = find_row(&s, "mlx").expect("mlx row");
    let cells = s.lines().nth(row as usize).unwrap_or_default();
    assert!(
        cells.contains('—') && cells.contains("registry"),
        "no fabricated enabled flag on a registry row:\n{cells}"
    );

    // The selected row's summary carries the probe's own words.
    let s = select_provider(&mut h, "ollama");
    assert!(
        s.contains("unreachable"),
        "an unreachable local server says so:\n{s}"
    );
    assert!(
        s.contains("k/e set the key") || s.contains("no key to set"),
        "the summary says what THIS row supports:\n{s}"
    );

    // The reserved `google` row is gone: the registry has no google
    // provider, so a registry-driven list cannot invent one.
    assert!(
        find_row(&s, "google").is_none(),
        "no provider row for a provider that does not exist:\n{s}"
    );
}

/// Per-row verbs, gateway's set: every one either acts or REFUSES WITH
/// THE REASON — an advertised key that silently does nothing reads as
/// a dead app.
#[test]
fn row_verbs_act_or_refuse_with_the_reason() {
    let mut h = providers_harness();
    h.load_fixtures();
    h.goto_screen(2);

    // `k` on a row with no key: the refusal names the provider and why.
    select_provider(&mut h, "mlx");
    h.drain_cmds();
    h.key(b"k");
    let s = h.turns(2);
    assert!(
        s.contains("mlx takes no API key"),
        "the refusal names the provider and why:\n{s}"
    );
    assert!(!s.contains("Secret — api_keys"), "no editor opened:\n{s}");
    assert!(h.drain_cmds().is_empty(), "nothing sent");

    // `d` on a builtin: only stored connections delete here.
    h.key(b"d");
    let s = h.turns(2);
    assert!(
        s.contains("only stored connections delete here"),
        "the delete refusal teaches the alternative:\n{s}"
    );
    assert!(h.drain_cmds().is_empty(), "nothing sent");

    // `e` on a key-taking builtin opens the masked api_keys editor —
    // that IS how a builtin is configured in core.
    select_provider(&mut h, "openai");
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("Secret — api_keys.openai"),
        "e configures the selected row:\n{s}"
    );
    h.press_escape();
    h.turns(2);

    // `m` asks for THIS row's models, by the name it answers to.
    select_provider(&mut h, "endpoint:ovh-provider");
    h.drain_cmds();
    h.key(b"m");
    let s = h.turns(2);
    assert!(s.contains("Models — endpoint:ovh-provider"), "{s}");
    let cmds = h.drain_cmds();
    assert!(
        matches!(cmds.as_slice(), [Cmd::LoadModels { provider }] if provider == "endpoint:ovh-provider"),
        "one discovery for the selected row: {cmds:?}"
    );
}

/// `d` on a stored connection confirms with the consequence spelled
/// out, then deletes through `config delete-provider` (verified by
/// re-read, like every write here).
#[test]
fn delete_connection_confirms_then_writes() {
    let mut h = providers_harness();
    h.load_fixtures();
    h.goto_screen(2);
    select_provider(&mut h, "endpoint:team-proxy");
    h.drain_cmds();
    h.key(b"d");
    let s = h.turns(2);
    assert!(
        s.contains("Delete provider connection 'team-proxy'?"),
        "the confirm names the connection:\n{s}"
    );
    assert!(
        s.contains("endpoint:team-proxy"),
        "…and the consequence for anything routing through it:\n{s}"
    );
    assert!(h.drain_cmds().is_empty(), "nothing written before the answer");
    // The danger confirm defaults to KEEP — Enter must not delete.
    h.key(b"\r");
    h.turns(2);
    assert!(h.drain_cmds().is_empty(), "the default answer keeps it");
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
    h.ui.profile_sel.set(0); // openai — the first row of the unified list
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

    // output.text (row 3) is the derived row — e refuses with the
    // teaching, in the gateway console's exact words.
    h.ui.route_sel.set(3);
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("output.text derives from input.text — edit that route instead"),
        "derived-row refusal teaches:\n{s}"
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
        s.contains("Clear the override on output.voice"),
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

/// `a` on Routes applies the framework recommendation — the other half
/// of the weights banner. Safe by default (routes the operator
/// configured are KEPT); replacing them is a separate, danger-tinted
/// answer, never the accidental one.
#[test]
fn a_on_routes_applies_the_recommended_defaults() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(3);
    assert!(
        s.contains("a applies the recommended routes"),
        "the banner names the verb:\n{s}"
    );
    h.drain_cmds();

    h.key(b"a");
    let s = h.turns(2);
    assert!(
        s.contains("Apply the framework's recommended routes"),
        "the prompt asks first:\n{s}"
    );
    // Default answer is the SAFE one.
    h.key(b"\r");
    h.turns(2);
    let cmds = h.drain_cmds();
    let dbg = format!("{cmds:?}");
    assert!(
        dbg.contains("apply-recommended"),
        "the honest CLI verb: {dbg}"
    );
    assert!(
        !dbg.contains("--force"),
        "the default answer never overrules the operator: {dbg}"
    );

    // The second answer is the explicit overrule.
    h.key(b"a");
    h.turns(2);
    h.key(b"\x1b[B");
    h.turn();
    h.key(b"\r");
    h.turns(2);
    let dbg = format!("{:?}", h.drain_cmds());
    assert!(
        dbg.contains("apply-recommended") && dbg.contains("--force"),
        "the danger answer forces: {dbg}"
    );
}

/// THE PARTIAL-UPDATE CONTRACT, end to end through the editor.
/// `set-default` preserves the fields a command does not name, so the
/// editor sends ONLY what the operator moved. Changing the reasoning
/// must produce a command with `--reasoning` and NOTHING else — no
/// echo of the provider/model the grid happened to render, which is
/// exactly how a value set from the gateway console between render and
/// save would get silently overwritten.
#[test]
fn route_editor_sends_only_the_field_the_operator_edited() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(3);
    h.ui.route_sel.set(0); // input.text — the text-generation route
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("Route — Text Input (input.text)"),
        "editor opens:\n{s}"
    );
    // The stored truth leads the form — never the local picks.
    assert!(
        s.contains("Applies now: lmstudio / qwen3-0.6b"),
        "the editor states what applies now:\n{s}"
    );
    assert!(s.contains("reasoning"), "the reasoning row is present:\n{s}");
    h.drain_cmds();

    // Saving an UNTOUCHED form refuses instead of rewriting the row
    // with its own rendered values. Tab order: provider, model, base
    // URL, reasoning, options, [Save].
    for _ in 0..5 {
        h.key(b"\t");
        h.turn();
    }
    h.type_text("\r");
    let s = h.turns(2);
    assert!(s.contains("nothing changed"), "untouched save refuses:\n{s}");
    assert!(
        h.drain_cmds().is_empty(),
        "an untouched form sends no write"
    );

    // Move ONLY the reasoning select: focus it and pick "high".
    h.press_escape();
    h.turns(2);
    h.key(b"e");
    h.turns(2);
    for _ in 0..3 {
        h.key(b"\t"); // provider → model → base URL → reasoning
        h.turn();
    }
    h.type_text("\r"); // open the select
    h.turns(2);
    for _ in 0..4 {
        h.key(b"\x1b[B"); // not set → minimal → low → medium → high
        h.turn();
    }
    h.type_text("\r"); // commit
    h.turns(2);
    for _ in 0..2 {
        h.key(b"\t"); // options → Save
        h.turn();
    }
    h.type_text("\r");
    h.turns(2);

    let cmds = h.drain_cmds();
    let dbg = format!("{cmds:?}");
    assert!(dbg.contains("set-default"), "a write went out: {dbg}");
    assert!(
        dbg.contains("--reasoning") && dbg.contains("high"),
        "the edited field is sent: {dbg}"
    );
    for unsent in ["--provider", "--model", "--base-url", "--option"] {
        assert!(
            !dbg.contains(unsent),
            "{unsent} must NOT be sent — the store keeps what the editor did not touch: {dbg}"
        );
    }
}

/// The reasoning effort is a property of TEXT GENERATION: the control
/// exists on the text route and on no other row.
#[test]
fn reasoning_row_appears_only_on_the_text_route() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(3);
    h.ui.route_sel.set(4); // output.voice
    h.key(b"e");
    let s = h.turns(2);
    assert!(
        s.contains("Route — Voice Output (output.voice)"),
        "editor opens:\n{s}"
    );
    assert!(
        !s.contains("reasoning"),
        "no reasoning control on a non-text route:\n{s}"
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

/// `t` on Providers tests the SELECTED row — the gateway's per-row
/// verb. It could not be selected-row before: the old top table WAS
/// the api_keys section, so keyless lmstudio/ollama (the wizard's own
/// recommended targets) had no row to select, and the verb had to open
/// a picker over a list the screen did not show. The unified list gives
/// every provider a row, so the picker is gone. `g` (anywhere) still
/// sends the default-route generation. Each posts exactly one Probe.
#[test]
fn test_verbs_send_probe_commands() {
    use abstractcore_console::probes::ProbeKind;
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(2); // Providers
    h.drain_cmds();

    // A keyless local server — unreachable through the old picker-less
    // path, and the reason the picker existed at all.
    select_provider(&mut h, "lmstudio");
    h.key(b"t");
    h.turns(2);
    let cmds = h.drain_cmds();
    let [Cmd::Probe(spec)] = cmds.as_slice() else {
        panic!("expected exactly one Probe, got {cmds:?}");
    };
    let ProbeKind::ListModels { target, reach } = &spec.kind else {
        panic!("expected ListModels, got {spec:?}");
    };
    assert_eq!(target, "lmstudio", "the verb tests the SELECTED row");
    assert_eq!(
        reach.as_ref().map(|hp| hp.port),
        Some(1234),
        "the row's own base URL feeds the reach check: {reach:?}"
    );
    h.store.probe_busy.set(false);

    // The single-flight guard: while a probe runs, the verb refuses at
    // the door with the teaching notice.
    h.store.probe_busy.set(true);
    h.key(b"t");
    let s = h.turns(2);
    assert!(h.drain_cmds().is_empty(), "no probe queued while busy");
    assert!(s.contains("a test is already running"), "{s}");
    h.store.probe_busy.set(false);

    // A stored connection answers to `endpoint:<id>` — the name the
    // first column prints and a route's provider field takes.
    select_provider(&mut h, "endpoint:ovh-provider");
    h.key(b"t");
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
// 0.3.0 posture: hover ink
// =======================================================================

/// Grid position (col, row) of the first occurrence of `needle` in a
/// `to_text()` dump — the text and the cell grid share coordinates.
fn find_text(screen: &str, needle: &str) -> Option<(i32, i32)> {
    screen.lines().enumerate().find_map(|(row, line)| {
        line.find(needle).map(|byte_col| {
            let col = line[..byte_col].chars().count();
            (col as i32, row as i32)
        })
    })
}

/// The engine has ALWAYS drawn Button hover ink; before 0.3.0 nothing
/// armed motion-with-no-button-held, so it never fired. Two halves,
/// and they guard DIFFERENT things — verified by flipping the flag off
/// and watching which assertion moved:
///
///   1. the armed mouse mode — the ONLY half that catches `hover_ink`
///      being dropped from the RunConfig;
///   2. the Save button's ink under the pointer — the widget-side
///      visual. `push_input` feeds the parser directly, so this half
///      passes even with the flag off (a real terminal, not the
///      harness, is what withholds the motion report). It guards the
///      ink, never the arming — do not read a green (2) as (1).
#[test]
fn hover_ink_is_armed_and_buttons_light_under_the_pointer() {
    let mut h = harness();
    h.load_fixtures();

    // Half 1 — RunConfig::hover_ink reached the terminal as mode 1003.
    // ButtonDrag (the default) would deliver motion only while a button
    // is held, which is exactly the state a hover never has.
    assert_eq!(
        h.term.enter_options().map(|o| o.mouse),
        Some(abstracttui::term::MouseMode::AnyMotion),
        "hover_ink must arm AnyMotion (1003), not the ButtonDrag default"
    );

    // Half 2 — the scalar editor's Save/Cancel row: the crate's densest
    // button surface, and the one a user meets on every write.
    h.goto_screen(4);
    h.ui.media_sel.set(12);
    h.key(b"e");
    let s = h.turns(2);
    let (col, row) = find_text(&s, "Save").unwrap_or_else(|| panic!("Save renders:\n{s}"));
    let ink_cold = h.term.screen().cell(col, row).expect("Save cell").paint.fg;

    // SGR motion, NO button held (the 1003 report), 1-based coords.
    h.key(format!("\x1b[<35;{};{}M", col + 1, row + 1).as_bytes());
    h.turns(2);
    let ink_hot = h.term.screen().cell(col, row).expect("Save cell").paint.fg;
    assert_ne!(
        ink_cold, ink_hot,
        "the hovered Save button shifts ink (accent garnish)"
    );

    // Hover is a garnish, not a latch: leaving restores the cold ink.
    h.key(b"\x1b[<35;1;1M");
    h.turns(2);
    let ink_left = h.term.screen().cell(col, row).expect("Save cell").paint.fg;
    assert_eq!(
        ink_cold, ink_left,
        "ink restores when the pointer leaves the button"
    );

    // The modal survived the whole gesture — hover must never disturb
    // the form it garnishes.
    let s = h.turns(1);
    assert!(s.contains("Edit video.max_frames"), "form intact:\n{s}");
}

// =======================================================================
// Row activation: Enter / Space / double-click open the SAME editor as
// the screen's edit verb, on every table that has one.
// =======================================================================

/// Position of a TABLE ROW whose first cell is `cell` — the row's own
/// line, not the first place the text happens to appear. Notices and
/// refusals quote route keys, so a plain substring search finds prose
/// and clicks the wrong line.
fn find_row(screen: &str, cell: &str) -> Option<(i32, i32)> {
    screen.lines().enumerate().find_map(|(row, line)| {
        let indent = line.len() - line.trim_start_matches([' ', '│']).len();
        let rest = &line[indent..];
        let after = rest.strip_prefix(cell)?;
        // The cell must END here — `input.text` must not match the
        // `input.text.foo` row, and column padding follows a real cell.
        if after.starts_with(' ') || after.is_empty() {
            Some((indent as i32, row as i32))
        } else {
            None
        }
    })
}

impl Harness {
    /// A true double-click on a row: press 1 selects, press 2 (inside
    /// the chain's timing window) activates. SGR press+release pairs,
    /// 1-based coords — the same bytes a terminal sends.
    fn double_click(&mut self, col: i32, row: i32) {
        let (cx, cy) = (col + 1, row + 1);
        for _ in 0..2 {
            self.key(format!("\x1b[<0;{cx};{cy}M").as_bytes()); // press
            self.key(format!("\x1b[<0;{cx};{cy}m").as_bytes()); // release
        }
        self.turns(2);
    }
}

/// Routes was the screen with NO `on_activate` at all: the footer
/// promised "Enter/e edit route" and neither Enter nor a double-click
/// did anything. Both triggers must now reach the same editor `e` does.
#[test]
fn routes_rows_activate_by_enter_and_double_click() {
    for (label, activate) in [
        ("enter", &(|h: &mut Harness| h.key(b"\r")) as &dyn Fn(&mut Harness)),
        ("double-click", &|h: &mut Harness| {
            let s = h.turns(1);
            let (c, r) = find_row(&s, "input.text").expect("input.text row on screen");
            h.double_click(c, r);
        }),
    ] {
        let mut h = harness();
        h.load_fixtures();
        h.goto_screen(3);
        h.ui.route_sel.set(0); // input.text — editable
        h.turns(1);
        activate(&mut h);
        let s = h.turns(2);
        assert!(
            s.contains("Route — Text Input (input.text)"),
            "{label} opens the route editor:\n{s}"
        );
    }
}

/// Activation is the `e` verb, so it inherits every refusal — a
/// double-click on the output.text alias must teach, not open a form
/// that would write to the wrong route.
#[test]
fn activation_inherits_the_edit_refusals() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(3);
    h.ui.route_sel.set(3); // output.text — the alias
    h.turns(1);
    h.key(b"\r");
    let s = h.turns(2);
    assert!(
        s.contains("output.text derives from input.text — edit that route instead"),
        "Enter refuses the derived row with the same teaching as e:\n{s}"
    );
    assert!(
        !s.contains("Route — Text Output"),
        "no editor opened for a read-only row:\n{s}"
    );
}

/// Table cells must carry no double-width emoji. The alias row's
/// "not editable" marker was `🔒` U+1F512 (Emoji=Yes, 2 cells): the
/// engine measured 2, terminals drew something else, and every column
/// after it on that row slid out of alignment — the one visibly broken
/// row on the Routes screen. `⊘` U+2298 is width 1 in both
/// unicode-width conventions and emoji-data never touches its block.
#[test]
fn locked_route_marker_is_not_a_double_width_emoji() {
    let mut h = harness();
    h.load_fixtures();
    let s = h.goto_screen(3);
    assert!(
        !s.contains('\u{1F512}'),
        "no padlock emoji in the routes table:\n{s}"
    );
    let (_, row) = find_row(&s, "output.text").expect("the alias row renders");
    let line = s.lines().nth(row as usize).expect("row line");
    assert!(
        line.contains('\u{2298}'),
        "the alias row still marks itself locked:\n{line}"
    );
    // Alignment is the point: the marked row's `state` cell must start
    // in the same screen column as an unmarked row's.
    let state_x = |row_key: &str, state: &str| -> usize {
        let (_, r) = find_row(&s, row_key).expect("row");
        let l = s.lines().nth(r as usize).expect("line");
        let byte = l.find(state).unwrap_or_else(|| panic!("{state} in {l:?}"));
        l[..byte].chars().count()
    };
    assert_eq!(
        state_x("output.text", "derived ← input.text"),
        state_x("output.voice", "configured"),
        "the marked row's state column lines up with an unmarked row's:\n{s}"
    );
}

/// `a` ADDS A CONNECTION — "we should have a way to configure as many
/// as necessary, like in the gateway console". The write lands in
/// CORE's own store: `config set-provider <id>` writes
/// `provider_profiles.profiles.<id>`, and the spec verifies by re-read
/// that the profile exists afterwards.
#[test]
fn add_connection_writes_a_provider_profile_to_core() {
    use abstractcore_console::writes::{Arg, Expect, WriteVerb};
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(2);
    h.drain_cmds();

    h.key(b"a");
    let s = h.turns(2);
    assert!(
        s.contains("Add a provider connection (endpoint:<id>)"),
        "the create door says what it creates:\n{s}"
    );

    // Tab order: id, family, base URL, API key, clear-key, display
    // name, description, enabled, [Save].
    h.type_text("paritytest");
    h.turns(1);
    h.key(b"\t");
    h.turn();
    // The family select opens with a PLACEHOLDER — never a fabricated
    // choice; `openai-compatible` is the 5th of PROFILE_FAMILIES.
    h.type_text("\r");
    h.turns(2);
    for _ in 0..5 {
        h.key(b"\x1b[B");
        h.turn();
    }
    h.type_text("\r");
    h.turns(2);
    h.key(b"\t");
    h.turn();
    h.type_text("http://127.0.0.1:1234/v1");
    h.turns(2);
    for _ in 0..6 {
        h.key(b"\t");
        h.turn();
    }
    h.type_text("\r"); // Save
    h.turns(2);

    let cmds = h.drain_cmds();
    let [Cmd::Write(spec)] = cmds.as_slice() else {
        panic!("expected exactly one Write, got {cmds:?}");
    };
    let [WriteVerb::Cli(args)] = spec.verbs.as_slice() else {
        panic!("expected one CLI verb, got {:?}", spec.verbs);
    };
    let argv: Vec<&str> = args.iter().map(Arg::value).collect();
    assert_eq!(
        argv,
        vec![
            "config",
            "set-provider",
            "paritytest",
            "--family",
            "openai-compatible",
            "--base-url",
            "http://127.0.0.1:1234/v1",
            "--enabled",
        ],
        "the add-connection argv writes CORE's provider_profiles"
    );
    assert_eq!(
        spec.expects,
        vec![Expect::ProfileExists {
            id: "paritytest".into()
        }],
        "the write proves itself by re-reading the store"
    );
}

/// The profiles table had selection but no activation — `e` worked,
/// Enter and double-click did not.
#[test]
fn profile_rows_activate_into_the_profile_editor() {
    let mut h = providers_harness();
    h.load_fixtures();
    h.goto_screen(2);
    let s = h.turns(1);
    // The row's first cell is the PROVIDER NAME — `endpoint:<id>`, the
    // spelling a route's provider field takes and the gateway console's
    // column too.
    let (c, r) = find_row(&s, "endpoint:ovh-provider").expect("profile row on screen");
    h.double_click(c, r);
    let s = h.turns(2);
    // Assert on the EDITOR's own title — the row's id is on screen
    // either way, so `contains("ovh-provider")` alone would pass with
    // no editor at all.
    assert!(
        s.contains("Edit profile ovh-provider"),
        "the profile editor opened:\n{s}"
    );
}

/// The section screens (Model/Media/Embeddings/Server/Logging) share
/// one table — activation there opens the field editor.
#[test]
fn section_field_rows_activate_into_the_field_editor() {
    let mut h = harness();
    h.load_fixtures();
    h.goto_screen(4);
    h.ui.media_sel.set(12); // video.max_frames — a scalar editor
    h.turns(1);
    h.key(b"\r");
    let s = h.turns(2);
    assert!(
        s.contains("Edit video.max_frames"),
        "Enter opens the field editor:\n{s}"
    );
}

/// The disposal bug behind the whole class: activation used to open the
/// modal on the DYN's scope, so the next config re-render (a reload
/// landing, a write completing) disposed the scope and took the open
/// form with it. Editing then silently lost the user's typing. The
/// modal belongs to the page scope, which outlives data churn.
///
/// This is the regression the `e` verb never had — it always captured
/// the page scope — which is exactly why it went unnoticed.
#[test]
fn a_form_opened_by_activation_survives_a_reload() {
    // Every screen whose activation opens a form, driven by Enter.
    let cases: [(usize, Option<usize>, &str); 3] = [
        (2, None, "Secret — api_keys."),      // providers / api_keys
        (3, Some(0), "Route — Text Input"),   // routes
        (4, Some(12), "Edit video.max_frames"), // media / section field
    ];
    for (screen, sel, needle) in cases {
        let mut h = harness();
        h.load_fixtures();
        h.goto_screen(screen);
        if let Some(i) = sel {
            match screen {
                3 => h.ui.route_sel.set(i),
                _ => h.ui.media_sel.set(i),
            }
            h.turns(1);
        }
        h.key(b"\r");
        let s = h.turns(2);
        assert!(s.contains(needle), "screen {screen} form opened:\n{s}");

        // A reload landing mid-edit re-renders the config dyn.
        h.store
            .cfg
            .set(Loadable::Ready(mirror_of(config_fixture_value())));
        h.store
            .routes
            .set(Loadable::Ready(RoutesData::from_value(&routes_fixture())));
        let s = h.turns(3);
        assert!(
            s.contains(needle),
            "screen {screen} form SURVIVES the reload:\n{s}"
        );
    }
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



// =======================================================================
// WEIGHTS: the `d` verb (model downloads)
// =======================================================================

/// The weights column and banner speak the SAME four words as the
/// gateway console and the gateway TUI, and the banner names the exact
/// artifact that is missing — the 4-bit build, not the served id the
/// route stores.
#[test]
fn routes_screen_shows_weight_availability_and_the_missing_artifact() {
    let mut h = harness_sized(Size::new(150, 40));
    h.load_fixtures();
    let s = h.goto_screen(3);
    assert!(s.contains("recommended models: 2 of 3 present"), "banner counts:\n{s}");
    assert!(
        s.contains("qwen/qwen3.5-9b@4bit"),
        "the banner names the ARTIFACT, not the served id:\n{s}"
    );
    assert!(s.contains("not downloaded"), "absent weights read plainly:\n{s}");
    assert!(s.contains("installed"), "present weights read plainly:\n{s}");
    assert!(s.contains("weights"), "the column is labelled:\n{s}");
    assert!(
        s.contains("download weights"),
        "the hint row offers the verb on THIS screen:\n{s}"
    );
    // The unconfigured route carries no weights word at all: an absent
    // answer must not read as an answer.
    assert!(
        !s.contains("output.image  not downloaded"),
        "an unconfigured route is not a missing download:\n{s}"
    );
}

/// `d` on an ABSENT row confirms first, names the artifact and the
/// provider, and only then sends the download — the one command in this
/// console that spends gigabytes never fires on a single keystroke.
#[test]
fn w_confirms_then_downloads_the_recommended_artifact() {
    let mut h = harness_sized(Size::new(150, 40));
    h.load_fixtures();
    h.goto_screen(3);
    h.drain_cmds();

    h.ui.route_sel.set(0); // input.text — weights absent
    h.key(b"w");
    let s = h.turns(2);
    assert!(
        s.contains("Download qwen/qwen3.5-9b@4bit with lmstudio"),
        "the confirm names artifact and provider:\n{s}"
    );
    assert!(
        h.drain_cmds().is_empty(),
        "nothing is downloaded before the operator confirms"
    );
    // Danger confirms default to the SAFE option; move to Download.
    h.key(b"\x1b[A");
    h.turn();
    h.key(b"\r");
    h.turns(2);
    let dbg = format!("{:?}", h.drain_cmds());
    assert!(
        dbg.contains("DownloadModel") && dbg.contains("qwen/qwen3.5-9b@4bit"),
        "the download names the artifact: {dbg}"
    );
    assert!(
        dbg.contains("LoadAvailability"),
        "the weights are re-probed right after: {dbg}"
    );
}

/// The three refusals, each with its reason — an installed model, a
/// provider with nothing to fetch, and an `unknown` answer. `unknown`
/// is the important one: guessing there spends the operator's disk.
#[test]
fn w_refuses_with_a_reason_instead_of_guessing() {
    let mut h = harness_sized(Size::new(150, 40));
    h.load_fixtures();
    h.goto_screen(3);

    h.ui.route_sel.set(4); // output.voice — installed
    h.key(b"w");
    let s = h.turns(2);
    assert!(s.contains("already installed"), "installed refusal:\n{s}");
    assert!(h.drain_cmds().is_empty(), "no download for an installed model");

    h.ui.route_sel.set(5); // embedding.text — unknown
    h.key(b"w");
    let s = h.turns(2);
    assert!(
        s.contains("availability is unknown"),
        "unknown is never treated as absent:\n{s}"
    );
    assert!(h.drain_cmds().is_empty(), "no download on an unknown answer");

    h.ui.route_sel.set(2); // input.video — no weights row at all
    h.key(b"w");
    let s = h.turns(2);
    assert!(
        s.contains("no weight information yet"),
        "an unprobed route says so:\n{s}"
    );
    assert!(h.drain_cmds().is_empty(), "no download without evidence");
}

/// The routes grid AS RENDERED: the header row and every data row under
/// it, up to the first blank line. Row LABELS belong to the row model;
/// this helper only cares which cells the width policy drew.
fn grid_rows(screen: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for l in screen.lines() {
        if l.trim_start().starts_with("route ") {
            inside = true;
            continue;
        }
        if inside {
            if l.trim().is_empty() {
                break;
            }
            out.push(l.to_string());
        }
    }
    out
}

/// THE OPERATOR'S BUG REPORT, rendered: "don't truncate the text in the
/// column when not necessary".
///
/// A 200-cell terminal has room for every model artifact and source
/// module the fixture carries — so the routes grid must print them
/// WHOLE. Before `ui::widths` this screen spent constants (`Cells(30)`,
/// `Cells(20)`, `ellipsize(model, 40)`) and handed the slack to a Flex
/// model column, so `text-embedding-qwen3-embedding-0.6b` and
/// `abstractcore.capability_defaults` wore ellipses beside seventy blank
/// cells.
#[test]
fn routes_grid_prints_whole_names_when_the_terminal_has_room() {
    let mut h = harness_sized(Size::new(200, 40));
    h.load_fixtures();
    let s = h.goto_screen(3);
    for whole in [
        "text-embedding-qwen3-embedding-0.6b",
        "abstractcore.capability_defaults",
        "covered by input.text",
    ] {
        assert!(s.contains(whole), "{whole:?} must print whole at 200:\n{s}");
    }
    // Not one cut anywhere in the grid — the banner above may still
    // elide an open-ended list; the GRID has no excuse at this width.
    let grid = grid_rows(&s);
    assert!(grid.len() >= 20, "the grid rendered:\n{s}");
    assert!(
        !grid.iter().any(|l| l.contains('\u{2026}')),
        "no ellipsis belongs in a 200-cell grid:\n{}",
        grid.join("\n")
    );
}

/// And when the terminal is genuinely too narrow, the cut keeps the end
/// that tells rows apart. `text-embedding-qwen3-embedding-0.6b` is the
/// fixture's long artifact: head-first it renders as its neighbours'
/// twin (`text-embed…`), middle-first it still ends in the size tag that
/// says WHICH embedding model this route runs.
#[test]
fn narrow_routes_grid_keeps_the_discriminating_tail() {
    let mut h = harness_sized(Size::new(84, 40));
    h.load_fixtures();
    let s = h.goto_screen(3);
    let row = grid_rows(&s)
        .into_iter()
        .find(|l| l.contains("embedding") && l.contains("0.6b"))
        .unwrap_or_else(|| panic!("no embedding row survived:\n{s}"));
    assert!(
        row.contains('\u{2026}'),
        "84 cells really is too narrow for this artifact: {row:?}"
    );
    assert!(
        row.contains("\u{2026}") && row.contains("embedding-0.6b"),
        "the discriminating tail survives the cut: {row:?}"
    );
    assert!(
        !row.contains("text-embedding-qwen3-embedding-0.6b"),
        "the cell really was cut (otherwise this test proves nothing): {row:?}"
    );
    // The closed vocabularies keep their whole word at every width.
    assert!(
        s.contains("not configured"),
        "the state vocabulary is not squeezed into nonsense:\n{s}"
    );
}
