//! The DISPLAY schema: the 17 config sections and their 85 scalar
//! fields, with dataclass defaults and the validation Python actually
//! enforces. Source: docs/config-surface-inventory.md §2 (every entry
//! cites abstractcore/config/manager.py at the probe date).
//!
//! This table is display metadata ONLY — it classifies fields as
//! set/default/broken and picks the right editor control per type. It
//! is NEVER a write schema: direct writes mutate the loaded
//! `serde_json::Value` in place and preserve every key this table does
//! not know (risk-map fact #1). A field unknown to this table renders
//! honestly as "unknown to this console" instead of being dropped.

/// How a field's value is typed, which also picks its editor control
/// (M2) and its validation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    /// Required string ("" is a real value, not "unset").
    Str,
    /// Nullable string.
    OptStr,
    Bool,
    /// Integer with an enforced range (inclusive).
    Int { min: i64, max: i64 },
    /// Nullable integer with an enforced range.
    OptInt { min: i64, max: i64 },
    /// Float with an enforced minimum (timeouts: 0 = unlimited,
    /// negatives rejected — manager.py:1738-1764).
    Float { min: f64 },
    /// Float with no Python-side validation (don't invent ranges).
    FloatFree,
    /// Closed value set (canonical spellings).
    Enum(&'static [&'static str]),
    /// Closed value set, nullable (Python types the field
    /// `Optional[str]` even though the default is a string — null
    /// loads fine there and must not read as broken here).
    OptEnum(&'static [&'static str]),
    /// Closed set + aliases the manager normalizes at load
    /// (audio.strategy: native→native_only, stt→speech_to_text;
    /// ""/disabled appear on disk historically — manager.py:380-403).
    EnumLoose {
        canon: &'static [&'static str],
        accepted: &'static [&'static str],
    },
    /// Filesystem path (string on disk; existence not validated).
    Path,
    OptPath,
    /// Plaintext secret at rest — NEVER rendered; folded to
    /// set/not-set + sha256[:8] fingerprint at parse time.
    Secret,
    /// An environment variable NAME (email password fields hold the
    /// var name, not the secret — manager.py:165-187).
    EnvVarName,
    /// vision/audio fallback_chain: list of {provider, model}.
    FallbackChain,
}

/// The dataclass default, comparable against the loaded JSON.
#[derive(Clone, Copy, Debug)]
pub enum Dv {
    Null,
    S(&'static str),
    B(bool),
    I(i64),
    F(f64),
    EmptyList,
}

impl Dv {
    pub fn matches(&self, v: &serde_json::Value) -> bool {
        match self {
            Dv::Null => v.is_null(),
            Dv::S(s) => v.as_str() == Some(s),
            Dv::B(b) => v.as_bool() == Some(*b),
            // JSON does not distinguish 3 from 3.0 — compare numerically.
            Dv::I(i) => v.as_f64().is_some_and(|f| (f - *i as f64).abs() < 1e-9),
            Dv::F(x) => v.as_f64().is_some_and(|f| (f - x).abs() < 1e-9),
            Dv::EmptyList => v.as_array().is_some_and(Vec::is_empty),
        }
    }

    /// Render the default for "resets to …" copy and default rows.
    pub fn render(&self) -> String {
        match self {
            Dv::Null => "—".into(),
            Dv::S(s) => (*s).to_string(),
            Dv::B(b) => b.to_string(),
            Dv::I(i) => i.to_string(),
            Dv::F(f) => f.to_string(),
            Dv::EmptyList => "[]".into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FieldSpec {
    /// The JSON key inside the section object.
    pub key: &'static str,
    pub kind: FieldKind,
    pub default: Dv,
    /// A short truth worth showing beside the value (safety flags,
    /// unit semantics, reserved status). None for self-evident fields.
    pub note: Option<&'static str>,
}

const fn f(key: &'static str, kind: FieldKind, default: Dv) -> FieldSpec {
    FieldSpec {
        key,
        kind,
        default,
        note: None,
    }
}

const fn fn_(key: &'static str, kind: FieldKind, default: Dv, note: &'static str) -> FieldSpec {
    FieldSpec {
        key,
        kind,
        default,
        note: Some(note),
    }
}

/// How a section's body is shaped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SectionKind {
    /// Flat scalar fields described by `fields`.
    Fields,
    /// capability_defaults: `{version, routes: {key -> route}}` — the
    /// authoritative display comes from `config defaults --json`.
    Routes,
    /// provider_profiles: `{profiles: {id -> profile}}` — displayed
    /// from the pre-redacted `config providers --json`.
    Profiles,
}

#[derive(Clone, Copy, Debug)]
pub struct SectionSpec {
    /// Top-level JSON key.
    pub name: &'static str,
    /// Human title for screens and the overview table.
    pub title: &'static str,
    pub kind: SectionKind,
    pub fields: &'static [FieldSpec],
}

pub const LOG_LEVELS: &[&str] = &["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NONE"];

/// Embedding providers — source of truth abstractcore/embeddings/
/// models.py:71-145 (NOT the wizard's own list, which omits vllm).
pub const EMBEDDING_PROVIDERS: &[&str] = &[
    "huggingface",
    "lmstudio",
    "ollama",
    "vllm",
    "openai",
    "openai-compatible",
    "openrouter",
    "portkey",
];

/// The 10 static providers in the registry (registry.py:117-261);
/// `endpoint:<id>` profiles extend this at runtime.
pub const STATIC_PROVIDERS: &[&str] = &[
    "anthropic",
    "huggingface",
    "lmstudio",
    "mlx",
    "ollama",
    "openai",
    "openai-compatible",
    "openrouter",
    "portkey",
    "vllm",
];

/// Profile provider families (provider_profiles.py:22-31 — exactly 8;
/// NOT huggingface/mlx).
pub const PROFILE_FAMILIES: &[&str] = &[
    "anthropic",
    "lmstudio",
    "ollama",
    "openai",
    "openai-compatible",
    "openrouter",
    "portkey",
    "vllm",
];

/// The five app slots of app_defaults (manager.py:1599-1623).
pub const APPS: &[&str] = &["cli", "summarizer", "extractor", "judge", "intent"];

const VISION_FIELDS: &[FieldSpec] = &[
    f(
        "strategy",
        FieldKind::Enum(&["two_stage", "disabled", "basic_metadata"]),
        Dv::S("disabled"),
    ),
    f("caption_provider", FieldKind::OptStr, Dv::Null),
    f("caption_model", FieldKind::OptStr, Dv::Null),
    fn_(
        "fallback_chain",
        FieldKind::FallbackChain,
        Dv::EmptyList,
        "CLI can only append (--add-vision-fallback); removal is a direct write",
    ),
    f("local_models_path", FieldKind::OptPath, Dv::Null),
];

const AUDIO_FIELDS: &[FieldSpec] = &[
    fn_(
        "strategy",
        FieldKind::EnumLoose {
            canon: &["native_only", "speech_to_text", "caption", "auto"],
            accepted: &[
                "native_only",
                "speech_to_text",
                "caption",
                "auto",
                "native",
                "stt",
                "disabled",
                "",
            ],
        },
        Dv::S("auto"),
        "effective value depends on audio_strategy_explicit (see note)",
    ),
    f("stt_backend_id", FieldKind::OptStr, Dv::Null),
    f("stt_language", FieldKind::OptStr, Dv::Null),
    fn_("caption_provider", FieldKind::OptStr, Dv::Null, "reserved"),
    fn_("caption_model", FieldKind::OptStr, Dv::Null, "reserved"),
    fn_(
        "fallback_chain",
        FieldKind::FallbackChain,
        Dv::EmptyList,
        "reserved",
    ),
];

const VIDEO_FIELDS: &[FieldSpec] = &[
    f(
        "strategy",
        FieldKind::Enum(&["native_only", "frames_caption", "auto"]),
        Dv::S("auto"),
    ),
    f(
        "max_frames",
        FieldKind::Int {
            min: 1,
            max: i64::MAX,
        },
        Dv::I(3),
    ),
    f(
        "max_frames_native",
        FieldKind::Int {
            min: 1,
            max: i64::MAX,
        },
        Dv::I(8),
    ),
    f("frame_format", FieldKind::Enum(&["jpg", "png"]), Dv::S("jpg")),
    f(
        "sampling_strategy",
        FieldKind::Enum(&["uniform", "keyframes"]),
        Dv::S("uniform"),
    ),
    f(
        "max_frame_side",
        FieldKind::Int {
            min: 1,
            max: i64::MAX,
        },
        Dv::I(1024),
    ),
    f(
        "max_video_size_bytes",
        FieldKind::OptInt {
            min: 1,
            max: i64::MAX,
        },
        Dv::Null,
    ),
];

const EMBEDDINGS_FIELDS: &[FieldSpec] = &[
    fn_(
        "provider",
        FieldKind::OptEnum(EMBEDDING_PROVIDERS),
        Dv::S("huggingface"),
        "legacy pair of route embedding.text — setters mirror both",
    ),
    f("model", FieldKind::OptStr, Dv::S("all-minilm-l6-v2")),
    f("base_url", FieldKind::OptStr, Dv::Null),
];

const DEFAULT_QWEN: Dv = Dv::S("unsloth/Qwen3-4B-Instruct-2507-GGUF");

// Python types every app_defaults field Optional[str] (manager.py:137-149)
// — null loads fine and must not read as broken (review P3-2).
const APP_DEFAULTS_FIELDS: &[FieldSpec] = &[
    f("cli_provider", FieldKind::OptStr, Dv::S("huggingface")),
    f("cli_model", FieldKind::OptStr, DEFAULT_QWEN),
    f("summarizer_provider", FieldKind::OptStr, Dv::S("huggingface")),
    f("summarizer_model", FieldKind::OptStr, DEFAULT_QWEN),
    f("extractor_provider", FieldKind::OptStr, Dv::S("huggingface")),
    f("extractor_model", FieldKind::OptStr, DEFAULT_QWEN),
    f("judge_provider", FieldKind::OptStr, Dv::S("huggingface")),
    f("judge_model", FieldKind::OptStr, DEFAULT_QWEN),
    f("intent_provider", FieldKind::OptStr, Dv::S("huggingface")),
    f("intent_model", FieldKind::OptStr, DEFAULT_QWEN),
];

const DEFAULT_MODELS_FIELDS: &[FieldSpec] = &[
    fn_(
        "global_provider",
        FieldKind::OptStr,
        Dv::Null,
        "coupled: --set-global-default also writes route input.text",
    ),
    f("global_model", FieldKind::OptStr, Dv::Null),
    fn_(
        "chat_model",
        FieldKind::OptStr,
        Dv::Null,
        "stored as one provider/model string",
    ),
    fn_(
        "code_model",
        FieldKind::OptStr,
        Dv::Null,
        "stored as one provider/model string",
    ),
];

const API_KEYS_FIELDS: &[FieldSpec] = &[
    f("openai", FieldKind::Secret, Dv::Null),
    f("anthropic", FieldKind::Secret, Dv::Null),
    f("openrouter", FieldKind::Secret, Dv::Null),
    f("portkey", FieldKind::Secret, Dv::Null),
    fn_(
        "openai_compatible",
        FieldKind::Secret,
        Dv::Null,
        "injects OPENAI_API_KEY (shared with openai; openai wins)",
    ),
    f("vllm", FieldKind::Secret, Dv::Null),
    fn_(
        "google",
        FieldKind::Secret,
        Dv::Null,
        "reserved — no google provider exists in the registry",
    ),
];

const SERVER_FIELDS: &[FieldSpec] = &[
    f("auth_token", FieldKind::Secret, Dv::Null),
    fn_(
        "allow_unauthenticated",
        FieldKind::Bool,
        Dv::B(false),
        "UNSAFE when true",
    ),
    fn_(
        "base_url_allowlist",
        FieldKind::OptStr,
        Dv::Null,
        "CSV; unset = loopback-only",
    ),
    f("url_fetch_allowlist", FieldKind::OptStr, Dv::Null),
    f("media_root", FieldKind::OptPath, Dv::Null),
    fn_(
        "allow_local_files",
        FieldKind::Bool,
        Dv::B(false),
        "UNSAFE when true",
    ),
    fn_(
        "host",
        FieldKind::OptStr,
        Dv::Null,
        "env HOST overrides config for server settings",
    ),
    f(
        "port",
        FieldKind::OptInt {
            min: 1,
            max: 65535,
        },
        Dv::Null,
    ),
];

const CACHE_FIELDS: &[FieldSpec] = &[
    f(
        "default_cache_dir",
        FieldKind::Path,
        Dv::S("~/.cache/abstractcore"),
    ),
    f(
        "huggingface_cache_dir",
        FieldKind::Path,
        Dv::S("~/.cache/huggingface"),
    ),
    f(
        "local_models_cache_dir",
        FieldKind::Path,
        Dv::S("~/.abstractcore/models"),
    ),
    fn_(
        "glyph_cache_dir",
        FieldKind::Path,
        Dv::S("~/.abstractcore/glyph_cache"),
        "no CLI setter — direct write only",
    ),
];

const LOGGING_FIELDS: &[FieldSpec] = &[
    f("console_level", FieldKind::Enum(LOG_LEVELS), Dv::S("ERROR")),
    f("file_level", FieldKind::Enum(LOG_LEVELS), Dv::S("DEBUG")),
    f("file_logging_enabled", FieldKind::Bool, Dv::B(false)),
    fn_(
        "log_base_dir",
        FieldKind::OptPath,
        Dv::Null,
        "unset = ~/.abstractcore/logs at runtime",
    ),
    f("verbatim_enabled", FieldKind::Bool, Dv::B(true)),
    f("console_json", FieldKind::Bool, Dv::B(false)),
    f("file_json", FieldKind::Bool, Dv::B(true)),
];

const STREAMING_FIELDS: &[FieldSpec] = &[f("cli_stream_default", FieldKind::Bool, Dv::B(false))];

const TIMEOUTS_FIELDS: &[FieldSpec] = &[
    fn_(
        "default_timeout",
        FieldKind::Float { min: 0.0 },
        Dv::F(7200.0),
        "seconds; 0 = unlimited, negatives rejected",
    ),
    fn_(
        "tool_timeout",
        FieldKind::Float { min: 0.0 },
        Dv::F(600.0),
        "seconds; 0 = unlimited",
    ),
];

const OFFLINE_FIELDS: &[FieldSpec] = &[
    f("offline_first", FieldKind::Bool, Dv::B(true)),
    f("allow_network", FieldKind::Bool, Dv::B(false)),
    f("force_local_files_only", FieldKind::Bool, Dv::B(true)),
];

const MAINTENANCE_FIELDS: &[FieldSpec] = &[
    f("triage_llm_enabled", FieldKind::Bool, Dv::B(false)),
    f(
        "triage_llm_base_url",
        FieldKind::Str,
        Dv::S("http://localhost:1234"),
    ),
    f(
        "triage_llm_model",
        FieldKind::Str,
        Dv::S("qwen/qwen3-next-80b"),
    ),
    f("triage_llm_temperature", FieldKind::FloatFree, Dv::F(0.2)),
    f(
        "triage_llm_max_tokens",
        FieldKind::Int {
            min: 1,
            max: i64::MAX,
        },
        Dv::I(800),
    ),
    f("triage_llm_timeout_s", FieldKind::FloatFree, Dv::F(30.0)),
];

const EMAIL_FIELDS: &[FieldSpec] = &[
    f("smtp_host", FieldKind::Str, Dv::S("")),
    f(
        "smtp_port",
        FieldKind::Int {
            min: 1,
            max: 65535,
        },
        Dv::I(587),
    ),
    f("smtp_username", FieldKind::Str, Dv::S("")),
    fn_(
        "smtp_password_env_var",
        FieldKind::EnvVarName,
        Dv::S("EMAIL_PASSWORD"),
        "an env var NAME, not a secret",
    ),
    f("smtp_use_starttls", FieldKind::Bool, Dv::B(true)),
    f("from_email", FieldKind::OptStr, Dv::Null),
    f("reply_to", FieldKind::OptStr, Dv::Null),
    f("imap_host", FieldKind::Str, Dv::S("")),
    f(
        "imap_port",
        FieldKind::Int {
            min: 1,
            max: 65535,
        },
        Dv::I(993),
    ),
    f("imap_username", FieldKind::Str, Dv::S("")),
    fn_(
        "imap_password_env_var",
        FieldKind::EnvVarName,
        Dv::S("EMAIL_PASSWORD"),
        "an env var NAME, not a secret",
    ),
    f("imap_folder", FieldKind::Str, Dv::S("INBOX")),
];

/// The 17 sections, in the display order of the overview. The two
/// non-scalar sections (Routes/Profiles) carry no field table — their
/// authoritative display is the CLI-derived view.
pub const SECTIONS: &[SectionSpec] = &[
    SectionSpec {
        name: "default_models",
        title: "Default models",
        kind: SectionKind::Fields,
        fields: DEFAULT_MODELS_FIELDS,
    },
    SectionSpec {
        name: "app_defaults",
        title: "App defaults",
        kind: SectionKind::Fields,
        fields: APP_DEFAULTS_FIELDS,
    },
    SectionSpec {
        name: "capability_defaults",
        title: "Capability routes",
        kind: SectionKind::Routes,
        fields: &[],
    },
    SectionSpec {
        name: "provider_profiles",
        title: "Provider profiles",
        kind: SectionKind::Profiles,
        fields: &[],
    },
    SectionSpec {
        name: "api_keys",
        title: "API keys",
        kind: SectionKind::Fields,
        fields: API_KEYS_FIELDS,
    },
    SectionSpec {
        name: "vision",
        title: "Vision",
        kind: SectionKind::Fields,
        fields: VISION_FIELDS,
    },
    SectionSpec {
        name: "audio",
        title: "Audio",
        kind: SectionKind::Fields,
        fields: AUDIO_FIELDS,
    },
    SectionSpec {
        name: "video",
        title: "Video",
        kind: SectionKind::Fields,
        fields: VIDEO_FIELDS,
    },
    SectionSpec {
        name: "embeddings",
        title: "Embeddings",
        kind: SectionKind::Fields,
        fields: EMBEDDINGS_FIELDS,
    },
    SectionSpec {
        name: "server",
        title: "Server",
        kind: SectionKind::Fields,
        fields: SERVER_FIELDS,
    },
    SectionSpec {
        name: "cache",
        title: "Cache",
        kind: SectionKind::Fields,
        fields: CACHE_FIELDS,
    },
    SectionSpec {
        name: "logging",
        title: "Logging",
        kind: SectionKind::Fields,
        fields: LOGGING_FIELDS,
    },
    SectionSpec {
        name: "streaming",
        title: "Streaming",
        kind: SectionKind::Fields,
        fields: STREAMING_FIELDS,
    },
    SectionSpec {
        name: "timeouts",
        title: "Timeouts",
        kind: SectionKind::Fields,
        fields: TIMEOUTS_FIELDS,
    },
    SectionSpec {
        name: "offline",
        title: "Offline",
        kind: SectionKind::Fields,
        fields: OFFLINE_FIELDS,
    },
    SectionSpec {
        name: "maintenance",
        title: "Maintenance",
        kind: SectionKind::Fields,
        fields: MAINTENANCE_FIELDS,
    },
    SectionSpec {
        name: "email",
        title: "Email",
        kind: SectionKind::Fields,
        fields: EMAIL_FIELDS,
    },
];

/// The top-level meta flag beside the sections (manager.py:514-521,
/// 616). A legacy nested `audio.strategy_explicit` is also accepted.
pub const META_FLAG: &str = "audio_strategy_explicit";

pub fn section(name: &str) -> Option<&'static SectionSpec> {
    SECTIONS.iter().find(|s| s.name == name)
}

/// Validate one field value against the kind Python actually enforces.
/// `Ok(())` for anything Python would load and use without misbehaving;
/// `Err(reason)` marks the field broken in the mirror. The reasons name
/// the accepted shape — they double as editor help.
pub fn validate(kind: &FieldKind, v: &serde_json::Value) -> Result<(), String> {
    use serde_json::Value;
    let type_err = |want: &str| -> Result<(), String> {
        Err(format!("expected {want}, file has {}", type_name(v)))
    };
    match kind {
        FieldKind::Str | FieldKind::Path | FieldKind::EnvVarName => match v {
            Value::String(_) => Ok(()),
            _ => type_err("a string"),
        },
        FieldKind::OptStr | FieldKind::OptPath | FieldKind::Secret => match v {
            Value::Null | Value::String(_) => Ok(()),
            _ => type_err("a string or null"),
        },
        FieldKind::Bool => match v {
            Value::Bool(_) => Ok(()),
            _ => type_err("true/false"),
        },
        FieldKind::Int { min, max } => int_in(v, *min, *max, false),
        FieldKind::OptInt { min, max } => int_in(v, *min, *max, true),
        FieldKind::Float { min } => match v.as_f64() {
            Some(f) if f >= *min => Ok(()),
            Some(f) => Err(format!("{f} is below the minimum {min}")),
            None if v.is_null() => type_err("a number"),
            None => type_err("a number"),
        },
        FieldKind::FloatFree => match v.as_f64() {
            Some(_) => Ok(()),
            None => type_err("a number"),
        },
        FieldKind::Enum(choices) => match v.as_str() {
            Some(s) if choices.contains(&s) => Ok(()),
            Some(s) => Err(format!("\"{s}\" is not one of: {}", choices.join(", "))),
            None => type_err("a string"),
        },
        FieldKind::OptEnum(choices) => match v {
            Value::Null => Ok(()),
            _ => match v.as_str() {
                Some(s) if choices.contains(&s) => Ok(()),
                Some(s) => Err(format!("\"{s}\" is not one of: {}", choices.join(", "))),
                None => type_err("a string or null"),
            },
        },
        FieldKind::EnumLoose { canon, accepted } => match v.as_str() {
            Some(s) if accepted.contains(&s) => Ok(()),
            Some(s) => Err(format!("\"{s}\" is not one of: {}", canon.join(", "))),
            None => type_err("a string"),
        },
        FieldKind::FallbackChain => match v {
            Value::Array(items) => {
                for (i, it) in items.iter().enumerate() {
                    let ok = it.get("provider").map(Value::is_string).unwrap_or(false)
                        && it.get("model").map(Value::is_string).unwrap_or(false);
                    if !ok {
                        return Err(format!("entry {i} is not {{provider, model}}"));
                    }
                }
                Ok(())
            }
            _ => type_err("a list of {provider, model}"),
        },
    }
}

fn int_in(v: &serde_json::Value, min: i64, max: i64, nullable: bool) -> Result<(), String> {
    if v.is_null() {
        return if nullable {
            Ok(())
        } else {
            Err("expected an integer, file has null".into())
        };
    }
    // Float-typed integers (3.0) are the same value to Python's
    // untyped dataclass load — accept them, exactly like Dv::matches
    // does (review P3-3: holding both stances was the defect).
    let as_int = v.as_i64().or_else(|| {
        v.as_f64()
            .filter(|f| f.fract() == 0.0 && f.is_finite())
            .map(|f| f as i64)
    });
    match as_int {
        Some(i) if i >= min && i <= max => Ok(()),
        Some(i) if max == i64::MAX => Err(format!("{i} is below the minimum {min}")),
        Some(i) => Err(format!("{i} is outside {min}..={max}")),
        None => Err(format!("expected an integer, file has {}", type_name(v))),
    }
}

fn type_name(v: &serde_json::Value) -> &'static str {
    use serde_json::Value;
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a bool",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn field_count_matches_the_inventory() {
        let scalar: usize = SECTIONS.iter().map(|s| s.fields.len()).sum();
        assert_eq!(scalar, 85, "inventory §2 counts 85 scalar fields");
        assert_eq!(SECTIONS.len(), 17, "17 sections (+ 1 top-level meta flag)");
    }

    #[test]
    fn defaults_match_values() {
        assert!(Dv::S("auto").matches(&json!("auto")));
        assert!(Dv::I(3).matches(&json!(3)));
        assert!(Dv::I(3).matches(&json!(3.0)), "JSON 3.0 == default 3");
        assert!(Dv::F(7200.0).matches(&json!(7200)));
        assert!(Dv::Null.matches(&json!(null)));
        assert!(Dv::EmptyList.matches(&json!([])));
        assert!(!Dv::EmptyList.matches(&json!([{"provider": "x"}])));
    }

    #[test]
    fn validation_names_the_accepted_shape() {
        let level = FieldKind::Enum(LOG_LEVELS);
        assert!(validate(&level, &json!("ERROR")).is_ok());
        let err = validate(&level, &json!("chatty")).unwrap_err();
        assert!(err.contains("DEBUG"), "error teaches the choices: {err}");

        let port = FieldKind::OptInt { min: 1, max: 65535 };
        assert!(validate(&port, &json!(null)).is_ok());
        assert!(validate(&port, &json!(8000)).is_ok());
        assert!(validate(&port, &json!(0)).is_err());
        assert!(validate(&port, &json!("8000")).is_err(), "strings are not ints");
        // Float-typed integers are the same value to Python's untyped
        // load (review P3-3 — validate now agrees with Dv::matches).
        assert!(validate(&port, &json!(8000.0)).is_ok());
        assert!(validate(&port, &json!(8000.5)).is_err());

        // Nullable enums/strings: Optional[str] fields load null fine
        // (review P3-2).
        let oe = FieldKind::OptEnum(EMBEDDING_PROVIDERS);
        assert!(validate(&oe, &json!(null)).is_ok());
        assert!(validate(&oe, &json!("vllm")).is_ok());
        assert!(validate(&oe, &json!("bogus")).is_err());

        let t = FieldKind::Float { min: 0.0 };
        assert!(validate(&t, &json!(0)).is_ok(), "0 = unlimited is legal");
        assert!(validate(&t, &json!(-1)).is_err(), "negatives rejected");

        // The audio strategy aliases load fine; garbage is named.
        let audio = AUDIO_FIELDS[0].kind;
        assert!(validate(&audio, &json!("stt")).is_ok());
        assert!(validate(&audio, &json!("")).is_ok());
        assert!(validate(&audio, &json!("loud")).is_err());

        let chain = FieldKind::FallbackChain;
        assert!(validate(&chain, &json!([{"provider": "ollama", "model": "x"}])).is_ok());
        assert!(validate(&chain, &json!([{"provider": "ollama"}])).is_err());
        assert!(validate(&chain, &json!("nope")).is_err());
    }
}
