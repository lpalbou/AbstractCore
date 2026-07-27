//! The write vocabulary: how every editable surface writes, and how
//! every write VERIFIES. Grounded in live-probed CLI behavior
//! (docs/backlog/proposed/0001_write_lane_design.md):
//!
//! - CLI setters first — they enforce the coupled-field invariants
//!   (global default ↔ route input.text, embeddings ↔ embedding.text,
//!   audio strategy ↔ the explicit flag) a direct write silently breaks.
//! - Direct read-modify-write ONLY for fields with no CLI setter;
//!   unknown keys preserved by construction (mutate the fresh raw
//!   Value in place).
//! - Neither of the CLI's success signals is trustworthy on its own
//!   (`--set-*` exits 0 on refusals; `--set-app-default` prints ✅ for
//!   writes it drops) — every WriteSpec carries value-level
//!   expectations checked against a FRESH re-read.

use serde_json::Value;

use crate::config::fingerprint;

/// One subprocess argument. Debug is test/journal surface — secrets
/// redact structurally so a new call site cannot leak one.
#[derive(Clone)]
pub enum Arg {
    Plain(String),
    Secret(String),
}

impl Arg {
    pub fn value(&self) -> &str {
        match self {
            Arg::Plain(s) | Arg::Secret(s) => s,
        }
    }
    pub fn p(s: impl Into<String>) -> Arg {
        Arg::Plain(s.into())
    }
}

impl std::fmt::Debug for Arg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Arg::Plain(s) => write!(f, "{s:?}"),
            Arg::Secret(_) => write!(f, "«redacted»"),
        }
    }
}

/// A direct file mutation (data, not closures — Debug-able, testable,
/// and structurally unable to smuggle behavior). Secrets never travel
/// this lane: every secret field has a CLI setter.
#[derive(Clone, Debug, PartialEq)]
pub enum RmwOp {
    /// `section.key = value`.
    SetField {
        section: String,
        key: String,
        value: Value,
    },
    /// Remove `section.key` (absent ≡ dataclass default).
    ClearField { section: String, key: String },
    /// Remove one `{provider, model}` entry from a fallback chain by
    /// index (the CLI can only append — inventory §2 vision).
    RemoveFallbackEntry { section: String, index: usize },
    /// Null both global default fields (no CLI clear exists — probed:
    /// `--set-global-default ''` is a silent no-op).
    ClearGlobalDefaultFields,
    /// Remove audio.strategy AND the explicit flag — both its
    /// top-level spelling and the legacy nested one
    /// (manager.py:514-521) — restoring the smart-default state.
    ResetAudioStrategy,
}

impl RmwOp {
    /// Apply to a fresh raw config object. Errors name what refused.
    pub fn apply(&self, raw: &mut Value) -> Result<(), String> {
        let obj = raw
            .as_object_mut()
            .ok_or("config root is not an object")?;
        match self {
            RmwOp::SetField {
                section,
                key,
                value,
            } => {
                let sec = obj
                    .entry(section.clone())
                    .or_insert_with(|| Value::Object(Default::default()));
                let sec = sec.as_object_mut().ok_or_else(|| {
                    format!("section {section} is not an object — fix it by hand first")
                })?;
                sec.insert(key.clone(), value.clone());
                Ok(())
            }
            RmwOp::ClearField { section, key } => {
                if let Some(sec) = obj.get_mut(section).and_then(Value::as_object_mut) {
                    sec.remove(key);
                }
                Ok(())
            }
            RmwOp::RemoveFallbackEntry { section, index } => {
                let chain = obj
                    .get_mut(section)
                    .and_then(Value::as_object_mut)
                    .and_then(|s| s.get_mut("fallback_chain"))
                    .and_then(Value::as_array_mut)
                    .ok_or_else(|| format!("{section}.fallback_chain is not a list"))?;
                if *index >= chain.len() {
                    return Err(format!(
                        "entry {index} no longer exists (the chain has {})",
                        chain.len()
                    ));
                }
                chain.remove(*index);
                Ok(())
            }
            RmwOp::ClearGlobalDefaultFields => {
                if let Some(dm) = obj.get_mut("default_models").and_then(Value::as_object_mut) {
                    dm.insert("global_provider".into(), Value::Null);
                    dm.insert("global_model".into(), Value::Null);
                }
                Ok(())
            }
            RmwOp::ResetAudioStrategy => {
                obj.remove(crate::schema::META_FLAG);
                if let Some(audio) = obj.get_mut("audio").and_then(Value::as_object_mut) {
                    audio.remove("strategy");
                    audio.remove("strategy_explicit");
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum WriteVerb {
    /// `abstractcore <args>` — exit-0 `❌ Error:` lines are failures
    /// (the flags CLI's exit codes lie; probed).
    Cli(Vec<Arg>),
    /// Direct read-modify-write: fresh read → mutate → tmp+rename 0600.
    Rmw(RmwOp),
}

/// A value-level expectation checked against the FRESH post-write
/// state. Never carries secret material — secret writes verify by
/// fingerprint.
#[derive(Clone, Debug, PartialEq)]
pub enum Expect {
    /// `path` in the config file equals `value`.
    Eq { path: Vec<String>, value: Value },
    /// `path` is absent, null, or the empty string.
    Cleared { path: Vec<String> },
    /// `path` is a string whose sha256[:8] matches.
    SecretFp { path: Vec<String>, fp: String },
    /// The derived routes view shows `key` configured with this pair.
    RouteEq {
        key: String,
        provider: Option<String>,
        model: Option<String>,
    },
    /// The derived routes view shows `key` unconfigured.
    RouteCleared { key: String },
    /// The derived profiles view contains `id`.
    ProfileExists { id: String },
    ProfileAbsent { id: String },
}

impl Expect {
    pub fn needs_routes(&self) -> bool {
        matches!(self, Expect::RouteEq { .. } | Expect::RouteCleared { .. })
    }
    pub fn needs_profiles(&self) -> bool {
        matches!(self, Expect::ProfileExists { .. } | Expect::ProfileAbsent { .. })
    }
}

/// One user-level write action: verbs run in order, then every
/// expectation must hold on a fresh re-read (+ fresh derived views
/// where needed). The whole spec refuses to run unless the file state
/// is Ready or Missing and unchanged since `base_stamp`.
#[derive(Clone, Debug)]
pub struct WriteSpec {
    /// Human line for the journal and the busy strip. Never a secret.
    pub label: String,
    pub verbs: Vec<WriteVerb>,
    pub expects: Vec<Expect>,
    /// The file identity (mtime, ino, size) of the snapshot the
    /// operator edited FROM — a different identity at write time means
    /// another writer landed in between (no lock exists; whole-file
    /// rewrites are last-writer-wins), so the write refuses and asks
    /// for a reload.
    pub base_stamp: Option<crate::config::FileStamp>,
    /// Present when a form modal awaits the outcome.
    pub form_id: Option<u64>,
}

impl WriteSpec {
    pub fn needs_routes(&self) -> bool {
        self.expects.iter().any(Expect::needs_routes)
    }
    pub fn needs_profiles(&self) -> bool {
        self.expects.iter().any(Expect::needs_profiles)
    }
}

/// Path walk that understands BOTH containers: object keys and array
/// indices. `Value::get(&String)` indexes only objects — numeric
/// segments against arrays returned None unconditionally, so every
/// fallback-chain expectation false-failed on add and vacuously passed
/// on remove (M2 review P1-1, live-proven).
fn walk<'v>(raw: &'v Value, path: &[String]) -> Option<&'v Value> {
    let mut cur = raw;
    for seg in path {
        cur = match cur {
            Value::Array(items) => items.get(seg.parse::<usize>().ok()?)?,
            other => other.get(seg)?,
        };
    }
    Some(cur)
}

/// Evaluate one FILE expectation against the raw post-write value.
/// Ok carries the human proof line for the journal.
pub fn eval_file_expect(raw: &Value, expect: &Expect) -> Result<String, String> {
    let get = |path: &[String]| -> Option<&Value> { walk(raw, path) };
    let dotted = |path: &[String]| path.join(".");
    match expect {
        Expect::Eq { path, value } => {
            let got = get(path);
            if got == Some(value) {
                Ok(format!("{} = {}", dotted(path), value))
            } else {
                Err(format!(
                    "{} is {} (expected {})",
                    dotted(path),
                    got.map(Value::to_string).unwrap_or_else(|| "absent".into()),
                    value
                ))
            }
        }
        Expect::Cleared { path } => match get(path) {
            None | Some(Value::Null) => Ok(format!("{} cleared", dotted(path))),
            Some(Value::String(s)) if s.trim().is_empty() => {
                Ok(format!("{} cleared", dotted(path)))
            }
            Some(v) => Err(format!("{} still holds {}", dotted(path), redact_scalar(v))),
        },
        Expect::SecretFp { path, fp } => match get(path).and_then(Value::as_str) {
            Some(s) if fingerprint(s) == *fp => {
                Ok(format!("{} stored (fp {fp})", dotted(path)))
            }
            Some(_) => Err(format!(
                "{} holds a DIFFERENT value (fingerprint mismatch)",
                dotted(path)
            )),
            None => Err(format!("{} is not set", dotted(path))),
        },
        // Route/profile expects are evaluated against derived views by
        // the worker, not here.
        other => Err(format!("not a file expectation: {other:?}")),
    }
}

/// Displayed in verification errors — a leftover SECRET value must not
/// echo (Cleared on a secret path that still holds a key).
fn redact_scalar(v: &Value) -> String {
    match v {
        Value::String(_) => "a value («redacted»)".into(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------
// The field-route table: how each scalar field writes (inventory §2's
// CLI column, live-probed). Pair-coupled fields and secrets route to
// dedicated editors.
// ---------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldRoute {
    /// One CLI flag; the value is appended as the next argv.
    Set(&'static str),
    /// Same, and `''` clears the field (probed: writes null).
    SetBlankClears(&'static str),
    /// Bool via two flags.
    Toggle {
        on: &'static str,
        off: &'static str,
    },
    /// No CLI setter — direct read-modify-write.
    Rmw,
    /// Provider/model pair coupling — a dedicated pair editor writes
    /// both halves through ONE setter.
    Pair,
    /// Plaintext secret — the masked editor (CLI setter, fp verify).
    Secret,
    /// vision.strategy: three values, three different verbs.
    VisionStrategy,
}

pub fn field_route(section: &str, key: &str) -> FieldRoute {
    use FieldRoute as R;
    match (section, key) {
        ("vision", "strategy") => R::VisionStrategy,
        ("vision", "caption_provider" | "caption_model") => R::Pair,
        ("vision", "fallback_chain") => R::Rmw, // append has a CLI; removal doesn't — the chain editor mixes both
        ("vision", "local_models_path") => R::Rmw,

        ("audio", "strategy") => R::Set("--set-audio-strategy"),
        ("audio", "stt_backend_id") => R::SetBlankClears("--set-stt-backend-id"),
        ("audio", "stt_language") => R::SetBlankClears("--set-stt-language"),
        ("audio", _) => R::Rmw, // reserved caption_*/fallback_chain

        ("video", "strategy") => R::Set("--set-video-strategy"),
        ("video", "max_frames") => R::Set("--set-video-max-frames"),
        ("video", "max_frames_native") => R::Set("--set-video-max-frames-native"),
        ("video", "frame_format") => R::Set("--set-video-frame-format"),
        ("video", "sampling_strategy") => R::Set("--set-video-sampling-strategy"),
        ("video", "max_frame_side") => R::Set("--set-video-max-frame-side"),
        ("video", "max_video_size_bytes") => R::Set("--set-video-max-size-bytes"),

        ("embeddings", "provider" | "model") => R::Pair,
        ("embeddings", "base_url") => R::SetBlankClears("--set-embeddings-base-url"),

        ("app_defaults", _) => R::Pair,

        ("default_models", "global_provider" | "global_model") => R::Pair,
        ("default_models", "chat_model") => R::Set("--set-chat-model"),
        ("default_models", "code_model") => R::Set("--set-code-model"),

        ("api_keys", _) => R::Secret,

        ("server", "auth_token") => R::Secret,
        ("server", "allow_unauthenticated") => R::Toggle {
            on: "--allow-unauthenticated-server",
            off: "--disallow-unauthenticated-server",
        },
        ("server", "base_url_allowlist") => R::SetBlankClears("--set-server-base-url-allowlist"),
        ("server", "url_fetch_allowlist") => R::SetBlankClears("--set-server-url-fetch-allowlist"),
        ("server", "media_root") => R::SetBlankClears("--set-server-media-root"),
        ("server", "allow_local_files") => R::Toggle {
            on: "--allow-server-local-files",
            off: "--disallow-server-local-files",
        },
        ("server", "host") => R::Set("--set-server-host"),
        ("server", "port") => R::Set("--set-server-port"),

        ("cache", "default_cache_dir") => R::Set("--set-default-cache-dir"),
        ("cache", "huggingface_cache_dir") => R::Set("--set-huggingface-cache-dir"),
        ("cache", "local_models_cache_dir") => R::Set("--set-local-models-cache-dir"),
        ("cache", "glyph_cache_dir") => R::Rmw,

        ("logging", "console_level") => R::Set("--set-console-log-level"),
        ("logging", "file_level") => R::Set("--set-file-log-level"),
        ("logging", "file_logging_enabled") => R::Toggle {
            on: "--enable-file-logging",
            off: "--disable-file-logging",
        },
        ("logging", "log_base_dir") => R::Set("--set-log-base-dir"),
        ("logging", _) => R::Rmw, // verbatim_enabled, console_json, file_json

        ("streaming", "cli_stream_default") => R::Toggle {
            on: "--enable-streaming",
            off: "--disable-streaming",
        },

        ("timeouts", "default_timeout") => R::Set("--set-default-timeout"),
        ("timeouts", "tool_timeout") => R::Set("--set-tool-timeout"),

        // offline, maintenance, email — config-file-only (this console
        // is the first UI for them).
        _ => R::Rmw,
    }
}

// ---------------------------------------------------------------------
// Spec builders — each returns the complete three-phase contract.
// ---------------------------------------------------------------------

fn path2(section: &str, key: &str) -> Vec<String> {
    vec![section.to_string(), key.to_string()]
}

/// A scalar field write through its route. `value` is the TYPED value
/// (the editor validated it against the FieldKind); `cli_value` its
/// argv spelling.
pub fn set_scalar(
    section: &str,
    key: &str,
    value: Value,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> Result<WriteSpec, String> {
    let label = format!("set {section}.{key} = {value}");
    let cli_value = match &value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let (verbs, expects) = match field_route(section, key) {
        FieldRoute::Set(flag) | FieldRoute::SetBlankClears(flag) => (
            vec![WriteVerb::Cli(vec![Arg::p(flag), Arg::p(cli_value)])],
            vec![Expect::Eq {
                path: path2(section, key),
                value,
            }],
        ),
        FieldRoute::Toggle { on, off } => {
            let is_on = value.as_bool().ok_or("toggle needs a bool")?;
            (
                vec![WriteVerb::Cli(vec![Arg::p(if is_on { on } else { off })])],
                vec![Expect::Eq {
                    path: path2(section, key),
                    value,
                }],
            )
        }
        FieldRoute::Rmw => (
            vec![WriteVerb::Rmw(RmwOp::SetField {
                section: section.into(),
                key: key.into(),
                value: value.clone(),
            })],
            vec![Expect::Eq {
                path: path2(section, key),
                value,
            }],
        ),
        other => return Err(format!("{section}.{key} routes to {other:?} — use its editor")),
    };
    Ok(WriteSpec {
        label,
        verbs,
        expects,
        base_stamp: base,
        form_id,
    })
}

/// Clear an optional scalar: blank-clearing setters go through the
/// CLI; everything else removes the key directly.
pub fn clear_scalar(
    section: &str,
    key: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> Result<WriteSpec, String> {
    let label = format!("clear {section}.{key}");
    let verbs = match field_route(section, key) {
        FieldRoute::SetBlankClears(flag) => {
            vec![WriteVerb::Cli(vec![Arg::p(flag), Arg::p("")])]
        }
        FieldRoute::Set(_) | FieldRoute::Rmw => vec![WriteVerb::Rmw(RmwOp::ClearField {
            section: section.into(),
            key: key.into(),
        })],
        other => return Err(format!("{section}.{key} routes to {other:?} — use its editor")),
    };
    Ok(WriteSpec {
        label,
        verbs,
        expects: vec![Expect::Cleared {
            path: path2(section, key),
        }],
        base_stamp: base,
        form_id,
    })
}

/// The global default pair — the COUPLED write (also writes route
/// input.text; manager.py:1323-1338).
pub fn set_global_default(
    provider: &str,
    model: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("set global default = {provider}/{model} (+ route input.text)"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-global-default"),
            Arg::p(format!("{provider}/{model}")),
        ])],
        expects: vec![
            Expect::Eq {
                path: path2("default_models", "global_provider"),
                value: Value::String(provider.into()),
            },
            Expect::Eq {
                path: path2("default_models", "global_model"),
                value: Value::String(model.into()),
            },
            Expect::RouteEq {
                key: "input.text".into(),
                provider: Some(provider.into()),
                model: Some(model.into()),
            },
        ],
        base_stamp: base,
        form_id,
    }
}

/// Clearing the global default has NO CLI (probed) — null the fields
/// directly AND clear the coupled route through the honest CLI verb.
/// VERB ORDER is load-bearing (M2 review P2-2): the fallible CLI half
/// runs FIRST — if it refuses, nothing has been applied; the RMW half
/// over a just-read fresh file is the near-infallible tail. The
/// reverse order could null the fields and then fail the route clear,
/// desyncing status from runtime with an error that implies nothing
/// happened.
pub fn clear_global_default(base: Option<crate::config::FileStamp>, form_id: Option<u64>) -> WriteSpec {
    WriteSpec {
        label: "clear global default (+ route input.text)".into(),
        verbs: vec![
            WriteVerb::Cli(vec![
                Arg::p("config"),
                Arg::p("clear-default"),
                Arg::p("input.text"),
            ]),
            WriteVerb::Rmw(RmwOp::ClearGlobalDefaultFields),
        ],
        expects: vec![
            Expect::Cleared {
                path: path2("default_models", "global_provider"),
            },
            Expect::Cleared {
                path: path2("default_models", "global_model"),
            },
            Expect::RouteCleared {
                key: "input.text".into(),
            },
        ],
        base_stamp: base,
        form_id,
    }
}

pub fn set_app_default(
    app: &str,
    provider: &str,
    model: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("set app default {app} = {provider}/{model}"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-app-default"),
            Arg::p(app),
            Arg::p(provider),
            Arg::p(model),
        ])],
        // The CLI prints ✅ even when it drops the write (probed) —
        // these expectations are the only truth.
        expects: vec![
            Expect::Eq {
                path: path2("app_defaults", &format!("{app}_provider")),
                value: Value::String(provider.into()),
            },
            Expect::Eq {
                path: path2("app_defaults", &format!("{app}_model")),
                value: Value::String(model.into()),
            },
        ],
        base_stamp: base,
        form_id,
    }
}

pub fn set_embeddings(
    provider: &str,
    model: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("set embeddings = {provider}/{model} (+ route embedding.text)"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-embeddings-model"),
            Arg::p(format!("{provider}/{model}")),
        ])],
        expects: vec![
            Expect::Eq {
                path: path2("embeddings", "provider"),
                value: Value::String(provider.into()),
            },
            Expect::Eq {
                path: path2("embeddings", "model"),
                value: Value::String(model.into()),
            },
            Expect::RouteEq {
                key: "embedding.text".into(),
                provider: Some(provider.into()),
                model: Some(model.into()),
            },
        ],
        base_stamp: base,
        form_id,
    }
}

pub fn set_vision_pair(
    provider: &str,
    model: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("set vision fallback = {provider}/{model} (strategy two_stage)"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-vision-provider"),
            Arg::p(provider),
            Arg::p(model),
        ])],
        expects: vec![
            Expect::Eq {
                path: path2("vision", "strategy"),
                value: Value::String("two_stage".into()),
            },
            Expect::Eq {
                path: path2("vision", "caption_provider"),
                value: Value::String(provider.into()),
            },
            Expect::Eq {
                path: path2("vision", "caption_model"),
                value: Value::String(model.into()),
            },
        ],
        base_stamp: base,
        form_id,
    }
}

/// vision.strategy: three values, three verbs (two_stage needs the
/// pair editor; disabled has a flag; basic_metadata has no CLI).
///
/// `--disable-vision` is DESTRUCTIVE beyond the enum: Python's
/// `disable_vision` nulls the caption pair AND empties the whole
/// fallback_chain (manager.py:1586-1596) — the spec verifies all of it
/// so the journal records what the verb really did (M2 review P2-1).
pub fn set_vision_strategy(
    strategy: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> Result<WriteSpec, String> {
    let (label, verbs, expects) = match strategy {
        "disabled" => (
            "set vision.strategy = disabled (clears caption pair + fallback chain)".to_string(),
            vec![WriteVerb::Cli(vec![Arg::p("--disable-vision")])],
            vec![
                Expect::Eq {
                    path: path2("vision", "strategy"),
                    value: Value::String("disabled".into()),
                },
                Expect::Cleared {
                    path: path2("vision", "caption_provider"),
                },
                Expect::Cleared {
                    path: path2("vision", "caption_model"),
                },
                Expect::Eq {
                    path: path2("vision", "fallback_chain"),
                    value: Value::Array(Vec::new()),
                },
            ],
        ),
        "basic_metadata" => (
            "set vision.strategy = basic_metadata".to_string(),
            vec![WriteVerb::Rmw(RmwOp::SetField {
                section: "vision".into(),
                key: "strategy".into(),
                value: Value::String("basic_metadata".into()),
            })],
            vec![Expect::Eq {
                path: path2("vision", "strategy"),
                value: Value::String("basic_metadata".into()),
            }],
        ),
        "two_stage" => {
            return Err(
                "two_stage needs a caption provider/model — use the vision pair editor".into(),
            )
        }
        other => return Err(format!("unknown vision strategy {other:?}")),
    };
    Ok(WriteSpec {
        label,
        verbs,
        expects,
        base_stamp: base,
        form_id,
    })
}

/// Reset audio.strategy to the TRUE default state: the value removed
/// AND the explicit flag cleared (both spellings). Going through the
/// CLI setter would set `audio_strategy_explicit = true` — after which
/// the smart default stops applying, which is NOT the default state
/// (M2 review P2-4).
pub fn reset_audio_strategy(base: Option<crate::config::FileStamp>, form_id: Option<u64>) -> WriteSpec {
    WriteSpec {
        label: "reset audio.strategy (clears the explicit flag — smart default applies again)"
            .into(),
        verbs: vec![WriteVerb::Rmw(RmwOp::ResetAudioStrategy)],
        expects: vec![
            Expect::Cleared {
                path: path2("audio", "strategy"),
            },
            Expect::Cleared {
                path: vec![crate::schema::META_FLAG.to_string()],
            },
        ],
        base_stamp: base,
        form_id,
    }
}

pub fn add_vision_fallback(
    provider: &str,
    model: &str,
    new_len: usize,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("add vision fallback {provider}/{model}"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--add-vision-fallback"),
            Arg::p(provider),
            Arg::p(model),
        ])],
        // Order/index verification would over-specify the CLI's append
        // — length + membership is what the operator asked for.
        expects: vec![Expect::Eq {
            path: vec![
                "vision".into(),
                "fallback_chain".into(),
                (new_len - 1).to_string(),
                "provider".into(),
            ],
            value: Value::String(provider.into()),
        }],
        base_stamp: base,
        form_id,
    }
}

pub fn remove_vision_fallback(
    index: usize,
    old_len: usize,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("remove vision fallback entry {index}"),
        verbs: vec![WriteVerb::Rmw(RmwOp::RemoveFallbackEntry {
            section: "vision".into(),
            index,
        })],
        // The removed slot's successor shifts down — verify the length.
        expects: vec![if old_len <= 1 {
            Expect::Cleared {
                path: vec![
                    "vision".into(),
                    "fallback_chain".into(),
                    "0".into(),
                    "provider".into(),
                ],
            }
        } else {
            Expect::Cleared {
                path: vec![
                    "vision".into(),
                    "fallback_chain".into(),
                    (old_len - 1).to_string(),
                    "provider".into(),
                ],
            }
        }],
        base_stamp: base,
        form_id,
    }
}

/// Set an api_keys entry (masked editor). Argv carries the key — a
/// documented, accepted local-tool tradeoff (risk map #3); it never
/// reaches the journal or the screen.
pub fn set_api_key(
    name: &str,
    key: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("set API key {name}"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-api-key"),
            Arg::p(name),
            Arg::Secret(key.to_string()),
        ])],
        expects: vec![Expect::SecretFp {
            path: path2("api_keys", name),
            fp: fingerprint(key),
        }],
        base_stamp: base,
        form_id,
    }
}

/// Clearing stores `""` (probed) — semantically not-set to Python.
pub fn clear_api_key(
    name: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("clear API key {name}"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-api-key"),
            Arg::p(name),
            Arg::p(""),
        ])],
        expects: vec![Expect::Cleared {
            path: path2("api_keys", name),
        }],
        base_stamp: base,
        form_id,
    }
}

pub fn set_server_auth_token(
    token: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: "set server auth token".into(),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-server-auth-token"),
            Arg::Secret(token.to_string()),
        ])],
        expects: vec![Expect::SecretFp {
            path: path2("server", "auth_token"),
            fp: fingerprint(token),
        }],
        base_stamp: base,
        form_id,
    }
}

pub fn clear_server_auth_token(
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: "clear server auth token".into(),
        verbs: vec![WriteVerb::Cli(vec![Arg::p("--clear-server-auth-token")])],
        expects: vec![Expect::Cleared {
            path: path2("server", "auth_token"),
        }],
        base_stamp: base,
        form_id,
    }
}

/// Set a capability route (`config set-default` — honest exit codes).
/// `output.text` writes redirect to input.text Python-side; the UI
/// blocks editing it instead of relying on the redirect.
#[allow(clippy::too_many_arguments)]
pub fn set_route(
    key: &str,
    provider: Option<&str>,
    model: Option<&str>,
    base_url: Option<&str>,
    reasoning: Option<&str>,
    options: &[(String, String)],
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    let mut args = vec![Arg::p("config"), Arg::p("set-default"), Arg::p(key)];
    if let Some(p) = provider {
        args.push(Arg::p("--provider"));
        args.push(Arg::p(p));
    }
    if let Some(m) = model {
        args.push(Arg::p("--model"));
        args.push(Arg::p(m));
    }
    if let Some(u) = base_url {
        args.push(Arg::p("--base-url"));
        args.push(Arg::p(u));
    }
    // set-default is REPLACE semantics — a field the editor doesn't
    // resend is a field it deletes (M2 review P2-5: reasoning).
    if let Some(r) = reasoning {
        args.push(Arg::p("--reasoning"));
        args.push(Arg::p(r));
    }
    for (k, v) in options {
        args.push(Arg::p("--option"));
        args.push(Arg::p(format!("{k}={v}")));
    }
    WriteSpec {
        label: format!(
            "set route {key} = {}/{}",
            provider.unwrap_or("—"),
            model.unwrap_or("—")
        ),
        verbs: vec![WriteVerb::Cli(args)],
        expects: vec![Expect::RouteEq {
            key: key.to_string(),
            provider: provider.map(str::to_string),
            model: model.map(str::to_string),
        }],
        base_stamp: base,
        form_id,
    }
}

pub fn clear_route(key: &str, base: Option<crate::config::FileStamp>, form_id: Option<u64>) -> WriteSpec {
    WriteSpec {
        label: format!("clear route {key}"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("config"),
            Arg::p("clear-default"),
            Arg::p(key),
        ])],
        expects: vec![Expect::RouteCleared {
            key: key.to_string(),
        }],
        base_stamp: base,
        form_id,
    }
}

/// Save a provider endpoint profile (`config set-provider` — honest
/// exit codes; the setter validates id/family/url Python-side).
#[allow(clippy::too_many_arguments)]
pub fn save_profile(
    id: &str,
    family: &str,
    base_url: &str,
    api_key: Option<&str>,
    clear_key: bool,
    display_name: &str,
    description: &str,
    enabled: bool,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    let mut args = vec![
        Arg::p("config"),
        Arg::p("set-provider"),
        Arg::p(id),
        Arg::p("--family"),
        Arg::p(family),
    ];
    if !base_url.is_empty() {
        args.push(Arg::p("--base-url"));
        args.push(Arg::p(base_url));
    }
    if let Some(k) = api_key {
        args.push(Arg::p("--api-key"));
        args.push(Arg::Secret(k.to_string()));
    } else if clear_key {
        args.push(Arg::p("--clear-api-key"));
    }
    if !display_name.is_empty() {
        args.push(Arg::p("--name"));
        args.push(Arg::p(display_name));
    }
    if !description.is_empty() {
        args.push(Arg::p("--description"));
        args.push(Arg::p(description));
    }
    args.push(Arg::p(if enabled { "--enabled" } else { "--disabled" }));
    WriteSpec {
        label: format!("save provider profile {id}"),
        verbs: vec![WriteVerb::Cli(args)],
        expects: vec![Expect::ProfileExists { id: id.to_string() }],
        base_stamp: base,
        form_id,
    }
}

pub fn delete_profile(id: &str, base: Option<crate::config::FileStamp>, form_id: Option<u64>) -> WriteSpec {
    WriteSpec {
        label: format!("delete provider profile {id}"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("config"),
            Arg::p("delete-provider"),
            Arg::p(id),
        ])],
        expects: vec![Expect::ProfileAbsent { id: id.to_string() }],
        base_stamp: base,
        form_id,
    }
}

pub fn set_audio_strategy(
    strategy: &str,
    base: Option<crate::config::FileStamp>,
    form_id: Option<u64>,
) -> WriteSpec {
    WriteSpec {
        label: format!("set audio.strategy = {strategy} (marks it explicit)"),
        verbs: vec![WriteVerb::Cli(vec![
            Arg::p("--set-audio-strategy"),
            Arg::p(strategy),
        ])],
        expects: vec![
            Expect::Eq {
                path: path2("audio", "strategy"),
                value: Value::String(strategy.into()),
            },
            // The coupled meta flag (manager.py:661-676).
            Expect::Eq {
                path: vec!["audio_strategy_explicit".into()],
                value: Value::Bool(true),
            },
        ],
        base_stamp: base,
        form_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn secrets_redact_in_debug_everywhere() {
        let spec = set_api_key("openai", "sk-very-secret", None, Some(1));
        let dbg = format!("{spec:?}");
        assert!(!dbg.contains("sk-very-secret"), "{dbg}");
        assert!(dbg.contains("«redacted»"));
        let spec = save_profile(
            "x",
            "openai",
            "https://x/v1",
            Some("sk-other-secret"),
            false,
            "",
            "",
            true,
            None,
            None,
        );
        let dbg = format!("{spec:?}");
        assert!(!dbg.contains("sk-other-secret"), "{dbg}");
    }

    #[test]
    fn rmw_ops_mutate_and_preserve() {
        let mut raw = json!({
            "future_section": {"keep": true},
            "logging": {"verbatim_enabled": true, "unknown_knob": 7},
            "vision": {"fallback_chain": [
                {"provider": "a", "model": "m1"},
                {"provider": "b", "model": "m2"}
            ]},
            "default_models": {"global_provider": "x", "global_model": "y"}
        });
        RmwOp::SetField {
            section: "logging".into(),
            key: "verbatim_enabled".into(),
            value: json!(false),
        }
        .apply(&mut raw)
        .unwrap();
        RmwOp::RemoveFallbackEntry {
            section: "vision".into(),
            index: 0,
        }
        .apply(&mut raw)
        .unwrap();
        RmwOp::ClearGlobalDefaultFields.apply(&mut raw).unwrap();
        RmwOp::SetField {
            section: "offline".into(),
            key: "allow_network".into(),
            value: json!(true),
        }
        .apply(&mut raw)
        .unwrap();

        assert_eq!(raw["logging"]["verbatim_enabled"], json!(false));
        assert_eq!(raw["logging"]["unknown_knob"], json!(7), "unknown key kept");
        assert_eq!(raw["future_section"]["keep"], json!(true), "unknown section kept");
        assert_eq!(raw["vision"]["fallback_chain"].as_array().unwrap().len(), 1);
        assert_eq!(raw["vision"]["fallback_chain"][0]["provider"], json!("b"));
        assert_eq!(raw["default_models"]["global_provider"], Value::Null);
        assert_eq!(raw["offline"]["allow_network"], json!(true), "absent section created");

        // Refusals name their reason.
        let err = RmwOp::RemoveFallbackEntry {
            section: "vision".into(),
            index: 9,
        }
        .apply(&mut raw)
        .unwrap_err();
        assert!(err.contains("no longer exists"), "{err}");
    }

    #[test]
    fn file_expects_evaluate_and_redact() {
        let raw = json!({
            "video": {"max_frames": 5},
            "api_keys": {"openai": "sk-live-secret"},
            "audio": {"stt_language": null}
        });
        assert!(eval_file_expect(
            &raw,
            &Expect::Eq {
                path: vec!["video".into(), "max_frames".into()],
                value: json!(5)
            }
        )
        .is_ok());
        let err = eval_file_expect(
            &raw,
            &Expect::Eq {
                path: vec!["video".into(), "max_frames".into()],
                value: json!(9)
            },
        )
        .unwrap_err();
        assert!(err.contains("is 5"), "{err}");

        assert!(eval_file_expect(
            &raw,
            &Expect::Cleared {
                path: vec!["audio".into(), "stt_language".into()]
            }
        )
        .is_ok());
        assert!(eval_file_expect(
            &raw,
            &Expect::Cleared {
                path: vec!["audio".into(), "never_there".into()]
            }
        )
        .is_ok());

        // A still-set secret path must not echo the value.
        let err = eval_file_expect(
            &raw,
            &Expect::Cleared {
                path: vec!["api_keys".into(), "openai".into()]
            },
        )
        .unwrap_err();
        assert!(!err.contains("sk-live-secret"), "{err}");
        assert!(err.contains("«redacted»"), "{err}");

        let ok = eval_file_expect(
            &raw,
            &Expect::SecretFp {
                path: vec!["api_keys".into(), "openai".into()],
                fp: fingerprint("sk-live-secret"),
            },
        )
        .unwrap();
        assert!(!ok.contains("sk-live-secret"), "{ok}");
    }

    /// Every scalar field the schema declares must resolve to SOME
    /// route, and the special ones to their special editors — the
    /// drift guard between schema.rs and this table.
    #[test]
    fn every_schema_field_has_a_route() {
        for section in crate::schema::SECTIONS {
            for fs in section.fields {
                let route = field_route(section.name, fs.key);
                match fs.kind {
                    crate::schema::FieldKind::Secret => {
                        assert_eq!(
                            route,
                            FieldRoute::Secret,
                            "{}.{} must route to the masked editor",
                            section.name,
                            fs.key
                        );
                    }
                    _ => {
                        // No panic = a route exists; pair/special
                        // routes are exercised by their builders.
                    }
                }
            }
        }
        assert_eq!(field_route("default_models", "chat_model"), FieldRoute::Set("--set-chat-model"));
        assert_eq!(field_route("offline", "allow_network"), FieldRoute::Rmw);
        assert_eq!(field_route("email", "smtp_host"), FieldRoute::Rmw);
    }

    /// The walker indexes BOTH containers (M2 review P1-1).
    #[test]
    fn expect_paths_walk_arrays() {
        let raw = json!({"vision": {"fallback_chain": [
            {"provider": "a", "model": "m1"},
            {"provider": "b", "model": "m2"}
        ]}});
        let ok = eval_file_expect(
            &raw,
            &Expect::Eq {
                path: vec![
                    "vision".into(),
                    "fallback_chain".into(),
                    "1".into(),
                    "provider".into(),
                ],
                value: json!("b"),
            },
        )
        .unwrap();
        assert!(ok.contains("fallback_chain.1.provider"), "{ok}");
        // Out-of-range and non-numeric segments stay honest absences.
        assert!(eval_file_expect(
            &raw,
            &Expect::Eq {
                path: vec!["vision".into(), "fallback_chain".into(), "9".into(), "provider".into()],
                value: json!("x"),
            },
        )
        .is_err());
        // And a STILL-PRESENT slot fails a Cleared expectation (the
        // vacuous-pass direction).
        assert!(eval_file_expect(
            &raw,
            &Expect::Cleared {
                path: vec!["vision".into(), "fallback_chain".into(), "0".into(), "provider".into()],
            },
        )
        .is_err());
    }

    #[test]
    fn reset_audio_strategy_clears_value_and_both_flag_spellings() {
        let mut raw = json!({
            "audio_strategy_explicit": true,
            "audio": {"strategy": "speech_to_text", "strategy_explicit": true,
                       "stt_language": "fr"},
        });
        RmwOp::ResetAudioStrategy.apply(&mut raw).unwrap();
        assert!(raw.get("audio_strategy_explicit").is_none());
        assert!(raw["audio"].get("strategy").is_none());
        assert!(raw["audio"].get("strategy_explicit").is_none());
        assert_eq!(raw["audio"]["stt_language"], json!("fr"), "siblings kept");

        let spec = reset_audio_strategy(None, None);
        assert!(matches!(spec.verbs[0], WriteVerb::Rmw(RmwOp::ResetAudioStrategy)));
        assert!(spec
            .expects
            .iter()
            .any(|e| matches!(e, Expect::Cleared { path } if path == &vec!["audio_strategy_explicit".to_string()])));
    }

    /// `--disable-vision` verifies the FULL blast radius (M2 review
    /// P2-1): pair nulled, chain emptied.
    #[test]
    fn disable_vision_spec_verifies_the_wipe() {
        let spec = set_vision_strategy("disabled", None, None).unwrap();
        assert!(spec.label.contains("clears caption pair + fallback chain"));
        let has = |p: &[&str]| {
            spec.expects.iter().any(|e| match e {
                Expect::Cleared { path } => path.iter().map(String::as_str).eq(p.iter().copied()),
                _ => false,
            })
        };
        assert!(has(&["vision", "caption_provider"]));
        assert!(has(&["vision", "caption_model"]));
        assert!(spec.expects.iter().any(|e| matches!(
            e,
            Expect::Eq { path, value } if path[1] == "fallback_chain" && value == &json!([])
        )));
    }

    #[test]
    fn coupled_specs_carry_their_couplings() {
        let g = set_global_default("lmstudio", "m", None, None);
        assert!(g.needs_routes(), "global default verifies route input.text");
        let e = set_embeddings("ollama", "nomic", None, None);
        assert!(e.needs_routes());
        let a = set_audio_strategy("speech_to_text", None, None);
        assert!(a
            .expects
            .iter()
            .any(|x| matches!(x, Expect::Eq { path, .. } if path == &vec!["audio_strategy_explicit".to_string()])));
        let c = clear_global_default(None, None);
        assert_eq!(c.verbs.len(), 2, "RMW fields + CLI route clear");
    }
}
