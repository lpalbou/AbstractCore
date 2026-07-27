//! App state: one struct of signals owned by the UI thread, plus the
//! typed rows parsed out of `abstractcore config … --json` payloads.
//! Nothing here does I/O.

use std::collections::HashMap;
use std::time::Instant;

use abstracttui::prelude::*;
use serde_json::Value;

use crate::cli::{CliError, CliInfo};
use crate::config::{ConfigPath, FileState};

/// Remote/disk data honesty: never render a guess.
#[derive(Clone, Debug, Default)]
pub enum Loadable<T> {
    #[default]
    NotAsked,
    Loading,
    Ready(T),
    Failed(CliError),
}

impl<T> Loadable<T> {
    pub fn ready(&self) -> Option<&T> {
        match self {
            Loadable::Ready(t) => Some(t),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------
// Typed rows (tolerant parses: unknown fields ignored, absent = None —
// the UI renders `—` with a reason, never a guess).
// ---------------------------------------------------------------------

fn s(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}
fn b(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(Value::as_bool)
}

/// The config-file mirror: resolved path + the file's honest state.
#[derive(Clone, Debug)]
pub struct ConfigMirror {
    pub path: ConfigPath,
    pub state: FileState,
    /// UTC HH:MM:SSZ of the last (re)load — the acknowledgment that r
    /// actually did something even when nothing changed.
    pub loaded_at: String,
}

/// One capability route from `config defaults --json` — the derived,
/// coverage-decorated view (main.py:1442-1452 + manager.py:1166-1182).
#[derive(Clone, Debug)]
pub struct RouteRow {
    pub key: String,
    pub kind: String,
    pub modality: String,
    pub label: String,
    pub configured: bool,
    pub source: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub options: Option<Value>,
    pub covered_by: Option<String>,
    pub read_only: bool,
    pub overrideable: bool,
    pub package_hint: Option<String>,
    pub option_examples: Option<Value>,
    /// set-default is REPLACE semantics — the editor must round-trip
    /// this or silently delete it (M2 review P2-5).
    pub reasoning: Option<String>,
}

impl RouteRow {
    pub fn from_value(v: &Value) -> Option<RouteRow> {
        let key = s(v, "key")?;
        Some(RouteRow {
            kind: s(v, "kind").unwrap_or_default(),
            modality: s(v, "modality").unwrap_or_default(),
            label: s(v, "label").unwrap_or_else(|| key.clone()),
            configured: b(v, "configured").unwrap_or(false),
            source: s(v, "source").unwrap_or_default(),
            provider: s(v, "provider").filter(|p| !p.is_empty()),
            model: s(v, "model").filter(|m| !m.is_empty()),
            base_url: s(v, "base_url").filter(|u| !u.is_empty()),
            options: v.get("options").filter(|o| !o.is_null()).cloned(),
            covered_by: s(v, "covered_by"),
            read_only: b(v, "read_only").unwrap_or(false),
            overrideable: b(v, "overrideable").unwrap_or(false),
            package_hint: s(v, "package_hint"),
            option_examples: v.get("option_examples").filter(|o| !o.is_null()).cloned(),
            reasoning: s(v, "reasoning").filter(|r| !r.is_empty()),
            key,
        })
    }

    /// output.text is a read-only alias of input.text
    /// (manager.py:1126-1139) — never render it independently editable.
    pub fn editable(&self) -> bool {
        if self.key == "output.text" {
            return false;
        }
        if self.covered_by.is_some() {
            return self.overrideable && !self.read_only;
        }
        !self.read_only
    }
}

#[derive(Clone, Debug, Default)]
pub struct RoutesData {
    pub ok: bool,
    pub writable: bool,
    /// The config file the CLI says it read — cross-checked against
    /// the console's own resolved path (one-file identity).
    pub config_file: Option<String>,
    pub rows: Vec<RouteRow>,
    pub errors: Vec<String>,
}

impl RoutesData {
    pub fn from_value(v: &Value) -> RoutesData {
        RoutesData {
            ok: b(v, "ok").unwrap_or(false),
            writable: b(v, "writable").unwrap_or(false),
            config_file: s(v, "config_file"),
            rows: v
                .get("routes")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(RouteRow::from_value).collect())
                .unwrap_or_default(),
            errors: v
                .get("errors")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|e| e.as_str().map(str::to_string).unwrap_or_else(|| e.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    pub fn configured_count(&self) -> usize {
        self.rows.iter().filter(|r| r.configured).count()
    }
}

/// One provider endpoint profile from `config providers --json` — the
/// pre-redacted surface (public_dict, provider_profiles.py:168-185).
#[derive(Clone, Debug)]
pub struct ProfileRow {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub family: String,
    pub base_url: String,
    pub api_key_set: bool,
    pub api_key_fingerprint: Option<String>,
    /// Env var NAME reference — wins over the stored key when set.
    pub api_key_env_var: Option<String>,
    pub allowed_models: Vec<String>,
    pub enabled: bool,
}

impl ProfileRow {
    pub fn from_value(v: &Value) -> Option<ProfileRow> {
        let id = s(v, "id")?;
        Some(ProfileRow {
            display_name: s(v, "display_name").unwrap_or_else(|| id.clone()),
            description: s(v, "description").unwrap_or_default(),
            family: s(v, "provider_family").unwrap_or_default(),
            base_url: s(v, "base_url").unwrap_or_default(),
            api_key_set: b(v, "api_key_set").unwrap_or(false),
            api_key_fingerprint: s(v, "api_key_fingerprint").filter(|f| !f.is_empty()),
            api_key_env_var: s(v, "api_key_env_var").filter(|e| !e.is_empty()),
            allowed_models: v
                .get("allowed_models")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
                .unwrap_or_default(),
            enabled: b(v, "enabled").unwrap_or(true),
            id,
        })
    }

    pub fn virtual_provider(&self) -> String {
        format!("endpoint:{}", self.id)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProfilesData {
    pub ok: bool,
    pub writable: bool,
    pub config_file: Option<String>,
    pub profiles: Vec<ProfileRow>,
}

impl ProfilesData {
    pub fn from_value(v: &Value) -> ProfilesData {
        ProfilesData {
            ok: b(v, "ok").unwrap_or(false),
            writable: b(v, "writable").unwrap_or(false),
            config_file: s(v, "config_file"),
            profiles: v
                .get("profiles")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(ProfileRow::from_value).collect())
                .unwrap_or_default(),
        }
    }
}

/// Model ids from `config models <provider> --json` ({count, errors,
/// models} — live-probed shape).
pub fn models_from_payload(v: &Value) -> Vec<String> {
    v.get("models")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// One applied action + its verification — the review screen's rows.
#[derive(Clone, Debug)]
pub struct JournalEntry {
    pub when: String,
    pub action: String,
    pub outcome: Result<String, String>,
}

/// An in-flight operation for the busy strip.
#[derive(Clone, Debug)]
pub struct BusyOp {
    pub id: u64,
    pub label: String,
    pub started: Instant,
}

#[derive(Clone, Copy)]
pub struct Store {
    /// The file mirror (path + parsed/missing/corrupt state).
    pub cfg: Signal<Loadable<ConfigMirror>>,
    /// Derived routes view (`config defaults --json`).
    pub routes: Signal<Loadable<RoutesData>>,
    /// Derived profiles view (`config providers --json`).
    pub profiles: Signal<Loadable<ProfilesData>>,
    /// The resolved abstractcore binary (None = not found; the mirror
    /// still works and every derived surface teaches the fix).
    pub cli: Signal<Option<CliInfo>>,
    /// Per-provider model lists for the pickers.
    pub models: Signal<HashMap<String, Loadable<Vec<String>>>>,
    pub journal: Signal<Vec<JournalEntry>>,
    pub busy: Signal<Vec<BusyOp>>,
    /// Monotonic tick driving elapsed displays while ops are in flight.
    pub tick: Signal<u64>,
    /// One-line transient notice (mirrored as a toast).
    pub notice: Signal<Option<String>>,
    /// The `#FALLBACK` stderr line from the LAST exit-0 CLI run —
    /// Python announcing it refused the config file and ran on
    /// defaults (each such run also mints a fresh `.corrupt-*.bak`).
    /// While set, the mirror must not vouch for the file (P1-1).
    pub python_fallback: Signal<Option<String>>,
    /// Current test evidence, latest result per label (M3 probes) —
    /// re-tests REPLACE their prior entry (evidence is about NOW).
    pub tests: Signal<Vec<crate::probes::TestResult>>,
    /// One probe at a time: tests run real generations/discoveries;
    /// a queued duplicate would silently double the cost.
    pub probe_busy: Signal<bool>,
}

impl Store {
    pub fn create(cx: Scope) -> Store {
        Store {
            cfg: cx.signal(Loadable::default()),
            routes: cx.signal(Loadable::default()),
            profiles: cx.signal(Loadable::default()),
            cli: cx.signal(None),
            models: cx.signal(HashMap::new()),
            journal: cx.signal(Vec::new()),
            busy: cx.signal(Vec::new()),
            tick: cx.signal(0),
            notice: cx.signal(None),
            python_fallback: cx.signal(None),
            tests: cx.signal(Vec::new()),
            probe_busy: cx.signal(false),
        }
    }

    /// Forget every loaded domain (full reload). EXHAUSTIVE destructure,
    /// no `..` — adding a Store field fails compilation here, forcing
    /// the reset-or-exempt decision at the one site it matters (the
    /// sibling console's stale-domain P1 class, made structural).
    pub fn reset_domains(&self) {
        let Store {
            cli: _,     // resolution survives: same process, same env
            journal: _, // session audit
            busy: _,    // transient op bookkeeping
            tick: _,    // clock
            notice: _,  // transient toast
            tests: _,   // dated live-provider evidence, not file state
            probe_busy: _, // owned by the worker's probe lifecycle
            cfg,
            routes,
            profiles,
            models,
            python_fallback,
        } = *self;
        cfg.set(Loadable::NotAsked);
        routes.set(Loadable::NotAsked);
        profiles.set(Loadable::NotAsked);
        models.update(|m| m.clear());
        python_fallback.set(None);
    }

    /// Latest-per-label test evidence (newest first).
    pub fn record_test(&self, result: crate::probes::TestResult) {
        self.tests.update(|v| {
            v.retain(|r| r.label != result.label);
            v.insert(0, result);
        });
    }

    pub fn push_journal(&self, entry: JournalEntry) {
        self.journal.update(|j| j.push(entry));
    }

    pub fn begin_busy(&self, id: u64, label: &str) {
        let label = label.to_string();
        self.busy.update(move |ops| {
            ops.push(BusyOp {
                id,
                label,
                started: Instant::now(),
            })
        });
    }

    pub fn end_busy(&self, id: u64) {
        self.busy.update(move |ops| ops.retain(|o| o.id != id));
    }
}

/// Timestamp for journal rows and load acknowledgments (UTC HH:MM:SSZ —
/// unambiguous in a config journal).
pub fn now_hms() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Shape-faithful to `abstractcore config defaults --json` (live
    /// output probed 2026-07-25).
    #[test]
    fn routes_fold_reads_the_live_shape() {
        let v = json!({
            "ok": true, "authority": "abstractcore.local",
            "config_file": "/home/u/.abstractcore/config/abstractcore.json",
            "writable": true, "errors": [],
            "routes": [
                {"key": "input.text", "kind": "input", "modality": "text",
                 "label": "Text Input", "task": "text_understanding",
                 "provider": "lmstudio", "model": "qwen3-0.6b",
                 "base_url": "http://localhost:1234/v1",
                 "source": "abstractcore.capability_defaults", "configured": true},
                {"key": "input.image", "kind": "input", "modality": "image",
                 "label": "Image Input", "provider": "lmstudio",
                 "model": "m", "configured": true,
                 "covered_by": "input.text", "read_only": true,
                 "source": "abstractcore.capability_defaults"},
                {"key": "output.text", "kind": "output", "modality": "text",
                 "label": "Text Output", "provider": "lmstudio", "model": "m",
                 "configured": true, "source": "abstractcore.capability_defaults"},
                {"key": "input.video", "kind": "input", "modality": "video",
                 "label": "Video Input", "configured": false,
                 "source": "not_configured",
                 "package_hint": "abstractvideo or a video-capable LLM"},
                {"no_key": true}
            ]
        });
        let d = RoutesData::from_value(&v);
        assert!(d.ok && d.writable);
        assert_eq!(d.rows.len(), 4, "the keyless row is dropped");
        assert_eq!(d.configured_count(), 3);
        assert_eq!(
            d.config_file.as_deref(),
            Some("/home/u/.abstractcore/config/abstractcore.json")
        );
        let by_key = |k: &str| d.rows.iter().find(|r| r.key == k).unwrap();
        assert!(by_key("input.text").editable());
        assert!(
            !by_key("input.image").editable(),
            "covered + read_only is locked"
        );
        assert!(
            !by_key("output.text").editable(),
            "the input.text alias is never independently editable"
        );
        assert_eq!(
            by_key("input.video").package_hint.as_deref(),
            Some("abstractvideo or a video-capable LLM")
        );
    }

    /// Shape-faithful to `abstractcore config providers --json`.
    #[test]
    fn profiles_fold_reads_the_live_shape() {
        let v = json!({
            "ok": true, "writable": true, "config_file": "/x.json",
            "profiles": [
                {"id": "ovh-provider", "display_name": "OVH Provider",
                 "description": "hosted endpoint", "provider_family": "openai-compatible",
                 "base_url": "https://oai.example/v1",
                 "api_key_set": true, "api_key_fingerprint": "35982521",
                 "api_key_env_var": "", "allowed_models": [], "enabled": true,
                 "virtual_provider": "endpoint:ovh-provider"},
                {"id": "local", "provider_family": "lmstudio",
                 "base_url": "http://localhost:1234/v1",
                 "api_key_set": false, "api_key_fingerprint": null,
                 "allowed_models": ["a", "b"], "enabled": false}
            ]
        });
        let d = ProfilesData::from_value(&v);
        assert_eq!(d.profiles.len(), 2);
        let ovh = &d.profiles[0];
        assert_eq!(ovh.virtual_provider(), "endpoint:ovh-provider");
        assert!(ovh.api_key_set);
        assert_eq!(ovh.api_key_fingerprint.as_deref(), Some("35982521"));
        assert_eq!(ovh.api_key_env_var, None, "blank env var reads as none");
        let local = &d.profiles[1];
        assert!(!local.enabled);
        assert_eq!(local.allowed_models, vec!["a", "b"]);
    }
}
