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
    /// The default reasoning effort carried on the text-generation
    /// route (`capability_default_reasoning`). Editable on the text
    /// route only; every other row carries None.
    pub reasoning: Option<String>,
    /// This row MIRRORS another route and is never independently
    /// editable — `output.text` reads `input.text`
    /// (manager.py:1138-1151). Read from the payload, not from a
    /// hardcoded key, so a second derived row needs no console edit.
    pub derived_from: Option<String>,
    /// THE ROUTE HIERARCHY, off the payload — never re-derived here.
    /// `output.image` is the PARENT of `output.image.*`: the one value
    /// that answers every image task without a row of its own (the
    /// simple path, and what the fresh-install seed writes). A task row
    /// carries `broad_key` = its parent; a parent carries `task_keys`.
    /// Core derives all three in `manager._decorate_route_hierarchy`, so
    /// the four grids that render this payload cannot disagree.
    pub broad_key: Option<String>,
    pub task_keys: Vec<String>,
    /// The parent is UNSET and every task row under it is configured, so
    /// nothing can reach it. Benign — not a missing setting. Deliberately
    /// separate from `covered_by`, which forces read-only: this row stays
    /// editable, because setting it is still the simple path.
    pub covered_by_tasks: bool,
    /// This TASK row is unset while its parent IS configured — the mirror
    /// of `covered_by_tasks`, and the shape a fresh install has (the seed
    /// writes `output.image` alone). The parent answers it, so it is not
    /// unconfigured in effect and must not be painted as a gap.
    pub inherits_broad: bool,
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
            derived_from: s(v, "derived_from").filter(|d| !d.is_empty()),
            broad_key: s(v, "broad_key").filter(|k| !k.is_empty()),
            task_keys: v
                .get("task_keys")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|k| k.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            covered_by_tasks: b(v, "covered_by_tasks").unwrap_or(false),
            inherits_broad: b(v, "inherits_broad").unwrap_or(false),
            key,
        })
    }

    /// This row is a modality cell that has task rows overriding it.
    pub fn is_task_parent(&self) -> bool {
        !self.task_keys.is_empty()
    }

    /// The ROUTE column text: a task row shows only its task segment,
    /// under a tree marker, so the grid reads as the hierarchy it is.
    /// The full key stays available on the detail line and in every
    /// write path — and dropping the repeated `output.image.` prefix
    /// gives the column back ~13 cells rather than eating them.
    pub fn display_key(&self) -> String {
        match self.broad_key.as_deref() {
            Some(parent) if self.key.len() > parent.len() + 1 => {
                format!("  └ {}", &self.key[parent.len() + 1..])
            }
            _ => self.key.clone(),
        }
    }

    /// Can this row be edited at all? Derived rows (`output.text`) are
    /// read-only; covered rows are editable only when overrideable.
    /// THE SAME BODY as the gateway console's `RouteRow::editable` —
    /// one editability law across both entry points.
    pub fn editable(&self) -> bool {
        if self.derived_from.is_some() {
            return false;
        }
        if self.covered_by.is_some() {
            return self.overrideable && !self.read_only;
        }
        !self.read_only
    }

    /// THE shared state vocabulary, one string per row state — the
    /// same four words the gateway console's routes table prints, so
    /// an operator reads one grid whichever entry point they opened.
    pub fn state_label(&self) -> String {
        if let Some(from) = self.derived_from.as_deref() {
            return format!("derived ← {from}");
        }
        if let Some(by) = self.covered_by.as_deref() {
            return format!("covered by {by}");
        }
        if self.configured {
            return "configured".to_string();
        }
        // AN UNSET PARENT WHOSE TASK ROWS ARE ALL SET IS NOT A PROBLEM.
        // Core proves it (`capability_route_tasks_cover_broad`): the task
        // rows are exactly the keys the output route table can produce
        // for that modality, so nothing can reach the parent. Printing
        // "not configured" there sent an operator hunting for dead code
        // ("why do we have output.image AND t2i/i2i/upscale?").
        if self.covered_by_tasks {
            return "not needed".to_string();
        }
        // ...and the MIRROR: a task row with no value of its own whose
        // parent IS set is answered by that parent. A fresh install is
        // exactly this shape (the seed writes `output.image` alone), so
        // three red "not configured" rows used to sit under a working
        // parent and read as "image editing is not set up".
        if self.inherits_broad {
            return "inherited".to_string();
        }
        "not configured".to_string()
    }

    /// The reasoning effort is a property of TEXT GENERATION:
    /// `output.text` is the canonical route and `input.text` is where
    /// the store keeps it, so the control belongs on both cells and
    /// nowhere else (console.py `isTextGenerationDefault`).
    pub fn is_text_generation(&self) -> bool {
        self.key == "output.text" || self.key == "input.text"
    }

    /// "provider / model" as the store currently answers it, or the
    /// honest absence.
    pub fn pair_text(&self) -> String {
        match (self.provider.as_deref(), self.model.as_deref()) {
            (None, None) => "—".to_string(),
            (p, m) => format!("{} / {}", p.unwrap_or("—"), m.unwrap_or("—")),
        }
    }
}

/// The shared reasoning vocabulary: the effort levels the web console
/// offers, index 0 = "not set" (the placeholder that clears). Both
/// consoles offer exactly this list, in this order.
///
/// NOT the engine's `ReasoningSelect` (abstracttui 0.3.0), deliberately.
/// That control is the FOOTER/per-request picker: its ladder adds
/// `none`/`xhigh`/`auto`, and it is capability-driven — without a
/// `ReasoningFacts` block it renders LOCKED behind a "set anyway" gate.
/// Neither console has per-model capability facts for a route, and a
/// locked control on the screen whose entire job is setting this
/// default would be a dead one. The reference surface for a stored
/// DEFAULT is the web console's select, and this is that list verbatim.
pub const REASONING_LEVELS: [&str; 4] = ["minimal", "low", "medium", "high"];

/// Select index for a stored reasoning value — 0 ("not set") when the
/// row carries none, or one the list does not offer (never fabricate a
/// selection the store does not hold).
pub fn reasoning_index(stored: Option<&str>) -> usize {
    stored
        .map(str::trim)
        .filter(|r| !r.is_empty())
        .and_then(|r| {
            REASONING_LEVELS
                .iter()
                .position(|l| l.eq_ignore_ascii_case(r))
        })
        .map(|i| i + 1)
        .unwrap_or(0)
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

// ---------------------------------------------------------------------
// WEIGHTS. A route can be perfectly configured and still unrunnable
// because the model is not on this machine. `abstractcore models status
// --json` answers that, in the SAME four words the gateway console and
// the gateway TUI print — installed / absent / unknown / not_applicable
// — and `unknown` is a real answer that must never be shown as either
// of its neighbours.
// ---------------------------------------------------------------------

/// One route's weight availability, plus the artifact that would fetch
/// it. The ARTIFACT is not the route's model: a served id drops the
/// quantization suffix, so `input.text` stores `qwen/qwen3.5-9b` while
/// the download reference is `qwen/qwen3.5-9b@4bit`.
#[derive(Clone, Debug, Default)]
pub struct WeightsRow {
    pub status: String,
    pub provider: String,
    pub artifact: String,
    pub detail: String,
    pub instruction: String,
    pub downloadable: bool,
}

impl WeightsRow {
    /// The short word the `weights` column prints. Empty for a route
    /// that names no model — there is nothing to be missing.
    pub fn label(&self) -> &str {
        match self.status.as_str() {
            "installed" => "installed",
            "absent" => "not downloaded",
            "not_applicable" => "remote",
            "unknown" => "unknown",
            _ => "",
        }
    }
}

/// `abstractcore models status --json`, folded for the routes screen.
#[derive(Clone, Debug, Default)]
pub struct AvailabilityData {
    pub by_route: HashMap<String, WeightsRow>,
    pub total: usize,
    pub installed: usize,
    pub absent: usize,
    pub unknown: usize,
    /// `(provider, artifact)` of every recommended model NOT present.
    pub missing: Vec<(String, String)>,
}

impl AvailabilityData {
    pub fn from_value(v: &Value) -> AvailabilityData {
        let mut by_route = HashMap::new();
        for row in v.get("routes").and_then(Value::as_array).into_iter().flatten() {
            let Some(key) = s(row, "key").filter(|k| !k.is_empty()) else {
                continue;
            };
            let a = row.get("availability").cloned().unwrap_or(Value::Null);
            let status = s(&a, "status").unwrap_or_default();
            // An unconfigured route reports `unknown` with this evidence;
            // it is NOT a missing download and must not be dressed as one.
            if s(&a, "evidence").as_deref() == Some("route not configured") {
                continue;
            }
            by_route.insert(
                key,
                WeightsRow {
                    status,
                    provider: s(row, "provider").unwrap_or_default(),
                    artifact: s(row, "download_artifact")
                        .or_else(|| s(&a, "artifact"))
                        .or_else(|| s(row, "model"))
                        .unwrap_or_default(),
                    detail: s(&a, "detail").unwrap_or_default(),
                    instruction: s(&a, "instruction").unwrap_or_default(),
                    downloadable: b(&a, "downloadable").unwrap_or(false),
                },
            );
        }
        let plan = v.get("recommended").cloned().unwrap_or(Value::Null);
        let n = |key: &str| plan.get(key).and_then(Value::as_u64).unwrap_or(0) as usize;
        let missing = plan
            .get("would_download")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|item| {
                        Some((s(item, "provider")?, s(item, "artifact")?))
                    })
                    .collect()
            })
            .unwrap_or_default();
        AvailabilityData {
            by_route,
            total: n("total"),
            installed: n("installed"),
            absent: n("absent"),
            unknown: n("unknown"),
            missing,
        }
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

// ---------------------------------------------------------------------
// THE UNIFIED PROVIDER LIST (operator ruling 2026-08-01: "I do not
// understand why the providers are displayed in a different fashion
// between gateway and core; they should have the exact same. Gateway is
// the one we want. Profiles are just indicated as profile of the
// openai-compatible endpoint, and we should have a way to configure as
// many as necessary, like in the gateway console.")
//
// ONE table with the gateway console-TUI's exact columns —
// `provider | family | base URL | API key | models | enabled | origin` —
// composed HERE, so the screen renders rows instead of performing a
// join. The sources are core's own, all already in this one payload:
// the provider INVENTORY (`config providers --probe --json`: every
// provider the registry knows, one `endpoint:<id>` row per stored
// profile, plus the local-server probe results) joined with the
// `provider_profiles` section rows, which are what give an endpoint row
// its family, its allowlist and its enabled flag.
// ---------------------------------------------------------------------

/// The `origin` column, in core's vocabulary. The gateway prints
/// `gateway`/`user`/`env`/`core`/`auto`; core has one store and no
/// scopes, so the same question ("where does this row come from?") has
/// four answers here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// Stored in THIS config file — an endpoint profile in
    /// `provider_profiles`, or a key in `api_keys`.
    Config,
    /// Resolved from the environment (`$OPENAI_API_KEY`, `$OLLAMA_HOST`).
    Env,
    /// Nothing configured, but a local server ANSWERED at its
    /// documented default address — the gateway's `auto` row, produced
    /// by the same kind of probe.
    Auto,
    /// Known to the registry, nothing configured, nothing answering.
    /// The gateway has no such row (it lists these names only in its
    /// "not configured yet" footer); core keeps them because a keyless
    /// local engine has nowhere else to appear — ruling 2026-08-01,
    /// "how come we don't have ollama, lmstudio, huggingface and mlx?".
    Registry,
}

impl Origin {
    pub fn label(self) -> &'static str {
        match self {
            Origin::Config => "config",
            Origin::Env => "env",
            Origin::Auto => "auto",
            Origin::Registry => "registry",
        }
    }
}

/// One row of the unified providers table — cells pre-rendered in the
/// gateway console's vocabulary, plus the objects the row's verbs act
/// on (`profile` for `e`/`d`, `api_key_field` for `k`).
#[derive(Clone, Debug)]
pub struct ConnectionRow {
    /// Column 1 — the provider NAME a route, a flow or `--provider`
    /// must spell: `endpoint:<id>` for a stored profile, the bare
    /// registry id for a builtin.
    pub provider: String,
    pub family: String,
    /// The configured address, or empty (the cell prints `—`).
    pub base_url: String,
    pub api_key: String,
    pub models: String,
    /// `None` — a registry provider has no enable switch in core, so
    /// the cell prints `—` instead of fabricating a toggle that has
    /// nothing to write to.
    pub enabled: Option<bool>,
    pub origin: Origin,
    /// The env var behind an `env` origin (key var or base-URL var).
    pub origin_env: String,
    /// The stored profile behind an `endpoint:<id>` row — what `e`
    /// edits and `d` deletes. `None` on builtin rows.
    pub profile: Option<ProfileRow>,
    /// The `api_keys` field `k` edits (empty = this row takes no key).
    pub api_key_field: String,
    /// Evidence that earns no column but answers "why is this row
    /// odd": the probe's own words, the registry note.
    pub detail: String,
}

impl ConnectionRow {
    fn compose(inv: &ProviderRow, profile: Option<&ProfileRow>) -> ConnectionRow {
        let origin = if profile.is_some() {
            Origin::Config
        } else if inv.api_key_source.starts_with("env:") || inv.base_url_source.starts_with("env:")
        {
            Origin::Env
        } else if inv.api_key_source == "config" {
            Origin::Config
        } else if inv.reachable == Some(true) {
            Origin::Auto
        } else {
            Origin::Registry
        };
        let origin_env = match origin {
            Origin::Env => inv
                .api_key_source
                .strip_prefix("env:")
                .or_else(|| inv.base_url_source.strip_prefix("env:"))
                .unwrap_or_default()
                .to_string(),
            _ => String::new(),
        };
        let mut detail = Vec::new();
        if inv.reachable == Some(false) && !inv.reachability.is_empty() {
            detail.push(inv.reachability.clone());
        }
        if !inv.note.is_empty() {
            detail.push(inv.note.clone());
        }
        ConnectionRow {
            provider: inv.provider.clone(),
            // A builtin provider IS its own family — the gateway says
            // exactly this for its builtin rows
            // (`provider_family = spec.provider_id`,
            // provider_connections.py:189).
            family: match profile {
                Some(p) if !p.family.is_empty() => p.family.clone(),
                Some(_) => "—".into(),
                None => inv.provider.clone(),
            },
            base_url: inv.base_url.clone(),
            api_key: key_cell(
                inv.api_key_set,
                &inv.api_key_source,
                &inv.api_key_env_var,
                &inv.api_key_fingerprint,
            ),
            models: models_cell(
                profile.map(|p| p.allowed_models.len()).unwrap_or(0),
                &inv.reachability,
            ),
            enabled: profile.map(|p| p.enabled),
            origin,
            origin_env,
            profile: profile.cloned(),
            api_key_field: inv.api_key_field.clone(),
            detail: detail.join(" · "),
        }
    }

    /// A stored profile with NO inventory row. The CLI builds the
    /// inventory from the same store, so this is a can't-happen — but a
    /// profile the operator just created must never be invisible, and
    /// an inventory that lost a row is exactly when it would be.
    fn from_profile(p: &ProfileRow) -> ConnectionRow {
        ConnectionRow {
            provider: p.virtual_provider(),
            family: if p.family.is_empty() {
                "—".into()
            } else {
                p.family.clone()
            },
            base_url: p.base_url.clone(),
            api_key: key_cell(
                p.api_key_set,
                &match &p.api_key_env_var {
                    Some(v) => format!("env:{v}"),
                    None => "profile".to_string(),
                },
                p.api_key_env_var.as_deref().unwrap_or_default(),
                p.api_key_fingerprint.as_deref().unwrap_or_default(),
            ),
            models: models_cell(p.allowed_models.len(), ""),
            enabled: Some(p.enabled),
            origin: Origin::Config,
            origin_env: String::new(),
            profile: Some(p.clone()),
            api_key_field: String::new(),
            detail: String::new(),
        }
    }

    /// This row IS a stored endpoint profile (`e` edits, `d` deletes).
    pub fn is_profile(&self) -> bool {
        self.profile.is_some()
    }

    /// This row has an `api_keys` field to edit (`k`).
    pub fn takes_key(&self) -> bool {
        !self.api_key_field.is_empty()
    }

    pub fn base_url_text(&self) -> String {
        if self.base_url.is_empty() {
            "—".into()
        } else {
            self.base_url.clone()
        }
    }

    /// `yes` / `NO` — the gateway's two words — or `—` for a registry
    /// row, which has no such flag anywhere in core.
    pub fn enabled_text(&self) -> String {
        match self.enabled {
            Some(true) => "yes".into(),
            Some(false) => "NO".into(),
            None => "—".into(),
        }
    }

    /// Where this row comes from, in operator words — the gateway's
    /// `synthetic_origin`, said in core's terms, with the probe's own
    /// evidence appended when there is any.
    pub fn origin_detail(&self) -> String {
        let mut parts = vec![match self.origin {
            Origin::Config if self.is_profile() => {
                "endpoint profile stored in this config".to_string()
            }
            Origin::Config if self.takes_key() => {
                format!("key stored in this config (api_keys.{})", self.api_key_field)
            }
            Origin::Config => "stored in this config".to_string(),
            Origin::Env => format!("resolved from the environment (${})", self.origin_env),
            Origin::Auto if self.base_url.is_empty() => "answering locally".to_string(),
            Origin::Auto => format!("a local server answering at {}", self.base_url),
            Origin::Registry => "known to the registry — nothing configured yet".to_string(),
        }];
        if !self.detail.is_empty() {
            parts.push(self.detail.clone());
        }
        parts.join(" · ")
    }
}

/// THE SHARED API-KEY VOCABULARY: `stored (…)` / `none (…)`, word for
/// word the gateway console's connection table — carrying the `$VAR`
/// attribution core knows and the gateway does not. `api_key_set` is
/// the RESOLVED answer (env value if that is what wins, else the stored
/// key), so a `$VAR` reference that resolves to nothing reads as `none`.
fn key_cell(set: bool, source: &str, env_var: &str, fingerprint: &str) -> String {
    let from_env = source.strip_prefix("env:").filter(|v| !v.is_empty());
    if set {
        if let Some(var) = from_env {
            return format!("stored (${var})");
        }
        return match fingerprint {
            "" => "stored".to_string(),
            fp => format!("stored ({fp})"),
        };
    }
    // `none` already says the key is absent; naming the var says WHERE
    // it would have come from.
    if env_var.is_empty() {
        "none".to_string()
    } else {
        format!("none (${env_var})")
    }
}

/// The `models` cell, gateway-for-gateway: an allowlist RESTRICTS the
/// row, a probe that counted models says how many are live, and
/// everything else serves live discovery.
fn models_cell(allowed: usize, reachability: &str) -> String {
    if allowed > 0 {
        return format!("{allowed} restr");
    }
    match models_reported(reachability) {
        Some(n) => format!("{n} live"),
        None => "live".to_string(),
    }
}

/// `reachable (43 models)` → 43. The probe's own sentence
/// (`model_materializer._probe_local_server`) is the only place this
/// payload carries a live model count.
pub fn models_reported(reachability: &str) -> Option<u64> {
    let open = reachability.find('(')? + 1;
    let rest = reachability.get(open..)?;
    let end = rest.find(" models)")?;
    rest.get(..end)?.trim().parse().ok()
}

/// One row of `config providers --json`'s `providers` array: EVERY
/// provider AbstractCore knows, not just the ones that take a key.
///
/// The screen used to enumerate the `api_keys` config section, so
/// ollama / lmstudio / mlx / huggingface and every media engine —
/// which need no key — had no row at all ("how come we don't have
/// ollama, lmstudio, huggingface and mlx?"). `api_keys` is a KEY
/// STORE; the provider list is AbstractCore's registry, and this is it.
#[derive(Clone, Debug, Default)]
pub struct ProviderRow {
    pub provider: String,
    /// `cloud_api` | `local_server` | `local_engine` | `endpoint_profile`.
    pub kind: String,
    /// `required` | `optional` | `none`.
    pub auth: String,
    /// The `api_keys` section field this row edits — empty when the
    /// provider takes no key, which is exactly when `k` must refuse.
    pub api_key_field: String,
    pub api_key_env_var: String,
    pub api_key_set: bool,
    pub api_key_source: String,
    /// Non-reversible 8 chars — the presence proof every surface shows
    /// in place of the key itself.
    pub api_key_fingerprint: String,
    pub base_url: String,
    pub base_url_source: String,
    /// `None` = not probed (or nothing to probe).
    pub reachable: Option<bool>,
    pub reachability: String,
    pub note: String,
}

impl ProviderRow {
    pub fn from_value(v: &Value) -> Option<ProviderRow> {
        let provider = s(v, "provider").filter(|p| !p.is_empty())?;
        Some(ProviderRow {
            provider,
            kind: s(v, "kind").unwrap_or_default(),
            auth: s(v, "auth").unwrap_or_default(),
            api_key_field: s(v, "api_key_field").unwrap_or_default(),
            api_key_env_var: s(v, "api_key_env_var").unwrap_or_default(),
            api_key_set: b(v, "api_key_set").unwrap_or(false),
            api_key_source: s(v, "api_key_source").unwrap_or_default(),
            api_key_fingerprint: s(v, "api_key_fingerprint").unwrap_or_default(),
            base_url: s(v, "base_url").unwrap_or_default(),
            base_url_source: s(v, "base_url_source").unwrap_or_default(),
            reachable: v.get("reachable").and_then(Value::as_bool),
            reachability: s(v, "reachability").unwrap_or_default(),
            note: s(v, "note").unwrap_or_default(),
        })
    }

    /// `k` edits a key only where a key exists to edit.
    pub fn editable_key(&self) -> bool {
        !self.api_key_field.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProfilesData {
    pub ok: bool,
    pub writable: bool,
    pub config_file: Option<String>,
    pub providers: Vec<ProviderRow>,
    pub probed: bool,
    pub profiles: Vec<ProfileRow>,
}

impl ProfilesData {
    pub fn from_value(v: &Value) -> ProfilesData {
        ProfilesData {
            ok: b(v, "ok").unwrap_or(false),
            writable: b(v, "writable").unwrap_or(false),
            config_file: s(v, "config_file"),
            providers: v
                .get("providers")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(ProviderRow::from_value).collect())
                .unwrap_or_default(),
            probed: b(v, "probed").unwrap_or(false),
            profiles: v
                .get("profiles")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(ProfileRow::from_value).collect())
                .unwrap_or_default(),
        }
    }

    /// THE ONE LIST the Providers screen renders (gateway parity).
    ///
    /// Order is the CLI's own — registry providers, then the
    /// `endpoint:<id>` rows, then the media engines — never re-sorted
    /// here: that order is core's authority order, and a console that
    /// re-ranks it would answer "where is my provider?" differently
    /// from the CLI that printed the same payload.
    pub fn connections(&self) -> Vec<ConnectionRow> {
        let mut rows: Vec<ConnectionRow> = self
            .providers
            .iter()
            .map(|inv| {
                let profile = inv
                    .provider
                    .strip_prefix("endpoint:")
                    .and_then(|id| self.profiles.iter().find(|p| p.id == id));
                ConnectionRow::compose(inv, profile)
            })
            .collect();
        // A stored profile the inventory did not carry still gets a row:
        // an operator-created connection that is invisible is the one
        // failure this screen must not have.
        for p in &self.profiles {
            let name = p.virtual_provider();
            if !rows.iter().any(|r| r.provider == name) {
                rows.push(ConnectionRow::from_profile(p));
            }
        }
        rows
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
    /// Local weight availability (`models status --json`) — whether the
    /// configured models are actually ON this machine.
    pub availability: Signal<Loadable<AvailabilityData>>,
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
            availability: cx.signal(Loadable::default()),
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
            availability,
            models,
            python_fallback,
        } = *self;
        cfg.set(Loadable::NotAsked);
        routes.set(Loadable::NotAsked);
        profiles.set(Loadable::NotAsked);
        // Weights are machine state, not file state — but a reload is
        // exactly when an operator expects a just-finished download to
        // show up, so this domain resets with the rest.
        availability.set(Loadable::NotAsked);
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

    /// THE ROUTE HIERARCHY, AT THE ROW-MODEL LEVEL (operator question
    /// 2026-08-01: "why do we have output.image AND t2i/i2i/upscale —
    /// are those remnants?"). They are not: `output.image` is the
    /// PARENT, the one value that serves every image task without a row
    /// of its own. The grid has to SAY that, so:
    ///   - a task row indents under its parent and drops the repeated
    ///     prefix (`  └ text_to_image`), and the full key survives on
    ///     the detail line;
    ///   - a parent that is UNSET while every task row under it is set
    ///     reads `not needed`, never the alarming `not configured`;
    ///   - a parent with an unset task row under it still reads `not
    ///     configured`, because it IS the thing that would answer it;
    ///   - the parent stays EDITABLE either way — setting it is the
    ///     simple path, and what the fresh-install seed writes.
    #[test]
    fn route_rows_render_the_broad_task_hierarchy() {
        let parent = |covered: bool| {
            RouteRow::from_value(&json!({
                "key": "output.image", "kind": "output", "modality": "image",
                "label": "Image Output", "configured": false,
                "source": "not_configured",
                "task_keys": ["output.image.text_to_image", "output.image.image_to_image",
                              "output.image.image_upscale"],
                "covered_by_tasks": covered
            }))
            .unwrap()
        };
        let child = RouteRow::from_value(&json!({
            "key": "output.image.text_to_image", "kind": "output", "modality": "image",
            "label": "Image Generation", "configured": true, "provider": "mlx-gen",
            "model": "AbstractFramework/flux.2-klein-9b-8bit",
            "source": "abstractcore.capability_defaults",
            "broad_key": "output.image"
        }))
        .unwrap();

        let covered = parent(true);
        assert!(covered.is_task_parent());
        assert_eq!(covered.task_keys.len(), 3);
        assert_eq!(covered.display_key(), "output.image", "the parent is not indented");
        assert_eq!(
            covered.state_label(),
            "not needed",
            "an unset parent every task row already covers is benign, not a red flag"
        );
        assert!(
            covered.editable(),
            "the parent stays settable — one image model for every task is the simple path"
        );

        assert_eq!(
            parent(false).state_label(),
            "not configured",
            "a parent with an uncovered task under it IS the missing setting"
        );

        assert!(!child.is_task_parent());
        assert_eq!(child.broad_key.as_deref(), Some("output.image"));
        assert_eq!(
            child.display_key(),
            "  └ text_to_image",
            "task rows indent under the parent and drop its repeated prefix"
        );
        assert_eq!(child.state_label(), "configured");
    }

    /// THE MIRROR, and the shape a FRESH INSTALL has: the seed writes
    /// `output.image` alone, so the three task rows have no value of
    /// their own while the parent answers every one of them. Painting
    /// them "not configured" says "image editing is not set up" about a
    /// machine where it demonstrably is.
    #[test]
    fn task_rows_answered_by_a_configured_parent_read_as_inherited() {
        let inherited = RouteRow::from_value(&json!({
            "key": "output.image.image_upscale", "kind": "output", "modality": "image",
            "label": "Image Restore / Upscale", "configured": false,
            "source": "not_configured", "broad_key": "output.image",
            "inherits_broad": true
        }))
        .unwrap();
        assert_eq!(inherited.state_label(), "inherited");
        assert_eq!(inherited.display_key(), "  └ image_upscale");
        assert!(inherited.editable(), "an inherited row is still settable");

        // Without a configured parent the row is honestly unconfigured.
        let orphan = RouteRow::from_value(&json!({
            "key": "output.video.text_to_video", "kind": "output", "modality": "video",
            "label": "Video Generation", "configured": false,
            "source": "not_configured", "broad_key": "output.video"
        }))
        .unwrap();
        assert_eq!(orphan.state_label(), "not configured");
    }

    /// A modality with NO task rows (voice/sound/music, every input
    /// route) is the PRIMARY key, not a fallback — it must carry no
    /// hierarchy decoration at all and keep its plain key and state.
    /// This is why the broad row shape can never be deleted.
    #[test]
    fn broad_only_modalities_carry_no_hierarchy() {
        let voice = RouteRow::from_value(&json!({
            "key": "output.voice", "kind": "output", "modality": "voice",
            "label": "Voice Output", "configured": true, "provider": "supertonic",
            "model": "supertonic-3", "source": "abstractcore.capability_defaults"
        }))
        .unwrap();
        assert!(!voice.is_task_parent());
        assert!(voice.broad_key.is_none());
        assert!(!voice.covered_by_tasks);
        assert_eq!(voice.display_key(), "output.voice");
        assert_eq!(voice.state_label(), "configured");
    }

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
                 "reasoning": "high", "configured": true, "read_only": true,
                 "derived_from": "input.text",
                 "source": "abstractcore.capability_defaults"},
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

        // THE SHARED VOCABULARY — the four strings both consoles print.
        // Derived-ness is read from the PAYLOAD's `derived_from`, never
        // from a hardcoded key list, so a second derived row would need
        // no console edit.
        assert_eq!(by_key("input.text").state_label(), "configured");
        assert_eq!(by_key("input.image").state_label(), "covered by input.text");
        assert_eq!(by_key("output.text").state_label(), "derived ← input.text");
        assert_eq!(by_key("input.video").state_label(), "not configured");
        assert_eq!(
            by_key("output.text").derived_from.as_deref(),
            Some("input.text")
        );

        // Reasoning rides on the text-generation cells and nowhere else.
        assert_eq!(by_key("output.text").reasoning.as_deref(), Some("high"));
        assert!(by_key("output.text").is_text_generation());
        assert!(by_key("input.text").is_text_generation());
        assert!(!by_key("input.image").is_text_generation());
        assert_eq!(reasoning_index(Some("HIGH")), 4, "case-insensitive");
        assert_eq!(reasoning_index(Some("  ")), 0, "blank = not set");
        assert_eq!(
            reasoning_index(Some("ludicrous")),
            0,
            "a level the list does not offer never fabricates a selection"
        );
        assert_eq!(reasoning_index(None), 0);

        assert_eq!(by_key("input.text").pair_text(), "lmstudio / qwen3-0.6b");
        assert_eq!(by_key("input.video").pair_text(), "—");
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

    /// The live shape of `config providers --probe --json`: the
    /// inventory already carries one `endpoint:<id>` row per stored
    /// profile, so the unified list is a JOIN, not a concatenation.
    fn unified_fixture() -> ProfilesData {
        ProfilesData::from_value(&json!({
            "ok": true, "writable": true, "probed": true,
            "providers": [
                // A cloud API whose key lives in this config file.
                {"provider": "openai", "kind": "cloud_api", "auth": "required",
                 "api_key_field": "openai", "api_key_env_var": "OPENAI_API_KEY",
                 "api_key_set": true, "api_key_source": "config",
                 "api_key_fingerprint": "9f1c33aa", "base_url": "", "base_url_source": "",
                 "reachable": null, "reachability": "", "note": ""},
                // A cloud API whose key comes from the ENVIRONMENT.
                {"provider": "anthropic", "kind": "cloud_api", "auth": "required",
                 "api_key_field": "anthropic", "api_key_env_var": "ANTHROPIC_API_KEY",
                 "api_key_set": true, "api_key_source": "env:ANTHROPIC_API_KEY",
                 "api_key_fingerprint": "aa11bb22", "base_url": "", "base_url_source": "",
                 "reachable": null, "reachability": "", "note": ""},
                // A local server ANSWERING at its documented default.
                {"provider": "lmstudio", "kind": "local_server", "auth": "none",
                 "api_key_field": "", "api_key_env_var": "", "api_key_set": false,
                 "api_key_source": "", "api_key_fingerprint": "",
                 "base_url": "http://localhost:1234/v1", "base_url_source": "default",
                 "reachable": true, "reachability": "reachable (43 models)", "note": ""},
                // A local server at its default with nothing behind it.
                {"provider": "ollama", "kind": "local_server", "auth": "none",
                 "api_key_field": "", "api_key_env_var": "", "api_key_set": false,
                 "api_key_source": "", "api_key_fingerprint": "",
                 "base_url": "http://localhost:11434", "base_url_source": "default",
                 "reachable": false, "reachability": "GET .../api/tags unreachable",
                 "note": ""},
                // A keyless local engine — no server, nothing to set.
                {"provider": "mlx", "kind": "local_engine", "auth": "none",
                 "api_key_field": "", "api_key_env_var": "", "api_key_set": false,
                 "api_key_source": "", "api_key_fingerprint": "",
                 "base_url": "", "base_url_source": "",
                 "reachable": null, "reachability": "",
                 "note": "Apple Silicon text/vision inference"},
                // The stored profiles, INLINE — the inventory's own rows.
                {"provider": "endpoint:ovh-provider", "kind": "endpoint_profile",
                 "auth": "optional", "api_key_field": "", "api_key_env_var": "",
                 "api_key_set": true, "api_key_source": "profile",
                 "api_key_fingerprint": "35982521",
                 "base_url": "https://oai.example.net/v1",
                 "base_url_source": "endpoint profile",
                 "reachable": true, "reachability": "reachable (22 models)",
                 "note": "endpoint profile (openai-compatible)"},
                {"provider": "endpoint:team-proxy", "kind": "endpoint_profile",
                 "auth": "optional", "api_key_field": "", "api_key_env_var": "TEAM_KEY",
                 "api_key_set": false, "api_key_source": "",
                 "api_key_fingerprint": "",
                 "base_url": "https://proxy.example/v1",
                 "base_url_source": "endpoint profile",
                 "reachable": null, "reachability": "", "note": ""}
            ],
            "profiles": [
                {"id": "ovh-provider", "display_name": "OVH Provider",
                 "provider_family": "openai-compatible",
                 "base_url": "https://oai.example.net/v1",
                 "api_key_set": true, "api_key_fingerprint": "35982521",
                 "api_key_env_var": "", "allowed_models": [], "enabled": true},
                {"id": "team-proxy", "display_name": "Team proxy",
                 "provider_family": "openai", "base_url": "https://proxy.example/v1",
                 "api_key_set": false, "api_key_fingerprint": null,
                 "api_key_env_var": "TEAM_KEY", "allowed_models": ["gpt-x"],
                 "enabled": false}
            ]
        }))
    }

    /// THE UNIFIED LIST (operator ruling 2026-08-01: "they should have
    /// the exact same [display]. Gateway is the one we want"): ONE row
    /// per provider, endpoint profiles INLINE as `endpoint:<id>` rows
    /// carrying their family — never a second table underneath.
    #[test]
    fn connections_compose_one_row_per_provider_in_payload_order() {
        let rows = unified_fixture().connections();
        assert_eq!(
            rows.iter().map(|r| r.provider.as_str()).collect::<Vec<_>>(),
            vec![
                "openai",
                "anthropic",
                "lmstudio",
                "ollama",
                "mlx",
                "endpoint:ovh-provider",
                "endpoint:team-proxy"
            ],
            "payload order is preserved and each profile appears ONCE, inline"
        );
        let row = |name: &str| {
            rows.iter()
                .find(|r| r.provider == name)
                .unwrap_or_else(|| panic!("{name} row"))
                .clone()
        };

        // FAMILY: a profile says which endpoint family it is a profile
        // OF ("profiles are just indicated as profile of the
        // openai-compatible endpoint"); a builtin IS its own family,
        // exactly as the gateway reports it for its builtin rows.
        assert_eq!(row("endpoint:ovh-provider").family, "openai-compatible");
        assert_eq!(row("endpoint:team-proxy").family, "openai");
        assert_eq!(row("lmstudio").family, "lmstudio");

        // ENABLED: profiles carry the flag; a registry provider has no
        // enable switch anywhere in core, so the cell is honestly blank
        // rather than a toggle with nothing behind it.
        assert_eq!(row("endpoint:ovh-provider").enabled_text(), "yes");
        assert_eq!(row("endpoint:team-proxy").enabled_text(), "NO");
        assert_eq!(row("lmstudio").enabled_text(), "—");
        assert_eq!(row("lmstudio").enabled, None);

        // MODELS: an allowlist restricts; a probe that counted says how
        // many are live; everything else serves live discovery.
        assert_eq!(row("endpoint:team-proxy").models, "1 restr");
        assert_eq!(row("lmstudio").models, "43 live");
        assert_eq!(row("endpoint:ovh-provider").models, "22 live");
        assert_eq!(row("openai").models, "live");

        // BASE URL: the honest em-dash, never a blank cell.
        assert_eq!(row("openai").base_url_text(), "—");
        assert_eq!(row("ollama").base_url_text(), "http://localhost:11434");

        // The verbs' targets ride the row.
        assert!(row("endpoint:ovh-provider").is_profile());
        assert!(!row("endpoint:ovh-provider").takes_key());
        assert!(!row("lmstudio").is_profile());
        assert!(!row("lmstudio").takes_key());
        assert!(row("openai").takes_key());
        assert_eq!(row("openai").api_key_field, "openai");
    }

    /// The `origin` column: one word for where a row lives. Core has
    /// one store and no scopes, so the gateway's five words collapse to
    /// four — and `registry` (known, nothing configured) is the one the
    /// gateway has no row for at all.
    #[test]
    fn origin_labels_say_where_each_row_comes_from() {
        let rows = unified_fixture().connections();
        let origin = |name: &str| {
            rows.iter()
                .find(|r| r.provider == name)
                .map(|r| r.origin)
                .unwrap_or_else(|| panic!("{name} row"))
        };
        assert_eq!(origin("openai"), Origin::Config, "key stored in this file");
        assert_eq!(origin("anthropic"), Origin::Env, "key from the environment");
        assert_eq!(
            origin("lmstudio"),
            Origin::Auto,
            "a local server that ANSWERED at its default address"
        );
        assert_eq!(
            origin("ollama"),
            Origin::Registry,
            "a default address with nothing behind it is not configured"
        );
        assert_eq!(origin("mlx"), Origin::Registry);
        assert_eq!(
            origin("endpoint:ovh-provider"),
            Origin::Config,
            "an endpoint profile lives in provider_profiles — always this file"
        );
        assert_eq!(origin("endpoint:team-proxy"), Origin::Config);
        assert_eq!(
            ["config", "env", "auto", "registry"],
            [
                Origin::Config.label(),
                Origin::Env.label(),
                Origin::Auto.label(),
                Origin::Registry.label()
            ]
        );

        // The selected-row phrase behind each label — and the probe's
        // own words survive on the row that failed.
        let detail = |name: &str| {
            rows.iter()
                .find(|r| r.provider == name)
                .map(ConnectionRow::origin_detail)
                .unwrap_or_default()
        };
        assert_eq!(
            detail("endpoint:ovh-provider"),
            "endpoint profile stored in this config · endpoint profile (openai-compatible)"
        );
        assert_eq!(
            detail("openai"),
            "key stored in this config (api_keys.openai)"
        );
        assert_eq!(
            detail("anthropic"),
            "resolved from the environment ($ANTHROPIC_API_KEY)"
        );
        assert_eq!(
            detail("lmstudio"),
            "a local server answering at http://localhost:1234/v1"
        );
        assert!(
            detail("ollama").starts_with("known to the registry — nothing configured yet · GET"),
            "an unreachable row keeps the probe's own words: {}",
            detail("ollama")
        );
    }

    /// THE SHARED API-KEY VOCABULARY, one function for both lanes:
    /// `stored (fp)` / `stored ($VAR)` / `none ($VAR)` / `none` — and a
    /// `$VAR` reference that resolves to NOTHING never reads as
    /// configured.
    #[test]
    fn api_key_cells_speak_one_vocabulary() {
        let rows = unified_fixture().connections();
        let key = |name: &str| {
            rows.iter()
                .find(|r| r.provider == name)
                .map(|r| r.api_key.clone())
                .unwrap_or_else(|| panic!("{name} row"))
        };
        assert_eq!(key("openai"), "stored (9f1c33aa)");
        assert_eq!(key("anthropic"), "stored ($ANTHROPIC_API_KEY)");
        assert_eq!(key("endpoint:ovh-provider"), "stored (35982521)");
        assert_eq!(key("endpoint:team-proxy"), "none ($TEAM_KEY)");
        assert_eq!(key("lmstudio"), "none");
        // No fingerprint on a set key is still "stored" — presence is
        // the claim, the fingerprint is only its proof.
        assert_eq!(key_cell(true, "profile", "", ""), "stored");
    }

    /// A stored profile the inventory did not carry still gets a row —
    /// an operator-created connection that is invisible is the one
    /// failure this screen must not have.
    #[test]
    fn a_profile_missing_from_the_inventory_still_gets_a_row() {
        let d = ProfilesData::from_value(&json!({
            "providers": [
                {"provider": "openai", "kind": "cloud_api", "api_key_field": "openai",
                 "api_key_env_var": "OPENAI_API_KEY"}
            ],
            "profiles": [
                {"id": "paritytest", "provider_family": "openai-compatible",
                 "base_url": "http://127.0.0.1:1234/v1", "api_key_set": false,
                 "allowed_models": [], "enabled": true}
            ]
        }));
        let rows = d.connections();
        assert_eq!(rows.len(), 2);
        let orphan = rows.last().unwrap();
        assert_eq!(orphan.provider, "endpoint:paritytest");
        assert_eq!(orphan.family, "openai-compatible");
        assert_eq!(orphan.origin, Origin::Config);
        assert_eq!(orphan.enabled_text(), "yes");
        assert_eq!(orphan.api_key, "none");
        assert!(orphan.is_profile(), "e/d still act on the stored profile");
    }

    /// The live count comes from the probe's own sentence — and only
    /// from a sentence that actually carries one.
    #[test]
    fn model_counts_are_read_not_guessed() {
        assert_eq!(models_reported("reachable (43 models)"), Some(43));
        assert_eq!(models_reported("reachable (0 models)"), Some(0));
        assert_eq!(models_reported(""), None);
        assert_eq!(models_reported("GET http://x/v1/models unreachable"), None);
        assert_eq!(models_reported("reachable (many models)"), None);
    }
}
