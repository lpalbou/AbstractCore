//! Tolerant view-models over the contract A–E documents.
//!
//! Parsing never fails: an unknown key is ignored, a missing or
//! mistyped field is `None` and renders as "unknown" — never a guess
//! (the contracts' own rule: unknown = `null`). The raw document is not
//! kept; everything the screens print is a named field here.
//!
//! Every view struct is `#[non_exhaustive]`: it mirrors a contract that
//! grows (a field per new contract key), so a host reads fields and
//! never builds one — the `from_value` parsers do.

use serde_json::Value;

fn s(v: &Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}
fn b(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(Value::as_bool)
}
fn u(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| {
        x.as_u64()
            .or_else(|| x.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64))
    })
}
fn f(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}
fn strings(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Human byte size (binary units, one decimal) — `—` for unknown.
pub fn bytes_label(n: Option<u64>) -> String {
    match n {
        None => "—".to_string(),
        Some(n) => {
            const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
            let mut v = n as f64;
            let mut i = 0;
            while v >= 1024.0 && i < UNITS.len() - 1 {
                v /= 1024.0;
                i += 1;
            }
            if i == 0 {
                format!("{n} B")
            } else {
                format!("{v:.1} {}", UNITS[i])
            }
        }
    }
}

/// Parameter count in the vocabulary model cards use (`8.2B`, `600M`).
pub fn params_label(n: Option<u64>) -> String {
    match n {
        None => "—".to_string(),
        Some(n) if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
        Some(n) if n >= 1_000_000 => format!("{}M", n / 1_000_000),
        Some(n) => n.to_string(),
    }
}

// ---------------------------------------------------------------------
// A. Host profile
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct HostProfile {
    pub os: Option<String>,
    pub arch: Option<String>,
    pub accelerator: Option<String>,
    pub gpu_name: Option<String>,
    pub unified_memory: Option<bool>,
    pub ram_bytes: Option<u64>,
    pub vram_bytes: Option<u64>,
    pub ceiling_bytes: Option<u64>,
    pub ceiling_source: Option<String>,
    pub free_now_bytes: Option<u64>,
    /// Free disk per model store (`hf_cache`, `ollama`, `lmstudio`).
    pub disk: Vec<(String, Option<u64>)>,
}

impl HostProfile {
    pub fn from_value(v: &Value) -> HostProfile {
        let disk = v
            .get("disk")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, d)| (k.clone(), u(d, "free_bytes")))
                    .collect()
            })
            .unwrap_or_default();
        HostProfile {
            os: s(v, "os"),
            arch: s(v, "arch"),
            accelerator: s(v, "accelerator"),
            gpu_name: s(v, "gpu_name"),
            unified_memory: b(v, "unified_memory"),
            ram_bytes: u(v, "ram_bytes"),
            vram_bytes: u(v, "vram_bytes"),
            ceiling_bytes: u(v, "ceiling_bytes"),
            ceiling_source: s(v, "ceiling_source"),
            free_now_bytes: u(v, "free_now_bytes"),
            disk,
        }
    }

    /// One line: `Apple M5 Max · metal · 128.0 GiB unified · models up to
    /// 96.0 GiB · 56.8 GiB free now`.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(g) = &self.gpu_name {
            parts.push(g.clone());
        } else if let (Some(os), Some(arch)) = (&self.os, &self.arch) {
            parts.push(format!("{os}/{arch}"));
        }
        if let Some(a) = &self.accelerator {
            parts.push(a.clone());
        }
        if let Some(r) = self.ram_bytes {
            let unified = if self.unified_memory == Some(true) {
                " unified"
            } else {
                " RAM"
            };
            parts.push(format!("{}{unified}", bytes_label(Some(r))));
        }
        if let Some(v) = self.vram_bytes {
            parts.push(format!("{} VRAM", bytes_label(Some(v))));
        }
        if let Some(c) = self.ceiling_bytes {
            parts.push(format!("models up to {}", bytes_label(Some(c))));
        }
        if let Some(fr) = self.free_now_bytes {
            parts.push(format!("{} free now", bytes_label(Some(fr))));
        }
        if parts.is_empty() {
            "host profile unknown".to_string()
        } else {
            parts.join(" · ")
        }
    }
}

// ---------------------------------------------------------------------
// B. Engines
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct InstallPlan {
    pub available: bool,
    /// `brew | script | winget | pip | download_page` (None = no plan).
    pub method: Option<String>,
    /// The FIXED argv the backend would run (empty for a download page).
    pub argv: Vec<String>,
    pub url: Option<String>,
    pub requires_confirmation: bool,
    pub estimated_bytes: Option<u64>,
    pub notes: Option<String>,
    /// Gateway v2: one step needs an administrator (`None` = not said).
    pub needs_admin: Option<bool>,
    pub admin_reason: Option<String>,
    /// Gateway v2: the plan's steps, in words.
    pub steps: Vec<String>,
    /// Gateway v2: where it would land (`/Applications/Ollama.app`).
    pub target: Option<String>,
}

/// One entry of a gateway v2 engine row's `actions` (install, start,
/// stop, open_page, recheck, docs) — `enabled` + `reason` carry the
/// gateway's own guard (admin-only, installs disabled, a job running).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct EngineAction {
    pub id: String,
    pub label: Option<String>,
    pub enabled: bool,
    pub reason: Option<String>,
    pub url: Option<String>,
}

/// A gateway v2 row's `active_job` pointer (the engine's live install).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ActiveJobRef {
    pub job_id: String,
    pub state: Option<String>,
    pub percent: Option<f64>,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct EngineRow {
    pub id: String,
    pub name: String,
    pub kind: Option<String>,
    pub supported_on_host: Option<bool>,
    pub unsupported_reason: Option<String>,
    pub installed: Option<bool>,
    pub version: Option<String>,
    pub install_location: Option<String>,
    pub running: Option<bool>,
    pub base_url: Option<String>,
    pub reachable: Option<bool>,
    pub models_count: Option<u64>,
    pub install: InstallPlan,
    pub docs_url: Option<String>,
    /// Gateway v2 `actions`; `None` = the backend sends no action list
    /// (contract B: the CLI), so no verb is gated by it.
    pub actions: Option<Vec<EngineAction>>,
    pub active_job: Option<ActiveJobRef>,
}

impl EngineRow {
    pub fn from_value(v: &Value) -> EngineRow {
        let i = v.get("install").cloned().unwrap_or(Value::Null);
        let id = s(v, "id").unwrap_or_default();
        EngineRow {
            name: s(v, "name").unwrap_or_else(|| id.clone()),
            id,
            kind: s(v, "kind"),
            supported_on_host: b(v, "supported_on_host"),
            unsupported_reason: s(v, "unsupported_reason"),
            installed: b(v, "installed"),
            version: s(v, "version"),
            install_location: s(v, "install_location"),
            running: b(v, "running"),
            base_url: s(v, "base_url"),
            reachable: b(v, "reachable"),
            models_count: u(v, "models_count"),
            install: InstallPlan {
                available: b(&i, "available").unwrap_or(false),
                method: s(&i, "method"),
                argv: strings(&i, "argv"),
                url: s(&i, "url"),
                requires_confirmation: b(&i, "requires_confirmation").unwrap_or(true),
                estimated_bytes: u(&i, "estimated_bytes").or_else(|| u(&i, "download_bytes")),
                notes: s(&i, "notes"),
                needs_admin: b(&i, "needs_admin"),
                admin_reason: s(&i, "admin_reason"),
                steps: strings(&i, "steps"),
                target: s(&i, "target"),
            },
            docs_url: s(v, "docs_url"),
            actions: v.get("actions").and_then(Value::as_array).map(|a| {
                a.iter()
                    .filter_map(|x| {
                        Some(EngineAction {
                            id: s(x, "id")?,
                            label: s(x, "label"),
                            enabled: b(x, "enabled").unwrap_or(false),
                            reason: s(x, "reason"),
                            url: s(x, "url"),
                        })
                    })
                    .collect()
            }),
            active_job: v.get("active_job").filter(|j| j.is_object()).and_then(|j| {
                Some(ActiveJobRef {
                    job_id: s(j, "job_id")?,
                    state: s(j, "state"),
                    percent: f(j, "percent"),
                    message: s(j, "message"),
                })
            }),
        }
    }

    /// The v2 action `id`, when the backend sent an action list.
    pub fn action(&self, id: &str) -> Option<&EngineAction> {
        self.actions.as_ref()?.iter().find(|a| a.id == id)
    }

    /// True when this is an app engine (Ollama / LM Studio on macOS) —
    /// the only installs where a location means anything.
    pub fn is_app_install(&self) -> bool {
        self.install.method.as_deref() == Some("app")
    }

    /// `installed` / `not installed` / `remote only` / `unsupported` /
    /// `unknown`. `remote only` = this host cannot run it, but a server
    /// of it on another machine can be used (vLLM on a CPU box).
    pub fn install_label(&self) -> &'static str {
        if self.supported_on_host == Some(false) {
            return if self.kind.as_deref() == Some("remote_only") {
                "remote only"
            } else {
                "unsupported"
            };
        }
        match self.installed {
            Some(true) => "installed",
            Some(false) => "not installed",
            None => "unknown",
        }
    }

    /// `running` / `stopped` / `—` (unknown or not a server).
    pub fn running_label(&self) -> &'static str {
        match (self.running, self.reachable) {
            (Some(true), Some(false)) => "unreachable",
            (Some(true), _) => "running",
            (Some(false), _) => "stopped",
            (None, _) => "—",
        }
    }

    /// The page `o` opens: the install plan's URL, else the docs.
    pub fn open_url(&self) -> Option<&str> {
        self.install.url.as_deref().or(self.docs_url.as_deref())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct EnginesData {
    pub engines: Vec<EngineRow>,
}

impl EnginesData {
    pub fn from_value(v: &Value) -> EnginesData {
        EnginesData {
            engines: v
                .get("engines")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(EngineRow::from_value).collect())
                .unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------
// C. Catalog
// ---------------------------------------------------------------------

/// Contract C/G weight label for a presence status.
pub fn weights_label(status: &str) -> &'static str {
    match status {
        "installed" => "installed",
        "absent" => "not downloaded",
        "not_applicable" => "remote",
        _ => "unknown",
    }
}

/// Contract G fit badge for a verdict.
pub fn fit_label(verdict: &str) -> &'static str {
    match verdict {
        "fits" => "fits",
        "tight" => "tight",
        "too_large" => "too large",
        "partial_offload" => "partial offload",
        "needs_gpu_limit" => "needs GPU limit",
        _ => "unknown",
    }
}

/// Core's `fit.gpu_limit` (backlog 0947): the model fits once macOS lets
/// the GPU wire this much memory — `command` sets it (admin; lasts until
/// the Mac restarts).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct GpuLimit {
    pub required_mb: u64,
    pub command: String,
}

impl GpuLimit {
    fn from_value(v: Option<&Value>) -> Option<GpuLimit> {
        let v = v.filter(|v| v.is_object())?;
        Some(GpuLimit {
            required_mb: u(v, "required_mb")?,
            command: s(v, "command")?,
        })
    }

    /// The sentence the detail line prints: the value, the exact
    /// command, and what it costs.
    pub fn instruction(&self) -> String {
        format!(
            "fits once macOS lets the GPU use {} GiB: run `{}` (asks for your password; \
             lasts until the Mac restarts)",
            self.required_mb / 1024,
            self.command
        )
    }
}

/// One downloadable artifact of one catalog model — the Models table's row.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ArtifactRow {
    pub model_id: String,
    pub model_name: String,
    pub params_total: Option<u64>,
    pub provider: String,
    pub artifact: String,
    pub quant: Option<String>,
    pub download_bytes: Option<u64>,
    pub size_source: Option<String>,
    /// `installed | absent | unknown | not_applicable`.
    pub presence: String,
    pub presence_location: Option<String>,
    /// `fits | tight | too_large | partial_offload | needs_gpu_limit | unknown`.
    pub fit: String,
    /// `needs_gpu_limit` only: the Mac's GPU memory limit that makes it
    /// fit (Core `fit.gpu_limit`), with the exact command.
    pub gpu_limit: Option<GpuLimit>,
    pub need_bytes: Option<u64>,
    pub fits_now: Option<bool>,
    pub disk_ok: Option<bool>,
    pub max_context: Option<u64>,
    pub confidence: Option<String>,
    pub fit_notes: Vec<String>,
    pub downloadable: bool,
    pub recommended: bool,
    pub tags: Vec<String>,
    /// What the host can give a model (`fit.usable_bytes`), the number
    /// the verdict compares `need_bytes` with.
    pub usable_bytes: Option<u64>,
    pub free_now_bytes: Option<u64>,
    /// `false` = its engine does not run on this host.
    pub supported_on_host: Option<bool>,
    /// The model's `capabilities.text` / `.embedding` (who may be the
    /// default TEXT model: text and not an embedder).
    pub text_capable: Option<bool>,
    pub embedding: Option<bool>,
    /// Part of the recommended starter set.
    pub starter: bool,
    /// The row's origin: `curated`, `hf_search` (a hub search hit)…
    pub source: Option<String>,
    /// The model's types ([`CATEGORIES`] ids) from its `capabilities`.
    pub categories: Vec<&'static str>,
}

/// The Models screen's type filter (`t`): `(id, label)` in the web
/// console's chip order (`MC_CAPS`, console_catalog.py), video included.
pub const CATEGORIES: &[(&str, &str)] = &[
    ("text", "Text"),
    ("thinking", "Thinking"),
    ("tools", "Tools"),
    ("vision", "Vision"),
    ("audio", "Audio"),
    ("embedding", "Embedding"),
    ("voice", "Voice"),
    ("image", "Image"),
    ("video", "Video"),
];

/// A catalog model's types from its contract-C `capabilities` — the web
/// console's `mcRowCaps` rule, flag for flag (`tools` is `native` or
/// `prompted`; a speech synthesizer is `voice`, not `audio`).
pub fn categories_of(caps: &Value) -> Vec<&'static str> {
    let on = |k: &str| b(caps, k) == Some(true);
    let tools = matches!(
        caps.get("tools").and_then(Value::as_str),
        Some("native" | "prompted")
    );
    let mut out = Vec::new();
    for (id, yes) in [
        ("text", on("text")),
        ("thinking", on("thinking")),
        ("tools", tools),
        ("vision", on("vision")),
        ("audio", on("audio") && !on("speech_synthesis")),
        ("embedding", on("embedding")),
        ("voice", on("speech_synthesis")),
        ("image", on("image_generation")),
        ("video", on("video_generation")),
    ] {
        if yes {
            out.push(id);
        }
    }
    out
}

impl ArtifactRow {
    /// The size a download may announce to the backend's disk pre-check:
    /// the catalog's `download_bytes` only when its `size_source` vouches
    /// for it (`catalog`, `hf_api`) — the web console's rule. An estimate
    /// is never sent as a promise.
    pub fn expected_bytes(&self) -> Option<u64> {
        self.download_bytes
            .filter(|n| *n > 0)
            .filter(|_| matches!(self.size_source.as_deref(), Some("catalog" | "hf_api")))
    }

    /// May this artifact be made the default text model? (The web
    /// console's rule: installed, text-capable, not an embedder.)
    pub fn can_be_text_default(&self) -> bool {
        self.presence == "installed"
            && self.text_capable == Some(true)
            && self.embedding != Some(true)
    }
}

/// The model id a capability route stores for an installed artifact:
/// LM Studio serves `org/model@quant` as `org/model` (the route drops
/// the quantization suffix); every other engine serves the artifact id
/// as is. The same rule as the web console's `servedModelId`.
pub fn served_model_id(provider: &str, artifact: &str) -> String {
    if provider == "lmstudio" {
        if let Some(i) = artifact.rfind('@') {
            return artifact[..i].to_string();
        }
    }
    artifact.to_string()
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct CatalogData {
    pub host: Option<HostProfile>,
    /// Flattened: one row per (model, artifact), catalog order.
    pub rows: Vec<ArtifactRow>,
    /// Distinct providers across all artifacts (the `e` cycle).
    pub providers: Vec<String>,
    /// A hub answer's `hub.ok` (`Some(false)` = Hugging Face answered in
    /// part only: offline, rate limited) and its `hub.errors`.
    pub hub_ok: Option<bool>,
    pub hub_errors: Vec<String>,
}

impl CatalogData {
    pub fn from_value(v: &Value) -> CatalogData {
        let mut rows = Vec::new();
        for m in v
            .get("rows")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let model_id = s(m, "id").unwrap_or_default();
            let model_name = s(m, "display_name").unwrap_or_else(|| model_id.clone());
            let params_total = u(m, "params_total");
            let tags = strings(m, "tags");
            let caps = m.get("capabilities").cloned().unwrap_or(Value::Null);
            let (text_capable, embedding) = (b(&caps, "text"), b(&caps, "embedding"));
            let categories = categories_of(&caps);
            let starter = b(m, "starter").unwrap_or(false);
            let source = s(m, "source");
            for a in m
                .get("artifacts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let p = a.get("presence").cloned().unwrap_or(Value::Null);
                let fit = a.get("fit").cloned().unwrap_or(Value::Null);
                rows.push(ArtifactRow {
                    model_id: model_id.clone(),
                    model_name: model_name.clone(),
                    params_total,
                    provider: s(a, "provider").unwrap_or_default(),
                    artifact: s(a, "artifact").unwrap_or_default(),
                    quant: s(a, "quant"),
                    download_bytes: u(a, "download_bytes"),
                    size_source: s(a, "size_source"),
                    presence: s(&p, "status").unwrap_or_else(|| "unknown".into()),
                    presence_location: s(&p, "location"),
                    fit: s(&fit, "verdict").unwrap_or_else(|| "unknown".into()),
                    gpu_limit: GpuLimit::from_value(fit.get("gpu_limit")),
                    need_bytes: u(&fit, "need_bytes"),
                    fits_now: b(&fit, "fits_now"),
                    disk_ok: b(&fit, "disk_ok"),
                    max_context: u(&fit, "max_context"),
                    confidence: s(&fit, "confidence"),
                    fit_notes: strings(&fit, "notes"),
                    downloadable: b(a, "downloadable").unwrap_or(false),
                    recommended: b(a, "recommended").unwrap_or(false),
                    tags: tags.clone(),
                    usable_bytes: u(&fit, "usable_bytes"),
                    free_now_bytes: u(&fit, "free_now_bytes"),
                    supported_on_host: b(a, "supported_on_host"),
                    text_capable,
                    embedding,
                    starter,
                    source: source.clone(),
                    categories: categories.clone(),
                });
            }
        }
        let mut providers: Vec<String> = Vec::new();
        for r in &rows {
            if !r.provider.is_empty() && !providers.contains(&r.provider) {
                providers.push(r.provider.clone());
            }
        }
        CatalogData {
            host: v
                .get("host_profile")
                .filter(|h| h.is_object())
                .map(HostProfile::from_value),
            rows,
            providers,
            hub_ok: v.get("hub").and_then(|h| b(h, "ok")),
            hub_errors: v
                .get("hub")
                .map(|h| strings(h, "errors"))
                .unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------
// D. Installed
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct InstalledRow {
    pub provider: String,
    pub artifact: String,
    pub quant: Option<String>,
    pub size_bytes: Option<u64>,
    pub params_total: Option<u64>,
    pub location: Option<String>,
    pub loaded: Option<bool>,
    pub catalog_id: Option<String>,
    pub deletable: bool,
    pub delete_blockers: Vec<String>,
}

impl InstalledRow {
    pub fn from_value(v: &Value) -> InstalledRow {
        InstalledRow {
            provider: s(v, "provider").unwrap_or_default(),
            artifact: s(v, "artifact").unwrap_or_default(),
            quant: s(v, "quant"),
            size_bytes: u(v, "size_bytes"),
            params_total: u(v, "params_total"),
            location: s(v, "location"),
            loaded: b(v, "loaded"),
            catalog_id: s(v, "catalog_id"),
            deletable: b(v, "deletable").unwrap_or(false),
            delete_blockers: strings(v, "delete_blockers"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct InstalledData {
    pub rows: Vec<InstalledRow>,
    pub engines_probed: Vec<String>,
    /// Per-engine read errors (`ollama: unreachable`) — shown, not hidden.
    pub errors: Vec<(String, String)>,
}

impl InstalledData {
    pub fn from_value(v: &Value) -> InstalledData {
        InstalledData {
            rows: v
                .get("rows")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(InstalledRow::from_value).collect())
                .unwrap_or_default(),
            engines_probed: strings(v, "engines_probed"),
            errors: v
                .get("errors")
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .map(|(k, e)| {
                            (
                                k.clone(),
                                e.as_str()
                                    .map(str::to_string)
                                    .unwrap_or_else(|| e.to_string()),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    pub fn find(&self, provider: &str, artifact: &str) -> Option<&InstalledRow> {
        self.rows
            .iter()
            .find(|r| r.provider == provider && r.artifact == artifact)
    }
}

// ---------------------------------------------------------------------
// E. Jobs
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct JobView {
    pub job_id: String,
    /// `download | delete | engine_install`.
    pub kind: String,
    /// `queued | running | completed | failed | cancelled`.
    pub status: String,
    pub provider: Option<String>,
    pub artifact: Option<String>,
    pub engine: Option<String>,
    pub percent: Option<f64>,
    pub downloaded_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    pub message: Option<String>,
    pub log_tail: Vec<String>,
    pub command: Vec<String>,
    pub dry_run: bool,
    pub error: Option<String>,
    pub cli_equivalent: Option<String>,
    /// The fine-grained state behind `status`: an engine install's
    /// `queued | downloading | installing | needs_admin | needs_tools |
    /// done | failed | cancelled`, a download's `resolving | stalled |
    /// verifying …`. A PAUSED job keeps `status: running`.
    pub state: Option<String>,
    pub engine_name: Option<String>,
    /// `needs_admin`: what must run as administrator, and where.
    pub admin_prompt: Option<AdminPrompt>,
    /// `needs_tools`: which tools are missing and how to get them.
    pub tools_prompt: Option<ToolsPrompt>,
    /// What `continue` accepts now (`approve_admin`, `install_tools`,
    /// `recheck`), in the backend's order.
    pub continue_actions: Vec<String>,
    /// A download's own sentence for how it ended ("admin cancelled this
    /// download in the console at 21:15").
    pub ended_reason: Option<String>,
    /// A child of a "download all" group names its parent.
    pub parent_job: Option<String>,
    pub bytes_per_second: Option<f64>,
    pub eta_s: Option<u64>,
}

/// A paused install's administrator step (gateway `admin_prompt`).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct AdminPrompt {
    pub reason: Option<String>,
    /// The EXACT command that runs elevated — for `method: manual`, the
    /// one the operator types in a terminal on the host.
    pub command: Option<String>,
    /// `osascript | pkexec | manual`.
    pub method: Option<String>,
    pub button: Option<String>,
    /// "a terminal on the gateway host" / "the gateway host's screen".
    pub where_: Option<String>,
}

/// A paused install's missing tools (gateway `tools_prompt`).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ToolsPrompt {
    pub reason: Option<String>,
    pub tools: Option<String>,
    pub command: Option<String>,
    pub available: bool,
    pub button: Option<String>,
    pub started: bool,
}

impl JobView {
    pub fn from_value(v: &Value) -> JobView {
        // Accept a bare job or a `{"job": {...}}` envelope (the gateway's
        // legacy `/models/download` answer).
        let v = match v.get("job") {
            Some(j) if j.is_object() => j,
            _ => v,
        };
        let admin_prompt = v
            .get("admin_prompt")
            .filter(|p| p.is_object())
            .map(|p| AdminPrompt {
                reason: s(p, "reason"),
                command: s(p, "command"),
                method: s(p, "method"),
                button: s(p, "button"),
                where_: s(p, "where"),
            });
        let tools_prompt = v.get("tools_prompt").filter(|p| p.is_object()).map(|p| {
            let a = p.get("action").cloned().unwrap_or(Value::Null);
            ToolsPrompt {
                reason: s(p, "reason"),
                tools: s(p, "tools"),
                command: s(&a, "command"),
                available: b(&a, "available").unwrap_or(false),
                button: s(&a, "button"),
                started: b(p, "started").unwrap_or(false),
            }
        });
        // An engine job's `error` is `{code, message}`; a download's a string.
        let error = s(v, "error").or_else(|| v.get("error").and_then(|e| s(e, "message")));
        JobView {
            // Download dicts carry the id as `job` too (older gateways
            // only that).
            job_id: s(v, "job_id").or_else(|| s(v, "job")).unwrap_or_default(),
            kind: s(v, "kind").unwrap_or_default(),
            status: s(v, "status").unwrap_or_else(|| "unknown".into()),
            provider: s(v, "provider"),
            artifact: s(v, "artifact"),
            engine: s(v, "engine"),
            percent: f(v, "percent"),
            downloaded_bytes: u(v, "downloaded_bytes").or_else(|| u(v, "bytes_done")),
            total_bytes: u(v, "total_bytes").or_else(|| u(v, "bytes_total")),
            message: s(v, "message"),
            log_tail: strings(v, "log_tail"),
            command: strings(v, "command"),
            dry_run: b(v, "dry_run").unwrap_or(false),
            error,
            cli_equivalent: s(v, "cli_equivalent"),
            state: s(v, "state"),
            engine_name: s(v, "engine_name"),
            admin_prompt,
            tools_prompt,
            continue_actions: strings(v, "continue_actions"),
            ended_reason: s(v, "ended_reason"),
            parent_job: s(v, "parent_job"),
            bytes_per_second: f(v, "bytes_per_second"),
            eta_s: u(v, "eta_s"),
        }
    }

    /// Waiting for a PERSON (an administrator step or missing tools):
    /// still `running` for the backend, but nothing moves until someone
    /// acts — it must never hold other work hostage.
    pub fn is_paused(&self) -> bool {
        matches!(self.state.as_deref(), Some("needs_admin" | "needs_tools"))
    }

    /// Same subject as another job (dedupe key in the jobs list).
    pub fn same_subject(&self, other: &JobView) -> bool {
        self.kind == other.kind
            && self.provider == other.provider
            && self.artifact == other.artifact
            && self.engine == other.engine
    }

    /// The paused job's one-line headline ("needs an administrator").
    pub fn paused_label(&self) -> Option<&'static str> {
        match self.state.as_deref() {
            Some("needs_admin") => Some("needs an administrator"),
            Some("needs_tools") => Some("needs developer tools"),
            _ => None,
        }
    }

    /// The command the operator can copy for a paused job: the admin
    /// command, else the tools command.
    pub fn copyable_command(&self) -> Option<&str> {
        if self.state.as_deref() == Some("needs_admin") {
            return self
                .admin_prompt
                .as_ref()
                .and_then(|p| p.command.as_deref());
        }
        if self.state.as_deref() == Some("needs_tools") {
            return self
                .tools_prompt
                .as_ref()
                .and_then(|p| p.command.as_deref());
        }
        None
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status.as_str(), "queued" | "running")
    }

    /// What the job is about: `ollama qwen3:8b` or the engine id.
    pub fn subject(&self) -> String {
        match (&self.provider, &self.artifact, &self.engine) {
            (Some(p), Some(a), _) => format!("{p} {a}"),
            (_, Some(a), _) => a.clone(),
            (_, _, Some(e)) => e.clone(),
            _ => self.job_id.clone(),
        }
    }

    pub fn verb(&self) -> &'static str {
        match self.kind.as_str() {
            "download" => "download",
            "delete" => "delete",
            "engine_install" => "install",
            "download_group" => "download all",
            _ => "job",
        }
    }

    /// Progress in 0..=1 when the backend knows it.
    pub fn fraction(&self) -> Option<f32> {
        if let Some(p) = self.percent {
            return Some((p / 100.0).clamp(0.0, 1.0) as f32);
        }
        match (self.downloaded_bytes, self.total_bytes) {
            (Some(d), Some(t)) if t > 0 => Some((d as f64 / t as f64).clamp(0.0, 1.0) as f32),
            _ => None,
        }
    }

    /// The one-line outcome the toast carries.
    pub fn outcome_line(&self) -> String {
        let what = format!("{} {}", self.verb(), self.subject());
        match self.status.as_str() {
            "completed" if self.dry_run => format!(
                "dry run: {what} would run `{}`",
                if self.command.is_empty() {
                    "(no command)".to_string()
                } else {
                    self.command.join(" ")
                }
            ),
            "completed" => format!("✓ {what} completed"),
            "cancelled" => match &self.ended_reason {
                Some(r) => format!("⊘ {what} cancelled: {r}"),
                None => format!("⊘ {what} cancelled"),
            },
            "failed" => format!(
                "✗ {what} failed: {}",
                self.error
                    .as_deref()
                    .or(self.message.as_deref())
                    .unwrap_or("no reason given")
            ),
            _ if self.is_paused() => format!(
                "⏸ {what} {}: {}",
                self.paused_label().unwrap_or("is waiting"),
                self.message.as_deref().unwrap_or("see the Engines screen")
            ),
            other => format!("{what}: {other}"),
        }
    }
}

/// The downloads feed (`{"jobs": [...]}`): every download job, newest
/// first, a "download all" group's children flattened after it (each
/// naming its `parent_job`).
pub fn download_jobs_from_value(v: &Value) -> Vec<JobView> {
    let mut out = Vec::new();
    for j in v
        .get("jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let view = JobView::from_value(j);
        if view.job_id.is_empty() {
            continue;
        }
        let parent = view.job_id.clone();
        let group = view.kind == "download_group";
        out.push(view);
        if group {
            for c in j
                .get("children")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let mut child = JobView::from_value(c);
                if child.job_id.is_empty() {
                    continue;
                }
                child.parent_job.get_or_insert(parent.clone());
                out.push(child);
            }
        }
    }
    out
}

/// The route the default text model lives on.
pub const TEXT_ROUTE: &str = "output.text";

/// The configured default text model (`provider`, `model`) in a
/// capability-defaults document — `None` when the route names none.
pub fn text_default_from_value(v: &Value) -> Option<(String, String)> {
    v.get("routes")
        .and_then(Value::as_array)?
        .iter()
        .find(|r| r.get("key").and_then(Value::as_str) == Some(TEXT_ROUTE))
        .and_then(|r| Some((s(r, "provider")?, s(r, "model")?)))
}

/// One install location's plan (from a dry run's `plan`).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct LocationPlan {
    pub target: Option<String>,
    pub needs_admin: bool,
    pub admin_reason: Option<String>,
}

impl LocationPlan {
    /// `None` when the dry run carried no `plan` (an old backend).
    pub fn from_dry_run(v: &Value) -> Option<LocationPlan> {
        let p = v.get("plan").filter(|p| p.is_object())?;
        Some(LocationPlan {
            target: s(p, "target"),
            needs_admin: b(p, "needs_admin").unwrap_or(false),
            admin_reason: s(p, "admin_reason"),
        })
    }
}

/// Both real plans for an app engine: just this account vs every one.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct InstallPlans {
    pub user: LocationPlan,
    pub system: LocationPlan,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn labels_speak_the_shared_vocabulary() {
        assert_eq!(weights_label("absent"), "not downloaded");
        assert_eq!(weights_label("not_applicable"), "remote");
        assert_eq!(weights_label("whatever"), "unknown");
        assert_eq!(fit_label("too_large"), "too large");
        assert_eq!(fit_label("partial_offload"), "partial offload");
        assert_eq!(fit_label("needs_gpu_limit"), "needs GPU limit");
        let gl = GpuLimit::from_value(Some(&json!({"required_mb": 117760,
            "command": "sudo sysctl iogpu.wired_limit_mb=117760"})))
        .unwrap();
        assert!(gl.instruction().starts_with(
            "fits once macOS lets the GPU use 115 GiB: run `sudo sysctl iogpu.wired_limit_mb=117760`"
        ));
        assert!(
            GpuLimit::from_value(Some(&json!({"required_mb": 1}))).is_none(),
            "no command, no line"
        );
        assert_eq!(bytes_label(Some(5_200_000_000)), "4.8 GiB");
        assert_eq!(bytes_label(None), "—");
        assert_eq!(params_label(Some(8_200_000_000)), "8.2B");
    }

    #[test]
    fn job_fraction_and_outcome() {
        let j = JobView::from_value(&json!({"job": {
            "job_id": "dl_1", "kind": "download", "status": "running",
            "provider": "ollama", "artifact": "qwen3:8b",
            "downloaded_bytes": 50, "total_bytes": 200
        }}));
        assert!(j.is_active());
        assert_eq!(j.fraction(), Some(0.25));
        let done = JobView {
            status: "completed".into(),
            ..j
        };
        assert_eq!(done.outcome_line(), "✓ download ollama qwen3:8b completed");
    }

    /// The gateway's `engine_install_job_v1` snapshot of Ollama on Linux
    /// as a non-root user (engines_install.py: manual sudo, recheck).
    fn paused_ollama() -> Value {
        json!({
            "schema": "host_job_v1", "engine_job_schema": "engine_install_job_v1",
            "job_id": "ei_7", "kind": "engine_install", "engine": "ollama",
            "engine_name": "Ollama", "state": "needs_admin", "status": "running",
            "percent": 10.0, "bytes_done": 5, "bytes_total": 50,
            "message": "Ollama's Linux installer writes /usr/local … Press \"I ran it -- re-check\" to continue.",
            "admin_prompt": {"key": "ollama-linux-script", "reason": "needs root",
                             "command": "sudo sh -c 'curl -fsSL https://ollama.com/install.sh | sh'",
                             "method": "manual", "button": "I ran it -- re-check",
                             "where": "a terminal on the gateway host"},
            "tools_prompt": null, "continue_actions": ["recheck"],
            "error": null, "can_cancel": true
        })
    }

    #[test]
    fn a_paused_install_is_active_paused_and_names_its_command() {
        let j = JobView::from_value(&paused_ollama());
        assert!(j.is_active(), "status stays running while paused");
        assert!(j.is_paused());
        assert_eq!(j.paused_label(), Some("needs an administrator"));
        assert_eq!(
            j.copyable_command(),
            Some("sudo sh -c 'curl -fsSL https://ollama.com/install.sh | sh'")
        );
        assert_eq!(j.continue_actions, vec!["recheck"]);
        assert_eq!(
            j.admin_prompt.as_ref().unwrap().where_.as_deref(),
            Some("a terminal on the gateway host")
        );
        // Engine jobs count bytes as bytes_done/bytes_total.
        assert_eq!(j.fraction(), Some(0.1));
        assert!(
            j.outcome_line()
                .starts_with("⏸ install ollama needs an administrator"),
            "{}",
            j.outcome_line()
        );
        // The same job running again is not paused.
        let mut v = paused_ollama();
        v["state"] = json!("installing");
        let j = JobView::from_value(&v);
        assert!(j.is_active() && !j.is_paused() && j.copyable_command().is_none());
        // An engine job's error object reads as its message.
        v["status"] = json!("failed");
        v["error"] = json!({"code": "checksum_mismatch", "message": "bad sha"});
        assert_eq!(JobView::from_value(&v).error.as_deref(), Some("bad sha"));
    }

    #[test]
    fn downloads_feed_flattens_groups_and_reads_the_job_alias() {
        let v = json!({"ok": true, "jobs": [
            {"job": "grp_1", "kind": "download_group", "status": "running",
             "children": [{"job_id": "dl_a", "kind": "download", "status": "running",
                           "provider": "ollama", "artifact": "qwen3:8b"}]},
            {"job": "dl_old", "kind": "download", "status": "completed",
             "provider": "huggingface", "artifact": "x/y", "ended_reason": "done"},
            {"kind": "download"}
        ]});
        let jobs = download_jobs_from_value(&v);
        let ids: Vec<&str> = jobs.iter().map(|j| j.job_id.as_str()).collect();
        assert_eq!(ids, vec!["grp_1", "dl_a", "dl_old"]);
        assert_eq!(jobs[0].verb(), "download all");
        assert_eq!(jobs[1].parent_job.as_deref(), Some("grp_1"));
    }

    #[test]
    fn text_default_and_served_ids_follow_the_route_rules() {
        let doc = json!({"routes": [
            {"key": "input.text", "provider": "mlx", "model": "a"},
            {"key": "output.text", "provider": "lmstudio", "model": "qwen/qwen3-8b"}
        ]});
        assert_eq!(
            text_default_from_value(&doc),
            Some(("lmstudio".into(), "qwen/qwen3-8b".into()))
        );
        assert_eq!(
            text_default_from_value(&json!({"routes": [{"key": "output.text"}]})),
            None
        );
        assert_eq!(
            served_model_id("lmstudio", "qwen/qwen3-8b@4bit"),
            "qwen/qwen3-8b"
        );
        assert_eq!(served_model_id("ollama", "qwen3:8b"), "qwen3:8b");
        assert_eq!(served_model_id("mlx", "a@b"), "a@b");
    }

    #[test]
    fn v2_engine_rows_carry_actions_active_jobs_and_location_plans() {
        let e = EngineRow::from_value(&json!({
            "id": "ollama", "installed": true, "running": false,
            "install": {"available": true, "method": "app", "needs_admin": true,
                        "admin_reason": "not writable", "steps": ["a", "b"],
                        "target": "/Applications/Ollama.app"},
            "actions": [{"id": "start", "label": "Start", "enabled": false,
                         "reason": "Only an administrator can install engines."},
                        {"id": "docs", "enabled": true, "url": "https://docs.ollama.com"}],
            "active_job": {"job_id": "ei_7", "state": "needs_admin", "percent": 10.0}
        }));
        assert!(e.is_app_install());
        assert_eq!(e.install.needs_admin, Some(true));
        assert_eq!(e.install.steps, vec!["a", "b"]);
        let start = e.action("start").unwrap();
        assert!(!start.enabled && start.reason.as_deref().unwrap().contains("administrator"));
        assert!(e.action("stop").is_none());
        assert_eq!(e.active_job.as_ref().unwrap().job_id, "ei_7");
        // Contract B (the CLI): no action list at all, nothing gated.
        assert!(EngineRow::from_value(&json!({"id": "mlx"}))
            .actions
            .is_none());
        let plan =
            LocationPlan::from_dry_run(&json!({"plan": {"target": "/Applications/Ollama.app",
            "needs_admin": true, "admin_reason": "not writable"}}))
            .unwrap();
        assert!(plan.needs_admin);
        assert!(LocationPlan::from_dry_run(&json!({"status": "completed"})).is_none());
    }

    #[test]
    fn missing_fields_are_unknown_not_guessed() {
        let e = EngineRow::from_value(&json!({"id": "vllm", "supported_on_host": false}));
        assert_eq!(e.name, "vllm");
        assert_eq!(e.install_label(), "unsupported");
        let remote = EngineRow::from_value(
            &json!({"id": "vllm", "kind": "remote_only", "supported_on_host": false}),
        );
        assert_eq!(remote.install_label(), "remote only");
        assert_eq!(e.running_label(), "—");
        assert!(!e.install.available);
        let c = CatalogData::from_value(&json!({"rows": [{"id": "m", "artifacts": [{}]}]}));
        assert_eq!(c.rows[0].presence, "unknown");
        assert_eq!(c.rows[0].fit, "unknown");
        assert!(c.host.is_none());
    }
}
