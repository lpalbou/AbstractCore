//! Tolerant view-models over the contract A–E documents.
//!
//! Parsing never fails: an unknown key is ignored, a missing or
//! mistyped field is `None` and renders as "unknown" — never a guess
//! (the contracts' own rule: unknown = `null`). The raw document is not
//! kept; everything the screens print is a named field here.

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
}

#[derive(Clone, Debug, Default, PartialEq)]
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
                estimated_bytes: u(&i, "estimated_bytes"),
                notes: s(&i, "notes"),
            },
            docs_url: s(v, "docs_url"),
        }
    }

    /// `installed` / `not installed` / `unsupported` / `unknown`.
    pub fn install_label(&self) -> &'static str {
        if self.supported_on_host == Some(false) {
            return "unsupported";
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
        _ => "unknown",
    }
}

/// One downloadable artifact of one catalog model — the Models table's row.
#[derive(Clone, Debug, Default, PartialEq)]
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
    /// `fits | tight | too_large | partial_offload | unknown`.
    pub fit: String,
    pub need_bytes: Option<u64>,
    pub fits_now: Option<bool>,
    pub disk_ok: Option<bool>,
    pub max_context: Option<u64>,
    pub confidence: Option<String>,
    pub fit_notes: Vec<String>,
    pub downloadable: bool,
    pub recommended: bool,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CatalogData {
    pub host: Option<HostProfile>,
    /// Flattened: one row per (model, artifact), catalog order.
    pub rows: Vec<ArtifactRow>,
    /// Distinct providers across all artifacts (the `e` cycle).
    pub providers: Vec<String>,
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
                    need_bytes: u(&fit, "need_bytes"),
                    fits_now: b(&fit, "fits_now"),
                    disk_ok: b(&fit, "disk_ok"),
                    max_context: u(&fit, "max_context"),
                    confidence: s(&fit, "confidence"),
                    fit_notes: strings(&fit, "notes"),
                    downloadable: b(a, "downloadable").unwrap_or(false),
                    recommended: b(a, "recommended").unwrap_or(false),
                    tags: tags.clone(),
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
        }
    }
}

// ---------------------------------------------------------------------
// D. Installed
// ---------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
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
}

impl JobView {
    pub fn from_value(v: &Value) -> JobView {
        // Accept a bare job or a `{"job": {...}}` envelope (the gateway's
        // legacy `/models/download` answer).
        let v = match v.get("job") {
            Some(j) if j.is_object() => j,
            _ => v,
        };
        JobView {
            job_id: s(v, "job_id").unwrap_or_default(),
            kind: s(v, "kind").unwrap_or_default(),
            status: s(v, "status").unwrap_or_else(|| "unknown".into()),
            provider: s(v, "provider"),
            artifact: s(v, "artifact"),
            engine: s(v, "engine"),
            percent: f(v, "percent"),
            downloaded_bytes: u(v, "downloaded_bytes"),
            total_bytes: u(v, "total_bytes"),
            message: s(v, "message"),
            log_tail: strings(v, "log_tail"),
            command: strings(v, "command"),
            dry_run: b(v, "dry_run").unwrap_or(false),
            error: s(v, "error"),
            cli_equivalent: s(v, "cli_equivalent"),
        }
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
            "cancelled" => format!("⊘ {what} cancelled"),
            "failed" => format!(
                "✗ {what} failed: {}",
                self.error
                    .as_deref()
                    .or(self.message.as_deref())
                    .unwrap_or("no reason given")
            ),
            other => format!("{what}: {other}"),
        }
    }
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

    #[test]
    fn missing_fields_are_unknown_not_guessed() {
        let e = EngineRow::from_value(&json!({"id": "vllm", "supported_on_host": false}));
        assert_eq!(e.name, "vllm");
        assert_eq!(e.install_label(), "unsupported");
        assert_eq!(e.running_label(), "—");
        assert!(!e.install.available);
        let c = CatalogData::from_value(&json!({"rows": [{"id": "m", "artifacts": [{}]}]}));
        assert_eq!(c.rows[0].presence, "unknown");
        assert_eq!(c.rows[0].fit, "unknown");
        assert!(c.host.is_none());
    }
}
