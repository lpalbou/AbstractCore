//! Reading `abstractcore.json` and folding it into a REDACTED display
//! model. No writes live here (M1 is read-only; M2's write paths will
//! re-read fresh at write time — the model below is display currency).
//!
//! Redaction is structural: secrets are folded to set/not-set +
//! sha256[:8] fingerprints AT PARSE TIME and the raw `Value` is dropped
//! — no signal ever holds key material. The fingerprint convention is
//! the Python side's own (`provider_profiles.py:120-124`: sha256 of the
//! trimmed value, hex, first 8), so both surfaces show one fingerprint
//! for one key.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::schema::{self, FieldKind, SectionSpec};

/// Where the config path came from — always shown in the header, so
/// "which config am I editing" has a standing answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathSource {
    EnvFile,
    EnvDir,
    Default,
}

impl PathSource {
    pub fn label(&self) -> &'static str {
        match self {
            PathSource::EnvFile => "$ABSTRACTCORE_CONFIG_FILE",
            PathSource::EnvDir => "$ABSTRACTCORE_CONFIG_DIR",
            PathSource::Default => "default path",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConfigPath {
    pub path: PathBuf,
    pub source: PathSource,
}

/// Resolve the config file exactly as the Python manager does
/// (manager.py:338-349): `ABSTRACTCORE_CONFIG_FILE` (full path) wins
/// over `ABSTRACTCORE_CONFIG_DIR` (directory containing
/// `abstractcore.json`), else `~/.abstractcore/config/abstractcore.json`.
/// Python semantics followed exactly (adversarial review P2-3): only
/// the EMPTY string counts as unset (Python truthiness — whitespace is
/// a real, weird path), and `~` expands (`Path(...).expanduser()` runs
/// on every branch there).
pub fn resolve_config_path(env: &dyn Fn(&str) -> Option<String>, home: &Path) -> ConfigPath {
    if let Some(file) = env("ABSTRACTCORE_CONFIG_FILE").filter(|v| !v.is_empty()) {
        return ConfigPath {
            path: expand_user(&file, home),
            source: PathSource::EnvFile,
        };
    }
    if let Some(dir) = env("ABSTRACTCORE_CONFIG_DIR").filter(|v| !v.is_empty()) {
        return ConfigPath {
            path: expand_user(&dir, home).join("abstractcore.json"),
            source: PathSource::EnvDir,
        };
    }
    ConfigPath {
        path: home.join(".abstractcore").join("config").join("abstractcore.json"),
        source: PathSource::Default,
    }
}

/// `~` / `~/rest` → the home dir, matching Python's `expanduser` for
/// the shapes env vars actually deliver (`~user` is left alone — it is
/// vanishingly rare and mis-expanding it would be worse).
fn expand_user(raw: &str, home: &Path) -> PathBuf {
    if raw == "~" {
        return home.to_path_buf();
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return home.join(rest);
    }
    PathBuf::from(raw)
}

pub fn resolve_config_path_from_env() -> ConfigPath {
    let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
    resolve_config_path(&|k| std::env::var(k).ok(), &home)
}

/// The file-level truth. Corrupt/missing are honest states of the
/// mirror, not load failures — the UI renders each distinctly and M2's
/// write paths refuse everything but `Ready` (risk-map fact #4).
#[derive(Clone, Debug)]
pub enum FileState {
    /// No file yet: every section is at its dataclass default in
    /// memory; nothing is written until the first setter runs.
    Missing,
    Ready(Snapshot),
    /// Parse failed. NEVER write from this state; the backups are the
    /// recovery artifacts (Python's own corrupt path mints
    /// `<file>.corrupt-<stamp>.bak`; `.bak-repair-*` also exists in
    /// the wild).
    Corrupt {
        error: String,
        backups: Vec<String>,
    },
    /// I/O error other than not-found (permissions, etc.).
    Unreadable { error: String },
}

/// File identity for the drift guard: mtime alone is NOT identity on
/// coarse-mtime filesystems, and Python's save mints a new inode every
/// time (tmp + replace) — `(mtime, ino, size)` catches same-second
/// rewrites at zero cost (the workspace's JsonFileRunStore lesson;
/// M2 review P3-3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FileStamp {
    pub mtime: Option<SystemTime>,
    pub ino: u64,
    pub size: u64,
}

impl FileStamp {
    pub fn of(path: &Path) -> Option<FileStamp> {
        let meta = std::fs::metadata(path).ok()?;
        #[cfg(unix)]
        let ino = {
            use std::os::unix::fs::MetadataExt;
            meta.ino()
        };
        #[cfg(not(unix))]
        let ino = 0u64;
        Some(FileStamp {
            mtime: meta.modified().ok(),
            ino,
            size: meta.len(),
        })
    }
}

/// One parsed file, folded per section. Display-only; secrets already
/// reduced to fingerprints.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub sections: Vec<SectionView>,
    /// Top-level keys this console does not know. Preserved by OUR
    /// direct writes; the overview names them because a PYTHON-side
    /// save deletes them (manager.py:609-634).
    pub unknown_sections: Vec<String>,
    /// The effective meta flag (top-level, or legacy nested
    /// `audio.strategy_explicit` — manager.py:514-521).
    pub audio_strategy_explicit: bool,
    /// Route keys present in the file's capability_defaults with a
    /// non-empty object value — an approximation of what Python loads
    /// (it additionally normalizes aliases and skips unparseable keys,
    /// capability_defaults.py:296-304; the CLI-derived view is the
    /// exact truth when it is up).
    pub routes_in_file: Vec<String>,
    pub profiles_in_file: usize,
    /// Shapes Python's own loader RAISES on (adversarial review P1-1):
    /// the whole file then loads as DEFAULTS after a fresh
    /// `.corrupt-*.bak` — on EVERY Python invocation. When non-empty,
    /// the mirror must not vouch for anything it shows.
    pub python_refusals: Vec<String>,
    pub bytes: u64,
    pub mtime: Option<SystemTime>,
    /// Unix permission bits (file should be 0600 — secrets at rest).
    pub mode: Option<u32>,
    /// The file identity at load time (None for in-memory folds) —
    /// the drift guard's base.
    pub stamp: Option<FileStamp>,
}

#[derive(Clone, Debug)]
pub struct SectionView {
    pub spec: &'static SectionSpec,
    pub present_in_file: bool,
    pub fields: Vec<FieldView>,
    /// Keys inside a KNOWN section that the schema does not know.
    /// Python's loader silently drops these (manager.py:368-372) — a
    /// warning worth surfacing, since the next Python save loses them.
    pub unknown_keys: Vec<String>,
}

impl SectionView {
    pub fn set_count(&self) -> usize {
        self.fields
            .iter()
            .filter(|f| matches!(f.state, FieldState::Set))
            .count()
    }
    pub fn broken_count(&self) -> usize {
        self.fields
            .iter()
            .filter(|f| matches!(f.state, FieldState::Broken(_)))
            .count()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldState {
    /// Equals the dataclass default (or absent — same effect).
    Default,
    /// Present and different from the default.
    Set,
    /// Present but Python would misload or misuse it (wrong type,
    /// unknown enum value, out of enforced range).
    Broken(String),
}

#[derive(Clone, Debug)]
pub struct FieldView {
    pub key: &'static str,
    pub kind: FieldKind,
    /// Rendered value, pre-redacted for secrets. `—` for null.
    pub display: String,
    pub state: FieldState,
    /// Schema note and/or computed honesty note (effective-value rule,
    /// UNSAFE flags).
    pub note: Option<String>,
    /// For list-typed fields (fallback chains): the REAL entry count
    /// from the raw array — deriving it from the display string
    /// miscounts `org/model` ids (M2 review P2-6).
    pub list_len: Option<usize>,
}

/// sha256[:8] of the NORMALIZED value — Python's exact convention
/// (`normalize_api_key`, provider_profiles.py:80-84: trim, and any
/// case variant of `EMPTY` canonicalizes to `"EMPTY"` before hashing).
pub fn fingerprint(secret: &str) -> String {
    let trimmed = secret.trim();
    let canonical = if trimmed.eq_ignore_ascii_case("EMPTY") {
        "EMPTY"
    } else {
        trimmed
    };
    let mut h = Sha256::new();
    h.update(canonical.as_bytes());
    let hex = format!("{:x}", h.finalize());
    hex[..8].to_string()
}

/// Load the file at `path` into a `FileState`. This is the ONE reader;
/// it never writes and never mutates anything on disk.
pub fn load(path: &Path) -> FileState {
    load_with_raw(path).0
}

/// The write lane's reader: the folded state PLUS the raw value (for
/// RMW mutation bases and expectation checks). The raw value holds
/// secrets — it lives only on the worker thread and is dropped after
/// the write; nothing raw ever crosses to the UI.
pub fn load_with_raw(path: &Path) -> (FileState, Option<Value>) {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return (FileState::Missing, None)
        }
        Err(e) => {
            // A directory at the config path is a config mistake, not
            // a permissions problem — the hint must not send the
            // operator chmod-ing a directory (review P3-9).
            let error = if path.is_dir() {
                format!(
                    "{} is a DIRECTORY — ABSTRACTCORE_CONFIG_FILE must name a file \
                     (ABSTRACTCORE_CONFIG_DIR names the directory)",
                    path.display()
                )
            } else {
                e.to_string()
            };
            return (FileState::Unreadable { error }, None);
        }
    };
    let raw: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                FileState::Corrupt {
                    error: e.to_string(),
                    backups: list_backups(path),
                },
                None,
            )
        }
    };
    if !raw.is_object() {
        return (
            FileState::Corrupt {
                error: "the file parses as JSON but is not an object".into(),
                backups: list_backups(path),
            },
            None,
        );
    }
    let meta = std::fs::metadata(path).ok();
    let mut snap = fold(
        &raw,
        bytes.len() as u64,
        meta.as_ref().and_then(|m| m.modified().ok()),
        meta.as_ref().map(unix_mode),
    );
    snap.stamp = FileStamp::of(path);
    (FileState::Ready(snap), Some(raw))
}

#[cfg(unix)]
fn unix_mode(m: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    m.permissions().mode() & 0o777
}

#[cfg(not(unix))]
fn unix_mode(_m: &std::fs::Metadata) -> u32 {
    0o600
}

/// Recovery artifacts beside a corrupt file, newest first (timestamps
/// in the names sort lexicographically).
pub fn list_backups(path: &Path) -> Vec<String> {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name().and_then(|n| n.to_str()))
    else {
        return Vec::new();
    };
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| {
            n.starts_with(&format!("{name}.")) && (n.contains(".corrupt-") || n.contains(".bak"))
        })
        .collect();
    out.sort();
    out.reverse();
    out
}

/// Fold the raw object into the display model. Public so tests and the
/// M2 verify-by-re-read path can fold in-memory values.
pub fn fold(raw: &Value, bytes: u64, mtime: Option<SystemTime>, mode: Option<u32>) -> Snapshot {
    let obj = raw.as_object().expect("caller guarantees an object");
    let explicit = read_strategy_explicit(raw);

    let mut sections = Vec::with_capacity(schema::SECTIONS.len());
    for spec in schema::SECTIONS {
        let body = obj.get(spec.name);
        sections.push(fold_section(spec, body, explicit));
    }

    let known: Vec<&str> = schema::SECTIONS.iter().map(|s| s.name).collect();
    let mut unknown_sections: Vec<String> = obj
        .keys()
        .filter(|k| !known.contains(&k.as_str()) && k.as_str() != schema::META_FLAG)
        .cloned()
        .collect();
    unknown_sections.sort();

    // Only route entries Python would keep count as present: a route
    // must be a non-empty object ({} drops as unconfigured,
    // capability_defaults.py:302-304).
    let routes_in_file = obj
        .get("capability_defaults")
        .and_then(|c| c.get("routes"))
        .and_then(Value::as_object)
        .map(|r| {
            r.iter()
                .filter(|(_, v)| v.as_object().is_some_and(|o| !o.is_empty()))
                .map(|(k, _)| k.clone())
                .collect()
        })
        .unwrap_or_default();
    let profiles_in_file = profile_rows(obj.get("provider_profiles"))
        .map(|rows| rows.len())
        .unwrap_or(0);

    Snapshot {
        sections,
        unknown_sections,
        audio_strategy_explicit: explicit,
        routes_in_file,
        profiles_in_file,
        python_refusals: python_refusals(obj),
        bytes,
        mtime,
        mode,
        stamp: None,
    }
}

/// Python truthiness over JSON values — the semantics `bool(value)`
/// applies to hand-edited flags (review P3-4).
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Key-PRESENCE semantics exactly as manager.py:514-521: a present
/// top-level key decides (by truthiness — `"false"` is true, null is
/// false) and SUPPRESSES the nested legacy key; only an absent
/// top-level key consults `audio.strategy_explicit`.
fn read_strategy_explicit(raw: &Value) -> bool {
    if let Some(v) = raw.get(schema::META_FLAG) {
        return truthy(v);
    }
    if let Some(v) = raw.get("audio").and_then(|a| a.get("strategy_explicit")) {
        return truthy(v);
    }
    false
}

/// The profile ROWS as Python reads them (provider_profiles.py:232-243):
/// the `profiles` key when it is an object, else THE SECTION OBJECT
/// ITSELF (the `data.get("profiles", data)` fallback); a non-object at
/// either level is tolerated as empty.
fn profile_rows(section: Option<&Value>) -> Option<&serde_json::Map<String, Value>> {
    let section = section?.as_object()?;
    match section.get("profiles") {
        Some(p) => p.as_object(),
        None => Some(section),
    }
}

/// The 11 fields `ProviderProfile(**payload)` accepts — anything else
/// is a TypeError and a whole-file refusal (provider_profiles.py:224-229).
const PROFILE_FIELDS: &[&str] = &[
    "id",
    "display_name",
    "description",
    "provider_family",
    "base_url",
    "api_key",
    "api_key_env_var",
    "allowed_models",
    "enabled",
    // ONE STORE FOR PROVIDER CONFIG (ruling 2026-08-01,
    // provider_profiles.py:140-165): `scope` and `capabilities` are
    // real dataclass fields — the two columns a hosted Gateway needs on
    // the same row — and Python's own writer stamps them onto EVERY
    // profile it saves. Missing here, the console cried "Python REFUSES
    // this file" the instant the operator added their first connection
    // through it (caught live by the parity pty run). Core reads
    // neither, so nothing on this screen edits them.
    "scope",
    "capabilities",
    "created_at",
    "updated_at",
];

/// Shapes on which Python's `_dict_to_config` RAISES — sending the
/// whole file down the corrupt-fallback path (backup + defaults) on
/// every load. Everything else in the file is tolerated silently
/// (dataclass sections filter unknown keys and default non-objects;
/// capability_defaults skips bad entries). The ONE raising surface is
/// profile-row construction (adversarial review P1-1, each rule cited
/// to provider_profiles.py in the fold below).
fn python_refusals(obj: &serde_json::Map<String, Value>) -> Vec<String> {
    let mut out = Vec::new();
    let Some(rows) = profile_rows(obj.get("provider_profiles")) else {
        return out;
    };
    for (id, row) in rows {
        let at = format!("provider_profiles row \"{id}\"");
        let Some(row) = row.as_object() else {
            // provider_profiles.py:225-226 — "row must be an object".
            out.push(format!("{at}: not an object"));
            continue;
        };
        for key in row.keys() {
            if !PROFILE_FIELDS.contains(&key.as_str()) {
                // TypeError: unexpected keyword argument.
                out.push(format!("{at}: unknown field \"{key}\""));
            }
        }
        // normalize_profile_id (provider_profiles.py:38-49): strip an
        // optional endpoint: prefix, then non-empty + the id regex.
        let effective_id = row
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .trim()
            .trim_start_matches("endpoint:");
        if !valid_profile_id(effective_id) {
            out.push(format!("{at}: invalid profile id \"{effective_id}\""));
        }
        // normalize_provider_family (provider_profiles.py:63-68):
        // lowercased, _→-, must be one of the 8 families; absent/null
        // defaults to openai-compatible.
        if let Some(fam) = row.get("provider_family") {
            let norm = fam
                .as_str()
                .map(|s| s.trim().to_ascii_lowercase().replace('_', "-"))
                .unwrap_or_default();
            let effective = if norm.is_empty() {
                "openai-compatible".to_string()
            } else {
                norm
            };
            if !schema::PROFILE_FAMILIES.contains(&effective.as_str()) {
                out.push(format!("{at}: unsupported provider family \"{effective}\""));
            }
        }
        // normalize_base_url (provider_profiles.py:71-77): non-empty
        // must start http(s)://.
        if let Some(url) = row.get("base_url") {
            let s = url.as_str().map(str::trim).unwrap_or("");
            let ok = s.is_empty() || s.starts_with("http://") || s.starts_with("https://");
            if !ok {
                out.push(format!("{at}: base_url must start with http:// or https://"));
            }
        }
        // normalize_api_key_env_var (provider_profiles.py:87-93).
        if let Some(ev) = row.get("api_key_env_var") {
            let s = ev.as_str().map(str::trim).unwrap_or("");
            if !s.is_empty() && !valid_env_var_name(s) {
                out.push(format!("{at}: api_key_env_var is not a valid env var name"));
            }
        }
    }
    out
}

/// `^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$` (provider_profiles.py:19).
fn valid_profile_id(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    if s.chars().count() > 96 {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// `^[A-Za-z_][A-Za-z0-9_]*$` (provider_profiles.py:21).
fn valid_env_var_name(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn fold_section(spec: &'static SectionSpec, body: Option<&Value>, explicit: bool) -> SectionView {
    let present_in_file = body.is_some();
    let body_obj = body.and_then(Value::as_object);

    // A present section that is not an object: Python TOLERATES it —
    // `_filter_dataclass_kwargs` returns {} for any non-dict
    // (manager.py:368-372), so the whole section silently runs on
    // defaults. Still broken config (the operator's values are not in
    // effect), but the reason must say what Python actually does.
    let section_broken = body.is_some() && body_obj.is_none();

    let mut fields = Vec::with_capacity(spec.fields.len());
    for fs in spec.fields {
        let val = body_obj.and_then(|o| o.get(fs.key));
        let mut view = fold_field(fs, val);
        if section_broken {
            view.state = FieldState::Broken(
                "section is not a JSON object — Python ignores it (defaults apply)".into(),
            );
        }
        if spec.name == "audio" && fs.key == "strategy" {
            attach_audio_effective_note(&mut view, val, explicit);
        }
        fields.push(view);
    }

    let mut unknown_keys: Vec<String> = body_obj
        .map(|o| {
            o.keys()
                .filter(|k| {
                    let known_field = spec.fields.iter().any(|f| f.key == k.as_str());
                    // The legacy nested meta flag is known, just not a field.
                    let legacy_meta = spec.name == "audio" && k.as_str() == "strategy_explicit";
                    !known_field && !legacy_meta
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    unknown_keys.sort();

    SectionView {
        spec,
        present_in_file,
        fields,
        unknown_keys,
    }
}

fn fold_field(fs: &'static crate::schema::FieldSpec, val: Option<&Value>) -> FieldView {
    let (display, state) = match val {
        None => (render_value(&fs.kind, &Value::Null, true), FieldState::Default),
        Some(v) => match schema::validate(&fs.kind, v) {
            Err(reason) => (render_value(&fs.kind, v, false), FieldState::Broken(reason)),
            Ok(()) => {
                // Empty/whitespace secrets ARE not-set to Python
                // (truthiness at injection and status,
                // manager.py:437, 933-942) — `--set-api-key P ''`
                // stores `""`; classifying it Set while displaying
                // "not set" was a self-contradicting row (review P2-4).
                let effectively_default = fs.default.matches(v)
                    || (matches!(fs.kind, FieldKind::Secret)
                        && v.as_str().map(str::trim) == Some(""));
                let state = if effectively_default {
                    FieldState::Default
                } else {
                    FieldState::Set
                };
                (render_value(&fs.kind, v, false), state)
            }
        },
    };
    let mut note = fs.note.map(str::to_string);
    // Safety flags are honest only when they fire.
    if let Some(n) = &note {
        if n.starts_with("UNSAFE") && val.and_then(Value::as_bool) != Some(true) {
            note = None;
        }
    }
    let list_len = if matches!(fs.kind, FieldKind::FallbackChain) {
        Some(val.and_then(Value::as_array).map(Vec::len).unwrap_or(0))
    } else {
        None
    };
    FieldView {
        key: fs.key,
        kind: fs.kind,
        display,
        state,
        note,
        list_len,
    }
}

/// Render one value for display. Secrets never render their content —
/// only presence + fingerprint. `absent` renders the dataclass default
/// the runtime will use.
fn render_value(kind: &FieldKind, v: &Value, absent: bool) -> String {
    if matches!(kind, FieldKind::Secret) {
        return match v.as_str().map(str::trim) {
            Some(s) if !s.is_empty() => format!("set · fp {}", fingerprint(s)),
            _ => "not set".into(),
        };
    }
    if absent || v.is_null() {
        return "—".into();
    }
    match v {
        Value::String(s) if s.is_empty() => "\"\"".into(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(items) if matches!(kind, FieldKind::FallbackChain) => {
            if items.is_empty() {
                "[]".into()
            } else {
                let pairs: Vec<String> = items
                    .iter()
                    .map(|it| {
                        format!(
                            "{}/{}",
                            it.get("provider").and_then(Value::as_str).unwrap_or("?"),
                            it.get("model").and_then(Value::as_str).unwrap_or("?")
                        )
                    })
                    .collect();
                format!("[{}]", pairs.join(", "))
            }
        }
        other => other.to_string(),
    }
}

/// The non-persisted smart default (manager.py:380-403): unless the
/// meta flag is true, Python rewrites audio.strategy IN MEMORY at every
/// load — with abstractvoice installed `""|native_only|native|disabled`
/// become `auto`; without it `auto|speech_to_text|stt` become
/// `native_only`. The console cannot know whether abstractvoice is
/// importable, so it shows BOTH projections honestly.
fn attach_audio_effective_note(view: &mut FieldView, val: Option<&Value>, explicit: bool) {
    if explicit {
        view.note = Some("explicit — used as-is at load".into());
        return;
    }
    let on_disk = val.and_then(Value::as_str).unwrap_or("auto");
    let with_voice = match on_disk {
        "" | "native_only" | "native" | "disabled" => "auto",
        other => other,
    };
    let without_voice = match on_disk {
        "auto" | "speech_to_text" | "stt" => "native_only",
        other => other,
    };
    view.note = Some(format!(
        "not explicit — effective: {with_voice} (abstractvoice installed) / {without_voice} (not installed)"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SectionKind;
    use serde_json::json;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn path_resolution_matches_python_precedence() {
        let home = Path::new("/home/u");
        let p = resolve_config_path(&env_of(&[]), home);
        assert_eq!(
            p.path,
            PathBuf::from("/home/u/.abstractcore/config/abstractcore.json")
        );
        assert_eq!(p.source, PathSource::Default);

        let p = resolve_config_path(&env_of(&[("ABSTRACTCORE_CONFIG_DIR", "/etc/ac")]), home);
        assert_eq!(p.path, PathBuf::from("/etc/ac/abstractcore.json"));
        assert_eq!(p.source, PathSource::EnvDir);

        // FILE wins over DIR; only the EMPTY string is unset.
        let p = resolve_config_path(
            &env_of(&[
                ("ABSTRACTCORE_CONFIG_FILE", "/tmp/x.json"),
                ("ABSTRACTCORE_CONFIG_DIR", "/etc/ac"),
            ]),
            home,
        );
        assert_eq!(p.path, PathBuf::from("/tmp/x.json"));
        assert_eq!(p.source, PathSource::EnvFile);
        let p = resolve_config_path(&env_of(&[("ABSTRACTCORE_CONFIG_FILE", "")]), home);
        assert_eq!(p.source, PathSource::Default);

        // Python parity (review P2-3): `~` expands on every branch;
        // whitespace is truthy — a real (weird) path, not unset.
        let p = resolve_config_path(&env_of(&[("ABSTRACTCORE_CONFIG_FILE", "~/x.json")]), home);
        assert_eq!(p.path, PathBuf::from("/home/u/x.json"));
        let p = resolve_config_path(&env_of(&[("ABSTRACTCORE_CONFIG_DIR", "~/cfg")]), home);
        assert_eq!(p.path, PathBuf::from("/home/u/cfg/abstractcore.json"));
        let p = resolve_config_path(&env_of(&[("ABSTRACTCORE_CONFIG_FILE", "   ")]), home);
        assert_eq!(p.path, PathBuf::from("   "), "whitespace is a real path to Python");
    }

    #[test]
    fn fingerprint_matches_the_python_convention() {
        // hashlib.sha256("sk-test".encode()).hexdigest()[:8] — computed
        // via the framework venv 2026-07-25.
        assert_eq!(fingerprint("sk-test"), "f3abf2a6");
        assert_eq!(fingerprint("  sk-test  "), "f3abf2a6", "trimmed first");
        // normalize_api_key canonicalizes any case of EMPTY (review
        // P3-1): sha256("EMPTY")[:8] both sides.
        assert_eq!(fingerprint("EMPTY"), fingerprint("empty"));
        assert_eq!(fingerprint("Empty"), fingerprint(" EMPTY "));
        assert_ne!(fingerprint("EMPTY"), fingerprint("emptyish"));
    }

    /// The P1-1 model: exactly the shapes ProviderProfile construction
    /// raises on — and nothing else — count as Python refusals.
    #[test]
    fn python_refusals_match_the_profile_raise_paths() {
        let refusals = |v: &Value| fold(v, 0, None, None).python_refusals;

        // A clean profile: no refusals.
        let clean = json!({"provider_profiles": {"profiles": {
            "ok-id": {"id": "ok-id", "provider_family": "openai-compatible",
                       "base_url": "https://x/v1", "api_key_env_var": "MY_KEY"}}}});
        assert!(refusals(&clean).is_empty());

        // Unknown field in a row → TypeError → whole-file refusal.
        let unknown = json!({"provider_profiles": {"profiles": {
            "p": {"id": "p", "future_field": true}}}});
        let r = refusals(&unknown);
        assert_eq!(r.len(), 1, "{r:?}");
        assert!(r[0].contains("unknown field \"future_field\""), "{r:?}");

        // …but the row PYTHON ITSELF WRITES is not a refusal. Every
        // `config set-provider` stamps `scope` + `capabilities` onto
        // every profile in the file (ONE STORE FOR PROVIDER CONFIG,
        // provider_profiles.py:140-165) — reading those as unknown
        // fields made the console cry "Python REFUSES this file" about
        // a file Python had just saved, one keystroke after the
        // operator added their first connection.
        let written_by_python = json!({"provider_profiles": {"profiles": {
            "paritytest": {
                "id": "paritytest", "display_name": "paritytest", "description": "",
                "provider_family": "openai-compatible",
                "base_url": "http://127.0.0.1:1234/v1", "api_key": "",
                "api_key_env_var": "", "allowed_models": [], "enabled": true,
                "scope": "gateway", "capabilities": ["text"],
                "created_at": "2026-08-01T19:04:51Z",
                "updated_at": "2026-08-01T19:04:51Z"}}}});
        assert!(
            refusals(&written_by_python).is_empty(),
            "{:?}",
            refusals(&written_by_python)
        );

        // Non-dict row; invalid family; non-http base_url; bad env var.
        let bad = json!({"provider_profiles": {"profiles": {
            "s": "not-an-object",
            "f": {"id": "f", "provider_family": "huggingface"},
            "u": {"id": "u", "base_url": "localhost:1234"},
            "e": {"id": "e", "api_key_env_var": "9BAD NAME"},
            "i": {"id": "***"}
        }}});
        let r = refusals(&bad);
        assert!(r.iter().any(|m| m.contains("\"s\": not an object")), "{r:?}");
        assert!(r.iter().any(|m| m.contains("unsupported provider family")), "{r:?}");
        assert!(r.iter().any(|m| m.contains("must start with http")), "{r:?}");
        assert!(r.iter().any(|m| m.contains("env var name")), "{r:?}");
        assert!(r.iter().any(|m| m.contains("invalid profile id")), "{r:?}");

        // The `data.get("profiles", data)` quirk: with the profiles
        // key ABSENT, the section object itself is the rows dict — a
        // stray scalar entry is a refusing "row".
        let quirk = json!({"provider_profiles": {"version": 1}});
        let r = refusals(&quirk);
        assert!(
            r.iter().any(|m| m.contains("\"version\": not an object")),
            "{r:?}"
        );

        // Tolerated shapes: non-object section, non-object profiles key.
        assert!(refusals(&json!({"provider_profiles": "junk"})).is_empty());
        assert!(refusals(&json!({"provider_profiles": {"profiles": 3}})).is_empty());
        // A family alias Python normalizes fine (case + underscore).
        let alias = json!({"provider_profiles": {"profiles": {
            "a": {"id": "a", "provider_family": "OpenAI_Compatible"}}}});
        assert!(refusals(&alias).is_empty());
        // An endpoint:-prefixed id normalizes fine.
        let pfx = json!({"provider_profiles": {"profiles": {
            "b": {"id": "endpoint:b"}}}});
        assert!(refusals(&pfx).is_empty());
    }

    /// Empty/whitespace secrets are not-set to Python — the state must
    /// agree with the display (review P2-4).
    #[test]
    fn empty_string_secret_classifies_as_default() {
        let raw = json!({"api_keys": {"openai": "", "vllm": "   "},
                          "server": {"auth_token": ""}});
        let snap = fold(&raw, 0, None, None);
        let keys = snap.sections.iter().find(|s| s.spec.name == "api_keys").unwrap();
        for k in ["openai", "vllm"] {
            let f = keys.fields.iter().find(|f| f.key == k).unwrap();
            assert_eq!(f.state, FieldState::Default, "{k} is not set");
            assert_eq!(f.display, "not set");
        }
        assert_eq!(keys.set_count(), 0);
    }

    /// Presence-then-truthiness, top-level suppressing nested — the
    /// exact manager.py:514-521 semantics (review P3-4).
    #[test]
    fn strategy_explicit_python_semantics() {
        let f = |v: Value| fold(&v, 0, None, None).audio_strategy_explicit;
        assert!(f(json!({"audio_strategy_explicit": "false"})), "strings are truthy");
        assert!(!f(json!({"audio_strategy_explicit": null})), "null is falsy");
        assert!(
            !f(json!({"audio_strategy_explicit": null,
                       "audio": {"strategy_explicit": true}})),
            "a PRESENT top-level key suppresses the nested one"
        );
        assert!(f(json!({"audio": {"strategy_explicit": 1}})));
        assert!(!f(json!({})));
    }

    /// Empty route objects drop as unconfigured on Python's load —
    /// the file-lane count must not claim them (review P3-5).
    #[test]
    fn empty_route_objects_do_not_count()  {
        let raw = json!({"capability_defaults": {"routes": {
            "input.text": {"provider": "x"},
            "input.voice": {},
            "output.music": "junk"
        }}});
        let snap = fold(&raw, 0, None, None);
        assert_eq!(snap.routes_in_file, vec!["input.text".to_string()]);
    }

    #[test]
    fn secrets_never_reach_the_display_model() {
        let raw = json!({
            "api_keys": {"openai": "sk-verysecret123", "anthropic": null},
            "server": {"auth_token": "tok-secret"},
        });
        let snap = fold(&raw, 0, None, None);
        let rendered = format!("{snap:?}");
        assert!(!rendered.contains("verysecret"), "no key material anywhere");
        assert!(!rendered.contains("tok-secret"));
        let keys = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "api_keys")
            .unwrap();
        let openai = keys.fields.iter().find(|f| f.key == "openai").unwrap();
        assert!(openai.display.starts_with("set · fp "), "{}", openai.display);
        assert_eq!(openai.state, FieldState::Set);
        let anthropic = keys.fields.iter().find(|f| f.key == "anthropic").unwrap();
        assert_eq!(anthropic.display, "not set");
        assert_eq!(anthropic.state, FieldState::Default);
    }

    #[test]
    fn set_default_broken_classification() {
        let raw = json!({
            "video": {
                "strategy": "auto",          // default
                "max_frames": 12,            // set
                "frame_format": "webp",     // broken: not a choice
                "max_frame_side": "big",    // broken: wrong type
            },
            "logging": {"console_level": "INFO"},
        });
        let snap = fold(&raw, 0, None, None);
        let video = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "video")
            .unwrap();
        let by_key = |k: &str| video.fields.iter().find(|f| f.key == k).unwrap();
        assert_eq!(by_key("strategy").state, FieldState::Default);
        assert_eq!(by_key("max_frames").state, FieldState::Set);
        assert!(matches!(by_key("frame_format").state, FieldState::Broken(_)));
        assert!(matches!(by_key("max_frame_side").state, FieldState::Broken(_)));
        // Absent fields are defaults, absent sections wholly default.
        assert_eq!(by_key("max_frames_native").state, FieldState::Default);
        let audio = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "audio")
            .unwrap();
        assert!(!audio.present_in_file);
        assert_eq!(audio.set_count(), 0);
        assert_eq!(video.set_count(), 1);
        assert_eq!(video.broken_count(), 2);
    }

    #[test]
    fn unknown_sections_and_keys_are_surfaced_not_dropped() {
        let raw = json!({
            "future_section": {"x": 1},
            "video": {"strategy": "auto", "brand_new_knob": true},
        });
        let snap = fold(&raw, 0, None, None);
        assert_eq!(snap.unknown_sections, vec!["future_section".to_string()]);
        let video = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "video")
            .unwrap();
        assert_eq!(video.unknown_keys, vec!["brand_new_knob".to_string()]);
    }

    #[test]
    fn audio_effective_note_tells_both_projections() {
        // Not explicit, on disk "auto": with voice stays auto, without
        // becomes native_only.
        let raw = json!({"audio": {"strategy": "auto"}});
        let snap = fold(&raw, 0, None, None);
        let audio = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "audio")
            .unwrap();
        let strat = audio.fields.iter().find(|f| f.key == "strategy").unwrap();
        let note = strat.note.as_deref().unwrap();
        assert!(note.contains("auto (abstractvoice installed)"), "{note}");
        assert!(note.contains("native_only (not installed)"), "{note}");

        // Explicit (top-level flag): used as-is.
        let raw = json!({"audio": {"strategy": "auto"}, "audio_strategy_explicit": true});
        let snap = fold(&raw, 0, None, None);
        assert!(snap.audio_strategy_explicit);
        let audio = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "audio")
            .unwrap();
        let strat = audio.fields.iter().find(|f| f.key == "strategy").unwrap();
        assert_eq!(strat.note.as_deref(), Some("explicit — used as-is at load"));

        // Legacy nested spelling is accepted, and is NOT an unknown key.
        let raw = json!({"audio": {"strategy": "stt", "strategy_explicit": true}});
        let snap = fold(&raw, 0, None, None);
        assert!(snap.audio_strategy_explicit);
        let audio = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "audio")
            .unwrap();
        assert!(audio.unknown_keys.is_empty());
    }

    #[test]
    fn corrupt_and_missing_files_are_distinct_states() {
        let dir = std::env::temp_dir().join(format!("acc-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("abstractcore.json");

        assert!(matches!(load(&file), FileState::Missing));

        std::fs::write(&file, b"{ not json").unwrap();
        std::fs::write(dir.join("abstractcore.json.corrupt-20260101-000000.bak"), b"x").unwrap();
        std::fs::write(dir.join("abstractcore.json.bak-repair-101010"), b"x").unwrap();
        match load(&file) {
            FileState::Corrupt { backups, .. } => {
                assert_eq!(backups.len(), 2, "both backup shapes listed: {backups:?}");
            }
            other => panic!("expected corrupt, got {other:?}"),
        }

        // Valid JSON that is NOT an object: the one corrupt shape
        // where console and Python agree exactly (review audit #4).
        std::fs::write(&file, b"[1, 2, 3]").unwrap();
        assert!(
            matches!(load(&file), FileState::Corrupt { .. }),
            "non-object JSON is corrupt"
        );

        std::fs::write(&file, br#"{"video": {"max_frames": 5}}"#).unwrap();
        match load(&file) {
            FileState::Ready(snap) => {
                assert!(snap.bytes > 0);
                let video = snap
                    .sections
                    .iter()
                    .find(|s| s.spec.name == "video")
                    .unwrap();
                assert_eq!(video.set_count(), 1);
            }
            other => panic!("expected ready, got {other:?}"),
        }

        // A DIRECTORY at the config path: unreadable with a hint that
        // names the actual mistake, not permissions (review P3-9).
        let as_dir = dir.join("iam-a-dir");
        std::fs::create_dir_all(&as_dir).unwrap();
        match load(&as_dir) {
            FileState::Unreadable { error } => {
                assert!(error.contains("DIRECTORY"), "{error}");
            }
            other => panic!("expected unreadable, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn routes_and_profiles_summarized_from_file() {
        let raw = json!({
            "capability_defaults": {"version": 1, "routes": {
                "input.text": {"provider": "lmstudio", "model": "m"},
                "output.voice": {"options": {"voice": "M2"}}
            }},
            "provider_profiles": {"profiles": {"a": {"id": "a"}, "b": {"id": "b"}}},
        });
        let snap = fold(&raw, 0, None, None);
        assert_eq!(snap.routes_in_file.len(), 2);
        assert_eq!(snap.profiles_in_file, 2);
        // Non-scalar sections carry no field rows.
        let routes = snap
            .sections
            .iter()
            .find(|s| s.spec.name == "capability_defaults")
            .unwrap();
        assert_eq!(routes.spec.kind, SectionKind::Routes);
        assert!(routes.fields.is_empty());
    }
}
