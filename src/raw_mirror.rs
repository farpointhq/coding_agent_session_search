use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const RAW_MIRROR_ROOT_DIR: &str = "raw-mirror";
const RAW_MIRROR_VERSION_DIR: &str = "v1";
const RAW_MIRROR_HASH_ALGORITHM: &str = "blake3";
const RAW_MIRROR_BLOB_EXTENSION: &str = "raw";

// NOTE: The raw-mirror WRITE PATH has been removed. `cass` no longer mirrors
// indexed source files into a content-addressed blob store: the whole-file
// re-copy on every index pass caused unbounded disk growth (users hit ~1TB).
// The search index (SQLite + Tantivy) is built by PARSING source files directly
// and never reads this mirror. What remains here is:
//   * the read-side manifest/path helpers (forensic side-archive readers in
//     doctor / evidence_bundle / crash_replay still understand the schema), and
//   * `purge_raw_mirror_root`, which unconditionally reclaims any leftover
//     `raw-mirror/` store on the next index pass so oversized stores shrink.

// ---------------------------------------------------------------------------
// Read-side manifest schema (forensic side-archive; never written anymore).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RawMirrorDbLink {
    pub conversation_id: Option<i64>,
    pub message_count: Option<usize>,
    pub source_path: Option<String>,
    pub started_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawMirrorCompressionEnvelope {
    state: String,
    algorithm: Option<String>,
    uncompressed_size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawMirrorEncryptionEnvelope {
    state: String,
    algorithm: Option<String>,
    key_id: Option<String>,
    envelope_version: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawMirrorVerificationRecord {
    status: String,
    verifier: String,
    content_blake3: Option<String>,
    verified_at_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawMirrorManifestFile {
    schema_version: u32,
    manifest_kind: String,
    manifest_id: String,
    blob_hash_algorithm: String,
    blob_relative_path: String,
    blob_blake3: String,
    blob_size_bytes: u64,
    provider: String,
    source_id: String,
    origin_kind: String,
    origin_host: Option<String>,
    original_path: String,
    redacted_original_path: String,
    original_path_blake3: String,
    captured_at_ms: i64,
    source_mtime_ms: Option<i64>,
    source_size_bytes: u64,
    compression: RawMirrorCompressionEnvelope,
    encryption: RawMirrorEncryptionEnvelope,
    db_links: Vec<RawMirrorDbLink>,
    verification: RawMirrorVerificationRecord,
    manifest_blake3: Option<String>,
}

// ---------------------------------------------------------------------------
// Purge: replaces the old end-of-pass retention sweep.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub(crate) struct RawMirrorPurgeStats {
    pub removed: bool,
    pub bytes_reclaimed: u64,
    pub entries_removed: u64,
    pub dry_run: bool,
}

/// Best-effort, lock-guarded removal of the deprecated raw-mirror store.
/// Path-contained to `raw_mirror_root(data_dir)`; NEVER touches the sibling
/// `agent_search.db` or the Tantivy index. Idempotent (absent dir -> no-op).
///
/// The write path is gone, so the mirror is pure leftover disk: every pass
/// reclaims it. A mutating purge (`!dry_run`) holds the index-run lock for its
/// whole duration and is *skipped* (not failed) if another run owns it, so it
/// never races a concurrent pass; the next pass reclaims. Dry-run only tallies.
pub(crate) fn purge_raw_mirror_root(data_dir: &Path, dry_run: bool) -> Result<RawMirrorPurgeStats> {
    let mut stats = RawMirrorPurgeStats {
        dry_run,
        ..Default::default()
    };

    let root = raw_mirror_root(data_dir);
    if !root.exists() {
        return Ok(stats);
    }

    // Containment hardening: refuse to delete anything that is not the canonical
    // raw-mirror root under `data_dir`. Re-derive the expected root from the same
    // helper and require an exact match before any removal. This makes a hostile
    // or surprising `data_dir` (e.g. one engineered so `root` resolves elsewhere)
    // a hard error rather than a deletion.
    assert_raw_mirror_root_contained(data_dir, &root)?;

    // Tally bytes/entries for the stats (and to decide dry-run reporting).
    let (bytes, entries) = tally_dir_best_effort(&root);
    stats.bytes_reclaimed = bytes;
    stats.entries_removed = entries;

    if dry_run {
        // Read-only: report what would be reclaimed, delete nothing.
        return Ok(stats);
    }

    // Hold the index-run lock for the duration so a concurrent index/watch run
    // can never observe a half-deleted store. If another run owns it, skip (the
    // next idle pass reclaims). `_lock` is kept bound (not `_`) so the flock is
    // held until the function returns.
    let _lock = match try_hold_index_run_lock(data_dir) {
        Some(file) => file,
        None => {
            tracing::info!(
                data_dir = %data_dir.display(),
                "skipping raw-mirror purge: an index run is active"
            );
            // Nothing removed; reset the tally so the stats reflect reality.
            stats.bytes_reclaimed = 0;
            stats.entries_removed = 0;
            return Ok(stats);
        }
    };

    match fs::remove_dir_all(&root) {
        Ok(()) => {
            stats.removed = true;
            tracing::info!(
                root = %root.display(),
                freed_bytes = stats.bytes_reclaimed,
                entries = stats.entries_removed,
                "purged deprecated raw-mirror store"
            );
        }
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Raced away between tally and remove; treat as reclaimed.
            stats.removed = true;
        }
        Err(err) => {
            // Best-effort: a purge failure must not fail the index pass. Leave the
            // store on disk; the next pass retries.
            stats.bytes_reclaimed = 0;
            stats.entries_removed = 0;
            tracing::warn!(
                error = %err,
                root = %root.display(),
                "failed to purge raw-mirror store; leaving on disk for next pass"
            );
        }
    }

    Ok(stats)
}

/// Refuse to operate on any path that is not the canonical raw-mirror root under
/// `data_dir`. The root must end in `raw-mirror/v1` and equal the re-derived
/// `raw_mirror_root(data_dir)`. Errors (never silently proceeds) on mismatch.
fn assert_raw_mirror_root_contained(data_dir: &Path, root: &Path) -> Result<()> {
    let expected = raw_mirror_root(data_dir);
    if root != expected {
        return Err(anyhow!(
            "refusing to purge raw mirror: resolved path {} is not the canonical root {}",
            root.display(),
            expected.display()
        ));
    }
    // Belt-and-suspenders: the canonical root must literally end in
    // `<RAW_MIRROR_ROOT_DIR>/<RAW_MIRROR_VERSION_DIR>`.
    let ends_in_canonical = root.file_name() == Some(std::ffi::OsStr::new(RAW_MIRROR_VERSION_DIR))
        && root
            .parent()
            .and_then(|p| p.file_name())
            == Some(std::ffi::OsStr::new(RAW_MIRROR_ROOT_DIR));
    if !ends_in_canonical {
        return Err(anyhow!(
            "refusing to purge raw mirror: path {} does not end in {}/{}",
            root.display(),
            RAW_MIRROR_ROOT_DIR,
            RAW_MIRROR_VERSION_DIR
        ));
    }
    Ok(())
}

/// Best-effort walk of a directory tree, summing on-disk file bytes and counting
/// every entry (files + dirs). Unreadable entries contribute 0 rather than
/// aborting the tally.
fn tally_dir_best_effort(path: &Path) -> (u64, u64) {
    let read = match fs::read_dir(path) {
        Ok(read) => read,
        Err(_) => return (0, 0),
    };
    let mut bytes = 0u64;
    let mut entries = 0u64;
    for entry in read.flatten() {
        entries = entries.saturating_add(1);
        let child = entry.path();
        match entry.file_type() {
            Ok(t) if t.is_dir() => {
                let (b, e) = tally_dir_best_effort(&child);
                bytes = bytes.saturating_add(b);
                entries = entries.saturating_add(e);
            }
            Ok(_) => {
                if let Ok(md) = fs::symlink_metadata(&child) {
                    bytes = bytes.saturating_add(md.len());
                }
            }
            Err(_) => {}
        }
    }
    (bytes, entries)
}

/// Hold the index-run lock for the duration of a mutating operation so it can
/// never race a concurrent index/watch run. Returns the held lock file (keep it
/// alive to keep the lock) on success, or `None` if another run currently owns it.
fn try_hold_index_run_lock(data_dir: &Path) -> Option<File> {
    let lock_path = data_dir.join("index-run.lock");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .ok()?;
    match fs2::FileExt::try_lock_exclusive(&file) {
        Ok(()) => Some(file),
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Read-side helpers (forensic readers still understand the schema). These are
// retained intentionally even where currently unreferenced: doctor /
// evidence_bundle / crash_replay reporting reads manifests/blobs and these are
// the canonical path/id/hash derivations the schema is defined by.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn read_raw_mirror_manifest(path: &Path) -> Result<RawMirrorManifestFile> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("stat raw mirror manifest {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(anyhow!(
            "refusing to read symlink raw mirror manifest {}",
            path.display()
        ));
    }
    if !metadata.is_file() {
        return Err(anyhow!(
            "refusing to read non-file raw mirror manifest {}",
            path.display()
        ));
    }
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read raw mirror manifest {}", path.display()))?,
    )
    .with_context(|| format!("parse raw mirror manifest {}", path.display()))
}

fn raw_mirror_root(data_dir: &Path) -> PathBuf {
    data_dir
        .join(RAW_MIRROR_ROOT_DIR)
        .join(RAW_MIRROR_VERSION_DIR)
}

#[allow(dead_code)]
fn raw_mirror_blob_relative_path(blob_blake3: &str) -> Option<String> {
    if blob_blake3.len() != 64 || !blob_blake3.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let lower = blob_blake3.to_ascii_lowercase();
    Some(format!(
        "blobs/{}/{}/{}.{}",
        RAW_MIRROR_HASH_ALGORITHM,
        &lower[..2],
        lower,
        RAW_MIRROR_BLOB_EXTENSION
    ))
}

#[allow(dead_code)]
fn raw_mirror_manifest_relative_path(manifest_id: &str) -> String {
    format!("manifests/{manifest_id}.json")
}

#[allow(dead_code)]
fn raw_mirror_manifest_path_from_relative(root: &Path, relative_path: &str) -> Result<PathBuf> {
    let relative = Path::new(relative_path);
    if relative.is_absolute() {
        return Err(anyhow!(
            "raw mirror manifest path must be relative: {relative_path}"
        ));
    }

    let mut normal_components = Vec::new();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(part) => normal_components.push(part),
            _ => {
                return Err(anyhow!(
                    "raw mirror manifest path must use only normal relative components: {relative_path}"
                ));
            }
        }
    }

    if normal_components.len() != 2
        || normal_components[0] != std::ffi::OsStr::new("manifests")
        || Path::new(normal_components[1])
            .extension()
            .and_then(|ext| ext.to_str())
            != Some("json")
    {
        return Err(anyhow!(
            "raw mirror manifest path must match manifests/<id>.json: {relative_path}"
        ));
    }

    Ok(root.join(relative))
}

#[allow(dead_code)]
fn raw_mirror_original_path_blake3(original_path: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"doctor-raw-mirror-original-path-v1");
    hasher.update(&[0]);
    hasher.update(original_path.as_bytes());
    hasher.finalize().to_hex().to_string()
}

#[allow(dead_code)]
fn raw_mirror_manifest_id(
    provider: &str,
    source_id: &str,
    origin_kind: &str,
    origin_host: Option<&str>,
    original_path_blake3: &str,
    blob_blake3: &str,
) -> String {
    canonical_blake3(
        "doctor-raw-mirror-manifest-id-v1",
        json!({
            "provider": provider,
            "source_id": source_id,
            "origin_kind": origin_kind,
            "origin_host": origin_host,
            "original_path_blake3": original_path_blake3,
            "blob_blake3": blob_blake3,
        }),
    )
}

#[allow(dead_code)]
fn raw_mirror_manifest_blake3(manifest: &RawMirrorManifestFile) -> String {
    let mut value = serde_json::to_value(manifest).unwrap_or_default();
    if let Value::Object(map) = &mut value {
        map.remove("manifest_blake3");
    }
    canonical_blake3("doctor-raw-mirror-manifest-v1", value)
}

#[allow(dead_code)]
fn canonical_blake3(prefix: &str, value: Value) -> String {
    let encoded = serde_json::to_vec(&canonical_json_value(value)).unwrap_or_default();
    let mut hasher = blake3::Hasher::new();
    hasher.update(prefix.as_bytes());
    hasher.update(&[0]);
    hasher.update(&encoded);
    format!("{prefix}-{}", hasher.finalize().to_hex())
}

#[allow(dead_code)]
fn canonical_json_value(value: Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(canonical_json_value).collect()),
        Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut canonical = serde_json::Map::new();
            for (key, value) in entries {
                canonical.insert(key, canonical_json_value(value));
            }
            Value::Object(canonical)
        }
        other => other,
    }
}

#[allow(dead_code)]
fn redacted_original_path(provider: &str, source_path: &Path) -> String {
    let file_name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("session");
    format!("[{provider}]/{file_name}")
}

#[allow(dead_code)]
fn file_blake3(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[allow(dead_code)]
fn ensure_private_dir(path: &Path) -> Result<()> {
    create_private_dir_all(path)
        .with_context(|| format!("create raw mirror dir {}", path.display()))?;
    set_private_dir_permissions(path)?;
    Ok(())
}

#[cfg(unix)]
fn create_private_dir_all(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    let mut builder = fs::DirBuilder::new();
    builder.recursive(true).mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_dir_all(path: &Path) -> std::io::Result<()> {
    fs::create_dir_all(path)
}

#[allow(dead_code)]
fn now_ms() -> i64 {
    system_time_to_ms(SystemTime::now()).unwrap_or(0)
}

#[allow(dead_code)]
fn system_time_to_ms(time: SystemTime) -> Option<i64> {
    time.duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
}

#[cfg(unix)]
fn set_private_dir_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("set raw mirror dir permissions {}", path.display()))
}

#[cfg(not(unix))]
fn set_private_dir_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_raw_mirror_store(data_dir: &Path) -> (PathBuf, u64) {
        let root = raw_mirror_root(data_dir);
        let blob_dir = root.join("blobs/blake3/ab");
        fs::create_dir_all(&blob_dir).unwrap();
        let blob = blob_dir.join(format!("{}.raw", "a".repeat(64)));
        let bytes = vec![0u8; 4096];
        fs::write(&blob, &bytes).unwrap();
        fs::create_dir_all(root.join("manifests")).unwrap();
        fs::write(root.join("manifests").join("m.json"), b"{}").unwrap();
        (root, bytes.len() as u64)
    }

    #[test]
    fn purge_removes_root_and_is_idempotent() {
        let temp = tempfile::TempDir::new().expect("tempdir");
        let data_dir = temp.path().join("cass-data");
        let (root, _) = seed_raw_mirror_store(&data_dir);
        assert!(root.exists());

        let stats = purge_raw_mirror_root(&data_dir, false).expect("purge");
        assert!(stats.removed, "store removed");
        assert!(!stats.dry_run);
        assert!(!root.exists(), "raw-mirror root gone after purge");
        assert!(stats.bytes_reclaimed >= 4096);
        assert!(stats.entries_removed > 0);

        // Idempotent: a second purge is a clean no-op.
        let again = purge_raw_mirror_root(&data_dir, false).expect("re-purge");
        assert!(!again.removed);
        assert_eq!(again.bytes_reclaimed, 0);
        assert_eq!(again.entries_removed, 0);
    }

    #[test]
    fn purge_dry_run_is_non_destructive() {
        let temp = tempfile::TempDir::new().expect("tempdir");
        let data_dir = temp.path().join("cass-data");
        let (root, _) = seed_raw_mirror_store(&data_dir);

        let stats = purge_raw_mirror_root(&data_dir, true).expect("dry-run purge");
        assert!(stats.dry_run);
        assert!(!stats.removed, "dry run never removes");
        assert!(root.exists(), "dry run must not delete the store");
        assert!(stats.bytes_reclaimed > 0, "dry run still tallies bytes");
        assert!(stats.entries_removed > 0, "dry run still tallies entries");
    }

    #[test]
    fn purge_refuses_path_outside_root() {
        let temp = tempfile::TempDir::new().expect("tempdir");
        let data_dir = temp.path().join("cass-data");

        // The canonical root is accepted by the containment guard.
        let canonical = raw_mirror_root(&data_dir);
        assert_raw_mirror_root_contained(&data_dir, &canonical)
            .expect("canonical root must pass containment");

        // A path that is NOT the canonical raw-mirror root is refused, even though
        // it is under data_dir (e.g. the sibling SQLite DB / Tantivy index dir).
        let hostile = data_dir.join("agent_search.db");
        let err = assert_raw_mirror_root_contained(&data_dir, &hostile)
            .expect_err("non-raw-mirror path must be refused");
        assert!(
            err.to_string().contains("refusing to purge raw mirror"),
            "unexpected containment error: {err}"
        );

        // A raw-mirror root for a DIFFERENT data_dir is likewise refused.
        let other = raw_mirror_root(&temp.path().join("other-data"));
        let err = assert_raw_mirror_root_contained(&data_dir, &other)
            .expect_err("foreign raw-mirror root must be refused");
        assert!(err.to_string().contains("refusing to purge raw mirror"));
    }
}
