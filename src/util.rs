//! Small helpers with no home of their own.

use std::hash::BuildHasher;

/// `dir`, created if missing.
pub fn ensure_dir(dir: std::path::PathBuf) -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// 16 hex digits, different on every call: for ids that only need to be unique, not secret.
pub fn random_id() -> String {
    format!("{:016x}", std::collections::hash_map::RandomState::new().hash_one(std::time::SystemTime::now()))
}

/// Length of the longest proper prefix of `tag` that `s` ends with.
pub fn partial_suffix(s: &str, tag: &str) -> usize {
    (1..tag.len()).rev().filter(|&k| tag.is_char_boundary(k)).find(|&k| s.ends_with(&tag[..k])).unwrap_or(0)
}
