//! Turns client- or model-supplied relative paths into real paths that cannot leave a root folder.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};

/// `rel` must be relative and made only of plain names: no `..`, no root, no NUL.
fn lexical(root: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel);
    if rel.is_empty() || rel.contains('\0') || p.is_absolute() {
        bail!("path must be relative to the folder: {rel:?}");
    }
    if !p.components().all(|c| matches!(c, Component::Normal(_) | Component::CurDir)) {
        bail!("path must stay inside the folder: {rel:?}");
    }
    Ok(root.join(p))
}

fn canonical_root(root: &Path) -> Result<PathBuf> {
    root.canonicalize().with_context(|| format!("folder {} is not available", root.display()))
}

/// An existing file or directory under `root`; symlinks pointing outside are refused.
pub fn existing(root: &Path, rel: &str) -> Result<PathBuf> {
    let root = canonical_root(root)?;
    let path = lexical(&root, rel)?.canonicalize().with_context(|| format!("{rel} does not exist"))?;
    if !path.starts_with(&root) {
        bail!("{rel} points outside the folder");
    }
    Ok(path)
}

/// A path under `root` that may not exist yet (for writing); its nearest existing ancestor must be inside.
pub fn creatable(root: &Path, rel: &str) -> Result<PathBuf> {
    let root = canonical_root(root)?;
    let path = lexical(&root, rel)?;
    let mut ancestor = path.as_path();
    while !ancestor.exists() {
        ancestor = ancestor.parent().context("path has no parent")?;
    }
    if !ancestor.canonicalize()?.starts_with(&root) {
        bail!("{rel} points outside the folder");
    }
    if path.is_symlink() && !path.canonicalize()?.starts_with(&root) {
        bail!("{rel} points outside the folder");
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lair-jail-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn plain_paths_resolve_and_escapes_are_refused() {
        let root = temp("plain");
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(root.join("a/b.txt"), "x").unwrap();
        assert!(existing(&root, "a/b.txt").is_ok());
        assert!(existing(&root, "./a/b.txt").is_ok());
        for bad in ["../x", "/etc/passwd", "a/../../x", ""] {
            assert!(existing(&root, bad).is_err(), "{bad}");
            assert!(creatable(&root, bad).is_err(), "{bad}");
        }
        assert!(creatable(&root, "new/dir/file.md").is_ok());
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_cannot_lead_outside() {
        let root = temp("link");
        let outside = temp("link-outside");
        std::fs::write(outside.join("secret"), "s").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("out")).unwrap();
        std::os::unix::fs::symlink(outside.join("secret"), root.join("secret")).unwrap();
        assert!(existing(&root, "out/secret").is_err());
        assert!(creatable(&root, "out/new.txt").is_err());
        assert!(creatable(&root, "secret").is_err());
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }
}
