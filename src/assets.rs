//! Local 3D assets (GLB/GLTF and their sibling buffers/textures) served to the UI.
//!
//! Paths are canonicalized and must stay under one of the configured roots.
//! Dotfiles, absolute paths and `..` components are rejected; only known 3D
//! extensions are served.

use std::path::{Path, PathBuf};

const MAX_BYTES: u64 = 64 * 1024 * 1024;

const ALLOWED: &[&str] = &["glb", "gltf", "bin", "png", "jpg", "jpeg", "webp", "ktx2", "gif", "avif"];

#[derive(Default)]
pub struct Assets {
    roots: Vec<PathBuf>,
    /// `.glb`/`.gltf` files, as paths relative to their root; what the model may reference.
    ids: Vec<String>,
}

impl Assets {
    /// `paths` are directories (searched recursively) or files (their parent is used).
    pub fn new(paths: &[PathBuf]) -> Self {
        let mut roots = Vec::new();
        for p in paths {
            let dir = if p.is_dir() { p.clone() } else { p.parent().map(Path::to_path_buf).unwrap_or_default() };
            let Ok(canon) = dir.canonicalize() else { continue };
            if !roots.contains(&canon) {
                roots.push(canon);
            }
        }
        let mut ids = Vec::new();
        for root in &roots {
            scan(root, root, &mut ids);
        }
        ids.sort();
        Self { roots, ids }
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// Resolves a client-supplied id to a real file, or `None` if it is unsafe.
    pub fn resolve(&self, id: &str) -> Option<PathBuf> {
        let ext = Path::new(id).extension()?.to_str()?.to_ascii_lowercase();
        if id.len() > 512 || id.split('/').any(|part| part.starts_with('.')) || !ALLOWED.contains(&ext.as_str()) {
            return None;
        }
        self.roots
            .iter()
            .filter_map(|root| crate::jail::existing(root, id).ok())
            .find(|p| p.is_file() && p.metadata().is_ok_and(|m| m.len() <= MAX_BYTES))
    }
}

fn scan(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let Ok(file_type) = entry.file_type() else { continue };
        if file_type.is_symlink() {
            continue; // never follow links out of the root (or into loops)
        }
        let path = entry.path();
        if file_type.is_dir() {
            scan(root, &path, out);
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).unwrap_or_default();
        if ext != "glb" && ext != "gltf" {
            continue;
        }
        let Ok(canon) = path.canonicalize() else { continue };
        if !canon.starts_with(root) {
            continue;
        }
        let Ok(rel) = canon.strip_prefix(root) else { continue };
        let id = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
        if out.contains(&id) {
            eprintln!("assets: {id} exists in two roots; using the first");
            continue;
        }
        out.push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lair-assets-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn lists_models_and_resolves_siblings() {
        let dir = temp("list");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("robot.glb"), b"glb").unwrap();
        std::fs::write(dir.join("sub/ship.gltf"), b"gltf").unwrap();
        std::fs::write(dir.join("sub/ship.bin"), b"bin").unwrap();
        std::fs::write(dir.join("notes.txt"), b"no").unwrap();
        std::fs::write(dir.join(".hidden.glb"), b"no").unwrap();

        let a = Assets::new(&[dir.clone()]);
        assert_eq!(a.ids(), ["robot.glb", "sub/ship.gltf"]);
        assert!(a.resolve("robot.glb").is_some());
        assert!(a.resolve("sub/ship.bin").is_some());
        assert!(a.resolve("sub/ship.gltf").is_some());
        assert!(a.resolve("notes.txt").is_none(), "extension allowlist");
        assert!(a.resolve(".hidden.glb").is_none(), "dotfiles are hidden");
        assert!(a.resolve("../etc/passwd").is_none(), "traversal");
        assert!(a.resolve("/etc/passwd").is_none(), "absolute path");
        assert!(a.resolve("missing.glb").is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_symlink_cannot_escape_the_root() {
        let dir = temp("symlink");
        let outside = temp("symlink-outside");
        std::fs::write(outside.join("secret.glb"), b"secret").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.join("secret.glb"), dir.join("link.glb")).unwrap();
        let a = Assets::new(&[dir.clone()]);
        assert!(a.resolve("link.glb").is_none());
        assert!(a.ids().is_empty());
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn a_file_path_uses_its_parent_as_root() {
        let dir = temp("file-root");
        std::fs::write(dir.join("a.glb"), b"a").unwrap();
        let a = Assets::new(&[dir.join("a.glb")]);
        assert_eq!(a.ids(), ["a.glb"]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
