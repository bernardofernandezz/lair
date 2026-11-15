//! Ollama's model store on disk: models it already downloaded run here with no second copy.

use std::path::{Path, PathBuf};

/// Where Ollama keeps its models (`OLLAMA_MODELS`, or `.ollama/models` in the home folder), if it has any.
pub fn dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("OLLAMA_MODELS") {
        Some(dir) => dir.into(),
        None => {
            Path::new(&std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?).join(".ollama/models")
        }
    };
    dir.join("manifests").is_dir().then_some(dir)
}

/// The models in an Ollama store, by the names Ollama gives them (`qwen3:8b`, `hf.co/owner/repo:Q4_K_M`).
/// Its manifests name the GGUF blob each one runs on; the blobs have no `.gguf` to be found by.
// shortcut: a vision model's projector layer is not read, so it loads without seeing; read
// `application/vnd.ollama.image.projector` too if Ollama vision models matter.
pub fn models(root: &Path) -> Vec<(String, PathBuf)> {
    let manifests = root.join("manifests");
    let mut found = Vec::new();
    let mut stack = vec![manifests.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for path in entries.flatten().map(|e| e.path()) {
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            // manifests/<host>/<namespace…>/<name>/<tag>; Ollama's own `library` namespace is left out.
            let Ok(rel) = path.strip_prefix(&manifests) else { continue };
            let parts: Vec<String> = rel.iter().map(|p| p.to_string_lossy().into_owned()).collect();
            let [host, name @ .., tag] = parts.as_slice() else { continue };
            if name.is_empty() {
                continue;
            }
            let name = match name {
                [ns, rest @ ..] if ns == "library" && host == "registry.ollama.ai" => rest.join("/"),
                _ if host == "registry.ollama.ai" => name.join("/"),
                _ => format!("{host}/{}", name.join("/")),
            };
            let Ok(manifest) = std::fs::read(&path) else { continue };
            let manifest: serde_json::Value = serde_json::from_slice(&manifest).unwrap_or_default();
            let layers = manifest["layers"].as_array().map(Vec::as_slice).unwrap_or_default();
            let model = layers.iter().find(|l| l["mediaType"] == "application/vnd.ollama.image.model");
            let Some(hex) = model.and_then(|l| l["digest"].as_str()?.strip_prefix("sha256:")) else { continue };
            // The digest names a file: never let it climb out of the blobs folder.
            if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                continue;
            }
            let blob = root.join("blobs").join(format!("sha256-{hex}"));
            if blob.is_file() {
                found.push((format!("{name}:{tag}"), blob));
            }
        }
    }
    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_manifests_name_their_blobs() {
        let root = std::env::temp_dir().join(format!("lair-ollama-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let manifest = |dir: &str, tag: &str, digest: &str| {
            let dir = root.join("manifests").join(dir);
            std::fs::create_dir_all(&dir).unwrap();
            let layers = serde_json::json!({ "layers": [
                { "mediaType": "application/vnd.ollama.image.template", "digest": "sha256:00" },
                { "mediaType": "application/vnd.ollama.image.model", "digest": digest }
            ] });
            std::fs::write(dir.join(tag), layers.to_string()).unwrap();
        };
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        for hex in ["ab12", "cd34"] {
            std::fs::write(root.join("blobs").join(format!("sha256-{hex}")), b"GGUF").unwrap();
        }
        manifest("registry.ollama.ai/library/qwen3", "8b", "sha256:ab12");
        manifest("hf.co/bartowski/Some-GGUF", "Q4_K_M", "sha256:cd34");
        manifest("registry.ollama.ai/library/gone", "latest", "sha256:ef56");
        manifest("registry.ollama.ai/library/evil", "x", "sha256:../../etc/passwd");
        let names: Vec<String> = models(&root).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["hf.co/bartowski/Some-GGUF:Q4_K_M", "qwen3:8b"]);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
