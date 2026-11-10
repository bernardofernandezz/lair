//! Files attached to a user message. An upload lands in the agent's workspace, under `anexos/`, so
//! the agent can open the original with `read_file`. A document's text goes into the prompt; an
//! image goes to a vision model as pixels.

use std::hash::BuildHasher;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::store::Attachment;

/// The extensions lair accepts, by what it does with them. The UI asks for this list.
#[derive(Serialize)]
pub struct Types {
    /// Text handed to the model (PDFs have theirs extracted).
    pub documents: &'static [&'static str],
    /// Shown to vision models.
    pub images: &'static [&'static str],
}

pub const TYPES: Types = Types {
    documents: &[
        "pdf", "txt", "log", "md", "markdown", "rst", "csv", "tsv", "json", "json5", "xml", "yml", "yaml", "toml",
        "ini", "html", "svg", "c", "h", "cpp", "hpp", "cc", "cs", "rs", "py", "js", "jsx", "ts", "tsx", "go", "java",
        "kt", "rb", "php", "lua", "sh", "bash", "zsh", "fish", "sql", "css",
    ],
    images: &["png", "jpg", "jpeg", "webp", "gif", "bmp"],
};

/// The largest upload accepted.
pub const MAX_UPLOAD: usize = 16 * 1024 * 1024;
/// How much of a document's text enters the prompt; the rest is for the agent's `read_file`.
const INLINE_CHARS: usize = 16_000;
const DIR: &str = "anexos";

/// What a turn gets from an attachment: the record kept with the message, and what the model sees.
pub struct Loaded {
    pub attachment: Attachment,
    pub content: Content,
}

pub enum Content {
    Text(String),
    Image(PathBuf),
}

fn ext(name: &str) -> String {
    Path::new(name).extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase()
}

fn is_image(name: &str) -> bool {
    TYPES.images.contains(&ext(name).as_str())
}

fn accepted(name: &str) -> bool {
    let e = ext(name);
    TYPES.documents.contains(&e.as_str()) || TYPES.images.contains(&e.as_str())
}

/// Stores an upload in the workspace and returns its id: `"<random>-<safe name>"`, a plain file name.
pub fn save(workspace: &Path, name: &str, data: &[u8]) -> Result<String> {
    if !accepted(name) {
        bail!("unsupported attachment type: .{}", ext(name));
    }
    if data.len() > MAX_UPLOAD {
        bail!("the file is over {} MB", MAX_UPLOAD / (1024 * 1024));
    }
    let id = new_id(name);
    let dir = workspace.join(DIR);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(&id), data)?;
    Ok(id)
}

/// The stored file with that id; refused when the id is not a plain, accepted file name.
pub fn path(workspace: &Path, id: &str) -> Result<PathBuf> {
    if !accepted(id) || id.contains(['/', '\\']) || id.starts_with('.') {
        bail!("invalid attachment id: {id}");
    }
    let path = workspace.join(DIR).join(id);
    std::fs::metadata(&path).with_context(|| format!("attachment {id} was not found"))?;
    Ok(path)
}

pub fn load(workspace: &Path, id: &str, name: &str) -> Result<Loaded> {
    let file = path(workspace, id)?;
    let image = is_image(id);
    let content = if image {
        Content::Image(file)
    } else if ext(id) == "pdf" {
        let data = std::fs::read(&file)?;
        Content::Text(pdf_extract::extract_text_from_mem(&data).context("extraindo texto do pdf")?)
    } else {
        Content::Text(std::fs::read_to_string(&file).with_context(|| format!("{name} is not a text file"))?)
    };
    let chars = match &content {
        Content::Text(t) => t.chars().count(),
        Content::Image(_) => 0,
    };
    Ok(Loaded { attachment: Attachment { id: id.to_owned(), name: name.to_owned(), chars, image }, content })
}

fn new_id(raw: &str) -> String {
    let mut name = raw
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .trim()
        .replace(|c: char| !(c.is_alphanumeric() || c == '.' || c == '-' || c == '_'), "-");
    if name.is_empty() || name.starts_with('.') {
        name = format!("documento{name}");
    }
    let dot = name.rfind('.').unwrap_or(name.len());
    let (stem, extension) = name.split_at(dot);
    let random = std::collections::hash_map::RandomState::new().hash_one(std::time::SystemTime::now());
    format!("{:08x}-{}{extension}", random as u32, stem.chars().take(48).collect::<String>())
}

/// Attribute values inside the block below; a quote in a file name must not end the attribute.
fn attr(s: &str) -> String {
    s.replace('\\', "/").replace('"', "''")
}

/// The documents' text, put before the user's question; the agent learns where each original lives.
pub fn sections(docs: &[(&Attachment, &str)]) -> String {
    let mut out = String::new();
    for (a, text) in docs {
        let (name, path) = (attr(&a.name), attr(&format!("{DIR}/{}", a.id)));
        let text: String = text.trim().chars().take(INLINE_CHARS).collect();
        if text.is_empty() {
            out.push_str(&format!("<attachment name=\"{name}\" path=\"{path}\" text=\"empty\">\n</attachment>\n\n"));
        } else {
            out.push_str(&format!(
                "<attachment name=\"{name}\" path=\"{path}\" chars=\"{}\">\n{text}\n</attachment>\n\n",
                a.chars
            ));
        }
    }
    out.push_str(
        "Attached document(s): use them to answer. The agent can also open the original files in the workspace.\n\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lair-attach-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn uploads_get_safe_unique_ids_inside_the_workspace() {
        let ws = workspace("save");
        let a = save(&ws, "Relatório Anual.md", b"# oi").unwrap();
        let b = save(&ws, "Relatório Anual.md", b"# oi").unwrap();
        assert!(a.ends_with("-Relatório-Anual.md"), "{a}");
        assert_ne!(a, b);
        let traversal = save(&ws, "../../etc/passwd.txt", b"x").unwrap();
        assert!(!traversal.contains('/') && path(&ws, &traversal).is_ok(), "{traversal}");
        assert!(save(&ws, "..md", b"x").unwrap().contains("-documento."), "a name of dots still names a file");
        assert!(save(&ws, "setup.exe", b"x").is_err());
        assert!(save(&ws, "big.txt", &vec![b'x'; MAX_UPLOAD + 1]).is_err());
        for bad in ["..", "../lair.db", "a/b.md", "nota.exe", "faltando.md"] {
            assert!(path(&ws, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn documents_give_text_and_images_give_a_file() {
        let ws = workspace("load");
        let md = save(&ws, "notas.md", "## O que mudou\n\ntudo.".as_bytes()).unwrap();
        let png = save(&ws, "foto.PNG", b"png").unwrap();
        let doc = load(&ws, &md, "notas.md").unwrap();
        assert!(matches!(&doc.content, Content::Text(t) if t.contains("tudo")));
        assert!(!doc.attachment.image && doc.attachment.chars > 0);
        let img = load(&ws, &png, "foto.PNG").unwrap();
        assert!(matches!(img.content, Content::Image(_)) && img.attachment.image);
    }

    #[test]
    fn sections_name_each_file_and_survive_odd_names() {
        let a = Attachment { id: "1-vazio.pdf".into(), name: "vazio.pdf".into(), chars: 0, image: false };
        let b = Attachment { id: "2-n.md".into(), name: "stra\"nge.md".into(), chars: 11, image: false };
        let s = sections(&[(&a, "   "), (&b, "duas coisas")]);
        assert!(s.contains("text=\"empty\""), "an empty document is still announced");
        assert!(s.contains("path=\"anexos/2-n.md\"") && s.contains("duas coisas"));
        assert!(s.contains("stra''nge"), "{s}");
    }
}
