//! Getting skills from the network: a `.zip`, a GitHub folder, or a folder URL holding `skill.json`
//! and `prompt.md`; and the browsable index of skills a user points lair at.
//!
//! Only those two files are read from a download, so nothing else in an archive ever reaches disk.

use std::io::{Cursor, Read};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{Manifest, slug};

/// A zip for one skill.
const ZIP_LIMIT: usize = 1024 * 1024;
/// A whole GitHub repository, of which one folder is used.
const REPO_LIMIT: usize = 25 * 1024 * 1024;
const FILE_LIMIT: u64 = 256 * 1024;

#[derive(Debug)]
pub struct Fetched {
    pub id: String,
    pub manifest: Manifest,
    pub prompt: String,
}

#[derive(Debug, PartialEq)]
enum Kind {
    Zip,
    GitHub { owner: String, repo: String, git_ref: String, path: String },
    Folder,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IndexEntry {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub url: String,
}

fn classify(url: &str) -> Result<Kind> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .context("the address must start with http:// or https://")?;
    if let Some(gh) = rest.strip_prefix("github.com/") {
        let parts: Vec<&str> = gh.trim_end_matches('/').split('/').collect();
        return match parts.as_slice() {
            [owner, repo] => Ok(Kind::GitHub {
                owner: owner.to_string(),
                repo: repo.to_string(),
                git_ref: "HEAD".into(),
                path: String::new(),
            }),
            [owner, repo, "tree", git_ref, path @ ..] => Ok(Kind::GitHub {
                owner: owner.to_string(),
                repo: repo.to_string(),
                git_ref: git_ref.to_string(),
                path: path.join("/"),
            }),
            _ => bail!("use a GitHub address like github.com/owner/repo/tree/branch/folder"),
        };
    }
    Ok(if rest.split(['?', '#']).next().unwrap_or("").ends_with(".zip") { Kind::Zip } else { Kind::Folder })
}

/// The id from the last path segment of a URL or file name, when nothing better names the skill.
fn last_segment(url: &str) -> String {
    let path = url.split(['?', '#']).next().unwrap_or(url).trim_end_matches('/');
    slug(path.rsplit('/').next().unwrap_or_default().trim_end_matches(".zip"))
}

/// Finds `skill.json` + `prompt.md` in a zip. With `under`, only inside that folder (a GitHub
/// archive's top folder is skipped first); otherwise the shallowest skill in the archive wins.
fn from_zip(bytes: &[u8], under: Option<&str>, fallback_id: &str) -> Result<Fetched> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).context("the download is not a valid .zip")?;
    let names: Vec<String> = zip.file_names().map(|n| n.map(|n| n.into_owned())).collect::<Result<_, _>>()?;
    let strip = |name: &str| -> String {
        match under {
            Some(_) => name.split_once('/').map_or("", |(_, rest)| rest).to_owned(),
            None => name.to_owned(),
        }
    };
    let manifest_path = names
        .iter()
        .filter(|n| n.ends_with("skill.json") && (n.len() == "skill.json".len() || n.ends_with("/skill.json")))
        .filter(|n| {
            under.is_none_or(|dir| {
                strip(n) == if dir.is_empty() { "skill.json".to_owned() } else { format!("{dir}/skill.json") }
            })
        })
        .min_by_key(|n| n.matches('/').count())
        .context("no skill.json found in the download")?
        .clone();
    let dir = manifest_path.strip_suffix("skill.json").unwrap_or("");
    let mut read = |name: &str| -> Result<String> {
        let entry = zip.by_name(name).with_context(|| format!("{name} is missing next to skill.json"))?;
        if entry.size() > FILE_LIMIT {
            bail!("{name} is larger than {} KB", FILE_LIMIT / 1024);
        }
        let mut s = String::new();
        entry.take(FILE_LIMIT).read_to_string(&mut s).with_context(|| format!("{name} is not text"))?;
        Ok(s)
    };
    let manifest: Manifest = serde_json::from_str(&read(&manifest_path)?).context("invalid skill.json")?;
    let prompt = read(&format!("{dir}prompt.md"))?;
    let folder =
        dir.trim_end_matches('/').rsplit('/').next().filter(|f| !f.is_empty() && under.is_none_or(|u| !u.is_empty()));
    Ok(Fetched { id: folder.map_or_else(|| fallback_id.to_owned(), slug), manifest, prompt })
}

/// Parses an uploaded `.zip` the same way an URL's would be: the shallowest `skill.json` wins.
/// Its source is recorded as an upload, so it shows as installed and tracks no remote URL.
pub fn from_zip_bytes(bytes: &[u8], file_name: &str) -> Result<Fetched> {
    let mut fetched = from_zip(bytes, None, &last_segment(file_name))?;
    fetched.manifest.source = Some(format!("upload:{file_name}"));
    Ok(fetched)
}

async fn download(http: &reqwest::Client, url: &str, limit: usize) -> Result<Vec<u8>> {
    let (_, body, cut) = crate::net::get_capped(http, url, limit).await?;
    if cut {
        bail!("the download is larger than {} KB", limit / 1024);
    }
    Ok(body)
}

async fn text(http: &reqwest::Client, url: &str) -> Result<String> {
    String::from_utf8(download(http, url, FILE_LIMIT as usize).await.with_context(|| format!("fetching {url}"))?)
        .with_context(|| format!("{url} is not text"))
}

/// Downloads the skill at `url`; its manifest records `url` as its source.
pub async fn fetch(http: &reqwest::Client, url: &str) -> Result<Fetched> {
    let url = url.trim();
    let mut fetched = match classify(url)? {
        Kind::Zip => from_zip(&download(http, url, ZIP_LIMIT).await?, None, &last_segment(url))?,
        Kind::GitHub { owner, repo, git_ref, path } => {
            let archive =
                download(http, &format!("https://codeload.github.com/{owner}/{repo}/zip/{git_ref}"), REPO_LIMIT)
                    .await?;
            let id = if path.is_empty() { repo.to_ascii_lowercase() } else { last_segment(&path) };
            from_zip(&archive, Some(&path), &id)?
        }
        Kind::Folder => {
            let base = url.trim_end_matches('/');
            let manifest = serde_json::from_str(&text(http, &format!("{base}/skill.json")).await?)
                .context("invalid skill.json")?;
            Fetched { id: last_segment(base), manifest, prompt: text(http, &format!("{base}/prompt.md")).await? }
        }
    };
    fetched.manifest.source = Some(url.to_owned());
    Ok(fetched)
}

/// The skills listed by an index file: `{ "skills": [{ "id", "name", "description", "url" }] }`.
pub async fn index(http: &reqwest::Client, url: &str) -> Result<Vec<IndexEntry>> {
    #[derive(Deserialize)]
    struct Index {
        skills: Vec<IndexEntry>,
    }
    let index: Index =
        serde_json::from_str(&text(http, url).await?).context("the skills index is not in the expected format")?;
    Ok(index.skills)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn zip_of(files: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        let mut w = zip::ZipWriter::new(&mut buf);
        for (name, body) in files {
            w.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            w.write_all(body.as_bytes()).unwrap();
        }
        w.finish().unwrap();
        buf.into_inner()
    }

    const MANIFEST: &str = r#"{"name": "Pirata", "when": ["pirata"]}"#;

    #[test]
    fn urls_are_classified() {
        assert_eq!(classify("https://x.dev/skills/pirate.zip?dl=1").unwrap(), Kind::Zip);
        assert_eq!(classify("https://x.dev/skills/pirate/").unwrap(), Kind::Folder);
        assert_eq!(
            classify("https://github.com/me/skills/tree/main/pirate").unwrap(),
            Kind::GitHub { owner: "me".into(), repo: "skills".into(), git_ref: "main".into(), path: "pirate".into() }
        );
        assert!(
            matches!(classify("https://github.com/me/pirate").unwrap(), Kind::GitHub { git_ref, path, .. } if git_ref == "HEAD" && path.is_empty())
        );
        assert!(classify("ftp://x").is_err());
        assert!(classify("https://github.com/me").is_err());
    }

    #[test]
    fn a_zip_skill_is_found_at_its_shallowest_folder() {
        let z =
            zip_of(&[("pirate/skill.json", MANIFEST), ("pirate/prompt.md", "Arr."), ("pirate/extra/skill.json", "{}")]);
        let f = from_zip(&z, None, "fallback").unwrap();
        assert_eq!((f.id.as_str(), f.manifest.name.as_str(), f.prompt.as_str()), ("pirate", "Pirata", "Arr."));
        let flat = from_zip(&zip_of(&[("skill.json", MANIFEST), ("prompt.md", "Arr.")]), None, "pirate-zip").unwrap();
        assert_eq!(flat.id, "pirate-zip");
    }

    #[test]
    fn a_github_archive_uses_only_the_requested_folder() {
        let z = zip_of(&[
            ("skills-main/README.md", "hi"),
            ("skills-main/other/skill.json", r#"{"name": "Outra", "when": ["x"]}"#),
            ("skills-main/other/prompt.md", "x"),
            ("skills-main/pirate/skill.json", MANIFEST),
            ("skills-main/pirate/prompt.md", "Arr."),
        ]);
        let f = from_zip(&z, Some("pirate"), "pirate").unwrap();
        assert_eq!(f.manifest.name, "Pirata");
        assert!(from_zip(&z, Some("missing"), "x").is_err());
        let root = zip_of(&[("pirate-main/skill.json", MANIFEST), ("pirate-main/prompt.md", "Arr.")]);
        assert_eq!(from_zip(&root, Some(""), "pirate").unwrap().id, "pirate", "repo root uses the repo name");
    }

    #[test]
    fn broken_archives_explain_what_is_missing() {
        assert!(from_zip(b"not a zip", None, "x").unwrap_err().to_string().contains("valid .zip"));
        let no_prompt = zip_of(&[("p/skill.json", MANIFEST)]);
        assert!(from_zip(&no_prompt, None, "x").unwrap_err().to_string().contains("prompt.md"));
        let big = zip_of(&[("p/skill.json", MANIFEST), ("p/prompt.md", &"a".repeat(FILE_LIMIT as usize + 1))]);
        assert!(from_zip(&big, None, "x").unwrap_err().to_string().contains("larger"));
    }

    #[test]
    fn an_uploaded_zip_becomes_an_installed_skill_with_a_safe_id() {
        let f = from_zip_bytes(
            &zip_of(&[("pirate/skill.json", MANIFEST), ("pirate/prompt.md", "Arr.")]),
            "Pirata Pack.zip",
        )
        .unwrap();
        assert_eq!(f.id, "pirate", "the folder inside the zip names the skill, as with a URL");
        assert_eq!(f.manifest.source.as_deref(), Some("upload:Pirata Pack.zip"));

        let f = from_zip_bytes(&zip_of(&[("skill.json", MANIFEST), ("prompt.md", "x")]), "===.zip").unwrap();
        assert_eq!(f.id, "skill", "no folder inside: the (cleaned) file name is the fallback");
        assert!(from_zip_bytes(b"not a zip", "x.zip").is_err());
    }
}
