//! Getting model files onto this machine and off it: downloads that resume where they stopped and
//! are checked against the hash Hugging Face publishes, and removing an installed model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use reqwest::header::{CONTENT_RANGE, RANGE};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::engine::{shard_name, shard_of};
use crate::hub::{self, Remote};
use crate::lair::Lair;

/// A download's progress, as the UI polls it.
#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    pub repo: String,
    pub file: String,
    pub done: u64,
    pub total: u64,
    pub finished: bool,
    /// Every byte is here and being checked against the published hash.
    pub verifying: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

struct Entry {
    progress: Progress,
    abort: tokio::task::AbortHandle,
    /// What it writes, to delete on cancel.
    files: Vec<PathBuf>,
}

/// The downloads started from the UI or an API, by `repo/file`.
#[derive(Default)]
pub struct Downloads {
    list: Mutex<HashMap<String, Entry>>,
}

fn key(repo: &str, file: &str) -> String {
    format!("{repo}/{file}")
}

fn part_of(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_owned();
    name.push(".part");
    dest.with_file_name(name)
}

impl Downloads {
    pub fn list(&self) -> Vec<Progress> {
        let mut out: Vec<Progress> = self.list.lock().unwrap().values().map(|e| e.progress.clone()).collect();
        out.sort_by(|a, b| (&a.repo, &a.file).cmp(&(&b.repo, &b.file)));
        out
    }

    pub fn get(&self, repo: &str, file: &str) -> Option<Progress> {
        self.list.lock().unwrap().get(&key(repo, file)).map(|e| e.progress.clone())
    }

    fn update(&self, key: &str, f: impl FnOnce(&mut Progress)) {
        if let Some(e) = self.list.lock().unwrap().get_mut(key) {
            f(&mut e.progress);
        }
    }

    /// Stops a running download and deletes what it wrote so far, or forgets one that ended. A
    /// download that failed keeps its partial file until then, so starting it again resumes.
    pub fn cancel(&self, repo: &str, file: &str) -> bool {
        let Some(e) = self.list.lock().unwrap().remove(&key(repo, file)) else {
            return false;
        };
        if !e.progress.finished {
            e.abort.abort();
            for f in &e.files {
                let _ = std::fs::remove_file(part_of(f));
            }
        }
        true
    }
}

/// Starts downloading `file` from `repo`, with the smallest projector if the repo has one. The file
/// must be one the repo lists; the URL is built here, never taken from the client.
pub async fn start(lair: Arc<Lair>, repo: String, file: String) -> Result<()> {
    let dir = lair.models.dir().context("no writable models folder")?.join(hub::folder(&repo));
    let token = lair.config().hf_token;
    let listed = hub::group(hub::tree(&lair.http, &repo, token.as_deref()).await?);
    if !listed.iter().any(|l| l.name == file) {
        bail!("{file} is not in {repo}");
    }
    let queue: Vec<Remote> = hub::with_projector(&listed, &file, &dir).into_iter().flat_map(|l| l.parts).collect();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let key = key(&repo, &file);
    // One lock from the check to the abort handle: a cancel can never miss a download that started.
    let mut list = lair.downloads.list.lock().unwrap();
    if list.get(&key).is_some_and(|e| !e.progress.finished && e.progress.error.is_none()) {
        bail!("{file} is already downloading");
    }
    let progress = Progress {
        repo: repo.clone(),
        file: file.clone(),
        done: 0,
        total: queue.iter().map(|r| r.size).sum(),
        finished: false,
        verifying: false,
        error: None,
    };
    let files: Vec<PathBuf> = queue.iter().map(|r| dir.join(r.name())).collect();
    let task = {
        let (lair, key) = (lair.clone(), key.clone());
        tokio::spawn(async move {
            let http = crate::net::download_client();
            let mut before = 0;
            for r in &queue {
                let (url, dest) = (hub::file_url(&repo, &r.path), dir.join(r.name()));
                let res = fetch(&http, &url, &dest, r.sha256.as_deref(), token.as_deref(), |done, total| {
                    lair.downloads.update(&key, |p| {
                        p.done = before + done;
                        p.verifying = r.sha256.is_some() && total > 0 && done == total;
                    });
                })
                .await;
                if let Err(e) = res {
                    return lair.downloads.update(&key, |p| {
                        p.verifying = false;
                        p.error = Some(format!("{e:#}"));
                    });
                }
                before += r.size;
            }
            lair.models.refresh().await;
            lair.downloads.update(&key, |p| {
                p.verifying = false;
                p.finished = true;
            });
        })
    };
    list.insert(key, Entry { progress, abort: task.abort_handle(), files });
    Ok(())
}

/// Downloads `url` to `dest`, reporting `(done, total)` bytes. Bytes go to `dest.part`, which a
/// later call continues from, and become `dest` only once they are all there and match `sha256`
/// (when given), so an interrupted or corrupted download never looks like a model. `token` is sent
/// only to Hugging Face. A `dest` that already exists is taken as done.
pub async fn fetch(
    http: &reqwest::Client,
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    token: Option<&str>,
    mut progress: impl FnMut(u64, u64),
) -> Result<()> {
    if dest.exists() {
        let size = dest.metadata().map_or(0, |m| m.len());
        progress(size, size);
        return Ok(());
    }
    let part = part_of(dest);
    let have = tokio::fs::metadata(&part).await.map_or(0, |m| m.len());
    let request = |from: u64| {
        let mut req = http.get(url);
        if from > 0 {
            req = req.header(RANGE, format!("bytes={from}-"));
        }
        if let Some(t) = token.filter(|_| hub::is_hf(url)) {
            req = req.bearer_auth(t);
        }
        req.send()
    };
    let mut resp = request(have).await.with_context(|| format!("downloading {url}"))?;
    // The server resumes only if it answers 206 from where we are; one that answers some other range
    // is asked again for the whole file.
    let resumed = resp.status() == StatusCode::PARTIAL_CONTENT
        && resp
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with(&format!("bytes {have}-")));
    if resp.status() == StatusCode::PARTIAL_CONTENT && !resumed {
        resp = request(0).await.with_context(|| format!("downloading {url}"))?;
    }
    let status = resp.status();
    if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) && hub::is_hf(url) {
        bail!(
            "Hugging Face refused the download ({status}): the repo is gated or private. Accept its terms on \
             huggingface.co and set a token (HF_TOKEN, or in Settings)"
        );
    }
    if status == StatusCode::RANGE_NOT_SATISFIABLE && have > 0 {
        // The partial file already holds every byte: the last run died before renaming it.
        progress(have, have);
    } else {
        let mut resp = resp.error_for_status().with_context(|| format!("downloading {url}"))?;
        let mut done = if resumed { have } else { 0 };
        let total = done + resp.content_length().unwrap_or(0);
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resumed)
            .truncate(!resumed)
            .open(&part)
            .await
            .with_context(|| format!("{} is not writable", part.display()))?;
        while let Some(chunk) = resp.chunk().await.with_context(|| format!("downloading {url}"))? {
            file.write_all(&chunk).await?;
            done += chunk.len() as u64;
            progress(done, total);
        }
        file.flush().await?;
        if total > done {
            bail!("download ended early ({done} of {total} bytes); start it again to continue from there");
        }
    }
    if let Some(expected) = sha256 {
        let path = part.clone();
        let actual = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut hash = Sha256::new();
            std::io::copy(&mut std::fs::File::open(path)?, &mut hash)?;
            Ok(format!("{:x}", hash.finalize()))
        })
        .await??;
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = tokio::fs::remove_file(&part).await;
            let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
            bail!("{name} doesn't match its published checksum; the download was corrupted and has been deleted");
        }
    }
    tokio::fs::rename(&part, dest).await?;
    Ok(())
}

/// Downloads the .gguf at `url` into `dir` under its own name: `lair pull <url>` and the default model.
pub async fn download(url: &str, dir: &Path, token: Option<&str>, progress: impl FnMut(u64, u64)) -> Result<PathBuf> {
    let name = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .filter(|n| n.ends_with(".gguf"))
        .context("the URL must point to a .gguf file")?;
    let dest = dir.join(name);
    fetch(&crate::net::download_client(), url, &dest, None, token, progress)
        .await
        .with_context(|| format!("saving into {}", dir.display()))?;
    Ok(dest)
}

/// Deletes the model at `path` (every part of a split one) and returns the bytes freed. `path` must
/// be inside `models`: lair only deletes from its own folder, never a file `-m` pointed at elsewhere.
/// A repo's folder left with no model in it goes too, with its projector and unfinished downloads.
pub fn remove(models: &Path, path: &Path) -> Result<u64> {
    let (root, path) = (models.canonicalize()?, path.canonicalize()?);
    if !path.starts_with(&root) || !path.is_file() {
        bail!("{} is not in lair's models folder; delete it yourself", path.display());
    }
    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let files: Vec<PathBuf> = match shard_of(&name) {
        Some((base, _, total)) => (1..=total).map(|i| path.with_file_name(shard_name(base, i, total))).collect(),
        None => vec![path.clone()],
    };
    let mut freed = 0;
    for f in files.iter().filter(|f| f.exists()) {
        freed += f.metadata().map_or(0, |m| m.len());
        std::fs::remove_file(f).with_context(|| format!("deleting {}", f.display()))?;
    }
    let dir = path.parent().unwrap_or(&root);
    if dir != root {
        let left: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        let is_model = |p: &PathBuf| {
            let n = p.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
            n.ends_with(".gguf") && !n.contains("mmproj")
        };
        if !left.iter().any(is_model) {
            for f in left.iter().filter(|f| f.is_file()) {
                freed += f.metadata().map_or(0, |m| m.len());
                let _ = std::fs::remove_file(f);
            }
            // Fails, harmlessly, if something else is still in there.
            let _ = std::fs::remove_dir(dir);
        }
    }
    Ok(freed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use axum::routing::get;

    const BODY: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    /// A server with one file that honours `Range` and counts what it was asked.
    async fn serve() -> (String, Arc<Mutex<Vec<String>>>) {
        let asked: Arc<Mutex<Vec<String>>> = Arc::default();
        let log = asked.clone();
        let app = axum::Router::new().route(
            "/m.gguf",
            get(move |headers: HeaderMap| async move {
                let range = headers.get("range").and_then(|v| v.to_str().ok()).unwrap_or("").to_owned();
                let auth = headers.contains_key("authorization");
                log.lock().unwrap().push(format!("{range}{}", if auth { " +auth" } else { "" }));
                match range
                    .strip_prefix("bytes=")
                    .and_then(|r| r.strip_suffix('-'))
                    .and_then(|n| n.parse::<usize>().ok())
                {
                    Some(from) if from >= BODY.len() => (StatusCode::RANGE_NOT_SATISFIABLE, HeaderMap::new(), vec![]),
                    Some(from) => {
                        let mut h = HeaderMap::new();
                        let range = format!("bytes {from}-{}/{}", BODY.len() - 1, BODY.len());
                        h.insert("content-range", range.parse().unwrap());
                        (StatusCode::PARTIAL_CONTENT, h, BODY[from..].to_vec())
                    }
                    None => (StatusCode::OK, HeaderMap::new(), BODY.to_vec()),
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/m.gguf", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await });
        (url, asked)
    }

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[tokio::test]
    async fn a_download_resumes_and_is_checked_before_it_becomes_a_model() {
        let dir = std::env::temp_dir().join(format!("lair-fetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (url, asked) = serve().await;
        let http = reqwest::Client::new();
        let (dest, part) = (dir.join("m.gguf"), dir.join("m.gguf.part"));

        // An interrupted download left the first ten bytes.
        std::fs::write(&part, &BODY[..10]).unwrap();
        let mut seen = Vec::new();
        fetch(&http, &url, &dest, Some(&sha(BODY)), Some("secret"), |d, t| seen.push((d, t))).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), BODY);
        assert!(!part.exists());
        assert_eq!(
            *asked.lock().unwrap(),
            ["bytes=10-"],
            "only the rest was asked for, and no token left for a stranger"
        );
        assert_eq!(seen.last(), Some(&(BODY.len() as u64, BODY.len() as u64)));
        assert!(seen[0].0 > 10, "progress counts what was already there");

        // Already there: nothing is asked.
        fetch(&http, &url, &dest, None, None, |_, _| {}).await.unwrap();
        assert_eq!(asked.lock().unwrap().len(), 1);

        // A wrong hash deletes the partial file and leaves no model behind.
        std::fs::remove_file(&dest).unwrap();
        let err = fetch(&http, &url, &dest, Some(&sha(b"other")), None, |_, _| {}).await.unwrap_err();
        assert!(err.to_string().contains("checksum"), "{err}");
        assert!(!dest.exists() && !part.exists());

        // Every byte was there but the rename never happened: verified and finished without downloading.
        std::fs::write(&part, BODY).unwrap();
        fetch(&http, &url, &dest, Some(&sha(BODY)), None, |_, _| {}).await.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), BODY);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn removing_a_model_takes_its_parts_and_stays_inside_the_models_folder() {
        let root = std::env::temp_dir().join(format!("lair-remove-{}", std::process::id()));
        let (models, repo) = (root.join("models"), root.join("models/Qwen__X"));
        std::fs::create_dir_all(&repo).unwrap();
        let touch = |p: PathBuf| {
            std::fs::write(&p, b"12345").unwrap();
            p
        };
        let outside = touch(root.join("mine.gguf"));
        let loose = touch(models.join("loose.gguf"));
        let first = touch(repo.join("big-00001-of-00002.gguf"));
        touch(repo.join("big-00002-of-00002.gguf"));
        let other = touch(repo.join("other.gguf"));
        touch(repo.join("mmproj-f16.gguf"));
        touch(repo.join("stale.gguf.part"));

        assert!(remove(&models, &outside).is_err(), "a model outside the folder is not lair's to delete");
        assert!(remove(&models, &models.join("../mine.gguf")).is_err());
        assert!(outside.exists());

        assert_eq!(remove(&models, &first).unwrap(), 10, "both parts");
        assert!(other.exists() && repo.join("mmproj-f16.gguf").exists(), "the folder still has a model");
        assert_eq!(remove(&models, &other).unwrap(), 15, "the last model takes the projector and leftovers");
        assert!(!repo.exists());
        assert_eq!(remove(&models, &loose).unwrap(), 5);
        assert!(models.exists(), "the models folder itself stays");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
