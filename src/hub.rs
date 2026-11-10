//! Models from Hugging Face: search GGUF repos, judge which files fit this machine, and download them
//! into the models folder so they show up without a restart.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::lair::Lair;

const HF: &str = "https://huggingface.co";

/// Open GGUF repos shown before any search, smallest first; the best fit among them is recommended.
pub const SUGGESTED: &[(&str, &str)] = &[
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Small and fast; good for modest machines."),
    ("Qwen/Qwen3-4B-GGUF", "Reasons; lair's default."),
    ("ggml-org/gemma-3-4b-it-GGUF", "Sees images; light."),
    ("ggml-org/Qwen2.5-VL-7B-Instruct-GGUF", "Sees images; reads documents and screens well."),
    ("Qwen/Qwen3-8B-GGUF", "Reasons; a good balance for daily use."),
    ("ggml-org/gemma-3-12b-it-GGUF", "Sees images; more careful answers."),
    ("Qwen/Qwen3-14B-GGUF", "Reasons; more capable, needs a bigger GPU."),
    ("Qwen/Qwen3-30B-A3B-GGUF", "Reasons; big but fast (MoE)."),
    ("Qwen/Qwen3-32B-GGUF", "Reasons; the most capable on the list."),
];

#[derive(Serialize, Deserialize)]
pub struct Repo {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

/// One GGUF file of a repo, judged against this machine.
#[derive(Clone, Serialize)]
pub struct File {
    pub name: String,
    pub size: u64,
    pub mmproj: bool,
    pub fit: Fit,
    /// A model with this id is already in the catalog.
    pub installed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    /// Runs entirely on the GPU (or in RAM, without one).
    Fits,
    /// Runs only with some layers on the CPU: works, slower.
    Tight,
    No,
}

#[derive(Serialize)]
pub struct Hardware {
    pub gpus: Vec<String>,
    /// GPU memory, summed; llama.cpp splits layers across GPUs.
    pub vram: u64,
    pub ram: u64,
}

pub async fn search(http: &reqwest::Client, q: &str) -> Result<Vec<Repo>> {
    let repos = http
        .get(format!("{HF}/api/models"))
        .query(&[("search", q), ("filter", "gguf"), ("sort", "downloads"), ("limit", "20")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .context("unexpected answer from Hugging Face")?;
    Ok(repos)
}

/// The repo's GGUF files at its root. Split models (`-00001-of-00003`) are left out.
pub async fn files(http: &reqwest::Client, repo: &str) -> Result<Vec<(String, u64)>> {
    #[derive(Deserialize)]
    struct Entry {
        path: String,
        #[serde(default)]
        size: u64,
        lfs: Option<Lfs>,
    }
    #[derive(Deserialize)]
    struct Lfs {
        size: u64,
    }
    check_repo(repo)?;
    let resp = http.get(format!("{HF}/api/models/{repo}/tree/main")).send().await?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND || resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        bail!("repo {repo} not found or gated");
    }
    let entries: Vec<Entry> = resp.error_for_status()?.json().await.context("unexpected answer from Hugging Face")?;
    Ok(entries
        .into_iter()
        .filter(|e| e.path.ends_with(".gguf") && !e.path.contains('/') && !is_shard(&e.path))
        .map(|e| {
            let size = e.lfs.map_or(e.size, |l| l.size);
            (e.path, size)
        })
        .collect())
}

// shortcut: split models are hidden; download every shard into one folder if people ask for big ones.
fn is_shard(name: &str) -> bool {
    let stem = name.trim_end_matches(".gguf");
    stem.rsplit_once("-of-").is_some_and(|(a, b)| {
        b.len() == 5 && b.bytes().all(|c| c.is_ascii_digit()) && a.len() > 6 && a[a.len() - 6..].starts_with('-')
    })
}

/// `owner/name`, each part starting with a letter or digit: never a path that climbs out.
pub fn check_repo(repo: &str) -> Result<()> {
    let ok = |part: &str| {
        !part.is_empty()
            && part.len() <= 96
            && part.starts_with(|c: char| c.is_ascii_alphanumeric())
            && part.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
    };
    match repo.split_once('/') {
        Some((owner, name)) if ok(owner) && ok(name) => Ok(()),
        _ => bail!("\"{repo}\" is not a Hugging Face repo (owner/name)"),
    }
}

pub fn hardware() -> Hardware {
    use llama_cpp_2::LlamaBackendDeviceType as T;
    let gpus: Vec<_> = llama_cpp_2::list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|d| matches!(d.device_type, T::Gpu | T::IntegratedGpu | T::Accelerator))
        .collect();
    Hardware {
        gpus: gpus
            .iter()
            .map(|d| if d.description.is_empty() { d.name.clone() } else { d.description.clone() })
            .collect(),
        vram: gpus.iter().map(|d| d.memory_total as u64).sum(),
        ram: ram_total(),
    }
}

// shortcut: reads /proc/meminfo, so RAM is 0 off Linux; Macs run on Metal, whose memory is counted as VRAM.
fn ram_total() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines().find_map(|l| l.strip_prefix("MemTotal:")?.trim().strip_suffix("kB")?.trim().parse().ok())
        })
        .map_or(0, |kb: u64| kb * 1024)
}

/// Without a GPU, speed is bound by memory bandwidth: past this much weight replies crawl even if RAM fits.
const CPU_COMFORT: u64 = 5_000_000_000;

/// Weights plus the context's KV cache and runtime buffers, roughly.
fn needs(size: u64) -> u64 {
    size + size / 5 + 1_000_000_000
}

pub fn fit(size: u64, hw: &Hardware) -> Fit {
    let need = needs(size) as f64;
    let (vram, ram) = (hw.vram as f64, hw.ram as f64);
    if vram > 0.0 {
        // Some VRAM is always taken by the desktop and the driver.
        if need <= vram * 0.9 {
            Fit::Fits
        } else if need <= vram * 0.9 + ram * 0.6 {
            Fit::Tight
        } else {
            Fit::No
        }
    } else if need <= ram * 0.6 && size <= CPU_COMFORT {
        Fit::Fits
    } else if need <= ram * 0.85 {
        Fit::Tight
    } else {
        Fit::No
    }
}

/// Unquantized weights double the size for little gain; they are never the recommendation.
fn full_precision(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["f16.gguf", "bf16.gguf", "f32.gguf"].iter().any(|s| n.ends_with(s))
}

/// The best-quality file that runs fully on this machine, else the smallest that runs at all. On a CPU,
/// Q4_K_M wins when it fits: bigger quants cost speed there more than they add quality.
pub fn recommend<'a>(files: &'a [File], hw: &Hardware) -> Option<&'a File> {
    let usable = || files.iter().filter(|f| !f.mmproj && !full_precision(&f.name));
    let fits = || usable().filter(|f| f.fit == Fit::Fits);
    let q4 = (hw.vram == 0).then(|| fits().find(|f| f.name.to_ascii_uppercase().contains("Q4_K_M"))).flatten();
    q4.or_else(|| fits().max_by_key(|f| f.size))
        .or_else(|| usable().filter(|f| f.fit == Fit::Tight).min_by_key(|f| f.size))
}

/// A repo's files judged for this machine, with what is already installed marked.
pub async fn judged(lair: &Lair, repo: &str) -> Result<Vec<File>> {
    let hw = hardware();
    let have: Vec<String> = lair.models.local().into_iter().map(|m| m.id).collect();
    let dir = lair.models.dir().map(|d| d.join(folder(repo)));
    Ok(files(&lair.http, repo)
        .await?
        .into_iter()
        .map(|(name, size)| {
            let mmproj = name.to_ascii_lowercase().contains("mmproj");
            let installed = if mmproj {
                dir.as_ref().is_some_and(|d| d.join(&name).exists())
            } else {
                have.iter().any(|id| Some(id.as_str()) == name.strip_suffix(".gguf"))
            };
            File { fit: fit(size, &hw), name, size, mmproj, installed }
        })
        .collect())
}

/// `file`, plus the repo's smallest vision projector when it has one that isn't in `dir` yet.
fn with_projector(listed: &[(String, u64)], file: &str, dir: &Path) -> Vec<String> {
    let mut queue = vec![file.to_owned()];
    if !file.to_ascii_lowercase().contains("mmproj")
        && let Some((proj, _)) =
            listed.iter().filter(|(n, _)| n.to_ascii_lowercase().contains("mmproj")).min_by_key(|(_, s)| *s)
        && !dir.join(proj).exists()
    {
        queue.push(proj.clone());
    }
    queue
}

/// Where a repo's file downloads from.
pub fn file_url(repo: &str, name: &str) -> String {
    format!("{HF}/{repo}/resolve/main/{name}")
}

/// What `lair pull <spec>` downloads: `owner/repo`, `owner/repo:QUANT` or just `repo` (searched).
/// Returns the repo, its folder under `models` and the files to fetch, model first.
pub async fn plan(http: &reqwest::Client, spec: &str, models: &Path) -> Result<(String, PathBuf, Vec<String>)> {
    let (name, quant) = match spec.rsplit_once(':') {
        Some((n, q)) => (n, Some(q.to_ascii_uppercase())),
        None => (spec, None),
    };
    let repo = if name.contains('/') {
        name.to_owned()
    } else {
        let found = search(http, name).await?;
        let pick = found.iter().find(|r| r.id.rsplit('/').next().is_some_and(|n| n.eq_ignore_ascii_case(name)));
        pick.or(found.first()).with_context(|| format!("no GGUF repo on Hugging Face matches {name:?}"))?.id.clone()
    };
    let listed = files(http, &repo).await?;
    let hw = hardware();
    let judged: Vec<File> = listed
        .iter()
        .map(|(n, size)| File {
            name: n.clone(),
            size: *size,
            mmproj: n.to_ascii_lowercase().contains("mmproj"),
            fit: fit(*size, &hw),
            installed: false,
        })
        .collect();
    let file = pick_file(&judged, quant.as_deref(), &hw).map_err(|e| anyhow::anyhow!("{repo}: {e}"))?;
    let dir = models.join(folder(&repo));
    let queue = with_projector(&listed, &file, &dir);
    Ok((repo, dir, queue))
}

/// The file a quant names, or the recommended one for this machine.
fn pick_file(files: &[File], quant: Option<&str>, hw: &Hardware) -> Result<String, String> {
    let models = || files.iter().filter(|f| !f.mmproj);
    match quant {
        Some(q) => {
            // `Q4_0` must pick `m-Q4_0.gguf` over `m-Q4_0_4_4.gguf`: an exact ending beats a substring.
            let upper = |f: &&File| f.name.to_ascii_uppercase();
            let exact = models().find(|f| upper(f).ends_with(&format!("{q}.GGUF")));
            exact.or_else(|| models().find(|f| upper(f).contains(q))).map(|f| f.name.clone()).ok_or_else(|| {
                let names: Vec<&str> = models().map(|f| f.name.as_str()).collect();
                format!("no file matches {q}; the repo has: {}", names.join(", "))
            })
        }
        None => recommend(files, hw)
            .map(|f| f.name.clone())
            .ok_or_else(|| "no file fits this machine; pick one with :QUANT".to_owned()),
    }
}

/// Each repo downloads into its own folder: repos reuse names like `mmproj-model-f16.gguf`.
fn folder(repo: &str) -> String {
    repo.replace('/', "__")
}

/// A download's progress, as the UI polls it.
#[derive(Clone, Serialize)]
pub struct Progress {
    pub repo: String,
    pub file: String,
    pub done: u64,
    pub total: u64,
    pub finished: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Default)]
pub struct Downloads {
    list: Mutex<HashMap<String, (Progress, Option<tokio::task::AbortHandle>)>>,
}

impl Downloads {
    pub fn list(&self) -> Vec<Progress> {
        let mut out: Vec<Progress> = self.list.lock().unwrap().values().map(|(p, _)| p.clone()).collect();
        out.sort_by(|a, b| a.file.cmp(&b.file));
        out
    }

    fn update(&self, file: &str, f: impl FnOnce(&mut Progress)) {
        if let Some((p, _)) = self.list.lock().unwrap().get_mut(file) {
            f(p);
        }
    }

    /// Stops a running download and deletes what it wrote, or forgets a finished one.
    pub fn cancel(&self, file: &str, dir: Option<&Path>) -> bool {
        let Some((p, abort)) = self.list.lock().unwrap().remove(file) else {
            return false;
        };
        if let (Some(abort), false) = (abort, p.finished) {
            abort.abort();
            if let Some(dir) = dir {
                let _ = std::fs::remove_file(dir.join(folder(&p.repo)).join(format!("{file}.part")));
            }
        }
        true
    }
}

/// Starts downloading `file` from `repo`, with the smallest projector if the repo has one. The file
/// must be one the repo lists; the URL is built here, never taken from the client.
pub async fn start(lair: Arc<Lair>, repo: String, file: String) -> Result<()> {
    let dir = lair.models.dir().context("no writable models folder")?.join(folder(&repo));
    let listed = files(&lair.http, &repo).await?;
    if !listed.iter().any(|(n, _)| *n == file) {
        bail!("{file} is not in {repo}");
    }
    let queue = with_projector(&listed, &file, &dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    // One lock from the check to the abort handle: a cancel can never miss a download that started.
    let mut list = lair.downloads.list.lock().unwrap();
    if list.get(&file).is_some_and(|(p, _)| !p.finished && p.error.is_none()) {
        bail!("{file} is already downloading");
    }
    let total = listed.iter().filter(|(n, _)| queue.contains(n)).map(|(_, s)| s).sum();
    let p = Progress { repo: repo.clone(), file: file.clone(), done: 0, total, finished: false, error: None };
    let task = {
        let lair = lair.clone();
        let file = file.clone();
        tokio::spawn(async move {
            let mut before = 0;
            for name in &queue {
                let url = file_url(&repo, name);
                let res =
                    download(&url, &dir, |done, _| lair.downloads.update(&file, |p| p.done = before + done)).await;
                match res {
                    Ok(path) => before += std::fs::metadata(path).map_or(0, |m| m.len()),
                    Err(e) => {
                        let _ = std::fs::remove_file(dir.join(format!("{name}.part")));
                        return lair.downloads.update(&file, |p| p.error = Some(format!("{e:#}")));
                    }
                }
            }
            lair.models.refresh().await;
            lair.downloads.update(&file, |p| p.finished = true);
        })
    };
    list.insert(file, (p, Some(task.abort_handle())));
    Ok(())
}

/// Downloads a .gguf into `dir`, reporting `(done, total)` bytes. Saved as `.part` and renamed only
/// when complete, so an interrupted download never looks like a model.
pub async fn download(url: &str, dir: &Path, mut progress: impl FnMut(u64, u64)) -> Result<PathBuf> {
    let name = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .filter(|n| n.ends_with(".gguf"))
        .context("the URL must point to a .gguf file")?;
    let (dest, part) = (dir.join(name), dir.join(format!("{name}.part")));
    let http = crate::net::download_client();
    let mut resp =
        http.get(url).send().await.and_then(|r| r.error_for_status()).with_context(|| format!("downloading {url}"))?;
    let total = resp.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&part).await.with_context(|| {
        format!("{} is not writable; put a .gguf in it or mount it with write access", dir.display())
    })?;
    let mut done = 0u64;
    while let Some(chunk) = resp.chunk().await.with_context(|| format!("downloading {url}"))? {
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
        done += chunk.len() as u64;
        progress(done, total);
    }
    tokio::io::AsyncWriteExt::flush(&mut file).await?;
    if total > 0 && done != total {
        bail!("download ended early ({done} of {total} bytes); run it again to retry");
    }
    tokio::fs::rename(&part, &dest).await?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn file(name: &str, size: u64, hw: &Hardware) -> File {
        File { name: name.into(), size, mmproj: name.contains("mmproj"), fit: fit(size, hw), installed: false }
    }

    #[test]
    fn fit_compares_the_model_with_gpu_then_ram() {
        let gpu = Hardware { gpus: vec!["x".into()], vram: 12 * GB, ram: 32 * GB };
        assert_eq!(fit(5 * GB, &gpu), Fit::Fits);
        assert_eq!(fit(9 * GB, &gpu), Fit::Tight, "spills a few layers to the CPU");
        assert_eq!(fit(40 * GB, &gpu), Fit::No);
        let cpu = Hardware { gpus: vec![], vram: 0, ram: 16 * GB };
        assert_eq!(fit(5 * GB, &cpu), Fit::Fits);
        assert_eq!(fit(9 * GB, &cpu), Fit::Tight);
        let big = Hardware { gpus: vec![], vram: 0, ram: 64 * GB };
        assert_eq!(fit(9 * GB, &big), Fit::Tight, "runs, but slowly without a GPU");
        assert_eq!(fit(20 * GB, &cpu), Fit::No);
    }

    #[test]
    fn the_recommendation_is_the_largest_quant_that_fits_never_full_precision() {
        let hw = Hardware { gpus: vec!["x".into()], vram: 12 * GB, ram: 32 * GB };
        let files = [
            file("m-Q4_K_M.gguf", 5 * GB, &hw),
            file("m-Q8_0.gguf", 8 * GB, &hw),
            file("m-Q6_K.gguf", 7 * GB, &hw),
            file("m-f16.gguf", 9 * GB / 2, &hw),
            file("mmproj-m-f16.gguf", GB, &hw),
        ];
        assert_eq!(recommend(&files, &hw).unwrap().name, "m-Q8_0.gguf");
        let cpu = Hardware { gpus: vec![], vram: 0, ram: 32 * GB };
        let small = [file("s-Q4_K_M.gguf", 2 * GB, &cpu), file("s-Q8_0.gguf", 4 * GB, &cpu)];
        assert_eq!(recommend(&small, &cpu).unwrap().name, "s-Q4_K_M.gguf", "on a CPU, Q4 is faster");
        let small = Hardware { gpus: vec!["x".into()], vram: 4 * GB, ram: 16 * GB };
        let tight = [file("m-Q4_K_M.gguf", 5 * GB, &small), file("m-Q8_0.gguf", 8 * GB, &small)];
        assert_eq!(recommend(&tight, &small).unwrap().name, "m-Q4_K_M.gguf", "nothing fits: the smallest that runs");
        let none = Hardware { gpus: vec![], vram: 0, ram: 2 * GB };
        assert!(recommend(&[file("m-Q4_K_M.gguf", 5 * GB, &none)], &none).is_none());
    }

    #[test]
    fn repo_names_cannot_climb_out_of_the_models_folder() {
        assert!(check_repo("Qwen/Qwen3-8B-GGUF").is_ok());
        assert!(check_repo("bartowski/Llama-3.2-3B-Instruct-GGUF").is_ok());
        for bad in ["..", "../x", "a/..", "a/b/c", "a", "/etc", "a/.hidden", "a/b c", "a/"] {
            assert!(check_repo(bad).is_err(), "{bad}");
        }
        assert_eq!(folder("Qwen/Qwen3-8B-GGUF"), "Qwen__Qwen3-8B-GGUF");
    }

    #[test]
    fn pull_picks_the_named_quant_or_the_recommendation() {
        let hw = Hardware { gpus: vec![], vram: 0, ram: 64_000_000_000 };
        let files = [
            file("m-Q4_0_4_4.gguf", 2_000_000_000, &hw),
            file("m-Q4_0.gguf", 2_000_000_000, &hw),
            file("m-Q4_K_M.gguf", 2_000_000_000, &hw),
            file("m-Q8_0.gguf", 4_000_000_000, &hw),
        ];
        assert_eq!(pick_file(&files, Some("Q4_0"), &hw).unwrap(), "m-Q4_0.gguf");
        assert_eq!(pick_file(&files, Some("Q8_0"), &hw).unwrap(), "m-Q8_0.gguf");
        assert_eq!(pick_file(&files, None, &hw).unwrap(), "m-Q4_K_M.gguf");
        let err = pick_file(&files, Some("Q2_K"), &hw).unwrap_err();
        assert!(err.contains("m-Q8_0.gguf"), "{err}");
    }

    #[test]
    fn a_vision_model_brings_its_projector() {
        let listed = [("m.gguf".to_owned(), 9), ("mmproj-f16.gguf".to_owned(), 2), ("mmproj-f32.gguf".to_owned(), 4)];
        let dir = std::env::temp_dir().join("lair-no-such-dir");
        assert_eq!(with_projector(&listed, "m.gguf", &dir), ["m.gguf", "mmproj-f16.gguf"]);
    }

    #[test]
    fn split_models_are_recognized() {
        assert!(is_shard("Qwen3-235B-Q4_K_M-00001-of-00003.gguf"));
        assert!(!is_shard("Qwen3-8B-Q4_K_M.gguf"));
        assert!(!is_shard("model-of-00003.gguf"));
    }
}
