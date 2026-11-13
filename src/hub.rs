//! Models from Hugging Face: what is popular there right now, which of it runs well on this machine,
//! and which file of a repo to take. Downloading is in [`crate::download`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::shard_of;
use crate::hw::{Hardware, hardware};
use crate::lair::Lair;

const HF: &str = "https://huggingface.co";

/// Open GGUF repos lair vouches for, smallest first: shown when Hugging Face's listing can't be
/// reached, and always among the candidates for the recommendation.
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

/// Publishers whose repos hold a lab's own models, converted with care. Only their repos (and the
/// curated ones) can be "the best for this machine": popularity on the Hub is easy to inflate, and
/// the busiest converters also publish every fine-tune and merge there is.
const TRUSTED: &[&str] = &[
    "Qwen",
    "google",
    "ggml-org",
    "lmstudio-community",
    "unsloth",
    "microsoft",
    "mistralai",
    "meta-llama",
    "openai",
    "LiquidAI",
    "ibm-granite",
    "HuggingFaceTB",
];

/// Below this many tokens per second a model is not offered as the best, however capable.
const FAST_ENOUGH: f32 = 15.0;
/// How many repos' file lists one recommendation reads.
const MAX_CANDIDATES: usize = 30;
/// How long what Hugging Face said is reused: its listings change by the hour, not the minute.
const CACHE_TTL: Duration = Duration::from_secs(3600);
/// A page of the model listing carries each repo's chat template; this bounds what is read.
const LISTING_LIMIT: usize = 8 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
pub struct Repo {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
}

/// One file of a repo, as Hugging Face lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct Remote {
    /// Path inside the repo: `file.gguf` or `folder/file.gguf`.
    pub path: String,
    pub size: u64,
    /// The content hash Hugging Face stores for large files.
    pub sha256: Option<String>,
}

impl Remote {
    /// The name it is saved under: repos keep split models in folders, lair keeps a repo flat.
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

/// A model of a repo: one file, or every part of a split one.
#[derive(Clone, Debug, PartialEq)]
pub struct Listed {
    /// The file's name, or the first part's.
    pub name: String,
    pub size: u64,
    pub parts: Vec<Remote>,
}

/// A model of a repo, judged against this machine.
#[derive(Clone, Debug, Serialize)]
pub struct File {
    pub name: String,
    pub size: u64,
    pub mmproj: bool,
    pub fit: Fit,
    /// A model with this id is already in the catalog.
    pub installed: bool,
    /// Files it downloads as; more than one for a split model.
    pub parts: u32,
    /// Tokens per second it should reach here, estimated from models that ran; absent until one has.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tps: Option<f32>,
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

fn with_token(req: reqwest::RequestBuilder, token: Option<&str>) -> reqwest::RequestBuilder {
    match token {
        Some(t) => req.bearer_auth(t),
        None => req,
    }
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

/// A name lair will write to disk: one path segment, no leading dot, plain characters.
fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 160
        && name.starts_with(|c: char| c.is_ascii_alphanumeric())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.+".contains(c))
}

/// Whether a repo's file is a model lair can offer: a `.gguf` at the root, or a part of a split model
/// one folder down (where repos keep them). Other files in folders are auxiliary pieces, not models:
/// a 2 GB speculative-decoding head (`MTP/mtp-….gguf`) must never be picked as "the model that
/// fits" of a 60 GB one. Names come from the network, so anything that isn't a plain name is out too.
fn listable(path: &str) -> bool {
    let mut segments = path.rsplit('/');
    let name = segments.next().unwrap_or_default();
    let folders = segments.count();
    name.ends_with(".gguf")
        && path.split('/').all(safe_name)
        && !name.to_ascii_lowercase().starts_with("mtp-")
        && (folders == 0 || (folders == 1 && shard_of(name).is_some()))
}

/// The repo's GGUF models: see [`listable`].
pub async fn tree(http: &reqwest::Client, repo: &str, token: Option<&str>) -> Result<Vec<Remote>> {
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
        oid: Option<String>,
    }
    check_repo(repo)?;
    let req = http.get(format!("{HF}/api/models/{repo}/tree/main")).query(&[("recursive", "true")]);
    let resp = with_token(req, token).send().await?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND || resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        bail!("repo {repo} not found, or private (set a Hugging Face token to reach it)");
    }
    let entries: Vec<Entry> = resp.error_for_status()?.json().await.context("unexpected answer from Hugging Face")?;
    Ok(entries
        .into_iter()
        .filter(|e| listable(&e.path))
        .map(|e| {
            let hex = |h: &String| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit());
            let sha256 = e.lfs.as_ref().and_then(|l| l.oid.clone()).filter(hex);
            Remote { size: e.lfs.map_or(e.size, |l| l.size), path: e.path, sha256 }
        })
        .collect())
}

/// The models among a repo's files: a split one is one model, listed only if all its parts are there.
/// Two files with the same name in different folders would land on the same path: the first is kept.
pub fn group(remotes: Vec<Remote>) -> Vec<Listed> {
    let mut out: Vec<Listed> = Vec::new();
    let mut sets: HashMap<(String, u32), Vec<(u32, Remote)>> = HashMap::new();
    for r in remotes {
        match shard_of(r.name()) {
            Some((base, index, total)) => {
                let dir = r.path.rsplit_once('/').map_or("", |(d, _)| d);
                sets.entry((format!("{dir}/{base}"), total)).or_default().push((index, r));
            }
            None if out.iter().any(|l| l.name == r.name()) => {}
            None => out.push(Listed { name: r.name().to_owned(), size: r.size, parts: vec![r] }),
        }
    }
    for ((_, total), mut parts) in sets {
        parts.sort_by_key(|(i, _)| *i);
        parts.dedup_by_key(|(i, _)| *i);
        let whole = parts.len() == total as usize && parts.iter().map(|(i, _)| *i).eq(1..=total);
        let name = parts[0].1.name().to_owned();
        if whole && !out.iter().any(|l| l.name == name) {
            let parts: Vec<Remote> = parts.into_iter().map(|(_, r)| r).collect();
            out.push(Listed { name, size: parts.iter().map(|p| p.size).sum(), parts });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The id a downloaded file gets in the catalog: its name without `.gguf`, or a split model's base name.
fn model_id(name: &str) -> &str {
    shard_of(name).map_or_else(|| name.trim_end_matches(".gguf"), |(base, _, _)| base)
}

/// Names from Ollama's library and the Hugging Face repo each one means here: a fixed list of known
/// publishers, never a search, so a name can't be squatted into choosing what gets downloaded.
/// A name without a tag is Ollama's default size.
const OLLAMA_LIBRARY: &[(&str, &str)] = &[
    ("qwen3:0.6b", "Qwen/Qwen3-0.6B-GGUF"),
    ("qwen3:1.7b", "Qwen/Qwen3-1.7B-GGUF"),
    ("qwen3:4b", "Qwen/Qwen3-4B-GGUF"),
    ("qwen3", "Qwen/Qwen3-8B-GGUF"),
    ("qwen3:8b", "Qwen/Qwen3-8B-GGUF"),
    ("qwen3:14b", "Qwen/Qwen3-14B-GGUF"),
    ("qwen3:30b", "Qwen/Qwen3-30B-A3B-GGUF"),
    ("qwen3:32b", "Qwen/Qwen3-32B-GGUF"),
    ("qwen2.5-coder:1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"),
    ("qwen2.5-coder", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"),
    ("qwen2.5-coder:7b", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"),
    ("qwen2.5-coder:14b", "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"),
    ("qwen2.5-coder:32b", "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF"),
    ("qwen2.5vl", "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"),
    ("qwen2.5vl:7b", "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"),
    ("llama3.2:1b", "bartowski/Llama-3.2-1B-Instruct-GGUF"),
    ("llama3.2", "bartowski/Llama-3.2-3B-Instruct-GGUF"),
    ("llama3.2:3b", "bartowski/Llama-3.2-3B-Instruct-GGUF"),
    ("llama3.1", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
    ("llama3.1:8b", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
    ("gemma3:1b", "ggml-org/gemma-3-1b-it-GGUF"),
    ("gemma3", "ggml-org/gemma-3-4b-it-GGUF"),
    ("gemma3:4b", "ggml-org/gemma-3-4b-it-GGUF"),
    ("gemma3:12b", "ggml-org/gemma-3-12b-it-GGUF"),
    ("gemma3:27b", "ggml-org/gemma-3-27b-it-GGUF"),
    ("mistral", "bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
    ("mistral:7b", "bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
    ("phi4", "bartowski/phi-4-GGUF"),
    ("phi4:14b", "bartowski/phi-4-GGUF"),
    ("deepseek-r1:1.5b", "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"),
    ("deepseek-r1:7b", "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF"),
    ("deepseek-r1:14b", "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"),
    ("deepseek-r1:32b", "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"),
    ("gpt-oss", "ggml-org/gpt-oss-20b-GGUF"),
    ("gpt-oss:20b", "ggml-org/gpt-oss-20b-GGUF"),
    ("gpt-oss:120b", "ggml-org/gpt-oss-120b-GGUF"),
    ("nomic-embed-text", "nomic-ai/nomic-embed-text-v1.5-GGUF"),
    ("bge-m3", crate::memory::EMBED_REPO),
];

/// The repo an Ollama library name (`qwen3`, `llama3.2:3b`, `gemma3:latest`) stands for, if lair knows it.
pub fn ollama_repo(name: &str) -> Option<&'static str> {
    let name = name.trim().to_ascii_lowercase();
    let name = name.strip_suffix(":latest").unwrap_or(&name);
    OLLAMA_LIBRARY.iter().find(|(n, _)| *n == name).map(|(_, repo)| *repo)
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

/// Without a GPU, speed is bound by memory bandwidth: past this much weight replies crawl even if RAM fits.
const CPU_COMFORT: u64 = 5_000_000_000;

/// Weights plus the context's KV cache and runtime buffers, roughly.
fn needs(size: u64) -> u64 {
    size + size / 5 + 1_000_000_000
}

/// The share of a model's layers this machine would put on the GPU, before the model is here to
/// measure: all of it when it fits, else what fits of it.
pub fn offload_guess(size: u64, hw: &Hardware) -> f32 {
    if hw.vram == 0 {
        return 0.0;
    }
    (hw.vram as f64 * 0.9 / needs(size) as f64).min(1.0) as f32
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

/// Quantized below 4 bits a model loses enough that a smaller one at 4 bits answers better.
fn low_quant(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    ["Q2", "Q3", "IQ1", "IQ2", "IQ3", "TQ1", "TQ2"]
        .iter()
        .any(|q| n.contains(&format!("-{q}")) || n.contains(&format!(".{q}")))
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

/// A repo's models judged for this machine: `have` are the installed model ids, `dir` the repo's
/// folder (to see a projector already there), `speed` the estimate for a model of a given size.
fn judge(
    listed: &[Listed],
    hw: &Hardware,
    have: &[String],
    dir: Option<&Path>,
    speed: impl Fn(u64) -> Option<f32>,
) -> Vec<File> {
    listed
        .iter()
        .map(|l| {
            let mmproj = l.name.to_ascii_lowercase().contains("mmproj");
            let installed = if mmproj {
                dir.is_some_and(|d| d.join(&l.name).exists())
            } else {
                have.iter().any(|id| id == model_id(&l.name))
            };
            File {
                fit: fit(l.size, hw),
                tps: if mmproj { None } else { speed(l.size) },
                name: l.name.clone(),
                size: l.size,
                mmproj,
                installed,
                parts: l.parts.len() as u32,
            }
        })
        .collect()
}

fn judge_here(lair: &Lair, repo: &str, listed: &[Listed]) -> Vec<File> {
    let have: Vec<String> = lair.models.local().into_iter().map(|m| m.id).collect();
    let dir = lair.models.dir().map(|d| d.join(folder(repo)));
    judge(listed, &hardware(), &have, dir.as_deref(), |size| lair.models.predict(size).map(|p| p.tps))
}

/// A repo's files judged for this machine, with what is already installed marked.
pub async fn judged(lair: &Lair, repo: &str) -> Result<Vec<File>> {
    let listed = group(tree(&lair.http, repo, lair.config().hf_token.as_deref()).await?);
    Ok(judge_here(lair, repo, &listed))
}

/// The file of `repo` to download: the one `quant` names, else the best for this machine.
pub async fn choose(lair: &Lair, repo: &str, quant: Option<&str>) -> Result<String> {
    let files = judged(lair, repo).await?;
    let quant = quant.map(str::to_ascii_uppercase);
    pick_file(&files, quant.as_deref(), &hardware()).map_err(|e| anyhow::anyhow!("{repo}: {e}"))
}

/// `file`, plus the repo's smallest vision projector when it has one that isn't in `dir` yet.
pub fn with_projector(listed: &[Listed], file: &str, dir: &Path) -> Vec<Listed> {
    let is_proj = |l: &&Listed| l.name.to_ascii_lowercase().contains("mmproj");
    let mut queue: Vec<Listed> = listed.iter().filter(|l| l.name == file).cloned().collect();
    if !file.to_ascii_lowercase().contains("mmproj")
        && let Some(proj) = listed.iter().filter(is_proj).min_by_key(|l| l.size)
        && !dir.join(&proj.name).exists()
    {
        queue.push(proj.clone());
    }
    queue
}

/// Where a repo's file downloads from.
pub fn file_url(repo: &str, path: &str) -> String {
    format!("{HF}/{repo}/resolve/main/{path}")
}

/// Whether `url` is Hugging Face's: the only place a Hugging Face token is ever sent.
pub fn is_hf(url: &str) -> bool {
    url.starts_with(&format!("{HF}/"))
}

/// What `lair pull <spec>` downloads: `owner/repo`, `owner/repo:QUANT` or just `repo` (searched).
/// Returns the repo, its folder under `models` and the files to fetch, model first.
pub async fn plan(
    http: &reqwest::Client,
    spec: &str,
    models: &Path,
    token: Option<&str>,
) -> Result<(String, PathBuf, Vec<Remote>)> {
    let (name, quant) = match (ollama_repo(spec), spec.rsplit_once(':')) {
        (Some(repo), _) => (repo, None),
        (None, Some((n, q))) => (n, Some(q.to_ascii_uppercase())),
        (None, None) => (spec, None),
    };
    let repo = if name.contains('/') {
        name.to_owned()
    } else {
        let found = search(http, name).await?;
        let pick = found.iter().find(|r| r.id.rsplit('/').next().is_some_and(|n| n.eq_ignore_ascii_case(name)));
        pick.or(found.first()).with_context(|| format!("no GGUF repo on Hugging Face matches {name:?}"))?.id.clone()
    };
    let listed = group(tree(http, &repo, token).await?);
    let hw = hardware();
    let judged = judge(&listed, &hw, &[], None, |_| None);
    let file = pick_file(&judged, quant.as_deref(), &hw).map_err(|e| anyhow::anyhow!("{repo}: {e}"))?;
    let dir = models.join(folder(&repo));
    let queue = with_projector(&listed, &file, &dir).into_iter().flat_map(|l| l.parts).collect();
    Ok((repo, dir, queue))
}

/// The file a quant names, or the recommended one for this machine.
fn pick_file(files: &[File], quant: Option<&str>, hw: &Hardware) -> Result<String, String> {
    let models = || files.iter().filter(|f| !f.mmproj);
    match quant {
        Some(q) => {
            // `Q4_0` must pick `m-Q4_0.gguf` over `m-Q4_0_4_4.gguf`: an exact ending beats a substring.
            // A split model's name ends with its part number, so the quant is looked for in its id.
            let upper = |f: &&File| model_id(&f.name).to_ascii_uppercase();
            let exact = models().find(|f| upper(f).ends_with(q));
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
pub fn folder(repo: &str) -> String {
    repo.replace('/', "__")
}

/// A GGUF repo as Hugging Face's listing describes it.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Candidate {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default, rename = "trendingScore")]
    pub trending: f64,
    /// `false`, or how access is granted (`"auto"`, `"manual"`): a gated repo needs a token to download.
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    gguf: Option<Gguf>,
}

/// What the Hub read from a repo's GGUF header.
#[derive(Clone, Debug, Default, Deserialize)]
struct Gguf {
    /// Parameter count.
    #[serde(default)]
    total: u64,
    #[serde(default)]
    chat_template: Option<String>,
}

impl Candidate {
    fn gated(&self) -> bool {
        !matches!(self.gated, Value::Null | Value::Bool(false))
    }

    fn params(&self) -> u64 {
        self.gguf.as_ref().map_or(0, |g| g.total)
    }

    fn thinks(&self) -> bool {
        let template = self.gguf.as_ref().and_then(|g| g.chat_template.as_deref()).unwrap_or("");
        template.contains("<think>") || template.contains("enable_thinking")
    }

    fn owner(&self) -> &str {
        self.id.split('/').next().unwrap_or("")
    }

    fn curated(&self) -> Option<&'static str> {
        SUGGESTED.iter().find(|(repo, _)| *repo == self.id).map(|(_, about)| *about)
    }

    /// May be offered as the best: a curated repo, or a publisher lair trusts.
    fn trusted(&self) -> bool {
        self.curated().is_some() || TRUSTED.contains(&self.owner())
    }
}

/// One page of Hugging Face's GGUF listing: chat models (`text-generation`) or ones that also see
/// images (`image-text-to-text`), by trend or by downloads.
async fn listing(http: &reqwest::Client, task: &str, sort: &str) -> Result<Vec<Candidate>> {
    let url = format!(
        "{HF}/api/models?filter=gguf&pipeline_tag={task}&sort={sort}&limit=40\
         &expand[]=gguf&expand[]=gated&expand[]=downloads&expand[]=likes&expand[]=trendingScore"
    );
    let (_, body, cut) = crate::net::get_capped(http, &url, LISTING_LIMIT).await?;
    if cut {
        bail!("Hugging Face's listing is larger than expected");
    }
    serde_json::from_slice(&body).context("unexpected answer from Hugging Face")
}

/// The repos worth judging: what is trending and most downloaded on the Hub, plus the curated ones.
/// Offline (or if the Hub's listing changes shape) the curated ones alone.
async fn candidates(http: &reqwest::Client) -> Vec<Candidate> {
    let pages = tokio::join!(
        listing(http, "text-generation", "trendingScore"),
        listing(http, "text-generation", "downloads"),
        listing(http, "image-text-to-text", "trendingScore"),
    );
    let mut all: Vec<Candidate> = Vec::new();
    for page in [pages.0, pages.1, pages.2] {
        match page {
            Ok(found) => all.extend(found.into_iter().filter(|c| check_repo(&c.id).is_ok())),
            Err(e) => eprintln!("Hugging Face's listing is unavailable: {e:#}"),
        }
    }
    for (repo, _) in SUGGESTED {
        all.push(Candidate { id: (*repo).to_owned(), ..Default::default() });
    }
    // A repo found twice keeps the entry that knows more about it.
    all.sort_by(|a, b| a.id.cmp(&b.id).then(b.params().cmp(&a.params())));
    all.dedup_by(|b, a| a.id == b.id);
    // Trusted repos first, then by trend: the ones whose files get read.
    all.sort_by(|a, b| b.trusted().cmp(&a.trusted()).then(b.trending.total_cmp(&a.trending)));
    all.truncate(MAX_CANDIDATES);
    all
}

/// The candidates and each one's models.
type Survey = Vec<(Candidate, Vec<Listed>)>;

/// What Hugging Face last said.
#[derive(Default)]
pub struct Cache {
    found: Mutex<Option<(Instant, Survey)>>,
}

/// The candidates with their file lists, from the cache when it is fresh. A repo whose files can't be
/// listed right now is left out rather than failing the whole list.
async fn surveyed(http: &reqwest::Client, token: Option<&str>, cache: Option<&Cache>) -> Survey {
    if let Some(cache) = cache
        && let Some((at, found)) = &*cache.found.lock().unwrap()
        && at.elapsed() < CACHE_TTL
    {
        return found.clone();
    }
    let mut tasks = tokio::task::JoinSet::new();
    for c in candidates(http).await {
        let (http, token) = (http.clone(), token.map(str::to_owned));
        tasks.spawn(async move {
            let listed = tree(&http, &c.id, token.as_deref()).await.map(group);
            (c, listed)
        });
    }
    let mut found = Vec::new();
    while let Some(res) = tasks.join_next().await {
        if let Ok((c, Ok(listed))) = res {
            found.push((c, listed));
        }
    }
    // Nothing found is most likely a network failure: not worth remembering for an hour.
    if let Some(cache) = cache
        && !found.is_empty()
    {
        *cache.found.lock().unwrap() = Some((Instant::now(), found.clone()));
    }
    found
}

/// A repo with the file to take from it on this machine.
#[derive(Clone, Debug, Serialize)]
pub struct Pick {
    pub repo: String,
    /// What lair says about a curated repo; empty for the others.
    pub about: &'static str,
    /// The recommended file; `None` when nothing in the repo runs here.
    pub pick: Option<File>,
    pub vision: bool,
    pub thinks: bool,
    /// Parameter count; 0 when the Hub doesn't say.
    pub params: u64,
    pub downloads: u64,
    pub likes: u64,
    /// Why it is listed where it is, as codes the UI words: `fits`, `tight`, `fast`, `slow`,
    /// `trusted`, `trending`, `gated`, `low_quant`.
    pub why: Vec<&'static str>,
    /// The one to get: see [`rank`].
    pub best: bool,
}

/// Models of about the same size are equals; what trends among them wins. Half-octave classes:
/// 3B, 4B, 6B, 8B, 12B, 16B, 24B, 32B…
fn size_class(params: u64, bytes: u64) -> i32 {
    // The Hub doesn't always give the parameter count; a 4-bit file is about 0.6 bytes per parameter.
    let params = if params > 0 { params as f64 } else { bytes as f64 / 0.6 };
    ((params / 1e9).max(0.1).log2() * 2.0).round() as i32
}

/// Orders the candidates for this machine and marks the best: **the biggest model from a publisher
/// lair trusts that runs fully on this machine and fast, at 4 bits or better; what is trending
/// breaks ties.** "Fast" is [`FAST_ENOUGH`] estimated tokens per second, checked only once some model
/// has run here to estimate from. The rest follow by trend, the ones that don't run here last.
pub fn rank(found: Vec<(Candidate, Vec<File>)>, hw: &Hardware, has_token: bool) -> Vec<Pick> {
    let mut picks: Vec<(bool, i32, f64, Pick)> = found
        .into_iter()
        .map(|(c, files)| {
            let pick = recommend(&files, hw).cloned();
            let mut why = Vec::new();
            let mut eligible = c.trusted() && (has_token || !c.gated());
            if c.trusted() {
                why.push("trusted");
            }
            if c.gated() {
                why.push("gated");
            }
            if c.trending >= 10.0 {
                why.push("trending");
            }
            match &pick {
                Some(f) => {
                    why.push(if f.fit == Fit::Fits { "fits" } else { "tight" });
                    eligible &= f.fit == Fit::Fits;
                    if low_quant(&f.name) {
                        why.push("low_quant");
                        eligible = false;
                    }
                    if let Some(tps) = f.tps {
                        why.push(if tps >= FAST_ENOUGH { "fast" } else { "slow" });
                        eligible &= tps >= FAST_ENOUGH;
                    }
                }
                None => eligible = false,
            }
            let class = pick.as_ref().map_or(i32::MIN, |f| size_class(c.params(), f.size));
            let vision = files.iter().any(|f| f.mmproj);
            let p = Pick {
                about: c.curated().unwrap_or(""),
                thinks: c.thinks(),
                params: c.params(),
                downloads: c.downloads,
                likes: c.likes,
                repo: c.id.clone(),
                pick,
                vision,
                why,
                best: false,
            };
            (eligible, class, c.trending, p)
        })
        .collect();
    // Eligible first, biggest class first, then by trend; among the rest, what runs here comes first.
    picks.sort_by(|a, b| {
        let runs = |p: &Pick| p.pick.is_some();
        (b.0, runs(&b.3)).cmp(&(a.0, runs(&a.3))).then_with(|| {
            let by_class = if a.0 { b.1.cmp(&a.1) } else { std::cmp::Ordering::Equal };
            by_class.then(b.2.total_cmp(&a.2)).then_with(|| a.3.repo.cmp(&b.3.repo))
        })
    });
    let best = picks.first().is_some_and(|p| p.0);
    let mut picks: Vec<Pick> = picks.into_iter().map(|(_, _, _, p)| p).collect();
    if let Some(first) = picks.first_mut() {
        first.best = best;
    }
    picks
}

/// The models to offer on this machine, best first.
pub async fn suggested(lair: &Lair) -> Vec<Pick> {
    let token = lair.config().hf_token;
    let found = surveyed(&lair.http, token.as_deref(), Some(&lair.hub)).await;
    let judged = found.into_iter().map(|(c, listed)| {
        let files = judge_here(lair, &c.id, &listed);
        (c, files)
    });
    rank(judged.collect(), &hardware(), token.is_some())
}

/// The model a first run downloads into `models`: the best for this machine no bigger than `max`
/// bytes, with the files to fetch (model first, then its projector) and the reasons, to tell the user.
pub async fn first_model(
    http: &reqwest::Client,
    token: Option<&str>,
    max: u64,
    models: &Path,
) -> Result<(String, Vec<Remote>, Vec<&'static str>)> {
    let hw = hardware();
    let found = surveyed(http, token, None).await;
    let lists: HashMap<String, Vec<Listed>> = found.iter().map(|(c, l)| (c.id.clone(), l.clone())).collect();
    let judged = found
        .into_iter()
        .map(|(c, listed)| {
            let files = judge(&listed, &hw, &[], None, |_| None);
            (c, files.into_iter().filter(|f| f.mmproj || f.size <= max).collect())
        })
        .collect();
    let best = rank(judged, &hw, token.is_some()).into_iter().find(|p| p.best).context("nothing suitable found")?;
    let file = best.pick.context("the best pick has a file")?;
    let dir = models.join(folder(&best.repo));
    let queue = with_projector(&lists[&best.repo], &file.name, &dir).into_iter().flat_map(|l| l.parts);
    Ok((best.repo, queue.collect(), best.why))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn machine(vram: u64, ram: u64) -> Hardware {
        Hardware { vram, ram, ..Default::default() }
    }

    fn file(name: &str, size: u64, hw: &Hardware) -> File {
        File {
            name: name.into(),
            size,
            mmproj: name.contains("mmproj"),
            fit: fit(size, hw),
            installed: false,
            parts: 1,
            tps: None,
        }
    }

    fn remote(path: &str, size: u64) -> Remote {
        Remote { path: path.into(), size, sha256: None }
    }

    #[test]
    fn fit_compares_the_model_with_gpu_then_ram() {
        let gpu = machine(12 * GB, 32 * GB);
        assert_eq!(fit(5 * GB, &gpu), Fit::Fits);
        assert_eq!(fit(9 * GB, &gpu), Fit::Tight, "spills a few layers to the CPU");
        assert_eq!(fit(40 * GB, &gpu), Fit::No);
        let cpu = machine(0, 16 * GB);
        assert_eq!(fit(5 * GB, &cpu), Fit::Fits);
        assert_eq!(fit(9 * GB, &cpu), Fit::Tight);
        let big = machine(0, 64 * GB);
        assert_eq!(fit(9 * GB, &big), Fit::Tight, "runs, but slowly without a GPU");
        assert_eq!(fit(20 * GB, &cpu), Fit::No);
        assert_eq!(offload_guess(5 * GB, &gpu), 1.0);
        assert!((0.3..0.9).contains(&offload_guess(20 * GB, &gpu)));
        assert_eq!(offload_guess(5 * GB, &cpu), 0.0);
    }

    #[test]
    fn the_recommendation_is_the_largest_quant_that_fits_never_full_precision() {
        let hw = machine(12 * GB, 32 * GB);
        let files = [
            file("m-Q4_K_M.gguf", 5 * GB, &hw),
            file("m-Q8_0.gguf", 8 * GB, &hw),
            file("m-Q6_K.gguf", 7 * GB, &hw),
            file("m-f16.gguf", 9 * GB / 2, &hw),
            file("mmproj-m-f16.gguf", GB, &hw),
        ];
        assert_eq!(recommend(&files, &hw).unwrap().name, "m-Q8_0.gguf");
        let cpu = machine(0, 32 * GB);
        let small = [file("s-Q4_K_M.gguf", 2 * GB, &cpu), file("s-Q8_0.gguf", 4 * GB, &cpu)];
        assert_eq!(recommend(&small, &cpu).unwrap().name, "s-Q4_K_M.gguf", "on a CPU, Q4 is faster");
        let small = machine(4 * GB, 16 * GB);
        let tight = [file("m-Q4_K_M.gguf", 5 * GB, &small), file("m-Q8_0.gguf", 8 * GB, &small)];
        assert_eq!(recommend(&tight, &small).unwrap().name, "m-Q4_K_M.gguf", "nothing fits: the smallest that runs");
        let none = machine(0, 2 * GB);
        assert!(recommend(&[file("m-Q4_K_M.gguf", 5 * GB, &none)], &none).is_none());
    }

    #[test]
    fn names_from_the_network_cannot_climb_out_of_the_models_folder() {
        assert!(check_repo("Qwen/Qwen3-8B-GGUF").is_ok());
        assert_eq!(ollama_repo("Qwen3:Latest"), Some("Qwen/Qwen3-8B-GGUF"));
        assert_eq!(ollama_repo("llama3.2:1b"), Some("bartowski/Llama-3.2-1B-Instruct-GGUF"));
        assert_eq!(ollama_repo("llama3"), None);
        assert!(OLLAMA_LIBRARY.iter().all(|(_, repo)| check_repo(repo).is_ok()));
        assert!(check_repo("bartowski/Llama-3.2-3B-Instruct-GGUF").is_ok());
        for bad in ["..", "../x", "a/..", "a/b/c", "a", "/etc", "a/.hidden", "a/b c", "a/"] {
            assert!(check_repo(bad).is_err(), "{bad}");
        }
        assert_eq!(folder("Qwen/Qwen3-8B-GGUF"), "Qwen__Qwen3-8B-GGUF");
        for bad in ["", "..", ".hidden.gguf", "a b.gguf", "a\\b.gguf", "a/b.gguf"] {
            assert!(!safe_name(bad), "{bad}");
        }
        assert!(safe_name("Qwen3-8B-Q4_K_M.gguf") && safe_name("IQ4_XS"));
        for model in ["m-Q4_K_M.gguf", "Q8_0/m-Q8_0-00001-of-00006.gguf"] {
            assert!(listable(model), "{model}");
        }
        let not_models = [
            "MTP/mtp-m-Q4_K_M.gguf",
            "mtp-m-Q4_K_M.gguf",
            "extra/m.gguf",
            "a/b/m-00001-of-00002.gguf",
            "../m.gguf",
            "m.bin",
            ".hidden/m-00001-of-00002.gguf",
        ];
        for other in not_models {
            assert!(!listable(other), "{other}");
        }
        assert!(is_hf("https://huggingface.co/a/b/resolve/main/m.gguf"));
        assert!(!is_hf("https://huggingface.co.evil.example/m.gguf") && !is_hf("http://huggingface.co/m.gguf"));
    }

    #[test]
    fn pull_picks_the_named_quant_or_the_recommendation() {
        let hw = machine(0, 64_000_000_000);
        let files = [
            file("m-Q4_0_4_4.gguf", 2_000_000_000, &hw),
            file("m-Q4_0.gguf", 2_000_000_000, &hw),
            file("m-Q4_K_M.gguf", 2_000_000_000, &hw),
            file("m-Q8_0.gguf", 4_000_000_000, &hw),
            file("big-Q6_K-00001-of-00002.gguf", 4_000_000_000, &hw),
        ];
        assert_eq!(pick_file(&files, Some("Q4_0"), &hw).unwrap(), "m-Q4_0.gguf");
        assert_eq!(pick_file(&files, Some("Q8_0"), &hw).unwrap(), "m-Q8_0.gguf");
        assert_eq!(pick_file(&files, Some("Q6_K"), &hw).unwrap(), "big-Q6_K-00001-of-00002.gguf");
        assert_eq!(pick_file(&files, None, &hw).unwrap(), "m-Q4_K_M.gguf");
        let err = pick_file(&files, Some("Q2_K"), &hw).unwrap_err();
        assert!(err.contains("m-Q8_0.gguf"), "{err}");
    }

    #[test]
    fn a_vision_model_brings_its_projector() {
        let listed = group(vec![remote("m.gguf", 9), remote("mmproj-f16.gguf", 2), remote("mmproj-f32.gguf", 4)]);
        let dir = std::env::temp_dir().join("lair-no-such-dir");
        let names: Vec<String> = with_projector(&listed, "m.gguf", &dir).into_iter().map(|l| l.name).collect();
        assert_eq!(names, ["m.gguf", "mmproj-f16.gguf"]);
    }

    #[test]
    fn a_split_model_is_one_model_when_all_its_parts_are_listed() {
        let listed = group(vec![
            remote("Q4/big-Q4-00002-of-00003.gguf", 5),
            remote("Q4/big-Q4-00001-of-00003.gguf", 5),
            remote("Q4/big-Q4-00003-of-00003.gguf", 2),
            remote("Q8/big-Q8-00001-of-00002.gguf", 9),
            remote("small.gguf", 3),
            remote("copy/small.gguf", 4),
        ]);
        let names: Vec<(&str, u64, usize)> = listed.iter().map(|l| (l.name.as_str(), l.size, l.parts.len())).collect();
        assert_eq!(names, [("big-Q4-00001-of-00003.gguf", 12, 3), ("small.gguf", 3, 1)], "an incomplete set is hidden");
        assert_eq!(listed[0].parts[1].path, "Q4/big-Q4-00002-of-00003.gguf");
        assert_eq!((model_id("big-Q4-00001-of-00003.gguf"), model_id("small.gguf")), ("big-Q4", "small"));
        let hw = machine(0, 64 * GB);
        let files = judge(&listed, &hw, &["big-Q4".into()], None, |size| Some(size as f32));
        assert!(files[0].installed && files[0].parts == 3 && !files[1].installed);
        assert_eq!(files[1].tps, Some(3.0));
    }

    fn cand(id: &str, params: u64, trending: f64) -> Candidate {
        let gguf = Some(Gguf { total: params, chat_template: None });
        Candidate { id: id.into(), trending, gguf, ..Default::default() }
    }

    #[test]
    fn the_best_is_the_biggest_trusted_model_that_runs_fully_and_fast() {
        let hw = machine(12 * GB, 32 * GB);
        let q4 = |name: &str, size: u64, tps: f32| File { tps: Some(tps), ..file(name, size, &hw) };
        let found = vec![
            (cand("Qwen/Small-GGUF", 4 * GB, 5.0), vec![q4("s-Q4_K_M.gguf", 3 * GB, 90.0)]),
            (cand("Qwen/Mid-GGUF", 8 * GB, 2.0), vec![q4("m-Q4_K_M.gguf", 5 * GB, 50.0)]),
            (cand("google/Mid2-GGUF", 8 * GB, 40.0), vec![q4("m2-Q4_K_M.gguf", 5 * GB, 50.0)]),
            (cand("Qwen/Slow-GGUF", 14 * GB, 9.0), vec![q4("slow-Q4_K_M.gguf", 8 * GB, 6.0)]),
            (cand("Qwen/Huge-GGUF", 70 * GB, 99.0), vec![q4("h-Q4_K_M.gguf", 60 * GB, 20.0)]),
            (cand("Qwen/Crushed-GGUF", 30 * GB, 9.0), vec![q4("c-Q2_K.gguf", 8 * GB, 30.0)]),
            (cand("someone/Hyped-GGUF", 12 * GB, 500.0), vec![q4("x-Q4_K_M.gguf", 7 * GB, 40.0)]),
            (
                Candidate { gated: Value::String("auto".into()), ..cand("google/Gated-GGUF", 12 * GB, 9.0) },
                vec![q4("g-Q4_K_M.gguf", 7 * GB, 40.0)],
            ),
        ];
        let ranked = rank(found.clone(), &hw, false);
        let order: Vec<&str> = ranked.iter().map(|p| p.repo.as_str()).collect();
        assert_eq!(order[..3], ["google/Mid2-GGUF", "Qwen/Mid-GGUF", "Qwen/Small-GGUF"], "{order:?}");
        assert!(ranked[0].best && ranked.iter().filter(|p| p.best).count() == 1);
        assert!(ranked[0].why.contains(&"fast") && ranked[0].why.contains(&"trending"));
        assert_eq!(order[3], "someone/Hyped-GGUF", "popular but not vouched for: listed, never the best");
        assert_eq!(*order.last().unwrap(), "Qwen/Huge-GGUF", "what doesn't run here goes last");
        let gated = ranked.iter().find(|p| p.repo == "google/Gated-GGUF").unwrap();
        assert!(gated.why.contains(&"gated"));

        let with_token = rank(found, &hw, true);
        assert_eq!(with_token[0].repo, "google/Gated-GGUF", "with a token, a gated repo can be the best");
        assert!(rank(vec![], &hw, false).is_empty());
        let nothing =
            rank(vec![(cand("Qwen/Huge-GGUF", 70 * GB, 1.0), vec![file("h.gguf", 90 * GB, &hw)])], &hw, false);
        assert!(!nothing[0].best && nothing[0].pick.is_none());
    }
}
