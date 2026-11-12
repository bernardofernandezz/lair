mod agent;
mod assets;
mod attach;
mod chat;
mod config;
mod download;
mod engine;
mod hub;
mod hw;
mod jail;
mod lair;
mod memory;
mod models;
mod net;
mod peers;
mod server;
mod skills;
mod speed;
mod sse;
mod store;
mod turn;
mod update;

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use agent::Approvals;
use anyhow::Context;
use assets::Assets;
use config::Config;
use engine::{Engine, Msg, Params, Reasoning, Tune};
use lair::Lair;
use models::Models;
use peers::Peers;
use skills::Catalog;
use store::Store;

/// lair — run LLMs locally (GGUF via llama.cpp). Nothing leaves this machine.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
// Parsed once at startup; boxing `Serve`'s flags would only make the clap derive harder to read.
#[allow(clippy::large_enum_variant)]
enum Cmd {
    /// Interactive chat in the terminal
    #[command(visible_alias = "run")]
    Chat {
        /// Model to chat with: any part of its name (default: the first one found)
        name: Option<String>,
        #[command(flatten)]
        model: ModelArgs,
        /// System prompt
        #[arg(long)]
        system: Option<String>,
    },
    /// Download a model: a Hugging Face repo (`Qwen/Qwen3-8B-GGUF`, `...:Q5_K_M`, or a name to search) or a .gguf URL
    Pull {
        /// `owner/repo`, `owner/repo:QUANT`, a repo name to search for, or the URL of a .gguf file
        model: String,
        /// Folder to save it in (default: ~/.local/share/lair/models, or %LOCALAPPDATA%\lair\models on Windows)
        #[arg(short = 'm', long = "models")]
        models: Option<PathBuf>,
        /// Hugging Face token, for gated or private repos
        #[arg(long, env = "HF_TOKEN", hide_env_values = true)]
        hf_token: Option<String>,
    },
    /// Delete a model from the models folder
    #[command(visible_alias = "remove")]
    Rm {
        /// The model's name as `lair list` shows it, or any part of it that matches only one
        name: String,
        /// Folder to delete from (default: ~/.local/share/lair/models, or %LOCALAPPDATA%\lair\models on Windows)
        #[arg(short = 'm', long = "models")]
        models: Option<PathBuf>,
    },
    /// Measure how fast a model runs on this machine; lair uses it to estimate the speed of others
    Bench {
        /// Model to measure: any part of its name (default: the first one found)
        name: Option<String>,
        #[command(flatten)]
        model: ModelArgs,
        /// Where the measurement is kept (default: ~/.local/share/lair, or %LOCALAPPDATA%\lair on Windows)
        #[arg(long, env = "LAIR_DATA")]
        data: Option<PathBuf>,
    },
    /// List the models in the models folder
    List {
        /// Folder to look in (default: ~/.local/share/lair/models, or %LOCALAPPDATA%\lair\models on Windows)
        #[arg(short = 'm', long = "models")]
        models: Option<PathBuf>,
    },
    /// Web UI and HTTP API (OpenAI-style under /v1, Ollama-style under /api).
    /// The UI is embedded at build time: run `bun install && bun run build` in web/ before `cargo build --release`.
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        skills: SkillArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Require `Authorization: Bearer <token>` on the API
        #[arg(long, env = "LAIR_TOKEN")]
        token: Option<String>,
        /// Where conversations and installed skills are kept (default: ~/.local/share/lair, or
        /// %LOCALAPPDATA%\lair on Windows)
        #[arg(long, env = "LAIR_DATA")]
        data: Option<PathBuf>,
        /// Folder the agent's file and shell tools work in (default: <data>/workspace)
        #[arg(long, env = "LAIR_WORKSPACE")]
        workspace: Option<PathBuf>,
        /// Base URL of a SearXNG instance for the agent's web search, e.g. http://localhost:8888
        #[arg(long, env = "LAIR_SEARXNG")]
        searxng: Option<String>,
        /// Hugging Face token, for downloading gated or private models
        #[arg(long, env = "HF_TOKEN", hide_env_values = true)]
        hf_token: Option<String>,
        #[command(flatten)]
        network: NetworkArgs,
    },
    /// Update lair to the latest release
    #[command(visible_alias = "self-update")]
    Update,
}

#[derive(Args)]
struct NetworkArgs {
    /// Another lair node to use, as host[:port] (repeatable); nodes on your tailnet are found on their own
    #[arg(long = "peer", env = "LAIR_PEERS", value_delimiter = ',')]
    peers: Vec<String>,
    /// tailscaled's socket, for finding and trusting other nodes on the tailnet
    #[arg(long, env = "LAIR_TAILSCALE_SOCKET", default_value = "/var/run/tailscale/tailscaled.sock")]
    tailscale_socket: PathBuf,
    /// This node's name on the network (default: its Tailscale host name)
    #[arg(long, env = "LAIR_NAME")]
    name: Option<String>,
}

impl NetworkArgs {
    /// The peers finder and this node's automatic name: its Tailscale host name, else the host name.
    async fn peers(&self, port: u16, http: reqwest::Client) -> (Peers, String) {
        let socket = self.tailscale_socket.exists().then(|| self.tailscale_socket.clone());
        let tailnet = socket.is_some();
        let peers = Peers::new(socket, port, http);
        let auto = match peers.status().await {
            Some(s) if !s.host.is_empty() => s.host,
            _ => std::fs::read_to_string("/etc/hostname")
                .map(|h| h.trim().to_owned())
                .ok()
                .or_else(|| std::env::var("COMPUTERNAME").ok())
                .unwrap_or_default(),
        };
        let auto = if auto.is_empty() { "lair".to_owned() } else { auto };
        eprintln!("node name {auto}{}", if tailnet { ", looking for peers on the tailnet" } else { "" });
        (peers, auto)
    }
}

#[derive(Args)]
struct SkillArgs {
    /// Turn off skills (prompt packages added when a message asks for them)
    #[arg(long)]
    no_skills: bool,
    /// Read-only folder with extra skills; a skill with the same id replaces the built-in
    #[arg(long, env = "LAIR_SKILLS")]
    skills_dir: Option<PathBuf>,
    /// URL of a JSON index of skills to browse and install from the UI
    #[arg(long, env = "LAIR_SKILLS_INDEX")]
    skills_index: Option<String>,
    /// Folders or files with .glb/.gltf models that 3D scenes may use (repeatable)
    #[arg(long, env = "LAIR_ASSETS")]
    assets: Vec<PathBuf>,
}

impl SkillArgs {
    /// Skills created or installed here live in `<data>/skills`.
    fn catalog(&self, data: &std::path::Path) -> Catalog {
        if self.no_skills { Catalog::default() } else { Catalog::load(self.skills_dir.clone(), data.join("skills")) }
    }
}

#[derive(Args)]
struct ModelArgs {
    /// .gguf model files, or directories to search for them (repeatable; default: <data>/models)
    #[arg(short = 'm', long = "models", visible_alias = "model")]
    models: Vec<PathBuf>,
    /// Layers to offload to the GPU, for every model (default: as many as fit, per model; 0 = CPU only)
    #[arg(long)]
    gpu_layers: Option<u32>,
    /// Context size in tokens, for every model (default: per model, what fits, from 8192 up to 32768)
    #[arg(long)]
    ctx: Option<u32>,
    /// Threads to generate with (default: one per physical core, up to 16)
    #[arg(long)]
    threads: Option<i32>,
    #[arg(long, default_value_t = 0.7)]
    temp: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// Sample only among the k likeliest tokens (0 = all)
    #[arg(long, default_value_t = 0)]
    top_k: i32,
    /// Drop tokens less likely than this share of the likeliest one (0 = off)
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// Penalty on tokens the reply already used (1 = off)
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    /// Max tokens per reply, thinking included
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,
    /// How much reasoning models think: none, low (~512 tokens) or high (unlimited)
    #[arg(long)]
    reasoning: Option<Reasoning>,
    /// Model downloaded into the first models folder when no .gguf is found there: "auto" picks the
    /// best one for this machine from Hugging Face, a URL names one, "" never downloads
    #[arg(long, env = "LAIR_DEFAULT_MODEL", default_value = "auto")]
    default_model: String,
}

/// What a first run downloads when Hugging Face can't be asked for something better.
const FALLBACK_MODEL: &str = "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf";
/// A first run never downloads more than this without being asked to: bigger models are a choice.
const FIRST_RUN_MAX: u64 = 6_000_000_000;
/// How long a first run waits for Hugging Face before settling for the fallback.
const FIRST_RUN_PATIENCE: std::time::Duration = std::time::Duration::from_secs(15);

impl ModelArgs {
    /// Without `-m`, models live in `<data>/models`, created on first use.
    fn resolve(&mut self, data: &std::path::Path) -> Result<()> {
        if self.models.is_empty() {
            self.models.push(models_dir(Some(data.join("models")))?);
        }
        Ok(())
    }

    /// First run with an empty models folder: download a model so lair starts with something to chat with.
    async fn ensure_model(&self, token: Option<&str>) -> Result<()> {
        let mut found = Vec::new();
        for p in &self.models {
            let _ = engine::collect_gguf(p, &mut found);
        }
        if !found.is_empty() || self.default_model.is_empty() {
            return Ok(());
        }
        let dir = self
            .models
            .iter()
            .find(|p| p.is_dir())
            .context("no .gguf found and no models folder to download one into")?;
        if self.default_model != models::AUTO {
            eprintln!("no models in {}; downloading the default one", dir.display());
            return download(&self.default_model, dir, token).await.map(drop);
        }
        eprintln!("no models in {}; asking Hugging Face for the best one for this machine", dir.display());
        let http = net::client();
        let best = tokio::time::timeout(FIRST_RUN_PATIENCE, hub::first_model(&http, token, FIRST_RUN_MAX, dir)).await;
        match best {
            Ok(Ok((repo, files, why))) => {
                eprintln!("picked {repo} ({})", why.join(", "));
                fetch_all(&repo, &dir.join(hub::folder(&repo)), &files, token).await
            }
            Ok(Err(e)) => {
                eprintln!("couldn't choose ({e:#}); downloading the default model instead");
                download(FALLBACK_MODEL, dir, token).await.map(drop)
            }
            Err(_) => {
                eprintln!("Hugging Face took too long; downloading the default model instead");
                download(FALLBACK_MODEL, dir, token).await.map(drop)
            }
        }
    }

    fn tune(&self) -> Tune {
        Tune { gpu_layers: self.gpu_layers, ctx: self.ctx, threads: self.threads }
    }

    /// The engine and the default generation params. `must_chat` refuses to start with no chat model:
    /// the terminal has nothing to do then, while the server can still download one.
    fn load(&self, must_chat: bool) -> Result<(Engine, Params)> {
        let engine = Engine::new(&self.models, self.tune())?;
        for m in engine.models() {
            let kind = if m.meta.embedding {
                " (embeddings)"
            } else if m.thinks {
                " (reasons)"
            } else {
                ""
            };
            eprintln!("found {}{kind}", m.id);
        }
        if must_chat && !engine.models().iter().any(|m| !m.meta.embedding) {
            anyhow::bail!("no usable .gguf chat models in {:?}", self.models);
        }
        let params = Params {
            temperature: self.temp,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            seed: None,
            reasoning: self.reasoning.unwrap_or_default(),
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            stop: Vec::new(),
            grammar: None,
            raw: false,
        };
        Ok((engine, params))
    }
}

/// The chat model whose name contains `name`, or the first one.
fn find_model(engine: &Engine, name: Option<&str>) -> Result<String> {
    let models = engine.models();
    let mut chat = models.iter().filter(|m| !m.meta.embedding);
    match name {
        Some(n) => chat.find(|m| m.id.contains(n)).with_context(|| format!("no model matching {n:?}; see `lair list`")),
        None => chat.next().context("no chat model found"),
    }
    .map(|m| m.id.clone())
}

/// `dir`, or `<data>/models` by default; created if missing.
fn models_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = dir.map_or_else(|| default_data_dir().map(|d| d.join("models")), Ok)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// Prints a download's progress every 5%.
fn progress() -> impl FnMut(u64, u64) {
    let mut shown = 0;
    move |done, total| {
        if total > 0 && done * 20 / total > shown {
            shown = done * 20 / total;
            eprintln!("  {}% of {} MB", shown * 5, total / 1_000_000);
        }
    }
}

/// Downloads a .gguf into `dir`.
async fn download(url: &str, dir: &std::path::Path, token: Option<&str>) -> Result<PathBuf> {
    eprintln!("downloading {url}");
    let dest = download::download(url, dir, token, progress()).await?;
    eprintln!("saved {}", dest.display());
    Ok(dest)
}

/// Downloads a repo's `files` into `dir`, resuming partial ones and checking each against its hash.
async fn fetch_all(repo: &str, dir: &std::path::Path, files: &[hub::Remote], token: Option<&str>) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let http = net::download_client();
    for f in files {
        let (url, dest) = (hub::file_url(repo, &f.path), dir.join(f.name()));
        eprintln!("downloading {url}");
        download::fetch(&http, &url, &dest, f.sha256.as_deref(), token, progress()).await?;
        eprintln!("saved {}", dest.display());
    }
    Ok(())
}

/// `lair pull`: a URL downloads as is; anything else names a Hugging Face repo, and the file that fits
/// this machine best (or the quant asked for) comes down with its vision projector, if it has one.
async fn pull(spec: &str, models: &std::path::Path, token: Option<&str>) -> Result<()> {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        return download(spec, models, token).await.map(drop);
    }
    let (repo, dir, files) = hub::plan(&net::client(), spec, models, token).await?;
    eprintln!("using {repo}");
    fetch_all(&repo, &dir, &files, token).await
}

/// `lair rm`: deletes the one model in `dir` that `name` names.
fn rm(name: &str, dir: &std::path::Path) -> Result<()> {
    let mut found = Vec::new();
    engine::collect_gguf(dir, &mut found)?;
    // A model is its file, or the first part of a split one; projectors go with their model.
    let models: Vec<(String, PathBuf)> = found
        .into_iter()
        .filter_map(|p| {
            let file = p.file_name()?.to_string_lossy().into_owned();
            let id = match engine::shard_of(&file) {
                Some((base, 1, _)) => base.to_owned(),
                Some(_) => return None,
                None => file.trim_end_matches(".gguf").to_owned(),
            };
            (!file.to_ascii_lowercase().contains("mmproj")).then_some((id, p))
        })
        .collect();
    let exact: Vec<_> = models.iter().filter(|(id, _)| id == name).collect();
    let matches = if exact.is_empty() { models.iter().filter(|(id, _)| id.contains(name)).collect() } else { exact };
    match matches.as_slice() {
        [(id, path)] => {
            let freed = download::remove(dir, path)?;
            println!("deleted {id} ({:.1} GB freed)", freed as f64 / 1e9);
            Ok(())
        }
        [] => anyhow::bail!("no model matching {name:?} in {}; see `lair list`", dir.display()),
        many => {
            let names: Vec<&str> = many.iter().map(|(id, _)| id.as_str()).collect();
            anyhow::bail!("{name:?} matches several models, name one: {}", names.join(", "))
        }
    }
}

/// `lair bench`: generates a fixed amount with the model as lair would run it, and again on the CPU
/// alone when a GPU took part, so both sides of the speed estimate get a measurement.
fn bench(mut engine: Engine, id: &str, params: &Params, store: &Store) -> Result<()> {
    const TOKENS: usize = 96;
    let msgs = [Msg {
        role: "user".into(),
        content: "Write a long, detailed essay about the history of computing.".into(),
        ..Default::default()
    }];
    let p = Params { max_tokens: TOKENS, reasoning: Reasoning::Off, temperature: 0.0, ..params.clone() };
    let mut bench: speed::Bench = store.kv("bench")?.and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
    let mut measure = |engine: &Engine, label: &str| -> Result<f32> {
        eprintln!("measuring {id} {label}…");
        let done = engine.generate(id, &msgs, &p, |_| true)?;
        let l = engine.loaded().into_iter().find(|l| l.id == id).context("the model just ran")?;
        let offload = l.gpu_layers as f32 / l.layers.max(1) as f32;
        bench.record(id, l.size, offload, done.tps());
        println!(
            "{id} {label}: {:.1} tokens/s ({} of {} layers on the GPU, context {}, loaded in {:.1} s)",
            done.tps(),
            l.gpu_layers,
            l.layers,
            l.ctx,
            done.load_us as f64 / 1e6
        );
        Ok(offload)
    };
    let offload = measure(&engine, "as lair runs it")?;
    if offload > 0.0 {
        // Kept under its own name: one sample per model would let the CPU run replace the GPU one.
        engine.set_tune(Tune { gpu_layers: Some(0), ..engine.tune() });
        let done = {
            eprintln!("measuring {id} on the CPU alone…");
            engine.generate(id, &msgs, &p, |_| true)?
        };
        let size = engine.loaded().into_iter().find(|l| l.id == id).map_or(0, |l| l.size);
        bench.record(&format!("{id} (cpu)"), size, 0.0, done.tps());
        println!("{id} on the CPU alone: {:.1} tokens/s", done.tps());
    }
    store.save_kv("bench", &serde_json::to_value(&bench)?)?;
    println!("saved; lair now estimates other models' speed on this machine from this");
    Ok(())
}

fn default_data_dir() -> Result<PathBuf> {
    if cfg!(windows) {
        let base = std::env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set; pass --data")?;
        return Ok(PathBuf::from(base).join("lair"));
    }
    let base = match std::env::var_os("XDG_DATA_HOME") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").context("HOME is not set; pass --data")?).join(".local/share"),
    };
    Ok(base.join("lair"))
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Chat { name, mut model, system } => {
            model.resolve(&default_data_dir()?)?;
            model.ensure_model(std::env::var("HF_TOKEN").ok().as_deref()).await?;
            let (engine, params) = model.load(true)?;
            let id = find_model(&engine, name.as_deref())?;
            tokio::task::block_in_place(|| chat::run(&engine, &id, params, system))
        }
        Cmd::Bench { name, mut model, data } => {
            let data = data.map_or_else(default_data_dir, Ok)?;
            std::fs::create_dir_all(&data).with_context(|| format!("creating {}", data.display()))?;
            model.resolve(&data)?;
            model.ensure_model(std::env::var("HF_TOKEN").ok().as_deref()).await?;
            let (engine, params) = model.load(true)?;
            let id = find_model(&engine, name.as_deref())?;
            let store = Store::open(&data.join("lair.db"))?;
            tokio::task::block_in_place(|| bench(engine, &id, &params, &store))
        }
        Cmd::Update => update::run().await,
        Cmd::Pull { model, models, hf_token } => pull(&model, &models_dir(models)?, hf_token.as_deref()).await,
        Cmd::Rm { name, models } => rm(&name, &models_dir(models)?),
        Cmd::List { models } => {
            let dir = models_dir(models)?;
            let mut found = Vec::new();
            engine::collect_gguf(&dir, &mut found)?;
            found.sort();
            for path in &found {
                let size = std::fs::metadata(path).map_or(0, |m| m.len());
                println!("{:<48} {:>6.1} GB", path.strip_prefix(&dir).unwrap_or(path).display(), size as f64 / 1e9);
            }
            if found.is_empty() {
                eprintln!(
                    "no models in {}; `lair pull <repo>` downloads one, or `lair serve` fetches the best for this machine",
                    dir.display()
                );
            }
            Ok(())
        }
        Cmd::Serve { mut model, skills, host, port, token, data, workspace, searxng, hf_token, network } => {
            if token.is_none() && !matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1") {
                eprintln!("warning: serving on {host} without --token; anyone who can reach it can use it");
            }
            let data = data.map_or_else(default_data_dir, Ok)?;
            let workspace = workspace.unwrap_or_else(|| data.join("workspace"));
            for dir in [&data, &workspace] {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            let store = Store::open(&data.join("lair.db"))?;
            let saved = lair::saved(&store);
            let flags = Config {
                searxng,
                skills_index: skills.skills_index.clone(),
                name: network.name.clone().filter(|n| !n.is_empty()),
                peers: network.peers.clone(),
                reasoning: model.reasoning,
                trust_tailnet: None,
                hf_token: hf_token.filter(|t| !t.is_empty()),
                memory: None,
            };
            let config = flags.over(&saved);
            model.resolve(&data)?;
            model.ensure_model(config.hf_token.as_deref()).await?;
            let (engine, defaults) = model.load(false)?;
            if engine.models().is_empty() {
                eprintln!("no models yet; download one from Settings → Models, or with `lair pull`");
            }
            let http = net::client();
            let (peers, auto_name) = network.peers(port, http.clone()).await;
            let bench = store.kv("bench")?.and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
            let peers = Arc::new(peers);
            let lair = Lair {
                models: Models::new(Arc::new(engine), peers.clone(), bench),
                peers,
                store,
                skills: RwLock::new(skills.catalog(&data)),
                http,
                assets: Assets::new(&skills.assets),
                workspace,
                approvals: Approvals::default(),
                downloads: download::Downloads::default(),
                hub: hub::Cache::default(),
                memory: memory::Memory::default(),
                params: defaults,
                config: RwLock::new(config),
                flags,
                auto_name,
                stopping: tokio::sync::watch::Sender::new(false),
            };
            server::run(lair, token, format!("{host}:{port}")).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(dir: &std::path::Path, url: &str) -> ModelArgs {
        match Cli::parse_from(["lair", "chat", "-m", dir.to_str().unwrap(), "--default-model", url]).cmd {
            Cmd::Chat { model, .. } => model,
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn the_default_model_is_only_fetched_into_an_empty_folder() {
        let dir = std::env::temp_dir().join(format!("lair-ensure-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(args(&dir, "").ensure_model(None).await.is_ok());
        assert!(args(&dir, "http://invalid.invalid/model.bin").ensure_model(None).await.is_err());
        std::fs::write(dir.join("m.gguf"), b"x").unwrap();
        assert!(args(&dir, "http://invalid.invalid/model.gguf").ensure_model(None).await.is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn without_m_models_live_in_the_data_folder() {
        let data = std::env::temp_dir().join(format!("lair-data-{}", std::process::id()));
        let mut model = match Cli::parse_from(["lair", "run", "qwen"]).cmd {
            Cmd::Chat { name, model, .. } => {
                assert_eq!(name.as_deref(), Some("qwen"));
                model
            }
            _ => unreachable!(),
        };
        assert_eq!((model.gpu_layers, model.ctx, model.default_model.as_str()), (None, None, "auto"));
        model.resolve(&data).unwrap();
        assert_eq!(model.models, [data.join("models")]);
        assert!(data.join("models").is_dir());
        std::fs::remove_dir_all(&data).unwrap();
    }

    #[test]
    fn rm_deletes_the_one_model_a_name_points_at() {
        let dir = std::env::temp_dir().join(format!("lair-rm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["qwen-Q4.gguf", "qwen-Q8.gguf", "gemma.gguf"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert!(rm("qwen", &dir).unwrap_err().to_string().contains("several"));
        assert!(rm("llama", &dir).is_err());
        rm("gem", &dir).unwrap();
        rm("qwen-Q4", &dir).unwrap();
        assert!(
            !dir.join("gemma.gguf").exists() && !dir.join("qwen-Q4.gguf").exists() && dir.join("qwen-Q8.gguf").exists()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
