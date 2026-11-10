mod agent;
mod assets;
mod attach;
mod chat;
mod config;
mod engine;
mod hub;
mod jail;
mod lair;
mod models;
mod net;
mod peers;
mod server;
mod skills;
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
use engine::{Engine, Params, Reasoning};
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
    },
    /// List the models in the models folder
    List {
        /// Folder to look in (default: ~/.local/share/lair/models, or %LOCALAPPDATA%\lair\models on Windows)
        #[arg(short = 'm', long = "models")]
        models: Option<PathBuf>,
    },
    /// Web UI and HTTP API (chat completions at /v1/chat/completions, models at /v1/models).
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
    /// Layers to offload to the GPU (0 = CPU only)
    #[arg(long, default_value_t = 99)]
    gpu_layers: u32,
    /// Context size in tokens
    #[arg(long, default_value_t = 8192)]
    ctx: u32,
    #[arg(long, default_value_t = 0.7)]
    temp: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// Max tokens per reply, thinking included
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,
    /// How much reasoning models think: none, low (~512 tokens) or high (unlimited)
    #[arg(long)]
    reasoning: Option<Reasoning>,
    /// Model downloaded into the first models folder when no .gguf is found there ("" = never download)
    #[arg(long, env = "LAIR_DEFAULT_MODEL", default_value = DEFAULT_MODEL)]
    default_model: String,
}

const DEFAULT_MODEL: &str = "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf";

impl ModelArgs {
    /// Without `-m`, models live in `<data>/models`, created on first use.
    fn resolve(&mut self, data: &std::path::Path) -> Result<()> {
        if self.models.is_empty() {
            self.models.push(models_dir(Some(data.join("models")))?);
        }
        Ok(())
    }

    /// First run with an empty models folder: download `default_model` so lair starts with something to chat with.
    async fn ensure_model(&self) -> Result<()> {
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
        eprintln!("no models in {}; downloading the default one", dir.display());
        download(&self.default_model, dir).await.map(drop)
    }

    fn load(&self) -> Result<(Engine, Params)> {
        let engine = Engine::new(&self.models, self.gpu_layers, self.ctx)?;
        for m in engine.models() {
            eprintln!("found {}{}", m.id, if m.thinks { " (reasons)" } else { "" });
        }
        let params = Params {
            temperature: self.temp,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            seed: None,
            reasoning: self.reasoning.unwrap_or_default(),
        };
        Ok((engine, params))
    }
}

/// `dir`, or `<data>/models` by default; created if missing.
fn models_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = dir.map_or_else(|| default_data_dir().map(|d| d.join("models")), Ok)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// Downloads a .gguf into `dir`, printing progress every 5%.
async fn download(url: &str, dir: &std::path::Path) -> Result<PathBuf> {
    eprintln!("downloading {url}");
    let mut shown = 0;
    let dest = hub::download(url, dir, |done, total| {
        if total > 0 && done * 20 / total > shown {
            shown = done * 20 / total;
            eprintln!("  {}% of {} MB", shown * 5, total / 1_000_000);
        }
    })
    .await?;
    eprintln!("saved {}", dest.display());
    Ok(dest)
}

/// `lair pull`: a URL downloads as is; anything else names a Hugging Face repo, and the file that fits
/// this machine best (or the quant asked for) comes down with its vision projector, if it has one.
async fn pull(spec: &str, models: &std::path::Path) -> Result<()> {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        return download(spec, models).await.map(drop);
    }
    let (repo, dir, files) = hub::plan(&net::client(), spec, models).await?;
    eprintln!("using {repo}");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    for name in &files {
        download(&hub::file_url(&repo, name), &dir).await?;
    }
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
            model.ensure_model().await?;
            let (mut engine, params) = model.load()?;
            let id = match &name {
                Some(n) => engine
                    .models()
                    .iter()
                    .find(|m| m.id.contains(n.as_str()))
                    .with_context(|| format!("no model matching {n:?}; see `lair list`"))?,
                None => &engine.models()[0],
            }
            .id
            .clone();
            tokio::task::block_in_place(|| chat::run(&mut engine, &id, params, system))
        }
        Cmd::Update => update::run().await,
        Cmd::Pull { model, models } => pull(&model, &models_dir(models)?).await,
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
                    "no models in {}; `lair pull <repo>` downloads one, or `lair serve` fetches a default",
                    dir.display()
                );
            }
            Ok(())
        }
        Cmd::Serve { mut model, skills, host, port, token, data, workspace, searxng, network } => {
            if token.is_none() && !matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1") {
                eprintln!("warning: serving on {host} without --token; anyone who can reach it can use it");
            }
            let data = data.map_or_else(default_data_dir, Ok)?;
            let workspace = workspace.unwrap_or_else(|| data.join("workspace"));
            for dir in [&data, &workspace] {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            model.resolve(&data)?;
            model.ensure_model().await?;
            let (engine, defaults) = model.load()?;
            let http = net::client();
            let (peers, auto_name) = network.peers(port, http.clone()).await;
            let flags = Config {
                searxng,
                skills_index: skills.skills_index.clone(),
                name: network.name.filter(|n| !n.is_empty()),
                peers: network.peers,
                reasoning: model.reasoning,
                trust_tailnet: None,
            };
            let store = Store::open(&data.join("lair.db"))?;
            let saved = lair::saved(&store);
            let peers = Arc::new(peers);
            let lair = Lair {
                models: Models::new(engine, peers.clone()),
                peers,
                store,
                skills: RwLock::new(skills.catalog(&data)),
                http,
                assets: Assets::new(&skills.assets),
                workspace,
                approvals: Approvals::default(),
                downloads: hub::Downloads::default(),
                params: defaults,
                // ~3 characters per token; half the context for history, half for the reply.
                history_chars: model.ctx as usize * 3 / 2,
                config: RwLock::new(flags.over(&saved)),
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
        assert!(args(&dir, "").ensure_model().await.is_ok());
        assert!(args(&dir, "http://invalid.invalid/model.bin").ensure_model().await.is_err());
        std::fs::write(dir.join("m.gguf"), b"x").unwrap();
        assert!(args(&dir, "http://invalid.invalid/model.gguf").ensure_model().await.is_ok());
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
        model.resolve(&data).unwrap();
        assert_eq!(model.models, [data.join("models")]);
        assert!(data.join("models").is_dir());
        std::fs::remove_dir_all(&data).unwrap();
    }
}
