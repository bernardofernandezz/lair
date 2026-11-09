mod agent;
mod assets;
mod chat;
mod engine;
mod jail;
mod lair;
mod models;
mod peers;
mod server;
mod skills;
mod store;
mod turn;

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use anyhow::Context;
use agent::tools::Tools;
use agent::Approvals;
use assets::Assets;
use lair::Lair;
use models::Models;
use peers::Peers;
use store::Store;
use engine::{Engine, Params, Reasoning};
use skills::Catalog;

/// lair — run LLMs locally (GGUF via llama.cpp). Nothing leaves this machine.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Interactive chat in the terminal, with the first model found
    Chat {
        #[command(flatten)]
        model: ModelArgs,
        /// System prompt
        #[arg(long)]
        system: Option<String>,
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
        /// Where conversations and installed skills are kept (default: ~/.local/share/lair)
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
    async fn peers(self, port: u16, http: reqwest::Client) -> Peers {
        let socket = self.tailscale_socket.exists().then_some(self.tailscale_socket);
        let tailscale = match &socket {
            Some(s) => Peers::tailscale_name(s).await,
            None => None,
        };
        let name = self
            .name
            .or(tailscale)
            .or_else(|| std::fs::read_to_string("/etc/hostname").ok().map(|h| h.trim().to_owned()))
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "lair".into());
        eprintln!("node name {name}{}", if socket.is_some() { ", looking for peers on the tailnet" } else { "" });
        Peers::new(name, socket, self.peers, port, http)
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
    /// .gguf model files, or directories to search for them (repeatable)
    #[arg(short = 'm', long = "models", visible_alias = "model", required = true)]
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
    #[arg(long, default_value = "high", value_parser = parse_reasoning)]
    reasoning: Reasoning,
}

fn parse_reasoning(s: &str) -> Result<Reasoning, String> {
    serde_json::from_value(serde_json::Value::String(s.into())).map_err(|_| "expected none, low or high".into())
}

impl ModelArgs {
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
            reasoning: self.reasoning,
        };
        Ok((engine, params))
    }
}

fn default_data_dir() -> Result<PathBuf> {
    let base = match std::env::var_os("XDG_DATA_HOME") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").context("HOME is not set; pass --data")?).join(".local/share"),
    };
    Ok(base.join("lair"))
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Chat { model, system } => {
            let (mut engine, params) = model.load()?;
            tokio::task::block_in_place(|| chat::run(&mut engine, params, system))
        }
        Cmd::Serve { model, skills, host, port, token, data, workspace, searxng, network } => {
            if token.is_none() && !matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1") {
                eprintln!("warning: serving on {host} without --token; anyone who can reach it can use it");
            }
            let data = data.map_or_else(default_data_dir, Ok)?;
            let workspace = workspace.unwrap_or_else(|| data.join("workspace"));
            for dir in [&data, &workspace] {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            let (engine, defaults) = model.load()?;
            let http = agent::tools::http_client();
            let peers = Arc::new(network.peers(port, http.clone()).await);
            let lair = Lair {
                models: Models::new(engine, peers),
                store: Store::open(&data.join("lair.db"))?,
                skills: RwLock::new(skills.catalog(&data)),
                skills_index: skills.skills_index.clone(),
                http,
                assets: Assets::new(&skills.assets),
                tools: Tools { workspace, searxng },
                approvals: Approvals::default(),
                defaults,
                // ~3 characters per token; half the context for history, half for the reply.
                history_chars: model.ctx as usize * 3 / 2,
                stopping: tokio::sync::watch::Sender::new(false),
            };
            server::run(lair, token, format!("{host}:{port}")).await
        }
    }
}
