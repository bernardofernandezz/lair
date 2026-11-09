mod agent;
mod assets;
mod chat;
mod engine;
mod jail;
mod lair;
mod models;
mod server;
mod skills;
mod store;
mod turn;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use anyhow::Context;
use agent::tools::Tools;
use agent::Approvals;
use assets::Assets;
use lair::Lair;
use models::Models;
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
    },
}

#[derive(Args)]
struct SkillArgs {
    /// Turn off skills (prompt packages added when a message asks for them)
    #[arg(long)]
    no_skills: bool,
    /// Folder with extra skills; a skill with the same id replaces the built-in
    #[arg(long, env = "LAIR_SKILLS")]
    skills_dir: Option<PathBuf>,
    /// Folders or files with .glb/.gltf models that 3D scenes may use (repeatable)
    #[arg(long, env = "LAIR_ASSETS")]
    assets: Vec<PathBuf>,
}

impl SkillArgs {
    fn catalog(&self) -> Catalog {
        if self.no_skills { Catalog::default() } else { Catalog::load(self.skills_dir.as_deref()) }
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
        Cmd::Serve { model, skills, host, port, token, data, workspace, searxng } => {
            if token.is_none() && !matches!(host.as_str(), "127.0.0.1" | "localhost" | "::1") {
                eprintln!("warning: serving on {host} without --token; anyone who can reach it can use it");
            }
            let data = data.map_or_else(default_data_dir, Ok)?;
            let workspace = workspace.unwrap_or_else(|| data.join("workspace"));
            for dir in [&data, &workspace] {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            let (engine, defaults) = model.load()?;
            let lair = Lair {
                models: Models::new(engine),
                store: Store::open(&data.join("lair.db"))?,
                skills: skills.catalog(),
                assets: Assets::new(&skills.assets),
                tools: Tools::new(workspace, searxng),
                approvals: Approvals::default(),
                defaults,
                // ~3 characters per token; half the context for history, half for the reply.
                history_chars: model.ctx as usize * 3 / 2,
            };
            server::run(lair, token, format!("{host}:{port}")).await
        }
    }
}
