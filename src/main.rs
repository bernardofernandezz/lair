mod agent;
mod assets;
mod attach;
mod bench;
mod chat;
mod config;
mod download;
mod engine;
mod generation;
mod gguf;
mod hub;
mod hw;
mod jail;
mod lair;
mod memory;
mod models;
mod net;
mod ollama_store;
mod peers;
mod server;
mod skills;
mod speed;
mod sse;
mod store;
mod turn;
mod update;
mod util;

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use assets::Assets;
use config::Config;
use download::cli;
use engine::{Engine, Tune};
use generation::{Params, Reasoning};
use lair::{Lair, Parts};
use peers::Peers;
use skills::Catalog;
use store::Store;

/// lair — run LLMs locally (GGUF via llama.cpp). Nothing leaves this machine.
#[derive(Parser)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// Where models, conversations, skills and measurements are kept (default: ~/.local/share/lair, or
    /// %LOCALAPPDATA%\lair on Windows)
    #[arg(long, env = "LAIR_DATA", global = true)]
    data: Option<PathBuf>,
    /// Hugging Face token, for gated or private models
    #[arg(long, env = "HF_TOKEN", hide_env_values = true, global = true)]
    hf_token: Option<String>,
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
        #[command(flatten)]
        dir: ModelsDir,
    },
    /// Delete a model from the models folder
    #[command(visible_alias = "remove")]
    Rm {
        /// The model's name as `lair list` shows it, or any part of it that matches only one
        name: String,
        #[command(flatten)]
        dir: ModelsDir,
    },
    /// Measure how fast a model runs on this machine; lair uses it to estimate the speed of others
    Bench {
        /// Model to measure: any part of its name (default: the first one found)
        name: Option<String>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// List the models in the models folder
    List {
        #[command(flatten)]
        dir: ModelsDir,
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
        /// Folder the agent's file and shell tools work in (default: <data>/workspace)
        #[arg(long, env = "LAIR_WORKSPACE")]
        workspace: Option<PathBuf>,
        /// Base URL of a SearXNG instance for the agent's web search, e.g. http://localhost:8888
        #[arg(long, env = "LAIR_SEARXNG")]
        searxng: Option<String>,
        /// Tool steps the agent may take in one reply (default: 20)
        #[arg(long, env = "LAIR_MAX_STEPS")]
        max_steps: Option<usize>,
        #[command(flatten)]
        network: NetworkArgs,
    },
    /// Update lair to the latest release
    #[command(visible_alias = "self-update")]
    Update,
}

#[derive(Args)]
struct ModelsDir {
    /// The models folder (default: <data>/models)
    #[arg(short = 'm', long = "models")]
    models: Option<PathBuf>,
}

impl ModelsDir {
    fn resolve(self, data: Option<PathBuf>) -> Result<PathBuf> {
        match self.models {
            Some(dir) => util::ensure_dir(dir),
            None => util::ensure_dir(config::data_dir(data)?.join("models")),
        }
    }
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
    fn catalog(&self, data: &Path, assets: &Assets) -> Catalog {
        if self.no_skills {
            Catalog::off()
        } else {
            Catalog::load(self.skills_dir.clone(), data.join("skills"), assets.ids().to_vec())
        }
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
    #[arg(long, default_value_t = Params::default().temperature)]
    temp: f32,
    #[arg(long, default_value_t = Params::default().top_p)]
    top_p: f32,
    /// Sample only among the k likeliest tokens (0 = all)
    #[arg(long, default_value_t = Params::default().top_k)]
    top_k: i32,
    /// Drop tokens less likely than this share of the likeliest one (0 = off)
    #[arg(long, default_value_t = Params::default().min_p)]
    min_p: f32,
    /// Penalty on tokens the reply already used (1 = off)
    #[arg(long, default_value_t = Params::default().repeat_penalty)]
    repeat_penalty: f32,
    /// Max tokens per reply, thinking included
    #[arg(long, default_value_t = Params::default().max_tokens)]
    max_tokens: usize,
    /// How much reasoning models think: none, low (~512 tokens) or high (unlimited)
    #[arg(long)]
    reasoning: Option<Reasoning>,
    /// Model downloaded into the first models folder when no .gguf is found there: "auto" picks the
    /// best one for this machine from Hugging Face, a URL names one, "" never downloads
    #[arg(long, env = "LAIR_DEFAULT_MODEL", default_value = models::AUTO)]
    default_model: String,
}

impl ModelArgs {
    /// The folders to look for models in: `-m`, or `<data>/models` plus what Ollama downloaded. With no
    /// model in any, the default one is downloaded first.
    async fn prepare(&mut self, data: &Path, token: Option<&str>) -> Result<()> {
        if self.models.is_empty() {
            self.models.push(util::ensure_dir(data.join("models"))?);
            self.models.extend(ollama_store::dir());
        }
        cli::first_run(&self.models, &self.default_model, token).await
    }

    /// The engine and the default generation params. `must_chat` refuses to start with no chat model:
    /// the terminal has nothing to do then, while the server can still download one.
    fn load(&self, must_chat: bool) -> Result<(Engine, Params)> {
        let tune = Tune { gpu_layers: self.gpu_layers, ctx: self.ctx, threads: self.threads };
        let engine = Engine::new(&self.models, tune)?;
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
            reasoning: self.reasoning.unwrap_or_default(),
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            ..Params::default()
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

#[tokio::main]
async fn main() -> Result<()> {
    let Cli { cmd, data, hf_token } = Cli::parse();
    let token = hf_token.filter(|t| !t.is_empty());
    match cmd {
        Cmd::Chat { name, mut model, system } => {
            model.prepare(&config::data_dir(data)?, token.as_deref()).await?;
            let (engine, params) = model.load(true)?;
            let id = find_model(&engine, name.as_deref())?;
            tokio::task::block_in_place(|| chat::run(&engine, &id, params, system))
        }
        Cmd::Bench { name, mut model } => {
            let data = config::data_dir(data)?;
            model.prepare(&data, token.as_deref()).await?;
            let (engine, params) = model.load(true)?;
            let id = find_model(&engine, name.as_deref())?;
            let store = Store::open(&data.join("lair.db"))?;
            tokio::task::block_in_place(|| bench::run(engine, &id, &params, &store))
        }
        Cmd::Update => update::run().await,
        Cmd::Pull { model, dir } => cli::pull(&model, &dir.resolve(data)?, token.as_deref()).await,
        Cmd::Rm { name, dir } => cli::rm(&name, &dir.resolve(data)?),
        Cmd::List { dir } => cli::list(&dir.resolve(data)?),
        Cmd::Serve { mut model, skills, host, port, token: api_token, workspace, searxng, max_steps, network } => {
            if api_token.is_none() && !host.parse::<IpAddr>().map_or(host == "localhost", |ip| ip.is_loopback()) {
                eprintln!("warning: serving on {host} without --token; anyone who can reach it can use it");
            }
            let data = config::data_dir(data)?;
            let workspace = util::ensure_dir(workspace.unwrap_or_else(|| data.join("workspace")))?;
            let flags = Config {
                searxng,
                skills_index: skills.skills_index.clone(),
                name: network.name,
                peers: network.peers,
                reasoning: model.reasoning,
                hf_token: token,
                max_steps,
                ..Config::default()
            }
            .validated()?;
            let store = Store::open(&data.join("lair.db"))?;
            let saved = Config::saved(&store);
            model.prepare(&data, flags.over(&saved).hf_token.as_deref()).await?;
            let (engine, params) = model.load(false)?;
            if engine.models().is_empty() {
                eprintln!("no models yet; download one from Settings → Models, or with `lair pull`");
            }
            let http = net::client();
            let peers = Peers::new(network.tailscale_socket, port, http.clone());
            let auto_name = peers.auto_name().await;
            eprintln!(
                "node name {auto_name}{}",
                if peers.on_tailnet() { ", looking for peers on the tailnet" } else { "" }
            );
            let assets = Assets::new(&skills.assets);
            let lair = Lair::new(Parts {
                engine: Arc::new(engine),
                peers,
                auto_name,
                store,
                skills: skills.catalog(&data, &assets),
                assets,
                workspace,
                params,
                flags,
                saved,
                http,
            })?;
            server::run(lair, api_token, format!("{host}:{port}")).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lair-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn the_default_model_is_only_fetched_into_an_empty_folder() {
        let dir = temp("first-run");
        let models = [dir.clone()];
        assert!(cli::first_run(&models, "", None).await.is_ok(), "\"\" never downloads");
        assert!(cli::first_run(&models, "http://invalid.invalid/model.gguf", None).await.is_err(), "it tried");
        std::fs::write(dir.join("m.gguf"), b"x").unwrap();
        assert!(cli::first_run(&models, "http://invalid.invalid/model.gguf", None).await.is_ok(), "it didn't try");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pull_keeps_models_in_the_data_folder_by_default() {
        let data = temp("data");
        let Cli { cmd: Cmd::Pull { dir, .. }, data: flag, .. } =
            Cli::parse_from(["lair", "pull", "qwen", "--data", data.to_str().unwrap()])
        else {
            unreachable!()
        };
        assert_eq!(dir.resolve(flag).unwrap(), data.join("models"));
        assert!(data.join("models").is_dir());
        std::fs::remove_dir_all(&data).unwrap();
    }

    #[test]
    fn rm_deletes_the_one_model_a_name_points_at() {
        let dir = temp("rm");
        for f in ["qwen-Q4.gguf", "qwen-Q8.gguf", "gemma.gguf"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        assert!(cli::rm("qwen", &dir).unwrap_err().to_string().contains("several"));
        assert!(cli::rm("llama", &dir).is_err());
        cli::rm("gem", &dir).unwrap();
        cli::rm("qwen-Q4", &dir).unwrap();
        assert!(
            !dir.join("gemma.gguf").exists() && !dir.join("qwen-Q4.gguf").exists() && dir.join("qwen-Q8.gguf").exists()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
