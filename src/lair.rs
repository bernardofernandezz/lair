//! What this node has: models, conversations, skills and assets. Shared by the HTTP API and the agent.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use serde_json::Value;

use crate::agent::Approvals;
use crate::assets::Assets;
use crate::config::Config;
use crate::engine::Engine;
use crate::generation::Params;
use crate::models::Models;
use crate::peers::Peers;
use crate::skills::Catalog;
use crate::speed::Bench;
use crate::store::Store;

pub struct Lair {
    pub models: Models,
    /// The other lair nodes; `models` routes generations to them too.
    pub peers: Arc<Peers>,
    pub store: Store,
    pub skills: RwLock<Catalog>,
    /// Shared by everything that reaches the network: tools, skill downloads, peers.
    pub http: reqwest::Client,
    pub assets: Assets,
    /// The folder the agent's tools work in; attachments land there too.
    pub workspace: PathBuf,
    pub approvals: Approvals,
    /// Tools of the MCP servers in `<data>/mcp.json`, offered to the agent next to its own.
    pub mcp: crate::agent::mcp::Mcp,
    /// Models being downloaded from Hugging Face.
    pub downloads: crate::download::Downloads,
    /// What Hugging Face last said about the models worth offering.
    pub hub: crate::hub::Cache,
    /// Wakes the indexer that keeps conversations and workspace files searchable by meaning.
    pub memory: crate::memory::Memory,
    /// Flips to `true` when the server is shutting down, so running turns save what they have and end.
    pub stopping: tokio::sync::watch::Sender<bool>,
    /// Generation params from the command line; reasoning comes from [`Config`].
    params: Params,
    /// What the command line set; these settings can't be changed from the UI.
    flags: Config,
    /// The live settings: `flags` over what the UI saved.
    config: RwLock<Config>,
    /// This node's name when no setting names it.
    auto_name: String,
}

/// What a node is made of; everything else in [`Lair`] starts empty.
pub struct Parts {
    pub engine: Arc<Engine>,
    pub peers: Peers,
    pub auto_name: String,
    pub store: Store,
    pub skills: Catalog,
    pub assets: Assets,
    pub workspace: PathBuf,
    /// Generation params from the command line.
    pub params: Params,
    /// Settings from the command line.
    pub flags: Config,
    /// What the UI saved ([`Config::saved`]).
    pub saved: Config,
    pub http: reqwest::Client,
    pub mcp: crate::agent::mcp::Mcp,
}

impl Lair {
    pub fn new(p: Parts) -> Result<Self> {
        let peers = Arc::new(p.peers);
        let config = p.flags.over(&p.saved);
        Ok(Self {
            models: Models::new(p.engine, peers.clone(), Bench::load(&p.store)?),
            peers,
            skills: RwLock::new(p.skills),
            http: p.http,
            assets: p.assets,
            workspace: p.workspace,
            approvals: Approvals::default(),
            mcp: p.mcp,
            downloads: Default::default(),
            hub: Default::default(),
            memory: Default::default(),
            stopping: tokio::sync::watch::Sender::new(false),
            params: p.params,
            flags: p.flags,
            config: RwLock::new(config),
            auto_name: p.auto_name,
            store: p.store,
        })
    }

    pub fn config(&self) -> Config {
        self.config.read().unwrap().clone()
    }

    /// What the command line set, which the UI can't change.
    pub fn flags(&self) -> &Config {
        &self.flags
    }

    /// Changes what the UI saved (see [`Config::patched`]); takes effect at once, under the flags.
    pub fn save(&self, change: Value) -> Result<()> {
        let saved = Config::saved(&self.store).patched(change)?;
        self.store.save_config(&serde_json::to_value(&saved)?)?;
        *self.config.write().unwrap() = self.flags.over(&saved);
        Ok(())
    }

    /// Params for a generation that says nothing else.
    pub fn params(&self) -> Params {
        Params { reasoning: self.config.read().unwrap().reasoning.unwrap_or_default(), ..self.params.clone() }
    }

    /// This node's name when no setting names it: its Tailscale host name, else the host name.
    pub fn auto_name(&self) -> &str {
        &self.auto_name
    }

    pub fn node_name(&self) -> String {
        self.config.read().unwrap().name.clone().unwrap_or_else(|| self.auto_name.clone())
    }
}
