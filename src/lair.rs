//! What this node has: models, conversations, skills and assets. Shared by the HTTP API and the agent.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use serde_json::Value;

use crate::agent::Approvals;
use crate::assets::Assets;
use crate::config::Config;
use crate::engine::Params;
use crate::models::Models;
use crate::peers::Peers;
use crate::skills::Catalog;
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
    /// Generation params from the command line; reasoning comes from [`Config`].
    pub params: Params,
    /// Rough size, in characters, of the history a prompt may carry; the rest of the context is for the reply.
    pub history_chars: usize,
    /// What the command line set; these settings can't be changed from the UI.
    pub flags: Config,
    /// The live settings: `flags` over what the UI saved. Changed only through [`Lair::save`].
    pub config: RwLock<Config>,
    /// This node's name when no setting names it: its Tailscale host name, else the host name.
    pub auto_name: String,
    /// Flips to `true` when the server is shutting down, so running turns save what they have and end.
    pub stopping: tokio::sync::watch::Sender<bool>,
}

impl Lair {
    pub fn config(&self) -> Config {
        self.config.read().unwrap().clone()
    }

    /// Changes what the UI saved (see [`Config::patched`]); takes effect at once, under the flags.
    pub fn save(&self, change: Value) -> Result<()> {
        let saved = saved(&self.store).patched(change)?;
        self.store.save_config(&serde_json::to_value(&saved)?)?;
        *self.config.write().unwrap() = self.flags.over(&saved);
        Ok(())
    }

    /// Params for a generation that says nothing else.
    pub fn params(&self) -> Params {
        Params { reasoning: self.config.read().unwrap().reasoning.unwrap_or_default(), ..self.params }
    }

    pub fn node_name(&self) -> String {
        self.config.read().unwrap().name.clone().unwrap_or_else(|| self.auto_name.clone())
    }
}

/// What the UI saved. A document lair can't read any more (an old field, a broken row) is set aside
/// with a warning, and the next save replaces it.
pub fn saved(store: &Store) -> Config {
    match store.config().and_then(|v| Ok(serde_json::from_value(v)?)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warning: ignoring the saved settings: {e:#}");
            Config::default()
        }
    }
}
