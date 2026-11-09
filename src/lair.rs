//! What this node has: models, conversations, skills and assets. Shared by the HTTP API and the agent.

use std::sync::RwLock;

use crate::agent::tools::Tools;
use crate::agent::Approvals;
use crate::assets::Assets;
use crate::engine::Params;
use crate::models::Models;
use crate::skills::Catalog;
use crate::store::Store;

pub struct Lair {
    pub models: Models,
    pub store: Store,
    pub skills: RwLock<Catalog>,
    /// Where the UI browses skills to install, when the user configured one.
    pub skills_index: Option<String>,
    /// Shared by everything that reaches the network: tools, skill downloads.
    pub http: reqwest::Client,
    pub assets: Assets,
    pub tools: Tools,
    pub approvals: Approvals,
    pub defaults: Params,
    /// Rough size, in characters, of the history a prompt may carry; the rest of the context is for the reply.
    pub history_chars: usize,
}
