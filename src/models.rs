//! Every model this node can use, local or on a peer, behind one streaming `generate`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex};

use crate::engine::{Done, Engine, ModelInfo, Msg, Params, Part};
use crate::peers::Peers;

/// One streamed piece of a generation. The stream always ends with `Done` or `Failed`.
#[derive(Debug)]
pub enum Event {
    Thinking(String),
    Answer(String),
    Done(Done),
    Failed(String),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelEntry {
    /// `model` for local ones, `model@node` for a peer's.
    pub id: String,
    /// The peer it runs on; `None` here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    pub thinks: bool,
    pub loaded: bool,
}

pub struct Models {
    // One model on one GPU: run requests one at a time.
    engine: Arc<Mutex<Engine>>,
    local: Vec<ModelInfo>,
    /// Model in memory after the last generation; readable while the engine is busy.
    loaded: Arc<std::sync::Mutex<Option<String>>>,
    pub peers: Arc<Peers>,
}

impl Models {
    pub fn new(engine: Engine, peers: Arc<Peers>) -> Self {
        Self {
            local: engine.models().to_vec(),
            engine: Arc::new(Mutex::new(engine)),
            loaded: Arc::default(),
            peers,
        }
    }

    /// Models on this machine only: what peers see of this node.
    pub fn local(&self) -> Vec<ModelEntry> {
        let loaded = self.loaded.lock().unwrap().clone();
        self.local
            .iter()
            .map(|m| ModelEntry { id: m.id.clone(), node: None, thinks: m.thinks, loaded: loaded.as_deref() == Some(m.id.as_str()) })
            .collect()
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        let mut all = self.local();
        all.extend(self.peers.models());
        all
    }

    pub fn contains(&self, id: &str) -> bool {
        self.local.iter().any(|m| m.id == id) || self.peers.locate(id).is_some()
    }

    /// The model in memory, else the first one.
    pub fn default_id(&self) -> String {
        self.loaded.lock().unwrap().clone().unwrap_or_else(|| self.local[0].id.clone())
    }

    /// Starts a generation, here or on the peer that has the model. Dropping the receiver stops it.
    pub fn generate(&self, id: &str, msgs: Vec<Msg>, params: Params) -> mpsc::UnboundedReceiver<Event> {
        if !self.local.iter().any(|m| m.id == id) {
            if let Some((base, model)) = self.peers.locate(id) {
                return self.peers.generate(base, model, msgs, params);
            }
        }
        self.generate_local(id, msgs, params)
    }

    /// Starts a generation on this machine; a peer asking for one never gets forwarded further.
    pub fn generate_local(&self, id: &str, msgs: Vec<Msg>, params: Params) -> mpsc::UnboundedReceiver<Event> {
        let (tx, rx) = mpsc::unbounded_channel();
        let (engine, loaded, id) = (self.engine.clone(), self.loaded.clone(), id.to_owned());
        tokio::spawn(async move {
            let mut engine = engine.lock_owned().await;
            let _ = tokio::task::spawn_blocking(move || {
                let res = engine.generate(&id, &msgs, params, |part| {
                    let event = match part {
                        Part::Thinking(t) => Event::Thinking(t.to_owned()),
                        Part::Answer(t) => Event::Answer(t.to_owned()),
                    };
                    tx.send(event).is_ok()
                });
                *loaded.lock().unwrap() = engine.loaded().map(str::to_owned);
                let _ = tx.send(match res {
                    Ok(done) => Event::Done(done),
                    Err(e) => Event::Failed(format!("{e:#}")),
                });
            })
            .await;
        });
        rx
    }
}
