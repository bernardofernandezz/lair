//! Every model this node can use, local or on a peer, behind one streaming `generate`.

use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

use crate::engine::{Engine, Event, ModelEntry, ModelInfo, Msg, Params, Part};
use crate::peers::Peers;

pub struct Models {
    // One model on one GPU: run requests one at a time.
    engine: Arc<Mutex<Engine>>,
    local: Vec<ModelInfo>,
    /// Model in memory after the last generation; readable while the engine is busy.
    loaded: Arc<std::sync::Mutex<Option<String>>>,
    peers: Arc<Peers>,
}

impl Models {
    pub fn new(engine: Engine, peers: Arc<Peers>) -> Self {
        Self { local: engine.models().to_vec(), engine: Arc::new(Mutex::new(engine)), loaded: Arc::default(), peers }
    }

    /// Models on this machine only: what peers see of this node.
    pub fn local(&self) -> Vec<ModelEntry> {
        let loaded = self.loaded.lock().unwrap().clone();
        self.local
            .iter()
            .map(|m| ModelEntry {
                id: m.id.clone(),
                node: None,
                thinks: m.thinks,
                loaded: loaded.as_deref() == Some(m.id.as_str()),
                vision: m.mmproj.is_some(),
            })
            .collect()
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        let mut all = self.local();
        all.extend(self.peers.models());
        all
    }

    /// Runs here and sees images; remote models never get images, their paths would be ours.
    pub fn sees_images(&self, id: &str) -> bool {
        self.local.iter().any(|m| m.id == id && m.mmproj.is_some())
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
                        Part::Thinking(t) => Event::Thinking { text: t.to_owned() },
                        Part::Answer(t) => Event::Answer { text: t.to_owned() },
                    };
                    tx.send(event).is_ok()
                });
                *loaded.lock().unwrap() = engine.loaded().map(str::to_owned);
                let _ = tx.send(match res {
                    Ok(done) => Event::Done(done),
                    Err(e) => Event::Failed { message: format!("{e:#}") },
                });
            })
            .await;
        });
        rx
    }
}
