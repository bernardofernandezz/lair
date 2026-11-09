//! Every model this node can use, wherever it runs, behind one streaming `generate`.

use std::sync::Arc;

use serde::Serialize;
use tokio::sync::{mpsc, Mutex};

use crate::engine::{Done, Engine, ModelInfo, Msg, Params, Part};

/// One streamed piece of a generation. The stream always ends with `Done` or `Failed`.
#[derive(Debug)]
pub enum Event {
    Thinking(String),
    Answer(String),
    Done(Done),
    Failed(String),
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub thinks: bool,
    pub loaded: bool,
}

pub struct Models {
    // One model on one GPU: run requests one at a time.
    engine: Arc<Mutex<Engine>>,
    local: Vec<ModelInfo>,
    /// Model in memory after the last generation; readable while the engine is busy.
    loaded: Arc<std::sync::Mutex<Option<String>>>,
}

impl Models {
    pub fn new(engine: Engine) -> Self {
        Self {
            local: engine.models().to_vec(),
            engine: Arc::new(Mutex::new(engine)),
            loaded: Arc::default(),
        }
    }

    pub fn list(&self) -> Vec<ModelEntry> {
        let loaded = self.loaded.lock().unwrap().clone();
        self.local
            .iter()
            .map(|m| ModelEntry { id: m.id.clone(), thinks: m.thinks, loaded: loaded.as_deref() == Some(m.id.as_str()) })
            .collect()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.local.iter().any(|m| m.id == id)
    }

    /// The model in memory, else the first one.
    pub fn default_id(&self) -> String {
        self.loaded.lock().unwrap().clone().unwrap_or_else(|| self.local[0].id.clone())
    }

    /// Starts a generation. Dropping the receiver stops it.
    pub fn generate(&self, id: &str, msgs: Vec<Msg>, params: Params) -> mpsc::UnboundedReceiver<Event> {
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
