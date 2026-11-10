//! Every model this node can use, local or on a peer, behind one streaming `generate`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use tokio::sync::{Mutex, mpsc};

use crate::engine::{Engine, Event, ModelEntry, ModelInfo, Msg, Params, Part};
use crate::peers::Peers;

pub struct Models {
    // One model on one GPU: run requests one at a time.
    engine: Arc<Mutex<Engine>>,
    /// Read without the engine lock, so listing never waits on a generation; replaced by [`Models::refresh`].
    local: RwLock<Vec<ModelInfo>>,
    ctx: u32,
    /// Where downloaded models go.
    dir: Option<PathBuf>,
    /// Model in memory after the last generation; readable while the engine is busy.
    loaded: Arc<std::sync::Mutex<Option<String>>>,
    peers: Arc<Peers>,
}

impl Models {
    pub fn new(engine: Engine, peers: Arc<Peers>) -> Self {
        Self {
            local: RwLock::new(engine.models().to_vec()),
            ctx: engine.ctx(),
            dir: engine.models_dir().map(PathBuf::from),
            engine: Arc::new(Mutex::new(engine)),
            loaded: Arc::default(),
            peers,
        }
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Looks for models on disk again, after a download; waits for a running generation to end.
    pub async fn refresh(&self) {
        let mut engine = self.engine.clone().lock_owned().await;
        let found = tokio::task::spawn_blocking(move || {
            engine.rescan();
            engine.models().to_vec()
        })
        .await
        .expect("rescanning models does not panic");
        *self.local.write().unwrap() = found;
    }

    /// Models on this machine only: what peers see of this node.
    pub fn local(&self) -> Vec<ModelEntry> {
        let loaded = self.loaded.lock().unwrap().clone();
        self.local
            .read()
            .unwrap()
            .iter()
            .map(|m| ModelEntry {
                id: m.id.clone(),
                node: None,
                thinks: m.thinks,
                loaded: loaded.as_deref() == Some(m.id.as_str()),
                vision: m.mmproj.is_some(),
                size: std::fs::metadata(&m.path).map_or(0, |f| f.len()),
                ctx: self.ctx,
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
        self.local.read().unwrap().iter().any(|m| m.id == id && m.mmproj.is_some())
    }

    pub fn contains(&self, id: &str) -> bool {
        self.local.read().unwrap().iter().any(|m| m.id == id) || self.peers.locate(id).is_some()
    }

    /// The model in memory, else the first one.
    pub fn default_id(&self) -> String {
        self.loaded.lock().unwrap().clone().unwrap_or_else(|| self.local.read().unwrap()[0].id.clone())
    }

    /// Starts a generation, here or on the peer that has the model. Dropping the receiver stops it.
    pub fn generate(&self, id: &str, msgs: Vec<Msg>, params: Params) -> mpsc::UnboundedReceiver<Event> {
        if !self.local.read().unwrap().iter().any(|m| m.id == id)
            && let Some((base, model)) = self.peers.locate(id)
        {
            return self.peers.generate(base, model, msgs, params);
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
