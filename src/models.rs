//! Every model this node can use, local or on a peer, behind one streaming `generate`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::engine::{Engine, Event, LoadedInfo, ModelEntry, Msg, Params, Part};
use crate::peers::Peers;
use crate::speed::{Bench, Cand, Prediction};

/// The model id that lets lair choose per request.
pub const AUTO: &str = "auto";

/// A generation shorter than this says little about speed.
const WORTH_TIMING: usize = 32;

pub struct Models {
    engine: Arc<Engine>,
    /// Where downloaded models go.
    dir: Option<PathBuf>,
    /// Local generations running or queued, told to peers so `auto` can avoid a busy node.
    active: Arc<AtomicU32>,
    /// Measured speeds; `unsaved` flips when a generation adds one.
    bench: Arc<Mutex<Bench>>,
    unsaved: Arc<AtomicBool>,
    peers: Arc<Peers>,
}

/// The model a request runs on, and why when lair chose it.
pub struct Route {
    pub id: String,
    pub why: Option<String>,
}

impl Models {
    pub fn new(engine: Arc<Engine>, peers: Arc<Peers>, bench: Bench) -> Self {
        Self {
            dir: engine.models_dir().map(PathBuf::from),
            engine,
            active: Arc::default(),
            bench: Arc::new(Mutex::new(bench)),
            unsaved: Arc::default(),
            peers,
        }
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Looks for models on disk again, after a download or a removal.
    pub async fn refresh(&self) {
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || engine.rescan()).await.expect("rescanning models does not panic");
    }

    /// Share of `m`'s layers on the GPU: as loaded, else as this machine would take a model its size.
    fn offload(&self, loaded: &[LoadedInfo], id: &str, size: u64) -> f32 {
        match loaded.iter().find(|l| l.id == id) {
            Some(l) => l.gpu_layers as f32 / l.layers.max(1) as f32,
            None => crate::hub::offload_guess(size, &crate::hw::hardware()),
        }
    }

    /// Models on this machine only: what peers see of this node.
    pub fn local(&self) -> Vec<ModelEntry> {
        let loaded = self.engine.loaded();
        let bench = self.bench.lock().unwrap();
        self.engine
            .models()
            .into_iter()
            .map(|m| {
                // Tokens per second is a chat model's measure; an embedding model has none.
                let speed = (!m.meta.embedding)
                    .then(|| bench.predict(Some(&m.id), m.size, self.offload(&loaded, &m.id, m.size)))
                    .flatten();
                ModelEntry {
                    node: None,
                    thinks: m.thinks,
                    loaded: loaded.iter().any(|l| l.id == m.id),
                    vision: m.mmproj.is_some(),
                    size: m.size,
                    ctx: self.engine.ctx_of(&m.id),
                    family: m.meta.arch,
                    params: m.meta.params,
                    quant: m.meta.quant,
                    embedding: m.meta.embedding,
                    tps: speed.map_or(0.0, |s| s.tps),
                    measured: speed.is_some_and(|s| s.measured),
                    id: m.id,
                }
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
        self.engine.models().iter().any(|m| m.id == id && m.mmproj.is_some())
    }

    fn is_local(&self, id: &str) -> bool {
        self.engine.models().iter().any(|m| m.id == id)
    }

    /// The context size `id` runs with, to size the history a prompt carries.
    pub fn ctx(&self, id: &str) -> u32 {
        if self.is_local(id) {
            return self.engine.ctx_of(id);
        }
        self.peers.models().iter().find(|m| m.id == id).map_or(0, |m| m.ctx)
    }

    /// The model for a request. A named one must exist. `auto` picks among every chat model, peers
    /// included; no name at all picks among the local ones, as clients that never name a model expect.
    pub fn resolve(&self, id: Option<&str>, vision: bool) -> Result<Route, String> {
        let id = id.filter(|id| !id.is_empty());
        if let Some(id) = id.filter(|id| *id != AUTO) {
            let known = self.is_local(id) || self.peers.locate(id).is_some();
            return if known {
                Ok(Route { id: id.to_owned(), why: None })
            } else {
                Err(format!("model {id} not found"))
            };
        }
        let here = self.active.load(Ordering::Relaxed);
        let models = if id.is_some() { self.list() } else { self.local() };
        let cands: Vec<Cand> = models
            .into_iter()
            .filter(|m| !m.embedding)
            .map(|m| Cand {
                local: m.node.is_none(),
                vision: m.vision,
                size: m.size,
                tps: m.tps,
                loaded: m.loaded,
                active: m.node.as_deref().map_or(here, |n| self.peers.active(n)),
                id: m.id,
            })
            .collect();
        match crate::speed::pick(&cands, vision) {
            Some((i, why)) => Ok(Route { id: cands[i].id.clone(), why: Some(why) }),
            None if vision => Err("no local model sees images; download a vision model".into()),
            None => Err("no chat model is installed; download one in Settings or with `lair pull`".into()),
        }
    }

    /// Starts a generation, here or on the peer that has the model. Dropping the receiver stops it.
    pub fn generate(&self, id: &str, msgs: Vec<Msg>, params: Params) -> mpsc::UnboundedReceiver<Event> {
        if !self.is_local(id)
            && let Some((base, model)) = self.peers.locate(id)
        {
            return self.peers.generate(base, model, msgs, params);
        }
        self.generate_local(id, msgs, params)
    }

    /// Starts a generation on this machine; a peer asking for one never gets forwarded further.
    /// Generations on different models run side by side; on the same model, one after the other.
    pub fn generate_local(&self, id: &str, msgs: Vec<Msg>, params: Params) -> mpsc::UnboundedReceiver<Event> {
        let (tx, rx) = mpsc::unbounded_channel();
        let (engine, active, id) = (self.engine.clone(), self.active.clone(), id.to_owned());
        let (bench, unsaved) = (self.bench.clone(), self.unsaved.clone());
        active.fetch_add(1, Ordering::Relaxed);
        tokio::task::spawn_blocking(move || {
            let res = engine.generate(&id, &msgs, &params, |part| {
                let event = match part {
                    Part::Thinking(t) => Event::Thinking { text: t.to_owned() },
                    Part::Answer(t) => Event::Answer { text: t.to_owned() },
                };
                tx.send(event).is_ok()
            });
            active.fetch_sub(1, Ordering::Relaxed);
            if let Ok(done) = &res
                && done.completion_tokens >= WORTH_TIMING
                && let Some(l) = engine.loaded().iter().find(|l| l.id == id)
            {
                let offload = l.gpu_layers as f32 / l.layers.max(1) as f32;
                bench.lock().unwrap().record(&id, l.size, offload, done.tps());
                unsaved.store(true, Ordering::Relaxed);
            }
            let _ = tx.send(match res {
                Ok(done) => Event::Done(done),
                Err(e) => Event::Failed { message: format!("{e:#}") },
            });
        });
        rx
    }

    /// The local embedding model to use when none is named.
    pub fn embedder(&self) -> Option<String> {
        self.engine.models().into_iter().find(|m| m.meta.embedding).map(|m| m.id)
    }

    /// One vector of length 1 per input, and the tokens read. Local embedding models only.
    pub async fn embed(&self, id: &str, inputs: Vec<String>) -> Result<(Vec<Vec<f32>>, usize)> {
        let (engine, id) = (self.engine.clone(), id.to_owned());
        tokio::task::spawn_blocking(move || engine.embed(&id, &inputs)).await?
    }

    /// Brings a local model into memory.
    pub async fn load(&self, id: &str) -> Result<()> {
        let (engine, id) = (self.engine.clone(), id.to_owned());
        tokio::task::spawn_blocking(move || engine.load(&id)).await?
    }

    /// Drops a local model from memory; `false` when it wasn't there.
    pub fn unload(&self, id: &str) -> bool {
        self.engine.unload(id)
    }

    pub fn keep_alive(&self, id: &str, keep: Option<Duration>) {
        self.engine.keep_alive(id, keep);
    }

    /// Unloads models idle past their keep-alive.
    pub fn reap(&self) {
        self.engine.reap();
    }

    /// The models in memory.
    pub fn ps(&self) -> Vec<LoadedInfo> {
        self.engine.loaded()
    }

    /// Where a local model's file is, to delete it.
    pub fn path(&self, id: &str) -> Option<PathBuf> {
        self.engine.models().into_iter().find(|m| m.id == id).map(|m| m.path)
    }

    /// Local generations running or queued.
    pub fn active(&self) -> u32 {
        self.active.load(Ordering::Relaxed)
    }

    /// How fast a model of `size` bytes not installed yet would run here; `None` until something ran.
    pub fn predict(&self, size: u64) -> Option<Prediction> {
        let offload = crate::hub::offload_guess(size, &crate::hw::hardware());
        self.bench.lock().unwrap().predict(None, size, offload)
    }

    /// Adds a measurement made outside a generation here (`lair bench`).
    pub fn bench(&self) -> Bench {
        self.bench.lock().unwrap().clone()
    }

    /// The measurements, if any were added since this was last asked: time to save them.
    pub fn unsaved_bench(&self) -> Option<Bench> {
        self.unsaved.swap(false, Ordering::Relaxed).then(|| self.bench())
    }
}
