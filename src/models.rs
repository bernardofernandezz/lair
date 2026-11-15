//! Every model this node can use, local or on a peer, behind one streaming `generate`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use serde::{Deserialize, Serialize};

use crate::engine::{CTX_FLOOR, Engine, LoadedInfo};
use crate::generation::{Event, Msg, Params, Part};
use crate::peers::Peers;
use crate::speed::{Bench, Cand, Prediction};

/// The model id that lets lair choose per request.
pub const AUTO: &str = "auto";

/// A model as the APIs list it. Peers on older versions send only the first fields.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ModelEntry {
    /// `model` for local ones, `model@node` for a peer's.
    pub id: String,
    /// The peer it runs on; `None` here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    pub thinks: bool,
    pub loaded: bool,
    /// Sees images (a projector was found next to it).
    #[serde(default)]
    pub vision: bool,
    /// File size in bytes, to show and to judge whether it fits.
    #[serde(default)]
    pub size: u64,
    /// Context size in tokens it runs with (what it will get, until it is loaded).
    #[serde(default)]
    pub ctx: u32,
    /// The GGUF's architecture, parameter count and quantization.
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub params: u64,
    #[serde(default)]
    pub quant: String,
    /// Turns text into vectors instead of chatting.
    #[serde(default)]
    pub embedding: bool,
    /// Tokens per second: measured here, or estimated from other models measured here; 0 when unknown.
    #[serde(default)]
    pub tps: f32,
    /// `tps` was measured running this very model.
    #[serde(default)]
    pub measured: bool,
}

/// A generation shorter than this says little about speed.
const WORTH_TIMING: usize = 32;

pub struct Models {
    engine: Arc<Engine>,
    /// Where downloaded models go.
    dir: Option<PathBuf>,
    /// Local generations running or queued, told to peers so `auto` can avoid a busy node.
    active: Arc<AtomicU32>,
    /// Measured speeds, and the models measured again since they were last saved.
    bench: Arc<Mutex<Bench>>,
    unsaved: Arc<Mutex<BTreeSet<String>>>,
    peers: Arc<Peers>,
}

/// Why no model can take a request.
#[derive(Debug)]
pub enum Unresolved {
    /// The model it names (this id) isn't here or on any peer.
    NotFound(String),
    /// It leaves the choice to lair, and nothing installed can do what it asks.
    NoneFits(&'static str),
}

impl std::fmt::Display for Unresolved {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "model {id} not found"),
            Self::NoneFits(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Unresolved {}

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
            Some(l) => l.offload(),
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
                    .then(|| {
                        bench.predict(Some(&m.id), m.meta.read_per_token(m.size), self.offload(&loaded, &m.id, m.size))
                    })
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
        // Older peers don't say; every model gets at least the floor.
        self.peers.models().iter().find(|m| m.id == id).map(|m| m.ctx).filter(|&c| c > 0).unwrap_or(CTX_FLOOR)
    }

    /// The model for a request. A named one must exist. `auto` picks among every chat model, peers
    /// included; no name at all picks among the local ones, as clients that never name a model expect.
    pub fn resolve(&self, id: Option<&str>, vision: bool) -> Result<Route, Unresolved> {
        let id = id.filter(|id| !id.is_empty());
        if let Some(id) = id.filter(|id| *id != AUTO) {
            let known = self.is_local(id) || self.peers.locate(id).is_some();
            return if known {
                Ok(Route { id: id.to_owned(), why: None })
            } else {
                Err(Unresolved::NotFound(id.to_owned()))
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
            None if vision => Err(Unresolved::NoneFits("no local model sees images; download a vision model")),
            None => {
                Err(Unresolved::NoneFits("no chat model is installed; download one in Settings or with `lair pull`"))
            }
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
                let reads = engine.info(&id).map_or(l.size, |m| m.meta.read_per_token(l.size));
                bench.lock().unwrap().record(&id, reads, l.offload(), done.tps());
                unsaved.lock().unwrap().insert(id.clone());
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

    pub fn ps(&self) -> Vec<LoadedInfo> {
        self.engine.loaded()
    }

    /// Deletes the local model `id` from disk and returns the bytes freed; `None` when there's no such model.
    pub async fn delete(&self, id: &str) -> Result<Option<u64>> {
        let Some(path) = self.engine.models().into_iter().find(|m| m.id == id).map(|m| m.path) else {
            return Ok(None);
        };
        let dir = self.dir().context("no models folder to delete from")?.to_owned();
        // Out of memory first: Windows refuses to delete a file that is mapped.
        self.unload(id);
        let freed = tokio::task::spawn_blocking(move || crate::download::remove(&dir, &path)).await??;
        self.refresh().await;
        Ok(Some(freed))
    }

    pub fn active(&self) -> u32 {
        self.active.load(Ordering::Relaxed)
    }

    /// How fast a model of `size` bytes not installed yet would run here, `reads` of them per token
    /// (fewer for mixture-of-experts); `None` until something ran.
    pub fn predict(&self, size: u64, reads: u64) -> Option<Prediction> {
        let offload = crate::hub::offload_guess(size, &crate::hw::hardware());
        self.bench.lock().unwrap().predict(None, reads, offload)
    }

    /// The models measured since this was last asked, if any: time to save them. Only those, so a
    /// `lair bench` run meanwhile keeps what it measured of the others.
    pub fn unsaved_bench(&self) -> Option<Bench> {
        let keys = std::mem::take(&mut *self.unsaved.lock().unwrap());
        let bench = self.bench.lock().unwrap();
        let samples: std::collections::BTreeMap<_, _> =
            keys.into_iter().filter_map(|k| Some((k.clone(), *bench.samples.get(&k)?))).collect();
        (!samples.is_empty()).then_some(Bench { samples })
    }
}
