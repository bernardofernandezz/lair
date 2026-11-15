use std::cell::Cell;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::pin;
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::{FitError, LlamaModelParams};
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde::Serialize;

use crate::generation::{Done, Finish, Msg, Params, Part, Reasoning};
use crate::gguf::{self, Meta};
use crate::ollama_store;

mod text;
use text::{StopCut, ThinkSplit, Utf8Buf};

pub(crate) const OPEN: &str = "<think>";
pub(crate) const CLOSE: &str = "</think>";
/// What we feed the model to end its reasoning block for it.
const FORCED_CLOSE: &str = "\n\n</think>\n\n";
/// Thinking tokens allowed in `Reasoning::Light` before the block is closed for the model.
const LIGHT_BUDGET: usize = 512;

/// The context a model gets when nothing says otherwise and memory is not measured (no GPU); also the
/// least a GPU fit may shrink it to, so history trimmed for it always fits.
pub const CTX_FLOOR: u32 = 8192;
/// The most context picked automatically; `--ctx` goes past it.
const CTX_CAP: u32 = 32768;
/// Tokens decoded per call; a batch as big as a 32k context would waste memory.
const BATCH: u32 = 2048;
/// GPU memory left free on each device when fitting: the desktop, the driver and other apps use some.
const VRAM_MARGIN: usize = 1 << 30;

/// The grammar that makes an answer match a JSON schema; `{}` accepts any JSON value.
pub fn schema_grammar(schema: &serde_json::Value) -> Result<String> {
    llama_cpp_2::json_schema_to_grammar(&schema.to_string()).map_err(|e| anyhow!("unusable JSON schema: {e}"))
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    /// The file, or the first part of a split model.
    pub path: PathBuf,
    /// Emits a `<think>` block before answering.
    pub thinks: bool,
    /// The multimodal projector found next to the model; with it the model sees images.
    pub mmproj: Option<PathBuf>,
    pub meta: Meta,
    /// Bytes on disk, every part of a split model counted.
    pub size: u64,
}

/// What the command line fixes for every model; `None` lets lair pick per model.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tune {
    pub gpu_layers: Option<u32>,
    pub ctx: Option<u32>,
    pub threads: Option<i32>,
}

/// A model in memory, as `ps` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct LoadedInfo {
    pub id: String,
    pub ctx: u32,
    /// Layers on the GPU, out of `layers`.
    pub gpu_layers: u32,
    pub layers: u32,
    pub size: u64,
    /// A generation is running on it or waiting for it.
    pub busy: bool,
    /// Seconds until it is unloaded for being idle; `None` keeps it until memory is needed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<u64>,
}

impl LoadedInfo {
    /// The share of its layers on the GPU.
    pub fn offload(&self) -> f32 {
        self.gpu_layers as f32 / self.layers.max(1) as f32
    }
}

/// A loaded model's one context, reused across generations instead of allocated for each.
struct Session<'m> {
    ctx: LlamaContext<'m>,
    /// Tokens whose state is in the context's memory, in order; the next prompt reuses the common prefix.
    cached: Vec<LlamaToken>,
    /// Image encoder of a vision model.
    mtmd: Option<MtmdContext>,
}

/// Stands in for image positions in `cached`; never matches a real token, so the next prompt starts over.
const NOT_A_TOKEN: LlamaToken = LlamaToken(-1);

self_cell::self_cell!(
    /// A model in memory together with the context that borrows it.
    struct Loaded {
        owner: LlamaModel,
        #[not_covariant]
        dependent: Session,
    }
);

impl Session<'_> {
    /// Keeps the longest prefix of `tokens` already in memory and returns its length. The last token is
    /// always decoded again, since sampling needs its logits. Models whose memory can't drop a suffix
    /// (hybrid or recurrent ones) start over from an empty context.
    fn reuse(&mut self, tokens: &[LlamaToken]) -> usize {
        let common = self.cached.iter().zip(tokens).take_while(|(a, b)| a == b).count().min(tokens.len() - 1);
        if common < self.cached.len() {
            if !self.ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None).unwrap_or(false) {
                self.ctx.clear_kv_cache();
                self.cached.clear();
                return 0;
            }
            self.cached.truncate(common);
        }
        common
    }

    /// Decodes `toks` after what is cached, a batch at a time, keeping logits for the last one so it
    /// can be sampled from.
    fn feed(&mut self, batch: &mut LlamaBatch, toks: &[LlamaToken]) -> Result<()> {
        let last = toks.len() - 1;
        let size = self.ctx.n_batch().max(1) as usize;
        for (c, chunk) in toks.chunks(size).enumerate() {
            batch.clear();
            let start = self.cached.len();
            for (k, t) in chunk.iter().enumerate() {
                batch.add(*t, (start + k) as i32, &[0], c * size + k == last)?;
            }
            if let Err(e) = self.ctx.decode(batch) {
                self.ctx.clear_kv_cache();
                self.cached.clear();
                return Err(e.into());
            }
            self.cached.extend_from_slice(chunk);
        }
        Ok(())
    }
}

/// A model in memory. Whoever generates with it holds a clone of `loaded`, so a slot can be dropped
/// from the list at any time: its memory is freed when the last generation on it ends.
struct Slot {
    id: String,
    loaded: Arc<Mutex<Loaded>>,
    ctx: u32,
    gpu_layers: u32,
    layers: u32,
    size: u64,
    last_used: Instant,
    /// How long it may sit idle before [`Engine::reap`] unloads it.
    keep: Option<Duration>,
}

impl Slot {
    fn busy(&self) -> bool {
        Arc::strong_count(&self.loaded) > 1
    }
}

/// The idle slot used longest ago, given each slot's `(last_used, busy)`.
fn victim(slots: &[(Instant, bool)]) -> Option<usize> {
    slots.iter().enumerate().filter(|(_, (_, busy))| !busy).min_by_key(|(_, (at, _))| *at).map(|(i, _)| i)
}

/// Threads for a machine with `logical` cores: the physical ones, as hyperthreads slow llama.cpp down.
// shortcut: assumes two threads per core; read the topology if hybrid or SMT-less CPUs get a bad count.
fn threads(logical: usize) -> i32 {
    (logical / 2).clamp(1, 16) as i32
}

/// Every model found on disk, and the ones in memory: as many as fit, different models generating at
/// the same time, the one idle longest making room when a new one doesn't fit.
pub struct Engine {
    backend: LlamaBackend,
    /// Where models were looked for, to look again after a download.
    paths: Vec<PathBuf>,
    tune: Tune,
    catalog: RwLock<Vec<ModelInfo>>,
    slots: Mutex<Vec<Slot>>,
    /// Held while a model loads: llama.cpp's fit and context creation must not run twice at once, and
    /// two loads deciding at the same time would both count the same free memory.
    loading: Mutex<()>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // A panicked generation must not take the model (or the whole list) down with it.
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Engine {
    /// `paths` are .gguf files or directories searched recursively for them. Finding none is not an
    /// error here: a server can start empty and download its first model.
    pub fn new(paths: &[PathBuf], tune: Tune) -> Result<Self> {
        // llama.cpp allows one backend per process, so it lives here rather than per model.
        let mut backend = LlamaBackend::init()?;
        if std::env::var_os("LAIR_VERBOSE").is_none() {
            backend.void_logs();
        }
        let catalog = scan(&backend, paths, &[]);
        Ok(Self {
            backend,
            paths: paths.to_vec(),
            tune,
            catalog: RwLock::new(catalog),
            slots: Mutex::default(),
            loading: Mutex::default(),
        })
    }

    pub fn tune(&self) -> Tune {
        self.tune
    }

    /// Changes what is fixed for every model; the ones in memory are dropped so it takes effect.
    pub fn set_tune(&mut self, tune: Tune) {
        self.tune = tune;
        lock(&self.slots).clear();
    }

    /// Looks for models again, unloading the ones whose file is gone.
    pub fn rescan(&self) {
        let known = self.models();
        let found = scan(&self.backend, &self.paths, &known);
        lock(&self.slots).retain(|s| found.iter().any(|m| m.id == s.id));
        *self.catalog.write().unwrap() = found;
    }

    /// The first folder models are looked for in: where downloads go.
    pub fn models_dir(&self) -> Option<&Path> {
        self.paths.iter().find(|p| p.is_dir()).map(PathBuf::as_path)
    }

    pub fn models(&self) -> Vec<ModelInfo> {
        self.catalog.read().unwrap().clone()
    }

    pub fn info(&self, id: &str) -> Result<ModelInfo> {
        self.catalog.read().unwrap().iter().find(|m| m.id == id).cloned().ok_or_else(|| anyhow!("unknown model {id}"))
    }

    /// The context `m` runs with if nothing measures memory for it: what `--ctx` says, else the floor.
    fn default_ctx(&self, m: &Meta) -> u32 {
        let trained = if m.ctx_train == 0 { CTX_FLOOR } else { m.ctx_train };
        self.tune.ctx.unwrap_or(CTX_FLOOR.min(trained))
    }

    /// The context size `id` runs with: the real one once loaded, before that the least it will get.
    pub fn ctx_of(&self, id: &str) -> u32 {
        if let Some(s) = lock(&self.slots).iter().find(|s| s.id == id) {
            return s.ctx;
        }
        self.info(id).map_or(CTX_FLOOR, |m| self.default_ctx(&m.meta))
    }

    pub fn loaded(&self) -> Vec<LoadedInfo> {
        lock(&self.slots)
            .iter()
            .map(|s| LoadedInfo {
                id: s.id.clone(),
                ctx: s.ctx,
                gpu_layers: s.gpu_layers,
                layers: s.layers,
                size: s.size,
                busy: s.busy(),
                expires_in: s.keep.map(|k| k.saturating_sub(s.last_used.elapsed()).as_secs()),
            })
            .collect()
    }

    /// Brings `id` into memory without generating.
    pub fn load(&self, id: &str) -> Result<()> {
        self.acquire(id).map(drop)
    }

    /// Drops `id` from memory; a generation running on it finishes first.
    pub fn unload(&self, id: &str) -> bool {
        let mut slots = lock(&self.slots);
        let before = slots.len();
        slots.retain(|s| s.id != id);
        slots.len() < before
    }

    /// How long `id` may sit idle before it is unloaded; `None` keeps it until memory is needed.
    pub fn keep_alive(&self, id: &str, keep: Option<Duration>) {
        if let Some(s) = lock(&self.slots).iter_mut().find(|s| s.id == id) {
            s.keep = keep;
        }
    }

    /// Unloads the models that sat idle past their keep-alive.
    pub fn reap(&self) {
        lock(&self.slots).retain(|s| s.busy() || s.keep.is_none_or(|k| s.last_used.elapsed() < k));
    }

    /// The model in memory, loading it if needed, and the microseconds that took.
    fn acquire(&self, id: &str) -> Result<(Arc<Mutex<Loaded>>, u64)> {
        let hit = || {
            lock(&self.slots).iter_mut().find(|s| s.id == id).map(|s| {
                s.last_used = Instant::now();
                s.loaded.clone()
            })
        };
        if let Some(loaded) = hit() {
            return Ok((loaded, 0));
        }
        let _one_at_a_time = lock(&self.loading);
        if let Some(loaded) = hit() {
            return Ok((loaded, 0));
        }
        let info = self.info(id)?;
        let start = Instant::now();
        let (loaded, gpu_layers) = loop {
            // Dropping the slot frees its memory before the next try; a busy one is never dropped.
            let evict = || {
                let mut slots = lock(&self.slots);
                let idle: Vec<_> = slots.iter().map(|s| (s.last_used, s.busy())).collect();
                victim(&idle).map(|i| slots.remove(i)).is_some()
            };
            let last_try = !lock(&self.slots).iter().any(|s| !s.busy());
            match self.open(&info, last_try) {
                Ok(Some(opened)) => break opened,
                Ok(None) => {
                    evict();
                }
                // shortcut: a backend that can't report free memory (Vulkan without the memory-budget
                // extension) fits as if nothing were loaded; the load then fails and lands here. One that
                // oversubscribes instead of failing just runs slow: keep a VRAM ledger if that shows up.
                Err(e) => {
                    if !evict() {
                        return Err(e);
                    }
                }
            }
        };
        let ctx = loaded.with_dependent(|_, s| s.ctx.n_ctx());
        let loaded = Arc::new(Mutex::new(loaded));
        lock(&self.slots).push(Slot {
            id: id.to_owned(),
            loaded: loaded.clone(),
            ctx,
            gpu_layers,
            layers: info.meta.layers + 1,
            size: info.size,
            last_used: Instant::now(),
            keep: None,
        });
        Ok((loaded, start.elapsed().as_micros() as u64))
    }

    /// Loads `info` with the GPU layers and context that fit this machine right now. `None` means it
    /// would run better with another model out of memory first; with `last_try` it loads regardless.
    fn open(&self, info: &ModelInfo, last_try: bool) -> Result<Option<(Loaded, u32)>> {
        let hw = crate::hw::now();
        let layers = info.meta.layers + 1;
        let fallback_ctx = self.default_ctx(&info.meta);
        let mut cparams = LlamaContextParams::default().with_n_ctx(self.tune.ctx.and_then(NonZeroU32::new));
        let mut mparams = pin!(LlamaModelParams::default());
        let (gpu_layers, ctx) = if let Some(n) = self.tune.gpu_layers {
            mparams.set(LlamaModelParams::default().with_n_gpu_layers(n));
            (n.min(layers), fallback_ctx)
        } else if hw.gpus.is_empty() {
            mparams.set(LlamaModelParams::default().with_n_gpu_layers(0));
            (0, fallback_ctx)
        } else {
            // llama.cpp measures what the weights and the context take on each device and picks how many
            // layers go to the GPU, shrinking an automatic context down to `fallback_ctx` to fit more.
            let path = CString::new(info.path.to_string_lossy().as_bytes())?;
            let projector = info.mmproj.as_ref().and_then(|p| p.metadata().ok()).map_or(0, |m| m.len() as usize);
            let mut margins = vec![VRAM_MARGIN + projector; llama_cpp_2::max_devices()];
            const LOG_ERRORS: u32 = 4; // GGML_LOG_LEVEL_ERROR
            match mparams.as_mut().fit_params(&path, &mut cparams, &mut margins, fallback_ctx, LOG_ERRORS as _) {
                Ok(fit) => {
                    let n = mparams.n_gpu_layers();
                    let n = if n < 0 { layers } else { (n as u32).min(layers) };
                    if n < layers && !last_try {
                        return Ok(None);
                    }
                    let ctx = if fit.n_ctx == 0 { fallback_ctx } else { fit.n_ctx };
                    (n, self.tune.ctx.unwrap_or(ctx.min(CTX_CAP)))
                }
                Err(FitError::Failure) if !last_try => return Ok(None),
                Err(e) => {
                    // No fit: run on the CPU if nothing fits, as before lair fitted if fitting itself broke.
                    let n = if e == FitError::Failure { 0 } else { 99 };
                    eprintln!("fitting {} to this machine failed ({e}); using {n} GPU layers", info.id);
                    mparams.set(LlamaModelParams::default().with_n_gpu_layers(n));
                    (n.min(layers), fallback_ctx)
                }
            }
        };
        // The fit only looks at GPUs; what stays on the CPU has to fit in free RAM.
        let on_cpu = info.size / u64::from(layers) * u64::from(layers - gpu_layers);
        if !last_try && hw.ram_free > 0 && on_cpu + on_cpu / 10 > hw.ram_free {
            return Ok(None);
        }

        let model = LlamaModel::load_from_file(&self.backend, &info.path, &mparams)
            .with_context(|| format!("loading model {}", info.path.display()))?;
        let n_threads = self.tune.threads.unwrap_or_else(|| threads(hw.cores));
        // An embedding model reads a whole text in one batch.
        let batch = if info.meta.embedding { ctx } else { ctx.min(BATCH) };
        let cparams = cparams
            .with_n_ctx(NonZeroU32::new(ctx))
            .with_n_batch(batch)
            .with_n_ubatch(if info.meta.embedding { batch } else { batch.min(512) })
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads)
            .with_embeddings(info.meta.embedding);
        let (backend, mmproj) = (&self.backend, info.mmproj.clone());
        let loaded = Loaded::try_new(model, |m| -> Result<Session> {
            let mtmd = match mmproj {
                Some(p) => {
                    let params = MtmdContextParams { use_gpu: gpu_layers > 0, n_threads, ..Default::default() };
                    Some(
                        MtmdContext::init_from_file(&p.to_string_lossy(), m, &params)
                            .with_context(|| format!("loading vision projector {}", p.display()))?,
                    )
                }
                None => None,
            };
            Ok(Session { ctx: m.new_context(backend, cparams)?, cached: Vec::new(), mtmd })
        })?;
        Ok(Some((loaded, gpu_layers)))
    }

    /// Runs one completion with `model_id`, loading it first if needed.
    /// `on_part` gets each decoded piece; return `false` to stop early.
    pub fn generate(
        &self,
        model_id: &str,
        msgs: &[Msg],
        p: &Params,
        mut on_part: impl for<'a> FnMut(Part<'a>) -> bool,
    ) -> Result<Done> {
        let info = self.info(model_id)?;
        if info.meta.embedding {
            bail!("{model_id} is an embedding model; it doesn't chat");
        }
        let images: Vec<&PathBuf> = msgs.iter().flat_map(|m| &m.images).collect();
        let (loaded, load_us) = self.acquire(model_id)?;
        // Raw completion has no reasoning block to sort out, and a grammar leaves no room for one.
        let thinks = info.thinks && !p.raw;
        let reasoning = if p.grammar.is_some() { Reasoning::Off } else { p.reasoning };
        let done = lock(&loaded).with_dependent_mut(|model, session| {
            let n_ctx = session.ctx.n_ctx() as usize;
            let started = Instant::now();
            let mut prompt = if p.raw {
                msgs.iter().map(|m| m.content.as_str()).collect::<String>()
            } else {
                render_prompt(model, msgs, !images.is_empty())?
            };
            // Some templates open the block themselves, so the output starts mid-thought.
            let opened = thinks && prompt.trim_end().ends_with(OPEN);
            if thinks && reasoning == Reasoning::Off {
                if !opened {
                    prompt.push_str(OPEN);
                }
                prompt.push_str(FORCED_CLOSE);
            }
            let mut split = ThinkSplit::new(opened && reasoning != Reasoning::Off);
            let budget = (thinks && reasoning == Reasoning::Light).then_some(LIGHT_BUDGET);

            let vocab = model.vocab();
            // Templates usually emit BOS themselves; avoid a double BOS.
            let bos = vocab.text(vocab.bos()).and_then(|s| s.to_str().ok()).unwrap_or("");
            let add_bos = bos.is_empty() || !prompt.starts_with(bos);
            let mut batch = LlamaBatch::new(session.ctx.n_batch() as usize, 1);
            let mut cached_tokens = 0;
            let prompt_tokens = if images.is_empty() {
                let tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);
                if tokens.is_empty() || tokens.len() >= n_ctx {
                    bail!("prompt is {} tokens, context is {n_ctx}", tokens.len());
                }
                cached_tokens = session.reuse(&tokens);
                session.feed(&mut batch, &tokens[cached_tokens..])?;
                tokens.len()
            } else {
                // shortcut: a prompt with images starts from an empty context instead of reusing a prefix; cache it if it gets slow.
                let mtmd = session.mtmd.as_mut().context("this model has no vision projector")?;
                let bitmaps = images
                    .iter()
                    .map(|p| {
                        MtmdBitmap::from_file(mtmd, &p.to_string_lossy(), false)
                            .with_context(|| format!("reading image {}", p.display()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let text = MtmdInputText { text: prompt.clone(), add_special: add_bos, parse_special: true };
                let chunks = mtmd.tokenize(text, &bitmaps.iter().collect::<Vec<_>>())?;
                if chunks.total_tokens() >= n_ctx {
                    bail!("prompt is {} tokens with the images, context is {n_ctx}", chunks.total_tokens());
                }
                session.ctx.clear_kv_cache();
                session.cached.clear();
                let n_batch = session.ctx.n_batch() as i32;
                let n_past = chunks.eval_chunks(mtmd, &mut session.ctx, 0, 0, n_batch, true)?;
                // Positions, not tokens, decide where the answer goes; `feed` counts them from `cached`.
                session.cached = vec![NOT_A_TOKEN; n_past as usize];
                // The image chunks were decoded outside `batch`; the sampler reads the last logits.
                batch.clear();
                chunks.total_tokens()
            };
            let prompt_us = started.elapsed().as_micros() as u64;
            let started = Instant::now();

            let mut chain = Vec::new();
            if let Some(g) = &p.grammar {
                chain.push(LlamaSampler::grammar(model, g, "root").map_err(|e| anyhow!("unusable grammar: {e}"))?);
            }
            if p.repeat_penalty > 0.0 && p.repeat_penalty != 1.0 {
                chain.push(LlamaSampler::penalties(model.n_vocab(), 64, p.repeat_penalty, 0.0, 0.0));
            }
            if p.temperature <= 0.0 {
                chain.push(LlamaSampler::greedy());
            } else {
                if p.top_k > 0 {
                    chain.push(LlamaSampler::top_k(p.top_k));
                }
                chain.push(LlamaSampler::top_p(p.top_p, 1));
                if p.min_p > 0.0 {
                    chain.push(LlamaSampler::min_p(p.min_p, 1));
                }
                chain.push(LlamaSampler::temp(p.temperature));
                chain.push(LlamaSampler::dist(p.seed.unwrap_or(u32::MAX))); // u32::MAX is llama.cpp's "seed randomly"
            }
            let mut sampler = LlamaSampler::chain_simple(chain);

            let mut produced = 0;
            let mut thought = 0;
            let mut utf8 = Utf8Buf::default();
            let mut stops = StopCut::new(&p.stop);
            let stopped = Cell::new(false);
            let mut emit = |part: Part| {
                if stopped.get() {
                    return;
                }
                let go_on = match part {
                    Part::Thinking(_) => on_part(part),
                    Part::Answer(text) => {
                        let (text, hit) = stops.push(text);
                        (text.is_empty() || on_part(Part::Answer(&text))) && !hit
                    }
                };
                stopped.set(!go_on);
            };
            // After images the logits to sample from are the context's last, not a slot of `batch`.
            let last_logits = |batch: &LlamaBatch| if batch.n_tokens() > 0 { batch.n_tokens() - 1 } else { -1 };
            let finish_reason = loop {
                if produced >= p.max_tokens || session.cached.len() >= n_ctx {
                    break Finish::Length;
                }
                let tok = sampler.sample(&session.ctx, last_logits(&batch));
                if vocab.is_eog(tok) {
                    break Finish::Stop;
                }
                produced += 1;
                split.push(&utf8.push(&vocab.token_to_piece(tok, false, None)), &mut emit);
                if stopped.get() {
                    break Finish::Stop;
                }
                session.feed(&mut batch, &[tok])?;
                if split.thinking() {
                    thought += 1;
                    if budget.is_some_and(|b| thought >= b) {
                        let close = vocab.tokenize(FORCED_CLOSE.as_bytes(), false, true);
                        if session.cached.len() + close.len() >= n_ctx {
                            break Finish::Length;
                        }
                        session.feed(&mut batch, &close)?;
                        split.close(&mut emit);
                    }
                }
            };
            split.finish(&mut emit);
            if !stopped.get() {
                let rest = stops.finish();
                if !rest.is_empty() {
                    on_part(Part::Answer(&rest));
                }
            }
            Ok(Done {
                prompt_tokens,
                completion_tokens: produced,
                finish_reason,
                cached_tokens,
                load_us,
                prompt_us,
                gen_us: started.elapsed().as_micros() as u64,
            })
        });
        self.touch(model_id);
        done
    }

    /// One vector per input, each of length 1, and the tokens read. Inputs longer than the model's
    /// context are cut to it.
    pub fn embed(&self, model_id: &str, inputs: &[String]) -> Result<(Vec<Vec<f32>>, usize)> {
        if !self.info(model_id)?.meta.embedding {
            bail!("{model_id} is not an embedding model");
        }
        let (loaded, _) = self.acquire(model_id)?;
        let out = lock(&loaded).with_dependent_mut(|model, session| {
            let n_ctx = session.ctx.n_ctx() as usize;
            let vocab = model.vocab();
            let mut batch = LlamaBatch::new(n_ctx, 1);
            let (mut vectors, mut read) = (Vec::with_capacity(inputs.len()), 0);
            for text in inputs {
                let mut tokens = vocab.tokenize(text.as_bytes(), true, false);
                tokens.truncate(n_ctx);
                if tokens.is_empty() {
                    bail!("an input to embed is empty");
                }
                read += tokens.len();
                session.ctx.clear_kv_cache();
                batch.clear();
                batch.add_sequence(&tokens, 0, false)?;
                session.ctx.decode(&mut batch)?;
                let v = session.ctx.embeddings_seq_ith(0).map_err(|_| anyhow!("{model_id} produced no embedding"))?;
                vectors.push(normalized(v));
            }
            Ok((vectors, read))
        });
        self.touch(model_id);
        out
    }

    /// Marks `id` as just used, so idle time counts from the end of a generation.
    fn touch(&self, id: &str) {
        if let Some(s) = lock(&self.slots).iter_mut().find(|s| s.id == id) {
            s.last_used = Instant::now();
        }
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(String::new, |s| s.to_string_lossy().into_owned())
}

/// `v` scaled to length 1, so a dot product of two is their cosine.
fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Every usable model under `paths`. Files already in `known` are not probed again.
fn scan(backend: &LlamaBackend, paths: &[PathBuf], known: &[ModelInfo]) -> Vec<ModelInfo> {
    let mut files = Vec::new();
    for p in paths {
        if let Err(e) = gguf::collect_gguf(p, &mut files) {
            eprintln!("reading {}: {e}", p.display());
        }
    }
    files.sort();
    let (projectors, files): (Vec<_>, Vec<_>) = files.into_iter().partition(|p| gguf::is_mmproj(&file_name(p)));
    let ollama = paths.iter().flat_map(|p| ollama_store::models(p)).map(|(id, path)| (path, Some(id)));
    let mut catalog: Vec<ModelInfo> = Vec::new();
    for (path, named) in files.into_iter().map(|p| (p, None)).chain(ollama) {
        let Some(id) = named.or_else(|| gguf::model_id(&file_name(&path)).map(str::to_owned)) else { continue };
        let Ok(size) = gguf::parts(&path).iter().map(|p| p.metadata().map(|m| m.len())).sum::<std::io::Result<u64>>()
        else {
            eprintln!("skipping {}: some of its parts are missing", path.display());
            continue;
        };
        if catalog.iter().any(|m| m.id == id) {
            eprintln!("skipping {}: another model is already called {id}", path.display());
            continue;
        }
        if let Some(m) = known.iter().find(|m| m.path == path) {
            catalog.push(ModelInfo { mmproj: None, size, ..m.clone() });
            continue;
        }
        match probe(backend, &path) {
            Ok((thinks, meta)) => catalog.push(ModelInfo { id, path, thinks, mmproj: None, meta, size }),
            Err(e) => eprintln!("skipping {}: {e:#}", path.display()),
        }
    }
    let all: Vec<PathBuf> = catalog.iter().map(|m| m.path.clone()).collect();
    for m in &mut catalog {
        m.mmproj = gguf::projector_for(&m.path, &all, &projectors);
    }
    catalog
}

/// Reads only metadata and vocabulary: what kind of model is this, and does it think? A file that
/// neither chats (no chat template) nor embeds is not a model lair can use.
fn probe(backend: &LlamaBackend, path: &Path) -> Result<(bool, Meta)> {
    let params = LlamaModelParams::default().with_vocab_only(true);
    let model = LlamaModel::load_from_file(backend, path, &params)?;
    let meta = gguf::meta_from(|key| model.meta_val_str(key).ok(), model.n_params());
    if meta.embedding {
        return Ok((false, meta));
    }
    let template = model.chat_template(None).context("no built-in chat template")?;
    let in_template = template.to_str().is_ok_and(|t| t.contains(OPEN));
    let special_token = model.vocab().tokenize(OPEN.as_bytes(), false, true).len() == 1;
    Ok((in_template || special_token, meta))
}

fn render_prompt(model: &LlamaModel, msgs: &[Msg], vision: bool) -> Result<String> {
    let tmpl = model.chat_template(None).context("model has no built-in chat template")?;
    let chat = msgs
        .iter()
        .map(|m| {
            // The image encoder replaces each marker with that image's embedding, in order.
            let marks = if vision { mtmd_default_marker().repeat(m.images.len()) } else { String::new() };
            LlamaChatMessage::new(m.role.clone(), format!("{marks}{}", m.content))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(model.apply_chat_template(&tmpl, &chat, true)?)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn the_idle_model_used_longest_ago_makes_room() {
        let now = Instant::now();
        let ago = |s| now - Duration::from_secs(s);
        assert_eq!(victim(&[(ago(5), false), (ago(90), true), (ago(30), false)]), Some(2), "never a busy one");
        assert_eq!(victim(&[(ago(5), true)]), None);
        assert_eq!((threads(1), threads(16), threads(128)), (1, 8, 16));
    }

    #[test]
    fn vectors_are_scaled_to_length_one() {
        let v = normalized(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert_eq!(normalized(&[0.0, 0.0]), [0.0, 0.0]);
    }

    /// The one engine a test process may have (llama.cpp allows a single backend), with the models
    /// the environment names, if any.
    pub(crate) fn test_engine() -> Arc<Engine> {
        static ENGINE: std::sync::OnceLock<Arc<Engine>> = std::sync::OnceLock::new();
        ENGINE
            .get_or_init(|| {
                let paths: Vec<PathBuf> = ["LAIR_TEST_MODEL", "LAIR_TEST_EMBED_MODEL"]
                    .iter()
                    .filter_map(std::env::var_os)
                    .map(Into::into)
                    .collect();
                Arc::new(Engine::new(&paths, Tune { ctx: Some(4096), ..Default::default() }).unwrap())
            })
            .clone()
    }

    /// Held by each test that runs a real model: they share the engine and assert on what is loaded.
    pub(crate) static REAL_MODEL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn test_params() -> Params {
        Params {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 600,
            seed: None,
            reasoning: Reasoning::Off,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 0.0,
            stop: vec![],
            grammar: None,
            raw: false,
        }
    }

    fn run(engine: &Engine, id: &str, msgs: &[Msg], p: &Params) -> (String, String, Done) {
        let (mut thought, mut answer) = (String::new(), String::new());
        let done = engine
            .generate(id, msgs, p, |part| {
                match part {
                    Part::Thinking(t) => thought.push_str(t),
                    Part::Answer(t) => answer.push_str(t),
                }
                true
            })
            .unwrap();
        (thought, answer, done)
    }

    fn chat_model(engine: &Engine) -> ModelInfo {
        engine.models().into_iter().find(|m| !m.meta.embedding).expect("LAIR_TEST_MODEL must be a chat model")
    }

    #[test]
    #[ignore = "needs LAIR_TEST_MODEL=/path/to/thinking-model.gguf"]
    fn reasoning_modes_with_a_real_model() {
        let _alone = REAL_MODEL.blocking_lock();
        let engine = &*test_engine();
        let model = chat_model(engine);
        let id = model.id.as_str();
        assert!(model.thinks, "test needs a model that thinks");
        assert!(model.meta.layers > 0 && model.meta.params > 0 && model.meta.ctx_train > 0, "{:?}", model.meta);
        let msgs = [Msg { role: "user".into(), content: "What is 2+2?".into(), ..Default::default() }];
        let base = test_params();

        let (thought, answer, done) = run(engine, id, &msgs, &base);
        assert!(thought.is_empty() && answer.contains('4'), "off: {thought:?} / {answer:?}");
        assert!(done.load_us > 0 && done.prompt_us > 0 && done.tps() > 0.0, "{done:?}");
        let (_, again, done) = run(engine, id, &msgs, &base);
        assert_eq!(again, answer, "a reused context answers like a fresh one");
        assert!(done.load_us == 0 && done.cached_tokens > 0, "{done:?}");

        let light = Params { reasoning: Reasoning::Light, max_tokens: LIGHT_BUDGET + 200, ..base.clone() };
        let (_, answer, done) = run(engine, id, &msgs, &light);
        assert!(!answer.is_empty(), "light must reach an answer within its budget");
        assert!(done.completion_tokens <= LIGHT_BUDGET + 200);

        let deep = Params { reasoning: Reasoning::Deep, max_tokens: 16, ..base.clone() };
        let (_, _, done) = run(engine, id, &msgs, &deep);
        assert_eq!((done.completion_tokens, done.finish_reason), (16, Finish::Length));
        // Other models (an embedding one, say) may be in memory beside it.
        let loaded = engine.loaded();
        assert_eq!(loaded.iter().find(|l| l.id == id).map(|l| l.ctx), Some(4096), "{loaded:?}");
    }

    #[test]
    #[ignore = "needs LAIR_TEST_MODEL=/path/to/model.gguf"]
    fn grammar_stop_and_raw_with_a_real_model() {
        let _alone = REAL_MODEL.blocking_lock();
        let engine = &*test_engine();
        let id = chat_model(engine).id;
        let ask = |text: &str| [Msg { role: "user".into(), content: text.into(), ..Default::default() }];
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "population": { "type": "integer" } },
            "required": ["city", "population"]
        });
        let json = Params { grammar: Some(schema_grammar(&schema).unwrap()), max_tokens: 200, ..test_params() };
        let (thought, answer, _) = run(engine, &id, &ask("Describe Paris as JSON."), &json);
        let v: serde_json::Value = serde_json::from_str(&answer).unwrap_or_else(|e| panic!("{e}: {answer:?}"));
        assert!(thought.is_empty() && v["city"].is_string() && v["population"].is_i64(), "{answer}");

        let stop = Params { stop: vec!["3".into()], max_tokens: 60, ..test_params() };
        let (_, answer, done) = run(engine, &id, &ask("Count from 1 to 9, digits separated by spaces."), &stop);
        assert!(answer.contains('2') && !answer.contains('3'), "{answer:?}");
        assert_eq!(done.finish_reason, Finish::Stop);

        let raw = Params { raw: true, max_tokens: 8, ..test_params() };
        let (thought, answer, _) = run(engine, &id, &ask("The capital of France is"), &raw);
        assert!(thought.is_empty() && answer.contains("Paris"), "{answer:?}");
    }

    #[test]
    #[ignore = "needs LAIR_TEST_MODEL=/path/to/model.gguf"]
    fn one_model_serves_two_generations_at_once() {
        let _alone = REAL_MODEL.blocking_lock();
        let engine = &*test_engine();
        let id = chat_model(engine).id;
        let msgs = [Msg { role: "user".into(), content: "Say hi.".into(), ..Default::default() }];
        let p = Params { max_tokens: 12, ..test_params() };
        std::thread::scope(|s| {
            let a = s.spawn(|| run(engine, &id, &msgs, &p).2);
            let b = s.spawn(|| run(engine, &id, &msgs, &p).2);
            assert!(a.join().unwrap().completion_tokens > 0 && b.join().unwrap().completion_tokens > 0);
        });
        let copies = |engine: &Engine| engine.loaded().iter().filter(|l| l.id == id).count();
        assert_eq!(copies(engine), 1, "they queued on one copy of the model");
        assert!(engine.unload(&id) && copies(engine) == 0);
    }

    #[test]
    #[ignore = "needs LAIR_TEST_EMBED_MODEL=/path/to/embedding-model.gguf"]
    fn embeddings_put_similar_texts_together() {
        let _alone = REAL_MODEL.blocking_lock();
        let engine = &*test_engine();
        let id = engine.models().into_iter().find(|m| m.meta.embedding).expect("an embedding model").id;
        let texts = ["How do I bake sourdough bread?", "Sourdough baking instructions", "The GPU has 12 GB of VRAM"];
        let (v, read) = engine.embed(&id, &texts.map(String::from)).unwrap();
        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(read > 10 && (cos(&v[0], &v[0]) - 1.0).abs() < 1e-3);
        assert!(cos(&v[0], &v[1]) > cos(&v[0], &v[2]) + 0.1, "{} vs {}", cos(&v[0], &v[1]), cos(&v[0], &v[2]));
    }
}
