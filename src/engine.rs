use std::cell::Cell;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde::{Deserialize, Serialize};

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
/// What we feed the model to end its reasoning block for it.
const FORCED_CLOSE: &str = "\n\n</think>\n\n";
/// Thinking tokens allowed in `Reasoning::Light` before the block is closed for the model.
const LIGHT_BUDGET: usize = 512;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Msg {
    pub role: String,
    pub content: String,
    /// Image files shown to a vision model with this message. Local paths only: never read from or
    /// sent over the wire, so no API client or peer can point the engine at a file.
    #[serde(skip)]
    pub images: Vec<PathBuf>,
}

/// Appends `extra` to the first system message, or adds one at the start.
pub fn with_system(msgs: &[Msg], extra: &str) -> Vec<Msg> {
    let mut out = msgs.to_vec();
    match out.iter_mut().find(|m| m.role == "system") {
        Some(sys) if sys.content.is_empty() => sys.content = extra.into(),
        Some(sys) => sys.content = format!("{}\n\n{extra}", sys.content),
        None => out.insert(0, Msg { role: "system".into(), content: extra.into(), ..Default::default() }),
    }
    out
}

/// How much a reasoning model may think. Deserializes from the chat API's `reasoning_effort` values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
pub enum Reasoning {
    #[serde(rename = "none")]
    Off,
    #[serde(rename = "low", alias = "minimal")]
    Light,
    #[default]
    #[serde(rename = "high", alias = "medium")]
    Deep,
}

impl std::fmt::Display for Reasoning {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let v = serde_json::to_value(self).map_err(|_| std::fmt::Error)?;
        f.write_str(v.as_str().unwrap_or_default())
    }
}

impl std::str::FromStr for Reasoning {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        serde_json::from_value(serde_json::Value::String(s.into())).map_err(|_| "expected none, low or high".into())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Params {
    pub temperature: f32,
    pub top_p: f32,
    /// Counts thinking and answer tokens together.
    pub max_tokens: usize,
    /// `None` picks a fresh random seed per generation.
    pub seed: Option<u32>,
    pub reasoning: Reasoning,
}

/// A piece of generated text, already sorted into the model's reasoning or its answer.
#[derive(Debug, PartialEq)]
pub enum Part<'a> {
    Thinking(&'a str),
    Answer(&'a str),
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct Done {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: Finish,
}

/// Why a generation ended, in the chat API's words.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Finish {
    /// The model ended its answer, or the caller stopped it.
    Stop,
    /// It hit `max_tokens` or the context size.
    Length,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    /// Emits a `<think>` block before answering.
    pub thinks: bool,
    /// The multimodal projector found next to the model; with it the model sees images.
    pub mmproj: Option<PathBuf>,
}

/// One streamed piece of a generation. The stream always ends with `Done` or `Failed`. Peers send
/// these to each other as server-sent events named after the variant.
#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "event", content = "data", rename_all = "snake_case")]
pub enum Event {
    Thinking { text: String },
    Answer { text: String },
    Done(Done),
    Failed { message: String },
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
    /// Sees images (a projector was found next to it). Peers on older versions don't send it.
    #[serde(default)]
    pub vision: bool,
}

/// The loaded model's one context, reused across generations instead of allocated for each.
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

    /// Decodes `toks` after what is cached, keeping logits for the last one so it can be sampled from.
    fn feed(&mut self, batch: &mut LlamaBatch, toks: &[LlamaToken]) -> Result<()> {
        batch.clear();
        let (start, last) = (self.cached.len(), toks.len() - 1);
        for (k, t) in toks.iter().enumerate() {
            batch.add(*t, (start + k) as i32, &[0], k == last)?;
        }
        if let Err(e) = self.ctx.decode(batch) {
            self.ctx.clear_kv_cache();
            self.cached.clear();
            return Err(e.into());
        }
        self.cached.extend_from_slice(toks);
        Ok(())
    }
}

/// Every model found on disk; at most one is in memory at a time.
pub struct Engine {
    backend: LlamaBackend,
    gpu_layers: u32,
    n_ctx: u32,
    catalog: Vec<ModelInfo>,
    loaded: Option<(usize, Loaded)>,
}

impl Engine {
    /// `paths` are .gguf files or directories searched recursively for them.
    pub fn new(paths: &[PathBuf], gpu_layers: u32, n_ctx: u32) -> Result<Self> {
        // llama.cpp allows one backend per process, so it lives here rather than per model.
        let mut backend = LlamaBackend::init()?;
        if std::env::var_os("LAIR_VERBOSE").is_none() {
            backend.void_logs();
        }
        let mut files = Vec::new();
        for p in paths {
            collect_gguf(p, &mut files).with_context(|| format!("reading {}", p.display()))?;
        }
        files.sort();
        let (projectors, files): (Vec<_>, Vec<_>) = files.into_iter().partition(|p| is_mmproj(p));
        let mut catalog: Vec<ModelInfo> = Vec::new();
        for path in files {
            let id = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            if catalog.iter().any(|m| m.id == id) {
                eprintln!("skipping {}: another model is already called {id}", path.display());
                continue;
            }
            match probe(&backend, &path) {
                Ok(thinks) => catalog.push(ModelInfo { id, path, thinks, mmproj: None }),
                Err(e) => eprintln!("skipping {}: {e:#}", path.display()),
            }
        }
        if catalog.is_empty() {
            bail!("no usable .gguf chat models in {paths:?}");
        }
        let all: Vec<PathBuf> = catalog.iter().map(|m| m.path.clone()).collect();
        for m in &mut catalog {
            m.mmproj = projector_for(&m.path, &all, &projectors);
        }
        Ok(Self { backend, gpu_layers, n_ctx, catalog, loaded: None })
    }

    pub fn models(&self) -> &[ModelInfo] {
        &self.catalog
    }

    pub fn loaded(&self) -> Option<&str> {
        self.loaded.as_ref().map(|(i, _)| self.catalog[*i].id.as_str())
    }

    fn load(&mut self, id: &str) -> Result<usize> {
        let i = self.catalog.iter().position(|m| m.id == id).ok_or_else(|| anyhow!("unknown model {id}"))?;
        if self.loaded.as_ref().is_some_and(|(j, _)| *j == i) {
            return Ok(i);
        }
        self.loaded = None; // free the old weights before loading new ones; VRAM rarely fits two
        let path = &self.catalog[i].path;
        let params = LlamaModelParams::default().with_n_gpu_layers(self.gpu_layers);
        let model = LlamaModel::load_from_file(&self.backend, path, &params)
            .with_context(|| format!("loading model {}", path.display()))?;
        let (backend, n_ctx, mmproj) = (&self.backend, self.n_ctx, self.catalog[i].mmproj.clone());
        let loaded = Loaded::try_new(model, |m| -> Result<Session> {
            let params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx)).with_n_batch(n_ctx);
            let mtmd = match mmproj {
                Some(p) => {
                    let params = MtmdContextParams { use_gpu: self.gpu_layers > 0, ..Default::default() };
                    Some(
                        MtmdContext::init_from_file(&p.to_string_lossy(), m, &params)
                            .with_context(|| format!("loading vision projector {}", p.display()))?,
                    )
                }
                None => None,
            };
            Ok(Session { ctx: m.new_context(backend, params)?, cached: Vec::new(), mtmd })
        })?;
        self.loaded = Some((i, loaded));
        Ok(i)
    }

    /// Runs one completion with `model_id`, loading it first if needed.
    /// `on_part` gets each decoded piece; return `false` to stop early.
    pub fn generate(
        &mut self,
        model_id: &str,
        msgs: &[Msg],
        p: Params,
        mut on_part: impl for<'a> FnMut(Part<'a>) -> bool,
    ) -> Result<Done> {
        let images: Vec<&PathBuf> = msgs.iter().flat_map(|m| &m.images).collect();
        let i = self.load(model_id)?;
        let thinks = self.catalog[i].thinks;
        let n_ctx = self.n_ctx as usize;
        let loaded = &mut self.loaded.as_mut().expect("load() just set it").1;
        loaded.with_dependent_mut(|model, session| {
            let mut prompt = render_prompt(model, msgs, !images.is_empty())?;
            // Some templates open the block themselves, so the output starts mid-thought.
            let opened = thinks && prompt.trim_end().ends_with(OPEN);
            if thinks && p.reasoning == Reasoning::Off {
                if !opened {
                    prompt.push_str(OPEN);
                }
                prompt.push_str(FORCED_CLOSE);
            }
            let mut split = ThinkSplit::new(opened && p.reasoning != Reasoning::Off);
            let budget = (thinks && p.reasoning == Reasoning::Light).then_some(LIGHT_BUDGET);

            let vocab = model.vocab();
            // Templates usually emit BOS themselves; avoid a double BOS.
            let bos = vocab.text(vocab.bos()).and_then(|s| s.to_str().ok()).unwrap_or("");
            let add_bos = bos.is_empty() || !prompt.starts_with(bos);
            let mut batch = LlamaBatch::new(n_ctx, 1);
            let prompt_tokens = if images.is_empty() {
                let tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);
                if tokens.is_empty() || tokens.len() >= n_ctx {
                    bail!("prompt is {} tokens, context is {n_ctx}", tokens.len());
                }
                let reused = session.reuse(&tokens);
                session.feed(&mut batch, &tokens[reused..])?;
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
                let n_past = chunks.eval_chunks(mtmd, &mut session.ctx, 0, 0, n_ctx as i32, true)?;
                // Positions, not tokens, decide where the answer goes; `feed` counts them from `cached`.
                session.cached = vec![NOT_A_TOKEN; n_past as usize];
                chunks.total_tokens()
            };

            let mut sampler = if p.temperature <= 0.0 {
                LlamaSampler::greedy()
            } else {
                LlamaSampler::chain_simple([
                    LlamaSampler::top_p(p.top_p, 1),
                    LlamaSampler::temp(p.temperature),
                    LlamaSampler::dist(p.seed.unwrap_or(u32::MAX)), // u32::MAX is llama.cpp's "seed randomly"
                ])
            };

            let mut produced = 0;
            let mut thought = 0;
            let mut utf8 = Utf8Buf::default();
            let stopped = Cell::new(false);
            let mut emit = |part: Part| {
                if !stopped.get() && !on_part(part) {
                    stopped.set(true);
                }
            };
            let finish_reason = loop {
                if produced >= p.max_tokens || session.cached.len() >= n_ctx {
                    break Finish::Length;
                }
                let tok = sampler.sample(&session.ctx, batch.n_tokens() - 1);
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
            Ok(Done { prompt_tokens, completion_tokens: produced, finish_reason })
        })
    }
}

pub(crate) fn collect_gguf(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            collect_gguf(&entry?.path(), out)?;
        }
    } else if path.extension().is_some_and(|e| e == "gguf") {
        out.push(path.to_owned());
    }
    Ok(())
}

/// Reads only metadata and vocabulary: is this a chat model, and does it think?
fn probe(backend: &LlamaBackend, path: &Path) -> Result<bool> {
    let params = LlamaModelParams::default().with_vocab_only(true);
    let model = LlamaModel::load_from_file(backend, path, &params)?;
    let template = model.chat_template(None).context("no built-in chat template")?;
    let in_template = template.to_str().is_ok_and(|t| t.contains(OPEN));
    let special_token = model.vocab().tokenize(OPEN.as_bytes(), false, true).len() == 1;
    Ok(in_template || special_token)
}

/// Projector files ship next to the weights as `mmproj-*.gguf` (or `*-mmproj-*.gguf`).
fn is_mmproj(path: &Path) -> bool {
    path.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj"))
}

/// The projector for `model`: in its folder, named after it, or the only one there beside a lone model.
fn projector_for(model: &Path, models: &[PathBuf], projectors: &[PathBuf]) -> Option<PathBuf> {
    let dir = model.parent();
    let here: Vec<&PathBuf> = projectors.iter().filter(|p| p.parent() == dir).collect();
    let stem = model.file_stem()?.to_string_lossy().to_ascii_lowercase();
    // `Qwen2.5-VL-3B-Q4_K_M` → `qwen2.5-vl-3b`: projectors don't carry the weights' quantization.
    let base = stem.rsplit_once(['-', '.']).map_or(stem.as_str(), |(b, q)| {
        if q.starts_with('q') || q.starts_with("f16") || q.starts_with("bf16") { b } else { stem.as_str() }
    });
    let named =
        here.iter().find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains(base)));
    let alone = models.iter().filter(|m| m.parent() == dir).count() == 1 && here.len() == 1;
    named.or(alone.then(|| &here[0])).map(|p| (*p).clone())
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    /// Nothing but whitespace yet; a `<think>` here opens the reasoning block.
    Start,
    Thinking,
    /// Just closed the block; drop the whitespace models put before the answer.
    Closed,
    Answer,
}

/// Sorts streamed text into reasoning and answer, even when a tag is split across tokens.
struct ThinkSplit {
    state: State,
    pending: String,
}

impl ThinkSplit {
    fn new(opened: bool) -> Self {
        Self { state: if opened { State::Thinking } else { State::Start }, pending: String::new() }
    }

    fn thinking(&self) -> bool {
        self.state == State::Thinking
    }

    fn push(&mut self, text: &str, emit: &mut impl FnMut(Part)) {
        self.pending.push_str(text);
        loop {
            match self.state {
                State::Start => {
                    let t = self.pending.trim_start();
                    if let Some(rest) = t.strip_prefix(OPEN) {
                        self.pending = rest.to_owned();
                        self.state = State::Thinking;
                    } else if OPEN.starts_with(t) {
                        return;
                    } else {
                        self.pending = t.to_owned();
                        self.state = State::Answer;
                    }
                }
                State::Thinking => {
                    if let Some(i) = self.pending.find(CLOSE) {
                        if i > 0 {
                            emit(Part::Thinking(&self.pending[..i]));
                        }
                        self.pending.drain(..i + CLOSE.len());
                        self.state = State::Closed;
                    } else {
                        let keep = partial_suffix(&self.pending, CLOSE);
                        let n = self.pending.len() - keep;
                        if n > 0 {
                            emit(Part::Thinking(&self.pending[..n]));
                        }
                        self.pending.drain(..n);
                        return;
                    }
                }
                State::Closed => {
                    let t = self.pending.trim_start();
                    if t.is_empty() {
                        self.pending.clear();
                        return;
                    }
                    self.pending = t.to_owned();
                    self.state = State::Answer;
                }
                State::Answer => {
                    // A stray closing tag after the block is never answer text.
                    while let Some(i) = self.pending.find(CLOSE) {
                        self.pending.replace_range(i..i + CLOSE.len(), "");
                    }
                    let n = self.pending.len() - partial_suffix(&self.pending, CLOSE);
                    if n > 0 {
                        emit(Part::Answer(&self.pending[..n]));
                    }
                    self.pending.drain(..n);
                    return;
                }
            }
        }
    }

    /// Ends the reasoning block early (the caller fed the model a closing tag).
    fn close(&mut self, emit: &mut impl FnMut(Part)) {
        if self.state == State::Thinking {
            if !self.pending.is_empty() {
                emit(Part::Thinking(&self.pending));
            }
            self.pending.clear();
            self.state = State::Closed;
        }
    }

    fn finish(&mut self, emit: &mut impl FnMut(Part)) {
        let rest = std::mem::take(&mut self.pending);
        match self.state {
            State::Thinking if !rest.is_empty() => emit(Part::Thinking(&rest)),
            State::Start | State::Answer if !rest.trim().is_empty() => emit(Part::Answer(&rest)),
            _ => {}
        }
    }
}

/// Length of the longest proper prefix of `tag` that `s` ends with.
pub(crate) fn partial_suffix(s: &str, tag: &str) -> usize {
    (1..tag.len()).rev().find(|&k| s.ends_with(&tag[..k])).unwrap_or(0)
}

/// Tokens can split multi-byte UTF-8 characters; hold incomplete bytes until the rest arrives.
#[derive(Default)]
struct Utf8Buf(Vec<u8>);

impl Utf8Buf {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.0.extend_from_slice(bytes);
        match std::str::from_utf8(&self.0) {
            Ok(s) => {
                let out = s.to_owned();
                self.0.clear();
                out
            }
            Err(e) if e.error_len().is_none() => {
                let n = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.0[..n]).into_owned();
                self.0.drain(..n);
                out
            }
            Err(_) => String::from_utf8_lossy(&std::mem::take(&mut self.0)).into_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(opened: bool, chunks: &[&str]) -> (String, String) {
        let (mut thought, mut answer) = (String::new(), String::new());
        let mut s = ThinkSplit::new(opened);
        let mut emit = |p: Part| match p {
            Part::Thinking(t) => thought.push_str(t),
            Part::Answer(t) => answer.push_str(t),
        };
        for c in chunks {
            s.push(c, &mut emit);
        }
        s.finish(&mut emit);
        (thought, answer)
    }

    #[test]
    fn think_tags_split_across_tokens() {
        let (t, a) = split(false, &["<th", "ink>", "hm", "m</th", "ink>", "\n\n", "Oi", "!"]);
        assert_eq!((t.as_str(), a.as_str()), ("hmm", "Oi!"));
    }

    #[test]
    fn template_already_opened_the_block() {
        let (t, a) = split(true, &["plan", "</think>", " done"]);
        assert_eq!((t.as_str(), a.as_str()), ("plan", "done"));
    }

    #[test]
    fn think_tag_after_answer_started_is_text() {
        let (t, a) = split(false, &["Use ", "<think>", " tags"]);
        assert_eq!((t.as_str(), a.as_str()), ("", "Use <think> tags"));
        let (_, a) = split(false, &["Feito.</th", "ink>Pronto"]);
        assert_eq!(a, "Feito.Pronto", "a stray closing tag is dropped");
    }

    #[test]
    fn no_block_and_unfinished_tag() {
        assert_eq!(split(false, &["  Oi"]), ("".into(), "Oi".into()));
        assert_eq!(split(false, &["<thi"]), ("".into(), "<thi".into()));
        assert_eq!(split(false, &["<think>still going"]), ("still going".into(), "".into()));
    }

    #[test]
    fn utf8_split_across_tokens() {
        let mut b = Utf8Buf::default();
        let e = "é".as_bytes();
        assert_eq!(b.push(&[b'a', e[0]]), "a");
        assert_eq!(b.push(&[e[1]]), "é");
        assert_eq!(b.push(&[0xff]), "\u{fffd}");
    }

    #[test]
    fn projector_pairs_by_name_or_lone_folder() {
        let p = |s: &str| PathBuf::from(s);
        let models = [p("/m/Qwen2.5-VL-3B-Q4_K_M.gguf"), p("/m/Qwen3-4B-Q4_K_M.gguf"), p("/v/gemma.gguf")];
        let projs = [p("/m/mmproj-Qwen2.5-VL-3B-f16.gguf"), p("/v/mmproj-model-f16.gguf")];
        assert_eq!(projector_for(&models[0], &models, &projs), Some(projs[0].clone()));
        assert_eq!(projector_for(&models[1], &models, &projs), None, "a text model in a shared folder stays text-only");
        assert_eq!(
            projector_for(&models[2], &models, &projs),
            Some(projs[1].clone()),
            "lone model takes the lone projector"
        );
        assert!(is_mmproj(&projs[0]) && !is_mmproj(&models[0]));
    }

    #[test]
    #[ignore = "needs LAIR_TEST_MODEL=/path/to/thinking-model.gguf"]
    fn reasoning_modes_with_a_real_model() {
        let path = PathBuf::from(std::env::var("LAIR_TEST_MODEL").unwrap());
        let mut engine = Engine::new(&[path], 99, 4096).unwrap();
        let id = engine.models()[0].id.clone();
        assert!(engine.models()[0].thinks, "test needs a model that thinks");
        let msgs = [Msg { role: "user".into(), content: "What is 2+2?".into(), ..Default::default() }];
        let base = Params { temperature: 0.0, top_p: 1.0, max_tokens: 600, seed: None, reasoning: Reasoning::Off };
        let run = |engine: &mut Engine, p| {
            let (mut thought, mut answer) = (String::new(), String::new());
            let done = engine
                .generate(&id, &msgs, p, |part| {
                    match part {
                        Part::Thinking(t) => thought.push_str(t),
                        Part::Answer(t) => answer.push_str(t),
                    }
                    true
                })
                .unwrap();
            (thought, answer, done)
        };

        let (thought, answer, _) = run(&mut engine, base);
        assert!(thought.is_empty() && answer.contains('4'), "off: {thought:?} / {answer:?}");
        let (_, again, _) = run(&mut engine, base);
        assert_eq!(again, answer, "a reused context answers like a fresh one");

        let light = Params { reasoning: Reasoning::Light, max_tokens: LIGHT_BUDGET + 200, ..base };
        let (_, answer, done) = run(&mut engine, light);
        assert!(!answer.is_empty(), "light must reach an answer within its budget");
        assert!(done.completion_tokens <= LIGHT_BUDGET + 200);

        let (_, _, done) = run(&mut engine, Params { reasoning: Reasoning::Deep, max_tokens: 16, ..base });
        assert_eq!((done.completion_tokens, done.finish_reason), (16, Finish::Length));
        assert_eq!(engine.loaded(), Some(id.as_str()));
    }
}
