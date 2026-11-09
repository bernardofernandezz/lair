use std::cell::Cell;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde::{Deserialize, Serialize};

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
/// What we feed the model to end its reasoning block for it.
const FORCED_CLOSE: &str = "\n\n</think>\n\n";
/// Thinking tokens allowed in `Reasoning::Light` before the block is closed for the model.
const LIGHT_BUDGET: usize = 512;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Msg {
    pub role: String,
    pub content: String,
}

/// Appends `extra` to the first system message, or adds one at the start.
pub fn with_system(msgs: &[Msg], extra: &str) -> Vec<Msg> {
    let mut out = msgs.to_vec();
    match out.iter_mut().find(|m| m.role == "system") {
        Some(sys) if sys.content.is_empty() => sys.content = extra.into(),
        Some(sys) => sys.content = format!("{}\n\n{extra}", sys.content),
        None => out.insert(0, Msg { role: "system".into(), content: extra.into() }),
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

#[derive(Debug)]
pub struct Done {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// "stop" (end-of-generation token or caller stopped) or "length" (hit max_tokens / context).
    pub finish_reason: &'static str,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    /// Emits a `<think>` block before answering.
    pub thinks: bool,
}

/// Every model found on disk; at most one is in memory at a time.
pub struct Engine {
    backend: LlamaBackend,
    gpu_layers: u32,
    n_ctx: u32,
    catalog: Vec<ModelInfo>,
    loaded: Option<(usize, LlamaModel)>,
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
        let mut catalog: Vec<ModelInfo> = Vec::new();
        for path in files {
            let id = path.file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            if catalog.iter().any(|m| m.id == id) {
                eprintln!("skipping {}: another model is already called {id}", path.display());
                continue;
            }
            match probe(&backend, &path) {
                Ok(thinks) => catalog.push(ModelInfo { id, path, thinks }),
                Err(e) => eprintln!("skipping {}: {e:#}", path.display()),
            }
        }
        if catalog.is_empty() {
            bail!("no usable .gguf chat models in {paths:?}");
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
        let i = self
            .catalog
            .iter()
            .position(|m| m.id == id)
            .ok_or_else(|| anyhow!("unknown model {id}"))?;
        if self.loaded.as_ref().is_some_and(|(j, _)| *j == i) {
            return Ok(i);
        }
        self.loaded = None; // free the old weights before loading new ones; VRAM rarely fits two
        let path = &self.catalog[i].path;
        let params = LlamaModelParams::default().with_n_gpu_layers(self.gpu_layers);
        let model = LlamaModel::load_from_file(&self.backend, path, &params)
            .with_context(|| format!("loading model {}", path.display()))?;
        self.loaded = Some((i, model));
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
        let i = self.load(model_id)?;
        let thinks = self.catalog[i].thinks;
        let model = &self.loaded.as_ref().expect("load() just set it").1;

        let mut prompt = render_prompt(model, msgs)?;
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
        let tokens = vocab.tokenize(prompt.as_bytes(), add_bos, true);

        let n_ctx = self.n_ctx as usize;
        if tokens.len() >= n_ctx {
            bail!("prompt is {} tokens, context is {n_ctx}", tokens.len());
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_ctx);
        let mut ctx = model.new_context(&self.backend, ctx_params)?;

        let mut batch = LlamaBatch::new(n_ctx, 1);
        let mut pos = 0;
        feed(&mut ctx, &mut batch, &tokens, &mut pos)?;

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
            if produced >= p.max_tokens || pos >= n_ctx {
                break "length";
            }
            let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
            if vocab.is_eog(tok) {
                break "stop";
            }
            produced += 1;
            split.push(&utf8.push(&vocab.token_to_piece(tok, false, None)), &mut emit);
            if stopped.get() {
                break "stop";
            }
            feed(&mut ctx, &mut batch, &[tok], &mut pos)?;
            if split.thinking() {
                thought += 1;
                if budget.is_some_and(|b| thought >= b) {
                    let close = vocab.tokenize(FORCED_CLOSE.as_bytes(), false, true);
                    if pos + close.len() >= n_ctx {
                        break "length";
                    }
                    feed(&mut ctx, &mut batch, &close, &mut pos)?;
                    split.close(&mut emit);
                }
            }
        };
        split.finish(&mut emit);
        Ok(Done { prompt_tokens: tokens.len(), completion_tokens: produced, finish_reason })
    }
}

/// Decodes `toks` at `pos`, keeping logits for the last one so it can be sampled from.
fn feed(ctx: &mut LlamaContext, batch: &mut LlamaBatch, toks: &[LlamaToken], pos: &mut usize) -> Result<()> {
    batch.clear();
    let last = toks.len() - 1;
    for (k, t) in toks.iter().enumerate() {
        batch.add(*t, (*pos + k) as i32, &[0], k == last)?;
    }
    ctx.decode(batch)?;
    *pos += toks.len();
    Ok(())
}

fn collect_gguf(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
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

fn render_prompt(model: &LlamaModel, msgs: &[Msg]) -> Result<String> {
    let tmpl = model.chat_template(None).context("model has no built-in chat template")?;
    let chat = msgs
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
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
    #[ignore = "needs LAIR_TEST_MODEL=/path/to/thinking-model.gguf"]
    fn reasoning_modes_with_a_real_model() {
        let path = PathBuf::from(std::env::var("LAIR_TEST_MODEL").unwrap());
        let mut engine = Engine::new(&[path], 99, 4096).unwrap();
        let id = engine.models()[0].id.clone();
        assert!(engine.models()[0].thinks, "test needs a model that thinks");
        let msgs = [Msg { role: "user".into(), content: "What is 2+2?".into() }];
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

        let light = Params { reasoning: Reasoning::Light, max_tokens: LIGHT_BUDGET + 200, ..base };
        let (_, answer, done) = run(&mut engine, light);
        assert!(!answer.is_empty(), "light must reach an answer within its budget");
        assert!(done.completion_tokens <= LIGHT_BUDGET + 200);

        let (_, _, done) = run(&mut engine, Params { reasoning: Reasoning::Deep, max_tokens: 16, ..base });
        assert_eq!((done.completion_tokens, done.finish_reason), (16, "length"));
        assert_eq!(engine.loaded(), Some(id.as_str()));
    }
}
