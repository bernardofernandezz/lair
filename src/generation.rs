//! What a generation is asked for and what it yields: shared by the engine, peers, the APIs and the store.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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

/// The fields after `reasoning` are absent from what older peers send; their defaults turn them off.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Params {
    pub temperature: f32,
    pub top_p: f32,
    /// Counts thinking and answer tokens together.
    pub max_tokens: usize,
    /// `None` picks a fresh random seed per generation.
    pub seed: Option<u32>,
    pub reasoning: Reasoning,
    /// 0 keeps every candidate.
    #[serde(default)]
    pub top_k: i32,
    #[serde(default)]
    pub min_p: f32,
    /// 0 or 1 leaves repeated tokens alone.
    #[serde(default)]
    pub repeat_penalty: f32,
    /// The answer ends before the first of these it would contain.
    #[serde(default)]
    pub stop: Vec<String>,
    /// GBNF the answer must match (see [`schema_grammar`]). Turns reasoning off: a thought is not in the grammar.
    #[serde(default)]
    pub grammar: Option<String>,
    /// The prompt is the messages' text as it is, with no chat template: plain completion.
    #[serde(default)]
    pub raw: bool,
}

impl Default for Params {
    /// What the command line starts from.
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 4096,
            seed: None,
            reasoning: Reasoning::default(),
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            stop: Vec::new(),
            grammar: None,
            raw: false,
        }
    }
}

/// A piece of generated text, already sorted into the model's reasoning or its answer.
#[derive(Debug, PartialEq)]
pub enum Part<'a> {
    Thinking(&'a str),
    Answer(&'a str),
}

/// The timings are absent from what older peers send.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct Done {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: Finish,
    /// Prompt tokens that were already in the context from the previous turn.
    #[serde(default)]
    pub cached_tokens: usize,
    /// Microseconds spent loading the model (0 when it was in memory), reading the prompt and generating.
    #[serde(default)]
    pub load_us: u64,
    #[serde(default)]
    pub prompt_us: u64,
    #[serde(default)]
    pub gen_us: u64,
}

impl Done {
    /// Generated tokens per second; 0 when nothing was timed.
    pub fn tps(&self) -> f32 {
        if self.gen_us == 0 { 0.0 } else { self.completion_tokens as f32 * 1e6 / self.gen_us as f32 }
    }
}

/// Why a generation ended, in the chat API's words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Finish {
    /// The model ended its answer, or the caller stopped it.
    #[default]
    Stop,
    /// It hit `max_tokens` or the context size.
    Length,
    /// It stopped to call a tool a chat API client offered.
    #[serde(rename = "tool_calls")]
    ToolCalls,
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
