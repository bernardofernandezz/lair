//! What the chat APIs share. Each one (OpenAI-, Ollama- and Anthropic-style) turns its request into a
//! [`Request`], [`start`]s it, and renders the [`Out`] pieces that come back in its own words. A
//! conversation's tool traffic becomes the agent's text protocol, so any model can call the tools a
//! client offers; [`start`] splits those calls back out of the answer.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use axum::http::StatusCode;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::ApiError;
use crate::agent::{self, Call, CallSplit, Piece};
use crate::engine::schema_grammar;
use crate::generation::{Done, Event, Finish, Msg, Params, Reasoning, with_system};
use crate::lair::Lair;
use crate::models::Unresolved;
use crate::store::Named;

/// A tool the client offers the model.
#[derive(Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// A JSON schema of its arguments.
    #[serde(default)]
    pub parameters: Value,
}

/// What the client says about using the tools it offers.
#[derive(Debug, Default, PartialEq)]
pub enum ToolChoice {
    #[default]
    Auto,
    /// The tools are not shown to the model.
    None,
    // shortcut: "required" is asked for in the prompt, not enforced; constrain the answer to a
    // tool-call grammar if models ignore it.
    Required,
    Named(String),
}

/// Sampling options; `None` keeps the server's default. Its field names (and the aliases other servers
/// use) are the OpenAI API's, which flattens it into its requests; the other APIs fill it by hand.
#[derive(Default, Deserialize)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    pub seed: Option<u32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    #[serde(alias = "repetition_penalty")]
    pub repeat_penalty: Option<f32>,
}

impl Sampling {
    fn over(self, d: Params) -> Params {
        Params {
            temperature: self.temperature.unwrap_or(d.temperature),
            top_p: self.top_p.unwrap_or(d.top_p),
            max_tokens: self.max_tokens.unwrap_or(d.max_tokens),
            seed: self.seed.or(d.seed),
            top_k: self.top_k.unwrap_or(d.top_k),
            min_p: self.min_p.unwrap_or(d.min_p),
            repeat_penalty: self.repeat_penalty.unwrap_or(d.repeat_penalty),
            ..d
        }
    }
}

/// One stop sequence or several, as clients send them.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

/// Stop sequences within sane bounds: a handful, each short.
pub fn stops(stop: Option<Stop>) -> Vec<String> {
    let all = match stop {
        Some(Stop::One(s)) => vec![s],
        Some(Stop::Many(v)) => v,
        None => vec![],
    };
    all.into_iter().filter(|s| !s.is_empty() && s.len() <= 256).take(16).collect()
}

/// One text or several, as clients send prompts and inputs to embed.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum Texts {
    One(String),
    Many(Vec<String>),
}

impl From<Texts> for Vec<String> {
    fn from(t: Texts) -> Self {
        match t {
            Texts::One(s) => vec![s],
            Texts::Many(v) => v,
        }
    }
}

/// The schema that accepts any JSON object, for clients that ask for "JSON" without saying which.
pub fn any_object() -> Value {
    json!({ "type": "object" })
}

/// The conversation a request carries, as engine messages: tool calls in the agent's text protocol,
/// tool results as user messages naming their tool, and inline images as temporary files.
#[derive(Default)]
pub struct Convo {
    msgs: Vec<Msg>,
    images: TempImages,
    /// The tool each call id named: results name the call they answer by id.
    calls: HashMap<String, String>,
}

impl Convo {
    pub fn push(&mut self, msg: Msg) {
        self.msgs.push(msg);
    }

    #[cfg(test)]
    pub fn into_msgs(self) -> Vec<Msg> {
        let Convo { msgs, .. } = self;
        msgs
    }

    /// A call the model made earlier, as the text it goes back into the prompt as. Arguments may come
    /// as an object, as the JSON text of one, or not at all.
    pub fn call(&mut self, id: Option<String>, name: &str, arguments: Value) -> String {
        if let Some(id) = id {
            self.calls.insert(id, name.to_owned());
        }
        let arguments = match arguments {
            Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
            Value::Null => json!({}),
            v => v,
        };
        agent::render_call(name, &arguments)
    }

    /// A tool's result (`msg`, from the client's `tool` message), named by the tool or by the id of the
    /// call it answers.
    pub fn result(&mut self, name: Option<String>, call_id: Option<&str>, msg: Msg) {
        let name = name.or_else(|| self.calls.get(call_id?).cloned()).unwrap_or_else(|| "tool".into());
        self.push(Msg { role: "user".into(), content: agent::render_result(&name, &msg.content), ..msg });
    }

    /// Saves a `data:` URL or plain base64 image for the model to see. Web URLs are refused: the
    /// server fetches nothing a client points it at.
    pub fn image(&mut self, src: &str) -> Result<PathBuf, String> {
        let (ext, b64) = match src.strip_prefix("data:") {
            Some(rest) => {
                let (meta, data) = rest.split_once(',').ok_or("malformed data: URL")?;
                let mime = meta.strip_suffix(";base64").ok_or("only base64 data: URLs are accepted")?;
                let ext = match mime {
                    "image/png" => "png",
                    "image/jpeg" | "image/jpg" => "jpg",
                    "image/webp" => "webp",
                    "image/gif" => "gif",
                    "image/bmp" => "bmp",
                    other => return Err(format!("unsupported image type {other}")),
                };
                (ext, data)
            }
            None if src.starts_with("http://") || src.starts_with("https://") => {
                return Err("images must be sent inline as base64 data: URLs, not links".into());
            }
            // Plain base64 says nothing of its type; the image decoder reads the content, not the name.
            None => ("png", src),
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .map_err(|e| format!("image is not valid base64: {e}"))?;
        let path = std::env::temp_dir().join(format!("lair-img-{}.{ext}", crate::util::random_id()));
        std::fs::write(&path, bytes).map_err(|e| format!("saving the image: {e}"))?;
        self.images.0.push(path.clone());
        Ok(path)
    }
}

/// Images decoded for one generation, deleted when it ends.
#[derive(Default)]
struct TempImages(Vec<PathBuf>);

impl Drop for TempImages {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// A generation as an API asks for it.
#[derive(Default)]
pub struct Request {
    /// A model id, `auto`, or nothing (a local model lair picks).
    pub model: Option<String>,
    pub convo: Convo,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    pub sampling: Sampling,
    pub reasoning: Option<Reasoning>,
    pub stop: Vec<String>,
    /// A JSON schema the answer must match. Ignored when tools are offered: a call is not in the schema.
    pub schema: Option<Value>,
    /// Plain completion: the prompt goes to the model as it is.
    pub raw: bool,
}

/// What a generation produced, with offered tool calls split out of the answer.
pub enum Out {
    Thinking(String),
    Text(String),
    /// The model called a tool; more calls may follow, then a [`Out::Done`] with [`Finish::ToolCalls`].
    Call(Call),
    Done(Done),
    Failed(String),
}

pub struct Started {
    /// The model that answers: the one asked for, or the one lair chose.
    pub model: String,
    /// Why lair chose it, when the request left the choice open.
    pub why: Option<String>,
    pub skill: Option<Named>,
    pub out: mpsc::UnboundedReceiver<Out>,
}

/// A model named that doesn't exist is not found; no model that can do the job is a bad request.
pub fn unresolved(e: Unresolved) -> ApiError {
    let status = match e {
        Unresolved::NotFound(_) => StatusCode::NOT_FOUND,
        Unresolved::NoneFits(_) => StatusCode::BAD_REQUEST,
    };
    ApiError(status, e.to_string())
}

/// Starts a generation for an API request.
pub fn start(lair: &Lair, r: Request) -> Result<Started, ApiError> {
    let bad = |m: String| ApiError(StatusCode::BAD_REQUEST, m);
    let Convo { msgs: mut messages, images, .. } = r.convo;
    if messages.is_empty() {
        return Err(bad("messages must not be empty".into()));
    }
    let route = lair.models.resolve(r.model.as_deref(), !images.0.is_empty()).map_err(unresolved)?;
    let model = route.id;
    if !images.0.is_empty() && !lair.models.sees_images(&model) {
        return Err(bad(format!("model {model} can't see images; use a local vision model")));
    }
    let tools = if r.tool_choice == ToolChoice::None { &[][..] } else { &r.tools };
    if !tools.is_empty() {
        messages = with_system(&messages, &tool_prompt(tools, &r.tool_choice));
    }
    let grammar = match &r.schema {
        Some(schema) if tools.is_empty() => Some(schema_grammar(schema).map_err(|e| bad(format!("{e:#}")))?),
        _ => None,
    };
    let d = lair.params();
    let reasoning = r.reasoning.unwrap_or(d.reasoning);
    let params = Params { stop: r.stop, grammar, raw: r.raw, reasoning, ..r.sampling.over(d) };
    // A plain completion is the client's text and nothing else: no skill adds a system prompt to it.
    let (skill, messages) = if r.raw { (None, messages) } else { lair.skills.read().unwrap().apply(&messages) };
    let rx = lair.models.generate(&model, messages, params);
    let names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
    let (tx, out) = mpsc::unbounded_channel();
    tokio::spawn(split(rx, tx, names, images, lair.stopping.subscribe()));
    Ok(Started { model, why: route.why, skill, out })
}

/// Forwards a generation, splitting out the calls to offered tools: a reply may hold several, and
/// ends with them. Dropping `rx` (the client left, the model went on writing after its calls, or the
/// server is stopping) stops the engine; `_images` lives until the generation is over.
async fn split(
    mut rx: mpsc::UnboundedReceiver<Event>,
    tx: mpsc::UnboundedSender<Out>,
    names: Vec<String>,
    _images: TempImages,
    mut stopping: watch::Receiver<bool>,
) {
    let mut calls = (!names.is_empty()).then(|| CallSplit::new(names.iter().map(String::as_str)));
    let mut called = false;
    let as_called = |d: Done, called: bool| if called { Done { finish_reason: Finish::ToolCalls, ..d } } else { d };
    loop {
        let event = tokio::select! {
            e = rx.recv() => e,
            _ = stopping.wait_for(|s| *s) => Some(Event::Done(Done::default())),
        };
        let Some(event) = event else { return };
        if tx.is_closed() {
            return;
        }
        let (pieces, done) = match event {
            Event::Thinking { text } => {
                let _ = tx.send(Out::Thinking(text));
                continue;
            }
            Event::Answer { text } => match calls.as_mut() {
                Some(c) => (c.push(&text), None),
                None => (vec![Piece::Text(text)], None),
            },
            Event::Done(d) => (calls.as_mut().and_then(CallSplit::finish).into_iter().collect(), Some(d)),
            Event::Failed { message } => {
                let _ = tx.send(Out::Failed(message));
                return;
            }
        };
        for piece in pieces {
            let text = match piece {
                Piece::Call { parsed: Ok(call), .. } => {
                    let _ = tx.send(Out::Call(call));
                    called = true;
                    continue;
                }
                // A call the parser couldn't read reaches the client as the text the model wrote.
                Piece::Call { raw, parsed: Err(_) } => raw,
                Piece::Text(t) => t,
            };
            if !called {
                let _ = tx.send(Out::Text(text));
            } else if !text.trim().is_empty() {
                // The model kept writing after its calls (often a made-up tool result): cut it off.
                // shortcut: when the engine is stopped mid-way, this reply's token counts are unknown.
                let _ = tx.send(Out::Done(as_called(done.unwrap_or_default(), true)));
                return;
            }
        }
        if let Some(d) = done {
            // A call found only at the very end (the model stopped before closing it) counts too.
            let _ = tx.send(Out::Done(as_called(d, called)));
            return;
        }
    }
}

/// The protocol the agent uses, listing the client's tools.
fn tool_prompt(tools: &[Tool], choice: &ToolChoice) -> String {
    let mut out = format!("{}\nTools:\n", agent::PROTOCOL);
    for t in tools {
        let _ = writeln!(out, "- {}: {} Arguments (JSON Schema): {}", t.name, t.description, t.parameters);
    }
    match choice {
        ToolChoice::Required => out.push_str("\nYou must call a tool in this reply."),
        ToolChoice::Named(name) => {
            let _ = write!(out, "\nYou must call the tool {name} in this reply.");
        }
        ToolChoice::Auto | ToolChoice::None => {}
    }
    out
}

/// A whole reply, for a client that doesn't stream.
#[derive(Default)]
pub struct Reply {
    pub thinking: String,
    pub text: String,
    pub calls: Vec<Call>,
    pub done: Done,
}

pub async fn collect(mut out: mpsc::UnboundedReceiver<Out>) -> Result<Reply, ApiError> {
    let mut reply = Reply::default();
    loop {
        match out.recv().await {
            Some(Out::Thinking(t)) => reply.thinking.push_str(&t),
            Some(Out::Text(t)) => reply.text.push_str(&t),
            Some(Out::Call(c)) => reply.calls.push(c),
            Some(Out::Done(done)) => return Ok(Reply { done, ..reply }),
            Some(Out::Failed(message)) => return Err(ApiError(StatusCode::BAD_REQUEST, message)),
            None => return Err(ApiError(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly".into())),
        }
    }
}

/// A streamed reply, each piece turned into the frames an API sends for it. The client leaving drops
/// the stream, which drops `out` and stops the generation.
pub fn frames<T: Send + 'static>(
    mut out: mpsc::UnboundedReceiver<Out>,
    mut encode: impl FnMut(Out) -> Vec<T> + Send + 'static,
) -> UnboundedReceiverStream<T> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(piece) = out.recv().await {
            for frame in encode(piece) {
                if tx.send(frame).is_err() {
                    return;
                }
            }
        }
    });
    UnboundedReceiverStream::new(rx)
}

/// The most texts one embeddings request may carry.
const MAX_INPUTS: usize = 2048;

pub struct Embedded {
    pub model: String,
    pub vectors: Vec<Vec<f32>>,
    /// Tokens read.
    pub tokens: usize,
}

/// Embeds `inputs` with `model`, or with the installed embedding model when none is named.
pub async fn embed(lair: &Lair, model: Option<String>, inputs: Vec<String>) -> Result<Embedded, ApiError> {
    let bad = |m: String| ApiError(StatusCode::BAD_REQUEST, m);
    if inputs.is_empty() || inputs.len() > MAX_INPUTS {
        return Err(bad(format!("send 1 to {MAX_INPUTS} texts to embed")));
    }
    let model = model
        .filter(|m| !m.is_empty())
        .or_else(|| lair.models.embedder())
        .ok_or_else(|| bad("no embedding model is installed; download one in Settings".into()))?;
    let (vectors, tokens) = lair.models.embed(&model, inputs).await.map_err(|e| bad(format!("{e:#}")))?;
    Ok(Embedded { model, vectors, tokens })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run_split(events: Vec<Event>, tools: &[&str]) -> Vec<Out> {
        let (etx, erx) = mpsc::unbounded_channel();
        let (tx, mut out) = mpsc::unbounded_channel();
        for e in events {
            etx.send(e).unwrap();
        }
        drop(etx);
        let (_stop, stopping) = watch::channel(false);
        split(erx, tx, tools.iter().map(|t| t.to_string()).collect(), TempImages::default(), stopping).await;
        let mut all = Vec::new();
        while let Some(o) = out.recv().await {
            all.push(o);
        }
        all
    }

    fn answer(text: &str) -> Event {
        Event::Answer { text: text.into() }
    }

    #[test]
    fn tool_traffic_becomes_the_agent_text_protocol() {
        let mut c = Convo::default();
        let call = c.call(Some("c1".into()), "weather", json!({ "city": "Rio" }));
        assert!(crate::agent::CallSplit::new(["weather"]).push(&call).iter().any(|p| matches!(
            p,
            Piece::Call { parsed: Ok(Call { name, arguments }), .. } if name == "weather" && arguments["city"] == "Rio"
        )));
        let said = |content: &str| Msg { role: "tool".into(), content: content.into(), ..Default::default() };
        c.result(None, Some("c1"), said("31C"));
        c.result(None, Some("unknown"), said("?"));
        assert_eq!(c.msgs[0].role, "user");
        assert_eq!(c.msgs[0].content, agent::render_result("weather", "31C"), "named after the call it answers");
        assert_eq!(c.msgs[1].content, agent::render_result("tool", "?"));
    }

    #[test]
    fn call_arguments_are_read_however_clients_send_them() {
        let mut c = Convo::default();
        let object = c.call(None, "f", json!({ "a": 1 }));
        assert_eq!(c.call(None, "f", json!("{\"a\": 1}")), object, "the JSON text of an object");
        assert_eq!(c.call(None, "f", Value::Null), c.call(None, "f", json!({})), "none at all");
    }

    #[test]
    fn images_are_inline_only_and_cleaned_up() {
        let mut c = Convo::default();
        assert!(c.image("https://example.com/a.png").is_err());
        assert!(c.image("data:text/html;base64,aGk=").is_err());
        let path = c.image("aGk=").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hi");
        drop(c);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn a_client_that_left_stops_the_generation() {
        let (etx, erx) = mpsc::unbounded_channel();
        let (tx, out) = mpsc::unbounded_channel();
        drop(out);
        etx.send(answer("one")).unwrap();
        let (_stop, stopping) = watch::channel(false);
        split(erx, tx, vec![], TempImages::default(), stopping).await;
        assert!(etx.send(answer("two")).is_err(), "the engine's receiver is gone");
    }

    #[tokio::test]
    async fn a_server_stopping_ends_the_reply() {
        let (etx, erx) = mpsc::unbounded_channel();
        let (tx, mut out) = mpsc::unbounded_channel();
        let (stop, stopping) = watch::channel(false);
        let task = tokio::spawn(split(erx, tx, vec![], TempImages::default(), stopping));
        etx.send(answer("partial")).unwrap();
        assert!(matches!(out.recv().await, Some(Out::Text(t)) if t == "partial"));
        stop.send_replace(true);
        assert!(matches!(out.recv().await, Some(Out::Done(_))));
        task.await.unwrap();
        assert!(etx.send(answer("more")).is_err(), "the engine is stopped");
    }

    #[tokio::test]
    async fn an_offered_tool_call_ends_the_answer() {
        let done = Done { prompt_tokens: 7, completion_tokens: 5, ..Default::default() };
        let out = run_split(
            vec![
                answer("Let me check. <tool_"),
                answer("call>{\"name\": \"weather\", \"arguments\": {}}</tool_call> ignored"),
                Event::Done(done.clone()),
            ],
            &["weather"],
        )
        .await;
        assert!(matches!(&out[0], Out::Text(t) if t == "Let me check. "));
        assert!(matches!(&out[1], Out::Call(c) if c.name == "weather"));
        assert!(matches!(&out[2], Out::Done(d) if d.finish_reason == Finish::ToolCalls));
        assert_eq!(out.len(), 3);
    }

    #[tokio::test]
    async fn a_reply_can_call_several_tools_and_keeps_its_token_counts() {
        let calls = "<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Rio\"}}</tool_call>\n\
                     <tool_call>{\"name\": \"time\", \"arguments\": {}}</tool_call>";
        let done = Done { prompt_tokens: 7, completion_tokens: 5, gen_us: 9, ..Default::default() };
        let out = run_split(vec![answer(calls), Event::Done(done)], &["weather", "time"]).await;
        assert!(matches!(&out[0], Out::Call(c) if c.name == "weather"));
        assert!(matches!(&out[1], Out::Call(c) if c.name == "time"));
        let Out::Done(d) = &out[2] else { panic!("the reply ends with its counts") };
        assert_eq!((d.finish_reason, d.prompt_tokens, d.completion_tokens, d.gen_us), (Finish::ToolCalls, 7, 5, 9));
        assert_eq!(out.len(), 3);
    }

    #[tokio::test]
    async fn text_after_a_call_found_at_the_end_keeps_the_token_counts() {
        let done = Done { prompt_tokens: 7, completion_tokens: 5, ..Default::default() };
        let out = run_split(
            // `<too` could open another call, so it is held until the end, when it turns out to be text.
            vec![answer("<tool_call>{\"name\": \"time\", \"arguments\": {}}</tool_call> <too"), Event::Done(done)],
            &["time"],
        )
        .await;
        let Some(Out::Done(d)) = out.last() else { panic!("the reply ends") };
        assert_eq!((d.finish_reason, d.completion_tokens), (Finish::ToolCalls, 5));
    }

    #[test]
    fn stop_sequences_are_bounded() {
        assert_eq!(stops(Some(Stop::One("END".into()))), ["END"]);
        let many = Stop::Many(vec!["a".into(), String::new(), "x".repeat(300), "b".into()]);
        assert_eq!(stops(Some(many)), ["a", "b"], "empty and overlong ones are dropped");
        assert!(stops(None).is_empty());
    }
}
