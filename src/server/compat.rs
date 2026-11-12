//! The stateless chat API, in the widely used `/v1/chat/completions` message format, so editors and
//! other tools can talk to lair. Conversations, the agent and peers live in the native API.
//!
//! [`start`] is shared with the Ollama-style API: it turns the clients' messages (text or content
//! parts, tool calls and results, base64 images) into a generation, and splits the tool calls a
//! client offered out of the answer.

use std::convert::Infallible;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{self, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{AppState, error, now};
use crate::agent::{self, CallSplit, Piece, tools};
use crate::engine::{Done, Event, Finish, Msg, Params, Reasoning, schema_grammar, with_system};
use crate::models::AUTO;
use crate::store::Named;

/// A chat message as API clients send it.
#[derive(Deserialize)]
pub struct WireMsg {
    role: String,
    #[serde(default)]
    content: Option<Content>,
    #[serde(default)]
    tool_calls: Vec<WireCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
    /// The tool a `tool` message answers (Ollama's `tool_name`; optional in the OpenAI format).
    #[serde(default, alias = "tool_name")]
    name: Option<String>,
    /// Ollama's images: plain base64.
    #[serde(default)]
    images: Vec<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct WireCall {
    #[serde(default)]
    id: Option<String>,
    function: WireFn,
}

#[derive(Deserialize)]
struct WireFn {
    name: String,
    /// A JSON string in the OpenAI format, an object in Ollama's.
    #[serde(default)]
    arguments: Value,
}

/// A tool the client offers the model; the same shape in both APIs.
#[derive(Deserialize)]
pub struct ToolDef {
    function: FnDef,
}

#[derive(Deserialize)]
struct FnDef {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: Value,
}

/// Sampling options; `None` keeps the server's default.
#[derive(Default)]
pub struct Opts {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u32>,
    pub reasoning: Option<Reasoning>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub stop: Vec<String>,
    /// A JSON schema the answer must match. Ignored when tools are offered: a call is not in the schema.
    pub schema: Option<Value>,
    /// Plain completion: the prompt goes to the model as it is.
    pub raw: bool,
    pub tool_choice: ToolChoice,
}

/// What the client says about using the tools it offers.
#[derive(Debug, Default, PartialEq)]
pub enum ToolChoice {
    #[default]
    Auto,
    /// The tools are not shown to the model.
    None,
    // ponytail: "required" is asked for in the prompt, not enforced; constrain the answer to a
    // tool-call grammar if models ignore it.
    Required,
    Named(String),
}

impl ToolChoice {
    /// From the OpenAI field: `"none"`, `"auto"`, `"required"` or `{"type":"function","function":{"name":…}}`.
    pub fn parse(v: &Value) -> Self {
        match (v.as_str(), v["function"]["name"].as_str()) {
            (Some("none"), _) => Self::None,
            (Some("required"), _) => Self::Required,
            (_, Some(name)) => Self::Named(name.to_owned()),
            _ => Self::Auto,
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

/// The schema an OpenAI `response_format` asks for: any JSON object for `json_object`, the given one
/// for `json_schema`, none for `text`.
pub fn response_schema(format: &Value) -> Result<Option<Value>, String> {
    match format["type"].as_str() {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({ "type": "object" }))),
        Some("json_schema") => match &format["json_schema"]["schema"] {
            Value::Null => Err("response_format.json_schema.schema is missing".into()),
            schema => Ok(Some(schema.clone())),
        },
        Some(other) => Err(format!("unknown response_format type {other}")),
    }
}

/// What a generation produced, with offered tool calls split out of the answer.
pub enum Out {
    Thinking(String),
    Text(String),
    /// The model called a tool; more calls may follow, then a [`Out::Done`] with [`Finish::ToolCalls`].
    Call(agent::Call),
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

/// Starts a generation for an API request. Errors carry the status and message to answer with.
pub fn start(
    s: &AppState,
    model: Option<String>,
    wire: Vec<WireMsg>,
    tools: &[ToolDef],
    o: Opts,
) -> Result<Started, (StatusCode, String)> {
    let bad = |m: String| (StatusCode::BAD_REQUEST, m);
    if wire.is_empty() {
        return Err(bad("messages must not be empty".into()));
    }
    let mut images = TempImages::default();
    let mut messages = to_msgs(wire, &mut images).map_err(bad)?;
    let named = model.as_deref().is_some_and(|m| !m.is_empty() && m != AUTO);
    let route = s
        .lair
        .models
        .resolve(model.as_deref(), !images.0.is_empty())
        .map_err(|e| (if named { StatusCode::NOT_FOUND } else { StatusCode::BAD_REQUEST }, e))?;
    let model = route.id;
    if !images.0.is_empty() && !s.lair.models.sees_images(&model) {
        return Err(bad(format!("model {model} can't see images; use a local vision model")));
    }
    let tools = if o.tool_choice == ToolChoice::None { &[] } else { tools };
    if !tools.is_empty() {
        let mut prompt = tool_prompt(tools);
        match &o.tool_choice {
            ToolChoice::Required => prompt.push_str("\nYou must call a tool in this reply."),
            ToolChoice::Named(name) => {
                let _ = write!(prompt, "\nYou must call the tool {name} in this reply.");
            }
            ToolChoice::Auto | ToolChoice::None => {}
        }
        messages = with_system(&messages, &prompt);
    }
    let grammar = match &o.schema {
        Some(schema) if tools.is_empty() => Some(schema_grammar(schema).map_err(|e| bad(format!("{e:#}")))?),
        _ => None,
    };
    let d = s.lair.params();
    let params = Params {
        temperature: o.temperature.unwrap_or(d.temperature),
        top_p: o.top_p.unwrap_or(d.top_p),
        max_tokens: o.max_tokens.unwrap_or(d.max_tokens),
        seed: o.seed.or(d.seed),
        reasoning: o.reasoning.unwrap_or(d.reasoning),
        top_k: o.top_k.unwrap_or(d.top_k),
        min_p: o.min_p.unwrap_or(d.min_p),
        repeat_penalty: o.repeat_penalty.unwrap_or(d.repeat_penalty),
        stop: o.stop,
        grammar,
        raw: o.raw,
    };
    // A plain completion is the client's text and nothing else: no skill adds a system prompt to it.
    let (skill, messages) =
        if o.raw { (None, messages) } else { s.lair.skills.read().unwrap().apply(&messages, s.lair.assets.ids()) };
    let rx = s.lair.models.generate(&model, messages, params);
    let names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
    let (tx, out) = mpsc::unbounded_channel();
    tokio::spawn(split(rx, tx, names, images));
    Ok(Started { model, why: route.why, skill, out })
}

/// Forwards a generation, splitting out the calls to offered tools: a reply may hold several, and
/// ends with them. Dropping `rx` (the client left, or the model went on writing after its calls)
/// stops the engine; `_images` lives until the generation is over.
async fn split(
    mut rx: mpsc::UnboundedReceiver<Event>,
    tx: mpsc::UnboundedSender<Out>,
    names: Vec<String>,
    _images: TempImages,
) {
    let mut calls = (!names.is_empty()).then(|| CallSplit::new(names.iter().map(String::as_str)));
    let mut called = false;
    while let Some(event) = rx.recv().await {
        if tx.is_closed() {
            return;
        }
        let (pieces, last) = match event {
            Event::Thinking { text } => {
                let _ = tx.send(Out::Thinking(text));
                continue;
            }
            Event::Answer { text } => match calls.as_mut() {
                Some(c) => (c.push(&text), None),
                None => (vec![Piece::Text(text)], None),
            },
            Event::Done(d) => {
                let finish_reason = if called { Finish::ToolCalls } else { d.finish_reason };
                let rest = calls.as_mut().and_then(CallSplit::finish).into_iter().collect();
                (rest, Some(Out::Done(Done { finish_reason, ..d })))
            }
            Event::Failed { message } => (vec![], Some(Out::Failed(message))),
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
                // shortcut: the engine is stopped mid-way, so this reply's token counts are unknown.
                let _ = tx.send(Out::Done(Done { finish_reason: Finish::ToolCalls, ..Default::default() }));
                return;
            }
        }
        if let Some(out) = last {
            // A call found only at the very end (the model stopped before closing it) counts too.
            let _ = tx.send(match out {
                Out::Done(d) if called => Out::Done(Done { finish_reason: Finish::ToolCalls, ..d }),
                out => out,
            });
            return;
        }
    }
}

/// The protocol the agent uses, listing the client's tools.
fn tool_prompt(tools: &[ToolDef]) -> String {
    let mut out = format!("{}\nTools:\n", tools::PROTOCOL);
    for t in tools {
        let f = &t.function;
        let _ = writeln!(out, "- {}: {} Arguments (JSON Schema): {}", f.name, f.description, f.parameters);
    }
    out
}

/// Client messages as engine messages: tool traffic becomes the text protocol the agent uses (calls in
/// `<tool_call>`, results in a `<tool_response>` user message) and images become temporary files.
fn to_msgs(wire: Vec<WireMsg>, images: &mut TempImages) -> Result<Vec<Msg>, String> {
    // Tool results name the call they answer by id in the OpenAI format; this finds the tool's name.
    let mut call_names: Vec<(String, String)> = Vec::new();
    let mut out = Vec::with_capacity(wire.len());
    for m in wire {
        let mut msg = Msg { role: m.role, ..Default::default() };
        match m.content {
            Some(Content::Text(t)) => msg.content = t,
            Some(Content::Parts(parts)) => {
                for p in parts {
                    match p {
                        ContentPart::Text { text } => {
                            if !msg.content.is_empty() {
                                msg.content.push('\n');
                            }
                            msg.content.push_str(&text);
                        }
                        ContentPart::ImageUrl { image_url } => msg.images.push(save_image(&image_url.url, images)?),
                        ContentPart::Other => {}
                    }
                }
            }
            None => {}
        }
        for b64 in &m.images {
            msg.images.push(save_image(b64, images)?);
        }
        for c in m.tool_calls {
            let arguments = match c.function.arguments {
                Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
                Value::Null => json!({}),
                v => v,
            };
            if let Some(id) = c.id {
                call_names.push((id, c.function.name.clone()));
            }
            let call = json!({ "name": c.function.name, "arguments": arguments });
            msg.content.push_str(&format!("<tool_call>{call}</tool_call>"));
        }
        if msg.role == "tool" {
            let name = m
                .name
                .or_else(|| {
                    let id = m.tool_call_id?;
                    call_names.iter().find(|(i, _)| *i == id).map(|(_, n)| n.clone())
                })
                .unwrap_or_else(|| "tool".into());
            msg = Msg { role: "user".into(), content: agent::render_result(&name, &msg.content), ..msg };
        }
        out.push(msg);
    }
    Ok(out)
}

/// Writes a `data:` URL or plain base64 image to a temporary file. Web URLs are refused: the server
/// fetches nothing a client points it at.
fn save_image(src: &str, images: &mut TempImages) -> Result<PathBuf, String> {
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
        None => ("png", src),
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("image is not valid base64: {e}"))?;
    let path = std::env::temp_dir().join(format!("lair-img-{}.{ext}", crate::turn::random_id()));
    std::fs::write(&path, bytes).map_err(|e| format!("saving the image: {e}"))?;
    images.0.push(path.clone());
    Ok(path)
}

#[derive(Deserialize)]
pub struct ChatRequest {
    messages: Vec<WireMsg>,
    model: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    #[serde(alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    seed: Option<u32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    #[serde(alias = "repetition_penalty")]
    repeat_penalty: Option<f32>,
    stop: Option<Stop>,
    reasoning_effort: Option<Reasoning>,
    #[serde(default)]
    tools: Vec<ToolDef>,
    #[serde(default)]
    tool_choice: Value,
    #[serde(default)]
    response_format: Value,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: StreamOptions,
}

#[derive(Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// A model as `/v1/models` lists it: lair's own fields plus the two the OpenAI format asks for.
fn model_json(m: crate::engine::ModelEntry) -> Value {
    let owner = m.node.clone().unwrap_or_else(|| "local".into());
    let mut v = json!(m);
    v["object"] = json!("model");
    v["owned_by"] = json!(owner);
    v
}

pub async fn models(State(s): State<Arc<AppState>>) -> Json<Value> {
    let data: Vec<Value> = s.lair.models.list().into_iter().map(model_json).collect();
    Json(json!({ "object": "list", "data": data }))
}

pub async fn model(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match s.lair.models.list().into_iter().find(|m| m.id == id) {
        Some(m) => Json(model_json(m)).into_response(),
        None => error(StatusCode::NOT_FOUND, format!("model {id} not found")),
    }
}

/// A tool call in the OpenAI format, where arguments travel as a JSON string.
fn openai_call(c: &agent::Call, index: usize) -> Value {
    json!({ "index": index, "id": format!("call_{}", crate::turn::random_id()), "type": "function",
            "function": { "name": c.name, "arguments": c.arguments.to_string() } })
}

fn usage(d: &Done) -> Value {
    json!({ "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens,
            "total_tokens": d.prompt_tokens + d.completion_tokens,
            "prompt_tokens_details": { "cached_tokens": d.cached_tokens } })
}

/// How long the generation took, in the shape llama.cpp's server reports it.
fn timings(d: &Done) -> Value {
    json!({ "load_ms": d.load_us as f64 / 1e3, "prompt_ms": d.prompt_us as f64 / 1e3,
            "predicted_ms": d.gen_us as f64 / 1e3, "predicted_per_second": d.tps() })
}

/// What lair adds to a response: the skill it applied and why it chose the model, when it did either.
fn extras(skill: &Option<Named>, why: &Option<String>) -> Option<Value> {
    let mut v = json!({});
    if let Some(skill) = skill {
        v["skills"] = json!([skill]);
    }
    if let Some(why) = why {
        v["model_choice"] = json!(why);
    }
    (skill.is_some() || why.is_some()).then_some(v)
}

/// The response to a streaming request: `chunks` turns each piece into the JSON chunks to send.
fn stream(
    mut out: mpsc::UnboundedReceiver<Out>,
    skill: Option<Named>,
    first: Option<Value>,
    mut chunks: impl FnMut(Out) -> Vec<Value> + Send + 'static,
) -> Response {
    // One piece can be several chunks. When the client leaves, `rx` drops, the next send fails and
    // `out` drops with this task, which stops the generation.
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(e) = out.recv().await {
            for chunk in chunks(e) {
                if tx.send(chunk).is_err() {
                    return;
                }
            }
        }
    });
    let body = UnboundedReceiverStream::new(rx);
    let skill_event = skill.map(|k| sse::Event::default().event("lair.skill").data(json!(k).to_string()));
    let events = tokio_stream::iter(skill_event)
        .chain(tokio_stream::iter(first).chain(body).map(|v| sse::Event::default().data(v.to_string())))
        .chain(tokio_stream::once(sse::Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

pub async fn chat_completions(State(s): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    let schema = match response_schema(&req.response_format) {
        Ok(schema) => schema,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let opts = Opts {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        seed: req.seed,
        reasoning: req.reasoning_effort,
        top_k: req.top_k,
        min_p: req.min_p,
        repeat_penalty: req.repeat_penalty,
        stop: stops(req.stop),
        schema,
        raw: false,
        tool_choice: ToolChoice::parse(&req.tool_choice),
    };
    let Started { model, why, skill, mut out } = match start(&s, req.model, req.messages, &req.tools, opts) {
        Ok(started) => started,
        Err((status, message)) => return error(status, message),
    };
    let created = now();
    let id = format!("chatcmpl-{}", crate::turn::random_id());

    if !req.stream {
        let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
        let done = loop {
            match out.recv().await {
                Some(Out::Thinking(t)) => reasoning.push_str(&t),
                Some(Out::Text(t)) => content.push_str(&t),
                Some(Out::Call(c)) => calls.push(openai_call(&c, calls.len())),
                Some(Out::Done(done)) => break done,
                Some(Out::Failed(message)) => return error(StatusCode::BAD_REQUEST, message),
                None => return error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly"),
            }
        };
        let mut message = json!({ "role": "assistant", "content": content });
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning);
        }
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
        }
        let mut body = json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{ "index": 0, "finish_reason": done.finish_reason, "message": message }],
            "usage": usage(&done), "timings": timings(&done)
        });
        if let Some(lair) = extras(&skill, &why) {
            body["lair"] = lair;
        }
        return Json(body).into_response();
    }

    let chunk = move |choices: Value| json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": choices });
    let delta = |delta: Value, finish: Option<Finish>| json!([{ "index": 0, "delta": delta, "finish_reason": finish }]);
    let mut first = chunk(delta(json!({ "role": "assistant" }), None));
    if let Some(why) = &why {
        first["lair"] = json!({ "model_choice": why });
    }
    let include_usage = req.stream_options.include_usage;
    let mut calls = 0;
    stream(out, skill, Some(first), move |e| match e {
        Out::Thinking(text) => vec![chunk(delta(json!({ "reasoning_content": text }), None))],
        Out::Text(text) => vec![chunk(delta(json!({ "content": text }), None))],
        Out::Call(c) => {
            calls += 1;
            vec![chunk(delta(json!({ "tool_calls": [openai_call(&c, calls - 1)] }), None))]
        }
        Out::Done(done) => {
            let mut last = chunk(delta(json!({}), Some(done.finish_reason)));
            last["timings"] = timings(&done);
            let mut all = vec![last];
            if include_usage {
                let mut total = chunk(json!([]));
                total["usage"] = usage(&done);
                all.push(total);
            }
            all
        }
        Out::Failed(message) => vec![json!({ "error": { "message": message } })],
    })
}

/// One prompt or several, as clients send them; several are joined, since lair answers with one choice.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum Texts {
    One(String),
    Many(Vec<String>),
}

impl Texts {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    prompt: Texts,
    model: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    seed: Option<u32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    #[serde(alias = "repetition_penalty")]
    repeat_penalty: Option<f32>,
    stop: Option<Stop>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: StreamOptions,
}

/// Plain text completion: the prompt reaches the model as written, with no chat template.
pub async fn completions(State(s): State<Arc<AppState>>, Json(req): Json<CompletionRequest>) -> Response {
    let opts = Opts {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        seed: req.seed,
        top_k: req.top_k,
        min_p: req.min_p,
        repeat_penalty: req.repeat_penalty,
        stop: stops(req.stop),
        raw: true,
        ..Default::default()
    };
    let prompt = json!([{ "role": "user", "content": req.prompt.into_vec().concat() }]);
    let wire = serde_json::from_value(prompt).expect("the message built above is valid");
    let Started { model, mut out, .. } = match start(&s, req.model, wire, &[], opts) {
        Ok(started) => started,
        Err((status, message)) => return error(status, message),
    };
    let created = now();
    let id = format!("cmpl-{}", crate::turn::random_id());
    let body = move |choices: Value| json!({ "id": id, "object": "text_completion", "created": created, "model": model, "choices": choices });
    let choice = |text: &str, finish: Option<Finish>| json!([{ "index": 0, "text": text, "finish_reason": finish }]);

    if !req.stream {
        let mut text = String::new();
        loop {
            match out.recv().await {
                Some(Out::Text(t)) => text.push_str(&t),
                Some(Out::Thinking(_) | Out::Call(_)) => {}
                Some(Out::Done(done)) => {
                    let mut v = body(choice(&text, Some(done.finish_reason)));
                    v["usage"] = usage(&done);
                    v["timings"] = timings(&done);
                    return Json(v).into_response();
                }
                Some(Out::Failed(message)) => return error(StatusCode::BAD_REQUEST, message),
                None => return error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly"),
            }
        }
    }
    let include_usage = req.stream_options.include_usage;
    stream(out, None, None, move |e| match e {
        Out::Text(text) => vec![body(choice(&text, None))],
        Out::Thinking(_) | Out::Call(_) => vec![],
        Out::Done(done) => {
            let mut last = body(choice("", Some(done.finish_reason)));
            last["timings"] = timings(&done);
            let mut all = vec![last];
            if include_usage {
                let mut total = body(json!([]));
                total["usage"] = usage(&done);
                all.push(total);
            }
            all
        }
        Out::Failed(message) => vec![json!({ "error": { "message": message } })],
    })
}

/// The most texts one embeddings request may carry.
const MAX_INPUTS: usize = 2048;

/// Embeds `inputs` with `model`, or with the installed embedding model when none is named. Shared
/// with the Ollama-style API.
pub async fn embed(
    s: &AppState,
    model: Option<String>,
    inputs: Vec<String>,
) -> Result<(String, Vec<Vec<f32>>, usize), (StatusCode, String)> {
    let bad = |m: String| (StatusCode::BAD_REQUEST, m);
    if inputs.is_empty() || inputs.len() > MAX_INPUTS {
        return Err(bad(format!("send 1 to {MAX_INPUTS} texts to embed")));
    }
    let model = model
        .filter(|m| !m.is_empty())
        .or_else(|| s.lair.models.embedder())
        .ok_or_else(|| bad("no embedding model is installed; download one in Settings".into()))?;
    let (vectors, read) = s.lair.models.embed(&model, inputs).await.map_err(|e| bad(format!("{e:#}")))?;
    Ok((model, vectors, read))
}

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    input: Texts,
    model: Option<String>,
    /// `float` (the default) or `base64`, which the OpenAI clients ask for.
    encoding_format: Option<String>,
}

pub async fn embeddings(State(s): State<Arc<AppState>>, Json(req): Json<EmbeddingsRequest>) -> Response {
    let (model, vectors, read) = match embed(&s, req.model, req.input.into_vec()).await {
        Ok(done) => done,
        Err((status, message)) => return error(status, message),
    };
    let base64 = req.encoding_format.as_deref() == Some("base64");
    let data: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, v)| {
            let embedding = if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(v)
            };
            json!({ "object": "embedding", "index": index, "embedding": embedding })
        })
        .collect();
    Json(json!({ "object": "list", "data": data, "model": model,
                 "usage": { "prompt_tokens": read, "total_tokens": read } }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(v: Value) -> Vec<WireMsg> {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn tool_traffic_becomes_the_agent_text_protocol() {
        let msgs = to_msgs(
            wire(json!([
                { "role": "user", "content": [{ "type": "text", "text": "weather?" }] },
                { "role": "assistant", "content": null, "tool_calls": [
                    { "id": "c1", "type": "function", "function": { "name": "weather", "arguments": "{\"city\":\"Rio\"}" } }
                ] },
                { "role": "tool", "tool_call_id": "c1", "content": "31C" }
            ])),
            &mut TempImages::default(),
        )
        .unwrap();
        assert_eq!(msgs[0].content, "weather?");
        assert_eq!(msgs[1].content, r#"<tool_call>{"arguments":{"city":"Rio"},"name":"weather"}</tool_call>"#);
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].content, agent::render_result("weather", "31C"));
    }

    #[test]
    fn images_are_inline_only_and_cleaned_up() {
        let mut imgs = TempImages::default();
        let link = wire(json!([{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
        ] }]));
        assert!(to_msgs(link, &mut imgs).is_err());
        let inline = wire(json!([{ "role": "user", "images": ["aGk="] }]));
        let msgs = to_msgs(inline, &mut imgs).unwrap();
        let path = msgs[0].images[0].clone();
        assert_eq!(std::fs::read(&path).unwrap(), b"hi");
        drop(imgs);
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn a_client_that_left_stops_the_generation() {
        let (etx, erx) = mpsc::unbounded_channel();
        let (tx, out) = mpsc::unbounded_channel();
        drop(out);
        etx.send(Event::Answer { text: "one".into() }).unwrap();
        split(erx, tx, vec![], TempImages::default()).await;
        assert!(etx.send(Event::Answer { text: "two".into() }).is_err(), "the engine's receiver is gone");
    }

    #[tokio::test]
    async fn an_offered_tool_call_ends_the_answer() {
        let (etx, erx) = mpsc::unbounded_channel();
        let (tx, mut out) = mpsc::unbounded_channel();
        for t in ["Let me check. <tool_", "call>{\"name\": \"weather\", \"arguments\": {}}</tool_call> ignored"] {
            etx.send(Event::Answer { text: t.into() }).unwrap();
        }
        split(erx, tx, vec!["weather".into()], TempImages::default()).await;
        assert!(matches!(out.recv().await, Some(Out::Text(t)) if t == "Let me check. "));
        assert!(matches!(out.recv().await, Some(Out::Call(c)) if c.name == "weather"));
        assert!(matches!(out.recv().await, Some(Out::Done(d)) if d.finish_reason == Finish::ToolCalls));
        assert!(out.recv().await.is_none());
    }

    #[tokio::test]
    async fn a_reply_can_call_several_tools_and_keeps_its_token_counts() {
        let (etx, erx) = mpsc::unbounded_channel();
        let (tx, mut out) = mpsc::unbounded_channel();
        let calls = "<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Rio\"}}</tool_call>\n\
                     <tool_call>{\"name\": \"time\", \"arguments\": {}}</tool_call>";
        etx.send(Event::Answer { text: calls.into() }).unwrap();
        let done = Done { prompt_tokens: 7, completion_tokens: 5, gen_us: 9, ..Default::default() };
        etx.send(Event::Done(done)).unwrap();
        split(erx, tx, vec!["weather".into(), "time".into()], TempImages::default()).await;
        assert!(matches!(out.recv().await, Some(Out::Call(c)) if c.name == "weather"));
        assert!(matches!(out.recv().await, Some(Out::Call(c)) if c.name == "time"));
        let Some(Out::Done(d)) = out.recv().await else { panic!("the reply ends with its counts") };
        assert_eq!((d.finish_reason, d.prompt_tokens, d.completion_tokens, d.gen_us), (Finish::ToolCalls, 7, 5, 9));
        assert!(out.recv().await.is_none());
    }

    #[test]
    fn request_options_are_read_as_clients_send_them() {
        assert_eq!(ToolChoice::parse(&json!("none")), ToolChoice::None);
        assert_eq!(ToolChoice::parse(&json!("required")), ToolChoice::Required);
        let named = json!({ "type": "function", "function": { "name": "weather" } });
        assert_eq!(ToolChoice::parse(&named), ToolChoice::Named("weather".into()));
        assert_eq!(ToolChoice::parse(&Value::Null), ToolChoice::Auto);
        assert_eq!(stops(Some(Stop::One("END".into()))), ["END"]);
        let many = Stop::Many(vec!["a".into(), String::new(), "x".repeat(300), "b".into()]);
        assert_eq!(stops(Some(many)), ["a", "b"], "empty and overlong ones are dropped");
        assert!(stops(None).is_empty());
        assert_eq!(response_schema(&Value::Null), Ok(None));
        assert_eq!(response_schema(&json!({ "type": "json_object" })), Ok(Some(json!({ "type": "object" }))));
        let schema = json!({ "type": "json_schema", "json_schema": { "name": "x", "schema": { "type": "array" } } });
        assert_eq!(response_schema(&schema), Ok(Some(json!({ "type": "array" }))));
        assert!(response_schema(&json!({ "type": "json_schema" })).is_err());
    }
}
