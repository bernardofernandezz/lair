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
use axum::extract::State;
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
use crate::engine::{Done, Event, Finish, Msg, Params, Reasoning, with_system};
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
}

/// What a generation produced, with offered tool calls split out of the answer.
pub enum Out {
    Thinking(String),
    Text(String),
    /// The model called a tool; nothing follows but a [`Out::Done`] with [`Finish::ToolCalls`].
    Call(agent::Call),
    Done(Done),
    Failed(String),
}

pub struct Started {
    pub model: String,
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
    let model = match model.filter(|m| !m.is_empty()) {
        Some(id) if s.lair.models.contains(&id) => id,
        Some(id) => return Err((StatusCode::NOT_FOUND, format!("model {id} not found"))),
        None => s.lair.models.default_id(),
    };
    let mut images = TempImages::default();
    let mut messages = to_msgs(wire, &mut images).map_err(bad)?;
    if !images.0.is_empty() && !s.lair.models.sees_images(&model) {
        return Err(bad(format!("model {model} can't see images; use a local vision model")));
    }
    if !tools.is_empty() {
        messages = with_system(&messages, &tool_prompt(tools));
    }
    let d = s.lair.params();
    let params = Params {
        temperature: o.temperature.unwrap_or(d.temperature),
        top_p: o.top_p.unwrap_or(d.top_p),
        max_tokens: o.max_tokens.unwrap_or(d.max_tokens),
        seed: o.seed.or(d.seed),
        reasoning: o.reasoning.unwrap_or(d.reasoning),
    };
    let (skill, messages) = s.lair.skills.read().unwrap().apply(&messages, s.lair.assets.ids());
    let rx = s.lair.models.generate(&model, messages, params);
    let names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
    let (tx, out) = mpsc::unbounded_channel();
    tokio::spawn(split(rx, tx, names, images));
    Ok(Started { model, skill, out })
}

/// Forwards a generation, cutting it at the first call to an offered tool. Dropping `rx` there stops
/// the engine; `_images` lives until the generation is over.
async fn split(
    mut rx: mpsc::UnboundedReceiver<Event>,
    tx: mpsc::UnboundedSender<Out>,
    names: Vec<String>,
    _images: TempImages,
) {
    let mut calls = (!names.is_empty()).then(|| CallSplit::new(names.iter().map(String::as_str)));
    while let Some(event) = rx.recv().await {
        let (pieces, last) = match event {
            Event::Thinking { text } => {
                let _ = tx.send(Out::Thinking(text));
                continue;
            }
            Event::Answer { text } => match calls.as_mut() {
                Some(c) => (c.push(&text), None),
                None => (vec![Piece::Text(text)], None),
            },
            Event::Done(d) => (calls.as_mut().and_then(CallSplit::finish).into_iter().collect(), Some(Out::Done(d))),
            Event::Failed { message } => (vec![], Some(Out::Failed(message))),
        };
        for piece in pieces {
            match piece {
                Piece::Text(t) => {
                    let _ = tx.send(Out::Text(t));
                }
                Piece::Call { parsed: Ok(call), .. } => {
                    let _ = tx.send(Out::Call(call));
                    // shortcut: token counts are unknown once the engine is cut off; count them if clients need usage here.
                    let done = Done { prompt_tokens: 0, completion_tokens: 0, finish_reason: Finish::ToolCalls };
                    let _ = tx.send(Out::Done(done));
                    return;
                }
                // A call the parser couldn't read reaches the client as the text the model wrote.
                Piece::Call { raw, parsed: Err(_) } => {
                    let _ = tx.send(Out::Text(raw));
                }
            }
        }
        if let Some(out) = last {
            let _ = tx.send(out);
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
    reasoning_effort: Option<Reasoning>,
    #[serde(default)]
    tools: Vec<ToolDef>,
    #[serde(default)]
    stream: bool,
}

pub async fn models(State(s): State<Arc<AppState>>) -> Json<Value> {
    let data: Vec<Value> = s
        .lair
        .models
        .list()
        .into_iter()
        .map(|m| {
            let owner = m.node.clone().unwrap_or_else(|| "local".into());
            let mut v = json!(m);
            v["object"] = json!("model");
            v["owned_by"] = json!(owner);
            v
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

/// A tool call in the OpenAI format, where arguments travel as a JSON string.
fn openai_call(c: &agent::Call, index: usize) -> Value {
    json!({ "index": index, "id": format!("call_{}", crate::turn::random_id()), "type": "function",
            "function": { "name": c.name, "arguments": c.arguments.to_string() } })
}

pub async fn chat_completions(State(s): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    let opts = Opts {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        seed: req.seed,
        reasoning: req.reasoning_effort,
    };
    let Started { model, skill, mut out } = match start(&s, req.model, req.messages, &req.tools, opts) {
        Ok(started) => started,
        Err((status, message)) => return error(status, message),
    };
    let created = now();
    let id = format!("chatcmpl-{created}");

    if !req.stream {
        let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
        let done = loop {
            match out.recv().await {
                Some(Out::Thinking(t)) => reasoning.push_str(&t),
                Some(Out::Text(t)) => content.push_str(&t),
                Some(Out::Call(c)) => calls.push(openai_call(&c, 0)),
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
            "usage": { "prompt_tokens": done.prompt_tokens, "completion_tokens": done.completion_tokens,
                       "total_tokens": done.prompt_tokens + done.completion_tokens }
        });
        if let Some(skill) = skill {
            body["lair"] = json!({ "skills": [skill] });
        }
        return Json(body).into_response();
    }

    let chunk = move |delta: Value, finish: Option<Finish>| {
        json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
    };
    let first = chunk(json!({ "role": "assistant" }), None);
    let body = UnboundedReceiverStream::new(out).map(move |e| match e {
        Out::Thinking(text) => chunk(json!({ "reasoning_content": text }), None),
        Out::Text(text) => chunk(json!({ "content": text }), None),
        Out::Call(c) => chunk(json!({ "tool_calls": [openai_call(&c, 0)] }), None),
        Out::Done(done) => chunk(json!({}), Some(done.finish_reason)),
        Out::Failed(message) => json!({ "error": { "message": message } }),
    });
    let skill_event = skill.map(|k| sse::Event::default().event("lair.skill").data(json!(k).to_string()));
    let events = tokio_stream::iter(skill_event)
        .chain(tokio_stream::once(first).chain(body).map(|v| sse::Event::default().data(v.to_string())))
        .chain(tokio_stream::once(sse::Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
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
}
