//! The Ollama-style API under `/api`, so the many apps that only speak it can use lair: point them at
//! lair's address instead of Ollama's. Generation goes through the same path as the OpenAI-style API.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::compat::{Opts, Out, Started, ToolDef, WireMsg, start};
use super::{AppState, now};
use crate::engine::{Finish, Reasoning};

/// The Ollama API version lair answers to, not lair's own: some clients refuse versions they see as old.
const API_VERSION: &str = "0.12.0";

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/version", get(version))
        .route("/tags", get(tags))
        .route("/show", post(show))
        .route("/chat", post(chat))
        .route("/generate", post(generate))
}

fn error(status: StatusCode, msg: impl ToString) -> Response {
    (status, Json(json!({ "error": msg.to_string() }))).into_response()
}

/// Ollama reads JSON whatever the content type says, and its docs' `curl -d` examples rely on that.
fn parse<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, String> {
    serde_json::from_slice(body).map_err(|e| format!("invalid request: {e}"))
}

async fn version() -> Json<Value> {
    Json(json!({ "version": API_VERSION }))
}

async fn tags(State(s): State<Arc<AppState>>) -> Json<Value> {
    let at = rfc3339(now());
    let models: Vec<Value> = s
        .lair
        .models
        .list()
        .into_iter()
        .map(|m| {
            json!({ "name": m.id, "model": m.id, "modified_at": at, "size": m.size, "digest": "",
                    "details": { "format": "gguf", "family": "", "parameter_size": "", "quantization_level": "" } })
        })
        .collect();
    Json(json!({ "models": models }))
}

#[derive(Deserialize)]
struct ShowReq {
    #[serde(alias = "name")]
    model: String,
}

async fn show(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: ShowReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let Some(m) = s.lair.models.list().into_iter().find(|m| m.id == req.model) else {
        return error(StatusCode::NOT_FOUND, format!("model {} not found", req.model));
    };
    // Tool calls are understood for every model: lair parses them out of plain text.
    let mut caps = vec!["completion", "tools"];
    caps.extend(m.thinks.then_some("thinking"));
    caps.extend(m.vision.then_some("vision"));
    Json(json!({ "modelfile": "", "parameters": "", "template": "", "capabilities": caps, "model_info": {},
                 "details": { "format": "gguf", "family": "", "parameter_size": "", "quantization_level": "" } }))
    .into_response()
}

#[derive(Default, Deserialize)]
struct Options {
    temperature: Option<f32>,
    top_p: Option<f32>,
    /// -1 means no limit.
    num_predict: Option<i64>,
    seed: Option<i64>,
}

impl Options {
    fn with(self, think: Option<Value>) -> Opts {
        Opts {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.num_predict.and_then(|n| usize::try_from(n).ok()),
            seed: self.seed.map(|n| n as u32),
            reasoning: think.and_then(|t| match t {
                Value::Bool(false) => Some(Reasoning::Off),
                Value::Bool(true) => Some(Reasoning::Deep),
                Value::String(s) if s == "low" => Some(Reasoning::Light),
                Value::String(_) => Some(Reasoning::Deep),
                _ => None,
            }),
        }
    }
}

#[derive(Deserialize)]
struct ChatReq {
    model: Option<String>,
    messages: Vec<WireMsg>,
    #[serde(default)]
    tools: Vec<ToolDef>,
    stream: Option<bool>,
    #[serde(default)]
    options: Options,
    think: Option<Value>,
}

async fn chat(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: ChatReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let started = start(&s, req.model, req.messages, &req.tools, req.options.with(req.think));
    reply(started, req.stream.unwrap_or(true), true).await
}

#[derive(Deserialize)]
struct GenReq {
    model: Option<String>,
    #[serde(default)]
    prompt: String,
    system: Option<String>,
    #[serde(default)]
    images: Vec<String>,
    stream: Option<bool>,
    #[serde(default)]
    options: Options,
    think: Option<Value>,
}

async fn generate(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: GenReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let mut msgs = Vec::new();
    if let Some(system) = req.system {
        msgs.push(json!({ "role": "system", "content": system }));
    }
    msgs.push(json!({ "role": "user", "content": req.prompt, "images": req.images }));
    let msgs = serde_json::from_value(Value::Array(msgs)).expect("messages built above are valid");
    let started = start(&s, req.model, msgs, &[], req.options.with(req.think));
    reply(started, req.stream.unwrap_or(true), false).await
}

/// One response object. `chat` picks the `/api/chat` shape (`message`) over `/api/generate`'s (`response`).
fn piece(model: &str, chat: bool, content: &str, thinking: &str, calls: &[Value]) -> Value {
    let mut v = json!({ "model": model, "created_at": rfc3339(now()), "done": false });
    if chat {
        let mut message = json!({ "role": "assistant", "content": content });
        if !thinking.is_empty() {
            message["thinking"] = json!(thinking);
        }
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
        }
        v["message"] = message;
    } else {
        v["response"] = json!(content);
        if !thinking.is_empty() {
            v["thinking"] = json!(thinking);
        }
    }
    v
}

fn finish(mut v: Value, d: &crate::engine::Done) -> Value {
    v["done"] = json!(true);
    v["done_reason"] = json!(if d.finish_reason == Finish::Length { "length" } else { "stop" });
    v["prompt_eval_count"] = json!(d.prompt_tokens);
    v["eval_count"] = json!(d.completion_tokens);
    v
}

fn call(c: &crate::agent::Call) -> Value {
    json!({ "function": { "name": c.name, "arguments": c.arguments } })
}

async fn reply(started: Result<Started, (StatusCode, String)>, stream: bool, chat: bool) -> Response {
    let Started { model, mut out, .. } = match started {
        Ok(s) => s,
        Err((status, message)) => return error(status, message),
    };
    if !stream {
        let (mut content, mut thinking, mut calls) = (String::new(), String::new(), Vec::new());
        loop {
            match out.recv().await {
                Some(Out::Thinking(t)) => thinking.push_str(&t),
                Some(Out::Text(t)) => content.push_str(&t),
                Some(Out::Call(c)) => calls.push(call(&c)),
                Some(Out::Done(d)) => {
                    return Json(finish(piece(&model, chat, &content, &thinking, &calls), &d)).into_response();
                }
                Some(Out::Failed(m)) => return error(StatusCode::BAD_REQUEST, m),
                None => return error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly"),
            }
        }
    }
    let lines = UnboundedReceiverStream::new(out).map(move |o| {
        let v = match o {
            Out::Thinking(t) => piece(&model, chat, "", &t, &[]),
            Out::Text(t) => piece(&model, chat, &t, "", &[]),
            Out::Call(c) => piece(&model, chat, "", "", &[call(&c)]),
            Out::Done(d) => finish(piece(&model, chat, "", "", &[]), &d),
            Out::Failed(m) => json!({ "error": m }),
        };
        Ok::<_, Infallible>(format!("{v}\n"))
    });
    ([(header::CONTENT_TYPE, "application/x-ndjson")], Body::from_stream(lines)).into_response()
}

/// A Unix time as RFC 3339 in UTC, the timestamp format Ollama clients parse.
fn rfc3339(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Howard Hinnant's days-to-civil algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_rfc3339() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_794_312_245), "2026-11-10T12:04:05Z");
    }

    #[test]
    fn think_maps_to_reasoning() {
        let r = |t: Value| Options::default().with(Some(t)).reasoning;
        assert_eq!(r(json!(false)), Some(Reasoning::Off));
        assert_eq!(r(json!(true)), Some(Reasoning::Deep));
        assert_eq!(r(json!("low")), Some(Reasoning::Light));
        assert_eq!(Options { num_predict: Some(-1), ..Default::default() }.with(None).max_tokens, None);
    }
}
