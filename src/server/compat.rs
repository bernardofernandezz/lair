//! The stateless chat API, in the widely used `/v1/chat/completions` message format, so editors and
//! other tools can talk to lair. Conversations, the agent and peers live in the native API.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{self, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{AppState, error, now};
use crate::engine::{Event, Finish, Msg, Params, Reasoning};

#[derive(Deserialize)]
pub struct ChatRequest {
    messages: Vec<Msg>,
    model: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    seed: Option<u32>,
    reasoning_effort: Option<Reasoning>,
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

pub async fn chat_completions(State(s): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    if req.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let model = match req.model {
        Some(id) if s.lair.models.contains(&id) => id,
        Some(id) => return error(StatusCode::NOT_FOUND, format!("model {id} not found")),
        None => s.lair.models.default_id(),
    };
    let d = s.lair.params();
    let params = Params {
        temperature: req.temperature.unwrap_or(d.temperature),
        top_p: req.top_p.unwrap_or(d.top_p),
        max_tokens: req.max_tokens.unwrap_or(d.max_tokens),
        seed: req.seed.or(d.seed),
        reasoning: req.reasoning_effort.unwrap_or(d.reasoning),
    };
    let (skill, messages) = s.lair.skills.read().unwrap().apply(&req.messages, s.lair.assets.ids());
    let mut rx = s.lair.models.generate(&model, messages, params);
    let created = now();
    let id = format!("chatcmpl-{created}");

    if !req.stream {
        let (mut reasoning, mut content) = (String::new(), String::new());
        let done = loop {
            match rx.recv().await {
                Some(Event::Thinking { text }) => reasoning.push_str(&text),
                Some(Event::Answer { text }) => content.push_str(&text),
                Some(Event::Done(done)) => break done,
                Some(Event::Failed { message }) => return error(StatusCode::BAD_REQUEST, message),
                None => {
                    return error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly");
                }
            }
        };
        let mut message = json!({ "role": "assistant", "content": content });
        if !reasoning.is_empty() {
            message["reasoning_content"] = json!(reasoning);
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
    let body = UnboundedReceiverStream::new(rx).map(move |e| match e {
        Event::Thinking { text } => chunk(json!({ "reasoning_content": text }), None),
        Event::Answer { text } => chunk(json!({ "content": text }), None),
        Event::Done(done) => chunk(json!({}), Some(done.finish_reason)),
        Event::Failed { message } => json!({ "error": { "message": message } }),
    });
    let skill_event = skill.map(|k| sse::Event::default().event("lair.skill").data(json!(k).to_string()));
    let events = tokio_stream::iter(skill_event)
        .chain(tokio_stream::once(first).chain(body).map(|v| sse::Event::default().data(v.to_string())))
        .chain(tokio_stream::once(sse::Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}
