//! The stateless chat API, in the widely used `/v1/chat/completions` message format, so editors and
//! other tools can talk to lair. Conversations, the agent and peers live in the native API.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{self, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt;

use super::core::{self, Convo, Out, Reply, Request, Sampling, Started, Stop, Texts, Tool, ToolChoice, stops};
use super::{ApiError, ApiResult, AppState, now};
use crate::agent::Call;
use crate::generation::{Done, Finish, Msg, Reasoning};
use crate::models::ModelEntry;
use crate::store::Named;
use crate::util::random_id;

/// A chat message as API clients send it.
#[derive(Deserialize)]
struct WireMsg {
    role: String,
    #[serde(default)]
    content: Option<Content>,
    #[serde(default)]
    tool_calls: Vec<WireCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
    /// The tool a `tool` message answers; optional, the call's id says it too.
    #[serde(default)]
    name: Option<String>,
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
    /// A JSON string; some clients send the object itself.
    #[serde(default)]
    arguments: Value,
}

#[derive(Deserialize)]
struct ToolDef {
    function: Tool,
}

fn convo(wire: Vec<WireMsg>) -> Result<Convo, String> {
    let mut c = Convo::default();
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
                        ContentPart::ImageUrl { image_url } => msg.images.push(c.image(&image_url.url)?),
                        ContentPart::Other => {}
                    }
                }
            }
            None => {}
        }
        for call in m.tool_calls {
            msg.content.push_str(&c.call(call.id, &call.function.name, call.function.arguments));
        }
        if msg.role == "tool" {
            c.result(m.name, m.tool_call_id.as_deref(), msg);
        } else {
            c.push(msg);
        }
    }
    Ok(c)
}

/// From `tool_choice`: `"none"`, `"auto"`, `"required"` or `{"type":"function","function":{"name":…}}`.
fn tool_choice(v: &Value) -> ToolChoice {
    match (v.as_str(), v["function"]["name"].as_str()) {
        (Some("none"), _) => ToolChoice::None,
        (Some("required"), _) => ToolChoice::Required,
        (_, Some(name)) => ToolChoice::Named(name.to_owned()),
        _ => ToolChoice::Auto,
    }
}

/// The schema a `response_format` asks for: any JSON object for `json_object`, the given one for
/// `json_schema`, none for `text`.
fn response_schema(format: &Value) -> Result<Option<Value>, String> {
    match format["type"].as_str() {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(core::any_object())),
        Some("json_schema") => match &format["json_schema"]["schema"] {
            Value::Null => Err("response_format.json_schema.schema is missing".into()),
            schema => Ok(Some(schema.clone())),
        },
        Some(other) => Err(format!("unknown response_format type {other}")),
    }
}

#[derive(Deserialize)]
pub struct ChatRequest {
    messages: Vec<WireMsg>,
    model: Option<String>,
    #[serde(flatten)]
    sampling: Sampling,
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
fn model_json(m: ModelEntry) -> Value {
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

pub async fn model(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    match s.lair.models.list().into_iter().find(|m| m.id == id) {
        Some(m) => Ok(Json(model_json(m))),
        None => Err(super::missing(format!("model {id} not found"))),
    }
}

/// A tool call in the OpenAI format, where arguments travel as a JSON string.
fn openai_call(c: &Call, index: usize) -> Value {
    json!({ "index": index, "id": format!("call_{}", random_id()), "type": "function",
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

/// The response to a streaming request: `head` first, then each piece as the chunks `chunks` makes of
/// it, then `[DONE]`.
fn stream(
    out: tokio::sync::mpsc::UnboundedReceiver<Out>,
    head: Vec<sse::Event>,
    mut chunks: impl FnMut(Out) -> Vec<Value> + Send + 'static,
) -> Response {
    let body =
        core::frames(out, move |o| chunks(o).into_iter().map(|v| sse::Event::default().data(v.to_string())).collect());
    let events = tokio_stream::iter(head)
        .chain(body)
        .chain(tokio_stream::once(sse::Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// The chunks that end a streamed reply: why it ended, and the token counts when the client asked.
fn last_chunks(done: &Done, include_usage: bool, chunk: impl Fn(Value) -> Value, end: Value) -> Vec<Value> {
    let mut last = chunk(end);
    last["timings"] = timings(done);
    let mut all = vec![last];
    if include_usage {
        let mut total = chunk(json!([]));
        total["usage"] = usage(done);
        all.push(total);
    }
    all
}

fn failed(message: String) -> Vec<Value> {
    vec![json!({ "error": { "message": message } })]
}

pub async fn chat_completions(State(s): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> ApiResult<Response> {
    let bad = |e: String| ApiError(StatusCode::BAD_REQUEST, e);
    let request = Request {
        model: req.model,
        convo: convo(req.messages).map_err(bad)?,
        tools: req.tools.into_iter().map(|t| t.function).collect(),
        tool_choice: tool_choice(&req.tool_choice),
        sampling: req.sampling,
        reasoning: req.reasoning_effort,
        stop: stops(req.stop),
        schema: response_schema(&req.response_format).map_err(bad)?,
        raw: false,
    };
    let Started { model, why, skill, out } = core::start(&s.lair, request)?;
    let created = now();
    let id = format!("chatcmpl-{}", random_id());

    if !req.stream {
        let Reply { thinking, text, calls, done } = core::collect(out).await?;
        let mut message = json!({ "role": "assistant", "content": text });
        if !thinking.is_empty() {
            message["reasoning_content"] = json!(thinking);
        }
        if !calls.is_empty() {
            let calls: Vec<Value> = calls.iter().enumerate().map(|(i, c)| openai_call(c, i)).collect();
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
        return Ok(Json(body).into_response());
    }

    let chunk = move |choices: Value| json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": choices });
    let delta = |delta: Value, finish: Option<Finish>| json!([{ "index": 0, "delta": delta, "finish_reason": finish }]);
    let mut first = chunk(delta(json!({ "role": "assistant" }), None));
    if let Some(why) = &why {
        first["lair"] = json!({ "model_choice": why });
    }
    let skill = skill.map(|k| sse::Event::default().event("lair.skill").data(json!(k).to_string()));
    let head = skill.into_iter().chain([sse::Event::default().data(first.to_string())]).collect();
    let include_usage = req.stream_options.include_usage;
    let mut calls = 0;
    Ok(stream(out, head, move |e| match e {
        Out::Thinking(text) => vec![chunk(delta(json!({ "reasoning_content": text }), None))],
        Out::Text(text) => vec![chunk(delta(json!({ "content": text }), None))],
        Out::Call(c) => {
            calls += 1;
            vec![chunk(delta(json!({ "tool_calls": [openai_call(&c, calls - 1)] }), None))]
        }
        Out::Done(done) => last_chunks(&done, include_usage, &chunk, delta(json!({}), Some(done.finish_reason))),
        Out::Failed(message) => failed(message),
    }))
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    /// Several prompts are joined: lair answers with one choice.
    prompt: Texts,
    model: Option<String>,
    #[serde(flatten)]
    sampling: Sampling,
    stop: Option<Stop>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: StreamOptions,
}

/// Plain text completion: the prompt reaches the model as written, with no chat template.
pub async fn completions(State(s): State<Arc<AppState>>, Json(req): Json<CompletionRequest>) -> ApiResult<Response> {
    let mut convo = Convo::default();
    let prompt: Vec<String> = req.prompt.into();
    convo.push(Msg { role: "user".into(), content: prompt.concat(), ..Default::default() });
    let request = Request {
        model: req.model,
        convo,
        sampling: req.sampling,
        stop: stops(req.stop),
        raw: true,
        ..Default::default()
    };
    let Started { model, out, .. } = core::start(&s.lair, request)?;
    let created = now();
    let id = format!("cmpl-{}", random_id());
    let body = move |choices: Value| json!({ "id": id, "object": "text_completion", "created": created, "model": model, "choices": choices });
    let choice = |text: &str, finish: Option<Finish>| json!([{ "index": 0, "text": text, "finish_reason": finish }]);

    if !req.stream {
        let Reply { text, done, .. } = core::collect(out).await?;
        let mut v = body(choice(&text, Some(done.finish_reason)));
        v["usage"] = usage(&done);
        v["timings"] = timings(&done);
        return Ok(Json(v).into_response());
    }
    let include_usage = req.stream_options.include_usage;
    Ok(stream(out, Vec::new(), move |e| match e {
        Out::Text(text) => vec![body(choice(&text, None))],
        Out::Thinking(_) | Out::Call(_) => vec![],
        Out::Done(done) => last_chunks(&done, include_usage, &body, choice("", Some(done.finish_reason))),
        Out::Failed(message) => failed(message),
    }))
}

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    input: Texts,
    model: Option<String>,
    /// `float` (the default) or `base64`, which the OpenAI clients ask for.
    encoding_format: Option<String>,
}

pub async fn embeddings(State(s): State<Arc<AppState>>, Json(req): Json<EmbeddingsRequest>) -> ApiResult<Json<Value>> {
    let e = core::embed(&s.lair, req.model, req.input.into()).await?;
    let base64 = req.encoding_format.as_deref() == Some("base64");
    let data: Vec<Value> = e
        .vectors
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
    Ok(Json(json!({ "object": "list", "data": data, "model": e.model,
                    "usage": { "prompt_tokens": e.tokens, "total_tokens": e.tokens } })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(v: Value) -> Vec<WireMsg> {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn tool_calls_and_results_travel_by_id() {
        let c = convo(wire(json!([
            { "role": "user", "content": [{ "type": "text", "text": "weather?" }] },
            { "role": "assistant", "content": null, "tool_calls": [
                { "id": "c1", "type": "function", "function": { "name": "weather", "arguments": "{\"city\":\"Rio\"}" } }
            ] },
            { "role": "tool", "tool_call_id": "c1", "content": "31C" }
        ])))
        .unwrap();
        let mut other = Convo::default();
        let call = other.call(None, "weather", json!({ "city": "Rio" }));
        other.result(Some("weather".into()), None, Msg { content: "31C".into(), ..Default::default() });
        let (a, b) = (c.into_msgs(), other.into_msgs());
        assert_eq!(a[0].content, "weather?");
        assert_eq!(a[1].content, call, "the string arguments are read as JSON");
        assert_eq!((a[2].role.as_str(), a[2].content.as_str()), (b[0].role.as_str(), b[0].content.as_str()));
        assert!(
            convo(wire(json!([{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
        ] }])))
            .is_err()
        );
    }

    #[test]
    fn request_options_are_read_as_clients_send_them() {
        assert_eq!(tool_choice(&json!("none")), ToolChoice::None);
        assert_eq!(tool_choice(&json!("required")), ToolChoice::Required);
        let named = json!({ "type": "function", "function": { "name": "weather" } });
        assert_eq!(tool_choice(&named), ToolChoice::Named("weather".into()));
        assert_eq!(tool_choice(&Value::Null), ToolChoice::Auto);
        assert_eq!(response_schema(&Value::Null), Ok(None));
        assert_eq!(response_schema(&json!({ "type": "json_object" })), Ok(Some(core::any_object())));
        let schema = json!({ "type": "json_schema", "json_schema": { "name": "x", "schema": { "type": "array" } } });
        assert_eq!(response_schema(&schema), Ok(Some(json!({ "type": "array" }))));
        assert!(response_schema(&json!({ "type": "json_schema" })).is_err());
        let req: ChatRequest = serde_json::from_value(json!({
            "messages": [], "max_completion_tokens": 9, "repetition_penalty": 1.1, "temperature": 0.2
        }))
        .unwrap();
        assert_eq!(
            (req.sampling.max_tokens, req.sampling.repeat_penalty),
            (Some(9), Some(1.1)),
            "other servers' names"
        );
    }
}
