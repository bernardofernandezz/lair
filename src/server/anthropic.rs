//! The Anthropic Messages API (`/v1/messages`), so Claude Code and other tools made for it run on local
//! models. Requests go through the same [`core::start`] as the other APIs; only the wire format differs.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt;

use super::core::{self, Convo, Out, Reply, Request, Sampling, Started, Tool, ToolChoice, stops};
use super::{ApiError, AppState, Dialect};
use crate::agent::Call;
use crate::generation::{Done, Finish, Msg, Reasoning};
use crate::models::AUTO;
use crate::util::random_id;

/// Anthropic content: a string or a list of typed blocks.
fn blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => vec![json!({ "type": "text", "text": s })],
        Value::Array(a) => a.clone(),
        _ => vec![],
    }
}

/// The text of a content value, blocks joined by newlines; non-text blocks are dropped.
fn text_of(content: &Value) -> String {
    let texts: Vec<String> = blocks(content).iter().filter_map(|b| b["text"].as_str().map(str::to_owned)).collect();
    texts.join("\n")
}

/// Anthropic `system` and `messages` as the conversation every API shares: `tool_use` blocks become
/// calls, each `tool_result` its own result message, base64 images files to see.
fn convo(system: &Value, messages: &[Value]) -> Result<Convo, String> {
    let mut c = Convo::default();
    let system = text_of(system);
    if !system.is_empty() {
        c.push(Msg { role: "system".into(), content: system, ..Default::default() });
    }
    for m in messages {
        let role = m["role"].as_str().ok_or("every message needs a role")?;
        let mut msg = Msg { role: role.into(), ..Default::default() };
        for b in blocks(&m["content"]) {
            match b["type"].as_str() {
                Some("text") => {
                    if !msg.content.is_empty() {
                        msg.content.push('\n');
                    }
                    msg.content.push_str(b["text"].as_str().unwrap_or_default());
                }
                Some("image") => {
                    let src = &b["source"];
                    if src["type"] != "base64" {
                        return Err("images must be sent inline as base64, not links".into());
                    }
                    let url = format!(
                        "data:{};base64,{}",
                        src["media_type"].as_str().unwrap_or_default(),
                        src["data"].as_str().unwrap_or_default()
                    );
                    msg.images.push(c.image(&url)?);
                }
                Some("tool_use") => {
                    let name = b["name"].as_str().ok_or("a tool_use block needs a name")?;
                    msg.content.push_str(&c.call(b["id"].as_str().map(str::to_owned), name, b["input"].clone()));
                }
                Some("tool_result") => {
                    let content = text_of(&b["content"]);
                    let content = if b["is_error"] == true { format!("Error: {content}") } else { content };
                    c.result(None, b["tool_use_id"].as_str(), Msg { content, ..Default::default() });
                }
                // Thinking from earlier turns, documents and anything newer: not shown to the model.
                _ => {}
            }
        }
        if !msg.content.is_empty() || !msg.images.is_empty() {
            c.push(msg);
        }
    }
    Ok(c)
}

#[derive(Deserialize)]
struct ToolDef {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    input_schema: Value,
}

fn tools(v: &Value) -> Result<Vec<Tool>, String> {
    if v.is_null() {
        return Ok(Vec::new());
    }
    let defs: Vec<ToolDef> = serde_json::from_value(v.clone()).map_err(|e| format!("invalid tools: {e}"))?;
    Ok(defs
        .into_iter()
        .map(|t| Tool { name: t.name, description: t.description, parameters: t.input_schema })
        .collect())
}

fn tool_choice(v: &Value) -> ToolChoice {
    match v["type"].as_str() {
        Some("none") => ToolChoice::None,
        Some("any") => ToolChoice::Required,
        Some("tool") => v["name"].as_str().map_or(ToolChoice::Auto, |n| ToolChoice::Named(n.to_owned())),
        _ => ToolChoice::Auto,
    }
}

fn stop_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "end_turn",
        Finish::Length => "max_tokens",
        Finish::ToolCalls => "tool_use",
    }
}

fn usage(d: &Done) -> Value {
    json!({ "input_tokens": d.prompt_tokens, "output_tokens": d.completion_tokens,
            "cache_read_input_tokens": d.cached_tokens })
}

fn tool_use(c: &Call) -> Value {
    json!({ "type": "tool_use", "id": format!("toolu_{}", random_id()), "name": c.name, "input": c.arguments })
}

fn request(req: &Value) -> Result<Request, String> {
    let msgs = req["messages"].as_array().ok_or("messages is required")?;
    // Tools made for Claude name Claude models; those mean "pick for me" here.
    let model = req["model"].as_str().filter(|m| !m.starts_with("claude-")).unwrap_or(AUTO).to_owned();
    let number = |key: &str| req[key].as_f64();
    Ok(Request {
        model: Some(model),
        convo: convo(&req["system"], msgs)?,
        tools: tools(&req["tools"])?,
        tool_choice: tool_choice(&req["tool_choice"]),
        sampling: Sampling {
            temperature: number("temperature").map(|v| v as f32),
            top_p: number("top_p").map(|v| v as f32),
            top_k: req["top_k"].as_i64().and_then(|v| i32::try_from(v).ok()),
            max_tokens: req["max_tokens"].as_u64().and_then(|v| usize::try_from(v).ok()),
            ..Default::default()
        },
        // shortcut: `budget_tokens` is not read; thinking is on or off.
        reasoning: match req["thinking"]["type"].as_str() {
            Some("enabled" | "adaptive") => Some(Reasoning::Deep),
            Some("disabled") => Some(Reasoning::Off),
            _ => None,
        },
        stop: stops(serde_json::from_value(req["stop_sequences"].clone()).ok()),
        ..Default::default()
    })
}

pub async fn messages(State(s): State<Arc<AppState>>, Json(req): Json<Value>) -> Response {
    let started = request(&req).map_err(|e| ApiError(StatusCode::BAD_REQUEST, e)).and_then(|r| core::start(&s.lair, r));
    let Started { model, out, .. } = match started {
        Ok(started) => started,
        Err(e) => return e.to(Dialect::Anthropic),
    };
    let id = format!("msg_{}", random_id());

    if req["stream"] != true {
        let Reply { thinking, text, calls, done } = match core::collect(out).await {
            Ok(reply) => reply,
            Err(e) => return e.to(Dialect::Anthropic),
        };
        let mut content = Vec::new();
        if !thinking.is_empty() {
            content.push(json!({ "type": "thinking", "thinking": thinking, "signature": "" }));
        }
        if !text.is_empty() {
            content.push(json!({ "type": "text", "text": text }));
        }
        content.extend(calls.iter().map(tool_use));
        return Json(json!({
            "id": id, "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason(done.finish_reason), "stop_sequence": null, "usage": usage(&done)
        }))
        .into_response();
    }

    let start = json!({ "type": "message_start", "message": {
        "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": null, "stop_sequence": null, "usage": { "input_tokens": 0, "output_tokens": 0 }
    } });
    let mut blocks = Blocks::default();
    let body = core::frames(out, move |o| blocks.frames(o));
    let events = tokio_stream::once(("message_start", start))
        .chain(body)
        .map(|(name, data)| Ok::<_, Infallible>(SseEvent::default().event(name).data(data.to_string())));
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// A kind of content block whose text arrives in deltas.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Thinking,
    Text,
}

/// The content blocks of a streamed message: a new block starts whenever the kind of text changes.
#[derive(Default)]
struct Blocks {
    index: usize,
    open: Option<Kind>,
}

type Frame = (&'static str, Value);

impl Blocks {
    fn frames(&mut self, o: Out) -> Vec<Frame> {
        let mut out = Vec::new();
        match o {
            Out::Thinking(t) => {
                self.delta(&mut out, Kind::Thinking, json!({ "type": "thinking_delta", "thinking": t }))
            }
            Out::Text(t) => self.delta(&mut out, Kind::Text, json!({ "type": "text_delta", "text": t })),
            Out::Call(c) => {
                self.close(&mut out);
                let mut block = tool_use(&c);
                let input = block["input"].take().to_string();
                block["input"] = json!({});
                let delta = json!({ "type": "input_json_delta", "partial_json": input });
                self.start(&mut out, block);
                out.push((
                    "content_block_delta",
                    json!({ "type": "content_block_delta", "index": self.index, "delta": delta }),
                ));
                self.stop(&mut out);
            }
            Out::Done(d) => {
                self.close(&mut out);
                let delta = json!({ "stop_reason": stop_reason(d.finish_reason), "stop_sequence": null });
                out.push(("message_delta", json!({ "type": "message_delta", "delta": delta, "usage": usage(&d) })));
                out.push(("message_stop", json!({ "type": "message_stop" })));
            }
            Out::Failed(m) => {
                out.push(("error", json!({ "type": "error", "error": { "type": "api_error", "message": m } })));
            }
        }
        out
    }

    fn start(&self, out: &mut Vec<Frame>, block: Value) {
        out.push((
            "content_block_start",
            json!({ "type": "content_block_start", "index": self.index, "content_block": block }),
        ));
    }

    fn stop(&mut self, out: &mut Vec<Frame>) {
        out.push(("content_block_stop", json!({ "type": "content_block_stop", "index": self.index })));
        self.index += 1;
    }

    fn close(&mut self, out: &mut Vec<Frame>) {
        if self.open.take().is_some() {
            self.stop(out);
        }
    }

    /// A delta for a thinking or text block, opening that block first if another kind is open.
    fn delta(&mut self, out: &mut Vec<Frame>, kind: Kind, delta: Value) {
        if self.open != Some(kind) {
            self.close(out);
            self.start(
                out,
                match kind {
                    Kind::Thinking => json!({ "type": "thinking", "thinking": "", "signature": "" }),
                    Kind::Text => json!({ "type": "text", "text": "" }),
                },
            );
            self.open = Some(kind);
        }
        out.push((
            "content_block_delta",
            json!({ "type": "content_block_delta", "index": self.index, "delta": delta }),
        ));
    }
}

/// Claude Code asks how long a prompt is before sending it.
// shortcut: about 4 characters per token, not the model's tokenizer; count with it if clients rely on precision.
pub async fn count_tokens(Json(req): Json<Value>) -> Json<Value> {
    let chars = req["system"].to_string().len() + req["messages"].to_string().len() + req["tools"].to_string().len();
    Json(json!({ "input_tokens": chars / 4 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_messages_become_the_shared_conversation() {
        let c = convo(
            &json!([{ "type": "text", "text": "Be brief." }]),
            &[
                json!({ "role": "user", "content": "weather in Rio?" }),
                json!({ "role": "assistant", "content": [
                    { "type": "thinking", "thinking": "hmm" },
                    { "type": "text", "text": "Checking." },
                    { "type": "tool_use", "id": "t1", "name": "weather", "input": { "city": "Rio" } }
                ] }),
                json!({ "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "t1", "content": [{ "type": "text", "text": "31C" }] },
                    { "type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": true },
                    { "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "aGk=" } }
                ] }),
            ],
        )
        .unwrap();
        let mut same = Convo::default();
        let call = same.call(None, "weather", json!({ "city": "Rio" }));
        let said = |content: &str| Msg { content: content.into(), ..Default::default() };
        same.result(Some("weather".into()), None, said("31C"));
        same.result(None, None, said("Error: boom"));
        let (msgs, want) = (c.into_msgs(), same.into_msgs());
        assert_eq!((msgs[0].role.as_str(), msgs[0].content.as_str()), ("system", "Be brief."));
        assert_eq!(msgs[1].content, "weather in Rio?");
        assert_eq!(msgs[2].content, format!("Checking.{call}"), "earlier thinking is left out");
        assert_eq!(msgs[3].content, want[0].content, "the result names the tool its call used");
        assert_eq!(msgs[4].content, want[1].content);
        assert_eq!(msgs[5].images.len(), 1);

        let link = json!({ "role": "user", "content": [{ "type": "image", "source": { "type": "url", "url": "https://x" } }] });
        assert!(convo(&Value::Null, &[link]).is_err());
    }

    #[test]
    fn tools_and_tool_choice_are_translated() {
        let t =
            tools(&json!([{ "name": "ls", "description": "List.", "input_schema": { "type": "object" } }])).unwrap();
        assert_eq!((t[0].name.as_str(), &t[0].parameters), ("ls", &json!({ "type": "object" })));
        assert!(tools(&json!([{ "description": "no name" }])).is_err());
        assert_eq!(tool_choice(&json!({ "type": "any" })), ToolChoice::Required);
        assert_eq!(tool_choice(&json!({ "type": "tool", "name": "ls" })), ToolChoice::Named("ls".into()));
        assert_eq!(tool_choice(&Value::Null), ToolChoice::Auto);
    }

    #[test]
    fn a_streamed_message_opens_a_block_per_kind_of_text() {
        let mut b = Blocks::default();
        let mut names = Vec::new();
        let call = Call { name: "ls".into(), arguments: json!({ "path": "." }) };
        let mut input = None;
        for o in [
            Out::Thinking("a".into()),
            Out::Text("b".into()),
            Out::Text("c".into()),
            Out::Call(call),
            Out::Done(Done::default()),
        ] {
            for (n, d) in b.frames(o) {
                input = input.or(d["delta"]["partial_json"].as_str().map(str::to_owned));
                names.push(format!("{n}:{}", d["index"]));
            }
        }
        assert_eq!(input.as_deref(), Some(r#"{"path":"."}"#), "a call's input streams as JSON text");
        assert_eq!(
            names,
            [
                "content_block_start:0",
                "content_block_delta:0",
                "content_block_stop:0",
                "content_block_start:1",
                "content_block_delta:1",
                "content_block_delta:1",
                "content_block_stop:1",
                "content_block_start:2",
                "content_block_delta:2",
                "content_block_stop:2",
                "message_delta:null",
                "message_stop:null",
            ]
        );
    }
}
