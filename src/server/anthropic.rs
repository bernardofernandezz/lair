//! The Anthropic Messages API (`/v1/messages`), so Claude Code and other tools made for it run on local
//! models. Requests are rewritten into the OpenAI shape and go through the same [`start`] as the
//! other APIs; only the wire format differs.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::AppState;
use super::compat::{Opts, Out, Started, ToolChoice, start, stops};
use crate::engine::{Done, Finish, Reasoning};
use crate::models::AUTO;

fn error(status: StatusCode, message: impl ToString) -> Response {
    let kind = match status {
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::BAD_REQUEST => "invalid_request_error",
        _ => "api_error",
    };
    let body = json!({ "type": "error", "error": { "type": kind, "message": message.to_string() } });
    (status, Json(body)).into_response()
}

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

/// Anthropic `system` and `messages` as OpenAI-format messages: `tool_use` blocks become `tool_calls`,
/// each `tool_result` its own `tool` message, base64 images `data:` URLs.
fn to_openai(system: &Value, messages: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    let system = text_of(system);
    if !system.is_empty() {
        out.push(json!({ "role": "system", "content": system }));
    }
    for m in messages {
        let role = m["role"].as_str().ok_or("every message needs a role")?;
        let (mut parts, mut calls) = (Vec::new(), Vec::new());
        for b in blocks(&m["content"]) {
            match b["type"].as_str() {
                Some("text") => parts.push(json!({ "type": "text", "text": b["text"] })),
                Some("image") => {
                    let src = &b["source"];
                    if src["type"] != "base64" {
                        return Err("images must be sent inline as base64, not links".into());
                    }
                    let url = format!(
                        "data:{};base64,{}",
                        src["media_type"].as_str().unwrap_or(""),
                        src["data"].as_str().unwrap_or("")
                    );
                    parts.push(json!({ "type": "image_url", "image_url": { "url": url } }));
                }
                Some("tool_use") => calls.push(json!({
                    "id": b["id"], "type": "function",
                    "function": { "name": b["name"], "arguments": b["input"] }
                })),
                Some("tool_result") => {
                    let mut content = text_of(&b["content"]);
                    if b["is_error"] == true {
                        content = format!("Error: {content}");
                    }
                    out.push(json!({ "role": "tool", "tool_call_id": b["tool_use_id"], "content": content }));
                }
                // Thinking from earlier turns, documents and anything newer: not shown to the model.
                _ => {}
            }
        }
        if !parts.is_empty() || !calls.is_empty() {
            let mut msg = json!({ "role": role, "content": parts });
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            out.push(msg);
        }
    }
    Ok(out)
}

/// Anthropic tools (`name`, `description`, `input_schema`) in the OpenAI shape.
fn to_openai_tools(tools: &Value) -> Value {
    let tools = tools.as_array().map(Vec::as_slice).unwrap_or_default();
    json!(
        tools
            .iter()
            .map(|t| json!({ "type": "function", "function": {
        "name": t["name"], "description": t["description"].as_str().unwrap_or(""), "parameters": t["input_schema"]
    } }))
            .collect::<Vec<_>>()
    )
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

fn tool_use(c: &crate::agent::Call) -> Value {
    json!({ "type": "tool_use", "id": format!("toolu_{}", crate::turn::random_id()), "name": c.name, "input": c.arguments })
}

pub async fn messages(State(s): State<Arc<AppState>>, Json(req): Json<Value>) -> Response {
    let Some(msgs) = req["messages"].as_array() else {
        return error(StatusCode::BAD_REQUEST, "messages is required");
    };
    let wire = match to_openai(&req["system"], msgs)
        .and_then(|v| serde_json::from_value(json!(v)).map_err(|e| format!("invalid messages: {e}")))
    {
        Ok(w) => w,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let tools = match serde_json::from_value::<Vec<_>>(to_openai_tools(&req["tools"])) {
        Ok(t) => t,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid tools: {e}")),
    };
    // Tools made for Claude name Claude models; those mean "pick for me" here.
    let model = req["model"].as_str().filter(|m| !m.starts_with("claude-")).unwrap_or(AUTO).to_owned();
    let stop = serde_json::from_value(req["stop_sequences"].clone()).ok();
    let opts = Opts {
        temperature: req["temperature"].as_f64().map(|v| v as f32),
        top_p: req["top_p"].as_f64().map(|v| v as f32),
        top_k: req["top_k"].as_i64().map(|v| v as i32),
        max_tokens: req["max_tokens"].as_u64().map(|v| v as usize),
        reasoning: match req["thinking"]["type"].as_str() {
            Some("enabled" | "adaptive") => Some(Reasoning::Deep),
            Some("disabled") => Some(Reasoning::Off),
            _ => None,
        },
        stop: stops(stop),
        tool_choice: tool_choice(&req["tool_choice"]),
        ..Default::default()
    };
    let Started { model, mut out, .. } = match start(&s, Some(model), wire, &tools, opts) {
        Ok(started) => started,
        Err((status, message)) => return error(status, message),
    };
    let id = format!("msg_{}", crate::turn::random_id());

    if req["stream"] != true {
        let (mut thinking, mut text, mut calls) = (String::new(), String::new(), Vec::new());
        let done = loop {
            match out.recv().await {
                Some(Out::Thinking(t)) => thinking.push_str(&t),
                Some(Out::Text(t)) => text.push_str(&t),
                Some(Out::Call(c)) => calls.push(tool_use(&c)),
                Some(Out::Done(d)) => break d,
                Some(Out::Failed(m)) => return error(StatusCode::BAD_REQUEST, m),
                None => return error(StatusCode::INTERNAL_SERVER_ERROR, "generation ended unexpectedly"),
            }
        };
        let mut content = Vec::new();
        if !thinking.is_empty() {
            content.push(json!({ "type": "thinking", "thinking": thinking, "signature": "" }));
        }
        if !text.is_empty() {
            content.push(json!({ "type": "text", "text": text }));
        }
        content.extend(calls);
        return Json(json!({
            "id": id, "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop_reason(done.finish_reason), "stop_sequence": null, "usage": usage(&done)
        }))
        .into_response();
    }

    let (tx, rx) = mpsc::unbounded_channel::<(&'static str, Value)>();
    tokio::spawn(async move {
        let start = json!({ "type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": null, "stop_sequence": null, "usage": { "input_tokens": 0, "output_tokens": 0 }
        } });
        let mut b = Blocks { tx, index: 0, open: None };
        if !b.send("message_start", start) {
            return;
        }
        // Dropping `out` when the client leaves (a send fails) stops the generation.
        while let Some(e) = out.recv().await {
            let ok = match e {
                Out::Thinking(t) => b.delta("thinking", json!({ "type": "thinking_delta", "thinking": t })),
                Out::Text(t) => b.delta("text", json!({ "type": "text_delta", "text": t })),
                Out::Call(c) => {
                    let mut block = tool_use(&c);
                    let input = block["input"].take().to_string();
                    block["input"] = json!({});
                    b.close()
                        && b.send(
                            "content_block_start",
                            json!({ "type": "content_block_start", "index": b.index, "content_block": block }),
                        )
                        && b.send(
                            "content_block_delta",
                            json!({ "type": "content_block_delta", "index": b.index,
                                  "delta": { "type": "input_json_delta", "partial_json": input } }),
                        )
                        && b.send("content_block_stop", json!({ "type": "content_block_stop", "index": b.index }))
                        && {
                            b.index += 1;
                            true
                        }
                }
                Out::Done(d) => {
                    let _ = b.close()
                        && b.send(
                            "message_delta",
                            json!({ "type": "message_delta",
                                  "delta": { "stop_reason": stop_reason(d.finish_reason), "stop_sequence": null },
                                  "usage": usage(&d) }),
                        )
                        && b.send("message_stop", json!({ "type": "message_stop" }));
                    return;
                }
                Out::Failed(m) => {
                    let _ = b.send("error", json!({ "type": "error", "error": { "type": "api_error", "message": m } }));
                    return;
                }
            };
            if !ok {
                return;
            }
        }
    });
    let events = UnboundedReceiverStream::new(rx)
        .map(|(name, data)| Ok::<_, Infallible>(SseEvent::default().event(name).data(data.to_string())));
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

/// The content blocks of a streamed message: a new block starts whenever the kind of text changes.
struct Blocks {
    tx: mpsc::UnboundedSender<(&'static str, Value)>,
    index: usize,
    open: Option<&'static str>,
}

impl Blocks {
    fn send(&self, name: &'static str, data: Value) -> bool {
        self.tx.send((name, data)).is_ok()
    }

    fn close(&mut self) -> bool {
        if self.open.take().is_none() {
            return true;
        }
        let ok = self.send("content_block_stop", json!({ "type": "content_block_stop", "index": self.index }));
        self.index += 1;
        ok
    }

    /// A delta for a `thinking` or `text` block, opening that block first if another kind is open.
    fn delta(&mut self, kind: &'static str, delta: Value) -> bool {
        if self.open != Some(kind) {
            let block = match kind {
                "thinking" => json!({ "type": "thinking", "thinking": "", "signature": "" }),
                _ => json!({ "type": "text", "text": "" }),
            };
            if !(self.close()
                && self.send(
                    "content_block_start",
                    json!({ "type": "content_block_start", "index": self.index, "content_block": block }),
                ))
            {
                return false;
            }
            self.open = Some(kind);
        }
        self.send("content_block_delta", json!({ "type": "content_block_delta", "index": self.index, "delta": delta }))
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
    fn anthropic_messages_become_openai_messages() {
        let msgs = to_openai(
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
        assert_eq!(msgs[0], json!({ "role": "system", "content": "Be brief." }));
        assert_eq!(msgs[1]["content"][0]["text"], "weather in Rio?");
        assert_eq!(msgs[2]["content"], json!([{ "type": "text", "text": "Checking." }]));
        assert_eq!(msgs[2]["tool_calls"][0]["function"], json!({ "name": "weather", "arguments": { "city": "Rio" } }));
        assert_eq!(msgs[3], json!({ "role": "tool", "tool_call_id": "t1", "content": "31C" }));
        assert_eq!(msgs[4]["content"], "Error: boom");
        assert_eq!(msgs[5]["content"][0]["image_url"]["url"], "data:image/png;base64,aGk=");
        // They parse as the messages every API shares.
        let _: Vec<super::super::compat::WireMsg> = serde_json::from_value(json!(msgs)).unwrap();

        let link = json!({ "role": "user", "content": [{ "type": "image", "source": { "type": "url", "url": "https://x" } }] });
        assert!(to_openai(&Value::Null, &[link]).is_err());
    }

    #[test]
    fn tools_and_tool_choice_are_translated() {
        let tools =
            to_openai_tools(&json!([{ "name": "ls", "description": "List.", "input_schema": { "type": "object" } }]));
        let _: Vec<super::super::compat::ToolDef> = serde_json::from_value(tools.clone()).unwrap();
        assert_eq!(tools[0]["function"]["parameters"], json!({ "type": "object" }));
        assert_eq!(tool_choice(&json!({ "type": "any" })), ToolChoice::Required);
        assert_eq!(tool_choice(&json!({ "type": "tool", "name": "ls" })), ToolChoice::Named("ls".into()));
        assert_eq!(tool_choice(&Value::Null), ToolChoice::Auto);
    }
}
