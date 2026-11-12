//! The Ollama-style API under `/api`, so the many apps that only speak it can use lair: point them at
//! lair's address instead of Ollama's. Generation goes through the same path as the OpenAI-style API.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::compat::{self, Opts, Out, Started, Stop, Texts, ToolDef, WireMsg, start, stops};
use super::{AppState, now};
use crate::engine::{Done, Finish, ModelEntry, Reasoning};
use crate::{download, hub};

/// The Ollama API version lair answers to, not lair's own: some clients refuse versions they see as old.
const API_VERSION: &str = "0.12.0";

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/version", get(version))
        .route("/tags", get(tags))
        .route("/ps", get(ps))
        .route("/show", post(show))
        .route("/chat", post(chat))
        .route("/generate", post(generate))
        .route("/embed", post(embed))
        .route("/embeddings", post(embeddings))
        .route("/pull", post(pull))
        .route("/delete", delete(remove))
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

/// `4.0B`, `494M`: a parameter count the way Ollama shows it; empty when unknown.
fn parameter_size(params: u64) -> String {
    match params {
        0 => String::new(),
        n if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
        n => format!("{:.0}M", n as f64 / 1e6),
    }
}

fn details(m: &ModelEntry) -> Value {
    json!({ "format": "gguf", "family": m.family, "families": [m.family],
            "parameter_size": parameter_size(m.params), "quantization_level": m.quant })
}

/// Stands in for Ollama's content digest, which clients only compare: hashing gigabytes on every
/// listing would be the alternative. It changes when the file's name or size does.
fn digest(m: &ModelEntry) -> String {
    format!("{:x}", Sha256::digest(format!("{}|{}", m.id, m.size)))
}

async fn tags(State(s): State<Arc<AppState>>) -> Json<Value> {
    let at = rfc3339(now());
    let models: Vec<Value> = s
        .lair
        .models
        .list()
        .into_iter()
        .map(|m| {
            json!({ "name": m.id, "model": m.id, "modified_at": at, "size": m.size, "digest": digest(&m),
                    "details": details(&m) })
        })
        .collect();
    Json(json!({ "models": models }))
}

/// The models in memory, as `ollama ps` shows them.
async fn ps(State(s): State<Arc<AppState>>) -> Json<Value> {
    let all = s.lair.models.local();
    let models: Vec<Value> = s
        .lair
        .models
        .ps()
        .into_iter()
        .filter_map(|l| {
            let m = all.iter().find(|m| m.id == l.id)?;
            // No keep-alive means "until memory is needed"; clients want a date, so a far one.
            let expires = now() + l.expires_in.unwrap_or(100 * 365 * 86_400);
            let on_gpu = l.size / u64::from(l.layers.max(1)) * u64::from(l.gpu_layers);
            Some(json!({ "name": m.id, "model": m.id, "size": l.size, "digest": digest(m), "details": details(m),
                         "expires_at": rfc3339(expires), "size_vram": on_gpu, "context_length": l.ctx }))
        })
        .collect();
    Json(json!({ "models": models }))
}

#[derive(Deserialize)]
struct Named {
    #[serde(alias = "name")]
    model: String,
}

async fn show(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Named = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let Some(m) = s.lair.models.list().into_iter().find(|m| m.id == req.model) else {
        return error(StatusCode::NOT_FOUND, format!("model {} not found", req.model));
    };
    let caps: Vec<&str> = if m.embedding {
        vec!["embedding"]
    } else {
        // Tool calls are understood for every chat model: lair parses them out of plain text.
        let mut caps = vec!["completion", "tools"];
        caps.extend(m.thinks.then_some("thinking"));
        caps.extend(m.vision.then_some("vision"));
        caps
    };
    let mut info = json!({ "general.architecture": m.family, "general.parameter_count": m.params });
    info[format!("{}.context_length", m.family)] = json!(m.ctx);
    Json(json!({ "modelfile": "", "parameters": "", "template": "", "capabilities": caps, "model_info": info,
                 "details": details(&m) }))
    .into_response()
}

#[derive(Default, Deserialize)]
struct Options {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    /// -1 means no limit.
    num_predict: Option<i64>,
    seed: Option<i64>,
    stop: Option<Stop>,
}

impl Options {
    /// `format` is Ollama's structured output: `"json"` for any object, or a JSON schema.
    fn with(self, think: Option<Value>, format: Option<Value>) -> Opts {
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
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            stop: stops(self.stop),
            schema: match format {
                Some(Value::String(f)) if f == "json" => Some(json!({ "type": "object" })),
                Some(schema @ Value::Object(_)) => Some(schema),
                _ => None,
            },
            ..Default::default()
        }
    }
}

/// How long a model stays in memory after a request, from Ollama's `keep_alive`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Keep {
    /// `0`: out of memory as soon as the request ends.
    Unload,
    For(Duration),
    /// Negative: until memory is needed, which is lair's default.
    Forever,
}

impl Keep {
    /// A number of seconds, or a duration like `"5m"`, `"1h"`, `"30s"`; anything else is ignored.
    fn parse(v: &Value) -> Option<Self> {
        let secs = match v {
            Value::Number(n) => n.as_f64()?,
            Value::String(s) => {
                let s = s.trim();
                let (number, unit) = s.split_at(s.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(s.len()));
                let scale = match unit {
                    "" | "s" => 1.0,
                    "ms" => 0.001,
                    "m" => 60.0,
                    "h" => 3600.0,
                    _ => return None,
                };
                number.trim().parse::<f64>().ok()? * scale
            }
            _ => return None,
        };
        Some(match secs {
            s if s < 0.0 => Self::Forever,
            0.0 => Self::Unload,
            s => Self::For(Duration::from_secs_f64(s)),
        })
    }

    fn apply(self, s: &AppState, model: &str) {
        match self {
            Self::Unload => drop(s.lair.models.unload(model)),
            Self::For(d) => s.lair.models.keep_alive(model, Some(d)),
            Self::Forever => s.lair.models.keep_alive(model, None),
        }
    }
}

/// A request with nothing to generate is Ollama's way to load a model, or with `keep_alive: 0` to
/// unload it.
async fn preload(s: &AppState, model: Option<String>, keep: Option<Keep>, chat: bool) -> Response {
    let route = match s.lair.models.resolve(model.as_deref(), false) {
        Ok(route) => route,
        Err(e) => return error(StatusCode::NOT_FOUND, e),
    };
    let reason = if keep == Some(Keep::Unload) {
        s.lair.models.unload(&route.id);
        "unload"
    } else {
        if let Err(e) = s.lair.models.load(&route.id).await {
            return error(StatusCode::BAD_REQUEST, format!("{e:#}"));
        }
        if let Some(keep) = keep {
            keep.apply(s, &route.id);
        }
        "load"
    };
    let mut v = piece(&route.id, chat, "", "", &[]);
    v["done"] = json!(true);
    v["done_reason"] = json!(reason);
    Json(v).into_response()
}

#[derive(Deserialize)]
struct ChatReq {
    model: Option<String>,
    #[serde(default)]
    messages: Vec<WireMsg>,
    #[serde(default)]
    tools: Vec<ToolDef>,
    stream: Option<bool>,
    #[serde(default)]
    options: Options,
    think: Option<Value>,
    format: Option<Value>,
    #[serde(default)]
    keep_alive: Value,
}

async fn chat(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: ChatReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let keep = Keep::parse(&req.keep_alive);
    if req.messages.is_empty() {
        return preload(&s, req.model, keep, true).await;
    }
    let started = start(&s, req.model, req.messages, &req.tools, req.options.with(req.think, req.format));
    reply(s, started, req.stream.unwrap_or(true), true, keep).await
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
    format: Option<Value>,
    /// The prompt goes to the model as written, with no chat template.
    #[serde(default)]
    raw: bool,
    #[serde(default)]
    keep_alive: Value,
}

async fn generate(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: GenReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let keep = Keep::parse(&req.keep_alive);
    if req.prompt.is_empty() && req.images.is_empty() {
        return preload(&s, req.model, keep, false).await;
    }
    let mut msgs = Vec::new();
    if let Some(system) = req.system.filter(|_| !req.raw) {
        msgs.push(json!({ "role": "system", "content": system }));
    }
    msgs.push(json!({ "role": "user", "content": req.prompt, "images": req.images }));
    let msgs = serde_json::from_value(Value::Array(msgs)).expect("messages built above are valid");
    let opts = Opts { raw: req.raw, ..req.options.with(req.think, req.format) };
    let started = start(&s, req.model, msgs, &[], opts);
    reply(s, started, req.stream.unwrap_or(true), false, keep).await
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

/// The last object of a reply: why it ended, token counts and durations (in nanoseconds, as Ollama has them).
fn finish(mut v: Value, d: &Done) -> Value {
    v["done"] = json!(true);
    v["done_reason"] = json!(if d.finish_reason == Finish::Length { "length" } else { "stop" });
    v["prompt_eval_count"] = json!(d.prompt_tokens);
    v["eval_count"] = json!(d.completion_tokens);
    v["total_duration"] = json!((d.load_us + d.prompt_us + d.gen_us) * 1000);
    v["load_duration"] = json!(d.load_us * 1000);
    v["prompt_eval_duration"] = json!(d.prompt_us * 1000);
    v["eval_duration"] = json!(d.gen_us * 1000);
    v
}

fn call(c: &crate::agent::Call) -> Value {
    json!({ "function": { "name": c.name, "arguments": c.arguments } })
}

async fn reply(
    s: Arc<AppState>,
    started: Result<Started, (StatusCode, String)>,
    stream: bool,
    chat: bool,
    keep: Option<Keep>,
) -> Response {
    let Started { model, mut out, .. } = match started {
        Ok(s) => s,
        Err((status, message)) => return error(status, message),
    };
    // The generation is over: now the keep-alive the request asked for takes effect.
    let settle = move |model: &str| {
        if let Some(keep) = keep {
            keep.apply(&s, model);
        }
    };
    if !stream {
        let (mut content, mut thinking, mut calls) = (String::new(), String::new(), Vec::new());
        loop {
            match out.recv().await {
                Some(Out::Thinking(t)) => thinking.push_str(&t),
                Some(Out::Text(t)) => content.push_str(&t),
                Some(Out::Call(c)) => calls.push(call(&c)),
                Some(Out::Done(d)) => {
                    settle(&model);
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
            Out::Done(d) => {
                settle(&model);
                finish(piece(&model, chat, "", "", &[]), &d)
            }
            Out::Failed(m) => json!({ "error": m }),
        };
        Ok::<_, Infallible>(format!("{v}\n"))
    });
    ndjson(lines)
}

fn ndjson(lines: impl tokio_stream::Stream<Item = Result<String, Infallible>> + Send + 'static) -> Response {
    ([(header::CONTENT_TYPE, "application/x-ndjson")], Body::from_stream(lines)).into_response()
}

#[derive(Deserialize)]
struct EmbedReq {
    model: Option<String>,
    /// `/api/embed` takes `input` (one text or several), the older `/api/embeddings` takes `prompt`.
    #[serde(alias = "prompt")]
    input: Texts,
}

async fn embed_request(s: &AppState, body: &[u8]) -> Result<(String, Vec<Vec<f32>>, usize), (StatusCode, String)> {
    let req: EmbedReq = parse(body).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    compat::embed(s, req.model, req.input.into_vec()).await
}

async fn embed(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    match embed_request(&s, &body).await {
        Ok((model, embeddings, read)) => {
            Json(json!({ "model": model, "embeddings": embeddings, "prompt_eval_count": read })).into_response()
        }
        Err((status, message)) => error(status, message),
    }
}

/// The older endpoint: one text in, one vector out.
async fn embeddings(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    match embed_request(&s, &body).await {
        Ok((_, mut vectors, _)) => Json(json!({ "embedding": vectors.swap_remove(0) })).into_response(),
        Err((status, message)) => error(status, message),
    }
}

async fn remove(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Named = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    match super::native::delete_model(&s.lair, &req.model).await {
        Ok(_) => Json(json!({})).into_response(),
        Err(e) => {
            let (status, message) = e.parts();
            error(status, message)
        }
    }
}

/// What `ollama pull <name>` means here: a Hugging Face GGUF repo, `owner/repo` or `owner/repo:QUANT`
/// (an `hf.co/` prefix, as Ollama writes them, is fine). A bare name like `llama3` belongs to
/// Ollama's own library, which lair doesn't mirror, and guessing a repo for it would let anyone who
/// names a repo that way choose what gets downloaded. URLs are refused: the server downloads from
/// Hugging Face only.
fn pull_spec(name: &str) -> Result<(String, Option<String>), String> {
    let name = name.trim();
    let name = ["https://huggingface.co/", "huggingface.co/", "hf.co/"]
        .iter()
        .find_map(|p| name.strip_prefix(p))
        .unwrap_or(name);
    let (repo, quant) = match name.split_once(':') {
        Some((repo, quant)) => (repo, Some(quant.to_owned()).filter(|q| !q.is_empty() && q != "latest")),
        None => (name, None),
    };
    match hub::check_repo(repo) {
        Ok(()) => Ok((repo.to_owned(), quant)),
        Err(_) => Err(format!(
            "\"{name}\" is not a Hugging Face repo; lair pulls GGUF repos by owner/name, e.g. Qwen/Qwen3-4B-GGUF"
        )),
    }
}

#[derive(Deserialize)]
struct PullReq {
    #[serde(alias = "name")]
    model: String,
    stream: Option<bool>,
}

/// Downloads a model as the UI would, reporting progress the way `ollama pull` shows it.
async fn pull(State(s): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: PullReq = match parse(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let (repo, quant) = match pull_spec(&req.model) {
        Ok(spec) => spec,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let file = match hub::choose(&s.lair, &repo, quant.as_deref()).await {
        Ok(file) => file,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    // Already downloading is fine: this request then follows the download that is running.
    let running = s.lair.downloads.get(&repo, &file).is_some_and(|p| !p.finished && p.error.is_none());
    if !running && let Err(e) = download::start(s.lair.clone(), repo.clone(), file.clone()).await {
        return error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let (tx, rx) = mpsc::unbounded_channel();
    let lair = s.lair.clone();
    tokio::spawn(async move {
        let _ = tx.send(json!({ "status": "pulling manifest" }));
        loop {
            let Some(p) = lair.downloads.get(&repo, &file) else {
                let _ = tx.send(json!({ "error": "the download was cancelled" }));
                return;
            };
            let line = if let Some(e) = &p.error {
                json!({ "error": e })
            } else if p.finished {
                json!({ "status": "success" })
            } else if p.verifying {
                json!({ "status": "verifying sha256 digest" })
            } else {
                json!({ "status": format!("pulling {file}"), "digest": file, "total": p.total, "completed": p.done })
            };
            // The client leaving does not stop the download; it shows in the UI and can be cancelled there.
            if tx.send(line).is_err() || p.finished || p.error.is_some() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    });
    let mut lines = UnboundedReceiverStream::new(rx);
    if req.stream.unwrap_or(true) {
        return ndjson(lines.map(|v| Ok::<_, Infallible>(format!("{v}\n"))));
    }
    let mut last = json!({});
    while let Some(v) = lines.next().await {
        last = v;
    }
    match last["error"].as_str() {
        Some(e) => error(StatusCode::BAD_GATEWAY, e),
        None => Json(last).into_response(),
    }
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
        let r = |t: Value| Options::default().with(Some(t), None).reasoning;
        assert_eq!(r(json!(false)), Some(Reasoning::Off));
        assert_eq!(r(json!(true)), Some(Reasoning::Deep));
        assert_eq!(r(json!("low")), Some(Reasoning::Light));
        assert_eq!(Options { num_predict: Some(-1), ..Default::default() }.with(None, None).max_tokens, None);
        let format = |f: Value| Options::default().with(None, Some(f)).schema;
        assert_eq!(format(json!("json")), Some(json!({ "type": "object" })));
        assert_eq!(format(json!({ "type": "array" })), Some(json!({ "type": "array" })));
        assert_eq!(format(json!("yaml")), None);
    }

    #[test]
    fn keep_alive_reads_seconds_and_durations() {
        assert_eq!(Keep::parse(&json!(0)), Some(Keep::Unload));
        assert_eq!(Keep::parse(&json!("0")), Some(Keep::Unload));
        assert_eq!(Keep::parse(&json!(-1)), Some(Keep::Forever));
        assert_eq!(Keep::parse(&json!(90)), Some(Keep::For(Duration::from_secs(90))));
        assert_eq!(Keep::parse(&json!("5m")), Some(Keep::For(Duration::from_secs(300))));
        assert_eq!(Keep::parse(&json!("1h")), Some(Keep::For(Duration::from_secs(3600))));
        assert_eq!(Keep::parse(&json!("soon")), None);
        assert_eq!(Keep::parse(&Value::Null), None);
    }

    #[test]
    fn pull_takes_hugging_face_repos_only() {
        assert_eq!(pull_spec("Qwen/Qwen3-4B-GGUF"), Ok(("Qwen/Qwen3-4B-GGUF".into(), None)));
        assert_eq!(
            pull_spec("hf.co/Qwen/Qwen3-4B-GGUF:Q5_K_M"),
            Ok(("Qwen/Qwen3-4B-GGUF".into(), Some("Q5_K_M".into())))
        );
        assert_eq!(pull_spec("a/b:latest"), Ok(("a/b".into(), None)));
        for bad in ["llama3", "llama3:8b", "https://evil.example/m.gguf", "http://10.0.0.1/a/b", "../x/y", ""] {
            assert!(pull_spec(bad).is_err(), "{bad}");
        }
        assert_eq!(
            (parameter_size(4_022_000_000), parameter_size(494_000_000), parameter_size(0)),
            ("4.0B".into(), "494M".into(), String::new())
        );
    }
}
