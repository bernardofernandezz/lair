use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Request, State};
use axum::http::{header, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rust_embed::RustEmbed;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::assets::Assets;
use crate::engine::{Engine, ModelInfo, Msg, Params, Part, Reasoning};
use crate::skills::Catalog;

#[derive(RustEmbed)]
#[folder = "web/dist"]
#[allow_missing = true]
struct AssetsUi;

struct AppState {
    // One model on one GPU: run requests one at a time.
    engine: Arc<Mutex<Engine>>,
    catalog: Vec<ModelInfo>,
    /// Model in memory after the last request; readable while the engine is busy generating.
    loaded: std::sync::Mutex<Option<String>>,
    defaults: Params,
    token: Option<String>,
    skills: Catalog,
    assets: Assets,
}

#[derive(Deserialize)]
struct ChatRequest {
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

pub async fn run(
    engine: Engine,
    defaults: Params,
    token: Option<String>,
    addr: String,
    skills: Catalog,
    assets: Assets,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        catalog: engine.models().to_vec(),
        loaded: std::sync::Mutex::new(None),
        engine: Arc::new(Mutex::new(engine)),
        defaults,
        token,
        skills,
        assets,
    });
    let api = Router::new()
        .route("/models", get(models))
        .route("/assets/{*id}", get(asset_file))
        .route("/session", post(create_session))
        .route("/chat/completions", post(chat_completions))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_token))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "no such API route") });
    let app = Router::new().nest("/v1", api).fallback(web_ui).with_state(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn require_token(State(s): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    if let Some(want) = &s.token {
        let bearer = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        // GLTF textures are fetched by <img>, which cannot send Authorization, so the UI also holds
        // the token in an HttpOnly cookie; its Path=/v1/assets keeps browsers from sending it elsewhere.
        let cookie = cookie_value(&req, "lair_asset").unwrap_or_default();
        let ok = constant_time_eq(bearer.as_bytes(), want.as_bytes()) || constant_time_eq(cookie.as_bytes(), want.as_bytes());
        if !ok {
            return error(StatusCode::UNAUTHORIZED, "invalid or missing token");
        }
    }
    next.run(req).await
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn cookie_value(req: &Request, name: &str) -> Option<String> {
    let raw = req.headers().get(header::COOKIE)?.to_str().ok()?;
    for part in raw.split(';') {
        if let Some((key, value)) = part.trim().split_once('=') {
            if key == name {
                return percent_decode(value);
            }
        }
    }
    None
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{b:02X}"));
        }
    }
    out
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Serves the embedded single-page app; unknown paths get index.html so client routes work.
async fn web_ui(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    match AssetsUi::get(path).or_else(|| AssetsUi::get("index.html")) {
        Some(f) => ([(header::CONTENT_TYPE, f.metadata.mimetype().to_owned())], f.data).into_response(),
        None => error(StatusCode::NOT_FOUND, "web UI not built: run `bun install && bun run build` in web/"),
    }
}

async fn models(State(s): State<Arc<AppState>>) -> Json<Value> {
    let loaded = s.loaded.lock().unwrap().clone();
    let data: Vec<Value> = s
        .catalog
        .iter()
        .map(|m| {
            json!({ "id": m.id, "object": "model", "owned_by": "local",
                    "thinks": m.thinks, "loaded": loaded.as_deref() == Some(m.id.as_str()) })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn asset_file(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Some(path) = s.assets.resolve(&id) else {
        return error(StatusCode::NOT_FOUND, "asset not found");
    };
    match tokio::fs::read(&path).await {
        Ok(data) => {
            let mime = mime_guess::from_path(&path).first_or_octet_stream();
            (
                [
                    (header::CONTENT_TYPE, mime.essence_str().to_owned()),
                    (header::CACHE_CONTROL, "private, max-age=3600".to_owned()),
                ],
                data,
            )
                .into_response()
        }
        Err(e) => error(StatusCode::NOT_FOUND, format!("cannot read asset: {e}")),
    }
}

/// Exchanges the Bearer token for the HttpOnly cookie used by GLTF texture loads.
async fn create_session(State(s): State<Arc<AppState>>) -> Response {
    match &s.token {
        None => StatusCode::NO_CONTENT.into_response(),
        Some(t) => {
            let cookie = format!(
                "lair_asset={}; HttpOnly; SameSite=Strict; Path=/v1/assets; Max-Age=604800",
                percent_encode(t)
            );
            ([(header::SET_COOKIE, cookie)], StatusCode::NO_CONTENT).into_response()
        }
    }
}

fn error(status: StatusCode, msg: impl ToString) -> Response {
    (status, Json(json!({ "error": { "message": msg.to_string() } }))).into_response()
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

async fn chat_completions(State(s): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    if req.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let model = match req.model.clone() {
        Some(id) if s.catalog.iter().any(|m| m.id == id) => id,
        Some(id) => return error(StatusCode::NOT_FOUND, format!("model {id} not found")),
        None => s.loaded.lock().unwrap().clone().unwrap_or_else(|| s.catalog[0].id.clone()),
    };
    let d = s.defaults;
    let params = Params {
        temperature: req.temperature.unwrap_or(d.temperature),
        top_p: req.top_p.unwrap_or(d.top_p),
        max_tokens: req.max_tokens.unwrap_or(d.max_tokens),
        seed: req.seed.or(d.seed),
        reasoning: req.reasoning_effort.unwrap_or(d.reasoning),
    };
    let (skill, messages) = s.skills.apply(&req.messages, s.assets.ids());
    let skill = skill.map(|k| json!({ "id": k.id, "name": k.name }));
    let created = now();
    let id = format!("chatcmpl-{created}");
    let mut engine = s.engine.clone().lock_owned().await;
    let (state, model_id) = (s.clone(), model.clone());
    let mut generate = move |on_part: &mut dyn for<'a> FnMut(Part<'a>) -> bool| {
        let res = engine.generate(&model_id, &messages, params, on_part);
        *state.loaded.lock().unwrap() = engine.loaded().map(str::to_owned);
        res
    };

    if !req.stream {
        let res = tokio::task::spawn_blocking(move || {
            let (mut reasoning, mut content) = (String::new(), String::new());
            let res = generate(&mut |part| {
                match part {
                    Part::Thinking(t) => reasoning.push_str(t),
                    Part::Answer(t) => content.push_str(t),
                }
                true
            });
            res.map(|done| (reasoning, content, done))
        })
        .await;
        return match res {
            Ok(Ok((reasoning, content, done))) => {
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
                Json(body).into_response()
            }
            Ok(Err(e)) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
            Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
    }

    let (tx, rx) = mpsc::unbounded_channel::<Value>();
    let chunk_model = model.clone();
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": chunk_model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
    };
    tokio::task::spawn_blocking(move || {
        let _ = tx.send(chunk(json!({ "role": "assistant" }), None));
        // send() fails once the client disconnects, which stops generation.
        let res = generate(&mut |part| {
            let delta = match part {
                Part::Thinking(t) => json!({ "reasoning_content": t }),
                Part::Answer(t) => json!({ "content": t }),
            };
            tx.send(chunk(delta, None)).is_ok()
        });
        let last = match res {
            Ok(done) => chunk(json!({}), Some(done.finish_reason)),
            Err(e) => json!({ "error": { "message": format!("{e:#}") } }),
        };
        let _ = tx.send(last);
    });

    let skill_event = skill.map(|k| Event::default().event("lair.skill").data(k.to_string()));
    let events = tokio_stream::iter(skill_event)
        .chain(UnboundedReceiverStream::new(rx).map(|v| Event::default().data(v.to_string())))
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);
    Sse::new(events).into_response()
}
