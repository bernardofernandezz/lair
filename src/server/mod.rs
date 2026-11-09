use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{header, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rust_embed::RustEmbed;
use serde_json::json;

use crate::lair::Lair;

#[derive(RustEmbed)]
#[folder = "web/dist"]
#[allow_missing = true]
struct AssetsUi;

mod compat;
mod native;

/// How often other lair nodes on the tailnet are looked for.
const PEER_REFRESH: std::time::Duration = std::time::Duration::from_secs(60);

pub struct AppState {
    pub lair: Arc<Lair>,
    token: Option<String>,
}

pub async fn run(lair: Lair, token: Option<String>, addr: String) -> anyhow::Result<()> {
    let state = Arc::new(AppState { lair: Arc::new(lair), token });
    let peers = state.lair.models.peers.clone();
    tokio::spawn(async move {
        loop {
            peers.refresh().await;
            tokio::time::sleep(PEER_REFRESH).await;
        }
    });
    let auth = middleware::from_fn_with_state(state.clone(), require_token);
    let v1 = Router::new()
        .route("/models", get(compat::models))
        .route("/chat/completions", post(compat::chat_completions))
        .route("/assets/{*id}", get(asset_file))
        .route("/session", post(create_session))
        .route_layer(auth.clone())
        .fallback(no_route);
    let native = native::routes().route_layer(auth).fallback(no_route);
    let app = Router::new().nest("/v1", v1).nest("/lair/v1", native).fallback(web_ui).with_state(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("listening on http://{addr}");
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}

/// With a token set, a request needs it, unless it comes from a machine in this node's tailnet.
async fn require_token(State(s): State<Arc<AppState>>, ConnectInfo(from): ConnectInfo<SocketAddr>, req: Request, next: Next) -> Response {
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
        let ok = constant_time_eq(bearer.as_bytes(), want.as_bytes())
            || constant_time_eq(cookie.as_bytes(), want.as_bytes())
            || s.lair.models.peers.trusts(from.ip()).await;
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

async fn asset_file(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Some(path) = s.lair.assets.resolve(&id) else {
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

async fn no_route() -> Response {
    error(StatusCode::NOT_FOUND, "no such API route")
}

fn error(status: StatusCode, msg: impl ToString) -> Response {
    (status, Json(json!({ "error": { "message": msg.to_string() } }))).into_response()
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}
