//! `/lair/v1/settings`: what the UI saved, what the command line fixed, and the facts the settings
//! panel shows about this node.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{Value, json};

use super::{ApiResult, AppState, bad};

type S = State<Arc<AppState>>;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/settings", get(read).patch(write))
}

async fn read(State(s): S) -> ApiResult<Json<Value>> {
    let lair = &s.lair;
    let peers = &lair.peers;
    let config = lair.config();
    Ok(Json(json!({
        "saved": crate::lair::saved(&lair.store),
        "flags": lair.flags,
        "tailscale": { "socket": peers.has_socket(), "status": peers.status().await },
        "discovered": peers.found(),
        "system": {
            "version": env!("CARGO_PKG_VERSION"),
            "token": s.token.is_some(),
            "addr": s.addr,
            "name": lair.node_name(),
            "autoName": lair.auto_name,
            "reasoning": lair.params().reasoning,
            "trustTailnet": config.trust_tailnet(),
            "workspace": lair.workspace.display().to_string(),
            "models": lair.models.local(),
        },
    })))
}

/// Changes some of what the UI saved; see [`crate::lair::Lair::save`].
async fn write(State(s): S, Json(change): Json<Value>) -> ApiResult<StatusCode> {
    s.lair.save(change).map_err(bad)?;
    Ok(StatusCode::NO_CONTENT)
}
