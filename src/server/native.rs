//! lair's own API, under `/lair/v1`: conversations and their turns.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use super::{error, AppState};
use crate::store::Settings;
use crate::turn;

type S = State<Arc<AppState>>;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/conversations", get(list).post(create))
        .route("/conversations/search", get(search))
        .route("/conversations/{id}", get(read).patch(update).delete(remove))
        .route("/conversations/{id}/messages", post(send))
}

fn reply<T: serde::Serialize>(res: anyhow::Result<T>) -> Response {
    match res {
        Ok(v) => Json(v).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

fn found<T: serde::Serialize>(res: anyhow::Result<Option<T>>) -> Response {
    match res {
        Ok(None) => error(StatusCode::NOT_FOUND, "conversation not found"),
        other => reply(other),
    }
}

async fn list(State(s): S) -> Response {
    reply(s.lair.store.list())
}

async fn create(State(s): S, Json(settings): Json<Settings>) -> Response {
    reply(s.lair.store.create(settings))
}

async fn read(State(s): S, Path(id): Path<String>) -> Response {
    let store = &s.lair.store;
    found(store.get(&id).and_then(|c| {
        c.map(|c| Ok(json!({ "conversation": c, "messages": store.messages(&id)? }))).transpose()
    }))
}

async fn update(State(s): S, Path(id): Path<String>, Json(settings): Json<Settings>) -> Response {
    found(s.lair.store.update(&id, settings))
}

async fn remove(State(s): S, Path(id): Path<String>) -> Response {
    match s.lair.store.delete(&id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        other => found(other.map(|_| None::<()>)),
    }
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
}

async fn search(State(s): S, Query(q): Query<SearchQuery>) -> Response {
    reply(s.lair.store.search(&q.q))
}

/// Streams one turn as server-sent events named after `turn::Out` (`skill`, `thinking`, `answer`, `done`, `error`).
async fn send(State(s): S, Path(id): Path<String>, Json(input): Json<turn::Input>) -> Response {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(turn::run(s.lair.clone(), id, input, tx));
    let events = UnboundedReceiverStream::new(rx)
        .map(|out| Ok::<_, Infallible>(Event::default().event(out.name()).data(out.data().to_string())));
    Sse::new(events).into_response()
}
