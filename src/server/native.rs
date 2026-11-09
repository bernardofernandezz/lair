//! lair's own API, under `/lair/v1`: conversations and their turns.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use super::{error, AppState};
use crate::agent::Decision;
use crate::skills::{install, Manifest};
use crate::store::Settings;
use crate::turn;

type S = State<Arc<AppState>>;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/conversations", get(list).post(create))
        .route("/conversations/search", get(search))
        .route("/conversations/{id}", get(read).patch(update).delete(remove))
        .route("/conversations/{id}/messages", post(send))
        .route("/approvals/{id}", post(approve))
        .route("/skills", get(skills))
        .route("/skills/index", get(skills_index))
        .route("/skills/install", post(install_skill))
        .route("/skills/{id}", put(save_skill).patch(toggle_skill).delete(remove_skill))
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

/// The user's answer to a tool call waiting for approval.
async fn approve(State(s): S, Path(id): Path<String>, Json(d): Json<Decision>) -> Response {
    if s.lair.approvals.answer(&id, d) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        error(StatusCode::NOT_FOUND, "no tool call is waiting with that id")
    }
}

/// Streams one turn as server-sent events named after `turn::Out` (`thinking`, `answer`, `tool_call`, …).
async fn send(State(s): S, Path(id): Path<String>, Json(input): Json<turn::Input>) -> Response {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(turn::run(s.lair.clone(), id, input, tx));
    let events = UnboundedReceiverStream::new(rx)
        .map(|out| Ok::<_, Infallible>(Event::default().event(out.name()).data(out.data().to_string())));
    Sse::new(events).into_response()
}

async fn skills(State(s): S) -> Response {
    Json(s.lair.skills.read().unwrap().list().to_vec()).into_response()
}

#[derive(Deserialize)]
struct SkillDraft {
    #[serde(flatten)]
    manifest: Manifest,
    prompt: String,
}

async fn save_skill(State(s): S, Path(id): Path<String>, Json(d): Json<SkillDraft>) -> Response {
    let manifest = Manifest { source: None, ..d.manifest };
    match s.lair.skills.write().unwrap().save(&id, &manifest, &d.prompt) {
        Ok(skill) => Json(skill).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

#[derive(Deserialize)]
struct Toggle {
    enabled: bool,
}

async fn toggle_skill(State(s): S, Path(id): Path<String>, Json(t): Json<Toggle>) -> Response {
    match s.lair.skills.write().unwrap().set_enabled(&id, t.enabled) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

async fn remove_skill(State(s): S, Path(id): Path<String>) -> Response {
    match s.lair.skills.write().unwrap().remove(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

#[derive(Deserialize)]
struct InstallRequest {
    url: String,
}

/// Downloads a skill from a URL (zip, GitHub folder or folder) into lair's skills folder.
async fn install_skill(State(s): S, Json(r): Json<InstallRequest>) -> Response {
    let saved = match install::fetch(&s.lair.http, &r.url).await {
        Ok(f) => s.lair.skills.write().unwrap().save(&f.id, &f.manifest, &f.prompt),
        Err(e) => Err(e),
    };
    match saved {
        Ok(skill) => Json(skill).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

/// The configured skills index, marking which entries are already installed.
async fn skills_index(State(s): S) -> Response {
    let Some(url) = s.lair.skills_index.as_deref() else {
        return Json(json!({ "configured": false, "skills": [] })).into_response();
    };
    match install::index(&s.lair.http, url).await {
        Ok(entries) => {
            let catalog = s.lair.skills.read().unwrap();
            let installed = |url: &str| catalog.list().iter().any(|k| k.source.as_deref() == Some(url));
            let skills: Vec<_> = entries.iter().map(|e| json!({ "entry": e, "installed": installed(&e.url) })).collect();
            Json(json!({ "configured": true, "skills": skills })).into_response()
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    }
}
