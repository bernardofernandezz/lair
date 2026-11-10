//! lair's own API, under `/lair/v1`: conversations and their turns, skills, attachments, and what
//! other nodes ask of this one.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use super::{bad, missing, settings, sse, ApiResult, AppState};
use crate::agent::Decision;
use crate::attach;
use crate::engine::{Msg, Params};
use crate::peers::NodeInfo;
use crate::skills::{install, slug, Manifest, Skill};
use crate::store::{Conversation, Hit, Settings};
use crate::turn;

type S = State<Arc<AppState>>;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/conversations", get(list).post(create))
        .route("/conversations/search", get(search))
        .route("/conversations/{id}", get(read).patch(update).delete(remove))
        .route("/conversations/{id}/messages", post(send))
        .route("/approvals/{id}", post(approve))
        .route("/node", get(node))
        .route("/generate", post(generate))
        .route("/skills", get(skills).post(create_skill))
        .route("/skills/index", get(skills_index))
        .route("/skills/install", post(install_skill))
        .route("/skills/upload", post(upload_skill))
        .route("/skills/{id}", put(save_skill).patch(toggle_skill).delete(remove_skill))
        .route("/files", get(file_types).post(upload_file))
        .route("/files/{id}", get(file))
        .layer(DefaultBodyLimit::max(attach::MAX_UPLOAD + 64 * 1024))
        .merge(settings::routes())
}

async fn list(State(s): S) -> ApiResult<Json<Vec<Conversation>>> {
    Ok(Json(s.lair.store.list()?))
}

async fn create(State(s): S, Json(settings): Json<Settings>) -> ApiResult<Json<Conversation>> {
    Ok(Json(s.lair.store.create(settings)?))
}

async fn read(State(s): S, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let store = &s.lair.store;
    let c = store.get(&id)?.ok_or_else(|| missing("conversation not found"))?;
    Ok(Json(json!({ "conversation": c, "messages": store.messages(&id)? })))
}

async fn update(State(s): S, Path(id): Path<String>, Json(settings): Json<Settings>) -> ApiResult<Json<Conversation>> {
    Ok(Json(s.lair.store.update(&id, settings)?.ok_or_else(|| missing("conversation not found"))?))
}

async fn remove(State(s): S, Path(id): Path<String>) -> ApiResult<StatusCode> {
    match s.lair.store.delete(&id)? {
        true => Ok(StatusCode::NO_CONTENT),
        false => Err(missing("conversation not found")),
    }
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
}

async fn search(State(s): S, Query(q): Query<SearchQuery>) -> ApiResult<Json<Vec<Hit>>> {
    Ok(Json(s.lair.store.search(&q.q)?))
}

/// The user's answer to a tool call waiting for approval.
async fn approve(State(s): S, Path(id): Path<String>, Json(d): Json<Decision>) -> ApiResult<StatusCode> {
    match s.lair.approvals.answer(&id, d) {
        true => Ok(StatusCode::NO_CONTENT),
        false => Err(missing("no tool call is waiting with that id")),
    }
}

/// Streams one turn as server-sent events, one per [`turn::Out`].
async fn send(State(s): S, Path(id): Path<String>, Json(input): Json<turn::Input>) -> Response {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(turn::run(s.lair.clone(), id, input, tx));
    let events = UnboundedReceiverStream::new(rx).map(|out| Ok::<_, Infallible>(sse(&out)));
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}

async fn skills(State(s): S) -> Json<Vec<Skill>> {
    Json(s.lair.skills.read().unwrap().list().to_vec())
}

#[derive(Deserialize)]
struct SkillDraft {
    #[serde(flatten)]
    manifest: Manifest,
    prompt: String,
}

/// A new skill, named after its name.
async fn create_skill(State(s): S, Json(d): Json<SkillDraft>) -> ApiResult<Json<Skill>> {
    save_skill(State(s), Path(slug(&d.manifest.name)), Json(d)).await
}

async fn save_skill(State(s): S, Path(id): Path<String>, Json(d): Json<SkillDraft>) -> ApiResult<Json<Skill>> {
    let manifest = Manifest { source: None, ..d.manifest };
    Ok(Json(s.lair.skills.write().unwrap().save(&id, &manifest, &d.prompt).map_err(bad)?))
}

#[derive(Deserialize)]
struct Toggle {
    enabled: bool,
}

async fn toggle_skill(State(s): S, Path(id): Path<String>, Json(t): Json<Toggle>) -> ApiResult<StatusCode> {
    s.lair.skills.write().unwrap().set_enabled(&id, t.enabled).map_err(bad)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn remove_skill(State(s): S, Path(id): Path<String>) -> ApiResult<StatusCode> {
    s.lair.skills.write().unwrap().remove(&id).map_err(bad)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct InstallRequest {
    url: String,
}

/// Downloads a skill from a URL (zip, GitHub folder or folder) into lair's skills folder.
async fn install_skill(State(s): S, Json(r): Json<InstallRequest>) -> ApiResult<Json<Skill>> {
    let f = install::fetch(&s.lair.http, &r.url).await.map_err(bad)?;
    Ok(Json(s.lair.skills.write().unwrap().save(&f.id, &f.manifest, &f.prompt).map_err(bad)?))
}

/// Installs a skill from an uploaded .zip, read the same way as one from a URL.
async fn upload_skill(State(s): S, form: Multipart) -> ApiResult<Json<Skill>> {
    let (name, data) = read_file(form).await?;
    let f = install::from_zip_bytes(&data, &name).map_err(bad)?;
    Ok(Json(s.lair.skills.write().unwrap().save(&f.id, &f.manifest, &f.prompt).map_err(bad)?))
}

async fn file_types() -> Json<&'static attach::Types> {
    Json(&attach::TYPES)
}

#[derive(Serialize)]
struct Uploaded {
    /// What a turn sends back to attach this file.
    id: String,
    name: String,
}

async fn upload_file(State(s): S, form: Multipart) -> ApiResult<Json<Uploaded>> {
    let (name, data) = read_file(form).await?;
    let id = attach::save(&s.lair.workspace, &name, &data).map_err(bad)?;
    Ok(Json(Uploaded { id, name }))
}

/// An uploaded attachment, for the chat's thumbnails.
async fn file(State(s): S, Path(id): Path<String>) -> ApiResult<Response> {
    let path = attach::path(&s.lair.workspace, &id).map_err(missing)?;
    let data = tokio::fs::read(&path).await.map_err(missing)?;
    let mime = mime_guess::from_path(&path).first_or_octet_stream();
    // Uploads are user content: never let one run as a page on this origin.
    let headers = [
        (header::CONTENT_TYPE, mime.essence_str().to_owned()),
        (header::CONTENT_SECURITY_POLICY, "default-src 'none'; sandbox".to_owned()),
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_owned()),
    ];
    Ok((headers, data).into_response())
}

/// The form's `file` field, whole, with the name the browser gave it.
async fn read_file(mut form: Multipart) -> ApiResult<(String, Vec<u8>)> {
    while let Some(field) = form.next_field().await.map_err(bad)? {
        if field.name() == Some("file") {
            let name = field.file_name().unwrap_or("arquivo").to_owned();
            return Ok((name, field.bytes().await.map_err(bad)?.to_vec()));
        }
    }
    Err(bad(anyhow::anyhow!("o formulário não tem o campo \"file\"")))
}

/// The configured skills index, marking which entries are already installed.
async fn skills_index(State(s): S) -> ApiResult<Json<Value>> {
    let Some(url) = s.lair.config().skills_index else {
        return Ok(Json(json!({ "configured": false, "skills": [] })));
    };
    let entries = install::index(&s.lair.http, &url)
        .await
        .map_err(|e| super::ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}")))?;
    let catalog = s.lair.skills.read().unwrap();
    let installed = |url: &str| catalog.list().iter().any(|k| k.source.as_deref() == Some(url));
    let skills: Vec<_> = entries.iter().map(|e| json!({ "entry": e, "installed": installed(&e.url) })).collect();
    Ok(Json(json!({ "configured": true, "skills": skills })))
}

/// Who this node is and which models it runs, for other nodes on the tailnet.
async fn node(State(s): S) -> Json<NodeInfo> {
    Json(NodeInfo { name: s.lair.node_name(), version: env!("CARGO_PKG_VERSION").into(), models: s.lair.models.local() })
}

#[derive(Deserialize)]
struct GenerateRequest {
    model: String,
    messages: Vec<Msg>,
    params: Params,
}

/// Plain generation on one of this machine's models, for a peer. The asking node keeps the
/// conversation, skills and tools; this node only generates.
async fn generate(State(s): S, Json(r): Json<GenerateRequest>) -> Response {
    let rx = s.lair.models.generate_local(&r.model, r.messages, r.params);
    let events = UnboundedReceiverStream::new(rx).map(|e| Ok::<_, Infallible>(sse(&e)));
    Sse::new(events).keep_alive(KeepAlive::default()).into_response()
}
