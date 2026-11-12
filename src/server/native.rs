//! lair's own API, under `/lair/v1`: conversations and their turns, skills, attachments, and what
//! other nodes ask of this one.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::{ApiResult, AppState, bad, missing, settings, sse};
use crate::agent::Decision;
use crate::attach;
use crate::engine::{Msg, Params};
use crate::peers::NodeInfo;
use crate::skills::{Manifest, Skill, install, slug};
use crate::store::{Conversation, Hit, Settings};
use crate::turn;
use crate::{download, hub, memory as mem};

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
        .route("/hub/hardware", get(hub_hardware))
        .route("/hub/suggested", get(hub_suggested))
        .route("/hub/search", get(hub_search))
        .route("/hub/files", get(hub_files))
        .route("/hub/downloads", get(hub_downloads).post(hub_download).delete(hub_cancel))
        .route("/models", get(models))
        .route("/models/{id}", axum::routing::delete(remove_model))
        .route("/models/{id}/load", post(load_model))
        .route("/models/{id}/unload", post(unload_model))
        .route("/memory", get(memory).delete(forget))
        .route("/memory/search", get(memory_search))
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
    s.lair.memory.wake();
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
            let name = field.file_name().unwrap_or("file").to_owned();
            return Ok((name, field.bytes().await.map_err(bad)?.to_vec()));
        }
    }
    Err(bad(anyhow::anyhow!("the form has no \"file\" field")))
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
    Json(NodeInfo {
        name: s.lair.node_name(),
        version: env!("CARGO_PKG_VERSION").into(),
        models: s.lair.models.local(),
        active: s.lair.models.active(),
    })
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

async fn hub_hardware() -> Json<crate::hw::Hardware> {
    Json(crate::hw::hardware())
}

/// The models to offer on this machine, best first: what is trending on Hugging Face and what lair
/// vouches for, each with the file to take and why; `best` marks the one to get (see [`hub::rank`]).
async fn hub_suggested(State(s): S) -> ApiResult<Json<Vec<hub::Pick>>> {
    let picks = hub::suggested(&s.lair).await;
    if picks.is_empty() {
        return Err(bad_gateway(anyhow::anyhow!("Hugging Face can't be reached right now")));
    }
    Ok(Json(picks))
}

#[derive(Deserialize)]
struct HubQuery {
    q: Option<String>,
    repo: Option<String>,
}

async fn hub_search(State(s): S, Query(q): Query<HubQuery>) -> ApiResult<Json<Vec<hub::Repo>>> {
    let q = q.q.unwrap_or_default();
    if q.trim().is_empty() || q.len() > 200 {
        return Err(bad(anyhow::anyhow!("the search is empty or too long")));
    }
    Ok(Json(hub::search(&s.lair.http, q.trim()).await.map_err(bad_gateway)?))
}

async fn hub_files(State(s): S, Query(q): Query<HubQuery>) -> ApiResult<Json<Value>> {
    let repo = q.repo.unwrap_or_default();
    hub::check_repo(&repo).map_err(bad)?;
    let files = hub::judged(&s.lair, &repo).await.map_err(bad_gateway)?;
    let pick = hub::recommend(&files, &crate::hw::hardware()).map(|f| f.name.clone());
    Ok(Json(json!({ "files": files, "pick": pick })))
}

async fn hub_downloads(State(s): S) -> Json<Vec<download::Progress>> {
    Json(s.lair.downloads.list())
}

#[derive(Deserialize)]
struct DownloadRequest {
    repo: String,
    file: String,
}

/// Starts a download, or continues one that stopped: what was already fetched is kept.
async fn hub_download(State(s): S, Json(r): Json<DownloadRequest>) -> ApiResult<StatusCode> {
    hub::check_repo(&r.repo).map_err(bad)?;
    download::start(s.lair.clone(), r.repo, r.file).await.map_err(bad)?;
    // Started, not done: progress is polled from GET /hub/downloads.
    Ok(StatusCode::NO_CONTENT)
}

async fn hub_cancel(State(s): S, Query(r): Query<DownloadRequest>) -> ApiResult<StatusCode> {
    match s.lair.downloads.cancel(&r.repo, &r.file) {
        true => Ok(StatusCode::NO_CONTENT),
        false => Err(missing("no download with that name")),
    }
}

/// This machine's models and the ones in memory right now.
async fn models(State(s): S) -> Json<Value> {
    Json(json!({ "models": s.lair.models.local(), "loaded": s.lair.models.ps() }))
}

/// Deletes an installed model's files. Shared with the Ollama-style API. Only models in lair's own
/// folder: the path comes from the catalog, never from the request.
pub async fn delete_model(lair: &crate::lair::Lair, id: &str) -> ApiResult<u64> {
    let path = lair.models.path(id).ok_or_else(|| missing(format!("model {id} not found")))?;
    let dir = lair.models.dir().ok_or_else(|| bad(anyhow::anyhow!("no models folder to delete from")))?.to_owned();
    // Out of memory first: Windows refuses to delete a file that is mapped.
    lair.models.unload(id);
    let freed = tokio::task::spawn_blocking(move || download::remove(&dir, &path)).await?.map_err(bad)?;
    lair.models.refresh().await;
    Ok(freed)
}

async fn remove_model(State(s): S, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    Ok(Json(json!({ "freed": delete_model(&s.lair, &id).await? })))
}

async fn load_model(State(s): S, Path(id): Path<String>) -> ApiResult<StatusCode> {
    s.lair.models.load(&id).await.map_err(bad)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn unload_model(State(s): S, Path(id): Path<String>) -> ApiResult<StatusCode> {
    match s.lair.models.unload(&id) {
        true => Ok(StatusCode::NO_CONTENT),
        false => Err(missing(format!("model {id} is not in memory"))),
    }
}

/// What memory holds, and what it needs to search by meaning.
async fn memory(State(s): S) -> ApiResult<Json<Value>> {
    let model = s.lair.models.embedder();
    let (pieces, embedded) = s.lair.store.chunk_counts(model.as_deref())?;
    Ok(Json(json!({
        "enabled": s.lair.config().memory(),
        "pieces": pieces,
        "embedded": embedded,
        "model": model,
        "suggested": { "repo": mem::EMBED_REPO, "file": mem::EMBED_FILE },
    })))
}

/// Forgets everything; chats and files are indexed again from scratch (unless memory is off).
async fn forget(State(s): S) -> ApiResult<StatusCode> {
    s.lair.store.clear_chunks()?;
    s.lair.store.save_kv("memory.chat", &json!(0))?;
    s.lair.memory.wake();
    Ok(StatusCode::NO_CONTENT)
}

async fn memory_search(State(s): S, Query(q): Query<SearchQuery>) -> ApiResult<Json<Vec<mem::Hit>>> {
    Ok(Json(mem::recall(&s.lair, &q.q, None, 10).await?))
}

/// Hugging Face failed or answered something unexpected: not the client's fault.
fn bad_gateway(e: anyhow::Error) -> super::ApiError {
    super::ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}"))
}
