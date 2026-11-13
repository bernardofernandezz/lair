//! The HTTP API driven through its router, without a socket. They share the process's one engine,
//! which has models only when `LAIR_TEST_MODEL` / `LAIR_TEST_EMBED_MODEL` name them; the tests that
//! need one are ignored otherwise.

use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

use axum::body::Body;
use axum::extract::connect_info::MockConnectInfo;
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

use super::{AppState, router};
use crate::config::Config;
use crate::engine::{Params, Reasoning};
use crate::lair::Lair;
use crate::models::Models;
use crate::peers::Peers;
use crate::store::Store;

fn state(token: Option<&str>) -> Arc<AppState> {
    let http = reqwest::Client::new();
    let peers = Arc::new(Peers::new(None, 8080, http.clone()));
    let workspace = std::env::temp_dir().join(format!("lair-api-{}", std::process::id()));
    std::fs::create_dir_all(&workspace).unwrap();
    let lair = Lair {
        models: Models::new(crate::engine::tests::test_engine(), peers.clone(), Default::default()),
        peers,
        store: Store::memory(),
        skills: RwLock::default(),
        http,
        assets: Default::default(),
        workspace,
        approvals: Default::default(),
        downloads: Default::default(),
        hub: Default::default(),
        memory: Default::default(),
        params: Params {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 64,
            seed: None,
            reasoning: Reasoning::Off,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 0.0,
            stop: vec![],
            grammar: None,
            raw: false,
        },
        flags: Config::default(),
        config: RwLock::default(),
        auto_name: "test".into(),
        stopping: tokio::sync::watch::Sender::new(false),
    };
    Arc::new(AppState { lair: Arc::new(lair), token: token.map(str::to_owned), addr: "127.0.0.1:0".into() })
}

/// One request through the whole router; the body comes back as JSON (or a string, if it isn't).
async fn call(
    s: &Arc<AppState>,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(path);
    if let Some(t) = token {
        req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let req = match body {
        Some(b) => req.header(header::CONTENT_TYPE, "application/json").body(Body::from(b.to_string())),
        None => req.body(Body::empty()),
    }
    .unwrap();
    let app = router(s.clone()).layer(MockConnectInfo(SocketAddr::from(([127, 0, 0, 1], 9))));
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024 * 1024).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).into_owned();
    (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

fn chat_model(s: &AppState) -> String {
    s.lair.models.local().into_iter().find(|m| !m.embedding).expect("LAIR_TEST_MODEL must be a chat model").id
}

#[tokio::test]
async fn the_apis_ask_for_the_token_and_answer_unknown_things_with_404() {
    let s = state(Some("secret"));
    for path in ["/v1/models", "/api/tags", "/lair/v1/models"] {
        assert_eq!(call(&s, "GET", path, None, None).await.0, StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(call(&s, "GET", path, Some("wrong"), None).await.0, StatusCode::UNAUTHORIZED, "{path}");
        assert_eq!(call(&s, "GET", path, Some("secret"), None).await.0, StatusCode::OK, "{path}");
    }
    let open = state(None);
    let (status, list) = call(&open, "GET", "/v1/models", None, None).await;
    assert_eq!((status, list["object"].as_str()), (StatusCode::OK, Some("list")));
    assert_eq!(call(&open, "GET", "/v1/models/no-such-model", None, None).await.0, StatusCode::NOT_FOUND);
    assert_eq!(call(&open, "GET", "/v1/nothing", None, None).await.0, StatusCode::NOT_FOUND);
    let (status, ps) = call(&open, "GET", "/api/ps", None, None).await;
    assert!(status == StatusCode::OK && ps["models"].is_array());
    assert_eq!(call(&open, "DELETE", "/lair/v1/models/no-such-model", None, None).await.0, StatusCode::NOT_FOUND);
    let gone = json!({ "model": "no-such-model" });
    assert_eq!(call(&open, "DELETE", "/api/delete", None, Some(gone.clone())).await.0, StatusCode::NOT_FOUND);
    assert_eq!(call(&open, "POST", "/api/show", None, Some(gone)).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn bad_requests_are_refused_before_anything_runs() {
    let s = state(None);
    let ask = |extra: Value| {
        let mut body = json!({ "messages": [{ "role": "user", "content": "hi" }] });
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        body
    };
    let (status, e) = call(&s, "POST", "/v1/chat/completions", None, Some(ask(json!({ "model": "nope" })))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{e}");
    let format = json!({ "response_format": { "type": "yaml" } });
    let (status, e) = call(&s, "POST", "/v1/chat/completions", None, Some(ask(format))).await;
    assert!(status == StatusCode::BAD_REQUEST && e["error"]["message"].as_str().unwrap().contains("yaml"), "{e}");
    let empty = json!({ "messages": [] });
    assert_eq!(call(&s, "POST", "/v1/chat/completions", None, Some(empty)).await.0, StatusCode::BAD_REQUEST);
    let (status, e) = call(&s, "POST", "/v1/messages", None, Some(ask(json!({ "model": "nope" })))).await;
    assert!(status == StatusCode::NOT_FOUND && e["type"] == "error", "{e}");
    assert_eq!(call(&s, "POST", "/v1/messages", None, Some(json!({}))).await.0, StatusCode::BAD_REQUEST);

    // Pulling takes Hugging Face repos only: a URL or an Ollama library name never reaches the network.
    for name in ["https://evil.example/m.gguf", "llama3"] {
        let (status, e) = call(&s, "POST", "/api/pull", None, Some(json!({ "model": name }))).await;
        assert!(status == StatusCode::BAD_REQUEST && e["error"].as_str().unwrap().contains("owner/name"), "{e}");
    }
    let bad = json!({ "repo": "../etc", "file": "passwd.gguf" });
    assert_eq!(call(&s, "POST", "/lair/v1/hub/downloads", None, Some(bad)).await.0, StatusCode::BAD_REQUEST);

    let (status, e) = call(&s, "POST", "/v1/embeddings", None, Some(json!({ "input": [] }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{e}");
    if s.lair.models.embedder().is_none() {
        let (status, e) = call(&s, "POST", "/v1/embeddings", None, Some(json!({ "input": "hi" }))).await;
        assert!(
            status == StatusCode::BAD_REQUEST && e["error"]["message"].as_str().unwrap().contains("embedding model")
        );
    }
}

#[tokio::test]
async fn settings_never_give_the_hugging_face_token_back() {
    let s = state(None);
    let patch = json!({ "hfToken": "hf_secret", "memory": false });
    assert_eq!(call(&s, "PATCH", "/lair/v1/settings", None, Some(patch)).await.0, StatusCode::NO_CONTENT);
    let (_, settings) = call(&s, "GET", "/lair/v1/settings", None, None).await;
    assert_eq!(settings["saved"]["hfToken"], json!(true));
    assert!(!settings.to_string().contains("hf_secret"));
    assert_eq!(s.lair.config().hf_token.as_deref(), Some("hf_secret"));
    // The UI sends back what it was shown; that must not replace the token with `true`.
    let echo = json!({ "hfToken": true, "memory": true });
    assert_eq!(call(&s, "PATCH", "/lair/v1/settings", None, Some(echo)).await.0, StatusCode::NO_CONTENT);
    assert_eq!(s.lair.config().hf_token.as_deref(), Some("hf_secret"));
    let (_, memory) = call(&s, "GET", "/lair/v1/memory", None, None).await;
    assert_eq!((memory["enabled"].as_bool(), memory["pieces"].as_u64()), (Some(true), Some(0)));
}

#[tokio::test]
#[ignore = "needs LAIR_TEST_MODEL=/path/to/model.gguf"]
async fn a_chat_completion_with_a_real_model() {
    let _alone = crate::engine::tests::REAL_MODEL.lock().await;
    let s = state(None);
    let model = chat_model(&s);
    let body = json!({
        "model": "auto",
        "messages": [{ "role": "user", "content": "Name a colour, as JSON with the key \"colour\"." }],
        "response_format": { "type": "json_object" },
        "max_tokens": 60
    });
    let (status, v) = call(&s, "POST", "/v1/chat/completions", None, Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["model"].as_str(), Some(model.as_str()), "auto names the model that answered");
    assert!(v["lair"]["model_choice"].is_string(), "{v}");
    let content: Value = serde_json::from_str(v["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
    assert!(content.is_object(), "{content}");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() > 0 && v["timings"]["predicted_per_second"].is_number());

    let stream = json!({
        "messages": [{ "role": "user", "content": "Say hi." }],
        "stream": true, "stream_options": { "include_usage": true }, "max_tokens": 8
    });
    let (status, text) = call(&s, "POST", "/v1/chat/completions", None, Some(stream)).await;
    let text = text.as_str().unwrap();
    assert_eq!(status, StatusCode::OK);
    let usage = text.lines().filter(|l| l.contains("\"usage\"")).count();
    assert!(usage == 1 && text.trim_end().ends_with("data: [DONE]"), "{text}");

    let plain = json!({ "prompt": "The capital of France is", "max_tokens": 6 });
    let (status, v) = call(&s, "POST", "/v1/completions", None, Some(plain)).await;
    assert!(status == StatusCode::OK && v["choices"][0]["text"].as_str().unwrap().contains("Paris"), "{v}");

    let gen_req = json!({ "model": model, "prompt": "Say hi.", "stream": false, "keep_alive": 0,
                          "options": { "num_predict": 8 } });
    let (status, v) = call(&s, "POST", "/api/generate", None, Some(gen_req)).await;
    assert!(status == StatusCode::OK && v["eval_duration"].as_u64().unwrap() > 0, "{v}");
    let (_, ps) = call(&s, "GET", "/api/ps", None, None).await;
    let in_memory = ps["models"].as_array().unwrap().iter().any(|m| m["name"] == model.as_str());
    assert!(!in_memory, "keep_alive 0 unloads after the reply: {ps}");
    let (_, tags) = call(&s, "GET", "/api/tags", None, None).await;
    assert!(!tags["models"][0]["details"]["quantization_level"].as_str().unwrap().is_empty(), "{tags}");
}

#[tokio::test]
#[ignore = "needs LAIR_TEST_EMBED_MODEL=/path/to/embedding-model.gguf"]
async fn embeddings_and_memory_with_a_real_model() {
    let _alone = crate::engine::tests::REAL_MODEL.lock().await;
    let s = state(None);
    let body = json!({ "input": ["sourdough bread", "baking bread at home"], "encoding_format": "base64" });
    let (status, v) = call(&s, "POST", "/v1/embeddings", None, Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(
        v["data"][1]["embedding"].as_str().unwrap().len() > 100 && v["usage"]["prompt_tokens"].as_u64().unwrap() > 0
    );
    let (status, v) = call(&s, "POST", "/api/embed", None, Some(json!({ "input": "hello" }))).await;
    assert!(status == StatusCode::OK && v["embeddings"][0].as_array().unwrap().len() > 100, "{v}");

    // A chat about bread is remembered and found again by meaning, in other words.
    let store = &s.lair.store;
    let bread = store.create(Default::default()).unwrap();
    let say = |conv: &str, role: &str, text: &str| {
        let m = crate::store::Message { role: role.into(), content: text.into(), ..Default::default() };
        store.append(conv, &m).unwrap();
    };
    say(&bread.id, "user", "My sourdough starter needs feeding every twelve hours with rye flour.");
    let other = store.create(Default::default()).unwrap();
    say(&other.id, "user", "The quarterly tax filing deadline for the company is in April.");
    std::fs::write(s.lair.workspace.join("gpu-notes.md"), "The workstation has an RTX 3060 with 12 GB of VRAM.")
        .unwrap();
    crate::memory::sync(&s.lair).unwrap();
    crate::memory::sync(&s.lair).unwrap();
    let (_, m) = call(&s, "GET", "/lair/v1/memory", None, None).await;
    assert_eq!(m["pieces"].as_u64(), Some(3), "indexing twice adds nothing: {m}");

    let hits = crate::memory::recall(&s.lair, "how often do I feed my bread culture?", None, 3).await.unwrap();
    assert!(hits.is_empty() || hits[0].similarity.is_none(), "nothing is embedded yet, so only words can match");
    // The indexer embeds what is pending; here by hand, through the same search route afterwards.
    let model = s.lair.models.embedder().unwrap();
    let pending = store.pending_chunks(&model, 16).unwrap();
    let (ids, texts): (Vec<i64>, Vec<String>) = pending.into_iter().unzip();
    let (vectors, _) = s.lair.models.embed(&model, texts).await.unwrap();
    store.set_vectors(&model, &ids.into_iter().zip(vectors).collect::<Vec<_>>()).unwrap();

    let (status, hits) =
        call(&s, "GET", "/lair/v1/memory/search?q=how+often+do+I+feed+my+bread+culture", None, None).await;
    assert_eq!(status, StatusCode::OK, "{hits}");
    assert_eq!(hits[0]["ref"].as_str(), Some(bread.id.as_str()), "{hits}");
    assert!(hits[0]["similarity"].as_f64().unwrap() > 0.45, "{hits}");
    let (_, hits) = call(&s, "GET", "/lair/v1/memory/search?q=which+graphics+card+is+in+my+computer", None, None).await;
    assert_eq!((hits[0]["source"].as_str(), hits[0]["ref"].as_str()), (Some("file"), Some("gpu-notes.md")), "{hits}");

    store.delete(&bread.id).unwrap();
    let (_, m) = call(&s, "GET", "/lair/v1/memory", None, None).await;
    assert_eq!(m["pieces"].as_u64(), Some(2), "a deleted chat is forgotten");
    assert_eq!(call(&s, "DELETE", "/lair/v1/memory", None, None).await.0, StatusCode::NO_CONTENT);
    let (_, m) = call(&s, "GET", "/lair/v1/memory", None, None).await;
    assert_eq!(m["pieces"].as_u64(), Some(0));
}
