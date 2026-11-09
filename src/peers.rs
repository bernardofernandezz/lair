//! Other lair nodes on the user's Tailscale network: finding them, trusting them, and generating on them.
//!
//! lair talks to the local `tailscaled` over its Unix socket. Discovery lists the tailnet's online
//! machines and asks each whether it runs lair; trust uses `whois`, so a request that arrives from a
//! machine in the same tailnet needs no token. Nothing here involves any service outside the tailnet.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::engine::{Done, Msg, Params};
use crate::models::{Event, ModelEntry};

const PROBE_TIMEOUT: Duration = Duration::from_millis(1500);
const WHOIS_TTL: Duration = Duration::from_secs(300);

/// What a node says about itself at `GET /lair/v1/node`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeInfo {
    pub name: String,
    pub version: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub info: NodeInfo,
    /// `http://host:port`, reached over the tailnet.
    pub base: String,
}

pub struct Peers {
    /// This node's name, used to skip itself during discovery and to name remote models `model@name`.
    pub name: String,
    socket: Option<PathBuf>,
    /// `host[:port]` addresses given with `--peer`.
    manual: Vec<String>,
    port: u16,
    http: reqwest::Client,
    nodes: RwLock<Vec<Node>>,
    whois: Mutex<HashMap<IpAddr, (Instant, bool)>>,
}

/// Tailscale addresses: 100.64.0.0/10 and fd7a:115c:a1e0::/48.
pub fn is_tailscale_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            a == 100 && (64..128).contains(&b)
        }
        IpAddr::V6(v6) => v6.segments()[..3] == [0xfd7a, 0x115c, 0xa1e0],
    }
}

impl Peers {
    pub fn new(name: String, socket: Option<PathBuf>, manual: Vec<String>, port: u16, http: reqwest::Client) -> Self {
        Self { name, socket, manual, port, http, nodes: RwLock::default(), whois: Mutex::default() }
    }

    fn local_api(&self) -> Result<reqwest::Client> {
        let socket = self.socket.as_ref().context("no Tailscale socket")?;
        Ok(reqwest::Client::builder().unix_socket(socket.as_path()).timeout(PROBE_TIMEOUT).build()?)
    }

    async fn local_get(&self, path: &str) -> Result<Value> {
        Ok(self.local_api()?.get(format!("http://local-tailscaled.sock/localapi/v0/{path}")).send().await?.error_for_status()?.json().await?)
    }

    /// This machine's Tailscale host name, when tailscaled is reachable.
    pub async fn tailscale_name(socket: &std::path::Path) -> Option<String> {
        let client = reqwest::Client::builder().unix_socket(socket).timeout(PROBE_TIMEOUT).build().ok()?;
        let status: Value = client.get("http://local-tailscaled.sock/localapi/v0/status").send().await.ok()?.json().await.ok()?;
        status["Self"]["HostName"].as_str().map(str::to_owned)
    }

    /// Asks every online tailnet machine (and every `--peer`) whether it runs lair.
    pub async fn refresh(&self) {
        let mut candidates: Vec<String> = self.manual.iter().map(|p| self.base(p)).collect();
        if let Ok(status) = self.local_get("status").await {
            for peer in status["Peer"].as_object().into_iter().flat_map(|m| m.values()) {
                if peer["Online"].as_bool() == Some(true) {
                    if let Some(dns) = peer["DNSName"].as_str() {
                        candidates.push(self.base(dns.trim_end_matches('.')));
                    }
                }
            }
        }
        let mut probes = tokio::task::JoinSet::new();
        for base in candidates {
            let http = self.http.clone();
            probes.spawn(async move {
                let info: NodeInfo = http.get(format!("{base}/lair/v1/node")).timeout(PROBE_TIMEOUT).send().await.ok()?.json().await.ok()?;
                Some(Node { info, base })
            });
        }
        let mut found: Vec<Node> = probes.join_all().await.into_iter().flatten().filter(|n| n.info.name != self.name).collect();
        found.sort_by(|a, b| a.info.name.cmp(&b.info.name));
        found.dedup_by(|a, b| a.info.name == b.info.name);
        *self.nodes.write().unwrap() = found;
    }

    fn base(&self, host: &str) -> String {
        if host.contains(':') { format!("http://{host}") } else { format!("http://{host}:{}", self.port) }
    }

    /// Models on other nodes, named `model@node`.
    pub fn models(&self) -> Vec<ModelEntry> {
        self.nodes
            .read()
            .unwrap()
            .iter()
            .flat_map(|n| n.info.models.iter().map(move |m| ModelEntry { id: format!("{}@{}", m.id, n.info.name), node: Some(n.info.name.clone()), ..m.clone() }))
            .collect()
    }

    /// Splits `model@node` and finds the node, if it is one we know.
    pub fn locate(&self, id: &str) -> Option<(String, String)> {
        let (model, node) = id.rsplit_once('@')?;
        let nodes = self.nodes.read().unwrap();
        let n = nodes.iter().find(|n| n.info.name == node)?;
        n.info.models.iter().any(|m| m.id == model).then(|| (n.base.clone(), model.to_owned()))
    }

    /// Whether `ip` is a machine in this node's tailnet, checked with tailscaled's `whois` and cached.
    pub async fn trusts(&self, ip: IpAddr) -> bool {
        if !is_tailscale_ip(ip) || self.socket.is_none() {
            return false;
        }
        if let Some((at, ok)) = self.whois.lock().unwrap().get(&ip) {
            if at.elapsed() < WHOIS_TTL {
                return *ok;
            }
        }
        let ok = self.local_get(&format!("whois?addr={ip}:0")).await.is_ok_and(|w| w["Node"].is_object());
        self.whois.lock().unwrap().insert(ip, (Instant::now(), ok));
        ok
    }

    /// Generates on another node. Dropping the receiver stops it (the remote sees the disconnect).
    pub fn generate(&self, base: String, model: String, msgs: Vec<Msg>, params: Params) -> mpsc::UnboundedReceiver<Event> {
        let (tx, rx) = mpsc::unbounded_channel();
        let http = self.http.clone();
        tokio::spawn(async move {
            let body = json!({ "model": model, "messages": msgs, "params": params });
            let result = async {
                let mut resp = http.post(format!("{base}/lair/v1/generate")).json(&body).timeout(Duration::from_secs(3600)).send().await?;
                if !resp.status().is_success() {
                    bail!("the remote node answered {}: {}", resp.status(), resp.text().await.unwrap_or_default());
                }
                let mut buf = String::new();
                while let Some(chunk) = resp.chunk().await? {
                    buf.push_str(&String::from_utf8_lossy(&chunk));
                    while let Some(end) = buf.find("\n\n") {
                        let frame: String = buf.drain(..end + 2).collect();
                        let Some(event) = remote_event(&frame) else { continue };
                        let last = matches!(event, Event::Done(_) | Event::Failed(_));
                        if tx.send(event).is_err() || last {
                            return Ok(());
                        }
                    }
                }
                bail!("the remote node closed the stream early")
            }
            .await;
            if let Err(e) = result {
                let _ = tx.send(Event::Failed(format!("{e:#}")));
            }
        });
        rx
    }
}

/// One SSE frame from a node's `/lair/v1/generate` as a generation event.
fn remote_event(frame: &str) -> Option<Event> {
    let mut name = "";
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(n) = line.strip_prefix("event:") {
            name = n.trim();
        } else if let Some(d) = line.strip_prefix("data:") {
            data.push_str(d.trim_start());
        }
    }
    let v: Value = serde_json::from_str(&data).ok()?;
    let text = || v["text"].as_str().unwrap_or_default().to_owned();
    Some(match name {
        "thinking" => Event::Thinking(text()),
        "answer" => Event::Answer(text()),
        "done" => Event::Done(Done {
            prompt_tokens: v["prompt_tokens"].as_u64().unwrap_or(0) as usize,
            completion_tokens: v["completion_tokens"].as_u64().unwrap_or(0) as usize,
            finish_reason: if v["finish_reason"] == "length" { "length" } else { "stop" },
        }),
        "failed" => Event::Failed(v["message"].as_str().unwrap_or("the remote node failed").to_owned()),
        _ => return None,
    })
}

/// The SSE frame a node sends for one local generation event; the inverse of [`remote_event`].
pub fn frame(event: &Event) -> (&'static str, Value) {
    match event {
        Event::Thinking(t) => ("thinking", json!({ "text": t })),
        Event::Answer(t) => ("answer", json!({ "text": t })),
        Event::Done(d) => ("done", json!({ "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens, "finish_reason": d.finish_reason })),
        Event::Failed(e) => ("failed", json!({ "message": e })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tailscale_ranges() {
        assert!(is_tailscale_ip("100.93.144.37".parse().unwrap()));
        assert!(is_tailscale_ip("100.127.255.255".parse().unwrap()));
        assert!(!is_tailscale_ip("100.128.0.1".parse().unwrap()));
        assert!(!is_tailscale_ip("192.168.0.10".parse().unwrap()));
        assert!(is_tailscale_ip("fd7a:115c:a1e0::429:9026".parse().unwrap()));
        assert!(!is_tailscale_ip("::1".parse().unwrap()));
    }

    #[test]
    fn events_survive_the_trip_between_nodes() {
        let events = [
            Event::Thinking("hm".into()),
            Event::Answer("Oi".into()),
            Event::Done(Done { prompt_tokens: 3, completion_tokens: 2, finish_reason: "length" }),
            Event::Failed("boom".into()),
        ];
        for e in events {
            let (name, data) = frame(&e);
            let back = remote_event(&format!("event: {name}\ndata: {data}\n\n")).unwrap();
            assert_eq!(format!("{e:?}"), format!("{back:?}"));
        }
        assert!(remote_event("event: other\ndata: {}\n\n").is_none());
    }

    #[test]
    fn remote_models_are_named_after_their_node() {
        let p = Peers::new("here".into(), None, vec![], 8080, reqwest::Client::new());
        let info = NodeInfo { name: "doom".into(), version: "0.1.0".into(), models: vec![ModelEntry { id: "qwen".into(), node: None, thinks: true, loaded: false }] };
        *p.nodes.write().unwrap() = vec![Node { info, base: "http://doom:8080".into() }];
        assert_eq!(p.models()[0].id, "qwen@doom");
        assert_eq!(p.locate("qwen@doom"), Some(("http://doom:8080".into(), "qwen".into())));
        assert_eq!(p.locate("other@doom"), None);
        assert_eq!(p.locate("qwen"), None);
    }
}
