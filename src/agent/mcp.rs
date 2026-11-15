//! MCP servers as agent tools: each server named in `<data>/mcp.json` (the `mcpServers` format other
//! apps use) runs as a child process, and its tools are offered as `mcp__<server>__<tool>`. They always
//! ask before running: they are someone else's code.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

const TIMEOUT: Duration = Duration::from_secs(60);
const PREFIX: &str = "mcp__";

#[derive(Deserialize)]
struct File {
    #[serde(rename = "mcpServers", default)]
    servers: HashMap<String, Launch>,
}

#[derive(Deserialize)]
struct Launch {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

/// A tool as the model sees it.
pub struct Tool {
    /// `mcp__<server>__<tool>`.
    pub name: String,
    pub usage: String,
}

struct Server {
    name: String,
    io: Mutex<Io>,
}

struct Io {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next: u64,
}

/// The servers of one `mcp.json`, started again when the file changes: an edit applies from the next
/// agent turn, without restarting lair. The servers it replaces stop once no turn uses them.
#[derive(Default)]
pub struct Live {
    path: Option<PathBuf>,
    current: Mutex<(Option<SystemTime>, Arc<Mcp>)>,
}

impl Live {
    pub fn new(path: PathBuf) -> Self {
        Self { path: Some(path), current: Default::default() }
    }

    pub async fn get(&self) -> Arc<Mcp> {
        let Some(path) = &self.path else { return Arc::default() };
        let changed = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        let mut current = self.current.lock().await;
        if current.0 != changed {
            *current = (changed, Arc::new(Mcp::start(path).await));
        }
        current.1.clone()
    }
}

#[derive(Default)]
pub struct Mcp {
    servers: Vec<Server>,
    tools: Vec<Tool>,
}

impl Mcp {
    /// Starts the servers `path` names; one that fails is reported and left out. No file, no servers.
    pub async fn start(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else { return Self::default() };
        let file: File = match serde_json::from_str(&text) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{}: {e}; no MCP servers started", path.display());
                return Self::default();
            }
        };
        let mut mcp = Self::default();
        for (name, launch) in file.servers {
            match Server::start(&name, &launch).await {
                Ok((server, tools)) => {
                    eprintln!("MCP server {name}: {} tools", tools.len());
                    mcp.servers.push(server);
                    mcp.tools.extend(tools);
                }
                Err(e) => eprintln!("MCP server {name} didn't start: {e:#}"),
            }
        }
        mcp
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    pub fn has(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }

    /// Runs `mcp__<server>__<tool>`; the text is what the server answered.
    pub async fn call(&self, name: &str, arguments: &Value) -> Result<String> {
        let (server, tool) = name.strip_prefix(PREFIX).and_then(|n| n.split_once("__")).context("not an MCP tool")?;
        let s = self.servers.iter().find(|s| s.name == server).context("no such MCP server")?;
        let args = if arguments.is_object() { arguments.clone() } else { json!({}) };
        let result = s.request("tools/call", json!({ "name": tool, "arguments": args })).await?;
        let text: Vec<&str> = result["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|c| if c["type"] == "text" { c["text"].as_str() } else { Some("[non-text content]") })
            .collect();
        let text = text.join("\n");
        if result["isError"] == true {
            bail!("{text}");
        }
        Ok(text)
    }
}

impl Server {
    async fn start(name: &str, launch: &Launch) -> Result<(Self, Vec<Tool>)> {
        if name.contains("__") {
            bail!("server names can't contain `__`");
        }
        let mut child = Command::new(&launch.command)
            .args(&launch.args)
            .envs(&launch.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("running {}", launch.command))?;
        let io = Io {
            stdin: child.stdin.take().context("no stdin")?,
            stdout: BufReader::new(child.stdout.take().context("no stdout")?),
            _child: child,
            next: 0,
        };
        let server = Self { name: name.to_owned(), io: Mutex::new(io) };
        let init = json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "lair", "version": env!("CARGO_PKG_VERSION") },
        });
        server.request("initialize", init).await?;
        server.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).await?;
        let mut tools = Vec::new();
        let mut cursor = Value::Null;
        // A server that keeps handing out cursors is cut off rather than followed forever.
        for _ in 0..100 {
            let params = if cursor.is_null() { json!({}) } else { json!({ "cursor": cursor }) };
            let page = server.request("tools/list", params).await?;
            tools.extend(page["tools"].as_array().into_iter().flatten().filter_map(|t| {
                let tool = t["name"].as_str()?;
                let about = t["description"].as_str().unwrap_or("").lines().next().unwrap_or("");
                Some(Tool {
                    name: format!("{PREFIX}{name}__{tool}"),
                    usage: format!("{about} Arguments: {}", t["inputSchema"]),
                })
            }));
            cursor = page["nextCursor"].clone();
            if !cursor.is_string() {
                break;
            }
        }
        Ok((server, tools))
    }

    async fn send(&self, msg: &Value) -> Result<()> {
        let mut io = self.io.lock().await;
        write(&mut io.stdin, msg).await
    }

    /// One request, answered by the line with its id. Other lines are notifications, ignored, or the
    /// server's own requests, refused, and late answers to requests that timed out.
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let mut io = self.io.lock().await;
        io.next += 1;
        let id = io.next;
        write(&mut io.stdin, &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })).await?;
        let wait = async {
            let mut line = String::new();
            loop {
                line.clear();
                if io.stdout.read_line(&mut line).await? == 0 {
                    bail!("the MCP server {} exited", self.name);
                }
                let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
                if msg.get("method").is_some() {
                    if let Some(theirs) = msg.get("id") {
                        let no = json!({ "jsonrpc": "2.0", "id": theirs, "error": { "code": -32601, "message": "not supported" } });
                        write(&mut io.stdin, &no).await?;
                    }
                } else if msg["id"] == id {
                    return match msg.get("error") {
                        Some(e) => Err(anyhow!("{}", e["message"].as_str().unwrap_or("MCP error"))),
                        None => Ok(msg["result"].clone()),
                    };
                }
            }
        };
        tokio::time::timeout(TIMEOUT, wait).await.map_err(|_| anyhow!("the MCP server took over {TIMEOUT:?}"))?
    }
}

async fn write(stdin: &mut ChildStdin, msg: &Value) -> Result<()> {
    let mut line = serde_json::to_vec(msg)?;
    line.push(b'\n');
    stdin.write_all(&line).await?;
    Ok(stdin.flush().await?)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A server in a few lines of shell: answers by request id, and says hello first, unasked.
    const FAKE: &str = r#"
        echo '{"jsonrpc":"2.0","method":"notifications/message","params":{}}'
        while read -r line; do
          case "$line" in
            *'"initialize"'*) echo '{"jsonrpc":"2.0","id":1,"result":{}}' ;;
            *'"cursor"'*) echo '{"jsonrpc":"2.0","id":3,"result":{"tools":[{"name":"two"}]}}' ;;
            *'"tools/list"'*) echo '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo","description":"Echoes.","inputSchema":{"type":"object"}}],"nextCursor":"p2"}}' ;;
            *'"tools/call"'*) echo '{"jsonrpc":"2.0","method":"roots/list","id":"r1"}'
                              echo '{"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"hi"}]}}' ;;
          esac
        done"#;

    #[tokio::test]
    async fn tools_are_listed_and_called() {
        let dir = std::env::temp_dir().join(format!("lair-mcp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcp.json");
        let config = json!({ "mcpServers": { "fake": { "command": "sh", "args": ["-c", FAKE] } } });
        std::fs::write(&path, config.to_string()).unwrap();
        let live = Live::new(path.clone());
        let mcp = live.get().await;
        let names: Vec<&str> = mcp.tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["mcp__fake__echo", "mcp__fake__two"], "every page");
        assert!(Arc::ptr_eq(&mcp, &live.get().await), "same file, same servers");
        assert!(mcp.tools()[0].usage.starts_with("Echoes."));
        assert_eq!(mcp.call("mcp__fake__echo", &json!({})).await.unwrap(), "hi");
        assert!(mcp.call("mcp__other__echo", &json!({})).await.is_err());
        assert!(Mcp::start(&dir.join("none.json")).await.tools().is_empty(), "no file, no servers");
        let later = std::fs::File::options().write(true).open(&path).unwrap();
        later.set_modified(SystemTime::now() + Duration::from_secs(5)).unwrap();
        assert!(!Arc::ptr_eq(&mcp, &live.get().await), "an edited file starts them again");
    }
}
