//! The tools an agent may call, what each needs from the user, and how each runs.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::jail;
use crate::lair::Lair;

/// Characters of tool output handed back to the model; small models have small contexts.
const MAX_OUTPUT: usize = 6000;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const FETCH_LIMIT: usize = 2 * 1024 * 1024;

pub struct Spec {
    pub name: &'static str,
    /// One line for the model: what the tool does and its arguments.
    pub usage: &'static str,
    /// Changes something or reaches outside this machine, so the user approves each call.
    pub needs_approval: bool,
}

pub const SPECS: &[Spec] = &[
    Spec { name: "list_dir", usage: r#"lista uma pasta do workspace. {"path": "."}"#, needs_approval: false },
    Spec { name: "read_file", usage: r#"lê um arquivo de texto do workspace. {"path": "notas.md"}"#, needs_approval: false },
    Spec {
        name: "write_file",
        usage: r#"cria ou substitui um arquivo do workspace. {"path": "notas.md", "content": "..."}"#,
        needs_approval: true,
    },
    Spec {
        name: "edit_file",
        usage: r#"troca um trecho exato de um arquivo. {"path": "a.txt", "find": "texto atual", "replace": "texto novo"}"#,
        needs_approval: true,
    },
    Spec { name: "delete_file", usage: r#"apaga um arquivo do workspace. {"path": "velho.txt"}"#, needs_approval: true },
    Spec {
        name: "run_command",
        usage: r#"roda um comando de shell na pasta do workspace (60 s no máximo). {"command": "ls -la"}"#,
        needs_approval: true,
    },
    Spec { name: "web_search", usage: r#"busca na web. {"query": "..."}"#, needs_approval: true },
    Spec { name: "fetch_url", usage: r#"baixa uma página e devolve o texto. {"url": "https://..."}"#, needs_approval: true },
    Spec {
        name: "search_conversations",
        usage: r#"procura nas conversas salvas do usuário. {"query": "..."}"#,
        needs_approval: false,
    },
    Spec { name: "read_conversation", usage: r#"lê uma conversa salva pelo id. {"id": "..."}"#, needs_approval: false },
];

pub fn spec(name: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|s| s.name == name)
}

/// The instructions that turn a chat model into an agent.
pub fn system_prompt() -> String {
    let mut out = String::from(
        "# Ferramentas\n\
         Você pode agir usando ferramentas. Para usar uma, escreva exatamente\n\
         <tool_call>{\"name\": \"NOME\", \"arguments\": {...}}</tool_call>\n\
         e pare. O resultado chega numa mensagem <tool_response>. Use uma ferramenta por vez, \
         só quando precisar. Se disser que vai fazer algo, faça a chamada logo em seguida, na mesma \
         resposta; não termine a resposta só anunciando. Com o resultado em mãos, responda ao usuário \
         normalmente, sem <tool_call>.\n\
         Caminhos são relativos à pasta de trabalho (workspace).\n\nFerramentas:\n",
    );
    for s in SPECS {
        out.push_str(&format!("- {}: {}\n", s.name, s.usage));
    }
    out
}

/// Where tools act. `searxng` is the base URL of a SearXNG instance, when web search is set up.
pub struct Tools {
    pub workspace: PathBuf,
    pub searxng: Option<String>,
    http: reqwest::Client,
}

impl Tools {
    pub fn new(workspace: PathBuf, searxng: Option<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent(concat!("lair/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("an HTTP client with default settings builds");
        Self { workspace, searxng, http }
    }

    /// Runs one call; the text is what the model reads back, already shortened.
    pub async fn run(&self, lair: &Lair, name: &str, args: &Value) -> Result<String> {
        let out = match name {
            "list_dir" => self.list_dir(str_arg(args, "path").unwrap_or(".")),
            "read_file" => self.read_file(req(args, "path")?),
            "write_file" => self.write_file(req(args, "path")?, req(args, "content")?),
            "edit_file" => self.edit_file(req(args, "path")?, req(args, "find")?, str_arg(args, "replace").unwrap_or("")),
            "delete_file" => {
                std::fs::remove_file(jail::existing(&self.workspace, req(args, "path")?)?)?;
                Ok("apagado".into())
            }
            "run_command" => self.run_command(req(args, "command")?).await,
            "web_search" => self.web_search(req(args, "query")?).await,
            "fetch_url" => self.fetch_url(req(args, "url")?).await,
            "search_conversations" => {
                let hits = lair.store.search(req(args, "query")?)?;
                Ok(if hits.is_empty() { "nenhuma conversa encontrada".into() } else { serde_json::to_string_pretty(&hits)? })
            }
            "read_conversation" => {
                let id = req(args, "id")?;
                let c = lair.store.get(id)?.with_context(|| format!("conversa {id} não existe"))?;
                let body: String = lair
                    .store
                    .messages(id)?
                    .iter()
                    .filter(|m| m.role != "tool")
                    .map(|m| format!("{}: {}\n", m.role, m.content))
                    .collect();
                Ok(format!("# {}\n{body}", c.title))
            }
            other => bail!("ferramenta desconhecida: {other}"),
        }?;
        Ok(shorten(out))
    }

    fn list_dir(&self, path: &str) -> Result<String> {
        let dir = jail::existing(&self.workspace, path)?;
        let mut names: Vec<String> = std::fs::read_dir(dir)?
            .flatten()
            .map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if e.path().is_dir() { format!("{name}/") } else { name }
            })
            .collect();
        names.sort();
        Ok(if names.is_empty() { "(pasta vazia)".into() } else { names.join("\n") })
    }

    fn read_file(&self, path: &str) -> Result<String> {
        let bytes = std::fs::read(jail::existing(&self.workspace, path)?)?;
        String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("{path} não é um arquivo de texto"))
    }

    fn write_file(&self, path: &str, content: &str) -> Result<String> {
        let target = jail::creatable(&self.workspace, path)?;
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, content)?;
        Ok(format!("{path} gravado ({} bytes)", content.len()))
    }

    fn edit_file(&self, path: &str, find: &str, replace: &str) -> Result<String> {
        let target = jail::existing(&self.workspace, path)?;
        let text = std::fs::read_to_string(&target)?;
        match text.matches(find).count() {
            0 => bail!("o trecho não aparece em {path}"),
            1 => {
                std::fs::write(&target, text.replacen(find, replace, 1))?;
                Ok(format!("{path} editado"))
            }
            n => bail!("o trecho aparece {n} vezes em {path}; inclua mais contexto para ser único"),
        }
    }

    async fn run_command(&self, command: &str) -> Result<String> {
        let child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.workspace)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let out = tokio::time::timeout(COMMAND_TIMEOUT, child.wait_with_output())
            .await
            .map_err(|_| anyhow::anyhow!("o comando passou de {} s e foi interrompido", COMMAND_TIMEOUT.as_secs()))??;
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        let err = String::from_utf8_lossy(&out.stderr);
        if !err.trim().is_empty() {
            text.push_str(&format!("\n[stderr]\n{err}"));
        }
        Ok(format!("[código de saída {}]\n{}", out.status.code().unwrap_or(-1), text.trim_end()))
    }

    async fn web_search(&self, query: &str) -> Result<String> {
        #[derive(Deserialize)]
        struct Results {
            results: Vec<Hit>,
        }
        #[derive(Deserialize)]
        struct Hit {
            title: String,
            url: String,
            #[serde(default)]
            content: String,
        }
        let base = self.searxng.as_deref().context("busca na web não configurada: defina LAIR_SEARXNG")?;
        let r: Results = self
            .http
            .get(format!("{}/search", base.trim_end_matches('/')))
            .query(&[("q", query), ("format", "json")])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let hits: Vec<Value> = r.results.into_iter().take(6).map(|h| json!({ "title": h.title, "url": h.url, "snippet": h.content })).collect();
        Ok(if hits.is_empty() { "nenhum resultado".into() } else { serde_json::to_string_pretty(&hits)? })
    }

    async fn fetch_url(&self, url: &str) -> Result<String> {
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            bail!("só URLs http(s)");
        }
        let resp = self.http.get(url).send().await?.error_for_status()?;
        let html = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.contains("html"));
        let mut body = Vec::new();
        let mut resp = resp;
        while let Some(chunk) = resp.chunk().await? {
            body.extend_from_slice(&chunk);
            if body.len() > FETCH_LIMIT {
                break;
            }
        }
        Ok(if html {
            html2text::from_read(body.as_slice(), 100).unwrap_or_default()
        } else {
            String::from_utf8_lossy(&body).into_owned()
        })
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn req<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    str_arg(args, key).with_context(|| format!("falta o argumento \"{key}\" (texto)"))
}

fn shorten(mut s: String) -> String {
    if let Some((i, _)) = s.char_indices().nth(MAX_OUTPUT) {
        let total = s.chars().count();
        s.truncate(i);
        s.push_str(&format!("\n… (cortado: {total} caracteres no total)"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools(name: &str) -> Tools {
        let p = std::env::temp_dir().join(format!("lair-tools-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Tools::new(p, None)
    }

    #[test]
    fn files_are_written_edited_and_listed_inside_the_workspace() {
        let t = tools("files");
        t.write_file("docs/a.md", "um dois um").unwrap();
        assert_eq!(t.list_dir(".").unwrap(), "docs/");
        assert_eq!(t.read_file("docs/a.md").unwrap(), "um dois um");
        assert!(t.edit_file("docs/a.md", "um", "três").unwrap_err().to_string().contains("2 vezes"));
        t.edit_file("docs/a.md", "dois", "2").unwrap();
        assert_eq!(t.read_file("docs/a.md").unwrap(), "um 2 um");
        assert!(t.write_file("../fora.md", "x").is_err());
        assert!(t.read_file("/etc/passwd").is_err());
    }

    #[tokio::test]
    async fn commands_run_in_the_workspace_with_exit_code_and_timeout_message() {
        let t = tools("cmd");
        let out = t.run_command("pwd; echo oops >&2; exit 3").await.unwrap();
        assert!(out.starts_with("[código de saída 3]"));
        assert!(out.contains(t.workspace.file_name().unwrap().to_str().unwrap()));
        assert!(out.contains("[stderr]\noops"));
    }

    #[test]
    fn long_output_is_cut_with_a_note() {
        let s = shorten("é".repeat(MAX_OUTPUT + 10));
        assert!(s.ends_with(&format!("(cortado: {} caracteres no total)", MAX_OUTPUT + 10)));
        assert_eq!(s.chars().filter(|&c| c == 'é').count(), MAX_OUTPUT);
        assert_eq!(shorten("curto".into()), "curto");
    }

    #[test]
    fn tools_that_change_things_or_go_online_need_approval() {
        for name in ["write_file", "edit_file", "delete_file", "run_command", "web_search", "fetch_url"] {
            assert!(spec(name).unwrap().needs_approval, "{name}");
        }
        for name in ["list_dir", "read_file", "search_conversations", "read_conversation"] {
            assert!(!spec(name).unwrap().needs_approval, "{name}");
        }
    }
}
