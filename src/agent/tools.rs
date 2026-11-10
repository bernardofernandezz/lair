//! The tools an agent may call, what each needs from the user, and how each runs.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::jail;
use crate::lair::Lair;
use crate::skills::Manifest;

/// Characters of tool output handed back to the model; small models have small contexts.
const MAX_OUTPUT: usize = 6000;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const FETCH_LIMIT: usize = 2 * 1024 * 1024;

/// Which handler runs a call; one per row of [`SPECS`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tool {
    ListDir,
    ReadFile,
    WriteFile,
    EditFile,
    DeleteFile,
    RunCommand,
    WebSearch,
    FetchUrl,
    SearchConversations,
    ReadConversation,
    ListSkills,
    SaveSkill,
}

pub struct Spec {
    pub tool: Tool,
    /// What the model writes in a call.
    pub name: &'static str,
    /// What the UI shows for a step.
    pub label: &'static str,
    /// The argument that says what a call is about, shown next to the label.
    pub subject: &'static str,
    /// One line for the model: what the tool does and its arguments.
    pub usage: &'static str,
    /// Changes something or reaches outside this machine, so the user approves each call.
    pub needs_approval: bool,
}

pub const SPECS: &[Spec] = &[
    Spec {
        tool: Tool::ListDir,
        name: "list_dir",
        label: "Listar pasta",
        subject: "path",
        usage: r#"lista uma pasta do workspace. {"path": "."}"#,
        needs_approval: false,
    },
    Spec {
        tool: Tool::ReadFile,
        name: "read_file",
        label: "Ler arquivo",
        subject: "path",
        usage: r#"lê um arquivo de texto do workspace. {"path": "notas.md"}"#,
        needs_approval: false,
    },
    Spec {
        tool: Tool::WriteFile,
        name: "write_file",
        label: "Escrever arquivo",
        subject: "path",
        usage: r#"cria ou substitui um arquivo do workspace. {"path": "notas.md", "content": "..."}"#,
        needs_approval: true,
    },
    Spec {
        tool: Tool::EditFile,
        name: "edit_file",
        label: "Editar arquivo",
        subject: "path",
        usage: r#"troca um trecho exato de um arquivo. {"path": "a.txt", "find": "texto atual", "replace": "texto novo"}"#,
        needs_approval: true,
    },
    Spec {
        tool: Tool::DeleteFile,
        name: "delete_file",
        label: "Apagar arquivo",
        subject: "path",
        usage: r#"apaga um arquivo do workspace. {"path": "velho.txt"}"#,
        needs_approval: true,
    },
    Spec {
        tool: Tool::RunCommand,
        name: "run_command",
        label: "Rodar comando",
        subject: "command",
        usage: r#"roda um comando de shell na pasta do workspace (60 s no máximo). {"command": "ls -la"}"#,
        needs_approval: true,
    },
    Spec {
        tool: Tool::WebSearch,
        name: "web_search",
        label: "Buscar na web",
        subject: "query",
        usage: r#"busca na web. {"query": "..."}"#,
        needs_approval: true,
    },
    Spec {
        tool: Tool::FetchUrl,
        name: "fetch_url",
        label: "Abrir página",
        subject: "url",
        usage: r#"baixa uma página e devolve o texto. {"url": "https://..."}"#,
        needs_approval: true,
    },
    Spec {
        tool: Tool::SearchConversations,
        name: "search_conversations",
        label: "Buscar conversas",
        subject: "query",
        usage: r#"procura nas conversas salvas do usuário. {"query": "..."}"#,
        needs_approval: false,
    },
    Spec {
        tool: Tool::ReadConversation,
        name: "read_conversation",
        label: "Ler conversa",
        subject: "id",
        usage: r#"lê uma conversa salva pelo id. {"id": "..."}"#,
        needs_approval: false,
    },
    Spec {
        tool: Tool::ListSkills,
        name: "list_skills",
        label: "Listar skills",
        subject: "",
        usage: "lista as skills e seus gatilhos. {}",
        needs_approval: false,
    },
    Spec {
        tool: Tool::SaveSkill,
        name: "save_skill",
        label: "Salvar skill",
        subject: "id",
        usage: r#"cria ou substitui uma skill (instruções extras ativadas por palavras). {"id": "pirata", "name": "Pirata", "description": "...", "when": ["pirata"], "prompt": "Responda como um pirata."}"#,
        needs_approval: true,
    },
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

/// Runs one call; the text is what the model reads back, already shortened.
pub async fn run(lair: &Lair, tool: Tool, args: &Value) -> Result<String> {
    let ws = lair.workspace.as_path();
    let out = match tool {
        Tool::ListDir => list_dir(ws, str_arg(args, "path").unwrap_or(".")),
        Tool::ReadFile => read_file(ws, req(args, "path")?),
        Tool::WriteFile => write_file(ws, req(args, "path")?, req(args, "content")?),
        Tool::EditFile => edit_file(ws, req(args, "path")?, req(args, "find")?, str_arg(args, "replace").unwrap_or("")),
        Tool::DeleteFile => {
            std::fs::remove_file(jail::existing(ws, req(args, "path")?)?)?;
            Ok("apagado".into())
        }
        Tool::RunCommand => run_command(ws, req(args, "command")?).await,
        Tool::WebSearch => web_search(lair.config().searxng.as_deref(), &lair.http, req(args, "query")?).await,
        Tool::FetchUrl => fetch_url(&lair.http, req(args, "url")?).await,
        Tool::ListSkills => {
            let catalog = lair.skills.read().unwrap();
            let lines: Vec<String> = catalog
                .list()
                .iter()
                .map(|s| {
                    format!(
                        "- {} ({}): {} [gatilhos: {}]{}",
                        s.id,
                        s.name,
                        s.description,
                        s.when.join(", "),
                        if s.enabled { "" } else { " (desligada)" }
                    )
                })
                .collect();
            Ok(lines.join("\n"))
        }
        Tool::SaveSkill => {
            let id = req(args, "id")?;
            let when = args
                .get("when")
                .and_then(Value::as_array)
                .map(|w| w.iter().filter_map(Value::as_str).map(str::to_owned).collect())
                .unwrap_or_default();
            let manifest = Manifest {
                name: req(args, "name")?.to_owned(),
                description: str_arg(args, "description").unwrap_or("").to_owned(),
                when,
                source: None,
            };
            lair.skills.write().unwrap().save(id, &manifest, req(args, "prompt")?)?;
            Ok(format!("skill {id} salva; ela roda quando a mensagem do usuário tiver um dos gatilhos"))
        }
        Tool::SearchConversations => {
            let hits = lair.store.search(req(args, "query")?)?;
            Ok(if hits.is_empty() {
                "nenhuma conversa encontrada".into()
            } else {
                serde_json::to_string_pretty(&hits)?
            })
        }
        Tool::ReadConversation => {
            let id = req(args, "id")?;
            let c = lair.store.get(id)?.with_context(|| format!("conversa {id} não existe"))?;
            Ok(format!("# {}\n{}", c.title, lair.store.transcript(id)?))
        }
    }?;
    Ok(shorten(out))
}

fn list_dir(ws: &Path, path: &str) -> Result<String> {
    let dir = jail::existing(ws, path)?;
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

fn read_file(ws: &Path, path: &str) -> Result<String> {
    let bytes = std::fs::read(jail::existing(ws, path)?)?;
    String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("{path} não é um arquivo de texto"))
}

fn write_file(ws: &Path, path: &str, content: &str) -> Result<String> {
    let target = jail::creatable(ws, path)?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&target, content)?;
    Ok(format!("{path} gravado ({} bytes)", content.len()))
}

fn edit_file(ws: &Path, path: &str, find: &str, replace: &str) -> Result<String> {
    let target = jail::existing(ws, path)?;
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

async fn run_command(ws: &Path, command: &str) -> Result<String> {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(ws)
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

async fn web_search(searxng: Option<&str>, http: &reqwest::Client, query: &str) -> Result<String> {
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
    let base = searxng.context("busca na web não configurada: defina o SearXNG nas configurações")?;
    let r: Results = http
        .get(format!("{}/search", base.trim_end_matches('/')))
        .query(&[("q", query), ("format", "json")])
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let hits: Vec<Value> = r
        .results
        .into_iter()
        .take(6)
        .map(|h| json!({ "title": h.title, "url": h.url, "snippet": h.content }))
        .collect();
    Ok(if hits.is_empty() { "nenhum resultado".into() } else { serde_json::to_string_pretty(&hits)? })
}

async fn fetch_url(http: &reqwest::Client, url: &str) -> Result<String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("só URLs http(s)");
    }
    // A cut page still reads well enough; the model only sees the first few thousand characters anyway.
    let (headers, body, _) = crate::net::get_capped(http, url, FETCH_LIMIT).await?;
    let html = headers.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.contains("html"));
    Ok(if html {
        html2text::from_read(body.as_slice(), 100).unwrap_or_default()
    } else {
        String::from_utf8_lossy(&body).into_owned()
    })
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

    fn workspace(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("lair-tools-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn files_are_written_edited_and_listed_inside_the_workspace() {
        let t = workspace("files");
        write_file(&t, "docs/a.md", "um dois um").unwrap();
        assert_eq!(list_dir(&t, ".").unwrap(), "docs/");
        assert_eq!(read_file(&t, "docs/a.md").unwrap(), "um dois um");
        assert!(edit_file(&t, "docs/a.md", "um", "três").unwrap_err().to_string().contains("2 vezes"));
        edit_file(&t, "docs/a.md", "dois", "2").unwrap();
        assert_eq!(read_file(&t, "docs/a.md").unwrap(), "um 2 um");
        assert!(write_file(&t, "../fora.md", "x").is_err());
        assert!(read_file(&t, "/etc/passwd").is_err());
    }

    #[tokio::test]
    async fn commands_run_in_the_workspace_and_report_exit_code_and_stderr() {
        let t = workspace("cmd");
        let out = run_command(&t, "pwd; echo oops >&2; exit 3").await.unwrap();
        assert!(out.starts_with("[código de saída 3]"));
        assert!(out.contains(t.file_name().unwrap().to_str().unwrap()));
        assert!(out.contains("[stderr]\noops"));
    }

    #[test]
    fn long_output_is_cut_with_a_note() {
        let s = shorten("é".repeat(MAX_OUTPUT + 10));
        assert!(s.ends_with(&format!("(cortado: {} caracteres no total)", MAX_OUTPUT + 10)));
        assert_eq!(s.chars().filter(|&c| c == 'é').count(), MAX_OUTPUT);
        assert_eq!(shorten("curto".into()), "curto");
    }
}
