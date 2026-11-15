//! The tools an agent may call, what each needs from the user, and how each runs.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::Config;
use crate::jail;
use crate::lair::Lair;
use crate::skills::{Catalog, Manifest};

/// Characters of tool output handed back to the model; small models have small contexts.
const MAX_OUTPUT: usize = 6000;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(60);
const FETCH_LIMIT: usize = 2 * 1024 * 1024;

/// What a tool's handler works with.
pub struct Ctx<'l> {
    pub lair: &'l Lair,
    /// Web results already numbered in this turn, so citations stay unique across searches.
    pub sources: &'l mut usize,
}

type Running<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

pub struct Spec {
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
    /// What the settings must allow for it to work, and what to say when they don't; a tool that can't
    /// work isn't offered to the model.
    ready: fn(&Config) -> Result<(), &'static str>,
    run: for<'a, 'l> fn(&'a mut Ctx<'l>, &'a Value) -> Running<'a>,
}

const NO_SEARXNG: &str = "web search is not set up: set SearXNG in the settings";

fn always(_: &Config) -> Result<(), &'static str> {
    Ok(())
}

pub static SPECS: &[Spec] = &[
    Spec {
        name: "list_dir",
        label: "List folder",
        subject: "path",
        usage: r#"lists a folder in the workspace. {"path": "."}"#,
        needs_approval: false,
        ready: always,
        run: |c, a| Box::pin(async move { list_dir(&c.lair.workspace, str_arg(a, "path").unwrap_or(".")) }),
    },
    Spec {
        name: "read_file",
        label: "Read file",
        subject: "path",
        usage: r#"reads a text file from the workspace. {"path": "notes.md"}"#,
        needs_approval: false,
        ready: always,
        run: |c, a| Box::pin(async move { read_file(&c.lair.workspace, req(a, "path")?) }),
    },
    Spec {
        name: "write_file",
        label: "Write file",
        subject: "path",
        usage: r#"creates or replaces a file in the workspace. {"path": "notes.md", "content": "..."}"#,
        needs_approval: true,
        ready: always,
        run: |c, a| Box::pin(async move { write_file(&c.lair.workspace, req(a, "path")?, req(a, "content")?) }),
    },
    Spec {
        name: "edit_file",
        label: "Edit file",
        subject: "path",
        usage: r#"replaces an exact piece of a file. {"path": "a.txt", "find": "current text", "replace": "new text"}"#,
        needs_approval: true,
        ready: always,
        run: |c, a| {
            Box::pin(async move {
                edit_file(&c.lair.workspace, req(a, "path")?, req(a, "find")?, str_arg(a, "replace").unwrap_or(""))
            })
        },
    },
    Spec {
        name: "delete_file",
        label: "Delete file",
        subject: "path",
        usage: r#"deletes a file from the workspace. {"path": "old.txt"}"#,
        needs_approval: true,
        ready: always,
        run: |c, a| {
            Box::pin(async move {
                std::fs::remove_file(jail::existing(&c.lair.workspace, req(a, "path")?)?)?;
                Ok("deleted".into())
            })
        },
    },
    Spec {
        name: "run_command",
        label: "Run command",
        subject: "command",
        usage: if cfg!(windows) {
            r#"runs a PowerShell command in the workspace folder (60 s at most). {"command": "Get-ChildItem"}"#
        } else {
            r#"runs a shell command in the workspace folder (60 s at most). {"command": "ls -la"}"#
        },
        needs_approval: true,
        ready: always,
        run: |c, a| Box::pin(async move { run_command(&c.lair.workspace, req(a, "command")?).await }),
    },
    Spec {
        name: "web_search",
        label: "Search the web",
        subject: "query",
        usage: r#"searches the web; each result comes with a number n to cite. {"query": "..."}"#,
        needs_approval: true,
        ready: |c| c.searxng.as_ref().map(drop).ok_or(NO_SEARXNG),
        run: |c, a| {
            Box::pin(async move {
                let searxng = c.lair.config().searxng.context(NO_SEARXNG)?;
                web_search(&searxng, &c.lair.http, req(a, "query")?, c.sources).await
            })
        },
    },
    Spec {
        name: "fetch_url",
        label: "Open page",
        subject: "url",
        usage: r#"downloads a page and returns its text. {"url": "https://..."}"#,
        needs_approval: true,
        ready: always,
        run: |c, a| Box::pin(async move { fetch_url(&c.lair.http, req(a, "url")?).await }),
    },
    Spec {
        name: "search_conversations",
        label: "Search chats",
        subject: "query",
        usage: r#"searches the user's saved chats. {"query": "..."}"#,
        needs_approval: false,
        ready: always,
        run: |c, a| {
            Box::pin(async move {
                let hits = c.lair.store.search(req(a, "query")?)?;
                Ok(if hits.is_empty() { "no chats found".into() } else { serde_json::to_string_pretty(&hits)? })
            })
        },
    },
    Spec {
        name: "search_memory",
        label: "Search memory",
        subject: "query",
        usage: r#"finds passages of the user's earlier chats and workspace files related to a question, by meaning. {"query": "..."}"#,
        needs_approval: false,
        ready: |c| if c.memory() { Ok(()) } else { Err(crate::memory::OFF) },
        run: |c, a| {
            Box::pin(async move {
                let hits = crate::memory::recall(c.lair, req(a, "query")?, None, 6).await?;
                Ok(if hits.is_empty() { "nothing related found".into() } else { serde_json::to_string_pretty(&hits)? })
            })
        },
    },
    Spec {
        name: "read_conversation",
        label: "Read chat",
        subject: "id",
        usage: r#"reads a saved chat by id. {"id": "..."}"#,
        needs_approval: false,
        ready: always,
        run: |c, a| {
            Box::pin(async move {
                let id = req(a, "id")?;
                let conv = c.lair.store.get(id)?.with_context(|| format!("chat {id} does not exist"))?;
                Ok(format!("# {}\n{}", conv.title, c.lair.store.transcript(id)?))
            })
        },
    },
    Spec {
        name: "update_plan",
        label: "Plan",
        subject: "",
        usage: r#"writes down the steps of a task with several steps, and marks them done as you go; call it first and after each step. {"items": [{"text": "read the config", "done": true}, {"text": "fix the bug", "done": false}]}"#,
        needs_approval: false,
        ready: always,
        run: |_, a| Box::pin(async move { update_plan(a) }),
    },
    Spec {
        name: "list_skills",
        label: "List skills",
        subject: "",
        usage: "lists the skills and their triggers. {}",
        needs_approval: false,
        ready: always,
        run: |c, _| Box::pin(async move { Ok(list_skills(&c.lair.skills.read().unwrap())) }),
    },
    Spec {
        name: "save_skill",
        label: "Save skill",
        subject: "id",
        usage: r#"creates or replaces a skill (extra instructions triggered by words). {"id": "pirate", "name": "Pirate", "description": "...", "when": ["pirate"], "prompt": "Answer like a pirate."}"#,
        needs_approval: true,
        ready: always,
        run: |c, a| Box::pin(async move { save_skill(&mut c.lair.skills.write().unwrap(), a) }),
    },
];

pub fn spec(name: &str) -> Option<&'static Spec> {
    SPECS.iter().find(|s| s.name == name)
}

/// The instructions that turn a chat model into an agent, offering the tools these settings let work.
pub fn system_prompt(config: &Config) -> String {
    let mut out = format!(
        "{}Paths are relative to the working folder (workspace).\n\
         When you use information from a web search, cite the source with its number in brackets right \
         after the passage, like this: [1]. Only cite numbers the search returned.\n\nTools:\n",
        super::PROTOCOL
    );
    for s in SPECS.iter().filter(|s| (s.ready)(config).is_ok()) {
        out.push_str(&format!("- {}: {}\n", s.name, s.usage));
    }
    out
}

impl Spec {
    /// Whether these settings let it work, and if not, why.
    pub fn ready(&self, config: &Config) -> Result<(), &'static str> {
        (self.ready)(config)
    }

    /// Runs one call; the text is what the model reads back, already shortened.
    pub async fn run(&self, c: &mut Ctx<'_>, args: &Value) -> Result<String> {
        Ok(shorten((self.run)(c, args).await?))
    }
}

/// The plan as a checklist; the model reads it back, and it shows in the step's card.
fn update_plan(args: &Value) -> Result<String> {
    let items = args.get("items").and_then(Value::as_array).context("`items` must be a list of steps")?;
    let lines: Vec<String> = items
        .iter()
        .map(|i| {
            let mark = if i.get("done").and_then(Value::as_bool).unwrap_or(false) { 'x' } else { ' ' };
            format!("- [{mark}] {}", i.get("text").and_then(Value::as_str).unwrap_or_default())
        })
        .collect();
    Ok(if lines.is_empty() { "The plan is empty.".into() } else { lines.join("\n") })
}

fn list_skills(catalog: &Catalog) -> String {
    let lines: Vec<String> = catalog
        .list()
        .iter()
        .map(|s| {
            let off = if s.enabled { "" } else { " (disabled)" };
            format!("- {} ({}): {} [triggers: {}]{off}", s.id, s.name, s.description, s.when.join(", "))
        })
        .collect();
    lines.join("\n")
}

fn save_skill(catalog: &mut Catalog, args: &Value) -> Result<String> {
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
    catalog.save(id, &manifest, req(args, "prompt")?)?;
    Ok(format!("skill {id} saved; it runs when the user's message has one of its triggers"))
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
    Ok(if names.is_empty() { "(empty folder)".into() } else { names.join("\n") })
}

fn read_file(ws: &Path, path: &str) -> Result<String> {
    let bytes = std::fs::read(jail::existing(ws, path)?)?;
    String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("{path} is not a text file"))
}

fn write_file(ws: &Path, path: &str, content: &str) -> Result<String> {
    let target = jail::creatable(ws, path)?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&target, content)?;
    Ok(format!("{path} written ({} bytes)", content.len()))
}

fn edit_file(ws: &Path, path: &str, find: &str, replace: &str) -> Result<String> {
    let target = jail::existing(ws, path)?;
    let text = std::fs::read_to_string(&target)?;
    match text.matches(find).count() {
        0 => bail!("the text does not appear in {path}"),
        1 => {
            std::fs::write(&target, text.replacen(find, replace, 1))?;
            Ok(format!("{path} edited"))
        }
        n => bail!("the text appears {n} times in {path}; include more context so it is unique"),
    }
}

async fn run_command(ws: &Path, command: &str) -> Result<String> {
    let shell = if cfg!(windows) { "powershell" } else { "sh" };
    let flags: &[&str] = if cfg!(windows) { &["-NoProfile", "-NonInteractive", "-Command"] } else { &["-c"] };
    let child = tokio::process::Command::new(shell)
        .args(flags)
        .arg(command)
        .current_dir(ws)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let out = tokio::time::timeout(COMMAND_TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| anyhow::anyhow!("the command ran past {} s and was stopped", COMMAND_TIMEOUT.as_secs()))??;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    let err = String::from_utf8_lossy(&out.stderr);
    if !err.trim().is_empty() {
        text.push_str(&format!("\n[stderr]\n{err}"));
    }
    Ok(format!("[exit code {}]\n{}", out.status.code().unwrap_or(-1), text.trim_end()))
}

async fn web_search(searxng: &str, http: &reqwest::Client, query: &str, sources: &mut usize) -> Result<String> {
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
    let r: Results = http
        .get(format!("{}/search", searxng.trim_end_matches('/')))
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
        .map(|h| {
            *sources += 1;
            json!({ "n": *sources, "title": h.title, "url": h.url, "snippet": h.content })
        })
        .collect();
    Ok(if hits.is_empty() { "no results".into() } else { serde_json::to_string_pretty(&hits)? })
}

/// The page's `<title>`, so the UI and the model know what was read.
fn html_title(html: &[u8]) -> Option<String> {
    let head = String::from_utf8_lossy(&html[..html.len().min(64 * 1024)]);
    let lower = head.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let start = start + lower[start..].find('>')? + 1;
    let end = start + lower[start..].find("</title")?;
    let title = html2text::from_read(head[start..end].as_bytes(), usize::MAX).ok()?;
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    (!title.is_empty()).then_some(title)
}

async fn fetch_url(http: &reqwest::Client, url: &str) -> Result<String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("only http(s) URLs");
    }
    // A cut page still reads well enough; the model only sees the first few thousand characters anyway.
    let (headers, body, _) = crate::net::get_capped(http, url, FETCH_LIMIT).await?;
    let html = headers.get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.contains("html"));
    Ok(if html {
        let text = html2text::from_read(body.as_slice(), 100).unwrap_or_default();
        match html_title(&body) {
            Some(title) => format!("Title: {title}\n\n{text}"),
            None => text,
        }
    } else {
        String::from_utf8_lossy(&body).into_owned()
    })
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn req<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    str_arg(args, key).with_context(|| format!("missing argument \"{key}\" (text)"))
}

fn shorten(mut s: String) -> String {
    if let Some((i, _)) = s.char_indices().nth(MAX_OUTPUT) {
        let total = s.chars().count();
        s.truncate(i);
        s.push_str(&format!("\n… (cut: {total} characters in total)"));
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
        assert!(edit_file(&t, "docs/a.md", "um", "três").unwrap_err().to_string().contains("2 times"));
        edit_file(&t, "docs/a.md", "dois", "2").unwrap();
        assert_eq!(read_file(&t, "docs/a.md").unwrap(), "um 2 um");
        assert!(write_file(&t, "../fora.md", "x").is_err());
        assert!(read_file(&t, "/etc/passwd").is_err());
    }

    #[test]
    fn only_tools_the_settings_let_work_are_offered() {
        let offered = |c: &Config| system_prompt(c).contains("- web_search:");
        assert!(!offered(&Config::default()), "no SearXNG, no web search");
        assert!(offered(&Config { searxng: Some("http://s".into()), ..Default::default() }));
        let memory = |c: &Config| system_prompt(c).contains("- search_memory:");
        assert!(memory(&Config::default()) && !memory(&Config { memory: Some(false), ..Default::default() }));
        assert!(SPECS.iter().all(|s| spec(s.name).is_some_and(|f| std::ptr::eq(f, s))), "names are unique");
    }

    #[test]
    fn page_titles_are_read_and_unescaped() {
        assert_eq!(
            html_title(b"<html><HEAD><title lang=pt>\n Ol&aacute;  mundo </title>").as_deref(),
            Some("Olá mundo")
        );
        assert_eq!(html_title(b"<html><title></title>"), None);
        assert_eq!(html_title(b"<p>sem titulo</p>"), None);
    }

    #[tokio::test]
    async fn commands_run_in_the_workspace_and_report_exit_code_and_stderr() {
        let t = workspace("cmd");
        let script = match cfg!(windows) {
            true => "(Get-Location).Path; [Console]::Error.WriteLine('oops'); exit 3",
            false => "pwd; echo oops >&2; exit 3",
        };
        let out = run_command(&t, script).await.unwrap();
        assert!(out.starts_with("[exit code 3]"));
        assert!(out.contains(t.file_name().unwrap().to_str().unwrap()));
        assert!(out.contains("[stderr]\noops"));
    }

    #[test]
    fn the_plan_comes_back_as_a_checklist() {
        let plan = json!({ "items": [{ "text": "read", "done": true }, { "text": "fix" }] });
        assert_eq!(update_plan(&plan).unwrap(), "- [x] read\n- [ ] fix");
        assert!(update_plan(&json!({})).is_err());
    }

    #[test]
    fn long_output_is_cut_with_a_note() {
        let s = shorten("é".repeat(MAX_OUTPUT + 10));
        assert_eq!(s.chars().filter(|&c| c == 'é').count(), MAX_OUTPUT, "cut by characters, not bytes");
        assert!(s.contains(&(MAX_OUTPUT + 10).to_string()), "the model learns how long it was: {s}");
        assert_eq!(shorten("curto".into()), "curto");
    }
}
