//! Conversations and their messages, in SQLite, with full-text search.

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::Reasoning;

#[derive(Clone, Debug, Serialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub model: Option<String>,
    pub effort: Option<Reasoning>,
    pub agent: bool,
    pub created: i64,
    pub updated: i64,
}

/// A conversation another message points at, kept with its title so the reference reads well later.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ref {
    pub id: String,
    pub title: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Message {
    pub id: i64,
    /// `user`, `assistant`, or `tool` (one agent step: the call and its result).
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
    /// The skill that shaped this reply.
    pub skill: Option<Named>,
    pub tool: Option<ToolStep>,
    pub refs: Vec<Ref>,
    pub attachments: Vec<Attachment>,
    pub created: i64,
}

/// Something named, kept with a message by id and display name: the skill a reply used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Named {
    pub id: String,
    pub name: String,
}

/// One agent step: the call as the model wrote it (`raw`), what lair understood, and the result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolStep {
    pub id: String,
    pub name: String,
    /// What the UI calls the tool; steps saved before labels show their name.
    #[serde(default)]
    pub label: String,
    /// The argument that says what the step is about: a path, a command, a query…
    #[serde(default)]
    pub subject: String,
    pub arguments: Value,
    pub raw: String,
    pub status: ToolStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolStatus {
    Ok,
    Error,
    Denied,
}

/// A file attached to a user message: its upload id, the name to show, and how much text it gave.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attachment {
    /// Rows from before ids stored the workspace path (`anexos/<id>`).
    #[serde(alias = "path", deserialize_with = "file_name")]
    pub id: String,
    pub name: String,
    pub chars: usize,
    #[serde(default)]
    pub image: bool,
}

fn file_name<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let s = String::deserialize(d)?;
    Ok(s.rsplit('/').next().unwrap_or_default().to_owned())
}

#[derive(Default, Deserialize)]
pub struct Settings {
    pub title: Option<String>,
    pub model: Option<String>,
    pub effort: Option<Reasoning>,
    pub agent: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct Hit {
    pub id: String,
    pub title: String,
    pub snippet: String,
}

pub struct Store {
    db: Mutex<Connection>,
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// A timestamp later than every stored one, so "most recently updated" never ties within a millisecond.
fn tick(db: &Connection) -> rusqlite::Result<i64> {
    let latest: i64 = db.query_row("SELECT coalesce(max(updated), 0) FROM conversations", [], |r| r.get(0))?;
    Ok(now().max(latest + 1))
}

/// The `config` row holding the settings the UI saved, as one JSON document.
const CONFIG_KEY: &str = "runtime";

/// Conversations made before attachments existed have no `attachments` column in `messages`.
fn migrate(db: &Connection) {
    let has = db
        .query_row("SELECT count(*) FROM pragma_table_info('messages') WHERE name = 'attachments'", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap_or(1);
    if has == 0 {
        db.execute_batch("ALTER TABLE messages ADD COLUMN attachments TEXT").expect("adding the attachments column");
    }
}

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    model TEXT,
    effort TEXT,
    agent INTEGER NOT NULL DEFAULT 0,
    created INTEGER NOT NULL,
    updated INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conv TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    reasoning TEXT,
    skill TEXT,
    tool TEXT,
    refs TEXT,
    attachments TEXT,
    created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conv ON messages(conv, id);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
";

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        db.execute_batch(SCHEMA)?;
        migrate(&db);
        Ok(Self { db: Mutex::new(db) })
    }

    #[cfg(test)]
    fn memory() -> Self {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(SCHEMA).unwrap();
        migrate(&db);
        Self { db: Mutex::new(db) }
    }

    /// The saved settings document; `{}` before anything was saved.
    pub fn config(&self) -> Result<Value> {
        let db = self.db.lock().unwrap();
        let raw = db
            .query_row("SELECT value FROM config WHERE key = ?1", [CONFIG_KEY], |r| r.get::<_, String>(0))
            .optional()?;
        Ok(raw.and_then(|v| serde_json::from_str(&v).ok()).unwrap_or_else(|| Value::Object(Default::default())))
    }

    pub fn save_config(&self, config: &Value) -> Result<()> {
        let db = self.db.lock().unwrap();
        db.execute(
            "INSERT INTO config (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![CONFIG_KEY, config.to_string()],
        )?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<Conversation>> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT * FROM conversations ORDER BY updated DESC")?;
        let rows = stmt.query_map([], conversation)?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn create(&self, s: Settings) -> Result<Conversation> {
        let db = self.db.lock().unwrap();
        let id: String = db.query_row("SELECT lower(hex(randomblob(8)))", [], |r| r.get(0))?;
        let t = tick(&db)?;
        db.execute(
            "INSERT INTO conversations (id, title, model, effort, agent, created, updated) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            params![id, s.title.unwrap_or_default(), s.model, s.effort.map(|e| e.to_string()), s.agent.unwrap_or(false), t],
        )?;
        Ok(db.query_row("SELECT * FROM conversations WHERE id = ?1", [&id], conversation)?)
    }

    pub fn get(&self, id: &str) -> Result<Option<Conversation>> {
        let db = self.db.lock().unwrap();
        Ok(db.query_row("SELECT * FROM conversations WHERE id = ?1", [id], conversation).optional()?)
    }

    pub fn messages(&self, id: &str) -> Result<Vec<Message>> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT * FROM messages WHERE conv = ?1 ORDER BY id")?;
        let rows = stmt.query_map([id], message)?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// The conversation as plain text for a model to read: who said what, without the tool steps.
    pub fn transcript(&self, id: &str) -> Result<String> {
        Ok(self
            .messages(id)?
            .iter()
            .filter(|m| m.role != "tool")
            .map(|m| format!("{}: {}\n", if m.role == "user" { "Usuário" } else { "Assistente" }, m.content))
            .collect())
    }

    /// Changes the given settings; returns `None` when the conversation does not exist.
    pub fn update(&self, id: &str, s: Settings) -> Result<Option<Conversation>> {
        let db = self.db.lock().unwrap();
        db.execute(
            "UPDATE conversations SET title = coalesce(?2, title), model = coalesce(?3, model),
             effort = coalesce(?4, effort), agent = coalesce(?5, agent) WHERE id = ?1",
            params![id, s.title, s.model, s.effort.map(|e| e.to_string()), s.agent],
        )?;
        Ok(db.query_row("SELECT * FROM conversations WHERE id = ?1", [id], conversation).optional()?)
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.db.lock().unwrap().execute("DELETE FROM conversations WHERE id = ?1", [id])? > 0)
    }

    /// Adds a message and bumps the conversation; the first user message also names an untitled one.
    pub fn append(&self, conv: &str, m: &Message) -> Result<i64> {
        let db = self.db.lock().unwrap();
        let t = tick(&db)?;
        let skill = m.skill.as_ref().map(serde_json::to_string).transpose()?;
        let tool = m.tool.as_ref().map(serde_json::to_string).transpose()?;
        let refs = (!m.refs.is_empty()).then(|| serde_json::to_string(&m.refs)).transpose()?;
        let attachments = (!m.attachments.is_empty()).then(|| serde_json::to_string(&m.attachments)).transpose()?;
        db.execute(
            "INSERT INTO messages (conv, role, content, reasoning, skill, tool, refs, attachments, created)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![conv, m.role, m.content, m.reasoning, skill, tool, refs, attachments, t],
        )?;
        let id = db.last_insert_rowid();
        let title = if m.role == "user" { title_from(&m.content) } else { String::new() };
        db.execute(
            "UPDATE conversations SET updated = ?2, title = CASE WHEN title = '' THEN ?3 ELSE title END WHERE id = ?1",
            params![conv, t, title],
        )?;
        Ok(id)
    }

    /// Conversations whose title or messages contain every word of `q` (prefix match), newest first.
    pub fn search(&self, q: &str) -> Result<Vec<Hit>> {
        let words: Vec<String> = q
            .split_whitespace()
            .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
            .filter(|w| !w.is_empty())
            .collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let fts = words.iter().map(|w| format!("\"{w}\"*")).collect::<Vec<_>>().join(" ");
        let like = format!("%{}%", q.trim());
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare(
            "SELECT c.id, c.title, coalesce(
                 (SELECT snippet(messages_fts, 0, '', '', '…', 12) FROM messages_fts
                  JOIN messages m ON m.id = messages_fts.rowid
                  WHERE messages_fts MATCH ?1 AND m.conv = c.id LIMIT 1), '') AS snippet
             FROM conversations c
             WHERE c.title LIKE ?2 OR c.id IN (
                 SELECT m.conv FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid
                 WHERE messages_fts MATCH ?1)
             ORDER BY c.updated DESC LIMIT 20",
        )?;
        let rows = stmt
            .query_map(params![fts, like], |r| Ok(Hit { id: r.get(0)?, title: r.get(1)?, snippet: r.get(2)? }))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }
}

fn title_from(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    match line.char_indices().nth(60) {
        Some((i, _)) => format!("{}…", line[..i].trim_end()),
        None => line.to_owned(),
    }
}

fn conversation(r: &Row) -> rusqlite::Result<Conversation> {
    Ok(Conversation {
        id: r.get("id")?,
        title: r.get("title")?,
        model: r.get("model")?,
        effort: r.get::<_, Option<String>>("effort")?.and_then(|e| e.parse().ok()),
        agent: r.get("agent")?,
        created: r.get("created")?,
        updated: r.get("updated")?,
    })
}

fn message(r: &Row) -> rusqlite::Result<Message> {
    fn json<T: serde::de::DeserializeOwned>(r: &Row, col: &str) -> rusqlite::Result<Option<T>> {
        Ok(r.get::<_, Option<String>>(col)?.and_then(|s| serde_json::from_str(&s).ok()))
    }
    Ok(Message {
        id: r.get("id")?,
        role: r.get("role")?,
        content: r.get("content")?,
        reasoning: r.get("reasoning")?,
        skill: json(r, "skill")?,
        tool: json(r, "tool")?,
        refs: json(r, "refs")?.unwrap_or_default(),
        attachments: json(r, "attachments")?.unwrap_or_default(),
        created: r.get("created")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> Message {
        Message { role: "user".into(), content: text.into(), ..Default::default() }
    }

    #[test]
    fn conversations_keep_order_title_and_settings() {
        let s = Store::memory();
        let a = s.create(Settings::default()).unwrap();
        let b = s.create(Settings { title: Some("Fixo".into()), ..Default::default() }).unwrap();
        s.append(&a.id, &user("\n  Como faço pão de fermentação natural em casa, passo a passo, sem batedeira?"))
            .unwrap();
        s.append(&b.id, &user("isto não renomeia")).unwrap();

        let list = s.list().unwrap();
        assert_eq!(list[0].id, b.id, "most recently updated first");
        assert_eq!(list[0].title, "Fixo");
        assert_eq!(list[1].title, "Como faço pão de fermentação natural em casa, passo a passo,…");

        let a2 = s
            .update(&a.id, Settings { model: Some("m".into()), agent: Some(true), ..Default::default() })
            .unwrap()
            .unwrap();
        assert_eq!((a2.model.as_deref(), a2.agent, a2.title.as_str()), (Some("m"), true, list[1].title.as_str()));
        assert!(s.update("nope", Settings::default()).unwrap().is_none());
    }

    #[test]
    fn messages_round_trip_and_delete_cascades() {
        let s = Store::memory();
        let c = s.create(Settings::default()).unwrap();
        let reply = Message {
            role: "assistant".into(),
            content: "Oi".into(),
            reasoning: Some("pensei".into()),
            skill: Some(Named { id: "3d".into(), name: "3D".into() }),
            ..Default::default()
        };
        let mut question = user("oi");
        question.refs = vec![Ref { id: "x".into(), title: "Outra".into() }];
        s.append(&c.id, &question).unwrap();
        s.append(&c.id, &reply).unwrap();

        let msgs = s.messages(&c.id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].refs[0].title, "Outra");
        assert_eq!(msgs[1].reasoning.as_deref(), Some("pensei"));
        assert_eq!(msgs[1].skill.as_ref().unwrap().id, "3d");

        assert!(s.delete(&c.id).unwrap());
        assert!(s.messages(&c.id).unwrap().is_empty());
        assert!(s.search("oi").unwrap().is_empty(), "search index follows deletes");
    }

    #[test]
    fn search_matches_words_by_prefix_and_titles() {
        let s = Store::memory();
        let bread = s.create(Settings::default()).unwrap();
        s.append(&bread.id, &user("receita de fermentação natural")).unwrap();
        let other = s.create(Settings { title: Some("Viagem ao Japão".into()), ..Default::default() }).unwrap();
        s.append(&other.id, &user("roteiro de 10 dias")).unwrap();

        let hits = s.search("ferment").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, bread.id);
        assert!(hits[0].snippet.contains("fermentação"));
        assert_eq!(s.search("japão").unwrap()[0].id, other.id);
        assert!(s.search("\" OR *").unwrap().is_empty(), "FTS syntax in the query is neutralized");
    }

    #[test]
    fn config_round_trips_and_attachments_replay() {
        let s = Store::memory();
        assert_eq!(s.config().unwrap(), serde_json::json!({}));
        s.save_config(&serde_json::json!({ "peers": ["doom:8080"] })).unwrap();
        assert_eq!(s.config().unwrap()["peers"][0], "doom:8080");

        let c = s.create(Settings::default()).unwrap();
        s.append(
            &c.id,
            &Message {
                role: "user".into(),
                content: "o que diz este pdf?".into(),
                attachments: vec![Attachment {
                    id: "1a-contrato.pdf".into(),
                    name: "contrato.pdf".into(),
                    chars: 400,
                    image: false,
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let stored = s.messages(&c.id).unwrap();
        assert_eq!(stored[0].attachments[0].id, "1a-contrato.pdf");
        assert_eq!(stored[0].attachments.len(), 1);
        s.append(&c.id, &user("sem anexo")).unwrap();
        assert!(s.messages(&c.id).unwrap()[1].attachments.is_empty());

        let old: Attachment = serde_json::from_str(r#"{"name":"a.pdf","path":"anexos/1a-a.pdf","chars":3}"#).unwrap();
        assert_eq!((old.id.as_str(), old.image), ("1a-a.pdf", false), "rows saved with a path still open");
    }
}
