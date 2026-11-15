//! Memory's index: pieces of chats and workspace files, found by their words and, once an embedding
//! model has turned them into vectors, by meaning.

use anyhow::Result;
use rusqlite::{OptionalExtension, params};

use super::Store;

/// What a piece was cut from. The schema (`super::SCHEMA`) deletes a chat's pieces with the chat by name.
pub const CHAT: &str = "chat";
pub const FILE: &str = "file";

/// A piece of a chat or of a workspace file, as memory keeps it.
#[derive(Debug)]
pub struct Chunk {
    /// `chat` or `file`.
    pub source: String,
    /// The conversation's id, or the file's path in the workspace.
    pub of: String,
    pub text: String,
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dot(blob: &[u8], q: &[f32]) -> f32 {
    blob.as_chunks::<4>().0.iter().zip(q).map(|(b, x)| f32::from_le_bytes(*b) * x).sum()
}

impl Store {
    /// Adds pieces of `of` to memory; with `replace`, in place of the ones it had (a file that changed).
    pub fn add_chunks(&self, source: &str, of: &str, stamp: i64, texts: &[String], replace: bool) -> Result<()> {
        let mut db = self.db.lock().unwrap();
        let tx = db.transaction()?;
        if replace {
            tx.execute("DELETE FROM chunks WHERE source = ?1 AND ref = ?2", params![source, of])?;
        }
        for text in texts {
            tx.execute(
                "INSERT INTO chunks (source, ref, stamp, text) VALUES (?1, ?2, ?3, ?4)",
                params![source, of, stamp, text],
            )?;
        }
        Ok(tx.commit()?)
    }

    pub fn drop_chunks(&self, source: &str, of: &str) -> Result<()> {
        self.db.lock().unwrap().execute("DELETE FROM chunks WHERE source = ?1 AND ref = ?2", params![source, of])?;
        Ok(())
    }

    /// What memory holds of `source`: each ref with its latest stamp.
    pub fn chunk_refs(&self, source: &str) -> Result<Vec<(String, i64)>> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT ref, max(stamp) FROM chunks WHERE source = ?1 GROUP BY ref")?;
        let rows = stmt.query_map([source], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// Pieces `model` has not embedded yet: `(id, text)`.
    pub fn pending_chunks(&self, model: &str, limit: usize) -> Result<Vec<(i64, String)>> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT id, text FROM chunks WHERE model IS NOT ?1 AND text != '' LIMIT ?2")?;
        let rows = stmt
            .query_map(params![model, limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn set_vectors(&self, model: &str, vectors: &[(i64, Vec<f32>)]) -> Result<()> {
        let mut db = self.db.lock().unwrap();
        let tx = db.transaction()?;
        for (id, v) in vectors {
            tx.execute("UPDATE chunks SET model = ?1, vec = ?2 WHERE id = ?3", params![model, to_blob(v), id])?;
        }
        Ok(tx.commit()?)
    }

    /// The pieces that share words with `q`, best match first.
    pub fn chunks_by_words(&self, q: &str, limit: usize) -> Result<Vec<i64>> {
        // Any of the words, not all: the question is a sentence, not a list of keywords.
        let words: Vec<String> = q
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.chars().count() > 2)
            .take(24)
            .map(|w| format!("\"{w}\""))
            .collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY rank LIMIT ?2")?;
        let rows = stmt
            .query_map(params![words.join(" OR "), limit as i64], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// The pieces closest in meaning to `q` (a vector from `model`, of length 1), each with its
    /// cosine, closest first; those under `min` are left out.
    // shortcut: reads every vector on each search. Fine for a person's chats and files (100k pieces is
    // ~half a second); store int8 vectors or add an index (sqlite-vec) past that.
    pub fn chunks_by_meaning(&self, model: &str, q: &[f32], limit: usize, min: f32) -> Result<Vec<(i64, f32)>> {
        let db = self.db.lock().unwrap();
        let mut stmt = db.prepare("SELECT id, vec FROM chunks WHERE model = ?1 AND vec IS NOT NULL")?;
        let mut rows = stmt.query([model])?;
        let mut found: Vec<(i64, f32)> = Vec::new();
        while let Some(r) = rows.next()? {
            let cos = dot(r.get_ref(1)?.as_blob()?, q);
            if cos >= min {
                found.push((r.get(0)?, cos));
            }
        }
        found.sort_by(|a, b| b.1.total_cmp(&a.1));
        found.truncate(limit);
        Ok(found)
    }

    pub fn chunk(&self, id: i64) -> Result<Option<Chunk>> {
        let db = self.db.lock().unwrap();
        Ok(db
            .query_row("SELECT source, ref, text FROM chunks WHERE id = ?1", [id], |r| {
                Ok(Chunk { source: r.get(0)?, of: r.get(1)?, text: r.get(2)? })
            })
            .optional()?)
    }

    /// How many pieces memory holds, and how many of them `model` has embedded.
    pub fn chunk_counts(&self, model: Option<&str>) -> Result<(u64, u64)> {
        let db = self.db.lock().unwrap();
        Ok(db.query_row("SELECT count(*), count(CASE WHEN model = ?1 THEN 1 END) FROM chunks", [model], |r| {
            Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64))
        })?)
    }

    pub fn clear_chunks(&self) -> Result<()> {
        self.db.lock().unwrap().execute("DELETE FROM chunks", [])?;
        Ok(())
    }
}
