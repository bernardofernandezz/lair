//! Skills: prompt packages added to a request when the user's message asks for one.
//!
//! A skill is a folder with `skill.json` (`name`, `when`) and `prompt.md`. Built-ins are embedded in
//! the binary; `--skills-dir` adds more and replaces built-ins with the same id. They are prompts,
//! not executable plugins. A `{{assets}}` slot in the prompt receives the local 3D asset list.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::engine::Msg;

const BUILTINS: &[(&str, &str, &str)] =
    &[("3d", include_str!("../skills/3d/skill.json"), include_str!("../skills/3d/prompt.md"))];

pub struct Skill {
    pub id: String,
    pub name: String,
    /// Trigger phrases, normalized with [`normalize`].
    when: Vec<String>,
    prompt: String,
}

#[derive(Deserialize)]
struct Manifest {
    name: String,
    #[serde(default)]
    when: Vec<String>,
}

#[derive(Default)]
pub struct Catalog {
    skills: Vec<Skill>,
}

impl Catalog {
    pub fn load(dir: Option<&Path>) -> Self {
        let mut skills: Vec<Skill> = BUILTINS
            .iter()
            .map(|(id, manifest, prompt)| Skill::parse(id, manifest, prompt).expect("built-in skills are valid"))
            .collect();
        for skill in dir.map(load_dir).unwrap_or_default() {
            match skills.iter().position(|s| s.id == skill.id) {
                Some(i) => {
                    eprintln!("skills: {} replaces the built-in", skill.id);
                    skills[i] = skill;
                }
                None => skills.push(skill),
            }
        }
        Self { skills }
    }

    /// The skill the last user message asks for, if any, and the messages with its instructions added.
    pub fn apply(&self, msgs: &[Msg], assets: &[String]) -> (Option<&Skill>, Vec<Msg>) {
        let Some(skill) = self.trigger(msgs) else {
            return (None, msgs.to_vec());
        };
        let prompt = skill.prompt.replace("{{assets}}", &assets_section(assets));
        (Some(skill), with_system(msgs, prompt.trim_end()))
    }

    fn trigger(&self, msgs: &[Msg]) -> Option<&Skill> {
        let text = normalize(&msgs.iter().rev().find(|m| m.role == "user")?.content);
        self.skills.iter().find(|s| s.when.iter().any(|w| contains_word(&text, w)))
    }
}

impl Skill {
    fn parse(id: &str, manifest: &str, prompt: &str) -> Result<Self> {
        if !valid_id(id) {
            bail!("invalid skill id {id:?} (use a-z, 0-9, - or _)");
        }
        let m: Manifest = serde_json::from_str(manifest).context("invalid skill.json")?;
        let name = m.name.trim().to_owned();
        if name.is_empty() {
            bail!("name must not be empty");
        }
        if prompt.trim().is_empty() {
            bail!("prompt.md is empty");
        }
        let mut when = Vec::new();
        for w in m.when.iter().map(|w| normalize(w.trim())) {
            if !w.is_empty() && !when.contains(&w) {
                when.push(w);
            }
        }
        if when.is_empty() {
            bail!("`when` needs at least one trigger phrase");
        }
        Ok(Self { id: id.to_owned(), name, when, prompt: prompt.to_owned() })
    }
}

fn valid_id(id: &str) -> bool {
    let mut chars = id.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn load_dir(dir: &Path) -> Vec<Skill> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skills: cannot read {}: {e}", dir.display());
            return Vec::new();
        }
    };
    let mut skills: Vec<Skill> = Vec::new();
    for path in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        let id = path.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
        let parsed = std::fs::read_to_string(path.join("skill.json"))
            .and_then(|m| Ok((m, std::fs::read_to_string(path.join("prompt.md"))?)))
            .map_err(anyhow::Error::from)
            .and_then(|(manifest, prompt)| Skill::parse(&id, &manifest, &prompt));
        match parsed {
            Ok(skill) if skills.iter().any(|s| s.id == skill.id) => {
                eprintln!("skills: skipping {}: another folder is also called {id}", path.display());
            }
            Ok(skill) => skills.push(skill),
            Err(e) => eprintln!("skills: skipping {}: {e:#}", path.display()),
        }
    }
    skills.sort_by(|a, b| a.id.cmp(&b.id));
    skills
}

const LISTED_ASSETS: usize = 40;

fn assets_section(assets: &[String]) -> String {
    if assets.is_empty() {
        return "Nenhum arquivo 3D local disponível: use apenas primitivas.".to_owned();
    }
    let mut out = String::from("Arquivos 3D locais disponíveis (use o id no campo `asset`):\n");
    for id in assets.iter().take(LISTED_ASSETS) {
        let _ = writeln!(out, "- {id}");
    }
    if assets.len() > LISTED_ASSETS {
        let _ = writeln!(out, "… e mais {}", assets.len() - LISTED_ASSETS);
    }
    out
}

/// Appends `extra` to the first system message, or adds one at the start.
fn with_system(msgs: &[Msg], extra: &str) -> Vec<Msg> {
    let mut out = msgs.to_vec();
    match out.iter_mut().find(|m| m.role == "system") {
        Some(sys) if sys.content.is_empty() => sys.content = extra.into(),
        Some(sys) => sys.content = format!("{}\n\n{extra}", sys.content),
        None => out.insert(0, Msg { role: "system".into(), content: extra.into() }),
    }
    out
}

/// Lowercase and remove the accents used by Portuguese triggers.
fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ã' | 'ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            'ñ' => 'n',
            'ý' | 'ÿ' => 'y',
            _ => c,
        })
        .collect()
}

/// Is `needle` inside `hay` with a non-alphanumeric character (or nothing) on both sides?
fn contains_word(hay: &str, needle: &str) -> bool {
    hay.match_indices(needle).any(|(a, m)| {
        let before = hay[..a].chars().next_back();
        let after = hay[a + m.len()..].chars().next();
        before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lair-skills-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn user(text: &str) -> Vec<Msg> {
        vec![Msg { role: "user".into(), content: text.into() }]
    }

    fn triggered(c: &Catalog, text: &str) -> Option<String> {
        c.apply(&user(text), &[]).0.map(|s| s.id.clone())
    }

    #[test]
    fn the_3d_skill_triggers_only_on_explicit_requests() {
        let c = Catalog::load(None);
        assert_eq!(triggered(&c, "mostre um sistema solar em 3D").as_deref(), Some("3d"));
        assert_eq!(triggered(&c, "Desenhe uma molécula em três dimensões").as_deref(), Some("3d"));
        assert_eq!(triggered(&c, "qual o maior planeta?"), None);
        assert_eq!(triggered(&c, "a fórmula da molécula de água"), None);
        assert_eq!(triggered(&c, "rode o modelo 13d2"), None);
    }

    #[test]
    fn apply_adds_the_prompt_with_the_asset_list() {
        let c = Catalog::load(None);
        let msgs = vec![
            Msg { role: "system".into(), content: "base".into() },
            Msg { role: "user".into(), content: "uma cena 3d".into() },
        ];
        let (_, out) = c.apply(&msgs, &["robot.glb".into()]);
        assert_eq!(out.len(), 2);
        assert!(out[0].content.starts_with("base\n\n"));
        assert!(out[0].content.contains("- robot.glb"));
        assert!(!out[0].content.contains("{{assets}}"));

        let (_, out) = c.apply(&user("uma cena 3d"), &[]);
        assert_eq!(out[0].role, "system");
        assert!(out[0].content.contains("Nenhum arquivo 3D local"));
        assert_eq!(out[1].content, "uma cena 3d");
    }

    #[test]
    fn no_trigger_leaves_the_conversation_untouched() {
        let c = Catalog::load(None);
        let (skill, out) = c.apply(&user("oi"), &[]);
        assert!(skill.is_none());
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn external_skills_load_replace_and_skip_bad_ones() {
        let dir = temp("external");
        let write = |id: &str, manifest: &str, prompt: &str| {
            std::fs::create_dir_all(dir.join(id)).unwrap();
            std::fs::write(dir.join(id).join("skill.json"), manifest).unwrap();
            std::fs::write(dir.join(id).join("prompt.md"), prompt).unwrap();
        };
        write("pirate", r#"{"name":"Pirata","when":["pirata"]}"#, "Responda como um pirata.");
        write("3d", r#"{"name":"3D custom","when":["cena"]}"#, "Custom.");
        write("bad", "{not json", "x");
        write("silent", r#"{"name":"Sem gatilho"}"#, "x");

        let c = Catalog::load(Some(&dir));
        assert_eq!(c.skills.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["3d", "pirate"]);
        assert_eq!(triggered(&c, "fale como um pirata").as_deref(), Some("pirate"));
        assert_eq!(c.apply(&user("uma cena"), &[]).0.map(|s| s.name.as_str()), Some("3D custom"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_prompt_example_is_strict_json() {
        let prompt = include_str!("../skills/3d/prompt.md");
        let example = &prompt[prompt.find("## Exemplo").expect("example section")..];
        let start = example.find("```scene").expect("scene fence");
        let rest = &example[start + "```scene".len()..];
        let end = rest.find("```").expect("closing fence");
        let v: serde_json::Value = serde_json::from_str(rest[..end].trim()).expect("example must be strict JSON");
        assert!(v["nodes"].as_array().is_some_and(|n| !n.is_empty()));
    }
}
