//! Skills: prompt packages added to a request when the user's message asks for one.
//!
//! A skill is a folder with `skill.json` (`name`, `when`, optional `description` and `source`) and
//! `prompt.md`. They come from three places, later ones replacing earlier ones with the same id:
//! the built-ins embedded in the binary, an optional read-only folder (`--skills-dir`), and lair's
//! own writable folder, where skills created in the UI and installed from a URL live. They are
//! prompts, not executable plugins. A `{{assets}}` slot in the prompt receives the local 3D asset list.

pub mod install;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::engine::{with_system, Msg};

const BUILTINS: &[(&str, &str, &str)] =
    &[("3d", include_str!("../../skills/3d/skill.json"), include_str!("../../skills/3d/prompt.md"))];

/// Ids of skills switched off in the UI, kept next to the writable skills.
const DISABLED_FILE: &str = "disabled.json";

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    Builtin,
    /// The read-only `--skills-dir` folder.
    Extra,
    Created,
    Installed,
}

#[derive(Clone, Debug, Serialize)]
pub struct Skill {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Trigger phrases as written; matching uses their [`normalize`]d form.
    pub when: Vec<String>,
    pub prompt: String,
    pub origin: Origin,
    /// Where an installed skill came from, so it can be updated.
    pub source: Option<String>,
    pub enabled: bool,
    #[serde(skip)]
    triggers: Vec<String>,
}

/// What a user (or the agent) writes to create or change a skill.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Manifest {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default)]
    pub when: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A skill's name and id, as announced to clients when it runs.
pub struct Used {
    pub id: String,
    pub name: String,
}

#[derive(Default)]
pub struct Catalog {
    skills: Vec<Skill>,
    extra: Option<PathBuf>,
    /// Writable folder; `None` when skills are turned off.
    own: Option<PathBuf>,
}

impl Catalog {
    pub fn load(extra: Option<PathBuf>, own: PathBuf) -> Self {
        let mut c = Self { skills: Vec::new(), extra, own: Some(own) };
        c.reload();
        c
    }

    fn reload(&mut self) {
        let disabled = self.disabled();
        let mut skills: Vec<Skill> = BUILTINS
            .iter()
            .map(|(id, manifest, prompt)| Skill::parse(id, manifest, prompt, Origin::Builtin).expect("built-in skills are valid"))
            .collect();
        let folders = [(self.extra.as_deref(), Origin::Extra), (self.own.as_deref(), Origin::Created)];
        for (dir, origin) in folders {
            for skill in dir.map(|d| load_dir(d, origin)).unwrap_or_default() {
                match skills.iter().position(|s| s.id == skill.id) {
                    Some(i) => skills[i] = skill,
                    None => skills.push(skill),
                }
            }
        }
        for s in &mut skills {
            s.enabled = !disabled.contains(&s.id);
        }
        self.skills = skills;
    }

    pub fn list(&self) -> &[Skill] {
        &self.skills
    }

    /// The skill the last user message asks for, if any, and the messages with its instructions added.
    pub fn apply(&self, msgs: &[Msg], assets: &[String]) -> (Option<Used>, Vec<Msg>) {
        let Some(skill) = self.trigger(msgs) else {
            return (None, msgs.to_vec());
        };
        let prompt = skill.prompt.replace("{{assets}}", &assets_section(assets));
        (Some(Used { id: skill.id.clone(), name: skill.name.clone() }), with_system(msgs, prompt.trim_end()))
    }

    fn trigger(&self, msgs: &[Msg]) -> Option<&Skill> {
        let text = normalize(&msgs.iter().rev().find(|m| m.role == "user")?.content);
        self.skills.iter().filter(|s| s.enabled).find(|s| s.triggers.iter().any(|w| contains_word(&text, w)))
    }

    fn own(&self) -> Result<&Path> {
        self.own.as_deref().context("skills are turned off on this server (--no-skills)")
    }

    /// Creates or replaces a skill in the writable folder. A built-in with the same id stays hidden behind it.
    pub fn save(&mut self, id: &str, manifest: &Manifest, prompt: &str) -> Result<Skill> {
        Skill::parse(id, &serde_json::to_string(manifest)?, prompt, Origin::Created)?;
        let dir = self.own()?.join(id);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("skill.json"), serde_json::to_string_pretty(manifest)?)?;
        std::fs::write(dir.join("prompt.md"), prompt)?;
        self.reload();
        Ok(self.skills.iter().find(|s| s.id == id).cloned().expect("just saved"))
    }

    /// Deletes a created or installed skill; a built-in it was replacing comes back.
    pub fn remove(&mut self, id: &str) -> Result<()> {
        let dir = self.own()?.join(id);
        if !valid_id(id) || !dir.is_dir() {
            bail!("only skills created here or installed can be deleted");
        }
        std::fs::remove_dir_all(dir)?;
        self.reload();
        Ok(())
    }

    pub fn set_enabled(&mut self, id: &str, enabled: bool) -> Result<()> {
        if !self.skills.iter().any(|s| s.id == id) {
            bail!("no skill called {id}");
        }
        let mut disabled = self.disabled();
        if enabled {
            disabled.remove(id);
        } else {
            disabled.insert(id.to_owned());
        }
        std::fs::create_dir_all(self.own()?)?;
        std::fs::write(self.own()?.join(DISABLED_FILE), serde_json::to_string(&disabled)?)?;
        self.reload();
        Ok(())
    }

    fn disabled(&self) -> BTreeSet<String> {
        self.own
            .as_deref()
            .and_then(|d| std::fs::read_to_string(d.join(DISABLED_FILE)).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
}

impl Skill {
    fn parse(id: &str, manifest: &str, prompt: &str, origin: Origin) -> Result<Self> {
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
        let mut triggers = Vec::new();
        for w in m.when.iter().map(|w| normalize(w.trim())) {
            if !w.is_empty() && !triggers.contains(&w) {
                triggers.push(w);
            }
        }
        if triggers.is_empty() {
            bail!("`when` needs at least one trigger phrase");
        }
        let origin = if origin == Origin::Created && m.source.is_some() { Origin::Installed } else { origin };
        Ok(Self {
            id: id.to_owned(),
            name,
            description: m.description.trim().to_owned(),
            when: m.when.iter().map(|w| w.trim().to_owned()).filter(|w| !w.is_empty()).collect(),
            prompt: prompt.to_owned(),
            origin,
            source: m.source,
            enabled: true,
            triggers,
        })
    }
}

pub fn valid_id(id: &str) -> bool {
    let mut chars = id.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn load_dir(dir: &Path, origin: Origin) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut skills: Vec<Skill> = Vec::new();
    for path in entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()) {
        let id = path.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
        let parsed = std::fs::read_to_string(path.join("skill.json"))
            .and_then(|m| Ok((m, std::fs::read_to_string(path.join("prompt.md"))?)))
            .map_err(anyhow::Error::from)
            .and_then(|(manifest, prompt)| Skill::parse(&id, &manifest, &prompt, origin));
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
        c.apply(&user(text), &[]).0.map(|s| s.id)
    }

    fn manifest(name: &str, when: &[&str]) -> Manifest {
        Manifest { name: name.into(), when: when.iter().map(|w| w.to_string()).collect(), ..Default::default() }
    }

    #[test]
    fn the_3d_skill_triggers_only_on_explicit_requests() {
        let c = Catalog::load(None, temp("3d"));
        assert_eq!(triggered(&c, "mostre um sistema solar em 3D").as_deref(), Some("3d"));
        assert_eq!(triggered(&c, "Desenhe uma molécula em três dimensões").as_deref(), Some("3d"));
        assert_eq!(triggered(&c, "qual o maior planeta?"), None);
        assert_eq!(triggered(&c, "a fórmula da molécula de água"), None);
        assert_eq!(triggered(&c, "rode o modelo 13d2"), None);
    }

    #[test]
    fn apply_adds_the_prompt_with_the_asset_list() {
        let c = Catalog::load(None, temp("apply"));
        let msgs = vec![
            Msg { role: "system".into(), content: "base".into() },
            Msg { role: "user".into(), content: "uma cena 3d".into() },
        ];
        let (_, out) = c.apply(&msgs, &["robot.glb".into()]);
        assert_eq!(out.len(), 2);
        assert!(out[0].content.starts_with("base\n\n"));
        assert!(out[0].content.contains("- robot.glb"));
        assert!(!out[0].content.contains("{{assets}}"));
        let (skill, out) = c.apply(&user("oi"), &[]);
        assert!(skill.is_none());
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn created_skills_replace_builtins_until_deleted_and_can_be_switched_off() {
        let own = temp("own");
        let mut c = Catalog::load(None, own.clone());
        let pirate = c.save("pirate", &manifest("Pirata", &["pirata"]), "Responda como um pirata.").unwrap();
        assert_eq!(pirate.origin, Origin::Created);
        assert_eq!(triggered(&c, "fale como um pirata").as_deref(), Some("pirate"));

        c.save("3d", &manifest("3D custom", &["cena"]), "Custom.").unwrap();
        assert_eq!(c.apply(&user("uma cena"), &[]).0.map(|s| s.name).as_deref(), Some("3D custom"));
        c.remove("3d").unwrap();
        assert_eq!(c.list().iter().find(|s| s.id == "3d").unwrap().origin, Origin::Builtin);
        assert!(c.remove("3d").is_err(), "built-ins cannot be deleted");

        c.set_enabled("pirate", false).unwrap();
        assert_eq!(triggered(&c, "fale como um pirata"), None);
        let reopened = Catalog::load(None, own);
        assert!(!reopened.list().iter().find(|s| s.id == "pirate").unwrap().enabled, "the switch survives a restart");
    }

    #[test]
    fn bad_skills_are_refused_or_skipped() {
        let own = temp("bad");
        let mut c = Catalog::load(None, own.clone());
        assert!(c.save("Bad Id", &manifest("x", &["x"]), "x").is_err());
        assert!(c.save("silent", &manifest("Sem gatilho", &[]), "x").is_err());
        std::fs::create_dir_all(own.join("broken")).unwrap();
        std::fs::write(own.join("broken/skill.json"), "{not json").unwrap();
        std::fs::write(own.join("broken/prompt.md"), "x").unwrap();
        assert!(Catalog::load(None, own).list().iter().all(|s| s.id != "broken"));
        assert!(Catalog::default().save("a", &manifest("A", &["a"]), "x").is_err(), "--no-skills");
    }

    #[test]
    fn the_prompt_example_is_strict_json() {
        let prompt = include_str!("../../skills/3d/prompt.md");
        let example = &prompt[prompt.find("## Exemplo").expect("example section")..];
        let start = example.find("```scene").expect("scene fence");
        let rest = &example[start + "```scene".len()..];
        let end = rest.find("```").expect("closing fence");
        let v: serde_json::Value = serde_json::from_str(rest[..end].trim()).expect("example must be strict JSON");
        assert!(v["nodes"].as_array().is_some_and(|n| !n.is_empty()));
    }
}
