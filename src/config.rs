//! Runtime settings. The command line and the settings UI write the same [`Config`]; the live one is
//! the flags laid over what the UI saved, so a flag wins until lair restarts without it.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::generation::Reasoning;
use crate::store::Store;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Config {
    /// Base URL of a SearXNG instance for the agent's web search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub searxng: Option<String>,
    /// JSON index of skills the UI can browse and install.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills_index: Option<String>,
    /// This node's name on the tailnet; unset means its Tailscale host name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `host[:port]` of other lair nodes, besides the ones discovered on the tailnet.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<String>,
    /// How much reasoning models do when a conversation doesn't say.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// Machines of this tailnet skip the token; unset means yes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trust_tailnet: Option<bool>,
    /// Hugging Face token, for gated or private models. Sent to huggingface.co only; never shown again.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_token: Option<String>,
    /// Chats recall related past conversations and workspace files; unset means yes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<bool>,
    /// Tool steps the agent may take in one reply; unset means [`crate::agent::MAX_STEPS`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<usize>,
}

impl Config {
    /// What the UI saved. A document lair can't read any more (an old field, a broken row) is set aside
    /// with a warning, and the next save replaces it.
    pub fn saved(store: &Store) -> Config {
        match store.config().and_then(|v| Ok(serde_json::from_value(v)?)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("warning: ignoring the saved settings: {e:#}");
                Config::default()
            }
        }
    }

    /// `self` (the flags) wins field by field; peers from both are kept.
    pub fn over(&self, saved: &Config) -> Config {
        // Unset fields aren't serialized, so laying one document over the other keeps what each sets.
        let (Value::Object(mut doc), Value::Object(flags)) = (json(saved), json(self)) else {
            unreachable!("a Config serializes to an object")
        };
        doc.extend(flags);
        let mut live: Config = serde_json::from_value(Value::Object(doc)).expect("two valid configs merge into one");
        live.peers = self.peers.clone();
        live.peers.extend(saved.peers.iter().filter(|p| !self.peers.contains(p)).cloned());
        live
    }

    /// `self` with the fields in `change` replaced; `null` or `""` clears one, the others stay.
    pub fn patched(&self, change: Value) -> Result<Config> {
        let mut doc = serde_json::to_value(self)?;
        if let (Value::Object(doc), Value::Object(change)) = (&mut doc, change) {
            // The UI is shown `true` in place of a saved token; sending that back changes nothing.
            doc.extend(change.into_iter().filter(|(k, v)| !(k == HF_TOKEN && v.is_boolean())));
            doc.retain(|_, v| !v.is_null());
        }
        serde_json::from_value::<Config>(doc)?.validated()
    }

    pub fn trust_tailnet(&self) -> bool {
        self.trust_tailnet.unwrap_or(true)
    }

    pub fn memory(&self) -> bool {
        self.memory.unwrap_or(true)
    }

    /// For showing: the token is replaced by `true`, so it never leaves the server again.
    pub fn redacted(&self) -> Value {
        let mut v = json(self);
        if self.hf_token.is_some() {
            v[HF_TOKEN] = Value::Bool(true);
        }
        v
    }

    /// Trims every value, drops empty ones, and refuses what lair could not use.
    pub fn validated(mut self) -> Result<Config> {
        let text = |v: Option<String>| v.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
        let url = |v: Option<String>, what: &str| -> Result<Option<String>> {
            let v = text(v);
            if v.as_deref().is_some_and(|u| !(u.starts_with("http://") || u.starts_with("https://"))) {
                bail!("{what} must start with http:// or https://");
            }
            Ok(v)
        };
        self.name = text(self.name);
        if self.name.as_deref().is_some_and(|n| n.len() > 64) {
            bail!("the node name is over 64 characters");
        }
        self.peers = self.peers.iter().map(|p| p.trim().to_owned()).filter(|p| !p.is_empty()).collect();
        for p in &self.peers {
            if p.contains(['/', ' ']) || p.starts_with("http") {
                bail!("\"{p}\" does not look like an address: use host:port (e.g. 10.1.2.3:8080)");
            }
            if p.split(':').nth(1).is_some_and(|port| port.parse::<u16>().ok().filter(|&n| n > 0).is_none()) {
                bail!("the port in \"{p}\" is not a number from 1 to 65535");
            }
        }
        if self.peers.len() > 64 {
            bail!("at most 64 peers");
        }
        if self.max_steps.is_some_and(|n| !STEPS.contains(&n)) {
            bail!("the agent's step limit must be from {} to {}", STEPS.start(), STEPS.end());
        }
        self.searxng = url(self.searxng, "the SearXNG URL")?;
        self.skills_index = url(self.skills_index, "the skills index")?;
        self.hf_token = text(self.hf_token);
        Ok(self)
    }
}

/// Where lair keeps its data: `dir` when given, else `$XDG_DATA_HOME/lair` (`~/.local/share/lair`), or
/// `%LOCALAPPDATA%\lair` on Windows. Created if missing.
pub fn data_dir(dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = match dir {
        Some(d) => d,
        None if cfg!(windows) => {
            PathBuf::from(std::env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set; pass --data")?)
                .join("lair")
        }
        None => match std::env::var_os("XDG_DATA_HOME") {
            Some(d) => PathBuf::from(d),
            None => {
                PathBuf::from(std::env::var_os("HOME").context("HOME is not set; pass --data")?).join(".local/share")
            }
        }
        .join("lair"),
    };
    crate::util::ensure_dir(dir)
}

/// The steps an agent may be allowed per reply.
const STEPS: std::ops::RangeInclusive<usize> = 1..=200;

/// The key the token goes by in the documents the UI reads and patches.
const HF_TOKEN: &str = "hfToken";

fn json(c: &Config) -> Value {
    serde_json::to_value(c).expect("a Config always serializes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_win_field_by_field_and_peers_add_up() {
        let flags = Config { searxng: Some("http://flag".into()), peers: vec!["a:1".into()], ..Default::default() };
        let saved = Config {
            searxng: Some("http://saved".into()),
            name: Some("doom".into()),
            peers: vec!["a:1".into(), "b:2".into()],
            trust_tailnet: Some(false),
            ..Default::default()
        };
        let live = flags.over(&saved);
        assert_eq!(live.searxng.as_deref(), Some("http://flag"));
        assert_eq!(live.name.as_deref(), Some("doom"));
        assert_eq!(live.peers, ["a:1", "b:2"]);
        assert!(!live.trust_tailnet());
        assert!(Config::default().trust_tailnet(), "trust is on unless turned off");
    }

    #[test]
    fn an_unset_field_is_left_out_so_saved_settings_show_through_flags() {
        assert_eq!(json(&Config::default()), serde_json::json!({}), "every field needs skip_serializing_if");
    }

    #[test]
    fn validation_trims_clears_and_refuses() {
        let c = Config {
            searxng: Some("  ".into()),
            name: Some(" doom ".into()),
            peers: vec!["10.0.0.1:8080".into(), " ".into()],
            ..Default::default()
        }
        .validated()
        .unwrap();
        assert_eq!((c.searxng, c.name.as_deref(), c.peers), (None, Some("doom"), vec!["10.0.0.1:8080".to_owned()]));
        for bad in [
            Config { searxng: Some("localhost:8888".into()), ..Default::default() },
            Config { max_steps: Some(0), ..Default::default() },
            Config { peers: vec!["http://doom".into()], ..Default::default() },
            Config { peers: vec!["doom:70000".into()], ..Default::default() },
            Config { peers: vec!["doom:0".into()], ..Default::default() },
            Config { name: Some("x".repeat(65)), ..Default::default() },
        ] {
            assert!(bad.validated().is_err());
        }
    }

    #[test]
    fn a_patch_changes_only_what_it_names() {
        let saved = Config { searxng: Some("http://s".into()), trust_tailnet: Some(false), ..Default::default() };
        let c = saved.patched(serde_json::json!({ "name": "doom", "trustTailnet": true })).unwrap();
        assert_eq!(
            (c.searxng.as_deref(), c.name.as_deref(), c.trust_tailnet),
            (Some("http://s"), Some("doom"), Some(true))
        );
        let cleared = c.patched(serde_json::json!({ "searxng": "", "name": null })).unwrap();
        assert_eq!((cleared.searxng, cleared.name), (None, None));
        assert!(c.patched(serde_json::json!({ "reasoning": "max" })).is_err(), "unknown values are refused");
    }

    #[test]
    fn saved_json_keeps_its_names() {
        let c: Config = serde_json::from_str(
            r#"{"searxng":"http://s","skillsIndex":"http://i","reasoning":"low","trustTailnet":false}"#,
        )
        .unwrap();
        assert_eq!(c.skills_index.as_deref(), Some("http://i"));
        assert_eq!(c.reasoning, Some(Reasoning::Light));
        assert_eq!(serde_json::to_value(&c).unwrap()["trustTailnet"], false);
    }
}
