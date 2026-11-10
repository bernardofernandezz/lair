//! Runtime settings. The command line and the settings UI write the same [`Config`]; the live one is
//! the flags laid over what the UI saved, so a flag wins until lair restarts without it.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::Reasoning;

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
}

impl Config {
    /// `self` (the flags) wins field by field; peers from both are kept.
    pub fn over(&self, saved: &Config) -> Config {
        let mut peers = self.peers.clone();
        peers.extend(saved.peers.iter().filter(|p| !self.peers.contains(p)).cloned());
        Config {
            searxng: self.searxng.clone().or_else(|| saved.searxng.clone()),
            skills_index: self.skills_index.clone().or_else(|| saved.skills_index.clone()),
            name: self.name.clone().or_else(|| saved.name.clone()),
            peers,
            reasoning: self.reasoning.or(saved.reasoning),
            trust_tailnet: self.trust_tailnet.or(saved.trust_tailnet),
        }
    }

    /// `self` with the fields in `change` replaced; `null` or `""` clears one, the others stay.
    pub fn patched(&self, change: Value) -> Result<Config> {
        let mut doc = serde_json::to_value(self)?;
        if let (Value::Object(doc), Value::Object(change)) = (&mut doc, change) {
            doc.extend(change);
            doc.retain(|_, v| !v.is_null());
        }
        serde_json::from_value::<Config>(doc)?.validated()
    }

    pub fn trust_tailnet(&self) -> bool {
        self.trust_tailnet.unwrap_or(true)
    }

    /// Trims every value, drops empty ones, and refuses what lair could not use.
    pub fn validated(self) -> Result<Config> {
        let text = |v: Option<String>| v.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty());
        let url = |v: Option<String>, what: &str| -> Result<Option<String>> {
            let v = text(v);
            if v.as_deref().is_some_and(|u| !(u.starts_with("http://") || u.starts_with("https://"))) {
                bail!("{what} deve começar com http:// ou https://");
            }
            Ok(v)
        };
        let name = text(self.name);
        if name.as_deref().is_some_and(|n| n.len() > 64) {
            bail!("o nome do nó passa de 64 caracteres");
        }
        let peers: Vec<String> = self.peers.iter().map(|p| p.trim().to_owned()).filter(|p| !p.is_empty()).collect();
        for p in &peers {
            if p.contains(['/', ' ']) || p.starts_with("http") {
                bail!("\"{p}\" não parece um endereço: use host:porta (ex.: 10.1.2.3:8080)");
            }
            if p.split(':').nth(1).is_some_and(|port| port.parse::<u16>().is_err()) {
                bail!("a porta de \"{p}\" não é um número de 1 a 65535");
            }
        }
        if peers.len() > 64 {
            bail!("no máximo 64 peers");
        }
        Ok(Config {
            searxng: url(self.searxng, "a URL do SearXNG")?,
            skills_index: url(self.skills_index, "o índice de skills")?,
            name,
            peers,
            reasoning: self.reasoning,
            trust_tailnet: self.trust_tailnet,
        })
    }
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
            Config { peers: vec!["http://doom".into()], ..Default::default() },
            Config { peers: vec!["doom:70000".into()], ..Default::default() },
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
