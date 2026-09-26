//! Prompt registry: loads skill/agent/design files once so callers send a name,
//! not 40 KB of markdown, on every request.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::PromptsConfig;
use crate::types::{Intent, Tier};

/// Routing defaults a skill or agent file declares about itself. Placed in a
/// `pankhllm:` block of the markdown front matter or of the agent YAML.
/// Request-level values override these; tags are merged.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptMeta {
    #[serde(default)]
    pub tier: Option<Tier>,
    #[serde(default)]
    pub intent: Option<Intent>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    /// Pin to a named model. Use sparingly: it bypasses tiering for every call using this skill.
    #[serde(default)]
    pub model: Option<String>,
    /// Keywords or phrases (case-insensitive) that pull this skill into a request
    /// under `prompts.auto_select`.
    #[serde(default)]
    pub triggers: Vec<String>,
    /// Always included under `prompts.auto_select` (agent definition, house rules).
    #[serde(default)]
    pub always: bool,
    /// Example questions this skill is for. They seed the skill-routing model; requests
    /// that name the skill explicitly keep teaching it.
    #[serde(default)]
    pub examples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Prompt {
    pub body: String,
    pub meta: PromptMeta,
}

#[derive(Debug, Default, Clone)]
pub struct PromptRegistry {
    prompts: HashMap<String, Prompt>,
    default: Option<String>,
}

#[derive(Deserialize, Default)]
struct FrontMatter {
    #[serde(default)]
    pankhllm: PromptMeta,
}

/// Split `---\n<yaml>\n---\n<body>`; returns (meta, body). No front matter: (default, whole text).
pub fn parse_front_matter(raw: &str) -> Result<(PromptMeta, String)> {
    let trimmed = raw.trim_start_matches('\u{feff}');
    if let Some(rest) = trimmed.strip_prefix("---") {
        let rest = rest.trim_start_matches(['\r', '\n']);
        if let Some(end) = rest.find("\n---") {
            let yaml = &rest[..end];
            let body = rest[end + 4..].trim_start_matches(['\r', '\n']).to_string();
            let fm: FrontMatter = serde_yaml::from_str(yaml).context("parsing prompt front matter")?;
            return Ok((fm.pankhllm, body));
        }
    }
    Ok((PromptMeta::default(), trimmed.to_string()))
}

const EXTENSIONS: &[&str] = &["md", "txt", "yaml", "yml"];

impl PromptRegistry {
    pub fn load(cfg: &PromptsConfig) -> Result<Self> {
        let mut prompts = HashMap::new();
        if let Some(dir) = &cfg.dir {
            let dir = Path::new(dir);
            for entry in std::fs::read_dir(dir).with_context(|| format!("reading prompts dir {}", dir.display()))? {
                let path = entry?.path();
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if !path.is_file() || !EXTENSIONS.contains(&ext) {
                    continue;
                }
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                let raw = std::fs::read_to_string(&path)?;
                let (meta, body) = if ext == "yaml" || ext == "yml" {
                    // Agent YAML: a top-level `pankhllm:` block is routing metadata; the
                    // whole file is wrapped so the model sees it as a definition, not prose.
                    let fm: FrontMatter = serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
                    (fm.pankhllm, format!("<agent_definition name=\"{stem}\">\n{raw}\n</agent_definition>"))
                } else {
                    parse_front_matter(&raw).with_context(|| format!("parsing {}", path.display()))?
                };
                prompts.insert(stem, Prompt { body, meta });
            }
        }
        if let Some(d) = &cfg.default {
            anyhow::ensure!(prompts.contains_key(d), "prompts.default '{d}' not found in prompts dir");
        }
        Ok(Self { prompts, default: cfg.default.clone() })
    }

    pub fn from_map(prompts: HashMap<String, String>) -> Self {
        let prompts = prompts.into_iter().map(|(k, body)| (k, Prompt { body, meta: PromptMeta::default() })).collect();
        Self { prompts, default: None }
    }

    pub fn from_prompts(prompts: HashMap<String, Prompt>) -> Self {
        Self { prompts, default: None }
    }

    /// Merged routing metadata of the named prompts (later ones override scalars, tags union).
    pub fn meta(&self, names: Option<&str>) -> Result<PromptMeta, String> {
        let names = names.map(|s| s.to_string()).or_else(|| self.default.clone());
        let mut out = PromptMeta::default();
        let Some(names) = names else { return Ok(out) };
        for n in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let p = self.prompts.get(n).ok_or_else(|| format!("unknown prompt '{n}'; known: {:?}", self.names()))?;
            let m = &p.meta;
            out.tier = m.tier.or(out.tier);
            out.intent = m.intent.or(out.intent);
            out.max_latency_ms = m.max_latency_ms.or(out.max_latency_ms);
            out.max_cost_usd = m.max_cost_usd.or(out.max_cost_usd);
            out.model = m.model.clone().or(out.model);
            for t in &m.tags {
                if !out.tags.contains(t) {
                    out.tags.push(t.clone());
                }
            }
        }
        Ok(out)
    }

    /// Skills that can be chosen (not `always`), sorted.
    pub fn choosable(&self) -> Vec<String> {
        let mut v: Vec<String> = self.prompts.iter().filter(|(_, p)| !p.meta.always).map(|(n, _)| n.clone()).collect();
        v.sort();
        v
    }

    pub fn is_always(&self, name: &str) -> bool {
        self.prompts.get(name).is_some_and(|p| p.meta.always)
    }

    /// Name plus the start of the body, for a teacher choosing among skills.
    pub fn summary(&self, name: &str, max_chars: usize) -> String {
        let body = self.prompts.get(name).map(|p| p.body.as_str()).unwrap_or("");
        let flat: String = body.split_whitespace().collect::<Vec<_>>().join(" ");
        flat.chars().take(max_chars).collect()
    }

    /// (skill, example question) pairs from front matter.
    pub fn skill_examples(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = self.prompts.iter().flat_map(|(n, p)| p.meta.examples.iter().map(move |e| (n.clone(), e.clone()))).collect();
        v.sort();
        v
    }

    /// Prompts to include for a question: `always` ones, then triggered ones in
    /// order of how many triggers matched, capped at `max_auto`.
    pub fn select(&self, query: &str, max_auto: usize) -> Vec<String> {
        let q = query.to_lowercase();
        let mut always: Vec<String> = Vec::new();
        let mut scored: Vec<(usize, String)> = Vec::new();
        for (name, p) in &self.prompts {
            if p.meta.always {
                always.push(name.clone());
                continue;
            }
            let hits = p.meta.triggers.iter().filter(|t| !t.is_empty() && q.contains(&t.to_lowercase())).count();
            if hits > 0 {
                scored.push((hits, name.clone()));
            }
        }
        always.sort();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        always.extend(scored.into_iter().take(max_auto).map(|(_, n)| n));
        always
    }

    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.prompts.keys().map(|s| s.as_str()).collect();
        v.sort();
        v
    }

    /// Resolve a comma-separated list of names (joined in order), or the default.
    pub fn resolve(&self, names: Option<&str>) -> Result<Option<String>, String> {
        let names = names.map(|s| s.to_string()).or_else(|| self.default.clone());
        let Some(names) = names else { return Ok(None) };
        let mut parts = Vec::new();
        for n in names.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match self.prompts.get(n) {
                Some(p) => parts.push(p.body.as_str()),
                None => return Err(format!("unknown prompt '{n}'; known: {:?}", self.names())),
            }
        }
        Ok(if parts.is_empty() { None } else { Some(parts.join("\n\n")) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_is_parsed_and_stripped() {
        let raw = "---\npankhllm:\n  tier: fast\n  tags: [private]\n  max_latency_ms: 5000\n---\n# Skill\nBe brief.";
        let (meta, body) = parse_front_matter(raw).unwrap();
        assert_eq!(meta.tier, Some(Tier::Fast));
        assert_eq!(meta.tags, vec!["private"]);
        assert_eq!(meta.max_latency_ms, Some(5000));
        assert_eq!(body, "# Skill\nBe brief.");
        let (meta, body) = parse_front_matter("plain text").unwrap();
        assert_eq!(meta, PromptMeta::default());
        assert_eq!(body, "plain text");
    }

    #[test]
    fn auto_select_by_triggers_and_always() {
        let mut m = HashMap::new();
        m.insert("agent".to_string(), Prompt { body: "AGENT".into(), meta: PromptMeta { always: true, ..Default::default() } });
        m.insert("kpi".to_string(), Prompt { body: "KPI".into(), meta: PromptMeta { triggers: vec!["call activity".into(), "reach".into()], ..Default::default() } });
        m.insert("alerts".to_string(), Prompt { body: "ALERTS".into(), meta: PromptMeta { triggers: vec!["alert".into()], ..Default::default() } });
        m.insert("sql".to_string(), Prompt { body: "SQL".into(), meta: PromptMeta { triggers: vec!["sql".into(), "query".into()], ..Default::default() } });
        let r = PromptRegistry::from_prompts(m);
        assert_eq!(r.select("What was call activity and reach for East?", 3), vec!["agent", "kpi"]);
        assert_eq!(r.select("hello", 3), vec!["agent"]);
        assert_eq!(r.select("Write a SQL query for alert counts", 1), vec!["agent", "sql"], "capped, highest trigger count first");
    }

    #[test]
    fn meta_merges_across_prompts() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), Prompt { body: "A".into(), meta: PromptMeta { tier: Some(Tier::Fast), tags: vec!["private".into()], ..Default::default() } });
        m.insert("b".to_string(), Prompt { body: "B".into(), meta: PromptMeta { tier: Some(Tier::Reasoning), tags: vec!["eu".into()], ..Default::default() } });
        let r = PromptRegistry::from_prompts(m);
        let meta = r.meta(Some("a, b")).unwrap();
        assert_eq!(meta.tier, Some(Tier::Reasoning));
        assert_eq!(meta.tags, vec!["private", "eu"]);
        assert_eq!(r.resolve(Some("a,b")).unwrap().unwrap(), "A\n\nB");
        assert!(r.meta(Some("zzz")).is_err());
    }
}
