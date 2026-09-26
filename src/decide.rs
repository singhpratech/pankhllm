//! System-1 decision engine client. Typed decisions (choice, score, yes/no) in one
//! forward pass from a non-autoregressive model, instead of generating text.
//!
//! Speaks the `POST /v1/systemone` shape served by `laya-serve` (open weights,
//! self-hosted) and by Jev-compatible endpoints. Every call has a hard timeout;
//! a slow or failing engine simply means the router uses its slower path.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::config::DecisionConfig;

/// One typed question.
#[derive(Debug, Clone)]
pub enum Question {
    /// Pick one label; each label carries a short description. Order is preserved on the
    /// wire: decision models show position bias, so callers put the default-reject option last.
    Choice { instructions: String, criteria: Vec<(String, String)> },
    /// Yes/no: calibrated probability that the statement is true.
    Noul { instructions: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Choice { label: String, confidence: f64 },
    Noul { p_true: f64 },
}

pub struct Verdict {
    pub answers: HashMap<String, Answer>,
    pub latency_ms: u128,
    pub model: Option<String>,
}

pub struct DecisionEngine {
    http: reqwest::Client,
    pub cfg: DecisionConfig,
    key: Option<String>,
}

impl DecisionEngine {
    pub fn new(cfg: DecisionConfig) -> Self {
        let key = cfg.api_key_env.as_ref().and_then(|e| std::env::var(e).ok()).filter(|k| !k.is_empty());
        let http = reqwest::Client::builder().pool_idle_timeout(Duration::from_secs(90)).tcp_nodelay(true).build().unwrap_or_default();
        Self { http, cfg, key }
    }

    fn body(&self, state: &str, questions: &[(String, Question)]) -> Value {
        let mut qs = Map::new();
        for (name, q) in questions {
            let v = match q {
                Question::Choice { instructions, criteria } => {
                    let mut c = Map::new();
                    for (k, v) in criteria {
                        c.insert(k.clone(), json!(v));
                    }
                    json!({"type": "choice", "instructions": instructions, "criteria": c})
                }
                Question::Noul { instructions } => json!({"type": "noul", "instructions": instructions}),
            };
            qs.insert(name.clone(), v);
        }
        let mut b = json!({"state": state, "questions": qs});
        if let Some(m) = &self.cfg.model {
            b["model"] = json!(m);
        }
        b
    }

    /// Ask several typed questions about one piece of text in a single call.
    pub async fn decide(&self, state: &str, questions: &[(String, Question)]) -> Result<Verdict, String> {
        let started = Instant::now();
        let url = self.cfg.url.as_deref().ok_or_else(|| "no external decision engine url".to_string())?;
        let mut rq = self.http.post(url).timeout(Duration::from_millis(self.cfg.timeout_ms.max(1))).json(&self.body(state, questions));
        if let Some(k) = &self.key {
            rq = rq.bearer_auth(k);
        }
        let resp = rq.send().await.map_err(|e| if e.is_timeout() { "decision engine timed out".to_string() } else { format!("decision engine unreachable: {e}") })?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| e.to_string())?;
        if !status.is_success() {
            return Err(format!("decision engine {status}: {}", crate::providers::truncate(&text, 200)));
        }
        let v: Value = serde_json::from_str(&text).map_err(|_| "decision engine did not return JSON".to_string())?;
        let answers = parse_answers(&v, questions);
        if answers.is_empty() {
            return Err("decision engine returned no usable answers".into());
        }
        Ok(Verdict { answers, latency_ms: started.elapsed().as_millis(), model: v["routing"]["model"].as_str().map(str::to_string) })
    }
}

fn num(v: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_f64()))
}

/// Tolerant parser: accepts `answers.<name>` objects with `choice`/`label` and
/// `answer_confidence`/`confidence` (or a probability map), and for yes/no any of
/// `p_true`/`probability`/`prob`/`value`/`answer`.
pub fn parse_answers(v: &Value, questions: &[(String, Question)]) -> HashMap<String, Answer> {
    let root = if v.get("answers").is_some() { &v["answers"] } else { v };
    let mut out = HashMap::new();
    for (name, q) in questions {
        let a = &root[name];
        if a.is_null() {
            continue;
        }
        match q {
            Question::Choice { criteria, .. } => {
                let label = a["choice"].as_str().or(a["label"].as_str()).or(a["answer"].as_str()).map(str::to_string);
                let probs: Option<(String, f64)> = a["probabilities"].as_object().and_then(|m| {
                    m.iter().filter_map(|(k, p)| p.as_f64().map(|p| (k.clone(), p))).max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                });
                let label = label.or_else(|| probs.as_ref().map(|p| p.0.clone()));
                let Some(label) = label.filter(|l| criteria.iter().any(|(k, _)| k == l)) else { continue };
                // Probability of the reported answer is the gate; normalised-entropy confidence is the fallback.
                let conf = num(a, &["answer_confidence"])
                    .or_else(|| a["probabilities"].get(&label).and_then(|p| p.as_f64()))
                    .or_else(|| num(a, &["confidence"]))
                    .unwrap_or(0.0);
                out.insert(name.clone(), Answer::Choice { label, confidence: conf.clamp(0.0, 1.0) });
            }
            Question::Noul { .. } => {
                let p = num(a, &["p_true", "probability", "prob", "value", "answer"]).or_else(|| a.as_f64());
                if let Some(p) = p {
                    out.insert(name.clone(), Answer::Noul { p_true: p.clamp(0.0, 1.0) });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qs() -> Vec<(String, Question)> {
        let c = vec![("billing".to_string(), "refunds".to_string()), ("tech".to_string(), "bugs".to_string())];
        vec![("dept".into(), Question::Choice { instructions: "which team?".into(), criteria: c }), ("churn".into(), Question::Noul { instructions: "cancel?".into() })]
    }

    #[test]
    fn criteria_order_is_preserved_on_the_wire() {
        let e = DecisionEngine::new(crate::config::DecisionConfig { models_file: None, engine: crate::config::DecisionEngineKind::External, url: Some("http://x".into()), target_precision: 0.98, api_key_env: None, timeout_ms: 1, min_confidence: 0.85, consensus_confidence: 0.55, min_known_words: 0.7, mode: Default::default(), model: None, planner: true, tools: false, allow_tools: vec![] });
        let q = vec![("op".to_string(), Question::Choice { instructions: "i".into(), criteria: vec![("zeta".into(), "z".into()), ("alpha".into(), "a".into()), ("UNSUPPORTED".into(), "none".into())] })];
        let body = e.body("s", &q).to_string();
        let (z, a, u) = (body.find("zeta").unwrap(), body.find("alpha").unwrap(), body.find("UNSUPPORTED").unwrap());
        assert!(z < a && a < u, "{body}");
    }

    #[test]
    fn parses_laya_and_jev_like_shapes() {
        let v = json!({"answers": {"dept": {"choice": "billing", "probabilities": {"billing": 0.91, "tech": 0.09}, "confidence": 0.6, "answer_confidence": 0.91}, "churn": {"probability": 0.8}}});
        let a = parse_answers(&v, &qs());
        assert_eq!(a["dept"], Answer::Choice { label: "billing".into(), confidence: 0.91 });
        assert_eq!(a["churn"], Answer::Noul { p_true: 0.8 });
        // probability map only
        let v = json!({"answers": {"dept": {"probabilities": {"billing": 0.2, "tech": 0.8}}}});
        assert_eq!(parse_answers(&v, &qs())["dept"], Answer::Choice { label: "tech".into(), confidence: 0.8 });
        // unknown label is ignored rather than trusted
        let v = json!({"answers": {"dept": {"choice": "legal", "answer_confidence": 0.99}}});
        assert!(!parse_answers(&v, &qs()).contains_key("dept"));
    }
}
