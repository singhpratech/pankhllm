//! Learned routes: the router generalises the planning model's tool calls into
//! slot-typed shapes and serves later questions of the same shape without a model.
//!
//! "call activity for T-112 in 2026-08" -> tool exec_query {territory: T-112, month: 2026-08}
//! becomes shape "call activity for <CODE> in <DATE>" with args {territory: {{s0}}, month: {{s1}}}.
//! "Call activity for T-207 in 2026-09?" then fills the slots and never reaches a model.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::cache::{cosine, hash64};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotType {
    Date,
    Code,
    Num,
    Text,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub kind: SlotType,
    pub value: String,
}

/// Typed spans found in a question, in order of appearance, non-overlapping.
pub fn typed_spans(q: &str) -> Vec<Span> {
    typed_spans_with_vocab(q, &[])
}

/// Same, with a vocabulary of known entity names (products, regions, segments)
/// that count as text slots whatever their casing. Longest names first.
pub fn typed_spans_with_vocab(q: &str, vocab: &[String]) -> Vec<Span> {
    let mut spans = typed_spans_inner(q);
    if !vocab.is_empty() {
        let lower = q.to_lowercase();
        let mut names: Vec<&String> = vocab.iter().collect();
        names.sort_by_key(|n| std::cmp::Reverse(n.len()));
        for name in names {
            let n = name.to_lowercase();
            if n.is_empty() {
                continue;
            }
            let mut from = 0;
            while let Some(pos) = lower[from..].find(&n) {
                let (s, e) = (from + pos, from + pos + n.len());
                from = e;
                let boundary = |i: usize| i == 0 || i == q.len() || !q.as_bytes()[i.min(q.len() - 1)].is_ascii_alphanumeric();
                let left_ok = s == 0 || !q.as_bytes()[s - 1].is_ascii_alphanumeric();
                let right_ok = e >= q.len() || !q.as_bytes()[e].is_ascii_alphanumeric();
                let _ = boundary;
                if !left_ok || !right_ok || spans.iter().any(|x| s < x.end && e > x.start) {
                    continue;
                }
                spans.push(Span { start: s, end: e, kind: SlotType::Text, value: q[s..e].to_string() });
            }
        }
        spans.sort_by_key(|s| s.start);
    }
    spans
}

fn typed_spans_inner(q: &str) -> Vec<Span> {
    static PATTERNS: std::sync::OnceLock<Vec<(SlotType, Regex)>> = std::sync::OnceLock::new();
    let pats = PATTERNS.get_or_init(|| {
        vec![
            (SlotType::Date, Regex::new(r"\b\d{4}-\d{2}(?:-\d{2})?\b").unwrap()),
            (SlotType::Code, Regex::new(r"\b[A-Za-z]{1,6}-\d{1,8}\b").unwrap()),
            (SlotType::Num, Regex::new(r"\b\d+(?:[.,]\d+)?%?\b").unwrap()),
            // Quoted strings, or capitalised words/phrases that are not the first word.
            (SlotType::Text, Regex::new(r#""([^"]{1,60})"|'([^']{1,60})'|[\s,;:(]([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)"#).unwrap()),
        ]
    });
    let mut spans: Vec<Span> = Vec::new();
    for (kind, re) in pats {
        for caps in re.captures_iter(q) {
            // Use the first capture group when the pattern has one (text slots), else the whole match.
            let m = (1..caps.len()).find_map(|i| caps.get(i)).unwrap_or_else(|| caps.get(0).unwrap());
            let (s, e) = (m.start(), m.end());
            if spans.iter().any(|x| s < x.end && e > x.start) {
                continue; // overlaps an earlier, higher-priority span
            }
            spans.push(Span { start: s, end: e, kind: *kind, value: m.as_str().to_string() });
        }
    }
    spans.sort_by_key(|s| s.start);
    spans
}

fn normalise(s: &str) -> String {
    s.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ").trim_end_matches(['?', '.', '!']).trim().to_string()
}

/// Replace chosen spans with typed tokens, producing the shape string.
pub fn abstract_question(q: &str, spans: &[Span]) -> String {
    let mut out = String::new();
    let mut last = 0;
    for s in spans {
        out.push_str(&q[last..s.start]);
        out.push_str(match s.kind {
            SlotType::Date => "<date>",
            SlotType::Code => "<code>",
            SlotType::Num => "<num>",
            SlotType::Text => "<text>",
        });
        last = s.end;
    }
    out.push_str(&q[last..]);
    normalise(&out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedRoute {
    pub shape: String,
    pub slot_types: Vec<SlotType>,
    pub tool: String,
    /// Arguments JSON with `{{s0}}`, `{{s1}}`... for slot values.
    pub args_template: String,
    pub answer_template: Option<String>,
    pub needs_analysis: bool,
    pub observations: u64,
    pub hits: u64,
    #[serde(skip, default = "Instant::now")]
    pub created: Instant,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// Scope hash the route belongs to (tenant key, prompt, tool set).
    #[serde(default)]
    pub scope: u64,
    /// Served calls whose tool result came back clean / as an error.
    #[serde(default)]
    pub successes: u64,
    #[serde(default)]
    pub failures: u64,
    /// Quarantined routes are kept for inspection but never served.
    #[serde(default)]
    pub quarantined: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quarantine_reason: Option<String>,
}

/// On-disk form of the route memory.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RouteSnapshot {
    pub routes: Vec<LearnedRoute>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct LearnStats {
    pub routes: usize,
    pub learned: u64,
    pub served: u64,
    pub rejected_shape: u64,
}

pub struct RouteMemory {
    ttl: Duration,
    max_routes: usize,
    min_similarity: f64,
    min_observations: u64,
    vocab: Vec<String>,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    routes: HashMap<u64, LearnedRoute>, // key: hash(scope + shape)
    stats: LearnStats,
}

/// Result of matching a question against learned routes.
pub struct Fill {
    /// Identifies the route for success/failure feedback.
    pub key: u64,
    pub tool: String,
    pub arguments: String,
    pub answer_template: Option<String>,
    pub needs_analysis: bool,
    pub shape: String,
    pub similarity: Option<f64>,
}

impl RouteMemory {
    pub fn new(ttl_secs: u64, max_routes: usize, min_similarity: f64, min_observations: u64) -> Self {
        Self::with_vocab(ttl_secs, max_routes, min_similarity, min_observations, Vec::new())
    }

    pub fn with_vocab(ttl_secs: u64, max_routes: usize, min_similarity: f64, min_observations: u64, vocab: Vec<String>) -> Self {
        Self { ttl: Duration::from_secs(ttl_secs.max(1)), max_routes: max_routes.max(1), min_similarity, min_observations: min_observations.max(1), vocab, inner: Mutex::new(Inner::default()) }
    }

    pub fn spans(&self, q: &str) -> Vec<Span> {
        typed_spans_with_vocab(q, &self.vocab)
    }

    pub fn vocab_list(&self) -> Vec<String> {
        self.vocab.clone()
    }

    pub fn is_vocab(&self, value: &str) -> bool {
        self.vocab.iter().any(|v| v.eq_ignore_ascii_case(value))
    }

    /// Everything learned, for persistence or shipping to another instance.
    pub fn export(&self) -> RouteSnapshot {
        let g = self.lock();
        RouteSnapshot { routes: g.routes.values().cloned().collect() }
    }

    /// Merge a snapshot in. Existing routes keep the higher observation count.
    pub fn import(&self, snap: RouteSnapshot) -> usize {
        let mut g = self.lock();
        let mut n = 0;
        for r in snap.routes {
            let key = hash64(&format!("{}\u{1}{}", r.scope, r.shape));
            match g.routes.get_mut(&key) {
                Some(existing) => {
                    existing.observations = existing.observations.max(r.observations);
                    existing.hits += r.hits;
                    if existing.answer_template.is_none() {
                        existing.answer_template = r.answer_template;
                    }
                    if existing.vector.is_none() {
                        existing.vector = r.vector;
                    }
                }
                None => {
                    g.routes.insert(key, LearnedRoute { created: Instant::now(), ..r });
                    n += 1;
                }
            }
        }
        n
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let snap = self.export();
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(&snap)?)?;
        std::fs::rename(tmp, path)
    }

    pub fn load(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let data = std::fs::read(path)?;
        let snap: RouteSnapshot = serde_json::from_slice(&data)?;
        Ok(self.import(snap))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn stats(&self) -> LearnStats {
        let g = self.lock();
        let mut s = g.stats.clone();
        s.routes = g.routes.len();
        s
    }

    /// Generalise one observed (question -> tool call [+ answer template]) pair.
    /// Returns the shape learned, or None when no argument value appears in the
    /// question (nothing to generalise) or the arguments are not JSON.
    #[allow(clippy::too_many_arguments)]
    pub fn learn(&self, scope: u64, question: &str, tool: &str, arguments: &str, answer_template: Option<&str>, needs_analysis: bool, vector: Option<Vec<f32>>) -> Option<String> {
        let args: serde_json::Value = serde_json::from_str(arguments).ok()?;
        let all = self.spans(question);
        // Keep only spans whose value appears among the argument values (case-insensitive).
        let mut arg_values: Vec<(String, String)> = Vec::new(); // (json pointer path, value)
        collect_scalars(&args, "", &mut arg_values);
        let mut used: Vec<Span> = Vec::new();
        let mut slot_of_path: HashMap<String, usize> = HashMap::new();
        for sp in &all {
            let matches: Vec<&(String, String)> = arg_values.iter().filter(|(_, v)| v.eq_ignore_ascii_case(&sp.value)).collect();
            if matches.is_empty() {
                continue;
            }
            let idx = used.len();
            for (path, _) in matches {
                slot_of_path.entry(path.clone()).or_insert(idx);
            }
            used.push(sp.clone());
        }
        if used.is_empty() {
            return None;
        }
        // Every scalar argument that is not a slot must be constant; that is fine, it is
        // part of the template. Build the args template by replacing slot values by path.
        let mut templ = args.clone();
        for (path, idx) in &slot_of_path {
            set_path(&mut templ, path, serde_json::Value::String(format!("{{{{s{idx}}}}}")));
        }
        let shape = abstract_question(question, &used);
        let slot_types: Vec<SlotType> = used.iter().map(|s| s.kind).collect();
        let key = hash64(&format!("{scope}\u{1}{shape}"));
        let mut g = self.lock();
        g.routes.retain(|_, r| r.created.elapsed() < self.ttl);
        let entry = g.routes.entry(key).or_insert_with(|| LearnedRoute {
            shape: shape.clone(), slot_types: slot_types.clone(), tool: tool.to_string(), args_template: templ.to_string(),
            answer_template: answer_template.map(str::to_string), needs_analysis, observations: 0, hits: 0, created: Instant::now(), vector: None, scope,
            successes: 0, failures: 0, quarantined: false, quarantine_reason: None,
        });
        entry.observations += 1;
        entry.tool = tool.to_string();
        entry.args_template = templ.to_string();
        if let Some(t) = answer_template {
            // Keep whichever template generalises more (more placeholders).
            let better = entry.answer_template.as_ref().is_none_or(|e| t.matches("{{").count() >= e.matches("{{").count());
            if better {
                entry.answer_template = Some(t.to_string());
                entry.needs_analysis = needs_analysis;
            }
        }
        if vector.is_some() {
            entry.vector = vector;
        }
        g.stats.learned += 1;
        if g.routes.len() > self.max_routes {
            // Drop the least used route.
            if let Some((&k, _)) = g.routes.iter().min_by_key(|(_, r)| (r.hits, r.observations)) {
                g.routes.remove(&k);
            }
        }
        Some(shape)
    }

    /// Find a route for a new question: exact shape first, then semantic on the
    /// abstracted text when a vector is supplied. Slot types must line up exactly.
    /// Text slots are noisy (acronyms, proper nouns), so the question is tried
    /// with them and without them.
    pub fn fill(&self, scope: u64, question: &str, vector: Option<&[f32]>) -> Option<Fill> {
        let all = self.spans(question);
        let no_text: Vec<Span> = all.iter().filter(|s| s.kind != SlotType::Text).cloned().collect();
        let candidates: Vec<&Vec<Span>> = if no_text.len() == all.len() { vec![&all] } else { vec![&all, &no_text] };
        let mut g = self.lock();
        let mut chosen: Option<(u64, Option<f64>, Vec<Span>)> = None;
        'outer: for spans in &candidates {
            let shape = abstract_question(question, spans);
            let types: Vec<SlotType> = spans.iter().map(|s| s.kind).collect();
            let key = hash64(&format!("{scope}\u{1}{shape}"));
            if let Some(r) = g.routes.get(&key) {
                if !r.quarantined && r.created.elapsed() < self.ttl && r.observations >= self.min_observations && r.slot_types == types {
                    chosen = Some((key, None, (*spans).clone()));
                    break 'outer;
                }
            }
        }
        if chosen.is_none() {
            if let Some(v) = vector {
                let mut best: Option<(u64, f64, Vec<Span>)> = None;
                for spans in &candidates {
                    let types: Vec<SlotType> = spans.iter().map(|s| s.kind).collect();
                    for (k, r) in g.routes.iter() {
                        if r.quarantined || r.created.elapsed() >= self.ttl || r.observations < self.min_observations || r.slot_types != types {
                            continue;
                        }
                        if let Some(rv) = &r.vector {
                            let sim = cosine(v, rv);
                            if sim >= self.min_similarity && best.as_ref().is_none_or(|b| sim > b.1) {
                                best = Some((*k, sim, (*spans).clone()));
                            }
                        }
                    }
                }
                chosen = best.map(|(k, s, sp)| (k, Some(s), sp));
            }
        }
        let Some((k, similarity, spans)) = chosen else {
            if !all.is_empty() {
                g.stats.rejected_shape += 1;
            }
            return None;
        };
        let fill = {
            let r = g.routes.get_mut(&k)?;
            let mut arguments = r.args_template.clone();
            for (i, sp) in spans.iter().enumerate() {
                let val = serde_json::to_string(&sp.value).unwrap_or_default();
                arguments = arguments.replace(&format!("\"{{{{s{i}}}}}\""), &val).replace(&format!("{{{{s{i}}}}}"), val.trim_matches('"'));
            }
            r.hits += 1;
            Fill { key: k, tool: r.tool.clone(), arguments, answer_template: r.answer_template.clone(), needs_analysis: r.needs_analysis, shape: r.shape.clone(), similarity }
        };
        g.stats.served += 1;
        Some(fill)
    }

    /// Feedback from a served call's tool result. Returns Some(shape) when this
    /// failure pushed the route into quarantine.
    pub fn feedback(&self, key: u64, ok: bool, quarantine_after: u32) -> Option<String> {
        let mut g = self.lock();
        let r = g.routes.get_mut(&key)?;
        if ok {
            r.successes += 1;
            return None;
        }
        r.failures += 1;
        if !r.quarantined && r.failures >= quarantine_after as u64 && r.failures > r.successes {
            r.quarantined = true;
            r.quarantine_reason = Some(format!("{} error results against {} clean", r.failures, r.successes));
            return Some(r.shape.clone());
        }
        None
    }

    /// Manually or offline-driven quarantine / release by shape.
    pub fn set_quarantine(&self, shape: &str, on: bool, reason: &str) -> usize {
        let mut g = self.lock();
        let mut n = 0;
        for r in g.routes.values_mut().filter(|r| r.shape == shape) {
            r.quarantined = on;
            r.quarantine_reason = on.then(|| reason.to_string());
            if !on {
                r.failures = 0;
            }
            n += 1;
        }
        n
    }

    /// Learn directly into a known scope (offline miner path).
    #[allow(clippy::too_many_arguments)]
    pub fn learn_scoped(&self, scope: u64, question: &str, tool: &str, arguments: &str, answer_template: Option<&str>, needs_analysis: bool) -> Option<String> {
        self.learn(scope, question, tool, arguments, answer_template, needs_analysis, None)
    }

    pub fn routes(&self) -> Vec<LearnedRoute> {
        let g = self.lock();
        let mut v: Vec<LearnedRoute> = g.routes.values().cloned().collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.hits));
        v
    }
}

/// Whether a tool call's arguments are flat typed parameters safe to generalise:
/// no nested objects/arrays, no denied (code-like) names, no long strings, no
/// newlines or statement separators inside strings.
pub fn learnable_arguments(arguments: &str, deny_names: &[String], max_len: usize) -> Result<(), String> {
    let v: serde_json::Value = serde_json::from_str(arguments).map_err(|_| "arguments are not JSON".to_string())?;
    let obj = v.as_object().ok_or_else(|| "arguments are not an object".to_string())?;
    for (k, x) in obj {
        let kl = k.to_lowercase();
        if deny_names.iter().any(|d| kl == d.to_lowercase() || kl.ends_with(&format!("_{}", d.to_lowercase()))) {
            return Err(format!("argument '{k}' is free-text code"));
        }
        match x {
            serde_json::Value::String(s) => {
                if s.chars().count() > max_len {
                    return Err(format!("argument '{k}' longer than {max_len} chars"));
                }
                if s.contains('\n') || s.contains(';') || s.contains("--") || s.contains("/*") {
                    return Err(format!("argument '{k}' looks like code"));
                }
            }
            serde_json::Value::Number(_) | serde_json::Value::Bool(_) | serde_json::Value::Null => {}
            _ => return Err(format!("argument '{k}' is nested; only flat parameters are learned")),
        }
    }
    Ok(())
}

fn collect_scalars(v: &serde_json::Value, path: &str, out: &mut Vec<(String, String)>) {
    match v {
        serde_json::Value::Object(o) => {
            for (k, x) in o {
                collect_scalars(x, &format!("{path}/{k}"), out);
            }
        }
        serde_json::Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                collect_scalars(x, &format!("{path}/{i}"), out);
            }
        }
        serde_json::Value::String(s) => out.push((path.to_string(), s.clone())),
        serde_json::Value::Number(n) => out.push((path.to_string(), n.to_string())),
        _ => {}
    }
}

fn set_path(v: &mut serde_json::Value, path: &str, new: serde_json::Value) {
    if let Some(slot) = v.pointer_mut(path) {
        *slot = new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_and_shapes() {
        let q = "Pull call activity for T-112 for 2026-08, top 5 reps in East.";
        let s = typed_spans(q);
        let kinds: Vec<SlotType> = s.iter().map(|x| x.kind).collect();
        assert_eq!(kinds, vec![SlotType::Code, SlotType::Date, SlotType::Num, SlotType::Text], "{s:?}");
        assert_eq!(abstract_question(q, &s), "pull call activity for <code> for <date>, top <num> reps in <text>");
    }

    #[test]
    fn learn_then_fill_new_entities() {
        let m = RouteMemory::new(600, 100, 0.9, 1);
        let shape = m.learn(7, "Pull call activity for T-112 for 2026-08.", "exec_query", r#"{"template":"kpi","territory":"T-112","month":"2026-08","limit":10}"#, Some("{{territory}}: {{calls}} calls"), false, None).unwrap();
        assert_eq!(shape, "pull call activity for <code> for <date>");
        let f = m.fill(7, "pull call activity for T-207 for 2026-09", None).expect("same shape, new entities");
        let args: serde_json::Value = serde_json::from_str(&f.arguments).unwrap();
        assert_eq!(args, serde_json::json!({"template": "kpi", "territory": "T-207", "month": "2026-09", "limit": 10}));
        assert_eq!(f.tool, "exec_query");
        assert_eq!(f.answer_template.as_deref(), Some("{{territory}}: {{calls}} calls"));
        // Different slot types or a different scope: no fill.
        assert!(m.fill(7, "pull call activity for East for 2026-09", None).is_none());
        assert!(m.fill(8, "pull call activity for T-207 for 2026-09", None).is_none());
        // Nothing to generalise when no argument value appears in the question.
        assert!(m.learn(7, "How are we doing?", "exec_query", r#"{"template":"overview"}"#, None, false, None).is_none());
        assert_eq!(m.stats().served, 1);
    }

    #[test]
    fn code_like_arguments_are_never_learnable() {
        let deny: Vec<String> = ["code", "sql", "query"].iter().map(|s| s.to_string()).collect();
        assert!(learnable_arguments(r#"{"template":"kpi","territory":"T-112","limit":10}"#, &deny, 80).is_ok());
        assert!(learnable_arguments(r#"{"code":"var x = Query<JsonObject>(\"select * from t where terr='T-112'\");"}"#, &deny, 80).is_err());
        assert!(learnable_arguments(r#"{"raw_sql":"select 1"}"#, &deny, 80).is_err(), "suffix match on denied names");
        assert!(learnable_arguments(r#"{"filter":"a; drop table x"}"#, &deny, 80).is_err(), "statement separator");
        assert!(learnable_arguments(&format!(r#"{{"note":"{}"}}"#, "x".repeat(200)), &deny, 80).is_err(), "too long");
        assert!(learnable_arguments(r#"{"params":{"a":1}}"#, &deny, 80).is_err(), "nested");
    }

    #[test]
    fn vocabulary_makes_product_and_region_names_slots() {
        let vocab: Vec<String> = ["Zelora", "Cardivex", "east", "West Coast"].iter().map(|s| s.to_string()).collect();
        let m = RouteMemory::with_vocab(600, 100, 0.9, 1, vocab);
        m.learn(1, "nrx for zelora in east for 2026-03", "kpi", r#"{"product":"zelora","region":"east","month":"2026-03"}"#, None, false, None).unwrap();
        let f = m.fill(1, "nrx for Cardivex in West Coast for 2026-04", None).expect("names are slots regardless of case");
        let a: serde_json::Value = serde_json::from_str(&f.arguments).unwrap();
        assert_eq!(a, serde_json::json!({"product": "Cardivex", "region": "West Coast", "month": "2026-04"}));
    }

    #[test]
    fn snapshot_round_trip() {
        let m = RouteMemory::new(600, 100, 0.9, 1);
        m.learn(3, "calls for T-001 in 2026-01", "kpi", r#"{"t":"T-001","m":"2026-01"}"#, Some("{{calls}}"), false, Some(vec![0.5, 0.5])).unwrap();
        let snap = m.export();
        assert_eq!(snap.routes.len(), 1);
        let m2 = RouteMemory::new(600, 100, 0.9, 1);
        assert_eq!(m2.import(serde_json::from_str(&serde_json::to_string(&snap).unwrap()).unwrap()), 1);
        assert!(m2.fill(3, "calls for T-002 in 2026-02", None).is_some());
        let dir = std::env::temp_dir().join(format!("pankh-routes-{}.json", std::process::id()));
        m2.save(&dir).unwrap();
        let m3 = RouteMemory::new(600, 100, 0.9, 1);
        assert_eq!(m3.load(&dir).unwrap(), 1);
        let _ = std::fs::remove_file(dir);
    }

    #[test]
    fn semantic_fill_requires_matching_slot_types() {
        let m = RouteMemory::new(600, 100, 0.9, 1);
        m.learn(1, "Call activity for T-112 in 2026-08?", "exec_query", r#"{"territory":"T-112","month":"2026-08"}"#, None, false, Some(vec![1.0, 0.0])).unwrap();
        let f = m.fill(1, "Show HCP calls for T-300 during 2026-01 please", Some(&[0.99, 0.05])).expect("semantic shape match");
        assert!(f.arguments.contains("T-300") && f.arguments.contains("2026-01"));
        assert!(f.similarity.unwrap() > 0.9);
        assert!(m.fill(1, "Show HCP calls for T-300 during 2026-01 and 2026-02", Some(&[0.99, 0.05])).is_none(), "extra slot must not be guessed");
    }
}
