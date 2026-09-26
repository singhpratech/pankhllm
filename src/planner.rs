//! Planner lane. The model reads; the executor runs; the router checks.
//!
//! 1. A learned plan route for the question's shape fills the plan with no model.
//! 2. Otherwise one small structured-output call picks an operation from a closed
//!    catalog and fills typed parameters (or says UNSUPPORTED).
//! 3. The plan is validated here: known operation, known parameters, types,
//!    enums, patterns, ranges. Nothing the user typed becomes an identifier.
//!
//! 4. Your executor (any language, one HTTP endpoint) runs it with the user's
//!    identity and returns data; the router renders the answer.
//!
//! Any failure falls through to the normal lanes, so the planner can only help.

use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::config::{PlanOperation, PlanParam, PlannerConfig};

/// A catalog with enum files resolved and patterns compiled.
pub struct Catalog {
    pub ops: Vec<PlanOperation>,
    patterns: HashMap<(String, String), regex::Regex>,
}

impl Catalog {
    pub fn load(cfg: &PlannerConfig) -> anyhow::Result<Self> {
        let mut ops = cfg.operations.clone();
        if let Some(path) = &cfg.catalog {
            let raw = std::fs::read_to_string(path)?;
            #[derive(serde::Deserialize)]
            struct File {
                operations: Vec<PlanOperation>,
            }
            let f: File = if path.ends_with(".json") { serde_json::from_str(&raw)? } else { serde_yaml::from_str(&raw)? };
            ops.extend(f.operations);
        }
        let mut patterns = HashMap::new();
        for op in ops.iter_mut() {
            for (k, p) in op.params.iter_mut() {
                if let Some(f) = &p.values_file {
                    let text = std::fs::read_to_string(f)?;
                    p.values.extend(text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string));
                }
                if let Some(pat) = &p.pattern {
                    patterns.insert((op.name.clone(), k.clone()), regex::Regex::new(&format!("^(?:{pat})$"))?);
                }
            }
        }
        Ok(Self { ops, patterns })
    }

    /// Slots for deterministic filling, with patterns compiled.
    pub fn slots(&self, op: &PlanOperation) -> Vec<crate::slots::Slot> {
        op.params
            .iter()
            .map(|(k, p)| crate::slots::Slot {
                name: k.clone(),
                kind: p.kind.clone(),
                values: p.values.clone(),
                aliases: p.aliases.clone(),
                pattern: self.patterns.get(&(op.name.clone(), k.clone())).cloned(),
                min: p.min,
                max: p.max,
                optional: p.optional,
                default: p.default.clone(),
            })
            .collect()
    }

    pub fn op(&self, name: &str) -> Option<&PlanOperation> {
        self.ops.iter().find(|o| o.name == name)
    }

    /// JSON schema for the planner's structured output: one branch per operation.
    pub fn schema(&self) -> Value {
        let mut branches: Vec<Value> = self
            .ops
            .iter()
            .map(|op| {
                let mut props = Map::new();
                let mut required = Vec::new();
                for (k, p) in &op.params {
                    props.insert(k.clone(), param_schema(p));
                    // Strict mode wants every property required; optional ones allow null.
                    required.push(Value::String(k.clone()));
                }
                json!({
                    "type": "object",
                    "properties": {"op": {"type": "string", "enum": [op.name]}, "params": {"type": "object", "properties": props, "required": required, "additionalProperties": false}},
                    "required": ["op", "params"],
                    "additionalProperties": false
                })
            })
            .collect();
        branches.push(json!({"type": "object", "properties": {"op": {"type": "string", "enum": ["UNSUPPORTED"]}, "params": {"type": "object", "properties": {}, "required": [], "additionalProperties": false}}, "required": ["op", "params"], "additionalProperties": false}));
        json!({"type": "object", "properties": {"plan": {"anyOf": branches}}, "required": ["plan"], "additionalProperties": false})
    }

    /// System prompt: the catalog in plain words.
    pub fn instructions(&self) -> String {
        let mut s = String::from(
            "You translate one question into one plan over a fixed catalog of operations. You never answer the question and never invent values.\n\
             Choose the single operation that answers the question exactly and fill its parameters from the question.\n\
             If no operation answers it exactly, or the question needs explanation, judgement, prediction or a calculation the operation does not perform, return op UNSUPPORTED.\n\
             Optional parameters the question does not mention are null.\n\nOperations:\n",
        );
        for op in &self.ops {
            s.push_str(&format!("- {}: {}\n", op.name, op.description));
            for (k, p) in &op.params {
                let mut line = format!("    {} ({}{})", k, p.kind, if p.optional { ", optional" } else { "" });
                if !p.description.is_empty() {
                    line.push_str(&format!(": {}", p.description));
                }
                if p.kind == "enum" && p.values.len() <= 40 {
                    line.push_str(&format!(" one of [{}]", p.values.join(", ")));
                    let al: Vec<String> = p.aliases.iter().map(|(v, a)| format!("{v} = {}", a.join(" / "))).collect();
                    if !al.is_empty() {
                        line.push_str(&format!(" (aliases: {})", al.join("; ")));
                    }
                }
                s.push_str(&line);
                s.push('\n');
            }
        }
        s
    }

    /// Check a plan against the catalog. Returns canonical params (defaults filled,
    /// enums in canonical spelling) or the reason it is rejected.
    pub fn validate(&self, op_name: &str, params: &Value) -> Result<Value, String> {
        let op = self.op(op_name).ok_or_else(|| format!("unknown operation {op_name}"))?;
        let given = params.as_object().cloned().unwrap_or_default();
        for k in given.keys() {
            if !op.params.contains_key(k) {
                return Err(format!("unknown parameter {k}"));
            }
        }
        let mut out = Map::new();
        for (k, p) in &op.params {
            let v = given.get(k).cloned().filter(|v| !v.is_null()).or_else(|| p.default.clone());
            let Some(v) = v else {
                if p.optional {
                    continue;
                }
                return Err(format!("missing parameter {k}"));
            };
            out.insert(k.clone(), self.check(op, k, p, v)?);
        }
        Ok(Value::Object(out))
    }

    fn check(&self, op: &PlanOperation, k: &str, p: &PlanParam, v: Value) -> Result<Value, String> {
        let range = |n: f64| -> Result<(), String> {
            if p.min.is_some_and(|m| n < m) || p.max.is_some_and(|m| n > m) {
                return Err(format!("{k} out of range"));
            }
            Ok(())
        };
        match p.kind.as_str() {
            "integer" => {
                let n = v.as_i64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).ok_or_else(|| format!("{k} is not an integer"))?;
                range(n as f64)?;
                Ok(json!(n))
            }
            "number" => {
                let n = v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).ok_or_else(|| format!("{k} is not a number"))?;
                range(n)?;
                Ok(json!(n))
            }
            "boolean" => v.as_bool().map(Value::Bool).ok_or_else(|| format!("{k} is not a boolean")),
            "enum" => {
                let s = v.as_str().ok_or_else(|| format!("{k} is not a string"))?;
                p.values.iter().find(|x| x.eq_ignore_ascii_case(s.trim())).map(|x| json!(x)).ok_or_else(|| format!("{k}={s} is not an allowed value"))
            }
            _ => {
                let s = v.as_str().ok_or_else(|| format!("{k} is not a string"))?.trim().to_string();
                if s.chars().count() > p.max_len.unwrap_or(80) {
                    return Err(format!("{k} too long"));
                }
                if s.contains(['\n', ';']) || s.contains("--") || s.contains("/*") {
                    return Err(format!("{k} contains control syntax"));
                }
                if let Some(re) = self.patterns.get(&(op.name.clone(), k.to_string())) {
                    if !re.is_match(&s) {
                        return Err(format!("{k} does not match its pattern"));
                    }
                }
                Ok(json!(s))
            }
        }
    }
}

fn param_schema(p: &PlanParam) -> Value {
    let base = match p.kind.as_str() {
        "integer" => json!({"type": "integer"}),
        "number" => json!({"type": "number"}),
        "boolean" => json!({"type": "boolean"}),
        "enum" if p.values.len() <= 500 => json!({"type": "string", "enum": p.values}),
        _ => json!({"type": "string"}),
    };
    if p.optional {
        json!({"anyOf": [base, {"type": "null"}], "description": p.description})
    } else {
        let mut b = base;
        b["description"] = json!(p.description);
        b
    }
}

/// Pull `{"plan": {...}}` (or a bare `{op, params}`) out of model text.
pub fn parse_plan(text: &str) -> Option<(String, Value)> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    let v: Value = serde_json::from_str(&text[start..=end]).ok()?;
    let plan = if v.get("plan").is_some() { &v["plan"] } else { &v };
    Some((plan["op"].as_str()?.to_string(), plan.get("params").cloned().unwrap_or(json!({}))))
}

/// Result of the executor call.
pub struct Executed {
    pub data: Value,
    pub answer: Option<String>,
}

#[allow(clippy::too_many_arguments)]
pub async fn execute(http: &reqwest::Client, cfg: &PlannerConfig, op: &str, params: &Value, question: &str, user: Option<&str>, tenant: Option<&str>, trace_id: &str, deadline: Instant) -> Result<Executed, String> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    let timeout = Duration::from_millis(cfg.executor.timeout_ms).min(remaining);
    if timeout < Duration::from_millis(50) {
        return Err("no time left for the executor".into());
    }
    let mut rq = http
        .post(&cfg.executor.url)
        .timeout(timeout)
        .json(&json!({"op": op, "params": params, "question": question, "user": user, "tenant": tenant, "trace_id": trace_id}));
    if let Some(u) = user {
        rq = rq.header("x-pankh-user", u);
    }
    if let Some(t) = tenant {
        rq = rq.header("x-pankh-tenant", t);
    }
    if let Some(k) = cfg.executor.api_key_env.as_ref().and_then(|e| std::env::var(e).ok()) {
        rq = rq.bearer_auth(k);
    }
    let resp = rq.send().await.map_err(|e| format!("executor unreachable: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("executor {status}: {}", crate::providers::truncate(&text, 200)));
    }
    let v: Value = serde_json::from_str(&text).map_err(|_| "executor did not return JSON".to_string())?;
    if v.get("ok") == Some(&Value::Bool(false)) || v.get("error").is_some_and(|e| !e.is_null()) {
        return Err(format!("executor error: {}", v["error"]));
    }
    let data = v.get("data").cloned().unwrap_or_else(|| v.clone());
    Ok(Executed { data, answer: v["answer"].as_str().map(str::to_string) })
}

/// Final text: the executor's own answer, else the operation's template, else a table
/// from `rows`, else compact JSON. Returns None when a template placeholder is missing.
pub fn render(op: &PlanOperation, params: &Value, ex: &Executed, max_rows: usize) -> Option<String> {
    if let Some(a) = ex.answer.as_ref().filter(|a| !a.trim().is_empty()) {
        return Some(a.clone());
    }
    let mut merged = match &ex.data {
        Value::Object(o) => Value::Object(o.clone()),
        other => json!({"rows": other}),
    };
    merged["params"] = params.clone();
    merged["args"] = params.clone();
    if let Some(t) = &op.answer_template {
        let placeholders = t.matches("{{").count();
        if crate::router::count_filled(t, &merged, "", max_rows) < placeholders {
            return None;
        }
        return Some(crate::router::render_template(t, &merged, "", &HashMap::new(), max_rows));
    }
    if merged.get("rows").is_some_and(|r| r.is_array()) {
        return Some(crate::router::render_template("{{rows|table}}", &merged, "", &HashMap::new(), max_rows));
    }
    Some(serde_json::to_string_pretty(&ex.data).unwrap_or_default())
}

/// Stable, sorted JSON for learning and cache keys.
pub fn canonical(params: &Value) -> String {
    match params.as_object() {
        Some(o) => serde_json::to_string(&o.iter().collect::<BTreeMap<_, _>>()).unwrap_or_default(),
        None => params.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat() -> Catalog {
        let y = r#"
model: small
executor: { url: "http://x" }
operations:
  - name: metric_by_period
    description: One metric for one entity and period.
    params:
      metric: { type: enum, values: [calls, reach, revenue] }
      entity: { type: string, max_len: 40 }
      period: { type: string, pattern: '\d{4}-\d{2}' }
      n: { type: integer, min: 1, max: 50, default: 10 }
      region: { type: string, optional: true }
    answer_template: "{{params.entity}} {{params.metric}} in {{params.period}}: {{value|N0}}"
"#;
        let cfg: PlannerConfig = serde_yaml::from_str(y).unwrap();
        Catalog::load(&cfg).unwrap()
    }

    #[test]
    fn validation_is_strict_and_canonicalises() {
        let c = cat();
        let ok = c.validate("metric_by_period", &json!({"metric": "REACH", "entity": "T-112", "period": "2026-08"})).unwrap();
        assert_eq!(ok, json!({"metric": "reach", "entity": "T-112", "period": "2026-08", "n": 10}));
        assert!(c.validate("metric_by_period", &json!({"metric": "margin", "entity": "x", "period": "2026-08"})).is_err(), "enum");
        assert!(c.validate("metric_by_period", &json!({"metric": "calls", "entity": "x", "period": "August"})).is_err(), "pattern");
        assert!(c.validate("metric_by_period", &json!({"metric": "calls", "entity": "x; drop table t", "period": "2026-08"})).is_err(), "control syntax");
        assert!(c.validate("metric_by_period", &json!({"metric": "calls", "entity": "x", "period": "2026-08", "n": 500})).is_err(), "range");
        assert!(c.validate("metric_by_period", &json!({"metric": "calls", "entity": "x", "period": "2026-08", "table": "users"})).is_err(), "unknown param");
        assert!(c.validate("drop_everything", &json!({})).is_err(), "unknown op");
        assert!(c.validate("metric_by_period", &json!({"metric": "calls", "period": "2026-08"})).is_err(), "missing required");
    }

    #[test]
    fn schema_and_parse_and_render() {
        let c = cat();
        let s = c.schema();
        assert_eq!(s["properties"]["plan"]["anyOf"].as_array().unwrap().len(), 2);
        assert!(c.instructions().contains("metric_by_period") && c.instructions().contains("one of [calls, reach, revenue]"));
        let (op, params) = parse_plan("Sure: {\"plan\":{\"op\":\"metric_by_period\",\"params\":{\"metric\":\"calls\"}}}").unwrap();
        assert_eq!(op, "metric_by_period");
        assert_eq!(params["metric"], "calls");
        let p = c.validate("metric_by_period", &json!({"metric": "calls", "entity": "T-112", "period": "2026-08"})).unwrap();
        let ex = Executed { data: json!({"value": 1842}), answer: None };
        assert_eq!(render(c.op("metric_by_period").unwrap(), &p, &ex, 20).unwrap(), "T-112 calls in 2026-08: 1,842");
        let ex = Executed { data: json!({"other": 1}), answer: None };
        assert!(render(c.op("metric_by_period").unwrap(), &p, &ex, 20).is_none(), "missing value must fail open");
    }
}
