//! Deterministic slot filling: typed parameters from the words of a question,
//! with no model. Used after a decision engine picks an operation or a tool.
//! Fills only when the assignment is unambiguous; otherwise it says what is
//! missing so the router can use a slower path.

use serde_json::{json, Map, Value};

use crate::learn::{typed_spans_with_vocab, Span, SlotType};

#[derive(Debug, Clone)]
pub struct Slot {
    pub name: String,
    /// string | integer | number | boolean | enum
    pub kind: String,
    pub values: Vec<String>,
    /// value -> other phrasings of it
    pub aliases: std::collections::BTreeMap<String, Vec<String>>,
    pub pattern: Option<regex::Regex>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub optional: bool,
    pub default: Option<Value>,
}

#[derive(Debug, PartialEq)]
pub enum FillError {
    /// Enum parameters the question does not name: a decision engine may resolve them.
    NeedsChoice(Vec<String>),
    /// Nothing or more than one candidate for this parameter.
    Ambiguous(String),
    Missing(String),
    /// A parameter type that cannot be filled from text (nested objects, arrays).
    Unfillable(String),
}

fn word_match(hay_lower: &str, needle: &str) -> Option<(usize, usize)> {
    let n = needle.to_lowercase();
    if n.is_empty() {
        return None;
    }
    let bytes = hay_lower.as_bytes();
    let mut from = 0;
    while let Some(p) = hay_lower[from..].find(&n) {
        let s = from + p;
        let e = s + n.len();
        let left = s == 0 || !bytes[s - 1].is_ascii_alphanumeric();
        let right = e >= hay_lower.len() || !bytes[e].is_ascii_alphanumeric() || (bytes[e] == b's' && (e + 1 >= hay_lower.len() || !bytes[e + 1].is_ascii_alphanumeric()));
        if left && right {
            return Some((s, e));
        }
        from = e;
    }
    None
}

/// Fill `slots` from `question`. `resolved` carries enum answers already decided
/// elsewhere (by a decision engine), keyed by slot name.
pub fn fill(question: &str, slots: &[Slot], vocab: &[String], resolved: &Map<String, Value>) -> Result<Map<String, Value>, FillError> {
    let lower = question.to_lowercase();
    let spans: Vec<Span> = typed_spans_with_vocab(question, vocab);
    let mut used: Vec<(usize, usize)> = Vec::new();
    let overlaps = |used: &Vec<(usize, usize)>, s: usize, e: usize| used.iter().any(|(a, b)| s < *b && e > *a);
    let mut out = Map::new();
    let mut need_choice = Vec::new();

    // 1. Enums: a value named in the question, or an engine answer.
    for sl in slots.iter().filter(|s| s.kind == "enum") {
        if let Some(v) = resolved.get(&sl.name) {
            out.insert(sl.name.clone(), v.clone());
            continue;
        }
        // A value is named by itself, by itself with spaces for underscores, or by an alias;
        // the longest phrase wins so "new prescriptions" is not also read as "prescriptions".
        let mut hits: Vec<(&String, (usize, usize))> = sl
            .values
            .iter()
            .filter_map(|v| {
                let mut forms: Vec<String> = vec![v.clone(), v.replace('_', " ")];
                if let Some(a) = sl.aliases.get(v) {
                    forms.extend(a.iter().cloned());
                }
                forms.iter().filter_map(|f| word_match(&lower, f)).max_by_key(|(s, e)| e - s).map(|m| (v, m))
            })
            .collect();
        // Drop hits contained in a longer hit of another value.
        let spans_of: Vec<(usize, usize)> = hits.iter().map(|(_, m)| *m).collect();
        hits.retain(|(_, (s, e))| !spans_of.iter().any(|(s2, e2)| s2 <= s && e <= e2 && (e2 - s2) > (e - s)));
        hits.dedup_by(|a, b| a.0 == b.0);
        match hits.len() {
            1 => {
                used.push(hits[0].1);
                out.insert(sl.name.clone(), json!(hits[0].0));
            }
            0 if sl.default.is_some() => {
                out.insert(sl.name.clone(), sl.default.clone().unwrap());
            }
            0 if sl.optional => {}
            _ => need_choice.push(sl.name.clone()),
        }
    }
    if !need_choice.is_empty() {
        return Err(FillError::NeedsChoice(need_choice));
    }

    // 2. Patterned strings (dates, codes, ids): candidates are spans matching the pattern.
    for sl in slots.iter().filter(|s| s.kind == "string" && s.pattern.is_some()) {
        let re = sl.pattern.as_ref().unwrap();
        let mut cands: Vec<&Span> = spans.iter().filter(|sp| re.is_match(&sp.value) && !overlaps(&used, sp.start, sp.end)).collect();
        cands.dedup_by(|a, b| a.value == b.value);
        match cands.len() {
            1 => {
                used.push((cands[0].start, cands[0].end));
                out.insert(sl.name.clone(), json!(cands[0].value));
            }
            0 if sl.default.is_some() => {
                out.insert(sl.name.clone(), sl.default.clone().unwrap());
            }
            0 if sl.optional => {}
            0 => return Err(FillError::Missing(sl.name.clone())),
            _ => return Err(FillError::Ambiguous(sl.name.clone())),
        }
    }

    // 3. Numbers: numeric spans in range.
    for sl in slots.iter().filter(|s| s.kind == "integer" || s.kind == "number") {
        let cands: Vec<(&Span, f64)> = spans
            .iter()
            .filter(|sp| sp.kind == SlotType::Num && !overlaps(&used, sp.start, sp.end))
            .filter_map(|sp| sp.value.trim_end_matches('%').replace(',', "").parse::<f64>().ok().map(|n| (sp, n)))
            .filter(|(_, n)| sl.min.is_none_or(|m| *n >= m) && sl.max.is_none_or(|m| *n <= m) && (sl.kind == "number" || n.fract() == 0.0))
            .collect();
        match cands.len() {
            1 => {
                used.push((cands[0].0.start, cands[0].0.end));
                let v = if sl.kind == "integer" { json!(cands[0].1 as i64) } else { json!(cands[0].1) };
                out.insert(sl.name.clone(), v);
            }
            0 if sl.default.is_some() => {
                out.insert(sl.name.clone(), sl.default.clone().unwrap());
            }
            0 if sl.optional => {}
            0 => return Err(FillError::Missing(sl.name.clone())),
            _ => return Err(FillError::Ambiguous(sl.name.clone())),
        }
    }

    // 4. Free strings: the one remaining code or name in the question.
    for sl in slots.iter().filter(|s| s.kind == "string" && s.pattern.is_none()) {
        let cands: Vec<&Span> = spans.iter().filter(|sp| matches!(sp.kind, SlotType::Code | SlotType::Text) && !overlaps(&used, sp.start, sp.end)).collect();
        match cands.len() {
            1 => {
                used.push((cands[0].start, cands[0].end));
                out.insert(sl.name.clone(), json!(cands[0].value));
            }
            0 if sl.default.is_some() => {
                out.insert(sl.name.clone(), sl.default.clone().unwrap());
            }
            0 if sl.optional => {}
            0 => return Err(FillError::Missing(sl.name.clone())),
            _ => return Err(FillError::Ambiguous(sl.name.clone())),
        }
    }

    for sl in slots.iter().filter(|s| s.kind == "boolean") {
        match (&sl.default, sl.optional) {
            (Some(d), _) => {
                out.insert(sl.name.clone(), d.clone());
            }
            (None, true) => {}
            (None, false) => return Err(FillError::Unfillable(sl.name.clone())),
        }
    }
    Ok(out)
}

/// Every phrase any of `slots` recognises in the question, by enum value, space form or
/// alias, with all the values it can stand for: "district" can be a ranking level and a
/// benchmark. Overlapping phrases resolve to the longest.
pub fn enum_evidence(question: &str, slots: &[Slot]) -> Vec<(usize, usize, Vec<String>)> {
    let lower = question.to_lowercase();
    let mut hits: Vec<(usize, usize, String)> = Vec::new();
    for sl in slots.iter().filter(|s| s.kind == "enum") {
        for v in &sl.values {
            let mut forms: Vec<String> = vec![v.clone(), v.replace('_', " ")];
            if let Some(a) = sl.aliases.get(v) {
                forms.extend(a.iter().cloned());
            }
            for f in forms {
                if let Some((s, e)) = word_match(&lower, &f) {
                    hits.push((s, e, v.to_lowercase()));
                }
            }
        }
    }
    hits.sort_by_key(|(s, e, _)| (*s, std::cmp::Reverse(*e)));
    let mut out: Vec<(usize, usize, Vec<String>)> = Vec::new();
    for (s, e, v) in hits {
        if let Some(existing) = out.iter_mut().find(|(s2, e2, _)| *s2 == s && *e2 == e) {
            if !existing.2.contains(&v) {
                existing.2.push(v);
            }
        } else if !out.iter().any(|(s2, e2, _)| s < *e2 && e > *s2) {
            out.push((s, e, vec![v]));
        }
    }
    out
}

/// True when every code, date and number in the question ended up in `params`:
/// the operation uses everything specific the question says.
pub fn uses_every_typed_value(question: &str, params: &Map<String, Value>, vocab: &[String]) -> bool {
    uses_every_typed_value_with(question, params, vocab, &[])
}

/// As above; a code, date or number inside a phrase the operation already uses (the "4"
/// in "past 4 weeks" when the window is last_4_weeks) counts as used.
pub fn uses_every_typed_value_with(question: &str, params: &Map<String, Value>, vocab: &[String], evidence: &[(usize, usize, Vec<String>)]) -> bool {
    let values: Vec<String> = params.values().map(|v| match v { Value::String(s) => s.to_lowercase(), other => other.to_string() }).collect();
    let used_phrases: Vec<(usize, usize)> = evidence.iter().filter(|(_, _, vs)| vs.iter().any(|v| values.contains(v))).map(|(s, e, _)| (*s, *e)).collect();
    typed_spans_with_vocab(question, vocab).iter().filter(|sp| matches!(sp.kind, SlotType::Code | SlotType::Date | SlotType::Num)).all(|sp| {
        if used_phrases.iter().any(|(s, e)| *s <= sp.start && sp.end <= *e) {
            return true;
        }
        let v = sp.value.to_lowercase();
        let n = v.trim_end_matches('%').replace(',', "").parse::<f64>().ok();
        values.iter().any(|x| *x == v || (n.is_some() && x.parse::<f64>().ok() == n))
    })
}

/// Slots from a tool's JSON schema. Returns Err when the tool takes nested or
/// array arguments, which cannot be filled from words.
pub fn slots_from_schema(schema: &Value) -> Result<Vec<Slot>, FillError> {
    let props = schema["properties"].as_object().cloned().unwrap_or_default();
    let required: Vec<&str> = schema["required"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
    let mut out = Vec::new();
    for (name, p) in props {
        let ty = p["type"].as_str().unwrap_or("string");
        let values: Vec<String> = p["enum"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
        let kind = match ty {
            _ if !values.is_empty() => "enum",
            "string" => "string",
            "integer" => "integer",
            "number" => "number",
            "boolean" => "boolean",
            _ => return Err(FillError::Unfillable(name)),
        };
        let pattern = p["pattern"]
            .as_str()
            .map(str::to_string)
            .or_else(|| match p["format"].as_str() {
                Some("date") => Some(r"\d{4}-\d{2}-\d{2}".into()),
                _ => None,
            })
            .and_then(|pat| regex::Regex::new(&format!("^(?:{pat})$")).ok());
        out.push(Slot {
            aliases: Default::default(),
            optional: !required.contains(&name.as_str()),
            default: p.get("default").cloned(),
            name,
            kind: kind.into(),
            values,
            pattern,
            min: p["minimum"].as_f64(),
            max: p["maximum"].as_f64(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(name: &str, kind: &str) -> Slot {
        Slot { name: name.into(), kind: kind.into(), values: vec![], aliases: Default::default(), pattern: None, min: None, max: None, optional: false, default: None }
    }

    #[test]
    fn aliases_name_values_and_longest_phrase_wins() {
        let mut m = s("metric", "enum");
        m.values = vec!["trx".into(), "nrx".into()];
        m.aliases.insert("trx".into(), vec!["total prescriptions".into(), "prescriptions".into()]);
        m.aliases.insert("nrx".into(), vec!["new prescriptions".into()]);
        assert_eq!(fill("total prescriptions for T-1", std::slice::from_ref(&m), &[], &Map::new()).unwrap()["metric"], "trx");
        assert_eq!(fill("new prescriptions for T-1", std::slice::from_ref(&m), &[], &Map::new()).unwrap()["metric"], "nrx");
    }

    #[test]
    fn fills_enums_patterns_numbers_and_names() {
        let mut metric = s("metric", "enum");
        metric.values = vec!["calls".into(), "reach".into(), "revenue".into()];
        let mut period = s("period", "string");
        period.pattern = Some(regex::Regex::new(r"^(?:\d{4}-\d{2})$").unwrap());
        let mut n = s("n", "integer");
        n.min = Some(1.0);
        n.max = Some(50.0);
        let entity = s("entity", "string");
        let slots = vec![metric, period, n, entity];
        let out = fill("Top 5 calls for T-112 in 2026-08", &slots, &[], &Map::new()).unwrap();
        assert_eq!(Value::Object(out), json!({"metric": "calls", "period": "2026-08", "n": 5, "entity": "T-112"}));
        // vocab names count as entities in any casing
        let out = fill("top 5 calls for zelora in 2026-08", &slots, &["Zelora".into()], &Map::new()).unwrap();
        assert_eq!(out["entity"], "zelora");
    }

    #[test]
    fn refuses_to_guess() {
        let mut metric = s("metric", "enum");
        metric.values = vec!["calls".into(), "reach".into()];
        assert_eq!(fill("calls and reach for T-1", &[metric.clone()], &[], &Map::new()), Err(FillError::NeedsChoice(vec!["metric".into()])));
        let mut resolved = Map::new();
        resolved.insert("metric".into(), json!("reach"));
        assert_eq!(fill("calls and reach for T-1", &[metric], &[], &resolved).unwrap()["metric"], "reach");
        let entity = s("entity", "string");
        assert_eq!(fill("compare T-112 and T-207", std::slice::from_ref(&entity), &[], &Map::new()), Err(FillError::Ambiguous("entity".into())));
        assert_eq!(fill("how are things", &[entity], &[], &Map::new()), Err(FillError::Missing("entity".into())));
    }

    #[test]
    fn exact_structural_fit() {
        let mut p = Map::new();
        p.insert("entity".into(), json!("T-112"));
        p.insert("period".into(), json!("2026-08"));
        assert!(uses_every_typed_value("calls for T-112 in 2026-08", &p, &[]));
        assert!(!uses_every_typed_value("top 5 by calls for T-112 in 2026-08", &p, &[]), "5 unused");
        p.insert("n".into(), json!(5));
        assert!(uses_every_typed_value("top 5 by calls for T-112 in 2026-08", &p, &[]));
    }

    #[test]
    fn schema_to_slots() {
        let schema = json!({"type": "object", "properties": {
            "territory": {"type": "string", "pattern": "[A-Z]-\\d{3}"},
            "month": {"type": "string", "pattern": "\\d{4}-\\d{2}"},
            "metric": {"type": "string", "enum": ["calls", "reach"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10}},
            "required": ["territory", "month", "metric"]});
        let slots = slots_from_schema(&schema).unwrap();
        let out = fill("reach for T-112 in 2026-08", &slots, &[], &Map::new()).unwrap();
        assert_eq!(Value::Object(out), json!({"territory": "T-112", "month": "2026-08", "metric": "reach", "limit": 10}));
        assert!(slots_from_schema(&json!({"properties": {"filters": {"type": "array"}}})).is_err());
    }
}
