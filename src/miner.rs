//! The overnight miner: reads the trace store, learns routes from clean turns the
//! live path did not learn, quarantines routes that keep failing, measures every
//! lane and model, proposes vocabulary and tuning, and writes a report.
//!
//! Nothing here runs on the request path.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::router::Router;
use crate::store::{now_ms, read_heal, read_traces, Trace};

/// Tuning the miner measured; applied at start when `heal.autotune` is set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tuning {
    pub hedge_after_ms: Option<u64>,
    /// Lowest decision-engine confidence whose verdicts agreed with the slow path often enough.
    #[serde(default)]
    pub decision_min_confidence: Option<f64>,
    /// model name -> timeout in seconds
    pub timeouts_secs: BTreeMap<String, u64>,
    pub measured_at_ms: i64,
}

impl Tuning {
    pub fn apply(&self, cfg: &mut Config) {
        if let Some(h) = self.hedge_after_ms {
            if cfg.routing.hedge.after_ms.is_some() {
                cfg.routing.hedge.after_ms = Some(h);
            }
        }
        for m in cfg.models.iter_mut() {
            if let Some(t) = self.timeouts_secs.get(&m.name) {
                m.timeout_secs = Some(*t);
            }
        }
        if let (Some(c), Some(d)) = (self.decision_min_confidence, cfg.decisions.as_mut()) {
            d.min_confidence = c;
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LaneStat {
    pub served_by: String,
    pub requests: usize,
    pub model_free: bool,
    pub p50_ms: i64,
    pub p95_ms: i64,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Miss {
    pub shape: String,
    pub count: usize,
    pub example: Option<String>,
}

/// Agreement between the decision engine and the slow path at one confidence threshold.
#[derive(Debug, Clone, Serialize)]
pub struct Calibration {
    pub kind: String,
    pub threshold: f64,
    /// Shadow rows with engine confidence at or above the threshold.
    pub covered: usize,
    pub total: usize,
    /// Share of covered rows where the engine agreed with the slow path.
    pub precision: f64,
}

/// What the overnight training of pankhllm's own decision models produced.
#[derive(Debug, Clone, Serialize)]
pub struct NativeTraining {
    pub task: String,
    pub examples: usize,
    pub labels: Vec<String>,
    pub holdout_accuracy: Option<f64>,
    pub recommended_threshold: Option<f64>,
    pub saved: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub window_hours: u64,
    pub traces: usize,
    pub model_free_share: f64,
    pub p50_ms: i64,
    pub p95_ms: i64,
    pub cost_usd: f64,
    pub lanes: Vec<LaneStat>,
    pub top_misses: Vec<Miss>,
    pub learned_from_logs: usize,
    pub quarantined: Vec<String>,
    pub vocab_suggestions: Vec<(String, usize)>,
    pub tuning: Tuning,
    pub heal_events: usize,
    pub pruned: usize,
    pub decision_calibration: Vec<Calibration>,
    pub native: Vec<NativeTraining>,
}

fn pct(mut v: Vec<i64>, q: f64) -> i64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() as f64 - 1.0) * q).round() as usize]
}

/// Answered without any model call.
pub fn model_free(served_by: &str) -> bool {
    served_by == "learned-route"
        || served_by == "speculative-template"
        || served_by.starts_with("rule:")
        || served_by.starts_with("cache:")
        || served_by.starts_with("plan-route:")
        || served_by.starts_with("decision:")
        || served_by.starts_with("decision-tool:")
}

pub fn mine(router: &Router, since_hours: u64, apply: bool) -> anyhow::Result<Report> {
    let store = router.store.as_ref().ok_or_else(|| anyhow::anyhow!("no `store` configured: nothing to mine"))?;
    store.flush();
    let conn = store.reader()?;
    let since = now_ms() - (since_hours as i64) * 3_600_000;
    let all_rows: Vec<Trace> = read_traces(&conn, since, 5_000_000)?;
    let (shadow, rest): (Vec<Trace>, Vec<Trace>) = all_rows.into_iter().partition(|t| t.surface.starts_with("shadow:"));
    // Teacher labels train the native models below; they are not requests.
    let traces: Vec<Trace> = rest.into_iter().filter(|t| !t.surface.starts_with("label:")).collect();
    let model_names: HashSet<&str> = router.cfg.models.iter().map(|m| m.name.as_str()).collect();

    // ---- lanes ----
    let mut by: HashMap<String, Vec<&Trace>> = HashMap::new();
    for t in &traces {
        if t.outcome != "error" {
            by.entry(t.served_by.clone()).or_default().push(t);
        }
    }
    let mut lanes: Vec<LaneStat> = by
        .iter()
        .map(|(k, v)| LaneStat {
            served_by: k.clone(),
            requests: v.len(),
            model_free: model_free(k),
            p50_ms: pct(v.iter().map(|t| t.latency_ms).collect(), 0.5),
            p95_ms: pct(v.iter().map(|t| t.latency_ms).collect(), 0.95),
            cost_usd: v.iter().map(|t| t.cost_usd).sum(),
        })
        .collect();
    lanes.sort_by_key(|l| std::cmp::Reverse(l.requests));
    let ok: Vec<&Trace> = traces.iter().filter(|t| t.outcome != "error").collect();
    let free = ok.iter().filter(|t| model_free(&t.served_by)).count();

    // ---- misses: shapes that needed a model ----
    let mut miss: HashMap<&str, (usize, Option<&str>)> = HashMap::new();
    for t in ok.iter().filter(|t| model_names.contains(t.served_by.as_str()) && t.result_for.is_none()) {
        let e = miss.entry(t.shape.as_str()).or_insert((0, None));
        e.0 += 1;
        if e.1.is_none() {
            e.1 = t.question.as_deref();
        }
    }
    let mut top_misses: Vec<Miss> = miss.into_iter().map(|(s, (c, ex))| Miss { shape: s.to_string(), count: c, example: ex.map(str::to_string) }).collect();
    top_misses.sort_by_key(|m| (std::cmp::Reverse(m.count), m.shape.clone()));
    top_misses.truncate(30);

    // ---- learn from clean model-made tool calls (join on call id) ----
    let results: HashMap<&str, bool> = traces.iter().filter_map(|t| Some((t.result_for.as_deref()?, t.result_error.unwrap_or(true)))).collect();
    let mut learned = 0;
    if router.cfg.routing.learned_routes.enabled {
        for t in traces.iter().filter(|t| model_names.contains(t.served_by.as_str())) {
            let (Some(call), Some(tool), Some(args), Some(q)) = (t.call_id.as_deref(), t.tool.as_deref(), t.tool_args.as_deref(), t.question.as_deref()) else { continue };
            if results.get(call) != Some(&false) {
                continue; // no result seen, or it was an error
            }
            let lr = &router.cfg.routing.learned_routes;
            if !lr.allow_tools.is_empty() && !lr.allow_tools.iter().any(|a| a == tool) {
                continue;
            }
            if router.routes.learn_scoped(t.scope as u64, q, tool, args, None, false).is_some() {
                learned += 1;
            }
        }
    }

    // ---- self-heal offline: learned routes whose served calls mostly failed ----
    let mut route_outcomes: HashMap<&str, (u32, u32)> = HashMap::new(); // shape -> (ok, err)
    for t in traces.iter().filter(|t| t.served_by == "learned-route") {
        if let Some(call) = t.call_id.as_deref() {
            if let Some(err) = results.get(call) {
                let e = route_outcomes.entry(t.shape.as_str()).or_insert((0, 0));
                if *err { e.1 += 1 } else { e.0 += 1 }
            }
        }
    }
    let mut quarantined = Vec::new();
    for (shape, (okc, errc)) in &route_outcomes {
        if *errc >= router.cfg.heal.quarantine_after_failures && errc > okc {
            quarantined.push(shape.to_string());
            if apply && router.routes.set_quarantine(shape, true, &format!("miner: {errc} errors vs {okc} clean")) > 0 {
                store.heal("quarantine", shape, &format!("miner: {errc} error results vs {okc} clean"));
            }
        }
    }
    quarantined.sort();

    // ---- vocabulary suggestions from questions that needed a model ----
    let mut words: HashMap<String, HashSet<&str>> = HashMap::new();
    for t in ok.iter().filter(|t| model_names.contains(t.served_by.as_str())) {
        let Some(q) = t.question.as_deref() else { continue };
        for (i, w) in q.split(|c: char| !c.is_alphanumeric() && c != '-').filter(|w| w.len() >= 3).enumerate() {
            let looks_like_name = w.chars().next().is_some_and(|c| c.is_uppercase()) && i > 0;
            let known = router.routes.is_vocab(w);
            if looks_like_name && !known && !w.chars().any(|c| c.is_ascii_digit()) {
                words.entry(w.to_string()).or_default().insert(q);
            }
        }
    }
    let mut vocab_suggestions: Vec<(String, usize)> = words.into_iter().map(|(w, qs)| (w, qs.len())).filter(|(_, n)| *n >= 3).collect();
    vocab_suggestions.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    vocab_suggestions.truncate(40);

    // ---- tuning ----
    let mut tuning = Tuning { measured_at_ms: now_ms(), ..Default::default() };
    for m in &router.cfg.models {
        let lat: Vec<i64> = ok.iter().filter(|t| t.served_by == m.name).map(|t| t.latency_ms).collect();
        if lat.len() >= 20 {
            let p99 = pct(lat, 0.99);
            let secs = ((p99 as f64 * 2.0) / 1000.0).ceil() as u64;
            tuning.timeouts_secs.insert(m.name.clone(), secs.clamp(5, 300));
        }
    }
    let fast: Vec<i64> = ok
        .iter()
        .filter(|t| router.cfg.model(&t.served_by).is_some_and(|m| m.tier == crate::types::Tier::Fast))
        .map(|t| t.latency_ms)
        .collect();
    if fast.len() >= 20 {
        let p90 = pct(fast, 0.9);
        tuning.hedge_after_ms = Some(((p90 as u64).div_ceil(100) * 100).max(300));
    }

    // ---- decision engine calibration from shadow traffic ----
    let mut decision_calibration = Vec::new();
    for kind in ["shadow:plan", "shadow:tool"] {
        let rows: Vec<&Trace> = shadow.iter().filter(|t| t.surface == kind).collect();
        if rows.is_empty() {
            continue;
        }
        for th in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99] {
            let covered: Vec<&&Trace> = rows.iter().filter(|t| t.confidence >= th).collect();
            let agree = covered.iter().filter(|t| t.outcome == "agree").count();
            decision_calibration.push(Calibration {
                kind: kind.trim_start_matches("shadow:").to_string(),
                threshold: th,
                covered: covered.len(),
                total: rows.len(),
                precision: if covered.is_empty() { 0.0 } else { agree as f64 / covered.len() as f64 },
            });
        }
    }
    // Lowest threshold with >= 98% agreement over at least 30 covered rows.
    tuning.decision_min_confidence = decision_calibration.iter().filter(|c| c.kind == "plan" && c.covered >= 30 && c.precision >= 0.98).map(|c| c.threshold).fold(None, |a: Option<f64>, t| Some(a.map_or(t, |x| x.min(t))));

    // ---- train pankhllm's own decision models from teacher labels ----
    let mut native = Vec::new();
    if router.cfg.decisions.is_some() || router.cfg.prompts.auto_select {
        let target = router.cfg.decisions.as_ref().map(|d| d.target_precision).unwrap_or(0.98);
        for task in ["plan", "tool", "skill"] {
            let mut ex: Vec<crate::native::Example> = read_traces(&conn, 0, 5_000_000)?
                .into_iter()
                .filter(|t| t.surface == format!("label:{task}"))
                .filter_map(|t| Some(crate::native::Example { shape: t.shape, label: t.tool? }))
                .collect();
            if task == "plan" {
                ex.extend(router.seed_plan_examples());
            }
            if task == "skill" {
                ex.extend(router.seed_skill_examples());
            }
            if let Some(m) = crate::native::train_with_holdout(&ex, target) {
                let saved = apply;
                if apply {
                    store.put_kv(&format!("native:{task}"), serde_json::to_vec(&m)?);
                }
                native.push(NativeTraining { task: task.into(), examples: ex.len(), labels: m.labels.clone(), holdout_accuracy: m.holdout_accuracy, recommended_threshold: m.recommended_threshold, saved });
            }
        }
    }

    let heal_events = read_heal(&conn, since).map(|v| v.len()).unwrap_or(0);
    drop(conn);
    let mut pruned = 0;
    if apply {
        store.put_kv("tuning", serde_json::to_vec(&tuning)?);
        if let Some(sc) = &router.cfg.store {
            if let Ok(w) = rusqlite::Connection::open(&sc.path) {
                pruned = crate::store::prune(&w, sc.retention_days).unwrap_or(0);
            }
        }
        let _ = router.persist();
    }

    let all_ok: Vec<i64> = ok.iter().map(|t| t.latency_ms).collect();
    Ok(Report {
        window_hours: since_hours,
        traces: traces.len(),
        model_free_share: if ok.is_empty() { 0.0 } else { free as f64 / ok.len() as f64 },
        p50_ms: pct(all_ok.clone(), 0.5),
        p95_ms: pct(all_ok, 0.95),
        cost_usd: ok.iter().map(|t| t.cost_usd).sum(),
        lanes,
        top_misses,
        learned_from_logs: learned,
        quarantined,
        vocab_suggestions,
        tuning,
        heal_events,
        pruned,
        decision_calibration,
        native,
    })
}

/// Training rows for fine-tuning a System-1 model on this deployment's own traffic:
/// the question, and the decision the slow path (generative planner or agent) made.
pub fn export_decisions(router: &Router, since_hours: u64, path: &std::path::Path) -> anyhow::Result<usize> {
    let store = router.store.as_ref().ok_or_else(|| anyhow::anyhow!("no `store` configured"))?;
    store.flush();
    let conn = store.reader()?;
    let since = now_ms() - (since_hours as i64) * 3_600_000;
    let rows = read_traces(&conn, since, 5_000_000)?;
    let ops: Vec<(String, String)> = router.planner_ops();
    let mut out = String::new();
    let mut n = 0;
    for t in rows.iter().filter(|t| t.surface.starts_with("shadow:")) {
        let (Some(q), Some(label)) = (t.question.as_deref(), t.tool_args.as_deref()) else { continue };
        let kind = t.surface.trim_start_matches("shadow:");
        let criteria: serde_json::Map<String, serde_json::Value> = if kind == "plan" {
            let mut m: serde_json::Map<String, serde_json::Value> = ops.iter().map(|(k, d)| (k.clone(), serde_json::json!(d))).collect();
            m.insert("UNSUPPORTED".into(), serde_json::json!("none of these operations answers it exactly"));
            m
        } else {
            serde_json::Map::new()
        };
        let row = serde_json::json!({"state": q, "question": {"type": "choice", "instructions": if kind == "plan" { "Which operation answers this request exactly?" } else { "Which tool should handle this request?" }, "criteria": criteria}, "label": label, "kind": kind});
        out.push_str(&row.to_string());
        out.push('\n');
        n += 1;
    }
    std::fs::write(path, out)?;
    Ok(n)
}

/// Human-readable report.
pub fn markdown(r: &Report) -> String {
    let mut s = String::new();
    s.push_str(&format!("# pankhllm miner report (last {} h)\n\n", r.window_hours));
    s.push_str(&format!(
        "{:.1}% of {} requests were answered without a model call. p50 {} ms, p95 {} ms, model spend ${:.4}.\n\n",
        r.model_free_share * 100.0, r.traces, r.p50_ms, r.p95_ms, r.cost_usd
    ));
    s.push_str("## Lanes\n\n| Served by | Requests | Model-free | p50 ms | p95 ms | Cost USD |\n| --- | ---: | --- | ---: | ---: | ---: |\n");
    for l in &r.lanes {
        s.push_str(&format!("| {} | {} | {} | {} | {} | {:.4} |\n", l.served_by, l.requests, if l.model_free { "yes" } else { "no" }, l.p50_ms, l.p95_ms, l.cost_usd));
    }
    s.push_str("\n## Shapes that still needed a model\n\n| Shape | Count | Example |\n| --- | ---: | --- |\n");
    for m in &r.top_misses {
        s.push_str(&format!("| {} | {} | {} |\n", m.shape.replace('|', "\\|"), m.count, m.example.clone().unwrap_or_default().replace('|', "\\|")));
    }
    s.push_str(&format!("\n## Learning and healing\n\n- Routes learned from logs this run: {}\n- Heal events in window: {}\n- Traces pruned by retention: {}\n", r.learned_from_logs, r.heal_events, r.pruned));
    if !r.quarantined.is_empty() {
        s.push_str("- Quarantined shapes:\n");
        for q in &r.quarantined {
            s.push_str(&format!("    - {q}\n"));
        }
    }
    if !r.vocab_suggestions.is_empty() {
        s.push_str("\n## Vocabulary suggestions\n\nNames seen in three or more questions that needed a model. Add real entities to `learned_routes.vocab_file`.\n\n");
        for (w, n) in &r.vocab_suggestions {
            s.push_str(&format!("- {w} ({n})\n"));
        }
    }
    if !r.native.is_empty() {
        s.push_str("\n## pankhllm's own decision models\n\n| Task | Examples | Labels | Held-out accuracy | Threshold | Saved |\n| --- | ---: | ---: | ---: | ---: | --- |\n");
        for n in &r.native {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                n.task,
                n.examples,
                n.labels.len(),
                n.holdout_accuracy.map(|a| format!("{:.1}%", a * 100.0)).unwrap_or_else(|| "too few to hold out".into()),
                n.recommended_threshold.map(|t| format!("{t:.2}")).unwrap_or_else(|| "-".into()),
                if n.saved { "yes" } else { "no (--apply)" }
            ));
        }
    }
    if !r.decision_calibration.is_empty() {
        s.push_str("\n## Decision engine calibration (shadow)\n\n| Kind | Threshold | Covered | Agreement |\n| --- | ---: | ---: | ---: |\n");
        for c in &r.decision_calibration {
            s.push_str(&format!("| {} | {:.2} | {} of {} | {:.1}% |\n", c.kind, c.threshold, c.covered, c.total, c.precision * 100.0));
        }
    }
    s.push_str(&format!("\n## Tuning\n\n- hedge.after_ms: {}\n", r.tuning.hedge_after_ms.map(|v| v.to_string()).unwrap_or_else(|| "not enough data".into())));
    if let Some(c) = r.tuning.decision_min_confidence {
        s.push_str(&format!("- decisions.min_confidence: {c:.2}\n"));
    }
    for (m, t) in &r.tuning.timeouts_secs {
        s.push_str(&format!("- {m} timeout_secs: {t}\n"));
    }
    s
}
