use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};
use futures::StreamExt;

use std::collections::BTreeMap;
use std::io::BufRead;
use std::time::Instant;

use pankhllm::providers::StreamEvent;
use pankhllm::types::{Chunk, Intent, Message, RouteRequest, Tier, ToolDef};
use pankhllm::{Config, Router};

#[derive(Parser)]
#[command(name = "pankhllm", version, about = "The LLM gateway that learns to skip the LLM")]
struct Cli {
    /// Path to the YAML config.
    #[arg(short, long, default_value = "pankhllm.yaml", global = true)]
    config: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the HTTP server.
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(short, long, default_value_t = 4000)]
        port: u16,
    },
    /// Explain where a question would go. No network calls.
    Route {
        question: String,
        /// Retrieval scores of the chunks you would pass, e.g. --scores 0.9,0.4
        #[arg(long, value_delimiter = ',')]
        scores: Vec<f64>,
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
    },
    /// Route and answer a question.
    Ask {
        question: String,
        /// Files whose contents are passed as retrieved context.
        #[arg(long, value_delimiter = ',')]
        context: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        tags: Vec<String>,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        stream: bool,
    },
    /// Validate the config and list models and prompts.
    Check,
    /// Overnight miner: read the trace store, learn from clean turns, quarantine failing
    /// routes, measure every lane and model, propose vocabulary and tuning, write a report.
    Mine {
        #[arg(long, default_value_t = 24)]
        since_hours: u64,
        /// Write tuning, apply quarantines, prune by retention, persist routes.
        #[arg(long)]
        apply: bool,
        /// Markdown report path (stdout when omitted).
        #[arg(long)]
        report: Option<String>,
        /// Print the report as JSON instead of markdown.
        #[arg(long)]
        json: bool,
        /// Also write decision-engine training rows (question + the slow path's decision) here.
        #[arg(long)]
        export_decisions: Option<String>,
    },
    /// Label questions for training pankhllm's own decision model. Lines with a "label" are
    /// recorded as given; lines with only a "question" are labelled by the teacher (the
    /// generative planner): decision only, validated against the catalog, nothing executed.
    Label {
        file: String,
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Write {"question","label"} lines here instead of recording training labels
        /// (to build an evaluation set labelled by the teacher).
        #[arg(long)]
        out: Option<String>,
        /// What the teacher labels unlabelled lines for: `plan` (catalog operation) or
        /// `skill` (which skill file the question needs).
        #[arg(long, default_value = "plan")]
        task: String,
        /// Teacher model for `--task skill`; defaults to the planner model, else the first model.
        #[arg(long)]
        teacher_model: Option<String>,
    },
    /// Write the decision models in use (plan, tool, skill) to one JSON file. Ship it and
    /// point `decisions.models_file` at it; no trace store is needed to serve.
    ExportModels {
        out: String,
    },
    /// Evaluate pankhllm's own decision model on labelled questions ({"question","label"}):
    /// accuracy, coverage, wrong decisions and decision time. Offline; no models called.
    DecideEval {
        file: String,
        #[arg(long, default_value_t = 15)]
        show: usize,
        /// `plan` (catalog operations) or `skill` (skill routing).
        #[arg(long, default_value = "plan")]
        task: String,
    },
    /// Generate varied training questions for the planner catalog with a generative model:
    /// many phrasings per operation, plus questions none of them should answer. Write them
    /// as JSONL, then run `label` so the planner (the teacher) labels them.
    Augment {
        /// Output JSONL (one {"question": ...} per line).
        out: String,
        /// Questions to request per operation (and for the negatives).
        #[arg(long, default_value_t = 30)]
        per_op: usize,
        /// Generator model; defaults to planner.model.
        #[arg(long)]
        model: Option<String>,
    },
    /// Offline training. Ingest completed turns from a JSONL log (question, tool,
    /// arguments, tool_result, answer...) into the learned-route memory: no model calls.
    /// With --replay, lines that carry only a question are run through the planner model
    /// and, when a tool_result is present, the result hop as well.
    Warm {
        file: String,
        #[arg(long)]
        replay: bool,
        #[arg(long, default_value_t = 8)]
        concurrency: usize,
    },
    /// Routing-accuracy evaluation over a labelled JSONL file. Offline: no model calls.
    /// Each line: {"question", "expected_intent", "expected_tier", "scores": [..], "tags": [..], "tools": [..]}
    Eval {
        file: String,
        /// Exit non-zero if tier accuracy is below this (0-100).
        #[arg(long, default_value_t = 99.0)]
        min_accuracy: f64,
        /// Print up to this many mismatches.
        #[arg(long, default_value_t = 25)]
        show: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let cli = Cli::parse();
    let cfg = Config::from_path(&cli.config)?;

    match cli.cmd {
        Cmd::Serve { host, port } => {
            let router = Arc::new(Router::new(cfg)?);
            if router.cfg.routing.learned_routes.persist_path.is_some() {
                let r = Arc::clone(&router);
                let every = router.cfg.routing.learned_routes.persist_interval_secs.max(5);
                tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(std::time::Duration::from_secs(every)).await;
                        if let Err(e) = r.persist() {
                            tracing::warn!(error = %e, "could not persist learned routes");
                        }
                    }
                });
            }
            if router.store.is_some() && router.cfg.decisions.is_some() {
                // Pick up models the overnight miner trained, without a restart.
                let r = Arc::clone(&router);
                tokio::spawn(async move {
                    loop {
                        tokio::time::sleep(std::time::Duration::from_secs(300)).await;
                        r.reload_native();
                    }
                });
            }
            let app = pankhllm::server::app(Arc::clone(&router));
            let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
            tracing::info!("pankhllm listening on http://{host}:{port}");
            let shutdown = async {
                let _ = tokio::signal::ctrl_c().await;
            };
            axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
            router.persist()?;
            tracing::info!("learned routes persisted; bye");
        }
        Cmd::Label { file, concurrency, out, task, teacher_model } => {
            anyhow::ensure!(task == "plan" || task == "skill", "--task must be plan or skill");
            let teacher = teacher_model.or_else(|| cfg.planner.as_ref().map(|p| p.model.clone())).or_else(|| cfg.models.first().map(|m| m.name.clone())).unwrap_or_default();
            let router = Arc::new(Router::new(cfg)?);
            let reader = std::io::BufReader::new(std::fs::File::open(&file)?);
            let sem = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
            let mut handles = Vec::new();
            let mut direct = 0usize;
            // With --out nothing is recorded for training: given labels pass through to the file.
            let mut given = Vec::new();
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(&line)?;
                let Some(q) = v["question"].as_str().map(str::to_string) else { continue };
                if let Some(l) = v["label"].as_str() {
                    let row_task = v["task"].as_str().unwrap_or("plan");
                    // One pass per task; given tool labels ride along with the default (plan) pass.
                    if !(row_task == task || (task == "plan" && row_task == "tool")) {
                        continue;
                    }
                    if out.is_some() {
                        given.push(serde_json::json!({"question": q, "label": l, "task": row_task}).to_string());
                    } else {
                        router.direct_label(row_task, &q, l);
                    }
                    direct += 1;
                    continue;
                }
                let r = Arc::clone(&router);
                let permit = Arc::clone(&sem).acquire_owned().await?;
                let key = v["cache_key"].as_str().map(str::to_string);
                let record = out.is_none();
                let (task, teacher) = (task.clone(), teacher.clone());
                handles.push(tokio::spawn(async move {
                    let _p = permit;
                    let l = if task == "skill" { r.teacher_skill_label(&teacher, &q, record).await } else { r.teacher_label_opt(&q, key.as_deref(), record).await };
                    l.ok().flatten().map(|l| (q, l))
                }));
            }
            let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
            let mut none = 0usize;
            let mut rows = given;
            for h in handles {
                match h.await.ok().flatten() {
                    Some((q, l)) => {
                        *counts.entry(l.clone()).or_default() += 1;
                        rows.push(serde_json::json!({"question": q, "label": l, "task": task}).to_string());
                    }
                    None => none += 1,
                }
            }
            if let Some(p) = &out {
                std::fs::write(p, rows.join("\n") + "\n")?;
                println!("wrote {} labelled questions to {p} ({direct} given, none recorded for training)", rows.len());
            }
            if let Some(st) = &router.store {
                st.flush();
            }
            println!("recorded {direct} given labels; teacher labelled {} ({counts:?}); {none} gave no valid decision", counts.values().sum::<usize>());
            println!("next: pankhllm mine --apply   (trains the decision model from these labels)");
        }
        Cmd::ExportModels { out } => {
            let router = Router::new(cfg)?;
            let f = router.export_models();
            let bytes = serde_json::to_vec(&f)?;
            std::fs::write(&out, &bytes)?;
            for (k, m) in &f.models {
                let weights: usize = m.weights.iter().map(|w| w.len()).sum();
                println!("{k:6} labels {:>3}  weights {weights:>8}  trained on {:>7}  held-out accuracy {}", m.labels.len(), m.trained_on, m.holdout_accuracy.map(|a| format!("{:.1}%", a * 100.0)).unwrap_or("-".into()));
            }
            println!("wrote {out} ({:.1} KB)", bytes.len() as f64 / 1024.0);
        }
        Cmd::DecideEval { file, show, task } => {
            let router = Router::new(cfg)?;
            let reader = std::io::BufReader::new(std::fs::File::open(&file)?);
            let (mut n, mut decided, mut correct, mut wrong, mut unsup_ok, mut fallback) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
            let mut us: Vec<u128> = Vec::new();
            let mut misses: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(&line)?;
                let (Some(q), Some(want)) = (v["question"].as_str(), v["label"].as_str()) else { continue };
                if v["task"].as_str().unwrap_or("plan") != task {
                    continue;
                }
                n += 1;
                let (got, p, t) = if task == "skill" {
                    let started = std::time::Instant::now();
                    let d = router.native_skill_decision(q);
                    let t = started.elapsed().as_micros();
                    d.map(|(s, p)| (s, p, t)).unwrap_or(("FALLBACK".into(), 0.0, t))
                } else {
                    router.native_decision(q).await
                };
                us.push(t);
                match got.as_str() {
                    "FALLBACK" => fallback += 1,
                    g if g == want => {
                        decided += 1;
                        correct += 1;
                        if g == "UNSUPPORTED" {
                            unsup_ok += 1;
                        }
                    }
                    g => {
                        decided += 1;
                        wrong += 1;
                        if misses.len() < show {
                            misses.push(format!("  wrong: said {g} ({p:.2}) wanted {want}: {q}"));
                        }
                    }
                }
            }
            anyhow::ensure!(n > 0, "no labelled {task} lines in {file}");
            us.sort_unstable();
            println!("questions: {n}");
            println!("decided without a model: {decided} ({:.1}%)", 100.0 * decided as f64 / n as f64);
            println!("  correct: {correct}  wrong: {wrong}  precision: {:.2}%", if decided > 0 { 100.0 * correct as f64 / decided as f64 } else { 0.0 });
            println!("  of which confident UNSUPPORTED (planner call skipped): {unsup_ok}");
            println!("sent to the generative planner: {fallback} ({:.1}%)", 100.0 * fallback as f64 / n as f64);
            println!("decision time: p50 {} us, p99 {} us", us[(us.len() - 1) / 2], us[(us.len() - 1) * 99 / 100]);
            for m in misses {
                println!("{m}");
            }
        }
        Cmd::Augment { out, per_op, model } => {
            let router = Router::new(cfg)?;
            let model = model.or_else(|| router.cfg.planner.as_ref().map(|p| p.model.clone())).ok_or_else(|| anyhow::anyhow!("no --model and no planner.model"))?;
            let ops = router.planner_catalog_text();
            anyhow::ensure!(!ops.is_empty(), "no planner operations to augment");
            let system = "You write realistic user questions for testing an assistant. Output only the questions, one per line, no numbering, no quotes, no commentary.";
            let mut seen = std::collections::HashSet::new();
            let mut lines = Vec::new();
            let clean = |l: &str| -> Option<String> {
                let t = l.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')' || c == '-' || c == '*' || c == ' ').trim_matches('"').trim();
                (t.len() >= 6 && t.len() <= 200 && !t.ends_with(':')).then(|| t.to_string())
            };
            let catalog = ops.iter().map(|(n, d, _)| format!("- {n}: {d}")).collect::<Vec<_>>().join("\n");
            let mut prompts: Vec<(String, String)> = ops
                .iter()
                .map(|(n, d, p)| (n.clone(), format!("Write {per_op} different questions a user could ask that this operation answers exactly:\n{d}\nParameters: {p}\nVary wording, word order, length, formality and small typos. Use realistic, varied values for every parameter.")))
                .collect();
            prompts.push(("negatives".into(), format!("An assistant can only do these operations:\n{catalog}\nWrite {per_op} questions its users might plausibly ask that NONE of these operations answers exactly: explanations, advice, trends and comparisons over time, other topics, small talk, writing tasks. Some should mention the same kinds of values the operations use.")));
            for (name, prompt) in prompts {
                let text = router.generate(&model, system, &prompt, 2000).await?;
                let mut n = 0;
                for l in text.lines().filter_map(clean) {
                    if seen.insert(l.to_lowercase()) {
                        lines.push(serde_json::json!({"question": l, "generated_for": name}).to_string());
                        n += 1;
                    }
                }
                eprintln!("{name}: {n} questions");
            }
            std::fs::write(&out, lines.join("\n") + "\n")?;
            println!("wrote {} questions to {out}; label them with: pankhllm label {out}", lines.len());
        }
        Cmd::Mine { since_hours, apply, report, json, export_decisions } => {
            let router = Router::new(cfg)?;
            let r = pankhllm::miner::mine(&router, since_hours, apply)?;
            if let Some(p) = &export_decisions {
                let n = pankhllm::miner::export_decisions(&router, since_hours, std::path::Path::new(p))?;
                eprintln!("wrote {n} decision training rows to {p}");
            }
            let out = if json { serde_json::to_string_pretty(&r)? } else { pankhllm::miner::markdown(&r) };
            match report {
                Some(p) => {
                    std::fs::write(&p, &out)?;
                    println!("report written to {p}: {:.1}% model-free over {} traces", r.model_free_share * 100.0, r.traces);
                }
                None => println!("{out}"),
            }
        }
        Cmd::Warm { file, replay, concurrency } => {
            let router = Arc::new(Router::new(cfg)?);
            let reader = std::io::BufReader::new(std::fs::File::open(&file)?);
            let mut ingested = 0usize;
            let mut replayed = 0usize;
            let mut skipped = 0usize;
            let sem = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
            let mut handles = Vec::new();
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(&line)?;
                if v.get("tool").is_some() && v.get("arguments").is_some() {
                    let turn: pankhllm::router::ObservedTurn = serde_json::from_value(v)?;
                    if router.ingest_turn(&turn).is_some() { ingested += 1 } else { skipped += 1 }
                    continue;
                }
                if !replay {
                    skipped += 1;
                    continue;
                }
                // Replay: run the tool-select hop (and the result hop if a result is logged).
                let r = Arc::clone(&router);
                let permit = Arc::clone(&sem).acquire_owned().await?;
                handles.push(tokio::spawn(async move {
                    let _p = permit;
                    let q = v["question"].as_str().unwrap_or("").to_string();
                    let tools: Vec<ToolDef> = v["tool_names"].as_array().into_iter().flatten().filter_map(|t| t.as_str()).map(|n| ToolDef { name: n.into(), description: String::new(), parameters: serde_json::json!({"type": "object"}) }).collect();
                    let mut req = RouteRequest { messages: vec![Message::text("user", q.clone())], tools: tools.clone(), ..Default::default() };
                    req.cache_key = v["cache_key"].as_str().map(str::to_string);
                    req.prompt = v["prompt"].as_str().map(str::to_string);
                    let Ok(c) = r.complete(&req).await else { return false };
                    if let (Some(call), Some(result)) = (c.tool_calls.first(), v["tool_result"].as_str()) {
                        req.messages.push(Message { role: "assistant".into(), content: String::new(), tool_calls: vec![call.clone()], tool_call_id: None });
                        req.messages.push(Message { role: "user".into(), content: result.to_string(), tool_calls: vec![], tool_call_id: Some(call.id.clone()) });
                        let _ = r.complete(&req).await;
                    }
                    true
                }));
            }
            for h in handles {
                if h.await.unwrap_or(false) { replayed += 1 } else { skipped += 1 }
            }
            router.persist()?;
            let st = router.learned_stats();
            println!("ingested {ingested}, replayed {replayed}, skipped {skipped}; routes now {} (learned {}, served {})", st.routes, st.learned, st.served);
            if let Some(p) = &router.cfg.routing.learned_routes.persist_path {
                println!("saved to {p}");
            }
        }
        Cmd::Route { question, scores, tags } => {
            let router = Router::offline(cfg);
            let req = RouteRequest {
                messages: vec![Message::text("user", question)],
                context: scores.into_iter().map(|s| Chunk { text: String::new(), score: Some(s), source: None }).collect(),
                tags,
                ..Default::default()
            };
            println!("{}", serde_json::to_string_pretty(&router.decide(&req))?);
        }
        Cmd::Ask { question, context, tags, prompt, stream } => {
            let router = Arc::new(Router::new(cfg)?);
            let mut chunks = Vec::new();
            for path in context {
                chunks.push(Chunk { text: std::fs::read_to_string(&path)?, score: None, source: Some(path) });
            }
            let req = RouteRequest {
                messages: vec![Message::text("user", question)],
                context: chunks,
                tags,
                prompt,
                ..Default::default()
            };
            if stream {
                let mut sc = router.stream(&req).await?;
                eprintln!("[pankhllm] streaming from {} ({})", sc.model, sc.provider_model);
                while let Some(ev) = sc.events.next().await {
                    match ev? {
                        StreamEvent::Delta(t) => print!("{t}"),
                        StreamEvent::ToolCallStart { name, .. } => eprint!("\n[tool call {name}] "),
                        StreamEvent::ToolCallDelta { arguments, .. } => eprint!("{arguments}"),
                        StreamEvent::Done { input_tokens, output_tokens, .. } => {
                            eprintln!("\n[pankhllm] tokens in={input_tokens} out={output_tokens}")
                        }
                    }
                }
                println!();
            } else {
                let c = router.complete(&req).await?;
                println!("{}", c.text);
                eprintln!(
                    "\n[pankhllm] model={} tier={} cost=${:.5} attempts={}",
                    c.model,
                    c.decision.target_tier,
                    c.usage.cost_usd,
                    c.attempts.len()
                );
            }
        }
        Cmd::Eval { file, min_accuracy, show } => {
            let router = Router::offline(cfg);
            let reader = std::io::BufReader::new(std::fs::File::open(&file)?);
            let mut total = 0usize;
            let mut tier_ok = 0usize;
            let mut intent_ok = 0usize;
            let mut per_intent: BTreeMap<String, (usize, usize)> = BTreeMap::new();
            let mut confusion: BTreeMap<(String, String), usize> = BTreeMap::new();
            let mut mismatches: Vec<String> = Vec::new();
            let mut total_ns: u128 = 0;
            let mut max_ns: u128 = 0;
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let rec: serde_json::Value = serde_json::from_str(&line)?;
                let q = rec["question"].as_str().unwrap_or("").to_string();
                let exp_tier: Tier = serde_json::from_value(rec["expected_tier"].clone())?;
                let exp_intent: Option<Intent> = serde_json::from_value(rec["expected_intent"].clone()).ok();
                let req = RouteRequest {
                    messages: vec![Message::text("user", q.clone())],
                    context: rec["scores"].as_array().into_iter().flatten().filter_map(|s| s.as_f64()).map(|s| Chunk { text: "passage".into(), score: Some(s), source: None }).collect(),
                    tags: rec["tags"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(str::to_string)).collect(),
                    tools: rec["tools"].as_array().into_iter().flatten().filter_map(|t| t.as_str()).map(|n| ToolDef { name: n.into(), description: String::new(), parameters: serde_json::json!({"type": "object"}) }).collect(),
                    ..Default::default()
                };
                let t = Instant::now();
                let d = router.decide(&req);
                let ns = t.elapsed().as_nanos();
                total_ns += ns;
                max_ns = max_ns.max(ns);
                total += 1;
                let got_intent = d.signals.intent;
                let ok_t = d.target_tier == exp_tier;
                let ok_i = exp_intent.is_none_or(|e| e == got_intent);
                tier_ok += ok_t as usize;
                intent_ok += ok_i as usize;
                let key = exp_intent.map(|i| i.to_string()).unwrap_or_else(|| "?".into());
                let e = per_intent.entry(key).or_insert((0, 0));
                e.0 += 1;
                e.1 += ok_t as usize;
                *confusion.entry((exp_tier.to_string(), d.target_tier.to_string())).or_insert(0) += 1;
                if (!ok_t || !ok_i) && mismatches.len() < show {
                    mismatches.push(format!("  expected {}/{} got {}/{}  {:?}", exp_intent.map(|i| i.to_string()).unwrap_or_default(), exp_tier, got_intent, d.target_tier, q));
                }
            }
            if total == 0 {
                anyhow::bail!("no records in {file}");
            }
            let acc = 100.0 * tier_ok as f64 / total as f64;
            println!("records: {total}");
            println!("tier accuracy:   {acc:.2}%  ({tier_ok}/{total})");
            println!("intent accuracy: {:.2}%  ({intent_ok}/{total})", 100.0 * intent_ok as f64 / total as f64);
            println!("decision latency: mean {:.1} us, max {:.1} us, total {:.1} ms", total_ns as f64 / total as f64 / 1000.0, max_ns as f64 / 1000.0, total_ns as f64 / 1e6);
            println!("per expected intent (tier accuracy):");
            for (k, (n, ok)) in &per_intent {
                println!("  {k:<14} {:>6.2}%  n={n}", 100.0 * *ok as f64 / *n as f64);
            }
            println!("tier confusion (expected -> got):");
            for ((e, g), n) in &confusion {
                if e != g {
                    println!("  {e:<10} -> {g:<10} {n}");
                }
            }
            if !mismatches.is_empty() {
                println!("mismatches (first {}):", mismatches.len());
                for m in &mismatches {
                    println!("{m}");
                }
            }
            if acc < min_accuracy {
                anyhow::bail!("tier accuracy {acc:.2}% is below the required {min_accuracy:.2}%");
            }
        }
        Cmd::Check => {
            let router = Router::new(cfg)?;
            println!("config ok");
            for m in &router.cfg.models {
                println!("  model {:<14} tier={:<9} provider={:<10} ctx={:<8} tags={:?}", m.name, m.tier, m.provider, m.context_window, m.tags);
            }
            println!("  prompts: {:?}", router.prompts.names());
        }
    }
    Ok(())
}
