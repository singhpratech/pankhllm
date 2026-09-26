//! The policy: signals in, an ordered list of models out. Then the executor
//! walks that list, escalating on errors, refusals and hedged answers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use futures::StreamExt;
use serde::Serialize;
use tokio::sync::Semaphore;
use tracing::{info, warn};

use crate::cache::{hash64, CachedAnswer, ResponseCache};
use crate::learn::RouteMemory;
use crate::config::{Config, ModelConfig, OnAmbiguous, ProviderKind, WeakRetrievalAction};
use crate::prompts::PromptRegistry;
use crate::providers::{
    anthropic::AnthropicProvider, openai::OpenAiProvider, EventStream, Provider, ProviderError,
    ProviderRequest, ProviderResponse, StopReason, StreamEvent,
};
use crate::signals;
use crate::types::*;

/// Rolling latency and outcome counters per model, exposed on `/v1/stats`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelStats {
    pub calls: u64,
    pub ok: u64,
    pub errors: u64,
    pub refusals: u64,
    pub low_confidence: u64,
    pub timeouts: u64,
    pub ewma_latency_ms: f64,
    pub total_cost_usd: f64,
    pub in_flight: usize,
}

/// A template the planning model produced for a pending tool call.
#[derive(Debug, Clone)]
struct PendingTemplate {
    template: String,
    needs_analysis: bool,
    created: Instant,
    /// The tool call's arguments, exposed to the template as `{{args.<name>}}`.
    args: serde_json::Value,
}

/// Token bucket plus a daily spend counter for one user.
#[derive(Debug, Clone)]
struct UserBudget {
    tokens: f64,
    last: Instant,
    day: u64,
    spent_usd: f64,
}

/// Why a request was refused before routing.
#[derive(Debug, Clone)]
pub struct BudgetDenied {
    pub message: String,
    pub retry_after_secs: u64,
}

/// A tool call observed on a tool-selection turn, waiting to see whether its
/// result comes back clean before the shape is learned.
#[derive(Debug, Clone)]
struct RouteCandidate {
    scope: u64,
    question: String,
    call: ToolCall,
    template: Option<PendingTemplate>,
    vector: Option<Vec<f32>>,
    created: Instant,
}

pub struct Router {
    pub cfg: Config,
    pub prompts: PromptRegistry,
    pub cache: ResponseCache,
    /// call id -> answer template waiting for the tool result.
    pending_templates: Mutex<HashMap<String, PendingTemplate>>,
    /// call id -> observed tool call waiting for a clean result before it is learned.
    route_candidates: Mutex<HashMap<String, RouteCandidate>>,
    pub routes: RouteMemory,
    /// Exact request key -> notifier for requests currently being answered, so
    /// identical concurrent requests share one model call.
    in_flight: Mutex<HashMap<u64, Arc<tokio::sync::Notify>>>,
    /// Per-user token buckets and daily spend.
    user_budgets: Mutex<HashMap<String, UserBudget>>,
    /// Shadow verdicts waiting for the slow path's decision: question hash -> (label, conf, ms, at).
    pending_shadow: Mutex<HashMap<u64, (String, f64, u128, Instant)>>,
    /// Embedded trace database, when configured.
    pub store: Option<crate::store::Store>,
    /// call id -> learned-route key, so a result hop can report back on the route.
    served_routes: Mutex<HashMap<String, (u64, Instant)>>,
    /// Planner lane config and its compiled catalog.
    planner: Option<(crate::config::PlannerConfig, crate::planner::Catalog)>,
    /// Pooled client for executor calls.
    http: reqwest::Client,
    /// Optional external decision engine adapter (`decisions.engine: external`).
    pub engine: Option<crate::decide::DecisionEngine>,
    /// pankhllm's own decision models, trained from its own traffic.
    native_plan: std::sync::RwLock<Option<crate::native::NativeModel>>,
    native_tool: std::sync::RwLock<Option<crate::native::NativeModel>>,
    /// Skill routing: which skill file a question needs, learned from explicit choices.
    native_skill: std::sync::RwLock<Option<crate::native::NativeModel>>,
    providers: HashMap<String, Arc<dyn Provider>>,
    cooldown: Mutex<HashMap<String, Instant>>,
    stats: Mutex<HashMap<String, ModelStats>>,
    limits: HashMap<String, Arc<Semaphore>>,
    /// Compiled `pattern` per rule, same order as `cfg.routing.rules`.
    rule_patterns: Vec<Option<regex::Regex>>,
}

/// A streaming answer: the decision is known up front, tokens arrive as they are generated.
pub struct StreamingCompletion {
    pub model: String,
    pub provider_model: String,
    pub decision: RoutingDecision,
    pub attempts: Vec<Attempt>,
    pub cache: Option<CacheInfo>,
    pub events: EventStream,
}

impl Router {
    pub fn new(cfg: Config) -> Result<Self> {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        for (name, p) in &cfg.providers {
            let provider: Arc<dyn Provider> = match p.kind {
                ProviderKind::Anthropic => Arc::new(AnthropicProvider::new(p)?),
                ProviderKind::Openai => Arc::new(OpenAiProvider::new(p)?),
                ProviderKind::Azure => Arc::new(OpenAiProvider::azure(p)?),
            };
            providers.insert(name.clone(), provider);
        }
        let prompts = PromptRegistry::load(&cfg.prompts)?;
        Ok(Self::assemble(cfg, prompts, providers))
    }

    /// Build without provider clients. Only `decide` works; used by tests and dry runs.
    pub fn offline(cfg: Config) -> Self {
        Self::assemble(cfg, PromptRegistry::default(), HashMap::new())
    }

    /// Build with caller-supplied providers (tests use this to inject mocks).
    pub fn with_providers(cfg: Config, providers: HashMap<String, Arc<dyn Provider>>) -> Self {
        Self::assemble(cfg, PromptRegistry::default(), providers)
    }

    /// Replace the prompt registry (tests, or loading prompts from another source).
    pub fn with_prompts(mut self, prompts: PromptRegistry) -> Self {
        // Seed from the new skill files only when no trained model was loaded (store or file).
        let mut slot = self.native_skill.write().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = crate::native::train(&skill_seed(&prompts), 30);
        }
        drop(slot);
        self.prompts = prompts;
        self
    }

    fn assemble(mut cfg: Config, prompts: PromptRegistry, providers: HashMap<String, Arc<dyn Provider>>) -> Self {
        let limits = cfg
            .models
            .iter()
            .filter_map(|m| m.max_concurrency.map(|n| (m.name.clone(), Arc::new(Semaphore::new(n.max(1))))))
            .collect();
        let rule_patterns = cfg.routing.rules.iter().map(|r| r.pattern.as_ref().and_then(|p| regex::Regex::new(p).ok())).collect();
        let store = cfg.store.as_ref().and_then(|sc| match crate::store::Store::open(&sc.path, sc.queue) {
            Ok(s) => Some(s),
            Err(e) => {
                warn!(path = %sc.path, error = %e, "trace store unavailable; continuing without it");
                None
            }
        });
        // Autotune: apply what the miner measured last night.
        if cfg.heal.autotune {
            if let Some(t) = store.as_ref().and_then(|s| s.get_kv("tuning")).and_then(|b| serde_json::from_slice::<crate::miner::Tuning>(&b).ok()) {
                t.apply(&mut cfg);
                info!(hedge_after_ms = ?cfg.routing.hedge.after_ms, "applied miner tuning");
            }
        }
        let cache = ResponseCache::new(cfg.cache.clone());
        let planner = cfg.planner.clone().filter(|p| p.enabled).and_then(|pc| match crate::planner::Catalog::load(&pc) {
            Ok(c) => Some((pc, c)),
            Err(e) => {
                warn!(error = %e, "planner catalog could not be loaded; planner lane off");
                None
            }
        });
        let engine = cfg.decisions.clone().filter(|d| d.engine == crate::config::DecisionEngineKind::External && d.url.is_some()).map(crate::decide::DecisionEngine::new);
        let load_native = |key: &str| newest_native(store.as_ref(), &cfg, key);
        let native_tool = load_native("native:tool");
        let native_skill = load_native("native:skill").or_else(|| crate::native::train(&skill_seed(&prompts), 30));
        // Our own plan model: last night's training if there is one, else seeded from catalog examples.
        let native_plan = load_native("native:plan").or_else(|| planner.as_ref().and_then(|(pc, cat)| crate::native::train(&seed_examples(pc, cat), 30)));
        let lr = &cfg.routing.learned_routes;
        let mut vocab = lr.vocab.clone();
        if let Some(f) = &lr.vocab_file {
            if let Ok(text) = std::fs::read_to_string(f) {
                vocab.extend(text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string));
            }
        }
        let routes = RouteMemory::with_vocab(lr.ttl_secs, lr.max_routes, lr.min_similarity, lr.min_observations, vocab);
        if let Some(p) = &lr.persist_path {
            match routes.load(std::path::Path::new(p)) {
                Ok(n) => info!(path = %p, routes = n, "loaded learned routes"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => warn!(path = %p, error = %e, "could not load learned routes"),
            }
        } else if let Some(snap) = store.as_ref().and_then(|s| s.get_kv("learned_routes")).and_then(|b| serde_json::from_slice::<crate::learn::RouteSnapshot>(&b).ok()) {
            let n = routes.import(snap);
            info!(routes = n, "loaded learned routes from the trace store");
        }
        Self {
            cfg,
            prompts,
            cache,
            pending_templates: Mutex::new(HashMap::new()),
            route_candidates: Mutex::new(HashMap::new()),
            routes,
            in_flight: Mutex::new(HashMap::new()),
            user_budgets: Mutex::new(HashMap::new()),
            pending_shadow: Mutex::new(HashMap::new()),
            store,
            served_routes: Mutex::new(HashMap::new()),
            planner,
            http: reqwest::Client::builder().pool_idle_timeout(Duration::from_secs(90)).build().unwrap_or_default(),
            engine,
            native_plan: std::sync::RwLock::new(native_plan),
            native_tool: std::sync::RwLock::new(native_tool),
            native_skill: std::sync::RwLock::new(native_skill),
            providers,
            cooldown: Mutex::new(HashMap::new()),
            stats: Mutex::new(HashMap::new()),
            limits,
            rule_patterns,
        }
    }

    /// A panic while holding a lock must not take the whole router down with it.
    fn lock<'a, T>(m: &'a Mutex<T>) -> std::sync::MutexGuard<'a, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.cache.stats()
    }

    pub fn learned_stats(&self) -> crate::learn::LearnStats {
        self.routes.stats()
    }

    pub fn stats(&self) -> HashMap<String, ModelStats> {
        let mut m = Self::lock(&self.stats).clone();
        for (name, sem) in &self.limits {
            if let Some(cap) = self.cfg.model(name).and_then(|c| c.max_concurrency) {
                m.entry(name.clone()).or_default().in_flight = cap - sem.available_permits();
            }
        }
        m
    }

    fn record(&self, model: &str, outcome: Outcome, latency_ms: u128, cost: f64, timed_out: bool) {
        let mut stats = Self::lock(&self.stats);
        let s = stats.entry(model.to_string()).or_default();
        s.calls += 1;
        match outcome {
            Outcome::Ok => s.ok += 1,
            Outcome::Error => s.errors += 1,
            Outcome::Refusal => s.refusals += 1,
            Outcome::LowConfidence => s.low_confidence += 1,
            Outcome::Empty => s.errors += 1,
        }
        if timed_out {
            s.timeouts += 1;
        }
        let l = latency_ms as f64;
        s.ewma_latency_ms = if s.calls == 1 { l } else { 0.8 * s.ewma_latency_ms + 0.2 * l };
        s.total_cost_usd += cost;
    }

    fn model_timeout(&self, m: &ModelConfig) -> Duration {
        Duration::from_secs(m.timeout_secs.unwrap_or(self.cfg.routing.default_timeout_secs))
    }

    fn in_cooldown(&self, model: &str) -> bool {
        let map = Self::lock(&self.cooldown);
        map.get(model).is_some_and(|until| *until > Instant::now())
    }

    fn mark_unhealthy(&self, model: &str) {
        let until = Instant::now() + Duration::from_secs(self.cfg.routing.error_cooldown_secs);
        Self::lock(&self.cooldown).insert(model.to_string(), until);
    }

    fn estimate_cost(&self, m: &ModelConfig, input_tokens: u32, max_tokens: u32) -> f64 {
        // Assume the answer uses a quarter of the allowance; better than assuming the max.
        let est_output = (max_tokens / 4).max(64) as f64;
        (input_tokens as f64 * m.cost.input + est_output * m.cost.output) / 1_000_000.0
    }

    /// Apply the routing metadata declared by the referenced skill/agent files.
    /// Request values win over skill values; tags are unioned.
    /// The skill-routing model's decision for a question, when it is confident and the
    /// wording is familiar: a skill name or `NO_SKILL`. None means "not sure".
    pub fn native_skill_decision(&self, q: &str) -> Option<(String, f64)> {
        let guard = self.native_skill.read().unwrap_or_else(|e| e.into_inner());
        let m = guard.as_ref()?;
        let shape = crate::learn::abstract_question(q, &self.routes.spans(q));
        let (min_conf, min_known) = self.cfg.decisions.as_ref().map(|d| (d.min_confidence, d.min_known_words)).unwrap_or((0.85, 0.7));
        let mut candidates = self.prompts.choosable();
        candidates.push(NO_SKILL.to_string());
        let (skill, p) = m.choose(&shape, &candidates)?;
        (p >= min_conf && m.known_ratio(&shape) >= min_known).then_some((skill, p))
    }

    /// The teacher picks the skill a question needs, or `NO_SKILL`. Nothing is executed.
    pub async fn teacher_skill_label(&self, model: &str, question: &str, record: bool) -> Result<Option<String>> {
        let skills = self.prompts.choosable();
        anyhow::ensure!(skills.len() >= 2, "skill labelling needs at least two skills that are not always-on");
        let m = self.cfg.model(model).ok_or_else(|| anyhow!("unknown teacher model {model}"))?;
        let provider = self.providers.get(&m.provider).ok_or_else(|| anyhow!("provider {} not initialised", m.provider))?;
        let list: Vec<String> = skills.iter().map(|n| format!("- {n}: {}", self.prompts.summary(n, 300))).collect();
        let system = format!(
            "You route a user's question to the skill whose instructions are needed to answer it.\nSkills:\n{}\n- none: no skill applies.\nAnswer with exactly one skill name.",
            list.join("\n")
        );
        let mut names: Vec<String> = skills.clone();
        names.push("none".into());
        let schema = serde_json::json!({"type": "object", "properties": {"skill": {"type": "string", "enum": names}}, "required": ["skill"], "additionalProperties": false});
        let preq = ProviderRequest {
            system_stable: Some(system),
            system: None,
            messages: vec![Message::text("user", question.to_string())],
            max_tokens: 60,
            temperature: None,
            tools: vec![],
            tool_choice: None,
            effort: None,
            response_format: Some(serde_json::json!({"type": "json_schema", "json_schema": {"name": "skill", "schema": schema, "strict": true}})),
        };
        let resp = tokio::time::timeout(Duration::from_secs(60), provider.complete(m, &preq)).await.map_err(|_| anyhow!("teacher timed out"))??;
        let text = resp.text.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```");
        let Some(choice) = serde_json::from_str::<serde_json::Value>(text).ok().and_then(|v| v["skill"].as_str().map(str::to_string)) else { return Ok(None) };
        let label = if choice == "none" { NO_SKILL.to_string() } else if skills.contains(&choice) { choice } else { return Ok(None) };
        if record {
            self.direct_label("skill", question, &label);
        }
        Ok(Some(label))
    }

    pub fn apply_prompt_meta(&self, req: &RouteRequest) -> Result<RouteRequest, String> {
        let mut r = req.clone();
        let wants_auto = self.cfg.prompts.auto_select && (r.prompt.is_none() || r.prompt.as_deref() == Some("auto"));
        if wants_auto || r.prompt.as_deref() == Some("auto") {
            let mut selected = self.prompts.select(r.query(), self.cfg.prompts.max_auto);
            // No trigger matched: ask the skill model trained from explicit choices and examples.
            if selected.iter().all(|n| self.prompts.is_always(n)) {
                if let Some((skill, _)) = self.native_skill_decision(r.query()).filter(|(s, _)| s != NO_SKILL) {
                    selected.push(skill);
                }
            }
            r.prompt = if selected.is_empty() { None } else { Some(selected.join(",")) };
        }
        let meta = self.prompts.meta(r.prompt.as_deref())?;
        r.tier = r.tier.or(meta.tier);
        r.intent = r.intent.or(meta.intent);
        r.max_latency_ms = r.max_latency_ms.or(meta.max_latency_ms);
        r.max_cost_usd = r.max_cost_usd.or(meta.max_cost_usd);
        r.model = r.model.or(meta.model);
        for t in meta.tags {
            if !r.tags.contains(&t) {
                r.tags.push(t);
            }
        }
        Ok(r)
    }

    /// Pure function of the request and config. Never touches the network.
    pub fn decide(&self, req: &RouteRequest) -> RoutingDecision {
        let merged;
        let req = match self.apply_prompt_meta(req) {
            Ok(r) => {
                merged = r;
                &merged
            }
            Err(_) => req, // unknown prompt is reported by complete()/stream() when the request runs
        };
        let sig = signals::extract(req, &self.cfg, self.matching_rule(req.query()));
        let mut reasons = Vec::new();
        if req.prompt.is_some() {
            reasons.push(format!("skill/agent prompt {:?} applied", req.prompt.as_deref().unwrap_or("")));
        }
        let max_tokens = req.max_tokens.unwrap_or(self.cfg.routing.default_max_tokens);
        let budget = req.max_cost_usd.or(self.cfg.routing.max_cost_usd);

        // 1. Target tier: request override > rule tier > agent-loop phase > intent map.
        let rule_tier = sig.matched_rule.as_ref().and_then(|n| self.cfg.routing.rules.iter().find(|r| &r.name == n)).and_then(|r| r.tier);
        let al = &self.cfg.routing.agent_loop;
        let phase_tier = match sig.phase {
            Phase::ToolSelect if al.enabled => Some(al.tool_select),
            Phase::AfterToolResult if al.enabled => Some(al.after_tool_result),
            _ => None,
        };
        let mut target = match (req.tier, rule_tier, phase_tier) {
            (Some(t), _, _) => {
                reasons.push(format!("tier forced to {t}"));
                t
            }
            (None, Some(t), _) => {
                reasons.push(format!("rule '{}' sets tier {t} (intent {})", sig.matched_rule.as_deref().unwrap_or(""), sig.intent));
                t
            }
            (None, None, Some(t)) => {
                reasons.push(format!("agent loop phase {:?} maps to tier {t}", sig.phase));
                t
            }
            (None, None, None) => {
                let t = self.cfg.tier_for(sig.intent);
                reasons.push(format!("intent {} ({}) maps to tier {t}", sig.intent, sig.intent_source));
                t
            }
        };

        let mut abstain = false;
        let mut clarify = false;
        // Uncertainty signals (vague question, weak retrieval) escalate at most one tier in total.
        let mut escalated = false;
        if sig.ambiguous && self.cfg.routing.confidence.enabled && req.tier.is_none() {
            match self.cfg.routing.confidence.on_ambiguous {
                OnAmbiguous::Clarify => {
                    clarify = true;
                    reasons.push("question is under-specified: will ask for clarification".into());
                }
                OnAmbiguous::Escalate => {
                    let bumped = target.up();
                    if bumped != target {
                        reasons.push(format!("question is under-specified: escalates {target} to {bumped}"));
                        target = bumped;
                        escalated = true;
                    }
                }
                OnAmbiguous::Ignore => {}
            }
        }
        if sig.weak_retrieval && sig.intent != Intent::Chitchat && req.tier.is_none() && !escalated && !clarify {
            match self.cfg.routing.weak_retrieval.action {
                WeakRetrievalAction::Escalate => {
                    let bumped = target.up();
                    if bumped != target {
                        reasons.push(format!(
                            "weak retrieval (top score {:?} < {}) escalates {target} to {bumped}",
                            sig.top_score, self.cfg.routing.weak_retrieval.threshold
                        ));
                        target = bumped;
                    }
                }
                WeakRetrievalAction::Abstain => {
                    abstain = true;
                    reasons.push("weak retrieval: policy says abstain".into());
                }
                WeakRetrievalAction::Ignore => {}
            }
        }

        // 2. Filter.
        let mut candidates = Vec::new();
        let mut rejected = Vec::new();

        for m in &self.cfg.models {
            // What this model would actually be asked to produce, not the raw request value.
            let out = max_tokens.min(m.max_output);
            let needed_context = ((sig.total_input_tokens + out) as f64 * 1.1) as u32;
            let cost = self.estimate_cost(m, sig.total_input_tokens, out);
            let mut reject = |why: String| {
                rejected.push(Candidate { name: m.name.clone(), tier: m.tier, estimated_cost_usd: cost, reason: why });
            };

            if let Some(forced) = &req.model {
                if &m.name != forced {
                    reject("not the forced model".into());
                    continue;
                }
            } else if !m.is_routable() {
                reject("not routable (routable: false or kind: embedding)".into());
                continue;
            }
            if let Some(missing) = sig.required_tags.iter().find(|t| !m.tags.contains(t)) {
                reject(format!("missing required tag '{missing}'"));
                continue;
            }
            if m.context_window < needed_context {
                reject(format!("context window {} < needed {}", m.context_window, needed_context));
                continue;
            }
            if let Some(b) = budget {
                if cost > b {
                    reject(format!("estimated cost ${cost:.5} exceeds budget ${b:.5}"));
                    continue;
                }
            }
            if self.in_cooldown(&m.name) {
                reject("in error cooldown".into());
                continue;
            }
            let reason = if m.tier == target {
                "matches target tier".to_string()
            } else if m.tier > target {
                "above target tier (escalation path)".to_string()
            } else {
                "below target tier (last resort)".to_string()
            };
            candidates.push(Candidate { name: m.name.clone(), tier: m.tier, estimated_cost_usd: cost, reason });
        }

        // 3. Rank: exact tier first, then higher tiers ascending, then lower tiers
        //    descending. Within a group, cheapest first.
        let key = |c: &Candidate| {
            let d = c.tier.rank() - target.rank();
            let group = if d >= 0 { d } else { 100 - d };
            (group, (c.estimated_cost_usd * 1e9) as i64)
        };
        candidates.sort_by_key(key);

        if let Some(forced) = &req.model {
            reasons.push(format!("model forced to {forced}"));
        }
        if req.model.is_none() && !sig.required_tags.is_empty() {
            reasons.push(format!("required tags {:?}", sig.required_tags));
        }
        if candidates.is_empty() {
            reasons.push("no model satisfies the constraints".into());
        } else {
            reasons.push(format!("selected {} ({})", candidates[0].name, candidates[0].reason));
        }

        RoutingDecision { target_tier: target, candidates, rejected, signals: sig, reasons, abstain, clarify }
    }

    fn grounding_for(&self, intent: Intent) -> &str {
        match intent {
            Intent::Lookup | Intent::Extraction | Intent::Summarization => &self.cfg.routing.grounding_strict,
            _ => &self.cfg.routing.grounding_open,
        }
    }

    fn build_provider_request(&self, req: &RouteRequest, m: &ModelConfig, intent: Intent) -> Result<ProviderRequest> {
        let system_stable = self.prompts.resolve(req.prompt.as_deref()).map_err(|e| anyhow!(e))?;
        let mut system_parts: Vec<String> = req
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| m.content.clone())
            .collect();

        if !req.context.is_empty() {
            let mut ctx = format!("{}\n\n<context>\n", self.grounding_for(intent));
            for (i, c) in req.context.iter().enumerate() {
                let src = c.source.as_deref().unwrap_or("unknown");
                ctx.push_str(&format!("<chunk id=\"{}\" source=\"{}\">\n{}\n</chunk>\n", i + 1, src, c.text));
            }
            ctx.push_str("</context>");
            system_parts.push(ctx);
        }

        let messages: Vec<Message> =
            req.messages.iter().filter(|m| m.role != "system").cloned().collect();

        let mut tools = req.tools.clone();
        let spec = &self.cfg.routing.speculative_answer;
        let is_tool_select = !req.tools.is_empty() && !req.messages.last().is_some_and(|m| m.tool_call_id.is_some());
        if spec.enabled && is_tool_select {
            tools.push(ToolDef {
                name: spec.tool_name.clone(),
                description: "Plan the final answer before the data arrives. Call this in the same turn as any data tool call.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "template": {"type": "string", "description": "Markdown for the final answer with placeholders for fields of the data tool's JSON result: {{field}}, {{a.b.0}}, {{rows|table}} for an array of objects, {{rows|count}}, {{_raw}} for the raw result."},
                        "needs_analysis": {"type": "boolean", "description": "true only if answering requires interpreting the data (trends, causes, comparisons, recommendations) rather than presenting it."}
                    },
                    "required": ["template", "needs_analysis"]
                }),
            });
            system_parts.push(format!(
                "Whenever you call a data tool, also call `{}` in the same turn with a template for the final answer written for that tool's expected JSON result, using {{{{field}}}} placeholders, {{{{rows|table}}}} for arrays of objects and {{{{rows|count}}}} for counts. Set needs_analysis to true only when the answer must interpret the data. Never mention templates or placeholders to the user.",
                spec.tool_name
            ));
        }
        Ok(ProviderRequest {
            system_stable,
            system: if system_parts.is_empty() { None } else { Some(system_parts.join("\n\n")) },
            messages,
            max_tokens: req.max_tokens.unwrap_or(self.cfg.routing.default_max_tokens).min(m.max_output),
            temperature: req.temperature,
            tools,
            tool_choice: req.tool_choice.clone(),
            effort: req.effort.clone(),
            response_format: req.response_format.clone(),
        })
    }

    /// Wall-clock budget for the whole request.
    fn deadline(&self, req: &RouteRequest) -> Instant {
        let ms = req.max_latency_ms.unwrap_or(self.cfg.routing.max_latency_ms).max(1);
        Instant::now() + Duration::from_millis(ms)
    }

    /// Time allowed for one attempt: the model's own timeout, capped by what is left.
    fn attempt_timeout(&self, m: &ModelConfig, deadline: Instant) -> Option<Duration> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining < Duration::from_millis(200) {
            return None;
        }
        Some(self.model_timeout(m).min(remaining))
    }

    /// A timeout only says the model is unhealthy when it was the model's own
    /// limit. Hitting the caller's request budget is the caller being impatient.
    fn timeout_is_models_own(&self, m: &ModelConfig, timeout: Duration) -> bool {
        timeout >= self.model_timeout(m)
    }

    /// Non-blocking admission. `None` means the model is saturated: move on.
    fn admit(&self, name: &str) -> Option<Option<tokio::sync::OwnedSemaphorePermit>> {
        match self.limits.get(name) {
            None => Some(None),
            Some(sem) => sem.clone().try_acquire_owned().ok().map(Some),
        }
    }

    fn is_hedged(&self, text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        self.cfg.routing.cascade.hedge_phrases.iter().any(|p| lower.contains(p.as_str()))
    }

    /// Heuristic confidence from signals and answer shape. No model call.
    fn heuristic_confidence(&self, sig: &Signals, text: &str, hedged: bool) -> Confidence {
        let grounding = match sig.top_score {
            Some(s) => s.clamp(0.0, 1.0),
            None if sig.n_chunks > 0 => 0.6,
            None => 0.35,
        };
        let cited = sig.n_chunks > 0 && text.contains('[') && text.chars().zip(text.chars().skip(1)).any(|(a, b)| a == '[' && b.is_ascii_digit());
        let mut score = 0.4 + 0.5 * grounding;
        if cited {
            score += 0.1;
        }
        if hedged {
            score *= 0.35;
        }
        if sig.ambiguous {
            score *= 0.6;
        }
        if text.trim().len() < 20 && sig.intent != Intent::Lookup && sig.intent != Intent::Chitchat {
            score *= 0.7;
        }
        // Tool-selection turn answered directly by a model below the intent's own tier:
        // the cheap model skipped the tools on a hard question. Be sceptical.
        if sig.phase == Phase::ToolSelect && self.cfg.tier_for(sig.intent) > self.cfg.routing.agent_loop.tool_select {
            score *= 0.6;
        }
        Confidence { score: score.clamp(0.0, 1.0), grounding, hedged, cited, ambiguous: sig.ambiguous, verifier: None }
    }

    /// Ask the configured verifier model to judge the answer. Returns None if it
    /// is not configured or its output is unusable; never fails the request.
    async fn verify(&self, req: &RouteRequest, answer: &str, deadline: Instant) -> Option<serde_json::Value> {
        let name = self.cfg.routing.confidence.verifier.as_ref()?;
        let m = self.cfg.model(name)?;
        if self.in_cooldown(&m.name) {
            return None;
        }
        let provider = self.providers.get(&m.provider)?;
        let timeout = self.attempt_timeout(m, deadline)?.min(Duration::from_millis(self.cfg.routing.confidence.verifier_timeout_ms.max(1)));
        let mut ctx = String::new();
        for (i, c) in req.context.iter().enumerate() {
            ctx.push_str(&format!("[{}] {}\n", i + 1, c.text));
        }
        let user = format!(
            "Question:\n{}\n\nRetrieved context:\n{}\nAnswer under review:\n{}\n\nJudge the answer. Reply with only a JSON object: {{\"grounded\": 0-1 (claims supported by the context), \"complete\": 0-1 (fully addresses the question), \"needs_clarification\": true|false (the question was too vague to answer well), \"issue\": \"one short sentence or empty\"}}",
            req.query(),
            if ctx.is_empty() { "(none)\n".to_string() } else { ctx },
            answer
        );
        let preq = ProviderRequest {
            system_stable: None,
            system: Some("You are a strict grader for a retrieval-augmented assistant. Output JSON only.".into()),
            messages: vec![Message::text("user", user)],
            max_tokens: 200,
            temperature: None,
            tools: vec![],
            tool_choice: None,
            effort: None,
            response_format: None,
        };
        let started = Instant::now();
        let resp = match tokio::time::timeout(timeout, provider.complete(m, &preq)).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                warn!(verifier = %m.name, error = %e, "verifier failed; using heuristic confidence");
                if matches!(e, ProviderError::Retryable(_) | ProviderError::Transport(_)) {
                    self.mark_unhealthy(&m.name);
                }
                self.record(&m.name, Outcome::Error, started.elapsed().as_millis(), 0.0, false);
                return None;
            }
            Err(_) => {
                warn!(verifier = %m.name, "verifier exceeded verifier_timeout_ms; using heuristic confidence");
                self.record(&m.name, Outcome::Error, started.elapsed().as_millis(), 0.0, true);
                return None;
            }
        };
        self.record(&m.name, Outcome::Ok, started.elapsed().as_millis(), (resp.input_tokens as f64 * m.cost.input + resp.output_tokens as f64 * m.cost.output) / 1_000_000.0, false);
        let text = resp.text;
        let start = text.find('{')?;
        let end = text.rfind('}')?;
        serde_json::from_str(&text[start..=end]).ok()
    }

    /// Full confidence: heuristic, blended with the verifier when configured.
    async fn confidence(&self, req: &RouteRequest, sig: &Signals, text: &str, hedged: bool, deadline: Instant) -> Confidence {
        let mut c = self.heuristic_confidence(sig, text, hedged);
        // Confident answers skip the verifier round-trip entirely.
        if c.score >= self.cfg.routing.confidence.verify_below {
            return c;
        }
        if let Some(v) = self.verify(req, text, deadline).await {
            let g = v["grounded"].as_f64().unwrap_or(0.5).clamp(0.0, 1.0);
            let comp = v["complete"].as_f64().unwrap_or(0.5).clamp(0.0, 1.0);
            c.score = (0.4 * c.score + 0.6 * (0.6 * g + 0.4 * comp)).clamp(0.0, 1.0);
            if v["needs_clarification"].as_bool() == Some(true) {
                c.ambiguous = true;
            }
            c.verifier = Some(v);
        }
        c
    }

    /// Have the cheapest candidate write one clarifying question.
    async fn clarify(&self, req: &RouteRequest, decision: RoutingDecision) -> Result<Completion> {
        let deadline = self.deadline(req);
        let mut attempts = Vec::new();
        // Writing one question is trivial: cheapest first, tier irrelevant.
        let mut order = decision.candidates.clone();
        order.sort_by_key(|c| ((c.estimated_cost_usd * 1e9) as i64, c.tier.rank()));
        for cand in &order {
            let m = self.cfg.model(&cand.name).expect("candidate came from config");
            let Some(provider) = self.providers.get(&m.provider) else { continue };
            let Some(timeout) = self.attempt_timeout(m, deadline) else { break };
            let mut preq = self.build_provider_request(req, m, Intent::Chitchat)?;
            preq.system = Some(format!(
                "{}\n\nThe user's question is under-specified. Do not answer it. Ask exactly one short clarifying question that would let you answer precisely (for example which region, product, period or metric).",
                preq.system.unwrap_or_default()
            ));
            preq.max_tokens = preq.max_tokens.min(120);
            let started = Instant::now();
            match tokio::time::timeout(timeout, provider.complete(m, &preq)).await {
                Ok(Ok(resp)) if !resp.text.trim().is_empty() => {
                    let usage = Usage { input_tokens: resp.input_tokens, output_tokens: resp.output_tokens, cost_usd: (resp.input_tokens as f64 * m.cost.input + resp.output_tokens as f64 * m.cost.output) / 1_000_000.0 };
                    self.record(&m.name, Outcome::Ok, started.elapsed().as_millis(), usage.cost_usd, false);
                    attempts.push(Attempt { model: m.name.clone(), outcome: Outcome::Ok, detail: "clarifying question".into(), latency_ms: started.elapsed().as_millis(), usage: usage.clone() });
                    return Ok(Completion {
                        text: resp.text,
                        tool_calls: vec![],
                        model: m.name.clone(),
                        provider_model: m.model.clone(),
                        usage,
                        confidence: Confidence { score: 0.0, ambiguous: true, ..Default::default() },
                        decision,
                        attempts,
                        abstained: false,
                        clarification: true,
                        cache: None,
                    });
                }
                Ok(Ok(_)) => attempts.push(Attempt { model: m.name.clone(), outcome: Outcome::Empty, detail: String::new(), latency_ms: started.elapsed().as_millis(), usage: Usage::default() }),
                Ok(Err(e)) => attempts.push(Attempt { model: m.name.clone(), outcome: Outcome::Error, detail: e.to_string(), latency_ms: started.elapsed().as_millis(), usage: Usage::default() }),
                Err(_) => attempts.push(Attempt { model: m.name.clone(), outcome: Outcome::Error, detail: "timed out".into(), latency_ms: timeout.as_millis(), usage: Usage::default() }),
            }
        }
        Err(anyhow!("could not produce a clarifying question: {:?}", attempts))
    }

    /// Index of the first rule matching the query, by keywords or pattern.
    pub fn matching_rule(&self, query: &str) -> Option<usize> {
        let lower = query.to_lowercase();
        self.cfg.routing.rules.iter().enumerate().find_map(|(i, r)| {
            let by_words = r.any.iter().any(|k| lower.contains(&k.to_lowercase()));
            let by_pattern = self.rule_patterns.get(i).and_then(|p| p.as_ref()).is_some_and(|p| p.is_match(query));
            (by_words || by_pattern).then_some(i)
        })
    }

    /// Deterministic tool call from a `respond_with_tool` rule: no model call at all.
    /// Only fires on a tool-selection turn when the request offers that tool.
    fn rule_tool_call(&self, req: &RouteRequest, decision: &RoutingDecision) -> Option<Completion> {
        if decision.signals.phase != Phase::ToolSelect {
            return None;
        }
        let i = self.matching_rule(req.query())?;
        let rule = &self.cfg.routing.rules[i];
        let rwt = rule.respond_with_tool.as_ref()?;
        if !req.tools.iter().any(|t| t.name == rwt.name) {
            return None;
        }
        let mut args = rwt.arguments.to_string();
        if let Some(re) = self.rule_patterns.get(i).and_then(|p| p.as_ref()) {
            if let Some(caps) = re.captures(req.query()) {
                for name in re.capture_names().flatten() {
                    if let Some(m) = caps.name(name) {
                        let val = serde_json::to_string(m.as_str()).unwrap_or_default();
                        let val = val.trim_matches('"');
                        args = args.replace(&format!("{{{{{name}}}}}"), val);
                    }
                }
            }
        }
        let call = ToolCall { id: format!("call_{}", uuid::Uuid::new_v4().simple()), name: rwt.name.clone(), arguments: args };
        let attempt = Attempt { model: format!("rule:{}", rule.name), outcome: Outcome::Ok, detail: "deterministic tool call from rule".into(), latency_ms: 0, usage: Usage::default() };
        Some(Completion {
            text: String::new(),
            tool_calls: vec![call],
            model: format!("rule:{}", rule.name),
            provider_model: String::new(),
            usage: Usage::default(),
            confidence: Confidence { score: 1.0, grounding: 1.0, ..Default::default() },
            decision: decision.clone(),
            attempts: vec![attempt],
            abstained: false,
            clarification: false,
            cache: None,
        })
    }

    /// Scope for learned routes: prompt, tenant key, tags and the offered tool set.
    fn route_scope(&self, req: &RouteRequest) -> u64 {
        let mut s = String::new();
        s.push_str(req.prompt.as_deref().unwrap_or(""));
        s.push('\u{1}');
        s.push_str(req.cache_key.as_deref().unwrap_or(""));
        s.push('\u{1}');
        s.push_str(&req.tags.join(","));
        s.push('\u{1}');
        let mut names: Vec<&str> = req.tools.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        s.push_str(&names.join(","));
        hash64(&s)
    }

    /// Serve a tool-selection turn from a learned route, if one fits the question's shape.
    async fn learned_call(&self, req: &RouteRequest, decision: &RoutingDecision, deadline: Instant) -> Option<Completion> {
        let lr = &self.cfg.routing.learned_routes;
        if !lr.enabled || decision.signals.phase != Phase::ToolSelect || req.model.is_some() || req.cache_mode == CacheMode::Bypass {
            return None;
        }
        let scope = self.route_scope(req);
        let q = req.query();
        // Exact shape needs no embedding; only embed when there is something to compare against.
        let vector = if self.routes.fill(scope, q, None).is_some() {
            None
        } else if self.cfg.cache.semantic.is_some() {
            let spans = self.routes.spans(q);
            self.embed_query(&crate::learn::abstract_question(q, &spans), deadline).await
        } else {
            return None;
        };
        // The first fill above consumed a hit counter only if it matched; call again to get the Fill.
        let fill = self.routes.fill(scope, q, vector.as_deref())?;
        if !req.tools.iter().any(|t| t.name == fill.tool) {
            return None;
        }
        let call = ToolCall { id: format!("call_{}", uuid::Uuid::new_v4().simple()), name: fill.tool.clone(), arguments: fill.arguments.clone() };
        {
            let mut g = Self::lock(&self.served_routes);
            if g.len() > 100_000 {
                g.retain(|_, (_, t)| t.elapsed() < Duration::from_secs(3600));
            }
            g.insert(call.id.clone(), (fill.key, Instant::now()));
        }
        if let Some(t) = &fill.answer_template {
            let args = serde_json::from_str(&fill.arguments).unwrap_or(serde_json::Value::Null);
            Self::lock(&self.pending_templates).insert(call.id.clone(), PendingTemplate { template: t.clone(), needs_analysis: fill.needs_analysis, created: Instant::now(), args });
        }
        let detail = match fill.similarity {
            Some(sim) => format!("learned route (shape \"{}\", similarity {sim:.3})", fill.shape),
            None => format!("learned route (shape \"{}\")", fill.shape),
        };
        let attempt = Attempt { model: "learned-route".into(), outcome: Outcome::Ok, detail, latency_ms: 0, usage: Usage::default() };
        Some(Completion {
            text: String::new(),
            tool_calls: vec![call],
            model: "learned-route".into(),
            provider_model: String::new(),
            usage: Usage::default(),
            confidence: Confidence { score: 0.95, grounding: 1.0, cited: false, hedged: false, ambiguous: false, verifier: None },
            decision: decision.clone(),
            attempts: vec![attempt],
            abstained: false,
            clarification: false,
            cache: None,
        })
    }

    /// Record a model-produced tool-selection turn as a route candidate. It is
    /// learned only when its result comes back clean (see `promote_candidate`).
    async fn learn_from(&self, req: &RouteRequest, visible: &[ToolCall], deadline: Instant) {
        let lr = &self.cfg.routing.learned_routes;
        if !(lr.enabled || self.decisions_for(false)) || visible.len() != 1 || req.cache_mode == CacheMode::Bypass {
            return;
        }
        let is_tool_select = !req.tools.is_empty() && !req.messages.last().is_some_and(|m| m.tool_call_id.is_some());
        if !is_tool_select {
            return;
        }
        let call = &visible[0];
        if !lr.allow_tools.is_empty() && !lr.allow_tools.iter().any(|t| t == &call.name) {
            return;
        }
        if let Err(why) = crate::learn::learnable_arguments(&call.arguments, &lr.deny_arg_names, lr.max_arg_len) {
            info!(tool = %call.name, %why, "not learnable");
            return;
        }
        // A turn that already contains earlier tool calls or error results is a retry: never learn from it.
        if req.messages.iter().any(|m| !m.tool_calls.is_empty() || m.tool_call_id.is_some()) {
            return;
        }
        let template = Self::lock(&self.pending_templates).get(&call.id).cloned();
        let scope = self.route_scope(req);
        let q = req.query();
        let vector = if self.cfg.cache.semantic.is_some() {
            let spans = self.routes.spans(q);
            self.embed_query(&crate::learn::abstract_question(q, &spans), deadline).await
        } else {
            None
        };
        let mut g = Self::lock(&self.route_candidates);
        let ttl = Duration::from_secs(lr.ttl_secs.max(1)).min(Duration::from_secs(3600));
        g.retain(|_, c| c.created.elapsed() < ttl);
        g.insert(call.id.clone(), RouteCandidate { scope, question: q.to_string(), call: call.clone(), template, vector, created: Instant::now() });
    }

    /// Does a tool result look like a failure? Error keys, error-ish status, or
    /// plain text starting with an error word.
    fn result_is_error(content: &str) -> bool {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return true;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(o) = v.as_object() {
                if o.get("error").is_some_and(|e| !e.is_null() && e != false) || o.get("errors").is_some_and(|e| !e.is_null()) {
                    return true;
                }
                if o.get("ok") == Some(&serde_json::Value::Bool(false)) || o.get("success") == Some(&serde_json::Value::Bool(false)) {
                    return true;
                }
                if let Some(st) = o.get("status").and_then(|s| s.as_str()) {
                    if ["error", "failed", "failure"].contains(&st.to_lowercase().as_str()) {
                        return true;
                    }
                }
            }
            return false;
        }
        let lower = trimmed.to_lowercase();
        ["error", "exception", "failed", "traceback"].iter().any(|w| lower.starts_with(w))
    }

    /// On the result hop: if the result for a candidate is clean and the turn is
    /// still a single clean call, learn the shape.
    fn promote_candidate(&self, req: &RouteRequest) {
        if !(self.cfg.routing.learned_routes.enabled || self.decisions_for(false)) {
            return;
        }
        let Some(last) = req.messages.last() else { return };
        let Some(call_id) = last.tool_call_id.as_deref() else { return };
        // Self-healing: a call we filled from a learned route reports back through its result.
        if let Some((key, _)) = Self::lock(&self.served_routes).remove(call_id) {
            let ok = !Self::result_is_error(&last.content);
            if let Some(shape) = self.routes.feedback(key, ok, self.cfg.heal.quarantine_after_failures) {
                warn!(shape = %shape, "learned route quarantined after error results");
                if let Some(st) = &self.store {
                    st.heal("quarantine", &shape, "served calls returned error results");
                }
            }
        }
        let Some(cand) = Self::lock(&self.route_candidates).remove(call_id) else { return };
        if !Self::result_is_error(&last.content) && self.decisions_for(false) {
            let first = RouteRequest { messages: vec![Message::text("user", cand.question.clone())], tools: req.tools.clone(), prompt: req.prompt.clone(), cache_key: req.cache_key.clone(), tags: req.tags.clone(), ..Default::default() };
            self.label(&first, "tool", &cand.call.name);
        }
        if Self::result_is_error(&last.content) {
            info!(tool = %cand.call.name, "tool result was an error; shape not learned");
            return;
        }
        // Exactly one tool call and one result in the transcript, and no earlier errors.
        let n_calls: usize = req.messages.iter().map(|m| m.tool_calls.len()).sum();
        let n_results = req.messages.iter().filter(|m| m.tool_call_id.is_some()).count();
        if n_calls != 1 || n_results != 1 {
            return;
        }
        if !self.cfg.routing.learned_routes.enabled {
            return;
        }
        if let Some(shape) = self.routes.learn(cand.scope, &cand.question, &cand.call.name, &cand.call.arguments, cand.template.as_ref().map(|t| t.template.as_str()), cand.template.as_ref().is_some_and(|t| t.needs_analysis), cand.vector) {
            info!(shape = %shape, tool = %cand.call.name, "learned route from a clean turn");
        }
    }

    /// A template must not carry data of its own. Strip placeholders and reject
    /// anything that still contains a code, date, number or a known entity name:
    /// such literals came from one user's answer and would be replayed to others.
    fn template_is_clean(&self, template: &str) -> bool {
        let mut stripped = String::new();
        let mut rest = template;
        while let Some(start) = rest.find("{{") {
            stripped.push_str(&rest[..start]);
            match rest[start..].find("}}") {
                Some(end) => rest = &rest[start + end + 2..],
                None => break,
            }
        }
        stripped.push_str(rest);
        // Markup attributes such as id="..." were handled by placeholders; what remains is prose.
        let spans = self.routes.spans(&stripped);
        !spans.iter().any(|sp| sp.kind != crate::learn::SlotType::Text || self.routes.is_vocab(&sp.value))
    }

    /// Per-user admission: token bucket on requests, daily cap on spend.
    pub fn admit_user(&self, user: Option<&str>) -> Result<(), BudgetDenied> {
        let Some(pu) = &self.cfg.server.per_user else { return Ok(()) };
        let Some(user) = user else {
            return match pu.anonymous {
                crate::config::AnonymousPolicy::Allow => Ok(()),
                crate::config::AnonymousPolicy::Reject => Err(BudgetDenied { message: "per-user budgets are on: send X-Pankh-User".into(), retry_after_secs: 0 }),
            };
        };
        let day = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() / 86_400).unwrap_or(0);
        let rate_per_sec = pu.requests_per_minute.max(1) as f64 / 60.0;
        let capacity = (pu.burst.max(1)) as f64;
        let mut g = Self::lock(&self.user_budgets);
        if g.len() > 100_000 {
            g.retain(|_, b| b.last.elapsed() < Duration::from_secs(3600));
        }
        let b = g.entry(user.to_string()).or_insert(UserBudget { tokens: capacity, last: Instant::now(), day, spent_usd: 0.0 });
        if b.day != day {
            b.day = day;
            b.spent_usd = 0.0;
        }
        let elapsed = b.last.elapsed().as_secs_f64();
        b.tokens = (b.tokens + elapsed * rate_per_sec).min(capacity);
        b.last = Instant::now();
        if let Some(cap) = pu.daily_cost_usd {
            if b.spent_usd >= cap {
                let secs_left = 86_400 - (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) % 86_400);
                return Err(BudgetDenied { message: format!("daily model spend cap of ${cap:.2} reached for this user"), retry_after_secs: secs_left });
            }
        }
        if b.tokens < 1.0 {
            let wait = ((1.0 - b.tokens) / rate_per_sec).ceil() as u64;
            return Err(BudgetDenied { message: format!("rate limit: {} requests/minute per user (burst {})", pu.requests_per_minute, pu.burst), retry_after_secs: wait.max(1) });
        }
        b.tokens -= 1.0;
        Ok(())
    }

    /// Charge model spend to the user's daily budget.
    pub fn charge_user(&self, user: Option<&str>, cost_usd: f64) {
        if self.cfg.server.per_user.is_none() || cost_usd <= 0.0 {
            return;
        }
        let Some(user) = user else { return };
        let mut g = Self::lock(&self.user_budgets);
        if let Some(b) = g.get_mut(user) {
            b.spent_usd += cost_usd;
        }
    }

    /// Save learned routes to `persist_path`, if configured.
    pub fn persist(&self) -> std::io::Result<()> {
        if let Some(st) = &self.store {
            st.put_kv("learned_routes", serde_json::to_vec(&self.routes.export()).unwrap_or_default());
            st.flush();
        }
        match &self.cfg.routing.learned_routes.persist_path {
            Some(p) => self.routes.save(std::path::Path::new(p)),
            None => Ok(()),
        }
    }

    /// Ingest one completed turn from a log: question, the tool call that was made,
    /// its (clean) result, and optionally the final answer or an answer template.
    /// No model is called. Returns the shape learned, if any.
    pub fn ingest_turn(&self, turn: &ObservedTurn) -> Option<String> {
        let lr = &self.cfg.routing.learned_routes;
        if !lr.enabled {
            return None;
        }
        if !lr.allow_tools.is_empty() && !lr.allow_tools.iter().any(|t| t == &turn.tool) {
            return None;
        }
        crate::learn::learnable_arguments(&turn.arguments, &lr.deny_arg_names, lr.max_arg_len).ok()?;
        if let Some(r) = &turn.tool_result {
            if Self::result_is_error(r) {
                return None;
            }
        }
        let req = RouteRequest {
            messages: vec![Message::text("user", turn.question.clone())],
            tags: turn.tags.clone(),
            prompt: turn.prompt.clone(),
            cache_key: turn.cache_key.clone(),
            tools: turn.tool_names.iter().map(|n| ToolDef { name: n.clone(), description: String::new(), parameters: serde_json::json!({"type": "object"}) }).collect(),
            ..Default::default()
        };
        let scope = self.route_scope(&req);
        let template = turn
            .answer_template
            .clone()
            .or_else(|| {
                // Derive a template from the final answer: replace result and argument values that appear in it.
                let (result, answer) = (turn.tool_result.as_ref()?, turn.answer.as_ref()?);
                derive_answer_template_with_args(answer, result, &turn.arguments)
            })
            .filter(|t| {
                let clean = self.template_is_clean(t);
                if !clean {
                    info!("answer template rejected: it still contains literal data");
                }
                clean
            });
        let shape = self.routes.learn(scope, &turn.question, &turn.tool, &turn.arguments, template.as_deref(), turn.needs_analysis, None)?;
        // A log line is evidence in itself: count it as an observation per line (weight is the caller's choice).
        Some(shape)
    }

    /// Single-flight: if an identical request is already being answered, wait for it
    /// (bounded by the deadline) and then read the cache. Returns Some(leader token)
    /// when this request is the leader and must answer.
    async fn single_flight_wait(&self, key: u64, deadline: Instant) -> Option<Arc<tokio::sync::Notify>> {
        let existing = {
            let mut g = Self::lock(&self.in_flight);
            match g.get(&key) {
                Some(n) => n.clone(),
                None => {
                    let n = Arc::new(tokio::sync::Notify::new());
                    g.insert(key, n.clone());
                    return Some(n); // this request leads
                }
            }
        };
        // Follower: wait for the leader (bounded by the deadline), then the caller
        // re-reads the cache. A miss there means the leader's answer was not
        // cacheable, and the caller answers itself.
        let remaining = deadline.saturating_duration_since(Instant::now());
        if !remaining.is_zero() {
            let _ = tokio::time::timeout(remaining, existing.notified()).await;
        }
        None
    }

    fn single_flight_done(&self, key: u64, token: &Arc<tokio::sync::Notify>) {
        let mut g = Self::lock(&self.in_flight);
        if g.get(&key).is_some_and(|n| Arc::ptr_eq(n, token)) {
            g.remove(&key);
        }
        token.notify_waiters();
    }

    /// Pull the hidden template call out of a tool-call list and remember the
    /// template for every real call in the same turn. Returns the visible calls.
    fn absorb_template_calls(&self, calls: Vec<ToolCall>) -> Vec<ToolCall> {
        let spec = &self.cfg.routing.speculative_answer;
        if !spec.enabled || !calls.iter().any(|c| c.name == spec.tool_name) {
            return calls;
        }
        let mut template: Option<PendingTemplate> = None;
        let mut visible = Vec::new();
        for c in calls {
            if c.name == spec.tool_name {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&c.arguments) {
                    if let Some(t) = v["template"].as_str().filter(|t| !t.trim().is_empty()) {
                        if self.template_is_clean(t) {
                            template = Some(PendingTemplate { template: t.to_string(), needs_analysis: v["needs_analysis"].as_bool().unwrap_or(false), created: Instant::now(), args: serde_json::Value::Null });
                        } else {
                            warn!("planning model's answer template contained literal data; discarded");
                        }
                    }
                }
            } else {
                visible.push(c);
            }
        }
        if let Some(t) = template {
            let mut g = Self::lock(&self.pending_templates);
            let ttl = Duration::from_secs(spec.ttl_secs.max(1));
            g.retain(|_, p| p.created.elapsed() < ttl);
            for c in &visible {
                let mut pt = t.clone();
                pt.args = serde_json::from_str(&c.arguments).unwrap_or(serde_json::Value::Null);
                g.insert(c.id.clone(), pt);
            }
        }
        visible
    }

    /// On the hop after a tool result: render the stored template if the model
    /// did not flag the data as needing analysis and every placeholder resolves.
    fn speculative_render(&self, req: &RouteRequest, decision: &RoutingDecision) -> Option<Completion> {
        let spec = &self.cfg.routing.speculative_answer;
        if !spec.enabled || decision.signals.phase != Phase::AfterToolResult {
            return None;
        }
        let last = req.messages.last()?;
        let call_id = last.tool_call_id.as_deref()?;
        let pending = Self::lock(&self.pending_templates).remove(call_id)?;
        if pending.needs_analysis {
            return None;
        }
        let output: serde_json::Value = serde_json::from_str(&last.content).unwrap_or(serde_json::Value::Null);
        // Result fields plus the call's arguments under "args", so templates can say {{args.region}}.
        let mut merged = match &output { serde_json::Value::Object(o) => serde_json::Value::Object(o.clone()), other => serde_json::json!({"_value": other}) };
        if let serde_json::Value::Object(m) = &mut merged {
            m.insert("args".into(), pending.args.clone());
        }
        let text = render_template(&pending.template, &merged, &last.content, &HashMap::new(), spec.max_rows);
        // Unfilled placeholders mean the model guessed the result shape wrong: let a model answer.
        let placeholders = pending.template.matches("{{").count();
        let unresolved = placeholders > 0 && text.trim().is_empty();
        let empties = placeholders.saturating_sub(count_filled(&pending.template, &merged, &last.content, spec.max_rows));
        if unresolved || empties > 0 {
            return None;
        }
        let attempt = Attempt { model: "speculative-template".into(), outcome: Outcome::Ok, detail: "rendered from the planning model's answer template".into(), latency_ms: 0, usage: Usage::default() };
        Some(Completion {
            text,
            tool_calls: vec![],
            model: "speculative-template".into(),
            provider_model: String::new(),
            usage: Usage::default(),
            confidence: Confidence { score: 0.95, grounding: 1.0, cited: false, hedged: false, ambiguous: false, verifier: None },
            decision: decision.clone(),
            attempts: vec![attempt],
            abstained: false,
            clarification: false,
            cache: None,
        })
    }

    /// Deterministic final answer on the hop after a rule-generated tool call
    /// returned: render the rule's `after_result` template from the tool output.
    fn rule_after_result(&self, req: &RouteRequest, decision: &RoutingDecision) -> Option<Completion> {
        if decision.signals.phase != Phase::AfterToolResult {
            return None;
        }
        let i = self.matching_rule(req.query())?;
        let rule = &self.cfg.routing.rules[i];
        let rwt = rule.respond_with_tool.as_ref()?;
        let after = rwt.after_result.as_ref()?;
        // The last message is the tool result; its call must be the rule's tool.
        let last = req.messages.last()?;
        let call_id = last.tool_call_id.as_deref()?;
        let called = req.messages.iter().rev().find_map(|m| m.tool_calls.iter().find(|t| t.id == call_id))?;
        if called.name != rwt.name {
            return None;
        }
        let output: serde_json::Value = serde_json::from_str(&last.content).unwrap_or(serde_json::Value::Null);
        let mut vars: HashMap<String, String> = HashMap::new();
        if let Some(re) = self.rule_patterns.get(i).and_then(|p| p.as_ref()) {
            if let Some(caps) = re.captures(req.query()) {
                for name in re.capture_names().flatten() {
                    if let Some(m) = caps.name(name) {
                        vars.insert(name.to_string(), m.as_str().to_string());
                    }
                }
            }
        }
        let text = render_template(&after.template, &output, &last.content, &vars, after.max_rows);
        let attempt = Attempt { model: format!("rule:{}", rule.name), outcome: Outcome::Ok, detail: "deterministic answer from after_result template".into(), latency_ms: 0, usage: Usage::default() };
        Some(Completion {
            text,
            tool_calls: vec![],
            model: format!("rule:{}", rule.name),
            provider_model: String::new(),
            usage: Usage::default(),
            confidence: Confidence { score: 1.0, grounding: 1.0, cited: false, hedged: false, ambiguous: false, verifier: None },
            decision: decision.clone(),
            attempts: vec![attempt],
            abstained: false,
            clarification: false,
            cache: None,
        })
    }

    /// (exact key, scope) for the cache. Scope excludes the question so semantic
    /// hits can match paraphrases with the same context, tools and prompt.
    fn cache_keys(&self, req: &RouteRequest) -> (u64, u64) {
        let mut scope = String::new();
        scope.push_str(req.prompt.as_deref().unwrap_or(""));
        scope.push('\u{1}');
        scope.push_str(req.cache_key.as_deref().unwrap_or(""));
        scope.push('\u{1}');
        scope.push_str(&req.tags.join(","));
        scope.push('\u{1}');
        for t in &req.tools {
            scope.push_str(&t.name);
            scope.push_str(&t.parameters.to_string());
            scope.push('\u{2}');
        }
        scope.push_str(&format!("{:?}{:?}{:?}{:?}{}", req.tier, req.intent, req.model, req.effort, req.response_format.as_ref().map(|v| v.to_string()).unwrap_or_default()));
        if self.cfg.cache.scope == crate::config::CacheScope::ExactContext {
            for c in &req.context {
                scope.push_str(&c.text);
                scope.push('\u{2}');
            }
        }
        // Prior turns are part of the exact key: the same last question after a
        // different history is a different request.
        let mut exact = scope.clone();
        for m in &req.messages {
            exact.push_str(&m.role);
            exact.push(':');
            exact.push_str(&m.content);
            for t in &m.tool_calls {
                exact.push_str(&t.name);
                exact.push_str(&t.arguments);
            }
            exact.push('\u{3}');
        }
        (hash64(&exact), hash64(&scope))
    }

    async fn embed_query(&self, query: &str, deadline: Instant) -> Option<Vec<f32>> {
        let sem = self.cfg.cache.semantic.as_ref()?;
        let m = self.cfg.model(&sem.embedding_model)?;
        if self.in_cooldown(&m.name) {
            return None;
        }
        let provider = self.providers.get(&m.provider)?;
        let timeout = self.attempt_timeout(m, deadline)?.min(Duration::from_millis(2000));
        let started = Instant::now();
        match tokio::time::timeout(timeout, provider.embed(m, query.trim())).await {
            Ok(Ok(v)) => {
                self.record(&m.name, Outcome::Ok, started.elapsed().as_millis(), 0.0, false);
                Some(v)
            }
            Ok(Err(e)) => {
                warn!(model = %m.name, error = %e, "embedding failed; semantic cache skipped");
                if matches!(e, ProviderError::Retryable(_) | ProviderError::Transport(_)) {
                    self.mark_unhealthy(&m.name);
                }
                None
            }
            Err(_) => {
                warn!(model = %m.name, "embedding timed out; semantic cache skipped");
                None
            }
        }
    }

    /// The cache may be used for this request at all. A question-only cache
    /// without a tenant/scope key is a data leak waiting to happen, so it is off.
    fn cache_allowed(&self, req: &RouteRequest) -> bool {
        if !self.cfg.cache.enabled || req.cache_mode == CacheMode::Bypass {
            return false;
        }
        if self.cfg.cache.scope == crate::config::CacheScope::QuestionOnly && self.cfg.cache.require_cache_key && req.cache_key.as_deref().unwrap_or("").is_empty() {
            warn!("cache skipped: scope is question_only and the request has no cache_key (tenant / scope / data version)");
            return false;
        }
        true
    }

    /// Cache lookup: exact first, then semantic (single-turn questions only, since a
    /// paraphrase of the last turn says nothing about the history).
    async fn cache_lookup(&self, req: &RouteRequest, decision: &RoutingDecision, deadline: Instant) -> (Option<Completion>, Option<Vec<f32>>) {
        if !self.cache_allowed(req) || req.cache_mode != CacheMode::Use || decision.clarify || decision.abstain {
            return (None, None);
        }
        let (exact, scope) = self.cache_keys(req);
        if let Some((a, info)) = self.cache.get_exact(exact, scope) {
            return (Some(self.completion_from_cache(a, info, decision)), None);
        }
        let single_turn = req.messages.iter().filter(|m| m.role != "system").count() == 1;
        let mut vector = None;
        if single_turn && self.cfg.cache.semantic.is_some() {
            vector = self.embed_query(req.query(), deadline).await;
            if let Some(v) = &vector {
                if let Some((a, info)) = self.cache.get_semantic(v, scope) {
                    return (Some(self.completion_from_cache(a, info, decision)), vector);
                }
            }
        }
        self.cache.miss();
        (None, vector)
    }

    fn completion_from_cache(&self, a: CachedAnswer, info: CacheInfo, decision: &RoutingDecision) -> Completion {
        Completion {
            text: a.text,
            tool_calls: a.tool_calls,
            model: a.model,
            provider_model: a.provider_model,
            usage: Usage { cost_usd: 0.0, ..a.usage },
            confidence: a.confidence,
            decision: decision.clone(),
            attempts: vec![],
            abstained: false,
            clarification: false,
            cache: Some(info),
        }
    }

    fn cache_store(&self, req: &RouteRequest, c: &Completion, vector: Option<Vec<f32>>) {
        if !self.cache_allowed(req) || c.cache.is_some() {
            return;
        }
        if !self.cache.cacheable(&c.confidence, c.clarification, c.abstained, !c.tool_calls.is_empty()) {
            return;
        }
        let (exact, scope) = self.cache_keys(req);
        let single_turn = req.messages.iter().filter(|m| m.role != "system").count() == 1;
        self.cache.put(exact, scope, CachedAnswer { text: c.text.clone(), tool_calls: c.tool_calls.clone(), model: c.model.clone(), provider_model: c.provider_model.clone(), usage: c.usage.clone(), confidence: c.confidence.clone() }, if single_turn { vector } else { None });
    }

    /// One provider call with admission, timeout and health bookkeeping.
    /// Returns the attempt record and the raw response when there was one.
    async fn run_attempt(&self, req: &RouteRequest, m: &ModelConfig, intent: Intent, deadline: Instant, attempt_no: usize) -> Result<(Attempt, Option<ProviderResponse>)> {
        let provider = self
            .providers
            .get(&m.provider)
            .ok_or_else(|| anyhow!("provider {} not initialised", m.provider))?;
        let Some(timeout) = self.attempt_timeout(m, deadline) else {
            return Ok((Attempt { model: m.name.clone(), outcome: Outcome::Error, detail: "request latency budget exhausted".into(), latency_ms: 0, usage: Usage::default() }, None));
        };
        let Some(_permit) = self.admit(&m.name) else {
            return Ok((Attempt { model: m.name.clone(), outcome: Outcome::Error, detail: "at max_concurrency, skipped".into(), latency_ms: 0, usage: Usage::default() }, None));
        };
        let preq = self.build_provider_request(req, m, intent)?;
        info!(model = %m.name, tier = %m.tier, attempt = attempt_no, timeout_ms = timeout.as_millis(), "calling");
        let started = Instant::now();
        let (result, budget_cut) = match tokio::time::timeout(timeout, provider.complete(m, &preq)).await {
            Ok(r) => (r, false),
            Err(_) => {
                let own = self.timeout_is_models_own(m, timeout);
                let why = if own { "model timeout" } else { "request latency budget" };
                (Err(ProviderError::Retryable(format!("timed out after {} ms ({why})", timeout.as_millis()))), !own)
            }
        };
        let latency_ms = started.elapsed().as_millis();
        match result {
            Err(e) => {
                warn!(model = %m.name, error = %e, "provider error");
                let timed_out = e.to_string().contains("timed out");
                if !budget_cut && matches!(e, ProviderError::Retryable(_) | ProviderError::Transport(_)) {
                    self.mark_unhealthy(&m.name);
                }
                self.record(&m.name, Outcome::Error, latency_ms, 0.0, timed_out);
                Ok((Attempt { model: m.name.clone(), outcome: Outcome::Error, detail: e.to_string(), latency_ms, usage: Usage::default() }, None))
            }
            Ok(mut resp) => {
                resp.tool_calls = self.absorb_template_calls(resp.tool_calls);
                self.learn_from(req, &resp.tool_calls, deadline).await;
                let usage = Usage {
                    input_tokens: resp.input_tokens,
                    output_tokens: resp.output_tokens,
                    cost_usd: (resp.input_tokens as f64 * m.cost.input + resp.output_tokens as f64 * m.cost.output) / 1_000_000.0,
                };
                let outcome = if resp.stop_reason == StopReason::Refusal {
                    Outcome::Refusal
                } else if !resp.tool_calls.is_empty() {
                    // A tool request is a complete, useful turn: never treat it as empty or hedged.
                    Outcome::Ok
                } else if resp.text.trim().is_empty() {
                    if resp.stop_reason == StopReason::MaxTokens {
                        warn!(model = %m.name, "empty answer: output budget exhausted (likely spent on thinking)");
                    }
                    Outcome::Empty
                } else if matches!(intent, Intent::Lookup | Intent::Extraction | Intent::Summarization) && self.is_hedged(&resp.text) {
                    Outcome::LowConfidence
                } else {
                    Outcome::Ok
                };
                self.record(&m.name, outcome, latency_ms, usage.cost_usd, false);
                let detail = match (outcome, resp.stop_reason) {
                    (Outcome::Empty, StopReason::MaxTokens) => "empty: max_tokens exhausted before any answer (thinking?)".to_string(),
                    _ => format!("stop_reason={:?}", resp.stop_reason),
                };
                Ok((Attempt { model: m.name.clone(), outcome, detail, latency_ms, usage }, Some(resp)))
            }
        }
    }

    /// Turn a successful provider response into a scored completion.
    #[allow(clippy::too_many_arguments)]
    async fn assess(&self, req: &RouteRequest, decision: &RoutingDecision, m: &ModelConfig, attempt: &mut Attempt, resp: ProviderResponse, attempts: &[Attempt], deadline: Instant) -> Completion {
        let mut completion = Completion {
            text: resp.text,
            tool_calls: resp.tool_calls,
            model: m.name.clone(),
            provider_model: m.model.clone(),
            usage: attempt.usage.clone(),
            confidence: Confidence::default(),
            decision: decision.clone(),
            attempts: attempts.to_vec(),
            abstained: false,
            clarification: false,
            cache: None,
        };
        if completion.tool_calls.is_empty() && matches!(attempt.outcome, Outcome::Ok | Outcome::LowConfidence) {
            let hedged = attempt.outcome == Outcome::LowConfidence || self.is_hedged(&completion.text);
            completion.confidence = if self.cfg.routing.confidence.enabled {
                self.confidence(req, &decision.signals, &completion.text, hedged, deadline).await
            } else {
                self.heuristic_confidence(&decision.signals, &completion.text, hedged)
            };
            if self.cfg.routing.confidence.enabled && attempt.outcome == Outcome::Ok && completion.confidence.score < self.cfg.routing.confidence.threshold {
                attempt.outcome = Outcome::LowConfidence;
                attempt.detail = format!("confidence {:.2} < {:.2}", completion.confidence.score, self.cfg.routing.confidence.threshold);
            }
        } else if !completion.tool_calls.is_empty() {
            completion.confidence = Confidence { score: 0.9, grounding: 1.0, ..Default::default() };
        }
        completion
    }

    /// (name, description) of the planner's operations.
    pub fn planner_ops(&self) -> Vec<(String, String)> {
        self.planner.as_ref().map(|(_, c)| c.ops.iter().map(|o| (o.name.clone(), o.description.clone())).collect()).unwrap_or_default()
    }

    /// Vocabulary known to the route memory (entity names that count as slots).
    fn vocab(&self) -> Vec<String> {
        self.routes.vocab_list()
    }

    /// Record a shadow comparison: what the decision engine said vs what the slow path did.
    fn shadow(&self, req: &RouteRequest, kind: &str, engine_label: &str, engine_conf: f64, actual: &str, engine_ms: u128) {
        let Some(st) = &self.store else { return };
        let redact = self.cfg.store.as_ref().is_some_and(|s| s.redact_questions);
        let q = req.query();
        let spans = self.routes.spans(q);
        st.record(crate::store::Trace {
            id: uuid::Uuid::new_v4().simple().to_string(),
            ts_ms: crate::store::now_ms(),
            surface: format!("shadow:{kind}"),
            scope: self.route_scope(req) as i64,
            question: if redact { None } else { Some(crate::providers::truncate(q, 2000)) },
            shape: crate::learn::abstract_question(q, &spans),
            served_by: "shadow".into(),
            tool: Some(engine_label.to_string()),
            tool_args: Some(actual.to_string()),
            confidence: engine_conf,
            latency_ms: engine_ms as i64,
            outcome: if engine_label == actual { "agree".into() } else { "disagree".into() },
            ..Default::default()
        });
    }

    /// Record a teacher label for the native model: what the generative planner or the agent
    /// decided for this question's shape. Only the shape is stored, never the text.
    fn label(&self, req: &RouteRequest, task: &str, label: &str) {
        let Some(st) = &self.store else { return };
        let q = req.query();
        st.record(crate::store::Trace {
            id: uuid::Uuid::new_v4().simple().to_string(),
            ts_ms: crate::store::now_ms(),
            surface: format!("label:{task}"),
            scope: self.route_scope(req) as i64,
            shape: crate::learn::abstract_question(q, &self.routes.spans(q)),
            served_by: "label".into(),
            tool: Some(label.to_string()),
            outcome: "label".into(),
            ..Default::default()
        });
    }

    /// Reload the native models from the store (after `pankhllm mine --apply`).
    pub fn reload_native(&self) -> (bool, bool) {
        let load = |k: &str| newest_native(self.store.as_ref(), &self.cfg, k);
        let (p, t) = (load("native:plan"), load("native:tool"));
        let (hp, ht) = (p.is_some(), t.is_some());
        if let Some(m) = load("native:skill") {
            *self.native_skill.write().unwrap_or_else(|e| e.into_inner()) = Some(m);
        }
        if let Some(m) = p {
            *self.native_plan.write().unwrap_or_else(|e| e.into_inner()) = Some(m);
        }
        if let Some(m) = t {
            *self.native_tool.write().unwrap_or_else(|e| e.into_inner()) = Some(m);
        }
        (hp, ht)
    }

    /// One plain generation from a configured model (used by `pankhllm augment`).
    pub async fn generate(&self, model: &str, system: &str, user: &str, max_tokens: u32) -> Result<String> {
        let m = self.cfg.model(model).ok_or_else(|| anyhow!("unknown model {model}"))?;
        let provider = self.providers.get(&m.provider).ok_or_else(|| anyhow!("provider {} not initialised", m.provider))?;
        let preq = ProviderRequest {
            system_stable: None,
            system: Some(system.to_string()),
            messages: vec![Message::text("user", user.to_string())],
            max_tokens,
            temperature: Some(1.0),
            tools: vec![],
            tool_choice: None,
            effort: None,
            response_format: None,
        };
        let r = tokio::time::timeout(self.model_timeout(m).max(Duration::from_secs(120)), provider.complete(m, &preq)).await.map_err(|_| anyhow!("generation timed out"))??;
        Ok(r.text)
    }

    /// Operation name, description and parameter summary for prompts.
    pub fn planner_catalog_text(&self) -> Vec<(String, String, String)> {
        self.planner
            .as_ref()
            .map(|(_, c)| {
                c.ops
                    .iter()
                    .map(|o| {
                        let params = o
                            .params
                            .iter()
                            .map(|(k, p)| {
                                let mut d = format!("{k} ({})", p.kind);
                                if p.kind == "enum" {
                                    d.push_str(&format!(" one of {}", p.values.join("/")));
                                }
                                // A plain description beats a regex: generators copy regexes literally.
                                if !p.description.is_empty() {
                                    d.push_str(&format!(": {}", p.description));
                                } else if let Some(pat) = &p.pattern {
                                    d.push_str(&format!(" matching the regular expression {pat}"));
                                }
                                d
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        (o.name.clone(), o.description.clone(), params)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Teacher-only labelling: ask the generative planner for its decision on a question,
    /// validate the plan against the catalog, record the label. Nothing is executed and no
    /// agent runs. Returns the label recorded, if any.
    pub async fn teacher_label(&self, question: &str, cache_key: Option<&str>) -> Result<Option<String>> {
        self.teacher_label_opt(question, cache_key, true).await
    }

    /// As `teacher_label`; with `record` false the label is returned but not stored.
    pub async fn teacher_label_opt(&self, question: &str, cache_key: Option<&str>, record: bool) -> Result<Option<String>> {
        let (pc, cat) = self.planner.as_ref().ok_or_else(|| anyhow!("no planner configured"))?;
        let m = self.cfg.model(&pc.model).ok_or_else(|| anyhow!("planner model missing"))?;
        let provider = self.providers.get(&m.provider).ok_or_else(|| anyhow!("provider {} not initialised", m.provider))?;
        let preq = ProviderRequest {
            system_stable: Some(cat.instructions()),
            system: None,
            messages: vec![Message::text("user", question.to_string())],
            max_tokens: 400,
            temperature: None,
            tools: vec![],
            tool_choice: None,
            effort: None,
            response_format: Some(serde_json::json!({"type": "json_schema", "json_schema": {"name": "plan", "schema": cat.schema(), "strict": true}})),
        };
        let timeout = Duration::from_millis(pc.timeout_ms.max(5_000));
        let resp = tokio::time::timeout(timeout, provider.complete(m, &preq)).await.map_err(|_| anyhow!("teacher timed out"))??;
        let Some((op, params)) = crate::planner::parse_plan(&resp.text) else { return Ok(None) };
        // An invalid plan is not evidence either way.
        if op != "UNSUPPORTED" && cat.validate(&op, &params).is_err() {
            return Ok(None);
        }
        let label = op;
        if record {
            let req = RouteRequest { messages: vec![Message::text("user", question.to_string())], cache_key: cache_key.map(str::to_string), ..Default::default() };
            self.label(&req, "plan", &label);
        }
        Ok(Some(label))
    }

    /// Record a label supplied by the caller (templates, curated sets, human review).
    pub fn direct_label(&self, task: &str, question: &str, label: &str) {
        let req = RouteRequest { messages: vec![Message::text("user", question.to_string())], ..Default::default() };
        self.label(&req, task, label);
    }

    /// Offline decision for evaluation: what pankhllm's own model would do with a question.
    /// Returns (decision, probability, micros) where decision is an op name, "UNSUPPORTED",
    /// or "FALLBACK" (the request would go to the generative planner).
    pub async fn native_decision(&self, question: &str) -> (String, f64, u128) {
        let Some((_, cat)) = self.planner.as_ref() else { return ("FALLBACK".into(), 0.0, 0) };
        let started = Instant::now();
        let (res, raw) = self.engine_plan(question, cat).await;
        let us = started.elapsed().as_micros();
        let p = raw.as_ref().map(|r| r.1).unwrap_or(0.0);
        match res {
            Ok(Some((op, _))) => (op, p, us),
            Ok(None) => ("UNSUPPORTED".into(), p, us),
            Err(_) => ("FALLBACK".into(), p, us),
        }
    }

    /// The decision models in use, as a portable file.
    pub fn export_models(&self) -> crate::native::ModelsFile {
        let mut models = std::collections::BTreeMap::new();
        for (k, l) in [("plan", &self.native_plan), ("tool", &self.native_tool), ("skill", &self.native_skill)] {
            if let Some(m) = l.read().unwrap_or_else(|e| e.into_inner()).clone() {
                models.insert(k.to_string(), m);
            }
        }
        crate::native::ModelsFile { format: crate::native::ModelsFile::FORMAT.into(), exported_at_ms: crate::store::now_ms() as u64, models }
    }

    pub fn seed_skill_examples(&self) -> Vec<crate::native::Example> {
        skill_seed(&self.prompts)
    }

    pub fn seed_plan_examples(&self) -> Vec<crate::native::Example> {
        self.planner.as_ref().map(|(pc, cat)| seed_examples(pc, cat)).unwrap_or_default()
    }

    pub fn native_summary(&self) -> serde_json::Value {
        let s = |m: &Option<crate::native::NativeModel>| m.as_ref().map(|m| serde_json::json!({"labels": m.labels, "trained_on": m.trained_on, "holdout_accuracy": m.holdout_accuracy, "recommended_threshold": m.recommended_threshold}));
        serde_json::json!({"plan": s(&self.native_plan.read().unwrap_or_else(|e| e.into_inner())), "tool": s(&self.native_tool.read().unwrap_or_else(|e| e.into_inner())), "skill": s(&self.native_skill.read().unwrap_or_else(|e| e.into_inner()))})
    }

    /// Pick one of `criteria` (order preserved; reject option last) for question `q`, with
    /// pankhllm's own model or the external adapter. Returns (label, probability, micros)
    /// and the threshold to use for a structurally-constrained choice.
    async fn choose(&self, task: &str, q: &str, instructions: &str, criteria: Vec<(String, String)>) -> Option<(String, f64, u128, f64)> {
        let d = self.cfg.decisions.as_ref()?;
        let started = Instant::now();
        match d.engine {
            crate::config::DecisionEngineKind::Native => {
                let guard = if task == "tool" { self.native_tool.read() } else { self.native_plan.read() };
                let guard = guard.unwrap_or_else(|e| e.into_inner());
                let m = guard.as_ref()?;
                let names: Vec<String> = criteria.iter().map(|(k, _)| k.clone()).collect();
                let shape = crate::learn::abstract_question(q, &self.routes.spans(q));
                let (l, p) = m.choose(&shape, &names)?;
                // Out-of-domain guard: an operation chosen for mostly unfamiliar wording is not
                // acted on (the threshold becomes unreachable); UNSUPPORTED is unaffected.
                let familiar = m.known_ratio(&shape) >= d.min_known_words;
                let th = if familiar || l == "UNSUPPORTED" || l == "NO_TOOL" { d.consensus_confidence.max(m.recommended_threshold.unwrap_or(0.0)) } else { f64::INFINITY };
                Some((l, p, started.elapsed().as_micros(), th))
            }
            crate::config::DecisionEngineKind::External => {
                use crate::decide::{Answer, Question};
                let eng = self.engine.as_ref()?;
                // Wire question names: "operation", "tool", or the parameter name for enums.
                let key = match task {
                    "plan" => "operation",
                    "tool" => "tool",
                    other => other.strip_prefix("enum:").unwrap_or(other),
                }
                .to_string();
                let v = eng.decide(q, &[(key.clone(), Question::Choice { instructions: instructions.into(), criteria })]).await.ok()?;
                match v.answers.get(&key)? {
                    Answer::Choice { label, confidence } => Some((label.clone(), *confidence, v.latency_ms * 1000, d.consensus_confidence)),
                    _ => None,
                }
            }
        }
    }

    fn decisions_for(&self, planner: bool) -> bool {
        self.cfg.decisions.as_ref().is_some_and(|d| if planner { d.planner } else { d.tools })
    }

    /// Operations whose parameters the question fills exactly (every code, date and number
    /// used). Returns (op name, params with enums resolved or placeholder, unresolved enums).
    fn structural_fits(&self, q: &str, cat: &crate::planner::Catalog) -> Vec<(String, serde_json::Map<String, serde_json::Value>, Vec<String>)> {
        let vocab = self.vocab();
        let all_slots: Vec<crate::slots::Slot> = cat.ops.iter().flat_map(|o| cat.slots(o)).collect();
        let evidence = crate::slots::enum_evidence(q, &all_slots);
        let mut out = Vec::new();
        for op in &cat.ops {
            let slots = cat.slots(op);
            match crate::slots::fill(q, &slots, &vocab, &serde_json::Map::new()) {
                Ok(params) => {
                    if crate::slots::uses_every_typed_value_with(q, &params, &vocab, &evidence) {
                        out.push((op.name.clone(), params, vec![]));
                    }
                }
                Err(crate::slots::FillError::NeedsChoice(names)) => {
                    // Check the rest of the structure with a placeholder for each unnamed enum.
                    let mut tmp = serde_json::Map::new();
                    for n in &names {
                        if let Some(v) = slots.iter().find(|s| &s.name == n).and_then(|s| s.values.first()) {
                            tmp.insert(n.clone(), serde_json::json!(v));
                        }
                    }
                    if let Ok(params) = crate::slots::fill(q, &slots, &vocab, &tmp) {
                        let mut check = params.clone();
                        for n in &names {
                            check.remove(n);
                        }
                        if crate::slots::uses_every_typed_value_with(q, &check, &vocab, &evidence) {
                            out.push((op.name.clone(), params, names));
                        }
                    }
                }
                Err(_) => {}
            }
        }
        // Evidence rule: prefer operations that explain the most recognised values. An
        // operation that fits but ignores "bottom" and "districts" loses to one that uses them.
        if out.len() > 1 {
            let used = |params: &serde_json::Map<String, serde_json::Value>, unnamed: &[String]| -> usize {
                let vals: Vec<String> = params.iter().filter(|(k, _)| !unnamed.contains(k)).filter_map(|(_, v)| v.as_str().map(|s| s.to_lowercase())).collect();
                evidence.iter().filter(|(_, _, vs)| vs.iter().any(|v| vals.contains(v))).count()
            };
            let best = out.iter().map(|(_, p, u)| used(p, u)).max().unwrap_or(0);
            out.retain(|(_, p, u)| used(p, u) == best);
        }
        out
    }

    /// Decide which catalog operation answers the question, choosing only among operations
    /// the question structurally fits, and fill parameters without a generative model.
    /// Ok(Some) = a plan; Ok(None) = confident that nothing fits (skip the LLM planner);
    /// Err = no confident verdict, use the LLM planner. The second value is the raw verdict
    /// (label, probability, micros) for shadowing.
    async fn engine_plan(&self, q: &str, cat: &crate::planner::Catalog) -> (Result<Option<(String, serde_json::Value)>, String>, Option<(String, f64, u128)>) {
        if !self.decisions_for(true) {
            return (Err("decisions off".into()), None);
        }
        let d = self.cfg.decisions.as_ref().expect("checked");
        let fits = self.structural_fits(q, cat);
        // Catalog order, with the reject option last: choice models are position-biased.
        let mut criteria: Vec<(String, String)> = cat
            .ops
            .iter()
            .filter(|o| fits.is_empty() || fits.iter().any(|(n, _, _)| *n == o.name))
            .map(|o| (o.name.clone(), o.description.clone()))
            .collect();
        criteria.push(("UNSUPPORTED".to_string(), "None of these operations answers it exactly, or it asks why, for an explanation, a judgement or a prediction.".to_string()));
        let Some((label, confidence, micros, threshold)) = self.choose("plan", q, "Which operation answers this request exactly?", criteria).await else {
            return (Err("no decision model available".into()), None);
        };
        let raw = Some((label.clone(), confidence, micros));
        if label == "UNSUPPORTED" {
            // "Everything else" is open-ended; a small model cannot be trusted to rule out an
            // operation the question structurally fits. It may skip the planner only when
            // nothing fits; otherwise the teacher decides. A wrong call is then slow, not wrong.
            return if fits.is_empty() && confidence >= d.min_confidence {
                (Ok(None), raw)
            } else {
                (Err(format!("unsupported at {confidence:.2} with {} structural fits", fits.len())), raw)
            };
        }
        // Disagreement veto: if the model, free to choose among every operation, prefers one
        // the question does not fit, the question carries meaning the slot filler could not
        // read ("since last month" for a comparison). The teacher decides instead.
        if matches!(d.engine, crate::config::DecisionEngineKind::Native) && !fits.is_empty() {
            let all: Vec<(String, String)> = cat.ops.iter().map(|o| (o.name.clone(), String::new())).chain(std::iter::once(("UNSUPPORTED".to_string(), String::new()))).collect();
            if let Some((free, fp, _, _)) = self.choose("plan", q, "", all).await {
                if free != label && free != "UNSUPPORTED" && !fits.iter().any(|(n, _, _)| *n == free) {
                    return (Err(format!("model prefers {free} ({fp:.2}), which the question does not fit")), raw);
                }
            }
        }
        // Without a structural fit the pick is never trusted: the parameters would not fill.
        let Some((_, params, unnamed)) = fits.into_iter().find(|(n, _, _)| *n == label) else {
            return (Err(format!("{label} does not fit the question's structure")), raw);
        };
        if confidence < threshold {
            return (Err(format!("{label} at {confidence:.2} below {threshold:.2}")), raw);
        }
        if unnamed.is_empty() {
            return (Ok(Some((label, serde_json::Value::Object(params)))), raw);
        }
        // Enum parameters the question does not name: decide each among its declared values.
        let Some(op) = cat.op(&label) else { return (Err("unknown op".into()), raw) };
        let slots = cat.slots(op);
        let mut resolved = serde_json::Map::new();
        for n in &unnamed {
            let sl = slots.iter().find(|s| &s.name == n).expect("slot exists");
            let crit: Vec<(String, String)> = sl.values.iter().map(|v| (v.clone(), v.replace('_', " "))).collect();
            if let Some((v, p, _, th)) = self.choose(&format!("enum:{n}"), q, &format!("Which {n} does the request ask about?"), crit).await {
                if p >= th {
                    resolved.insert(n.clone(), serde_json::json!(v));
                }
            }
        }
        if resolved.len() < unnamed.len() {
            return (Err("could not resolve every unnamed parameter".into()), raw);
        }
        match crate::slots::fill(q, &slots, &self.vocab(), &resolved) {
            Ok(params) => (Ok(Some((label, serde_json::Value::Object(params)))), raw),
            Err(e) => (Err(format!("slots: {e:?}")), raw),
        }
    }

    /// Answer an agent's tool-selection turn without a generative model when the chosen
    /// tool's arguments are flat typed values the question fills unambiguously.
    async fn engine_tool_call(&self, req: &RouteRequest, decision: &RoutingDecision) -> Option<Completion> {
        if !self.decisions_for(false) || decision.signals.phase != Phase::ToolSelect || req.model.is_some() {
            return None;
        }
        let d = self.cfg.decisions.as_ref()?;
        let q = req.query();
        let vocab = self.vocab();
        // Tools whose arguments the question fills exactly: the structural constraint.
        let fitting: Vec<(&ToolDef, serde_json::Map<String, serde_json::Value>)> = req
            .tools
            .iter()
            .filter(|t| d.allow_tools.is_empty() || d.allow_tools.contains(&t.name))
            .filter_map(|t| {
                let slots = crate::slots::slots_from_schema(&t.parameters).ok()?;
                let args = crate::slots::fill(q, &slots, &vocab, &serde_json::Map::new()).ok()?;
                crate::slots::uses_every_typed_value(q, &args, &vocab).then_some((t, args))
            })
            .collect();
        if fitting.is_empty() {
            return None;
        }
        let mut criteria: Vec<(String, String)> = fitting.iter().map(|(t, _)| (t.name.clone(), if t.description.is_empty() { t.name.replace('_', " ") } else { t.description.clone() })).collect();
        criteria.push(("NO_TOOL".into(), "Answer directly, or a tool not listed here, or the request needs explanation or judgement.".into()));
        let (label, confidence, micros, threshold) = self.choose("tool", q, "Which tool should handle this request?", criteria).await?;
        if d.mode == crate::config::DecisionMode::Shadow {
            Self::lock(&self.pending_shadow).insert(hash64(q), (label, confidence, micros, Instant::now()));
            return None;
        }
        if label == "NO_TOOL" || confidence < threshold {
            return None;
        }
        let (tool, args) = fitting.into_iter().find(|(t, _)| t.name == label)?;
        let call = ToolCall { id: format!("call_{}", uuid::Uuid::new_v4().simple()), name: tool.name.clone(), arguments: serde_json::Value::Object(args).to_string() };
        let served_by = format!("decision-tool:{}", tool.name);
        Some(Completion {
            text: String::new(),
            tool_calls: vec![call],
            model: served_by.clone(),
            provider_model: String::new(),
            usage: Usage::default(),
            confidence: Confidence { score: confidence, grounding: 1.0, cited: false, hedged: false, ambiguous: false, verifier: None },
            decision: decision.clone(),
            attempts: vec![Attempt { model: served_by, outcome: Outcome::Ok, detail: format!("decided at {confidence:.2} in {micros} us; arguments filled from the question"), latency_ms: micros / 1000, usage: Usage::default() }],
            abstained: false,
            clarification: false,
            cache: None,
        })
    }

    /// The planner lane. Returns None whenever it cannot prove an answer, so the
    /// request carries on through the normal lanes untouched.
    async fn planner_lane(&self, req: &RouteRequest, decision: &RoutingDecision, deadline: Instant) -> Option<Completion> {
        let (pc, cat) = self.planner.as_ref()?;
        if !req.tools.is_empty() || req.model.is_some() || decision.clarify || decision.abstain {
            return None;
        }
        if !pc.multi_turn && req.messages.iter().filter(|m| m.role != "system").count() != 1 {
            return None;
        }
        if !pc.intents.contains(&decision.signals.intent) {
            return None;
        }
        let q = req.query();
        let scope = self.route_scope(req);
        let learning = pc.learn && self.cfg.routing.learned_routes.enabled;
        let started = Instant::now();

        // 1. A learned plan for this question shape: no model call.
        let mut planned: Option<(String, serde_json::Value, Usage, Option<u64>)> = None;
        if learning {
            if let Some(fill) = self.routes.fill(scope, q, None) {
                if let (Some(op), Ok(params)) = (fill.tool.strip_prefix("plan:"), serde_json::from_str::<serde_json::Value>(&fill.arguments)) {
                    planned = Some((op.to_string(), params, Usage::default(), Some(fill.key)));
                }
            }
        }
        // 2. A System-1 decision engine: operation + parameters with no generative model.
        let mut engine_verdict: Option<(String, f64, u128)> = None;
        let mut from_engine = false;
        if planned.is_none() && self.decisions_for(true) {
            let (res, raw) = self.engine_plan(q, cat).await;
            engine_verdict = raw;
            let shadow = self.cfg.decisions.as_ref().is_some_and(|d| d.mode == crate::config::DecisionMode::Shadow);
            if !shadow {
                match res {
                    Ok(Some((op, params))) => {
                        planned = Some((op, params, Usage::default(), None));
                        from_engine = true;
                    }
                    Ok(None) => {
                        info!("decision engine: no operation fits; skipping the planner call");
                        return None;
                    }
                    Err(why) => info!(%why, "decision engine not confident; using the planner model"),
                }
            }
        }
        // 3. Otherwise one small structured-output call.
        if planned.is_none() {
            let m = self.cfg.model(&pc.model)?;
            if self.in_cooldown(&m.name) {
                return None;
            }
            let provider = self.providers.get(&m.provider)?;
            let timeout = Duration::from_millis(pc.timeout_ms).min(deadline.saturating_duration_since(Instant::now()));
            let preq = ProviderRequest {
                system_stable: Some(cat.instructions()),
                system: None,
                messages: vec![Message::text("user", q.to_string())],
                max_tokens: 400,
                temperature: None,
                tools: vec![],
                tool_choice: None,
                effort: None,
                response_format: Some(serde_json::json!({"type": "json_schema", "json_schema": {"name": "plan", "schema": cat.schema(), "strict": true}})),
            };
            let resp = match tokio::time::timeout(timeout, provider.complete(m, &preq)).await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => {
                    warn!(model = %m.name, error = %e, "planner call failed; falling through");
                    self.record(&m.name, Outcome::Error, started.elapsed().as_millis(), 0.0, false);
                    return None;
                }
                Err(_) => {
                    warn!(model = %m.name, "planner call timed out; falling through");
                    self.record(&m.name, Outcome::Error, started.elapsed().as_millis(), 0.0, true);
                    return None;
                }
            };
            let usage = Usage { input_tokens: resp.input_tokens, output_tokens: resp.output_tokens, cost_usd: (resp.input_tokens as f64 * m.cost.input + resp.output_tokens as f64 * m.cost.output) / 1_000_000.0 };
            self.record(&m.name, Outcome::Ok, started.elapsed().as_millis(), usage.cost_usd, false);
            let (op, params) = crate::planner::parse_plan(&resp.text)?;
            if let Some((label, conf, ms)) = &engine_verdict {
                self.shadow(req, "plan", label, *conf, &op, *ms);
            }
            if op == "UNSUPPORTED" {
                self.label(req, "plan", "UNSUPPORTED");
                info!("planner: unsupported; falling through to the agent");
                return None;
            }
            planned = Some((op, params, usage, None));
        }
        let (op, params, usage, route_key) = planned?;
        let quarantine_after = self.cfg.heal.quarantine_after_failures;
        let params = match cat.validate(&op, &params) {
            Ok(p) => p,
            Err(why) => {
                warn!(%op, %why, "plan rejected by the catalog; falling through");
                if let Some(k) = route_key {
                    self.routes.feedback(k, false, quarantine_after);
                }
                return None;
            }
        };
        let trace_id = uuid::Uuid::new_v4().simple().to_string();
        let ex = match crate::planner::execute(&self.http, pc, &op, &params, q, req.user.as_deref(), req.cache_key.as_deref(), &trace_id, deadline).await {
            Ok(e) => e,
            Err(why) => {
                warn!(%op, %why, "executor failed; falling through");
                if let Some(k) = route_key {
                    if let Some(shape) = self.routes.feedback(k, false, quarantine_after) {
                        if let Some(st) = &self.store {
                            st.heal("quarantine", &shape, &format!("plan route: {why}"));
                        }
                    }
                }
                return None;
            }
        };
        let text = crate::planner::render(cat.op(&op)?, &params, &ex, 50)?;
        if route_key.is_none() && !from_engine {
            // The generative planner's plan executed cleanly: a teacher label for our own model.
            self.label(req, "plan", &op);
        }
        if let Some(k) = route_key {
            self.routes.feedback(k, true, quarantine_after);
        } else if learning {
            let args = crate::planner::canonical(&params);
            let lr = &self.cfg.routing.learned_routes;
            if crate::learn::learnable_arguments(&args, &lr.deny_arg_names, lr.max_arg_len).is_ok() {
                if let Some(shape) = self.routes.learn(scope, q, &format!("plan:{op}"), &args, None, false, None) {
                    info!(%shape, %op, "learned plan route");
                }
            }
        }
        let served_by = if route_key.is_some() {
            format!("plan-route:{op}")
        } else if from_engine {
            format!("decision:{op}")
        } else {
            format!("planner:{op}")
        };
        let decided = match (&engine_verdict, from_engine) {
            (Some((_, p, us)), true) => format!("decided at {p:.2} in {us} us; "),
            _ => String::new(),
        };
        let attempt = Attempt { model: served_by.clone(), outcome: Outcome::Ok, detail: format!("{decided}plan {op} {} executed", crate::planner::canonical(&params)), latency_ms: started.elapsed().as_millis(), usage: usage.clone() };
        Some(Completion {
            text,
            tool_calls: vec![],
            model: served_by,
            provider_model: String::new(),
            usage,
            confidence: Confidence { score: 0.95, grounding: 1.0, cited: false, hedged: false, ambiguous: false, verifier: None },
            decision: decision.clone(),
            attempts: vec![attempt],
            abstained: false,
            clarification: false,
            cache: None,
        })
    }

    /// Record one routed request in the trace store (non-blocking).
    pub fn trace(&self, surface: &str, req: &RouteRequest, c: Option<&Completion>, err: Option<&str>, latency_ms: u128) {
        let Some(st) = &self.store else { return };
        let redact = self.cfg.store.as_ref().is_some_and(|s| s.redact_questions);
        let q = req.query();
        let spans = self.routes.spans(q);
        let last = req.messages.last();
        let (result_for, result_error) = match last.and_then(|m| m.tool_call_id.as_ref()) {
            Some(id) => (Some(id.clone()), Some(Self::result_is_error(&last.map(|m| m.content.clone()).unwrap_or_default()))),
            None => (None, None),
        };
        let mut t = crate::store::Trace {
            id: uuid::Uuid::new_v4().simple().to_string(),
            ts_ms: crate::store::now_ms(),
            surface: surface.to_string(),
            scope: self.route_scope(req) as i64,
            user_hash: req.user.as_ref().map(|u| format!("{:016x}", hash64(u))),
            question: if redact { None } else { Some(crate::providers::truncate(q, 2000)) },
            shape: crate::learn::abstract_question(q, &spans),
            result_for,
            result_error,
            latency_ms: latency_ms as i64,
            error: err.map(|e| crate::providers::truncate(e, 500)),
            outcome: if err.is_some() { "error".into() } else { "ok".into() },
            ..Default::default()
        };
        if let Some(c) = c {
            t.served_by = match &c.cache { Some(ci) => format!("cache:{}", ci.kind), None => c.model.clone() };
            t.intent = c.decision.signals.intent.to_string();
            t.tier = c.decision.target_tier.to_string();
            t.phase = format!("{:?}", c.decision.signals.phase).to_lowercase();
            t.input_tokens = c.usage.input_tokens as i64;
            t.output_tokens = c.usage.output_tokens as i64;
            t.cost_usd = c.usage.cost_usd;
            t.confidence = c.confidence.score;
            if c.clarification { t.outcome = "clarification".into(); }
            if c.abstained { t.outcome = "abstained".into(); }
            if let Some(tc) = c.tool_calls.first() {
                t.tool = Some(tc.name.clone());
                t.call_id = Some(tc.id.clone());
                let lr = &self.cfg.routing.learned_routes;
                // Only flat, learnable arguments are stored; code-like arguments never reach disk.
                if crate::learn::learnable_arguments(&tc.arguments, &lr.deny_arg_names, lr.max_arg_len).is_ok() {
                    t.tool_args = Some(tc.arguments.clone());
                }
            }
        }
        st.record(t);
    }

    /// Route, call, and cascade. Returns the first acceptable answer. Every call is traced.
    /// A request that names its skills explicitly is a label for skill routing.
    fn label_skill_choice(&self, req: &RouteRequest) {
        let Some(names) = req.prompt.as_deref().filter(|p| *p != "auto") else { return };
        if let Some(first) = names.split(',').map(str::trim).find(|n| !n.is_empty() && !self.prompts.is_always(n)) {
            self.label(req, "skill", first);
        }
    }

    pub async fn complete(&self, req: &RouteRequest) -> Result<Completion> {
        self.label_skill_choice(req);
        let started = Instant::now();
        let r = self.complete_traced(req).await;
        let ms = started.elapsed().as_millis();
        match &r {
            Ok(c) => self.trace("complete", req, Some(c), None, ms),
            Err(e) => self.trace("complete", req, None, Some(&e.to_string()), ms),
        }
        r
    }

    async fn complete_traced(&self, req: &RouteRequest) -> Result<Completion> {
        let req = &self.apply_prompt_meta(req).map_err(|e| anyhow!(e))?;
        let decision = self.decide(req);
        let attempts: Vec<Attempt> = Vec::new();

        if decision.abstain {
            return Ok(Completion {
                text: "I don't have enough relevant information to answer that.".into(),
                tool_calls: Vec::new(),
                model: String::new(),
                provider_model: String::new(),
                usage: Usage::default(),
                confidence: Confidence { score: 0.0, ..Default::default() },
                decision,
                attempts,
                abstained: true,
                clarification: false,
                cache: None,
            });
        }
        if decision.candidates.is_empty() {
            return Err(anyhow!("no model satisfies the constraints: {:?}", decision.reasons));
        }
        if decision.clarify {
            return self.clarify(req, decision).await;
        }
        self.promote_candidate(req);
        if let Some(c) = self.rule_tool_call(req, &decision).or_else(|| self.rule_after_result(req, &decision)).or_else(|| self.speculative_render(req, &decision)) {
            return Ok(c);
        }
        let deadline = self.deadline(req);
        if let Some(c) = self.learned_call(req, &decision, deadline).await {
            return Ok(c);
        }
        if let Some(c) = self.engine_tool_call(req, &decision).await {
            return Ok(c);
        }
        let (hit, vector) = self.cache_lookup(req, &decision, deadline).await;
        if let Some(c) = hit {
            return Ok(c);
        }
        // Identical concurrent requests: one leader calls the model, followers wait and re-read the cache.
        let sf_key = if self.cache_allowed(req) && req.cache_mode == CacheMode::Use { Some(self.cache_keys(req).0) } else { None };
        let mut token = None;
        if let Some(k) = sf_key {
            match self.single_flight_wait(k, deadline).await {
                Some(t) => token = Some(t),
                None => {
                    if let (Some(c), _) = self.cache_lookup(req, &decision, deadline).await {
                        let mut c = c;
                        c.attempts.push(Attempt { model: c.model.clone(), outcome: Outcome::Ok, detail: "coalesced with an identical in-flight request".into(), latency_ms: 0, usage: Usage::default() });
                        return Ok(c);
                    }
                    // Leader's answer was not cacheable (or timed out): answer ourselves.
                }
            }
        }
        let result = match self.planner_lane(req, &decision, deadline).await {
            Some(c) => Ok(c),
            None => self.complete_uncached(req, decision, deadline).await,
        };
        if let Ok(r) = &result {
            let tool_select = !req.tools.is_empty() && !req.messages.last().is_some_and(|m| m.tool_call_id.is_some());
            if tool_select && r.tool_calls.is_empty() && self.cfg.model(&r.model).is_some() && self.decisions_for(false) && r.confidence.score >= 0.7 {
                self.label(req, "tool", "NO_TOOL");
            }
            if let Some((label, conf, ms, _)) = Self::lock(&self.pending_shadow).remove(&hash64(req.query())) {
                let actual = r.tool_calls.first().map(|t| t.name.clone()).unwrap_or_else(|| "NO_TOOL".into());
                self.shadow(req, "tool", &label, conf, &actual, ms);
            }
        }
        if let Ok(r) = &result {
            self.charge_user(req.user.as_deref(), r.usage.cost_usd);
        }
        if let (Some(k), Some(t)) = (sf_key, &token) {
            if let Ok(r) = &result {
                self.cache_store(req, r, vector);
            }
            self.single_flight_done(k, t);
            return result;
        }
        let result = result?;
        self.cache_store(req, &result, vector);
        Ok(result)
    }

    async fn complete_uncached(&self, req: &RouteRequest, decision: RoutingDecision, deadline: Instant) -> Result<Completion> {
        let mut attempts: Vec<Attempt> = Vec::new();
        let max_attempts = if self.cfg.routing.cascade.enabled { 1 + self.cfg.routing.cascade.max_steps } else { 1 };
        let intent = decision.signals.intent;
        let hedge_after = self.cfg.routing.hedge.after_ms.map(Duration::from_millis);
        let mut best_hedged: Option<Completion> = None;
        let cands = &decision.candidates;
        let mut idx = 0usize;

        // Each round yields one or two (candidate index, attempt, response) results,
        // in completion order. A round is a single call, or a hedged race of two.
        while idx < cands.len() && attempts.len() < max_attempts {
            let mut round: Vec<(usize, Attempt, Option<ProviderResponse>)> = Vec::new();
            let m0 = self.cfg.model(&cands[idx].name).expect("candidate came from config");
            let race_with = match hedge_after {
                Some(_) if idx + 1 < cands.len() && attempts.len() + 1 < max_attempts && attempts.is_empty() => Some(idx + 1),
                _ => None,
            };
            match race_with {
                None => {
                    let (a, r) = self.run_attempt(req, m0, intent, deadline, attempts.len() + 1).await?;
                    round.push((idx, a, r));
                    idx += 1;
                }
                Some(j) => {
                    let m1 = self.cfg.model(&cands[j].name).expect("candidate came from config");
                    let first = self.run_attempt(req, m0, intent, deadline, attempts.len() + 1);
                    tokio::pin!(first);
                    let wait = tokio::time::sleep(hedge_after.unwrap());
                    tokio::pin!(wait);
                    let mut first_done: Option<(Attempt, Option<ProviderResponse>)> = None;
                    tokio::select! {
                        r = &mut first => { first_done = Some(r?); }
                        _ = &mut wait => {}
                    }
                    match first_done {
                        Some((a, r)) => {
                            round.push((idx, a, r));
                            idx += 1; // the hedge never started; j is tried in the next round if needed
                        }
                        None => {
                            info!(primary = %m0.name, hedge = %m1.name, after_ms = hedge_after.unwrap().as_millis(), "hedging");
                            let second = self.run_attempt(req, m1, intent, deadline, attempts.len() + 2);
                            tokio::pin!(second);
                            // Whichever finishes first is examined first; if it is not usable, wait for the other.
                            let (winner, pending): ((usize, Attempt, Option<ProviderResponse>), Option<usize>) = tokio::select! {
                                r = &mut first => { let (a, r) = r?; ((idx, a, r), Some(j)) }
                                r = &mut second => { let (a, r) = r?; ((j, a, r), Some(idx)) }
                            };
                            let usable = winner.2.as_ref().is_some_and(|_| winner.1.outcome == Outcome::Ok);
                            round.push(winner);
                            if !usable {
                                let (a, r) = if pending == Some(j) { second.await? } else { first.await? };
                                round.push((pending.unwrap(), a, r));
                            } else {
                                // Loser is dropped: its in-flight request is cancelled.
                                let loser = self.cfg.model(&cands[pending.unwrap()].name).expect("candidate came from config");
                                round.push((pending.unwrap(), Attempt { model: loser.name.clone(), outcome: Outcome::Error, detail: "cancelled: hedge lost the race".into(), latency_ms: 0, usage: Usage::default() }, None));
                            }
                            idx = j + 1;
                        }
                    }
                }
            }

            // Records without a response (errors, cancelled hedges) are filed first so
            // they survive an early return on a usable answer.
            round.sort_by_key(|(_, _, r)| r.is_some());
            let mut stop_cascade = false;
            for (ci, mut attempt, resp) in round {
                let m = self.cfg.model(&cands[ci].name).expect("candidate came from config");
                let Some(resp) = resp else {
                    if self.cfg.routing.cascade.stop_on_invalid_request && attempt.detail.starts_with("invalid request") {
                        stop_cascade = true;
                    }
                    attempts.push(attempt);
                    continue;
                };
                attempts.push(attempt.clone());
                let mut completion = self.assess(req, &decision, m, &mut attempt, resp, &attempts, deadline).await;
                if let Some(last) = attempts.last_mut() {
                    *last = attempt.clone();
                }
                completion.attempts = attempts.clone();
                match attempt.outcome {
                    Outcome::Ok => return Ok(completion),
                    Outcome::LowConfidence => {
                        // Only worth escalating if a stronger model is still ahead.
                        let stronger_ahead = cands[idx.min(cands.len())..].iter().any(|c| c.tier > cands[ci].tier);
                        if !self.cfg.routing.cascade.enabled || !stronger_ahead {
                            return Ok(completion);
                        }
                        if best_hedged.as_ref().is_none_or(|b| completion.confidence.score > b.confidence.score) {
                            best_hedged = Some(completion);
                        }
                    }
                    _ => {}
                }
            }

            if stop_cascade {
                warn!("upstream rejected the request as invalid; not retrying on other candidates");
                break;
            }
        }

        // Every attempt hedged: the honest answer is the hedge itself.
        if let Some(mut c) = best_hedged {
            c.attempts = attempts;
            return Ok(c);
        }
        Err(anyhow!("all candidates failed: {:?}", attempts))
    }

    /// Open a stream and wait for its first event. `Err(attempt)` means this
    /// candidate is not usable and the record explains why.
    async fn open_stream(&self, req: &RouteRequest, m: &ModelConfig, intent: Intent, deadline: Instant) -> Result<Result<OpenedStream, Attempt>> {
        let provider = self.providers.get(&m.provider).ok_or_else(|| anyhow!("provider {} not initialised", m.provider))?.clone();
        let rec = |outcome: Outcome, detail: String, latency_ms: u128| Attempt { model: m.name.clone(), outcome, detail, latency_ms, usage: Usage::default() };
        let Some(timeout) = self.attempt_timeout(m, deadline) else {
            return Ok(Err(rec(Outcome::Error, "request latency budget exhausted".into(), 0)));
        };
        let Some(permit) = self.admit(&m.name) else {
            return Ok(Err(rec(Outcome::Error, "at max_concurrency, skipped".into(), 0)));
        };
        let preq = self.build_provider_request(req, m, intent)?;
        let started = Instant::now();
        let mut events = match tokio::time::timeout(timeout, provider.stream(m, &preq)).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                warn!(model = %m.name, error = %e, "stream open failed");
                if matches!(e, ProviderError::Retryable(_) | ProviderError::Transport(_)) {
                    self.mark_unhealthy(&m.name);
                }
                self.record(&m.name, Outcome::Error, started.elapsed().as_millis(), 0.0, false);
                return Ok(Err(rec(Outcome::Error, e.to_string(), started.elapsed().as_millis())));
            }
            Err(_) => {
                if self.timeout_is_models_own(m, timeout) {
                    self.mark_unhealthy(&m.name);
                }
                self.record(&m.name, Outcome::Error, timeout.as_millis(), 0.0, true);
                return Ok(Err(rec(Outcome::Error, "timed out opening stream".into(), timeout.as_millis())));
            }
        };
        let remaining = deadline.saturating_duration_since(Instant::now()).min(timeout);
        let first = match tokio::time::timeout(remaining, events.next()).await {
            Ok(Some(Ok(ev))) => ev,
            Ok(Some(Err(e))) => {
                warn!(model = %m.name, error = %e, "stream failed before first token");
                if matches!(e, ProviderError::Retryable(_) | ProviderError::Transport(_)) {
                    self.mark_unhealthy(&m.name);
                }
                self.record(&m.name, Outcome::Error, started.elapsed().as_millis(), 0.0, false);
                return Ok(Err(rec(Outcome::Error, e.to_string(), started.elapsed().as_millis())));
            }
            Ok(None) => {
                self.record(&m.name, Outcome::Empty, started.elapsed().as_millis(), 0.0, false);
                return Ok(Err(rec(Outcome::Empty, "stream ended with no events".into(), started.elapsed().as_millis())));
            }
            Err(_) => {
                if self.timeout_is_models_own(m, timeout) {
                    self.mark_unhealthy(&m.name);
                }
                self.record(&m.name, Outcome::Error, started.elapsed().as_millis(), 0.0, true);
                return Ok(Err(rec(Outcome::Error, "timed out waiting for first token".into(), started.elapsed().as_millis())));
            }
        };
        if let StreamEvent::Done { stop_reason: StopReason::Refusal, .. } = first {
            self.record(&m.name, Outcome::Refusal, started.elapsed().as_millis(), 0.0, false);
            return Ok(Err(rec(Outcome::Refusal, "refused before any output".into(), started.elapsed().as_millis())));
        }
        Ok(Ok(OpenedStream { model: m.clone(), events, first, permit, started, timeout }))
    }

    /// Commit to an opened stream: prepend its first event, hold the permit and the
    /// hard timeout for its whole life, and record stats when it finishes.
    fn commit_stream(self: &Arc<Self>, opened: OpenedStream, decision: RoutingDecision, mut attempts: Vec<Attempt>, req: &RouteRequest) -> StreamingCompletion {
        let OpenedStream { model: m, events, first, permit, started, timeout } = opened;
        let ttft = started.elapsed().as_millis();
        attempts.push(Attempt { model: m.name.clone(), outcome: Outcome::Ok, detail: format!("streaming, ttft_ms={ttft}"), latency_ms: ttft, usage: Usage::default() });
        let me = Arc::clone(self);
        let name = m.name.clone();
        let cost = m.cost;
        let hard_deadline = tokio::time::sleep(timeout);
        let tail = futures::stream::unfold(
            (events, Box::pin(hard_deadline), Some(permit), false),
            move |(mut events, mut sleep, permit, done)| {
                let me = me.clone();
                let name = name.clone();
                async move {
                    if done {
                        return None;
                    }
                    tokio::select! {
                        _ = &mut sleep => {
                            me.record(&name, Outcome::Error, 0, 0.0, true);
                            Some((Err(ProviderError::Retryable("stream exceeded model timeout".into())), (events, sleep, None, true)))
                        }
                        next = events.next() => match next {
                            None => None,
                            Some(Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason })) => {
                                let c = (input_tokens as f64 * cost.input + output_tokens as f64 * cost.output) / 1_000_000.0;
                                me.record(&name, Outcome::Ok, started.elapsed().as_millis(), c, false);
                                Some((Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason }), (events, sleep, None, true)))
                            }
                            Some(other) => Some((other, (events, sleep, permit, false))),
                        }
                    }
                }
            },
        );
        let events: EventStream = futures::stream::once(async move { Ok(first) }).chain(tail).boxed();
        let events = self.strip_hidden_tool_stream(events, req);
        StreamingCompletion { model: m.name.clone(), provider_model: m.model.clone(), decision, attempts, cache: None, events }
    }

    /// Hide the speculative-template tool from a streamed tool-call sequence: its
    /// start/delta events are swallowed, its arguments accumulated, and on Done the
    /// template is stored for the visible calls of the same turn.
    fn strip_hidden_tool_stream(self: &Arc<Self>, events: EventStream, req: &RouteRequest) -> EventStream {
        let spec = &self.cfg.routing.speculative_answer;
        if !spec.enabled && !self.cfg.routing.learned_routes.enabled {
            return events;
        }
        let me = Arc::clone(self);
        let req_c = Arc::new(req.clone());
        let hidden_name = spec.tool_name.clone();
        // (hidden index, hidden args, visible calls seen, index remap)
        let state = (events, None::<usize>, String::new(), Vec::<ToolCall>::new(), HashMap::<usize, usize>::new());
        futures::stream::unfold(state, move |(mut events, mut hidden_idx, mut hidden_args, mut visible, mut remap)| {
            let me = me.clone();
            let req_c = Arc::clone(&req_c);
            let hidden_name = hidden_name.clone();
            async move {
                loop {
                    let ev = events.next().await?;
                    match ev {
                        Ok(StreamEvent::ToolCallStart { index, id: _, name }) if name == hidden_name => {
                            hidden_idx = Some(index);
                            continue;
                        }
                        Ok(StreamEvent::ToolCallDelta { index, arguments }) if hidden_idx == Some(index) => {
                            hidden_args.push_str(&arguments);
                            continue;
                        }
                        Ok(StreamEvent::ToolCallStart { index, id, name }) => {
                            let new_index = remap.len();
                            remap.insert(index, new_index);
                            visible.push(ToolCall { id: id.clone(), name: name.clone(), arguments: String::new() });
                            return Some((Ok(StreamEvent::ToolCallStart { index: new_index, id, name }), (events, hidden_idx, hidden_args, visible, remap)));
                        }
                        Ok(StreamEvent::ToolCallDelta { index, arguments }) => {
                            let new_index = *remap.get(&index).unwrap_or(&index);
                            if let Some(c) = visible.get_mut(new_index) {
                                c.arguments.push_str(&arguments);
                            }
                            return Some((Ok(StreamEvent::ToolCallDelta { index: new_index, arguments }), (events, hidden_idx, hidden_args, visible, remap)));
                        }
                        Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason }) => {
                            if hidden_idx.is_some() {
                                let mut all = visible.clone();
                                all.push(ToolCall { id: "hidden".into(), name: hidden_name.clone(), arguments: hidden_args.clone() });
                                let _ = me.absorb_template_calls(all);
                            }
                            if !visible.is_empty() {
                                let me2 = me.clone();
                                let req2 = Arc::clone(&req_c);
                                let vis = visible.clone();
                                tokio::spawn(async move {
                                    let deadline = Instant::now() + Duration::from_secs(5);
                                    me2.learn_from(&req2, &vis, deadline).await;
                                });
                            }
                            // If the only call was the hidden one, the turn is a plain answer.
                            let stop = if stop_reason == StopReason::ToolUse && visible.is_empty() { StopReason::EndTurn } else { stop_reason };
                            return Some((Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason: stop }), (events, hidden_idx, hidden_args, visible, remap)));
                        }
                        other => return Some((other, (events, hidden_idx, hidden_args, visible, remap))),
                    }
                }
            }
        })
        .boxed()
    }

    /// Streaming variant. Fallback (and hedging) happens until the first token
    /// arrives; after that the stream is committed to one model. Hedge detection
    /// needs the full text, so it is not applied here.
    pub async fn stream(self: &Arc<Self>, req: &RouteRequest) -> Result<StreamingCompletion> {
        self.label_skill_choice(req);
        let started = Instant::now();
        let r = self.stream_traced(req).await;
        let ms = started.elapsed().as_millis();
        match &r {
            Ok(sc) => {
                let c = Completion {
                    text: String::new(), tool_calls: vec![], model: sc.model.clone(), provider_model: sc.provider_model.clone(),
                    usage: Usage::default(), confidence: Confidence::default(), decision: sc.decision.clone(), attempts: sc.attempts.clone(),
                    abstained: false, clarification: false, cache: sc.cache.clone(),
                };
                self.trace("stream", req, Some(&c), None, ms);
            }
            Err(e) => self.trace("stream", req, None, Some(&e.to_string()), ms),
        }
        r
    }

    async fn stream_traced(self: &Arc<Self>, req: &RouteRequest) -> Result<StreamingCompletion> {
        let req = &self.apply_prompt_meta(req).map_err(|e| anyhow!(e))?;
        let decision = self.decide(req);
        if decision.abstain {
            return Err(anyhow!("policy abstained: weak retrieval"));
        }
        if decision.candidates.is_empty() {
            return Err(anyhow!("no model satisfies the constraints: {:?}", decision.reasons));
        }
        let deadline = self.deadline(req);
        self.promote_candidate(req);
        // Deterministic rule and cache hits are replayed as a short event stream.
        let served = match self.rule_tool_call(req, &decision).or_else(|| self.rule_after_result(req, &decision)).or_else(|| self.speculative_render(req, &decision)) {
            Some(c) => Some(c),
            None => match self.learned_call(req, &decision, deadline).await {
                Some(c) => Some(c),
                None => match self.engine_tool_call(req, &decision).await {
                    Some(c) => Some(c),
                    None => self.cache_lookup(req, &decision, deadline).await.0,
                },
            },
        };
        let served = match served {
            Some(c) => Some(c),
            None => {
                let p = self.planner_lane(req, &decision, deadline).await;
                if let Some(c) = &p {
                    self.charge_user(req.user.as_deref(), c.usage.cost_usd);
                    self.cache_store(req, c, None);
                }
                p
            }
        };
        if let Some(c) = served {
            let mut evs: Vec<Result<StreamEvent, ProviderError>> = Vec::new();
            if !c.text.is_empty() {
                evs.push(Ok(StreamEvent::Delta(c.text.clone())));
            }
            for (i, t) in c.tool_calls.iter().enumerate() {
                evs.push(Ok(StreamEvent::ToolCallStart { index: i, id: t.id.clone(), name: t.name.clone() }));
                evs.push(Ok(StreamEvent::ToolCallDelta { index: i, arguments: t.arguments.clone() }));
            }
            let stop = if c.tool_calls.is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };
            evs.push(Ok(StreamEvent::Done { input_tokens: c.usage.input_tokens, output_tokens: c.usage.output_tokens, stop_reason: stop }));
            return Ok(StreamingCompletion { model: c.model, provider_model: c.provider_model, decision: c.decision, attempts: c.attempts, cache: c.cache, events: futures::stream::iter(evs).boxed() });
        }
        let mut sc = self.stream_uncached(req, decision, deadline).await?;
        // Write-through: accumulate the committed stream and store it on Done.
        if self.cache_allowed(req) {
            let me = Arc::clone(self);
            let req_c = req.clone();
            let model = sc.model.clone();
            let provider_model = sc.provider_model.clone();
            let sig = sc.decision.signals.clone();
            let tail = futures::stream::unfold((sc.events, String::new(), Vec::<ToolCall>::new()), move |(mut events, mut text, mut calls)| {
                let me = me.clone();
                let req_c = req_c.clone();
                let model = model.clone();
                let provider_model = provider_model.clone();
                let sig = sig.clone();
                async move {
                    let ev = events.next().await?;
                    match &ev {
                        Ok(StreamEvent::Delta(t)) => text.push_str(t),
                        Ok(StreamEvent::ToolCallStart { index, id, name }) => {
                            while calls.len() <= *index {
                                calls.push(ToolCall { id: String::new(), name: String::new(), arguments: String::new() });
                            }
                            calls[*index].id = id.clone();
                            calls[*index].name = name.clone();
                        }
                        Ok(StreamEvent::ToolCallDelta { index, arguments }) => {
                            if let Some(c) = calls.get_mut(*index) {
                                c.arguments.push_str(arguments);
                            }
                        }
                        Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason }) => {
                            if *stop_reason != StopReason::Refusal {
                                let hedged = me.is_hedged(&text);
                                let confidence = me.heuristic_confidence(&sig, &text, hedged);
                                let c = Completion {
                                    text: text.clone(), tool_calls: calls.clone(), model: model.clone(), provider_model: provider_model.clone(),
                                    usage: Usage { input_tokens: *input_tokens, output_tokens: *output_tokens, cost_usd: 0.0 },
                                    confidence, decision: RoutingDecision { target_tier: Tier::Fast, candidates: vec![], rejected: vec![], signals: sig.clone(), reasons: vec![], abstain: false, clarify: false },
                                    attempts: vec![], abstained: false, clarification: false, cache: None,
                                };
                                me.cache_store(&req_c, &c, None);
                            }
                        }
                        Err(_) => {}
                    }
                    Some((ev, (events, text, calls)))
                }
            });
            sc.events = tail.boxed();
        }
        Ok(sc)
    }

    async fn stream_uncached(self: &Arc<Self>, req: &RouteRequest, decision: RoutingDecision, deadline: Instant) -> Result<StreamingCompletion> {
        let mut attempts: Vec<Attempt> = Vec::new();
        let max_attempts = if self.cfg.routing.cascade.enabled { 1 + self.cfg.routing.cascade.max_steps } else { 1 };
        let intent = decision.signals.intent;
        let hedge_after = self.cfg.routing.hedge.after_ms.map(Duration::from_millis);
        let cands = decision.candidates.clone();
        let mut idx = 0usize;

        while idx < cands.len() && attempts.len() < max_attempts {
            let m0 = self.cfg.model(&cands[idx].name).expect("candidate came from config").clone();
            let race = hedge_after.is_some() && idx + 1 < cands.len() && attempts.is_empty() && attempts.len() + 1 < max_attempts;
            if !race {
                match self.open_stream(req, &m0, intent, deadline).await? {
                    Ok(opened) => return Ok(self.commit_stream(opened, decision, attempts, req)),
                    Err(a) => attempts.push(a),
                }
                idx += 1;
                continue;
            }

            // Hedged: wait `after_ms` for the primary's first token, then race the next candidate.
            let j = idx + 1;
            let m1 = self.cfg.model(&cands[j].name).expect("candidate came from config").clone();
            let first = self.open_stream(req, &m0, intent, deadline);
            tokio::pin!(first);
            let wait = tokio::time::sleep(hedge_after.unwrap());
            tokio::pin!(wait);
            let mut early: Option<Result<OpenedStream, Attempt>> = None;
            tokio::select! {
                r = &mut first => { early = Some(r?); }
                _ = &mut wait => {}
            }
            if let Some(r) = early {
                match r {
                    Ok(opened) => return Ok(self.commit_stream(opened, decision, attempts, req)),
                    Err(a) => attempts.push(a),
                }
                idx += 1;
                continue;
            }
            info!(primary = %m0.name, hedge = %m1.name, after_ms = hedge_after.unwrap().as_millis(), "hedging stream");
            let second = self.open_stream(req, &m1, intent, deadline);
            tokio::pin!(second);
            let (winner, loser_pending) = tokio::select! {
                r = &mut first => (r?, j),
                r = &mut second => (r?, idx),
            };
            match winner {
                Ok(opened) => {
                    let loser = self.cfg.model(&cands[loser_pending].name).expect("candidate came from config");
                    attempts.push(Attempt { model: loser.name.clone(), outcome: Outcome::Error, detail: "cancelled: hedge lost the race".into(), latency_ms: 0, usage: Usage::default() });
                    return Ok(self.commit_stream(opened, decision, attempts, req));
                }
                Err(a) => {
                    attempts.push(a);
                    // The other one is still running: wait for it.
                    let other = if loser_pending == j { second.await? } else { first.await? };
                    match other {
                        Ok(opened) => return Ok(self.commit_stream(opened, decision, attempts, req)),
                        Err(a) => attempts.push(a),
                    }
                }
            }
            idx = j + 1;
        }
        Err(anyhow!("all candidates failed: {:?}", attempts))
    }
}

/// Fill `{{path}}`, `{{path|table}}` and `{{_raw}}` placeholders from a JSON value.
/// The newest trained model for `key` ("native:plan"...): the trace store (retrained by the
/// miner) or `decisions.models_file` (shipped), whichever was trained later.
fn newest_native(store: Option<&crate::store::Store>, cfg: &Config, key: &str) -> Option<crate::native::NativeModel> {
    let from_store = store.and_then(|s| s.get_kv(key)).and_then(|b| serde_json::from_slice::<crate::native::NativeModel>(&b).ok());
    let from_file = cfg.decisions.as_ref().and_then(|d| d.models_file.as_deref()).and_then(|p| match crate::native::ModelsFile::read(p) {
        Ok(f) => f.models.get(key.trim_start_matches("native:")).cloned(),
        Err(e) => {
            tracing::warn!("decisions.models_file {p}: {e}");
            None
        }
    });
    match (from_store, from_file) {
        (Some(a), Some(b)) => Some(if b.trained_at_ms > a.trained_at_ms { b } else { a }),
        (a, b) => a.or(b),
    }
}

/// Skill-routing label for "no skill applies".
pub const NO_SKILL: &str = "NO_SKILL";

/// Skill front-matter `examples` as training rows for the skill-routing model.
pub fn skill_seed(prompts: &PromptRegistry) -> Vec<crate::native::Example> {
    prompts
        .skill_examples()
        .into_iter()
        .map(|(skill, q)| crate::native::Example { shape: crate::learn::abstract_question(&q, &crate::learn::typed_spans(&q)), label: skill })
        .collect()
}

/// Catalog `examples` and `unsupported_examples` as training rows for the native model.
pub fn seed_examples(pc: &crate::config::PlannerConfig, cat: &crate::planner::Catalog) -> Vec<crate::native::Example> {
    let shape = |q: &str| crate::learn::abstract_question(q, &crate::learn::typed_spans(q));
    let mut out: Vec<crate::native::Example> = cat.ops.iter().flat_map(|o| o.examples.iter().map(move |q| crate::native::Example { shape: shape(q), label: o.name.clone() })).collect();
    out.extend(pc.unsupported_examples.iter().map(|q| crate::native::Example { shape: shape(q), label: "UNSUPPORTED".into() }));
    out
}

/// One completed agent turn as found in application logs, for offline training.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ObservedTurn {
    pub question: String,
    pub tool: String,
    /// JSON string of the arguments the model produced.
    pub arguments: String,
    #[serde(default)]
    pub tool_names: Vec<String>,
    #[serde(default)]
    pub tool_result: Option<String>,
    #[serde(default)]
    pub answer: Option<String>,
    #[serde(default)]
    pub answer_template: Option<String>,
    #[serde(default)]
    pub needs_analysis: bool,
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Turn a final answer into a template by replacing values that also appear in
/// the tool result with `{{path}}` placeholders. Longest values first so "1,842"
/// is replaced before "42". Returns None when nothing in the answer came from
/// the result (the answer was not data-driven).
pub fn derive_answer_template(answer: &str, result_json: &str) -> Option<String> {
    derive_answer_template_with_args(answer, result_json, "{}")
}

/// As above, also replacing tool-argument values (product, region, period...) with
/// `{{args.<name>}}` so entities from the question never stay literal.
pub fn derive_answer_template_with_args(answer: &str, result_json: &str, args_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(result_json).ok()?;
    let mut pairs: Vec<(String, String)> = Vec::new();
    fn walk(v: &serde_json::Value, path: &str, out: &mut Vec<(String, String)>) {
        match v {
            serde_json::Value::Object(o) => {
                for (k, x) in o {
                    walk(x, &if path.is_empty() { k.clone() } else { format!("{path}.{k}") }, out);
                }
            }
            serde_json::Value::Array(a) => {
                for (i, x) in a.iter().enumerate().take(50) {
                    walk(x, &format!("{path}.{i}"), out);
                }
            }
            serde_json::Value::String(s) if s.len() >= 2 => out.push((path.to_string(), s.clone())),
            serde_json::Value::Number(n) => out.push((path.to_string(), n.to_string())),
            _ => {}
        }
    }
    walk(&v, "", &mut pairs);
    // Argument values come first (longest first): "East" from the question must not stay literal.
    let mut arg_pairs: Vec<(String, String)> = Vec::new();
    if let Ok(serde_json::Value::Object(a)) = serde_json::from_str::<serde_json::Value>(args_json) {
        for (k, x) in a {
            match x {
                serde_json::Value::String(sv) if sv.len() >= 2 => arg_pairs.push((format!("args.{k}"), sv)),
                serde_json::Value::Number(n) => arg_pairs.push((format!("args.{k}"), n.to_string())),
                _ => {}
            }
        }
    }
    // Candidate renderings for each value: raw, thousands-grouped, percent (0-1 fractions).
    let mut forms: Vec<(String, String)> = Vec::new(); // (placeholder, literal)
    for (path, val) in arg_pairs.into_iter().chain(pairs) {
        let is_num = val.parse::<f64>().is_ok();
        if val.chars().all(|c| c.is_ascii_digit()) && val.len() < 2 {
            continue; // single digits are too ambiguous
        }
        forms.push((format!("{{{{{path}}}}}"), val.clone()));
        if is_num {
            let n: f64 = val.parse().unwrap_or(0.0);
            forms.push((format!("{{{{{path}|N0}}}}"), group_thousands(&format!("{:.0}", (n).round()))));
            if val.contains('.') {
                forms.push((format!("{{{{{path}|N1}}}}"), group_thousands(&format!("{:.1}", (n * 10.0).round() / 10.0))));
            }
            if n > 0.0 && n <= 1.0 && val.contains('.') {
                forms.push((format!("{{{{{path}|P0}}}}"), format!("{:.0}%", (n * 100.0).round())));
                forms.push((format!("{{{{{path}|P1}}}}"), format!("{:.1}%", (n * 1000.0).round() / 10.0)));
            }
        }
    }
    // Longest literals first so "1,842" beats "42" and "74%" beats "74".
    forms.sort_by_key(|(_, lit)| std::cmp::Reverse(lit.len()));
    let mut out = answer.to_string();
    let mut replaced = 0;
    for (ph, lit) in forms {
        if lit.is_empty() || !out.contains(&lit) {
            continue;
        }
        // Whole-token matches only, so "42" inside "1,842" or a placeholder is left alone.
        let mut rebuilt = String::new();
        let mut last = 0;
        let bytes = out.as_bytes();
        let mut idx = 0;
        while let Some(pos) = out[idx..].find(&lit) {
            let s = idx + pos;
            let e = s + lit.len();
            let left_ok = s == 0 || !(bytes[s - 1].is_ascii_alphanumeric() || bytes[s - 1] == b'{' || bytes[s - 1] == b'.' || bytes[s - 1] == b',');
            let right_ok = e >= out.len() || !(bytes[e].is_ascii_alphanumeric() || bytes[e] == b'}' || (bytes[e] == b'.' && e + 1 < out.len() && bytes[e + 1].is_ascii_digit()) || (bytes[e] == b',' && e + 1 < out.len() && bytes[e + 1].is_ascii_digit()));
            if left_ok && right_ok {
                rebuilt.push_str(&out[last..s]);
                rebuilt.push_str(&ph);
                last = e;
                replaced += 1;
            }
            idx = e;
        }
        rebuilt.push_str(&out[last..]);
        out = rebuilt;
    }
    (replaced > 0).then_some(out)
}

/// Dotted-path lookup into a JSON value (`rows.0.rep`).
pub fn json_path<'a>(v: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cur = v;
    for part in path.split('.') {
        cur = match cur {
            serde_json::Value::Object(o) => o.get(part)?,
            serde_json::Value::Array(a) => a.get(part.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

pub fn render_template(template: &str, output: &serde_json::Value, raw: &str, vars: &HashMap<String, String>, max_rows: usize) -> String {
    let lookup = json_path;
    fn scalar(v: &serde_json::Value) -> String {
        match v {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Null => String::new(),
            other => other.to_string(),
        }
    }
    fn table(v: &serde_json::Value, max_rows: usize) -> String {
        let rows = match v.as_array() { Some(r) => r, None => return scalar(v) };
        let mut cols: Vec<String> = Vec::new();
        for r in rows.iter().take(max_rows) {
            if let Some(o) = r.as_object() {
                for k in o.keys() {
                    if !cols.contains(k) {
                        cols.push(k.clone());
                    }
                }
            }
        }
        if cols.is_empty() {
            return rows.iter().take(max_rows).map(scalar).collect::<Vec<_>>().join(", ");
        }
        let mut out = format!("| {} |\n| {} |\n", cols.join(" | "), cols.iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
        for r in rows.iter().take(max_rows) {
            let cells: Vec<String> = cols.iter().map(|c| r.get(c).map(scalar).unwrap_or_default().replace('|', "\\|")).collect();
            out.push_str(&format!("| {} |\n", cells.join(" | ")));
        }
        if rows.len() > max_rows {
            out.push_str(&format!("\n({} of {} rows shown)", max_rows, rows.len()));
        }
        out
    }
    let mut out = String::new();
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let Some(end) = rest[start..].find("}}") else { out.push_str(&rest[start..]); return out; };
        let inner = rest[start + 2..start + end].trim();
        let (path, filter) = match inner.split_once('|') { Some((p, f)) => (p.trim(), Some(f.trim())), None => (inner, None) };
        let value = if path == "_raw" {
            raw.to_string()
        } else if let Some(v) = vars.get(path) {
            v.clone()
        } else {
            match (lookup(output, path), filter) {
                (Some(v), Some("table")) => table(v, max_rows),
                (Some(v), Some("count")) => v.as_array().map(|a| a.len().to_string()).unwrap_or_else(|| scalar(v)),
                (Some(v), Some(f)) if f.len() == 2 && (f.starts_with('N') || f.starts_with('P')) && f[1..].chars().all(|c| c.is_ascii_digit()) => {
                    // .NET-style numeric formats: N0/N1/N2 thousands-grouped, P0/P1 percent of a 0-1 fraction.
                    let decimals: usize = f[1..].parse().unwrap_or(0);
                    match v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim_end_matches('%').parse().ok())) {
                        Some(n) => {
                            let (n, suffix) = if f.starts_with('P') { (n * 100.0, "%") } else { (n, "") };
                            // Round half away from zero like .NET, not banker's rounding.
                            let scale = 10f64.powi(decimals as i32);
                            let n = (n * scale).round() / scale;
                            format!("{}{}", group_thousands(&format!("{n:.decimals$}")), suffix)
                        }
                        None => scalar(v),
                    }
                }
                (Some(v), _) => scalar(v),
                (None, _) => String::new(),
            }
        };
        out.push_str(&value);
        rest = &rest[start + end + 2..];
    }
    out.push_str(rest);
    out
}

fn group_thousands(s: &str) -> String {
    let (neg, s) = match s.strip_prefix('-') { Some(r) => (true, r), None => (false, s) };
    let (int, frac) = match s.split_once('.') { Some((i, f)) => (i, Some(f)), None => (s, None) };
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    let mut r = if neg { format!("-{out}") } else { out };
    if let Some(f) = frac {
        r.push('.');
        r.push_str(f);
    }
    r
}

/// How many placeholders in `template` resolve: the path exists in the result
/// (an empty table or zero is a real value) or renders to non-empty text.
pub fn count_filled(template: &str, output: &serde_json::Value, raw: &str, max_rows: usize) -> usize {
    let mut n = 0;
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        let Some(end) = rest[start..].find("}}") else { break };
        let inner = rest[start + 2..start + end].trim();
        let path = inner.split('|').next().unwrap_or(inner).trim();
        let exists = path == "_raw" || json_path(output, path).is_some_and(|v| !v.is_null());
        let one = format!("{{{{{inner}}}}}");
        if exists || !render_template(&one, output, raw, &HashMap::new(), max_rows).trim().is_empty() {
            n += 1;
        }
        rest = &rest[start + end + 2..];
    }
    n
}

/// A stream that produced its first event; everything needed to commit to it.
pub struct OpenedStream {
    pub model: ModelConfig,
    pub events: EventStream,
    pub first: StreamEvent,
    pub permit: Option<tokio::sync::OwnedSemaphorePermit>,
    pub started: Instant,
    pub timeout: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: &str = r#"
providers:
  anthropic: { kind: anthropic, api_key_env: ANTHROPIC_API_KEY }
  local: { kind: openai, base_url: http://localhost:11434/v1 }
models:
  - { name: local-llama, provider: local, model: llama3.1:8b, tier: fast, context_window: 32000, tags: [private, local] }
  - { name: haiku, provider: anthropic, model: claude-haiku-4-5, tier: fast, context_window: 200000, cost: { input: 1, output: 5 } }
  - { name: sonnet, provider: anthropic, model: claude-sonnet-5, tier: balanced, context_window: 1000000, cost: { input: 2, output: 10 } }
  - { name: opus, provider: anthropic, model: claude-opus-5, tier: reasoning, context_window: 1000000, cost: { input: 5, output: 25 } }
routing:
  weak_retrieval: { threshold: 0.3, action: escalate }
"#;

    fn router() -> Router {
        Router::offline(Config::from_yaml(CFG).unwrap())
    }

    fn req(q: &str, scores: &[f64]) -> RouteRequest {
        RouteRequest {
            messages: vec![Message::text("user", q)],
            context: scores.iter().map(|s| Chunk { text: "some passage".into(), score: Some(*s), source: None }).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn lookup_with_good_retrieval_goes_cheap() {
        let d = router().decide(&req("What is the refund window?", &[0.9, 0.7]));
        assert_eq!(d.target_tier, Tier::Fast);
        // local-llama has zero cost, so it wins the fast tier.
        assert_eq!(d.candidates[0].name, "local-llama");
        assert_eq!(d.candidates[1].name, "haiku");
        assert_eq!(d.candidates[2].name, "sonnet");
    }

    #[test]
    fn weak_retrieval_escalates() {
        let d = router().decide(&req("What is the refund window?", &[0.1]));
        assert_eq!(d.target_tier, Tier::Balanced);
        assert_eq!(d.candidates[0].name, "sonnet");
    }

    #[test]
    fn reasoning_goes_to_top_tier() {
        let d = router().decide(&req("Why did the migration fail?", &[0.8]));
        assert_eq!(d.target_tier, Tier::Reasoning);
        assert_eq!(d.candidates[0].name, "opus");
    }

    #[test]
    fn private_tag_restricts_to_local() {
        let mut r = req("Why did the migration fail?", &[0.8]);
        r.tags = vec!["private".into()];
        let d = router().decide(&r);
        assert_eq!(d.candidates.len(), 1);
        assert_eq!(d.candidates[0].name, "local-llama");
        assert!(d.rejected.iter().all(|c| c.reason.contains("tag")));
    }

    #[test]
    fn huge_context_excludes_small_windows() {
        let mut r = req("What is the refund window?", &[0.9]);
        r.context[0].text = "x".repeat(900_000); // ~250k tokens
        let d = router().decide(&r);
        assert!(d.candidates.iter().all(|c| c.name == "sonnet" || c.name == "opus"));
    }

    #[test]
    fn budget_prunes_expensive_models() {
        let mut r = req("Why did the migration fail?", &[0.8]);
        r.max_cost_usd = Some(0.0001);
        let d = router().decide(&r);
        assert_eq!(d.candidates[0].name, "local-llama");
        assert!(d.rejected.iter().any(|c| c.name == "opus" && c.reason.contains("budget")));
    }

    #[test]
    fn forced_model_bypasses_policy() {
        let mut r = req("hello", &[]);
        r.model = Some("opus".into());
        let d = router().decide(&r);
        assert_eq!(d.candidates.len(), 1);
        assert_eq!(d.candidates[0].name, "opus");
    }

    #[test]
    fn config_rules_beat_heuristics() {
        let mut cfg = Config::from_yaml(CFG).unwrap();
        cfg.routing.rules = vec![
            crate::config::Rule { name: "kpi-lookup".into(), any: vec!["call activity".into(), "fte".into(), "trx".into(), "nrx".into()], pattern: None, intent: Some(Intent::Lookup), tier: Some(Tier::Fast), respond_with_tool: None },
            crate::config::Rule { name: "alerts".into(), any: vec!["alert".into()], pattern: None, intent: None, tier: Some(Tier::Reasoning), respond_with_tool: None },
        ];
        let r = Router::offline(cfg);
        // "why" would normally be reasoning; the KPI rule pins it to a cheap lookup.
        let d = r.decide(&req("Why is call activity down for the east region this month?", &[0.9]));
        assert_eq!(d.signals.matched_rule.as_deref(), Some("kpi-lookup"));
        assert_eq!(d.target_tier, Tier::Fast);
        assert_eq!(d.signals.intent_source, "rule");
        let d = r.decide(&req("Which alerts fired for territory 12?", &[0.9]));
        assert_eq!(d.target_tier, Tier::Reasoning);
        // request override still wins over rules
        let mut rr = req("What is the FTE count?", &[0.9]);
        rr.tier = Some(Tier::Balanced);
        assert_eq!(r.decide(&rr).target_tier, Tier::Balanced);
    }

    #[test]
    fn tool_requests_without_context_are_not_weak_retrieval() {
        let r = router();
        let mut rq = req("What is the refund window?", &[]);
        assert!(r.decide(&rq).signals.weak_retrieval);
        rq.tools = vec![ToolDef { name: "lookup_policy".into(), description: String::new(), parameters: serde_json::json!({"type": "object"}) }];
        let d = r.decide(&rq);
        assert!(!d.signals.weak_retrieval);
        assert_eq!(d.target_tier, Tier::Fast);
    }

    #[test]
    fn grounding_instruction_follows_intent() {
        let r = router();
        let m = r.cfg.model("haiku").unwrap();
        let rq = req("What is the refund window?", &[0.9]);
        let p = r.build_provider_request(&rq, m, Intent::Lookup).unwrap();
        assert!(p.system.as_deref().unwrap().contains("I don't know"));
        let p = r.build_provider_request(&rq, m, Intent::Reasoning).unwrap();
        assert!(p.system.as_deref().unwrap().contains("not supported by the context"));
        assert!(p.system.as_deref().unwrap().contains("<chunk id=\"1\""));
    }

    #[test]
    fn agent_loop_phases_pick_tiers() {
        let r = router();
        let tool = ToolDef { name: "get_kpi".into(), description: String::new(), parameters: serde_json::json!({"type": "object"}) };
        // Tool selection on a reasoning-sounding question still goes fast: picking a tool is easy.
        let mut rq = req("Why did NRx fall in T-112? Investigate.", &[]);
        rq.tools = vec![tool.clone()];
        let d = r.decide(&rq);
        assert_eq!(d.signals.phase, Phase::ToolSelect);
        assert_eq!(d.target_tier, Tier::Fast);
        // After the tool result arrives, synthesis goes to balanced.
        rq.messages.push(Message { role: "assistant".into(), content: String::new(), tool_calls: vec![ToolCall { id: "c1".into(), name: "get_kpi".into(), arguments: "{}".into() }], tool_call_id: None });
        rq.messages.push(Message { role: "user".into(), content: "{\"nrx\": -18}".into(), tool_calls: vec![], tool_call_id: Some("c1".into()) });
        let d = r.decide(&rq);
        assert_eq!(d.signals.phase, Phase::AfterToolResult);
        assert_eq!(d.target_tier, Tier::Balanced);
        // A rule still beats the phase.
        let mut cfg = Config::from_yaml(CFG).unwrap();
        cfg.routing.rules = vec![crate::config::Rule { name: "alerts".into(), any: vec!["investigate".into()], pattern: None, intent: None, tier: Some(Tier::Reasoning), respond_with_tool: None }];
        assert_eq!(Router::offline(cfg).decide(&rq).target_tier, Tier::Reasoning);
    }

    #[test]
    fn skill_front_matter_drives_routing() {
        let mut prompts = HashMap::new();
        prompts.insert("field-kpi".to_string(), crate::prompts::Prompt { body: "KPI skill".into(), meta: crate::prompts::PromptMeta { tier: Some(Tier::Fast), tags: vec!["private".into()], max_latency_ms: Some(5000), ..Default::default() } });
        let r = Router::offline(Config::from_yaml(CFG).unwrap()).with_prompts(crate::prompts::PromptRegistry::from_prompts(prompts));
        let mut rq = req("Why did NRx fall in T-112?", &[0.9]);
        rq.prompt = Some("field-kpi".into());
        let d = r.decide(&rq);
        assert_eq!(d.target_tier, Tier::Fast, "{:?}", d.reasons);
        assert_eq!(d.candidates.len(), 1);
        assert_eq!(d.candidates[0].name, "local-llama");
        // Request-level values override the skill's.
        rq.tier = Some(Tier::Reasoning);
        rq.tags = vec![];
        let d = r.decide(&rq);
        assert_eq!(d.target_tier, Tier::Reasoning);
        assert_eq!(d.candidates[0].name, "local-llama", "tags still merged from the skill");
    }

    #[test]
    fn template_renderer_paths_tables_and_captures() {
        let out = serde_json::json!({"calls": 312, "reach": 0.74, "rows": [{"rep": "A", "calls": 200}, {"rep": "B", "calls": 112}], "meta": {"period": "2026-08"}});
        let mut vars = HashMap::new();
        vars.insert("territory".to_string(), "T-112".to_string());
        let t = "{{territory}} in {{meta.period}}: {{calls}} calls, reach {{reach}}, {{rows|count}} reps, top {{rows.0.rep}}\n{{rows|table}}";
        let s = render_template(t, &out, "raw", &vars, 1);
        assert!(s.starts_with("T-112 in 2026-08: 312 calls, reach 0.74, 2 reps, top A\n| rep | calls |"), "{s}");
        assert!(s.contains("| A | 200 |") && !s.contains("| B |") && s.contains("(1 of 2 rows shown)"), "{s}");
        assert_eq!(render_template("x {{missing}} y {{_raw}}", &serde_json::Value::Null, "RAW", &HashMap::new(), 5), "x  y RAW");
        let o = serde_json::json!({"ref": "tbl_91", "calls": 1234567.891, "share": 0.4567, "pct": "12.5%"});
        assert_eq!(render_template("<x-table id=\"{{ref}}\"></x-table> {{calls|N0}} {{calls|N1}} {{share|P1}} {{pct|N0}}", &o, "", &HashMap::new(), 5), "<x-table id=\"tbl_91\"></x-table> 1,234,568 1,234,567.9 45.7% 13");
        assert_eq!(render_template("unterminated {{oops", &out, "", &HashMap::new(), 5), "unterminated {{oops");
    }

    #[test]
    fn answer_templates_are_derived_from_logged_answers() {
        let result = r#"{"territory":"T-112","calls":1842,"reach":0.71,"rows":[{"rep":"Ann","calls":900}]}"#;
        let t = derive_answer_template("T-112 logged 1,842 HCP calls; top rep Ann with 900.", result).unwrap();
        assert_eq!(t, "{{territory}} logged {{calls|N0}} HCP calls; top rep {{rows.0.rep}} with {{rows.0.calls}}.");
        assert!(derive_answer_template("Thanks for asking!", result).is_none());
        // Entities from the question (arguments) and percent renderings of fractions.
        let t = derive_answer_template_with_args("T-207 made 88 calls in 2026-09 (reach 61%). <x-table id=\"t2\"></x-table>",
            r#"{"calls":88,"reach":0.61,"ref":"t2"}"#, r#"{"template":"calls_by_territory_month","territory":"T-207","month":"2026-09"}"#).unwrap();
        assert_eq!(t, "{{args.territory}} made {{calls}} calls in {{args.month}} (reach {{reach|P0}}). <x-table id=\"{{ref}}\"></x-table>");
    }

    #[test]
    fn abstain_policy() {
        let mut cfg = Config::from_yaml(CFG).unwrap();
        cfg.routing.weak_retrieval.action = WeakRetrievalAction::Abstain;
        let d = Router::offline(cfg).decide(&req("What is the refund window?", &[0.05]));
        assert!(d.abstain);
    }
}
