//! YAML configuration: providers, models, and the routing policy.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::types::{Intent, Tier};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    /// OpenAI itself, and anything speaking its API: Ollama, vLLM, Groq, Together, LM Studio.
    Openai,
    /// Azure OpenAI. `base_url` is the resource endpoint; each model's `model` is the deployment name.
    Azure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>` (OpenAI, Entra ID tokens on Azure).
    Bearer,
    /// `api-key: <key>` (Azure OpenAI keys).
    ApiKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    /// Environment variable holding the API key. Optional for local servers.
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// TCP/TLS connect timeout. Keep it short: a provider that cannot be reached
    /// in a few seconds should be skipped, not waited on.
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,
    /// Azure only. Set for the classic `/openai/deployments/<name>/chat/completions?api-version=` path;
    /// leave unset to use the newer `/openai/v1/chat/completions` surface.
    #[serde(default)]
    pub api_version: Option<String>,
    /// Header style. Defaults: `bearer` for openai, `api_key` for azure.
    #[serde(default)]
    pub auth: Option<AuthStyle>,
    /// Upstream API for openai/azure providers: `chat` (chat completions, default) or
    /// `responses` (the Responses API, required by some newer deployments).
    #[serde(default = "default_upstream_api")]
    pub api: UpstreamApi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamApi {
    Chat,
    Responses,
}

fn default_upstream_api() -> UpstreamApi {
    UpstreamApi::Chat
}

fn default_timeout() -> u64 {
    120
}
fn default_connect_timeout() -> u64 {
    3
}

impl ProviderConfig {
    pub fn api_key(&self) -> Option<String> {
        self.api_key_env
            .as_ref()
            .and_then(|k| std::env::var(k).ok())
            .filter(|v| !v.is_empty())
    }
}

/// USD per one million tokens.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Chat,
    /// An embeddings endpoint (OpenAI-compatible `/embeddings`); never routed to.
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Name you route to, e.g. `haiku`, `local-llama`.
    pub name: String,
    #[serde(default = "default_model_kind")]
    pub kind: ModelKind,
    pub provider: String,
    /// The provider's model id, e.g. `claude-haiku-4-5`.
    pub model: String,
    pub tier: Tier,
    pub context_window: u32,
    #[serde(default = "default_max_output")]
    pub max_output: u32,
    #[serde(default)]
    pub cost: Cost,
    /// Free-form capabilities: `private`, `local`, `vision`, `eu`.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Anthropic only: send `thinking: {type: adaptive}`. Defaults to on except for Haiku.
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Reasoning depth. Anthropic: `output_config.effort` (low | medium | high | xhigh | max).
    /// OpenAI / Azure: `reasoning_effort` (minimal | low | medium | high); setting it also
    /// marks the model as a reasoning model so sampling params are not sent.
    #[serde(default)]
    pub effort: Option<String>,
    /// Anthropic only: pass `temperature` through. Off by default because the
    /// current 5-series and 4.7+ models reject sampling parameters.
    #[serde(default)]
    pub allow_sampling: bool,
    /// Anthropic only: opt into server-side refusal fallbacks (`fallbacks: "default"`).
    #[serde(default)]
    pub server_fallbacks: bool,
    /// Hard per-call timeout. A model that blows this is skipped and the next candidate runs.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// In-flight cap. When full, the router moves to the next candidate instead of queueing.
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    /// False keeps the model out of automatic routing (verifier / judge models,
    /// embeddings). It can still be forced by name.
    #[serde(default = "default_true")]
    pub routable: bool,
}

fn default_max_output() -> u32 {
    8192
}
fn default_model_kind() -> ModelKind {
    ModelKind::Chat
}

impl ModelConfig {
    /// Eligible for automatic routing: chat models with `routable: true`.
    pub fn is_routable(&self) -> bool {
        self.routable && self.kind == ModelKind::Chat
    }

    pub fn thinking_enabled(&self) -> bool {
        self.thinking
            .unwrap_or_else(|| !self.model.to_ascii_lowercase().contains("haiku"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeakRetrievalAction {
    /// Bump the tier: a stronger model is more likely to notice the context does not answer.
    Escalate,
    /// Do not call any model; return an abstain decision.
    Abstain,
    Ignore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeakRetrieval {
    /// Top chunk score below this counts as weak. Set to your retriever's scale.
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_weak_action")]
    pub action: WeakRetrievalAction,
}

fn default_threshold() -> f64 {
    0.3
}
fn default_weak_action() -> WeakRetrievalAction {
    WeakRetrievalAction::Escalate
}

impl Default for WeakRetrieval {
    fn default() -> Self {
        Self { threshold: default_threshold(), action: default_weak_action() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cascade {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// A 400/422 from an upstream is deterministic for that request: stop instead of
    /// paying for the same failure on the next candidate. Default true.
    #[serde(default = "default_true")]
    pub stop_on_invalid_request: bool,
    /// Extra attempts after the first.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// If the answer contains one of these, treat it as low confidence and escalate.
    #[serde(default = "default_hedges")]
    pub hedge_phrases: Vec<String>,
}

fn default_true() -> bool {
    true
}
fn default_max_steps() -> usize {
    2
}
fn default_hedges() -> Vec<String> {
    [
        "i don't know",
        "i do not know",
        "not enough information",
        "cannot determine",
        "can't determine",
        "does not contain",
        "doesn't contain",
        "no information",
        "not mentioned in the",
        "unable to answer",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl Default for Cascade {
    fn default() -> Self {
        Self { enabled: true, stop_on_invalid_request: true, max_steps: default_max_steps(), hedge_phrases: default_hedges() }
    }
}

/// Render the final answer from the tool's output with a template instead of a
/// model call. Placeholders: `{{field}}` / `{{a.b.0}}` paths into the JSON output,
/// `{{field|table}}` renders an array of objects as a markdown table (first
/// `max_rows` rows), `{{_raw}}` is the output text, and regex captures from the
/// question (`{{territory}}`) are available too.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AfterResult {
    pub template: String,
    #[serde(default = "default_max_rows")]
    pub max_rows: usize,
}

fn default_max_rows() -> usize {
    50
}

/// Answer a matching question with a tool call directly, skipping the model.
/// `arguments` is a JSON template; `{{name}}` is filled from the rule's regex
/// named captures. Applies only when the request offers a tool with that name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespondWithTool {
    pub name: String,
    #[serde(default = "default_schema_obj")]
    pub arguments: serde_json::Value,
    /// When the tool result comes back, answer from it deterministically.
    #[serde(default)]
    pub after_result: Option<AfterResult>,
}

fn default_schema_obj() -> serde_json::Value {
    serde_json::json!({})
}

/// A domain rule. Matches when the query contains any of `any` (case-insensitive)
/// or matches `pattern` (a regex, use `(?i)` for case-insensitive and named
/// captures for tool arguments). Rules run before the built-in heuristics;
/// first match wins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub name: String,
    #[serde(default)]
    pub any: Vec<String>,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub intent: Option<Intent>,
    #[serde(default)]
    pub tier: Option<Tier>,
    #[serde(default)]
    pub respond_with_tool: Option<RespondWithTool>,
}

/// Tiers for the two phases of an agent loop. Choosing a tool is easy; making
/// sense of tool results is where quality matters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLoop {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Request carries tools and the last message is from the user: the model picks a tool.
    #[serde(default = "default_tool_select_tier")]
    pub tool_select: Tier,
    /// The last message is a tool result: the model synthesises it.
    #[serde(default = "default_after_tool_tier")]
    pub after_tool_result: Tier,
}

fn default_tool_select_tier() -> Tier {
    Tier::Fast
}
fn default_after_tool_tier() -> Tier {
    Tier::Balanced
}

impl Default for AgentLoop {
    fn default() -> Self {
        Self { enabled: true, tool_select: default_tool_select_tier(), after_tool_result: default_after_tool_tier() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnAmbiguous {
    /// Ask one clarifying question (cheapest candidate writes it) instead of guessing.
    Clarify,
    /// Route the vague question one tier up.
    Escalate,
    Ignore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Answers scoring below this escalate when a stronger candidate remains.
    #[serde(default = "default_conf_threshold")]
    pub threshold: f64,
    /// Optional model name that judges text answers (grounded / complete /
    /// needs clarification). Use a fast or local model.
    #[serde(default)]
    pub verifier: Option<String>,
    /// Only run the verifier when the heuristic score is below this. Confident
    /// answers skip the extra round-trip entirely.
    #[serde(default = "default_verify_below")]
    pub verify_below: f64,
    /// Hard cap on the verifier call. When it is slow, the answer is returned
    /// on the heuristic score alone.
    #[serde(default = "default_verifier_timeout")]
    pub verifier_timeout_ms: u64,
    #[serde(default = "default_on_ambiguous")]
    pub on_ambiguous: OnAmbiguous,
}

fn default_verify_below() -> f64 {
    0.85
}
fn default_verifier_timeout() -> u64 {
    3000
}

fn default_conf_threshold() -> f64 {
    0.55
}
fn default_on_ambiguous() -> OnAmbiguous {
    OnAmbiguous::Clarify
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self { enabled: true, threshold: default_conf_threshold(), verifier: None, verify_below: default_verify_below(), verifier_timeout_ms: default_verifier_timeout(), on_ambiguous: default_on_ambiguous() }
    }
}

/// Hedged requests: if the first candidate has not answered after `after_ms`,
/// the next candidate is started in parallel and the first good answer wins.
/// Trades some duplicate tokens for tail latency.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Hedge {
    #[serde(default)]
    pub after_ms: Option<u64>,
}

/// HTTP server limits. All of them exist to fail fast instead of queueing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Request body limit. Large retrieved contexts need more than axum's 2 MB default.
    #[serde(default = "default_body_mb")]
    pub max_body_mb: usize,
    /// Requests in flight before new ones get an immediate 503.
    #[serde(default = "default_in_flight")]
    pub max_in_flight: usize,
    /// Whole-request wall clock at the HTTP layer; defaults to routing.max_latency_ms + 5000.
    #[serde(default)]
    pub request_timeout_ms: Option<u64>,
    /// Per-user fairness. Identity comes from `X-Pankh-User` or `pankhllm.user`.
    #[serde(default)]
    pub per_user: Option<PerUserBudget>,
    /// Environment variable holding comma-separated inbound API keys. When set and
    /// non-empty, every request except /health must carry one of them as
    /// `Authorization: Bearer <key>`, `api-key: <key>` or `x-api-key: <key>`.
    #[serde(default)]
    pub api_keys_env: Option<String>,
    /// Browser origins allowed to call the API (CORS), e.g. ["https://app.example.com"].
    /// Empty (the default) sends no CORS headers. "*" allows any origin; avoid it with api keys.
    #[serde(default)]
    pub cors_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerUserBudget {
    /// Sustained request rate per user.
    #[serde(default = "default_rpm")]
    pub requests_per_minute: u32,
    /// Short-term allowance above the sustained rate.
    #[serde(default = "default_burst")]
    pub burst: u32,
    /// Model spend per user per UTC day; requests beyond it get 429.
    #[serde(default)]
    pub daily_cost_usd: Option<f64>,
    /// Requests without a user identity: `allow` (default) or `reject`.
    #[serde(default)]
    pub anonymous: AnonymousPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnonymousPolicy {
    #[default]
    Allow,
    Reject,
}

fn default_rpm() -> u32 {
    60
}
fn default_burst() -> u32 {
    20
}

fn default_body_mb() -> usize {
    32
}
fn default_in_flight() -> usize {
    512
}

impl ServerConfig {
    /// Inbound keys resolved from the environment. Empty means open access.
    pub fn api_keys(&self) -> Vec<String> {
        self.api_keys_env
            .as_ref()
            .and_then(|k| std::env::var(k).ok())
            .map(|v| v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect())
            .unwrap_or_default()
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { max_body_mb: default_body_mb(), max_in_flight: default_in_flight(), request_timeout_ms: None, per_user: None, api_keys_env: None, cors_origins: vec![] }
    }
}

/// Speculative answer templates. On a tool-selection turn the model is also
/// asked, through a hidden tool, for a template of the final answer shaped for
/// the tool's JSON result. When the result arrives the router renders the
/// answer itself; the synthesis model runs only when the model flagged that the
/// data needs interpretation, or when the template cannot be filled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeAnswer {
    #[serde(default)]
    pub enabled: bool,
    /// How long a stored template waits for its tool result.
    #[serde(default = "default_spec_ttl")]
    pub ttl_secs: u64,
    /// Name of the hidden tool the model calls. Never shown to clients.
    #[serde(default = "default_spec_tool")]
    pub tool_name: String,
    #[serde(default = "default_max_rows")]
    pub max_rows: usize,
}

fn default_spec_ttl() -> u64 {
    900
}
fn default_spec_tool() -> String {
    "pankh_answer_template".into()
}

impl Default for SpeculativeAnswer {
    fn default() -> Self {
        Self { enabled: false, ttl_secs: default_spec_ttl(), tool_name: default_spec_tool(), max_rows: default_max_rows() }
    }
}

/// Learned routes: generalise the planning model's tool calls into slot-typed
/// shapes and serve later questions of the same shape with no model call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedRoutes {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_lr_ttl")]
    pub ttl_secs: u64,
    #[serde(default = "default_lr_max")]
    pub max_routes: usize,
    /// Cosine threshold on the abstracted question when an embedding model is configured
    /// under `cache.semantic`; without one only exact shapes match.
    #[serde(default = "default_lr_sim")]
    pub min_similarity: f64,
    /// Successful observations of a shape before it is served. Default 2: the
    /// first, possibly messy attempt never becomes the template on its own.
    #[serde(default = "default_lr_obs")]
    pub min_observations: u64,
    /// Only these tools may be learned (empty = any tool that passes the argument checks).
    #[serde(default)]
    pub allow_tools: Vec<String>,
    /// Argument names that mark free-text code: such calls are never learned, because
    /// slot-filling into code or SQL text is string interpolation of user input.
    #[serde(default = "default_deny_args")]
    pub deny_arg_names: Vec<String>,
    /// Longest string argument a learnable call may carry. Flat typed parameters are
    /// short; programs and queries are not.
    #[serde(default = "default_max_arg_len")]
    pub max_arg_len: usize,
    /// Known entity names (products, regions, segments, metrics) that count as slots
    /// regardless of casing, so "nrx for zelora" and "NRx for Cardivex" share a shape.
    #[serde(default)]
    pub vocab: Vec<String>,
    /// Optional file with one entity name per line, merged into `vocab`.
    #[serde(default)]
    pub vocab_file: Option<String>,
    /// JSON file the route memory is loaded from at start and saved to every
    /// `persist_interval_secs` and on shutdown. This is the trained state.
    #[serde(default)]
    pub persist_path: Option<String>,
    #[serde(default = "default_persist_interval")]
    pub persist_interval_secs: u64,
}

fn default_persist_interval() -> u64 {
    60
}

fn default_deny_args() -> Vec<String> {
    ["code", "sql", "query", "script", "program", "command", "expression", "statement", "source"].iter().map(|s| s.to_string()).collect()
}
fn default_max_arg_len() -> usize {
    80
}

fn default_lr_ttl() -> u64 {
    30 * 86_400
}
fn default_lr_max() -> usize {
    5_000
}
fn default_lr_sim() -> f64 {
    0.93
}
fn default_lr_obs() -> u64 {
    2
}

impl Default for LearnedRoutes {
    fn default() -> Self {
        Self { enabled: false, ttl_secs: default_lr_ttl(), max_routes: default_lr_max(), min_similarity: default_lr_sim(), min_observations: default_lr_obs(), allow_tools: Vec::new(), deny_arg_names: default_deny_args(), max_arg_len: default_max_arg_len(), vocab: Vec::new(), vocab_file: None, persist_path: None, persist_interval_secs: default_persist_interval() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    #[serde(default)]
    pub rules: Vec<Rule>,
    #[serde(default)]
    pub learned_routes: LearnedRoutes,
    #[serde(default)]
    pub speculative_answer: SpeculativeAnswer,
    #[serde(default)]
    pub hedge: Hedge,
    #[serde(default)]
    pub agent_loop: AgentLoop,
    #[serde(default)]
    pub confidence: ConfidenceConfig,
    #[serde(default = "default_tier")]
    pub default_tier: Tier,
    #[serde(default = "default_intent_map")]
    pub intents: HashMap<Intent, Tier>,
    #[serde(default)]
    pub weak_retrieval: WeakRetrieval,
    #[serde(default)]
    pub cascade: Cascade,
    /// Hard ceiling per request unless the request sets its own.
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    /// Seconds a model is skipped after a provider error.
    #[serde(default = "default_cooldown")]
    pub error_cooldown_secs: u64,
    /// Default output allowance when the request does not set `max_tokens`.
    #[serde(default = "default_output_tokens")]
    pub default_max_tokens: u32,
    /// Per-call timeout used when a model does not set its own.
    #[serde(default = "default_model_timeout")]
    pub default_timeout_secs: u64,
    /// Whole-request wall-clock budget (all cascade steps). Requests may lower it.
    #[serde(default = "default_request_budget")]
    pub max_latency_ms: u64,
    /// Instruction placed above retrieved chunks for lookup, extraction and
    /// summarization: the answer must come from the context.
    #[serde(default = "default_grounding_strict")]
    pub grounding_strict: String,
    /// Instruction for synthesis, reasoning, multi-hop and code: prefer the
    /// context but reason beyond it, labelling what is not in the context.
    #[serde(default = "default_grounding_open")]
    pub grounding_open: String,
}

fn default_grounding_strict() -> String {
    "Answer only from the retrieved context below and cite chunk ids like [1]. If the context does not contain the specific answer, start with \"I don't know\", then state the closest relevant fact the context does contain (for example a different period, region or an aggregate) so the reader knows what is available. Never derive or estimate a figure the context does not state.".into()
}
fn default_grounding_open() -> String {
    "Use the retrieved context below as your primary evidence and cite chunk ids like [1]. If it does not fully answer the question, reason from general knowledge and say clearly which parts are not supported by the context.".into()
}

fn default_model_timeout() -> u64 {
    60
}
fn default_request_budget() -> u64 {
    90_000
}

fn default_tier() -> Tier {
    Tier::Balanced
}
fn default_cooldown() -> u64 {
    30
}
fn default_output_tokens() -> u32 {
    2048
}

pub fn default_intent_map() -> HashMap<Intent, Tier> {
    HashMap::from([
        (Intent::Chitchat, Tier::Fast),
        (Intent::Lookup, Tier::Fast),
        (Intent::Extraction, Tier::Fast),
        (Intent::Summarization, Tier::Balanced),
        (Intent::Synthesis, Tier::Balanced),
        (Intent::Reasoning, Tier::Reasoning),
        (Intent::MultiHop, Tier::Reasoning),
        (Intent::Code, Tier::Reasoning),
    ])
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            learned_routes: LearnedRoutes::default(),
            speculative_answer: SpeculativeAnswer::default(),
            hedge: Hedge::default(),
            agent_loop: AgentLoop::default(),
            confidence: ConfidenceConfig::default(),
            default_tier: default_tier(),
            intents: default_intent_map(),
            weak_retrieval: WeakRetrieval::default(),
            cascade: Cascade::default(),
            max_cost_usd: None,
            error_cooldown_secs: default_cooldown(),
            default_max_tokens: default_output_tokens(),
            default_timeout_secs: default_model_timeout(),
            max_latency_ms: default_request_budget(),
            grounding_strict: default_grounding_strict(),
            grounding_open: default_grounding_open(),
        }
    }
}

/// Server-held system prompts (skill markdown, agent YAML, design docs). Clients
/// reference them by file stem instead of re-sending them on every call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptsConfig {
    /// Directory scanned at startup for `*.md`, `*.txt`, `*.yaml`, `*.yml`.
    #[serde(default)]
    pub dir: Option<String>,
    /// Prompt applied when a request names none. Optional.
    #[serde(default)]
    pub default: Option<String>,
    /// When a request names no prompt (or names "auto"), include the prompts whose
    /// front-matter `triggers` match the question plus those marked `always`.
    /// Keeps the system prompt small no matter how many skill files exist.
    #[serde(default)]
    pub auto_select: bool,
    /// Cap on auto-selected prompts per request (always-on ones do not count).
    #[serde(default = "default_max_auto")]
    pub max_auto: usize,
}

fn default_max_auto() -> usize {
    3
}

/// Paraphrase matching on top of the exact cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCache {
    /// A model with `kind: embedding`.
    pub embedding_model: String,
    /// Cosine similarity at or above this counts as the same question.
    #[serde(default = "default_sem_threshold")]
    pub threshold: f64,
}

fn default_sem_threshold() -> f64 {
    0.95
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
    /// Hit only when the retrieved context is byte-identical too (safe default for RAG).
    ExactContext,
    /// Key on the question alone plus the caller's `cache_key`; for pipelines whose
    /// context is stable (agent + skill) and whose data version is in `cache_key`.
    QuestionOnly,
}

/// Response cache. Exact hits return in microseconds; semantic hits after one
/// embedding call. Only confident, non-hedged answers and tool calls are stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cache_ttl")]
    pub ttl_secs: u64,
    #[serde(default = "default_cache_entries")]
    pub max_entries: usize,
    #[serde(default = "default_cache_scope")]
    pub scope: CacheScope,
    /// Answers below this confidence are not cached.
    #[serde(default = "default_cache_min_conf")]
    pub min_confidence: f64,
    /// With `scope: question_only`, refuse to read or write the cache unless the
    /// request carries a `cache_key`. The key is where tenant, business unit,
    /// row-level scope and data version must go; without it a question-only
    /// cache would serve one tenant's answer to another. Default true.
    #[serde(default = "default_true")]
    pub require_cache_key: bool,
    #[serde(default)]
    pub semantic: Option<SemanticCache>,
}

fn default_cache_ttl() -> u64 {
    3600
}
fn default_cache_entries() -> usize {
    20_000
}
fn default_cache_scope() -> CacheScope {
    CacheScope::ExactContext
}
fn default_cache_min_conf() -> f64 {
    0.7
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { enabled: false, ttl_secs: default_cache_ttl(), max_entries: default_cache_entries(), scope: default_cache_scope(), min_confidence: default_cache_min_conf(), require_cache_key: true, semantic: None }
    }
}

/// Fast typed decisions (which operation, which tool) without a generative model.
/// `native` (default): pankhllm's own model, trained from its own traffic by the
/// miner, seeded by catalog examples, running in-process in microseconds.
/// `external`: an optional adapter to any `POST /v1/systemone` service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionConfig {
    #[serde(default)]
    pub engine: DecisionEngineKind,
    /// Native: trained models exported with `pankhllm export-models`. Used when the trace
    /// store holds no newer ones, so a trained model ships as one reviewable file.
    #[serde(default)]
    pub models_file: Option<String>,
    /// External engine only, e.g. http://127.0.0.1:8000/v1/systemone
    #[serde(default)]
    pub url: Option<String>,
    /// Native: held-out precision the miner aims for when recommending a threshold.
    #[serde(default = "default_target_precision")]
    pub target_precision: f64,
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Hard cap per call. Past it the router takes its slower path.
    #[serde(default = "default_decision_timeout")]
    pub timeout_ms: u64,
    /// Probability needed to act on an unconstrained verdict (for example "no operation
    /// fits", which skips the planner call). Calibrate with shadow mode.
    #[serde(default = "default_decision_conf")]
    pub min_confidence: f64,
    /// Probability needed when the question's structure already agrees: the engine only
    /// chose among operations whose parameters the question fills exactly.
    #[serde(default = "default_consensus_conf")]
    pub consensus_confidence: f64,
    /// Native: share of the question's words that must be familiar from answerable
    /// questions. Mostly unfamiliar wording goes to the teacher even when the structure fits.
    #[serde(default = "default_min_known")]
    pub min_known_words: f64,
    /// `act` uses verdicts; `shadow` records them next to what the slower path decided,
    /// for calibration, without changing any answer.
    #[serde(default)]
    pub mode: DecisionMode,
    /// Checkpoint override passed through as `model`.
    #[serde(default)]
    pub model: Option<String>,
    /// Pick planner operations and resolve enum parameters.
    #[serde(default = "default_true")]
    pub planner: bool,
    /// Answer agent tool-selection turns when the chosen tool's arguments are flat
    /// typed parameters that the question fills unambiguously. Opt in.
    #[serde(default)]
    pub tools: bool,
    /// Restrict `tools` to these tool names (empty = any eligible tool).
    #[serde(default)]
    pub allow_tools: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionEngineKind {
    #[default]
    Native,
    External,
}

fn default_target_precision() -> f64 {
    0.98
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMode {
    #[default]
    Act,
    Shadow,
}

fn default_decision_timeout() -> u64 {
    150
}
fn default_decision_conf() -> f64 {
    0.85
}
fn default_consensus_conf() -> f64 {
    0.55
}
fn default_min_known() -> f64 {
    0.7
}

/// One parameter of a planner operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanParam {
    /// string | integer | number | boolean | enum
    #[serde(rename = "type", default = "default_param_type")]
    pub kind: String,
    #[serde(default)]
    pub description: String,
    /// Allowed values for `enum` (matched case-insensitively, returned canonical).
    #[serde(default)]
    pub values: Vec<String>,
    /// File with one allowed value per line, merged into `values`.
    #[serde(default)]
    pub values_file: Option<String>,
    /// Other ways people say each value: `trx: [total prescriptions, scripts]`.
    /// The slot filler recognises them without a model.
    #[serde(default)]
    pub aliases: std::collections::BTreeMap<String, Vec<String>>,
    /// Regex a string must match in full.
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub max_len: Option<usize>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub default: Option<serde_json::Value>,
}

fn default_param_type() -> String {
    "string".into()
}

/// An operation the planner may choose. The executor implements it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOperation {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub params: std::collections::BTreeMap<String, PlanParam>,
    /// Final answer rendered from the executor's `data` (+ `params.<name>`). Optional:
    /// the executor may return `answer` itself, or `rows` for a table.
    #[serde(default)]
    pub answer_template: Option<String>,
    /// Example questions. They seed pankhllm's own decision model before any traffic.
    #[serde(default)]
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// POST target. Receives {op, params, question, user, tenant, trace_id}; returns
    /// {ok?, data?, answer?, rows?}. Implement it in any language.
    pub url: String,
    #[serde(default = "default_exec_timeout")]
    pub timeout_ms: u64,
    /// Env var with a bearer token for the executor.
    #[serde(default)]
    pub api_key_env: Option<String>,
}

fn default_exec_timeout() -> u64 {
    3000
}

/// The planner lane: "the model reads, the executor runs". One small structured-output
/// call turns a question into a plan over a closed catalog of operations; the plan is
/// validated here, executed by your service, rendered, and learned by shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    #[serde(default)]
    pub enabled: bool,
    /// A configured model; small and fast is the point.
    pub model: String,
    #[serde(default)]
    pub operations: Vec<PlanOperation>,
    /// YAML/JSON file with `operations:`, merged with the inline list.
    #[serde(default)]
    pub catalog: Option<String>,
    pub executor: ExecutorConfig,
    /// Cap on the planning call; past it the request goes on to the normal lanes.
    #[serde(default = "default_planner_timeout")]
    pub timeout_ms: u64,
    /// Intents the planner is tried for. Reasoning, multi-hop and code go straight to the agent.
    #[serde(default = "default_planner_intents")]
    pub intents: Vec<Intent>,
    /// Also try multi-turn conversations (the planner sees only the last question).
    #[serde(default)]
    pub multi_turn: bool,
    /// Learn question shapes -> plans so the next question of that shape needs no model.
    #[serde(default = "default_true")]
    pub learn: bool,
    /// Example questions no operation should take; they seed the UNSUPPORTED class.
    #[serde(default)]
    pub unsupported_examples: Vec<String>,
}

fn default_planner_timeout() -> u64 {
    2500
}
fn default_planner_intents() -> Vec<Intent> {
    vec![Intent::Lookup, Intent::Extraction, Intent::Synthesis]
}

/// Embedded trace database. Everything routed is logged here off the hot path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreConfig {
    /// SQLite file. Relative paths are resolved from the working directory.
    pub path: String,
    /// Store shapes and hashes only, never the question text.
    #[serde(default)]
    pub redact_questions: bool,
    /// Traces older than this are deleted by `pankhllm mine`.
    #[serde(default = "default_retention")]
    pub retention_days: u64,
    /// In-memory queue in front of the writer; traces beyond it are dropped, not waited on.
    #[serde(default = "default_store_queue")]
    pub queue: usize,
}

fn default_retention() -> u64 {
    30
}
fn default_store_queue() -> usize {
    50_000
}

/// Online self-healing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealConfig {
    /// A learned route whose served calls come back as errors this many times is quarantined.
    #[serde(default = "default_quarantine_after")]
    pub quarantine_after_failures: u32,
    /// Load tuning written by `pankhllm mine --apply` (hedge timing, per-model timeouts) at start.
    #[serde(default)]
    pub autotune: bool,
}

fn default_quarantine_after() -> u32 {
    2
}

impl Default for HealConfig {
    fn default() -> Self {
        Self { quarantine_after_failures: default_quarantine_after(), autotune: false }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub providers: HashMap<String, ProviderConfig>,
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub store: Option<StoreConfig>,
    #[serde(default)]
    pub heal: HealConfig,
    #[serde(default)]
    pub planner: Option<PlannerConfig>,
    #[serde(default)]
    pub decisions: Option<DecisionConfig>,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub prompts: PromptsConfig,
    #[serde(default)]
    pub server: ServerConfig,
}

impl Config {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let raw = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("reading {}", path.as_ref().display()))?;
        Self::from_yaml(&raw)
    }

    pub fn from_yaml(raw: &str) -> Result<Self> {
        let cfg: Config = serde_yaml::from_str(raw).context("parsing config yaml")?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(!self.models.is_empty(), "config has no models");
        if let Some(v) = &self.routing.confidence.verifier {
            anyhow::ensure!(self.model(v).is_some(), "confidence.verifier '{v}' is not a configured model");
        }
        for r in &self.routing.rules {
            anyhow::ensure!(!r.any.is_empty() || r.pattern.is_some(), "rule {} has neither keywords nor a pattern", r.name);
            anyhow::ensure!(r.intent.is_some() || r.tier.is_some() || r.respond_with_tool.is_some(), "rule {} sets neither intent, tier nor respond_with_tool", r.name);
            if let Some(p) = &r.pattern {
                regex::Regex::new(p).map_err(|e| anyhow::anyhow!("rule {} pattern: {e}", r.name))?;
            }
        }
        if let Some(p) = &self.planner {
            anyhow::ensure!(self.model(&p.model).is_some(), "planner.model '{}' is not a configured model", p.model);
            for op in &p.operations {
                anyhow::ensure!(op.name != "UNSUPPORTED" && !op.name.is_empty(), "planner operation needs a name other than UNSUPPORTED");
                for (k, prm) in &op.params {
                    anyhow::ensure!(["string", "integer", "number", "boolean", "enum"].contains(&prm.kind.as_str()), "planner {}.{}: unknown type {}", op.name, k, prm.kind);
                    if let Some(pat) = &prm.pattern {
                        regex::Regex::new(pat).map_err(|e| anyhow::anyhow!("planner {}.{} pattern: {e}", op.name, k))?;
                    }
                }
            }
        }
        if let Some(d) = &self.decisions {
            anyhow::ensure!(d.engine != DecisionEngineKind::External || d.url.is_some(), "decisions.engine external needs decisions.url");
        }
        if let Some(sem) = &self.cache.semantic {
            let m = self.model(&sem.embedding_model).ok_or_else(|| anyhow::anyhow!("cache.semantic.embedding_model '{}' is not a configured model", sem.embedding_model))?;
            anyhow::ensure!(m.kind == ModelKind::Embedding, "cache.semantic.embedding_model '{}' must have kind: embedding", sem.embedding_model);
        }
        let mut seen = std::collections::HashSet::new();
        for m in &self.models {
            anyhow::ensure!(seen.insert(&m.name), "duplicate model name {}", m.name);
            anyhow::ensure!(
                self.providers.contains_key(&m.provider),
                "model {} references unknown provider {}",
                m.name,
                m.provider
            );
        }
        Ok(())
    }

    pub fn model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.name == name)
    }

    pub fn tier_for(&self, intent: Intent) -> Tier {
        self.routing.intents.get(&intent).copied().unwrap_or(self.routing.default_tier)
    }
}
