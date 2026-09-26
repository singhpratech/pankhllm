//! Shared data types. Everything the router reads or produces lives here.

use serde::{Deserialize, Serialize};

/// Capability tiers, ordered cheapest to strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Fast,
    Balanced,
    Reasoning,
}

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::Fast, Tier::Balanced, Tier::Reasoning];

    pub fn rank(self) -> i32 {
        match self {
            Tier::Fast => 0,
            Tier::Balanced => 1,
            Tier::Reasoning => 2,
        }
    }

    pub fn up(self) -> Tier {
        match self {
            Tier::Fast => Tier::Balanced,
            _ => Tier::Reasoning,
        }
    }

    pub fn parse(s: &str) -> Option<Tier> {
        match s.to_ascii_lowercase().as_str() {
            "fast" => Some(Tier::Fast),
            "balanced" => Some(Tier::Balanced),
            "reasoning" => Some(Tier::Reasoning),
            _ => None,
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Tier::Fast => "fast",
            Tier::Balanced => "balanced",
            Tier::Reasoning => "reasoning",
        };
        f.write_str(s)
    }
}

/// What the question asks the model to do with the retrieved context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    /// "What is the refund window?" - the answer is a span inside the context.
    Lookup,
    /// "List every date mentioned" - structured pull from the context.
    Extraction,
    /// "Summarize this policy."
    Summarization,
    /// "Compare X and Y" - combine across chunks.
    Synthesis,
    /// "Why does X imply Y" - multi-step inference.
    Reasoning,
    /// Chains of lookups across documents.
    MultiHop,
    /// Generate, fix or explain code.
    Code,
    /// Greetings, thanks - no context needed.
    Chitchat,
}

impl std::fmt::Display for Intent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Intent::Lookup => "lookup",
            Intent::Extraction => "extraction",
            Intent::Summarization => "summarization",
            Intent::Synthesis => "synthesis",
            Intent::Reasoning => "reasoning",
            Intent::MultiHop => "multi_hop",
            Intent::Code => "code",
            Intent::Chitchat => "chitchat",
        };
        f.write_str(s)
    }
}

/// One retrieved passage. `score` is whatever your retriever returns, higher is better.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub text: String,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
}

/// One function call requested by the model (OpenAI shape; Anthropic tool_use maps onto it).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments, exactly as the wire format carries them.
    pub arguments: String,
}

/// A function the caller exposes to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// JSON schema for the arguments.
    #[serde(default = "default_schema")]
    pub parameters: serde_json::Value,
}

fn default_schema() -> serde_json::Value {
    serde_json::json!({"type": "object", "properties": {}})
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// Assistant turns that requested tools.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// `tool` role: which call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn text(role: &str, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), tool_calls: Vec::new(), tool_call_id: None }
    }
}

/// The unit of work pankhllm routes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouteRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub context: Vec<Chunk>,
    /// Constraints the chosen model must satisfy, e.g. `["private"]`.
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    /// Force an intent instead of classifying.
    #[serde(default)]
    pub intent: Option<Intent>,
    /// Force a tier instead of deriving one.
    #[serde(default)]
    pub tier: Option<Tier>,
    /// Force a named model; bypasses routing but keeps fallbacks.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Vec<ToolDef>,
    /// "auto" | "none" | "required" | a function name.
    #[serde(default)]
    pub tool_choice: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Wall-clock budget for the whole request, cascade included.
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    /// Name of a server-held prompt (see `prompts.dir`) to prepend as the stable system block.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Per-request reasoning effort (minimal | low | medium | high | xhigh | max).
    /// Overrides the model's configured `effort` for this call only.
    #[serde(default)]
    pub effort: Option<String>,
    /// Caller-supplied cache partition, e.g. a data version or tenant. Changing it
    /// invalidates everything cached under the old value.
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub cache_mode: CacheMode,
    /// OpenAI-style structured output: {"type":"json_object"} or
    /// {"type":"json_schema","json_schema":{"name","schema","strict"}}. Forwarded as-is
    /// to OpenAI-compatible upstreams, translated for Anthropic and Responses.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Caller identity for per-user budgets and for executors (row-level scope).
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub stream: bool,
}

impl RouteRequest {
    pub fn query(&self) -> &str {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "user" && m.tool_call_id.is_none())
            .map(|m| m.content.as_str())
            .unwrap_or("")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheMode {
    #[default]
    Use,
    /// Neither read nor write the cache for this request.
    Bypass,
    /// Skip the read, recompute, overwrite.
    Refresh,
}

/// How an answer was served from the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    pub hit: bool,
    /// "exact" | "semantic"
    pub kind: String,
    pub similarity: Option<f64>,
    pub age_secs: u64,
}

/// Where in an agent loop this request sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// No tools involved.
    Plain,
    /// Tools offered, model must decide whether and what to call.
    ToolSelect,
    /// Tool results are in; model must synthesise them.
    AfterToolResult,
}

/// Everything the policy looks at, kept so decisions are explainable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signals {
    pub intent: Intent,
    pub phase: Phase,
    pub intent_source: String,
    /// Name of the config rule that matched, if any.
    pub matched_rule: Option<String>,
    pub query_tokens: u32,
    pub context_tokens: u32,
    pub history_tokens: u32,
    pub total_input_tokens: u32,
    pub n_chunks: usize,
    pub top_score: Option<f64>,
    pub weak_retrieval: bool,
    pub has_code: bool,
    pub required_tags: Vec<String>,
    /// Query looks under-specified: very short, or leans on an unresolved reference.
    pub ambiguous: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub name: String,
    pub tier: Tier,
    pub estimated_cost_usd: f64,
    pub reason: String,
}

/// Ordered candidates plus the reasons. Index 0 is what gets called first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub target_tier: Tier,
    pub candidates: Vec<Candidate>,
    pub rejected: Vec<Candidate>,
    pub signals: Signals,
    pub reasons: Vec<String>,
    pub abstain: bool,
    /// Policy says: ask a clarifying question instead of answering.
    #[serde(default)]
    pub clarify: bool,
}

/// How much to trust an answer, and why. Drives escalation and clarification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Confidence {
    /// 0..1. Below `routing.confidence.threshold` the router escalates if it can.
    pub score: f64,
    /// Retrieval strength component (top score, presence of context or tools).
    pub grounding: f64,
    pub hedged: bool,
    /// Answer cites chunk ids like [1] when context was supplied.
    pub cited: bool,
    /// The question itself was judged too vague to answer well.
    pub ambiguous: bool,
    /// Raw judgement from the verifier model, when one is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ok,
    Error,
    Refusal,
    LowConfidence,
    Empty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    pub model: String,
    pub outcome: Outcome,
    pub detail: String,
    pub latency_ms: u128,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completion {
    pub text: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// pankhllm model name that produced the answer.
    pub model: String,
    /// The provider's model id.
    pub provider_model: String,
    pub usage: Usage,
    pub decision: RoutingDecision,
    pub attempts: Vec<Attempt>,
    pub abstained: bool,
    #[serde(default)]
    pub confidence: Confidence,
    /// The answer is a clarifying question, not an attempt to answer.
    #[serde(default)]
    pub clarification: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheInfo>,
}
