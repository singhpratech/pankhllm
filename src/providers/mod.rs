//! Provider adapters. Each one speaks its vendor's wire protocol over raw HTTP
//! and normalises the result into `ProviderResponse` / `StreamEvent`.

pub mod anthropic;
pub mod openai;
pub mod sse;

use async_trait::async_trait;
use futures::stream::BoxStream;
use thiserror::Error;

use crate::config::ModelConfig;
use crate::types::{Message, ToolCall, ToolDef};

#[derive(Debug, Clone)]
pub struct ProviderRequest {
    /// Byte-stable across calls (agent definition, skills). Providers that
    /// support prompt caching mark this block cacheable.
    pub system_stable: Option<String>,
    /// Per-request system content (retrieved context, caller's system message).
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: Option<String>,
    /// Per-request effort; falls back to the model's configured effort.
    pub effort: Option<String>,
    pub response_format: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    Refusal,
    ToolUse,
    Other,
}

#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Delta(String),
    /// A tool call begins; `index` is its position among the turn's tool calls.
    ToolCallStart { index: usize, id: String, name: String },
    /// A fragment of that tool call's JSON arguments.
    ToolCallDelta { index: usize, arguments: String },
    Done { input_tokens: u32, output_tokens: u32, stop_reason: StopReason },
}

pub type EventStream = BoxStream<'static, Result<StreamEvent, ProviderError>>;

#[derive(Debug, Error)]
pub enum ProviderError {
    /// 429 / 5xx / timeouts: the model is unhealthy right now, cool it down.
    #[error("retryable provider error: {0}")]
    Retryable(String),
    /// Network-level failure.
    #[error("transport error: {0}")]
    Transport(String),
    /// 4xx we caused: auth, missing model, quota shape. Not the model's fault; the
    /// next candidate may still work.
    #[error("request rejected: {0}")]
    Rejected(String),
    /// 400 / 422: the request itself is invalid for this upstream. Deterministic:
    /// retrying elsewhere with the same request wastes time, so the cascade stops.
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("malformed provider response: {0}")]
    Malformed(String),
    #[error("missing API key: set {0}")]
    MissingKey(String),
}

impl ProviderError {
    pub fn from_status(status: reqwest::StatusCode, body: &str) -> Self {
        let body = truncate(body, 400);
        if status.as_u16() == 429 || status.is_server_error() || status.as_u16() == 408 {
            ProviderError::Retryable(format!("{status}: {body}"))
        } else if status.as_u16() == 400 || status.as_u16() == 422 {
            ProviderError::Invalid(format!("{status}: {body}"))
        } else {
            ProviderError::Rejected(format!("{status}: {body}"))
        }
    }
}

impl From<reqwest::Error> for ProviderError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            ProviderError::Retryable(format!("timeout: {e}"))
        } else {
            ProviderError::Transport(e.to_string())
        }
    }
}

pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let cut: String = s.chars().take(n).collect();
        format!("{cut}...")
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, model: &ModelConfig, req: &ProviderRequest) -> Result<ProviderResponse, ProviderError>;
    async fn stream(&self, model: &ModelConfig, req: &ProviderRequest) -> Result<EventStream, ProviderError>;
    /// Embedding for the semantic cache. Providers without an embeddings API reject.
    async fn embed(&self, model: &ModelConfig, _input: &str) -> Result<Vec<f32>, ProviderError> {
        Err(ProviderError::Rejected(format!("provider for {} has no embeddings endpoint", model.name)))
    }
}
