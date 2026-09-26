//! Anthropic Messages API over raw HTTP (there is no official Rust SDK).

use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::{sse, EventStream, Provider, ProviderError, ProviderRequest, ProviderResponse, StopReason, StreamEvent};
use crate::config::{ModelConfig, ProviderConfig};
use crate::types::ToolCall;

const API_VERSION: &str = "2023-06-01";
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

pub struct AnthropicProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    key_env: String,
}

impl AnthropicProvider {
    pub fn new(cfg: &ProviderConfig) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .build()?;
        Ok(Self {
            http,
            base_url: cfg
                .base_url
                .clone()
                .unwrap_or_else(|| "https://api.anthropic.com".into())
                .trim_end_matches('/')
                .to_string(),
            api_key: cfg.api_key(),
            key_env: cfg.api_key_env.clone().unwrap_or_else(|| "ANTHROPIC_API_KEY".into()),
        })
    }

    fn body(&self, model: &ModelConfig, req: &ProviderRequest, stream: bool) -> Value {
        // Anthropic wants tool results as user-role tool_result blocks, and all
        // results for one assistant turn inside a single user message.
        let mut messages: Vec<Value> = Vec::new();
        for m in &req.messages {
            match m.role.as_str() {
                "assistant" => {
                    let mut blocks: Vec<Value> = Vec::new();
                    if !m.content.is_empty() {
                        blocks.push(json!({"type": "text", "text": m.content}));
                    }
                    for t in &m.tool_calls {
                        let input: Value = serde_json::from_str(&t.arguments).unwrap_or_else(|_| json!({}));
                        blocks.push(json!({"type": "tool_use", "id": t.id, "name": t.name, "input": input}));
                    }
                    if blocks.is_empty() {
                        continue;
                    }
                    messages.push(json!({"role": "assistant", "content": blocks}));
                }
                "user" if m.tool_call_id.is_some() => {
                    let block = json!({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content});
                    match messages.last_mut() {
                        Some(last) if last["role"] == "user" && last["content"].is_array() => {
                            last["content"].as_array_mut().unwrap().push(block);
                        }
                        _ => messages.push(json!({"role": "user", "content": [block]})),
                    }
                }
                "user" => messages.push(json!({"role": "user", "content": m.content})),
                _ => {}
            }
        }
        let mut body = json!({
            "model": model.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
            "stream": stream,
        });
        // Stable block first with a cache breakpoint, volatile block after it, so
        // the agent/skill prefix is a cache hit on every call.
        let mut system_blocks: Vec<Value> = Vec::new();
        if let Some(stable) = &req.system_stable {
            system_blocks.push(json!({"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}));
        }
        if let Some(sys) = &req.system {
            system_blocks.push(json!({"type": "text", "text": sys}));
        }
        if !system_blocks.is_empty() {
            body["system"] = json!(system_blocks);
        }
        if model.thinking_enabled() {
            body["thinking"] = json!({"type": "adaptive"});
        }
        // Anthropic takes low/medium/high/xhigh/max; "minimal" maps to low.
        if let Some(effort) = req.effort.as_ref().or(model.effort.as_ref()) {
            let e = if effort == "minimal" { "low" } else { effort.as_str() };
            body["output_config"] = json!({"effort": e});
        }
        if model.allow_sampling {
            if let Some(t) = req.temperature {
                body["temperature"] = json!(t);
            }
        }
        if model.server_fallbacks {
            body["fallbacks"] = json!("default");
        }
        if let Some(rf) = &req.response_format {
            match rf["type"].as_str() {
                Some("json_schema") => {
                    let mut oc = body["output_config"].clone();
                    if oc.is_null() { oc = json!({}); }
                    oc["format"] = json!({"type": "json_schema", "schema": rf["json_schema"]["schema"]});
                    body["output_config"] = oc;
                }
                Some("json_object") => {
                    // No native json_object mode: ask for it in the volatile system block.
                    if let Some(arr) = body["system"].as_array_mut() {
                        arr.push(json!({"type": "text", "text": "Respond with a single JSON object and nothing else."}));
                    } else {
                        body["system"] = json!([{"type": "text", "text": "Respond with a single JSON object and nothing else."}]);
                    }
                }
                _ => {}
            }
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(req.tools.iter().map(|t| json!({
                "name": t.name, "description": t.description, "input_schema": t.parameters
            })).collect::<Vec<_>>());
            // Forced choices ("required" / a name) are rejected by Claude Fable 5.1 and
            // Opus 5.5; those callers should use "auto" plus an instruction.
            if let Some(tc) = &req.tool_choice {
                body["tool_choice"] = match tc.as_str() {
                    "auto" => json!({"type": "auto"}),
                    "none" => json!({"type": "none"}),
                    "required" => json!({"type": "any"}),
                    name => json!({"type": "tool", "name": name}),
                };
            }
        }
        body
    }

    fn request(&self, model: &ModelConfig, body: &Value) -> Result<reqwest::RequestBuilder, ProviderError> {
        let key = self.api_key.as_ref().ok_or_else(|| ProviderError::MissingKey(self.key_env.clone()))?;
        let mut r = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", key)
            .header("anthropic-version", API_VERSION)
            .json(body);
        if model.server_fallbacks {
            r = r.header("anthropic-beta", FALLBACK_BETA);
        }
        Ok(r)
    }

    fn stop_reason(s: Option<&str>) -> StopReason {
        match s {
            Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("refusal") => StopReason::Refusal,
            Some("tool_use") => StopReason::ToolUse,
            _ => StopReason::Other,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn complete(&self, model: &ModelConfig, req: &ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let resp = self.request(model, &self.body(model, req, false))?.send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(ProviderError::from_status(status, &text));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Malformed(e.to_string()))?;
        let mut out = String::new();
        let mut tool_calls = Vec::new();
        for block in v["content"].as_array().into_iter().flatten() {
            match block["type"].as_str() {
                Some("text") => out.push_str(block["text"].as_str().unwrap_or("")),
                Some("tool_use") => tool_calls.push(ToolCall {
                    id: block["id"].as_str().unwrap_or("").to_string(),
                    name: block["name"].as_str().unwrap_or("").to_string(),
                    arguments: block["input"].to_string(),
                }),
                _ => {}
            }
        }
        Ok(ProviderResponse {
            text: out,
            tool_calls,
            input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
            stop_reason: Self::stop_reason(v["stop_reason"].as_str()),
        })
    }

    async fn stream(&self, model: &ModelConfig, req: &ProviderRequest) -> Result<EventStream, ProviderError> {
        let resp = self.request(model, &self.body(model, req, true))?.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(status, &text));
        }
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut stop = StopReason::Other;
        // Anthropic content-block index -> position among this turn's tool calls.
        let mut tool_index_of: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let events = sse::data_lines(resp.bytes_stream()).filter_map(move |item| {
            let out = match item {
                Err(e) => Some(Err(e)),
                Ok(data) => match serde_json::from_str::<Value>(&data) {
                    Err(e) => Some(Err(ProviderError::Malformed(e.to_string()))),
                    Ok(v) => match v["type"].as_str() {
                        Some("message_start") => {
                            input_tokens = v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
                            None
                        }
                        Some("content_block_start") if v["content_block"]["type"] == "tool_use" => {
                            let block = v["index"].as_u64().unwrap_or(0);
                            let index = tool_index_of.len();
                            tool_index_of.insert(block, index);
                            Some(Ok(StreamEvent::ToolCallStart {
                                index,
                                id: v["content_block"]["id"].as_str().unwrap_or("").to_string(),
                                name: v["content_block"]["name"].as_str().unwrap_or("").to_string(),
                            }))
                        }
                        Some("content_block_delta") if v["delta"]["type"] == "text_delta" => v["delta"]["text"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(|s| Ok(StreamEvent::Delta(s.to_string()))),
                        Some("content_block_delta") if v["delta"]["type"] == "input_json_delta" => {
                            let block = v["index"].as_u64().unwrap_or(0);
                            let index = *tool_index_of.get(&block).unwrap_or(&0);
                            v["delta"]["partial_json"].as_str().filter(|s| !s.is_empty()).map(|s| Ok(StreamEvent::ToolCallDelta { index, arguments: s.to_string() }))
                        }
                        Some("message_delta") => {
                            if let Some(o) = v["usage"]["output_tokens"].as_u64() {
                                output_tokens = o as u32;
                            }
                            if let Some(sr) = v["delta"]["stop_reason"].as_str() {
                                stop = Self::stop_reason(Some(sr));
                            }
                            None
                        }
                        Some("message_stop") => {
                            Some(Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason: stop }))
                        }
                        Some("error") => Some(Err(ProviderError::Retryable(
                            v["error"]["message"].as_str().unwrap_or("stream error").to_string(),
                        ))),
                        _ => None,
                    },
                },
            };
            async move { out }
        });
        Ok(events.boxed())
    }
}
