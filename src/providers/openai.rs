//! OpenAI Chat Completions, and every server that imitates it (Ollama, vLLM,
//! LM Studio, Groq, Together, OpenRouter, Azure-compatible gateways).

use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::{sse, EventStream, Provider, ProviderError, ProviderRequest, ProviderResponse, StopReason, StreamEvent};
use crate::config::{AuthStyle, ModelConfig, ProviderConfig, UpstreamApi};
use crate::types::ToolCall;

pub struct OpenAiProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    key_env: Option<String>,
    azure: bool,
    api_version: Option<String>,
    auth: AuthStyle,
    api: UpstreamApi,
}

impl OpenAiProvider {
    pub fn new(cfg: &ProviderConfig) -> anyhow::Result<Self> {
        Self::build(cfg, false)
    }

    /// Azure OpenAI: `base_url` is `https://<resource>.openai.azure.com`, model = deployment name.
    pub fn azure(cfg: &ProviderConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(cfg.base_url.is_some(), "azure provider needs base_url (the resource endpoint)");
        Self::build(cfg, true)
    }

    fn build(cfg: &ProviderConfig, azure: bool) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .connect_timeout(Duration::from_secs(cfg.connect_timeout_secs))
            .build()?;
        Ok(Self {
            http,
            base_url: cfg
                .base_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".into())
                .trim_end_matches('/')
                .to_string(),
            api_key: cfg.api_key(),
            key_env: cfg.api_key_env.clone(),
            azure,
            api_version: cfg.api_version.clone(),
            auth: cfg.auth.unwrap_or(if azure { AuthStyle::ApiKey } else { AuthStyle::Bearer }),
            api: cfg.api,
        })
    }

    fn effort_for(&self, model: &ModelConfig, req: &ProviderRequest) -> Option<String> {
        req.effort.as_ref().or(model.effort.as_ref()).map(|e| match e.as_str() { "xhigh" | "max" => "high".to_string(), other => other.to_string() })
    }

    /// Responses API request body, built from the same normalised request.
    fn responses_body(&self, model: &ModelConfig, req: &ProviderRequest, stream: bool) -> Value {
        let mut input: Vec<Value> = Vec::new();
        for m in &req.messages {
            if let Some(id) = &m.tool_call_id {
                input.push(json!({"type": "function_call_output", "call_id": id, "output": m.content}));
                continue;
            }
            if !m.content.is_empty() || m.tool_calls.is_empty() {
                input.push(json!({"role": m.role, "content": m.content}));
            }
            for t in &m.tool_calls {
                input.push(json!({"type": "function_call", "call_id": t.id, "name": t.name, "arguments": t.arguments}));
            }
        }
        let mut body = json!({"model": model.model, "input": input, "stream": stream, "store": false, "max_output_tokens": req.max_tokens});
        let sys: Vec<&str> = [req.system_stable.as_deref(), req.system.as_deref()].into_iter().flatten().collect();
        if !sys.is_empty() {
            body["instructions"] = json!(sys.join("\n\n"));
        }
        if let Some(e) = self.effort_for(model, req) {
            body["reasoning"] = json!({"effort": e});
        } else if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(req.tools.iter().map(|t| json!({"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters})).collect::<Vec<_>>());
            if let Some(tc) = &req.tool_choice {
                body["tool_choice"] = match tc.as_str() {
                    "auto" | "none" | "required" => json!(tc),
                    name => json!({"type": "function", "name": name}),
                };
            }
        }
        if let Some(rf) = &req.response_format {
            // chat {"type":"json_schema","json_schema":{name,schema,strict}} -> responses text.format
            let fmt = match rf["type"].as_str() {
                Some("json_schema") => {
                    let js = &rf["json_schema"];
                    json!({"type": "json_schema", "name": js["name"].as_str().unwrap_or("output"), "schema": js["schema"], "strict": js["strict"].as_bool().unwrap_or(true)})
                }
                Some("json_object") => json!({"type": "json_object"}),
                _ => json!({"type": "text"}),
            };
            body["text"] = json!({"format": fmt});
        }
        if let Some(stable) = &req.system_stable {
            body["prompt_cache_key"] = json!(format!("pankh-{:016x}", crate::cache::hash64(stable)));
        }
        body
    }

    fn responses_url(&self) -> String {
        if !self.azure {
            return format!("{}/responses", self.base_url);
        }
        match &self.api_version {
            Some(v) => format!("{}/openai/responses?api-version={v}", self.base_url),
            None => format!("{}/openai/v1/responses", self.base_url),
        }
    }

    fn parse_responses(v: &Value) -> Result<ProviderResponse, ProviderError> {
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for item in v["output"].as_array().into_iter().flatten() {
            match item["type"].as_str() {
                Some("message") => {
                    for part in item["content"].as_array().into_iter().flatten() {
                        if part["type"] == "output_text" {
                            text.push_str(part["text"].as_str().unwrap_or(""));
                        } else if part["type"] == "refusal" {
                            return Ok(ProviderResponse { text: String::new(), tool_calls: vec![], input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32, output_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32, stop_reason: StopReason::Refusal });
                        }
                    }
                }
                Some("function_call") => tool_calls.push(ToolCall {
                    id: item["call_id"].as_str().or(item["id"].as_str()).unwrap_or("").to_string(),
                    name: item["name"].as_str().unwrap_or("").to_string(),
                    arguments: match &item["arguments"] { Value::String(s) => s.clone(), other => other.to_string() },
                }),
                _ => {}
            }
        }
        let stop = if !tool_calls.is_empty() {
            StopReason::ToolUse
        } else if v["status"] == "incomplete" && v["incomplete_details"]["reason"] == "max_output_tokens" {
            StopReason::MaxTokens
        } else if v["status"] == "completed" {
            StopReason::EndTurn
        } else {
            StopReason::Other
        };
        Ok(ProviderResponse { text, tool_calls, input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32, output_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32, stop_reason: stop })
    }

    fn url(&self, model: &ModelConfig) -> String {
        if !self.azure {
            return format!("{}/chat/completions", self.base_url);
        }
        match &self.api_version {
            Some(v) => format!("{}/openai/deployments/{}/chat/completions?api-version={v}", self.base_url, model.model),
            None => format!("{}/openai/v1/chat/completions", self.base_url),
        }
    }

    fn body(&self, model: &ModelConfig, req: &ProviderRequest, stream: bool) -> Value {
        let mut messages: Vec<Value> = Vec::new();
        let sys: Vec<&str> = [req.system_stable.as_deref(), req.system.as_deref()].into_iter().flatten().collect();
        if !sys.is_empty() {
            messages.push(json!({"role": "system", "content": sys.join("\n\n")}));
        }
        for m in &req.messages {
            // Tool results travel as role "tool" on this wire format.
            let role = if m.tool_call_id.is_some() { "tool" } else { m.role.as_str() };
            let mut v = json!({"role": role, "content": m.content});
            if !m.tool_calls.is_empty() {
                v["tool_calls"] = json!(m.tool_calls.iter().map(|t| json!({
                    "id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}
                })).collect::<Vec<_>>());
                if m.content.is_empty() {
                    v["content"] = Value::Null;
                }
            }
            if let Some(id) = &m.tool_call_id {
                v["tool_call_id"] = json!(id);
            }
            messages.push(v);
        }
        let mut body = json!({
            "model": model.model,
            "messages": messages,
            "stream": stream,
        });
        // OpenAI and Azure moved to max_completion_tokens (reasoning models reject
        // max_tokens); most OpenAI-compatible local servers still want max_tokens.
        if self.azure || self.base_url.starts_with("https://api.openai.com") {
            body["max_completion_tokens"] = json!(req.max_tokens);
            // Route requests with the same stable prefix to the same cache shard.
            if let Some(stable) = &req.system_stable {
                body["prompt_cache_key"] = json!(format!("pankh-{:016x}", crate::cache::hash64(stable)));
            }
        } else {
            body["max_tokens"] = json!(req.max_tokens);
        }
        // Reasoning models: `effort` maps to reasoning_effort and sampling params are rejected.
        // OpenAI accepts minimal/low/medium/high; anything above is clamped to high.
        if let Some(e) = req.effort.as_ref().or(model.effort.as_ref()) {
            let e = match e.as_str() { "xhigh" | "max" => "high", other => other };
            body["reasoning_effort"] = json!(e);
        }
        if !req.tools.is_empty() {
            body["tools"] = json!(req.tools.iter().map(|t| json!({
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters}
            })).collect::<Vec<_>>());
            if let Some(tc) = &req.tool_choice {
                body["tool_choice"] = match tc.as_str() {
                    "auto" | "none" | "required" => json!(tc),
                    name => json!({"type": "function", "function": {"name": name}}),
                };
            }
        }
        if let (Some(t), None, None) = (req.temperature, &model.effort, &req.effort) {
            body["temperature"] = json!(t);
        }
        if let Some(rf) = &req.response_format {
            body["response_format"] = rf.clone();
        }
        if stream {
            body["stream_options"] = json!({"include_usage": true});
        }
        body
    }

    fn request(&self, model: &ModelConfig, body: &Value) -> reqwest::RequestBuilder {
        let url = if self.api == UpstreamApi::Responses { self.responses_url() } else { self.url(model) };
        let mut r = self.http.post(url).json(body);
        if let Some(k) = &self.api_key {
            r = match self.auth {
                AuthStyle::Bearer => r.bearer_auth(k),
                AuthStyle::ApiKey => r.header("api-key", k),
            };
        }
        r
    }

    fn stop_reason(s: Option<&str>) -> StopReason {
        match s {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::Refusal,
            Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
            _ => StopReason::Other,
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn embed(&self, model: &ModelConfig, input: &str) -> Result<Vec<f32>, ProviderError> {
        let url = if self.azure {
            match &self.api_version {
                Some(v) => format!("{}/openai/deployments/{}/embeddings?api-version={v}", self.base_url, model.model),
                None => format!("{}/openai/v1/embeddings", self.base_url),
            }
        } else {
            format!("{}/embeddings", self.base_url)
        };
        let mut r = self.http.post(url).json(&json!({"model": model.model, "input": input}));
        if let Some(k) = &self.api_key {
            r = match self.auth {
                AuthStyle::Bearer => r.bearer_auth(k),
                AuthStyle::ApiKey => r.header("api-key", k),
            };
        }
        let resp = r.send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(ProviderError::from_status(status, &text));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Malformed(e.to_string()))?;
        let vec: Vec<f32> = v["data"][0]["embedding"].as_array().ok_or_else(|| ProviderError::Malformed("no embedding".into()))?.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
        if vec.is_empty() {
            return Err(ProviderError::Malformed("empty embedding".into()));
        }
        Ok(vec)
    }

    async fn complete(&self, model: &ModelConfig, req: &ProviderRequest) -> Result<ProviderResponse, ProviderError> {
        let body = if self.api == UpstreamApi::Responses { self.responses_body(model, req, false) } else { self.body(model, req, false) };
        let resp = self.request(model, &body).send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(ProviderError::from_status(status, &text));
        }
        let v: Value = serde_json::from_str(&text).map_err(|e| ProviderError::Malformed(e.to_string()))?;
        if self.api == UpstreamApi::Responses {
            return Self::parse_responses(&v);
        }
        let choice = v["choices"].get(0).ok_or_else(|| ProviderError::Malformed("no choices".into()))?;
        let mut content = choice["message"]["content"].as_str().unwrap_or("").to_string();
        // Reasoning models on local servers may spend the whole budget thinking and
        // return an empty `content` with the thoughts in `reasoning`/`reasoning_content`.
        // Keep the answer empty (the cascade escalates) but say why in the text tag.
        if content.trim().is_empty() {
            let thought = choice["message"]["reasoning_content"].as_str().or(choice["message"]["reasoning"].as_str());
            if thought.is_some_and(|t| !t.trim().is_empty()) && choice["finish_reason"] == "length" {
                content.clear();
                tracing::warn!(model = %model.model, "reasoning consumed max_tokens before any answer; raise max_output or lower thinking");
            }
        }
        let tool_calls: Vec<ToolCall> = choice["message"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| {
                let f = &t["function"];
                let name = f["name"].as_str()?.to_string();
                // Some local servers emit arguments as an object instead of a string; normalise.
                let arguments = match &f["arguments"] {
                    Value::String(s) => s.clone(),
                    Value::Null => "{}".to_string(),
                    other => other.to_string(),
                };
                let id = t["id"].as_str().map(str::to_string).unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
                Some(ToolCall { id, name, arguments })
            })
            .collect();
        // OpenAI returns an explicit refusal field; treat it as a refusal stop.
        let refused = choice["message"]["refusal"].as_str().is_some_and(|r| !r.is_empty());
        let stop = if refused {
            StopReason::Refusal
        } else if !tool_calls.is_empty() {
            StopReason::ToolUse
        } else {
            Self::stop_reason(choice["finish_reason"].as_str())
        };
        Ok(ProviderResponse {
            text: content,
            tool_calls,
            input_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            stop_reason: stop,
        })
    }

    async fn stream(&self, model: &ModelConfig, req: &ProviderRequest) -> Result<EventStream, ProviderError> {
        if self.api == UpstreamApi::Responses {
            return self.stream_responses(model, req).await;
        }
        let resp = self.request(model, &self.body(model, req, true)).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(status, &text));
        }
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut stop = StopReason::Other;
        let mut finished = false;
        let mut saw_tool_call = false;
        let events = sse::data_lines(resp.bytes_stream()).flat_map(move |item| {
            let mut out: Vec<Result<StreamEvent, ProviderError>> = Vec::new();
            match item {
                Err(e) => out.push(Err(e)),
                Ok(data) if data == "[DONE]" => {
                    if !finished {
                        finished = true;
                        if saw_tool_call && stop == StopReason::Other {
                            stop = StopReason::ToolUse;
                        }
                        out.push(Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason: stop }));
                    }
                }
                Ok(data) => match serde_json::from_str::<Value>(&data) {
                    Err(e) => out.push(Err(ProviderError::Malformed(e.to_string()))),
                    Ok(v) => {
                        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                            input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                            output_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as u32;
                        }
                        let choice = &v["choices"][0];
                        if let Some(fr) = choice["finish_reason"].as_str() {
                            stop = Self::stop_reason(Some(fr));
                        }
                        if let Some(s) = choice["delta"]["content"].as_str().filter(|s| !s.is_empty()) {
                            out.push(Ok(StreamEvent::Delta(s.to_string())));
                        }
                        for tc in choice["delta"]["tool_calls"].as_array().into_iter().flatten() {
                            let index = tc["index"].as_u64().unwrap_or(0) as usize;
                            if let Some(name) = tc["function"]["name"].as_str().filter(|n| !n.is_empty()) {
                                saw_tool_call = true;
                                let id = tc["id"].as_str().map(str::to_string).unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
                                out.push(Ok(StreamEvent::ToolCallStart { index, id, name: name.to_string() }));
                            }
                            if let Some(args) = tc["function"]["arguments"].as_str().filter(|a| !a.is_empty()) {
                                out.push(Ok(StreamEvent::ToolCallDelta { index, arguments: args.to_string() }));
                            }
                        }
                    }
                },
            }
            futures::stream::iter(out)
        });
        Ok(events.boxed())
    }
}

impl OpenAiProvider {
    #[allow(dead_code)]
    pub fn key_env(&self) -> Option<&str> {
        self.key_env.as_deref()
    }

    /// Responses API streaming: named events -> normalised StreamEvents.
    async fn stream_responses(&self, model: &ModelConfig, req: &ProviderRequest) -> Result<EventStream, ProviderError> {
        let resp = self.request(model, &self.responses_body(model, req, true)).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(status, &text));
        }
        // output_index of function_call items -> position among tool calls
        let mut call_index: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        let mut finished = false;
        let events = sse::data_lines(resp.bytes_stream()).flat_map(move |item| {
            let mut out: Vec<Result<StreamEvent, ProviderError>> = Vec::new();
            match item {
                Err(e) => out.push(Err(e)),
                Ok(data) if data == "[DONE]" => {}
                Ok(data) => match serde_json::from_str::<Value>(&data) {
                    Err(e) => out.push(Err(ProviderError::Malformed(e.to_string()))),
                    Ok(v) => match v["type"].as_str() {
                        Some("response.output_text.delta") => {
                            if let Some(d) = v["delta"].as_str().filter(|d| !d.is_empty()) {
                                out.push(Ok(StreamEvent::Delta(d.to_string())));
                            }
                        }
                        Some("response.output_item.added") if v["item"]["type"] == "function_call" => {
                            let oi = v["output_index"].as_u64().unwrap_or(0);
                            let index = call_index.len();
                            call_index.insert(oi, index);
                            out.push(Ok(StreamEvent::ToolCallStart {
                                index,
                                id: v["item"]["call_id"].as_str().or(v["item"]["id"].as_str()).unwrap_or("").to_string(),
                                name: v["item"]["name"].as_str().unwrap_or("").to_string(),
                            }));
                        }
                        Some("response.function_call_arguments.delta") => {
                            let oi = v["output_index"].as_u64().unwrap_or(0);
                            let index = *call_index.get(&oi).unwrap_or(&0);
                            if let Some(d) = v["delta"].as_str().filter(|d| !d.is_empty()) {
                                out.push(Ok(StreamEvent::ToolCallDelta { index, arguments: d.to_string() }));
                            }
                        }
                        Some("response.completed") | Some("response.incomplete") if !finished => {
                            finished = true;
                            let r = &v["response"];
                            let stop = if !call_index.is_empty() {
                                StopReason::ToolUse
                            } else if r["status"] == "incomplete" {
                                StopReason::MaxTokens
                            } else {
                                StopReason::EndTurn
                            };
                            out.push(Ok(StreamEvent::Done { input_tokens: r["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32, output_tokens: r["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32, stop_reason: stop }));
                        }
                        Some("response.failed") | Some("error") => {
                            let msg = v["response"]["error"]["message"].as_str().or(v["error"]["message"].as_str()).or(v["message"].as_str()).unwrap_or("upstream stream failed");
                            out.push(Err(ProviderError::Retryable(msg.to_string())));
                        }
                        _ => {}
                    },
                },
            }
            futures::stream::iter(out)
        });
        Ok(events.boxed())
    }
}
