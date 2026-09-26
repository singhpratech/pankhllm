//! HTTP surface. `/v1/chat/completions` is OpenAI-shaped so any language's
//! OpenAI client, Semantic Kernel, LangChain or LlamaIndex can point at it by
//! changing the base URL. Routing hints travel in the `pankhllm` body field or,
//! for clients that cannot add body fields, in `X-Pankh-*` headers.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse, Response},
    routing::{get, post},
    Json, Router as AxumRouter,
};
use tokio::sync::Semaphore;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::providers::StreamEvent;
use crate::router::Router;
use crate::types::*;

pub type AppState = Arc<Router>;

/// Chat request as OpenAI clients send it, plus an optional `pankhllm` extension.
/// Unknown fields (tools, response_format, stream_options...) are accepted and ignored.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// OpenAI chat-completions style per-request effort.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub response_format: Option<Value>,
    /// OpenAI's own end-user field; used as the budget identity when no header is set.
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub pankhllm: Option<PankhExt>,
}

const EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

#[derive(Debug, Default, Deserialize)]
pub struct PankhExt {
    #[serde(default)]
    pub context: Vec<Chunk>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    #[serde(default)]
    pub max_latency_ms: Option<u64>,
    #[serde(default)]
    pub intent: Option<Intent>,
    #[serde(default)]
    pub tier: Option<Tier>,
    #[serde(default)]
    pub prompt: Option<String>,
    /// Cache partition (data version, tenant). See `cache` in the config.
    #[serde(default)]
    pub cache_key: Option<String>,
    /// "use" (default) | "bypass" | "refresh"
    #[serde(default)]
    pub cache: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ErrorBody,
}
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub message: String,
    pub r#type: String,
    pub code: u16,
}

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    let body = ApiError { error: ErrorBody { message: msg.into(), r#type: kind.into(), code: status.as_u16() } };
    (status, Json(body)).into_response()
}

const MAX_MESSAGES: usize = 512;
const MAX_CHUNKS: usize = 256;
const MAX_TOKENS_CAP: u32 = 128_000;
const MAX_TOOLS: usize = 128;

/// Flatten OpenAI content (string or array of parts) to text; images are dropped.
fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name).and_then(|v| v.to_str().ok()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Translate the wire request into a `RouteRequest`, applying header hints.
fn to_route_request(body: ChatRequest, headers: &HeaderMap) -> Result<RouteRequest, Box<Response>> {
    if body.messages.is_empty() {
        return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", "messages must not be empty")));
    }
    if body.messages.len() > MAX_MESSAGES {
        return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("too many messages (max {MAX_MESSAGES})"))));
    }
    let mut messages = Vec::with_capacity(body.messages.len());
    for m in &body.messages {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let role = match role {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            // tool results are folded into the transcript as user text
            "tool" | "function" => "user",
            other => return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("unsupported role '{other}'")))),
        };
        let tool_calls: Vec<ToolCall> = m["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| {
                let f = &t["function"];
                Some(ToolCall {
                    id: t["id"].as_str()?.to_string(),
                    name: f["name"].as_str()?.to_string(),
                    arguments: match &f["arguments"] {
                        Value::String(a) => a.clone(),
                        Value::Null => "{}".into(),
                        other => other.to_string(),
                    },
                })
            })
            .collect();
        let tool_call_id = m["tool_call_id"].as_str().map(str::to_string);
        if role == "user" && m["role"] == "tool" && tool_call_id.is_none() {
            return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", "tool message requires tool_call_id")));
        }
        messages.push(Message { role: role.into(), content: content_text(m.get("content").unwrap_or(&Value::Null)), tool_calls, tool_call_id });
    }

    if body.tools.len() > MAX_TOOLS {
        return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("too many tools (max {MAX_TOOLS})"))));
    }
    let mut tools = Vec::with_capacity(body.tools.len());
    for t in &body.tools {
        let f = if t["type"] == "function" || t.get("function").is_some() { &t["function"] } else { t };
        let Some(name) = f["name"].as_str().filter(|n| !n.is_empty()) else {
            return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", "each tool needs function.name")));
        };
        tools.push(ToolDef {
            name: name.to_string(),
            description: f["description"].as_str().unwrap_or("").to_string(),
            parameters: f.get("parameters").cloned().filter(|p| p.is_object()).unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        });
    }
    let tool_choice = match &body.tool_choice {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if ["auto", "none", "required"].contains(&s.as_str()) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("function").and_then(|f| f["name"].as_str()).map(str::to_string),
        Some(other) => return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("unsupported tool_choice {other}")))),
    };
    if tools.is_empty() {
        // tool_choice without tools is meaningless; drop it rather than confuse upstreams.
        let _ = &tool_choice;
    }
    if !messages.iter().any(|m| m.role == "user") {
        return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", "at least one user message is required")));
    }

    let ext = body.pankhllm.unwrap_or_default();
    if ext.context.len() > MAX_CHUNKS {
        return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("too many context chunks (max {MAX_CHUNKS})"))));
    }

    // `model` is routed unless it names a configured model. "auto", "auto/fast", "pankhllm" all route.
    let (forced_model, tier_from_model) = match body.model.as_deref() {
        None | Some("") | Some("auto") | Some("pankhllm") => (None, None),
        Some(m) if m.starts_with("auto/") => (None, Tier::parse(&m[5..])),
        Some(m) => (Some(m.to_string()), None),
    };

    let mut tags = ext.tags;
    if let Some(h) = header(headers, "x-pankh-tags") {
        tags.extend(h.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
    }
    tags.sort();
    tags.dedup();

    let max_tokens = body.max_tokens.or(body.max_completion_tokens).map(|t| t.clamp(1, MAX_TOKENS_CAP));
    let intent = ext.intent.or_else(|| header(headers, "x-pankh-intent").and_then(|s| serde_json::from_value(json!(s)).ok()));
    let tier = ext.tier.or(tier_from_model).or_else(|| header(headers, "x-pankh-tier").and_then(|s| Tier::parse(&s)));
    let max_cost_usd = ext.max_cost_usd.or_else(|| header(headers, "x-pankh-max-cost-usd").and_then(|s| s.parse().ok())).filter(|c: &f64| c.is_finite() && *c >= 0.0);
    let max_latency_ms = ext.max_latency_ms.or_else(|| header(headers, "x-pankh-max-latency-ms").and_then(|s| s.parse().ok()));
    let prompt = ext.prompt.or_else(|| header(headers, "x-pankh-prompt"));
    let temperature = body.temperature.filter(|t| t.is_finite() && (0.0..=2.0).contains(t));
    let cache_key = ext.cache_key.or_else(|| header(headers, "x-pankh-cache-key"));
    let cache_mode = match ext.cache.or_else(|| header(headers, "x-pankh-cache")).as_deref().map(|s| s.to_ascii_lowercase()) {
        None | Some(_) if false => CacheMode::Use,
        Some(ref m) if m == "bypass" || m == "no-cache" || m == "no-store" => CacheMode::Bypass,
        Some(ref m) if m == "refresh" => CacheMode::Refresh,
        Some(ref m) if m == "use" => CacheMode::Use,
        None => CacheMode::Use,
        Some(other) => return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("unsupported cache mode '{other}'")))),
    };
    let effort = match body.reasoning_effort.or_else(|| header(headers, "x-pankh-effort")) {
        None => None,
        Some(e) if EFFORTS.contains(&e.as_str()) => Some(e),
        Some(other) => return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("unsupported reasoning effort '{other}'")))),
    };

    let response_format = match &body.response_format {
        None | Some(Value::Null) => None,
        Some(v) if matches!(v["type"].as_str(), Some("json_object") | Some("json_schema") | Some("text")) => Some(v.clone()),
        Some(other) => return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("unsupported response_format {other}")))),
    };
    let user = ext.user.or_else(|| header(headers, "x-pankh-user")).or(body.user).filter(|u| !u.is_empty() && u.len() <= 256);

    Ok(RouteRequest {
        messages,
        context: ext.context,
        tags,
        max_tokens,
        max_cost_usd,
        intent,
        tier,
        model: forced_model,
        tool_choice: if tools.is_empty() { None } else { tool_choice },
        tools,
        temperature,
        max_latency_ms,
        prompt,
        effort,
        cache_key,
        cache_mode,
        response_format,
        user,
        stream: body.stream,
    })
}

fn tool_calls_json(calls: &[ToolCall]) -> Value {
    json!(calls.iter().enumerate().map(|(i, t)| json!({
        "index": i, "id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}
    })).collect::<Vec<_>>())
}

fn openai_response(id: &str, c: &Completion) -> Value {
    let mut message = json!({"role": "assistant", "content": c.text});
    let finish = if c.tool_calls.is_empty() { "stop" } else { "tool_calls" };
    if !c.tool_calls.is_empty() {
        message["tool_calls"] = tool_calls_json(&c.tool_calls);
        if c.text.is_empty() {
            message["content"] = Value::Null;
        }
    }
    json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": if c.provider_model.is_empty() { "pankhllm".to_string() } else { c.provider_model.clone() },
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish
        }],
        "usage": {
            "prompt_tokens": c.usage.input_tokens,
            "completion_tokens": c.usage.output_tokens,
            "total_tokens": c.usage.input_tokens + c.usage.output_tokens
        },
        "pankhllm": {
            "model": c.model,
            "abstained": c.abstained,
            "clarification": c.clarification,
            "confidence": c.confidence,
            "cost_usd": c.usage.cost_usd,
            "cache": c.cache,
            "decision": c.decision,
            "attempts": c.attempts
        }
    })
}

fn budget_response(b: crate::router::BudgetDenied) -> Response {
    let mut resp = err(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", b.message);
    if let Ok(v) = axum::http::HeaderValue::from_str(&b.retry_after_secs.to_string()) {
        resp.headers_mut().insert("retry-after", v);
    }
    resp
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

async fn chat(State(router): State<AppState>, headers: HeaderMap, Json(body): Json<ChatRequest>) -> Response {
    let req = match to_route_request(body, &headers) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    if let Err(b) = router.admit_user(req.user.as_deref()) {
        return budget_response(b);
    }

    let needs_full_answer = req.tools.is_empty() && router.decide(&req).clarify;
    if req.stream && needs_full_answer {
        // A clarifying question is short and comes from the full-answer path;
        // clients still see a valid chunk stream.
        return match router.complete(&req).await {
            Ok(c) => {
                let model = if c.provider_model.is_empty() { "pankhllm".to_string() } else { c.provider_model.clone() };
                let created = now();
                let mk = |delta: Value, finish: Option<&str>, extra: Option<Value>| {
                    let mut v = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
                    if let Some(x) = extra { v["pankhllm"] = x; }
                    Ok::<_, Infallible>(Event::default().data(v.to_string()))
                };
                let meta = json!({"model": c.model, "decision": c.decision, "attempts": c.attempts, "cost_usd": c.usage.cost_usd});
                let mut events = vec![mk(json!({"role": "assistant", "content": ""}), None, Some(meta))];
                if !c.text.is_empty() {
                    events.push(mk(json!({"content": c.text}), None, None));
                }
                if !c.tool_calls.is_empty() {
                    events.push(mk(json!({"tool_calls": tool_calls_json(&c.tool_calls)}), None, None));
                }
                let finish = if c.tool_calls.is_empty() { "stop" } else { "tool_calls" };
                events.push(mk(json!({}), Some(finish), None));
                events.push(Ok(Event::default().data(json!({"object": "chat.completion.chunk", "choices": [],
                    "usage": {"prompt_tokens": c.usage.input_tokens, "completion_tokens": c.usage.output_tokens, "total_tokens": c.usage.input_tokens + c.usage.output_tokens}}).to_string())));
                events.push(Ok(Event::default().data("[DONE]")));
                Sse::new(futures::stream::iter(events)).keep_alive(KeepAlive::default()).into_response()
            }
            Err(e) => err(StatusCode::BAD_GATEWAY, "upstream_error", e.to_string()),
        };
    }

    if req.stream {
        return match router.stream(&req).await {
            Ok(sc) => {
                let model = if sc.provider_model.is_empty() { "pankhllm".to_string() } else { sc.provider_model.clone() };
                let meta = json!({"model": sc.model, "decision": sc.decision, "attempts": sc.attempts, "cache": sc.cache});
                let stream = sse_stream(id, model, meta, sc.events);
                Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
            }
            Err(e) => err(StatusCode::BAD_GATEWAY, "routing_error", e.to_string()),
        };
    }

    match router.complete(&req).await {
        Ok(c) => Json(openai_response(&id, &c)).into_response(),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("no model satisfies") || msg.contains("unknown prompt") {
                err(StatusCode::UNPROCESSABLE_ENTITY, "routing_error", msg)
            } else {
                err(StatusCode::BAD_GATEWAY, "upstream_error", msg)
            }
        }
    }
}

fn sse_stream(
    id: String,
    model: String,
    meta: Value,
    events: crate::providers::EventStream,
) -> impl Stream<Item = Result<Event, Infallible>> + Send {
    let created = now();
    let chunk = move |delta: Value, finish: Option<&str>, extra: Option<Value>| {
        let mut v = json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        });
        if let Some(x) = extra {
            v["pankhllm"] = x;
        }
        Event::default().data(v.to_string())
    };
    let head = futures::stream::once({
        let chunk = chunk.clone();
        async move { Ok(chunk(json!({"role": "assistant", "content": ""}), None, Some(meta))) }
    });
    let body = events.map(move |ev| {
        Ok(match ev {
            Ok(StreamEvent::Delta(t)) => chunk(json!({"content": t}), None, None),
            Ok(StreamEvent::ToolCallStart { index, id, name }) => chunk(
                json!({"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}),
                None,
                None,
            ),
            Ok(StreamEvent::ToolCallDelta { index, arguments }) => {
                chunk(json!({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]}), None, None)
            }
            Ok(StreamEvent::Done { input_tokens, output_tokens, stop_reason }) => {
                let finish = match stop_reason {
                    crate::providers::StopReason::ToolUse => "tool_calls",
                    crate::providers::StopReason::MaxTokens => "length",
                    crate::providers::StopReason::Refusal => "content_filter",
                    _ => "stop",
                };
                // finish chunk carries usage, OpenAI style
                Event::default().data(
                    json!({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                           "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}})
                        .to_string(),
                )
            }
            Err(e) => chunk(json!({}), Some("error"), Some(json!({"error": e.to_string()}))),
        })
    });
    let tail = futures::stream::once(async { Ok(Event::default().data("[DONE]")) });
    head.chain(body).chain(tail)
}

async fn route(State(router): State<AppState>, headers: HeaderMap, Json(body): Json<ChatRequest>) -> Response {
    match to_route_request(body, &headers) {
        Ok(req) => Json(router.decide(&req)).into_response(),
        Err(resp) => *resp,
    }
}

async fn models(State(router): State<AppState>) -> Json<Value> {
    let mut data: Vec<Value> = vec![json!({"id": "auto", "object": "model", "owned_by": "pankhllm"})];
    for t in Tier::ALL {
        data.push(json!({"id": format!("auto/{t}"), "object": "model", "owned_by": "pankhllm"}));
    }
    for m in &router.cfg.models {
        data.push(json!({"id": m.name, "object": "model", "owned_by": m.provider, "tier": m.tier, "tags": m.tags, "context_window": m.context_window}));
    }
    Json(json!({"object": "list", "data": data}))
}

async fn stats(State(router): State<AppState>) -> Json<Value> {
    Json(json!({"models": router.stats(), "cache": router.cache_stats(), "learned_routes": router.learned_stats(), "decision_models": router.native_summary(), "prompts": router.prompts.names()}))
}

/// What the router has learned: shapes, tools, hit counts. Read-only.
async fn learned_routes(State(router): State<AppState>) -> Json<Value> {
    Json(json!({"routes": router.routes.routes()}))
}

/// Offline training: ingest completed turns from application logs. Body: {"turns": [ObservedTurn, ...]}.
async fn learn_batch(State(router): State<AppState>, Json(body): Json<Value>) -> Response {
    let Some(turns) = body["turns"].as_array() else {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", "body must be {\"turns\": [...]}");
    };
    if turns.len() > 10_000 {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", "at most 10000 turns per call");
    }
    let mut learned = 0usize;
    let mut skipped = 0usize;
    let mut shapes: Vec<String> = Vec::new();
    for t in turns {
        match serde_json::from_value::<crate::router::ObservedTurn>(t.clone()) {
            Ok(turn) => match router.ingest_turn(&turn) {
                Some(shape) => {
                    learned += 1;
                    if shapes.len() < 50 && !shapes.contains(&shape) {
                        shapes.push(shape);
                    }
                }
                None => skipped += 1,
            },
            Err(_) => skipped += 1,
        }
    }
    let _ = router.persist();
    Json(json!({"learned": learned, "skipped": skipped, "shapes": shapes, "stats": router.learned_stats()})).into_response()
}

/// Recent traces from the store: `?since_ms=&limit=`. Behind inbound auth when configured.
async fn traces(State(router): State<AppState>, axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    let Some(st) = &router.store else {
        return err(StatusCode::NOT_FOUND, "not_configured", "no `store` configured");
    };
    st.flush();
    let since = q.get("since_ms").and_then(|v| v.parse().ok()).unwrap_or(crate::store::now_ms() - 3_600_000);
    let limit = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(200usize).min(5000);
    // Request traces by default; teacher labels and shadow comparisons with ?all=1.
    let all = q.get("all").is_some_and(|v| v == "1" || v == "true");
    match st.reader().and_then(|c| crate::store::read_traces(&c, since, limit)) {
        Ok(t) => {
            let t: Vec<_> = t.into_iter().filter(|t| all || !(t.surface.starts_with("label:") || t.surface.starts_with("shadow:"))).collect();
            Json(json!({"traces": t, "dropped": st.dropped.load(std::sync::atomic::Ordering::Relaxed)})).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, "store_error", e.to_string()),
    }
}

/// Reload pankhllm's own decision models after overnight training.
async fn decisions_reload(State(router): State<AppState>) -> Json<Value> {
    let (plan, tool) = router.reload_native();
    Json(json!({"reloaded": {"plan": plan, "tool": tool}, "models": router.native_summary()}))
}

async fn routes_export(State(router): State<AppState>) -> Json<Value> {
    Json(serde_json::to_value(router.routes.export()).unwrap_or(json!({"routes": []})))
}

async fn routes_import(State(router): State<AppState>, Json(body): Json<Value>) -> Response {
    match serde_json::from_value::<crate::learn::RouteSnapshot>(body) {
        Ok(snap) => {
            let n = router.routes.import(snap);
            let _ = router.persist();
            Json(json!({"imported": n, "stats": router.learned_stats()})).into_response()
        }
        Err(e) => err(StatusCode::BAD_REQUEST, "invalid_request_error", e.to_string()),
    }
}

async fn health(State(router): State<AppState>) -> Response {
    let stats = router.stats();
    let healthy = router.cfg.models.iter().filter(|m| m.routable).count();
    let body = json!({"status": "ok", "routable_models": healthy, "models_tracked": stats.len()});
    Json(body).into_response()
}

#[derive(Clone)]
struct Limits {
    in_flight: Arc<Semaphore>,
    timeout: std::time::Duration,
    api_keys: Arc<Vec<String>>,
}

fn presented_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
        .or_else(|| headers.get("api-key").and_then(|v| v.to_str().ok()))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .map(str::trim)
}

/// Constant-time comparison so key checks do not leak length or prefix timing.
fn key_matches(presented: &str, keys: &[String]) -> bool {
    keys.iter().any(|k| {
        let (a, b) = (presented.as_bytes(), k.as_bytes());
        let mut diff = a.len() ^ b.len();
        for i in 0..a.len().max(b.len()) {
            diff |= (*a.get(i).unwrap_or(&0) ^ *b.get(i).unwrap_or(&0)) as usize;
        }
        diff == 0
    })
}

/// Load shedding: when the server is full, answer 503 in microseconds instead
/// of queueing and inflating everyone's latency.
async fn shed(State(l): State<Limits>, req: Request, next: Next) -> Response {
    if !l.api_keys.is_empty() && req.uri().path() != "/health" {
        match presented_key(req.headers()) {
            Some(k) if key_matches(k, &l.api_keys) => {}
            _ => return err(StatusCode::UNAUTHORIZED, "authentication_error", "missing or invalid API key"),
        }
    }
    let Ok(_permit) = l.in_flight.clone().try_acquire_owned() else {
        return err(StatusCode::SERVICE_UNAVAILABLE, "overloaded", "server at max_in_flight; retry shortly");
    };
    match tokio::time::timeout(l.timeout, next.run(req)).await {
        Ok(resp) => resp,
        Err(_) => err(StatusCode::GATEWAY_TIMEOUT, "timeout", format!("request exceeded server.request_timeout_ms ({} ms)", l.timeout.as_millis())),
    }
}

/// CORS for the origins in `server.cors_origins`. Outermost, so a browser's preflight is
/// answered before authentication; the real request still needs its key.
async fn cors(State(allowed): State<Arc<Vec<String>>>, req: Request, next: Next) -> Response {
    let origin = req.headers().get(header::ORIGIN).and_then(|v| v.to_str().ok()).filter(|o| allowed.iter().any(|a| a == "*" || a == o)).map(str::to_string);
    let Some(origin) = origin else { return next.run(req).await };
    let Ok(origin_value) = HeaderValue::from_str(&origin) else { return next.run(req).await };
    if req.method() == Method::OPTIONS {
        let asked = req.headers().get(header::ACCESS_CONTROL_REQUEST_HEADERS).cloned().unwrap_or(HeaderValue::from_static("authorization, content-type"));
        let mut resp = StatusCode::NO_CONTENT.into_response();
        let h = resp.headers_mut();
        h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin_value);
        h.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
        h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, asked);
        h.insert(header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("600"));
        h.insert(header::VARY, HeaderValue::from_static("Origin"));
        return resp;
    }
    let mut resp = next.run(req).await;
    let h = resp.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin_value);
    h.insert(header::VARY, HeaderValue::from_static("Origin"));
    resp
}

pub fn app(router: Arc<Router>) -> AxumRouter {
    let sc = &router.cfg.server;
    let limits = Limits {
        in_flight: Arc::new(Semaphore::new(sc.max_in_flight.max(1))),
        timeout: std::time::Duration::from_millis(sc.request_timeout_ms.unwrap_or(router.cfg.routing.max_latency_ms + 5_000).max(1)),
        api_keys: Arc::new(sc.api_keys()),
    };
    let body_limit = sc.max_body_mb.max(1) * 1024 * 1024;
    let cors_origins = Arc::new(sc.cors_origins.clone());
    let app = AxumRouter::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/stats", get(stats))
        .route("/v1/routes", get(learned_routes))
        .route("/v1/routes/export", get(routes_export))
        .route("/v1/routes/import", post(routes_import))
        .route("/v1/learn", post(learn_batch))
        .route("/v1/traces", get(traces))
        .route("/v1/decisions/reload", post(decisions_reload))
        .route("/v1/route", post(route))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses))
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(middleware::from_fn_with_state(limits, shed))
        .with_state(router);
    if cors_origins.is_empty() {
        app
    } else {
        app.layer(middleware::from_fn_with_state(cors_origins, cors))
    }
}

// ---------------------------------------------------------------------------
// OpenAI Responses API surface: POST /v1/responses
// Stateless: `previous_response_id` is rejected; send the full `input` each time.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// `reasoning: {effort: "low"}` in the Responses shape.
    #[serde(default)]
    pub reasoning: Option<Value>,
    /// `text: {format: {...}}` in the Responses shape.
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub pankhllm: Option<PankhExt>,
}

/// Flatten Responses content (string, or parts of input_text / output_text) to text.
fn responses_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|p| matches!(p["type"].as_str(), Some("input_text") | Some("output_text") | Some("text") | None))
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Translate a Responses request into the chat shape, then reuse the chat path.
fn responses_to_chat(r: ResponsesRequest) -> Result<ChatRequest, Box<Response>> {
    if r.previous_response_id.is_some() {
        return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", "previous_response_id is not supported: the router is stateless, send the full input")));
    }
    let mut messages: Vec<Value> = Vec::new();
    if let Some(i) = r.instructions.filter(|s| !s.trim().is_empty()) {
        messages.push(json!({"role": "system", "content": i}));
    }
    match &r.input {
        Value::String(s) => messages.push(json!({"role": "user", "content": s})),
        Value::Array(items) => {
            for it in items {
                match it["type"].as_str() {
                    // Plain {role, content} items are messages without an explicit type.
                    Some("message") | None => {
                        let role = it["role"].as_str().unwrap_or("user");
                        messages.push(json!({"role": role, "content": responses_text(&it["content"])}));
                    }
                    Some("function_call") => {
                        let call = json!({"id": it["call_id"], "type": "function", "function": {"name": it["name"], "arguments": it["arguments"]}});
                        // Merge consecutive calls into one assistant turn, as chat expects.
                        match messages.last_mut() {
                            Some(last) if last["role"] == "assistant" && last["tool_calls"].is_array() => {
                                last["tool_calls"].as_array_mut().unwrap().push(call);
                            }
                            _ => messages.push(json!({"role": "assistant", "content": Value::Null, "tool_calls": [call]})),
                        }
                    }
                    Some("function_call_output") => {
                        let out = match &it["output"] { Value::String(s) => s.clone(), other => other.to_string() };
                        messages.push(json!({"role": "tool", "tool_call_id": it["call_id"], "content": out}));
                    }
                    Some(other) => return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", format!("unsupported input item type '{other}'")))),
                }
            }
        }
        _ => return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid_request_error", "input must be a string or an array of items"))),
    }
    // Responses tools are flat; chat tools nest under "function".
    let tools: Vec<Value> = r
        .tools
        .iter()
        .map(|t| if t.get("function").is_some() { t.clone() } else { json!({"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}) })
        .collect();
    let tool_choice = match r.tool_choice {
        Some(Value::Object(o)) if o.get("type").and_then(|t| t.as_str()) == Some("function") && o.get("function").is_none() => {
            Some(json!({"type": "function", "function": {"name": o.get("name").cloned().unwrap_or(Value::Null)}}))
        }
        other => other,
    };
    Ok(ChatRequest {
        model: r.model,
        messages,
        max_tokens: r.max_output_tokens,
        max_completion_tokens: None,
        temperature: r.temperature,
        stream: r.stream,
        tools,
        tool_choice,
        reasoning_effort: r.reasoning.as_ref().and_then(|v| v["effort"].as_str()).map(str::to_string),
        response_format: r.text.as_ref().and_then(|t| t.get("format")).and_then(|f| match f["type"].as_str() {
            Some("json_schema") => Some(json!({"type": "json_schema", "json_schema": {"name": f["name"].as_str().unwrap_or("output"), "schema": f["schema"], "strict": f["strict"].as_bool().unwrap_or(true)}})),
            Some("json_object") => Some(json!({"type": "json_object"})),
            Some("text") => Some(json!({"type": "text"})),
            _ => None,
        }),
        user: r.user,
        pankhllm: r.pankhllm,
    })
}

fn responses_output_items(id: &str, text: &str, calls: &[ToolCall]) -> Vec<Value> {
    let mut out = Vec::new();
    if !text.is_empty() {
        out.push(json!({"type": "message", "id": format!("msg_{id}"), "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}));
    }
    for (i, c) in calls.iter().enumerate() {
        out.push(json!({"type": "function_call", "id": format!("fc_{id}_{i}"), "call_id": c.id, "name": c.name, "arguments": c.arguments, "status": "completed"}));
    }
    out
}

fn responses_body(id: &str, c: &Completion) -> Value {
    let model = if c.provider_model.is_empty() { "pankhllm".to_string() } else { c.provider_model.clone() };
    json!({
        "id": id,
        "object": "response",
        "created_at": now(),
        "status": "completed",
        "model": model,
        "output": responses_output_items(id, &c.text, &c.tool_calls),
        "usage": {"input_tokens": c.usage.input_tokens, "output_tokens": c.usage.output_tokens, "total_tokens": c.usage.input_tokens + c.usage.output_tokens},
        "pankhllm": {"model": c.model, "abstained": c.abstained, "clarification": c.clarification, "confidence": c.confidence,
                     "cost_usd": c.usage.cost_usd, "cache": c.cache, "decision": c.decision, "attempts": c.attempts}
    })
}

async fn responses(State(router): State<AppState>, headers: HeaderMap, Json(body): Json<ResponsesRequest>) -> Response {
    let chat_req = match responses_to_chat(body) {
        Ok(c) => c,
        Err(resp) => return *resp,
    };
    let req = match to_route_request(chat_req, &headers) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };
    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    if let Err(resp) = router.admit_user(req.user.as_deref()) {
        return budget_response(resp);
    }

    let needs_full_answer = req.tools.is_empty() && router.decide(&req).clarify;
    if req.stream && !needs_full_answer {
        return match router.stream(&req).await {
            Ok(sc) => {
                let model = if sc.provider_model.is_empty() { "pankhllm".to_string() } else { sc.provider_model.clone() };
                let meta = json!({"model": sc.model, "decision": sc.decision, "attempts": sc.attempts, "cache": sc.cache});
                Sse::new(responses_sse(id, model, meta, sc.events)).keep_alive(KeepAlive::default()).into_response()
            }
            Err(e) => err(StatusCode::BAD_GATEWAY, "routing_error", e.to_string()),
        };
    }
    match router.complete(&req).await {
        Ok(c) if req.stream => {
            // Clarifying question over the streaming surface: replay it as events.
            let model = if c.provider_model.is_empty() { "pankhllm".to_string() } else { c.provider_model.clone() };
            let meta = json!({"model": c.model, "decision": c.decision, "attempts": c.attempts, "clarification": c.clarification});
            let text = c.text.clone();
            let usage = (c.usage.input_tokens, c.usage.output_tokens);
            let events = futures::stream::iter(vec![
                Ok::<_, crate::providers::ProviderError>(StreamEvent::Delta(text)),
                Ok(StreamEvent::Done { input_tokens: usage.0, output_tokens: usage.1, stop_reason: crate::providers::StopReason::EndTurn }),
            ])
            .boxed();
            Sse::new(responses_sse(id, model, meta, events)).keep_alive(KeepAlive::default()).into_response()
        }
        Ok(c) => Json(responses_body(&id, &c)).into_response(),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("no model satisfies") || msg.contains("unknown prompt") {
                err(StatusCode::UNPROCESSABLE_ENTITY, "routing_error", msg)
            } else {
                err(StatusCode::BAD_GATEWAY, "upstream_error", msg)
            }
        }
    }
}

/// Responses streaming: named events with a running sequence number.
fn responses_sse(id: String, model: String, meta: Value, events: crate::providers::EventStream) -> impl Stream<Item = Result<Event, Infallible>> + Send {
    struct St {
        id: String,
        model: String,
        seq: u64,
        text: String,
        text_open: bool,
        calls: Vec<(String, String, String)>, // (call_id, name, arguments so far)
        open_call: Option<usize>,
        meta: Value,
    }

    fn ev(seq: &mut u64, kind: &str, mut body: Value) -> Event {
        *seq += 1;
        body["type"] = json!(kind);
        body["sequence_number"] = json!(*seq);
        Event::default().event(kind).data(body.to_string())
    }
    fn close_text(st: &mut St, out: &mut Vec<Event>) {
        if st.text_open {
            let msg_id = format!("msg_{}", st.id);
            let text = st.text.clone();
            out.push(ev(&mut st.seq, "response.output_text.done", json!({"item_id": msg_id, "output_index": 0, "content_index": 0, "text": text})));
            let item = json!({"type": "message", "id": msg_id, "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}]});
            out.push(ev(&mut st.seq, "response.output_item.done", json!({"output_index": 0, "item": item})));
            st.text_open = false;
        }
    }
    fn close_call(st: &mut St, out: &mut Vec<Event>) {
        if let Some(i) = st.open_call.take() {
            let (call_id, name, args) = st.calls[i].clone();
            let idx = i + usize::from(!st.text.is_empty());
            let fc_id = format!("fc_{}_{i}", st.id);
            out.push(ev(&mut st.seq, "response.function_call_arguments.done", json!({"item_id": fc_id, "output_index": idx, "arguments": args})));
            let item = json!({"type": "function_call", "id": fc_id, "call_id": call_id, "name": name, "arguments": args, "status": "completed"});
            out.push(ev(&mut st.seq, "response.output_item.done", json!({"output_index": idx, "item": item})));
        }
    }

    let mut seq = 0u64;
    let created = json!({"response": {"id": id, "object": "response", "created_at": now(), "status": "in_progress", "model": model, "output": [], "pankhllm": meta}});
    let e1 = ev(&mut seq, "response.created", created);
    let e2 = ev(&mut seq, "response.in_progress", json!({"response": {"id": id, "object": "response", "status": "in_progress"}}));
    let head = futures::stream::iter(vec![Ok::<_, Infallible>(e1), Ok(e2)]);

    let mut st = St { id, model, seq, text: String::new(), text_open: false, calls: Vec::new(), open_call: None, meta };
    let body = events.flat_map(move |item| {
        let mut out: Vec<Event> = Vec::new();
        match item {
            Ok(StreamEvent::Delta(t)) => {
                let msg_id = format!("msg_{}", st.id);
                if !st.text_open {
                    close_call(&mut st, &mut out);
                    let item = json!({"type": "message", "id": msg_id, "status": "in_progress", "role": "assistant", "content": []});
                    out.push(ev(&mut st.seq, "response.output_item.added", json!({"output_index": 0, "item": item})));
                    out.push(ev(&mut st.seq, "response.content_part.added", json!({"item_id": msg_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})));
                    st.text_open = true;
                }
                st.text.push_str(&t);
                out.push(ev(&mut st.seq, "response.output_text.delta", json!({"item_id": msg_id, "output_index": 0, "content_index": 0, "delta": t})));
            }
            Ok(StreamEvent::ToolCallStart { index, id, name }) => {
                close_text(&mut st, &mut out);
                close_call(&mut st, &mut out);
                while st.calls.len() <= index {
                    st.calls.push((String::new(), String::new(), String::new()));
                }
                st.calls[index] = (id.clone(), name.clone(), String::new());
                st.open_call = Some(index);
                let idx = index + usize::from(!st.text.is_empty());
                let fc_id = format!("fc_{}_{index}", st.id);
                let item = json!({"type": "function_call", "id": fc_id, "call_id": id, "name": name, "arguments": "", "status": "in_progress"});
                out.push(ev(&mut st.seq, "response.output_item.added", json!({"output_index": idx, "item": item})));
            }
            Ok(StreamEvent::ToolCallDelta { index, arguments }) => {
                if let Some(c) = st.calls.get_mut(index) {
                    c.2.push_str(&arguments);
                }
                let idx = index + usize::from(!st.text.is_empty());
                let fc_id = format!("fc_{}_{index}", st.id);
                out.push(ev(&mut st.seq, "response.function_call_arguments.delta", json!({"item_id": fc_id, "output_index": idx, "delta": arguments})));
            }
            Ok(StreamEvent::Done { input_tokens, output_tokens, .. }) => {
                close_text(&mut st, &mut out);
                close_call(&mut st, &mut out);
                let calls: Vec<ToolCall> = st.calls.iter().filter(|c| !c.1.is_empty()).map(|c| ToolCall { id: c.0.clone(), name: c.1.clone(), arguments: c.2.clone() }).collect();
                let output = responses_output_items(&st.id, &st.text, &calls);
                let resp = json!({"id": st.id, "object": "response", "created_at": now(), "status": "completed", "model": st.model, "output": output,
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}, "pankhllm": st.meta});
                out.push(ev(&mut st.seq, "response.completed", json!({"response": resp})));
            }
            Err(e) => {
                let failed = json!({"response": {"id": st.id, "object": "response", "status": "failed", "error": {"code": "upstream_error", "message": e.to_string()}}});
                out.push(ev(&mut st.seq, "response.failed", failed));
            }
        }
        futures::stream::iter(out.into_iter().map(Ok::<_, Infallible>))
    });
    head.chain(body)
}
