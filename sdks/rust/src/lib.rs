//! Client for the pankhllm router. Async, reqwest-based, no router internals.

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Chunk {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub messages: Option<Vec<Message>>,
    pub context: Vec<Chunk>,
    pub tags: Vec<String>,
    pub prompt: Option<String>,
    pub tier: Option<String>,
    pub intent: Option<String>,
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub max_cost_usd: Option<f64>,
    pub max_latency_ms: Option<u64>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    pub text: String,
    pub model: String,
    pub provider_model: String,
    pub cost_usd: f64,
    pub abstained: bool,
    pub clarification: bool,
    pub confidence: f64,
    pub decision: Value,
    pub attempts: Value,
    pub raw: Value,
}

#[derive(Debug)]
pub enum Error {
    Api { status: u16, kind: String, message: String },
    Transport(reqwest::Error),
    Malformed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Api { status, kind, message } => write!(f, "{status} {kind}: {message}"),
            Error::Transport(e) => write!(f, "transport: {e}"),
            Error::Malformed(m) => write!(f, "malformed response: {m}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Transport(e)
    }
}

pub enum StreamEvent {
    Meta(Value),
    Delta(String),
}

pub struct Client {
    http: reqwest::Client,
    base: String,
}

impl Client {
    pub fn new(base_url: &str) -> Self {
        Self { http: reqwest::Client::new(), base: base_url.trim_end_matches('/').to_string() }
    }

    fn body(question: Option<&str>, o: &Options, stream: bool) -> Value {
        let mut ext = json!({});
        if !o.context.is_empty() { ext["context"] = json!(o.context); }
        if !o.tags.is_empty() { ext["tags"] = json!(o.tags); }
        if let Some(p) = &o.prompt { ext["prompt"] = json!(p); }
        if let Some(t) = &o.tier { ext["tier"] = json!(t); }
        if let Some(i) = &o.intent { ext["intent"] = json!(i); }
        if let Some(c) = o.max_cost_usd { ext["max_cost_usd"] = json!(c); }
        if let Some(l) = o.max_latency_ms { ext["max_latency_ms"] = json!(l); }
        let messages = o.messages.clone().unwrap_or_else(|| vec![Message { role: "user".into(), content: question.unwrap_or("").into() }]);
        let mut b = json!({"model": o.model.clone().unwrap_or_else(|| "auto".into()), "messages": messages, "stream": stream, "pankhllm": ext});
        if let Some(m) = o.max_tokens { b["max_tokens"] = json!(m); }
        if let Some(t) = o.temperature { b["temperature"] = json!(t); }
        b
    }

    async fn post(&self, path: &str, body: &Value) -> Result<reqwest::Response, Error> {
        let resp = self.http.post(format!("{}{path}", self.base)).json(body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            let v: Value = serde_json::from_str(&text).unwrap_or(json!({}));
            return Err(Error::Api {
                status,
                kind: v["error"]["type"].as_str().unwrap_or("error").into(),
                message: v["error"]["message"].as_str().unwrap_or(&text).into(),
            });
        }
        Ok(resp)
    }

    pub async fn ask(&self, question: &str, o: &Options) -> Result<Answer, Error> {
        let v: Value = self.post("/v1/chat/completions", &Self::body(Some(question), o, false)).await?.json().await?;
        let p = &v["pankhllm"];
        Ok(Answer {
            text: v["choices"][0]["message"]["content"].as_str().unwrap_or("").into(),
            model: p["model"].as_str().unwrap_or("").into(),
            provider_model: v["model"].as_str().unwrap_or("").into(),
            cost_usd: p["cost_usd"].as_f64().unwrap_or(0.0),
            abstained: p["abstained"].as_bool().unwrap_or(false),
            clarification: p["clarification"].as_bool().unwrap_or(false),
            confidence: p["confidence"]["score"].as_f64().unwrap_or(0.0),
            decision: p["decision"].clone(),
            attempts: p["attempts"].clone(),
            raw: v.clone(),
        })
    }

    /// Text deltas as they arrive; the first item is the routing metadata.
    pub async fn stream(&self, question: &str, o: &Options) -> Result<impl Stream<Item = Result<StreamEvent, Error>>, Error> {
        let resp = self.post("/v1/chat/completions", &Self::body(Some(question), o, true)).await?;
        let mut buf = String::new();
        let mut meta_sent = false;
        Ok(resp.bytes_stream().flat_map(move |chunk| {
            let mut out = Vec::new();
            match chunk {
                Err(e) => out.push(Err(Error::Transport(e))),
                Ok(b) => {
                    buf.push_str(&String::from_utf8_lossy(&b));
                    while let Some(pos) = buf.find("\n\n") {
                        let block = buf[..pos].to_string();
                        buf.drain(..pos + 2);
                        for line in block.lines() {
                            let Some(data) = line.strip_prefix("data:") else { continue };
                            let data = data.trim();
                            if data == "[DONE]" { continue; }
                            let Ok(ev) = serde_json::from_str::<Value>(data) else { continue };
                            if !meta_sent && ev.get("pankhllm").is_some() {
                                meta_sent = true;
                                out.push(Ok(StreamEvent::Meta(ev["pankhllm"].clone())));
                            }
                            if ev["choices"][0]["finish_reason"] == "error" {
                                out.push(Err(Error::Malformed(ev["pankhllm"]["error"].to_string())));
                            }
                            if let Some(t) = ev["choices"][0]["delta"]["content"].as_str() {
                                if !t.is_empty() { out.push(Ok(StreamEvent::Delta(t.to_string()))); }
                            }
                        }
                    }
                }
            }
            futures::stream::iter(out)
        }))
    }

    /// Dry run: the routing decision, no model call.
    pub async fn route(&self, question: &str, o: &Options) -> Result<Value, Error> {
        Ok(self.post("/v1/route", &Self::body(Some(question), o, false)).await?.json().await?)
    }
}
