/** pankhllm client. Uses fetch only, so it runs in Node 18+, Deno, Bun and browsers. */

export interface Chunk { text: string; score?: number; source?: string }
export type Tier = "fast" | "balanced" | "reasoning";
export type Intent = "lookup" | "extraction" | "summarization" | "synthesis" | "reasoning" | "multi_hop" | "code" | "chitchat";
export interface ChatMessage { role: "system" | "user" | "assistant"; content: string }

export interface AskOptions {
  messages?: ChatMessage[];
  context?: Chunk[];
  tags?: string[];
  prompt?: string;
  tier?: Tier;
  intent?: Intent;
  model?: string;
  maxTokens?: number;
  maxCostUsd?: number;
  maxLatencyMs?: number;
  temperature?: number;
  /** Caller identity: per-user budgets on the router, row-level scope in executors. */
  user?: string;
  /** Tenant / data-version partition for caching and learned routes. */
  cacheKey?: string;
  /** OpenAI-style structured output. */
  responseFormat?: Record<string, unknown>;
  signal?: AbortSignal;
}

export interface Answer {
  text: string;
  model: string;
  providerModel: string;
  costUsd: number;
  decision: Record<string, unknown>;
  attempts: unknown[];
  abstained: boolean;
  raw: Record<string, unknown>;
}

export class PankhError extends Error {
  constructor(public status: number, message: string, public kind = "error") {
    super(`${status} ${kind}: ${message}`);
  }
}

export class PankhClient {
  private base: string;
  constructor(baseUrl = "http://localhost:4000", private headers: Record<string, string> = {}) {
    this.base = baseUrl.replace(/\/+$/, "");
  }

  private body(question: string | undefined, o: AskOptions, stream: boolean) {
    const ext: Record<string, unknown> = {};
    if (o.context?.length) ext.context = o.context;
    if (o.tags?.length) ext.tags = o.tags;
    if (o.prompt) ext.prompt = o.prompt;
    if (o.tier) ext.tier = o.tier;
    if (o.intent) ext.intent = o.intent;
    if (o.maxCostUsd !== undefined) ext.max_cost_usd = o.maxCostUsd;
    if (o.maxLatencyMs !== undefined) ext.max_latency_ms = o.maxLatencyMs;
    if (o.user) ext.user = o.user;
    if (o.cacheKey) ext.cache_key = o.cacheKey;
    const b: Record<string, unknown> = {
      model: o.model ?? "auto",
      messages: o.messages ?? [{ role: "user", content: question ?? "" }],
      stream,
      pankhllm: ext,
    };
    if (o.maxTokens !== undefined) b.max_tokens = o.maxTokens;
    if (o.temperature !== undefined) b.temperature = o.temperature;
    if (o.responseFormat) b.response_format = o.responseFormat;
    return b;
  }

  private async post(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
    const resp = await fetch(`${this.base}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", ...this.headers },
      body: JSON.stringify(body),
      signal,
    });
    if (!resp.ok) {
      let kind = "error", message = await resp.text();
      try { const e = JSON.parse(message).error; kind = e.type ?? kind; message = e.message ?? message; } catch { /* plain text */ }
      throw new PankhError(resp.status, message, kind);
    }
    return resp;
  }

  async ask(question?: string, o: AskOptions = {}): Promise<Answer> {
    const v = await (await this.post("/v1/chat/completions", this.body(question, o, false), o.signal)).json();
    const p = v.pankhllm ?? {};
    return {
      text: v.choices?.[0]?.message?.content ?? "", model: p.model ?? "", providerModel: v.model ?? "",
      costUsd: p.cost_usd ?? 0, decision: p.decision ?? {}, attempts: p.attempts ?? [], abstained: !!p.abstained, raw: v,
    };
  }

  /** Async iterator of text deltas. The first yielded item is the routing metadata. */
  async *stream(question?: string, o: AskOptions = {}): AsyncGenerator<{ meta?: Record<string, unknown>; delta?: string }> {
    const resp = await this.post("/v1/chat/completions", this.body(question, o, true), o.signal);
    const reader = resp.body!.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let metaSent = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buf += dec.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") return;
          const ev = JSON.parse(data);
          if (!metaSent && ev.pankhllm) { metaSent = true; yield { meta: ev.pankhllm }; }
          for (const ch of ev.choices ?? []) {
            if (ch.finish_reason === "error") throw new PankhError(502, String(ev.pankhllm?.error ?? "stream error"), "upstream_error");
            if (ch.delta?.content) yield { delta: ch.delta.content };
          }
        }
      }
    }
  }

  /** Dry run: the routing decision, no model call. */
  async route(question?: string, o: AskOptions = {}): Promise<Record<string, unknown>> {
    return (await this.post("/v1/route", this.body(question, o, false), o.signal)).json();
  }

  async stats(): Promise<Record<string, unknown>> {
    const r = await fetch(`${this.base}/v1/stats`, { headers: this.headers });
    if (!r.ok) throw new PankhError(r.status, await r.text());
    return r.json();
  }
}

export { createExecutor, serveExecutor, handlePlan } from "./executor.js";
export type { Op, PlanContext, PlanPayload, ExecResult } from "./executor.js";

export default PankhClient;
