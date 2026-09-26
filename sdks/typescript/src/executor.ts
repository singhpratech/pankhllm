/**
 * Planner-lane executor for JavaScript and TypeScript. No dependencies.
 *
 * The router validates a plan against your catalog, then POSTs it:
 *   { op, params, question, user, tenant, trace_id }
 * Return { ok: true, data } | { ok: true, answer } | { ok: true, data: { rows } } | { ok: false, error }.
 *
 * `createExecutor(ops)` gives a fetch-style handler for Node 18+, Deno, Bun, Cloudflare Workers,
 * Next.js route handlers and Hono. `serveExecutor(ops, port)` starts a plain node:http server.
 */

export interface PlanContext { question?: string; user?: string | null; tenant?: string | null; trace_id?: string }
export type Op = (params: Record<string, unknown>, ctx: PlanContext) => unknown | Promise<unknown>;
export interface PlanPayload extends PlanContext { op: string; params?: Record<string, unknown> }
export type ExecResult =
  | { ok: true; data?: unknown; answer?: string }
  | { ok: false; error: string };

/** Run one plan. Never throws: failures become { ok: false } so the router falls through. */
export async function handlePlan(payload: PlanPayload, ops: Record<string, Op>): Promise<ExecResult> {
  const fn = ops[payload?.op];
  if (typeof fn !== "function") return { ok: false, error: `unknown operation ${JSON.stringify(payload?.op)}` };
  try {
    const out = await fn({ ...(payload.params ?? {}) }, { question: payload.question, user: payload.user, tenant: payload.tenant, trace_id: payload.trace_id });
    if (typeof out === "string") return { ok: true, answer: out };
    if (Array.isArray(out)) return { ok: true, data: { rows: out } };
    if (out && typeof out === "object" && ("data" in out || "answer" in out)) return { ok: true, ...(out as object) } as ExecResult;
    return { ok: true, data: out };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? `${e.name}: ${e.message}` : String(e) };
  }
}

/** Fetch-style handler: (Request) => Promise<Response>. */
export function createExecutor(ops: Record<string, Op>, opts: { apiKey?: string } = {}) {
  return async (req: Request): Promise<Response> => {
    const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
    if (req.method !== "POST") return json(405, { ok: false, error: "POST only" });
    if (opts.apiKey && req.headers.get("authorization") !== `Bearer ${opts.apiKey}`) return json(401, { ok: false, error: "unauthorized" });
    let payload: PlanPayload;
    try { payload = await req.json(); } catch { return json(400, { ok: false, error: "bad json" }); }
    return json(200, await handlePlan(payload, ops));
  };
}

/** Standalone node:http server on `port`. Resolves once listening. */
export async function serveExecutor(ops: Record<string, Op>, port = 7000, opts: { apiKey?: string; host?: string } = {}) {
  const http = await import("node:http");
  const handler = createExecutor(ops, opts);
  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const request = new Request(`http://localhost${req.url}`, { method: req.method, headers: req.headers as Record<string, string>, body: req.method === "POST" ? Buffer.concat(chunks) : undefined });
    const response = await handler(request);
    res.writeHead(response.status, { "content-type": "application/json" });
    res.end(await response.text());
  });
  await new Promise<void>((resolve) => server.listen(port, opts.host ?? "127.0.0.1", resolve));
  return server;
}
