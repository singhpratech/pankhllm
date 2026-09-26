"""Planner-lane executor for Python. Standard library only.

The router validates a plan against your catalog, then POSTs it here:
    {"op": "...", "params": {...}, "question": "...", "user": "...", "tenant": "...", "trace_id": "..."}
You run it (parameterized query, API call, search...) with the user's identity and return
    {"ok": true, "data": {...}}            # rendered by the operation's answer_template
    {"ok": true, "answer": "..."}          # or your own final text
    {"ok": true, "data": {"rows": [...]}}  # or rows, rendered as a table
    {"ok": false, "error": "..."}          # the router falls through to the agent

Use it inside any framework via `handle()`, or standalone via `serve()`:

    from pankhllm.executor import serve

    def metric_by_period(params, ctx):
        rows = db.execute("SELECT value FROM kpi WHERE metric=? AND entity=? AND period=?",
                          (params["metric"], params["entity"], params["period"]), user=ctx["user"])
        return {"value": rows[0][0]}

    serve({"metric_by_period": metric_by_period}, port=7000)
"""
from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Mapping, Optional

Op = Callable[[Dict[str, Any], Dict[str, Any]], Any]


def handle(payload: Mapping[str, Any], ops: Mapping[str, Op]) -> Dict[str, Any]:
    """Run one plan. Never raises: errors become {"ok": false} so the router falls through."""
    name = payload.get("op")
    fn = ops.get(name) if isinstance(name, str) else None
    if fn is None:
        return {"ok": False, "error": f"unknown operation {name!r}"}
    ctx = {k: payload.get(k) for k in ("question", "user", "tenant", "trace_id")}
    try:
        out = fn(dict(payload.get("params") or {}), ctx)
    except Exception as e:  # noqa: BLE001 - the router must see a clean failure, not a 500 page
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if isinstance(out, str):
        return {"ok": True, "answer": out}
    if isinstance(out, list):
        return {"ok": True, "data": {"rows": out}}
    if isinstance(out, dict) and ("data" in out or "answer" in out or "ok" in out):
        return {"ok": True, **out}
    return {"ok": True, "data": out}


def serve(ops: Mapping[str, Op], host: str = "127.0.0.1", port: int = 7000, path: str = "/execute",
          api_key: Optional[str] = None) -> None:
    """Blocking HTTP server. `api_key` (or env PANKH_EXECUTOR_KEY) requires a matching bearer token."""
    key = api_key or os.environ.get("PANKH_EXECUTOR_KEY")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: Dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):  # noqa: N802
            if self.path.split("?")[0] != path:
                return self._send(404, {"ok": False, "error": "not found"})
            if key:
                got = self.headers.get("authorization", "")
                if not hmac.compare_digest(got, f"Bearer {key}"):
                    return self._send(401, {"ok": False, "error": "unauthorized"})
            try:
                n = int(self.headers.get("content-length") or 0)
                payload = json.loads(self.rfile.read(min(n, 1_000_000)) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"ok": False, "error": "bad json"})
            self._send(200, handle(payload, ops))

    ThreadingHTTPServer((host, port), H).serve_forever()


__all__ = ["handle", "serve"]
