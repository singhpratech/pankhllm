"""pankhllm client. The server is OpenAI-compatible, so the official `openai`
package also works (set base_url); this client adds the routing extras."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

try:  # the client needs httpx; the executor (pankhllm.executor) is stdlib-only and must import without it
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


@dataclass
class Chunk:
    text: str
    score: Optional[float] = None
    source: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"text": self.text}
        if self.score is not None:
            d["score"] = self.score
        if self.source is not None:
            d["source"] = self.source
        return d


@dataclass
class Answer:
    text: str
    model: str
    provider_model: str
    cost_usd: float
    decision: dict
    attempts: list
    abstained: bool
    raw: dict = field(repr=False, default_factory=dict)


class PankhError(RuntimeError):
    def __init__(self, status: int, message: str, kind: str = "error"):
        super().__init__(f"{status} {kind}: {message}")
        self.status, self.kind, self.message = status, kind, message


class Client:
    def __init__(self, base_url: str = "http://localhost:4000", timeout: float = 120.0, headers: Optional[dict] = None):
        if httpx is None:
            raise ImportError("pankhllm.Client needs httpx: pip install httpx")
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers or {})

    def _body(self, question, messages, context, tags, prompt, tier, intent, max_tokens, max_cost_usd, max_latency_ms, model, stream, temperature,
              user=None, cache_key=None, response_format=None):
        msgs = messages or [{"role": "user", "content": question}]
        ext: dict[str, Any] = {}
        if context:
            ext["context"] = [c.to_dict() if isinstance(c, Chunk) else c for c in context]
        if tags:
            ext["tags"] = list(tags)
        for k, v in (("prompt", prompt), ("tier", tier), ("intent", intent), ("max_cost_usd", max_cost_usd), ("max_latency_ms", max_latency_ms),
                     ("user", user), ("cache_key", cache_key)):
            if v is not None:
                ext[k] = v
        body: dict[str, Any] = {"model": model or "auto", "messages": msgs, "stream": stream, "pankhllm": ext}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if response_format is not None:
            body["response_format"] = response_format
        return body

    @staticmethod
    def _raise(resp) -> None:
        if resp.status_code >= 400:
            try:
                e = resp.json()["error"]
                raise PankhError(resp.status_code, e.get("message", ""), e.get("type", "error"))
            except (ValueError, KeyError):
                raise PankhError(resp.status_code, resp.text)

    def ask(self, question: Optional[str] = None, *, messages: Optional[list] = None, context: Optional[list] = None,
            tags: Optional[list] = None, prompt: Optional[str] = None, tier: Optional[str] = None, intent: Optional[str] = None,
            max_tokens: Optional[int] = None, max_cost_usd: Optional[float] = None, max_latency_ms: Optional[int] = None,
            model: Optional[str] = None, temperature: Optional[float] = None, user: Optional[str] = None,
            cache_key: Optional[str] = None, response_format: Optional[dict] = None) -> Answer:
        body = self._body(question, messages, context, tags, prompt, tier, intent, max_tokens, max_cost_usd, max_latency_ms, model, False, temperature,
                          user, cache_key, response_format)
        resp = self._http.post("/v1/chat/completions", json=body)
        self._raise(resp)
        v = resp.json()
        p = v.get("pankhllm", {})
        return Answer(
            text=v["choices"][0]["message"]["content"], model=p.get("model", ""), provider_model=v.get("model", ""),
            cost_usd=p.get("cost_usd", 0.0), decision=p.get("decision", {}), attempts=p.get("attempts", []),
            abstained=p.get("abstained", False), raw=v,
        )

    def stream(self, question: Optional[str] = None, **kw) -> Iterator[str]:
        """Yields text deltas. Routing metadata is available on `.last_meta` after the first chunk."""
        body = self._body(question, kw.get("messages"), kw.get("context"), kw.get("tags"), kw.get("prompt"), kw.get("tier"), kw.get("intent"),
                          kw.get("max_tokens"), kw.get("max_cost_usd"), kw.get("max_latency_ms"), kw.get("model"), True, kw.get("temperature"))
        self.last_meta: dict = {}
        with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code >= 400:
                resp.read()
                self._raise(resp)
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                ev = json.loads(data)
                if "pankhllm" in ev and not self.last_meta:
                    self.last_meta = ev["pankhllm"]
                for ch in ev.get("choices", []):
                    if ch.get("finish_reason") == "error":
                        raise PankhError(502, str(ev.get("pankhllm", {}).get("error", "stream error")), "upstream_error")
                    delta = ch.get("delta", {}).get("content")
                    if delta:
                        yield delta

    def route(self, question: Optional[str] = None, **kw) -> dict:
        """Dry run: the routing decision, no model call."""
        body = self._body(question, kw.get("messages"), kw.get("context"), kw.get("tags"), kw.get("prompt"), kw.get("tier"), kw.get("intent"),
                          kw.get("max_tokens"), kw.get("max_cost_usd"), kw.get("max_latency_ms"), kw.get("model"), False, None)
        resp = self._http.post("/v1/route", json=body)
        self._raise(resp)
        return resp.json()

    def stats(self) -> dict:
        resp = self._http.get("/v1/stats")
        self._raise(resp)
        return resp.json()


__all__ = ["Client", "Chunk", "Answer", "PankhError"]
