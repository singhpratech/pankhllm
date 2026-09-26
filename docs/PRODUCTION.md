# Production notes: latency and failure modes

Everything here came out of an adversarial pass over the request path with one question: what adds
milliseconds, and what falls over. Each gap lists what was found, what changed, and the test that pins it.

## Router overhead

Routing is a pure function: no I/O, no allocation beyond small strings. Measured with the offline eval
over 10,000 questions and with the HTTP benchmark (`cargo test --release --test adversarial bench_ --
--ignored --nocapture`). The benchmark runs the load generator, an in-process mock upstream and the
router in one process on a laptop, so it measures router plus HTTP overhead, not model time.

| Measurement | Value |
|---|---|
| Routing decision, mean (release build) | 2.9 us |
| Routing decision, max (release build) | 197 us |
| HTTP round trip incl. mock upstream, c=64, p50 / p90 / p99 | 0.86 / 1.20 / 7.15 ms |
| Throughput at c=64 | ~62,000 requests/s |
| HTTP round trip, c=256, p50 / p99 | 4.2 / 50.7 ms |

Model calls dominate every real request by three to four orders of magnitude. The router's job is to
keep those calls few, short and parallel.

## Gaps found and fixed

| # | Gap | Effect before | Fix | Test |
|---|---|---|---|---|
| 1 | Verifier ran serially after every answer | Every answer paid a second model round trip | `confidence.verify_below` (default 0.85): confident answers skip it; `verifier_timeout_ms` (default 3000) caps it; a verifier in cooldown is skipped | `confident_answers_skip_the_verifier`, `slow_verifier_is_cut_by_its_own_timeout` |
| 2 | Slow-but-alive primary blocked the cascade until its timeout | Tail latency equal to the slowest model's timeout | `routing.hedge.after_ms`: start the next candidate in parallel, first good answer wins, loser cancelled | `hedged_race_beats_a_slow_primary`, `hedge_falls_through_when_both_fail` |
| 3 | No load shedding | Overload queued requests; latency inflated for everyone | `server.max_in_flight` (default 512): immediate 503 when full | `oversized_body_gets_413_and_overload_gets_503` |
| 4 | No HTTP-layer timeout | A stuck handler held the connection forever | `server.request_timeout_ms` (default `max_latency_ms` + 5 s): 504 | `server_request_timeout_returns_504` |
| 5 | Body limit was axum's 2 MB default | Large retrieved contexts got 413 | `server.max_body_mb` (default 32) | same as 3 |
| 6 | Request-budget timeouts cooled models down | An impatient caller took a healthy model out for 30 s | Only a model's own timeout marks it unhealthy | `request_budget_timeout_does_not_cool_down_the_model` |
| 7 | `Mutex::lock().unwrap()` | One panic while holding a lock poisoned it; every later request panicked | Poison-tolerant lock helper | covered by all tests |
| 8 | 10 s connect timeout | Unreachable provider cost 10 s before cascade | `providers.<name>.connect_timeout_secs` (default 3) | config |
| 9 | HTTP/1.1 only to providers | One connection per in-flight request | reqwest `http2` feature; HTTP/2 negotiated where the provider supports it | build |
| 10 | Tool result read as "the question" after a tool turn | Wrong intent, wrong tier on synthesis turns | `RouteRequest::query` skips tool results | `agent_loop_phases_pick_tiers` |
| 11 | Ambiguity and weak retrieval escalated twice | Cheap questions landed on the top tier | Uncertainty signals escalate at most one tier | `confidence_can_be_disabled_or_set_to_escalate` |
| 12 | Verifier model was itself a routing candidate | The judge answered user questions | `routable: false` | `verifier_low_score_escalates_and_is_reported` |

## Latency knobs, in the order to tune them

1. **Per-model `timeout_secs`.** Set it to the p99 you are willing to wait for that model, not its worst case.
2. **`routing.max_latency_ms`.** The whole cascade must fit. Requests can lower it per call.
3. **`max_concurrency` on the GPU model.** When it is full the router moves on instead of queueing.
4. **`routing.hedge.after_ms`.** Start at roughly the fast model's p90. Costs duplicate tokens on the slow tail only.
5. **`confidence.verify_below`.** Lower it to verify fewer answers; set `verifier` to a fast local model.
6. **`server.max_in_flight`.** Size it to what the slowest tier can absorb; shedding early is cheaper than timing out late.
7. **Streaming.** Time to first token is what users feel. The first-token fallback keeps dead models from blocking it.

## Streaming

- Tool calls stream token by token in OpenAI's delta format (`delta.tool_calls[].function.arguments`
  fragments), translated from Anthropic `tool_use` / `input_json_delta` blocks as well.
- The hedge applies to streams too: it races on time-to-first-token. If the primary has not produced
  its first event by `hedge.after_ms`, the next candidate's stream is opened and whichever yields
  first is committed; the other is dropped, which cancels its request.
- After the first token the stream is committed to one model. Hedge-phrase and confidence checks need
  the full text, so they do not apply to streamed answers.

## Package size

| Artifact | Size |
|---|---|
| Release binary (LTO, stripped, Linux x86-64) | 10.6 MB |
| Same, gzip-compressed | 4.8 MB |
| pip wheel `pankhllm-server` | 5.6 MB |
| Resident memory at idle | ~10 MB |

The binary embeds SQLite, TLS and the HTTP stack; it has no runtime dependencies.

## Still open

- The response cache and learned routes are per instance; the store persists them, but instances do not share them live.
- No persistent stats or dashboard; `/v1/stats` is in-memory and resets on restart.
- Token estimates use 3.6 characters per token; a real tokenizer would tighten window and cost decisions.
- No per-tenant budgets or rate limits; put the router behind your gateway for that.
- The Anthropic and Azure adapters are mock-tested; run one real request against each before relying on them.
- Inbound auth is static API keys only; put OAuth or mTLS at your gateway if you need identity.
