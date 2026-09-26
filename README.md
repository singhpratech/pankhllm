# pankhllm

[![ci](https://github.com/singhpratech/pankhllm/actions/workflows/ci.yml/badge.svg)](https://github.com/singhpratech/pankhllm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg?logo=rust)](Cargo.toml)
[![OpenAI-compatible](https://img.shields.io/badge/API-OpenAI--compatible-412991.svg)](#call-it-from-anything)
[![Decision](https://img.shields.io/badge/decision-0.2%20ms%20on%20CPU-brightgreen.svg)](docs/BENCHMARK-DECISIONS.md)
[![Model size](https://img.shields.io/badge/model-262%20KB-brightgreen.svg)](docs/TRAINING.md#resource-needs)
[![No GPU](https://img.shields.io/badge/GPU-not%20required-brightgreen.svg)](docs/TRAINING.md#resource-needs)
[![Train in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/singhpratech/pankhllm/blob/main/notebooks/train_pankhllm.ipynb)
[![Agent skill](https://img.shields.io/badge/skill-Claude%20Code%20%7C%20Codex-8A2BE2.svg)](.claude/skills/pankhllm-train/SKILL.md)

[![crates.io](https://img.shields.io/static/v1?label=crates.io&message=cargo%20install%20pankhllm&color=e6822a&logo=rust&logoColor=white)](#install)
[![PyPI](https://img.shields.io/static/v1?label=PyPI&message=pip%20install%20pankhllm&color=3776ab&logo=python&logoColor=white)](#install)
[![npm](https://img.shields.io/static/v1?label=npm&message=npx%20pankhllm%20serve&color=cb3837&logo=npm&logoColor=white)](#install)
[![NuGet](https://img.shields.io/static/v1?label=NuGet&message=dotnet%20add%20package%20Pankhllm&color=004880&logo=nuget&logoColor=white)](#install)
[![Go](https://img.shields.io/static/v1?label=Go&message=go%20get%20github.com%2Fsinghpratech%2Fpankhllm%2Fsdks%2Fgo&color=00add8&logo=go&logoColor=white)](#install)
[![Docker](https://img.shields.io/static/v1?label=Docker&message=docker%20build%20-t%20pankhllm%20.&color=2496ed&logo=docker&logoColor=white)](#install)
<!-- After publishing, swap in live version badges:
[![crates.io](https://img.shields.io/crates/v/pankhllm?logo=rust)](https://crates.io/crates/pankhllm)
[![PyPI](https://img.shields.io/pypi/v/pankhllm?logo=python&logoColor=white)](https://pypi.org/project/pankhllm/)
[![npm](https://img.shields.io/npm/v/pankhllm?logo=npm)](https://www.npmjs.com/package/pankhllm)
[![NuGet](https://img.shields.io/nuget/v/Pankhllm?logo=nuget)](https://www.nuget.org/packages/Pankhllm)
[![Go Reference](https://pkg.go.dev/badge/github.com/singhpratech/pankhllm/sdks/go.svg)](https://pkg.go.dev/github.com/singhpratech/pankhllm/sdks/go)
-->

### The LLM gateway that learns to skip the LLM.

Most of your agent's LLM calls aren't writing anything. They're decisions: which tool, which
skill, which report, which parameters. Your app pays a large model seconds and tokens to make
the same decision thousands of times a day.

pankhllm sits where your app already calls an LLM. It watches those decisions, trains its own
tiny model on them, and starts making them itself in **0.2 ms on a CPU**. It makes no LLM call
for those questions. What it isn't sure about still goes to your LLM, exactly as before, and
that becomes tomorrow's training data. One Rust binary. OpenAI-compatible. Change one base URL.

| Measured | |
|---|---|
| Same held-out questions, median answer | **1,743 ms → 4 ms**, 14 of 14 correct |
| One decision, in-process on a CPU | **0.2 ms** |
| 2,000 unseen pharma KPI questions | **63% answered with no LLM call, 0 wrong**; the rest go to the LLM |
| Model size, all decision models | **262 KB** |
| Train on 20,000 questions | **~1 second** on one CPU core |
| Server | **11 MB** binary, **17 MB** RAM idle, **1,750 req/s** on a laptop |

Every number comes from a run in this repository. How we measured, including where it's
weaker (96.5% precision on phrasing written by a different model), is in
[docs/BENCHMARK-DECISIONS.md](docs/BENCHMARK-DECISIONS.md) and [docs/JOURNAL.md](docs/JOURNAL.md).

**Drop it in.** Nothing in your agent changes:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="...")   # was: your LLM provider
```

The same one-line change works in LangChain, Semantic Kernel, Microsoft Agent Framework, the
Vercel AI SDK, and any OpenAI client in Python, TypeScript, JavaScript, C#, Go, Java, Rust, Ruby, PHP, C or C++.

**Teach it your domain in one command.** Give it your skill files, your prompts and your logs.
Pick the teacher: Claude, OpenAI, Azure OpenAI / AI Foundry, or a local Ollama model. You get
back a ready-to-run bundle:

```sh
python3 notebooks/run_trainer.py --skills ./skills --prompts agent.yaml --logs app.jsonl --teacher ollama
```

Or open [the notebook](notebooks/train_pankhllm.ipynb) in Colab. In Claude Code or OpenAI
Codex, say "train pankhllm on these files" and the bundled `pankhllm-train` skill runs it for you.

**What it isn't.** It isn't an LLM, and it doesn't replace yours. Explanations, judgement and
open-ended writing still go to your model, with faster routing, hedging, caching and budgets
around them. pankhllm removes the calls that never needed a model.

## Why it exists

Every agent turns one question into several model hops: load instructions, choose a tool, write its arguments, read the result, write the answer. Each hop is a large-model call of seconds, and every hop resends the growing conversation. Picking a cheaper model helps a little. Not calling a model at all is what changes the numbers. pankhllm decides, per request, the cheapest path that can still be proven correct:

| Lane | What answers | Typical latency | Model calls |
|---|---|---|---|
| Cache | an identical question in the same tenant and data version | ~1 ms | 0 |
| Learned route | a question whose shape was seen before, with new entities or phrasing | ~1 ms + your tool | 0 |
| Rule | a pattern you pinned | ~1 ms + your tool | 0 |
| Decision | a new question or tool-selection turn pankhllm's own model can classify, with arguments the question states | a ~0.2 ms in-process decision + your executor or tool | 0 |
| Planner | a new question that maps to an operation in your catalog | one small-model call + your executor | 1 small |
| Agent | anything else: explanation, judgement, open-ended work | as fast as the best eligible model, hedged and streamed | as needed |

Every lane fails open to the next one, so adding a lane can only help. Measured on a laptop, 14 labelled questions, Python executor: a local 12B generative planner answers first asks at a 1,743 ms median; pankhllm's own decision model, trained overnight from that planner's decisions, answers the same held-out questions at a 4 ms median, 14 of 14 correct, deciding in under 0.3 ms. See `docs/BENCHMARK-DECISIONS.md`.

## How it learns

- **Online.** Tool calls the model makes are generalised into typed shapes (codes, dates, numbers, names from your vocabulary) once their results come back clean, twice. Plans the planner makes are learned the same way. A route whose calls start returning errors is quarantined on the spot.
- **Overnight.** `pankhllm mine` reads the trace database, learns from clean turns the live path missed, quarantines failing routes, measures every lane and model, proposes vocabulary and timeouts, and writes a report. `pankhllm warm` trains from your historical logs before go-live.
- **Safely.** Only flat typed arguments are ever learned, never free-text code or query strings. Templates that still carry literal data are rejected. Everything is scoped by tenant key. Question text can be redacted from the store.

## What it routes on

LiteLLM routes by model name and load. pankhllm routes on what an application knows and a generic gateway does not:

| Signal | Effect |
|---|---|
| Intent of the question (lookup, extraction, summary, synthesis, reasoning, multi-hop, code) | Picks a tier: `fast`, `balanced`, `reasoning` |
| Agent-loop phase (choosing a tool vs reading its result) | Tool selection goes to the fast tier, synthesis to the balanced tier |
| Retrieval quality (top chunk score) | Weak retrieval escalates a tier, or abstains |
| Context size | Drops models whose window is too small |
| Tags on the request (`private`, `eu`, ...) | Only models carrying every tag are eligible |
| Cost and latency budgets, per request and per user | Drops, cuts or sheds before they are exceeded |
| Answer quality | Refusals, empty, hedged and low-confidence answers escalate |

Every response carries the decision, the reasons, and every attempt, so you can see why a call went where it went.

Tool calling, structured output (`response_format`, `text.format`) and streaming pass through on both the chat-completions and Responses surfaces, translated for Anthropic and for upstreams that only speak the Responses API. Agent frameworks run their loops through the router without changes.

## Install

| Ecosystem | Command | What you get |
|---|---|---|
| Rust | `cargo install pankhllm` | the router |
| npm | `npx pankhllm serve` | the router, prebuilt per platform |
| npm | `npm install @pankhllm/client` | TypeScript and JavaScript client and executor |
| pip | `pip install pankhllm-server pankhllm` | the router, plus the Python client and executor |
| NuGet | `Pankhllm` / `Pankhllm.Server` | .NET client and executor; the router binary with `PankhllmServer.Start()` |
| Go | `go get github.com/singhpratech/pankhllm/sdks/go` | Go client and executor |
| Docker | `docker build -t pankhllm .` | the router |

Packaging sources are in `packaging/`. Nothing is published yet.

## Run

```bash
cargo build --release
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
./target/release/pankhllm check                     # validate config, list models and prompts
./target/release/pankhllm route "Why did the migration fail?" --scores 0.2 --tags private
./target/release/pankhllm serve --port 4000
```

Edit `pankhllm.yaml` to match your models. Prices are per million tokens; check them against your vendor.

## Call it from anything

The server speaks the OpenAI chat-completions wire format. Set `model` to `auto` (or `auto/fast`, `auto/reasoning`, or a configured model name).

```bash
curl localhost:4000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "auto",
  "messages": [{"role": "user", "content": "What is the refund window?"}],
  "pankhllm": {
    "context": [{"text": "Refunds within 30 days ...", "score": 0.91, "source": "policy.md"}],
    "tags": ["private"],
    "max_latency_ms": 8000,
    "max_cost_usd": 0.02,
    "prompt": "support-agent"
  }
}'
```

Clients that cannot add body fields (Semantic Kernel, LangChain, Spring AI) send the same hints as headers:

| Header | Meaning |
|---|---|
| `X-Pankh-Tags: private,eu` | Required model tags |
| `X-Pankh-Prompt: support-agent` | Server-held prompt to prepend |
| `X-Pankh-Tier: fast` | Force a tier |
| `X-Pankh-Intent: lookup` | Force an intent |
| `X-Pankh-Max-Cost-Usd: 0.02` | Cost ceiling |
| `X-Pankh-Max-Latency-Ms: 8000` | Wall-clock budget |

Endpoints:

- `POST /v1/chat/completions` answer (set `"stream": true` for SSE)
- `POST /v1/responses` the OpenAI Responses API shape: string or item input, `instructions`, flat function tools, `function_call` / `function_call_output` items, named streaming events. Stateless, so `previous_response_id` is rejected.
- `POST /v1/route` decision only, no model call
- `GET /v1/models`, `GET /v1/stats`, `GET /health`

## Clients

Every client is thin: the server does the work. Use the official OpenAI client of your language if you prefer; these add the `pankhllm` extras and typed results.

| Language | Path | Notes |
|---|---|---|
| Python | `sdks/python` | httpx |
| TypeScript / JavaScript | `sdks/typescript` | fetch, works in Node 18+, Deno, Bun, browsers |
| Go | `sdks/go` | net/http only |
| Rust | `sdks/rust` | reqwest, async |
| Java | `sdks/java` | JDK 17 HTTP client, no dependencies |
| C# / .NET | `sdks/csharp` | System.Net.Http; Semantic Kernel and SignalR example in `examples/csharp` |
| Ruby | `sdks/ruby` | net/http |
| PHP | `sdks/php` | ext-curl |
| C | `sdks/c` | libcurl |
| C++ | `sdks/cpp` | header-only over libcurl |
| Shell | `sdks/curl/pankh.sh` | curl, optional jq |

## Your own vocabulary

Domain rules run before the built-in classifier and pin phrases to an intent or tier:

```yaml
routing:
  rules:
    - { name: kpi-lookup, any: ["call activity", "fte", "trx", "nrx"], intent: lookup, tier: fast }
    - { name: alerts, any: ["alert", "anomal"], tier: reasoning }
```

Set `intent` on a rule as well as `tier`: the intent picks the grounding instruction. Lookup, extraction and summarization get `routing.grounding_strict` (answer only from context, say "I don't know" otherwise, and a hedge escalates). Synthesis, reasoning, multi-hop and code get `routing.grounding_open` (context first, reason beyond it, label what is unsupported). Both strings are configurable.

Rules are checked in order and the first match wins, so list the rules that must win (alerts, escalations) above broad KPI rules. The matched rule is reported in `decision.signals.matched_rule`.

## Providers

| `kind` | Notes |
|---|---|
| `anthropic` | Messages API. Adaptive thinking on by default (off for Haiku), optional `effort`, stable system block cache-marked. |
| `openai` | OpenAI and anything speaking its API: Ollama, vLLM, LM Studio, Groq, Together, OpenRouter. |
| `azure` | Azure OpenAI. `base_url` is the resource endpoint, `model` is the deployment name, `api-key` header. Set `api_version` for the classic deployments path or omit it for `/openai/v1`. |

## Sub-second paths

Model calls take seconds; the router's job is to avoid them where it can. Nothing below needs a change in the agent framework: it keeps making its hops, the router just answers most of them itself.

- **Learned routes.** The router watches the planning model's tool calls, abstracts the entities in the question into typed slots (codes, dates, numbers, names) and stores the shape with the argument template. The next question of the same shape, with different entities or paraphrased, gets the tool call filled in by the router with no model call. Guardrails: a shape is learned only from a turn that ended with one tool call and a clean result, after `min_observations` (default 2) such turns; only flat typed arguments are learned, never free-text `code` / `sql` / `query` arguments, long strings, or anything with statement separators, so slot-filling can never become code injection. Routes are scoped by tenant key, prompt and tool set. `GET /v1/routes` shows what was learned.
- **Train from logs before go-live.** `pankhllm warm log.jsonl` ingests completed turns from your application logs (question, tool call, result, final answer) straight into the route memory with no model calls, deriving answer templates from the logged answers. Routes persist to `learned_routes.persist_path`, survive restarts, and can be exported and imported between instances (`/v1/routes/export`, `/v1/routes/import`, `/v1/learn`). Product, region and segment names go in `learned_routes.vocab` so they are recognised as slots whatever their casing. After an overnight run, the first live question of any shape in your history is already router-served.
- **Identical concurrent requests cost one model call.** Followers wait for the leader's answer and read it from the cache.
- **Speculative answer templates.** On the tool-selection hop the model is also asked, through a hidden tool, for a template of the final answer shaped for the tool's JSON result. When the result comes back the router renders the answer itself. Templates can carry references (`<x-table id="{{ref}}">`) and .NET-style number formats (`{{calls|N0}}`, `{{share|P1}}`). The synthesis model only runs when the model flagged the question as needing interpretation or the template does not fit the data. With learned routes, a whole two-hop turn completes with zero model calls.

- **Response cache.** Exact hits return in about a millisecond with no model call. With an embedding model configured, paraphrases hit too (one embedding call, no chat model). Only confident, non-hedged answers and tool calls are stored. Streams are written through. Partition by `cache_key` (data version, tenant, business unit, row-level scope) and bump it to invalidate; with `scope: question_only` a request without a key is not cached at all. `X-Pankh-Cache: bypass | refresh` per request.
- **Rule-to-tool shortcuts.** A rule with `pattern` and `respond_with_tool` answers a tool-selection turn with the tool call itself, arguments filled from named captures. No model call for pre-resolved routes.
- **Phase routing, hedging, streaming** for everything else.

```yaml
cache:
  enabled: true
  ttl_secs: 604800
  scope: question_only            # or exact_context for retrieval pipelines
  semantic: { embedding_model: embed, threshold: 0.95 }
routing:
  learned_routes: { enabled: true, min_similarity: 0.93 }
  speculative_answer: { enabled: true }
  rules:
    - name: kpi-direct
      pattern: '(?i)call activity for (?P<territory>T-\d{3}) for (?P<month>\d{4}-\d{2})'
      respond_with_tool: { name: get_call_activity, arguments: { territory: "{{territory}}", month: "{{month}}" } }
```

A worked setup for a tool-using agent is in `docs/RECIPE-AGENTS.md` with the config in `examples/configs/agent-azure.yaml`.

## Confidence and clarification

Every text answer gets a confidence score from retrieval strength, hedging, citations and question clarity. Below `routing.confidence.threshold` the router escalates to a stronger candidate if one remains. Add a `verifier` model (a fast or local one, marked `routable: false`) and every answer is graded for groundedness and completeness before it is accepted:

```yaml
routing:
  confidence:
    threshold: 0.55
    verifier: judge          # optional; grades grounded / complete / needs_clarification
    on_ambiguous: clarify    # clarify | escalate | ignore
```

A question that is too vague to answer well ("what about west?", "numbers?") is not guessed at. With `clarify`, the cheapest model writes one clarifying question and the response carries `pankhllm.clarification: true`. The score and its parts are returned in `pankhllm.confidence`.

## Skills, agents and tool loops

**Skill files carry their own routing.** A `pankhllm:` block in the markdown front matter, or at the top level of an agent YAML, declares tier, tags, budgets or a pinned model. Requests that reference the skill inherit them; request values override, tags merge.

```markdown
---
pankhllm:
  tier: fast
  tags: [private]
  max_latency_ms: 5000
---
# Field KPI skill
Answer from the retrieved KPI extracts only ...
```

**Agent loops have phases.** When a request carries tools, picking the tool is routed to `agent_loop.tool_select` (default fast) and the turn after tool results to `agent_loop.after_tool_result` (default balanced). A cheap model that answers a hard question directly instead of calling a tool gets a confidence penalty, so it escalates rather than bluffs.

## Skills that grow without growing the prompt

Put skill markdown, agent YAML and design docs in `prompts/`. Reference them by file stem with `prompt: "support-agent,design"`, or turn on `prompts.auto_select` and give each skill `triggers` in its front matter: the router then sends the `always` skills plus only the skills whose triggers match the question, capped by `max_auto`. Fifty skill files cost the same per request as three. When no trigger matches, a skill-routing model picks the skill. It is seeded by `examples:` in each skill's front matter and learns from every request that names its skill explicitly; the nightly miner retrains it. The stable block is byte-identical across calls and is sent with a `prompt_cache_key` (OpenAI, Azure) or `cache_control` (Anthropic), so provider-side prompt caching hits every time. On turns served by learned routes, rules or the cache, no prompt is sent at all.

## Latency controls

- `timeout_secs` per model: a model that blows it is skipped and cooled down.
- `max_concurrency` per model: when the GPU box is full, the router moves on instead of queueing.
- `max_latency_ms` per request: the whole cascade must fit.
- `hedge.after_ms`: if the first model has not answered by then, the next one starts in parallel and the first good answer wins.
- `confidence.verify_below` and `verifier_timeout_ms`: the verifier only runs on borderline answers and never for long.
- `server.max_in_flight`, `server.request_timeout_ms`, `server.max_body_mb`: shed load with an immediate 503, cap the wall clock at the HTTP layer, and accept large contexts.
- Streaming falls back, and hedges, until the first token arrives, then commits. Tool calls stream token by token.

Routing itself takes about 15 microseconds; the HTTP layer adds under a millisecond at p50. Numbers and the full gap list are in `docs/PRODUCTION.md`.

## Decisions: pankhllm teaches itself

Most of an agent's model calls are decisions, not writing: which tool, which operation, is this answerable. pankhllm learns to make those decisions itself.

1. **The generative planner is the teacher.** Every plan it makes that executes cleanly, and every question it declines, is recorded as a label: the question's shape and the decision.
2. **The miner trains pankhllm's own model overnight.** A hashed-feature logistic regression over the question's shape, class-balanced, with a held-out split that sets the confidence threshold for a target precision. It trains in well under a second and is stored in the trace database. Catalog `examples` seed it before any traffic.
3. **At request time it decides in-process, in microseconds,** choosing only among the operations the question structurally fits plus `UNSUPPORTED`. Then the slot filler fills parameters deterministically: enum values the question names, codes and dates matching a pattern, numbers in range, the one remaining entity. Two candidates for one slot, or a slot with none, sends the request to the planner instead of guessing. A confident `UNSUPPORTED` skips the planner call only when no operation fits the question structurally. Wording the model has rarely seen (`min_known_words`) always goes to the planner. So does a question where the model, free to choose among all operations, prefers one the question does not fit.

```yaml
decisions:
  engine: native            # pankhllm's own model (default)
  min_confidence: 0.85      # for "nothing fits"; calibrate with shadow mode
  consensus_confidence: 0.55  # floor when the structure already agrees; the miner may raise it
  mode: act                 # or shadow, to compare against the planner first
  planner: true
  tools: false              # opt in: answer tool-selection turns the same way
```

On a pharma KPI catalog with seven operations, trained on 20,000 generated questions plus 183 teacher-labelled ones, it decided 63% of 2,000 held-out questions on its own with none wrong, in 0.2 ms each. On held-out questions written by a model it decided 61% at 96.5% precision. Everything else went to the planner as before. Details: [docs/BENCHMARK-DECISIONS.md](docs/BENCHMARK-DECISIONS.md).

**Train it on your own data** with [notebooks/train_pankhllm.ipynb](notebooks/train_pankhllm.ipynb) (Google Colab or Jupyter). It loads your skills and logs, trains, evaluates, and exports a ready-to-run bundle. The trained models are a 262 KB file (`pankhllm export-models`, `decisions.models_file`). Training 20,000 questions takes about a second on one CPU core, and the server runs in under 32 MB with no GPU. Pick Claude, OpenAI, Azure OpenAI / AI Foundry or a local Ollama model as the teacher. The steps and file formats are in [docs/TRAINING.md](docs/TRAINING.md). Which public datasets are legal to use is in [docs/DATASETS.md](docs/DATASETS.md).

Tool-selection turns use the same loop: the agent model is the teacher, clean tool calls and direct answers are the labels, and each tool's JSON schema is the slot specification. Any `POST /v1/systemone` service can be plugged in instead with `engine: external`.

## The planner lane

The model reads; your executor runs; the router checks. Declare a catalog of operations with typed parameters:

```yaml
planner:
  enabled: true
  model: small                      # a fast model, routable: false
  executor: { url: "http://127.0.0.1:7000/execute", timeout_ms: 2000 }
  operations:
    - name: metric_by_period
      description: One metric for one entity in one month.
      params:
        metric: { type: enum, values: [calls, revenue, tickets] }
        entity: { type: string, max_len: 40 }
        period: { type: string, pattern: '\d{4}-\d{2}' }
      answer_template: "{{params.entity}} {{params.metric}} in {{params.period}}: {{value|N0}}"
```

For a question in an eligible intent, one structured-output call returns a plan or `UNSUPPORTED`. The router rejects any plan with an unknown operation or parameter, a wrong type, a value outside an enum, pattern or range, or control syntax in a string. A valid plan is POSTed to your executor with the user's identity for row-level scope; the executor returns data and the router renders the answer. Your executor is one HTTP endpoint in any language; helpers exist for Python (`pankhllm.executor`), TypeScript and JavaScript (`createExecutor`, `serveExecutor`), C# (`MapPankhExecutor`) and Go (`Executor`). The router never builds a query itself.

## Logging and the miner

```yaml
store: { path: ./state/pankhllm.db, redact_questions: false, retention_days: 30 }
heal: { quarantine_after_failures: 2, autotune: true }
```

Every request is written to an embedded SQLite database off the hot path: served-by lane, model, shape, intent, tier, phase, tool, latency, tokens, cost, confidence and outcome. User ids are stored hashed; code-like tool arguments are never stored. `GET /v1/traces` reads recent traces. Run the miner nightly:

```bash
pankhllm mine --since-hours 24 --apply --report reports/nightly.md
```

## Testing and evaluation

```bash
cargo test                                                        # 99 tests, no network
pankhllm --config evals/pharma.yaml eval evals/pharma-10k.jsonl  # routing accuracy, offline
python3 evals/domain_adversarial.py http://localhost:4125         # domain scenarios, real models
```

See `docs/EVALUATION.md` for what each layer checks and the latest results.

## Layout

```
src/router.rs       lanes, decision, cascade, hedging, budgets, tracing
src/planner.rs      planner lane: catalog, schema, validation, executor, rendering
src/learn.rs        learned routes: typed shapes, fill, feedback, quarantine
src/cache.rs        exact and semantic response cache
src/store.rs        embedded trace database with a non-blocking writer
src/miner.rs        overnight miner and autotune
src/classifier.rs   intent heuristics
src/signals.rs      tokens, retrieval quality, ambiguity, agent-loop phase
src/prompts.rs      skill registry with front matter and auto-selection
src/providers/      anthropic, openai (chat or Responses; Azure; Ollama, vLLM, ...), SSE
src/server.rs       chat-completions and Responses surfaces, auth, shedding, per-user budgets
```

Inbound auth: set `server.api_keys_env` and clients send `Authorization: Bearer`, `api-key` or `x-api-key`. Deployment on Linux, Windows Server and Docker is in `docs/DEPLOY.md`.

## Prior art, and what is ours

pankhllm does not ship a model. It orchestrates other people's models, and says so:

- **Typed, non-autoregressive decisions** ("System 1") come from [Jev](https://www.langchain.com/blog/building-a-harness-with-jev) (TypeSafe AI) and [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations). pankhllm calls them over their `/v1/systemone` interface as a pluggable engine, exactly as it calls OpenAI or Anthropic for generation. It works without one.
- **Routing across providers** is the category [LiteLLM](https://github.com/BerriAI/litellm) defined; the OpenAI wire format is OpenAI's.

What pankhllm adds is the layer none of those provide: deciding per request which lane can *prove* an answer, and learning to prove more of them over time.

- **Structural constraint of decisions.** Before a decision model votes, the slot filler works out which operations the question can satisfy exactly and the model only chooses among those. On our benchmark this is what took Laya from 8 of 14 zero-shot to 8 of 10 answerable questions accepted with no wrong answers.
- **Learned routes and speculative answer templates.** Tool calls and plans generalised into typed shapes from clean turns, so a seen shape needs no model at all, decision model included.
- **A caller-owned executor contract** that keeps data access, row-level scope and query building in the application, in any language.
- **Tenant-scoped caching, shadow calibration, quarantine and the overnight miner**, which turn traffic into thresholds, vocabulary and fine-tuning data for whichever decision model you use.

## Roadmap

- Shared state (cache, learned routes) across instances behind a load balancer
- Racing the planner against the agent for questions near the planner's edge
- A review page for the miner's proposals
