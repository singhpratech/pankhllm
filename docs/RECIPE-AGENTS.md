# Recipe: sub-second answers for a tool-using agent

Target: an assistant where each turn is several model hops: load instructions, pick a tool and
write its arguments, run the tool, then write the answer from the result. Each hop is a multi-second
large-model call, so a turn can take 10 to 20 seconds, and the floor with two inferences is still several seconds.
Sub-second is not a tuning problem. It is a "do not call the large model" problem, and that is what
the router is for. The recipe applies to any domain: reporting, support, search, operations.

Config: `examples/configs/agent-azure.yaml`.

## The three hops, and how the router removes them without touching the framework

The framework keeps its loop. What changes is who answers each hop.

| Hop | Today | With the router |
|---|---|---|
| load instructions | LLM call, seconds | Gone: skills live in the router and are auto-selected by triggers; the model never asks for them |
| tool selection (choose a tool, write arguments) | LLM call, seconds | First sight of a shape: mini at low effort, 1 to 2 s. Every later question of that shape, new entities or paraphrased: learned route, ~0 ms |
| synthesis after the result | LLM call, seconds | Speculative template rendered by the router, ~0 ms; model only when the planner flagged "needs analysis" |

So a turn whose shape has been seen once completes with zero model calls: routing plus your query time. The first sight of a new shape costs one cheap-model call, and the router learns it for everyone in that tenant scope.

## Where the time goes and what removes it

| Turn shape | Before | With the router | How |
|---|---|---|---|
| Same shape, new entities or paraphrase | a full turn | query time only | learned route fills the tool call; speculative template renders the answer |
| Same question again (any user, same data version) | a full turn | ~1 ms | exact cache hit; `cache_key` = data refresh id |
| Paraphrase of a known question | a full turn | ~50-150 ms | semantic cache hit: one embedding call, no chat model |
| Pre-resolved route (regex matches, tool offered) | 5-6 s for the tool-select hop | ~0 ms for that hop | `respond_with_tool` rule emits the tool call itself |
| Novel question, tool-select hop | 5-6 s on the big model | 1-2 s on mini at low effort | `agent_loop.tool_select: fast` |
| Synthesis after the tool result | seconds | shorter, streamed | `after_tool_result: balanced`, streaming, shorter `max_tokens` |
| Mini stalls | waits for its timeout | big starts after 1.2 s | `hedge.after_ms` |

The first two rows are where "under one second" comes from. In a reporting workload most
questions repeat across users and days with the same data version, so the cache hit rate is the
number that decides your p50. Measure it on `/v1/stats` (`cache.exact_hits`, `semantic_hits`, `misses`).

## Turn-by-turn

1. **Client sends the turn** to `/v1/responses` (or chat completions) with `prompt: "agent,skill"`,
   the tools, `X-Pankh-Cache-Key: <data-version>`, and for a pre-resolved route
   `X-Pankh-Tier: fast` plus `X-Pankh-Effort: low`.
2. **Rule shortcut.** If a `respond_with_tool` rule matches and the tool is offered, the router
   returns the function call with captured arguments. No model. The app executes the query.
3. **Cache.** Otherwise the exact key (prompt, tools, tags, cache_key, full message history) is
   checked, then, for single-turn questions, the semantic index within the same scope.
4. **Model.** On a miss the tool-select hop goes to mini; the hop after the tool result goes to
   big, streamed. A confident final answer is written through to the cache, including streamed ones.
5. **Invalidate** by changing `cache_key` on the weekly refresh. Old entries expire by TTL.

## Inline the skill

If the framework loads instructions through a tool call, that is a whole extra LLM hop per turn. Put the skill in `prompts/` and reference it by name instead; the router prepends it as a
byte-stable system block on every call, so the turn becomes tool hop then synthesis.

## What stays on the big model

Novel analytical questions and the synthesis of fresh results. For those the levers are effort
(`medium` not `high`), a tighter `max_tokens`, streaming so the user sees the first token in about
a second, and the hedge so a slow start never blocks.

## Train it overnight, before anyone asks

The first sight of a shape costs a model call. Do not pay it in production. Export completed
turns from your logs as JSONL, one per line:

```json
{"question": "NRx for Zelora in East for 2026-03",
 "tool": "exec_template", "tool_names": ["exec_template"],
 "arguments": "{\"template\":\"nrx_by_product_region_month\",\"product\":\"Zelora\",\"region\":\"East\",\"month\":\"2026-03\"}",
 "tool_result": "{\"product\":\"Zelora\",\"nrx\":1842,\"ref\":\"tbl_1\"}",
 "answer": "Zelora wrote 1,842 NRx in East. <x-table id=\"tbl_1\"></x-table>",
 "cache_key": "v38|bu-a"}
```

Then:

```bash
pankhllm --config pankhllm.yaml warm turns.jsonl        # ingest: no model calls
pankhllm --config pankhllm.yaml warm questions.jsonl --replay --concurrency 8   # question-only lines: run the planner now, not at 9am
```

Ingestion abstracts each question into a shape, generalises the arguments, and derives an answer
template by replacing result values that appear in the logged answer with placeholders. Error
results and code-argument tools are skipped. Put product, region and segment names in
`learned_routes.vocab` (or `vocab_file`) so they become slots whatever their casing; dates, codes
and numbers are typed automatically. Set `persist_path`: the memory is saved every minute and on
shutdown, loaded at start, and can be copied between instances. `/v1/learn` does the same over HTTP
for a nightly job, and `/v1/routes` shows what is known.

What the trained state contains, and what it never contains: shapes, argument templates with
slot placeholders, and answer templates whose literals have been replaced by placeholders. A
template that still carries a code, date, number or a vocabulary name after placeholder removal is
rejected at ingestion and when the planning model proposes one, so a name or figure from one
user's answer is never persisted or replayed to another. The response cache does hold full
answers, in memory only, partitioned by `cache_key`. Treat `persist_path` and `/v1/routes/export`
with the same access controls as your conversation store anyway; `/v1/learn`, `/v1/routes/*` are
behind inbound auth when `server.api_keys_env` is set.

If your historical logs only hold free-text code calls, they cannot warm anything; start warm-up
with `--replay` of the historical questions once the structured tool exists.

Concurrency: identical requests in flight at the same time share one model call. Different
entities of a known shape never reach a model. Paraphrases of a known shape match semantically
when an embedding model is configured. Only genuinely new shapes, or questions the planner flagged
as needing analysis, cost a model call.

## Multi-tenant safety: the cache key is not optional

In a multi-tenant data assistant the same question means different things per tenant: a business-unit
selector changes the data sources, and row-level security changes the rows each user may see. A question-only cache that ignored that would hand one manager's scoped rows to another.

Hard requirement: with `scope: question_only`, put **data version, business unit and the user's
row-level scope** into `cache_key`, for example `v38|bu-a|region-east`. The router enforces the
minimum: with `require_cache_key: true` (the default) it will neither read nor write the cache for a
question-only request that carries no key, and logs a warning. Semantic hits are likewise confined
to the same key. The key is opaque to the router; getting its contents right is the application's job.

## Honest expectations

- The first sight of a new question shape costs one planning-model call (mini, low effort, 1 to
  2 s) plus your query. From the second sight on, both hops are router-served. Sub-second p50 needs
  most traffic to be shapes already seen, which is what reporting traffic looks like; watch
  `learned_routes.served` versus `learned` on `/v1/stats`.
- Speculative templates are written before the data exists. They present data; they do not
  interpret it. The planner sets `needs_analysis` for questions that need judgement and those go
  to the synthesis model as before. If a result does not fit the template, the model answers.
- Learned routes are conservative by default: a shape is learned only from a turn with exactly one
  tool call whose result came back clean, and it is served only after `min_observations` (default 2)
  such turns. Retries and error turns never teach shapes.
- Give the router a structured tool to learn. A tool whose only argument is a program or a SQL
  string (`exec_code(code)`) is never learned: filling entities into code text is string
  interpolation of user input. Expose `exec_template(id, params)` backed by parameterized queries and
  keep the free-text tool for novel questions; the router learns the structured one.
- If your UI renders tables from a cached reference rather than inline rows, let the tool return
  that reference and put it in the template: `<x-table id="{{ref}}"></x-table> {{total|N0}} rows,
  share {{share|P1}}`.

- p50 under a second requires a cache hit rate above 50 percent, or most turns resolving by rule.
  Both are properties of your traffic; measure before promising.
- Semantic hits reuse an answer computed for a paraphrase. With `scope: question_only` that is
  correct only if the data version is in `cache_key` and the agent context is stable. Keep
  `scope: exact_context` for retrieval pipelines where chunks vary.
- The cache is in memory per instance. Behind a load balancer use sticky routing or accept a
  per-instance hit rate. A shared store is on the roadmap.
