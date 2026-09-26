# Evaluation

Three layers, each with a different question.

## 1. Automated suite (no network)

`cargo test` runs 21 unit tests and 28 integration tests. The integration tests drive the real HTTP
handler against a mock upstream that is slow, returns 500 and 429, refuses, hedges, returns HTML, dies
mid-stream, saturates, grades answers as a verifier, and asks clarifying questions. Inputs include
malformed JSON, hostile roles, oversized bodies, NaN, unicode control characters and prompt-injection
text. Every case must produce a 4xx, a correct cascade, or a correct answer, and no bad input may reach
an upstream.

What it answers: does the router behave correctly under failure and abuse.

## 2. Routing accuracy at scale (no network)

```
python3 evals/gen_pharma_questions.py 10000 > evals/pharma-10k.jsonl
pankhllm --config evals/pharma.yaml eval evals/pharma-10k.jsonl --min-accuracy 90
```

`pankhllm eval` reads labelled JSONL (`question`, `expected_intent`, `expected_tier`, optional
`scores`, `tags`, `tools`), runs the pure routing decision for each line, and reports tier and intent
accuracy, per-intent accuracy, a tier confusion table, sample mismatches, and decision latency. It exits
non-zero below `--min-accuracy`, so it can gate CI.

Result on the generated 10,000-question set (2026-09-25):

| Metric | Value |
|---|---|
| Tier accuracy | 99.20% |
| Intent accuracy | 99.20% |
| Mean decision latency | 15.5 us |
| Max decision latency | 133 us |

The remaining misses are label ambiguity ("explain the relationship between X and Y" is labelled
reasoning and routed as synthesis). The generator covers lookups, extraction, summaries, synthesis,
reasoning, multi-hop, code, alerts and chit-chat across regions, territories, products, metrics,
periods and segments.

**Synthetic accuracy is an upper bound.** To claim a number for production, export real questions,
label a sample (500 to 1,000 is enough to start), put them in `evals/real/` (git-ignored) and run the
same command. Then tune `routing.rules` and the classifier lists until the real set passes.

What it answers: does the policy send each kind of question to the intended tier, and how fast.

## 3. Domain adversarial scenarios (real models)

```
pankhllm --config evals/pharma.yaml serve --port 4125 &
python3 evals/domain_adversarial.py http://localhost:4125
```

Eleven scenarios aimed at what breaks in a pharma reporting RAG, checked against routing decisions and
answer behaviour:

1. Prompt injection hidden in a retrieved chunk is ignored.
2. Contradictory extracts are surfaced, not silently resolved.
3. No number is invented for a region absent from the context.
4. PII-bearing chunks with a `private` tag never reach a cloud model.
5. A vague question gets a clarifying question, not a guess.
6. A quarterly figure is not split into a monthly one.
7. The newest of two dated figures is preferred and dated.
8. An oversized table excludes small-window models.
9. Skill front matter pins tier and tags.
10. Tool-selection turns go fast, post-tool synthesis goes balanced.
11. The verifier's judgement is attached to the answer.

Latest run: 11/11 with gpt-5.6-terra (medium effort), gpt-5.6-luna, local gemma4:12b, and luna as
the verifier. Answers are model-dependent, so treat this as a regression gate for your configuration,
not a benchmark of the models.

What it answers: does the combination of routing, grounding, confidence and cascade hold up on the
failure modes of this domain.

## Assumptions

- Retrieval scores are on a 0-1 scale with higher meaning better; set `weak_retrieval.threshold` to
  your retriever's scale.
- Token counts are estimated at 3.6 characters per token for window and cost decisions.
- Prices in the example configs are placeholders; verify against vendor pages.
- Hedge detection is phrase-based and English; extend `cascade.hedge_phrases` for other languages.
- The heuristic classifier is English keyword-based; domain vocabulary belongs in `routing.rules`.
