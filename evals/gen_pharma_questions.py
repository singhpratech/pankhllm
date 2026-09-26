#!/usr/bin/env python3
"""Generate labelled pharma commercial-analytics questions for routing evaluation.

Each line: {"question", "expected_intent", "expected_tier", "scores", "tags", "tools"}.
Templates x entities x phrasings give tens of thousands of distinct questions. The
labels are what a domain owner would expect the router to do under the rules in
evals/pharma.yaml (alerts -> reasoning/balanced, KPI lookups -> fast).

    python3 evals/gen_pharma_questions.py 10000 > evals/pharma-10k.jsonl
"""
import json, random, sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 7)

REGIONS = ["East", "West", "Central", "Northeast", "Southwest", "Mid-Atlantic", "Pacific", "Great Lakes"]
TERR = [f"T-{i:03d}" for i in range(101, 140)]
PRODUCTS = ["Product X", "Zelora", "Cardivex", "Neurolin", "Oncora", "the 40mg SKU", "Brand A", "the biosimilar"]
METRICS = ["call activity", "HCP calls", "reach", "frequency", "FTE count", "TRx", "NRx", "NBRx", "share of voice", "detail rate", "sample drops", "speaker programs"]
PERIODS = ["August 2026", "Q3 2026", "last month", "week 37", "September", "the last 4 weeks", "H1 2026", "YTD"]
SEGS = ["cardiologists", "PCPs", "oncologists", "high decilers", "tier-1 targets", "KOLs", "hospital accounts", "IDNs"]

def pick(xs): return random.choice(xs)

# (template, intent). Placeholders drawn per sample.
LOOKUP = [
    "What was {metric} for {region} in {period}?",
    "How many {metric} did {terr} log in {period}?",
    "Give me {metric} for {product} in {region}, {period}.",
    "{metric} for {terr}, {period}?",
    "What is the {metric} number for {segs} in {region}?",
    "Show {metric} for {product} among {segs} in {period}.",
    "How much did {metric} change for {region} in {period}?",
    "What is the current {metric} for {terr}?",
    "Who is the top rep by {metric} in {region}?",
    "When was the last field visit to the {segs} in {terr}?",
]
EXTRACTION = [
    "List all territories in {region} with {metric} below target in {period}.",
    "Extract the {metric} for every territory in {region} as a table.",
    "Enumerate the {segs} in {terr} with zero calls in {period}.",
    "Pull out every alerted account and its {metric} as JSON.",
    "List every product with declining {metric} in {region}.",
    "Give me all the reps in {region} and their {metric} for {period} as CSV.",
]
SUMMARY = [
    "Summarize {metric} performance for {region} in {period}.",
    "Give me the key points from the {period} field report for {region}.",
    "TL;DR of the {product} launch tracker for {period}.",
    "Brief overview of {metric} trends across {segs} in {period}.",
    "Recap the {terr} business review for {period}.",
]
SYNTHESIS = [
    "Compare {metric} for {region} versus {region2} in {period}.",
    "What is the difference between {metric} and {metric2} for {product} in {period}?",
    "How does {product} {metric} relate to {metric2} among {segs}?",
    "Contrast {terr} and {terr2} on {metric} and {metric2}.",
    "Pros and cons of shifting {segs} calls from {region} to {region2}.",
    "How do {metric} and {metric2} trend together for {product} in {period}?",
]
REASONING = [
    "Why did {metric} drop for {product} in {terr} in {period}?",
    "What would happen to {metric} if we cut {segs} calls in {region} by 20%?",
    "Should we reallocate FTEs from {region} to {region2} given {metric} in {period}?",
    "What is the most likely cause of the {metric} decline in {terr}?",
    "Explain the relationship between {metric} and {metric2} for {product} and what drives it.",
    "Estimate the {metric} impact of the {product} formulary change in {region}.",
    "What should the MSL check first after {metric} fell in {terr}?",
    "Diagnose the root cause of low {metric} among {segs} in {region}.",
    "Recommend a plan to lift {metric} for {product} in {terr} next quarter.",
    "Predict {metric} for {region} in {period} given the current trend.",
]
MULTIHOP = [
    "Which rep covers {terr}, and what was their {metric} in {period}?",
    "Who approved the {region} FTE change, and what did they say about {metric}?",
    "Which territory in {region} had the highest {metric}? What was its {metric2}?",
    "Which {segs} switched to {product} in {period}, and then which of them stopped prescribing?",
    "What was {metric} for {terr} in {period} and how many of those were {segs}?",
]
CODE = [
    "Write a SQL query for {metric} by territory in {region} for {period}.",
    "Fix this python function that aggregates {metric} per rep; it returns None for {terr}.",
    "Write a DAX measure for {metric} share of {product} among {segs}.",
    "Generate a pandas snippet to pivot {metric} by region and {period}.",
    "Implement a function that flags territories where {metric} fell 15% week over week.",
]
ALERT = [  # config rule: alerts -> reasoning intent, balanced tier
    "An alert fired: {metric} for {product} fell in {terr}. What is going on?",
    "Explain the {metric} anomaly in {region} for {period}.",
    "Which alerts fired for {terr} this week and why?",
    "Alert: {metric} spiked for {segs} in {region}. Causes?",
    "Investigate the anomaly in {product} {metric} for {period}.",
]
CHITCHAT = ["hello", "thanks!", "hi there", "ok", "good morning", "thank you", "hey", "bye"]

# KPI-lookup rule words force fast lookup even when phrasing looks analytical.
KPI_RULE_WORDS = ["call activity", "fte", "trx", "nrx"]

CLASSES = [
    (LOOKUP, "lookup", "fast", 0.30),
    (EXTRACTION, "extraction", "fast", 0.10),
    (SUMMARY, "summarization", "balanced", 0.08),
    (SYNTHESIS, "synthesis", "balanced", 0.12),
    (REASONING, "reasoning", "reasoning", 0.18),
    (MULTIHOP, "multi_hop", "reasoning", 0.07),
    (CODE, "code", "reasoning", 0.06),
    (ALERT, "reasoning", "balanced", 0.06),
    (CHITCHAT, "chitchat", "fast", 0.03),
]

def fill(t):
    r1, r2 = random.sample(REGIONS, 2)
    m1, m2 = random.sample(METRICS, 2)
    t1, t2 = random.sample(TERR, 2)
    return t.format(metric=m1, metric2=m2, region=r1, region2=r2, terr=t1, terr2=t2, product=pick(PRODUCTS), period=pick(PERIODS), segs=pick(SEGS))

def expected(intent, tier, q):
    ql = q.lower()
    # Mirror the config rules: alerts first, then KPI lookup words.
    if any(w in ql for w in ["alert", "anomal"]):
        return "reasoning", "balanced"
    if any(w in ql for w in KPI_RULE_WORDS):
        return "lookup", "fast"
    return intent, tier

out = 0
weights = [c[3] for c in CLASSES]
while out < N:
    templates, intent, tier, _ = random.choices(CLASSES, weights)[0]
    q = fill(pick(templates)) if intent != "chitchat" else pick(CHITCHAT)
    e_intent, e_tier = expected(intent, tier, q)
    # Retrieval quality: chitchat has none; others mostly good so tiering is tested, not escalation.
    scores = [] if intent == "chitchat" else [round(random.uniform(0.55, 0.95), 2)]
    tools = []
    rec = {"question": q, "expected_intent": e_intent, "expected_tier": e_tier, "scores": scores, "tags": [], "tools": tools}
    print(json.dumps(rec))
    out += 1
