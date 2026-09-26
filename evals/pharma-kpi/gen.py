#!/usr/bin/env python3
"""Labelled pharma KPI questions for training and evaluating pankhllm's own decision model.

Templates are split per operation into a training group and a held-out group, so the
test phrasings are ones the model never saw. Entity values (codes, months, numbers,
products) are drawn at random; the model learns shapes, not values.

    python3 evals/pharma-kpi/gen.py 20000 > state/pk-train.jsonl      # training templates
    python3 evals/pharma-kpi/gen.py 2000 --heldout > state/pk-test.jsonl
    python3 evals/pharma-kpi/gen.py --count                             # size of the space
"""
import json, random, sys

METRIC = {  # surface forms users type
    "trx": ["TRx", "total prescriptions", "total scripts", "scripts", "prescriptions"],
    "nrx": ["NRx", "new prescriptions", "new scripts"],
    "nbrx": ["NBRx", "new to brand prescriptions", "new-to-brand"],
    "calls": ["calls", "call activity", "HCP calls", "visits", "details"],
    "reach": ["reach", "call reach"],
    "frequency": ["frequency", "call frequency"],
    "market_share": ["market share", "share"],
    "samples": ["samples", "sample drops"],
    "new_writers": ["new writers", "new prescribers"],
}
PRODUCTS = ["Zelora", "Cardivex", "Neurolin", "Oncora"]
WINDOWS = ["month to date", "quarter to date", "year to date", "last month", "last quarter", "past 4 weeks", "last 13 weeks", "MTD", "QTD", "YTD"]
LEVELS = ["territories", "districts", "regions", "prescribers", "HCPs", "doctors"]
TOP = ["top", "highest", "best", "leading"]
BOTTOM = ["bottom", "lowest", "worst", "weakest"]
PEERS = ["my peers", "peers", "other reps", "the national average", "the nation", "my region", "the district", "my peer group"]
PRIOR = ["month over month", "vs last month", "compared to last month", "quarter over quarter", "vs last quarter"]
YOY = ["year over year", "vs last year", "compared to last year", "YoY", "versus last year"]
GRAIN = ["weekly", "monthly", "quarterly", "weeks", "months", "quarters"]
SEGMENT = ["high decile", "target", "new writer", "lapsed", "no-see", "top decile", "lapsed writers"]
SPECIALTY = ["cardiologists", "oncologists", "PCPs", "neurologists", "psychiatrists", "primary care physicians"]

def geo():
    k = random.random()
    return f"T-{random.randint(100, 999)}" if k < 0.6 else f"D-{random.randint(10, 99)}" if k < 0.9 else f"R-{random.randint(1, 9)}"

def month():
    return f"{random.choice([2024, 2025, 2026])}-{random.randint(1, 12):02d}"

T = {
    "kpi_value": [
        "What were {m} for {g} in {d}?", "{m} for {g} {d}", "show me {m} in {g} for {d}", "how many {m} did {g} have in {d}?",
        "{p} {m} in {g}, {d}", "give me {m} for {g} {w}", "what's my {m} {w}?", "{m} {w} for {p}", "pull {m} for {g} in {d}",
        "what is the {m} number for {g} this month", "tell me {g} {m} for {d}", "{g}: {m} {w}", "how are {m} looking for {p} in {g} {w}?",
        "can you get me {p} {m} for {d}", "I need {m} for {g}", "current {m} in {g}",
    ],
    "kpi_rank": [
        "{t} {n} {l} by {m} in {d}", "which {n} {l} had the {t} {m} {w}?", "rank {l} by {m} for {d}", "{b} {n} {l} on {m} {w}",
        "show the {t} {n} {l} for {p} {m}", "list the {n} {b} {l} by {m} in {g}", "who are my {t} {n} {l} for {m}?",
        "{n} {l} with the {b} {m} in {d}", "top {n} {l} in {g} by {m}", "which {l} are {b} on {m} {w}?", "leaderboard of {l} by {m} {w}",
        "{b} performing {l} for {p} {m} in {d}",
    ],
    "kpi_vs_benchmark": [
        "how am I doing on {m} vs {pe}?", "how does {g} compare to {pe} on {m}?", "{m} for {g} versus {pe} in {d}",
        "am I ahead of {pe} on {m} {w}?", "compare my {m} with {pe}", "where do I stand against {pe} on {p} {m}?",
        "is {g} above or below {pe} for {m}?", "benchmark {g} {m} against {pe}", "how do my {m} stack up against {pe} {w}?",
        "{p} {m} in {g} relative to {pe}", "gap between {g} and {pe} on {m} in {d}", "my {m} compared with {pe}",
    ],
    "kpi_change": [
        "how did {m} change {c} in {g}?", "{m} for {g} {c}", "what's the change in {p} {m} {c}?", "{g} {m} growth {c}",
        "did {m} go up or down {c} for {g}?", "{m} {c} in {d}", "percent change in {m} {c} for {g}", "show {m} trend {c} for {p}",
        "is {g} {m} growing {c}?", "difference in {m} {c} for {g} in {d}", "how much did {m} grow {c}?", "{p} {m} delta {c}",
    ],
    "kpi_trend": [
        "{m} trend over the last {n} {gr} for {g}", "show {gr} {m} for {g} over the last {n} {gr}", "plot {m} for {p} over {n} {gr}",
        "how has {m} trended over the past {n} {gr} in {g}?", "{gr} {m} trend for {g}", "trend of {m} in {g} for the last {n} {gr}",
        "{m} over time for {p}", "chart {g} {m} by {gr}", "give me the last {n} {gr} of {m} for {g}", "{m} history for {g} by {gr}",
        "time series of {p} {m} in {g}", "what does the {m} trend look like in {g}?",
    ],
    "prescriber_list": [
        "list {s} prescribers in {g}", "who are the {s} {sp} in {g}?", "show {n} {s} {sp} in {g}", "{sp} in {g} that are {s}",
        "give me my {s} HCPs in {g}", "which {sp} in {g} are {s} for {p}?", "names of {s} prescribers in {g}", "{s} doctors in {g}",
        "export the {n} {s} {sp} in {g}", "who should I call in {g}? {s} {sp}", "find {s} {sp} in {g} for {p}", "{sp} list for {g}",
    ],
    "coverage_owner": [
        "who covers {g}?", "who owns {g}", "which rep is assigned to {g}?", "owner of {g}", "who is the rep for {g}",
        "who manages {g}?", "tell me who covers territory {g}", "rep for {g}", "who's responsible for {g}?", "{g} owner",
        "which representative handles {g}?", "who looks after {g}?",
    ],
    "UNSUPPORTED": [
        "draft an email to my manager about {m} in {g}", "what does {m} stand for?", "how do I log a call in the CRM?",
        "remind me to visit {g} tomorrow", "what is the formulary status of {p}?", "book a meeting with the district manager",
        "translate this call note into Spanish", "what's the weather like in {g}?", "tell me a joke about sales reps",
        "what is the patient copay for {p}?", "summarize the latest sales training", "what are the side effects of {p}?",
        "how do I reset my password?", "write talking points for {p}", "what's our company holiday schedule?",
        "which conferences are coming up this quarter?", "how do I expense a lunch program?", "what is the capital of France?",
    ],
}

def fill(t):
    mk = random.choice(list(METRIC))
    return t.format(m=random.choice(METRIC[mk]), g=geo(), d=month(), w=random.choice(WINDOWS), p=random.choice(PRODUCTS),
                    n=random.randint(3, 25), l=random.choice(LEVELS), t=random.choice(TOP), b=random.choice(BOTTOM),
                    pe=random.choice(PEERS), c=random.choice(PRIOR + YOY), gr=random.choice(GRAIN), s=random.choice(SEGMENT),
                    sp=random.choice(SPECIALTY))

def split(templates, heldout):
    # Every third template (by position) is held out, so each operation keeps phrasings for testing.
    return [t for i, t in enumerate(templates) if (i % 3 == 2) == heldout]

def main():
    args = sys.argv[1:]
    if "--count" in args:
        n_templates = sum(len(v) for v in T.values())
        surfaces = sum(len(v) for v in METRIC.values())
        print(f"operations: {len(T) - 1} + UNSUPPORTED; templates: {n_templates}; metric phrasings: {surfaces}")
        print(f"distinct question shapes (template x metric phrasing x modifier phrasing): roughly {n_templates * surfaces * 8:,}")
        print(f"distinct questions with entity values (x ~900 geos x 36 months x products x N): well over {n_templates * surfaces * 900 * 36:,}")
        return
    n = int(args[0]) if args and args[0].isdigit() else 1000
    heldout = "--heldout" in args
    random.seed(7 if not heldout else 99)
    pools = {op: split(ts, heldout) for op, ts in T.items()}
    ops = list(pools)
    for i in range(n):
        op = ops[i % len(ops)]
        print(json.dumps({"question": fill(random.choice(pools[op])), "label": op}))

main()
