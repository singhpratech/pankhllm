"""Demo executor for the pharma KPI catalog. Fictional, deterministic numbers.

    python3 evals/pharma-kpi/executor.py        # listens on 127.0.0.1:7125

Your real executor runs the query against your warehouse and applies the caller's
row-level scope; pankhllm only sends the operation name and validated parameters.
"""
import sys, zlib
sys.path.insert(0, "sdks/python")
from pankhllm.executor import serve

def num(*parts, lo=100, hi=5000):
    return lo + zlib.crc32("|".join(map(str, parts)).encode()) % (hi - lo)

def kpi_value(p, ctx):
    return {"metric": p["metric"], "geo": p.get("geo"), "product": p.get("product"),
            "period": p.get("period") or p.get("window"), "value": num(p["metric"], p.get("geo"), p.get("product"), p.get("period"))}

def kpi_rank(p, ctx):
    n = p.get("n", 5)
    one = {"territories": "territory", "districts": "district", "regions": "region", "prescribers": "prescriber"}.get(p["level"], p["level"])
    return {"rows": [{"name": f"{one} {i + 1}", p["metric"]: num(p["metric"], i)} for i in range(n)]}

def kpi_vs_benchmark(p, ctx):
    mine, bench = num(p["metric"], p.get("geo")), num(p["metric"], p.get("benchmark"), "bench")
    return {"metric": p["metric"], "geo": p.get("geo"), "value": mine, "benchmark": p.get("benchmark"), "benchmark_value": bench, "index": round(100 * mine / bench)}

def kpi_change(p, ctx):
    now, before = num(p["metric"], p.get("geo")), num(p["metric"], p.get("geo"), "prior")
    return {"metric": p["metric"], "compare": p["compare"], "value": now, "prior": before, "change_pct": round(100 * (now - before) / before, 1)}

def kpi_trend(p, ctx):
    return {"metric": p["metric"], "grain": p.get("grain"), "points": [num(p["metric"], p.get("geo"), i) for i in range(p.get("n", 6))]}

def prescriber_list(p, ctx):
    return {"rows": [{"npi": f"FAKE{num(p['geo'], i, lo=10000, hi=99999)}", "segment": p.get("segment"), "specialty": p.get("specialty")} for i in range(min(p.get("n", 20), 5))]}

def coverage_owner(p, ctx):
    return {"geo": p["geo"], "owner": f"Rep {num(p['geo'], lo=1, hi=400)}"}

serve({f.__name__: f for f in [kpi_value, kpi_rank, kpi_vs_benchmark, kpi_change, kpi_trend, prescriber_list, coverage_owner]}, port=7125)
