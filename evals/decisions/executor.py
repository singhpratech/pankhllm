import sys
sys.path.insert(0, "sdks/python")
from pankhllm.executor import serve

VAL = {"calls": 1842, "reach": 0.71, "revenue": 125000}
def metric_by_period(p, ctx):
    return {"value": VAL[p["metric"]] + hash(p["entity"]) % 100}
def top_entities(p, ctx):
    return {"rows": [{"territory": f"T-{100+i}", p["metric"]: 1000 - i * 37} for i in range(p["n"])]}
def entity_owner(p, ctx):
    return {"owner": "Rep " + p["entity"][-3:]}
serve({"metric_by_period": metric_by_period, "top_entities": top_entities, "entity_owner": entity_owner}, port=7124)
