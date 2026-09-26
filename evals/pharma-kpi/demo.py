"""Ask pankhllm pharma KPI questions through the stock OpenAI SDK and show which lane answered.

    python3 evals/pharma-kpi/demo.py [port] [questions.txt]

Any framework that speaks the OpenAI API (LangChain, Semantic Kernel, Microsoft Agent
Framework, LlamaIndex, the Vercel AI SDK) talks to pankhllm the same way: change the base URL.
"""
import sys, time
from openai import OpenAI

port = sys.argv[1] if len(sys.argv) > 1 else "4150"
qs = [l.strip() for l in open(sys.argv[2])] if len(sys.argv) > 2 else [
    "what were total scripts for Zelora in T-112 in 2025-07?",
    "how is D-12 doing against my peers on call reach?",
    "top 5 territories by NBRx for Cardivex last quarter",
    "who covers R-3?",
    "which cardiologists in T-240 are lapsed writers?",
    "did new scripts for Oncora in T-77 go up since last month?",
    "why did calls fall in T-112 in 2025-08?",
]
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local")
for q in qs:
    if not q:
        continue
    t = time.perf_counter()
    r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": q}], max_tokens=200)
    ms = (time.perf_counter() - t) * 1000
    lane = (r.model_extra or {}).get("pankhllm", {}).get("model", r.model)
    text = (r.choices[0].message.content or "").replace("\n", " ")
    print(f"{ms:8.1f} ms  {lane:<30} {q[:60]:<61} {text[:70]}")
