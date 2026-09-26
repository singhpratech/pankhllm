# pankhllm (Python)

```python
from pankhllm import Client, Chunk

pk = Client("http://localhost:4000")
chunks = [Chunk(text=hit.text, score=hit.score, source=hit.doc_id) for hit in my_retriever.search(q)]

a = pk.ask(q, context=chunks, tags=["private"], prompt="support-agent", max_latency_ms=8000)
print(a.text, a.model, a.cost_usd, a.decision["target_tier"])

for delta in pk.stream(q, context=chunks):
    print(delta, end="")
```

Already using the `openai` package? Point it at the router and keep your code:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="unused",
                default_headers={"X-Pankh-Tags": "private", "X-Pankh-Prompt": "support-agent"})
client.chat.completions.create(model="auto", messages=[...])
```
