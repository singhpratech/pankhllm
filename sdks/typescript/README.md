# pankhllm (TypeScript / JavaScript)

```ts
import { PankhClient } from "@pankhllm/client";

const pk = new PankhClient("http://localhost:4000");
const a = await pk.ask(q, { context: hits.map(h => ({ text: h.text, score: h.score })), tags: ["private"], maxLatencyMs: 8000 });
console.log(a.text, a.model, a.decision);

for await (const ev of pk.stream(q, { context })) {
  if (ev.meta) console.error("routed to", ev.meta.model);
  if (ev.delta) process.stdout.write(ev.delta);
}
```

Or keep the official `openai` package: `new OpenAI({ baseURL: "http://localhost:4000/v1", apiKey: "unused", defaultHeaders: { "X-Pankh-Tags": "private" } })` and call `chat.completions.create({ model: "auto", ... })`.
