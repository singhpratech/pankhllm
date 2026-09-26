# Pankhllm

.NET client and executor helper for [pankhllm](https://github.com/singhpratech/pankhllm), the
LLM gateway that learns to skip the LLM. It trains its own tiny decision model on your agent's
traffic and answers repeat decisions in about 0.2 ms on a CPU with no LLM call; everything
unfamiliar still goes to the LLM you already use. OpenAI-compatible, so Semantic Kernel and
Microsoft Agent Framework work by changing the endpoint.

```csharp
var pk = new Pankhllm.PankhClient("http://localhost:4000");
var a = await pk.AskAsync("who covers R-3?", new AskOptions { Tags = ["private"] });
Console.WriteLine($"{a.Text} via {a.Model}");   // "decision:coverage_owner" when no LLM was called
```

The router binary for .NET projects is the `Pankhllm.Server` package. Live demo and docs:
https://singhpratech.github.io/pankhllm/
