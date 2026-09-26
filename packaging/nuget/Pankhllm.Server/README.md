# Pankhllm.Server

The [pankhllm](https://github.com/singhpratech/pankhllm) router binary for .NET teams, with
`PankhllmServer.Start()`. pankhllm is the LLM gateway that learns to skip the LLM: it trains
its own tiny decision model on your agent's traffic and answers repeat decisions in about
0.2 ms on a CPU with no LLM call. Everything unfamiliar still goes to the LLM you already use.
OpenAI-compatible: point Semantic Kernel or Microsoft Agent Framework at http://localhost:4000/v1.

The client library is the `Pankhllm` package. Live demo and docs: https://singhpratech.github.io/pankhllm/
