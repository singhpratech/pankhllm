# pankhllm-server

The [pankhllm](https://github.com/singhpratech/pankhllm) router binary, installed with pip.
pankhllm is the LLM gateway that learns to skip the LLM: it trains its own tiny decision model
on your agent's traffic and answers repeat decisions in about 0.2 ms on a CPU with no LLM
call. Everything unfamiliar still goes to the LLM you already use. OpenAI-compatible.

```sh
pip install pankhllm-server
pankhllm serve --port 4000      # then point any OpenAI client at http://localhost:4000/v1
```

The pure-Python client and executor helper are in the `pankhllm` package:
`pip install pankhllm`. Live demo and docs: https://singhpratech.github.io/pankhllm/
