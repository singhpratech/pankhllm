# pankhllm

The [pankhllm](https://github.com/singhpratech/pankhllm) router, installed through npm.
pankhllm is the LLM gateway that learns to skip the LLM: it trains its own tiny decision model
on your agent's traffic and answers repeat decisions in about 0.2 ms on a CPU with no LLM
call. Everything unfamiliar still goes to the LLM you already use. OpenAI-compatible.

```sh
npx pankhllm serve --port 4000     # then point any OpenAI client at http://localhost:4000/v1
```

The prebuilt binary for your platform is installed as an optional dependency
(`@pankhllm/linux-x64`, `darwin-arm64`, `win32-x64`, ...). The TypeScript and JavaScript client
is `@pankhllm/client`. Live demo and docs: https://singhpratech.github.io/pankhllm/
