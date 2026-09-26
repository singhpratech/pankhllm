# Deploying pankhllm

One static binary plus one YAML file. No database, no runtime, no Docker requirement.

## Build

```bash
cargo build --release            # target/release/pankhllm (Linux, macOS, Windows)
```

TLS uses rustls, so there is no OpenSSL dependency on any platform. On Windows build with the
MSVC toolchain (`rustup default stable-msvc`) and you get `pankhllm.exe`.

## Configure

Copy `pankhllm.yaml`, keep only the providers and models you have, and put secrets in the
environment, never in the file:

```yaml
providers:
  azure:
    kind: azure
    base_url: https://my-resource.openai.azure.com
    api_key_env: AZURE_OPENAI_API_KEY
    api_version: "2024-10-21"        # omit to use the /openai/v1 surface
    timeout_secs: 90
    connect_timeout_secs: 3

models:
  - name: large                      # what clients see in decision.model
    provider: azure
    model: my-large-deployment       # your Azure deployment name
    tier: balanced
    context_window: 400000
    effort: medium                   # reasoning_effort; also suppresses sampling params
    cost: { input: 2.5, output: 15 } # per 1M tokens, verify
    timeout_secs: 60
  - name: small
    provider: azure
    model: my-small-deployment
    tier: fast
    context_window: 400000
    effort: low
    cost: { input: 0.25, output: 2 }
    timeout_secs: 20

routing:
  agent_loop: { tool_select: fast, after_tool_result: balanced }
  hedge: { after_ms: 1500 }
  max_latency_ms: 45000

server:
  api_keys_env: PANKH_API_KEYS       # comma-separated inbound keys; unset = open
  max_in_flight: 512
```

Validate with `pankhllm --config pankhllm.yaml check`.

## Inbound authentication

Set `server.api_keys_env` to an environment variable holding one or more comma-separated keys.
Clients send `Authorization: Bearer <key>`, `api-key: <key>` (Azure SDK style) or `x-api-key`.
`/health` stays open for probes. Comparison is constant-time. Terminate TLS in front of the router
(IIS ARR, nginx, a cloud load balancer) or keep it on a private network.

## Run

### Linux (systemd)

```ini
[Unit]
Description=pankhllm router
After=network-online.target

[Service]
ExecStart=/opt/pankhllm/pankhllm --config /opt/pankhllm/pankhllm.yaml serve --port 4000
EnvironmentFile=/etc/pankhllm.env      # AZURE_OPENAI_API_KEY=..., PANKH_API_KEYS=...
Restart=always
RestartSec=2
User=pankhllm
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

### Windows Server

The binary runs as a console app or a service. Two zero-dependency options:

```powershell
# as a service via sc.exe (runs pankhllm.exe directly; use NSSM if you want log rotation)
sc.exe create pankhllm binPath= "C:\pankhllm\pankhllm.exe --config C:\pankhllm\pankhllm.yaml serve --port 4000" start= auto
[Environment]::SetEnvironmentVariable("AZURE_OPENAI_API_KEY", "...", "Machine")
[Environment]::SetEnvironmentVariable("PANKH_API_KEYS", "...", "Machine")
sc.exe start pankhllm
```

Behind IIS, use Application Request Routing as a reverse proxy to `http://localhost:4000` and let
IIS own TLS. Keep the router on localhost or a private interface.

### Docker

```bash
docker build -t pankhllm .
docker run -p 4000:4000 -e AZURE_OPENAI_API_KEY=... -e PANKH_API_KEYS=... \
  -v $PWD/pankhllm.yaml:/app/pankhllm.yaml -v $PWD/prompts:/app/prompts pankhllm
```

## Point clients at it

- OpenAI SDKs: base URL `http://<router>:4000/v1`, model `auto`.
- Microsoft.Extensions.AI / OpenAI SDK with provider type `Responses`: endpoint `http://<router>:4000/v1`,
  `/v1/responses` is served with streaming, function tools and `reasoning.effort` accepted.
  `previous_response_id` is rejected; send the full input (set `store=false` on the client).
- Semantic Kernel, LangChain, Spring AI: chat-completions base URL, model `auto`, routing hints in
  `X-Pankh-*` headers.

## Health and operations

- `GET /health`: liveness, no auth.
- `GET /v1/stats`: per-model calls, errors, EWMA latency, cost, in-flight. In-memory; resets on restart.
- Logs: `RUST_LOG=info` for one line per model call, `RUST_LOG=warn` for errors only.
- Capacity: a single instance handles tens of thousands of routing decisions per second; model
  throughput is the limit. Run several instances behind a load balancer; they share nothing.

## With a single upstream model

Everything still applies except tiering:

- Timeouts, the request budget, load shedding and inbound auth need nothing else.
- Hedging: list the same deployment twice under two names; the second entry is the hedge.
- Phase routing and cost tiering need a cheaper deployment. Provision a mini-class model in the
  same resource and mark it `tier: fast`; tool-selection turns and lookups move there.
