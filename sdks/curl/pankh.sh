#!/usr/bin/env bash
# pankhllm from the shell. Requires curl; jq is optional for pretty output.
#   PANKH_URL=http://localhost:4000 ./pankh.sh ask "What was call activity for East in August?" '[{"text":"East Aug: 1,842 calls","score":0.9}]' '["private"]'
#   ./pankh.sh route "Why did NRx fall in T-112?"
#   ./pankh.sh stream "Define the reach KPI."
#   ./pankh.sh stats
set -euo pipefail
URL="${PANKH_URL:-http://localhost:4000}"
cmd="${1:-}"; q="${2:-}"; ctx="${3:-[]}"; tags="${4:-[]}"; prompt="${PANKH_PROMPT:-}"
body() {
  local stream="$1"
  printf '{"model":"%s","stream":%s,"messages":[{"role":"user","content":%s}],"pankhllm":{"context":%s,"tags":%s%s}}' \
    "${PANKH_MODEL:-auto}" "$stream" "$(printf '%s' "$q" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || printf '"%s"' "$q")" \
    "$ctx" "$tags" "$( [ -n "$prompt" ] && printf ',"prompt":"%s"' "$prompt" )"
}
case "$cmd" in
  ask)    curl -sS "$URL/v1/chat/completions" -H 'content-type: application/json' -d "$(body false)" | (jq . 2>/dev/null || cat) ;;
  route)  curl -sS "$URL/v1/route" -H 'content-type: application/json' -d "$(body false)" | (jq . 2>/dev/null || cat) ;;
  stream) curl -sSN "$URL/v1/chat/completions" -H 'content-type: application/json' -d "$(body true)" \
            | sed -un 's/^data: //p' | grep -v '^\[DONE\]' | (jq -rj '.choices[0].delta.content // empty' 2>/dev/null || cat); echo ;;
  stats)  curl -sS "$URL/v1/stats" | (jq . 2>/dev/null || cat) ;;
  models) curl -sS "$URL/v1/models" | (jq . 2>/dev/null || cat) ;;
  *) echo "usage: $0 ask|route|stream|stats|models \"question\" [context_json] [tags_json]" >&2; exit 2 ;;
esac
