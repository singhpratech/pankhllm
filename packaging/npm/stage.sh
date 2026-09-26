#!/usr/bin/env bash
# Stage a platform package from a built binary:
#   packaging/npm/stage.sh linux-x64 target/release/pankhllm 0.1.0
set -euo pipefail
platform="$1"; binary="$2"; version="${3:-0.1.0}"
os="${platform%-*}"; cpu="${platform#*-}"
dir="$(dirname "$0")/dist/@pankhllm/$platform"
mkdir -p "$dir/bin"
exe="pankhllm"; [ "$os" = "win32" ] && exe="pankhllm.exe"
cp "$binary" "$dir/bin/$exe"; chmod +x "$dir/bin/$exe"
sed -e "s/__PLATFORM__/$platform/g" -e "s/__VERSION__/$version/g" -e "s/__OS__/$os/g" -e "s/__CPU__/$cpu/g" \
  "$(dirname "$0")/platform/package.json.tmpl" > "$dir/package.json"
echo "staged $dir"
