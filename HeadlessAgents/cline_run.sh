#!/usr/bin/env bash
set -euo pipefail
if [ -z "${PROMPT:-}" ]; then
  if [ "$#" -gt 0 ]; then PROMPT="$*"; else echo "Need PROMPT" >&2; exit 2; fi
fi
mkdir -p /results
echo "[INFO] Starting code-server with Cline extension (no automation yet)." >&2
code-server --auth none --port 13337 /workspace &
PID=$!
sleep 5
echo '{"status":"started","message":"Cline automation TBD"}' > /results/metrics.json
wait $PID || true
