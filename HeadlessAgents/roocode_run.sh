#!/usr/bin/env bash
set -euo pipefail
if [ -z "${PROMPT:-}" ]; then
  if [ "$#" -gt 0 ]; then PROMPT="$*"; else echo "Need PROMPT" >&2; exit 2; fi
fi
mkdir -p /results
echo "[INFO] Starting code-server for RooCode experimental harness." >&2
code-server --auth none --port 13338 /workspace &
PID=$!
sleep 5
echo '{"status":"started","message":"RooCode automation pending"}' > /results/metrics.json
wait $PID || true
