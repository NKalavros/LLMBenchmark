#!/usr/bin/env bash
set -euo pipefail
if [ -z "${PROMPT:-}" ]; then
  if [ "$#" -gt 0 ]; then PROMPT="$*"; else echo "Need PROMPT" >&2; exit 2; fi
fi
mkdir -p /results
write_metrics() {
  local tmpjson
  tmpjson=$(mktemp)
  echo '{"status":"started","message":"RooCode automation pending"}' > "$tmpjson"
  if cp -f "$tmpjson" /results/metrics.json 2>/dev/null; then
    :
  else
    cp -f "$tmpjson" /tmp/metrics.json 2>/dev/null || true
    cp -f /tmp/metrics.json /results/metrics.json 2>/dev/null || true
  fi
  rm -f "$tmpjson" 2>/dev/null || true
}
write_metrics

# Best-effort code-server start (non-fatal)
(
  set +e
  echo "[INFO] Starting code-server for RooCode experimental harness." >&2
  code-server --auth none --port 13338 /workspace
) >/tmp/roocode-codeserver.log 2>&1 &
sleep 1
exit 0
