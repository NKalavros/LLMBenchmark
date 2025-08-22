#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${QWEN_METRICS_DIR:-/results}
mkdir -p "$OUT_DIR"

if [ -z "${PROMPT:-}" ] && [ "$#" -gt 0 ]; then
  PROMPT="$*"
fi

MODEL_FLAG=""
if [ -n "${QWEN_MODEL:-}" ]; then
  MODEL_FLAG=( -m "${QWEN_MODEL}" )
fi

start_ts=$(date +%s.%N || date +%s)
if [ -z "${PROMPT:-}" ]; then
  if qwen --help >/dev/null 2>&1; then
    printf '{"status":"noop","message":"no_prompt","wall_time_sec":0.000}\n' > "$OUT_DIR/metrics.json"
    exit 0
  else
    echo '{"status":"error","message":"cli_help_failed"}' > "$OUT_DIR/metrics.json"
    exit 1
  fi
fi

# Decide whether we have any auth – if none, still attempt (may prompt) but we mark missing key.
AUTH_STATUS=present
if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  AUTH_STATUS=absent
fi

set +e
RAW=$(qwen ${MODEL_FLAG:+"${MODEL_FLAG[@]}"} -p "$PROMPT" 2>&1)
rc=$?
set -e
end_ts=$(date +%s.%N || date +%s)
if command -v python >/dev/null 2>&1; then
  duration=$(python - <<PY
import os
start=float(os.environ['START_TS']); end=float(os.environ['END_TS']); print(f"{end-start:.3f}")
PY START_TS="$start_ts" END_TS="$end_ts")
else
  duration="0"
fi

status=success
msg=ok
if [ $rc -ne 0 ]; then
  status=error
  msg="cli_exit_${rc}"
fi
if [ "$AUTH_STATUS" = absent ]; then
  status=error
  msg="no_auth"
fi
echo "${RAW}" | head -c 400 > "$OUT_DIR/last_output.txt" || true
escaped=$(printf '%s' "$RAW" | head -c 1000 | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
printf '{"status":"%s","message":"%s","wall_time_sec":%s,"response_excerpt":%s}\n' "$status" "$msg" "$duration" "$escaped" > "$OUT_DIR/metrics.json" || echo '{"status":"error","message":"write_failed"}' > "$OUT_DIR/metrics.json"
exit 0
