#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${GEMINI_METRICS_DIR:-/results}
mkdir -p "$OUT_DIR"

# Accept prompt via env or args
if [ -z "${PROMPT:-}" ] && [ "$#" -gt 0 ]; then
    PROMPT="$*"
fi

MODEL_FLAG=""
if [ -n "${GEMINI_MODEL:-}" ]; then
    MODEL_FLAG=( -m "${GEMINI_MODEL}" )
fi

API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"

start_ts=$(date +%s.%N || date +%s)
if [ -z "${PROMPT:-}" ]; then
    # No prompt: just show help for observability
    if gemini --help >/dev/null 2>&1; then
        printf '{"status":"noop","message":"no_prompt","wall_time_sec":%.3f}\n' "$(echo "$(date +%s.%N || date +%s) - $start_ts" | bc 2>/dev/null || awk -v s=$start_ts 'BEGIN{print 0}')" > "$OUT_DIR/metrics.json"
        exit 0
    else
        echo '{"status":"error","message":"cli_help_failed"}' > "$OUT_DIR/metrics.json"
        exit 1
    fi
fi

if [ -z "$API_KEY" ]; then
    echo '{"status":"error","message":"no_api_key"}' > "$OUT_DIR/metrics.json"
    exit 0
fi

# Non-interactive single prompt
set +e
RAW=$(gemini ${MODEL_FLAG:+"${MODEL_FLAG[@]}"} -p "$PROMPT" 2>&1)
rc=$?
set -e
end_ts=$(date +%s.%N || date +%s)
dur=$(python - <<PY
import os,sys
try:
    import time
    start=float(os.environ.get('GEM_RUN_START','0'))
except Exception:
    start=0
PY)
# Compute duration in shell if bc exists
if command -v python >/dev/null 2>&1; then
    duration=$(python - <<PY
import os,time
start=float(os.environ['START_TS'])
end=float(os.environ['END_TS'])
print(f"{end-start:.3f}")
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
echo "${RAW}" | head -c 400 > "$OUT_DIR/last_output.txt" || true
escaped=$(printf '%s' "$RAW" | head -c 1000 | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
printf '{"status":"%s","message":"%s","wall_time_sec":%s,"response_excerpt":%s}\n' "$status" "$msg" "${duration}" "$escaped" > "$OUT_DIR/metrics.json" || echo '{"status":"error","message":"write_failed"}' > "$OUT_DIR/metrics.json"
exit 0
