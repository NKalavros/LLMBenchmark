#!/usr/bin/env bash
set -euo pipefail
START_TS=$(date +%s)
echo "Baseline (no agent) run. Prompt: ${PROMPT:-<none>}" >&2
sleep 1
END_TS=$(date +%s)
mkdir -p /results
cat > /results/metrics.json <<EOF
{
  "status": "noop",
  "wall_time_sec": $((END_TS-START_TS))
}
EOF
