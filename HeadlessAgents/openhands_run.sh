#!/usr/bin/env bash
set -euo pipefail
STATUS=0
if [ -z "${PROMPT:-}" ]; then
  if [ "$#" -gt 0 ]; then PROMPT="$*"; else echo "ERROR: Provide PROMPT env or args" >&2; exit 2; fi
fi
ITERATIONS_FLAG="-i ${MAX_TURNS:-40}"
START_TS=$(date +%s)
if ! poetry run python -m openhands.core.main ${ITERATIONS_FLAG} -t "${PROMPT}"; then STATUS=$?; fi
END_TS=$(date +%s)
mkdir -p /results
cat > /results/metrics.json <<EOF
{
  "status": "${STATUS}",
  "wall_time_sec": $((END_TS-START_TS))
}
EOF
echo "Run complete"
