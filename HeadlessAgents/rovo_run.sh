#!/usr/bin/env bash
set -euo pipefail

PROMPT_VALUE="${ROVO_PROMPT:-${PROMPT:-}}"
if [ -z "$PROMPT_VALUE" ] && [ $# -gt 0 ]; then
  PROMPT_VALUE="$*"
fi

mkdir -p /results

# Verify ACLI exists
if ! command -v acli >/dev/null 2>&1; then
  echo "ERROR: acli binary not found" >&2
  echo '{"status":"error","error":"acli_missing"}' > /results/metrics.json
  exit 127
fi

# Guard against interactive mode for automation (unless explicitly allowed)
if [ -z "$PROMPT_VALUE" ]; then
  if [ "${ROVO_MODE}" = "interactive" ] || [ "${ROVO_ALLOW_INTERACTIVE}" = "1" ]; then
    echo "[INFO] Starting interactive mode (NOT recommended for CI)." >&2
    acli rovodev run
    echo '{"status":"interactive_completed"}' > /results/metrics.json
    exit 0
  else
    echo "ERROR: No prompt supplied for non-interactive run. Set PROMPT or ROVO_PROMPT." >&2
    echo '{"status":"error","error":"no_prompt"}' > /results/metrics.json
    exit 2
  fi
fi

# Pre-run git snapshot
if [ ! -d .git ]; then
  git init -q
  git add . >/dev/null 2>&1 || true
  git commit -m "initial" >/dev/null 2>&1 || true
fi
BASE_COMMIT=$(git rev-parse --short HEAD || echo "unknown")

START_TS=$(date +%s)
STATUS="success"
ERROR_MSG=""

set +e
acli rovodev run "$PROMPT_VALUE" > /results/agent_stdout.log 2>&1
RC=$?
set -e
if [ $RC -ne 0 ]; then
  STATUS="error"
  ERROR_MSG="acli_exit_${RC}"
fi

END_TS=$(date +%s)

# Collect diff
git add -A >/dev/null 2>&1 || true
DIFF_FILE=/results/diff.patch
git diff --staged > "$DIFF_FILE" || echo "(diff failed)" >&2

# Basic diff stats
INSERTIONS=0
DELETIONS=0
FILES_CHANGED=0
if command -v awk >/dev/null 2>&1; then
  while read -r ins del file; do
    if [[ "$ins" =~ ^[0-9]+$ ]]; then INSERTIONS=$((INSERTIONS+ins)); fi
    if [[ "$del" =~ ^[0-9]+$ ]]; then DELETIONS=$((DELETIONS+del)); fi
    FILES_CHANGED=$((FILES_CHANGED+1))
  done < <(git diff --staged --numstat || true)
fi

cat > /results/metrics.json <<EOF
{
  "status": "${STATUS}",
  "error": "${ERROR_MSG}",
  "prompt": $(printf '%s' "$PROMPT_VALUE" | jq -R .),
  "wall_time_sec": $((END_TS-START_TS)),
  "base_commit": "${BASE_COMMIT}",
  "files_changed": ${FILES_CHANGED},
  "insertions": ${INSERTIONS},
  "deletions": ${DELETIONS}
}
EOF

echo "Rovo Dev run completed: status=${STATUS} files=${FILES_CHANGED} +${INSERTIONS} -${DELETIONS}" >&2
