#!/usr/bin/env bash
set -euo pipefail

PROMPT_VAL=""

# Parse container-level args: -p|--prompt "..."
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prompt)
      shift
      PROMPT_VAL="${1:-}"
      shift || true
      ;;
    --)
      shift
      break
      ;;
    *)
      # Ignore unknown args (reserved for future); stop at first non-option
      break
      ;;
  esac
done

# If not provided via -p, allow PROMPT env
if [[ -z "${PROMPT_VAL}" ]]; then
  PROMPT_VAL="${PROMPT:-}"
fi

export PROMPT="${PROMPT_VAL}"

if [[ -z "${AGENT_CMD:-}" ]]; then
  echo "[run-agent] AGENT_CMD env variable is required. It should be a shell command that can reference $PROMPT." >&2
  echo "Example: docker run --rm -e AGENT_CMD='cursor-agent -p "$PROMPT"' code-agent-image -p 'Fix tests'" >&2
  echo "Fallback to interactive shell..." >&2
  exec "/bin/bash"
fi

echo "[run-agent] Using PROMPT: ${PROMPT}" >&2
echo "[run-agent] Executing: ${AGENT_CMD}" >&2

# Run via bash -lc to support shell features and $PROMPT expansion
exec bash -lc "${AGENT_CMD}"
