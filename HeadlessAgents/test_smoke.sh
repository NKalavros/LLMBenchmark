#!/usr/bin/env bash
# Smoke tests for HeadlessAgents Dockerfiles and run scripts.
# - Builds each image
# - Runs a fast check
# - Verifies /results/metrics.json or CLI availability
# macOS/zsh-friendly; requires Docker and (optionally) jq.

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

JQ=${JQ:-jq}
HAVE_JQ=1
if ! command -v "$JQ" >/dev/null 2>&1; then
  HAVE_JQ=0
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_CT=0
FAIL_CT=0
SKIP_CT=0

DOCKER=${DOCKER:-docker}
if ! command -v "$DOCKER" >/dev/null 2>&1; then
  echo -e "${RED}Docker is required but not found on PATH.${NC}" >&2
  exit 127
fi

require_env() {
  local name=$1
  if [ -z "${!name:-}" ]; then
    echo -e "${YELLOW}WARN: Env var ${name} is not set. Tests that rely on it may fail or be skipped.${NC}" >&2
  fi
}

require_env OPENAI_API_KEY
require_env ANTHROPIC_API_KEY
require_env GEMINI_API_KEY
require_env GOOGLE_API_KEY
require_env DASHSCOPE_API_KEY

tmpdir() {
  mktemp -d 2>/dev/null || mktemp -d -t "ha-smoke"
}

die() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }

note() { echo -e "${YELLOW}[*]${NC} $*"; }
ok() { echo -e "${GREEN}[PASS]${NC} $*"; PASS_CT=$((PASS_CT+1)); }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAIL_CT=$((FAIL_CT+1)); }
skip() { echo -e "${YELLOW}[SKIP]${NC} $*"; SKIP_CT=$((SKIP_CT+1)); }

json_get() {
  local file=$1 key=$2
  if [ $HAVE_JQ -eq 1 ]; then
    "$JQ" -r ".${key}" "$file" 2>/dev/null || true
  else
    # naive fallback: grep the key; acceptable for smoke tests
    grep -o '"'"$key"'"[[:space:]]*:[[:space:]]*"\?[^",}]*' "$file" | head -n1 | sed 's/.*:[[:space:]]*\"\?//; s/\"$//' || true
  fi
}

build_image() {
  local name=$1 df=$2
  note "Building ${name} from ${df}"
  local args=()
  if [ "${FORCE_REBUILD:-0}" = "1" ]; then
    args+=(--no-cache --pull)
  fi
  if [ -n "${DOCKER_BUILD_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    args+=(${DOCKER_BUILD_ARGS})
  fi
  (cd "$SCRIPT_DIR" && "$DOCKER" build ${args:+"${args[@]}"} -f "$df" -t "$name" .) >/dev/null
}

run_and_wait_metrics() {
  # Run container detached, wait for /results/metrics.json to appear, then stop container.
  # Usage: run_and_wait_metrics name image prompt [timeout] [as_root]
  local name=$1 image=$2 prompt=$3 timeout=${4:-30} as_root=${5:-0}
  local rdir
  rdir=$(tmpdir)
  local cname="${name}-$(date +%s)"
  if [ "$as_root" = "1" ]; then
    "$DOCKER" run -d --user 0:0 --name "$cname" -e PROMPT="$prompt" -v "$rdir":/results "$image" >/dev/null || return 1
  else
    "$DOCKER" run -d --name "$cname" -e PROMPT="$prompt" -v "$rdir":/results "$image" >/dev/null || return 1
  fi
  local ms="$rdir/metrics.json"
  local waited=0
  while [ $waited -lt $timeout ]; do
    if [ -s "$ms" ]; then
  # Intentionally avoid printing to stdout here to keep command substitution clean.
  # A status note goes to stderr; the caller will mark PASS/FAIL.
  echo "${name}: metrics.json created" >&2
      "$DOCKER" rm -f "$cname" >/dev/null 2>&1 || true
      echo "$ms"
      return 0
    fi
    sleep 1; waited=$((waited+1))
  done
  echo "$ms"
  return 2
}

run_ephemeral_and_expect_metrics() {
  # Run container in foreground; expect it to exit and write /results/metrics.json.
  local name=$1 image=$2 prompt=$3
  local rdir
  rdir=$(tmpdir)
  "$DOCKER" run --rm -e PROMPT="$prompt" -v "$rdir":/results "$image" >/dev/null || true
  echo "$rdir/metrics.json"
}

run_simple_cli() {
  # Run a simple CLI check inside the container image
  local image=$1 shift_arg=$2
  "$DOCKER" run --rm "$image" "$shift_arg" >/dev/null
}

SUMMARY=()

# 1) Baseline code-server (control)
if [ "${SKIP_BASELINE:-0}" != "1" ]; then
  IMG_BASELINE="ha-code-server:local"
  if build_image "$IMG_BASELINE" "code-server.docker"; then
    ms=$(run_ephemeral_and_expect_metrics "baseline" "$IMG_BASELINE" "hello baseline")
    if [ -s "$ms" ]; then
      status=$(json_get "$ms" status)
      if [ "$status" = "noop" ]; then ok "baseline status=noop"; else fail "baseline unexpected status: $status"; fi
    else
      fail "baseline metrics.json missing"
    fi
  else
    fail "baseline build failed"
  fi
else
  skip "baseline (SKIP_BASELINE=1)"
fi

# 2) Cline via code-server (experimental)
if [ "${SKIP_CLINE:-0}" != "1" ]; then
  IMG_CLINE="ha-cline:local"
  if build_image "$IMG_CLINE" "cline.docker"; then
    ms=$(run_and_wait_metrics "cline" "$IMG_CLINE" "hello cline" 40 1) || true
    if [ -s "$ms" ]; then
      status=$(json_get "$ms" status)
      if [ "$status" = "started" ]; then ok "cline started"; else fail "cline unexpected status: $status"; fi
    else
      fail "cline metrics.json not found within timeout"
    fi
  else
    fail "cline build failed"
  fi
else
  skip "cline (SKIP_CLINE=1)"
fi

# 3) RooCode via code-server (experimental)
if [ "${SKIP_ROOCODE:-0}" != "1" ]; then
  IMG_ROOCODE="ha-roocode:local"
  if build_image "$IMG_ROOCODE" "roocode.docker"; then
    ms=$(run_and_wait_metrics "roocode" "$IMG_ROOCODE" "hello roocode" 40 1) || true
    if [ -s "$ms" ]; then
      status=$(json_get "$ms" status)
      if [ "$status" = "started" ]; then ok "roocode started"; else fail "roocode unexpected status: $status"; fi
    else
      fail "roocode metrics.json not found within timeout"
    fi
  else
    fail "roocode build failed"
  fi
else
  skip "roocode (SKIP_ROOCODE=1)"
fi

# 4) OpenHands headless
if [ "${SKIP_OPENHANDS:-0}" != "1" ]; then
  IMG_OPENHANDS="ha-openhands:local"
  if build_image "$IMG_OPENHANDS" "openhands.docker"; then
    work=$(tmpdir)
    # Keep workspace empty to avoid heavy git ops inside the container
    ms_dir=$(tmpdir)
    note "Running OpenHands with MAX_TURNS=1 (this may still take time if images/models need pulling)."
    # Determine LLM_API_KEY from available keys; prefer Anthropic for default anthropic model
    LLM_KEY="${LLM_API_KEY:-}"
    if [ -z "$LLM_KEY" ]; then
      if [ -n "${ANTHROPIC_API_KEY:-}" ]; then LLM_KEY="$ANTHROPIC_API_KEY"; fi
    fi
    if [ -z "$LLM_KEY" ] && [ -n "${OPENAI_API_KEY:-}" ]; then LLM_KEY="$OPENAI_API_KEY"; fi
    set +e
    "$DOCKER" run --rm \
      -e PROMPT="List files in workspace and exit" \
      -e MAX_TURNS=1 \
      -e LLM_API_KEY="$LLM_KEY" \
      -v "$work":/workspace -v "$ms_dir":/results \
      "$IMG_OPENHANDS" >/dev/null
    rc=$?
    set -e
    ms="$ms_dir/metrics.json"
    if [ -s "$ms" ]; then
      status=$(json_get "$ms" status)
      if [ -n "$status" ]; then ok "openhands wrote metrics (status=${status}, rc=$rc)"; else fail "openhands metrics missing status"; fi
    else
      fail "openhands metrics.json missing"
    fi
  else
    fail "openhands build failed (image pull might have failed)"
  fi
else
  skip "openhands (SKIP_OPENHANDS=1)"
fi

# 5) Rovo Dev (ACLI)
if [ "${SKIP_ROVO:-0}" != "1" ]; then
  IMG_ROVO="ha-rovo:local"
  if build_image "$IMG_ROVO" "rovo.docker"; then
    work=$(tmpdir)
    ms_dir=$(tmpdir)
    # Run detached with a cap on wait; the harness writes metrics even on error
    cname="rovo-$(date +%s)"
    "$DOCKER" run -d --name "$cname" -e PROMPT="Add a hello.txt file" -v "$work":/workspace -v "$ms_dir":/results "$IMG_ROVO" >/dev/null || true
    waited=0; timeout=60; ms="$ms_dir/metrics.json"
    while [ $waited -lt $timeout ]; do
      if [ -s "$ms" ]; then break; fi
      sleep 2; waited=$((waited+2))
    done
    "$DOCKER" logs "$cname" >/dev/null 2>&1 || true
    "$DOCKER" rm -f "$cname" >/dev/null 2>&1 || true
    if [ -s "$ms" ]; then
      status=$(json_get "$ms" status)
      if [ -n "$status" ]; then ok "rovo wrote metrics (status=${status})"; else fail "rovo metrics missing status"; fi
    else
      fail "rovo metrics.json not found within timeout"
    fi
  else
    fail "rovo build failed (ACLI download may have failed)"
  fi
else
  skip "rovo (SKIP_ROVO=1)"
fi

# 6) Cursor CLI image
if [ "${SKIP_CURSOR:-0}" != "1" ]; then
  IMG_CURSOR="ha-cursor:local"
  if build_image "$IMG_CURSOR" "cursor.docker"; then
    if run_simple_cli "$IMG_CURSOR" "--help"; then
      ok "cursor CLI responds to --help"
    else
      fail "cursor CLI did not respond successfully"
    fi
  else
    fail "cursor build failed"
  fi
else
  skip "cursor (SKIP_CURSOR=1)"
fi

# 7) VS Code base image
if [ "${SKIP_CODEIMG:-1}" = "1" ]; then
  skip "code (SKIP_CODEIMG=1 by default due to heavy install)"
else
  IMG_CODE="ha-code:local"
  if build_image "$IMG_CODE" "code.docker"; then
    if "$DOCKER" run --rm "$IMG_CODE" code --version >/dev/null 2>&1; then
      ok "code installed in image"
    else
      fail "code not available in image"
    fi
  else
    fail "code image build failed"
  fi
fi

# 8) Claude Code CLI image
if [ "${SKIP_CLAUDECODE:-0}" != "1" ]; then
  IMG_CLAUDECODE="ha-claudecode:local"
  if build_image "$IMG_CLAUDECODE" "claudecode.docker"; then
    if "$DOCKER" run --rm "$IMG_CLAUDECODE" claude --help >/dev/null 2>&1; then
      ok "claude-code CLI responds to --help"
    else
      fail "claude-code CLI not working"
    fi
  else
    fail "claude-code image build failed"
  fi
else
  skip "claude-code (SKIP_CLAUDECODE=1)"
fi

# 9) Gemini (official gemini-cli) harness
if [ "${SKIP_GEMINI:-0}" != "1" ]; then
  if [ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" ]; then
    skip "gemini (no GEMINI_API_KEY/GOOGLE_API_KEY)"
  else
    IMG_GEMINI="ha-gemini:local"
    if build_image "$IMG_GEMINI" "gemini.docker"; then
      rdir=$(tmpdir)
      set +e
      GEM_FLAGS=()
      [ -n "${GEMINI_API_KEY:-}" ] && GEM_FLAGS+=( -e GEMINI_API_KEY="$GEMINI_API_KEY" )
      [ -n "${GOOGLE_API_KEY:-}" ] && GEM_FLAGS+=( -e GOOGLE_API_KEY="$GOOGLE_API_KEY" )
      "$DOCKER" run --rm -e PROMPT="List files briefly" "${GEM_FLAGS[@]}" -v "$rdir":/results "$IMG_GEMINI" >/dev/null 2>&1
      rc=$?
      set -e
      ms="$rdir/metrics.json"
      if [ -s "$ms" ]; then
        status=$(json_get "$ms" status)
        case "$status" in
          success) ok "gemini success" ;;
          error) fail "gemini error (msg=$(json_get $ms message))" ;;
          *) ok "gemini status=$status (non-fatal)" ;;
        esac
      else
        fail "gemini metrics.json missing"
      fi
    else
      fail "gemini build failed"
    fi
  fi
else
  skip "gemini (SKIP_GEMINI=1)"
fi

# 10) Qwen (qwen-code CLI) harness
if [ "${SKIP_QWEN:-0}" != "1" ]; then
  if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    skip "qwen (no DASHSCOPE_API_KEY)"
  else
    IMG_QWEN="ha-qwen:local"
    if build_image "$IMG_QWEN" "qwen.docker"; then
      rdir=$(tmpdir)
      set +e
      QWEN_FLAGS=()
      [ -n "${OPENAI_API_KEY:-}" ] && QWEN_FLAGS+=( -e OPENAI_API_KEY="$OPENAI_API_KEY" )
      [ -n "${DASHSCOPE_API_KEY:-}" ] && QWEN_FLAGS+=( -e DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY" )
      "$DOCKER" run --rm -e PROMPT="Summarize repository layout" "${QWEN_FLAGS[@]}" -v "$rdir":/results "$IMG_QWEN" >/dev/null 2>&1
      rc=$?
      set -e
      ms="$rdir/metrics.json"
      if [ -s "$ms" ]; then
        status=$(json_get "$ms" status)
        case "$status" in
          success) ok "qwen success" ;;
          error) fail "qwen error (msg=$(json_get $ms message))" ;;
          *) ok "qwen status=$status (non-fatal)" ;;
        esac
      else
        fail "qwen metrics.json missing"
      fi
    else
      fail "qwen build failed"
    fi
  fi
else
  skip "qwen (SKIP_QWEN=1)"
fi

echo ""
echo "==== Summary ===="
echo -e "${GREEN}PASS: $PASS_CT${NC}  ${RED}FAIL: $FAIL_CT${NC}  ${YELLOW}SKIP: $SKIP_CT${NC}"
if [ $FAIL_CT -gt 0 ]; then exit 1; fi
exit 0
