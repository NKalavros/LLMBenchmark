# HeadlessAgents Smoke Tests

This folder contains Dockerfiles and run harnesses for headless agent experiments. Use the provided smoke test to quickly verify builds and basic behavior.

## Prereqs
- Docker installed and running
- Optional: `jq` for nicer JSON parsing
- Environment variables set if available:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`

## Quick start
Run the smoke tests from this folder:

```bash
./test_smoke.sh
```

The script:
- Builds each image
- Runs a tiny check
- Looks for `/results/metrics.json` or CLI `--help` output
- Prints a PASS/FAIL/SKIP summary

## Tuning
You can skip heavy or flaky tests via env flags:

- `SKIP_BASELINE=1` – skip baseline code-server
- `SKIP_CLINE=1` – skip Cline adapter
- `SKIP_ROOCODE=1` – skip RooCode adapter
- `SKIP_OPENHANDS=1` – skip OpenHands (pulls large images)
- `SKIP_ROVO=1` – skip Rovo Dev ACLI
- `SKIP_CURSOR=1` – skip Cursor CLI
- `SKIP_CODEIMG=1` – skip VS Code base image (default)
- `SKIP_CLAUDECODE=1` – skip Claude Code CLI
- `SKIP_GEMINI=1` – skip Gemini harness (needs GEMINI_API_KEY or GOOGLE_API_KEY)
- `SKIP_QWEN=1` – skip Qwen harness (needs DASHSCOPE_API_KEY)

Example:

```bash
SKIP_OPENHANDS=1 SKIP_CODEIMG=1 SKIP_GEMINI=1 ./test_smoke.sh
```

## Notes
- OpenHands test runs with `MAX_TURNS=1` to keep it fast, but the initial image pull can still be slow.
- Rovo harness writes metrics even when ACLI fails; check `/results/agent_stdout.log` if debugging.
- Cline and RooCode adapters currently just bring up `code-server` and write a stub metrics file.
- Gemini & Qwen harnesses execute a single prompt and write basic metrics; they will report `status:error` if API keys are absent.
