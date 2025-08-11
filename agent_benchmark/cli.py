import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

from .core.discovery import discover_files
from .core.analyze import analyze_codebase
from .core.evaluate import evaluate_models
from .core.quality import quality_scores
from .core.report import build_report
from .core.cluster import cluster_codebase


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-benchmark",
        description="Semantic + performance benchmarking across agent-generated code",
    )
    p.add_argument("command", choices=["discover", "analyze", "evaluate", "quality", "cluster", "report"], help="Action to run")
    p.add_argument("--root", default=".", help="Root directory containing agent folders")
    p.add_argument("--outdir", default="analyses", help="Directory to write artifacts")
    p.add_argument("--agent-filter", nargs="*", help="Subset of agent folder names to include")
    p.add_argument("--mock-llm", action="store_true", help="Force mock mode even if API keys exist")
    p.add_argument("--openai-model", default="text-embedding-3-small", help="OpenAI embedding model name")
    p.add_argument("--max-files", type=int, help="Optional cap for number of files during analysis (debug)")
    p.add_argument("--store-full-embeddings", action="store_true", help="Store full embedding vectors (may create large JSON). Also writes embeddings.npz.")
    return p


def main(argv=None):  # pragma: no cover - thin wrapper
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat()
    meta_path = outdir / "run_meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({"created": timestamp, "root": str(root)}, indent=2))

    if args.command == "discover":
        index = discover_files(root, outdir, agent_filter=args.agent_filter)
        print(f"Indexed {len(index['files'])} files across {len(index['agents'])} agents.")
    elif args.command == "analyze":
        analyze_codebase(
            root,
            outdir,
            agent_filter=args.agent_filter,
            mock=args.mock_llm,
            model=args.openai_model,
            max_files=args.max_files,
            store_full=args.store_full_embeddings,
        )
    elif args.command == "evaluate":
        evaluate_models(root, outdir, agent_filter=args.agent_filter)
    elif args.command == "quality":
        quality_scores(root, outdir, agent_filter=args.agent_filter, mock=args.mock_llm)
    elif args.command == "cluster":
        cluster_codebase(outdir)
    elif args.command == "report":
        build_report(outdir)
    else:  # pragma: no cover - argparse prevents
        raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
