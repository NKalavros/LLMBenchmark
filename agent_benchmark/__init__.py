"""Agent Benchmarking Toolkit

Provides CLI utilities to:
 - discover: index code files across agent folders
 - analyze: produce lightweight semantic summaries & placeholder embeddings
 - evaluate: (skeleton) collect / compute RMSE metrics for model outputs
 - quality: (skeleton) heuristic + optional LLM quality scoring
 - report: aggregate artifacts into a markdown report

Real LLM-based embeddings & scoring are only activated when OPENAI_API_KEY
is present and corresponding libraries are installed. Otherwise a deterministic
mock mode is used so the pipeline remains reproducible offline.
"""

__all__ = [
    "main",
]

from .cli import main  # noqa: E402
