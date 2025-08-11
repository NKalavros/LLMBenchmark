Make sure I gitignore big files with: find * -size +1M | cat >> .gitignore

# Agent Benchmark

This adds an initial benchmarking toolkit (`agent_benchmark`) to:

1. Discover code files across agent directories.
2. Produce semantic summaries (DSPy if available) & embeddings (OpenAI or deterministic hash fallback).
3. (Skeleton) Evaluate RMSE from existing prediction CSVs. This does not work yet.
4. Generate heuristic quality scores plus optional DeepEval (GEval) LLM-based metrics.
5. Assemble a simple markdown report.
6. Cluster code by embedding similarity & generate a heatmap.
7. Provide advanced clustering analytics: per-cluster stats (avg LOC, size, intra-cluster similarity, agent diversity), silhouette score, and hierarchical (Ward) multi‑k summaries.

## Usage

Install dependencies (Mostly DeepEval and DSPy)

```bash
pip install -r requirements.txt
```

Run the pipeline (mock / offline mode):

```bash
python -m agent_benchmark discover --root .
python -m agent_benchmark analyze --root . --store-full-embeddings
python -m agent_benchmark cluster --root .   # build similarity clusters, stats, heatmap, hierarchy
python -m agent_benchmark evaluate --root . # Currently does not work because no runs returns CSV files for RMSE. I will remake the prompts for this
python -m agent_benchmark quality --root . 
python -m agent_benchmark plots --root . 
python -m agent_benchmark report --root .
```

To generate REAL OpenAI embeddings (requires your `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=YOUR_KEY
python -m agent_benchmark analyze --root . --openai-model text-embedding-3-small
```

Store full embedding vectors (instead of only a 16-d preview) and also write a compressed `embeddings.npz` matrix:

```bash
python -m agent_benchmark analyze --root . --store-full-embeddings
```

Artifacts land in `analyses/` (configurable with `--outdir`). Key outputs now include:

- `files_index.json` – discovered files metadata (loc, size).
- `code_semantics.json` – summaries + embeddings (preview / full).
- `clusters.json` – cluster assignments with stats (agent_diversity, avg_loc, avg_intra_similarity, silhouette, hierarchical levels).
- `similarity_matrix.npz` – cosine similarity matrix + file list.
- `similarity_heatmap.png` – visual snapshot of top-N similarity matrix slice.
- `quality_scores.json` – heuristic + optional LLM quality metrics.
- `rmse_results.json` – (when prediction CSVs present) RMSE values.
- `report.md` – consolidated human-readable summary.

## Embeddings, Summaries & LLM Metrics

If `OPENAI_API_KEY` is set and libraries are installed:

- Analyzer uses OpenAI embeddings (model configurable via `--openai-model`).
- DSPy summarization attempts richer structured summaries (set `AGENT_BENCHMARK_DSPY_MODEL` to override default model).
- Quality scoring can invoke DeepEval GEval metrics (readability, modularity) automatically.

Without an API key or with `--mock-llm`, deterministic hash embeddings and heuristic summaries/quality scores are produced for reproducibility.

## Clustering & Similarity Analysis

The `cluster` command performs:

- Vector clustering (KMeans if available; cosine fallback) with heuristic `k = sqrt(n/2)` when not provided.
- Per-cluster statistics: size, representative file (highest mean intra similarity), agent diversity, modeling approaches, avg LOC, avg size, avg intra-cluster similarity.
- Global silhouette score (if multiple clusters & sklearn metrics available).
- Hierarchical (Ward) clustering at multiple k values (k/2, k, k+1) to expose structural granularity.
- Similarity artifacts: compressed matrix (`similarity_matrix.npz`) and optional heatmap image.

Report excerpts display top clusters (by size) plus silhouette and a brief hierarchical level summary.

## Next Steps

- Integrate additional DeepEval custom metrics (maintainability, testability).
- Automatic optimal k selection (silhouette sweep / gap statistic) & dendrogram visualization.
- Implement active model execution wrapper & standardized prediction artifact format.
- Expand reporting visuals (cluster distribution charts, similarity histograms, per-agent comparative tables).
- Interactive exploration (optional future: lightweight Streamlit dashboard).

---

This is an evolving scaffold; contributions & iterative enhancements welcome.
