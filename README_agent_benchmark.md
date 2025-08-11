# Agent Benchmark Skeleton

This adds an initial benchmarking toolkit (`agent_benchmark`) to:

1. Discover code files across agent directories.
2. Produce semantic summaries (DSPy if available) & embeddings (OpenAI or deterministic hash fallback).
3. (Skeleton) Evaluate RMSE from existing prediction CSVs.
4. Generate heuristic quality scores plus optional DeepEval (GEval) LLM-based metrics.
5. Assemble a simple markdown report.
6. Cluster code by embedding similarity & generate a heatmap.
7. Provide advanced clustering analytics: per-cluster stats (avg LOC, size, intra-cluster similarity, agent diversity), silhouette score, and hierarchical (Ward) multi‑k summaries.
8. Generate comparative visualization plots (quality distributions, LLM metric relationships, cluster characteristics).

## Usage

Install dependencies (optional at this stage; discovery & heuristic analysis rely only on stdlib):

```bash
pip install -r requirements.txt
```

Run the pipeline (mock / offline mode):

```bash
python -m agent_benchmark discover --root .
python -m agent_benchmark analyze --root . 
python -m agent_benchmark cluster --root .   # build similarity clusters, stats, heatmap, hierarchy
python -m agent_benchmark plots --root .     # generate visualization PNGs from analysis artifacts
python -m agent_benchmark evaluate --root . # Currently does not work because no runs returns CSV files for RMSE.
python -m agent_benchmark quality --root . 
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
- `similarity_heatmap_labels.csv` – mapping of heatmap tick labels to full file paths.
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

## Visualization / Plots

After running `quality` and `cluster`, invoke:

```bash
python -m agent_benchmark plots --root .
```

This produces (when underlying data are present):

- `quality_distribution.png` – Histogram (with KDE) of heuristic quality scores.
- `readability_vs_modularity.png` – Scatter of LLM readability vs modularity per file, colored by agent.
- `readability_modularity_density.png` – (Large datasets) Hexbin density for readability vs modularity.
- `agent_quality_boxplot.png` – Distribution of heuristic scores per agent.
- `agent_avg_metrics.png` – Side‑by‑side bar chart of average readability / modularity per agent.
- `cluster_size_vs_similarity.png` – Cluster size vs average intra‑cluster similarity scatter (colored by agent diversity).
- `cluster_agent_diversity.png` – Histogram of agent diversity (unique agents / cluster size).
- `similarity_heatmap.png` – Now annotated with disambiguated short file labels (if <= 40 files displayed). Full mapping in `similarity_heatmap_labels.csv`.

All plots are written to the same analyses output directory. Plots auto‑skip if required metrics are absent (e.g. no LLM metrics -> skip readability/modularity plots).

### Tips

- Rerun `plots` after re‑running earlier stages; images are overwritten.
- Adjust the number of files displayed in the heatmap by editing `max_show` inside `cluster.py` if needed for deeper inspection.
- The density hexbin activates only when ≥ 50 files contain both readability and modularity metrics.

## Next Steps

- Integrate additional DeepEval custom metrics (maintainability, testability).
- Automatic optimal k selection (silhouette sweep / gap statistic) & dendrogram visualization.
- Implement active model execution wrapper & standardized prediction artifact format.
- Expand reporting visuals (cluster distribution charts, similarity histograms, per-agent comparative tables).
- Interactive exploration (optional future: lightweight Streamlit dashboard).

---

This is an evolving scaffold; contributions & iterative enhancements welcome.
