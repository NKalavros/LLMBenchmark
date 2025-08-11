import json
from pathlib import Path
from typing import List, Dict, Any
import math


def _load(outdir: Path, name: str):
    p = outdir / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def generate_plots(outdir: Path):
    """Generate static PNG plots from analysis artifacts.

    Creates (if data present):
      - quality_distribution.png (hist of heuristic scores)
      - readability_vs_modularity.png (scatter, colored by agent)
      - agent_quality_boxplot.png (boxplot heuristic by agent)
      - readability_modularity_density.png (2D hexbin) when enough points
      - cluster_size_vs_similarity.png (scatter cluster size vs avg intra similarity)
      - agent_avg_metrics.png (bar chart per-agent mean readability / modularity)
      - cluster_agent_diversity.png (hist of agent diversity)
    """
    quality = _load(outdir, 'quality_scores.json')
    clusters = _load(outdir, 'clusters.json')

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"Plot dependencies missing: {e}")
        return

    sns.set_context('talk')
    sns.set_style('whitegrid')

    if quality and quality.get('files'):
        qdf_rows = []
        for row in quality['files']:
            base = {k: v for k, v in row.items() if k != 'llm_metrics'}
            metrics = row.get('llm_metrics') or {}
            for mk, mv in metrics.items():
                base[mk] = mv
            qdf_rows.append(base)
        qdf = pd.DataFrame(qdf_rows)

        # Histogram of heuristic score
        if 'heuristic_score' in qdf.columns:
            plt.figure(figsize=(8,5))
            sns.histplot(data=qdf, x='heuristic_score', kde=True, bins=20, color='#4C72B0')
            plt.title('Heuristic Quality Score Distribution')
            plt.xlabel('Heuristic Score')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(outdir / 'quality_distribution.png', dpi=140)
            plt.close()

        # Readability vs Modularity scatter
        if 'readability' in qdf.columns and 'modularity' in qdf.columns:
            plt.figure(figsize=(8,6))
            sns.scatterplot(data=qdf, x='readability', y='modularity', hue='agent', alpha=0.75, s=50)
            plt.title('Readability vs Modularity')
            plt.tight_layout()
            plt.savefig(outdir / 'readability_vs_modularity.png', dpi=140)
            plt.close()

            # Hexbin density if enough points
            if len(qdf) >= 50:
                plt.figure(figsize=(8,6))
                hb = plt.hexbin(qdf['readability'], qdf['modularity'], gridsize=25, cmap='viridis', mincnt=1)
                plt.colorbar(hb, label='count')
                plt.xlabel('readability')
                plt.ylabel('modularity')
                plt.title('Readability vs Modularity Density')
                plt.tight_layout()
                plt.savefig(outdir / 'readability_modularity_density.png', dpi=140)
                plt.close()

        # Boxplot of heuristic score by agent
        if 'agent' in qdf.columns and 'heuristic_score' in qdf.columns:
            agent_counts = qdf['agent'].value_counts()
            top_agents = agent_counts.index.tolist()
            if len(top_agents) > 1:
                plt.figure(figsize=(min(12, 1.2*len(top_agents)+2),6))
                sns.boxplot(data=qdf, x='agent', y='heuristic_score', order=top_agents)
                plt.xticks(rotation=45, ha='right')
                plt.title('Heuristic Score by Agent')
                plt.tight_layout()
                plt.savefig(outdir / 'agent_quality_boxplot.png', dpi=150)
                plt.close()

        # Per-agent average readability/modularity
        metric_cols = [c for c in ['readability', 'modularity'] if c in qdf.columns]
        if metric_cols:
            agg = qdf.groupby('agent')[metric_cols].mean().reset_index()
            plt.figure(figsize=(min(12, 1.2*len(agg)+2), 6))
            width = 0.35
            x = np.arange(len(agg))
            for i, m in enumerate(metric_cols):
                offset = (i - (len(metric_cols)-1)/2) * width
                plt.bar(x + offset, agg[m], width=width, label=m, alpha=0.8)
            plt.xticks(x, list(agg['agent']), rotation=45, ha='right')
            plt.ylim(0, min(1.05, max(1.0, agg[metric_cols].max().max()+0.05)))
            plt.title('Average LLM Metrics per Agent')
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / 'agent_avg_metrics.png', dpi=150)
            plt.close()

    # Cluster plots
    if clusters and clusters.get('clusters'):
        cdf = pd.DataFrame(clusters['clusters'])
        if not cdf.empty:
            if 'avg_intra_similarity' in cdf.columns and 'size' in cdf.columns:
                plt.figure(figsize=(8,6))
                sns.scatterplot(data=cdf, x='size', y='avg_intra_similarity', hue='agent_diversity', palette='viridis', s=70)
                plt.title('Cluster Size vs Intra Similarity')
                plt.xlabel('Cluster Size')
                plt.ylabel('Avg Intra Similarity')
                plt.tight_layout()
                plt.savefig(outdir / 'cluster_size_vs_similarity.png', dpi=140)
                plt.close()
            if 'agent_diversity' in cdf.columns:
                plt.figure(figsize=(7,4))
                sns.histplot(data=cdf, x='agent_diversity', bins=10, kde=True, color='#2A9D8F')
                plt.title('Agent Diversity Across Clusters')
                plt.xlabel('Agent Diversity (unique agents / cluster size)')
                plt.tight_layout()
                plt.savefig(outdir / 'cluster_agent_diversity.png', dpi=140)
                plt.close()

    print("Plots generated.")
