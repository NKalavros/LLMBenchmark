import json
from pathlib import Path


def build_report(outdir: Path):
    report_path = outdir / "report.md"
    lines = ["# Agent Benchmark Report", ""]
    # Load artifacts if present
    def load(name):
        p = outdir / name
        return json.loads(p.read_text()) if p.exists() else None

    index = load("files_index.json")
    semantics = load("code_semantics.json")
    rmse = load("rmse_results.json")
    quality = load("quality_scores.json")
    clusters = load("clusters.json")

    if index:
        lines.append(f"Indexed Agents: {', '.join(index['agents'])}")
        lines.append(f"Total Code Files: {len(index['files'])}")
        lines.append("")

    if rmse and rmse.get('results'):
        lines.append("## RMSE Results")
        for r in sorted(rmse['results'], key=lambda x: x['rmse']):
            lines.append(f"- {r['agent']}: {r['rmse']:.4f} ({r['file']})")
        lines.append("")
    else:
        lines.append("_No RMSE results available yet._\n")

    if quality:
        lines.append("## Quality Heuristic (Top 5)")
        top = sorted(quality['files'], key=lambda x: x['heuristic_score'], reverse=True)[:5]
        for q in top:
            lines.append(f"- {q['agent']} :: {q['path']} => {q['heuristic_score']}")
        lines.append("")
        # Aggregate LLM metrics if available
        if quality.get('used_llm') and quality.get('files'):
            lines.append("## LLM Quality Metrics (Averages)")
            metric_names = quality.get('llm_metric_names', [])
            if metric_names:
                accum = {m: [] for m in metric_names}
                for fentry in quality['files']:
                    lm = fentry.get('llm_metrics') or {}
                    for m in metric_names:
                        v = lm.get(m)
                        if isinstance(v, (int, float)):
                            accum[m].append(v)
                for m in metric_names:
                    vals = accum[m]
                    if vals:
                        avg = sum(vals)/len(vals)
                        lines.append(f"- {m}: {avg:.3f} (n={len(vals)})")
                lines.append("")

    if clusters:
        lines.append("## Similarity Clusters")
        lines.append(f"Total Clusters: {clusters['n_clusters']} (files clustered: {clusters['n_files']})")
        if clusters.get('silhouette') is not None:
            lines.append(f"Silhouette Score: {clusters['silhouette']:.3f}")
        show_clusters = sorted(clusters['clusters'], key=lambda x: (-x['size'], x['cluster_id']))[:10]
        for c in show_clusters:
            lines.append(
                f"- Cluster {c['cluster_id']} size={c['size']} agents={c['agent_count']} div={c['agent_diversity']:.2f} "
                f"avg_loc={c['avg_loc']:.1f} intra_sim={c['avg_intra_similarity']:.2f} rep={c['representative']}"
            )
        if clusters['n_clusters'] > 10:
            lines.append(f"... ({clusters['n_clusters'] - 10} more clusters) ...")
        levels = clusters.get('hierarchical', {}).get('levels') or []
        if levels:
            lines.append("")
            lines.append("Hierarchical Levels (k -> cluster sizes):")
            for lvl in levels[:5]:
                sizes = sorted([len(v) for v in lvl['clusters'].values()], reverse=True)
                sizes_str = ','.join(map(str, sizes[:6])) + ("..." if len(sizes) > 6 else "")
                lines.append(f"- k={lvl['k']}: sizes={sizes_str}")
        lines.append("")

    if semantics:
        lines.append("## Sample Semantic Summaries")
        for f in semantics['files'][:5]:
            approach = f.get('modeling_approach')
            prefix = f"[{approach}] " if approach else ''
            lines.append(f"- {prefix}{f['path']}: {f['summary'][:120]}...")
        lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")
