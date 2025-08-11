import json
import math
from pathlib import Path
from typing import List, Dict, Optional


def cluster_codebase(outdir: Path, n_clusters: Optional[int] = None):
    """Cluster code files using stored embeddings.

    Produces:
      - clusters.json (cluster metadata)
      - similarity_matrix.npz (cosine similarity matrix + file list)
      - similarity_heatmap.png (if matplotlib/seaborn available)
    """
    sem_path = outdir / "code_semantics.json"
    if not sem_path.exists():  # pragma: no cover - invalid usage
        raise SystemExit("Run analyze before clustering")
    data = json.loads(sem_path.read_text())
    files = data.get('files', [])
    if not files:  # pragma: no cover - edge
        raise SystemExit("No files to cluster")

    # Load index for LOC/size stats if present
    index_path = outdir / 'files_index.json'
    idx_map = {}
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
            for rec in idx.get('files', []):
                idx_map[rec['path']] = rec
        except Exception:  # pragma: no cover
            idx_map = {}

    # Prefer full embeddings if any are present
    embed_key = 'embedding_full' if any('embedding_full' in f for f in files) else 'embedding_preview'
    vectors = []
    kept_files = []
    for f in files:
        vec = f.get(embed_key)
        if not vec:
            continue
        kept_files.append(f)
        vectors.append(vec)

    import numpy as np
    X = np.array(vectors, dtype=float)
    n = X.shape[0]
    if n == 0:  # pragma: no cover - defensive
        raise SystemExit("No embeddings available for clustering")
    if n == 1:
        single_cluster = [{
            "cluster_id": 0,
            "files": [kept_files[0]['path']],
            "size": 1,
            "representative": kept_files[0]['path'],
            "agents": [kept_files[0]['agent']],
            "modeling_approaches": list(filter(None, [kept_files[0].get('modeling_approach')]))
        }]
        out = {"clusters": single_cluster, "n_clusters": 1, "embedding_source": embed_key, "n_files": 1}
        (outdir / 'clusters.json').write_text(json.dumps(out, indent=2))
        return out

    if not n_clusters:
        # heuristic: sqrt(n/2) bounded
        n_clusters = max(2, min(n - 1, int(math.sqrt(n / 2)) or 2))

    # Try sklearn KMeans else cosine one-pass
    try:
        from sklearn.cluster import KMeans  # type: ignore
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
    except Exception:  # pragma: no cover - fallback path
        centroids = X[:n_clusters]
        labels = []
        import numpy as np
        for vec in X:
            sims = [float(vec.dot(c) / (np.linalg.norm(vec) * np.linalg.norm(c) + 1e-9)) for c in centroids]
            labels.append(int(max(range(len(sims)), key=lambda i: sims[i])))

    clusters: Dict[int, List[dict]] = {}
    for lbl, meta in zip(labels, kept_files):
        clusters.setdefault(int(lbl), []).append(meta)

    import numpy as np
    cluster_entries = []
    for cid, flist in clusters.items():
        idxs = [kept_files.index(f) for f in flist]
        subX = X[idxs]
        sim_mat = subX @ subX.T
        norms = (np.linalg.norm(subX, axis=1) + 1e-9)
        sim_norm = sim_mat / (norms[:, None] * norms[None, :])
        avg_sims = sim_norm.mean(axis=1)
        rep_i = int(avg_sims.argmax())
        rep_file = flist[rep_i]['path']
        agents = sorted({f['agent'] for f in flist})
        approaches = sorted({a for a in (f.get('modeling_approach') for f in flist) if a})
        # Stats
        paths = [f['path'] for f in flist]
        locs = [idx_map.get(p, {}).get('loc', 0) for p in paths]
        sizes = [idx_map.get(p, {}).get('size_bytes', 0) for p in paths]
        avg_loc = float(sum(locs) / len(locs)) if locs else 0.0
        avg_size = float(sum(sizes) / len(sizes)) if sizes else 0.0
        avg_similarity = float(avg_sims.mean()) if len(flist) > 1 else 1.0
        cluster_entries.append({
            'cluster_id': cid,
            'size': len(flist),
            'representative': rep_file,
            'agents': agents,
            'agent_count': len(agents),
            'agent_diversity': len(agents) / len(flist),
            'modeling_approaches': approaches,
            'modeling_approach_count': len(approaches),
            'avg_loc': avg_loc,
            'avg_size_bytes': avg_size,
            'avg_intra_similarity': avg_similarity,
            'files': paths,
        })

    # Full similarity artifacts
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim_matrix = (Xn @ Xn.T).astype('float32')
    np.savez_compressed(outdir / 'similarity_matrix.npz', similarity=sim_matrix, files=[f['path'] for f in kept_files])
    try:  # visualization optional
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        sns.set_context('paper')
        max_show = min(40, sim_matrix.shape[0])

        # Build short, unique labels for displayed files
        shown_paths = [f['path'] for f in kept_files[:max_show]]
        basenames = [Path(p).name for p in shown_paths]
        # Disambiguate duplicate basenames by prefixing parent directory
        duplicates = {name for name in basenames if basenames.count(name) > 1}
        short_labels = []
        for p, base in zip(shown_paths, basenames):
            if base in duplicates:
                parent = Path(p).parent.name
                short_labels.append(f"{parent}/{base}")
            else:
                short_labels.append(base)

        # Further trim very long labels
        def trim(label: str, limit: int = 28) -> str:
            return label if len(label) <= limit else label[:limit-3] + '…'

        display_labels = [trim(l) for l in short_labels]

        fig_w = min(14, max(6, 0.4 * max_show + 2))
        fig_h = min(14, max(4, 0.35 * max_show + 2))
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            sim_matrix[:max_show, :max_show],
            cmap='viridis',
            cbar=True,
            xticklabels=display_labels,
            yticklabels=display_labels,
            square=False
        )
        plt.title('Code Similarity (cosine)')
        plt.xticks(rotation=60, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / 'similarity_heatmap.png', dpi=150)
        plt.close()

        # Save mapping of displayed labels to full paths for reference
        try:
            mapping_lines = ["index,short_label,full_path"] + [f"{i},{display_labels[i]},{shown_paths[i]}" for i in range(len(display_labels))]
            (outdir / 'similarity_heatmap_labels.csv').write_text("\n".join(mapping_lines))
        except Exception:  # pragma: no cover
            pass
    except Exception:  # pragma: no cover
        pass

    # Silhouette score
    silhouette_val = None
    if len(set(labels)) > 1:
        try:
            from sklearn.metrics import silhouette_score  # type: ignore
            silhouette_val = float(silhouette_score(X, labels))
        except Exception:  # pragma: no cover
            silhouette_val = None

    # Hierarchical clustering (optional)
    hierarchical = {}
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
        candidate_ks = sorted({k for k in [max(2, n_clusters//2), n_clusters, min(n_clusters+1, n-1)] if 1 < k < n})
        levels = []
        for k in candidate_ks:
            try:
                ac = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
            except TypeError:  # older sklearn
                ac = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')  # type: ignore
            h_labels = ac.fit_predict(X)
            cluster_map: Dict[int, List[str]] = {}
            for lbl, meta in zip(h_labels, kept_files):
                cluster_map.setdefault(int(lbl), []).append(meta['path'])
            levels.append({'k': k, 'clusters': cluster_map})
        hierarchical = {'levels': levels, 'method': 'ward'}
    except Exception:  # pragma: no cover
        hierarchical = {}

    out = {
        'clusters': sorted(cluster_entries, key=lambda x: x['cluster_id']),
        'n_clusters': len(cluster_entries),
        'embedding_source': embed_key,
        'n_files': len(kept_files),
        'silhouette': silhouette_val,
        'hierarchical': hierarchical,
    }
    (outdir / 'clusters.json').write_text(json.dumps(out, indent=2))
    print(f"Generated {len(cluster_entries)} clusters from {len(kept_files)} files using {embed_key} vectors.")
    return out
