import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_benchmark.core.cluster import cluster_codebase


def test_cluster_stats_and_hierarchical(tmp_path):
    analyses = tmp_path / 'analyses'
    analyses.mkdir()
    # Create semantics and index with loc/size
    files_sem = []
    index_files = []
    for i in range(8):
        emb = [float(i % 4), 0.5, 0.1]
        path = f'Agent/file_{i}.py'
        files_sem.append({'agent': 'A' if i < 4 else 'B', 'path': path, 'embedding_preview': emb, 'modeling_approach': None})
        index_files.append({'agent': 'A' if i < 4 else 'B', 'path': path, 'language': 'python', 'size_bytes': 100 + i, 'loc': 10 + i})
    (analyses / 'code_semantics.json').write_text(json.dumps({'files': files_sem}))
    (analyses / 'files_index.json').write_text(json.dumps({'agents': ['A','B'], 'files': index_files}))
    out = cluster_codebase(analyses, n_clusters=3)
    assert out['n_clusters'] >= 2
    first = out['clusters'][0]
    assert 'avg_loc' in first and 'avg_intra_similarity' in first
    assert 'hierarchical' in out
