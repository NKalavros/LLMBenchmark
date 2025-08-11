import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_benchmark.core.cluster import cluster_codebase


def test_cluster_basic(tmp_path):
    analyses = tmp_path / 'analyses'
    analyses.mkdir()
    # Minimal code_semantics with pseudo embeddings
    files = []
    for i in range(6):
        files.append({
            'agent': 'A' if i < 3 else 'B',
            'path': f'Agent/file_{i}.py',
            'embedding_preview': [float(i % 3), 1.0, 0.0, 0.0],
            'modeling_approach': None,
        })
    (analyses / 'code_semantics.json').write_text(json.dumps({'files': files}))
    out = cluster_codebase(analyses, n_clusters=2)
    assert out['n_clusters'] == 2
    assert (analyses / 'clusters.json').exists()
