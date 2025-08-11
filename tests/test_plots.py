import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_benchmark.core.plots import generate_plots

def test_generate_plots(tmp_path):
    analyses = tmp_path / 'analyses'
    analyses.mkdir()
    quality = {
        'files': [
            {'agent':'A','path':'A/f1.py','heuristic_score':3.1,'llm_metrics':{'readability':0.8,'modularity':0.3}},
            {'agent':'A','path':'A/f2.py','heuristic_score':2.9,'llm_metrics':{'readability':0.82,'modularity':0.35}},
            {'agent':'B','path':'B/f3.py','heuristic_score':2.2,'llm_metrics':{'readability':0.7,'modularity':0.55}},
            {'agent':'B','path':'B/f4.py','heuristic_score':3.5,'llm_metrics':{'readability':0.9,'modularity':0.4}},
        ],
        'used_llm': True,
        'llm_metric_names': ['readability','modularity']
    }
    clusters = {
        'clusters': [
            {'cluster_id':0,'size':2,'avg_intra_similarity':0.9,'agent_diversity':0.5},
            {'cluster_id':1,'size':2,'avg_intra_similarity':0.7,'agent_diversity':1.0},
        ],
        'n_clusters':2,'n_files':4
    }
    (analyses/'quality_scores.json').write_text(json.dumps(quality))
    (analyses/'clusters.json').write_text(json.dumps(clusters))
    generate_plots(analyses)
    # Basic existence checks
    assert (analyses/'quality_distribution.png').exists()
    assert (analyses/'readability_vs_modularity.png').exists()
    assert (analyses/'agent_quality_boxplot.png').exists()
    assert (analyses/'agent_avg_metrics.png').exists()
    assert (analyses/'cluster_size_vs_similarity.png').exists()
    assert (analyses/'cluster_agent_diversity.png').exists()
